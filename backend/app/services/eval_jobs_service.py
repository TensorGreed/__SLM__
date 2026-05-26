"""Job-runner wrappers for long-running evaluation endpoints.

Held-out + LLM-judge eval both take 10–20 min on real workloads (200
gold rows × per-row inference / LLM-judge round-trip). Routing them
through the Jobs framework frees the browser from holding the
connection open and lets the notification bell surface progress.

The runners use an elapsed-time heartbeat rather than a per-row
fraction — neither `run_heldout_evaluation` nor `evaluate_with_llm_judge`
exposes a progress_callback, and instrumenting them would mean
plumbing through the per-row inference + judge loops, which we'd
rather not own here. The heartbeat is the same pattern used by the
synth-playbook + auto-rag/comparison runners.

Idempotency: refuse if there's already a heldout / llm-judge Job for
the same (project, experiment) tuple in QUEUED / RUNNING — two
parallel runs would double the GPU load + race the EvalResult write.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models.job import Job, JobStatus
from app.services.jobs_service import (
    JobProgressHandle,
    start_job,
)


_HEARTBEAT_INTERVAL_SEC = 5.0


async def ensure_no_in_flight_heldout_job(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> None:
    """Raise 409 when a heldout-eval Job for the same (project, experiment)
    is already QUEUED or RUNNING."""
    await _ensure_no_in_flight_job(
        db,
        kind="heldout_evaluation",
        project_id=project_id,
        experiment_id=experiment_id,
        error_code="HELDOUT_EVAL_ALREADY_RUNNING",
        human_label=f"heldout evaluation for experiment #{experiment_id}",
    )


async def ensure_no_in_flight_llm_judge_job(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> None:
    await _ensure_no_in_flight_job(
        db,
        kind="llm_judge_evaluation",
        project_id=project_id,
        experiment_id=experiment_id,
        error_code="LLM_JUDGE_EVAL_ALREADY_RUNNING",
        human_label=f"LLM-judge evaluation for experiment #{experiment_id}",
    )


async def _ensure_no_in_flight_job(
    db: AsyncSession,
    *,
    kind: str,
    project_id: int,
    experiment_id: int,
    error_code: str,
    human_label: str,
) -> None:
    stmt = (
        select(Job)
        .where(
            Job.kind == kind,
            Job.project_id == project_id,
            Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
        .order_by(Job.queued_at.desc())
    )
    result = await db.execute(stmt)
    for candidate in result.scalars().all():
        params = candidate.params or {}
        if int(params.get("experiment_id") or -1) == int(experiment_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": error_code,
                    "message": (
                        f"A {human_label} is already in flight. Watch the "
                        f"notification bell for completion."
                    ),
                    "metadata": {
                        "existing_job_id": candidate.id,
                        "existing_job_status": candidate.status.value,
                        "existing_job_queued_at": (
                            candidate.queued_at.isoformat()
                            if candidate.queued_at
                            else None
                        ),
                    },
                },
            )


async def start_heldout_eval_job(
    db: AsyncSession,
    *,
    kind: str,
    title: str,
    project_id: int,
    run_kwargs: dict[str, Any],
) -> Job:
    """Spawn the Job that wraps ``run_heldout_evaluation``.

    ``kind`` distinguishes the entry point ("heldout_evaluation" for
    /evaluation/run-heldout, "quickstart_baseline_eval" + …) so the
    audit trail shows which surface triggered the run. The runner
    itself is identical.
    """

    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        from app.services.evaluation_service import run_heldout_evaluation

        async def _work() -> dict[str, Any]:
            async with async_session_factory() as runner_db:
                result = await run_heldout_evaluation(
                    db=runner_db,
                    **run_kwargs,
                )
                await runner_db.commit()
                return _summarize_eval_result(result, run_kwargs)

        return await _run_with_heartbeat(
            handle,
            describe=f"Held-out eval (max {run_kwargs.get('max_samples')} rows)",
            work=_work,
        )

    return await start_job(
        db,
        kind=kind,
        title=title,
        runner=_runner,
        project_id=project_id,
        params=dict(run_kwargs),
    )


async def start_llm_judge_eval_job(
    db: AsyncSession,
    *,
    title: str,
    project_id: int,
    experiment_id: int,
    dataset_name: str,
    judge_model: str,
    predictions: list[dict],
) -> Job:
    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        from app.services.evaluation_service import evaluate_with_llm_judge

        async def _work() -> dict[str, Any]:
            async with async_session_factory() as runner_db:
                result = await evaluate_with_llm_judge(
                    runner_db,
                    project_id,
                    experiment_id,
                    dataset_name,
                    judge_model,
                    predictions,
                )
                await runner_db.commit()
                return _summarize_eval_result(
                    result,
                    {
                        "project_id": project_id,
                        "experiment_id": experiment_id,
                        "dataset_name": dataset_name,
                        "judge_model": judge_model,
                        "row_count": len(predictions),
                    },
                )

        return await _run_with_heartbeat(
            handle,
            describe=f"LLM-judge eval ({len(predictions)} rows · {judge_model})",
            work=_work,
        )

    return await start_job(
        db,
        kind="llm_judge_evaluation",
        title=title,
        runner=_runner,
        project_id=project_id,
        params={
            "project_id": project_id,
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "judge_model": judge_model,
            "row_count": len(predictions),
        },
    )


async def _run_with_heartbeat(
    handle: JobProgressHandle,
    *,
    describe: str,
    work,
) -> dict[str, Any]:
    started = time.monotonic()
    stop_heartbeat = asyncio.Event()

    async def _heartbeat() -> None:
        # Emit an "X · Ys elapsed" message every heartbeat-interval
        # until the work coroutine finishes. The first message fires
        # before the timeout so the bell doesn't sit on "queued" for
        # the full interval while we wait for the first tick.
        await handle.set_progress(message=f"{describe} · starting…")
        while not stop_heartbeat.is_set():
            try:
                await asyncio.wait_for(
                    stop_heartbeat.wait(), timeout=_HEARTBEAT_INTERVAL_SEC,
                )
                return
            except asyncio.TimeoutError:
                elapsed = int(time.monotonic() - started)
                await handle.set_progress(
                    message=f"{describe} · {elapsed}s elapsed",
                )

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        payload = await work()
    finally:
        stop_heartbeat.set()
        try:
            await heartbeat_task
        except Exception:  # noqa: BLE001 — heartbeat best-effort
            pass

    elapsed_total = int(time.monotonic() - started)
    await handle.set_progress(
        fraction=1.0,
        message=f"{describe} · finished in {elapsed_total}s",
    )
    return payload


def _summarize_eval_result(result: Any, run_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pointer-only summary for the Job row — keeps the row cheap; the
    full EvalResult is persisted by the service itself."""
    return {
        "experiment_id": run_kwargs.get("experiment_id"),
        "dataset_name": run_kwargs.get("dataset_name"),
        "eval_type": run_kwargs.get("eval_type"),
        "eval_result_id": getattr(result, "id", None),
        "pass_rate": getattr(result, "pass_rate", None),
        "max_samples": run_kwargs.get("max_samples"),
    }

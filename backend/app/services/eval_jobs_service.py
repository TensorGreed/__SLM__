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
``/evaluation/run-heldout`` narrows that to the same *evaluated target*
(``heldout_eval_target``: model path, dataset name, prompt variant), so
"Both, base vs fine-tuned" can queue its base leg next to the fine-tuned
one (UI simplification phase 7b, gap N1: the base leg used to get 409
``HELDOUT_EVAL_ALREADY_RUNNING`` every time). Held-out eval Jobs of one
(project, experiment) still never run at the same time: each waits for
the one queued before it (``_EVAL_CHAINS``), in the order they were
queued.
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
# How often a queued eval re-checks (in the DB) whether the eval ahead of it ended without its runner
# ever running (cancelled while queued, lost): the in-memory signal covers every other case.
_CHAIN_RECHECK_SEC = 5.0


def heldout_eval_target(params: dict[str, Any] | None) -> tuple[str, str, str]:
    """What a held-out eval request evaluates: (model path, dataset name, prompt variant). An empty
    model path is the run's own weights; a trailing "/" on a path is ignored."""
    params = params or {}
    model_path = str(params.get("model_path") or "").strip().rstrip("/")
    dataset_name = str(params.get("dataset_name") or "").strip()
    variant = str(params.get("prompt_variant") or "").strip()
    return model_path, dataset_name, variant


async def ensure_no_in_flight_heldout_job(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    *,
    target: dict[str, Any] | None = None,
) -> None:
    """Raise 409 when a heldout-eval Job for the same (project, experiment) is already QUEUED or
    RUNNING. With ``target`` (the request's ``model_path`` / ``dataset_name`` / ``prompt_variant``),
    only a Job evaluating the same target conflicts: another model or set on the same run queues
    behind it instead (``start_heldout_eval_job`` runs them one at a time)."""
    await _ensure_no_in_flight_job(
        db,
        kind="heldout_evaluation",
        project_id=project_id,
        experiment_id=experiment_id,
        error_code="HELDOUT_EVAL_ALREADY_RUNNING",
        human_label=(
            f"heldout evaluation of the same model, set and prompt format for experiment #{experiment_id}"
            if target is not None else f"heldout evaluation for experiment #{experiment_id}"
        ),
        target=heldout_eval_target(target) if target is not None else None,
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
    target: tuple[str, str, str] | None = None,
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
        if int(params.get("experiment_id") or -1) != int(experiment_id):
            continue
        if target is not None and heldout_eval_target(params) != target:
            continue
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


# One chain of held-out eval Jobs per (event loop, project, experiment): the newest entry is
# ``[job_id | None, done future]``. A new Job waits for the entry it replaced. In memory only: a restart
# fails every live Job anyway (the startup sweep), so nothing is left to wait for.
_EVAL_CHAINS: dict[tuple[int, int, int], list[Any]] = {}


async def _job_is_terminal(job_id: int | None) -> bool:
    if job_id is None:
        return False
    async with async_session_factory() as db:
        row = await db.get(Job, job_id)
        return row is None or row.status not in (JobStatus.QUEUED, JobStatus.RUNNING)


async def _wait_for_previous(handle: JobProgressHandle, previous: list[Any] | None) -> None:
    """Wait until the eval queued before this one (same project + experiment) has ended."""
    if previous is None or previous[1].done():
        return
    ahead = f"eval job #{previous[0]}" if previous[0] is not None else "the eval queued before it"
    await handle.set_progress(message=f"Waiting for {ahead} on the same run to finish (one eval at a time)…")
    while not previous[1].done():
        try:
            await asyncio.wait_for(asyncio.shield(previous[1]), timeout=_CHAIN_RECHECK_SEC)
        except asyncio.TimeoutError:
            # Its runner may never run (cancelled while queued): then its row is terminal.
            if await _job_is_terminal(previous[0]):
                return


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

    Evals of one (project, experiment) run one at a time, in the order they were queued: a Job queued
    while another is in flight waits for it ("Waiting for eval job #N …" on the bell), then runs.
    """
    loop = asyncio.get_running_loop()
    chain_key = (id(loop), int(project_id), int(run_kwargs.get("experiment_id") or -1))
    previous = _EVAL_CHAINS.get(chain_key)
    if previous is not None and previous[1].done():
        previous = None
    entry: list[Any] = [None, loop.create_future()]
    _EVAL_CHAINS[chain_key] = entry

    def _release() -> None:
        if not entry[1].done():
            entry[1].set_result(None)
        if _EVAL_CHAINS.get(chain_key) is entry:
            del _EVAL_CHAINS[chain_key]

    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        try:
            await _wait_for_previous(handle, previous)
            return await _evaluate(handle)
        finally:
            _release()

    async def _evaluate(handle: JobProgressHandle) -> dict[str, Any]:
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
            describe=(
                f"Eval on {run_kwargs.get('dataset_name')} (max {run_kwargs.get('max_samples')} rows)"
                if str(run_kwargs.get("dataset_name") or "").startswith("eval_set:")
                else f"Eval on test examples (max {run_kwargs.get('max_samples')} rows)"
            ),
            work=_work,
        )

    try:
        job = await start_job(
            db,
            kind=kind,
            title=title,
            runner=_runner,
            project_id=project_id,
            params=dict(run_kwargs),
        )
    except BaseException:
        _release()
        raise
    entry[0] = job.id
    return job


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
    full EvalResult is persisted by the service itself. When the judge left
    rows unjudged the summary also carries ``judge_status`` & co.
    (``judge_call_guard.job_result_fields``: the bell's "done, with a warning")."""
    from app.services.judge_call_guard import job_result_fields

    return {
        "experiment_id": run_kwargs.get("experiment_id"),
        "dataset_name": run_kwargs.get("dataset_name"),
        "eval_type": run_kwargs.get("eval_type"),
        "eval_result_id": getattr(result, "id", None),
        "pass_rate": getattr(result, "pass_rate", None),
        "max_samples": run_kwargs.get("max_samples"),
        # An eval-set run: the stored label (set, version, variant) for the bell's outcome line.
        **({"eval_set_label": result.dataset_name}
           if str(getattr(result, "dataset_name", "") or "").startswith("eval_set:") else {}),
        **job_result_fields(result),
    }

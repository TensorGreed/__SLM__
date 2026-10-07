"""Spawn the auto-RAG comparison as a background Job.

Shared by ``POST /auto-rag/comparison/run`` (user-triggered) and the
reroute-to-RAG clone (a RAG sibling created because document passages
beat the fine-tune gets its base + passages check on the test examples
automatically — its "did it help?" number, the way a trained run gets a
lift check). The Job runs ``scripts.auto_rag_ab.run_project_comparison``
on a worker thread and mirrors its per-row progress into the bell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.job import Job, JobStatus


class AutoRagComparisonInFlight(Exception):
    """A comparison Job for the project is already queued / running."""

    def __init__(self, job: Job) -> None:
        super().__init__(f"auto-RAG comparison job {job.id} already in flight")
        self.job = job


def lift_evidence(rows: list[Any], *, score_key: str = "f1") -> dict[str, Any]:
    from app.api.auto_rag import _lift_evidence

    return _lift_evidence(rows, score_key=score_key)


async def start_auto_rag_comparison_job(
    db: AsyncSession,
    project_id: int,
    *,
    recipe_id: str | None,
    base_only: bool,
    corpus: str = "qa",
    split: str = "val",
    sweep_retrieval: bool = False,
) -> Job:
    """Queue the comparison Job (``kind="auto_rag_comparison"``). Raises
    ``AutoRagComparisonInFlight`` instead of racing an existing one.
    ``sweep_retrieval``: try top-3 / top-5 ± reranker, keep the judge's best
    and write it to the project's ``runtime_config.auto_rag_retrieval`` so
    the playground serves it (``apply_retrieval_choice``)."""
    from datetime import datetime, timezone

    from app.services.jobs_service import JobProgressHandle, start_job

    _lift_evidence = lift_evidence
    # Idempotency — refuse if there's already a comparison Job for
    # this project in QUEUED or RUNNING. Two simultaneous runs would
    # race the comparison.json write + double the GPU load.
    in_flight_result = await db.execute(
        select(Job)
        .where(
            Job.kind == "auto_rag_comparison",
            Job.project_id == project_id,
            Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
        .order_by(Job.queued_at.desc())
        .limit(1)
    )
    in_flight = in_flight_result.scalar_one_or_none()
    if in_flight is not None:
        raise AutoRagComparisonInFlight(in_flight)


    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        import asyncio
        import time

        # The comparison work is GPU-heavy + uses sync code paths
        # (torch model load, sqlite3 read inside the script). Run it
        # on a worker thread; bridge the script's sync
        # progress_callback into JobProgressHandle.set_progress via
        # a shared mutable state + a polling drainer running on the
        # event loop.
        progress_state: dict[str, Any] = {
            "scored": 0,
            "total": 0,
            "condition": "without-RAG",
            "passes_done": 0,
        }

        def _sync_callback(scored: int, total: int, condition: str) -> None:
            progress_state["scored"] = scored
            progress_state["total"] = total
            # The without-RAG pass finishes first; once we see scored
            # reset back to 1 on the with-RAG pass, increment
            # passes_done so the overall fraction reflects 2 passes.
            if (
                progress_state["condition"] != condition
                and progress_state["condition"] == "without-RAG"
                and condition == "with-RAG"
            ):
                progress_state["passes_done"] = 1
            progress_state["condition"] = condition

        async def _drainer(stop: asyncio.Event) -> None:
            started = time.monotonic()
            while not stop.is_set():
                state = dict(progress_state)
                total = state["total"]
                scored = state["scored"]
                passes_done = state["passes_done"]
                condition = state["condition"]
                elapsed = int(time.monotonic() - started)
                if total > 0:
                    completed = passes_done * total + scored
                    overall_total = 2 * total
                    fraction = max(0.0, min(1.0, completed / overall_total))
                    msg = (
                        f"scoring row {scored}/{total} ({condition}) · "
                        f"pass {passes_done + 1}/2 · {elapsed}s elapsed"
                    )
                else:
                    fraction = None
                    msg = f"loading model · {elapsed}s elapsed"
                await handle.set_progress(fraction=fraction, message=msg)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass

        # Import inside the runner so a missing torch / peft install
        # (CPU-only dev box) fails inside the Job not at app boot.
        import sys as _sys

        backend_root = str(Path(__file__).resolve().parents[2])
        if backend_root not in _sys.path:
            _sys.path.insert(0, backend_root)
        from scripts.auto_rag_ab import comparison_file_name, run_project_comparison

        stop_event = asyncio.Event()
        drain_task = asyncio.create_task(_drainer(stop_event))
        try:
            payload = await asyncio.to_thread(
                run_project_comparison,
                project_id,
                progress_callback=_sync_callback,
                base_only=base_only,
                corpus=corpus,
                split=split,
                sweep_retrieval=sweep_retrieval,
            )
        finally:
            stop_event.set()
            try:
                await drain_task
            except Exception:  # noqa: BLE001 — drainer is best-effort
                pass

        summary = payload.get("summary") or {}
        retrieval_applied = None
        if sweep_retrieval and isinstance(summary.get("retrieval"), dict):
            retrieval_applied = await apply_retrieval_choice(project_id, summary["retrieval"])
        # Compact row-level evidence for the bell line, so "+29% lift" on a
        # few rows isn't announced without "within noise".
        full_evidence = _lift_evidence(payload.get("rows") or [])
        evidence = {k: full_evidence.get(k) for k in ("verdict", "n", "better", "worse", "same")}
        judge = summary.get("judge") if isinstance(summary.get("judge"), dict) else None
        judge_evidence = None
        if judge:
            full_judge = _lift_evidence(payload.get("rows") or [], score_key="judge")
            judge_evidence = {k: full_judge.get(k) for k in ("verdict", "n", "better", "worse", "same")}
        # Pointers-only result so the Job row stays cheap; the
        # comparison.json on disk is the canonical full payload.
        return {
            "project_id": project_id,
            "model": "base" if base_only else "fine_tuned",
            "corpus": corpus,
            "split": split,
            "retrieval": summary.get("retrieval"),
            "retrieval_sweep": summary.get("retrieval_sweep"),
            "retrieval_applied": retrieval_applied,
            "base_model": payload.get("base_model"),
            "experiment_id": payload.get("experiment_id"),
            "judge": (
                {
                    "judge": judge.get("judge"),
                    "off_score": (judge.get("without_rag") or {}).get("score"),
                    "on_score": (judge.get("with_rag") or {}).get("score"),
                }
                if judge
                else None
            ),
            "judge_evidence": judge_evidence,
            "off_mean_f1": summary.get("off_mean_f1"),
            "on_mean_f1": summary.get("on_mean_f1"),
            "absolute_lift": summary.get("absolute_lift"),
            "relative_lift_pct": summary.get("relative_lift_pct"),
            "evidence": evidence,
            "n_val_rows": summary.get("n_val_rows"),
            "comparison_path": str(
                settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
                / comparison_file_name(base_only=base_only, corpus=corpus)
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    job = await start_job(
        db,
        kind="auto_rag_comparison",
        title=(
            f"Auto-RAG comparison · {'base model' if base_only else 'fine-tuned'}"
            f"{' · document passages' if corpus == 'documents' else ''} · project #{project_id}"
        ),
        runner=_runner,
        project_id=project_id,
        params={
            "project_id": project_id,
            "recipe_id": recipe_id,
            "model": "base" if base_only else "fine_tuned",
            "corpus": corpus,
            "split": split,
            "sweep_retrieval": sweep_retrieval,
        },
    )
    return job


async def apply_retrieval_choice(project_id: int, retrieval: dict[str, Any]) -> dict[str, Any] | None:
    """Write the sweep's chosen ``{"k", "reranker"}`` to the project's
    ``runtime_config.auto_rag_retrieval`` (own session — the Job runner's
    thread work is over). Returns what was written, None on failure."""
    from app.database import async_session_factory
    from app.models.project import Project
    from app.services.auto_rag_service import RETRIEVAL_SETTINGS_KEY
    from app.services.retrieval_reranker import normalize_reranker

    chosen = {"k": int(retrieval.get("k") or 3), "reranker": normalize_reranker(retrieval.get("reranker"))}
    try:
        async with async_session_factory() as db:
            project = await db.get(Project, project_id)
            if project is None:
                return None
            cfg = dict(project.runtime_config or {})
            cfg[RETRIEVAL_SETTINGS_KEY] = {**chosen, "source": "retrieval_sweep"}
            project.runtime_config = cfg
            await db.commit()
        return chosen
    except Exception:  # noqa: BLE001 — the comparison result stands either way
        return None

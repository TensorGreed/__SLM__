"""Generic background-job framework (Hardening Phase H1).

Decouples long-running operations (LLM-driven synth, model cloning,
training, eval comparison) from the HTTP request that starts them.
The user sees the job land in the top-bar notification bell and
keeps working — instead of staring at a spinner that might time out.

Architecture:

  * Endpoints that opt in call ``start_job(...)`` and return 202 +
    the Job id. The framework spawns an ``asyncio.create_task`` that
    runs the supplied coroutine factory in the background. The
    runner owns its own DB session (the HTTP request's session is
    request-scoped and closes when the 202 is sent).

  * The runner receives a ``JobProgressHandle`` which it uses to
    bump ``progress`` / ``progress_message`` as it works. Bumps go
    straight to the DB so the UI's polled ``/jobs/active`` endpoint
    surfaces them on the next poll (3-5s lag).

  * On completion the runner writes ``status=SUCCEEDED`` + ``result``
    (a small JSON pointer dict — not bulk data) OR
    ``status=FAILED`` + ``error``. Both terminal transitions are
    wrapped in a catch-all so a buggy runner never leaves a job
    stuck in RUNNING forever.

  * Cancellation is *cooperative*. ``cancel_job`` sets
    ``status=CANCELLED`` but the runner has to check the flag at
    safe points to actually stop. Phase H1 records the request;
    per-migration cooperative wiring is a follow-on.

Persistence is the load-bearing design choice: training runs go for
30+ minutes and survive a developer-side ``uvicorn`` restart. An
in-memory dict would lose the job state exactly when the user
needs it most.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models.job import Job, JobStatus


_LOG = logging.getLogger("jobs")

# Completed jobs older than this are no longer surfaced by
# ``list_active_jobs`` — keeps the bell's "recently done" section
# from growing without bound. Older jobs remain queryable by id
# for audit / debug.
_RECENTLY_COMPLETED_WINDOW = timedelta(minutes=15)

# Strong references to runner tasks. asyncio.create_task only weakly
# tracks tasks; without an explicit ref the GC can collect a task
# whose only reference was the local variable in start_job, killing
# the runner mid-flight. Drained on completion via the done callback.
_RUNNING_TASKS: set[asyncio.Task] = set()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────
# JobProgressHandle — what the runner uses to publish updates
# ─────────────────────────────────────────────────────────────────────


class JobProgressHandle:
    """The progress-publishing interface handed to runner coroutines.

    Each call opens a fresh DB session so the runner doesn't need to
    keep a session of its own around (long-running runners that
    held a session for the full duration would block other writes).
    """

    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        self._cancelled = False

    async def set_progress(
        self,
        *,
        fraction: float | None = None,
        message: str | None = None,
    ) -> None:
        """Persist a progress update. Accepts a 0.0-1.0 fraction
        and/or a short human-readable message. Either can be omitted
        to update only the other (e.g. "still working, no quantitative
        estimate" → message only)."""
        async with async_session_factory() as db:
            job = await db.get(Job, self.job_id)
            if job is None:
                return
            if fraction is not None:
                job.progress = max(0.0, min(1.0, float(fraction)))
            if message is not None:
                job.progress_message = str(message)[:1024]
            await db.commit()

    async def check_cancelled(self) -> bool:
        """Returns True if a cancellation request landed for this job.
        Runners can call this at safe checkpoints to bail out early.
        Caching is per-handle so we don't hit the DB on every call —
        the runner re-reads when it actually wants to stop."""
        if self._cancelled:
            return True
        async with async_session_factory() as db:
            job = await db.get(Job, self.job_id)
            if job is None:
                return False
            if job.status == JobStatus.CANCELLED:
                self._cancelled = True
                return True
        return False


# Type alias for runner coroutines. They take a JobProgressHandle and
# return a dict (the job's ``result`` payload). Raise any exception to
# signal failure — the framework records the exception text in
# ``error`` and transitions the job to FAILED.
JobRunner = Callable[[JobProgressHandle], Awaitable[dict[str, Any] | None]]


# ─────────────────────────────────────────────────────────────────────
# start_job — the entry-point endpoints call
# ─────────────────────────────────────────────────────────────────────


async def start_job(
    db: AsyncSession,
    *,
    kind: str,
    title: str,
    runner: JobRunner,
    project_id: int | None = None,
    user_id: int | None = None,
    params: dict[str, Any] | None = None,
) -> Job:
    """Create a Job row, schedule its runner on the event loop, and
    return the persisted Job (status=QUEUED). The HTTP handler can
    immediately return ``{job_id: job.id}`` with 202 — the runner
    keeps going in the background.

    The runner is wrapped in ``_runner_wrapper`` which handles the
    QUEUED→RUNNING→SUCCEEDED/FAILED transitions + error capture.
    Never raises — even runner bugs land as ``status=FAILED`` rows.
    """
    job = Job(
        kind=kind,
        title=title,
        status=JobStatus.QUEUED,
        progress=None,
        progress_message=None,
        project_id=project_id,
        user_id=user_id,
        params=dict(params or {}),
    )
    db.add(job)
    await db.flush()
    await db.commit()
    await db.refresh(job)

    job_id = job.id
    # asyncio.create_task ties the runner to the running event loop;
    # uvicorn keeps it alive until completion or process exit. We
    # also hold a strong reference in _RUNNING_TASKS — Python's GC
    # can otherwise collect a task whose only ref was the local
    # variable here, silently killing the runner before it gets CPU.
    task = asyncio.create_task(_runner_wrapper(job_id, runner))
    _RUNNING_TASKS.add(task)
    task.add_done_callback(_RUNNING_TASKS.discard)
    return job


async def _runner_wrapper(job_id: int, runner: JobRunner) -> None:
    """Owns the lifecycle of a single job invocation. Catches every
    exception so a buggy runner can't leave a job stuck in RUNNING."""
    from sqlalchemy import update

    handle = JobProgressHandle(job_id)
    # Atomic QUEUED→RUNNING transition. Conditional WHERE ensures
    # we don't clobber a CANCELLED status that landed between the
    # initial start_job commit and this point (race window is small
    # but real, especially in tests that cancel immediately).
    async with async_session_factory() as db:
        result = await db.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == JobStatus.QUEUED)
            .values(status=JobStatus.RUNNING, started_at=_utcnow())
        )
        await db.commit()
        if result.rowcount == 0:
            # Either job vanished or status moved off QUEUED already
            # (most likely CANCELLED). Honor it.
            row = await db.get(Job, job_id)
            if row is None:
                _LOG.warning("Job %s vanished before runner started", job_id)
                return
            if row.status == JobStatus.CANCELLED:
                row.completed_at = _utcnow()
                await db.commit()
                return
            # Unexpected — log and bail so we don't run twice.
            _LOG.warning(
                "Job %s started in unexpected status %s; not running runner",
                job_id, row.status,
            )
            return

    result_payload: dict[str, Any] | None = None
    error_text: str | None = None
    try:
        result_payload = await runner(handle)
    except asyncio.CancelledError:
        # Honor task cancellation explicitly so we don't mis-record
        # it as a generic failure.
        async with async_session_factory() as db:
            job = await db.get(Job, job_id)
            if job is not None:
                job.status = JobStatus.CANCELLED
                job.completed_at = _utcnow()
                await db.commit()
        raise
    except Exception as exc:  # noqa: BLE001 — boundary
        error_text = f"{type(exc).__name__}: {exc}"
        _LOG.exception("Job %s failed: %s", job_id, error_text)
        # Keep the traceback in the progress_message slot so the UI
        # has something to surface to a power user. Truncated.
        tb = traceback.format_exc()[-1000:]
        async with async_session_factory() as db:
            job = await db.get(Job, job_id)
            if job is not None:
                job.status = JobStatus.FAILED
                job.error = error_text
                job.progress_message = tb
                job.completed_at = _utcnow()
                await db.commit()
        return

    # Success path — write the result + final status. If the runner
    # honored a cancellation it may have already been marked CANCELLED;
    # don't overwrite that.
    async with async_session_factory() as db:
        job = await db.get(Job, job_id)
        if job is None:
            return
        if job.status == JobStatus.CANCELLED:
            return
        job.status = JobStatus.SUCCEEDED
        job.result = dict(result_payload or {})
        job.progress = 1.0
        job.completed_at = _utcnow()
        await db.commit()


# ─────────────────────────────────────────────────────────────────────
# Read-side helpers — what the API endpoints / UI poll
# ─────────────────────────────────────────────────────────────────────


async def reconcile_orphaned_jobs(db: AsyncSession) -> dict[str, int]:
    """Mark jobs left in QUEUED / RUNNING from a previous process
    as FAILED. Called at server startup — those runners are gone
    (the asyncio task didn't survive the restart) and leaving the
    rows in RUNNING means the bell spins forever.

    Returns a small report: ``{queued_swept: N, running_swept: N}``.
    Best-effort; never raises.
    """
    from sqlalchemy import update

    queued = await db.execute(
        update(Job)
        .where(Job.status == JobStatus.QUEUED)
        .values(
            status=JobStatus.FAILED,
            error="lost_during_restart",
            progress_message="The server restarted before this job started running.",
            completed_at=_utcnow(),
        )
    )
    running = await db.execute(
        update(Job)
        .where(Job.status == JobStatus.RUNNING)
        .values(
            status=JobStatus.FAILED,
            error="lost_during_restart",
            progress_message="The server restarted mid-run. Re-trigger the operation from the source surface.",
            completed_at=_utcnow(),
        )
    )
    await db.commit()
    return {
        "queued_swept": int(queued.rowcount or 0),
        "running_swept": int(running.rowcount or 0),
    }


async def get_job(db: AsyncSession, job_id: int) -> Job | None:
    return await db.get(Job, job_id)


async def list_active_jobs(
    db: AsyncSession,
    *,
    project_id: int | None = None,
    include_recently_completed: bool = True,
    limit: int = 50,
) -> list[Job]:
    """Returns jobs in QUEUED / RUNNING + jobs that completed within
    the past ``_RECENTLY_COMPLETED_WINDOW`` (and weren't dismissed).
    What the notification bell renders.

    ``project_id`` scoping is opt-in — callers that want a global
    "show me everything running across all projects" view pass
    ``project_id=None``."""
    cutoff = _utcnow() - _RECENTLY_COMPLETED_WINDOW

    stmt = select(Job)
    if project_id is not None:
        stmt = stmt.where(Job.project_id == project_id)

    in_flight = (Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
    if include_recently_completed:
        recently_done = (
            Job.status.in_([JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED])
            & (Job.completed_at >= cutoff)
            & (Job.dismissed_at.is_(None))
        )
        stmt = stmt.where(in_flight | recently_done)
    else:
        stmt = stmt.where(in_flight)

    stmt = stmt.order_by(Job.queued_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars())


async def dismiss_job(db: AsyncSession, job_id: int) -> Job | None:
    """Mark a completed job as dismissed so it stops appearing in
    the bell. In-flight jobs can't be dismissed (caller should
    cancel instead)."""
    job = await db.get(Job, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        # Dismiss-while-running is a UX foot-gun (looks like cancel,
        # but doesn't stop the work). Refuse and let the caller pick.
        raise ValueError("cannot_dismiss_in_flight_job")
    job.dismissed_at = _utcnow()
    await db.commit()
    return job


async def cancel_job(db: AsyncSession, job_id: int) -> Job | None:
    """Request cancellation. Sets status=CANCELLED so the runner can
    notice on its next ``check_cancelled()`` call. For runners that
    don't cooperate, this is effectively a "stop reporting progress"
    marker — the work still completes server-side. Worth doing
    anyway so the bell stops showing it as in-flight."""
    job = await db.get(Job, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
        # Idempotent: cancelling an already-done job is a no-op.
        return job
    job.status = JobStatus.CANCELLED
    job.completed_at = _utcnow()
    await db.commit()
    return job


# ─────────────────────────────────────────────────────────────────────
# Serialization — keep the API shape in one place
# ─────────────────────────────────────────────────────────────────────


def serialize_job(job: Job) -> dict[str, Any]:
    """The JSON shape the UI reads. Frozen — frontend types in
    ``frontend/src/api/jobs.ts`` mirror this exactly."""
    return {
        "id": job.id,
        "kind": job.kind,
        "title": job.title,
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "progress": job.progress,
        "progress_message": job.progress_message,
        "project_id": job.project_id,
        "user_id": job.user_id,
        "params": dict(job.params or {}),
        "result": dict(job.result or {}) if job.result else None,
        "error": job.error,
        "queued_at": job.queued_at.isoformat() if job.queued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "dismissed_at": job.dismissed_at.isoformat() if job.dismissed_at else None,
    }

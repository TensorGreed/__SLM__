"""Jobs API routes (Hardening Phase H1).

Read-only surface that the frontend's notification bell polls.
Mutations (start, cancel, dismiss) live here too so the UI doesn't
need to talk to two routers.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.jobs_service import (
    cancel_job,
    dismiss_job,
    get_job,
    list_active_jobs,
    serialize_job,
    serialize_job_with_live_metrics,
)


router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/active")
async def get_active_jobs(
    project_id: int | None = None,
    include_recently_completed: bool = True,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List in-flight + recently-completed jobs.

    Frontend's notification bell polls this every 3-5 seconds while
    the user has any tab open. Optionally scope by ``project_id``
    when the bell is shown on a project page; pass ``project_id=null``
    (omit) for the global "all my jobs" view.
    """
    jobs = await list_active_jobs(
        db,
        project_id=project_id,
        include_recently_completed=include_recently_completed,
        limit=max(1, min(limit, 200)),
    )
    return {
        "count": len(jobs),
        # Bell consumers get the live-metrics enrichment so the
        # sparkline can render during a training run.
        "jobs": [serialize_job_with_live_metrics(j) for j in jobs],
    }


@router.get("/{job_id}")
async def get_job_by_id(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Fetch a single job by id. Useful for the "Open" deep-link
    from a notification when the bell payload is stale."""
    job = await get_job(db, job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return serialize_job_with_live_metrics(job)


@router.post("/{job_id}/dismiss")
async def dismiss(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Hide a completed job from the notification bell.
    In-flight jobs can't be dismissed — caller should cancel."""
    try:
        job = await dismiss_job(db, job_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return serialize_job(job)


@router.post("/{job_id}/cancel")
async def cancel(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Request job cancellation. Cooperative — runner has to honor
    it at its next checkpoint. For runners that don't yet
    cooperate, this is a "stop showing me" flag (the work
    still completes server-side)."""
    job = await cancel_job(db, job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return serialize_job(job)

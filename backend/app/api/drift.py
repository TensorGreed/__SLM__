"""Project-scoped drift API (E4).

Two surfaces:
  * Manual trap refresh — ``POST /api/projects/{id}/drift/refresh-traps``
    spins fresh hallucination traps via the trap-refresh service. Honors
    ``?count=`` override + ``?simulate=true`` for dev where no LLM
    credentials are wired.
  * Queue triage — list pending rows, accept (→ append to gold_test),
    reject (→ mark with a note).

Distinct from the per-deployment ``/deployments/{id}/drift/check``
surface, which evaluates a live endpoint's pass-rate; this one is
project-scoped and writes into the review queue rather than the
DeploymentDriftCheck table.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.gold_drift_review_queue import GoldDriftQueueStatus
from app.models.project import Project


router = APIRouter(
    prefix="/projects/{project_id}/drift",
    tags=["Drift"],
)


class RefreshTrapsResponse(BaseModel):
    project_id: int
    generated: int
    clusters_targeted: list[str]
    simulated: bool
    row_ids: list[int]


@router.post("/refresh-traps", status_code=201, response_model=RefreshTrapsResponse)
async def refresh_traps(
    project_id: int,
    count: int | None = Query(None, ge=1, le=20),
    simulate: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    """Generate fresh hallucination traps targeting recent failure
    clusters; persists them to the drift review queue.

    Query params:
      * ``count`` — override the per-project default (1-20).
      * ``simulate`` — bypass the LLM and use deterministic placeholder
        traps. Useful for dev + tests where no API key is wired.

    Errors:
      * 404 — project not found.
      * 400 — project has no recipe selected.
    """
    from app.services.drift_trap_refresh_service import refresh_traps_for_project

    try:
        return await refresh_traps_for_project(
            db,
            project_id=project_id,
            count=count,
            simulate=simulate,
        )
    except ValueError as exc:
        code = str(exc)
        if code == "project_not_found":
            raise HTTPException(404, code)
        raise HTTPException(400, code)


@router.get("/review-queue")
async def list_drift_review_queue(
    project_id: int,
    status: GoldDriftQueueStatus | None = Query(GoldDriftQueueStatus.PENDING),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Newest-first list of drift-queue rows for the project. Filters
    by ``status`` (default ``PENDING``). Pass ``status=`` (empty) to
    see every row including triaged ones for audit."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "project_not_found")

    from app.services.drift_trap_refresh_service import list_review_queue

    rows = await list_review_queue(
        db,
        project_id=project_id,
        status=status,
        limit=limit,
    )
    return {"project_id": project_id, "rows": rows}


class TriageRequest(BaseModel):
    accept: bool = Field(...)
    note: str | None = Field(default=None, max_length=2000)


@router.post("/review-queue/{row_id}/triage")
async def triage_drift_queue_row(
    project_id: int,
    row_id: int,
    payload: TriageRequest,
    db: AsyncSession = Depends(get_db),
):
    """Accept / reject one drift-queue row. Accepting appends the
    row to the project's gold_test JSONL; rejecting leaves the row
    in place with the user's reason captured.

    Errors:
      * 404 — row not found in this project.
      * 409 — row already triaged (no re-triage; rejected rows stay
        rejected, accepted rows stay accepted).
    """
    from app.services.drift_trap_refresh_service import triage_queue_row

    try:
        row = await triage_queue_row(
            db,
            project_id=project_id,
            row_id=row_id,
            accept=payload.accept,
            note=payload.note,
        )
    except ValueError as exc:
        code = str(exc)
        if code == "queue_row_not_found":
            raise HTTPException(404, code)
        if code == "queue_row_already_triaged":
            raise HTTPException(409, code)
        raise HTTPException(400, code)

    return {
        "id": row.id,
        "status": row.status.value,
        "triaged_at": row.triaged_at.isoformat() if row.triaged_at else None,
        "triage_note": row.triage_note,
    }

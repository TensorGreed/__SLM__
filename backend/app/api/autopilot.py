"""Autopilot API — decision-log query + rollback surface.

This router owns the `/autopilot/*` path space. Priorities covered:

- P1 — decision-log reads (`GET /autopilot/decisions`, run-scoped reads).
- P2 — snapshot + rollback (`GET /autopilot/snapshots`,
  `POST /autopilot/rollback/{decision_id}` and its `/preview` sibling).

Follow-up priorities P3 (repair preview/apply) and P4 (CLI parity) will add
more endpoints here so the autopilot surface lives in one file.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.autopilot_decision_service import (
    get_run_decisions,
    list_decisions,
)
from app.services.autopilot_snapshot_service import (
    get_snapshot_for_decision,
    list_snapshots,
    preview_rollback,
    purge_expired_snapshots,
    rollback_decision,
)


_ROLLBACK_FAILURE_STATUS: dict[str, int] = {
    "decision_not_found": 404,
    "no_snapshot": 409,
    "already_rolled_back": 409,
    "snapshot_expired": 410,
}


class AutopilotRollbackRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=1024)
    actor: Optional[str] = Field(default=None, max_length=64)


router = APIRouter(prefix="/autopilot", tags=["Autopilot"])


@router.get("/decisions")
async def list_autopilot_decisions(
    project_id: Optional[int] = Query(default=None, description="Filter by project."),
    run_id: Optional[str] = Query(default=None, description="Filter by autopilot run id."),
    stage: Optional[str] = Query(default=None, description="Filter by pipeline stage / step."),
    status: Optional[str] = Query(default=None, description="Filter by decision status."),
    action: Optional[str] = Query(default=None, description="Filter by derived action."),
    reason_code: Optional[str] = Query(default=None, description="Filter by reason code."),
    since: Optional[datetime] = Query(default=None, description="Created at >= (ISO-8601)."),
    until: Optional[datetime] = Query(default=None, description="Created at <= (ISO-8601)."),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List autopilot decision-log entries with optional filters."""
    return await list_decisions(
        db,
        project_id=project_id,
        run_id=run_id,
        stage=stage,
        status=status,
        action=action,
        reason_code=reason_code,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )


@router.get("/runs/{run_id}/decisions")
async def list_decisions_for_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return every decision-log entry for one autopilot run, ordered by sequence."""
    payload = await get_run_decisions(db, run_id)
    if not payload.get("items"):
        raise HTTPException(404, f"No autopilot decisions recorded for run_id {run_id!r}.")
    return payload


@router.get("/snapshots")
async def list_autopilot_snapshots(
    project_id: Optional[int] = Query(default=None, description="Filter by project."),
    run_id: Optional[str] = Query(default=None, description="Filter by autopilot run id."),
    include_restored: bool = Query(default=True, description="Include already-rolled-back snapshots."),
    include_expired: bool = Query(default=True, description="Include expired snapshots."),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List captured autopilot snapshots (rollback candidates)."""
    return await list_snapshots(
        db,
        project_id=project_id,
        run_id=run_id,
        include_restored=include_restored,
        include_expired=include_expired,
        limit=limit,
        offset=offset,
    )


@router.get("/decisions/{decision_id}/snapshot")
async def get_decision_snapshot(
    decision_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the snapshot associated with a decision, or 404 if none exists."""
    snapshot = await get_snapshot_for_decision(db, decision_id)
    if snapshot is None:
        raise HTTPException(
            404, f"No snapshot recorded for decision {decision_id}."
        )
    return snapshot


@router.post("/rollback/{decision_id}/preview")
async def preview_autopilot_rollback(
    decision_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a non-mutating preview of what rolling back `decision_id` would do."""
    return await preview_rollback(db, decision_id)


@router.post("/rollback/{decision_id}")
async def execute_autopilot_rollback(
    decision_id: int,
    req: Optional[AutopilotRollbackRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    """Roll back the given autopilot decision if a valid snapshot exists.

    Returns a structured error response with HTTP 404/409/410 when the decision
    is not reversible (see `_ROLLBACK_FAILURE_STATUS` for the mapping). On
    success, returns the updated snapshot, the new rollback-decision row, and
    per-step outcomes so the caller can render a confirmation UI.
    """
    actor = (req.actor if req and req.actor else "api").strip() or "api"
    reason = (req.reason if req else None)

    result = await rollback_decision(
        db,
        decision_id,
        actor=actor,
        reason=reason,
    )

    if not result.get("ok"):
        status_code = _ROLLBACK_FAILURE_STATUS.get(
            str(result.get("reason") or ""), 409
        )
        raise HTTPException(status_code=status_code, detail=result)

    return result


@router.post("/snapshots/purge-expired")
async def purge_expired_autopilot_snapshots():
    """Admin operation: delete snapshots past their TTL.

    Returns the number of rows removed. Intended for cron/scheduled use; safe
    to call ad-hoc because purge logic swallows its own errors.
    """
    removed = await purge_expired_snapshots()
    return {"removed": int(removed)}

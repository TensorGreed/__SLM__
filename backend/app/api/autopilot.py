"""Autopilot API — decision-log query surface.

This router owns the `/autopilot/*` path space. Today it only exposes the
decision-log reads introduced by priority.md P1; follow-up priorities P2
(rollback), P3 (repair preview/apply), and P4 (CLI parity) will add more
endpoints here so the autopilot surface lives in one file.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.autopilot_decision_service import (
    get_run_decisions,
    list_decisions,
)

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

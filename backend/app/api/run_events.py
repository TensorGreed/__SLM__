"""Run-events API (priority.md P31, Wave G).

Read-only HTTP surface over the canonical ``run_events`` table:

- ``GET /api/projects/{id}/run-events`` — project-scoped list, with
  filters for ``run_id`` / ``parent_run_id`` / ``stage`` / ``severity``
  / ``since`` / ``until``. Ordered newest-first.
- ``GET /api/run-events/run/{run_id}`` — every event for a single
  ``run_id`` ordered oldest-first; used for per-experiment timeline
  drill-in surfaces.

Stable reason codes mapped to HTTP:
- ``project_not_found`` (404)
- ``invalid_stage`` / ``invalid_severity`` / ``invalid_window`` (400)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.run_event_service import (
    list_run_events,
    list_run_events_for_run,
)


project_router = APIRouter(
    prefix="/projects/{project_id}/run-events", tags=["RunEvents"]
)
router = APIRouter(prefix="/run-events", tags=["RunEvents"])


_NOT_FOUND_CODES = {"project_not_found"}


def _raise_for(exc: ValueError) -> None:
    detail = str(exc) or "run_event_error"
    head = detail.split(":", 1)[0]
    if head in _NOT_FOUND_CODES:
        raise HTTPException(404, detail=detail) from exc
    raise HTTPException(400, detail=detail) from exc


@project_router.get("")
async def list_for_project(
    project_id: int,
    run_id: str | None = Query(default=None),
    parent_run_id: str | None = Query(default=None),
    stage: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    since: str | None = Query(default=None),
    until: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await list_run_events(
            db,
            project_id=project_id,
            run_id=run_id,
            parent_run_id=parent_run_id,
            stage=stage,
            severity=severity,
            since=since,
            until=until,
            limit=limit,
        )
    except ValueError as exc:
        _raise_for(exc)


@router.get("/run/{run_id}")
async def list_for_run(
    run_id: str,
    limit: int = Query(default=200, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    return await list_run_events_for_run(db, run_id=run_id, limit=limit)

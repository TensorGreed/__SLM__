"""Project-wide failure-cluster API (priority.md P33, Wave G).

Sits at ``/api/projects/{id}/failure-clusters`` — distinct from the P12
per-eval-result clusters at
``/api/projects/{id}/evaluation/{eval_result_id}/failure-clusters``.
Both surfaces ship the same noun ("failure clusters") but at different
abstraction levels: P12 clusters per-row prediction failures inside an
eval run; P33 clusters cross-stage RunEvent failures across the whole
project.

Routes:

- ``GET /api/projects/{id}/failure-clusters`` — list persisted
  clusters, optionally filtered by ``stage`` / ``reason_code``,
  ordered by failure_count DESC.
- ``POST /api/projects/{id}/failure-clusters/recompute`` — recompute
  from the RunEvent log; idempotent. Optional ``since`` / ``until``
  body fields scope the recompute window.

Stable reason codes:
- ``project_not_found`` (404)
- ``invalid_window`` (400)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.run_event_clustering_service import (
    compute_failure_clusters,
    list_failure_clusters,
)


project_router = APIRouter(
    prefix="/projects/{project_id}/failure-clusters",
    tags=["FailureClusters"],
)


_NOT_FOUND_CODES = {"project_not_found"}


def _raise_for(exc: ValueError) -> None:
    detail = str(exc) or "failure_cluster_error"
    head = detail.split(":", 1)[0]
    if head in _NOT_FOUND_CODES:
        raise HTTPException(404, detail=detail) from exc
    raise HTTPException(400, detail=detail) from exc


class RecomputeRequest(BaseModel):
    since: str | None = Field(default=None, max_length=64)
    until: str | None = Field(default=None, max_length=64)


@project_router.get("")
async def list_for_project(
    project_id: int,
    stage: str | None = Query(default=None),
    reason_code: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await list_failure_clusters(
            db,
            project_id=project_id,
            stage=stage,
            reason_code=reason_code,
            limit=limit,
        )
    except ValueError as exc:
        _raise_for(exc)


@project_router.post("/recompute")
async def recompute_for_project(
    project_id: int,
    req: RecomputeRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    payload = req or RecomputeRequest()
    try:
        return await compute_failure_clusters(
            db,
            project_id=project_id,
            since=payload.since,
            until=payload.until,
        )
    except ValueError as exc:
        _raise_for(exc)

"""Remediation event tracking API (E2).

Project-scoped POST to record user clicks on suggested-action buttons
(forecast panel + failure-cluster cards). The companion admin
aggregation endpoint lives in ``api/admin.py``.

The client sends ``{kind, params, outcome}``; the server hashes the
(kind, params) pair via ``remediation_tracking_service.compute_params_hash``
so re-clicks of the same suggested fix collapse for aggregation
without storing the full params blob.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.models.remediation_action_event import RemediationOutcome


router = APIRouter(
    prefix="/projects/{project_id}/remediation",
    tags=["Remediation"],
)


class RecordEventRequest(BaseModel):
    kind: str = Field(..., min_length=1, max_length=64)
    params: Any = None
    outcome: RemediationOutcome = RemediationOutcome.CLICKED


class RecordEventResponse(BaseModel):
    id: int
    project_id: int
    action_kind: str
    params_hash: str
    outcome: RemediationOutcome
    observed_at: str


@router.post("/events", status_code=201)
async def record_event(
    project_id: int,
    payload: RecordEventRequest,
    db: AsyncSession = Depends(get_db),
):
    """Record one remediation-action click. Returns the persisted
    event so the client can verify the hash + observed_at if needed."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    from app.services.remediation_tracking_service import record_action_event

    event = await record_action_event(
        db,
        project_id=project_id,
        kind=payload.kind,
        params=payload.params,
        outcome=payload.outcome,
    )
    return RecordEventResponse(
        id=event.id,
        project_id=event.project_id,
        action_kind=event.action_kind,
        params_hash=event.params_hash,
        outcome=event.outcome,
        observed_at=event.observed_at.isoformat(),
    )

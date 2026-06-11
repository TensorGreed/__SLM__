"""Training Config Gap API — Coach-stage-2 phase 1.

Single read-only endpoint that returns the same panel-friendly payload
shape as ``/data-health`` so the frontend can reuse rendering. Mounts at
``/api/projects/{project_id}/training-config-gaps``.

Phase 2 will add ``POST /apply-patch`` (preview + apply) once we know
which signals deserve a one-click remediation button.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.training_config_gap_service import scan_training_config_gaps


router = APIRouter(
    prefix="/projects/{project_id}/training-config-gaps",
    tags=["Training Config Gaps"],
)


@router.get("")
async def get_training_config_gaps(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the training-config gap report for a project."""
    try:
        return await scan_training_config_gaps(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

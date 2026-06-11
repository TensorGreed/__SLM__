"""Training Config Gap API — Coach-stage-2 phases 1 + 2.

Mounts at ``/api/projects/{project_id}/training-config-gaps``:

Phase 1 (read-only):
- ``GET /`` — aggregated gap report (same panel shape as data-health).

Phase 2 (preview → apply):
- ``GET /overrides`` — the current persisted overrides dict, for the
  TrainingPanel mount-time prefill.
- ``POST /patch/preview`` — return the before → after diff for a patch,
  without mutating anything.
- ``POST /patch/apply`` — persist the patch under
  ``project.runtime_config["training_config_overrides"]``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services.training_config_gap_service import (
    apply_patch,
    preview_patch,
    read_overrides,
    scan_training_config_gaps,
)


router = APIRouter(
    prefix="/projects/{project_id}/training-config-gaps",
    tags=["Training Config Gaps"],
)


class PatchRequest(BaseModel):
    """Body for the preview + apply endpoints."""
    signal_id: str = Field(..., min_length=1)


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


@router.get("/overrides")
async def get_training_config_overrides(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the persisted ``training_config_overrides`` block. Used by
    TrainingPanel on mount to prefill the form so the visible config
    matches the gap scanner's effective config.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return {
        "project_id": int(project_id),
        "overrides": read_overrides(project),
    }


@router.post("/patch/preview")
async def post_patch_preview(
    project_id: int,
    req: PatchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Return the before → after diff a patch *would* produce, without
    mutating anything. Unknown signal_id / signals without an
    apply_patch_kind return 400; missing project returns 404.
    """
    try:
        return await preview_patch(db, project_id, req.signal_id)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("Project"):
            raise HTTPException(404, msg)
        raise HTTPException(400, msg)


@router.post("/patch/apply")
async def post_patch_apply(
    project_id: int,
    req: PatchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist the patch onto the project's runtime_config and return
    the updated overrides. Idempotent — re-applying writes the same
    value.
    """
    try:
        result = await apply_patch(db, project_id, req.signal_id)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("Project"):
            raise HTTPException(404, msg)
        raise HTTPException(400, msg)
    await db.commit()
    return result

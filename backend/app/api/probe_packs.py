"""Probe Pack API — Coach-stage-2 phase 8.

Mounts at ``/api/projects/{project_id}/probe-pack``:

- ``GET /`` — the platform-authored, recipe-keyed adversarial probe pack
  for the project (the held-out ruler the user did not author). Returns
  an ``applicable=False`` payload when the project has no recipe or no
  pack exists for its task shape yet.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.probe_pack_service import get_probe_pack_for_project

router = APIRouter(prefix="/projects/{project_id}/probe-pack", tags=["Probe Pack"])


@router.get("")
async def get_probe_pack(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the platform-authored probe pack for a project's recipe."""
    try:
        return await get_probe_pack_for_project(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

"""Eval Gap API — Coach-stage-2 phase 3.

Mounts at ``/api/projects/{project_id}/eval-gaps``. Read-only in
phase 3 — every signal's ``suggested_action`` is a navigate pointer.
Phase 4 may add patch actions for the eval-side gaps that have an
unambiguous one-click fix (e.g. snapshot last green checkpoint as the
regression baseline).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.eval_gap_service import scan_eval_gaps


router = APIRouter(
    prefix="/projects/{project_id}/eval-gaps",
    tags=["Eval Gaps"],
)


@router.get("")
async def get_eval_gaps(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the eval-side gap report for a project."""
    try:
        return await scan_eval_gaps(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

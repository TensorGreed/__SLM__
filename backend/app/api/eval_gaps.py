"""Eval Gap API — Coach-stage-2 phases 3 + 5.

Mounts at ``/api/projects/{project_id}/eval-gaps``.

Phase 3 (read-only):
- ``GET /`` — aggregated gap report (same panel shape as data-health).

Phase 5 (preview → apply for the eval-side patches):
- ``POST /patch/preview`` — return the would-change diff for a patch
  without mutating anything.
- ``POST /patch/apply`` — apply the patch. Two patches today:
  * ``regression_baseline_promote_last_green`` — promote the best
    Checkpoint of the most recent green-passing run as the baseline.
  * ``label_kl_rebalance_eval`` — trim GOLD_DEV to match train label
    distribution (GOLD_TEST is never touched).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.eval_gap_service import (
    apply_patch,
    preview_patch,
    scan_eval_gaps,
)


router = APIRouter(
    prefix="/projects/{project_id}/eval-gaps",
    tags=["Eval Gaps"],
)


class PatchRequest(BaseModel):
    signal_id: str = Field(..., min_length=1)


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


@router.post("/patch/preview")
async def post_eval_patch_preview(
    project_id: int,
    req: PatchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Return the before → after diff for an eval-gap patch. Unknown
    signal_id / signals without a patch return 400; missing project
    returns 404."""
    try:
        return await preview_patch(db, project_id, req.signal_id)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("Project"):
            raise HTTPException(404, msg)
        raise HTTPException(400, msg)


@router.post("/patch/apply")
async def post_eval_patch_apply(
    project_id: int,
    req: PatchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Apply the patch. The mutation surface depends on the patch:
    ``regression_baseline_promote_last_green`` writes
    ``Checkpoint.promoted_at``; ``label_kl_rebalance_eval`` rewrites
    the GOLD_DEV JSONL file in place."""
    try:
        result = await apply_patch(db, project_id, req.signal_id)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("Project"):
            raise HTTPException(404, msg)
        raise HTTPException(400, msg)
    await db.commit()
    return result

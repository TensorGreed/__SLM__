"""Data Health Report API — D1/D3/D4 of the data-quality arc.

Mounts at ``/api/projects/{project_id}/data-health``:

- ``GET /`` — aggregated Data Health Report (D1+D2).
- ``POST /autofix/preview`` — return per-item diff for a fix (D3.2/D4).
- ``POST /autofix`` — apply the fix after the user confirms the preview.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.data_health_autofix_service import (
    SUPPORTED_FIX_KINDS,
    apply_autofix,
    preview_autofix,
)
from app.services.data_health_service import compute_data_health_report


router = APIRouter(prefix="/projects/{project_id}/data-health", tags=["Data Health"])


class AutofixRequest(BaseModel):
    """Body for ``POST /data-health/autofix``. D3 only takes ``fix_kind``
    (no per-fix params yet — every supported fix runs across the
    project's whole data corpus). D4 will add per-fix params (e.g.
    ``max_seq_length`` for the truncation fix)."""
    fix_kind: str = Field(..., min_length=1)


@router.get("")
async def get_data_health_report(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the aggregated Data Health Report for a project."""
    try:
        return await compute_data_health_report(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/autofix/preview")
async def post_data_health_autofix_preview(
    project_id: int,
    req: AutofixRequest,
    db: AsyncSession = Depends(get_db),
):
    """Return the per-item diff a fix *would* produce, without
    mutating anything. The UI renders this so the user sees the
    exact rows that will change before clicking Apply. Unknown
    ``fix_kind`` returns 400; safety-blocked previews (e.g. PII on a
    span-extraction recipe) return 200 with ``safe_to_apply=False``
    so the frontend can show the explanation rather than a generic
    error."""
    try:
        return await preview_autofix(db, project_id, req.fix_kind)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("Project"):
            raise HTTPException(404, msg)
        raise HTTPException(400, msg)


@router.post("/autofix")
async def post_data_health_autofix(
    project_id: int,
    req: AutofixRequest,
    db: AsyncSession = Depends(get_db),
):
    """Apply a safe auto-fix transform. Each fix is idempotent — calling
    again after applying returns ``applied_count=0`` rather than
    double-applying. Unknown ``fix_kind`` returns 400. The UI is
    expected to call ``/autofix/preview`` first; calling this
    endpoint directly is allowed for scripted use but the
    preview-then-apply contract is what protects users."""
    try:
        result = await apply_autofix(db, project_id, req.fix_kind)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("Project"):
            raise HTTPException(404, msg)
        raise HTTPException(400, msg)
    await db.commit()
    return result


@router.get("/autofix/supported")
def get_supported_autofixes(project_id: int):  # noqa: ARG001
    """List the fix kinds D3 supports. The panel calls this to decide
    which signals get an Auto-fix button."""
    return {"fix_kinds": list(SUPPORTED_FIX_KINDS)}

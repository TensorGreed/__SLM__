"""Data Studio API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.data_studio_service import build_data_studio_overview


router = APIRouter(prefix="/projects/{project_id}/data-studio", tags=["Data Studio"])


@router.get("/overview")
async def get_data_studio_overview(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the additive Data Studio readiness overview."""

    try:
        return await build_data_studio_overview(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

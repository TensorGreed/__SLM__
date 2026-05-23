"""Coach Mode API routes (USER-SUCCESS Epic 4 Phase 1)."""

from typing import get_args

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.coach_service import CoachStage, suggest_for_stage

router = APIRouter(prefix="/projects/{project_id}/coach", tags=["Coach"])


@router.get("/{stage}")
async def get_coach_suggestions(
    project_id: int,
    stage: str,
    db: AsyncSession = Depends(get_db),
):
    """Return top suggestions for one workflow stage.

    ``stage`` must be one of the values declared in
    ``CoachStage``. Phase 1 only handles ``"data"``; other stages
    resolve to an empty list with ``handler_available=False`` so the
    UI can mount the strip ahead of the backend rollout.
    """
    valid_stages = set(get_args(CoachStage))
    if stage not in valid_stages:
        raise HTTPException(
            400,
            f"Unknown stage {stage!r}. Valid stages: {sorted(valid_stages)}",
        )
    try:
        return await suggest_for_stage(db, project_id, stage)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(404, str(exc))

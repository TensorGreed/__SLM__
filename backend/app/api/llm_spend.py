"""Project cloud-LLM spend tally (``llm_spend_service``)."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services import llm_spend_service

router = APIRouter(prefix="/projects/{project_id}/llm-spend", tags=["LLM spend"])


@router.get("")
async def get_llm_spend(
    project_id: int,
    days: int = Query(llm_spend_service.DEFAULT_DAYS, ge=1, le=llm_spend_service.MAX_DAYS),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Tokens and list-price estimate of the project's cloud-LLM calls over
    the last ``days``, by model and by operation kind."""
    if await db.get(Project, project_id) is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return await llm_spend_service.project_llm_spend(db, project_id, days)

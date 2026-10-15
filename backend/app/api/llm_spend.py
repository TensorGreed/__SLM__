"""Project cloud-LLM spend tally + monthly budget (``llm_spend_service``)."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services import llm_spend_service

router = APIRouter(prefix="/projects/{project_id}/llm-spend", tags=["LLM spend"])
budget_router = APIRouter(prefix="/projects/{project_id}/llm-budget", tags=["LLM spend"])


@router.get("")
async def get_llm_spend(
    project_id: int,
    days: int = Query(llm_spend_service.DEFAULT_DAYS, ge=1, le=llm_spend_service.MAX_DAYS),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Tokens and list-price estimate of the project's cloud-LLM calls over
    the last ``days``, by model and by operation kind, plus the calendar
    month-to-date total and the budget status."""
    if await db.get(Project, project_id) is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return await llm_spend_service.project_llm_spend(db, project_id, days)


class LlmBudgetUpdate(BaseModel):
    """``monthly`` in US dollars (null clears the budget); ``warn_at`` the
    month-to-date share that turns the status to ``warning`` (default 0.8)."""

    monthly: float | None = None
    warn_at: float | None = None


async def _budget_payload(db: AsyncSession, project: Project) -> dict[str, Any]:
    return {
        "project_id": project.id,
        "budget": llm_spend_service.read_budget(project),
        "status": await llm_spend_service.month_budget_status(db, project),
    }


@budget_router.get("")
async def get_llm_budget(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """The project's monthly cloud-LLM budget and this month's status."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return await _budget_payload(db, project)


@budget_router.put("")
async def put_llm_budget(project_id: int, body: LlmBudgetUpdate, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Set (or clear, with ``monthly: null``) the monthly budget. Advisory
    only — nothing is ever blocked by it."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    try:
        budget = llm_spend_service.normalize_budget(body.monthly, body.warn_at)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    cfg = dict(project.runtime_config or {})
    if budget is None:
        cfg.pop(llm_spend_service.BUDGET_KEY, None)
    else:
        cfg[llm_spend_service.BUDGET_KEY] = budget
    project.runtime_config = cfg
    await db.commit()
    await db.refresh(project)
    return await _budget_payload(db, project)

"""Preference pairs built from evaluation results."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services.preference_pairs_from_eval_service import PreferencePairsError, preference_pairs_from_eval

router = APIRouter(prefix="/projects/{project_id}/alignment", tags=["Alignment"])


class PairsFromEvalRequest(BaseModel):
    eval_result_id: int = Field(..., ge=1)
    include: list[Literal["wrong", "partial"]] = Field(default_factory=lambda: ["wrong", "partial"], min_length=1)
    dataset_name: str | None = Field(default=None, max_length=255)
    # Pairs from the held-out test split teach the model its own test rows; refused unless set.
    allow_test_split: bool = False


@router.post("/preference-pairs/from-eval")
async def build_preference_pairs_from_eval(
    project_id: int, req: PairsFromEvalRequest, db: AsyncSession = Depends(get_db)
):
    """Wrong / partial answers of an eval -> ``{prompt, chosen: answer key, rejected: model answer}``
    pairs. 400 ``no_pairs_built`` (with ``metadata.skipped``, the count per skip reason) when nothing was built;
    400 ``test_split_refused`` for an eval on the test split unless ``allow_test_split`` is true."""
    if await db.get(Project, project_id) is None:
        raise HTTPException(404, f"Project {project_id} not found")
    try:
        return await preference_pairs_from_eval(
            db, project_id, req.eval_result_id, tuple(req.include), req.dataset_name, req.allow_test_split
        )
    except PreferencePairsError as exc:
        status = 404 if exc.code == "eval_result_not_found" else 400
        raise HTTPException(status, {
            "error_code": exc.code, "message": str(exc),
            "metadata": {"skipped": exc.skipped, "eval_result_id": req.eval_result_id},
        })

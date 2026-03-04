"""Evaluation API routes."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.evaluation import EvalResultResponse, LLMJudgeRequest
from app.services.evaluation_service import (
    generate_safety_scorecard,
    get_eval_results,
    run_evaluation,
    evaluate_with_llm_judge,
)

router = APIRouter(prefix="/projects/{project_id}/evaluation", tags=["Evaluation"])


class EvalRunRequest(BaseModel):
    experiment_id: int
    dataset_name: str = Field(..., min_length=1, max_length=255)
    eval_type: Literal["exact_match", "f1", "safety"]
    predictions: list[dict]


@router.post("/run", status_code=201)
async def run_eval(
    project_id: int,
    req: EvalRunRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run an evaluation."""
    try:
        result = await run_evaluation(
            db, project_id, req.experiment_id, req.dataset_name, req.eval_type, req.predictions,
        )
        return EvalResultResponse.model_validate(result)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/llm-judge", status_code=201)
async def run_llm_judge_eval(
    project_id: int,
    req: LLMJudgeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run LLM-as-a-judge evaluation."""
    try:
        result = await evaluate_with_llm_judge(
            db,
            project_id,
            req.experiment_id,
            req.dataset_name,
            req.judge_model,
            [p.model_dump() for p in req.predictions],
        )
        return EvalResultResponse.model_validate(result)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/results/{experiment_id}", response_model=list[EvalResultResponse])
async def get_results(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get evaluation results for an experiment."""
    try:
        results = await get_eval_results(db, project_id, experiment_id)
        return [EvalResultResponse.model_validate(r) for r in results]
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/safety-scorecard/{experiment_id}")
async def safety_scorecard(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Generate safety scorecard for an experiment."""
    try:
        return await generate_safety_scorecard(db, project_id, experiment_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

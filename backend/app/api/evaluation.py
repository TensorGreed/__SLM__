"""Evaluation API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.evaluation import EvalResultResponse, SafetyScorecardResponse, LLMJudgeRequest
from app.services.evaluation_service import (
    generate_safety_scorecard,
    get_eval_results,
    run_evaluation,
    evaluate_with_llm_judge,
)

router = APIRouter(prefix="/projects/{project_id}/evaluation", tags=["Evaluation"])


class EvalRunRequest(BaseModel):
    experiment_id: int
    dataset_name: str
    eval_type: str
    predictions: list[dict]


@router.post("/run", status_code=201)
async def run_eval(
    project_id: int,
    req: EvalRunRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run an evaluation."""
    result = await run_evaluation(
        db, req.experiment_id, req.dataset_name, req.eval_type, req.predictions,
    )
    return EvalResultResponse.model_validate(result)


@router.post("/llm-judge", status_code=201)
async def run_llm_judge_eval(
    project_id: int,
    req: LLMJudgeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run LLM-as-a-judge evaluation."""
    result = await evaluate_with_llm_judge(
        db, req.experiment_id, req.dataset_name, req.judge_model, req.predictions
    )
    return EvalResultResponse.model_validate(result)


@router.get("/results/{experiment_id}", response_model=list[EvalResultResponse])
async def get_results(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get evaluation results for an experiment."""
    results = await get_eval_results(db, experiment_id)
    return [EvalResultResponse.model_validate(r) for r in results]


@router.get("/safety-scorecard/{experiment_id}")
async def safety_scorecard(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Generate safety scorecard for an experiment."""
    return await generate_safety_scorecard(db, experiment_id)

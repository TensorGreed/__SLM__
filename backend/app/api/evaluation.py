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
    run_heldout_evaluation,
)

router = APIRouter(prefix="/projects/{project_id}/evaluation", tags=["Evaluation"])


class EvalRunRequest(BaseModel):
    experiment_id: int
    dataset_name: str = Field(..., min_length=1, max_length=255)
    eval_type: Literal["exact_match", "f1", "safety"]
    predictions: list[dict]


class HeldoutEvalRunRequest(BaseModel):
    experiment_id: int
    dataset_name: str = Field(default="test", min_length=1, max_length=255)
    eval_type: Literal["exact_match", "f1", "llm_judge"] = "exact_match"
    max_samples: int = Field(default=100, ge=1, le=5000)
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    model_path: str | None = None
    judge_model: str = Field(default="meta-llama/Meta-Llama-3-70B-Instruct", min_length=1, max_length=255)


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


@router.post("/run-heldout", status_code=201)
async def run_eval_on_heldout(
    project_id: int,
    req: HeldoutEvalRunRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run end-to-end evaluation by generating predictions from held-out dataset rows."""
    try:
        result = await run_heldout_evaluation(
            db=db,
            project_id=project_id,
            experiment_id=req.experiment_id,
            dataset_name=req.dataset_name,
            eval_type=req.eval_type,
            max_samples=req.max_samples,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            model_path=req.model_path,
            judge_model=req.judge_model,
        )
        return EvalResultResponse.model_validate(result)
    except ValueError as e:
        raise HTTPException(400, str(e))


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

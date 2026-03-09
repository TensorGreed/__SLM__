"""Evaluation API routes."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.schemas.evaluation import EvalResultResponse, LLMJudgeRequest
from app.services.evaluation_pack_service import (
    DEFAULT_EVALUATION_PACK_ID,
    DOMAIN_PROFILE_EVAL_PACK_ID,
    evaluate_experiment_auto_gates,
    is_supported_evaluation_pack_id,
    list_evaluation_packs,
    normalize_evaluation_pack_id,
    resolve_project_evaluation_pack,
)
from app.services.evaluation_service import (
    evaluate_with_llm_judge,
    generate_safety_scorecard,
    get_eval_results,
    run_evaluation,
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


class EvaluationPackPreferenceUpdateRequest(BaseModel):
    pack_id: str | None = Field(default=None, max_length=128)


@router.post("/run", status_code=201)
async def run_eval(
    project_id: int,
    req: EvalRunRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run an evaluation."""
    try:
        result = await run_evaluation(
            db,
            project_id,
            req.experiment_id,
            req.dataset_name,
            req.eval_type,
            req.predictions,
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


@router.get("/packs")
async def list_eval_packs(
    project_id: int,
    include_gates: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """List built-in evaluation packs and project-resolved active pack."""
    try:
        resolved = await resolve_project_evaluation_pack(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

    packs = list_evaluation_packs(include_gates=include_gates)
    if bool(resolved.get("dynamic_pack_available")):
        dynamic = await resolve_project_evaluation_pack(
            db,
            project_id,
            preferred_pack_id=DOMAIN_PROFILE_EVAL_PACK_ID,
        )
        if str(dynamic.get("active_pack_id") or "") == DOMAIN_PROFILE_EVAL_PACK_ID:
            dynamic_pack = dict(dynamic.get("pack") or {})
            dynamic_pack["gate_count"] = len(list(dynamic_pack.get("gates") or []))
            if not include_gates:
                dynamic_pack.pop("gates", None)
            packs.append(dynamic_pack)

    return {
        "project_id": project_id,
        "default_pack_id": DEFAULT_EVALUATION_PACK_ID,
        "domain_profile_pack_id": DOMAIN_PROFILE_EVAL_PACK_ID,
        "active_pack_id": resolved.get("active_pack_id"),
        "active_pack_source": resolved.get("source"),
        "dynamic_pack_available": bool(resolved.get("dynamic_pack_available")),
        "warnings": list(resolved.get("warnings") or []),
        "packs": packs,
    }


@router.get("/pack-preference")
async def get_eval_pack_preference(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Read project evaluation pack preference and resolved effective pack."""
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    resolved = await resolve_project_evaluation_pack(db, project_id)
    return {
        "project_id": project_id,
        "preferred_pack_id": normalize_evaluation_pack_id(project.evaluation_preferred_pack_id),
        "active_pack_id": resolved.get("active_pack_id"),
        "active_pack_source": resolved.get("source"),
        "dynamic_pack_available": bool(resolved.get("dynamic_pack_available")),
        "default_pack_id": DEFAULT_EVALUATION_PACK_ID,
        "domain_profile_pack_id": DOMAIN_PROFILE_EVAL_PACK_ID,
        "warnings": list(resolved.get("warnings") or []),
        "active_pack": resolved.get("pack"),
    }


@router.put("/pack-preference")
async def set_eval_pack_preference(
    project_id: int,
    req: EvaluationPackPreferenceUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist project evaluation pack preference."""
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    normalized = normalize_evaluation_pack_id(req.pack_id)
    if normalized is not None and not is_supported_evaluation_pack_id(normalized):
        builtin = ", ".join(sorted(item["pack_id"] for item in list_evaluation_packs(include_gates=False)))
        raise HTTPException(
            400,
            (
                f"Unsupported evaluation pack '{normalized}'. "
                f"Allowed values: {builtin}, {DOMAIN_PROFILE_EVAL_PACK_ID}, or null to clear."
            ),
        )

    project.evaluation_preferred_pack_id = normalized
    await db.flush()

    resolved = await resolve_project_evaluation_pack(db, project_id)
    return {
        "project_id": project_id,
        "preferred_pack_id": normalize_evaluation_pack_id(project.evaluation_preferred_pack_id),
        "active_pack_id": resolved.get("active_pack_id"),
        "active_pack_source": resolved.get("source"),
        "dynamic_pack_available": bool(resolved.get("dynamic_pack_available")),
        "default_pack_id": DEFAULT_EVALUATION_PACK_ID,
        "domain_profile_pack_id": DOMAIN_PROFILE_EVAL_PACK_ID,
        "warnings": list(resolved.get("warnings") or []),
        "active_pack": resolved.get("pack"),
    }


@router.get("/gates/{experiment_id}")
async def evaluate_gates_for_experiment(
    project_id: int,
    experiment_id: int,
    pack_id: str | None = None,
    task_profile: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Evaluate experiment metrics against active/requested evaluation pack gates.

    Optional `task_profile` lets clients force a specific task-aware spec from
    the evaluation contract v2 pack.
    """
    normalized_pack = normalize_evaluation_pack_id(pack_id)
    if normalized_pack is not None and not is_supported_evaluation_pack_id(normalized_pack):
        builtin = ", ".join(sorted(item["pack_id"] for item in list_evaluation_packs(include_gates=False)))
        raise HTTPException(
            400,
            f"Unsupported pack_id '{normalized_pack}'. Allowed values: {builtin}, {DOMAIN_PROFILE_EVAL_PACK_ID}.",
        )

    try:
        return await evaluate_experiment_auto_gates(
            db,
            project_id=project_id,
            experiment_id=experiment_id,
            pack_id=normalized_pack,
            task_profile=task_profile,
        )
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

"""Evaluation API routes."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.schemas.evaluation import (
    EvalResultResponse,
    LLMJudgeRequest,
    RemediationPlanGenerateRequest,
    RemediationPlanIndexResponse,
)
from app.services.evaluation_pack_service import (
    DEFAULT_EVALUATION_PACK_ID,
    DOMAIN_PROFILE_EVAL_PACK_ID,
    evaluate_experiment_auto_gates,
    is_supported_evaluation_pack_id,
    list_evaluation_packs,
    normalize_evaluation_pack_id,
    resolve_project_evaluation_pack,
)
from app.services.pack_generation_service import generate_starter_eval_pack
from app.services.evaluation_service import (
    evaluate_with_llm_judge,
    generate_safety_scorecard,
    get_eval_results,
    run_evaluation,
    run_heldout_evaluation,
)
from app.services.evaluation_remediation_service import (
    RemediationPlanBlockedError,
    generate_remediation_plan,
    get_remediation_plan,
    list_remediation_plan_index,
)
from app.services.serve_runtime_service import list_serve_runs

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


class EvaluationPackGenerateRequest(BaseModel):
    blueprint_id: int | None = Field(default=None, ge=1)
    dataset_id: int | None = Field(default=None, ge=1)
    adapter_id: int | None = Field(default=None, ge=1)
    include_judge_rubric: bool = Field(default=True)


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


@router.get("/local-judge/serve-runs")
async def list_local_judge_serve_runs(
    project_id: int,
    limit: int = 30,
):
    """List serve runs that can be selected as local LLM-judge backends."""
    payload = await list_serve_runs(project_id=project_id, limit=limit, logs_tail=0)
    runs = [item for item in list(payload.get("runs") or []) if isinstance(item, dict)]
    candidates: list[dict] = []
    for run in runs:
        telemetry = dict(run.get("telemetry") or {})
        smoke_url = str(telemetry.get("smoke_url") or "").strip()
        first_token_url = str(telemetry.get("first_token_url") or "").strip()
        if not (smoke_url or first_token_url):
            continue
        candidates.append(
            {
                "run_id": str(run.get("run_id") or "").strip(),
                "status": str(run.get("status") or "").strip(),
                "source": str(run.get("source") or "").strip(),
                "template_id": str(run.get("template_id") or "").strip(),
                "template_name": str(run.get("template_name") or "").strip(),
                "export_id": run.get("export_id"),
                "model_id": run.get("model_id"),
                "smoke_url": smoke_url or None,
                "first_token_url": first_token_url or None,
                "first_healthy_at": telemetry.get("first_healthy_at"),
                "startup_latency_ms": telemetry.get("startup_latency_ms"),
            }
        )
    return {
        "project_id": project_id,
        "count": len(candidates),
        "runs": candidates,
    }


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


_PACK_GENERATE_ERROR_STATUS: dict[str, int] = {
    "project_not_found": 404,
    "blueprint_not_found": 404,
    "dataset_not_found": 404,
    "adapter_not_found": 404,
}


@router.post("/packs/generate", status_code=201)
async def generate_eval_pack(
    project_id: int,
    req: EvaluationPackGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Auto-generate a starter evaluation pack from blueprint + dataset + adapter.

    Input identifiers are optional — omitting `blueprint_id` uses the latest
    blueprint revision for the project, omitting `dataset_id` picks the most
    recent project dataset (if any), and omitting `adapter_id` skips the
    adapter-driven task-profile override.
    """
    try:
        pack = await generate_starter_eval_pack(
            db,
            project_id=project_id,
            blueprint_id=req.blueprint_id,
            dataset_id=req.dataset_id,
            adapter_id=req.adapter_id,
            include_judge_rubric=req.include_judge_rubric,
        )
    except ValueError as e:
        reason = str(e)
        status = _PACK_GENERATE_ERROR_STATUS.get(reason, 400)
        raise HTTPException(status, reason)
    return pack


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


@router.get("/scorecard/{experiment_id}")
async def experiment_scorecard(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Detailed scorecard for a specific experiment including Ship/No-Ship decision."""
    try:
        report = await evaluate_experiment_auto_gates(
            db,
            project_id=project_id,
            experiment_id=experiment_id,
        )

        project_stmt = select(Project).where(Project.id == project_id)
        project_res = await db.execute(project_stmt)
        project = project_res.scalar_one_or_none()
        if not project:
            raise HTTPException(404, "Project not found")

        policy = project.gate_policy or {}
        must_pass = policy.get("must_pass", True)
        blocked_if_missing = policy.get("blocked_if_missing", True)

        failed_gates = report.get("failed_gate_ids", [])
        missing_metrics = report.get("missing_required_metrics", [])

        is_ship = True
        reasons = []

        if must_pass and not report.get("passed"):
            is_ship = False
            reasons.append(f"Failed {len(failed_gates)} mandatory gates.")

        if blocked_if_missing and missing_metrics:
            is_ship = False
            reasons.append(f"Missing {len(missing_metrics)} required metrics.")

        return {
            "experiment_id": experiment_id,
            "is_ship": is_ship,
            "decision": "SHIP" if is_ship else "NO-SHIP",
            "reasons": reasons,
            "failed_gates": failed_gates,
            "missing_metrics": missing_metrics,
            "gate_report": report,
            "policy": policy
        }
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


@router.post("/remediation-plans/generate", status_code=201)
async def generate_eval_remediation_plan(
    project_id: int,
    req: RemediationPlanGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate and persist a remediation plan from evaluation failures."""
    try:
        return await generate_remediation_plan(
            db,
            project_id=project_id,
            experiment_id=req.experiment_id,
            evaluation_result_id=req.evaluation_result_id,
            max_failures=req.max_failures,
        )
    except RemediationPlanBlockedError as e:
        raise HTTPException(e.status_code, e.detail)
    except ValueError as e:
        detail = str(e)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code, detail)


@router.get("/remediation-plans", response_model=RemediationPlanIndexResponse)
async def list_eval_remediation_plans(
    project_id: int,
    experiment_id: int | None = None,
    evaluation_result_id: int | None = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """List persisted remediation plans for project/experiment/evaluation scopes."""
    try:
        plans = await list_remediation_plan_index(
            db,
            project_id=project_id,
            experiment_id=experiment_id,
            evaluation_result_id=evaluation_result_id,
            limit=limit,
        )
        return RemediationPlanIndexResponse(
            project_id=project_id,
            count=len(plans),
            plans=plans,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/remediation-plans/{plan_id}")
async def get_eval_remediation_plan(
    project_id: int,
    plan_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Fetch a previously generated remediation plan by plan id."""
    payload = await get_remediation_plan(
        db,
        project_id=project_id,
        plan_id=plan_id,
    )
    if payload is None:
        raise HTTPException(404, f"Remediation plan '{plan_id}' not found in project {project_id}")
    return payload

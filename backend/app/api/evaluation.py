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
from app.services.active_learning_service import (
    DEFAULT_PROPOSE_ROWS,
    MAX_PROPOSE_ROWS,
    promote_active_learning_batch,
    propose_active_learning_batch,
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


@router.get("/{eval_result_id}/failure-clusters")
async def failure_clusters(
    project_id: int,  # noqa: ARG001 — scoped by router prefix, used for route matching
    eval_result_id: int,
    max_failures: int = 200,
    max_exemplars_per_cluster: int = 3,
    db: AsyncSession = Depends(get_db),
):
    """Cluster row-level eval failures by `(reason_code, output_pattern)` (P12)."""
    from app.services.failure_cluster_service import cluster_eval_result_failures

    try:
        return await cluster_eval_result_failures(
            db,
            eval_result_id=eval_result_id,
            max_failures=max_failures,
            max_exemplars_per_cluster=max_exemplars_per_cluster,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "eval_result_not_found":
            raise HTTPException(404, detail) from exc
        raise HTTPException(400, detail) from exc


@router.get("/{eval_result_id}/reroute-analysis")
async def get_reroute_analysis(
    project_id: int,
    eval_result_id: int,
    refresh: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """USER-SUCCESS Epic 7 Phase 7a — post-eval decision-engine analysis.

    Inspects the eval, the project brief, and the gold-set shape and
    returns a categorical reroute recommendation
    (try_rag / try_prompt_engineering / expand_data / stay_the_course)
    plus the three signal verdicts that drove it.

    Result is cached on ``EvalResult.details["reroute_analysis"]``.
    Pass ``?refresh=true`` to recompute and overwrite the cache.
    """
    from app.models.experiment import EvalResult, Experiment
    from app.services.post_eval_decision_engine_service import (
        analyze_eval_for_reroute,
    )

    eval_result = await db.get(EvalResult, eval_result_id)
    if eval_result is None:
        raise HTTPException(404, f"EvalResult {eval_result_id} not found")
    experiment = await db.get(Experiment, eval_result.experiment_id)
    if experiment is None:
        raise HTTPException(404, "experiment_not_found")
    if experiment.project_id != project_id:
        raise HTTPException(400, "eval_result_not_in_project")

    try:
        return await analyze_eval_for_reroute(
            db,
            eval_result_id=eval_result_id,
            use_cache=not refresh,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {"eval_result_not_found", "project_not_found", "experiment_not_found"}:
            raise HTTPException(404, detail) from exc
        raise HTTPException(400, detail) from exc


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


# ── Active-learning recommender (Theme 8 Epic 2) ────────────────────


class ActiveLearningPromoteRequest(BaseModel):
    row_indexes: list[int] = Field(
        default_factory=list,
        description=(
            "Row indexes (into the eval result's predictions array) "
            "to promote into the synthetic training dataset. The "
            "service ignores duplicates / out-of-range indexes; "
            "already-promoted rows are reported as skipped."
        ),
    )


@router.get("/active-learning/{experiment_id}/proposal")
async def get_active_learning_proposal(
    project_id: int,
    experiment_id: int,
    max_rows: int = DEFAULT_PROPOSE_ROWS,
    db: AsyncSession = Depends(get_db),
):
    """Propose up to `max_rows` failed eval rows for the given
    experiment, ranked by failure severity. Used by the
    'Add these N examples to improve most' panel on the Eval tab."""
    capped = max(1, min(MAX_PROPOSE_ROWS, int(max_rows or DEFAULT_PROPOSE_ROWS)))
    return await propose_active_learning_batch(
        db,
        project_id=project_id,
        experiment_id=experiment_id,
        max_rows=capped,
    )


@router.post("/active-learning/{experiment_id}/promote", status_code=201)
async def promote_active_learning_rows(
    project_id: int,
    experiment_id: int,
    data: ActiveLearningPromoteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Append the selected rows' gold answers to the project's
    SYNTHETIC dataset JSONL. Idempotent — re-running with the same
    indexes is a no-op."""
    try:
        result = await promote_active_learning_batch(
            db,
            project_id=project_id,
            experiment_id=experiment_id,
            row_indexes=list(data.row_indexes or []),
        )
    except ValueError as e:
        detail = str(e)
        if detail.startswith("project_not_found:") or detail.startswith("eval_result_not_found:"):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)
    return {
        "status": "ok",
        "experiment_id": experiment_id,
        **result,
    }


# ── Per-cluster failure explanations (Theme 8 Epic 3) ───────────────


@router.post("/{eval_result_id}/clusters/{cluster_id}/explain", status_code=200)
async def explain_failure_cluster_endpoint(
    project_id: int,
    eval_result_id: int,
    cluster_id: str,
    force: bool = False,
    max_exemplars: int = 5,
    judge_model: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Generate (or return cached) a one-line LLM-judge explanation
    for a failure cluster. Used by the Eval tab's
    `FailureClustersPanel` when the user expands a cluster.

    Returns 200 with `status="judge_unavailable"` when no judge
    model is configured — the UI then renders a soft fallback
    rather than breaking the expand interaction.
    """
    # Local import to avoid the circular dep at module load time
    # (cluster_explanation_service depends on evaluation_service).
    from app.services.cluster_explanation_service import (
        explain_failure_cluster,
    )

    try:
        payload = await explain_failure_cluster(
            db,
            project_id=project_id,
            eval_result_id=eval_result_id,
            cluster_id=cluster_id,
            max_exemplars=max(1, min(20, int(max_exemplars))),
            force_refresh=bool(force),
            judge_model=judge_model,
        )
    except ValueError as e:
        detail = str(e)
        if detail.startswith("eval_result_not_found:"):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)
    return {
        "project_id": project_id,
        "eval_result_id": eval_result_id,
        **payload,
    }


@router.post("/{eval_result_id}/clusters/{cluster_id}/augment", status_code=200)
async def augment_failure_cluster_endpoint(
    project_id: int,
    eval_result_id: int,
    cluster_id: str,
    target_count: int = 30,
    backend: str | None = None,
    async_job: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Generate cluster-targeted synthetic training data for a
    specific failure cluster (USER-SUCCESS Epic 2b).

    Calls the CLUSTER_TARGETED playbook for the project's recipe with
    the cluster's exemplars + reason code as context. Generated rows
    land in the project's synthetic dataset with provenance
    ``synth_source = "playbook:<recipe>:cluster_targeted:cluster=<cluster_id>"``
    and ``review_status = "pending"`` so they pass through the synth
    review queue before entering training.

    Returns the PlaybookResult dict (rows + backend_used +
    elapsed_sec + prompt_snippet).

    Hardening — pass ``?async_job=true`` to fire the generation as a
    background Job. Endpoint returns 202 + Job stub immediately; the
    bell tracks progress with an elapsed-time heartbeat and flips to
    FAILED when 0 rows are produced (silent-failure guard).

    503 when no synth backend is reachable. 404 when the eval /
    cluster / project isn't found. 400 for everything else.
    """
    from app.services.active_learning_service import augment_from_cluster
    from app.services.synth_backends import SynthBackendError

    if target_count < 1 or target_count > 500:
        raise HTTPException(400, "target_count must be between 1 and 500")

    if async_job:
        import asyncio
        import time

        from fastapi.responses import JSONResponse

        from app.services.jobs_service import (
            JobProgressHandle,
            serialize_job,
            start_job,
        )

        async def _runner(handle: JobProgressHandle) -> dict:
            # Elapsed-time heartbeat — matches the synth-playbook
            # pattern so the bell never sits on a stale message
            # while Ollama / the teacher backend is mid-call.
            started = time.monotonic()
            stop_heartbeat = asyncio.Event()

            async def _heartbeat() -> None:
                while not stop_heartbeat.is_set():
                    try:
                        await asyncio.wait_for(
                            stop_heartbeat.wait(), timeout=5.0,
                        )
                        return
                    except asyncio.TimeoutError:
                        elapsed = int(time.monotonic() - started)
                        await handle.set_progress(
                            message=(
                                f"Augmenting cluster {cluster_id} "
                                f"({target_count} rows) · {elapsed}s elapsed"
                            ),
                        )

            heartbeat_task = asyncio.create_task(_heartbeat())
            try:
                await handle.set_progress(
                    message=(
                        f"Augmenting cluster {cluster_id} · "
                        f"{target_count} rows requested"
                    ),
                )
                from app.database import async_session_factory

                async with async_session_factory() as runner_db:
                    result = await augment_from_cluster(
                        runner_db,
                        project_id=project_id,
                        eval_result_id=eval_result_id,
                        cluster_id=cluster_id,
                        target_count=target_count,
                        backend=backend,
                    )
                    await runner_db.commit()
            finally:
                stop_heartbeat.set()
                try:
                    await heartbeat_task
                except Exception:  # noqa: BLE001 — heartbeat best-effort
                    pass

            row_count = len(result.get("rows") or [])
            backend_used = result.get("backend_used") or "auto"
            elapsed_sec = result.get("elapsed_sec") or 0
            prompt_snippet = result.get("prompt_snippet") or ""

            # 0-rows-succeeded is a silent failure (parse / validate
            # rejected everything). Raise so the Job lands as FAILED
            # with an actionable diagnostic — matches the synth
            # playbook guard from earlier.
            if row_count == 0:
                raise RuntimeError(
                    f"Cluster-augment produced 0 accepted rows via "
                    f"{backend_used} in {elapsed_sec:.1f}s. The LLM "
                    f"returned output that didn't parse OR every "
                    f"generated row failed cluster validation (e.g. "
                    f"hard-negative emitted with the target class "
                    f"again). Check the server logs for the backend's "
                    f"raw response. Prompt (first 200 chars): "
                    f"{prompt_snippet[:200]!r}"
                )
            await handle.set_progress(
                fraction=1.0,
                message=(
                    f"Generated {row_count} rows via {backend_used} "
                    f"in {elapsed_sec:.1f}s · queued for review"
                ),
            )
            return {
                "rows_generated": row_count,
                "backend_used": backend_used,
                "elapsed_sec": elapsed_sec,
                "cluster_id": cluster_id,
                "eval_result_id": eval_result_id,
                "target_count": target_count,
            }

        job = await start_job(
            db,
            kind="synth_augment_from_cluster",
            title=(
                f"Augment cluster {cluster_id[:16]} · "
                f"{target_count} rows"
            ),
            runner=_runner,
            project_id=project_id,
            params={
                "eval_result_id": eval_result_id,
                "cluster_id": cluster_id,
                "target_count": target_count,
                "backend": backend,
            },
        )
        return JSONResponse(
            status_code=202,
            content=serialize_job(job),
        )

    try:
        return await augment_from_cluster(
            db,
            project_id=project_id,
            eval_result_id=eval_result_id,
            cluster_id=cluster_id,
            target_count=target_count,
            backend=backend,
        )
    except SynthBackendError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        message = str(e)
        if "not found" in message.lower():
            raise HTTPException(404, message)
        raise HTTPException(400, message)


# ── "Did SFT help?" lift summary (Theme 8 Epic 4) ───────────────────


@router.get("/sft-lift-summary")
async def get_sft_lift_summary(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Compute the baseline → trained lift summary for this project.
    Used by the Eval tab's "Did SFT help?" panel. Returns
    `status="no_baseline"` / `"no_trained"` / `"no_overlap"` when
    the comparison can't be built — never 404 unless the project is
    missing — so the UI can render an informative fallback."""
    from app.services.sft_lift_summary_service import compute_sft_lift_summary

    try:
        return await compute_sft_lift_summary(db, project_id)
    except ValueError as e:
        detail = str(e)
        if detail.startswith("project_not_found:"):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

"""Training API routes."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.dataset import DatasetType
from app.models.experiment import Checkpoint, Experiment, ExperimentStatus, TrainingMode
from app.models.project import Project
from app.schemas.training import ExperimentCreate, ExperimentResponse, TrainingConfig
from app.services.dataset_service import (
    preview_project_data_adapter,
    resolve_project_dataset_adapter_preference,
    save_project_dataset_adapter_preference,
    split_dataset,
)
from app.services.ingestion_service import list_documents, process_document
from app.services.job_service import get_task_status
from app.services.domain_profile_service import get_training_defaults
from app.services.domain_runtime_service import resolve_project_domain_runtime
from app.services.training_service import (
    cancel_training,
    create_experiment,
    get_training_status,
    list_experiments,
    start_training,
    register_websocket,
    unregister_websocket,
)
from app.services.training_preflight_service import (
    TRAINING_PLAN_PROFILES,
    normalize_training_plan_profile,
    run_training_preflight,
    run_training_preflight_plan,
)
from app.services.alignment_service import (
    list_alignment_recipes,
    resolve_alignment_recipe,
    score_preference_rows,
    validate_preference_rows,
)
from app.services.alignment_dataset_service import (
    compose_alignment_training_dataset,
    filter_preference_dataset_by_quality,
    import_preference_dataset_rows,
    summarize_playground_active_learning,
    materialize_playground_preference_pairs,
    summarize_preference_dataset,
)
from app.services.model_selection_service import (
    introspect_training_base_model,
    list_model_catalog,
    recommend_training_base_models,
)
from app.services.model_benchmark_service import benchmark_model_sweep
from app.services.training_telemetry_service import (
    build_model_acceptance_bias,
    build_model_benchmark_bias,
    list_model_benchmark_runs,
    list_observability_events,
    record_model_benchmark_run,
    record_model_wizard_event,
    record_observability_event,
    summarize_model_benchmark_runs,
    summarize_observability_events,
    summarize_model_wizard_events,
)
from app.services.playground_service import (
    list_playground_provider_catalog,
    normalize_playground_messages,
    run_playground_chat,
    stream_playground_chat,
)
from app.services.playground_log_service import (
    list_playground_feedback,
    record_playground_feedback,
    summarize_playground_feedback,
)
from app.services.rag_sandbox_service import (
    build_rag_context_block,
    retrieve_project_rag_snippets,
)
from app.services.playground_session_service import (
    delete_playground_session,
    get_playground_session,
    list_playground_model_options,
    list_playground_sessions,
    resolve_playground_model_runtime,
    save_playground_session_transcript,
    serialize_playground_session_detail,
    serialize_playground_session_summary,
)
from app.services.training_recipe_service import (
    list_training_recipes,
    resolve_training_recipe,
)
from app.services.training_runtime_service import (
    list_runtime_catalog,
    reload_runtime_plugins_from_settings,
    runtime_plugin_status,
)
from app.services.newbie_autopilot_service import (
    build_newbie_autopilot_intent_clarification,
    evaluate_newbie_autopilot_dataset_readiness,
    estimate_newbie_autopilot_run,
    resolve_newbie_autopilot_intent,
)
from app.services.autopilot_decision_service import persist_decision_log
from app.services.autopilot_snapshot_service import (
    capture_project_snapshot,
    capture_snapshot,
    update_snapshot_post_state,
)
from app.services.readiness_service import get_project_readiness
from app.services.target_profile_service import (
    check_compatibility,
    get_target_by_id,
    list_targets,
    resolve_target_device,
)
from app.services.capability_contract_service import build_training_capability_contract
from app.services.cloud_burst_service import (
    build_cloud_burst_launch_plan,
    cancel_cloud_burst_job,
    estimate_cloud_burst_quote,
    get_cloud_burst_job_logs,
    get_cloud_burst_job_status,
    list_cloud_burst_catalog,
    list_cloud_burst_jobs,
    submit_cloud_burst_job,
    sync_cloud_burst_job_artifacts,
)
from app.services.vibe_check_service import (
    capture_vibe_check_snapshot,
    load_project_vibe_check_config,
    load_vibe_check_timeline,
    save_project_vibe_check_config,
)

router = APIRouter(prefix="/projects/{project_id}/training", tags=["Training"])


class TrainingEffectiveConfigRequest(BaseModel):
    config: dict[str, object] = Field(default_factory=dict)


class TrainingPreferencesUpdateRequest(BaseModel):
    preferred_plan_profile: str = Field(..., min_length=1, max_length=32)


class VibeCheckConfigUpdateRequest(BaseModel):
    enabled: bool | None = None
    interval_steps: int | None = Field(default=None, ge=10, le=1000)
    prompts: list[str] | None = Field(default=None, min_length=1, max_length=5)
    provider: str | None = Field(default=None, max_length=64)
    model_name: str | None = Field(default=None, max_length=255)
    api_url: str | None = Field(default=None, max_length=2048)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=32, le=4096)


class VibeCheckSnapshotRequest(BaseModel):
    step: int | None = Field(default=None, ge=1)
    epoch: float | None = Field(default=None, ge=0.0)
    train_loss: float | None = None
    eval_loss: float | None = None
    api_key: str | None = Field(default=None, max_length=8192)


class TrainingRecipeResolveRequest(BaseModel):
    recipe_id: str = Field(..., min_length=1, max_length=128)
    base_config: dict[str, object] = Field(default_factory=dict)
    overrides: dict[str, object] = Field(default_factory=dict)
    include_preflight: bool = True


class ModelSelectionRecommendRequest(BaseModel):
    target_device: str = Field(default="laptop", min_length=1, max_length=32)
    primary_language: str = Field(default="english", min_length=1, max_length=32)
    available_vram_gb: float | None = Field(default=None, ge=0)
    task_profile: str | None = Field(default=None, max_length=64)
    top_k: int = Field(default=3, ge=1, le=5)


class ModelSelectionIntrospectRequest(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=255)
    allow_network: bool = True


class ModelSelectionTelemetryRequest(BaseModel):
    action: str = Field(default="recommend", pattern="^(recommend|apply)$")
    source: str = Field(default="training_setup_wizard", min_length=1, max_length=64)
    apply_source: str | None = Field(
        default=None,
        pattern="^(recommendation|benchmark|consensus)$",
        description="Decision pathway for apply events.",
    )
    auto_run: bool | None = None
    target_device: str | None = Field(default=None, max_length=32)
    primary_language: str | None = Field(default=None, max_length=32)
    available_vram_gb: float | None = Field(default=None, ge=0)
    task_profile: str | None = Field(default=None, max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=5)
    recommendation_count: int | None = Field(default=None, ge=0, le=20)
    recommendation_model_ids: list[str] | None = None
    selected_model_id: str | None = Field(default=None, max_length=255)
    selected_rank: int | None = Field(default=None, ge=1, le=50)
    selected_score: float | None = None


class ModelBenchmarkSweepRequest(BaseModel):
    target_device: str = Field(default="laptop", min_length=1, max_length=32)
    primary_language: str = Field(default="english", min_length=1, max_length=32)
    available_vram_gb: float | None = Field(default=None, ge=0)
    task_profile: str | None = Field(default=None, max_length=64)
    model_ids: list[str] = Field(default_factory=list)
    max_models: int = Field(default=3, ge=1, le=5)
    sample_size: int = Field(default=96, ge=10, le=500)
    allow_network_tokenizer: bool = False
    persist_run: bool = True


class ObservabilityTelemetryRequest(BaseModel):
    experiment_id: int | None = Field(default=None, ge=1)
    step: int | None = Field(default=None, ge=1)
    epoch: float | None = Field(default=None, ge=0)
    split: str | None = Field(default=None, max_length=32)
    layer_gradients: list[dict[str, object]] = Field(default_factory=list)
    attention_focus: list[dict[str, object]] = Field(default_factory=list)
    gradient_anomaly: bool | None = None
    hallucination_signal: bool | None = None
    notes: str | None = Field(default=None, max_length=2000)


class PlaygroundChatMessage(BaseModel):
    role: str = Field(..., min_length=1, max_length=16)
    content: str = Field(..., min_length=1, max_length=32000)


class PlaygroundChatRequest(BaseModel):
    provider: str = Field(default="openai_compatible", min_length=1, max_length=64)
    model_name: str | None = Field(default=None, max_length=255)
    api_url: str | None = Field(default=None, max_length=2048)
    api_key: str | None = Field(default=None, max_length=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=16, le=4096)
    system_prompt: str | None = Field(default=None, max_length=8000)
    auto_runtime_provider: bool = True
    session_id: int | None = Field(default=None, ge=1)
    session_title: str | None = Field(default=None, max_length=255)
    save_history: bool = True
    messages: list[PlaygroundChatMessage] = Field(default_factory=list, min_length=1)


class PlaygroundSessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    provider: str = Field(default="openai_compatible", min_length=1, max_length=64)
    model_name: str | None = Field(default=None, max_length=255)
    api_url: str | None = Field(default=None, max_length=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=16, le=4096)
    system_prompt: str | None = Field(default=None, max_length=8000)
    messages: list[PlaygroundChatMessage] = Field(default_factory=list)


class PlaygroundFeedbackRequest(BaseModel):
    session_id: int | None = Field(default=None, ge=1)
    message_index: int | None = Field(default=None, ge=0)
    provider: str | None = Field(default=None, max_length=64)
    model_name: str | None = Field(default=None, max_length=512)
    preset_id: str | None = Field(default=None, max_length=128)
    prompt: str = Field(..., min_length=1, max_length=32000)
    reply: str = Field(..., min_length=1, max_length=64000)
    rating: int | None = Field(default=None, ge=-1, le=1)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=4000)
    preferred_reply: str | None = Field(default=None, max_length=64000)


class PlaygroundRagCompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20000)
    provider: str = Field(default="mock", min_length=1, max_length=64)
    base_model_name: str | None = Field(default=None, max_length=255)
    tuned_model_name: str | None = Field(default=None, max_length=255)
    api_url: str | None = Field(default=None, max_length=2048)
    api_key: str | None = Field(default=None, max_length=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=16, le=4096)
    top_k: int = Field(default=4, ge=1, le=10)


class AlignmentRecipeResolveRequest(BaseModel):
    recipe_id: str = Field(..., min_length=1, max_length=128)
    base_config: dict[str, object] = Field(default_factory=dict)
    overrides: dict[str, object] = Field(default_factory=dict)


class AlignmentRetrainFromFeedbackRequest(BaseModel):
    recipe_id: str = Field(default="recipe.alignment.dpo.fast", min_length=1, max_length=128)
    include_playground_pairs: bool = True
    max_playground_pairs: int = Field(default=5000, ge=1, le=50000)
    quality_threshold: float = Field(default=3.0, ge=1.0, le=5.0)
    preferred_plan_profile: str = "balanced"


class AlignmentContractValidateRequest(BaseModel):
    rows: list[dict[str, object]] = Field(default_factory=list, min_length=1)
    min_coverage: float = Field(default=0.85, ge=0.0, le=1.0)
    max_rows: int = Field(default=2000, ge=1, le=10000)


class AlignmentJudgeScoreRequest(BaseModel):
    rows: list[dict[str, object]] = Field(default_factory=list, min_length=1)
    quality_threshold: float = Field(default=3.0, ge=1.0, le=5.0)
    max_rows: int = Field(default=1000, ge=1, le=5000)


class AlignmentDatasetImportRequest(BaseModel):
    rows: list[dict[str, object]] = Field(default_factory=list)
    raw_text: str | None = Field(default=None, max_length=2_000_000)
    mode: str = Field(default="replace", pattern="^(replace|append)$")
    target: str = Field(default="prepared_train", pattern="^(prepared_train|alignment_workspace)$")


class AlignmentDatasetFilterRequest(BaseModel):
    quality_threshold: float = Field(default=3.0, ge=1.0, le=5.0)
    min_keep_ratio: float = Field(default=0.4, ge=0.05, le=1.0)
    apply_to_train_file: bool = False
    source_path: str | None = Field(default=None, max_length=4096)
    target_path: str | None = Field(default=None, max_length=4096)


class AlignmentActiveLearningComposeRequest(BaseModel):
    source_path: str | None = Field(default=None, max_length=4096)
    target_path: str | None = Field(default=None, max_length=4096)
    include_playground_pairs: bool = True
    max_playground_pairs: int = Field(default=5000, ge=1, le=50000)


class CloudBurstQuoteRequest(BaseModel):
    provider_id: str = Field(..., min_length=1, max_length=64)
    gpu_sku: str = Field(..., min_length=1, max_length=64)
    duration_hours: float = Field(default=2.0, ge=0.25, le=72.0)
    storage_gb: int = Field(default=50, ge=10, le=2000)
    egress_gb: float = Field(default=0.0, ge=0.0, le=5000.0)
    spot: bool = True


class CloudBurstLaunchPlanRequest(BaseModel):
    provider_id: str = Field(..., min_length=1, max_length=64)
    gpu_sku: str = Field(..., min_length=1, max_length=64)
    duration_hours: float = Field(default=2.0, ge=0.25, le=72.0)
    experiment_id: int | None = Field(default=None, ge=1)
    region: str | None = Field(default=None, max_length=64)
    image: str = Field(default="", max_length=512)
    startup_script: str = Field(default="", max_length=8000)
    spot: bool = True


class CloudBurstJobSubmitRequest(BaseModel):
    provider_id: str = Field(..., min_length=1, max_length=64)
    gpu_sku: str = Field(..., min_length=1, max_length=64)
    duration_hours: float = Field(default=2.0, ge=0.25, le=72.0)
    experiment_id: int | None = Field(default=None, ge=1)
    region: str | None = Field(default=None, max_length=64)
    image: str = Field(default="", max_length=512)
    startup_script: str = Field(default="", max_length=8000)
    spot: bool = True
    auto_artifact_sync: bool = True
    artifact_sync_policy: str = Field(default="smart", pattern="^(smart|all)$")
    artifact_include_globs: list[str] = Field(default_factory=list, max_length=64)
    artifact_exclude_globs: list[str] = Field(default_factory=list, max_length=64)
    execution_mode: str = Field(default="auto", pattern="^(auto|live|simulate)$")
    allow_fallback_to_simulation: bool = True
    idempotency_key: str | None = Field(default=None, max_length=256)


class CloudBurstArtifactSyncRequest(BaseModel):
    policy: str = Field(default="smart", pattern="^(smart|all)$")
    include_globs: list[str] = Field(default_factory=list, max_length=64)
    exclude_globs: list[str] = Field(default_factory=list, max_length=64)
    dry_run: bool = False
    max_files: int = Field(default=2000, ge=1, le=10000)
    cursor: str | None = Field(default=None, max_length=4096)


class NewbieAutopilotIntentRequest(BaseModel):
    intent: str = Field(..., min_length=3, max_length=4000)
    target_profile_id: str = Field(default="vllm_server", min_length=1, max_length=64)
    target_device: str | None = Field(default=None, pattern="^(mobile|laptop|server)$")
    primary_language: str = Field(default="english", min_length=1, max_length=32)
    available_vram_gb: float | None = Field(default=None, ge=0)
    base_model: str | None = Field(default=None, min_length=1, max_length=255)
    auto_prepare_data: bool = True


class NewbieAutopilotOneClickRequest(NewbieAutopilotIntentRequest):
    run_name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=1000)
    auto_apply_rewrite: bool = True
    intent_rewrite: str | None = Field(default=None, min_length=3, max_length=4000)
    plan_profile: str = Field(
        default="balanced",
        pattern="^(safe|balanced|max_quality|fastest|best_quality)$",
    )


class NewbieAutopilotEstimateRequest(BaseModel):
    plan_profile: str = Field(
        ...,
        pattern="^(safe|balanced|max_quality|fastest|best_quality)$",
    )
    target_profile_id: str = Field(default="vllm_server", min_length=1, max_length=64)
    dataset_size_rows: int = Field(default=1000, ge=1)
    task_profile: str | None = Field(default=None, min_length=1, max_length=64)


class AutopilotPlanV2Response(BaseModel):
    project_id: int
    intent: str
    intent_rewrite: str | None = None
    resolved_target_device: str
    plans: list[dict[str, object]]
    recommended_profile: str
    guardrails: dict[str, object]
    dataset_readiness: dict[str, object]
    intent_clarification: dict[str, object]
    target_compatibility: dict[str, object] | None = None


class AutopilotV2OrchestrationRequest(NewbieAutopilotOneClickRequest):
    dry_run: bool = True
    allow_target_fallback: bool = True
    allow_profile_autotune: bool = True


class AutopilotV2OrchestrationResponse(BaseModel):
    project_id: int
    run_id: str = ""
    dry_run: bool
    strict_mode: bool
    intent: str
    effective_target_profile_id: str
    resolved_target_device: str
    selected_profile: str | None = None
    guardrails: dict[str, object] = Field(default_factory=dict)
    readiness: dict[str, object] = Field(default_factory=dict)
    decision_log: list[dict[str, object]] = Field(default_factory=list)
    repairs: dict[str, object] = Field(default_factory=dict)
    plan_v2: dict[str, object] = Field(default_factory=dict)
    experiment: dict[str, object] | None = None
    started: bool = False
    start_result: dict[str, object] | None = None
    start_error: str | None = None


def _sse_json(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=True)
    return f"data: {serialized}\n\n"


async def _get_project_or_404(db: AsyncSession, project_id: int) -> Project:
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _resolve_autopilot_target_device(
    *,
    target_profile_id: str | None,
    target_device: str | None,
) -> str:
    return resolve_target_device(target_profile_id, fallback=target_device)


def _guardrail_reason_code(message: str, *, blocker: bool) -> str:
    token = str(message or "").strip().lower()
    if not token:
        return "GUARDRAIL_BLOCKED" if blocker else "GUARDRAIL_WARNING"
    if "target" in token and ("compat" in token or "not fit" in token):
        return "TARGET_INCOMPATIBLE"
    if "target" in token and "vram" in token:
        return "TARGET_INCOMPATIBLE"
    if "dataset" in token or "prepared" in token or "ingest" in token:
        return "DATASET_NOT_READY"
    if "preflight" in token:
        return "TRAINING_PREFLIGHT_FAILED"
    if "intent" in token and "clarification" in token:
        return "INTENT_NEEDS_CLARIFICATION"
    if "credential" in token or "secret" in token or "token" in token:
        return "MISSING_CREDENTIALS"
    return "GUARDRAIL_BLOCKED" if blocker else "GUARDRAIL_WARNING"


def _has_dataset_guardrail_blocker(blockers: list[str]) -> bool:
    for item in blockers:
        if _guardrail_reason_code(str(item), blocker=True) == "DATASET_NOT_READY":
            return True
    return False


def _compat_params_b(payload: dict[str, object] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("parameters_billions")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _find_recommendation_by_model_id(
    recommendations: list[dict[str, object]],
    model_id: str,
) -> dict[str, object] | None:
    token = str(model_id or "").strip().lower()
    if not token:
        return None
    for item in recommendations:
        candidate = str(item.get("model_id") or "").strip().lower()
        if candidate == token:
            return item
    return None


def _attempt_compatibility_auto_repair(
    *,
    model_id: str,
    target_profile_id: str,
    recommendations: list[dict[str, object]],
) -> tuple[str, dict[str, object], dict[str, object] | None]:
    initial = check_compatibility(model_id, target_profile_id)
    if bool(initial.get("compatible", False)):
        return model_id, dict(initial), None
    initial_reasons = [
        str(item).strip()
        for item in list(initial.get("reasons") or [])
        if str(item).strip()
    ]
    initial_warnings = [
        str(item).strip()
        for item in list(initial.get("warnings") or [])
        if str(item).strip()
    ]

    base_params = _compat_params_b(initial)
    best_candidate_id = ""
    best_candidate_compat: dict[str, object] | None = None
    best_score: tuple[float, int] | None = None

    for idx, rec in enumerate(recommendations):
        candidate_id = str(rec.get("model_id") or "").strip()
        if not candidate_id:
            continue
        if candidate_id.lower() == model_id.lower():
            continue
        candidate_compat = check_compatibility(candidate_id, target_profile_id)
        if not bool(candidate_compat.get("compatible", False)):
            continue

        candidate_params = _compat_params_b(candidate_compat)
        if candidate_params is None:
            try:
                candidate_params = float(rec.get("params_b")) if rec.get("params_b") is not None else None
            except (TypeError, ValueError):
                candidate_params = None

        if base_params is not None and candidate_params is not None:
            distance = abs(candidate_params - base_params)
        else:
            # Fall back to recommendation rank if model sizes cannot be compared.
            distance = float(idx)
        score = (float(distance), int(idx))
        if best_score is None or score < best_score:
            best_score = score
            best_candidate_id = candidate_id
            best_candidate_compat = dict(candidate_compat)

    if best_candidate_id and best_candidate_compat is not None:
        return (
            best_candidate_id,
            best_candidate_compat,
            {
                "applied": True,
                "original_model_id": model_id,
                "repaired_model_id": best_candidate_id,
                "target_profile_id": target_profile_id,
                "strategy": "nearest_compatible_recommendation",
                "incompatibility_reasons": initial_reasons,
                "incompatibility_warnings": initial_warnings,
                "initial_vram_check": dict(initial.get("vram_check") or {}),
            },
        )

    return model_id, dict(initial), None


def _normalize_autopilot_task_profile(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    alias = {
        "chat_sft": "instruction_sft",
        "instruction": "instruction_sft",
        "qa_assistant": "qa",
    }
    return alias.get(token, token)


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _coerce_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


async def _load_autopilot_run_history(
    db: AsyncSession,
    *,
    project_id: int,
    fallback_target_profile_id: str | None = None,
    limit: int = 300,
) -> list[dict[str, object]]:
    safe_limit = max(1, min(int(limit), 1000))
    result = await db.execute(
        select(Experiment).where(
            Experiment.project_id == project_id,
            Experiment.status == ExperimentStatus.COMPLETED,
        ).order_by(Experiment.completed_at.desc()).limit(safe_limit)
    )
    experiments = result.scalars().all()
    rows: list[dict[str, object]] = []
    for exp in experiments:
        started_at = exp.started_at
        completed_at = exp.completed_at
        if started_at is None or completed_at is None:
            continue
        duration_seconds = (completed_at - started_at).total_seconds()
        if duration_seconds <= 0:
            continue

        config = dict(exp.config or {})
        plan_profile = (
            normalize_training_plan_profile(config.get("training_plan_profile"))
            or normalize_training_plan_profile(config.get("plan_profile"))
            or None
        )
        target_profile_id = (
            str(config.get("target_profile_id") or "").strip().lower()
            or str(fallback_target_profile_id or "").strip().lower()
            or None
        )
        task_profile = _normalize_autopilot_task_profile(config.get("task_profile"))
        if not task_profile:
            task_type = str(config.get("task_type") or "").strip().lower()
            if task_type == "classification":
                task_profile = "classification"
            elif task_type == "causal_lm":
                task_profile = "instruction_sft"

        dataset_size_rows = (
            _coerce_positive_int(config.get("_autopilot_dataset_rows"))
            or _coerce_positive_int(config.get("dataset_size_rows"))
            or None
        )

        runtime = dict(config.get("_runtime") or {})
        cost_usd = (
            _coerce_positive_float(runtime.get("estimated_cost_usd"))
            or _coerce_positive_float(runtime.get("cost_usd"))
            or _coerce_positive_float(runtime.get("cloud_burst_quote_total_usd"))
            or _coerce_positive_float(runtime.get("quote_total_usd"))
            or None
        )
        rows.append(
            {
                "experiment_id": int(exp.id),
                "plan_profile": plan_profile,
                "target_profile_id": target_profile_id,
                "task_profile": task_profile,
                "dataset_size_rows": dataset_size_rows,
                "duration_seconds": round(float(duration_seconds), 4),
                "cost_usd": cost_usd,
            }
        )
    return rows


async def _attempt_autopilot_auto_process_raw_documents(
    *,
    db: AsyncSession,
    project_id: int,
) -> dict[str, object]:
    report: dict[str, object] = {
        "attempted": True,
        "accepted_before": 0,
        "pending_before": 0,
        "accepted_after": 0,
        "processed_count": 0,
        "failed_count": 0,
        "failed_document_ids": [],
        "succeeded": False,
    }
    try:
        docs = await list_documents(db, project_id)
    except ValueError as exc:
        report["error"] = str(exc)
        return report

    accepted_before = 0
    pending_ids: list[int] = []
    for doc in docs:
        raw_status = getattr(doc, "status", None)
        status_token = str(getattr(raw_status, "value", raw_status) or "").strip().lower()
        if status_token == "accepted":
            accepted_before += 1
        elif status_token in {"pending", "processing", "error"}:
            try:
                pending_ids.append(int(getattr(doc, "id")))
            except (TypeError, ValueError):
                continue

    report["accepted_before"] = accepted_before
    report["pending_before"] = len(pending_ids)
    if not pending_ids:
        report["accepted_after"] = accepted_before
        report["succeeded"] = accepted_before > 0
        return report

    processed = 0
    failed: list[int] = []
    for document_id in pending_ids:
        try:
            processed_doc = await process_document(db, project_id, document_id)
            raw_status_after = getattr(processed_doc, "status", None)
            status_after = str(getattr(raw_status_after, "value", raw_status_after) or "").strip().lower()
            if status_after == "accepted":
                processed += 1
            else:
                failed.append(document_id)
        except Exception:  # noqa: BLE001
            failed.append(document_id)
            await db.rollback()

    report["processed_count"] = processed
    report["failed_count"] = len(failed)
    report["failed_document_ids"] = failed[:20]
    report["accepted_after"] = accepted_before + processed
    report["succeeded"] = (accepted_before + processed) > 0
    return report


async def _attempt_autopilot_auto_prepare_data(
    *,
    db: AsyncSession,
    project_id: int,
    task_profile: str | None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "attempted": True,
        "succeeded": False,
        "method": None,
        "error": None,
    }
    raw_document_report = await _attempt_autopilot_auto_process_raw_documents(
        db=db,
        project_id=project_id,
    )
    result["raw_documents"] = raw_document_report

    try:
        preset = await resolve_project_dataset_adapter_preference(db, project_id)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    adapter_id = str(preset.get("adapter_id") or "default-canonical").strip() or "default-canonical"
    adapter_config = dict(preset.get("adapter_config") or {})
    field_mapping = dict(preset.get("field_mapping") or {})
    resolved_task_profile = str(task_profile or preset.get("task_profile") or "").strip() or None

    try:
        preview = await preview_project_data_adapter(
            db=db,
            project_id=project_id,
            dataset_type=DatasetType.RAW,
            sample_size=250,
            adapter_id="auto",
            adapter_config=adapter_config,
            field_mapping=field_mapping,
            task_profile=resolved_task_profile,
            preview_limit=10,
        )
    except ValueError:
        preview = None
    except Exception:  # noqa: BLE001
        preview = None

    if isinstance(preview, dict):
        mapped_records = int(preview.get("mapped_records") or 0)
        if mapped_records > 0:
            adapter_id = str(preview.get("resolved_adapter_id") or adapter_id).strip() or adapter_id
            preview_task_profile = str(preview.get("resolved_task_profile") or "").strip() or None
            if preview_task_profile:
                resolved_task_profile = preview_task_profile
            try:
                await save_project_dataset_adapter_preference(
                    db,
                    project_id,
                    adapter_id=adapter_id,
                    adapter_config=adapter_config,
                    field_mapping=field_mapping,
                    task_profile=resolved_task_profile,
                )
            except ValueError:
                pass
            try:
                manifest = await split_dataset(
                    db=db,
                    project_id=project_id,
                    include_types=[DatasetType.RAW.value],
                    adapter_id=adapter_id,
                    adapter_config=adapter_config,
                    field_mapping=field_mapping,
                    task_profile=resolved_task_profile,
                )
                result["succeeded"] = True
                result["method"] = "raw_auto_detect"
                result["manifest"] = manifest
                return result
            except ValueError:
                await db.rollback()

    try:
        manifest = await split_dataset(
            db=db,
            project_id=project_id,
            include_types=[
                DatasetType.CLEANED.value,
                DatasetType.SYNTHETIC.value,
                DatasetType.GOLD_DEV.value,
            ],
            adapter_id=adapter_id,
            adapter_config=adapter_config,
            field_mapping=field_mapping,
            task_profile=resolved_task_profile,
        )
        result["succeeded"] = True
        result["method"] = "prepared_sources"
        result["manifest"] = manifest
        return result
    except ValueError as exc:
        await db.rollback()
        result["error"] = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001
        await db.rollback()
        result["error"] = str(exc)
        return result


def _build_guardrail_reason_codes(
    *,
    blockers: list[str],
    warnings: list[str],
) -> list[str]:
    reason_codes: list[str] = []
    reason_codes.extend([_guardrail_reason_code(item, blocker=True) for item in blockers])
    reason_codes.extend([_guardrail_reason_code(item, blocker=False) for item in warnings])
    return _dedupe_preserve(reason_codes)


def _build_guardrail_unblock_actions(
    *,
    project_id: int,
    blockers: list[str],
    dataset_readiness: dict[str, object],
    intent_clarification: dict[str, object],
    target_compatibility: dict[str, object] | None,
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []

    auto_fixes = [item for item in list(dataset_readiness.get("auto_fixes") or []) if isinstance(item, dict)]
    for idx, fix in enumerate(auto_fixes):
        label = str(fix.get("label") or fix.get("id") or f"Open suggested fix {idx + 1}").strip()
        if not label:
            continue
        navigate_to = str(fix.get("navigate_to") or "").strip()
        action: dict[str, object] = {
            "reason_code": "DATASET_NOT_READY",
            "label": label,
            "description": str(fix.get("description") or "").strip() or "Open guided remediation.",
            "navigate_to": navigate_to or f"/project/{project_id}/pipeline/ingestion",
            "one_click_available": bool(navigate_to),
        }
        actions.append(action)

    has_dataset_blocker = any(_guardrail_reason_code(item, blocker=True) == "DATASET_NOT_READY" for item in blockers)
    if has_dataset_blocker and not auto_fixes:
        actions.append(
            {
                "reason_code": "DATASET_NOT_READY",
                "label": "Complete data prep",
                "description": "Import and prepare a training split before launching Autopilot.",
                "navigate_to": f"/project/{project_id}/pipeline/ingestion",
                "one_click_available": True,
            }
        )

    if target_compatibility is not None and not bool(target_compatibility.get("compatible", False)):
        reasons = [
            str(item).strip()
            for item in list(target_compatibility.get("reasons") or [])
            if str(item).strip()
        ]
        actions.append(
            {
                "reason_code": "TARGET_INCOMPATIBLE",
                "label": "Change target or base model",
                "description": reasons[0] if reasons else "Selected base model does not fit target constraints.",
                "navigate_to": f"/project/{project_id}/wizard",
                "one_click_available": True,
            }
        )

    if bool(intent_clarification.get("required", False)):
        rewrite_rows = [item for item in list(intent_clarification.get("rewrite_suggestions") or []) if isinstance(item, dict)]
        if rewrite_rows:
            top = dict(rewrite_rows[0])
            rewritten_intent = str(top.get("rewritten_intent") or "").strip()
            actions.append(
                {
                    "reason_code": "INTENT_NEEDS_CLARIFICATION",
                    "label": str(top.get("label") or "Apply suggested rewrite").strip() or "Apply suggested rewrite",
                    "description": str(top.get("reason") or "Clarifies intent and expected outputs.").strip(),
                    "rewritten_intent": rewritten_intent or None,
                    "one_click_available": bool(rewritten_intent),
                }
            )

    deduped: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str]] = set()
    for action in actions:
        reason_code = str(action.get("reason_code") or "").strip() or "GUARDRAIL_BLOCKED"
        label = str(action.get("label") or "").strip() or "Open remediation"
        dedupe_key = (reason_code, label)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        action["reason_code"] = reason_code
        action["label"] = label
        deduped.append(action)
    return deduped


def _build_target_incompatibility_actions(
    *,
    project_id: int,
    target_profile_id: str,
    compatibility: dict[str, Any],
) -> list[dict[str, str]]:
    reasons = [
        str(item).strip()
        for item in list(compatibility.get("reasons") or [])
        if str(item).strip()
    ]
    vram_status = str((compatibility.get("vram_check") or {}).get("status") or "").strip().lower()
    actions: list[dict[str, str]] = []

    if vram_status == "blocked" or any("vram" in reason.lower() for reason in reasons):
        actions.append(
            {
                "id": "use_higher_memory_target",
                "label": "Use higher-memory target",
                "description": (
                    f"Switch target profile from '{target_profile_id}' to a higher-memory option "
                    "such as 'vllm_server'."
                ),
            }
        )
        actions.append(
            {
                "id": "choose_smaller_or_more_quantized_model",
                "label": "Choose smaller/quantized base model",
                "description": (
                    "Pick a base model with lower estimated minimum VRAM or apply stronger quantization "
                    "before training/export."
                ),
            }
        )

    actions.append(
        {
            "id": "review_target_compatibility_report",
            "label": "Review compatibility report",
            "description": "Inspect model metadata and target constraints to resolve incompatibility reasons.",
        }
    )
    actions.append(
        {
            "id": "open_wizard_target_step",
            "label": "Update target in wizard",
            "description": f"Open /project/{project_id}/wizard and re-select target/base model pairing.",
        }
    )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for action in actions:
        label = str(action.get("label") or "").strip()
        action_id = str(action.get("id") or "").strip()
        if not label or not action_id:
            continue
        key = (action_id, label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def _append_autopilot_decision(
    decision_log: list[dict[str, object]],
    *,
    step: str,
    status: str,
    summary: str,
    changed: bool = False,
    safe: bool = True,
    why: str | None = None,
    before: dict[str, object] | None = None,
    after: dict[str, object] | None = None,
    blocker: bool = False,
    fixes: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    entry: dict[str, object] = {
        "step": str(step or "").strip() or "unknown_step",
        "status": str(status or "").strip() or "info",
        "summary": str(summary or "").strip() or "Autopilot step updated.",
        "changed": bool(changed),
        "safe": bool(safe),
        "blocker": bool(blocker),
    }
    if why:
        entry["why"] = str(why).strip()
    if before:
        entry["before"] = dict(before)
    if after:
        entry["after"] = dict(after)
    if fixes:
        entry["fixes"] = [dict(item) for item in fixes if isinstance(item, dict)]
    if metadata:
        entry["metadata"] = dict(metadata)
    decision_log.append(entry)


# ---------------------------------------------------------------------------
# Strict-mode refusal reason codes (priority.md P5).
#
# When `settings.STRICT_EXECUTION_MODE` is true, the autopilot must NOT silently
# apply any of its auto-repairs. Instead, each repair that would have fired
# becomes a blocker with one of these reason codes, so operators can see
# exactly why the plan is unrunnable and what to fix.
# ---------------------------------------------------------------------------

STRICT_MODE_REFUSED_INTENT_REWRITE = "STRICT_MODE_REFUSED_INTENT_REWRITE"
STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE = "STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE"
STRICT_MODE_REFUSED_TARGET_FALLBACK = "STRICT_MODE_REFUSED_TARGET_FALLBACK"
STRICT_MODE_REFUSED_PROFILE_AUTOTUNE = "STRICT_MODE_REFUSED_PROFILE_AUTOTUNE"

_STRICT_REFUSAL_MESSAGES: dict[str, str] = {
    STRICT_MODE_REFUSED_INTENT_REWRITE: (
        "Strict mode refused an autopilot intent rewrite. Supply "
        "`intent_rewrite` explicitly or disable strict mode."
    ),
    STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE: (
        "Strict mode refused automatic dataset preparation. Prepare the "
        "dataset splits manually before retrying."
    ),
    STRICT_MODE_REFUSED_TARGET_FALLBACK: (
        "Strict mode refused to swap target profile. Choose a compatible "
        "target explicitly or disable strict mode."
    ),
    STRICT_MODE_REFUSED_PROFILE_AUTOTUNE: (
        "Strict mode refused to auto-tune the plan profile. Choose a runnable "
        "plan profile explicitly or disable strict mode."
    ),
}


def _merge_strict_mode_refusals(
    launch_guardrails: dict[str, object],
    refusals: list[str],
) -> dict[str, object]:
    """Append strict-mode refusal reason codes + blocker messages to guardrails.

    Mutates `launch_guardrails` for consistency with the surrounding call sites
    and returns it for chainability. When any refusal is present, forces
    `can_run=False` in the merged dict.
    """
    if not refusals:
        return launch_guardrails
    blockers = [
        str(item).strip()
        for item in list(launch_guardrails.get("blockers") or [])
        if str(item).strip()
    ]
    reason_codes = [
        str(item).strip()
        for item in list(launch_guardrails.get("reason_codes") or [])
        if str(item).strip()
    ]
    for code in refusals:
        message = _STRICT_REFUSAL_MESSAGES.get(code, f"Strict mode refused {code}.")
        if message not in blockers:
            blockers.append(message)
        if code not in reason_codes:
            reason_codes.append(code)
    launch_guardrails["blockers"] = _dedupe_preserve(blockers)
    launch_guardrails["reason_codes"] = _dedupe_preserve(reason_codes)
    launch_guardrails["can_run"] = False
    return launch_guardrails


def _choose_target_fallback_id(current_target_profile_id: str) -> str | None:
    current = str(current_target_profile_id or "").strip().lower() or "vllm_server"
    preferred = ("vllm_server", "edge_gpu", "mobile_cpu", "browser_webgpu")
    for target_id in preferred:
        if target_id == current:
            continue
        if get_target_by_id(target_id) is not None:
            return target_id
    for row in list_targets():
        candidate = str(getattr(row, "id", "") or "").strip().lower()
        if candidate and candidate != current:
            return candidate
    return None


def _has_target_incompatibility_blocker(
    *,
    guardrails: dict[str, object],
    compatibility: dict[str, object] | None,
) -> bool:
    reason_codes = {
        str(item).strip().upper()
        for item in list(guardrails.get("reason_codes") or [])
        if str(item).strip()
    }
    if "TARGET_INCOMPATIBLE" in reason_codes:
        return True
    if isinstance(compatibility, dict) and not bool(compatibility.get("compatible", False)):
        return True
    blockers = [
        str(item).strip().lower()
        for item in list(guardrails.get("blockers") or [])
        if str(item).strip()
    ]
    return any("compatib" in item or ("target" in item and "vram" in item) for item in blockers)


def _first_runnable_plan(plans: list[dict[str, object]]) -> dict[str, object] | None:
    for plan in plans:
        preflight = dict(plan.get("preflight") or {})
        if bool(preflight.get("ok", False)):
            return plan
    return None


def _plan_profile_token(value: Any, *, default: str = "balanced") -> str:
    return normalize_training_plan_profile(value) or str(default or "balanced")


def _contains_simulation_guardrail_signal(guardrails: dict[str, object]) -> bool:
    combined = [
        str(item).strip().lower()
        for item in (
            list(guardrails.get("blockers") or [])
            + list(guardrails.get("warnings") or [])
        )
        if str(item).strip()
    ]
    for message in combined:
        if "simulate" in message or "simulated" in message or "stub" in message:
            return True
    return False


async def _orchestrate_newbie_autopilot_v2(
    *,
    db: AsyncSession,
    project_id: int,
    req: AutopilotV2OrchestrationRequest,
) -> AutopilotV2OrchestrationResponse:
    await _get_project_or_404(db, project_id)

    run_id = str(uuid.uuid4())
    decision_log: list[dict[str, object]] = []
    try:
        return await _orchestrate_newbie_autopilot_v2_impl(
            db=db,
            project_id=project_id,
            req=req,
            run_id=run_id,
            decision_log=decision_log,
        )
    finally:
        if decision_log:
            await persist_decision_log(
                run_id=run_id,
                project_id=project_id,
                dry_run=bool(req.dry_run),
                intent=str(req.intent or ""),
                entries=decision_log,
            )


async def _orchestrate_newbie_autopilot_v2_impl(
    *,
    db: AsyncSession,
    project_id: int,
    req: AutopilotV2OrchestrationRequest,
    run_id: str,
    decision_log: list[dict[str, object]],
) -> AutopilotV2OrchestrationResponse:
    strict_mode = bool(settings.STRICT_EXECUTION_MODE)
    _append_autopilot_decision(
        decision_log,
        step="strict_mode_policy",
        status="active" if strict_mode else "inactive",
        summary=(
            "Strict execution mode is enabled; simulated/stub fallback paths are blocked."
            if strict_mode
            else "Strict execution mode is disabled; safe local automation is permitted."
        ),
        safe=True,
        metadata={"strict_mode": strict_mode},
    )

    readiness = await get_project_readiness(project_id)
    readiness_status = str(readiness.get("status") or "unknown").strip().lower() or "unknown"
    _append_autopilot_decision(
        decision_log,
        step="runtime_readiness",
        status=readiness_status,
        summary=(
            "Runtime readiness checks completed."
            if readiness_status in {"pass", "warn"}
            else "Runtime readiness has failing checks."
        ),
        blocker=readiness_status == "fail",
        fixes=[
            {
                "label": str(item.get("name") or item.get("id") or "Check readiness item"),
                "description": str(item.get("fix") or item.get("message") or "").strip(),
            }
            for item in list(readiness.get("checks") or [])
            if isinstance(item, dict) and str(item.get("fix") or "").strip()
        ],
        metadata={"status": readiness_status},
    )

    effective_intent = str(req.intent or "").strip()
    effective_target_profile_id = str(req.target_profile_id or "vllm_server").strip().lower() or "vllm_server"
    resolved_target_device = _resolve_autopilot_target_device(
        target_profile_id=effective_target_profile_id,
        target_device=req.target_device,
    )
    requested_profile = _plan_profile_token(req.plan_profile, default="balanced")
    selected_profile = requested_profile

    repairs: dict[str, object] = {
        "intent_rewrite": {
            "applied": False,
            "original_intent": effective_intent,
            "rewritten_intent": None,
            "source": None,
        },
        "dataset_auto_prepare": None,
        "target_fallback": {
            "applied": False,
            "from_target_profile_id": effective_target_profile_id,
            "to_target_profile_id": None,
            "reason": None,
        },
        "profile_autotune": {
            "applied": False,
            "from_profile": requested_profile,
            "to_profile": requested_profile,
            "reason": None,
        },
    }

    # Reason codes accumulated when strict mode refuses an otherwise-silent
    # auto-repair. Merged into the final `launch_guardrails` at the end.
    strict_mode_refusals: list[str] = []

    effective_req = NewbieAutopilotIntentRequest(
        intent=effective_intent,
        target_profile_id=effective_target_profile_id,
        target_device=resolved_target_device,
        primary_language=req.primary_language,
        available_vram_gb=req.available_vram_gb,
        base_model=req.base_model,
        auto_prepare_data=req.auto_prepare_data,
    )

    try:
        plan_v2_resp = await _build_newbie_autopilot_plan_v2(
            db=db,
            project_id=project_id,
            req=effective_req,
        )
    except HTTPException as exc:
        detail_payload = dict(exc.detail or {}) if isinstance(exc.detail, dict) else {}
        message = str(
            detail_payload.get("message")
            or detail_payload.get("detail")
            or exc.detail
            or "Autopilot planning failed."
        ).strip() or "Autopilot planning failed."
        actionable_fix = str(
            detail_payload.get("actionable_fix")
            or "Review runtime readiness and training configuration, then retry."
        ).strip()
        reason_code = str(detail_payload.get("error_code") or "TRAINING_PREFLIGHT_FAILED").strip() or "TRAINING_PREFLIGHT_FAILED"
        unblock_actions = [
            {
                "reason_code": reason_code,
                "label": "Resolve planning blocker",
                "description": actionable_fix,
                "one_click_available": False,
            }
        ]
        guardrails = {
            "can_run": False,
            "blockers": [message],
            "warnings": [],
            "reason_codes": _dedupe_preserve([reason_code, "STRICT_EXECUTION_MODE" if strict_mode else ""]),
            "unblock_actions": unblock_actions,
            "one_click_fix_available": False,
        }
        _append_autopilot_decision(
            decision_log,
            step="initial_planning",
            status="blocked",
            summary="Autopilot planning failed before repairs could run.",
            changed=False,
            safe=True,
            blocker=True,
            why=message,
            fixes=unblock_actions,
            metadata={"error_code": reason_code},
        )
        _append_autopilot_decision(
            decision_log,
            step="final_guardrails",
            status="blocked",
            summary="Autopilot stopped because planning prerequisites are not satisfied.",
            changed=False,
            safe=True,
            blocker=True,
            why=message,
            fixes=unblock_actions,
        )
        return AutopilotV2OrchestrationResponse(
            project_id=project_id,
            run_id=run_id,
            dry_run=bool(req.dry_run),
            strict_mode=strict_mode,
            intent=effective_intent,
            effective_target_profile_id=effective_target_profile_id,
            resolved_target_device=resolved_target_device,
            selected_profile=selected_profile,
            guardrails=guardrails,
            readiness=dict(readiness or {}),
            decision_log=decision_log,
            repairs=repairs,
            plan_v2={},
            experiment=None,
            started=False,
            start_result=None,
            start_error=f"Autopilot blocked run: {message}",
        )
    _append_autopilot_decision(
        decision_log,
        step="initial_planning",
        status="completed",
        summary="Built initial autopilot plan v2 with guardrail diagnostics.",
        safe=True,
        metadata={
            "requested_profile": requested_profile,
            "can_run": bool((plan_v2_resp.guardrails or {}).get("can_run", False)),
        },
    )

    rewrite_intent = str(req.intent_rewrite or "").strip()
    rewrite_source = "request.intent_rewrite"
    if not rewrite_intent and bool(req.auto_apply_rewrite):
        clarification = dict(plan_v2_resp.intent_clarification or {})
        rewrite_rows = [item for item in list(clarification.get("rewrite_suggestions") or []) if isinstance(item, dict)]
        if rewrite_rows:
            top = dict(rewrite_rows[0])
            rewrite_intent = str(top.get("rewritten_intent") or "").strip()
            rewrite_source = str(top.get("id") or "intent_clarification.rewrite_suggestions[0]")

    if rewrite_intent and rewrite_intent.lower() != effective_intent.lower():
        # Strict mode refuses autopilot-suggested rewrites (user-supplied
        # rewrites, identified by `rewrite_source == "request.intent_rewrite"`,
        # are still honored — the user made them explicit).
        if strict_mode and rewrite_source != "request.intent_rewrite":
            repairs["intent_rewrite"] = {
                "applied": False,
                "strict_mode_blocked": True,
                "original_intent": effective_intent,
                "proposed_intent": rewrite_intent,
                "source": rewrite_source,
                "reason_code": STRICT_MODE_REFUSED_INTENT_REWRITE,
            }
            strict_mode_refusals.append(STRICT_MODE_REFUSED_INTENT_REWRITE)
            _append_autopilot_decision(
                decision_log,
                step="intent_rewrite",
                status="blocked",
                summary="Strict mode refused an autopilot-suggested intent rewrite.",
                changed=False,
                safe=True,
                blocker=True,
                why="Strict mode requires the caller to supply intent_rewrite explicitly.",
                before={"intent": effective_intent},
                after={"intent": rewrite_intent},
                metadata={
                    "reason_code": STRICT_MODE_REFUSED_INTENT_REWRITE,
                    "source": rewrite_source,
                },
            )
        else:
            previous_intent = effective_intent
            effective_intent = rewrite_intent
            repairs["intent_rewrite"] = {
                "applied": True,
                "original_intent": previous_intent,
                "rewritten_intent": effective_intent,
                "source": rewrite_source,
            }
            effective_req = NewbieAutopilotIntentRequest(
                intent=effective_intent,
                target_profile_id=effective_target_profile_id,
                target_device=resolved_target_device,
                primary_language=req.primary_language,
                available_vram_gb=req.available_vram_gb,
                base_model=req.base_model,
                auto_prepare_data=req.auto_prepare_data,
            )
            plan_v2_resp = await _build_newbie_autopilot_plan_v2(
                db=db,
                project_id=project_id,
                req=effective_req,
            )
            _append_autopilot_decision(
                decision_log,
                step="intent_rewrite",
                status="applied",
                summary="Applied intent rewrite suggestion and rebuilt autopilot plan.",
                changed=True,
                safe=True,
                why="Clarifies task output for higher confidence planning.",
                before={"intent": previous_intent},
                after={"intent": effective_intent},
            )
    else:
        _append_autopilot_decision(
            decision_log,
            step="intent_rewrite",
            status="skipped",
            summary="Intent rewrite was not required.",
            safe=True,
        )

    launch_guardrails = dict(plan_v2_resp.guardrails or {})
    blockers = [
        str(item).strip()
        for item in list(launch_guardrails.get("blockers") or [])
        if str(item).strip()
    ]
    can_run = bool(launch_guardrails.get("can_run", False))
    auto_prepare_result: dict[str, object] | None = None

    if (
        not can_run
        and bool(req.auto_prepare_data)
        and _has_dataset_guardrail_blocker(blockers)
        and strict_mode
    ):
        # Strict mode refuses to silently generate or prepare dataset artifacts
        # — the operator must prepare splits manually.
        repairs["dataset_auto_prepare"] = {
            "applied": False,
            "strict_mode_blocked": True,
            "reason_code": STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE,
            "reason": "Strict mode refused automatic dataset preparation.",
        }
        strict_mode_refusals.append(STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE)
        _append_autopilot_decision(
            decision_log,
            step="dataset_auto_prepare",
            status="blocked",
            summary="Strict mode refused automatic dataset preparation.",
            changed=False,
            safe=True,
            blocker=True,
            why="Strict mode requires datasets to be prepared manually before training.",
            metadata={"reason_code": STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE},
        )
    elif (
        not can_run
        and bool(req.auto_prepare_data)
        and _has_dataset_guardrail_blocker(blockers)
    ):
        plan_options_for_hint = [dict(item) for item in list(plan_v2_resp.plans or []) if isinstance(item, dict)]
        selected_plan_for_hint = next(
            (
                p
                for p in plan_options_for_hint
                if _plan_profile_token(p.get("profile"), default="balanced") == requested_profile
            ),
            plan_options_for_hint[0] if plan_options_for_hint else {},
        )
        config_hint = dict(selected_plan_for_hint.get("config") or {})
        task_profile_hint = str(config_hint.get("task_profile") or "").strip() or None
        auto_prepare_result = await _attempt_autopilot_auto_prepare_data(
            db=db,
            project_id=project_id,
            task_profile=task_profile_hint,
        )
        repairs["dataset_auto_prepare"] = dict(auto_prepare_result)
        if bool(auto_prepare_result.get("succeeded")):
            plan_v2_resp = await _build_newbie_autopilot_plan_v2(
                db=db,
                project_id=project_id,
                req=effective_req,
            )
            launch_guardrails = dict(plan_v2_resp.guardrails or {})
            can_run = bool(launch_guardrails.get("can_run", False))
            blockers = [
                str(item).strip()
                for item in list(launch_guardrails.get("blockers") or [])
                if str(item).strip()
            ]
            _append_autopilot_decision(
                decision_log,
                step="dataset_auto_prepare",
                status="applied",
                summary="Automatically prepared dataset artifacts and rebuilt plan.",
                changed=True,
                safe=True,
                why="Dataset readiness blockers were detected.",
                metadata={
                    "method": str(auto_prepare_result.get("method") or ""),
                    "succeeded": True,
                },
            )
        else:
            _append_autopilot_decision(
                decision_log,
                step="dataset_auto_prepare",
                status="failed",
                summary="Automatic dataset preparation did not resolve dataset blockers.",
                changed=False,
                safe=True,
                blocker=True,
                why=str(auto_prepare_result.get("error") or "Dataset auto-prepare failed."),
            )
    else:
        _append_autopilot_decision(
            decision_log,
            step="dataset_auto_prepare",
            status="skipped",
            summary="Dataset auto-prepare was not required.",
            safe=True,
        )

    compatibility = dict(plan_v2_resp.target_compatibility or {})
    if (
        not can_run
        and bool(req.allow_target_fallback)
        and _has_target_incompatibility_blocker(
            guardrails=launch_guardrails,
            compatibility=compatibility,
        )
        and strict_mode
    ):
        # Strict mode refuses to silently swap the user's chosen target.
        repairs["target_fallback"] = {
            "applied": False,
            "strict_mode_blocked": True,
            "from_target_profile_id": effective_target_profile_id,
            "to_target_profile_id": None,
            "reason": "Strict mode refused to swap target profile.",
            "reason_code": STRICT_MODE_REFUSED_TARGET_FALLBACK,
        }
        strict_mode_refusals.append(STRICT_MODE_REFUSED_TARGET_FALLBACK)
        _append_autopilot_decision(
            decision_log,
            step="target_fallback",
            status="blocked",
            summary="Strict mode refused automatic target-profile fallback.",
            changed=False,
            safe=True,
            blocker=True,
            why=(
                "Strict mode requires the caller to choose a compatible target "
                "explicitly rather than accepting a silent downgrade."
            ),
            metadata={
                "reason_code": STRICT_MODE_REFUSED_TARGET_FALLBACK,
                "requested_target_profile_id": effective_target_profile_id,
            },
        )
    elif (
        not can_run
        and bool(req.allow_target_fallback)
        and _has_target_incompatibility_blocker(
            guardrails=launch_guardrails,
            compatibility=compatibility,
        )
    ):
        fallback_target_id = _choose_target_fallback_id(effective_target_profile_id)
        if fallback_target_id:
            fallback_req = NewbieAutopilotIntentRequest(
                intent=effective_intent,
                target_profile_id=fallback_target_id,
                target_device=req.target_device,
                primary_language=req.primary_language,
                available_vram_gb=req.available_vram_gb,
                base_model=req.base_model,
                auto_prepare_data=req.auto_prepare_data,
            )
            fallback_plan_v2 = await _build_newbie_autopilot_plan_v2(
                db=db,
                project_id=project_id,
                req=fallback_req,
            )
            fallback_guardrails = dict(fallback_plan_v2.guardrails or {})
            fallback_can_run = bool(fallback_guardrails.get("can_run", False))
            if fallback_can_run:
                previous_target = effective_target_profile_id
                effective_target_profile_id = fallback_target_id
                resolved_target_device = _resolve_autopilot_target_device(
                    target_profile_id=effective_target_profile_id,
                    target_device=req.target_device,
                )
                effective_req = fallback_req
                plan_v2_resp = fallback_plan_v2
                launch_guardrails = fallback_guardrails
                can_run = fallback_can_run
                blockers = [
                    str(item).strip()
                    for item in list(launch_guardrails.get("blockers") or [])
                    if str(item).strip()
                ]
                repairs["target_fallback"] = {
                    "applied": True,
                    "from_target_profile_id": previous_target,
                    "to_target_profile_id": effective_target_profile_id,
                    "reason": "Target incompatibility persisted after safe model repair.",
                }
                _append_autopilot_decision(
                    decision_log,
                    step="target_fallback",
                    status="applied",
                    summary="Switched to a fallback target profile and rebuilt plan.",
                    changed=True,
                    safe=True,
                    why="Initial target constraints blocked a runnable plan.",
                    before={"target_profile_id": previous_target},
                    after={"target_profile_id": effective_target_profile_id},
                )
            else:
                _append_autopilot_decision(
                    decision_log,
                    step="target_fallback",
                    status="skipped",
                    summary="Fallback target profile did not produce a runnable plan.",
                    changed=False,
                    safe=True,
                    why=f"Candidate target '{fallback_target_id}' is still blocked.",
                )
        else:
            _append_autopilot_decision(
                decision_log,
                step="target_fallback",
                status="skipped",
                summary="No fallback target profile was available.",
                changed=False,
                safe=True,
            )
    else:
        _append_autopilot_decision(
            decision_log,
            step="target_fallback",
            status="skipped",
            summary="Target fallback was not required.",
            safe=True,
        )

    plan_options = [dict(item) for item in list(plan_v2_resp.plans or []) if isinstance(item, dict)]
    requested_plan = next(
        (plan for plan in plan_options if _plan_profile_token(plan.get("profile")) == requested_profile),
        None,
    )
    selected_plan = requested_plan or (plan_options[0] if plan_options else None)
    if (
        bool(req.allow_profile_autotune)
        and selected_plan is not None
        and not bool(dict(selected_plan.get("preflight") or {}).get("ok", False))
    ):
        runnable = _first_runnable_plan(plan_options)
        if runnable is not None and strict_mode:
            # Strict mode refuses to silently downgrade the user's chosen
            # profile to a different runnable option.
            previous_profile = _plan_profile_token(selected_plan.get("profile"))
            candidate_profile = _plan_profile_token(runnable.get("profile"))
            repairs["profile_autotune"] = {
                "applied": False,
                "strict_mode_blocked": True,
                "from_profile": previous_profile,
                "to_profile": candidate_profile,
                "reason": "Strict mode refused to auto-tune the plan profile.",
                "reason_code": STRICT_MODE_REFUSED_PROFILE_AUTOTUNE,
            }
            strict_mode_refusals.append(STRICT_MODE_REFUSED_PROFILE_AUTOTUNE)
            _append_autopilot_decision(
                decision_log,
                step="profile_autotune",
                status="blocked",
                summary="Strict mode refused automatic plan-profile autotune.",
                changed=False,
                safe=True,
                blocker=True,
                why=(
                    "Strict mode requires the caller to pick a runnable profile "
                    "explicitly rather than accepting a silent downgrade."
                ),
                before={"profile": previous_profile},
                after={"profile": candidate_profile},
                metadata={"reason_code": STRICT_MODE_REFUSED_PROFILE_AUTOTUNE},
            )
        elif runnable is not None:
            previous_profile = _plan_profile_token(selected_plan.get("profile"))
            selected_plan = dict(runnable)
            selected_profile = _plan_profile_token(selected_plan.get("profile"))
            repairs["profile_autotune"] = {
                "applied": True,
                "from_profile": previous_profile,
                "to_profile": selected_profile,
                "reason": "Requested profile failed preflight; switched to a runnable profile.",
            }
            _append_autopilot_decision(
                decision_log,
                step="profile_autotune",
                status="applied",
                summary="Switched plan profile to the first preflight-passing option.",
                changed=True,
                safe=True,
                why="Requested profile failed preflight checks.",
                before={"profile": previous_profile},
                after={"profile": selected_profile},
            )
        else:
            _append_autopilot_decision(
                decision_log,
                step="profile_autotune",
                status="blocked",
                summary="No runnable preflight profile is available for automatic tuning.",
                changed=False,
                safe=True,
                blocker=True,
            )
    else:
        selected_profile = _plan_profile_token(
            (selected_plan or {}).get("profile"),
            default=requested_profile,
        )
        _append_autopilot_decision(
            decision_log,
            step="profile_autotune",
            status="skipped",
            summary="Requested profile is already usable or profile autotune is disabled.",
            safe=True,
        )

    strict_simulation_conflict = bool(
        _contains_simulation_guardrail_signal(launch_guardrails)
        or str(settings.TRAINING_BACKEND or "").strip().lower() == "simulate"
    )
    if strict_mode and strict_simulation_conflict:
        reason_codes = [
            str(item).strip()
            for item in list(launch_guardrails.get("reason_codes") or [])
            if str(item).strip()
        ]
        if "STRICT_EXECUTION_MODE" not in reason_codes:
            reason_codes.append("STRICT_EXECUTION_MODE")
        launch_guardrails["reason_codes"] = _dedupe_preserve(reason_codes)
        launch_guardrails["can_run"] = False
        can_run = False
        _append_autopilot_decision(
            decision_log,
            step="strict_mode_guardrail",
            status="blocked",
            summary="Strict mode blocked simulated/stub fallback behavior.",
            changed=False,
            safe=True,
            blocker=True,
            why="Provide live runtime dependencies or disable strict mode for demo-only workflows.",
        )

    launch_guardrails = dict(plan_v2_resp.guardrails or {}) | {
        "can_run": bool(can_run),
        "blockers": [
            str(item).strip()
            for item in list((plan_v2_resp.guardrails or {}).get("blockers") or [])
            if str(item).strip()
        ],
        "warnings": [
            str(item).strip()
            for item in list((plan_v2_resp.guardrails or {}).get("warnings") or [])
            if str(item).strip()
        ],
        "reason_codes": _dedupe_preserve(
            [
                str(item).strip()
                for item in list((plan_v2_resp.guardrails or {}).get("reason_codes") or [])
                if str(item).strip()
            ]
        ),
    }

    # Merge strict-mode refusals recorded by each repair block into the final
    # guardrails so every refused auto-repair surfaces as a stable blocker.
    if strict_mode_refusals:
        launch_guardrails = _merge_strict_mode_refusals(
            launch_guardrails, strict_mode_refusals
        )
        can_run = False

    strict_simulation_conflict = bool(
        _contains_simulation_guardrail_signal(launch_guardrails)
        or str(settings.TRAINING_BACKEND or "").strip().lower() == "simulate"
    )
    if strict_mode and strict_simulation_conflict:
        reason_codes = [str(item).strip() for item in list(launch_guardrails.get("reason_codes") or []) if str(item).strip()]
        launch_guardrails["reason_codes"] = _dedupe_preserve(reason_codes + ["STRICT_EXECUTION_MODE"])
        launch_guardrails["can_run"] = False
        can_run = False

    blockers = [
        str(item).strip()
        for item in list(launch_guardrails.get("blockers") or [])
        if str(item).strip()
    ]
    if not can_run:
        _append_autopilot_decision(
            decision_log,
            step="final_guardrails",
            status="blocked",
            summary="Autopilot stopped after safe repairs because blockers remain.",
            changed=False,
            safe=True,
            blocker=True,
            why=blockers[0] if blockers else "Guardrails did not pass.",
            fixes=[item for item in list(launch_guardrails.get("unblock_actions") or []) if isinstance(item, dict)],
        )
    else:
        _append_autopilot_decision(
            decision_log,
            step="final_guardrails",
            status="ready",
            summary="Autopilot guardrails passed after orchestration.",
            changed=True,
            safe=True,
        )

    plan_payload = plan_v2_resp.model_dump()
    if req.dry_run:
        return AutopilotV2OrchestrationResponse(
            project_id=project_id,
            run_id=run_id,
            dry_run=True,
            strict_mode=strict_mode,
            intent=effective_intent,
            effective_target_profile_id=effective_target_profile_id,
            resolved_target_device=resolved_target_device,
            selected_profile=selected_profile,
            guardrails=launch_guardrails,
            readiness=dict(readiness or {}),
            decision_log=decision_log,
            repairs=repairs,
            plan_v2=plan_payload,
            experiment=None,
            started=False,
            start_result=None,
            start_error=None if can_run else f"Autopilot blocked run: {blockers[0] if blockers else 'Unknown blocker.'}",
        )

    if not can_run:
        return AutopilotV2OrchestrationResponse(
            project_id=project_id,
            run_id=run_id,
            dry_run=False,
            strict_mode=strict_mode,
            intent=effective_intent,
            effective_target_profile_id=effective_target_profile_id,
            resolved_target_device=resolved_target_device,
            selected_profile=selected_profile,
            guardrails=launch_guardrails,
            readiness=dict(readiness or {}),
            decision_log=decision_log,
            repairs=repairs,
            plan_v2=plan_payload,
            experiment=None,
            started=False,
            start_result=None,
            start_error=f"Autopilot blocked run: {blockers[0] if blockers else 'Unknown blocker.'}",
        )

    selected_plan_final = next(
        (
            plan
            for plan in plan_options
            if _plan_profile_token(plan.get("profile"), default=selected_profile) == selected_profile
        ),
        _first_runnable_plan(plan_options) or (plan_options[0] if plan_options else None),
    )
    if selected_plan_final is None:
        raise HTTPException(400, "Autopilot could not select a runnable plan.")

    safe_config = dict(selected_plan_final.get("config") or {})
    safe_config["training_plan_profile"] = selected_profile
    safe_config["target_profile_id"] = effective_target_profile_id
    if not str(safe_config.get("task_profile") or "").strip():
        first_plan_cfg = dict(plan_options[0].get("config") or {}) if plan_options else {}
        safe_config["task_profile"] = str(first_plan_cfg.get("task_profile") or "instruction_sft").strip() or "instruction_sft"
    safe_config["_autopilot_dataset_rows"] = (
        _coerce_positive_int((plan_v2_resp.dataset_readiness or {}).get("prepared_row_count"))
        or _coerce_positive_int(safe_config.get("_autopilot_dataset_rows"))
        or 1000
    )
    base_model = str(safe_config.get("base_model") or "").strip() or "microsoft/phi-2"
    training_mode_token = str(safe_config.get("training_mode") or "sft").strip().lower()
    if training_mode_token == TrainingMode.DPO.value:
        training_mode = TrainingMode.DPO
    elif training_mode_token == TrainingMode.ORPO.value:
        training_mode = TrainingMode.ORPO
    else:
        training_mode = TrainingMode.SFT

    suggested_name = str(selected_plan_final.get("title") or "").strip()
    if not suggested_name:
        suggested_name = f"Autopilot - {selected_profile.replace('_', ' ').title()}"
    run_name = str(req.run_name or "").strip() or suggested_name or "Autopilot Run"
    description = str(req.description or "").strip() or f"Autopilot v2 run from intent: {effective_intent[:120]}"

    # Capture a pre-change snapshot so the upcoming experiment launch can be
    # rolled back via /autopilot/rollback/{decision_id}. The snapshot is keyed
    # to the sequence the `start_training` decision will occupy (current length
    # of decision_log), so rollback can resolve snapshot from decision id.
    start_training_sequence = len(decision_log)
    project_pre_config = await capture_project_snapshot(db, project_id=project_id)
    snapshot_id = await capture_snapshot(
        run_id=run_id,
        project_id=project_id,
        decision_sequence=start_training_sequence,
        snapshot_type="autopilot_training_launch",
        pre_state={
            "project": project_pre_config,
            "selected_profile": selected_profile,
            "safe_config": safe_config,
        },
        rollback_actions=[
            {"kind": "cancel_experiment"},
            {
                "kind": "restore_project_config",
                "fields": [
                    "base_model_name",
                    "target_profile_id",
                    "training_preferred_plan_profile",
                    "evaluation_preferred_pack_id",
                    "dataset_adapter_preset",
                    "active_domain_blueprint_version",
                ],
            },
        ],
    )

    exp = await create_experiment(
        db,
        project_id,
        run_name,
        base_model,
        safe_config,
        description,
        training_mode,
    )
    experiment_payload = ExperimentResponse.model_validate(exp).model_dump()
    started = False
    start_result: dict[str, object] | None = None
    start_error = ""
    try:
        start_result = await start_training(db, project_id, exp.id)
        started = True
        _append_autopilot_decision(
            decision_log,
            step="start_training",
            status="completed",
            summary="Created experiment and started training successfully.",
            changed=True,
            safe=True,
            metadata={
                "experiment_id": int(exp.id),
                "rollback_available": bool(snapshot_id),
                "snapshot_id": int(snapshot_id) if snapshot_id else None,
            },
        )
    except ValueError as exc:
        start_error = str(exc)
        _append_autopilot_decision(
            decision_log,
            step="start_training",
            status="failed",
            summary="Experiment was created, but training start failed.",
            changed=False,
            safe=True,
            blocker=True,
            why=start_error,
            metadata={
                "experiment_id": int(exp.id),
                "rollback_available": bool(snapshot_id),
                "snapshot_id": int(snapshot_id) if snapshot_id else None,
            },
        )

    # Record what actually happened so the rollback service knows which
    # experiment/task to unwind. Best-effort: a persistence error here does
    # not break the orchestration response.
    await update_snapshot_post_state(
        snapshot_id,
        post_state={
            "experiment_id": int(exp.id),
            "training_started": bool(started),
            "task_id": str((start_result or {}).get("task_id") or "") or None,
            "start_error": start_error or None,
        },
    )

    return AutopilotV2OrchestrationResponse(
        project_id=project_id,
        run_id=run_id,
        dry_run=False,
        strict_mode=strict_mode,
        intent=effective_intent,
        effective_target_profile_id=effective_target_profile_id,
        resolved_target_device=resolved_target_device,
        selected_profile=selected_profile,
        guardrails=launch_guardrails,
        readiness=dict(readiness or {}),
        decision_log=decision_log,
        repairs=repairs,
        plan_v2=plan_payload,
        experiment=experiment_payload,
        started=started,
        start_result=start_result,
        start_error=start_error or None,
    )


def _build_newbie_autopilot_plan(
    *,
    project_id: int,
    req: NewbieAutopilotIntentRequest,
) -> dict[str, object]:
    resolved_target_device = _resolve_autopilot_target_device(
        target_profile_id=req.target_profile_id,
        target_device=req.target_device,
    )
    plan = resolve_newbie_autopilot_intent(
        intent=req.intent,
        target_profile_id=req.target_profile_id,
        primary_language=req.primary_language,
        available_vram_gb=req.available_vram_gb,
    )
    task_profile = str(plan.get("task_profile") or "instruction_sft")
    recommendation = recommend_training_base_models(
        target_device=resolved_target_device,
        primary_language=req.primary_language,
        available_vram_gb=req.available_vram_gb,
        task_profile=task_profile,
        top_k=1,
    )
    top_recommendation = None
    recommendation_rows = list(recommendation.get("recommendations") or [])
    if recommendation_rows and isinstance(recommendation_rows[0], dict):
        top_recommendation = dict(recommendation_rows[0])

    requested_base_model = str(req.base_model or "").strip()
    safe_config = dict(plan.get("safe_training_config") or {})
    if requested_base_model:
        safe_config["base_model"] = requested_base_model
    elif top_recommendation is not None:
        model_id = str(top_recommendation.get("model_id") or "").strip()
        if model_id:
            safe_config["base_model"] = model_id
        suggested_defaults = dict(top_recommendation.get("suggested_defaults") or {})
        # Keep task_type from autopilot preset as authoritative unless it is missing.
        if not str(safe_config.get("task_type") or "").strip():
            suggested_task_type = str(suggested_defaults.get("task_type") or "").strip()
            if suggested_task_type:
                safe_config["task_type"] = suggested_task_type
        for key in ("chat_template", "batch_size", "max_seq_length"):
            if key in suggested_defaults and suggested_defaults[key] is not None:
                safe_config[key] = suggested_defaults[key]

    preflight = run_training_preflight(
        project_id=project_id,
        config=safe_config,
        base_model=str(safe_config.get("base_model") or ""),
    )
    confidence_value = float(plan.get("confidence") or 0.0)
    matched_keywords = [str(item) for item in list(plan.get("matched_keywords") or [])]
    intent_clarification = build_newbie_autopilot_intent_clarification(
        intent=req.intent,
        confidence=confidence_value,
        task_profile=task_profile,
        matched_keywords=matched_keywords,
    )
    dataset_readiness = evaluate_newbie_autopilot_dataset_readiness(project_id=project_id, min_rows=20)
    guardrail_blockers: list[str] = []
    guardrail_warnings: list[str] = []
    guardrail_blockers.extend([str(item) for item in list(dataset_readiness.get("blockers") or [])])
    if not bool(preflight.get("ok", False)):
        preflight_errors = [
            str(item).strip()
            for item in list(preflight.get("errors") or [])
            if str(item).strip()
        ]
        if preflight_errors:
            guardrail_blockers.append(f"Training preflight failed: {preflight_errors[0]}")
    preflight_warnings = [
        str(item).strip()
        for item in list(preflight.get("warnings") or [])
        if str(item).strip()
    ]
    guardrail_warnings.extend([str(item) for item in list(dataset_readiness.get("warnings") or [])])
    guardrail_warnings.extend(preflight_warnings[:3])
    if bool(intent_clarification.get("required", False)):
        reason = str(intent_clarification.get("reason") or "").strip()
        if reason:
            guardrail_warnings.append(f"Intent clarification recommended: {reason}.")

    reason_codes = _build_guardrail_reason_codes(
        blockers=guardrail_blockers,
        warnings=guardrail_warnings,
    )
    unblock_actions = _build_guardrail_unblock_actions(
        project_id=project_id,
        blockers=guardrail_blockers,
        dataset_readiness=dataset_readiness,
        intent_clarification=intent_clarification,
        target_compatibility=None,
    )

    can_one_click_run = len(guardrail_blockers) == 0
    launch_guardrails = {
        "can_one_click_run": can_one_click_run,
        "blockers": guardrail_blockers,
        "warnings": guardrail_warnings,
        "reason_codes": reason_codes,
        "unblock_actions": unblock_actions,
    }
    return {
        "plan": plan,
        "resolved_target_device": resolved_target_device,
        "intent_clarification": intent_clarification,
        "dataset_readiness": dataset_readiness,
        "launch_guardrails": launch_guardrails,
        "safe_training_config": safe_config,
        "model_recommendation": None if requested_base_model else top_recommendation,
        "recommendation_context": recommendation.get("request"),
        "preflight": preflight,
    }


async def _build_newbie_autopilot_plan_v2(
    *,
    db: AsyncSession,
    project_id: int,
    req: NewbieAutopilotIntentRequest,
) -> AutopilotPlanV2Response:
    resolved_target_device = _resolve_autopilot_target_device(
        target_profile_id=req.target_profile_id,
        target_device=req.target_device,
    )
    # 1. Resolve basic intent preset
    plan_meta = resolve_newbie_autopilot_intent(
        intent=req.intent,
        target_profile_id=req.target_profile_id,
        primary_language=req.primary_language,
        available_vram_gb=req.available_vram_gb,
    )
    task_profile = str(plan_meta.get("task_profile") or "instruction_sft")
    run_history = await _load_autopilot_run_history(
        db,
        project_id=project_id,
        fallback_target_profile_id=req.target_profile_id,
    )

    # 2. Get base model recommendation
    recommendation = recommend_training_base_models(
        target_device=resolved_target_device,
        primary_language=req.primary_language,
        available_vram_gb=req.available_vram_gb,
        task_profile=task_profile,
        top_k=5,
    )
    top_recommendation = None
    recommendation_rows = [
        dict(item)
        for item in list(recommendation.get("recommendations") or [])
        if isinstance(item, dict)
    ]
    if recommendation_rows and isinstance(recommendation_rows[0], dict):
        top_recommendation = dict(recommendation_rows[0])

    requested_base_model = str(req.base_model or "").strip()
    if requested_base_model:
        model_id = requested_base_model
    else:
        model_id = (
            str(top_recommendation.get("model_id") or "microsoft/phi-2")
            if top_recommendation
            else "microsoft/phi-2"
        )
    repaired_model_id, repaired_compatibility, auto_repair = _attempt_compatibility_auto_repair(
        model_id=model_id,
        target_profile_id=req.target_profile_id,
        recommendations=recommendation_rows,
    )
    model_id = repaired_model_id

    # 3. Build base config for preflight plan
    base_config = dict(plan_meta.get("safe_training_config") or {})
    base_config["base_model"] = model_id
    effective_recommendation = _find_recommendation_by_model_id(recommendation_rows, model_id) or top_recommendation
    auto_repair_applied = bool(auto_repair and auto_repair.get("applied"))
    if effective_recommendation and (not requested_base_model or auto_repair_applied):
        suggested_defaults = dict(effective_recommendation.get("suggested_defaults") or {})
        for key in ("chat_template", "batch_size", "max_seq_length"):
            if key in suggested_defaults and suggested_defaults[key] is not None:
                base_config[key] = suggested_defaults[key]

    # 4. Generate all 3 profile suggestions
    preflight_plan = run_training_preflight_plan(
        project_id=project_id,
        config=base_config,
        base_model=model_id,
    )

    # 5. Enrich suggestions with estimates and labels
    dataset_readiness = evaluate_newbie_autopilot_dataset_readiness(project_id=project_id)
    dataset_rows = _coerce_positive_int(dataset_readiness.get("prepared_row_count")) or 1000

    enriched_plans = []
    for suggestion in preflight_plan.get("suggestions", []):
        profile = suggestion["profile"]
        normalized_profile = normalize_training_plan_profile(profile) or "balanced"
        config = dict(suggestion.get("config") or {})
        config["training_plan_profile"] = normalized_profile
        config["target_profile_id"] = str(req.target_profile_id or "vllm_server").strip() or "vllm_server"
        config["task_profile"] = task_profile
        config["_autopilot_dataset_rows"] = int(dataset_rows)
        suggestion["config"] = config
        estimate = estimate_newbie_autopilot_run(
            plan_profile=normalized_profile,
            target_profile_id=req.target_profile_id,
            dataset_size_rows=dataset_rows,
            task_profile=task_profile,
            run_history=run_history,
        )
        suggestion["estimate"] = estimate
        enriched_plans.append(suggestion)

    # 6. Build Guardrails
    guardrail_blockers = []
    guardrail_warnings = []
    
    # Target compatibility check
    compatibility = dict(repaired_compatibility)
    if auto_repair_applied:
        compatibility["auto_repair"] = dict(auto_repair or {})
        original_model_id = str(auto_repair.get("original_model_id") or "").strip()
        repaired_model_id = str(auto_repair.get("repaired_model_id") or "").strip()
        if original_model_id and repaired_model_id and original_model_id != repaired_model_id:
            incompatibility_reasons = [
                str(item).strip()
                for item in list(auto_repair.get("incompatibility_reasons") or [])
                if str(item).strip()
            ]
            reason_suffix = f" Reason: {incompatibility_reasons[0]}" if incompatibility_reasons else ""
            guardrail_warnings.append(
                (
                    f"Base model '{original_model_id}' was incompatible with target "
                    f"'{req.target_profile_id}'. Autopilot switched to '{repaired_model_id}'."
                    f"{reason_suffix}"
                )
            )

    if not compatibility.get("compatible"):
        guardrail_blockers.extend(
            compatibility.get("reasons") or ["Model is not compatible with target hardware constraints."]
        )
    compatibility_warnings = [
        str(item).strip()
        for item in list(compatibility.get("warnings") or [])
        if str(item).strip()
    ]
    if compatibility_warnings:
        guardrail_warnings.extend(compatibility_warnings)
    
    guardrail_blockers.extend([str(item) for item in list(dataset_readiness.get("blockers") or [])])
    guardrail_warnings.extend([str(item) for item in list(dataset_readiness.get("warnings") or [])])

    intent_clarification = build_newbie_autopilot_intent_clarification(
        intent=req.intent,
        confidence=float(plan_meta.get("confidence") or 0.0),
        task_profile=task_profile,
        matched_keywords=[str(k) for k in list(plan_meta.get("matched_keywords") or [])],
    )
    if bool(intent_clarification.get("required", False)):
        reason = str(intent_clarification.get("reason") or "").strip()
        if reason:
            guardrail_warnings.append(f"Intent clarification recommended: {reason}.")

    any_runnable_plan = any(p.get("preflight", {}).get("ok", False) for p in enriched_plans)
    if not any_runnable_plan:
        guardrail_blockers.append("Training preflight failed: no generated plan passed preflight checks.")

    # Check if any plan is actually runnable
    can_run = any_runnable_plan and not guardrail_blockers
    reason_codes = _build_guardrail_reason_codes(
        blockers=[str(item) for item in guardrail_blockers],
        warnings=[str(item) for item in guardrail_warnings],
    )
    unblock_actions = _build_guardrail_unblock_actions(
        project_id=project_id,
        blockers=[str(item) for item in guardrail_blockers],
        dataset_readiness=dataset_readiness,
        intent_clarification=intent_clarification,
        target_compatibility=compatibility,
    )

    return AutopilotPlanV2Response(
        project_id=project_id,
        intent=req.intent,
        resolved_target_device=resolved_target_device,
        plans=enriched_plans,
        recommended_profile=preflight_plan.get("recommended_profile", "balanced"),
        target_compatibility=compatibility,
        guardrails={
            "can_run": can_run,
            "blockers": guardrail_blockers,
            "warnings": guardrail_warnings,
            "reason_codes": reason_codes,
            "unblock_actions": unblock_actions,
            "one_click_fix_available": any(bool(item.get("one_click_available")) for item in unblock_actions),
        },
        dataset_readiness=dataset_readiness,
        intent_clarification=intent_clarification,
    )


@router.get("/runtimes")
async def get_training_runtime_catalog(
    project_id: int,
):
    """List registered training runtime plugins and server default runtime."""
    return {
        "project_id": project_id,
        **list_runtime_catalog(),
    }


@router.get("/runtimes/plugins/status")
async def get_training_runtime_plugin_status(
    project_id: int,
):
    """Read runtime plugin loader status (modules/errors/count)."""
    return {
        "project_id": project_id,
        **runtime_plugin_status(),
    }


@router.post("/runtimes/plugins/reload")
async def reload_training_runtime_plugins(
    project_id: int,
):
    """Reload training runtime plugins from settings and return fresh catalog."""
    return {
        "project_id": project_id,
        **reload_runtime_plugins_from_settings(),
    }


@router.get("/capability-contract")
async def get_training_capability_contract(
    project_id: int,
):
    """Return shared task/backend/modality capability contract metadata."""
    return {
        "project_id": project_id,
        **build_training_capability_contract(),
    }


@router.post("/autopilot/intent-resolve")
async def resolve_training_autopilot_intent(
    project_id: int,
    req: NewbieAutopilotIntentRequest,
    db: AsyncSession = Depends(get_db),
):
    """Map plain-language user intent to a safe starter training preset."""
    await _get_project_or_404(db, project_id)
    try:
        payload = _build_newbie_autopilot_plan(project_id=project_id, req=req)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "project_id": project_id,
        **payload,
    }


@router.post("/autopilot/plan-v2", response_model=AutopilotPlanV2Response)
async def resolve_training_autopilot_plan_v2(
    project_id: int,
    req: NewbieAutopilotIntentRequest,
    db: AsyncSession = Depends(get_db),
):
    """Return three plan options (Fastest, Balanced, Best Quality) with cost/time estimates."""
    await _get_project_or_404(db, project_id)
    try:
        return await _build_newbie_autopilot_plan_v2(
            db=db,
            project_id=project_id,
            req=req,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/autopilot/estimate")
async def estimate_training_autopilot_run(
    project_id: int,
    req: NewbieAutopilotEstimateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Estimate training time and cost for a selected plan profile and hardware."""
    await _get_project_or_404(db, project_id)
    plan_profile = normalize_training_plan_profile(req.plan_profile) or "balanced"
    run_history = await _load_autopilot_run_history(
        db,
        project_id=project_id,
        fallback_target_profile_id=req.target_profile_id,
    )
    return {
        "project_id": project_id,
        "estimate": estimate_newbie_autopilot_run(
            plan_profile=plan_profile,
            target_profile_id=req.target_profile_id,
            dataset_size_rows=req.dataset_size_rows,
            task_profile=req.task_profile,
            run_history=run_history,
        ),
    }


@router.post("/autopilot/one-click-run")
async def run_training_autopilot_one_click(
    project_id: int,
    req: NewbieAutopilotOneClickRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create and start a training experiment from plain-language intent in one click."""
    await _get_project_or_404(db, project_id)

    effective_intent = str(req.intent).strip()
    resolved_target_device = _resolve_autopilot_target_device(
        target_profile_id=req.target_profile_id,
        target_device=req.target_device,
    )
    effective_req = NewbieAutopilotIntentRequest(
        intent=effective_intent,
        target_profile_id=req.target_profile_id,
        target_device=resolved_target_device,
        primary_language=req.primary_language,
        available_vram_gb=req.available_vram_gb,
        base_model=req.base_model,
        auto_prepare_data=req.auto_prepare_data,
    )
    applied_intent_rewrite: dict[str, object] = {
        "applied": False,
        "original_intent": effective_intent,
        "rewritten_intent": None,
        "source": None,
    }

    # Use v2 logic to get the right plan
    try:
        plan_v2_resp = await _build_newbie_autopilot_plan_v2(
            db=db,
            project_id=project_id,
            req=effective_req,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    rewrite_intent = str(req.intent_rewrite or "").strip()
    rewrite_source = "request.intent_rewrite"
    if not rewrite_intent and bool(req.auto_apply_rewrite):
        clarification = plan_v2_resp.intent_clarification
        rewrite_rows = list(clarification.get("rewrite_suggestions") or [])
        if rewrite_rows and isinstance(rewrite_rows[0], dict):
            top = dict(rewrite_rows[0])
            rewrite_intent = str(top.get("rewritten_intent") or "").strip()
            rewrite_source = str(top.get("id") or "intent_clarification.rewrite_suggestions[0]")

    if rewrite_intent and rewrite_intent.lower() != effective_intent.lower():
        effective_intent = rewrite_intent
        effective_req = NewbieAutopilotIntentRequest(
            intent=effective_intent,
            target_profile_id=req.target_profile_id,
            target_device=resolved_target_device,
            primary_language=req.primary_language,
            available_vram_gb=req.available_vram_gb,
            base_model=req.base_model,
            auto_prepare_data=req.auto_prepare_data,
        )
        try:
            plan_v2_resp = await _build_newbie_autopilot_plan_v2(
                db=db,
                project_id=project_id,
                req=effective_req,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        applied_intent_rewrite = {
            "applied": True,
            "original_intent": str(req.intent).strip(),
            "rewritten_intent": effective_intent,
            "source": rewrite_source,
        }

    plan_payload = plan_v2_resp.model_dump()
    requested_profile = normalize_training_plan_profile(req.plan_profile) or "balanced"
    launch_guardrails = plan_v2_resp.guardrails
    auto_prepare_result: dict[str, object] | None = None
    can_one_click_run = bool(launch_guardrails.get("can_run", False))
    blockers = [
        str(item).strip()
        for item in list(launch_guardrails.get("blockers") or [])
        if str(item).strip()
    ]

    if (
        not can_one_click_run
        and bool(req.auto_prepare_data)
        and _has_dataset_guardrail_blocker(blockers)
    ):
        plan_options_for_hint = [dict(item) for item in list(plan_v2_resp.plans or []) if isinstance(item, dict)]
        selected_plan_for_hint = next(
            (
                p
                for p in plan_options_for_hint
                if (normalize_training_plan_profile(p.get("profile")) or str(p.get("profile") or "").strip().lower())
                == requested_profile
            ),
            plan_options_for_hint[0] if plan_options_for_hint else {},
        )
        config_hint = dict(selected_plan_for_hint.get("config") or {})
        task_profile_hint = str(config_hint.get("task_profile") or "").strip() or None
        auto_prepare_result = await _attempt_autopilot_auto_prepare_data(
            db=db,
            project_id=project_id,
            task_profile=task_profile_hint,
        )
        if bool(auto_prepare_result.get("succeeded")):
            try:
                plan_v2_resp = await _build_newbie_autopilot_plan_v2(
                    db=db,
                    project_id=project_id,
                    req=effective_req,
                )
            except ValueError as e:
                raise HTTPException(400, str(e))
            plan_payload = plan_v2_resp.model_dump()
            launch_guardrails = plan_v2_resp.guardrails
            can_one_click_run = bool(launch_guardrails.get("can_run", False))
            blockers = [
                str(item).strip()
                for item in list(launch_guardrails.get("blockers") or [])
                if str(item).strip()
            ]

    if not can_one_click_run:
        error_preview = blockers[0] if blockers else "Autopilot guardrails blocked this run."
        if auto_prepare_result is not None and not bool(auto_prepare_result.get("succeeded")):
            prep_error = str(auto_prepare_result.get("error") or "").strip()
            if prep_error:
                error_preview = f"{error_preview} Auto data prep failed: {prep_error}"
        return {
            "project_id": project_id,
            "experiment": None,
            "started": False,
            "start_result": None,
            "start_error": f"Autopilot blocked one-click run: {error_preview}",
            "applied_intent_rewrite": applied_intent_rewrite,
            "auto_prepare": auto_prepare_result,
            "plan_v2": plan_payload,
        }

    plan_options = [dict(item) for item in list(plan_v2_resp.plans or []) if isinstance(item, dict)]
    if not plan_options:
        raise HTTPException(400, "Autopilot could not generate a runnable plan.")

    selected_plan = next(
        (
            p
            for p in plan_options
            if (normalize_training_plan_profile(p.get("profile")) or str(p.get("profile") or "").strip().lower())
            == requested_profile
        ),
        plan_options[0],
    )
    selected_profile = (
        normalize_training_plan_profile(selected_plan.get("profile"))
        or requested_profile
        or "balanced"
    )
    safe_config = dict(selected_plan.get("config") or {})
    safe_config["training_plan_profile"] = selected_profile
    safe_config["target_profile_id"] = str(req.target_profile_id or "vllm_server").strip() or "vllm_server"
    if not str(safe_config.get("task_profile") or "").strip():
        first_plan_cfg = dict(plan_options[0].get("config") or {})
        safe_config["task_profile"] = str(first_plan_cfg.get("task_profile") or "instruction_sft").strip() or "instruction_sft"
    safe_config["_autopilot_dataset_rows"] = (
        _coerce_positive_int((plan_v2_resp.dataset_readiness or {}).get("prepared_row_count"))
        or _coerce_positive_int(safe_config.get("_autopilot_dataset_rows"))
        or 1000
    )
    base_model = str(safe_config.get("base_model") or "").strip() or "microsoft/phi-2"
    training_mode_token = str(safe_config.get("training_mode") or "sft").strip().lower()
    if training_mode_token == TrainingMode.DPO.value:
        training_mode = TrainingMode.DPO
    elif training_mode_token == TrainingMode.ORPO.value:
        training_mode = TrainingMode.ORPO
    else:
        training_mode = TrainingMode.SFT

    suggested_name = str(selected_plan.get("title") or "").strip()
    if not suggested_name:
        suggested_name = f"Autopilot - {selected_profile.replace('_', ' ').title()}"
    run_name = str(req.run_name or "").strip() or suggested_name or "Autopilot Run"
    description = str(req.description or "").strip() or (
        f"Autopilot run from intent: {effective_intent[:120]}"
    )

    try:
        exp = await create_experiment(
            db,
            project_id,
            run_name,
            base_model,
            safe_config,
            description,
            training_mode,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    experiment_payload = ExperimentResponse.model_validate(exp).model_dump()
    started = False
    start_result: dict[str, object] | None = None
    start_error = ""
    try:
        start_result = await start_training(db, project_id, exp.id)
        started = True
    except ValueError as e:
        start_error = str(e)

    return {
        "project_id": project_id,
        "experiment": experiment_payload,
        "started": started,
        "start_result": start_result,
        "start_error": start_error or None,
        "applied_intent_rewrite": applied_intent_rewrite,
        "auto_prepare": auto_prepare_result,
        "plan_v2": plan_payload,
    }


@router.post("/autopilot/v2/orchestrate", response_model=AutopilotV2OrchestrationResponse)
async def orchestrate_training_autopilot_v2(
    project_id: int,
    req: AutopilotV2OrchestrationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run autopilot orchestration with safe auto-repairs and optional dry-run mode."""
    await _get_project_or_404(db, project_id)
    try:
        return await _orchestrate_newbie_autopilot_v2(
            db=db,
            project_id=project_id,
            req=req,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/autopilot/v2/orchestrate/run", response_model=AutopilotV2OrchestrationResponse)
async def run_training_autopilot_v2(
    project_id: int,
    req: AutopilotV2OrchestrationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run autopilot orchestration and launch training when guardrails pass."""
    await _get_project_or_404(db, project_id)
    try:
        execute_req = req.model_copy(update={"dry_run": False})
        return await _orchestrate_newbie_autopilot_v2(
            db=db,
            project_id=project_id,
            req=execute_req,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/cloud-burst/catalog")
async def get_cloud_burst_catalog(
    project_id: int,
):
    """List supported cloud burst providers and GPU SKU options."""
    return {
        "project_id": project_id,
        **list_cloud_burst_catalog(),
    }


@router.post("/cloud-burst/quote")
async def get_cloud_burst_quote(
    project_id: int,
    req: CloudBurstQuoteRequest,
):
    """Estimate cloud burst lease cost for selected provider/GPU."""
    try:
        quote = estimate_cloud_burst_quote(
            provider_id=req.provider_id,
            gpu_sku=req.gpu_sku,
            duration_hours=req.duration_hours,
            storage_gb=req.storage_gb,
            egress_gb=req.egress_gb,
            spot=req.spot,
        )
        return {
            "project_id": project_id,
            **quote,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/cloud-burst/launch-plan")
async def get_cloud_burst_launch_plan(
    project_id: int,
    req: CloudBurstLaunchPlanRequest,
    db: AsyncSession = Depends(get_db),
):
    """Build provider-specific burst launch plan + credential readiness report."""
    await _get_project_or_404(db, project_id)
    try:
        plan = await build_cloud_burst_launch_plan(
            db,
            project_id=project_id,
            provider_id=req.provider_id,
            gpu_sku=req.gpu_sku,
            duration_hours=req.duration_hours,
            experiment_id=req.experiment_id,
            region=req.region,
            image=req.image,
            startup_script=req.startup_script,
            spot=req.spot,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return plan


@router.post("/cloud-burst/jobs/submit")
async def submit_managed_cloud_burst_job(
    project_id: int,
    req: CloudBurstJobSubmitRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit a managed cloud burst job and start lifecycle tracking."""
    await _get_project_or_404(db, project_id)
    try:
        return await submit_cloud_burst_job(
            db,
            project_id=project_id,
            provider_id=req.provider_id,
            gpu_sku=req.gpu_sku,
            duration_hours=req.duration_hours,
            experiment_id=req.experiment_id,
            region=req.region,
            image=req.image,
            startup_script=req.startup_script,
            spot=req.spot,
            auto_artifact_sync=req.auto_artifact_sync,
            artifact_sync_policy=req.artifact_sync_policy,
            artifact_include_globs=req.artifact_include_globs,
            artifact_exclude_globs=req.artifact_exclude_globs,
            execution_mode=req.execution_mode,
            allow_fallback_to_simulation=req.allow_fallback_to_simulation,
            idempotency_key=req.idempotency_key,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/cloud-burst/jobs")
async def list_managed_cloud_burst_jobs(
    project_id: int,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """List managed cloud burst jobs for a project."""
    await _get_project_or_404(db, project_id)
    return list_cloud_burst_jobs(project_id=project_id, limit=limit)


@router.get("/cloud-burst/jobs/{run_id}")
async def get_managed_cloud_burst_job_status(
    project_id: int,
    run_id: str,
    logs_tail: int = 200,
    db: AsyncSession = Depends(get_db),
):
    """Get one managed cloud burst job status + logs tail."""
    await _get_project_or_404(db, project_id)
    try:
        return get_cloud_burst_job_status(
            project_id=project_id,
            run_id=run_id,
            logs_tail=logs_tail,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/cloud-burst/jobs/{run_id}/logs")
async def get_managed_cloud_burst_job_logs(
    project_id: int,
    run_id: str,
    tail: int = 200,
    db: AsyncSession = Depends(get_db),
):
    """Read managed cloud burst job logs."""
    await _get_project_or_404(db, project_id)
    try:
        return get_cloud_burst_job_logs(
            project_id=project_id,
            run_id=run_id,
            tail=tail,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/cloud-burst/jobs/{run_id}/cancel")
async def cancel_managed_cloud_burst_job(
    project_id: int,
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Request cancellation for a managed cloud burst job."""
    await _get_project_or_404(db, project_id)
    try:
        return await cancel_cloud_burst_job(project_id=project_id, run_id=run_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/cloud-burst/jobs/{run_id}/sync-artifacts")
async def sync_managed_cloud_burst_job_artifacts(
    project_id: int,
    run_id: str,
    req: CloudBurstArtifactSyncRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Sync artifacts for a managed cloud burst job into project storage."""
    await _get_project_or_404(db, project_id)
    payload = req or CloudBurstArtifactSyncRequest()
    try:
        return sync_cloud_burst_job_artifacts(
            project_id=project_id,
            run_id=run_id,
            policy=payload.policy,
            include_globs=payload.include_globs,
            exclude_globs=payload.exclude_globs,
            dry_run=payload.dry_run,
            max_files=payload.max_files,
            cursor=payload.cursor,
            fail_if_missing_source=True,
            reason="manual_api",
        )
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/recipes")
async def get_training_recipe_catalog(
    project_id: int,
):
    """List built-in training recipes."""
    return {
        "project_id": project_id,
        "recipe_count": len(list_training_recipes(include_patch=False)),
        "recipes": list_training_recipes(include_patch=False),
    }


@router.post("/recipes/resolve")
async def resolve_training_recipe_config(
    project_id: int,
    req: TrainingRecipeResolveRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resolve recipe config with runtime defaults and optional preflight."""
    try:
        recipe_resolution = resolve_training_recipe(
            req.recipe_id,
            base_config=dict(req.base_config or {}),
            overrides=dict(req.overrides or {}),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    candidate_config = dict(recipe_resolution.get("resolved_config") or {})
    (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    ) = await _resolve_effective_training_preview(
        project_id=project_id,
        source_config=candidate_config,
        db=db,
    )

    preflight = None
    if bool(req.include_preflight):
        preflight = run_training_preflight(
            project_id=project_id,
            config=resolved_config,
            base_model=str(resolved_config.get("base_model", "")),
        )

    return {
        "project_id": project_id,
        "recipe": recipe_resolution.get("recipe"),
        "recipe_missing_required_fields": recipe_resolution.get("missing_required_fields", []),
        "recipe_config": candidate_config,
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "profile_training_defaults": (
            dict(profile_training_defaults) if profile_training_defaults else None
        ),
        "resolved_training_config": resolved_config,
        "resolved_training_mode": resolved_training_mode.value,
        "profile_defaults_applied": profile_defaults_applied,
        "preflight": preflight,
    }


@router.post("/model-selection/recommend")
async def recommend_training_models(
    project_id: int,
    req: ModelSelectionRecommendRequest,
    db: AsyncSession = Depends(get_db),
):
    """Recommend base models for a project based on hardware/task wizard inputs."""
    project_result = await db.execute(select(Project.id).where(Project.id == project_id))
    if project_result.scalar_one_or_none() is None:
        raise HTTPException(404, f"Project {project_id} not found")

    acceptance_bias = build_model_acceptance_bias(
        project_id,
        target_device=req.target_device,
        task_profile=req.task_profile,
    )
    benchmark_bias = build_model_benchmark_bias(
        project_id,
        target_device=req.target_device,
        task_profile=req.task_profile,
    )
    combined_bias: dict[str, float] = {}
    for source_bias in (
        dict(acceptance_bias.get("bias_by_model") or {}),
        dict(benchmark_bias.get("bias_by_model") or {}),
    ):
        for model_id, raw_value in source_bias.items():
            token = str(model_id or "").strip()
            if not token:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            combined_bias[token] = round(min(1.5, combined_bias.get(token, 0.0) + max(0.0, value)), 4)

    adaptive_label_parts: list[str] = []
    if combined_bias and acceptance_bias.get("enabled"):
        adaptive_label_parts.append(
            f"acceptance:{str(acceptance_bias.get('context_label') or 'global')}"
        )
    if combined_bias and benchmark_bias.get("enabled"):
        adaptive_label_parts.append(
            f"benchmark:{str(benchmark_bias.get('context_label') or 'global')}"
        )
    adaptive_label = " + ".join(adaptive_label_parts) if adaptive_label_parts else ""
    payload = recommend_training_base_models(
        target_device=req.target_device,
        primary_language=req.primary_language,
        available_vram_gb=req.available_vram_gb,
        task_profile=req.task_profile,
        top_k=req.top_k,
        adaptive_model_bias=combined_bias,
        adaptive_bias_label=adaptive_label,
    )
    return {
        "project_id": project_id,
        **payload,
        "adaptive_ranking": {
            "enabled": bool(combined_bias),
            "context_label": (
                str(acceptance_bias.get("context_label") or benchmark_bias.get("context_label") or "global")
            ),
            "global_apply_events": acceptance_bias.get("global_apply_events"),
            "context_apply_events": acceptance_bias.get("context_apply_events"),
            "benchmark_run_count": benchmark_bias.get("run_count"),
            "boosted_model_count": len(combined_bias),
            "acceptance": {
                "enabled": bool(acceptance_bias.get("enabled")),
                "context_label": acceptance_bias.get("context_label"),
                "global_apply_events": acceptance_bias.get("global_apply_events"),
                "context_apply_events": acceptance_bias.get("context_apply_events"),
                "boosted_model_count": len(dict(acceptance_bias.get("bias_by_model") or {})),
            },
            "benchmark": {
                "enabled": bool(benchmark_bias.get("enabled")),
                "context_label": benchmark_bias.get("context_label"),
                "run_count": benchmark_bias.get("run_count"),
                "boosted_model_count": len(dict(benchmark_bias.get("bias_by_model") or {})),
            },
        },
    }


@router.get("/model-selection/catalog")
async def get_model_selection_catalog(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Expose model catalog entries with source/version provenance metadata."""
    project_result = await db.execute(select(Project.id).where(Project.id == project_id))
    if project_result.scalar_one_or_none() is None:
        raise HTTPException(404, f"Project {project_id} not found")

    return {
        "project_id": project_id,
        **list_model_catalog(),
    }


@router.post("/model-selection/introspect")
async def introspect_training_model(
    project_id: int,
    req: ModelSelectionIntrospectRequest,
    db: AsyncSession = Depends(get_db),
):
    """Inspect a model id/path and return architecture/context/license/memory hints."""
    project_result = await db.execute(select(Project.id).where(Project.id == project_id))
    if project_result.scalar_one_or_none() is None:
        raise HTTPException(404, f"Project {project_id} not found")

    model_id = str(req.model_id or "").strip()
    if not model_id:
        raise HTTPException(400, "model_id is required")

    payload = introspect_training_base_model(
        model_id=model_id,
        allow_network=bool(req.allow_network),
    )
    return {
        "project_id": project_id,
        "introspection": payload,
    }


@router.post("/model-selection/benchmark-sweep")
async def model_selection_benchmark_sweep(
    project_id: int,
    req: ModelBenchmarkSweepRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run a short sampled benchmark sweep across top base-model candidates."""
    project_result = await db.execute(select(Project.id).where(Project.id == project_id))
    if project_result.scalar_one_or_none() is None:
        raise HTTPException(404, f"Project {project_id} not found")
    benchmark = benchmark_model_sweep(
        project_id=project_id,
        target_device=req.target_device,
        primary_language=req.primary_language,
        available_vram_gb=req.available_vram_gb,
        task_profile=req.task_profile,
        model_ids=list(req.model_ids or []),
        max_models=req.max_models,
        sample_size=req.sample_size,
        allow_network_tokenizer=bool(req.allow_network_tokenizer),
    )
    if not bool(req.persist_run):
        return benchmark

    persisted = record_model_benchmark_run(
        project_id,
        payload=benchmark,
    )
    return {
        **benchmark,
        "persisted": {
            "event_id": str(dict(persisted.get("event") or {}).get("event_id") or ""),
            "path": persisted.get("path"),
            "summary": persisted.get("summary"),
        },
    }


@router.get("/model-selection/benchmark-sweep/history")
async def model_selection_benchmark_sweep_history(
    project_id: int,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Return persisted benchmark sweep runs for model ranking diagnostics."""
    project_result = await db.execute(select(Project.id).where(Project.id == project_id))
    if project_result.scalar_one_or_none() is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return list_model_benchmark_runs(project_id, limit=limit)


@router.get("/model-selection/benchmark-sweep/summary")
async def model_selection_benchmark_sweep_summary(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return aggregate benchmark sweep metrics for the project."""
    project_result = await db.execute(select(Project.id).where(Project.id == project_id))
    if project_result.scalar_one_or_none() is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return summarize_model_benchmark_runs(project_id)


@router.post("/model-selection/telemetry")
async def model_selection_telemetry(
    project_id: int,
    req: ModelSelectionTelemetryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist model selection wizard telemetry event."""
    await _get_project_or_404(db, project_id)
    return record_model_wizard_event(
        project_id,
        payload=req.model_dump(),
    )


@router.get("/model-selection/telemetry")
async def model_selection_telemetry_summary(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return aggregated model selection wizard telemetry metrics."""
    await _get_project_or_404(db, project_id)
    return summarize_model_wizard_events(project_id)


@router.post("/observability/telemetry")
async def observability_telemetry_record(
    project_id: int,
    req: ObservabilityTelemetryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist training observability telemetry event (gradients/attention focus)."""
    await _get_project_or_404(db, project_id)
    return record_observability_event(
        project_id,
        payload=req.model_dump(),
    )


@router.get("/observability/telemetry")
async def observability_telemetry_summary(
    project_id: int,
    experiment_id: int | None = None,
    limit: int = 120,
    db: AsyncSession = Depends(get_db),
):
    """Return observability telemetry summary with recent events for debugging."""
    await _get_project_or_404(db, project_id)
    return {
        "summary": summarize_observability_events(project_id, experiment_id=experiment_id),
        "recent": list_observability_events(
            project_id,
            experiment_id=experiment_id,
            limit=limit,
        ),
    }


@router.get("/playground/providers")
async def playground_provider_catalog(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List supported playground runtime providers."""
    await _get_project_or_404(db, project_id)
    return {
        "project_id": project_id,
        **list_playground_provider_catalog(),
    }


@router.post("/playground/chat")
async def playground_chat(
    project_id: int,
    req: PlaygroundChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run a chat completion request for the project playground."""
    project = await _get_project_or_404(db, project_id)

    requested_model_name = str(req.model_name or project.base_model_name or "").strip() or "microsoft/phi-2"
    runtime_resolution = resolve_playground_model_runtime(
        model_name=requested_model_name,
        provider=req.provider,
    )
    runtime_provider = str(req.provider or "").strip() or "openai_compatible"
    if bool(req.auto_runtime_provider):
        recommended_provider = str(runtime_resolution.get("recommended_provider") or "").strip()
        if recommended_provider and runtime_provider in {"", "openai_compatible"}:
            runtime_provider = recommended_provider
    resolved_model_name = str(runtime_resolution.get("resolved_model_name") or requested_model_name)
    normalized_messages = normalize_playground_messages(
        messages=[item.model_dump() for item in req.messages],
        system_prompt=req.system_prompt,
    )
    if not normalized_messages:
        raise HTTPException(400, "At least one non-empty chat message is required.")
    try:
        result = await run_playground_chat(
            provider=runtime_provider,
            model_name=resolved_model_name,
            messages=normalized_messages,
            api_url=req.api_url,
            api_key=req.api_key,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            system_prompt=None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    session_payload = None
    if bool(req.save_history):
        transcript = list(normalized_messages)
        transcript.append({"role": "assistant", "content": str(result.get("reply") or "").strip()})
        try:
            session = await save_playground_session_transcript(
                db,
                project_id=project_id,
                session_id=req.session_id,
                title=req.session_title,
                provider=runtime_provider,
                model_name=requested_model_name,
                api_url=req.api_url,
                system_prompt=req.system_prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                transcript=transcript,
                metadata={
                    "message_count": len(req.messages),
                    "last_latency_ms": result.get("latency_ms"),
                    "requested_model_name": requested_model_name,
                    "resolved_model_name": resolved_model_name,
                    "runtime_hint": runtime_resolution.get("runtime_hint"),
                },
            )
            session_payload = serialize_playground_session_summary(session)
        except ValueError as e:
            raise HTTPException(400, str(e))

    return {
        "project_id": project_id,
        "message_count": len(req.messages),
        "normalized_message_count": len(normalized_messages),
        "session_id": session_payload.get("id") if isinstance(session_payload, dict) else None,
        "session_summary": session_payload,
        "requested_model_name": requested_model_name,
        "resolved_model_name": resolved_model_name,
        "resolved_provider": runtime_provider,
        "runtime_hint": runtime_resolution.get("runtime_hint"),
        **result,
    }


@router.post("/playground/chat/stream")
async def playground_chat_stream(
    project_id: int,
    req: PlaygroundChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Stream incremental chat completion chunks (SSE)."""
    project = await _get_project_or_404(db, project_id)
    requested_model_name = str(req.model_name or project.base_model_name or "").strip() or "microsoft/phi-2"
    runtime_resolution = resolve_playground_model_runtime(
        model_name=requested_model_name,
        provider=req.provider,
    )
    runtime_provider = str(req.provider or "").strip() or "openai_compatible"
    if bool(req.auto_runtime_provider):
        recommended_provider = str(runtime_resolution.get("recommended_provider") or "").strip()
        if recommended_provider and runtime_provider in {"", "openai_compatible"}:
            runtime_provider = recommended_provider
    resolved_model_name = str(runtime_resolution.get("resolved_model_name") or requested_model_name)
    normalized_messages = normalize_playground_messages(
        messages=[item.model_dump() for item in req.messages],
        system_prompt=req.system_prompt,
    )
    if not normalized_messages:
        raise HTTPException(400, "At least one non-empty chat message is required.")

    async def stream_events() -> AsyncIterator[str]:
        try:
            async for event in stream_playground_chat(
                provider=runtime_provider,
                model_name=resolved_model_name,
                messages=normalized_messages,
                api_url=req.api_url,
                api_key=req.api_key,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                system_prompt=None,
            ):
                event_type = str(event.get("type") or "").strip().lower()
                if event_type != "final":
                    yield _sse_json(event)
                    continue

                final_event = dict(event)
                session_payload = None
                if bool(req.save_history):
                    transcript = list(normalized_messages)
                    transcript.append(
                        {
                            "role": "assistant",
                            "content": str(final_event.get("reply") or "").strip(),
                        }
                    )
                    try:
                        session = await save_playground_session_transcript(
                            db,
                            project_id=project_id,
                            session_id=req.session_id,
                            title=req.session_title,
                            provider=runtime_provider,
                            model_name=requested_model_name,
                            api_url=req.api_url,
                            system_prompt=req.system_prompt,
                            temperature=req.temperature,
                            max_tokens=req.max_tokens,
                            transcript=transcript,
                            metadata={
                                "message_count": len(req.messages),
                                "last_latency_ms": final_event.get("latency_ms"),
                                "stream": True,
                                "requested_model_name": requested_model_name,
                                "resolved_model_name": resolved_model_name,
                                "runtime_hint": runtime_resolution.get("runtime_hint"),
                            },
                        )
                        session_payload = serialize_playground_session_summary(session)
                    except ValueError as e:
                        yield _sse_json({"type": "error", "detail": str(e)})
                        return

                final_event.update(
                    {
                        "project_id": project_id,
                        "message_count": len(req.messages),
                        "normalized_message_count": len(normalized_messages),
                        "session_id": session_payload.get("id") if isinstance(session_payload, dict) else None,
                        "session_summary": session_payload,
                        "requested_model_name": requested_model_name,
                        "resolved_model_name": resolved_model_name,
                        "resolved_provider": runtime_provider,
                        "runtime_hint": runtime_resolution.get("runtime_hint"),
                    }
                )
                yield _sse_json(final_event)
        except ValueError as e:
            yield _sse_json({"type": "error", "detail": str(e)})

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/playground/models")
async def playground_models(
    project_id: int,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List model candidates for playground picker from project/registry/artifacts."""
    try:
        return await list_playground_model_options(db, project_id=project_id, limit=limit)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/playground/sessions")
async def playground_session_list(
    project_id: int,
    limit: int = 30,
    db: AsyncSession = Depends(get_db),
):
    """List project playground sessions."""
    await _get_project_or_404(db, project_id)
    rows = await list_playground_sessions(db, project_id=project_id, limit=limit)
    sessions = [serialize_playground_session_summary(item) for item in rows]
    return {
        "project_id": project_id,
        "count": len(sessions),
        "sessions": sessions,
    }


@router.post("/playground/sessions", status_code=201)
async def create_playground_session(
    project_id: int,
    req: PlaygroundSessionCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a playground session with optional initial history."""
    project = await _get_project_or_404(db, project_id)
    model_name = str(req.model_name or project.base_model_name or "").strip() or "microsoft/phi-2"
    transcript = normalize_playground_messages(
        messages=[item.model_dump() for item in req.messages],
        system_prompt=req.system_prompt,
    )
    session = await save_playground_session_transcript(
        db,
        project_id=project_id,
        session_id=None,
        title=req.title,
        provider=req.provider,
        model_name=model_name,
        api_url=req.api_url,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        transcript=transcript,
        metadata={"created_manually": True},
    )
    return serialize_playground_session_detail(session)


@router.get("/playground/sessions/{session_id}")
async def playground_session_get(
    project_id: int,
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Read one playground session with full transcript."""
    await _get_project_or_404(db, project_id)
    session = await get_playground_session(db, project_id=project_id, session_id=session_id)
    if session is None:
        raise HTTPException(404, f"Playground session {session_id} not found")
    return serialize_playground_session_detail(session)


@router.delete("/playground/sessions/{session_id}", status_code=204)
async def playground_session_remove(
    project_id: int,
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete one playground session."""
    await _get_project_or_404(db, project_id)
    deleted = await delete_playground_session(db, project_id=project_id, session_id=session_id)
    if not deleted:
        raise HTTPException(404, f"Playground session {session_id} not found")


@router.post("/playground/logs")
async def playground_feedback_log(
    project_id: int,
    req: PlaygroundFeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist prompt/eval feedback log for playground responses."""
    await _get_project_or_404(db, project_id)
    return record_playground_feedback(project_id, payload=req.model_dump())


@router.get("/playground/logs")
async def playground_feedback_list(
    project_id: int,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Read recent playground feedback logs with aggregate summary."""
    await _get_project_or_404(db, project_id)
    logs = list_playground_feedback(project_id, limit=limit)
    return {
        **logs,
        "summary": summarize_playground_feedback(project_id),
    }


@router.post("/playground/rag-compare")
async def playground_rag_compare(
    project_id: int,
    req: PlaygroundRagCompareRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run side-by-side RAG answers for base model vs tuned model."""
    project = await _get_project_or_404(db, project_id)

    snippets = await retrieve_project_rag_snippets(
        db,
        project_id=project_id,
        query=req.query,
        top_k=req.top_k,
    )
    if not snippets:
        raise HTTPException(
            400,
            "No retrieval snippets found. Upload and ingest project documents before using RAG compare.",
        )

    context_block = build_rag_context_block(snippets)
    user_prompt = (
        "Use only the provided snippets. Cite snippet IDs like [s1]. "
        "If the answer is not present in snippets, say you do not have enough context.\n\n"
        f"SNIPPETS:\n{context_block}\n\n"
        f"QUESTION:\n{req.query}"
    )
    messages = [{"role": "user", "content": user_prompt}]

    provider = str(req.provider or "mock").strip() or "mock"
    base_model_name = str(req.base_model_name or project.base_model_name or "").strip() or "microsoft/phi-2"
    tuned_model_name = str(req.tuned_model_name or base_model_name).strip() or base_model_name

    try:
        base_response = await run_playground_chat(
            provider=provider,
            model_name=base_model_name,
            messages=messages,
            api_url=req.api_url,
            api_key=req.api_key,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            system_prompt=(
                "You are the base model baseline. Answer with concise factual style and cite snippet IDs."
            ),
        )
        tuned_response = await run_playground_chat(
            provider=provider,
            model_name=tuned_model_name,
            messages=messages,
            api_url=req.api_url,
            api_key=req.api_key,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            system_prompt=(
                "You are the fine-tuned model candidate. Answer with domain-aware detail and cite snippet IDs."
            ),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "project_id": project_id,
        "query": req.query,
        "provider": provider,
        "retrieved_snippets": snippets,
        "base": {
            "model_name": base_model_name,
            "reply": str(base_response.get("reply") or "").strip(),
            "latency_ms": base_response.get("latency_ms"),
        },
        "tuned": {
            "model_name": tuned_model_name,
            "reply": str(tuned_response.get("reply") or "").strip(),
            "latency_ms": tuned_response.get("latency_ms"),
        },
    }


@router.get("/alignment/recipes")
async def alignment_recipe_catalog(
    project_id: int,
):
    """List DPO/ORPO alignment recipe presets."""
    return {
        "project_id": project_id,
        "recipe_count": len(list_alignment_recipes(include_patch=False)),
        "recipes": list_alignment_recipes(include_patch=False),
    }


@router.post("/alignment/recipes/resolve")
async def alignment_recipe_resolve(
    project_id: int,
    req: AlignmentRecipeResolveRequest,
):
    """Resolve one alignment recipe into effective training config patch."""
    try:
        resolution = resolve_alignment_recipe(
            req.recipe_id,
            base_config=dict(req.base_config or {}),
            overrides=dict(req.overrides or {}),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "project_id": project_id,
        "recipe": resolution.get("recipe"),
        "resolved_config": resolution.get("resolved_config"),
    }


@router.post("/alignment/preference-contract/validate")
async def alignment_contract_validate(
    project_id: int,
    req: AlignmentContractValidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Validate prompt/chosen/rejected pair contract for DPO/ORPO data."""
    await _get_project_or_404(db, project_id)
    report = validate_preference_rows(
        [dict(item) for item in req.rows],
        min_coverage=req.min_coverage,
        max_rows=req.max_rows,
    )
    return {
        "project_id": project_id,
        **report,
    }


@router.post("/alignment/judge/score")
async def alignment_judge_score(
    project_id: int,
    req: AlignmentJudgeScoreRequest,
    db: AsyncSession = Depends(get_db),
):
    """Heuristic judge scoring for preference rows before DPO/ORPO training."""
    await _get_project_or_404(db, project_id)
    report = score_preference_rows(
        [dict(item) for item in req.rows],
        quality_threshold=req.quality_threshold,
        max_rows=req.max_rows,
    )
    return {
        "project_id": project_id,
        **report,
    }


@router.get("/alignment/preference-dataset")
async def alignment_preference_dataset_summary(
    project_id: int,
    sample_size: int = 400,
    quality_threshold: float = 3.0,
    source_path: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Inspect current preference dataset contract + quality summary."""
    await _get_project_or_404(db, project_id)
    try:
        report = summarize_preference_dataset(
            project_id,
            source_path=source_path,
            sample_size=sample_size,
            quality_threshold=quality_threshold,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return report


@router.post("/alignment/preference-dataset/import")
async def alignment_preference_dataset_import(
    project_id: int,
    req: AlignmentDatasetImportRequest,
    db: AsyncSession = Depends(get_db),
):
    """Import preference rows from JSON array/JSONL text into project dataset files."""
    await _get_project_or_404(db, project_id)
    try:
        report = import_preference_dataset_rows(
            project_id,
            rows=[dict(item) for item in req.rows],
            raw_text=req.raw_text,
            mode=req.mode,
            target=req.target,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return report


@router.post("/alignment/preference-dataset/filter")
async def alignment_preference_dataset_filter(
    project_id: int,
    req: AlignmentDatasetFilterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run judge-quality filtering and optionally apply filtered dataset to train split."""
    await _get_project_or_404(db, project_id)
    try:
        report = filter_preference_dataset_by_quality(
            project_id,
            quality_threshold=req.quality_threshold,
            min_keep_ratio=req.min_keep_ratio,
            apply_to_train_file=req.apply_to_train_file,
            source_path=req.source_path,
            target_path=req.target_path,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return report


@router.post("/alignment/retrain-from-feedback")
async def alignment_retrain_from_feedback(
    project_id: int,
    req: AlignmentRetrainFromFeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    1. Materialize playground pairs.
    2. Compose merged alignment dataset.
    3. Filter by quality.
    4. Create experiment with alignment recipe.
    5. Start training.
    """
    project = await _get_project_or_404(db, project_id)

    # 1. & 2. Materialize and Compose
    try:
        composed = compose_alignment_training_dataset(
            project_id,
            include_playground_pairs=req.include_playground_pairs,
            max_playground_pairs=req.max_playground_pairs,
        )
    except ValueError as e:
        raise HTTPException(400, f"Dataset composition failed: {e}")

    effective_train_path = composed.get("effective_train_path")
    if not effective_train_path:
        raise HTTPException(400, "No training dataset available after composition.")

    # 3. Filter by quality
    try:
        filtered = filter_preference_dataset_by_quality(
            project_id,
            quality_threshold=req.quality_threshold,
            source_path=effective_train_path,
        )
    except ValueError as e:
        raise HTTPException(400, f"Quality filtering failed: {e}")

    final_train_path = filtered.get("target_path")

    # 4. Resolve alignment recipe
    try:
        resolved = resolve_alignment_recipe(req.recipe_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

    recipe_config = resolved["resolved_config"]
    recipe_config["train_file"] = final_train_path
    recipe_config["base_model"] = project.base_model_name or "microsoft/phi-2"

    # 5. Create and Start Experiment
    experiment_data = ExperimentCreate(
        name=f"Retrain from Feedback - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
        description=f"Auto-retrain run using {composed.get('playground_rows', 0)} playground feedback pairs.",
        training_mode=TrainingMode(resolved["recipe"]["training_mode"]),
        config=recipe_config,
    )

    experiment = await create_experiment(
        db,
        project_id=project_id,
        experiment_data=experiment_data,
    )

    # Add metadata for audit trail
    experiment.metadata_ = {
        "source": "retrain_from_feedback",
        "playground_rows": composed.get("playground_rows", 0),
        "quality_threshold": req.quality_threshold,
        "recipe_id": req.recipe_id,
        "composition_report": composed,
    }
    await db.commit()

    await start_training(db, project_id=project_id, experiment_id=experiment.id)

    return {
        "experiment_id": experiment.id,
        "status": experiment.status,
        "playground_rows_included": composed.get("playground_rows", 0),
        "total_rows_after_filter": filtered.get("keep_count", 0),
        "composition_report": composed,
        "filter_report": filtered,
    }


@router.get("/alignment/active-learning")
async def alignment_active_learning_summary(
    project_id: int,
    refresh_pairs: bool = True,
    max_playground_pairs: int = 5000,
    db: AsyncSession = Depends(get_db),
):
    """Inspect active-learning artifacts derived from playground downvotes."""
    await _get_project_or_404(db, project_id)
    if bool(refresh_pairs):
        materialize_playground_preference_pairs(
            project_id,
            max_pairs=max_playground_pairs,
        )
    return summarize_playground_active_learning(project_id)


@router.post("/alignment/active-learning/compose")
async def alignment_active_learning_compose(
    project_id: int,
    req: AlignmentActiveLearningComposeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compose train preference dataset merged with auto-materialized playground pairs."""
    await _get_project_or_404(db, project_id)
    try:
        return compose_alignment_training_dataset(
            project_id,
            source_path=req.source_path,
            include_playground_pairs=req.include_playground_pairs,
            target_path=req.target_path,
            max_playground_pairs=req.max_playground_pairs,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


def _resolve_training_config(
    *,
    config_payload: dict,
    provided_config_fields: set[str],
    profile_training_defaults: dict,
    fallback_training_mode: TrainingMode,
) -> tuple[dict, TrainingMode, list[str]]:
    resolved_config = dict(config_payload)
    profile_defaults_applied: list[str] = []

    if profile_training_defaults:
        for key, value in profile_training_defaults.items():
            if key == "training_mode":
                continue
            if key not in provided_config_fields:
                resolved_config[key] = value
                profile_defaults_applied.append(key)

    resolved_training_mode = fallback_training_mode
    if (
        "training_mode" not in provided_config_fields
        and isinstance(profile_training_defaults.get("training_mode"), str)
    ):
        candidate_mode = str(profile_training_defaults["training_mode"]).strip().lower()
        try:
            resolved_training_mode = TrainingMode(candidate_mode)
            resolved_config["training_mode"] = resolved_training_mode.value
            profile_defaults_applied.append("training_mode")
        except ValueError:
            pass

    return resolved_config, resolved_training_mode, sorted(set(profile_defaults_applied))


def _normalize_preferred_plan_profile(value: str) -> str:
    candidate = normalize_training_plan_profile(value)
    if candidate in TRAINING_PLAN_PROFILES:
        return str(candidate)
    allowed = ", ".join(TRAINING_PLAN_PROFILES)
    raise HTTPException(
        400,
        f"Invalid preferred_plan_profile '{str(value or '').strip().lower()}'. Allowed values: {allowed}",
    )


async def _resolve_effective_training_preview(
    *,
    project_id: int,
    source_config: dict[str, object],
    db: AsyncSession,
) -> tuple[
    dict,
    dict,
    dict,
    TrainingMode,
    list[str],
]:
    provided_config_fields = set(source_config.keys())
    if "base_model" not in source_config:
        source_config["base_model"] = "microsoft/phi-2"

    try:
        parsed_config = TrainingConfig.model_validate(source_config)
    except Exception as e:
        raise HTTPException(400, f"Invalid training config: {e}")

    try:
        runtime = await resolve_project_domain_runtime(db, project_id)
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

    effective_contract = runtime.get("effective_contract")
    profile_training_defaults = get_training_defaults(effective_contract)
    resolved_config, resolved_training_mode, profile_defaults_applied = _resolve_training_config(
        config_payload=parsed_config.model_dump(),
        provided_config_fields=provided_config_fields,
        profile_training_defaults=profile_training_defaults,
        fallback_training_mode=parsed_config.training_mode,
    )
    return (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    )


@router.get("/preferences")
async def get_training_preferences(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Read persisted training UI/runtime preferences for a project."""
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    stored = str(project.training_preferred_plan_profile or "").strip().lower()
    normalized = normalize_training_plan_profile(stored)
    preferred = normalized if normalized in TRAINING_PLAN_PROFILES else "balanced"
    source = "project" if normalized in TRAINING_PLAN_PROFILES else "default"
    return {
        "project_id": project_id,
        "preferred_plan_profile": preferred,
        "profile_options": list(TRAINING_PLAN_PROFILES),
        "source": source,
    }


@router.put("/preferences")
async def set_training_preferences(
    project_id: int,
    req: TrainingPreferencesUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist training preferences for a project."""
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    preferred = _normalize_preferred_plan_profile(req.preferred_plan_profile)
    project.training_preferred_plan_profile = preferred
    await db.flush()

    return {
        "project_id": project_id,
        "preferred_plan_profile": preferred,
        "profile_options": list(TRAINING_PLAN_PROFILES),
        "source": "project",
    }


@router.get("/vibe-check/config")
async def get_vibe_check_config(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Read project-level vibe-check prompt configuration."""
    await _get_project_or_404(db, project_id)
    config = load_project_vibe_check_config(project_id)
    return {
        "project_id": project_id,
        **config,
    }


@router.put("/vibe-check/config")
async def set_vibe_check_config(
    project_id: int,
    req: VibeCheckConfigUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist project-level vibe-check prompt configuration."""
    await _get_project_or_404(db, project_id)
    updates = req.model_dump(exclude_none=True)
    config = save_project_vibe_check_config(project_id, updates)
    return {
        "project_id": project_id,
        **config,
    }


@router.get("/experiments/{experiment_id}/vibe-check/timeline")
async def vibe_check_timeline(
    project_id: int,
    experiment_id: int,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    """Read qualitative vibe-check snapshots captured during training."""
    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    exp = exp_result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found in project {project_id}")
    output_dir_token = str(exp.output_dir or "").strip()
    if not output_dir_token:
        raise HTTPException(400, "Experiment output directory is not available.")
    output_dir = Path(output_dir_token).expanduser()
    if not output_dir.exists():
        raise HTTPException(400, "Experiment output directory is not available.")
    timeline = load_vibe_check_timeline(output_dir, limit=limit)
    timeline["config"] = load_project_vibe_check_config(
        project_id,
        experiment_config=dict(exp.config or {}),
    )
    return {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "status": exp.status.value,
        **timeline,
    }


@router.post("/experiments/{experiment_id}/vibe-check/snapshot")
async def create_vibe_check_snapshot(
    project_id: int,
    experiment_id: int,
    req: VibeCheckSnapshotRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Capture and persist one vibe-check snapshot immediately."""
    payload = req or VibeCheckSnapshotRequest()
    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    exp = exp_result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found in project {project_id}")

    output_dir_token = str(exp.output_dir or "").strip()
    if not output_dir_token:
        raise HTTPException(400, "Experiment output directory is not available.")
    output_dir = Path(output_dir_token).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    step_value: int | None = payload.step
    if step_value is None:
        ckpt_result = await db.execute(
            select(Checkpoint)
            .where(Checkpoint.experiment_id == experiment_id)
            .order_by(Checkpoint.step.desc())
        )
        latest = ckpt_result.scalars().first()
        if latest is not None:
            step_value = int(latest.step)
            if payload.epoch is None:
                payload.epoch = float(latest.epoch)
            if payload.train_loss is None and latest.train_loss is not None:
                payload.train_loss = float(latest.train_loss)
            if payload.eval_loss is None and latest.eval_loss is not None:
                payload.eval_loss = float(latest.eval_loss)
    if step_value is None:
        step_value = 1

    total_steps = int(exp.total_steps or 0)
    if total_steps <= 0:
        total_steps = int((dict(exp.config or {}).get("num_epochs") or 1) * 100)
    if total_steps <= 0:
        total_steps = max(1, step_value)

    result = await capture_vibe_check_snapshot(
        project_id=project_id,
        experiment_id=experiment_id,
        output_dir=output_dir,
        step=step_value,
        total_steps=total_steps,
        base_model=str(exp.base_model or "microsoft/phi-2"),
        epoch=payload.epoch,
        train_loss=payload.train_loss,
        eval_loss=payload.eval_loss,
        experiment_config=dict(exp.config or {}),
        api_key=payload.api_key,
    )
    return {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "status": exp.status.value,
        **result,
    }


@router.websocket("/ws/{experiment_id}")
async def ws_training_status(
    websocket: WebSocket,
    project_id: int, # Keep project_id for consistency with other routes, though not directly used in this new implementation
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket endpoint for real-time training metrics and terminal logs.
    """
    await websocket.accept()

    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    exp = exp_result.scalar_one_or_none()
    if not exp:
        await websocket.close(code=1008, reason="Experiment not found")
        return

    register_websocket(experiment_id, websocket)

    ckpt_result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.experiment_id == experiment_id)
        .order_by(Checkpoint.step.asc())
    )
    checkpoints = ckpt_result.scalars().all()
    metrics = [
        {
            "experiment_id": experiment_id,
            "epoch": c.epoch,
            "step": c.step,
            "train_loss": c.train_loss,
            "eval_loss": c.eval_loss,
        }
        for c in checkpoints
    ]
    try:
        await websocket.send_json({"type": "init", "metrics": metrics})
    except Exception:
        unregister_websocket(experiment_id, websocket)
        return

    # Background task to stream logs and metric envelopes from Redis PubSub.
    async def log_stream_loop():
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"log:experiment:{experiment_id}")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    raw = message.get("data", "")
                    envelope = None
                    if isinstance(raw, str):
                        try:
                            parsed = json.loads(raw)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict) and isinstance(parsed.get("type"), str):
                            envelope = parsed

                    try:
                        if envelope is None:
                            await websocket.send_json({"type": "log", "text": str(raw)})
                            continue
                        event_type = str(envelope.get("type", "")).strip().lower()
                        if event_type == "metric" and isinstance(envelope.get("metric"), dict):
                            metric = dict(envelope["metric"])
                            metric.setdefault("experiment_id", experiment_id)
                            await websocket.send_json({"type": "metric", "metric": metric})
                        elif event_type == "observability" and isinstance(envelope.get("payload"), dict):
                            payload = dict(envelope.get("payload") or {})
                            payload.setdefault("experiment_id", experiment_id)
                            await websocket.send_json({"type": "observability", "payload": payload})
                        elif event_type == "vibe_check":
                            snapshot = envelope.get("snapshot")
                            if isinstance(snapshot, dict):
                                await websocket.send_json(
                                    {
                                        "type": "vibe_check",
                                        "snapshot": snapshot,
                                        "timeline_path": envelope.get("timeline_path"),
                                        "snapshot_count": envelope.get("snapshot_count"),
                                    }
                                )
                            else:
                                await websocket.send_json({"type": "log", "text": str(raw)})
                        elif event_type == "status":
                            payload = {"type": "status", "status": envelope.get("status", "")}
                            if "error" in envelope:
                                payload["error"] = envelope.get("error")
                            if "returncode" in envelope:
                                payload["returncode"] = envelope.get("returncode")
                            await websocket.send_json(payload)
                        elif event_type == "log":
                            await websocket.send_json({"type": "log", "text": str(envelope.get("text", ""))})
                        else:
                            await websocket.send_json({"type": "log", "text": str(raw)})
                    except Exception:
                        break  # client disconnected
        finally:
            await pubsub.unsubscribe()
            await redis_client.aclose()

    log_task = asyncio.create_task(log_stream_loop())

    try:
        while True:
            # Keep connection alive; client might send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        log_task.cancel()
        unregister_websocket(experiment_id, websocket)


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create(
    project_id: int,
    data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new training experiment."""
    try:
        config_payload = data.config.model_dump()
        provided_config_fields = set(data.config.model_fields_set)

        runtime = await resolve_project_domain_runtime(db, project_id)
        effective_contract = runtime.get("effective_contract")
        profile_training_defaults = get_training_defaults(effective_contract)
        config_payload, resolved_training_mode, profile_defaults_applied = _resolve_training_config(
            config_payload=config_payload,
            provided_config_fields=provided_config_fields,
            profile_training_defaults=profile_training_defaults,
            fallback_training_mode=data.config.training_mode,
        )

        exp = await create_experiment(
            db,
            project_id,
            data.name,
            str(config_payload.get("base_model", data.config.base_model)),
            config_payload,
            data.description,
            resolved_training_mode,
        )
        response_payload = ExperimentResponse.model_validate(exp).model_dump()
        response_payload["domain_pack_applied"] = runtime.get("domain_pack_applied")
        response_payload["domain_pack_source"] = runtime.get("domain_pack_source")
        response_payload["domain_profile_applied"] = runtime.get("domain_profile_applied")
        response_payload["domain_profile_source"] = runtime.get("domain_profile_source")
        response_payload["profile_training_defaults"] = (
            dict(profile_training_defaults) if profile_training_defaults else None
        )
        response_payload["resolved_training_config"] = dict(config_payload)
        response_payload["profile_defaults_applied"] = sorted(set(profile_defaults_applied))
        return ExperimentResponse.model_validate(response_payload)
    except HTTPException:
        raise
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project ") and detail.endswith(" not found"):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.post("/experiments/effective-config")
async def effective_training_config(
    project_id: int,
    req: TrainingEffectiveConfigRequest,
    db: AsyncSession = Depends(get_db),
):
    """Preview effective training config after domain runtime defaults are applied."""
    source_config = dict(req.config or {})
    (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    ) = await _resolve_effective_training_preview(
        project_id=project_id,
        source_config=source_config,
        db=db,
    )

    return {
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "profile_training_defaults": (
            dict(profile_training_defaults) if profile_training_defaults else None
        ),
        "resolved_training_config": resolved_config,
        "resolved_training_mode": resolved_training_mode.value,
        "profile_defaults_applied": profile_defaults_applied,
    }


@router.post("/experiments/preflight")
async def training_preflight(
    project_id: int,
    req: TrainingEffectiveConfigRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resolve effective config and run capability/runtime preflight checks."""
    source_config = dict(req.config or {})
    (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    ) = await _resolve_effective_training_preview(
        project_id=project_id,
        source_config=source_config,
        db=db,
    )

    preflight = run_training_preflight(
        project_id=project_id,
        config=resolved_config,
        base_model=str(resolved_config.get("base_model", "")),
    )

    return {
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "profile_training_defaults": (
            dict(profile_training_defaults) if profile_training_defaults else None
        ),
        "resolved_training_config": resolved_config,
        "resolved_training_mode": resolved_training_mode.value,
        "profile_defaults_applied": profile_defaults_applied,
        "preflight": preflight,
    }


@router.post("/experiments/preflight/plan")
async def training_preflight_plan(
    project_id: int,
    req: TrainingEffectiveConfigRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resolve effective config and generate suggested preflight tuning plans."""
    source_config = dict(req.config or {})
    (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    ) = await _resolve_effective_training_preview(
        project_id=project_id,
        source_config=source_config,
        db=db,
    )

    plan = run_training_preflight_plan(
        project_id=project_id,
        config=resolved_config,
        base_model=str(resolved_config.get("base_model", "")),
    )

    return {
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "profile_training_defaults": (
            dict(profile_training_defaults) if profile_training_defaults else None
        ),
        "resolved_training_config": resolved_config,
        "resolved_training_mode": resolved_training_mode.value,
        "profile_defaults_applied": profile_defaults_applied,
        "plan": plan,
    }


@router.get("/experiments/{experiment_id}/preflight")
async def training_preflight_for_experiment(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Run preflight checks for an existing experiment config."""
    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    exp = exp_result.scalar_one_or_none()
    if not exp:
        raise HTTPException(404, f"Experiment {experiment_id} not found in project {project_id}")

    raw_config = dict(exp.config or {})
    raw_config.setdefault("base_model", exp.base_model)
    try:
        parsed = TrainingConfig.model_validate(raw_config)
    except Exception as e:
        raise HTTPException(400, f"Experiment {experiment_id} has invalid training config: {e}")

    resolved_config = parsed.model_dump()
    preflight = run_training_preflight(
        project_id=project_id,
        config=resolved_config,
        base_model=exp.base_model,
    )
    return {
        "experiment_id": exp.id,
        "status": exp.status.value,
        "resolved_training_config": resolved_config,
        "preflight": preflight,
    }


@router.post("/experiments/{experiment_id}/start")
async def start(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Start training for an experiment."""
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    exp = exp_result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found in project {project_id}")

    target_profile_id = str(project.target_profile_id or "vllm_server").strip() or "vllm_server"
    compatibility = check_compatibility(exp.base_model, target_profile_id)
    if not bool(compatibility.get("compatible", False)):
        reasons = [str(item).strip() for item in list(compatibility.get("reasons") or []) if str(item).strip()]
        warnings = [str(item).strip() for item in list(compatibility.get("warnings") or []) if str(item).strip()]
        unblock_actions = _build_target_incompatibility_actions(
            project_id=project_id,
            target_profile_id=target_profile_id,
            compatibility=dict(compatibility),
        )
        actionable_fix = (
            str(unblock_actions[0].get("description") or "").strip()
            if unblock_actions
            else (reasons[0] if reasons else "Choose a smaller base model or a less constrained target profile.")
        )
        raise HTTPException(
            400,
            {
                "error_code": "TARGET_INCOMPATIBLE",
                "stage": "training",
                "message": (
                    f"Base model '{exp.base_model}' is not compatible with target profile '{target_profile_id}'."
                ),
                "actionable_fix": actionable_fix,
                "docs_url": "/docs/targets/compatibility",
                "metadata": {
                    "target_profile_id": target_profile_id,
                    "reasons": reasons,
                    "warnings": warnings,
                    "vram_check": dict(compatibility.get("vram_check") or {}),
                    "model_metadata": dict(compatibility.get("model_metadata") or {}),
                    "unblock_actions": unblock_actions,
                },
            },
        )

    try:
        return await start_training(db, project_id, experiment_id)
    except ValueError as e:
        detail = str(e)
        if "already running" in detail or "already completed" in detail:
            raise HTTPException(409, detail)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.post("/experiments/{experiment_id}/cancel")
async def cancel(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running training experiment."""
    try:
        return await cancel_training(db, project_id, experiment_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(409, detail)


@router.get("/experiments/{experiment_id}/status")
async def status(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get training status and metrics."""
    try:
        return await get_training_status(db, project_id, experiment_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/tasks/{task_id}")
async def task_status(
    project_id: int,
    task_id: str,
):
    """Read Celery task state for training-related jobs."""
    try:
        return get_task_status(task_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/experiments", response_model=list[ExperimentResponse])
async def list_all(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all experiments for a project."""
    exps = await list_experiments(db, project_id)
    return [ExperimentResponse.model_validate(e) for e in exps]

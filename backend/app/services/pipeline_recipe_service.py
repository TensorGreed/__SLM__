"""End-to-end pipeline recipe (blueprint) service.

A pipeline recipe is a reusable project-level blueprint that can wire:
- domain pack/profile
- workflow graph template
- dataset adapter preset
- training recipe + base config
- evaluation gate pack
- export targets metadata
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.domain_pack import DomainPack
from app.models.domain_profile import DomainProfile
from app.models.project import Project
from app.services.artifact_registry_service import publish_artifact, serialize_artifact
from app.services.data_adapter_service import normalize_task_profile
from app.services.dataset_service import save_project_dataset_adapter_preference
from app.services.evaluation_pack_service import normalize_evaluation_pack_id
from app.services.training_preflight_service import run_training_preflight
from app.services.training_recipe_service import (
    list_training_recipes,
    resolve_training_recipe,
)
from app.services.workflow_graph_service import (
    get_workflow_graph_templates,
    save_workflow_graph_override,
)


DEFAULT_PIPELINE_RECIPE_ID = "recipe.pipeline.sft_default"


_BUILTIN_PIPELINE_RECIPES: list[dict[str, Any]] = [
    {
        "recipe_id": "recipe.pipeline.sft_default",
        "display_name": "SFT Default Blueprint",
        "description": "Balanced general-purpose supervised fine-tuning pipeline from ingestion to export.",
        "version": "1.0.0",
        "category": "general",
        "tags": ["default", "sft", "end_to_end"],
        "supports_task_profiles": [
            "instruction_sft",
            "chat_sft",
            "qa",
            "rag_qa",
            "tool_calling",
            "structured_extraction",
            "summarization",
            "language_modeling",
        ],
        "speed_profile": "balanced",
        "domain": {
            "domain_pack_id": "general-pack-v1",
            "domain_profile_id": "generic-domain-v1",
        },
        "workflow": {
            "template_id": "template.sft",
        },
        "dataset_adapter": {
            "adapter_id": "default-canonical",
            "adapter_config": {},
            "field_mapping": {},
            "task_profile": "instruction_sft",
        },
        "training": {
            "training_recipe_id": "recipe.sft.balanced",
            "base_config": {"base_model": "microsoft/phi-2"},
            "overrides": {},
            "preferred_plan_profile": "balanced",
        },
        "evaluation": {
            "preferred_pack_id": "evalpack.general.default",
        },
        "export": {
            "target_formats": ["gguf", "onnx"],
        },
    },
    {
        "recipe_id": "recipe.pipeline.lora_fast",
        "display_name": "LoRA Fast Blueprint",
        "description": "Fast LoRA iteration pipeline with lighter training defaults.",
        "version": "1.0.0",
        "category": "lora",
        "tags": ["lora", "fast", "iteration"],
        "supports_task_profiles": [
            "instruction_sft",
            "chat_sft",
            "qa",
            "rag_qa",
            "tool_calling",
            "summarization",
            "language_modeling",
        ],
        "speed_profile": "fast",
        "domain": {
            "domain_pack_id": "general-pack-v1",
            "domain_profile_id": "generic-domain-v1",
        },
        "workflow": {
            "template_id": "template.lora",
        },
        "dataset_adapter": {
            "adapter_id": "default-canonical",
            "adapter_config": {},
            "field_mapping": {},
            "task_profile": "instruction_sft",
        },
        "training": {
            "training_recipe_id": "recipe.lora.fast",
            "base_config": {"base_model": "microsoft/phi-2"},
            "overrides": {},
            "preferred_plan_profile": "safe",
        },
        "evaluation": {
            "preferred_pack_id": "evalpack.fast.iteration",
        },
        "export": {
            "target_formats": ["gguf"],
        },
    },
    {
        "recipe_id": "recipe.pipeline.eval_gate",
        "display_name": "Eval Gate Blueprint",
        "description": "Evaluation-focused workflow for validating existing checkpoints before release.",
        "version": "1.0.0",
        "category": "evaluation",
        "tags": ["eval", "gate", "release"],
        "supports_task_profiles": [
            "instruction_sft",
            "chat_sft",
            "qa",
            "rag_qa",
            "tool_calling",
            "structured_extraction",
            "summarization",
            "classification",
            "language_modeling",
            "preference",
        ],
        "speed_profile": "balanced",
        "domain": {
            "domain_pack_id": "general-pack-v1",
            "domain_profile_id": "generic-domain-v1",
        },
        "workflow": {
            "template_id": "template.eval_only",
        },
        "dataset_adapter": {
            "adapter_id": "default-canonical",
            "adapter_config": {},
            "field_mapping": {},
            "task_profile": "instruction_sft",
        },
        "training": {
            "training_recipe_id": "recipe.sft.safe",
            "base_config": {"base_model": "microsoft/phi-2"},
            "overrides": {},
            "preferred_plan_profile": "safe",
        },
        "evaluation": {
            "preferred_pack_id": "evalpack.quality.strict",
        },
        "export": {
            "target_formats": ["gguf", "onnx"],
        },
    },
]


_DOMAIN_TASK_TO_PROFILE: dict[str, str] = {
    "qa": "qa",
    "question_answering": "qa",
    "classification": "classification",
    "sequence_classification": "classification",
    "seq2seq": "seq2seq",
    "summarization": "summarization",
    "chat": "chat_sft",
    "chat_sft": "chat_sft",
    "rag": "rag_qa",
    "rag_qa": "rag_qa",
    "retrieval_qa": "rag_qa",
    "tool_calling": "tool_calling",
    "function_calling": "tool_calling",
    "preference": "preference",
    "language_modeling": "language_modeling",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lower()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = _deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = _deepcopy(value)
    return merged


def _recipe_summary(recipe: dict[str, Any], *, include_blueprint: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "recipe_id": str(recipe.get("recipe_id", "")),
        "display_name": str(recipe.get("display_name", "")),
        "description": str(recipe.get("description", "")),
        "version": str(recipe.get("version", "")),
        "category": str(recipe.get("category", "general")),
        "tags": [str(item) for item in list(recipe.get("tags") or []) if str(item).strip()],
        "supports_task_profiles": [
            normalize_task_profile(str(item), default="")
            for item in list(recipe.get("supports_task_profiles") or [])
            if normalize_task_profile(str(item), default="")
        ],
        "speed_profile": str(recipe.get("speed_profile") or "balanced").strip().lower() or "balanced",
    }
    if include_blueprint:
        payload["blueprint"] = _deepcopy(recipe)
    return payload


def list_pipeline_recipes(*, include_blueprint: bool = False) -> list[dict[str, Any]]:
    return [_recipe_summary(item, include_blueprint=include_blueprint) for item in _BUILTIN_PIPELINE_RECIPES]


def get_pipeline_recipe(recipe_id: str) -> dict[str, Any] | None:
    token = _normalize_token(recipe_id)
    if not token:
        return None
    for recipe in _BUILTIN_PIPELINE_RECIPES:
        if _normalize_token(str(recipe.get("recipe_id"))) == token:
            return _deepcopy(recipe)
    return None


def _extract_profile_from_domain_contract(contract: Any) -> str | None:
    if not isinstance(contract, dict):
        return None
    tasks = contract.get("tasks")
    if not isinstance(tasks, list):
        return None
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "").strip().lower()
        mapped = _DOMAIN_TASK_TO_PROFILE.get(task_id, task_id)
        normalized = normalize_task_profile(mapped, default="")
        if normalized:
            return normalized
    return None


def _extract_project_task_profile(project: Project) -> tuple[str | None, str]:
    preset = project.dataset_adapter_preset if isinstance(project.dataset_adapter_preset, dict) else {}
    preset_profile = normalize_task_profile(str(preset.get("task_profile") or ""), default="")
    if preset_profile:
        return preset_profile, "project.dataset_adapter_preset.task_profile"

    domain_profile = getattr(project, "domain_profile", None)
    contract = domain_profile.contract if getattr(domain_profile, "contract", None) else None
    contract_profile = _extract_profile_from_domain_contract(contract)
    if contract_profile:
        return contract_profile, "project.domain_profile.contract.tasks[0].task_id"

    return None, "default"


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return None


def _score_recipe_for_context(
    recipe: dict[str, Any],
    *,
    task_profile: str | None,
    preferred_plan_profile: str | None,
    prefer_fast: bool | None,
    project_base_model: str | None,
    default_bias: bool = False,
) -> tuple[float, list[str], bool]:
    score = 0.0
    reasons: list[str] = []
    summary = _recipe_summary(recipe, include_blueprint=False)
    supports_profiles = [str(item) for item in list(summary.get("supports_task_profiles") or []) if str(item).strip()]

    compatible = True
    if task_profile:
        if task_profile in supports_profiles:
            score += 4.0
            reasons.append(f"supports task_profile '{task_profile}'")
        elif supports_profiles:
            compatible = False
            score -= 1.5
            reasons.append(f"does not explicitly list task_profile '{task_profile}'")

    training = dict(recipe.get("training") or {})
    recipe_plan_profile = str(training.get("preferred_plan_profile") or "").strip().lower() or None
    if preferred_plan_profile and recipe_plan_profile == preferred_plan_profile:
        score += 1.8
        reasons.append(f"matches preferred plan profile '{preferred_plan_profile}'")

    tags = {str(item).strip().lower() for item in list(recipe.get("tags") or []) if str(item).strip()}
    speed_profile = str(recipe.get("speed_profile") or "balanced").strip().lower() or "balanced"
    is_fast = speed_profile == "fast" or "fast" in tags
    if prefer_fast is True and is_fast:
        score += 2.2
        reasons.append("optimized for fast iteration")
    elif prefer_fast is False and is_fast:
        score -= 0.6
        reasons.append("fast iteration profile deprioritized by context")
    elif prefer_fast is False and speed_profile == "balanced":
        score += 0.4
        reasons.append("balanced profile preferred")

    base_config = dict(training.get("base_config") or {})
    recipe_base_model = str(base_config.get("base_model") or "").strip().lower() or None
    project_model = str(project_base_model or "").strip().lower() or None
    if recipe_base_model and project_model and recipe_base_model == project_model:
        score += 0.8
        reasons.append("base model aligns with current project")

    if default_bias:
        score += 0.2
        reasons.append("default fallback preference")

    if not reasons:
        reasons.append("neutral fit")
    return round(score, 4), reasons, compatible


async def recommend_pipeline_recipes_for_project(
    db: AsyncSession,
    *,
    project_id: int,
    task_profile: str | None = None,
    preferred_plan_profile: str | None = None,
    prefer_fast: bool | None = None,
) -> dict[str, Any]:
    project = await _get_project(db, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    resolved_task_profile = normalize_task_profile(task_profile, default="")
    task_profile_source = "request"
    if not resolved_task_profile:
        extracted_profile, extracted_source = _extract_project_task_profile(project)
        resolved_task_profile = extracted_profile or ""
        task_profile_source = extracted_source

    resolved_plan_profile = str(preferred_plan_profile or project.training_preferred_plan_profile or "").strip().lower() or None
    resolved_prefer_fast = _coerce_bool(prefer_fast)
    project_base_model = str(project.base_model_name or "").strip() or None

    scored: list[dict[str, Any]] = []
    for recipe in _BUILTIN_PIPELINE_RECIPES:
        recipe_id = str(recipe.get("recipe_id") or "").strip()
        if not recipe_id:
            continue
        score, reasons, compatible = _score_recipe_for_context(
            recipe,
            task_profile=resolved_task_profile or None,
            preferred_plan_profile=resolved_plan_profile,
            prefer_fast=resolved_prefer_fast,
            project_base_model=project_base_model,
            default_bias=(recipe_id == DEFAULT_PIPELINE_RECIPE_ID),
        )
        row = _recipe_summary(recipe, include_blueprint=False)
        row.update(
            {
                "score": score,
                "reasons": reasons,
                "task_profile_compatible": compatible,
            }
        )
        scored.append(row)

    order = {str(item.get("recipe_id") or ""): idx for idx, item in enumerate(_BUILTIN_PIPELINE_RECIPES)}
    scored.sort(key=lambda item: (-float(item.get("score") or 0.0), order.get(str(item.get("recipe_id") or ""), 9999)))

    recommended_recipe_id = DEFAULT_PIPELINE_RECIPE_ID
    if scored:
        recommended_recipe_id = str(scored[0].get("recipe_id") or DEFAULT_PIPELINE_RECIPE_ID)

    return {
        "project_id": project_id,
        "recommended_recipe_id": recommended_recipe_id,
        "context": {
            "task_profile": resolved_task_profile or None,
            "task_profile_source": task_profile_source,
            "preferred_plan_profile": resolved_plan_profile,
            "prefer_fast": resolved_prefer_fast,
            "project_base_model": project_base_model,
        },
        "recommendations": scored,
    }


def _project_recipe_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "pipeline_recipes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_recipe_state_path(project_id: int) -> Path:
    return _project_recipe_dir(project_id) / "state.json"


def _project_recipe_runs_dir(project_id: int) -> Path:
    path = _project_recipe_dir(project_id) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_recipe_executions_dir(project_id: int) -> Path:
    path = _project_recipe_dir(project_id) / "executions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_recipe_execution_state_path(project_id: int, recipe_run_id: str) -> Path:
    token = str(recipe_run_id or "").strip()
    if not token:
        raise ValueError("recipe_run_id is required")
    return _project_recipe_executions_dir(project_id) / token / "run.json"


def load_pipeline_recipe_state(project_id: int) -> dict[str, Any] | None:
    path = _project_recipe_state_path(project_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_pipeline_recipe_state(project_id: int, payload: dict[str, Any]) -> str:
    path = _project_recipe_state_path(project_id)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path)


def patch_pipeline_recipe_state(
    project_id: int,
    *,
    patch: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    state = load_pipeline_recipe_state(project_id) or {}
    state.update(dict(patch or {}))
    path = _write_pipeline_recipe_state(project_id, state)
    return state, path


def save_pipeline_recipe_execution(
    project_id: int,
    *,
    recipe_run_id: str,
    payload: dict[str, Any],
) -> str:
    path = _project_recipe_execution_state_path(project_id, recipe_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path)


def load_pipeline_recipe_execution(
    project_id: int,
    *,
    recipe_run_id: str,
) -> dict[str, Any] | None:
    try:
        path = _project_recipe_execution_state_path(project_id, recipe_run_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def patch_pipeline_recipe_execution(
    project_id: int,
    *,
    recipe_run_id: str,
    patch: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    payload = load_pipeline_recipe_execution(project_id, recipe_run_id=recipe_run_id) or {}
    payload.update(dict(patch or {}))
    path = save_pipeline_recipe_execution(
        project_id,
        recipe_run_id=recipe_run_id,
        payload=payload,
    )
    return payload, path


def list_pipeline_recipe_executions(
    project_id: int,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    root = _project_recipe_executions_dir(project_id)
    rows: list[dict[str, Any]] = []
    for run_file in sorted(root.glob("*/run.json")):
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)

    def _sort_key(item: dict[str, Any]) -> str:
        return str(
            item.get("updated_at")
            or item.get("finished_at")
            or item.get("started_at")
            or item.get("created_at")
            or ""
        )

    rows.sort(key=_sort_key, reverse=True)
    safe_limit = max(1, min(int(limit), 200))
    return rows[:safe_limit]


async def _get_project(db: AsyncSession, project_id: int) -> Project | None:
    row = await db.execute(
        select(Project)
        .options(selectinload(Project.domain_profile))
        .where(Project.id == project_id)
    )
    return row.scalar_one_or_none()


async def _resolve_domain_ids(
    db: AsyncSession,
    *,
    pack_id: str | None,
    profile_id: str | None,
) -> tuple[int | None, int | None, list[str]]:
    warnings: list[str] = []
    resolved_pack_db_id: int | None = None
    resolved_profile_db_id: int | None = None

    if pack_id:
        pack_row = await db.execute(select(DomainPack).where(DomainPack.pack_id == pack_id.strip().lower()))
        pack = pack_row.scalar_one_or_none()
        if pack is None:
            warnings.append(f"Domain pack '{pack_id}' not found; keeping current project domain pack.")
        else:
            resolved_pack_db_id = int(pack.id)

    if profile_id:
        profile_row = await db.execute(
            select(DomainProfile).where(DomainProfile.profile_id == profile_id.strip().lower())
        )
        profile = profile_row.scalar_one_or_none()
        if profile is None:
            warnings.append(f"Domain profile '{profile_id}' not found; keeping current project domain profile.")
        else:
            resolved_profile_db_id = int(profile.id)

    return resolved_pack_db_id, resolved_profile_db_id, warnings


def _resolve_workflow_template(
    *,
    project_id: int,
    current_stage: Any,
    template_id: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    templates = get_workflow_graph_templates(project_id, current_stage)
    for item in templates:
        if str(item.get("template_id", "")).strip().lower() == template_id.strip().lower():
            graph = item.get("graph")
            if isinstance(graph, dict):
                return _deepcopy(graph), []
            return None, [f"Workflow template '{template_id}' has invalid graph payload."]
    available = sorted(str(item.get("template_id", "")) for item in templates)
    return None, [f"Workflow template '{template_id}' not found. Available: {', '.join(available)}"]


def _resolve_training_config(
    *,
    training_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    recipe_id = str(training_spec.get("training_recipe_id") or "").strip()
    base_config = dict(training_spec.get("base_config") or {})
    overrides = dict(training_spec.get("overrides") or {})

    if recipe_id:
        resolved = resolve_training_recipe(
            recipe_id,
            base_config=base_config,
            overrides=overrides,
        )
        recipe = dict(resolved.get("recipe") or {})
        config = dict(resolved.get("resolved_config") or {})
        missing_required = [str(item) for item in list(resolved.get("missing_required_fields") or [])]
        return recipe, config, missing_required, []

    merged = dict(base_config)
    merged.update(overrides)
    warning = ["training.training_recipe_id is empty; using base_config/overrides directly"]
    return {}, merged, [], warning


async def resolve_pipeline_recipe_blueprint(
    db: AsyncSession,
    *,
    project_id: int,
    recipe_id: str,
    overrides: dict[str, Any] | None = None,
    include_preflight: bool = True,
) -> dict[str, Any]:
    project = await _get_project(db, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    recipe = get_pipeline_recipe(recipe_id)
    if recipe is None:
        available = ", ".join(sorted(item["recipe_id"] for item in list_pipeline_recipes(include_blueprint=False)))
        raise ValueError(f"Unknown recipe_id '{recipe_id}'. Available: {available}")

    resolved_recipe = _deep_merge(recipe, dict(overrides or {}))
    warnings: list[str] = []

    domain_spec = dict(resolved_recipe.get("domain") or {})
    workflow_spec = dict(resolved_recipe.get("workflow") or {})
    adapter_spec = dict(resolved_recipe.get("dataset_adapter") or {})
    training_spec = dict(resolved_recipe.get("training") or {})
    evaluation_spec = dict(resolved_recipe.get("evaluation") or {})
    export_spec = dict(resolved_recipe.get("export") or {})

    pack_id = str(domain_spec.get("domain_pack_id") or "").strip().lower() or None
    profile_id = str(domain_spec.get("domain_profile_id") or "").strip().lower() or None
    pack_db_id, profile_db_id, domain_warnings = await _resolve_domain_ids(
        db,
        pack_id=pack_id,
        profile_id=profile_id,
    )
    warnings.extend(domain_warnings)

    template_id = str(workflow_spec.get("template_id") or "").strip()
    graph: dict[str, Any] | None = None
    if template_id:
        graph, graph_warnings = _resolve_workflow_template(
            project_id=project_id,
            current_stage=project.pipeline_stage,
            template_id=template_id,
        )
        warnings.extend(graph_warnings)
    else:
        warnings.append("workflow.template_id is empty; workflow graph override will not be updated")

    training_recipe, training_config, missing_required, training_warnings = _resolve_training_config(
        training_spec=training_spec
    )
    warnings.extend(training_warnings)

    preflight = None
    if include_preflight and training_config:
        preflight = run_training_preflight(
            project_id=project_id,
            config=training_config,
            base_model=str(training_config.get("base_model", "")),
        )

    preferred_plan_profile = str(training_spec.get("preferred_plan_profile") or "").strip().lower() or None
    evaluation_pack_id = normalize_evaluation_pack_id(str(evaluation_spec.get("preferred_pack_id") or "").strip() or None)

    return {
        "project_id": project_id,
        "recipe": _recipe_summary(resolved_recipe, include_blueprint=True),
        "resolved": {
            "domain": {
                "domain_pack_id": pack_id,
                "domain_pack_db_id": pack_db_id,
                "domain_profile_id": profile_id,
                "domain_profile_db_id": profile_db_id,
            },
            "workflow": {
                "template_id": template_id or None,
                "graph": graph,
            },
            "dataset_adapter": {
                "adapter_id": str(adapter_spec.get("adapter_id") or "").strip() or None,
                "adapter_config": dict(adapter_spec.get("adapter_config") or {}),
                "field_mapping": dict(adapter_spec.get("field_mapping") or {}),
                "task_profile": str(adapter_spec.get("task_profile") or "").strip() or None,
            },
            "training": {
                "preferred_plan_profile": preferred_plan_profile,
                "training_recipe": training_recipe or None,
                "resolved_training_config": training_config,
                "missing_required_fields": sorted(set(missing_required)),
            },
            "evaluation": {
                "preferred_pack_id": evaluation_pack_id,
            },
            "export": {
                "target_formats": [str(item).strip().lower() for item in list(export_spec.get("target_formats") or []) if str(item).strip()],
            },
        },
        "preflight": preflight,
        "warnings": warnings,
    }


async def apply_pipeline_recipe_blueprint(
    db: AsyncSession,
    *,
    project_id: int,
    recipe_id: str,
    overrides: dict[str, Any] | None = None,
    include_preflight: bool = True,
    enforce_preflight_ok: bool = False,
    mark_active: bool = True,
) -> dict[str, Any]:
    project = await _get_project(db, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    resolved = await resolve_pipeline_recipe_blueprint(
        db,
        project_id=project_id,
        recipe_id=recipe_id,
        overrides=overrides,
        include_preflight=include_preflight,
    )
    payload = dict(resolved.get("resolved") or {})
    preflight = resolved.get("preflight") if isinstance(resolved.get("preflight"), dict) else None
    if enforce_preflight_ok and preflight and not bool(preflight.get("ok")):
        errors = [str(item) for item in list(preflight.get("errors") or [])]
        preview = "; ".join(errors[:5]) if errors else "unknown preflight failure"
        raise ValueError(f"Pipeline recipe preflight failed: {preview}")

    domain_payload = dict(payload.get("domain") or {})
    workflow_payload = dict(payload.get("workflow") or {})
    adapter_payload = dict(payload.get("dataset_adapter") or {})
    training_payload = dict(payload.get("training") or {})
    evaluation_payload = dict(payload.get("evaluation") or {})

    if domain_payload.get("domain_pack_db_id") is not None:
        project.domain_pack_id = int(domain_payload["domain_pack_db_id"])
    if domain_payload.get("domain_profile_db_id") is not None:
        project.domain_profile_id = int(domain_payload["domain_profile_db_id"])

    resolved_training_config = dict(training_payload.get("resolved_training_config") or {})
    base_model = str(resolved_training_config.get("base_model") or "").strip()
    if base_model:
        project.base_model_name = base_model

    preferred_plan_profile = str(training_payload.get("preferred_plan_profile") or "").strip().lower()
    if preferred_plan_profile in {"safe", "balanced", "max_quality"}:
        project.training_preferred_plan_profile = preferred_plan_profile

    evaluation_pack_id = normalize_evaluation_pack_id(evaluation_payload.get("preferred_pack_id"))
    project.evaluation_preferred_pack_id = evaluation_pack_id

    adapter_id = str(adapter_payload.get("adapter_id") or "").strip()
    if adapter_id:
        await save_project_dataset_adapter_preference(
            db,
            project_id,
            adapter_id=adapter_id,
            adapter_config=dict(adapter_payload.get("adapter_config") or {}),
            field_mapping=dict(adapter_payload.get("field_mapping") or {}),
            task_profile=str(adapter_payload.get("task_profile") or "").strip() or None,
        )

    graph = workflow_payload.get("graph")
    saved_graph_path: str | None = None
    if isinstance(graph, dict):
        saved_graph_path = save_workflow_graph_override(project_id, graph)

    run_id = f"run-{_utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = _project_recipe_runs_dir(project_id) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "applied_at": _utcnow().isoformat(),
        "project_id": project_id,
        "recipe_id": str((resolved.get("recipe") or {}).get("recipe_id") or recipe_id),
        "resolved": payload,
        "preflight": preflight,
        "warnings": list(resolved.get("warnings") or []),
        "saved_workflow_graph_path": saved_graph_path,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    artifact = await publish_artifact(
        db=db,
        project_id=project_id,
        artifact_key="manifest.pipeline_recipe",
        uri=str(manifest_path),
        schema_ref="slm.pipeline-recipe-manifest/v1",
        producer_stage="pipeline_recipe",
        producer_run_id=run_id,
        metadata={
            "recipe_id": manifest["recipe_id"],
            "saved_workflow_graph_path": saved_graph_path,
            "preflight_ok": bool(preflight.get("ok")) if isinstance(preflight, dict) else None,
        },
    )

    state_payload = load_pipeline_recipe_state(project_id) or {}
    if mark_active:
        state_payload.update(
            {
                "project_id": project_id,
                "updated_at": _utcnow().isoformat(),
                "active_recipe_id": manifest["recipe_id"],
                "last_run_id": run_id,
                "last_manifest_path": str(manifest_path),
                "last_artifact_id": artifact.id,
            }
        )
        state_path = _write_pipeline_recipe_state(project_id, state_payload)
    else:
        state_path = str(_project_recipe_state_path(project_id))

    await db.flush()
    await db.refresh(project)

    return {
        "project_id": project_id,
        "recipe": resolved.get("recipe"),
        "resolved": payload,
        "preflight": preflight,
        "warnings": list(resolved.get("warnings") or []),
        "saved_workflow_graph_path": saved_graph_path,
        "manifest_path": str(manifest_path),
        "state_path": state_path,
        "state": state_payload if mark_active else load_pipeline_recipe_state(project_id),
        "artifact": serialize_artifact(artifact),
        "project": {
            "id": project.id,
            "name": project.name,
            "base_model_name": project.base_model_name,
            "domain_pack_id": project.domain_pack_id,
            "domain_profile_id": project.domain_profile_id,
            "training_preferred_plan_profile": project.training_preferred_plan_profile,
            "evaluation_preferred_pack_id": project.evaluation_preferred_pack_id,
            "dataset_adapter_preset": project.dataset_adapter_preset or {},
        },
    }


async def get_pipeline_recipe_state(
    db: AsyncSession,
    *,
    project_id: int,
) -> dict[str, Any]:
    project = await _get_project(db, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    state = load_pipeline_recipe_state(project_id)
    return {
        "project_id": project_id,
        "has_state": state is not None,
        "state": state,
    }


def list_available_training_recipe_ids() -> list[str]:
    return sorted(
        str(item.get("recipe_id"))
        for item in list_training_recipes(include_patch=False)
        if str(item.get("recipe_id", "")).strip()
    )

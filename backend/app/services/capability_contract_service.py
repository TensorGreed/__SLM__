"""Shared training capability contract helpers.

Phase 1 goal:
- keep a single source of truth for task/backend support
- evaluate adapter + runtime + task compatibility in one place
- expose a serializable matrix for API/UI consumers
"""

from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.services.data_adapter_service import (
    DEFAULT_ADAPTER_ID,
    is_training_task_compatible,
    normalize_task_profile,
    normalize_training_task_type,
    resolve_data_adapter_contract,
)
from app.services.training_runtime_service import (
    get_runtime_spec,
    list_runtime_specs,
)


SUPPORTED_TRAINING_TASK_TYPES: tuple[str, ...] = ("causal_lm", "seq2seq", "classification")
SUPPORTED_TRAINER_BACKENDS: tuple[str, ...] = ("auto", "hf_trainer", "trl_sft")
SUPPORTED_DATASET_MODALITIES: tuple[str, ...] = ("text", "vision_language", "audio_text", "multimodal")


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalized_set(items: list[Any] | tuple[Any, ...] | set[Any] | None) -> set[str]:
    out: set[str] = set()
    for item in list(items or []):
        token = _normalize_token(item)
        if token:
            out.add(token)
    return out


def infer_adapter_modality(contract: dict[str, Any] | None) -> str:
    """Infer dataset modality from adapter contract metadata."""
    payload = dict(contract or {})
    schema_hint = payload.get("schema_hint")
    if isinstance(schema_hint, dict):
        explicit = _normalize_token(schema_hint.get("modality"))
        if explicit in {"vision_language", "audio_text", "multimodal", "text"}:
            return explicit

    output_contract = payload.get("output_contract")
    required_fields: list[str] = []
    optional_fields: list[str] = []
    if isinstance(output_contract, dict):
        required_fields = [str(item).strip() for item in list(output_contract.get("required_fields") or []) if str(item).strip()]
        optional_fields = [str(item).strip() for item in list(output_contract.get("optional_fields") or []) if str(item).strip()]
    fields = {str(item).strip().lower() for item in (required_fields + optional_fields) if str(item).strip()}

    has_image = "image_path" in fields
    has_audio = "audio_path" in fields
    if has_image and has_audio:
        return "multimodal"
    if has_image:
        return "vision_language"
    if has_audio:
        return "audio_text"
    return "text"


def runtime_supported_modalities(runtime_id: str | None) -> dict[str, Any]:
    """Resolve runtime modality metadata for a runtime id."""
    token = str(runtime_id or "").strip().lower()
    if not token:
        return {
            "runtime_id": None,
            "known": False,
            "is_builtin": False,
            "modalities_declared": False,
            "supported_modalities": {"text"},
        }

    try:
        spec = get_runtime_spec(token)
    except Exception:
        # Unknown runtime ids fall back to text assumption only.
        return {
            "runtime_id": token,
            "known": False,
            "is_builtin": False,
            "modalities_declared": False,
            "supported_modalities": {"text"},
        }

    modalities = _normalized_set(list(getattr(spec, "supported_modalities", []) or []))
    if not modalities:
        modalities = {"text"}
    return {
        "runtime_id": token,
        "known": True,
        "is_builtin": bool(getattr(spec, "is_builtin", False)),
        "modalities_declared": bool(getattr(spec, "declares_supported_modalities", False)),
        "supported_modalities": modalities,
    }


def _prepared_manifest(project_id: int) -> dict[str, Any]:
    prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_training_adapter_context(
    *,
    project_id: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve adapter/task-profile context used by the latest prepared split."""
    cfg = dict(config or {})
    manifest = _prepared_manifest(project_id)

    raw_adapter_id = cfg.get("adapter_id")
    adapter_source = "config" if raw_adapter_id else "prepared_manifest"
    adapter_id = str(raw_adapter_id or manifest.get("adapter_id") or DEFAULT_ADAPTER_ID).strip() or DEFAULT_ADAPTER_ID

    raw_task_profile = cfg.get("task_profile")
    task_profile_source = "config" if raw_task_profile else "prepared_manifest"
    task_profile = normalize_task_profile(
        str(raw_task_profile or manifest.get("task_profile") or ""),
        default="",
    ) or None

    contract = resolve_data_adapter_contract(adapter_id)
    modality = infer_adapter_modality(contract)

    return {
        "adapter_id": adapter_id,
        "adapter_source": adapter_source,
        "task_profile": task_profile,
        "task_profile_source": task_profile_source,
        "adapter_contract": contract,
        "adapter_modality": modality,
        "prepared_manifest_found": bool(manifest),
        "prepared_manifest_path": str(settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "manifest.json"),
    }


def evaluate_training_capability_contract(
    *,
    task_type: str | None,
    training_mode: str | None,
    trainer_backend_requested: str | None,
    runtime_id: str | None,
    runtime_backend: str | None,
    adapter_id: str,
    adapter_contract: dict[str, Any] | None,
    adapter_task_profile: str | None,
) -> dict[str, Any]:
    """Evaluate task/runtime/adapter compatibility and return diagnostics."""
    normalized_task_type = normalize_training_task_type(task_type, default="causal_lm")
    normalized_training_mode = _normalize_token(training_mode) or "sft"
    normalized_backend = _normalize_token(trainer_backend_requested) or "auto"
    normalized_runtime_id = str(runtime_id or "").strip().lower()
    normalized_runtime_backend = _normalize_token(runtime_backend) or "unknown"

    contract = dict(adapter_contract or {})
    preferred_training_tasks = _normalized_set(list(contract.get("preferred_training_tasks") or []))
    declared_task_profiles = _normalized_set(list(contract.get("task_profiles") or []))
    adapter_modality = infer_adapter_modality(contract)

    errors: list[str] = []
    warnings: list[str] = []
    hints: list[str] = []

    if normalized_task_type not in SUPPORTED_TRAINING_TASK_TYPES:
        errors.append(
            f"task_type='{normalized_task_type}' is not supported. "
            f"Supported task types: {', '.join(SUPPORTED_TRAINING_TASK_TYPES)}."
        )

    if normalized_backend not in SUPPORTED_TRAINER_BACKENDS:
        errors.append(
            f"trainer_backend='{normalized_backend}' is not supported. "
            f"Supported trainer backends: {', '.join(SUPPORTED_TRAINER_BACKENDS)}."
        )

    if preferred_training_tasks and normalized_task_type not in preferred_training_tasks:
        warnings.append(
            (
                f"adapter_id='{adapter_id}' does not declare task_type='{normalized_task_type}' in "
                f"preferred_training_tasks ({', '.join(sorted(preferred_training_tasks))})."
            )
        )
        hints.append(
            "Run Adapter Lab preview and choose an adapter/task_profile aligned with your training task."
        )

    normalized_profile = normalize_task_profile(adapter_task_profile, default="")
    if normalized_profile:
        if declared_task_profiles and normalized_profile not in declared_task_profiles:
            warnings.append(
                (
                    f"adapter task_profile='{normalized_profile}' is not declared by adapter '{adapter_id}'. "
                    f"Declared profiles: {', '.join(sorted(declared_task_profiles))}."
                )
            )
        if not is_training_task_compatible(normalized_profile, normalized_task_type):
            warnings.append(
                (
                    f"task_profile='{normalized_profile}' is not directly compatible with "
                    f"task_type='{normalized_task_type}'."
                )
            )
            hints.append(
                "Adjust task_type or adapter task_profile before launching training."
            )

    runtime_meta = runtime_supported_modalities(normalized_runtime_id)
    runtime_modalities = set(runtime_meta.get("supported_modalities") or {"text"})
    runtime_modalities_declared = bool(runtime_meta.get("modalities_declared", False))
    runtime_known = bool(runtime_meta.get("known", False))
    runtime_is_builtin = bool(runtime_meta.get("is_builtin", False))
    if normalized_runtime_id and adapter_modality not in runtime_modalities:
        if runtime_modalities_declared:
            errors.append(
                (
                    f"runtime '{normalized_runtime_id}' supports modalities "
                    f"{', '.join(sorted(runtime_modalities))}, but dataset adapter '{adapter_id}' "
                    f"resolves modality '{adapter_modality}'."
                )
            )
            if runtime_is_builtin:
                hints.append(
                    (
                        "Use a text adapter/task profile for this built-in runtime, "
                        "or switch to a multimodal runtime plugin."
                    )
                )
            else:
                hints.append(
                    "Select a runtime that declares support for this adapter modality."
                )
        else:
            warnings.append(
                (
                    f"runtime '{normalized_runtime_id or 'unknown'}' does not declare supported_modalities; "
                    f"assuming text-only while adapter modality is '{adapter_modality}'."
                )
            )

    if normalized_training_mode in {"dpo", "orpo"} and adapter_modality != "text":
        errors.append(
            (
                f"training_mode='{normalized_training_mode}' currently requires text preference pairs, "
                f"but adapter modality is '{adapter_modality}'."
            )
        )
    if normalized_backend == "trl_sft" and adapter_modality != "text":
        errors.append(
            (
                f"trainer_backend='{normalized_backend}' is currently text-only, "
                f"but adapter modality is '{adapter_modality}'."
            )
        )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "hints": hints,
        "summary": {
            "task_type": normalized_task_type,
            "training_mode": normalized_training_mode,
            "trainer_backend_requested": normalized_backend,
            "runtime_id": normalized_runtime_id or None,
            "runtime_backend": normalized_runtime_backend,
            "runtime_known": runtime_known,
            "runtime_supported_modalities": sorted(runtime_modalities),
            "runtime_modalities_declared": runtime_modalities_declared,
            "adapter_id": adapter_id,
            "adapter_task_profile": normalized_profile or None,
            "adapter_declared_task_profiles": sorted(declared_task_profiles),
            "adapter_preferred_training_tasks": sorted(preferred_training_tasks),
            "adapter_modality": adapter_modality,
            "contract_version": str(contract.get("version") or ""),
        },
    }


def build_training_capability_contract() -> dict[str, Any]:
    """Serialize static capability matrix for API/UI consumption."""
    runtime_modality_support: dict[str, list[str]] = {}
    runtime_modality_declared: dict[str, bool] = {}
    for spec in list_runtime_specs():
        runtime_id = str(spec.runtime_id or "").strip().lower()
        if not runtime_id:
            continue
        modalities = _normalized_set(list(getattr(spec, "supported_modalities", []) or []))
        runtime_modality_support[runtime_id] = sorted(modalities or {"text"})
        runtime_modality_declared[runtime_id] = bool(
            getattr(spec, "declares_supported_modalities", False)
        )

    return {
        "contract_version": "slm.training-capability/v1",
        "supported_task_types": list(SUPPORTED_TRAINING_TASK_TYPES),
        "supported_trainer_backends": list(SUPPORTED_TRAINER_BACKENDS),
        "supported_dataset_modalities": list(SUPPORTED_DATASET_MODALITIES),
        "runtime_modality_support": runtime_modality_support,
        "runtime_modality_declared": runtime_modality_declared,
    }

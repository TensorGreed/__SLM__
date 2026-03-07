"""Training capability matrix and preflight checks."""

from __future__ import annotations

import importlib.util
from typing import Any

from app.config import settings
from app.services.dataset_contract_service import analyze_prepared_dataset_contract
from app.services.training_runtime_service import (
    get_runtime_spec,
    resolve_training_runtime_id,
    validate_runtime,
)


MODEL_CAPABILITY_MATRIX_VERSION = "v1"

SUPPORTED_TASK_TYPES = {"causal_lm", "seq2seq", "classification"}
SUPPORTED_TRAINER_BACKENDS = {"auto", "hf_trainer", "trl_sft"}

_MODEL_CAPABILITY_MATRIX: list[dict[str, Any]] = [
    {
        "family": "llama",
        "tokens": ("llama",),
        "architecture": "causal_lm",
        "supported_task_types": ["causal_lm", "classification"],
        "supported_trainer_backends": ["auto", "hf_trainer", "trl_sft"],
        "recommended_chat_templates": ["llama3", "chatml"],
        "max_seq_length_hint": 8192,
    },
    {
        "family": "mistral",
        "tokens": ("mistral", "mixtral"),
        "architecture": "causal_lm",
        "supported_task_types": ["causal_lm", "classification"],
        "supported_trainer_backends": ["auto", "hf_trainer", "trl_sft"],
        "recommended_chat_templates": ["chatml", "zephyr", "llama3"],
        "max_seq_length_hint": 8192,
    },
    {
        "family": "qwen",
        "tokens": ("qwen",),
        "architecture": "causal_lm",
        "supported_task_types": ["causal_lm", "classification"],
        "supported_trainer_backends": ["auto", "hf_trainer", "trl_sft"],
        "recommended_chat_templates": ["chatml", "llama3"],
        "max_seq_length_hint": 32768,
    },
    {
        "family": "phi",
        "tokens": ("phi-", "phi/", "microsoft/phi"),
        "architecture": "causal_lm",
        "supported_task_types": ["causal_lm", "classification"],
        "supported_trainer_backends": ["auto", "hf_trainer", "trl_sft"],
        "recommended_chat_templates": ["phi3", "chatml", "llama3"],
        "max_seq_length_hint": 4096,
    },
    {
        "family": "gemma",
        "tokens": ("gemma",),
        "architecture": "causal_lm",
        "supported_task_types": ["causal_lm", "classification"],
        "supported_trainer_backends": ["auto", "hf_trainer", "trl_sft"],
        "recommended_chat_templates": ["chatml", "llama3"],
        "max_seq_length_hint": 8192,
    },
    {
        "family": "encoder_decoder",
        "tokens": ("flan-t5", "t5", "bart", "mbart", "pegasus"),
        "architecture": "seq2seq",
        "supported_task_types": ["seq2seq", "classification"],
        "supported_trainer_backends": ["auto", "hf_trainer"],
        "recommended_chat_templates": ["chatml", "llama3"],
        "max_seq_length_hint": 1024,
    },
    {
        "family": "encoder_only",
        "tokens": ("bert", "roberta", "deberta", "distilbert", "electra", "albert"),
        "architecture": "encoder",
        "supported_task_types": ["classification"],
        "supported_trainer_backends": ["auto", "hf_trainer"],
        "recommended_chat_templates": [],
        "max_seq_length_hint": 512,
    },
]

_DEFAULT_MODEL_CAPABILITY: dict[str, Any] = {
    "family": "unknown",
    "architecture": "unknown",
    "supported_task_types": sorted(SUPPORTED_TASK_TYPES),
    "supported_trainer_backends": sorted(SUPPORTED_TRAINER_BACKENDS),
    "recommended_chat_templates": ["llama3", "chatml", "zephyr", "phi3"],
    "max_seq_length_hint": None,
}

_PLAN_PROFILE_META: dict[str, dict[str, str]] = {
    "safe": {
        "title": "Safe",
        "description": "Highest stability and lowest VRAM pressure for first successful run.",
    },
    "balanced": {
        "title": "Balanced",
        "description": "Good default tradeoff for speed, stability, and quality.",
    },
    "max_quality": {
        "title": "Max Quality",
        "description": "Preserves most quality-oriented settings; may require more VRAM.",
    },
}

TRAINING_PLAN_PROFILES: tuple[str, ...] = tuple(_PLAN_PROFILE_META.keys())


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _coerce_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _coerce_float(value: Any, default: float, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off", ""}:
            return False
    return default


def _infer_model_capability(base_model: str) -> dict[str, Any]:
    normalized = str(base_model or "").strip().lower()
    if not normalized:
        return dict(_DEFAULT_MODEL_CAPABILITY)

    for item in _MODEL_CAPABILITY_MATRIX:
        tokens = item.get("tokens", ())
        if any(token in normalized for token in tokens):
            return {
                "family": item.get("family"),
                "architecture": item.get("architecture"),
                "supported_task_types": list(item.get("supported_task_types", [])),
                "supported_trainer_backends": list(item.get("supported_trainer_backends", [])),
                "recommended_chat_templates": list(item.get("recommended_chat_templates", [])),
                "max_seq_length_hint": item.get("max_seq_length_hint"),
            }
    return dict(_DEFAULT_MODEL_CAPABILITY)


def _normalize_task_type(value: Any, warnings: list[str]) -> str:
    candidate = str(value or "causal_lm").strip().lower()
    if candidate in SUPPORTED_TASK_TYPES:
        return candidate
    warnings.append(f"Unknown task_type '{candidate}', defaulting to causal_lm for preflight.")
    return "causal_lm"


def _normalize_trainer_backend(value: Any, warnings: list[str]) -> str:
    candidate = str(value or "auto").strip().lower()
    if candidate in SUPPORTED_TRAINER_BACKENDS:
        return candidate
    warnings.append(f"Unknown trainer_backend '{candidate}', defaulting to auto for preflight.")
    return "auto"


def _resolve_backend(requested_backend: str) -> str:
    if requested_backend == "auto":
        return "hf_trainer"
    return requested_backend


def _dependency_status_snapshot() -> dict[str, bool]:
    return {
        "torch": _module_available("torch"),
        "transformers": _module_available("transformers"),
        "datasets": _module_available("datasets"),
        "accelerate": _module_available("accelerate"),
        "trl": _module_available("trl"),
        "peft": _module_available("peft"),
        "bitsandbytes": _module_available("bitsandbytes"),
    }


def run_training_preflight(
    *,
    project_id: int,
    config: dict[str, Any],
    base_model: str | None = None,
) -> dict[str, Any]:
    """Run preflight checks before launching a training experiment."""
    resolved_config = dict(config or {})
    if base_model:
        resolved_config.setdefault("base_model", base_model)

    warnings: list[str] = []
    errors: list[str] = []
    hints: list[str] = []

    model_id = str(resolved_config.get("base_model") or base_model or "").strip()
    if not model_id:
        errors.append("base_model is required for training preflight.")

    capability = _infer_model_capability(model_id)
    architecture = str(capability.get("architecture") or "unknown")

    task_type = _normalize_task_type(resolved_config.get("task_type", "causal_lm"), warnings)
    trainer_backend_requested = _normalize_trainer_backend(
        resolved_config.get("trainer_backend", "auto"),
        warnings,
    )
    trainer_backend_effective = _resolve_backend(trainer_backend_requested)

    supported_task_types = list(capability.get("supported_task_types", []))
    if supported_task_types and task_type not in supported_task_types:
        errors.append(
            (
                f"task_type={task_type} is incompatible with base_model '{model_id}' "
                f"(family={capability.get('family')}, architecture={architecture}). "
                f"Supported: {', '.join(supported_task_types)}."
            )
        )

    supported_backends = list(capability.get("supported_trainer_backends", []))
    if supported_backends and trainer_backend_requested not in supported_backends:
        errors.append(
            (
                f"trainer_backend={trainer_backend_requested} is not supported for model family "
                f"{capability.get('family')}. Supported: {', '.join(supported_backends)}."
            )
        )

    if trainer_backend_effective == "trl_sft" and task_type != "causal_lm":
        errors.append("trainer_backend=trl_sft supports task_type=causal_lm only.")

    if architecture == "seq2seq" and task_type == "causal_lm":
        errors.append(
            (
                f"base_model '{model_id}' appears encoder-decoder/seq2seq; "
                "use task_type=seq2seq (or choose a causal-lm base model)."
            )
        )
    if architecture == "encoder" and task_type in {"causal_lm", "seq2seq"}:
        errors.append(
            (
                f"base_model '{model_id}' appears encoder-only; "
                "use task_type=classification or choose a generative base model."
            )
        )

    max_seq_length = _coerce_int(resolved_config.get("max_seq_length"), 2048, minimum=128)
    max_seq_length_hint = capability.get("max_seq_length_hint")
    seq_length_status = "unknown"
    if isinstance(max_seq_length_hint, int) and max_seq_length_hint > 0:
        if max_seq_length <= max_seq_length_hint:
            seq_length_status = "ok"
        else:
            seq_length_status = "above_hint"
            warnings.append(
                (
                    f"max_seq_length={max_seq_length} exceeds model family hint "
                    f"({max_seq_length_hint}). This may trigger OOM or truncation."
                )
            )
    elif max_seq_length >= 32768:
        seq_length_status = "high"
        warnings.append(
            "Very high max_seq_length configured without a known model limit; verify model context window."
        )

    chat_template = str(resolved_config.get("chat_template", "llama3")).strip().lower()
    recommended_templates = list(capability.get("recommended_chat_templates", []))
    if task_type == "classification":
        warnings.append(
            "task_type=classification ignores chat-template formatting for labels in current runtime."
        )
    elif chat_template and recommended_templates and chat_template not in recommended_templates:
        warnings.append(
            (
                f"chat_template='{chat_template}' is not in recommended templates for "
                f"family={capability.get('family')}: {', '.join(recommended_templates)}."
            )
        )

    use_fp16 = bool(resolved_config.get("fp16", False))
    use_bf16 = bool(resolved_config.get("bf16", True))
    use_flash_attention = bool(resolved_config.get("flash_attention", True))
    if use_fp16 and use_bf16:
        errors.append("fp16 and bf16 cannot both be true.")

    runtime_requested = str(resolved_config.get("training_runtime_id") or "auto").strip().lower() or "auto"
    runtime_id = ""
    runtime_source = "unresolved"
    runtime_backend = "unknown"
    runtime_required_dependencies: list[str] = []
    try:
        runtime_id, runtime_source = resolve_training_runtime_id(resolved_config)
        runtime_spec = get_runtime_spec(runtime_id)
        runtime_backend = str(runtime_spec.execution_backend or "unknown")
        runtime_required_dependencies = list(runtime_spec.required_dependencies or [])
        for item in validate_runtime(runtime_id):
            text = str(item).strip()
            if text and text not in errors:
                errors.append(text)
    except ValueError as runtime_error:
        errors.append(str(runtime_error))

    dependencies = _dependency_status_snapshot()
    for package_name in runtime_required_dependencies:
        if not dependencies.get(package_name, False):
            errors.append(
                f"Missing required dependency for runtime '{runtime_id}': {package_name}."
            )
    if trainer_backend_effective == "trl_sft" and not dependencies.get("trl", False):
        errors.append(
            "trainer_backend=trl_sft requested but dependency 'trl' is not installed."
        )
    if bool(resolved_config.get("use_lora", False)) and not dependencies.get("peft", False):
        errors.append(
            "use_lora=true requested but dependency 'peft' is not installed."
        )
    if str(resolved_config.get("optimizer", "adamw_torch")).strip().lower() == "paged_adamw_8bit":
        if not dependencies.get("bitsandbytes", False):
            warnings.append(
                "bitsandbytes is not installed; optimizer will fall back to adamw_torch."
            )

    runtime_environment: dict[str, Any] = {
        "cuda_available": False,
        "bf16_supported": False,
        "device_name": None,
        "device_capability": None,
    }
    if dependencies.get("torch", False):
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            runtime_environment["cuda_available"] = cuda_available
            if cuda_available:
                try:
                    runtime_environment["device_name"] = torch.cuda.get_device_name(0)
                except Exception:
                    runtime_environment["device_name"] = None
                try:
                    capability_tuple = torch.cuda.get_device_capability(0)
                    runtime_environment["device_capability"] = (
                        f"sm_{int(capability_tuple[0])}{int(capability_tuple[1])}"
                    )
                except Exception:
                    runtime_environment["device_capability"] = None
                try:
                    runtime_environment["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
                except Exception:
                    runtime_environment["bf16_supported"] = False

            if use_bf16 and cuda_available and not runtime_environment["bf16_supported"]:
                warnings.append(
                    "bf16 requested but current GPU/runtime reports bf16 unsupported; runtime may fall back."
                )
            if use_bf16 and not cuda_available:
                warnings.append(
                    "bf16 requested but CUDA is not available; runtime will train in non-bf16 mode."
                )
            if use_flash_attention and not cuda_available:
                warnings.append(
                    "flash_attention requested but CUDA is not available; runtime will fall back."
                )
        except Exception as torch_error:
            warnings.append(f"Unable to inspect torch runtime details: {torch_error}")

    project_prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    train_file = project_prepared_dir / "train.jsonl"
    val_file = project_prepared_dir / "val.jsonl"
    train_exists = train_file.exists()
    val_exists = val_file.exists()
    dataset_contract: dict[str, Any] | None = None
    if not train_exists:
        errors.append(
            (
                f"Prepared training dataset missing at {train_file}. "
                "Run dataset split/prep before starting training."
            )
        )
        hints.append(
            "Go to Dataset Prep, run Split, and verify train.jsonl is created before starting training."
        )
    if not val_exists:
        warnings.append(
            f"Validation dataset not found at {val_file}; evaluation during training may be skipped."
        )
    if train_exists:
        dataset_contract = analyze_prepared_dataset_contract(
            project_id=project_id,
            task_type=task_type,
            sample_size=400,
            min_coverage=0.9,
        )
        for item in dataset_contract.get("errors", []):
            text = str(item).strip()
            if text and text not in errors:
                errors.append(text)
        for item in dataset_contract.get("warnings", []):
            text = str(item).strip()
            if text and text not in warnings:
                warnings.append(text)
        for item in dataset_contract.get("hints", []):
            text = str(item).strip()
            if text and text not in hints:
                hints.append(text)

    capability_summary = {
        "matrix_version": MODEL_CAPABILITY_MATRIX_VERSION,
        "model": {
            "id": model_id,
            "family": capability.get("family"),
            "architecture": architecture,
            "supported_task_types": supported_task_types,
            "supported_trainer_backends": supported_backends,
            "recommended_chat_templates": recommended_templates,
        },
        "task_type": task_type,
        "trainer_backend_requested": trainer_backend_requested,
        "trainer_backend_effective": trainer_backend_effective,
        "sequence_length": {
            "requested": max_seq_length,
            "family_hint": max_seq_length_hint,
            "status": seq_length_status,
        },
        "runtime_backend": runtime_backend,
        "runtime": {
            "requested_runtime_id": runtime_requested,
            "resolved_runtime_id": runtime_id,
            "source": runtime_source,
            "required_dependencies": runtime_required_dependencies,
        },
        "runtime_environment": runtime_environment,
        "dependencies": dependencies,
        "dataset": {
            "prepared_dir": str(project_prepared_dir),
            "train_file": str(train_file),
            "train_file_exists": train_exists,
            "val_file": str(val_file),
            "val_file_exists": val_exists,
            "contract": dataset_contract or {},
        },
        "command_template_present": bool(settings.TRAINING_EXTERNAL_CMD.strip()),
    }

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "hints": hints,
        "capability_summary": capability_summary,
    }


def _set_planned_field(
    cfg: dict[str, Any],
    changes: list[dict[str, Any]],
    *,
    field: str,
    to_value: Any,
    reason: str,
) -> None:
    old_value = cfg.get(field)
    if old_value == to_value:
        return
    cfg[field] = to_value
    changes.append(
        {
            "field": field,
            "from": old_value,
            "to": to_value,
            "reason": reason,
        }
    )


def _apply_common_autofixes(
    cfg: dict[str, Any],
    changes: list[dict[str, Any]],
    *,
    capability_summary: dict[str, Any],
) -> None:
    dependencies = dict(capability_summary.get("dependencies") or {})
    runtime_env = dict(capability_summary.get("runtime_environment") or {})

    task_type = str(cfg.get("task_type", "causal_lm")).strip().lower()
    trainer_backend = str(cfg.get("trainer_backend", "auto")).strip().lower()

    if trainer_backend == "trl_sft" and task_type != "causal_lm":
        _set_planned_field(
            cfg,
            changes,
            field="trainer_backend",
            to_value="hf_trainer",
            reason="trl_sft supports causal_lm only",
        )
        trainer_backend = "hf_trainer"

    if trainer_backend == "trl_sft" and not bool(dependencies.get("trl", False)):
        _set_planned_field(
            cfg,
            changes,
            field="trainer_backend",
            to_value="hf_trainer",
            reason="trl not installed",
        )

    if _coerce_bool(cfg.get("use_lora", False)) and not bool(dependencies.get("peft", False)):
        _set_planned_field(
            cfg,
            changes,
            field="use_lora",
            to_value=False,
            reason="peft not installed",
        )

    optimizer = str(cfg.get("optimizer", "adamw_torch")).strip().lower()
    if optimizer == "paged_adamw_8bit" and not bool(dependencies.get("bitsandbytes", False)):
        _set_planned_field(
            cfg,
            changes,
            field="optimizer",
            to_value="adamw_torch",
            reason="bitsandbytes not installed",
        )

    use_fp16 = _coerce_bool(cfg.get("fp16", False))
    use_bf16 = _coerce_bool(cfg.get("bf16", True))
    bf16_supported = _coerce_bool(runtime_env.get("bf16_supported", False))
    if use_fp16 and use_bf16:
        if bf16_supported:
            _set_planned_field(
                cfg,
                changes,
                field="fp16",
                to_value=False,
                reason="fp16 and bf16 cannot both be enabled",
            )
        else:
            _set_planned_field(
                cfg,
                changes,
                field="bf16",
                to_value=False,
                reason="fp16 and bf16 cannot both be enabled",
            )

    cuda_available = _coerce_bool(runtime_env.get("cuda_available", False))
    if not cuda_available:
        if _coerce_bool(cfg.get("bf16", False)):
            _set_planned_field(
                cfg,
                changes,
                field="bf16",
                to_value=False,
                reason="CUDA unavailable",
            )
        if _coerce_bool(cfg.get("fp16", False)):
            _set_planned_field(
                cfg,
                changes,
                field="fp16",
                to_value=False,
                reason="CUDA unavailable",
            )
        if _coerce_bool(cfg.get("flash_attention", False)):
            _set_planned_field(
                cfg,
                changes,
                field="flash_attention",
                to_value=False,
                reason="CUDA unavailable",
            )


def _apply_profile_tuning(
    profile: str,
    cfg: dict[str, Any],
    changes: list[dict[str, Any]],
    *,
    capability_summary: dict[str, Any],
) -> None:
    max_seq_current = _coerce_int(cfg.get("max_seq_length"), 2048, minimum=128)
    batch_size_current = _coerce_int(cfg.get("batch_size"), 4, minimum=1)
    grad_accum_current = _coerce_int(cfg.get("gradient_accumulation_steps"), 4, minimum=1)
    max_retries_current = _coerce_int(cfg.get("max_oom_retries"), 2, minimum=0)
    cuda_available = _coerce_bool(
        dict(capability_summary.get("runtime_environment") or {}).get("cuda_available", False)
    )

    if profile == "safe":
        _set_planned_field(
            cfg,
            changes,
            field="batch_size",
            to_value=1,
            reason="safe profile lowers memory pressure",
        )
        _set_planned_field(
            cfg,
            changes,
            field="gradient_accumulation_steps",
            to_value=max(8, grad_accum_current),
            reason="safe profile preserves effective batch size while keeping batch_size=1",
        )
        _set_planned_field(
            cfg,
            changes,
            field="max_seq_length",
            to_value=min(max_seq_current, 1024),
            reason="safe profile reduces context length for OOM resilience",
        )
        _set_planned_field(
            cfg,
            changes,
            field="gradient_checkpointing",
            to_value=True,
            reason="safe profile reduces activation memory",
        )
        _set_planned_field(
            cfg,
            changes,
            field="sequence_packing",
            to_value=False,
            reason="safe profile prioritizes stability over throughput",
        )
        _set_planned_field(
            cfg,
            changes,
            field="flash_attention",
            to_value=False,
            reason="safe profile favors broad runtime compatibility",
        )
        _set_planned_field(
            cfg,
            changes,
            field="auto_oom_retry",
            to_value=True,
            reason="safe profile enables automatic OOM recovery",
        )
        _set_planned_field(
            cfg,
            changes,
            field="max_oom_retries",
            to_value=max(2, max_retries_current),
            reason="safe profile increases retry budget",
        )
        return

    if profile == "balanced":
        _set_planned_field(
            cfg,
            changes,
            field="batch_size",
            to_value=max(1, min(batch_size_current, 2)),
            reason="balanced profile controls peak VRAM while keeping throughput",
        )
        _set_planned_field(
            cfg,
            changes,
            field="gradient_accumulation_steps",
            to_value=max(4, grad_accum_current),
            reason="balanced profile preserves effective batch size",
        )
        _set_planned_field(
            cfg,
            changes,
            field="max_seq_length",
            to_value=min(max_seq_current, 2048),
            reason="balanced profile avoids common long-context OOM failures",
        )
        _set_planned_field(
            cfg,
            changes,
            field="gradient_checkpointing",
            to_value=True,
            reason="balanced profile improves memory efficiency",
        )
        _set_planned_field(
            cfg,
            changes,
            field="auto_oom_retry",
            to_value=True,
            reason="balanced profile enables automatic OOM recovery",
        )
        _set_planned_field(
            cfg,
            changes,
            field="max_oom_retries",
            to_value=max(2, max_retries_current),
            reason="balanced profile keeps retry budget ready",
        )
        if not cuda_available:
            _set_planned_field(
                cfg,
                changes,
                field="flash_attention",
                to_value=False,
                reason="CUDA unavailable",
            )
        return

    # max_quality keeps user intent mostly intact, with conservative guards.
    if not cuda_available and batch_size_current > 2:
        _set_planned_field(
            cfg,
            changes,
            field="batch_size",
            to_value=2,
            reason="CPU-only runtime with large batch size risks instability",
        )
    _set_planned_field(
        cfg,
        changes,
        field="auto_oom_retry",
        to_value=True,
        reason="max_quality profile still keeps OOM safety net",
    )
    _set_planned_field(
        cfg,
        changes,
        field="max_oom_retries",
        to_value=max(1, max_retries_current),
        reason="max_quality profile keeps minimal retry budget",
    )


def _estimate_vram_risk(config: dict[str, Any], capability_summary: dict[str, Any]) -> dict[str, Any]:
    runtime_env = dict(capability_summary.get("runtime_environment") or {})
    if not _coerce_bool(runtime_env.get("cuda_available", False)):
        return {
            "level": "cpu",
            "score": 0,
            "note": "CUDA not detected; VRAM pressure is not applicable.",
        }

    score = 0
    batch_size = _coerce_int(config.get("batch_size"), 4, minimum=1)
    max_seq_length = _coerce_int(config.get("max_seq_length"), 2048, minimum=128)
    grad_accum = _coerce_int(config.get("gradient_accumulation_steps"), 4, minimum=1)

    if batch_size >= 8:
        score += 3
    elif batch_size >= 4:
        score += 2
    elif batch_size >= 2:
        score += 1

    if max_seq_length >= 8192:
        score += 4
    elif max_seq_length >= 4096:
        score += 3
    elif max_seq_length >= 2048:
        score += 2
    elif max_seq_length >= 1024:
        score += 1

    if not _coerce_bool(config.get("use_lora", False), default=False):
        score += 2
    if not _coerce_bool(config.get("gradient_checkpointing", True), default=True):
        score += 1
    if _coerce_bool(config.get("flash_attention", False), default=False):
        score -= 1
    if grad_accum >= 8:
        score -= 1

    family_hint = capability_summary.get("sequence_length", {}).get("family_hint")
    if isinstance(family_hint, int) and family_hint > 0 and max_seq_length > family_hint:
        score += 2

    score = max(0, int(score))
    if score <= 2:
        level = "low"
    elif score <= 5:
        level = "medium"
    else:
        level = "high"

    return {
        "level": level,
        "score": score,
    }


def _pick_recommended_profile(suggestions: list[dict[str, Any]]) -> str:
    if not suggestions:
        return "balanced"

    priority = {"balanced": 0, "safe": 1, "max_quality": 2}
    candidates = [item for item in suggestions if bool(item.get("preflight", {}).get("ok", False))]
    pool = candidates if candidates else suggestions

    best = min(
        pool,
        key=lambda item: (
            int(item.get("estimated_vram_score", 999)),
            int(priority.get(str(item.get("profile", "")), 99)),
        ),
    )
    return str(best.get("profile", "balanced"))


def run_training_preflight_plan(
    *,
    project_id: int,
    config: dict[str, Any],
    base_model: str | None = None,
) -> dict[str, Any]:
    """Return suggested training configs tuned for stability/performance tradeoffs."""
    resolved_config = dict(config or {})
    if base_model:
        resolved_config.setdefault("base_model", base_model)
    model_id = str(resolved_config.get("base_model") or base_model or "")

    base_preflight = run_training_preflight(
        project_id=project_id,
        config=resolved_config,
        base_model=model_id,
    )
    base_summary = dict(base_preflight.get("capability_summary") or {})

    suggestions: list[dict[str, Any]] = []
    for profile in TRAINING_PLAN_PROFILES:
        profile_cfg = dict(resolved_config)
        changes: list[dict[str, Any]] = []
        _apply_common_autofixes(
            profile_cfg,
            changes,
            capability_summary=base_summary,
        )
        _apply_profile_tuning(
            profile,
            profile_cfg,
            changes,
            capability_summary=base_summary,
        )

        # normalize numeric/string drift
        profile_cfg["batch_size"] = _coerce_int(profile_cfg.get("batch_size"), 4, minimum=1)
        profile_cfg["gradient_accumulation_steps"] = _coerce_int(
            profile_cfg.get("gradient_accumulation_steps"),
            4,
            minimum=1,
        )
        profile_cfg["max_seq_length"] = _coerce_int(profile_cfg.get("max_seq_length"), 2048, minimum=128)
        profile_cfg["learning_rate"] = _coerce_float(profile_cfg.get("learning_rate"), 2e-4, minimum=1e-12)
        profile_cfg["max_oom_retries"] = _coerce_int(profile_cfg.get("max_oom_retries"), 2, minimum=0)
        profile_cfg["oom_retry_seq_shrink"] = _coerce_float(
            profile_cfg.get("oom_retry_seq_shrink"),
            0.75,
            minimum=0.1,
        )

        profile_preflight = run_training_preflight(
            project_id=project_id,
            config=profile_cfg,
            base_model=model_id,
        )
        risk = _estimate_vram_risk(profile_cfg, dict(profile_preflight.get("capability_summary") or {}))

        meta = dict(_PLAN_PROFILE_META.get(profile) or {})
        suggestions.append(
            {
                "profile": profile,
                "title": meta.get("title", profile),
                "description": meta.get("description", ""),
                "config": profile_cfg,
                "changes": changes,
                "estimated_vram_risk": risk.get("level", "unknown"),
                "estimated_vram_score": int(risk.get("score", 0) or 0),
                "estimated_vram_note": risk.get("note"),
                "preflight": profile_preflight,
            }
        )

    recommended_profile = _pick_recommended_profile(suggestions)
    return {
        "base_preflight": base_preflight,
        "suggestions": suggestions,
        "recommended_profile": recommended_profile,
        "profile_order": list(TRAINING_PLAN_PROFILES),
    }

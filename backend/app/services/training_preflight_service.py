"""Training capability matrix and preflight checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.alignment_service import (
    analyze_preference_dataset_contract,
    analyze_preference_dataset_quality,
)
from app.services.capability_contract_service import (
    SUPPORTED_TRAINING_TASK_TYPES,
    SUPPORTED_TRAINER_BACKENDS,
    evaluate_training_capability_contract,
    resolve_training_adapter_context,
)
from app.services.dataset_contract_service import analyze_prepared_dataset_contract
from app.services.model_introspection_service import introspect_hf_model
from app.services.training_runtime_service import (
    BUILTIN_SIMULATE_RUNTIME_ID,
    get_runtime_spec,
    resolve_training_runtime_id,
    validate_runtime,
)


MODEL_CAPABILITY_MATRIX_VERSION = "v1"

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
    "supported_task_types": sorted(SUPPORTED_TRAINING_TASK_TYPES),
    "supported_trainer_backends": sorted(SUPPORTED_TRAINER_BACKENDS),
    "recommended_chat_templates": ["llama3", "chatml", "zephyr", "phi3"],
    "max_seq_length_hint": None,
}

_SUPPORTED_MODEL_ARCHITECTURES: tuple[str, ...] = (
    "causal_lm",
    "seq2seq",
    "classification",
    "encoder",
)

_ARCHITECTURE_MODALITY_SUPPORT: dict[str, set[str]] = {
    "causal_lm": {"text", "vision_language", "audio_text", "multimodal"},
    "seq2seq": {"text", "vision_language", "audio_text"},
    "classification": {"text"},
    "encoder": {"text"},
}

_MEDIA_IMAGE_FIELD_CANDIDATES: tuple[str, ...] = (
    "image_path",
    "image",
    "image_url",
    "image_file",
    "path",
)
_MEDIA_AUDIO_FIELD_CANDIDATES: tuple[str, ...] = (
    "audio_path",
    "audio",
    "audio_url",
    "audio_file",
    "path",
)
_MEDIA_IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".avif",
}
_MEDIA_AUDIO_EXTENSIONS: set[str] = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".webm",
    ".aiff",
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

_PLAN_PROFILE_ALIASES: dict[str, str] = {
    "safe": "safe",
    "balanced": "balanced",
    "max_quality": "max_quality",
    "fastest": "safe",
    "best_quality": "max_quality",
}

TRAINING_PLAN_PROFILES: tuple[str, ...] = tuple(_PLAN_PROFILE_META.keys())


def normalize_training_plan_profile(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    return _PLAN_PROFILE_ALIASES.get(token)


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

    introspection = introspect_hf_model(
        model_id=str(base_model or "").strip(),
        allow_network=True,
        timeout_seconds=1.2,
    )
    architecture = str(introspection.get("architecture") or "unknown").strip().lower()
    if architecture not in {"causal_lm", "seq2seq", "classification"}:
        fallback = dict(_DEFAULT_MODEL_CAPABILITY)
        family_hint = str(introspection.get("model_type") or "").strip().lower()
        if family_hint:
            fallback["family"] = family_hint
        fallback["max_seq_length_hint"] = introspection.get("context_length")
        fallback["introspection"] = introspection
        return fallback

    if architecture == "causal_lm":
        supported_task_types = ["causal_lm", "classification"]
        supported_trainer_backends = ["auto", "hf_trainer", "trl_sft"]
        recommended_chat_templates = ["llama3", "chatml"]
    elif architecture == "seq2seq":
        supported_task_types = ["seq2seq", "classification"]
        supported_trainer_backends = ["auto", "hf_trainer"]
        recommended_chat_templates = ["chatml", "llama3"]
    else:
        supported_task_types = ["classification"]
        supported_trainer_backends = ["auto", "hf_trainer"]
        recommended_chat_templates = []

    family_hint = str(introspection.get("model_type") or "").strip().lower() or "introspected"
    return {
        "family": family_hint,
        "architecture": architecture,
        "supported_task_types": supported_task_types,
        "supported_trainer_backends": supported_trainer_backends,
        "recommended_chat_templates": recommended_chat_templates,
        "max_seq_length_hint": introspection.get("context_length"),
        "introspection": introspection,
    }


def evaluate_training_base_model_compatibility(
    *,
    base_model: str | None,
) -> dict[str, Any]:
    """Validate that base model architecture is supported by current training stack."""
    model_id = str(base_model or "").strip()
    errors: list[str] = []
    warnings: list[str] = []
    hints: list[str] = []

    if not model_id:
        errors.append("base_model is required for training preflight.")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "hints": hints,
            "capability": dict(_DEFAULT_MODEL_CAPABILITY),
        }

    capability = _infer_model_capability(model_id)
    architecture = str(capability.get("architecture") or "unknown").strip().lower()
    introspection = dict(capability.get("introspection") or {})
    supported_architectures = sorted(set(_SUPPORTED_MODEL_ARCHITECTURES))

    if architecture not in _SUPPORTED_MODEL_ARCHITECTURES:
        errors.append(
            (
                f"base_model '{model_id}' has unsupported or unresolved architecture '{architecture}'. "
                f"Supported architectures: {', '.join(supported_architectures)}."
            )
        )
        source = str(introspection.get("source") or "none").strip().lower() or "none"
        if source == "none":
            hints.append(
                "Model metadata could not be resolved. Verify Hugging Face model id or local path to config.json."
            )
        hints.append(
            "Use Training > Config > Introspect Model to verify architecture/context before launching."
        )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "hints": hints,
        "capability": capability,
    }


def _normalize_task_type(value: Any, warnings: list[str]) -> str:
    candidate = str(value or "causal_lm").strip().lower()
    if candidate in SUPPORTED_TRAINING_TASK_TYPES:
        return candidate
    warnings.append(f"Unknown task_type '{candidate}', defaulting to causal_lm for preflight.")
    return "causal_lm"


def _normalize_trainer_backend(value: Any, warnings: list[str]) -> str:
    candidate = str(value or "auto").strip().lower()
    if candidate in SUPPORTED_TRAINER_BACKENDS:
        return candidate
    warnings.append(f"Unknown trainer_backend '{candidate}', defaulting to auto for preflight.")
    return "auto"


def _normalize_training_mode(value: Any) -> str:
    raw = value
    if hasattr(value, "value"):
        raw = getattr(value, "value")
    candidate = str(raw or "sft").strip().lower()
    if candidate in {"sft", "domain_pretrain", "dpo", "orpo"}:
        return candidate
    return "sft"


def _resolve_backend(requested_backend: str) -> str:
    if requested_backend == "auto":
        return "hf_trainer"
    return requested_backend


def _evaluate_model_dataset_modality_compatibility(
    *,
    architecture: str | None,
    adapter_modality: str | None,
) -> dict[str, Any]:
    model_arch = str(architecture or "unknown").strip().lower()
    modality = str(adapter_modality or "text").strip().lower() or "text"
    errors: list[str] = []
    warnings: list[str] = []
    hints: list[str] = []

    supported_modalities = set(_ARCHITECTURE_MODALITY_SUPPORT.get(model_arch) or {"text"})
    if modality not in {"text", "vision_language", "audio_text", "multimodal"}:
        modality = "text"

    if modality not in supported_modalities:
        errors.append(
            (
                f"base_model architecture '{model_arch}' supports modalities "
                f"{', '.join(sorted(supported_modalities))}, but adapter resolves modality '{modality}'."
            )
        )
        hints.append(
            "Choose a text adapter/task profile for this model architecture, or switch to a multimodal-capable base model."
        )

    if model_arch == "causal_lm" and modality in {"vision_language", "audio_text", "multimodal"}:
        warnings.append(
            (
                f"Adapter modality '{modality}' selected with causal_lm architecture. "
                "Verify the base model exposes multimodal forward inputs (pixel_values/input_features/input_values); "
                "otherwise runtime may fall back to text-only markers."
            )
        )
        hints.append(
            "Prefer explicitly multimodal checkpoints when training on image/audio rows."
        )

    if model_arch == "seq2seq" and modality == "multimodal":
        warnings.append(
            (
                "seq2seq multimodal beta currently expects single-modality batches "
                "(vision_language or audio_text). Mixed image+audio rows may use fallback handling."
            )
        )
        hints.append(
            "Split mixed-modality datasets into single-modality runs for more predictable behavior."
        )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "hints": hints,
        "summary": {
            "architecture": model_arch,
            "adapter_modality": modality,
            "supported_modalities": sorted(supported_modalities),
        },
    }


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


def _pick_media_ref(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _sample_jsonl_rows(path: Path, *, sample_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= max(1, int(sample_size)):
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _is_remote_media_ref(value: str) -> bool:
    token = str(value or "").strip().lower()
    return token.startswith(("http://", "https://", "s3://", "gs://"))


def _resolve_local_media_ref(
    value: str,
    *,
    search_roots: list[Path],
) -> Path | None:
    token = str(value or "").strip()
    if not token or _is_remote_media_ref(token):
        return None

    candidate = Path(token).expanduser()
    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
        except Exception:
            return None
        return resolved if resolved.exists() else None

    for root in search_roots:
        try:
            resolved = (root / candidate).expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return resolved
    return None


def _analyze_multimodal_media_contract(
    *,
    train_file: Path,
    adapter_modality: str | None,
    require_media: bool = False,
    sample_size: int = 400,
) -> dict[str, Any]:
    strict_require_media = bool(require_media)
    expected_modality = str(adapter_modality or "text").strip().lower() or "text"
    if expected_modality not in {"text", "vision_language", "audio_text", "multimodal"}:
        expected_modality = "text"

    rows = _sample_jsonl_rows(train_file, sample_size=max(40, min(5000, int(sample_size or 400))))
    sampled_rows = len(rows)
    media_rows = 0
    image_rows = 0
    audio_rows = 0
    multimodal_rows = 0
    resolved_local_images = 0
    resolved_local_audios = 0
    missing_local_images = 0
    missing_local_audios = 0
    remote_image_refs = 0
    remote_audio_refs = 0
    errors: list[str] = []
    warnings: list[str] = []
    hints: list[str] = []

    search_roots = [
        train_file.parent,
        train_file.parent.parent,
        settings.DATA_DIR,
    ]

    for row in rows:
        image_ref = _pick_media_ref(row, _MEDIA_IMAGE_FIELD_CANDIDATES)
        audio_ref = _pick_media_ref(row, _MEDIA_AUDIO_FIELD_CANDIDATES)
        if image_ref and audio_ref and image_ref == audio_ref:
            suffix = Path(image_ref).suffix.strip().lower()
            if suffix in _MEDIA_AUDIO_EXTENSIONS:
                image_ref = ""
            elif suffix in _MEDIA_IMAGE_EXTENSIONS:
                audio_ref = ""
            else:
                audio_ref = ""

        has_image = bool(image_ref)
        has_audio = bool(audio_ref)
        if has_image or has_audio:
            media_rows += 1
        if has_image:
            image_rows += 1
            if _is_remote_media_ref(image_ref):
                remote_image_refs += 1
            elif _resolve_local_media_ref(image_ref, search_roots=search_roots) is not None:
                resolved_local_images += 1
            else:
                missing_local_images += 1
        if has_audio:
            audio_rows += 1
            if _is_remote_media_ref(audio_ref):
                remote_audio_refs += 1
            elif _resolve_local_media_ref(audio_ref, search_roots=search_roots) is not None:
                resolved_local_audios += 1
            else:
                missing_local_audios += 1
        if has_image and has_audio:
            multimodal_rows += 1

    if expected_modality in {"vision_language", "multimodal"} and image_rows <= 0:
        errors.append(
            (
                f"Adapter modality '{expected_modality}' expects image references, "
                "but sampled train rows contain no image_path/image fields."
            )
        )
        hints.append("Re-run Dataset Prep with a vision adapter or ensure image_path is present in mapped rows.")
    if expected_modality in {"audio_text", "multimodal"} and audio_rows <= 0:
        errors.append(
            (
                f"Adapter modality '{expected_modality}' expects audio references, "
                "but sampled train rows contain no audio_path/audio fields."
            )
        )
        hints.append("Re-run Dataset Prep with an audio adapter or ensure audio_path is present in mapped rows.")

    total_local_media_refs = (image_rows - remote_image_refs) + (audio_rows - remote_audio_refs)
    total_missing_local_refs = missing_local_images + missing_local_audios
    missing_ratio = (
        float(total_missing_local_refs / total_local_media_refs)
        if total_local_media_refs > 0
        else 0.0
    )
    missing_error_threshold = max(3, int(total_local_media_refs * 0.35))
    if total_missing_local_refs > 0:
        message = (
            f"Missing local media assets for {total_missing_local_refs}/{max(1, total_local_media_refs)} "
            "sampled local media references; multimodal batches may fall back to text-only markers."
        )
        if strict_require_media or total_missing_local_refs >= missing_error_threshold:
            errors.append(message)
        else:
            warnings.append(message)
        hints.append(
            "Place media files under the prepared dataset directory (or provide absolute paths) before training."
        )

    total_remote_refs = remote_image_refs + remote_audio_refs
    if total_remote_refs > 0:
        bucket = errors if strict_require_media else warnings
        bucket.append(
            (
                f"Detected {total_remote_refs} remote media URL reference(s) in sampled rows. "
                "Current multimodal runtime resolves local files only and will use text-only fallback for remote assets."
            )
        )
        hints.append(
            "Download remote media to local paths inside prepared data, then rerun preflight."
        )

    if image_rows > 0 and audio_rows > 0:
        mixed_message = (
            "Sampled dataset includes both image and audio references; mixed-modality batches may use beta fallback handling."
        )
        if strict_require_media:
            errors.append(
                (
                    "Sampled dataset includes both image and audio references, but multimodal_require_media=true "
                    "expects no text-fallback path for mixed-modality batches."
                )
            )
        else:
            warnings.append(mixed_message)
        hints.append(
            "Prefer separate single-modality runs (vision-only or audio-only) for predictable training behavior."
        )

    return {
        "ok": len(errors) == 0,
        "require_media": strict_require_media,
        "expected_modality": expected_modality,
        "sampled_rows": sampled_rows,
        "media_rows": media_rows,
        "image_rows": image_rows,
        "audio_rows": audio_rows,
        "multimodal_rows": multimodal_rows,
        "resolved_local_images": resolved_local_images,
        "resolved_local_audios": resolved_local_audios,
        "missing_local_images": missing_local_images,
        "missing_local_audios": missing_local_audios,
        "remote_image_refs": remote_image_refs,
        "remote_audio_refs": remote_audio_refs,
        "missing_local_ratio": round(missing_ratio, 6),
        "errors": errors,
        "warnings": warnings,
        "hints": hints,
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
    model_compat = evaluate_training_base_model_compatibility(base_model=model_id)
    capability = dict(model_compat.get("capability") or _DEFAULT_MODEL_CAPABILITY)
    for item in model_compat.get("errors", []):
        token = str(item).strip()
        if token and token not in errors:
            errors.append(token)
    for item in model_compat.get("warnings", []):
        token = str(item).strip()
        if token and token not in warnings:
            warnings.append(token)
    for item in model_compat.get("hints", []):
        token = str(item).strip()
        if token and token not in hints:
            hints.append(token)

    architecture = str(capability.get("architecture") or "unknown")

    task_type = _normalize_task_type(resolved_config.get("task_type", "causal_lm"), warnings)
    trainer_backend_requested = _normalize_trainer_backend(
        resolved_config.get("trainer_backend", "auto"),
        warnings,
    )
    trainer_backend_effective = _resolve_backend(trainer_backend_requested)
    training_mode = _normalize_training_mode(resolved_config.get("training_mode"))
    multimodal_require_media = _coerce_bool(
        resolved_config.get("multimodal_require_media"),
        False,
    )

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
    if training_mode in {"dpo", "orpo"} and task_type != "causal_lm":
        errors.append(f"training_mode={training_mode} supports task_type=causal_lm only.")

    distillation_enabled = _coerce_bool(resolved_config.get("distillation_enabled"), False)
    distillation_teacher_model = str(resolved_config.get("distillation_teacher_model") or "").strip()
    if distillation_enabled:
        if training_mode in {"dpo", "orpo"}:
            errors.append(
                "distillation_enabled is incompatible with training_mode=dpo/orpo."
            )
        if task_type != "causal_lm":
            errors.append(
                "distillation_enabled currently supports task_type=causal_lm only."
            )
        if not distillation_teacher_model:
            errors.append(
                "distillation_enabled=true requires distillation_teacher_model."
            )
        elif distillation_teacher_model == model_id:
            warnings.append(
                "distillation_teacher_model matches base_model; transfer gain may be limited."
            )

        distillation_temperature = _coerce_float(
            resolved_config.get("distillation_temperature"),
            2.0,
            minimum=0.1,
        )
        if distillation_temperature <= 0.0:
            errors.append("distillation_temperature must be > 0.")
        distillation_alpha = max(
            0.0,
            min(_coerce_float(resolved_config.get("distillation_alpha"), 0.6, minimum=0.0), 1.0),
        )
        if distillation_alpha <= 0.0:
            warnings.append(
                "distillation_alpha is 0, so training uses only distillation loss and no supervised CE."
            )
        if distillation_alpha >= 1.0:
            warnings.append(
                "distillation_alpha is 1, so distillation loss is disabled and run behaves like standard SFT."
            )

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
    runtime_supported_modalities: list[str] = []
    runtime_modalities_declared = False
    try:
        runtime_id, runtime_source = resolve_training_runtime_id(resolved_config)
        runtime_spec = get_runtime_spec(runtime_id)
        runtime_backend = str(runtime_spec.execution_backend or "unknown")
        runtime_required_dependencies = list(runtime_spec.required_dependencies or [])
        runtime_supported_modalities = [
            str(item).strip().lower()
            for item in list(getattr(runtime_spec, "supported_modalities", []) or [])
            if str(item).strip()
        ]
        runtime_modalities_declared = bool(
            getattr(runtime_spec, "declares_supported_modalities", False)
        )
        for item in validate_runtime(runtime_id):
            text = str(item).strip()
            if text and text not in errors:
                errors.append(text)
    except ValueError as runtime_error:
        errors.append(str(runtime_error))

    is_simulate_runtime = runtime_id == BUILTIN_SIMULATE_RUNTIME_ID

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
    if training_mode in {"dpo", "orpo"} and not dependencies.get("trl", False):
        errors.append(
            (
                f"training_mode={training_mode} requires dependency 'trl' "
                "for native pairwise objective trainers."
            )
        )
        hints.append("Install trl in the backend/worker environment before DPO/ORPO runs.")
    if distillation_enabled and not dependencies.get("torch", False):
        errors.append(
            "distillation_enabled=true requires dependency 'torch'."
        )
    if distillation_enabled and not dependencies.get("transformers", False):
        errors.append(
            "distillation_enabled=true requires dependency 'transformers'."
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
    media_contract: dict[str, Any] | None = None
    alignment_contract: dict[str, Any] | None = None
    alignment_quality: dict[str, Any] | None = None
    if not train_exists:
        if is_simulate_runtime:
            warnings.append(
                (
                    f"Prepared training dataset missing at {train_file}. "
                    "Built-in simulate runtime can still run without dataset artifacts."
                )
            )
            hints.append(
                "Run Dataset Prep split to exercise full data-contract checks before switching to non-simulated runtimes."
            )
        else:
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

    if train_exists and training_mode in {"dpo", "orpo"}:
        alignment_contract = analyze_preference_dataset_contract(
            project_id=project_id,
            sample_size=400,
            min_coverage=0.85,
        )
        for item in alignment_contract.get("errors", []):
            text = str(item).strip()
            if text and text not in errors:
                errors.append(text)
        for item in alignment_contract.get("warnings", []):
            text = str(item).strip()
            if text and text not in warnings:
                warnings.append(text)
        for item in alignment_contract.get("hints", []):
            text = str(item).strip()
            if text and text not in hints:
                hints.append(text)

        quality_threshold = _coerce_float(
            resolved_config.get("alignment_quality_threshold"),
            3.0,
            minimum=1.0,
        )
        min_keep_ratio = max(
            0.05,
            min(
                _coerce_float(
                    resolved_config.get("alignment_min_keep_ratio"),
                    0.4,
                    minimum=0.05,
                ),
                1.0,
            ),
        )
        alignment_quality = analyze_preference_dataset_quality(
            project_id=project_id,
            sample_size=400,
            quality_threshold=quality_threshold,
        )
        scored_count = int(alignment_quality.get("scored_count", 0))
        keep_count = int(alignment_quality.get("keep_count", 0))
        keep_ratio = float(keep_count / scored_count) if scored_count > 0 else 0.0

        if scored_count <= 0:
            errors.append(
                (
                    "No scorable preference rows found for alignment mode. "
                    "Provide prompt/chosen/rejected rows before DPO/ORPO training."
                )
            )
        elif keep_count <= 0:
            if is_simulate_runtime:
                warnings.append(
                    (
                        "Judge quality threshold would drop all preference pairs. "
                        "Simulate runtime will continue, but lower alignment_quality_threshold "
                        "or improve dataset quality before real training."
                    )
                )
            else:
                errors.append(
                    (
                        "Judge quality threshold would drop all preference pairs. "
                        "Lower alignment_quality_threshold or improve dataset quality."
                    )
                )
        elif keep_ratio < min_keep_ratio:
            warnings.append(
                (
                    f"Alignment keep ratio {keep_ratio:.1%} is below configured minimum "
                    f"{min_keep_ratio:.0%}."
                )
            )
            hints.append(
                "Lower alignment_quality_threshold or import cleaner preference pairs."
            )

    adapter_context = resolve_training_adapter_context(
        project_id=project_id,
        config=resolved_config,
    )
    if train_exists:
        media_contract = _analyze_multimodal_media_contract(
            train_file=train_file,
            adapter_modality=adapter_context.get("adapter_modality"),
            require_media=multimodal_require_media,
            sample_size=400,
        )
        for item in media_contract.get("errors", []):
            text = str(item).strip()
            if text and text not in errors:
                errors.append(text)
        for item in media_contract.get("warnings", []):
            text = str(item).strip()
            if text and text not in warnings:
                warnings.append(text)
        for item in media_contract.get("hints", []):
            text = str(item).strip()
            if text and text not in hints:
                hints.append(text)

    model_modality_contract = _evaluate_model_dataset_modality_compatibility(
        architecture=architecture,
        adapter_modality=adapter_context.get("adapter_modality"),
    )
    for item in model_modality_contract.get("errors", []):
        text = str(item).strip()
        if text and text not in errors:
            errors.append(text)
    for item in model_modality_contract.get("warnings", []):
        text = str(item).strip()
        if text and text not in warnings:
            warnings.append(text)
    for item in model_modality_contract.get("hints", []):
        text = str(item).strip()
        if text and text not in hints:
            hints.append(text)

    capability_contract = evaluate_training_capability_contract(
        task_type=task_type,
        training_mode=training_mode,
        trainer_backend_requested=trainer_backend_requested,
        runtime_id=runtime_id,
        runtime_backend=runtime_backend,
        adapter_id=str(adapter_context.get("adapter_id") or ""),
        adapter_contract=dict(adapter_context.get("adapter_contract") or {}),
        adapter_task_profile=adapter_context.get("task_profile"),
    )
    for item in capability_contract.get("errors", []):
        text = str(item).strip()
        if text and text not in errors:
            errors.append(text)
    for item in capability_contract.get("warnings", []):
        text = str(item).strip()
        if text and text not in warnings:
            warnings.append(text)
    for item in capability_contract.get("hints", []):
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
            "introspection": capability.get("introspection"),
            "compatibility_gate": {
                "ok": bool(model_compat.get("ok", False)),
                "errors": list(model_compat.get("errors", [])),
                "hints": list(model_compat.get("hints", [])),
                "supported_architectures": sorted(set(_SUPPORTED_MODEL_ARCHITECTURES)),
            },
        },
        "task_type": task_type,
        "training_mode": training_mode,
        "distillation": {
            "enabled": distillation_enabled,
            "teacher_model": distillation_teacher_model or None,
            "alpha": _coerce_float(resolved_config.get("distillation_alpha"), 0.6, minimum=0.0),
            "temperature": _coerce_float(resolved_config.get("distillation_temperature"), 2.0, minimum=0.1),
            "hidden_state_weight": _coerce_float(
                resolved_config.get("distillation_hidden_state_weight"),
                0.0,
                minimum=0.0,
            ),
            "hidden_state_loss": str(resolved_config.get("distillation_hidden_state_loss") or "mse").strip().lower() or "mse",
        },
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
            "supported_modalities": runtime_supported_modalities,
            "modalities_declared": runtime_modalities_declared,
        },
        "runtime_environment": runtime_environment,
        "dependencies": dependencies,
        "dataset": {
            "prepared_dir": str(project_prepared_dir),
            "train_file": str(train_file),
            "train_file_exists": train_exists,
            "val_file": str(val_file),
            "val_file_exists": val_exists,
            "adapter_context": adapter_context,
            "contract": dataset_contract or {},
            "media_contract": media_contract or {},
            "alignment_contract": alignment_contract or {},
            "alignment_quality": alignment_quality or {},
        },
        "model_modality_contract": {
            **dict(model_modality_contract.get("summary") or {}),
            "ok": bool(model_modality_contract.get("ok", False)),
            "errors": [str(item) for item in list(model_modality_contract.get("errors") or []) if str(item).strip()],
            "warnings": [str(item) for item in list(model_modality_contract.get("warnings") or []) if str(item).strip()],
            "hints": [str(item) for item in list(model_modality_contract.get("hints") or []) if str(item).strip()],
        },
        "capability_contract": capability_contract.get("summary", {}),
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
    runtime_summary = dict(capability_summary.get("runtime") or {})
    capability_contract = dict(capability_summary.get("capability_contract") or {})
    dataset_summary = dict(capability_summary.get("dataset") or {})
    media_contract = dict(dataset_summary.get("media_contract") or {})
    adapter_context = dict(dataset_summary.get("adapter_context") or {})

    task_type = str(cfg.get("task_type", "causal_lm")).strip().lower()
    trainer_backend = str(cfg.get("trainer_backend", "auto")).strip().lower()
    training_mode = _normalize_training_mode(cfg.get("training_mode"))

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

    multimodal_require_media = _coerce_bool(cfg.get("multimodal_require_media"), False)
    if multimodal_require_media:
        adapter_modality = str(
            capability_contract.get("adapter_modality")
            or adapter_context.get("adapter_modality")
            or "text"
        ).strip().lower()
        if adapter_modality not in {"text", "vision_language", "audio_text", "multimodal"}:
            adapter_modality = "text"

        runtime_modalities_raw = capability_contract.get("runtime_supported_modalities")
        if not isinstance(runtime_modalities_raw, (list, tuple, set)):
            runtime_modalities_raw = runtime_summary.get("supported_modalities")

        runtime_supported_modalities: set[str] = set()
        if isinstance(runtime_modalities_raw, (list, tuple, set)):
            for item in runtime_modalities_raw:
                token = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
                if token == "multi_modal":
                    token = "multimodal"
                elif token in {"vision", "image", "image_text"}:
                    token = "vision_language"
                elif token == "audio":
                    token = "audio_text"
                if token in {"text", "vision_language", "audio_text", "multimodal"}:
                    runtime_supported_modalities.add(token)
        if not runtime_supported_modalities:
            runtime_supported_modalities = {"text"}

        runtime_modalities_declared = _coerce_bool(
            capability_contract.get("runtime_modalities_declared"),
            _coerce_bool(runtime_summary.get("modalities_declared"), False),
        )

        missing_local_refs = (
            _coerce_int(media_contract.get("missing_local_images"), 0, minimum=0)
            + _coerce_int(media_contract.get("missing_local_audios"), 0, minimum=0)
        )
        remote_media_refs = (
            _coerce_int(media_contract.get("remote_image_refs"), 0, minimum=0)
            + _coerce_int(media_contract.get("remote_audio_refs"), 0, minimum=0)
        )
        mixed_modality_rows = _coerce_int(media_contract.get("multimodal_rows"), 0, minimum=0)

        if training_mode in {"dpo", "orpo"}:
            _set_planned_field(
                cfg,
                changes,
                field="multimodal_require_media",
                to_value=False,
                reason="multimodal strict media mode supports sft/domain_pretrain only",
            )
        elif mixed_modality_rows > 0:
            _set_planned_field(
                cfg,
                changes,
                field="multimodal_require_media",
                to_value=False,
                reason="strict media mode does not support sampled mixed image+audio rows",
            )
        elif missing_local_refs > 0:
            _set_planned_field(
                cfg,
                changes,
                field="multimodal_require_media",
                to_value=False,
                reason="strict media mode requires all sampled local media assets to resolve",
            )
        elif remote_media_refs > 0:
            _set_planned_field(
                cfg,
                changes,
                field="multimodal_require_media",
                to_value=False,
                reason="strict media mode requires local media files (remote URLs detected)",
            )
        elif (
            adapter_modality != "text"
            and runtime_modalities_declared
            and adapter_modality not in runtime_supported_modalities
        ):
            _set_planned_field(
                cfg,
                changes,
                field="multimodal_require_media",
                to_value=False,
                reason="runtime modality contract does not include adapter modality for strict media mode",
            )
        elif (
            adapter_modality != "text"
            and not runtime_modalities_declared
        ):
            _set_planned_field(
                cfg,
                changes,
                field="multimodal_require_media",
                to_value=False,
                reason="runtime does not declare supported_modalities for strict media mode",
            )


def _apply_profile_tuning(
    profile: str,
    cfg: dict[str, Any],
    changes: list[dict[str, Any]],
    *,
    capability_summary: dict[str, Any],
) -> None:
    resolved_profile = normalize_training_plan_profile(profile) or "balanced"
    max_seq_current = _coerce_int(cfg.get("max_seq_length"), 2048, minimum=128)
    batch_size_current = _coerce_int(cfg.get("batch_size"), 4, minimum=1)
    grad_accum_current = _coerce_int(cfg.get("gradient_accumulation_steps"), 4, minimum=1)
    max_retries_current = _coerce_int(cfg.get("max_oom_retries"), 2, minimum=0)
    cuda_available = _coerce_bool(
        dict(capability_summary.get("runtime_environment") or {}).get("cuda_available", False)
    )

    if resolved_profile == "safe":
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

    if resolved_profile == "balanced":
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
            int(priority.get(str(normalize_training_plan_profile(item.get("profile")) or ""), 99)),
        ),
    )
    return str(normalize_training_plan_profile(best.get("profile")) or "balanced")


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

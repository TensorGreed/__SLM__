"""Universal base model registry normalization + compatibility engine."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import parse, request

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base_model_registry import (
    BaseModelRegistryEntry,
    BaseModelSourceType,
)
from app.models.domain_blueprint import DomainBlueprintRevision
from app.models.project import Project
from app.schemas.base_model_registry import CompatibilityReason
from app.services.artifact_registry_service import publish_artifact
from app.services.data_adapter_service import DEFAULT_ADAPTER_ID, list_data_adapter_catalog
from app.services.dataset_service import resolve_project_dataset_adapter_preference
from app.services.model_introspection_service import introspect_hf_model
from app.services.model_selection_service import list_model_catalog_entries
from app.services.target_profile_service import check_compatibility, get_target_by_id, list_targets
from app.services.training_runtime_service import list_runtime_catalog


BASE_MODEL_REGISTRY_CONTRACT_VERSION = "slm.base_model_registry/v1"
MODEL_COMPATIBILITY_ENGINE_VERSION = "slm.model_compatibility_engine/v1"
_HTTP_USER_AGENT = "slm-platform/base-model-registry-v1"

_SUPPORTED_SOURCE_TYPES = {"huggingface", "local_path", "catalog"}
_TASK_FAMILY_ALIASES: dict[str, str] = {
    "instruction_sft": "instruction_sft",
    "instruction": "instruction_sft",
    "qa": "qa",
    "question_answering": "qa",
    "structured_extraction": "structured_extraction",
    "extraction": "structured_extraction",
    "classification": "classification",
    "moderation": "classification",
    "routing": "classification",
    "summarization": "summarization",
    "translation": "translation",
    "causal_lm": "instruction_sft",
    "seq2seq": "summarization",
}
_ARCH_TO_TASK_FAMILIES: dict[str, list[str]] = {
    "causal_lm": [
        "instruction_sft",
        "qa",
        "structured_extraction",
        "summarization",
        "classification",
    ],
    "seq2seq": [
        "summarization",
        "translation",
        "qa",
        "structured_extraction",
    ],
    "classification": [
        "classification",
        "routing",
        "moderation",
    ],
}
_ARCH_TO_CHAT_TEMPLATE_HINT: dict[str, str] = {
    "causal_lm": "chatml_or_family_template",
    "seq2seq": "plain_text_prompting",
    "classification": "plain_text_prompting",
}
_MODALITY_ALIASES: dict[str, str] = {
    "text": "text",
    "image": "image",
    "vision": "image",
    "audio": "audio",
    "vision_language": "image",
    "audio_text": "audio",
    "multimodal": "multimodal",
}
_MODALITY_TO_RUNTIME_MODALITIES: dict[str, str] = {
    "text": "text",
    "image": "vision_language",
    "audio": "audio_text",
    "multimodal": "multimodal",
}
_TRAINING_TASK_BY_FAMILY: dict[str, list[str]] = {
    "instruction_sft": ["causal_lm"],
    "qa": ["causal_lm", "seq2seq"],
    "structured_extraction": ["causal_lm", "seq2seq"],
    "summarization": ["seq2seq", "causal_lm"],
    "translation": ["seq2seq"],
    "classification": ["classification"],
    "routing": ["classification"],
    "moderation": ["classification"],
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_source_type(value: str) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token not in _SUPPORTED_SOURCE_TYPES:
        allowed = ", ".join(sorted(_SUPPORTED_SOURCE_TYPES))
        raise ValueError(f"Unsupported source_type '{value}'. Expected one of: {allowed}")
    return token


def _normalize_source_ref(source_type: str, source_ref: str) -> str:
    token = str(source_ref or "").strip()
    if not token:
        raise ValueError("source_ref is required")
    if source_type == "local_path":
        return str(Path(token).expanduser().resolve())
    return token


def _safe_model_token(value: str, *, fallback: str = "model") -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip().lower()).strip("-")
    return token[:48] if token else fallback


def build_model_key(source_type: str, source_ref: str) -> str:
    normalized_ref = _normalize_source_ref(source_type, source_ref)
    digest = hashlib.sha1(f"{source_type}|{normalized_ref}".encode("utf-8")).hexdigest()[:14]
    tail = Path(normalized_ref).name if source_type == "local_path" else normalized_ref.split("/")[-1]
    return f"{source_type}:{_safe_model_token(tail)}:{digest}"


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _read_json_url(url: str, *, timeout_seconds: float = 2.5) -> dict[str, Any] | None:
    req = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": _HTTP_USER_AGENT,
        },
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=max(0.2, float(timeout_seconds))) as resp:
            data = resp.read()
    except Exception:
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_model_family(source_ref: str, config_payload: dict[str, Any] | None) -> str:
    config = dict(config_payload or {})
    model_type = str(config.get("model_type") or "").strip().lower()
    if model_type:
        return model_type

    token = str(source_ref or "").strip().lower()
    if "/" in token:
        owner = token.split("/", 1)[0]
        if owner:
            return owner
    if Path(token).name:
        return _safe_model_token(Path(token).name, fallback="unknown")
    return "unknown"


def _extract_modalities(
    *,
    source_ref: str,
    config_payload: dict[str, Any] | None,
    hf_info_payload: dict[str, Any] | None,
) -> list[str]:
    values: list[str] = []
    haystack = " ".join(
        [
            str(source_ref or ""),
            json.dumps(config_payload or {}, ensure_ascii=True),
            json.dumps(hf_info_payload or {}, ensure_ascii=True),
        ]
    ).lower()

    if any(marker in haystack for marker in ("llava", "vision", "image", "clip")):
        values.append("image")
    if any(marker in haystack for marker in ("audio", "whisper", "wav2vec", "speech")):
        values.append("audio")
    values.append("text")

    deduped: list[str] = []
    for raw in values:
        token = _MODALITY_ALIASES.get(str(raw).strip().lower(), "")
        if token and token not in deduped:
            deduped.append(token)

    if {"text", "image", "audio"}.issubset(set(deduped)):
        return ["multimodal", "text", "image", "audio"]
    return deduped or ["text"]


def _extract_chat_template(
    *,
    architecture: str,
    tokenizer_payload: dict[str, Any] | None,
    source_ref: str,
) -> str | None:
    tokenizer = dict(tokenizer_payload or {})
    chat_template = tokenizer.get("chat_template")
    if isinstance(chat_template, str) and chat_template.strip():
        return chat_template.strip()

    token = str(source_ref or "").lower()
    if "llama" in token:
        return "llama3"
    if "qwen" in token:
        return "chatml"
    if "phi" in token:
        return "phi3"
    return _ARCH_TO_CHAT_TEMPLATE_HINT.get(architecture)


def _extract_task_families(
    *,
    architecture: str,
    hf_info_payload: dict[str, Any] | None,
    modalities: list[str],
) -> list[str]:
    out: list[str] = list(_ARCH_TO_TASK_FAMILIES.get(architecture, ["instruction_sft", "qa"]))
    info = dict(hf_info_payload or {})
    pipeline_tag = str(info.get("pipeline_tag") or "").strip().lower()
    mapped = _TASK_FAMILY_ALIASES.get(pipeline_tag)
    if mapped and mapped not in out:
        out.append(mapped)
    tags = [str(item).strip().lower() for item in list(info.get("tags") or [])]
    for tag in tags:
        mapped = _TASK_FAMILY_ALIASES.get(tag)
        if mapped and mapped not in out:
            out.append(mapped)

    if "image" in modalities and "classification" not in out:
        out.append("classification")
    return out


def _extract_quantization_support(
    *,
    architecture: str,
    modalities: list[str],
) -> dict[str, Any]:
    formats = ["huggingface", "onnx", "int8"]
    if architecture == "causal_lm":
        formats.extend(["gguf", "gptq", "awq"])
    if "image" in modalities or "audio" in modalities:
        formats = [fmt for fmt in formats if fmt not in {"gguf"}]

    deduped: list[str] = []
    for item in formats:
        token = str(item).strip().lower()
        if token and token not in deduped:
            deduped.append(token)
    return {
        "formats": deduped,
        "notes": [
            "Quantization support is inferred from architecture family and may require runtime-specific checks.",
        ],
    }


def _extract_training_mode_support(architecture: str) -> list[str]:
    if architecture == "classification":
        return ["sft", "domain_pretrain"]
    return ["sft", "domain_pretrain", "dpo", "orpo"]


def _estimate_hardware_needs(
    *,
    params_b: float | None,
    context_length: int | None,
    modalities: list[str],
    min_vram_hint: float | None = None,
    ideal_vram_hint: float | None = None,
) -> dict[str, Any]:
    params = float(params_b or 0.0)
    min_vram = float(min_vram_hint or 0.0)
    ideal_vram = float(ideal_vram_hint or 0.0)
    if min_vram <= 0 and params > 0:
        min_vram = round(max(2.0, (params * 1.9) + 1.5), 2)
    if ideal_vram <= 0 and min_vram > 0:
        ideal_vram = round(max(min_vram + 2.0, min_vram * 1.25), 2)

    recommended_device_classes = ["laptop", "server"]
    if params > 0 and params <= 2.0 and min_vram <= 6.5:
        recommended_device_classes = ["mobile", "laptop", "server"]
    elif params > 0 and params >= 10.0:
        recommended_device_classes = ["server"]

    if "image" in modalities or "audio" in modalities:
        recommended_device_classes = [item for item in recommended_device_classes if item != "mobile"]

    return {
        "estimated_min_vram_gb": round(min_vram, 2) if min_vram > 0 else None,
        "estimated_ideal_vram_gb": round(ideal_vram, 2) if ideal_vram > 0 else None,
        "recommended_device_classes": recommended_device_classes,
        "context_length_hint": int(context_length) if isinstance(context_length, int) and context_length > 0 else None,
        "cpu_only_viable": bool(min_vram > 0 and min_vram <= 5.0 and "image" not in modalities and "audio" not in modalities),
    }


def _coerce_parameter_count(params_estimate_b: float | None) -> int | None:
    if params_estimate_b is None:
        return None
    try:
        parsed = int(float(params_estimate_b) * 1_000_000_000.0)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _build_cache_fingerprint(payload: dict[str, Any]) -> str:
    material = {
        "source_type": payload.get("source_type"),
        "source_ref": payload.get("source_ref"),
        "architecture": payload.get("architecture"),
        "context_length": payload.get("context_length"),
        "params_estimate_b": payload.get("params_estimate_b"),
        "license": payload.get("license"),
        "supported_task_families": payload.get("supported_task_families"),
        "modalities": payload.get("modalities"),
        "deployment_target_compatibility": payload.get("deployment_target_compatibility"),
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _infer_required_context(task_family: str | None) -> int:
    token = _TASK_FAMILY_ALIASES.get(str(task_family or "").strip().lower(), "instruction_sft")
    if token == "structured_extraction":
        return 4096
    if token == "summarization":
        return 3072
    return 1024


def _safe_artifact_key_for_model(model_key: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(model_key or "").strip().lower())
    return f"compatibility.base_models.{slug[:120] or 'unknown'}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        token = str(raw or "").strip()
        if not token:
            continue
        marker = token.lower()
        if marker in seen:
            continue
        seen.add(marker)
        out.append(token)
    return out


def _build_reason(
    code: str,
    *,
    severity: str,
    message: str,
    unblock_actions: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> CompatibilityReason:
    return CompatibilityReason(
        code=code,
        severity=severity,
        message=message,
        unblock_actions=list(unblock_actions or []),
        evidence=dict(evidence or {}),
    )


def _resolve_runtime_modalities(runtime_id: str | None) -> tuple[str | None, list[str]]:
    catalog = list_runtime_catalog()
    runtimes = [item for item in list(catalog.get("runtimes") or []) if isinstance(item, dict)]
    default_runtime = str(catalog.get("default_runtime_id") or "").strip() or None
    selected_runtime = str(runtime_id or "").strip() or default_runtime
    if not selected_runtime:
        return None, []
    for row in runtimes:
        if str(row.get("runtime_id") or "").strip() == selected_runtime:
            return selected_runtime, [str(item).strip() for item in list(row.get("supported_modalities") or []) if str(item).strip()]
    return selected_runtime, []


def _extract_domain_blueprint_context(
    project: Project,
    blueprint: DomainBlueprintRevision | None,
) -> tuple[str | None, str | None, str | None]:
    if blueprint is not None:
        return (
            str(blueprint.task_family or "").strip() or None,
            str(blueprint.input_modality or "").strip() or None,
            str((blueprint.deployment_target_constraints or {}).get("target_profile_id") or "").strip() or None,
        )
    return None, None, None


def _extract_target_compatibility_summary(
    *,
    source_ref: str,
    allow_network: bool,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for target in list_targets():
        result = check_compatibility(source_ref, target.id, allow_network=allow_network)
        summaries.append(
            {
                "target_profile_id": target.id,
                "compatible": bool(result.get("compatible")),
                "reason_count": len(list(result.get("reasons") or [])),
                "warning_count": len(list(result.get("warnings") or [])),
            }
        )
    return summaries


def _extract_hf_info_payload(model_id: str, *, allow_network: bool) -> dict[str, Any]:
    token = str(model_id or "").strip()
    if not token or not allow_network:
        return {}
    escaped = parse.quote(token, safe="/")
    return _read_json_url(f"https://huggingface.co/api/models/{escaped}") or {}


def _extract_hf_tokenizer_payload(model_id: str, *, allow_network: bool) -> dict[str, Any]:
    token = str(model_id or "").strip()
    if not token or not allow_network:
        return {}
    escaped = parse.quote(token, safe="/")
    payload = _read_json_url(f"https://huggingface.co/{escaped}/raw/main/tokenizer_config.json")
    return payload or {}


def _normalize_from_huggingface(
    *,
    source_ref: str,
    allow_network: bool,
) -> dict[str, Any]:
    introspection = introspect_hf_model(
        model_id=source_ref,
        allow_network=allow_network,
        timeout_seconds=2.5,
    )
    hf_info = _extract_hf_info_payload(source_ref, allow_network=allow_network)
    tokenizer_payload = _extract_hf_tokenizer_payload(source_ref, allow_network=allow_network)
    architecture = str(introspection.get("architecture") or "unknown").strip() or "unknown"
    params_estimate_b = introspection.get("params_estimate_b")
    if isinstance(params_estimate_b, (int, float)):
        params_estimate_b = float(params_estimate_b)
    else:
        params_estimate_b = None
    context_length = introspection.get("context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        context_length = None

    modalities = _extract_modalities(
        source_ref=source_ref,
        config_payload={},
        hf_info_payload=hf_info,
    )
    task_families = _extract_task_families(
        architecture=architecture,
        hf_info_payload=hf_info,
        modalities=modalities,
    )
    memory_profile = dict(introspection.get("memory_profile") or {})
    min_vram = memory_profile.get("estimated_min_vram_gb")
    ideal_vram = memory_profile.get("estimated_ideal_vram_gb")
    hardware = _estimate_hardware_needs(
        params_b=params_estimate_b,
        context_length=context_length,
        modalities=modalities,
        min_vram_hint=float(min_vram) if isinstance(min_vram, (int, float)) else None,
        ideal_vram_hint=float(ideal_vram) if isinstance(ideal_vram, (int, float)) else None,
    )
    tokenizer_name = str(tokenizer_payload.get("tokenizer_class") or "").strip() or None
    chat_template = _extract_chat_template(
        architecture=architecture,
        tokenizer_payload=tokenizer_payload,
        source_ref=source_ref,
    )
    license_value = str(introspection.get("license") or "").strip() or str(
        (hf_info.get("cardData") or {}).get("license") or ""
    ).strip() or None

    payload = {
        "source_type": "huggingface",
        "source_ref": source_ref,
        "display_name": source_ref,
        "model_family": _extract_model_family(source_ref, {}),
        "architecture": architecture,
        "tokenizer": tokenizer_name,
        "chat_template": chat_template,
        "context_length": context_length,
        "parameter_count": _coerce_parameter_count(params_estimate_b),
        "params_estimate_b": params_estimate_b,
        "license": license_value,
        "modalities": modalities,
        "quantization_support": _extract_quantization_support(
            architecture=architecture,
            modalities=modalities,
        ),
        "peft_support": architecture in {"causal_lm", "seq2seq"},
        "full_finetune_support": architecture in {"causal_lm", "seq2seq", "classification"},
        "supported_task_families": task_families,
        "training_mode_support": _extract_training_mode_support(architecture),
        "estimated_hardware_needs": hardware,
        "deployment_target_compatibility": _extract_target_compatibility_summary(
            source_ref=source_ref,
            allow_network=allow_network,
        ),
        "normalized_metadata": {
            "introspection": introspection,
            "hf_info": hf_info,
            "tokenizer_payload": tokenizer_payload,
        },
    }
    payload["model_key"] = build_model_key(payload["source_type"], payload["source_ref"])
    payload["cache_fingerprint"] = _build_cache_fingerprint(payload)
    return payload


def _normalize_from_local_path(
    *,
    source_ref: str,
) -> dict[str, Any]:
    path = Path(source_ref).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Local path not found: {path}")
    config_path = path if path.is_file() else (path / "config.json")
    if config_path.is_file() and config_path.name != "config.json":
        raise ValueError(f"Expected a model directory or config.json path. Got: {path}")
    model_dir = config_path.parent if config_path.is_file() else path

    config_payload = _read_json_file(config_path) if config_path.exists() else {}
    tokenizer_payload = _read_json_file(model_dir / "tokenizer_config.json") or {}
    introspection = introspect_hf_model(
        model_id=str(model_dir),
        allow_network=False,
        timeout_seconds=0.5,
    )
    architecture = str(introspection.get("architecture") or "unknown").strip() or "unknown"
    params_estimate_b = introspection.get("params_estimate_b")
    if isinstance(params_estimate_b, (int, float)):
        params_estimate_b = float(params_estimate_b)
    else:
        params_estimate_b = None
    context_length = introspection.get("context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        context_length = None

    modalities = _extract_modalities(
        source_ref=str(model_dir),
        config_payload=config_payload,
        hf_info_payload={},
    )
    task_families = _extract_task_families(
        architecture=architecture,
        hf_info_payload={},
        modalities=modalities,
    )
    memory_profile = dict(introspection.get("memory_profile") or {})
    min_vram = memory_profile.get("estimated_min_vram_gb")
    ideal_vram = memory_profile.get("estimated_ideal_vram_gb")
    hardware = _estimate_hardware_needs(
        params_b=params_estimate_b,
        context_length=context_length,
        modalities=modalities,
        min_vram_hint=float(min_vram) if isinstance(min_vram, (int, float)) else None,
        ideal_vram_hint=float(ideal_vram) if isinstance(ideal_vram, (int, float)) else None,
    )
    tokenizer_name = str(tokenizer_payload.get("tokenizer_class") or "").strip() or None
    chat_template = _extract_chat_template(
        architecture=architecture,
        tokenizer_payload=tokenizer_payload,
        source_ref=str(model_dir),
    )
    license_value = str(introspection.get("license") or "").strip() or str(
        (config_payload or {}).get("license") or ""
    ).strip() or None

    payload = {
        "source_type": "local_path",
        "source_ref": str(model_dir),
        "display_name": model_dir.name,
        "model_family": _extract_model_family(str(model_dir), config_payload),
        "architecture": architecture,
        "tokenizer": tokenizer_name,
        "chat_template": chat_template,
        "context_length": context_length,
        "parameter_count": _coerce_parameter_count(params_estimate_b),
        "params_estimate_b": params_estimate_b,
        "license": license_value,
        "modalities": modalities,
        "quantization_support": _extract_quantization_support(
            architecture=architecture,
            modalities=modalities,
        ),
        "peft_support": architecture in {"causal_lm", "seq2seq"},
        "full_finetune_support": architecture in {"causal_lm", "seq2seq", "classification"},
        "supported_task_families": task_families,
        "training_mode_support": _extract_training_mode_support(architecture),
        "estimated_hardware_needs": hardware,
        "deployment_target_compatibility": _extract_target_compatibility_summary(
            source_ref=str(model_dir),
            allow_network=False,
        ),
        "normalized_metadata": {
            "introspection": introspection,
            "config_payload": config_payload,
            "tokenizer_payload": tokenizer_payload,
        },
    }
    payload["model_key"] = build_model_key(payload["source_type"], payload["source_ref"])
    payload["cache_fingerprint"] = _build_cache_fingerprint(payload)
    return payload


def _normalize_from_catalog(
    *,
    source_ref: str,
    allow_network: bool,
) -> dict[str, Any]:
    entries = [item for item in list_model_catalog_entries() if isinstance(item, dict)]
    source_token = str(source_ref or "").strip()
    selected = next(
        (
            item
            for item in entries
            if str(item.get("model_id") or "").strip().lower() == source_token.lower()
        ),
        None,
    )
    if selected is None:
        available = sorted({str(item.get("model_id") or "").strip() for item in entries if str(item.get("model_id") or "").strip()})
        raise ValueError(
            "Catalog model not found. Use a known model_id from training catalog. "
            f"Received='{source_ref}'. Known entries={available[:20]}"
        )

    model_id = str(selected.get("model_id") or "").strip()
    introspection = introspect_hf_model(
        model_id=model_id,
        allow_network=allow_network,
        timeout_seconds=2.0,
    )
    architecture = str(introspection.get("architecture") or "unknown").strip() or "unknown"
    params_estimate_b = introspection.get("params_estimate_b")
    if isinstance(params_estimate_b, (int, float)):
        params_estimate_b = float(params_estimate_b)
    else:
        params_estimate_b = float(selected.get("params_b") or 0.0) or None
    context_length = introspection.get("context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        context_length = None

    modalities = _extract_modalities(
        source_ref=model_id,
        config_payload={},
        hf_info_payload={},
    )
    task_families = _extract_task_families(
        architecture=architecture,
        hf_info_payload={},
        modalities=modalities,
    )
    min_vram = selected.get("estimated_min_vram_gb")
    ideal_vram = selected.get("estimated_ideal_vram_gb")
    hardware = _estimate_hardware_needs(
        params_b=params_estimate_b,
        context_length=context_length,
        modalities=modalities,
        min_vram_hint=float(min_vram) if isinstance(min_vram, (int, float)) else None,
        ideal_vram_hint=float(ideal_vram) if isinstance(ideal_vram, (int, float)) else None,
    )

    payload = {
        "source_type": "catalog",
        "source_ref": model_id,
        "display_name": model_id,
        "model_family": str(selected.get("family") or _extract_model_family(model_id, {})).strip().lower() or "unknown",
        "architecture": architecture,
        "tokenizer": None,
        "chat_template": str(selected.get("preferred_chat_template") or "").strip() or _extract_chat_template(
            architecture=architecture,
            tokenizer_payload={},
            source_ref=model_id,
        ),
        "context_length": context_length,
        "parameter_count": _coerce_parameter_count(params_estimate_b),
        "params_estimate_b": params_estimate_b,
        "license": str(introspection.get("license") or "").strip() or None,
        "modalities": modalities,
        "quantization_support": _extract_quantization_support(
            architecture=architecture,
            modalities=modalities,
        ),
        "peft_support": architecture in {"causal_lm", "seq2seq"},
        "full_finetune_support": architecture in {"causal_lm", "seq2seq", "classification"},
        "supported_task_families": task_families,
        "training_mode_support": _extract_training_mode_support(architecture),
        "estimated_hardware_needs": hardware,
        "deployment_target_compatibility": _extract_target_compatibility_summary(
            source_ref=model_id,
            allow_network=allow_network,
        ),
        "normalized_metadata": {
            "catalog_entry": selected,
            "introspection": introspection,
        },
    }
    payload["model_key"] = build_model_key(payload["source_type"], payload["source_ref"])
    payload["cache_fingerprint"] = _build_cache_fingerprint(payload)
    return payload


def normalize_base_model_metadata(
    *,
    source_type: str,
    source_ref: str,
    allow_network: bool,
) -> dict[str, Any]:
    normalized_type = _normalize_source_type(source_type)
    normalized_ref = _normalize_source_ref(normalized_type, source_ref)
    if normalized_type == "huggingface":
        return _normalize_from_huggingface(source_ref=normalized_ref, allow_network=allow_network)
    if normalized_type == "local_path":
        return _normalize_from_local_path(source_ref=normalized_ref)
    if normalized_type == "catalog":
        return _normalize_from_catalog(source_ref=normalized_ref, allow_network=allow_network)
    raise ValueError(f"Unsupported source_type '{source_type}'")


def _serialize_target_compatibility(value: Any) -> list[dict[str, Any]]:
    rows = []
    for item in list(value or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "target_profile_id": str(item.get("target_profile_id") or ""),
                "compatible": bool(item.get("compatible")),
                "reason_count": int(item.get("reason_count") or 0),
                "warning_count": int(item.get("warning_count") or 0),
            }
        )
    return rows


def serialize_base_model_record(record: BaseModelRegistryEntry) -> dict[str, Any]:
    return {
        "id": record.id,
        "model_key": record.model_key,
        "source_type": record.source_type.value,
        "source_ref": record.source_ref,
        "display_name": record.display_name,
        "model_family": record.model_family,
        "architecture": record.architecture,
        "tokenizer": record.tokenizer,
        "chat_template": record.chat_template,
        "context_length": record.context_length,
        "parameter_count": record.parameter_count,
        "params_estimate_b": record.params_estimate_b,
        "license": record.license,
        "modalities": list(record.modalities or []),
        "quantization_support": dict(record.quantization_support or {}),
        "peft_support": bool(record.peft_support),
        "full_finetune_support": bool(record.full_finetune_support),
        "supported_task_families": list(record.supported_task_families or []),
        "training_mode_support": list(record.training_mode_support or []),
        "estimated_hardware_needs": dict(record.estimated_hardware_needs or {}),
        "deployment_target_compatibility": _serialize_target_compatibility(record.deployment_target_compatibility),
        "normalization_contract_version": record.normalization_contract_version,
        "normalized_metadata": dict(record.normalized_metadata or {}),
        "provenance": dict(record.provenance or {}),
        "cache_fingerprint": record.cache_fingerprint,
        "cache_status": record.cache_status,
        "refresh_count": int(record.refresh_count or 0),
        "imported_at": record.imported_at.isoformat() if record.imported_at else None,
        "last_refreshed_at": record.last_refreshed_at.isoformat() if record.last_refreshed_at else None,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }


def _append_provenance_event(
    provenance: dict[str, Any] | None,
    *,
    event_type: str,
    allow_network: bool,
    payload: dict[str, Any],
) -> dict[str, Any]:
    base = dict(provenance or {})
    events = [item for item in list(base.get("events") or []) if isinstance(item, dict)]
    events.append(
        {
            "event_type": event_type,
            "at": _utcnow().isoformat(),
            "allow_network": bool(allow_network),
            "cache_fingerprint": payload.get("cache_fingerprint"),
            "contract_version": BASE_MODEL_REGISTRY_CONTRACT_VERSION,
            "source_type": payload.get("source_type"),
            "source_ref": payload.get("source_ref"),
        }
    )
    base["events"] = events[-30:]
    base["last_event_type"] = event_type
    base["last_event_at"] = _utcnow().isoformat()
    base["engine_version"] = BASE_MODEL_REGISTRY_CONTRACT_VERSION
    return base


async def import_base_model_record(
    db: AsyncSession,
    *,
    source_type: str,
    source_ref: str,
    allow_network: bool,
    overwrite: bool = True,
) -> tuple[BaseModelRegistryEntry, bool]:
    payload = normalize_base_model_metadata(
        source_type=source_type,
        source_ref=source_ref,
        allow_network=allow_network,
    )
    model_key = str(payload.get("model_key") or "").strip()
    if not model_key:
        raise ValueError("Failed to derive model_key from normalized payload.")

    result = await db.execute(
        select(BaseModelRegistryEntry).where(BaseModelRegistryEntry.model_key == model_key)
    )
    row = result.scalar_one_or_none()
    now = _utcnow()
    created = False

    if row is None:
        row = BaseModelRegistryEntry(
            model_key=model_key,
            source_type=BaseModelSourceType(str(payload.get("source_type") or "huggingface")),
            source_ref=str(payload.get("source_ref") or source_ref),
            display_name=str(payload.get("display_name") or source_ref),
            model_family=str(payload.get("model_family") or "unknown"),
            architecture=str(payload.get("architecture") or "unknown"),
            tokenizer=payload.get("tokenizer"),
            chat_template=payload.get("chat_template"),
            context_length=payload.get("context_length"),
            parameter_count=payload.get("parameter_count"),
            params_estimate_b=payload.get("params_estimate_b"),
            license=payload.get("license"),
            modalities=list(payload.get("modalities") or []),
            quantization_support=dict(payload.get("quantization_support") or {}),
            peft_support=bool(payload.get("peft_support", True)),
            full_finetune_support=bool(payload.get("full_finetune_support", True)),
            supported_task_families=list(payload.get("supported_task_families") or []),
            training_mode_support=list(payload.get("training_mode_support") or []),
            estimated_hardware_needs=dict(payload.get("estimated_hardware_needs") or {}),
            deployment_target_compatibility=list(payload.get("deployment_target_compatibility") or []),
            normalization_contract_version=BASE_MODEL_REGISTRY_CONTRACT_VERSION,
            normalized_metadata=dict(payload.get("normalized_metadata") or {}),
            provenance=_append_provenance_event(
                {},
                event_type="import",
                allow_network=allow_network,
                payload=payload,
            ),
            cache_fingerprint=str(payload.get("cache_fingerprint") or ""),
            cache_status="fresh",
            refresh_count=0,
            imported_at=now,
            last_refreshed_at=now,
        )
        db.add(row)
        created = True
    elif overwrite:
        row.source_type = BaseModelSourceType(str(payload.get("source_type") or row.source_type.value))
        row.source_ref = str(payload.get("source_ref") or row.source_ref)
        row.display_name = str(payload.get("display_name") or row.display_name)
        row.model_family = str(payload.get("model_family") or row.model_family)
        row.architecture = str(payload.get("architecture") or row.architecture)
        row.tokenizer = payload.get("tokenizer")
        row.chat_template = payload.get("chat_template")
        row.context_length = payload.get("context_length")
        row.parameter_count = payload.get("parameter_count")
        row.params_estimate_b = payload.get("params_estimate_b")
        row.license = payload.get("license")
        row.modalities = list(payload.get("modalities") or [])
        row.quantization_support = dict(payload.get("quantization_support") or {})
        row.peft_support = bool(payload.get("peft_support", row.peft_support))
        row.full_finetune_support = bool(payload.get("full_finetune_support", row.full_finetune_support))
        row.supported_task_families = list(payload.get("supported_task_families") or [])
        row.training_mode_support = list(payload.get("training_mode_support") or [])
        row.estimated_hardware_needs = dict(payload.get("estimated_hardware_needs") or {})
        row.deployment_target_compatibility = list(payload.get("deployment_target_compatibility") or [])
        row.normalization_contract_version = BASE_MODEL_REGISTRY_CONTRACT_VERSION
        row.normalized_metadata = dict(payload.get("normalized_metadata") or {})
        row.provenance = _append_provenance_event(
            dict(row.provenance or {}),
            event_type="import_overwrite",
            allow_network=allow_network,
            payload=payload,
        )
        row.cache_fingerprint = str(payload.get("cache_fingerprint") or row.cache_fingerprint or "")
        row.cache_status = "fresh"
        row.refresh_count = int(row.refresh_count or 0) + 1
        row.last_refreshed_at = now

    await db.flush()
    await db.refresh(row)
    return row, created


async def refresh_base_model_record(
    db: AsyncSession,
    *,
    model_id: int | None = None,
    model_key: str | None = None,
    allow_network: bool = True,
) -> BaseModelRegistryEntry:
    row = await get_base_model_record(
        db,
        model_id=model_id,
        model_key=model_key,
    )
    if row is None:
        raise ValueError("Base model record not found.")

    payload = normalize_base_model_metadata(
        source_type=row.source_type.value,
        source_ref=row.source_ref,
        allow_network=allow_network,
    )
    now = _utcnow()
    row.display_name = str(payload.get("display_name") or row.display_name)
    row.model_family = str(payload.get("model_family") or row.model_family)
    row.architecture = str(payload.get("architecture") or row.architecture)
    row.tokenizer = payload.get("tokenizer")
    row.chat_template = payload.get("chat_template")
    row.context_length = payload.get("context_length")
    row.parameter_count = payload.get("parameter_count")
    row.params_estimate_b = payload.get("params_estimate_b")
    row.license = payload.get("license")
    row.modalities = list(payload.get("modalities") or [])
    row.quantization_support = dict(payload.get("quantization_support") or {})
    row.peft_support = bool(payload.get("peft_support", row.peft_support))
    row.full_finetune_support = bool(payload.get("full_finetune_support", row.full_finetune_support))
    row.supported_task_families = list(payload.get("supported_task_families") or [])
    row.training_mode_support = list(payload.get("training_mode_support") or [])
    row.estimated_hardware_needs = dict(payload.get("estimated_hardware_needs") or {})
    row.deployment_target_compatibility = list(payload.get("deployment_target_compatibility") or [])
    row.normalization_contract_version = BASE_MODEL_REGISTRY_CONTRACT_VERSION
    row.normalized_metadata = dict(payload.get("normalized_metadata") or {})
    row.provenance = _append_provenance_event(
        dict(row.provenance or {}),
        event_type="refresh",
        allow_network=allow_network,
        payload=payload,
    )
    row.cache_fingerprint = str(payload.get("cache_fingerprint") or row.cache_fingerprint or "")
    row.cache_status = "fresh"
    row.refresh_count = int(row.refresh_count or 0) + 1
    row.last_refreshed_at = now

    await db.flush()
    await db.refresh(row)
    return row


async def get_base_model_record(
    db: AsyncSession,
    *,
    model_id: int | None = None,
    model_key: str | None = None,
    source_ref: str | None = None,
) -> BaseModelRegistryEntry | None:
    if model_id is not None:
        result = await db.execute(
            select(BaseModelRegistryEntry).where(BaseModelRegistryEntry.id == int(model_id))
        )
        row = result.scalar_one_or_none()
        if row is not None:
            return row

    token = str(model_key or "").strip()
    if token:
        result = await db.execute(
            select(BaseModelRegistryEntry).where(BaseModelRegistryEntry.model_key == token)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            return row

    source_token = str(source_ref or "").strip()
    if source_token:
        result = await db.execute(
            select(BaseModelRegistryEntry).where(BaseModelRegistryEntry.source_ref == source_token)
        )
        return result.scalar_one_or_none()
    return None


async def list_base_model_records(
    db: AsyncSession,
    *,
    family: str | None = None,
    license_token: str | None = None,
    hardware_fit: str | None = None,
    min_context_length: int | None = None,
    max_params_b: float | None = None,
    training_mode: str | None = None,
    search: str | None = None,
) -> list[BaseModelRegistryEntry]:
    result = await db.execute(
        select(BaseModelRegistryEntry).order_by(
            BaseModelRegistryEntry.updated_at.desc(),
            BaseModelRegistryEntry.id.desc(),
        )
    )
    rows = list(result.scalars().all())
    out: list[BaseModelRegistryEntry] = []
    family_token = str(family or "").strip().lower()
    license_filter = str(license_token or "").strip().lower()
    hardware_fit_token = str(hardware_fit or "").strip().lower()
    training_mode_filter = str(training_mode or "").strip().lower()
    search_token = str(search or "").strip().lower()

    for row in rows:
        if family_token and family_token not in str(row.model_family or "").strip().lower():
            continue
        if license_filter and license_filter not in str(row.license or "").strip().lower():
            continue
        if hardware_fit_token:
            hardware_profile = dict(row.estimated_hardware_needs or {})
            device_classes = {
                str(item).strip().lower()
                for item in list(hardware_profile.get("recommended_device_classes") or [])
                if str(item).strip()
            }
            if hardware_fit_token not in device_classes:
                continue
        if min_context_length is not None:
            ctx = int(row.context_length or 0)
            if ctx <= 0 or ctx < int(min_context_length):
                continue
        if max_params_b is not None:
            params = float(row.params_estimate_b or 0.0)
            if params <= 0 or params > float(max_params_b):
                continue
        if training_mode_filter:
            modes = {str(item).strip().lower() for item in list(row.training_mode_support or []) if str(item).strip()}
            if training_mode_filter not in modes:
                continue
        if search_token:
            haystack = " ".join(
                [
                    str(row.display_name or ""),
                    str(row.model_key or ""),
                    str(row.source_ref or ""),
                    str(row.model_family or ""),
                    str(row.architecture or ""),
                ]
            ).lower()
            if search_token not in haystack:
                continue
        out.append(row)
    return out


async def _resolve_project_blueprint(
    db: AsyncSession,
    project: Project,
) -> DomainBlueprintRevision | None:
    active_version = int(project.active_domain_blueprint_version or 0)
    if active_version > 0:
        result = await db.execute(
            select(DomainBlueprintRevision).where(
                DomainBlueprintRevision.project_id == project.id,
                DomainBlueprintRevision.version == active_version,
            )
        )
        row = result.scalar_one_or_none()
        if row is not None:
            return row

    result = await db.execute(
        select(DomainBlueprintRevision)
        .where(DomainBlueprintRevision.project_id == project.id)
        .order_by(DomainBlueprintRevision.version.desc(), DomainBlueprintRevision.id.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def evaluate_project_model_compatibility(
    db: AsyncSession,
    *,
    project_id: int,
    model: BaseModelRegistryEntry,
    dataset_adapter_id: str | None = None,
    runtime_id: str | None = None,
    target_profile_id: str | None = None,
    allow_network: bool = False,
    persist_lineage: bool = False,
) -> dict[str, Any]:
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    blueprint = await _resolve_project_blueprint(db, project)
    blueprint_task_family, blueprint_input_modality, blueprint_target_from_constraints = _extract_domain_blueprint_context(
        project,
        blueprint,
    )

    adapter_pref = await resolve_project_dataset_adapter_preference(db, project_id)
    effective_adapter_id = str(dataset_adapter_id or adapter_pref.get("adapter_id") or DEFAULT_ADAPTER_ID).strip() or DEFAULT_ADAPTER_ID
    adapter_catalog = list_data_adapter_catalog()
    adapters = dict(adapter_catalog.get("adapters") or {})
    adapter_contract = dict((adapters.get(effective_adapter_id) or {}).get("contract") or {})
    adapter_training_tasks = {
        str(item).strip().lower()
        for item in list(adapter_contract.get("preferred_training_tasks") or [])
        if str(item).strip()
    }

    effective_target_profile_id = (
        str(target_profile_id or "").strip()
        or str(project.target_profile_id or "").strip()
        or str(blueprint_target_from_constraints or "").strip()
    )
    effective_target_profile_id = effective_target_profile_id or None
    effective_runtime_id, runtime_modalities = _resolve_runtime_modalities(runtime_id)
    runtime_modalities_set = {str(item).strip().lower() for item in runtime_modalities if str(item).strip()}

    model_task_families = {
        _TASK_FAMILY_ALIASES.get(str(item).strip().lower(), str(item).strip().lower())
        for item in list(model.supported_task_families or [])
        if str(item).strip()
    }
    model_modalities = {
        _MODALITY_ALIASES.get(str(item).strip().lower(), str(item).strip().lower())
        for item in list(model.modalities or [])
        if str(item).strip()
    } or {"text"}

    reasons: list[CompatibilityReason] = []
    reason_codes: list[str] = []

    def push(reason: CompatibilityReason) -> None:
        reasons.append(reason)
        reason_codes.append(reason.code)

    normalized_task = _TASK_FAMILY_ALIASES.get(str(blueprint_task_family or "").strip().lower())
    if normalized_task:
        if normalized_task in model_task_families:
            push(
                _build_reason(
                    "TASK_FAMILY_SUPPORTED",
                    severity="pass",
                    message=f"Model supports task family '{normalized_task}'.",
                    evidence={"task_family": normalized_task},
                )
            )
        else:
            push(
                _build_reason(
                    "TASK_FAMILY_UNSUPPORTED",
                    severity="blocker",
                    message=f"Model task families {sorted(model_task_families)} do not include required '{normalized_task}'.",
                    unblock_actions=[
                        "Choose a model whose supported_task_families includes the project task family.",
                        "Refine Domain Blueprint task_family if it was inferred incorrectly.",
                    ],
                    evidence={
                        "required_task_family": normalized_task,
                        "model_supported_task_families": sorted(model_task_families),
                    },
                )
            )
    else:
        push(
            _build_reason(
                "TASK_FAMILY_CONTEXT_MISSING",
                severity="warning",
                message="Project has no active Domain Blueprint task family; task fit is partially inferred.",
                unblock_actions=[
                    "Create/apply a Domain Blueprint revision with an explicit task family.",
                ],
            )
        )

    normalized_input_modality = _MODALITY_ALIASES.get(str(blueprint_input_modality or "text").strip().lower(), "text")
    if normalized_input_modality in model_modalities or "multimodal" in model_modalities:
        push(
            _build_reason(
                "INPUT_MODALITY_SUPPORTED",
                severity="pass",
                message=f"Model supports input modality '{normalized_input_modality}'.",
                evidence={
                    "required_input_modality": normalized_input_modality,
                    "model_modalities": sorted(model_modalities),
                },
            )
        )
    else:
        push(
            _build_reason(
                "INPUT_MODALITY_UNSUPPORTED",
                severity="blocker",
                message=f"Model modalities {sorted(model_modalities)} do not support required '{normalized_input_modality}'.",
                unblock_actions=[
                    "Select a model that supports the required modality.",
                    "Adjust Domain Blueprint input modality to match available model capabilities.",
                ],
                evidence={
                    "required_input_modality": normalized_input_modality,
                    "model_modalities": sorted(model_modalities),
                },
            )
        )

    model_training_tasks: set[str] = set()
    for family in model_task_families:
        model_training_tasks.update(_TRAINING_TASK_BY_FAMILY.get(family, []))
    if adapter_training_tasks:
        if model_training_tasks.intersection(adapter_training_tasks):
            push(
                _build_reason(
                    "ADAPTER_TASK_COMPATIBLE",
                    severity="pass",
                    message=f"Dataset adapter '{effective_adapter_id}' is compatible with model task support.",
                    evidence={
                        "adapter_id": effective_adapter_id,
                        "adapter_training_tasks": sorted(adapter_training_tasks),
                        "model_training_tasks": sorted(model_training_tasks),
                    },
                )
            )
        else:
            push(
                _build_reason(
                    "ADAPTER_TASK_MISMATCH",
                    severity="blocker",
                    message=(
                        f"Dataset adapter '{effective_adapter_id}' expects tasks {sorted(adapter_training_tasks)} "
                        f"but model supports {sorted(model_training_tasks)}."
                    ),
                    unblock_actions=[
                        "Switch to a compatible dataset adapter/task profile.",
                        "Choose a model aligned with the dataset adapter contract.",
                    ],
                    evidence={
                        "adapter_id": effective_adapter_id,
                        "adapter_training_tasks": sorted(adapter_training_tasks),
                        "model_training_tasks": sorted(model_training_tasks),
                    },
                )
            )

    required_runtime_modalities = {
        _MODALITY_TO_RUNTIME_MODALITIES.get(modality, "text")
        for modality in model_modalities
    }
    if runtime_modalities_set:
        missing_runtime_modalities = {
            item for item in required_runtime_modalities
            if item not in runtime_modalities_set and "multimodal" not in runtime_modalities_set
        }
        if missing_runtime_modalities:
            push(
                _build_reason(
                    "RUNTIME_MODALITY_UNSUPPORTED",
                    severity="blocker",
                    message=(
                        f"Runtime '{effective_runtime_id}' does not support required modalities "
                        f"{sorted(missing_runtime_modalities)}."
                    ),
                    unblock_actions=[
                        "Select a runtime that supports the model modalities.",
                        "Switch to a text-only model/runtime combination for first iteration.",
                    ],
                    evidence={
                        "runtime_id": effective_runtime_id,
                        "runtime_modalities": sorted(runtime_modalities_set),
                        "required_runtime_modalities": sorted(required_runtime_modalities),
                    },
                )
            )
        else:
            push(
                _build_reason(
                    "RUNTIME_MODALITY_SUPPORTED",
                    severity="pass",
                    message=f"Runtime '{effective_runtime_id}' supports required model modalities.",
                    evidence={
                        "runtime_id": effective_runtime_id,
                        "runtime_modalities": sorted(runtime_modalities_set),
                    },
                )
            )
    else:
        push(
            _build_reason(
                "RUNTIME_CONTEXT_MISSING",
                severity="warning",
                message="Runtime modality contract was not found; runtime compatibility is assumed.",
                unblock_actions=["Set an explicit training runtime before launch."],
                evidence={"runtime_id": effective_runtime_id},
            )
        )

    if effective_target_profile_id:
        target_result = check_compatibility(
            model.source_ref,
            effective_target_profile_id,
            allow_network=allow_network,
        )
        if bool(target_result.get("compatible")):
            push(
                _build_reason(
                    "TARGET_PROFILE_COMPATIBLE",
                    severity="pass",
                    message=f"Model is compatible with target profile '{effective_target_profile_id}'.",
                    evidence={"target_profile_id": effective_target_profile_id},
                )
            )
            for warning in [str(item).strip() for item in list(target_result.get("warnings") or []) if str(item).strip()]:
                push(
                    _build_reason(
                        "TARGET_PROFILE_WARNING",
                        severity="warning",
                        message=warning,
                        unblock_actions=[
                            "Consider a smaller model or stronger quantization for tighter target fit.",
                        ],
                        evidence={"target_profile_id": effective_target_profile_id},
                    )
                )
        else:
            reasons_text = [str(item).strip() for item in list(target_result.get("reasons") or []) if str(item).strip()]
            push(
                _build_reason(
                    "TARGET_PROFILE_INCOMPATIBLE",
                    severity="blocker",
                    message=reasons_text[0] if reasons_text else f"Model is not compatible with target profile '{effective_target_profile_id}'.",
                    unblock_actions=[
                        "Choose a target profile with sufficient VRAM/capability.",
                        "Select a smaller base model.",
                    ],
                    evidence={
                        "target_profile_id": effective_target_profile_id,
                        "target_reasons": reasons_text,
                    },
                )
            )
    else:
        push(
            _build_reason(
                "TARGET_PROFILE_UNSET",
                severity="warning",
                message="No deployment target profile is set; deployment compatibility cannot be fully scored.",
                unblock_actions=[
                    "Set project target_profile_id or blueprint deployment target constraints.",
                ],
            )
        )

    if not str(model.tokenizer or "").strip():
        push(
            _build_reason(
                "TOKENIZER_METADATA_MISSING",
                severity="warning",
                message="Tokenizer metadata is missing; tokenization behavior may differ at train/serve time.",
                unblock_actions=[
                    "Refresh model metadata or specify tokenizer explicitly in training config.",
                ],
            )
        )
    else:
        push(
            _build_reason(
                "TOKENIZER_METADATA_PRESENT",
                severity="pass",
                message="Tokenizer metadata is available.",
            )
        )

    if model.architecture == "causal_lm" and not str(model.chat_template or "").strip():
        push(
            _build_reason(
                "CHAT_TEMPLATE_MISSING",
                severity="warning",
                message="Chat template is missing for a causal LM; instruction formatting may be unstable.",
                unblock_actions=[
                    "Set an explicit chat template in training config before launch.",
                ],
            )
        )
    else:
        push(
            _build_reason(
                "CHAT_TEMPLATE_OK",
                severity="pass",
                message="Chat template signal is present or not required for this architecture.",
            )
        )

    if bool(project.beginner_mode) and not bool(model.peft_support):
        push(
            _build_reason(
                "PEFT_UNSUPPORTED_FOR_BEGINNER",
                severity="warning",
                message="Model does not advertise PEFT support; beginner flow may require full fine-tune resources.",
                unblock_actions=[
                    "Prefer a PEFT-compatible model for first project iterations.",
                ],
            )
        )
    else:
        push(
            _build_reason(
                "PEFT_SUPPORT_OK",
                severity="pass",
                message="Model supports PEFT or project mode does not require it.",
            )
        )

    required_context = _infer_required_context(blueprint_task_family)
    model_context = int(model.context_length or 0)
    if model_context > 0 and model_context < required_context:
        push(
            _build_reason(
                "CONTEXT_LENGTH_LOW",
                severity="warning",
                message=f"Model context length {model_context} is lower than recommended {required_context} for this task family.",
                unblock_actions=[
                    "Use shorter chunking windows or select a longer-context model.",
                ],
                evidence={
                    "model_context_length": model_context,
                    "recommended_context_length": required_context,
                },
            )
        )
    else:
        push(
            _build_reason(
                "CONTEXT_LENGTH_OK",
                severity="pass",
                message="Context length is sufficient for the inferred task requirements.",
                evidence={
                    "model_context_length": model_context or None,
                    "recommended_context_length": required_context,
                },
            )
        )

    blocker_count = sum(1 for item in reasons if item.severity == "blocker")
    warning_count = sum(1 for item in reasons if item.severity == "warning")
    pass_count = sum(1 for item in reasons if item.severity == "pass")
    raw_score = 1.0 + (0.025 * pass_count) - (0.08 * warning_count) - (0.26 * blocker_count)
    compatibility_score = round(max(0.0, min(1.0, raw_score)), 4)
    compatible = blocker_count == 0

    why_recommended = [item for item in reasons if item.severity == "pass"][:6]
    why_risky = [item for item in reasons if item.severity in {"warning", "blocker"}][:8]
    recommended_next_actions = _dedupe_preserve_order(
        [action for reason in why_risky for action in reason.unblock_actions]
    )
    unresolved_questions: list[str] = []
    if blueprint is None:
        unresolved_questions.append("Project has no saved Domain Blueprint revision.")
    if not effective_target_profile_id:
        unresolved_questions.append("Target profile is not set.")
    if not runtime_modalities_set:
        unresolved_questions.append("Runtime modality contract was not resolved.")

    payload = {
        "project_id": project_id,
        "model_id": model.id,
        "model_key": model.model_key,
        "compatibility_score": compatibility_score,
        "compatible": compatible,
        "reason_codes": [item.code for item in reasons],
        "reasons": [item.model_dump() for item in reasons],
        "why_recommended": [item.model_dump() for item in why_recommended],
        "why_risky": [item.model_dump() for item in why_risky],
        "unresolved_questions": unresolved_questions,
        "recommended_next_actions": recommended_next_actions,
        "context": {
            "task_family": blueprint_task_family,
            "input_modality": blueprint_input_modality,
            "dataset_adapter_id": effective_adapter_id,
            "runtime_id": effective_runtime_id,
            "runtime_modalities": sorted(runtime_modalities_set),
            "target_profile_id": effective_target_profile_id,
            "compatibility_engine_version": MODEL_COMPATIBILITY_ENGINE_VERSION,
        },
        "generated_at": _utcnow().isoformat(),
    }

    if persist_lineage:
        try:
            await publish_artifact(
                db=db,
                project_id=project_id,
                artifact_key=_safe_artifact_key_for_model(model.model_key),
                schema_ref="slm.base_model_compatibility/v1",
                producer_stage="training",
                metadata={
                    "model_key": model.model_key,
                    "model_id": model.id,
                    "compatibility_score": compatibility_score,
                    "compatible": compatible,
                    "reason_codes": payload.get("reason_codes"),
                    "target_profile_id": effective_target_profile_id,
                    "runtime_id": effective_runtime_id,
                    "dataset_adapter_id": effective_adapter_id,
                },
            )
        except Exception:
            # Compatibility evaluation should still succeed even when lineage publishing fails.
            pass

    return payload


async def recommend_models_for_project(
    db: AsyncSession,
    *,
    project_id: int,
    limit: int = 10,
    include_incompatible: bool = False,
    family: str | None = None,
    license_token: str | None = None,
    hardware_fit: str | None = None,
    min_context_length: int | None = None,
    max_params_b: float | None = None,
    training_mode: str | None = None,
    search: str | None = None,
    target_profile_id: str | None = None,
    runtime_id: str | None = None,
    dataset_adapter_id: str | None = None,
    allow_network: bool = False,
) -> list[dict[str, Any]]:
    rows = await list_base_model_records(
        db,
        family=family,
        license_token=license_token,
        hardware_fit=hardware_fit,
        min_context_length=min_context_length,
        max_params_b=max_params_b,
        training_mode=training_mode,
        search=search,
    )
    scored: list[dict[str, Any]] = []
    for row in rows:
        report = await evaluate_project_model_compatibility(
            db,
            project_id=project_id,
            model=row,
            dataset_adapter_id=dataset_adapter_id,
            runtime_id=runtime_id,
            target_profile_id=target_profile_id,
            allow_network=allow_network,
            persist_lineage=False,
        )
        report["model"] = serialize_base_model_record(row)
        scored.append(report)

    scored.sort(
        key=lambda item: (
            bool(item.get("compatible")),
            float(item.get("compatibility_score") or 0.0),
            float(((item.get("model") or {}).get("params_estimate_b") or 0.0)),
        ),
        reverse=True,
    )
    if not include_incompatible:
        scored = [item for item in scored if bool(item.get("compatible"))]
    max_items = max(1, min(int(limit or 10), 100))
    return scored[:max_items]

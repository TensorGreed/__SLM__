"""Heuristic base-model recommendation service for the training wizard."""

from __future__ import annotations

import copy
import importlib
import inspect
import threading
from typing import Any

from app.config import settings
from app.services.capability_contract_service import SUPPORTED_TRAINING_TASK_TYPES
from app.services.data_adapter_service import normalize_task_profile, task_profile_training_tasks
from app.services.model_introspection_service import introspect_hf_model

SUPPORTED_TARGET_DEVICES: tuple[str, ...] = ("mobile", "laptop", "server")
SUPPORTED_PRIMARY_LANGUAGES: tuple[str, ...] = ("english", "multilingual", "coding")

MODEL_CATALOG_REGISTRY_VERSION = "model_catalog.dynamic_registry/v1"

_TARGET_DEVICE_ALIASES: dict[str, str] = {
    "phone": "mobile",
    "tablet": "mobile",
    "mobile": "mobile",
    "mobile_cpu": "mobile",
    "laptop": "laptop",
    "desktop": "laptop",
    "edge_gpu": "laptop",
    "browser_webgpu": "mobile",
    "workstation": "server",
    "cloud": "server",
    "server": "server",
    "vllm_server": "server",
}

_PRIMARY_LANGUAGE_ALIASES: dict[str, str] = {
    "en": "english",
    "english": "english",
    "multi": "multilingual",
    "multilingual": "multilingual",
    "code": "coding",
    "coding": "coding",
}

_DEFAULT_DEVICE_VRAM_BUDGET_GB: dict[str, float] = {
    "mobile": 6.0,
    "laptop": 16.0,
    "server": 80.0,
}

_SUPPORTED_TRAINING_TASK_TYPES = set(SUPPORTED_TRAINING_TASK_TYPES)

_BUILTIN_MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "family": "llama",
        "params_b": 1.0,
        "estimated_min_vram_gb": 4.0,
        "estimated_ideal_vram_gb": 6.0,
        "preferred_chat_template": "llama3",
        "supported_languages": ("english", "multilingual"),
        "strengths": (
            "Very low VRAM footprint for quick iteration",
            "Good baseline for instruction tuning",
        ),
        "caveats": (
            "Smaller context understanding than 7B+ models",
            "May underperform on complex coding tasks",
        ),
    },
    {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "qwen",
        "params_b": 1.5,
        "estimated_min_vram_gb": 4.0,
        "estimated_ideal_vram_gb": 6.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english", "multilingual", "coding"),
        "strengths": (
            "Strong multilingual coverage at small size",
            "Good coding quality per VRAM",
        ),
        "caveats": (
            "May need careful prompt formatting for best output",
        ),
    },
    {
        "model_id": "google/gemma-2-2b-it",
        "family": "gemma",
        "params_b": 2.0,
        "estimated_min_vram_gb": 5.0,
        "estimated_ideal_vram_gb": 7.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english",),
        "strengths": (
            "Stable instruction quality for compact model size",
            "Easy to run on laptop GPUs",
        ),
        "caveats": (
            "English-first coverage",
        ),
    },
    {
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "family": "qwen",
        "params_b": 3.0,
        "estimated_min_vram_gb": 7.0,
        "estimated_ideal_vram_gb": 10.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english", "coding"),
        "strengths": (
            "Best coding-focused option for low-to-mid VRAM",
            "Good balance between quality and speed",
        ),
        "caveats": (
            "Narrower multilingual coverage than general Qwen instruct variants",
        ),
    },
    {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "llama",
        "params_b": 3.0,
        "estimated_min_vram_gb": 7.0,
        "estimated_ideal_vram_gb": 10.0,
        "preferred_chat_template": "llama3",
        "supported_languages": ("english", "multilingual"),
        "strengths": (
            "Strong instruction quality for small model footprint",
            "Reliable baseline for many text QA workloads",
        ),
        "caveats": (
            "Coding quality is behind coding-specialized variants",
        ),
    },
    {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "family": "phi",
        "params_b": 3.8,
        "estimated_min_vram_gb": 8.0,
        "estimated_ideal_vram_gb": 12.0,
        "preferred_chat_template": "phi3",
        "supported_languages": ("english", "coding"),
        "strengths": (
            "Strong reasoning density for model size",
            "Solid coding and function-style output",
        ),
        "caveats": (
            "Context window may require tuning for long documents",
        ),
    },
    {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "qwen",
        "params_b": 7.0,
        "estimated_min_vram_gb": 14.0,
        "estimated_ideal_vram_gb": 20.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english", "multilingual", "coding"),
        "strengths": (
            "High quality across multilingual and coding tasks",
            "Strong general-purpose instruct performance",
        ),
        "caveats": (
            "Needs mid-to-high VRAM for comfortable training",
        ),
    },
    {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "mistral",
        "params_b": 7.0,
        "estimated_min_vram_gb": 14.0,
        "estimated_ideal_vram_gb": 20.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english", "coding"),
        "strengths": (
            "Strong long-form generation quality",
            "Mature ecosystem and tuning recipes",
        ),
        "caveats": (
            "Multilingual behavior is weaker than Qwen at same size",
        ),
    },
]

_MODEL_CATALOG_LOCK = threading.RLock()
_MODEL_CATALOG_REGISTRY: dict[str, dict[str, Any]] = {}
_MODEL_CATALOG_ORDER: list[str] = []
_MODEL_CATALOG_INITIALIZED = False
_MODEL_CATALOG_PLUGIN_MODULES_LOADED: set[str] = set()
_MODEL_CATALOG_PLUGIN_LOAD_ERRORS: dict[str, str] = {}
_MODEL_CATALOG_REGISTRATION_SOURCE_CONTEXT: str | None = None


def _normalize_source_module(value: str | None) -> str:
    token = str(value or "").strip()
    return token or "custom"


def _normalize_catalog_version(value: str | None, *, is_builtin: bool) -> str:
    token = str(value or "").strip()
    if token:
        return token
    return "builtin-v1" if is_builtin else "plugin-v1"


def _coerce_positive_float(value: Any, *, precision: int = 2) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return round(parsed, precision)


def _normalize_language_collection(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = []

    items: list[str] = []
    for raw in raw_items:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        normalized = _PRIMARY_LANGUAGE_ALIASES.get(token, token)
        if normalized not in SUPPORTED_PRIMARY_LANGUAGES:
            continue
        if normalized not in items:
            items.append(normalized)
    if not items:
        items = ["english"]
    return tuple(items)


def _normalize_text_collection(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = []
    items = [str(item).strip() for item in raw_items if str(item).strip()]
    return tuple(items)


def _coerce_model_entry_payload(
    payload: dict[str, Any],
    *,
    source_module: str | None,
    catalog_version: str | None,
    is_builtin: bool,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Model catalog payload must be a mapping.")

    data = dict(payload)
    model_id = str(data.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("model_id is required for model catalog entries.")

    params_b = _coerce_positive_float(data.get("params_b"), precision=2)
    if params_b is None:
        raise ValueError(f"Model catalog entry '{model_id}' must define params_b > 0.")

    min_vram = _coerce_positive_float(data.get("estimated_min_vram_gb"), precision=2)
    if min_vram is None:
        raise ValueError(
            f"Model catalog entry '{model_id}' must define estimated_min_vram_gb > 0."
        )

    ideal_vram = _coerce_positive_float(data.get("estimated_ideal_vram_gb"), precision=2)
    if ideal_vram is None:
        ideal_vram = round(max(min_vram + 1.0, min_vram * 1.25), 2)

    normalized_is_builtin = bool(data.get("is_builtin", is_builtin))
    normalized_source = _normalize_source_module(
        data.get("catalog_source")
        or source_module
        or _MODEL_CATALOG_REGISTRATION_SOURCE_CONTEXT
        or ("builtin" if normalized_is_builtin else "custom")
    )
    normalized_version = _normalize_catalog_version(
        data.get("catalog_version") or catalog_version,
        is_builtin=normalized_is_builtin,
    )

    normalized: dict[str, Any] = {
        "model_id": model_id,
        "family": str(data.get("family") or "unknown").strip().lower() or "unknown",
        "params_b": params_b,
        "estimated_min_vram_gb": min_vram,
        "estimated_ideal_vram_gb": max(min_vram, ideal_vram),
        "preferred_chat_template": str(data.get("preferred_chat_template") or "llama3").strip() or "llama3",
        "supported_languages": _normalize_language_collection(data.get("supported_languages")),
        "strengths": _normalize_text_collection(data.get("strengths")),
        "caveats": _normalize_text_collection(data.get("caveats")),
        "catalog_source": normalized_source,
        "catalog_version": normalized_version,
        "is_builtin": normalized_is_builtin,
    }

    for key, value in data.items():
        if key not in normalized:
            normalized[key] = value
    return normalized


def _register_model_catalog_entry(entry: dict[str, Any]) -> None:
    model_id = str(entry.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("model_id is required for model catalog entries.")

    if model_id not in _MODEL_CATALOG_ORDER:
        _MODEL_CATALOG_ORDER.append(model_id)
    _MODEL_CATALOG_REGISTRY[model_id] = entry


def register_model_catalog_entry(
    payload: dict[str, Any],
    *,
    source_module: str | None = None,
    catalog_version: str | None = None,
    is_builtin: bool = False,
) -> None:
    """Public SDK entry-point for registering a model catalog entry."""
    entry = _coerce_model_entry_payload(
        payload,
        source_module=source_module,
        catalog_version=catalog_version,
        is_builtin=is_builtin,
    )
    with _MODEL_CATALOG_LOCK:
        _register_model_catalog_entry(entry)


def register_model_catalog_entries(
    payloads: list[dict[str, Any]],
    *,
    source_module: str | None = None,
    catalog_version: str | None = None,
    is_builtin: bool = False,
) -> int:
    count = 0
    for payload in list(payloads or []):
        register_model_catalog_entry(
            payload,
            source_module=source_module,
            catalog_version=catalog_version,
            is_builtin=is_builtin,
        )
        count += 1
    return count


def _iter_model_catalog_payload(raw_payload: Any) -> list[dict[str, Any]]:
    if isinstance(raw_payload, dict):
        rows: list[dict[str, Any]] = []
        for key, value in raw_payload.items():
            if not isinstance(value, dict):
                continue
            if not str(value.get("model_id") or "").strip():
                value = {**value, "model_id": str(key)}
            rows.append(value)
        return rows
    if isinstance(raw_payload, (list, tuple, set)):
        return [item for item in list(raw_payload) if isinstance(item, dict)]
    return []


def _extract_module_model_catalog_entries(module: Any) -> list[dict[str, Any]]:
    if hasattr(module, "get_model_catalog_entries") and callable(module.get_model_catalog_entries):
        payload = module.get_model_catalog_entries()
        return _iter_model_catalog_payload(payload)
    return _iter_model_catalog_payload(getattr(module, "MODEL_CATALOG_ENTRIES", []))


def _call_register_model_catalog_fn(module: Any, module_name: str) -> int:
    register_fn = getattr(module, "register_model_catalog_entries", None)
    if register_fn is None:
        return 0
    if not callable(register_fn):
        raise ValueError(
            f"Model catalog plugin module '{module_name}' has non-callable register_model_catalog_entries."
        )

    count = 0

    def register(
        entry_payload: dict[str, Any],
        *,
        catalog_version: str | None = None,
    ) -> None:
        nonlocal count
        register_model_catalog_entry(
            entry_payload,
            source_module=module_name,
            catalog_version=catalog_version,
            is_builtin=False,
        )
        count += 1

    signature = inspect.signature(register_fn)
    if len(signature.parameters) == 0:
        register_fn()
    elif len(signature.parameters) == 1:
        register_fn(register)
    else:
        register_fn(register, {"settings": settings})
    return count


def load_model_catalog_plugins(
    module_paths: list[str],
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Load model catalog plugins and register entries into the dynamic registry."""
    requested_modules = [str(item).strip() for item in list(module_paths or []) if str(item).strip()]
    loaded_modules: list[str] = []
    skipped_modules: list[str] = []
    errors: dict[str, str] = {}

    global _MODEL_CATALOG_REGISTRATION_SOURCE_CONTEXT
    with _MODEL_CATALOG_LOCK:
        for module_name in requested_modules:
            if module_name in _MODEL_CATALOG_PLUGIN_MODULES_LOADED and not force_reload:
                skipped_modules.append(module_name)
                continue

            try:
                module = importlib.import_module(module_name)
                if force_reload:
                    module = importlib.reload(module)

                registered_count = 0
                prior_context = _MODEL_CATALOG_REGISTRATION_SOURCE_CONTEXT
                _MODEL_CATALOG_REGISTRATION_SOURCE_CONTEXT = module_name
                try:
                    registered_count += _call_register_model_catalog_fn(module, module_name)
                    for payload in _extract_module_model_catalog_entries(module):
                        register_model_catalog_entry(
                            payload,
                            source_module=module_name,
                            catalog_version=None,
                            is_builtin=False,
                        )
                        registered_count += 1
                finally:
                    _MODEL_CATALOG_REGISTRATION_SOURCE_CONTEXT = prior_context

                if registered_count == 0:
                    raise ValueError(
                        "No model catalog entries registered. Provide register_model_catalog_entries(...) or MODEL_CATALOG_ENTRIES/get_model_catalog_entries()."
                    )

                _MODEL_CATALOG_PLUGIN_MODULES_LOADED.add(module_name)
                _MODEL_CATALOG_PLUGIN_LOAD_ERRORS.pop(module_name, None)
                loaded_modules.append(module_name)
            except Exception as exc:  # noqa: PERF203
                message = str(exc)
                _MODEL_CATALOG_PLUGIN_LOAD_ERRORS[module_name] = message
                errors[module_name] = message

    return {
        "requested_modules": requested_modules,
        "loaded_modules": sorted(set(loaded_modules)),
        "skipped_modules": sorted(set(skipped_modules)),
        "errors": errors,
    }


def load_model_catalog_plugins_from_settings(*, force_reload: bool = False) -> dict[str, Any]:
    modules = [str(item).strip() for item in list(settings.MODEL_CATALOG_PLUGIN_MODULES or []) if str(item).strip()]
    if not modules:
        return {
            "requested_modules": [],
            "loaded_modules": [],
            "skipped_modules": [],
            "errors": {},
        }
    return load_model_catalog_plugins(modules, force_reload=force_reload)


def clear_model_catalog_plugins() -> None:
    global _MODEL_CATALOG_INITIALIZED
    with _MODEL_CATALOG_LOCK:
        _MODEL_CATALOG_REGISTRY.clear()
        _MODEL_CATALOG_ORDER.clear()
        _MODEL_CATALOG_PLUGIN_MODULES_LOADED.clear()
        _MODEL_CATALOG_PLUGIN_LOAD_ERRORS.clear()
        _MODEL_CATALOG_INITIALIZED = False


def _model_catalog_metadata(*, entry_count: int | None = None) -> dict[str, Any]:
    count = int(entry_count) if entry_count is not None else len(_MODEL_CATALOG_ORDER)
    return {
        "catalog_version": MODEL_CATALOG_REGISTRY_VERSION,
        "entry_count": count,
        "loaded_plugin_modules": sorted(_MODEL_CATALOG_PLUGIN_MODULES_LOADED),
        "plugin_load_errors": copy.deepcopy(_MODEL_CATALOG_PLUGIN_LOAD_ERRORS),
        "has_plugin_entries": any(
            not bool(_MODEL_CATALOG_REGISTRY.get(model_id, {}).get("is_builtin"))
            for model_id in _MODEL_CATALOG_ORDER
        ),
    }


def model_catalog_plugin_status() -> dict[str, Any]:
    _ensure_model_catalog_loaded()
    with _MODEL_CATALOG_LOCK:
        return {
            "requested_modules": [
                str(item).strip()
                for item in list(settings.MODEL_CATALOG_PLUGIN_MODULES or [])
                if str(item).strip()
            ],
            "loaded_modules": sorted(_MODEL_CATALOG_PLUGIN_MODULES_LOADED),
            "failed_modules": copy.deepcopy(_MODEL_CATALOG_PLUGIN_LOAD_ERRORS),
            "registered_entry_count": len(_MODEL_CATALOG_ORDER),
        }


def _register_builtin_model_catalog_entries() -> None:
    for payload in _BUILTIN_MODEL_CATALOG:
        entry = _coerce_model_entry_payload(
            payload,
            source_module="builtin",
            catalog_version="builtin-v1",
            is_builtin=True,
        )
        _register_model_catalog_entry(entry)


def _ensure_model_catalog_loaded() -> None:
    global _MODEL_CATALOG_INITIALIZED
    if _MODEL_CATALOG_INITIALIZED:
        return

    with _MODEL_CATALOG_LOCK:
        if _MODEL_CATALOG_INITIALIZED:
            return
        _MODEL_CATALOG_REGISTRY.clear()
        _MODEL_CATALOG_ORDER.clear()
        _MODEL_CATALOG_PLUGIN_MODULES_LOADED.clear()
        _MODEL_CATALOG_PLUGIN_LOAD_ERRORS.clear()
        _register_builtin_model_catalog_entries()
        load_model_catalog_plugins_from_settings(force_reload=False)
        _MODEL_CATALOG_INITIALIZED = True


def list_model_catalog_entries() -> list[dict[str, Any]]:
    _ensure_model_catalog_loaded()
    with _MODEL_CATALOG_LOCK:
        return [
            copy.deepcopy(_MODEL_CATALOG_REGISTRY[model_id])
            for model_id in _MODEL_CATALOG_ORDER
            if model_id in _MODEL_CATALOG_REGISTRY
        ]


def list_model_catalog() -> dict[str, Any]:
    entries = list_model_catalog_entries()
    with _MODEL_CATALOG_LOCK:
        return {
            **_model_catalog_metadata(entry_count=len(entries)),
            "models": entries,
        }


def _normalize_target_device(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "laptop"
    return _TARGET_DEVICE_ALIASES.get(token, "laptop")


def _normalize_primary_language(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "english"
    return _PRIMARY_LANGUAGE_ALIASES.get(token, "english")


def _coerce_vram(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return round(parsed, 2)


def _coerce_top_k(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 3
    return max(1, min(5, parsed))


def _resolve_training_task(task_profile: str | None) -> tuple[str | None, str]:
    normalized_profile = normalize_task_profile(task_profile, default="auto")
    if normalized_profile == "auto":
        normalized_profile = None

    candidate_tasks = task_profile_training_tasks(normalized_profile or "instruction_sft")
    for task in candidate_tasks:
        if task in _SUPPORTED_TRAINING_TASK_TYPES:
            return normalized_profile, task
    return normalized_profile, "causal_lm"


def _suggest_batch_size(*, available_vram_gb: float | None, min_vram_gb: float, params_b: float) -> int:
    if available_vram_gb is not None:
        if available_vram_gb < min_vram_gb:
            return 1
        headroom = available_vram_gb - min_vram_gb
        if headroom >= 10:
            return 8
        if headroom >= 5:
            return 4
        if headroom >= 2:
            return 2
        return 1
    if params_b <= 2:
        return 8
    if params_b <= 4:
        return 4
    return 2


def _suggest_max_seq_length(*, available_vram_gb: float | None, params_b: float) -> int:
    if available_vram_gb is not None and available_vram_gb <= 8:
        return 1024
    if params_b <= 2:
        return 2048
    if params_b <= 4:
        return 2048
    return 1536


def _score_model(
    *,
    model: dict[str, Any],
    target_device: str,
    primary_language: str,
    available_vram_gb: float | None,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    supported_languages = {
        str(item).strip().lower()
        for item in list(model.get("supported_languages") or [])
        if str(item).strip()
    }
    if primary_language in supported_languages:
        score += 2.5
        reasons.append(f"matches primary language goal ({primary_language})")
    elif primary_language == "english":
        score += 1.0
        reasons.append("good default for English tasks")
    else:
        score -= 0.5
        reasons.append("language fit is weaker than top alternatives")

    params_b = float(model.get("params_b") or 0.0)
    if target_device == "mobile":
        if params_b <= 2.0:
            score += 2.0
            reasons.append("parameter size is mobile-friendly")
        elif params_b <= 4.0:
            score += 0.5
            reasons.append("parameter size may work on higher-end mobile hardware")
        else:
            score -= 2.0
            reasons.append("parameter size is heavy for mobile targets")
    elif target_device == "laptop":
        if params_b <= 4.0:
            score += 1.5
            reasons.append("good balance for laptop fine-tuning")
        elif params_b <= 7.0:
            score += 0.5
            reasons.append("workable on laptop with tuned batch/sequence settings")
        else:
            score -= 1.0
            reasons.append("likely expensive for laptop workflows")
    else:
        if params_b >= 3.0:
            score += 1.0
            reasons.append("uses server headroom for better model capacity")
        else:
            score += 0.25
            reasons.append("easy to scale for fast server iteration")

    min_vram_gb = float(model.get("estimated_min_vram_gb") or 0.0)
    if available_vram_gb is not None:
        if available_vram_gb >= min_vram_gb:
            score += 2.0
            reasons.append(f"fits available VRAM ({available_vram_gb:g} GB)")
        else:
            score -= 3.0 + (min_vram_gb - available_vram_gb) * 0.15
            reasons.append(
                f"estimated minimum VRAM is {min_vram_gb:g} GB (above available {available_vram_gb:g} GB)"
            )
    else:
        budget = _DEFAULT_DEVICE_VRAM_BUDGET_GB.get(target_device, 16.0)
        if min_vram_gb <= budget:
            score += 1.0
            reasons.append(f"within typical {target_device} VRAM budget ({budget:g} GB)")
        else:
            score -= 0.75
            reasons.append(f"above typical {target_device} VRAM budget ({budget:g} GB)")

    return score, reasons


def introspect_training_base_model(
    *,
    model_id: str,
    allow_network: bool = True,
) -> dict[str, Any]:
    """Expose model introspection metadata for API consumers."""
    return introspect_hf_model(
        model_id=model_id,
        allow_network=allow_network,
        timeout_seconds=2.5,
    )


def recommend_training_base_models(
    *,
    target_device: str,
    primary_language: str,
    available_vram_gb: float | None = None,
    task_profile: str | None = None,
    top_k: int = 3,
    adaptive_model_bias: dict[str, float] | None = None,
    adaptive_bias_label: str | None = None,
) -> dict[str, Any]:
    """Recommend base models from catalog registry using hardware/task heuristics."""
    resolved_device = _normalize_target_device(target_device)
    resolved_language = _normalize_primary_language(primary_language)
    resolved_vram = _coerce_vram(available_vram_gb)
    resolved_top_k = _coerce_top_k(top_k)
    resolved_task_profile, suggested_task_type = _resolve_training_task(task_profile)

    catalog_entries = list_model_catalog_entries()
    catalog_meta = _model_catalog_metadata(entry_count=len(catalog_entries))

    bias_map: dict[str, float] = {}
    for raw_model_id, raw_bias in dict(adaptive_model_bias or {}).items():
        model_id = str(raw_model_id or "").strip()
        if not model_id:
            continue
        try:
            parsed_bias = float(raw_bias)
        except (TypeError, ValueError):
            continue
        if parsed_bias <= 0:
            continue
        bias_map[model_id] = parsed_bias

    scored: list[tuple[float, dict[str, Any], list[str], float]] = []
    fits_vram_count = 0
    for model in catalog_entries:
        score, reasons = _score_model(
            model=model,
            target_device=resolved_device,
            primary_language=resolved_language,
            available_vram_gb=resolved_vram,
        )
        model_id = str(model.get("model_id") or "")
        adaptive_bias = float(bias_map.get(model_id, 0.0))
        if adaptive_bias > 0:
            score += adaptive_bias
            if adaptive_bias_label:
                reasons.append(
                    f"project adaptive trend boost (+{adaptive_bias:.2f}; {adaptive_bias_label})"
                )
            else:
                reasons.append(f"project adaptive trend boost (+{adaptive_bias:.2f})")
        min_vram_gb = float(model.get("estimated_min_vram_gb") or 0.0)
        if resolved_vram is not None and resolved_vram >= min_vram_gb:
            fits_vram_count += 1
        scored.append((score, model, reasons, adaptive_bias))

    scored.sort(
        key=lambda item: (
            float(item[0]),
            -float(item[1].get("estimated_min_vram_gb") or 0.0),
            -float(item[1].get("params_b") or 0.0),
        ),
        reverse=True,
    )

    recommendations: list[dict[str, Any]] = []
    introspection_warnings: list[str] = []
    for score, model, reasons, adaptive_bias in scored[:resolved_top_k]:
        min_vram_gb = float(model.get("estimated_min_vram_gb") or 0.0)
        params_b = float(model.get("params_b") or 0.0)
        model_id = str(model.get("model_id") or "")
        introspection = introspect_hf_model(
            model_id=model_id,
            allow_network=True,
            timeout_seconds=1.2,
        )
        memory_profile = dict(introspection.get("memory_profile") or {})
        introspection_min_vram = float(memory_profile.get("estimated_min_vram_gb") or 0.0)
        introspection_ideal_vram = float(memory_profile.get("estimated_ideal_vram_gb") or 0.0)
        metadata_source = str(introspection.get("source") or "none")
        if metadata_source == "none":
            introspection_warnings.append(
                f"unable to introspect model metadata for '{model_id}' (using curated defaults)"
            )

        recommendations.append(
            {
                "model_id": model_id,
                "family": str(model.get("family") or "unknown"),
                "params_b": round(params_b, 2),
                "estimated_min_vram_gb": round(min_vram_gb, 2),
                "estimated_ideal_vram_gb": round(
                    float(model.get("estimated_ideal_vram_gb") or min_vram_gb), 2
                ),
                "catalog_source": str(model.get("catalog_source") or "builtin"),
                "catalog_version": str(model.get("catalog_version") or "builtin-v1"),
                "catalog_entry_is_builtin": bool(model.get("is_builtin", False)),
                "introspection_estimated_min_vram_gb": (
                    round(introspection_min_vram, 2) if introspection_min_vram > 0 else None
                ),
                "introspection_estimated_ideal_vram_gb": (
                    round(introspection_ideal_vram, 2) if introspection_ideal_vram > 0 else None
                ),
                "supported_languages": list(model.get("supported_languages") or []),
                "strengths": list(model.get("strengths") or []),
                "caveats": list(model.get("caveats") or []),
                "match_reasons": reasons[:6],
                "match_score": round(float(score), 4),
                "adaptive_bias": round(float(adaptive_bias), 4),
                "architecture": str(introspection.get("architecture") or "unknown"),
                "context_length": introspection.get("context_length"),
                "license": introspection.get("license"),
                "metadata_source": metadata_source,
                "suggested_defaults": {
                    "task_type": suggested_task_type,
                    "chat_template": str(model.get("preferred_chat_template") or "llama3"),
                    "use_lora": True,
                    "batch_size": _suggest_batch_size(
                        available_vram_gb=resolved_vram,
                        min_vram_gb=min_vram_gb,
                        params_b=params_b,
                    ),
                    "max_seq_length": _suggest_max_seq_length(
                        available_vram_gb=resolved_vram,
                        params_b=params_b,
                    ),
                },
            }
        )

    warnings: list[str] = []
    if resolved_vram is not None and fits_vram_count == 0:
        warnings.append(
            (
                f"No catalog model fits available_vram_gb={resolved_vram:g}. "
                "Recommendations are still provided using lowest-VRAM options."
            )
        )
    if resolved_task_profile == "preference":
        warnings.append(
            "task_profile=preference currently maps to task_type=causal_lm in this wizard; DPO/ORPO flow is planned for a later phase."
        )
    if introspection_warnings:
        warnings.extend(introspection_warnings[:2])
    if catalog_meta.get("plugin_load_errors"):
        warnings.append(
            "Some model catalog plugins failed to load; recommendations include built-in fallback entries."
        )

    return {
        "catalog_strategy": "curated_defaults_with_introspection_v2",
        "catalog_registry_strategy": "dynamic_registry_with_builtin_fallback_v1",
        "catalog_metadata": catalog_meta,
        "request": {
            "target_device": resolved_device,
            "primary_language": resolved_language,
            "available_vram_gb": resolved_vram,
            "task_profile": resolved_task_profile,
            "top_k": resolved_top_k,
        },
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "warnings": warnings,
    }

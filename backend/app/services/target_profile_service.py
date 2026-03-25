import copy
import importlib
import inspect
import re
import threading
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.services.model_introspection_service import introspect_hf_model

TARGET_PROFILE_CATALOG_VERSION = "target_profiles.dynamic_registry/v1"


class TargetConstraint(BaseModel):
    max_parameters_billions: float | None = None
    min_vram_gb: float | None = None
    preferred_formats: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)


class TargetProfile(BaseModel):
    id: str
    name: str
    description: str
    device_class: str = "laptop"
    constraints: TargetConstraint
    inference_runner_default: str | None = None
    # Catalog provenance metadata.
    catalog_source: str = "builtin"
    catalog_version: str = "builtin-v1"
    is_builtin: bool = True


_BUILTIN_TARGET_PROFILES = [
    {
        "id": "vllm_server",
        "name": "vLLM Server",
        "description": "High-throughput GPU server using vLLM.",
        "device_class": "server",
        "constraints": {
            "min_vram_gb": 16.0,
            "preferred_formats": ["huggingface"],
        },
        "inference_runner_default": "runner.vllm",
    },
    {
        "id": "mobile_cpu",
        "name": "Mobile (CPU)",
        "description": "On-device inference using mobile CPU (llama.cpp/GGUF).",
        "device_class": "mobile",
        "constraints": {
            "max_parameters_billions": 4.0,
            "preferred_formats": ["gguf"],
        },
        "inference_runner_default": "runner.ollama",
    },
    {
        "id": "edge_gpu",
        "name": "Edge GPU (NVIDIA Jetson/Desktop)",
        "description": "Inference on edge devices with NVIDIA GPUs (TensorRT).",
        "device_class": "laptop",
        "constraints": {
            "min_vram_gb": 4.0,
            "preferred_formats": ["tensorrt", "onnx"],
        },
        "inference_runner_default": "exporter.tensorrt",
    },
    {
        "id": "browser_webgpu",
        "name": "Browser (WebGPU)",
        "description": "In-browser inference using WebGPU and ONNX.",
        "device_class": "mobile",
        "constraints": {
            "max_parameters_billions": 2.0,
            "preferred_formats": ["onnx"],
        },
        "inference_runner_default": "exporter.onnx",
    },
]

_TARGET_REGISTRY_LOCK = threading.RLock()
_TARGET_PROFILE_REGISTRY: dict[str, TargetProfile] = {}
_TARGET_PROFILE_ORDER: list[str] = []
_TARGET_PLUGINS_INITIALIZED = False
_TARGET_PLUGIN_MODULES_LOADED: set[str] = set()
_TARGET_PLUGIN_LOAD_ERRORS: dict[str, str] = {}
_TARGET_REGISTRATION_SOURCE_CONTEXT: str | None = None


def _normalize_target_id(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_source_module(value: str | None) -> str:
    token = str(value or "").strip()
    return token or "custom"


def _normalize_catalog_version(value: str | None, *, is_builtin: bool) -> str:
    token = str(value or "").strip()
    if token:
        return token
    return "builtin-v1" if is_builtin else "plugin-v1"


def _coerce_target_profile_payload(
    payload: TargetProfile | dict[str, Any],
    *,
    source_module: str | None,
    catalog_version: str | None,
    is_builtin: bool,
) -> TargetProfile:
    if isinstance(payload, TargetProfile):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = dict(payload)
    else:
        raise ValueError("Target profile payload must be a dict or TargetProfile instance.")

    target_id = _normalize_target_id(data.get("id"))
    if not target_id:
        raise ValueError("Target profile id is required.")
    data["id"] = target_id

    source = _normalize_source_module(
        data.get("catalog_source")
        or source_module
        or _TARGET_REGISTRATION_SOURCE_CONTEXT
        or ("builtin" if is_builtin else "custom")
    )
    data["catalog_source"] = source
    data["catalog_version"] = _normalize_catalog_version(
        data.get("catalog_version") or catalog_version,
        is_builtin=bool(data.get("is_builtin", is_builtin)),
    )
    data["is_builtin"] = bool(data.get("is_builtin", is_builtin))
    return TargetProfile(**data)


def _register_target_profile(profile: TargetProfile) -> None:
    target_id = _normalize_target_id(profile.id)
    if not target_id:
        raise ValueError("Target profile id is required.")

    if target_id not in _TARGET_PROFILE_ORDER:
        _TARGET_PROFILE_ORDER.append(target_id)
    _TARGET_PROFILE_REGISTRY[target_id] = profile


def register_target_profile(
    payload: TargetProfile | dict[str, Any],
    *,
    source_module: str | None = None,
    catalog_version: str | None = None,
    is_builtin: bool = False,
) -> None:
    """Public SDK entry-point for registering a target profile."""
    profile = _coerce_target_profile_payload(
        payload,
        source_module=source_module,
        catalog_version=catalog_version,
        is_builtin=is_builtin,
    )
    with _TARGET_REGISTRY_LOCK:
        _register_target_profile(profile)


def register_target_profiles(
    payloads: list[TargetProfile | dict[str, Any]],
    *,
    source_module: str | None = None,
    catalog_version: str | None = None,
    is_builtin: bool = False,
) -> int:
    count = 0
    for item in list(payloads or []):
        register_target_profile(
            item,
            source_module=source_module,
            catalog_version=catalog_version,
            is_builtin=is_builtin,
        )
        count += 1
    return count


def _iter_target_profile_payloads(raw_payload: Any) -> list[TargetProfile | dict[str, Any]]:
    if isinstance(raw_payload, dict):
        rows: list[TargetProfile | dict[str, Any]] = []
        for key, value in raw_payload.items():
            if isinstance(value, dict) and not str(value.get("id") or "").strip():
                value = {**value, "id": str(key)}
            rows.append(value)
        return rows
    if isinstance(raw_payload, (list, tuple, set)):
        return [item for item in list(raw_payload)]
    return []


def _extract_module_target_profiles(module: Any) -> list[TargetProfile | dict[str, Any]]:
    if hasattr(module, "get_target_profiles") and callable(module.get_target_profiles):
        payload = module.get_target_profiles()
        return _iter_target_profile_payloads(payload)
    return _iter_target_profile_payloads(getattr(module, "TARGET_PROFILES", []))


def _call_register_target_profiles_fn(module: Any, module_name: str) -> int:
    register_fn = getattr(module, "register_target_profiles", None)
    if register_fn is None:
        return 0
    if not callable(register_fn):
        raise ValueError(
            f"Target profile plugin module '{module_name}' has non-callable register_target_profiles."
        )

    count = 0

    def register(
        profile_payload: TargetProfile | dict[str, Any],
        *,
        catalog_version: str | None = None,
    ) -> None:
        nonlocal count
        register_target_profile(
            profile_payload,
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


def load_target_profile_plugins(
    module_paths: list[str],
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Load target profile plugins and register their catalog entries."""
    requested_modules = [str(item).strip() for item in list(module_paths or []) if str(item).strip()]
    loaded_modules: list[str] = []
    skipped_modules: list[str] = []
    errors: dict[str, str] = {}

    global _TARGET_REGISTRATION_SOURCE_CONTEXT
    with _TARGET_REGISTRY_LOCK:
        for module_name in requested_modules:
            if module_name in _TARGET_PLUGIN_MODULES_LOADED and not force_reload:
                skipped_modules.append(module_name)
                continue

            try:
                module = importlib.import_module(module_name)
                if force_reload:
                    module = importlib.reload(module)

                registered_count = 0
                prior_context = _TARGET_REGISTRATION_SOURCE_CONTEXT
                _TARGET_REGISTRATION_SOURCE_CONTEXT = module_name
                try:
                    registered_count += _call_register_target_profiles_fn(module, module_name)

                    for payload in _extract_module_target_profiles(module):
                        register_target_profile(
                            payload,
                            source_module=module_name,
                            catalog_version=None,
                            is_builtin=False,
                        )
                        registered_count += 1
                finally:
                    _TARGET_REGISTRATION_SOURCE_CONTEXT = prior_context

                if registered_count == 0:
                    raise ValueError(
                        "No target profiles registered. Provide register_target_profiles(...) or TARGET_PROFILES/get_target_profiles()."
                    )

                _TARGET_PLUGIN_MODULES_LOADED.add(module_name)
                _TARGET_PLUGIN_LOAD_ERRORS.pop(module_name, None)
                loaded_modules.append(module_name)
            except Exception as exc:  # noqa: PERF203
                message = str(exc)
                _TARGET_PLUGIN_LOAD_ERRORS[module_name] = message
                errors[module_name] = message

    return {
        "requested_modules": requested_modules,
        "loaded_modules": sorted(set(loaded_modules)),
        "skipped_modules": sorted(set(skipped_modules)),
        "errors": errors,
    }


def load_target_profile_plugins_from_settings(*, force_reload: bool = False) -> dict[str, Any]:
    modules = [str(item).strip() for item in list(settings.TARGET_PROFILE_PLUGIN_MODULES or []) if str(item).strip()]
    if not modules:
        return {
            "requested_modules": [],
            "loaded_modules": [],
            "skipped_modules": [],
            "errors": {},
        }
    return load_target_profile_plugins(modules, force_reload=force_reload)


def clear_target_profile_plugins() -> None:
    global _TARGET_PLUGINS_INITIALIZED
    with _TARGET_REGISTRY_LOCK:
        _TARGET_PROFILE_REGISTRY.clear()
        _TARGET_PROFILE_ORDER.clear()
        _TARGET_PLUGIN_MODULES_LOADED.clear()
        _TARGET_PLUGIN_LOAD_ERRORS.clear()
        _TARGET_PLUGINS_INITIALIZED = False


def _target_catalog_metadata() -> dict[str, Any]:
    return {
        "catalog_version": TARGET_PROFILE_CATALOG_VERSION,
        "target_count": len(_TARGET_PROFILE_ORDER),
        "loaded_plugin_modules": sorted(_TARGET_PLUGIN_MODULES_LOADED),
        "plugin_load_errors": copy.deepcopy(_TARGET_PLUGIN_LOAD_ERRORS),
        "has_plugin_targets": any(
            not bool(_TARGET_PROFILE_REGISTRY.get(target_id).is_builtin)
            for target_id in _TARGET_PROFILE_ORDER
            if target_id in _TARGET_PROFILE_REGISTRY
        ),
    }


def target_profile_plugin_status() -> dict[str, Any]:
    _ensure_target_profiles_loaded()
    with _TARGET_REGISTRY_LOCK:
        return {
            "requested_modules": [
                str(item).strip()
                for item in list(settings.TARGET_PROFILE_PLUGIN_MODULES or [])
                if str(item).strip()
            ],
            "loaded_modules": sorted(_TARGET_PLUGIN_MODULES_LOADED),
            "failed_modules": copy.deepcopy(_TARGET_PLUGIN_LOAD_ERRORS),
            "registered_target_count": len(_TARGET_PROFILE_ORDER),
        }


def _register_builtin_target_profiles() -> None:
    for payload in _BUILTIN_TARGET_PROFILES:
        profile = _coerce_target_profile_payload(
            payload,
            source_module="builtin",
            catalog_version="builtin-v1",
            is_builtin=True,
        )
        _register_target_profile(profile)


def _ensure_target_profiles_loaded() -> None:
    global _TARGET_PLUGINS_INITIALIZED
    if _TARGET_PLUGINS_INITIALIZED:
        return

    with _TARGET_REGISTRY_LOCK:
        if _TARGET_PLUGINS_INITIALIZED:
            return
        _TARGET_PROFILE_REGISTRY.clear()
        _TARGET_PROFILE_ORDER.clear()
        _TARGET_PLUGIN_MODULES_LOADED.clear()
        _TARGET_PLUGIN_LOAD_ERRORS.clear()
        _register_builtin_target_profiles()
        load_target_profile_plugins_from_settings(force_reload=False)
        _TARGET_PLUGINS_INITIALIZED = True


def list_targets() -> list[TargetProfile]:
    _ensure_target_profiles_loaded()
    with _TARGET_REGISTRY_LOCK:
        return [
            _TARGET_PROFILE_REGISTRY[target_id].model_copy(deep=True)
            for target_id in _TARGET_PROFILE_ORDER
            if target_id in _TARGET_PROFILE_REGISTRY
        ]


def list_target_catalog() -> dict[str, Any]:
    _ensure_target_profiles_loaded()
    rows = [target.model_dump() for target in list_targets()]
    with _TARGET_REGISTRY_LOCK:
        return {
            **_target_catalog_metadata(),
            "targets": rows,
        }


def get_target_by_id(target_id: str) -> TargetProfile | None:
    token = _normalize_target_id(target_id)
    if not token:
        return None

    _ensure_target_profiles_loaded()
    with _TARGET_REGISTRY_LOCK:
        target = _TARGET_PROFILE_REGISTRY.get(token)
        return target.model_copy(deep=True) if target is not None else None


def resolve_target_device(
    target_id: str | None,
    *,
    fallback: str | None = None,
) -> str:
    token = str(target_id or "").strip().lower()
    target = get_target_by_id(token) if token else None
    if target is not None:
        device = str(target.device_class or "").strip().lower()
        if device in {"mobile", "laptop", "server"}:
            return device

    fallback_device = str(fallback or "").strip().lower()
    if fallback_device in {"mobile", "laptop", "server"}:
        return fallback_device
    return "laptop"


def _coerce_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return round(parsed, 4)


def _fallback_params_from_name(model_name: str) -> float | None:
    token = str(model_name or "").strip().lower()
    if not token:
        return None
    # Common model-size hints: "1b", "1.5b", "7b", "70b".
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", token)
    if match is None:
        return None
    try:
        parsed = float(match.group(1))
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return round(parsed, 4)


def _fallback_memory_from_params(params_b: float | None) -> float | None:
    if params_b is None:
        return None
    # Coarse 4-bit-ish inference footprint heuristic.
    memory_gb = (float(params_b) * 0.7) + 1.0
    if memory_gb <= 0:
        return None
    return round(memory_gb, 2)


def _vram_block_margin_gb(target_min_vram_gb: float) -> float:
    # Keep a small uncertainty band so only clearly-over-baseline cases hard-block.
    return round(max(0.5, float(target_min_vram_gb) * 0.1), 3)


def _is_clearly_over_target_vram(*, estimated_min_vram_gb: float, target_min_vram_gb: float) -> bool:
    return float(estimated_min_vram_gb) > float(target_min_vram_gb) + _vram_block_margin_gb(float(target_min_vram_gb))


def _resolve_model_metadata(model_name: str) -> dict[str, Any]:
    introspection = introspect_hf_model(
        model_id=str(model_name or "").strip(),
        allow_network=True,
        timeout_seconds=1.8,
    )
    params_b = _coerce_positive_float(introspection.get("params_estimate_b"))
    params_source = "introspection"
    if params_b is None:
        params_b = _fallback_params_from_name(model_name)
        if params_b is not None:
            params_source = "name_hint"
        else:
            params_source = "unknown"

    memory_profile = dict(introspection.get("memory_profile") or {})
    estimated_min_vram = _coerce_positive_float(memory_profile.get("estimated_min_vram_gb"))
    estimated_ideal_vram = _coerce_positive_float(memory_profile.get("estimated_ideal_vram_gb"))
    if estimated_min_vram is None:
        estimated_min_vram = _fallback_memory_from_params(params_b)
    if estimated_ideal_vram is None and estimated_min_vram is not None:
        estimated_ideal_vram = round(max(estimated_min_vram + 1.0, estimated_min_vram * 1.25), 2)

    return {
        "model_id": str(model_name or "").strip(),
        "parameters_billions": params_b,
        "parameters_source": params_source,
        "estimated_min_vram_gb": estimated_min_vram,
        "estimated_ideal_vram_gb": estimated_ideal_vram,
        "architecture": str(introspection.get("architecture") or "").strip() or None,
        "context_length": introspection.get("context_length"),
        "license": str(introspection.get("license") or "").strip() or None,
        "source": str(introspection.get("source") or "none").strip() or "none",
        "resolved": bool(introspection.get("resolved", False)),
        "introspection": introspection,
    }


def check_compatibility(model_name: str, target_id: str) -> dict[str, Any]:
    target = get_target_by_id(target_id)
    if not target:
        return {
            "compatible": False,
            "reason": "Target profile not found",
            "reasons": ["Target profile not found."],
            "warnings": [],
            "target": None,
            "model_metadata": _resolve_model_metadata(model_name),
        }

    metadata = _resolve_model_metadata(model_name)
    parameters_billions = _coerce_positive_float(metadata.get("parameters_billions"))
    estimated_min_vram_gb = _coerce_positive_float(metadata.get("estimated_min_vram_gb"))

    reasons: list[str] = []
    warnings: list[str] = []

    max_parameters = _coerce_positive_float(target.constraints.max_parameters_billions)
    if max_parameters is not None:
        if parameters_billions is None:
            reasons.append(
                (
                    "Model parameter size could not be inferred; unable to validate "
                    f"max_parameters limit ({max_parameters:g}B) for target '{target.id}'."
                )
            )
        elif parameters_billions > max_parameters:
            reasons.append(
                (
                    f"Model size ({parameters_billions:g}B) exceeds target limit "
                    f"({max_parameters:g}B)."
                )
            )

    min_target_vram = _coerce_positive_float(target.constraints.min_vram_gb)
    vram_check: dict[str, Any] = {
        "status": "not_applicable",
        "target_min_vram_gb": min_target_vram,
        "estimated_min_vram_gb": estimated_min_vram_gb,
    }
    if min_target_vram is not None:
        if estimated_min_vram_gb is None:
            warnings.append(
                (
                    "Unable to estimate minimum VRAM for this model; "
                    f"VRAM compatibility check was skipped for target baseline ({min_target_vram:g} GB)."
                )
            )
            vram_check.update(
                {
                    "status": "unknown",
                    "message": (
                        "Estimated minimum VRAM is unavailable; compatibility is inferred from "
                        "other constraints only."
                    ),
                }
            )
        else:
            gap_gb = round(float(estimated_min_vram_gb) - float(min_target_vram), 3)
            block_margin_gb = _vram_block_margin_gb(float(min_target_vram))
            if _is_clearly_over_target_vram(
                estimated_min_vram_gb=float(estimated_min_vram_gb),
                target_min_vram_gb=float(min_target_vram),
            ):
                reasons.append(
                    (
                        f"Estimated minimum VRAM ({estimated_min_vram_gb:g} GB) exceeds target baseline "
                        f"({min_target_vram:g} GB) by {gap_gb:g} GB."
                    )
                )
                vram_check.update(
                    {
                        "status": "blocked",
                        "gap_gb": gap_gb,
                        "block_margin_gb": block_margin_gb,
                    }
                )
            elif gap_gb > 0:
                warnings.append(
                    (
                        f"Estimated minimum VRAM ({estimated_min_vram_gb:g} GB) is close to target baseline "
                        f"({min_target_vram:g} GB); deployment may require tighter quantization/runtime tuning."
                    )
                )
                vram_check.update(
                    {
                        "status": "warning",
                        "gap_gb": gap_gb,
                        "block_margin_gb": block_margin_gb,
                    }
                )
            else:
                vram_check.update(
                    {
                        "status": "pass",
                        "gap_gb": gap_gb,
                        "block_margin_gb": block_margin_gb,
                    }
                )

    return {
        "compatible": len(reasons) == 0,
        "reasons": reasons,
        "warnings": warnings,
        "vram_check": vram_check,
        "target": target.model_dump(),
        "model_metadata": metadata,
    }


def estimate_metrics(model_name: str, target_id: str) -> dict[str, Any]:
    metadata = _resolve_model_metadata(model_name)
    params_b = _coerce_positive_float(metadata.get("parameters_billions")) or 7.0
    memory_gb = _coerce_positive_float(metadata.get("estimated_min_vram_gb")) or _fallback_memory_from_params(params_b) or 5.9

    # Latency estimation (tokens/sec) with coarse target scaling.
    tps = 50.0
    if target_id == "mobile_cpu":
        tps = 5.0
    elif target_id == "browser_webgpu":
        tps = 10.0
    elif target_id == "edge_gpu":
        tps = 25.0

    # Smaller models generally decode faster on the same target.
    tps = tps * (7.0 / max(0.25, float(params_b)))

    return {
        "estimated_memory_gb": round(float(memory_gb), 2),
        "estimated_latency_tps": round(tps, 2),
        "target_id": target_id,
        "model_name": model_name,
        "model_metadata": metadata,
    }

"""Domain hook registry, plugin loading, and safe execution helpers."""

from __future__ import annotations

import copy
import importlib
import inspect
import re
import threading
from typing import Any, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.domain_runtime_service import resolve_project_domain_runtime

DEFAULT_NORMALIZER_HOOK_ID = "default-normalizer"
DEFAULT_VALIDATOR_HOOK_ID = "default-validator"
DEFAULT_EVALUATOR_HOOK_ID = "default-evaluator"

NormalizerHook = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any] | None]
ValidatorHook = Callable[[list[dict[str, Any]], dict[str, Any], dict[str, Any]], dict[str, Any]]
EvaluatorHook = Callable[[str, dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def _normalize_hook_spec(spec: dict[str, Any] | None, default_id: str) -> dict[str, Any]:
    payload = spec if isinstance(spec, dict) else {}
    hook_id_raw = payload.get("id") or payload.get("hook_id") or default_id
    hook_id = _normalize_token(str(hook_id_raw)) or default_id
    config = payload.get("config")
    return {
        "id": hook_id,
        "config": dict(config) if isinstance(config, dict) else {},
    }


def _normalizer_default(
    _raw_record: dict[str, Any],
    canonical_record: dict[str, Any],
    _config: dict[str, Any],
) -> dict[str, Any] | None:
    return dict(canonical_record) if isinstance(canonical_record, dict) else None


def _normalizer_qa_required(
    _raw_record: dict[str, Any],
    canonical_record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(canonical_record, dict):
        return None
    require_question = _to_bool(config.get("require_question"), True)
    require_answer = _to_bool(config.get("require_answer"), True)
    question = str(canonical_record.get("question") or "").strip()
    answer = str(canonical_record.get("answer") or "").strip()
    if require_question and not question:
        return None
    if require_answer and not answer:
        return None
    return dict(canonical_record)


def _normalizer_min_text_length(
    _raw_record: dict[str, Any],
    canonical_record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(canonical_record, dict):
        return None
    min_chars = max(0, _to_int(config.get("min_chars"), 1))
    text = str(canonical_record.get("text") or "")
    if len(text) < min_chars:
        return None
    return dict(canonical_record)


def _normalizer_strip_markdown(
    _raw_record: dict[str, Any],
    canonical_record: dict[str, Any],
    _config: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(canonical_record, dict):
        return None
    cleaned = dict(canonical_record)
    for key in ("text", "question", "answer"):
        value = cleaned.get(key)
        if not isinstance(value, str):
            continue
        text = re.sub(r"[`*_>#]", "", value)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned[key] = text
    return cleaned


def _validator_default(
    records: list[dict[str, Any]],
    profile: dict[str, Any],
    _config: dict[str, Any],
) -> dict[str, Any]:
    total = len(records)
    return {
        "status": "ok",
        "total_records": total,
        "valid_records": total,
        "failed_records": 0,
        "pass_rate": 1.0 if total else 0.0,
        "base_normalization_coverage": profile.get("normalization_coverage"),
    }


def _validator_min_text_length(
    records: list[dict[str, Any]],
    profile: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    min_chars = max(0, _to_int(config.get("min_chars"), 30))
    failures: list[int] = []
    valid = 0
    for idx, row in enumerate(records):
        text = str(row.get("text") or "")
        if len(text) >= min_chars:
            valid += 1
        else:
            failures.append(idx)
    total = len(records)
    failed = total - valid
    pass_rate = valid / total if total else 0.0
    return {
        "status": "ok" if failed == 0 else "warning",
        "rule": "min_text_length",
        "min_chars": min_chars,
        "total_records": total,
        "valid_records": valid,
        "failed_records": failed,
        "pass_rate": round(pass_rate, 6),
        "failed_indices_preview": failures[:20],
        "base_normalization_coverage": profile.get("normalization_coverage"),
    }


def _validator_qa_pair_presence(
    records: list[dict[str, Any]],
    profile: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    min_ratio = min(1.0, max(0.0, _to_float(config.get("min_ratio"), 0.8)))
    qa_count = 0
    for row in records:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if question and answer:
            qa_count += 1
    total = len(records)
    ratio = (qa_count / total) if total else 0.0
    return {
        "status": "ok" if ratio >= min_ratio else "warning",
        "rule": "qa_pair_presence",
        "min_ratio": min_ratio,
        "qa_pair_count": qa_count,
        "total_records": total,
        "qa_pair_ratio": round(ratio, 6),
        "base_normalization_coverage": profile.get("normalization_coverage"),
    }


def _evaluator_default(
    _eval_type: str,
    metrics: dict[str, Any],
    _config: dict[str, Any],
    _context: dict[str, Any],
) -> dict[str, Any]:
    return dict(metrics)


def _evaluator_pass_rate_band(
    _eval_type: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    _context: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(metrics)
    score = enriched.get("pass_rate")
    if score is None:
        score = enriched.get("exact_match")
    if score is None:
        score = enriched.get("f1")
    if score is None and isinstance(enriched.get("average_score"), (int, float)):
        score = float(enriched["average_score"]) / 5.0
    if not isinstance(score, (int, float)):
        return enriched

    high_threshold = min(1.0, max(0.0, _to_float(config.get("high_threshold"), 0.8)))
    medium_threshold = min(high_threshold, max(0.0, _to_float(config.get("medium_threshold"), 0.6)))
    value = float(score)
    if value >= high_threshold:
        band = "high"
    elif value >= medium_threshold:
        band = "medium"
    else:
        band = "low"
    enriched["quality_band"] = band
    return enriched


def _evaluator_weighted_score(
    _eval_type: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    _context: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(metrics)
    weights = config.get("weights")
    if not isinstance(weights, dict):
        return enriched

    weighted_sum = 0.0
    total_weight = 0.0
    for metric_name, weight in weights.items():
        metric_value = metrics.get(str(metric_name))
        if not isinstance(metric_value, (int, float)):
            continue
        weight_value = max(0.0, _to_float(weight, 0.0))
        if weight_value == 0:
            continue
        weighted_sum += float(metric_value) * weight_value
        total_weight += weight_value

    if total_weight > 0:
        enriched["weighted_score"] = round(weighted_sum / total_weight, 6)
    return enriched


BUILTIN_NORMALIZER_HOOKS: dict[str, NormalizerHook] = {
    "default-normalizer": _normalizer_default,
    "qa-required-normalizer": _normalizer_qa_required,
    "min-text-length-normalizer": _normalizer_min_text_length,
    "strip-markdown-normalizer": _normalizer_strip_markdown,
}

BUILTIN_VALIDATOR_HOOKS: dict[str, ValidatorHook] = {
    "default-validator": _validator_default,
    "min-text-length-validator": _validator_min_text_length,
    "qa-pair-validator": _validator_qa_pair_presence,
}

BUILTIN_EVALUATOR_HOOKS: dict[str, EvaluatorHook] = {
    "default-evaluator": _evaluator_default,
    "pass-rate-band-evaluator": _evaluator_pass_rate_band,
    "weighted-score-evaluator": _evaluator_weighted_score,
}

BUILTIN_HOOK_CATALOG = {
    "normalizers": {
        "default-normalizer": "No-op normalizer; uses canonical record as-is.",
        "qa-required-normalizer": "Drops records missing question/answer (config: require_question, require_answer).",
        "min-text-length-normalizer": "Drops records with text shorter than min_chars.",
        "strip-markdown-normalizer": "Removes basic markdown tokens from text/question/answer.",
    },
    "validators": {
        "default-validator": "No-op validator with pass-through summary.",
        "min-text-length-validator": "Reports validation failures for records shorter than min_chars.",
        "qa-pair-validator": "Checks ratio of records containing both question and answer.",
    },
    "evaluators": {
        "default-evaluator": "No-op evaluator; metrics unchanged.",
        "pass-rate-band-evaluator": "Adds quality_band from pass_rate/exact_match/f1.",
        "weighted-score-evaluator": "Adds weighted_score using config.weights.",
    },
}

_PLUGIN_NORMALIZER_HOOKS: dict[str, NormalizerHook] = {}
_PLUGIN_VALIDATOR_HOOKS: dict[str, ValidatorHook] = {}
_PLUGIN_EVALUATOR_HOOKS: dict[str, EvaluatorHook] = {}
_PLUGIN_HOOK_CATALOG: dict[str, dict[str, str]] = {
    "normalizers": {},
    "validators": {},
    "evaluators": {},
}
_PLUGIN_HOOK_SOURCE: dict[str, dict[str, str]] = {
    "normalizers": {},
    "validators": {},
    "evaluators": {},
}
_LOADED_PLUGIN_MODULES: set[str] = set()
_PLUGIN_LOAD_ERRORS: dict[str, str] = {}
_HOOK_REGISTRY_LOCK = threading.Lock()


def _normalize_hook_kind(kind: str) -> str:
    token = _normalize_token(kind)
    aliases = {
        "normalizer": "normalizers",
        "normalizers": "normalizers",
        "validator": "validators",
        "validators": "validators",
        "evaluator": "evaluators",
        "evaluators": "evaluators",
    }
    return aliases.get(token, token)


def _normalizer_registry() -> dict[str, NormalizerHook]:
    merged = dict(BUILTIN_NORMALIZER_HOOKS)
    merged.update(_PLUGIN_NORMALIZER_HOOKS)
    return merged


def _validator_registry() -> dict[str, ValidatorHook]:
    merged = dict(BUILTIN_VALIDATOR_HOOKS)
    merged.update(_PLUGIN_VALIDATOR_HOOKS)
    return merged


def _evaluator_registry() -> dict[str, EvaluatorHook]:
    merged = dict(BUILTIN_EVALUATOR_HOOKS)
    merged.update(_PLUGIN_EVALUATOR_HOOKS)
    return merged


def _register_plugin_hook(
    hook_kind: str,
    hook_id: str,
    handler: Callable[..., Any],
    *,
    source_module: str,
    description: str = "",
) -> None:
    hook_kind = _normalize_hook_kind(hook_kind)
    normalized_hook_id = _normalize_token(hook_id)
    if not normalized_hook_id:
        raise ValueError(f"Invalid hook id '{hook_id}'")
    if not callable(handler):
        raise ValueError(f"Hook '{normalized_hook_id}' handler is not callable")

    if hook_kind == "normalizers":
        _PLUGIN_NORMALIZER_HOOKS[normalized_hook_id] = handler  # type: ignore[assignment]
    elif hook_kind == "validators":
        _PLUGIN_VALIDATOR_HOOKS[normalized_hook_id] = handler  # type: ignore[assignment]
    elif hook_kind == "evaluators":
        _PLUGIN_EVALUATOR_HOOKS[normalized_hook_id] = handler  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported hook kind '{hook_kind}'")

    _PLUGIN_HOOK_CATALOG[hook_kind][normalized_hook_id] = (
        description.strip() or f"Plugin hook from {source_module}"
    )
    _PLUGIN_HOOK_SOURCE[hook_kind][normalized_hook_id] = source_module


def _register_hooks_from_mapping(
    hook_kind: str,
    mapping: dict[str, Any],
    *,
    source_module: str,
) -> int:
    count = 0
    for hook_id, payload in mapping.items():
        description = ""
        handler: Callable[..., Any] | None = None

        if callable(payload):
            handler = payload
        elif isinstance(payload, dict):
            candidate = payload.get("handler") or payload.get("callable") or payload.get("fn")
            if callable(candidate):
                handler = candidate
            description = str(payload.get("description") or "").strip()

        if handler is None:
            continue
        _register_plugin_hook(
            hook_kind,
            str(hook_id),
            handler,
            source_module=source_module,
            description=description,
        )
        count += 1
    return count


def _extract_module_hook_mappings(module: Any) -> dict[str, dict[str, Any]]:
    """Resolve plugin hook payload from module exports."""
    if hasattr(module, "get_domain_hooks") and callable(module.get_domain_hooks):
        payload = module.get_domain_hooks()
        if isinstance(payload, dict):
            normalizers = payload.get("normalizers")
            if not isinstance(normalizers, dict):
                normalizers = payload.get("normalizer")
            validators = payload.get("validators")
            if not isinstance(validators, dict):
                validators = payload.get("validator")
            evaluators = payload.get("evaluators")
            if not isinstance(evaluators, dict):
                evaluators = payload.get("evaluator")
            return {
                "normalizers": normalizers if isinstance(normalizers, dict) else {},
                "validators": validators if isinstance(validators, dict) else {},
                "evaluators": evaluators if isinstance(evaluators, dict) else {},
            }

    # Legacy/simple constants.
    return {
        "normalizers": getattr(module, "NORMALIZER_HOOKS", {}) if isinstance(getattr(module, "NORMALIZER_HOOKS", {}), dict) else {},
        "validators": getattr(module, "VALIDATOR_HOOKS", {}) if isinstance(getattr(module, "VALIDATOR_HOOKS", {}), dict) else {},
        "evaluators": getattr(module, "EVALUATOR_HOOKS", {}) if isinstance(getattr(module, "EVALUATOR_HOOKS", {}), dict) else {},
    }


def _call_register_function(module: Any, source_module: str) -> int:
    register_fn = getattr(module, "register_domain_hooks", None)
    if not callable(register_fn):
        return 0

    count = 0

    def register(
        hook_kind: str,
        hook_id: str,
        handler: Callable[..., Any],
        *,
        description: str = "",
    ) -> None:
        nonlocal count
        _register_plugin_hook(
            hook_kind,
            hook_id,
            handler,
            source_module=source_module,
            description=description,
        )
        count += 1

    sig = inspect.signature(register_fn)
    if len(sig.parameters) == 0:
        register_fn()
        return count
    if len(sig.parameters) == 1:
        register_fn(register)
        return count
    # Optional compatibility for register_domain_hooks(register, context)
    register_fn(register, {"settings": settings})
    return count


def load_hook_plugins(
    module_paths: list[str],
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Load hook plugin modules and register their hooks.

    Plugin module contract (any one is enough):
    1) `def register_domain_hooks(register): ...`
    2) `def get_domain_hooks() -> {"normalizers": {...}, "validators": {...}, "evaluators": {...}}`
    3) module constants `NORMALIZER_HOOKS`, `VALIDATOR_HOOKS`, `EVALUATOR_HOOKS`
    """
    loaded_now: list[str] = []
    errors: dict[str, str] = {}
    skipped: list[str] = []

    with _HOOK_REGISTRY_LOCK:
        for raw_path in module_paths:
            module_path = str(raw_path).strip()
            if not module_path:
                continue
            if module_path in _LOADED_PLUGIN_MODULES and not force_reload:
                skipped.append(module_path)
                continue

            try:
                module = importlib.import_module(module_path)
                if force_reload:
                    module = importlib.reload(module)

                registered_count = _call_register_function(module, module_path)

                mappings = _extract_module_hook_mappings(module)
                registered_count += _register_hooks_from_mapping(
                    "normalizers",
                    mappings.get("normalizers", {}),
                    source_module=module_path,
                )
                registered_count += _register_hooks_from_mapping(
                    "validators",
                    mappings.get("validators", {}),
                    source_module=module_path,
                )
                registered_count += _register_hooks_from_mapping(
                    "evaluators",
                    mappings.get("evaluators", {}),
                    source_module=module_path,
                )

                if registered_count == 0:
                    raise ValueError(
                        "No hooks registered. Define register_domain_hooks(...) or hook mappings."
                    )

                _LOADED_PLUGIN_MODULES.add(module_path)
                _PLUGIN_LOAD_ERRORS.pop(module_path, None)
                loaded_now.append(module_path)
            except Exception as e:
                message = str(e)
                _PLUGIN_LOAD_ERRORS[module_path] = message
                errors[module_path] = message

    return {
        "requested_modules": [m.strip() for m in module_paths if str(m).strip()],
        "loaded_modules": loaded_now,
        "skipped_modules": skipped,
        "errors": errors,
    }


def load_hook_plugins_from_settings(*, force_reload: bool = False) -> dict[str, Any]:
    modules = [str(item).strip() for item in settings.DOMAIN_HOOK_PLUGIN_MODULES if str(item).strip()]
    if not modules:
        return {
            "requested_modules": [],
            "loaded_modules": [],
            "skipped_modules": [],
            "errors": {},
        }
    return load_hook_plugins(modules, force_reload=force_reload)


def clear_plugin_hooks() -> None:
    with _HOOK_REGISTRY_LOCK:
        _PLUGIN_NORMALIZER_HOOKS.clear()
        _PLUGIN_VALIDATOR_HOOKS.clear()
        _PLUGIN_EVALUATOR_HOOKS.clear()
        _PLUGIN_HOOK_CATALOG["normalizers"].clear()
        _PLUGIN_HOOK_CATALOG["validators"].clear()
        _PLUGIN_HOOK_CATALOG["evaluators"].clear()
        _PLUGIN_HOOK_SOURCE["normalizers"].clear()
        _PLUGIN_HOOK_SOURCE["validators"].clear()
        _PLUGIN_HOOK_SOURCE["evaluators"].clear()
        _LOADED_PLUGIN_MODULES.clear()
        _PLUGIN_LOAD_ERRORS.clear()


def list_domain_hook_catalog() -> dict[str, Any]:
    catalog = {
        "normalizers": {
            **copy.deepcopy(BUILTIN_HOOK_CATALOG["normalizers"]),
            **copy.deepcopy(_PLUGIN_HOOK_CATALOG["normalizers"]),
        },
        "validators": {
            **copy.deepcopy(BUILTIN_HOOK_CATALOG["validators"]),
            **copy.deepcopy(_PLUGIN_HOOK_CATALOG["validators"]),
        },
        "evaluators": {
            **copy.deepcopy(BUILTIN_HOOK_CATALOG["evaluators"]),
            **copy.deepcopy(_PLUGIN_HOOK_CATALOG["evaluators"]),
        },
        "plugin_modules_loaded": sorted(_LOADED_PLUGIN_MODULES),
        "plugin_load_errors": copy.deepcopy(_PLUGIN_LOAD_ERRORS),
        "plugin_hook_sources": copy.deepcopy(_PLUGIN_HOOK_SOURCE),
    }
    return catalog


async def resolve_project_domain_hooks(db: AsyncSession, project_id: int) -> dict[str, Any]:
    runtime = await resolve_project_domain_runtime(db, project_id)
    raw_hooks = runtime.get("pack_hooks")
    pack_hooks = raw_hooks if isinstance(raw_hooks, dict) else {}
    return {
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "normalizer": _normalize_hook_spec(pack_hooks.get("normalizer"), DEFAULT_NORMALIZER_HOOK_ID),
        "validator": _normalize_hook_spec(pack_hooks.get("validator"), DEFAULT_VALIDATOR_HOOK_ID),
        "evaluator": _normalize_hook_spec(pack_hooks.get("evaluator"), DEFAULT_EVALUATOR_HOOK_ID),
    }


def apply_normalizer_hook(
    raw_record: dict[str, Any],
    canonical_record: dict[str, Any] | None,
    hook_spec: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if canonical_record is None:
        return None
    spec = _normalize_hook_spec(hook_spec, DEFAULT_NORMALIZER_HOOK_ID)
    hook_id = spec["id"]
    config = spec["config"]
    hook = _normalizer_registry().get(hook_id, _normalizer_default)
    try:
        result = hook(raw_record, canonical_record, config)
        if isinstance(result, dict):
            return result
        return None
    except Exception:
        return dict(canonical_record)


def run_validator_hook(
    records: list[dict[str, Any]],
    base_profile: dict[str, Any] | None,
    hook_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = _normalize_hook_spec(hook_spec, DEFAULT_VALIDATOR_HOOK_ID)
    hook_id = spec["id"]
    config = spec["config"]
    profile = base_profile if isinstance(base_profile, dict) else {}
    hook = _validator_registry().get(hook_id, _validator_default)
    try:
        report = hook(records, profile, config)
        if not isinstance(report, dict):
            report = {}
        report["hook_id"] = hook_id
        return report
    except Exception as e:
        return {
            "hook_id": hook_id,
            "status": "error",
            "error": str(e),
            "total_records": len(records),
        }


def apply_evaluator_hook(
    eval_type: str,
    metrics: dict[str, Any] | None,
    hook_spec: dict[str, Any] | None,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = _normalize_hook_spec(hook_spec, DEFAULT_EVALUATOR_HOOK_ID)
    hook_id = spec["id"]
    config = spec["config"]
    source_metrics = metrics if isinstance(metrics, dict) else {}
    hook = _evaluator_registry().get(hook_id, _evaluator_default)
    try:
        enriched = hook(eval_type, dict(source_metrics), config, dict(context or {}))
        if not isinstance(enriched, dict):
            enriched = dict(source_metrics)
        enriched["hook_evaluator_id"] = hook_id
        return enriched
    except Exception as e:
        fallback = dict(source_metrics)
        fallback["hook_evaluator_id"] = hook_id
        fallback["hook_evaluator_error"] = str(e)
        return fallback

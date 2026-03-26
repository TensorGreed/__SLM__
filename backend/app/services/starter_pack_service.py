"""Dynamic starter-pack registry for domain-specific onboarding defaults."""

from __future__ import annotations

import copy
import importlib
import inspect
import threading
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings

STARTER_PACK_CATALOG_VERSION = "starter_packs.dynamic_registry/v1"


class StarterPack(BaseModel):
    id: str
    display_name: str
    description: str
    domain: str
    recommended_model_families: list[str] = Field(default_factory=list)
    recommended_models: list[str] = Field(default_factory=list)
    default_base_model_name: str | None = None
    adapter_task_defaults: dict[str, Any] = Field(default_factory=dict)
    evaluation_gate_defaults: dict[str, Any] = Field(default_factory=dict)
    safety_compliance_reminders: list[str] = Field(default_factory=list)
    target_profile_default: str = "vllm_server"
    catalog_source: str = "builtin"
    catalog_version: str = "builtin-v1"
    is_builtin: bool = True


_BUILTIN_STARTER_PACKS: list[dict[str, Any]] = [
    {
        "id": "legal",
        "display_name": "Legal Starter",
        "description": "Defaults for contract review, policy QA, and citation-grounded legal assistants.",
        "domain": "legal",
        "recommended_model_families": ["llama", "qwen", "mistral"],
        "recommended_models": [
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        "default_base_model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "adapter_task_defaults": {
            "adapter_id": "default-canonical",
            "task_profile": "instruction_sft",
            "adapter_config": {
                "expect_citations": True,
                "preserve_section_headers": True,
            },
        },
        "evaluation_gate_defaults": {
            "must_pass": True,
            "blocked_if_missing": True,
            "min_score": 0.78,
            "required_metrics": {
                "llm_judge_pass_rate": 0.82,
                "citation_coverage": 0.8,
                "hallucination_rate_max": 0.1,
            },
        },
        "safety_compliance_reminders": [
            "Do not represent outputs as legal advice.",
            "Require citation to trusted sources for high-risk answers.",
            "Add human review for policy-impacting recommendations.",
        ],
        "target_profile_default": "edge_gpu",
    },
    {
        "id": "customer_support",
        "display_name": "Customer Support Starter",
        "description": "Defaults for support deflection, ticket triage, and workflow-aware assistant responses.",
        "domain": "customer_support",
        "recommended_model_families": ["qwen", "llama", "phi"],
        "recommended_models": [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
        ],
        "default_base_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_task_defaults": {
            "adapter_id": "default-canonical",
            "task_profile": "instruction_sft",
            "adapter_config": {
                "include_resolution_tags": True,
                "normalize_sla_labels": True,
            },
        },
        "evaluation_gate_defaults": {
            "must_pass": True,
            "blocked_if_missing": False,
            "min_score": 0.72,
            "required_metrics": {
                "intent_accuracy": 0.8,
                "response_quality": 0.78,
                "escalation_precision": 0.75,
            },
        },
        "safety_compliance_reminders": [
            "Mask or redact personal data in support transcripts.",
            "Escalate billing, legal, or account-security issues to human agents.",
        ],
        "target_profile_default": "edge_gpu",
    },
    {
        "id": "healthcare_generic",
        "display_name": "Healthcare Starter (Non-Diagnostic)",
        "description": "Defaults for administrative and educational healthcare assistants; not for diagnosis.",
        "domain": "healthcare",
        "recommended_model_families": ["llama", "qwen", "gemma"],
        "recommended_models": [
            "meta-llama/Llama-3.2-3B-Instruct",
            "google/gemma-2-2b-it",
        ],
        "default_base_model_name": "google/gemma-2-2b-it",
        "adapter_task_defaults": {
            "adapter_id": "default-canonical",
            "task_profile": "instruction_sft",
            "adapter_config": {
                "normalize_medical_abbreviations": True,
                "retain_source_attribution": True,
            },
        },
        "evaluation_gate_defaults": {
            "must_pass": True,
            "blocked_if_missing": True,
            "min_score": 0.82,
            "required_metrics": {
                "factuality": 0.88,
                "safety_pass_rate": 0.95,
                "hallucination_rate_max": 0.06,
            },
        },
        "safety_compliance_reminders": [
            "Keep usage non-diagnostic and informational only.",
            "Route symptom interpretation and treatment decisions to licensed clinicians.",
            "Avoid storing protected health information without compliance controls.",
        ],
        "target_profile_default": "vllm_server",
    },
    {
        "id": "finance_generic",
        "display_name": "Finance Starter (Non-Advisory)",
        "description": "Defaults for financial document QA and operations workflows; not investment advice.",
        "domain": "finance",
        "recommended_model_families": ["qwen", "llama", "mistral"],
        "recommended_models": [
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "default_base_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_task_defaults": {
            "adapter_id": "default-canonical",
            "task_profile": "instruction_sft",
            "adapter_config": {
                "normalize_currency_fields": True,
                "preserve_table_context": True,
            },
        },
        "evaluation_gate_defaults": {
            "must_pass": True,
            "blocked_if_missing": True,
            "min_score": 0.8,
            "required_metrics": {
                "numeric_accuracy": 0.9,
                "llm_judge_pass_rate": 0.84,
                "policy_compliance": 0.9,
            },
        },
        "safety_compliance_reminders": [
            "Do not present outputs as investment or tax advice.",
            "Require human approval for transactions, pricing changes, and disclosures.",
            "Track provenance for generated financial summaries.",
        ],
        "target_profile_default": "vllm_server",
    },
]

_STARTER_PACK_LOCK = threading.RLock()
_STARTER_PACK_REGISTRY: dict[str, StarterPack] = {}
_STARTER_PACK_ORDER: list[str] = []
_STARTER_PACKS_INITIALIZED = False
_STARTER_PACK_PLUGIN_MODULES_LOADED: set[str] = set()
_STARTER_PACK_PLUGIN_LOAD_ERRORS: dict[str, str] = {}
_STARTER_PACK_REGISTRATION_SOURCE_CONTEXT: str | None = None


def _normalize_pack_id(value: str | None) -> str:
    token = str(value or "").strip().lower()
    return token.replace(" ", "_")


def _normalize_source_module(value: str | None) -> str:
    token = str(value or "").strip()
    return token or "custom"


def _normalize_catalog_version(value: str | None, *, is_builtin: bool) -> str:
    token = str(value or "").strip()
    if token:
        return token
    return "builtin-v1" if is_builtin else "plugin-v1"


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = []
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _normalize_model_family_list(value: Any) -> list[str]:
    return [item.lower() for item in _normalize_text_list(value)]


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _coerce_starter_pack_payload(
    payload: StarterPack | dict[str, Any],
    *,
    source_module: str | None,
    catalog_version: str | None,
    is_builtin: bool,
) -> StarterPack:
    if isinstance(payload, StarterPack):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = dict(payload)
    else:
        raise ValueError("Starter pack payload must be a dict or StarterPack instance.")

    pack_id = _normalize_pack_id(data.get("id"))
    if not pack_id:
        raise ValueError("Starter pack id is required.")

    normalized_is_builtin = bool(data.get("is_builtin", is_builtin))
    normalized_source = _normalize_source_module(
        data.get("catalog_source")
        or source_module
        or _STARTER_PACK_REGISTRATION_SOURCE_CONTEXT
        or ("builtin" if normalized_is_builtin else "custom")
    )
    normalized_version = _normalize_catalog_version(
        data.get("catalog_version") or catalog_version,
        is_builtin=normalized_is_builtin,
    )

    normalized: dict[str, Any] = {
        "id": pack_id,
        "display_name": str(data.get("display_name") or "").strip() or pack_id.replace("_", " ").title(),
        "description": str(data.get("description") or "").strip(),
        "domain": str(data.get("domain") or "general").strip().lower() or "general",
        "recommended_model_families": _normalize_model_family_list(
            data.get("recommended_model_families")
        ),
        "recommended_models": _normalize_text_list(data.get("recommended_models")),
        "default_base_model_name": str(data.get("default_base_model_name") or "").strip() or None,
        "adapter_task_defaults": _normalize_mapping(data.get("adapter_task_defaults")),
        "evaluation_gate_defaults": _normalize_mapping(data.get("evaluation_gate_defaults")),
        "safety_compliance_reminders": _normalize_text_list(data.get("safety_compliance_reminders")),
        "target_profile_default": str(data.get("target_profile_default") or "vllm_server").strip() or "vllm_server",
        "catalog_source": normalized_source,
        "catalog_version": normalized_version,
        "is_builtin": normalized_is_builtin,
    }

    for key, value in data.items():
        if key not in normalized:
            normalized[key] = value
    return StarterPack(**normalized)


def _register_starter_pack(pack: StarterPack) -> None:
    pack_id = _normalize_pack_id(pack.id)
    if not pack_id:
        raise ValueError("Starter pack id is required.")

    if pack_id not in _STARTER_PACK_ORDER:
        _STARTER_PACK_ORDER.append(pack_id)
    _STARTER_PACK_REGISTRY[pack_id] = pack


def register_starter_pack(
    payload: StarterPack | dict[str, Any],
    *,
    source_module: str | None = None,
    catalog_version: str | None = None,
    is_builtin: bool = False,
) -> None:
    pack = _coerce_starter_pack_payload(
        payload,
        source_module=source_module,
        catalog_version=catalog_version,
        is_builtin=is_builtin,
    )
    with _STARTER_PACK_LOCK:
        _register_starter_pack(pack)


def register_starter_packs(
    payloads: list[StarterPack | dict[str, Any]],
    *,
    source_module: str | None = None,
    catalog_version: str | None = None,
    is_builtin: bool = False,
) -> int:
    count = 0
    for payload in list(payloads or []):
        register_starter_pack(
            payload,
            source_module=source_module,
            catalog_version=catalog_version,
            is_builtin=is_builtin,
        )
        count += 1
    return count


def _iter_starter_pack_payloads(raw_payload: Any) -> list[StarterPack | dict[str, Any]]:
    if isinstance(raw_payload, dict):
        rows: list[StarterPack | dict[str, Any]] = []
        for key, value in raw_payload.items():
            if isinstance(value, dict) and not str(value.get("id") or "").strip():
                value = {**value, "id": str(key)}
            rows.append(value)
        return rows
    if isinstance(raw_payload, (list, tuple, set)):
        return [item for item in list(raw_payload)]
    return []


def _extract_module_starter_packs(module: Any) -> list[StarterPack | dict[str, Any]]:
    if hasattr(module, "get_starter_packs") and callable(module.get_starter_packs):
        payload = module.get_starter_packs()
        return _iter_starter_pack_payloads(payload)
    return _iter_starter_pack_payloads(getattr(module, "STARTER_PACKS", []))


def _call_register_starter_packs_fn(module: Any, module_name: str) -> int:
    register_fn = getattr(module, "register_starter_packs", None)
    if register_fn is None:
        return 0
    if not callable(register_fn):
        raise ValueError(
            f"Starter-pack plugin module '{module_name}' has non-callable register_starter_packs."
        )

    count = 0

    def register(
        pack_payload: StarterPack | dict[str, Any],
        *,
        catalog_version: str | None = None,
    ) -> None:
        nonlocal count
        register_starter_pack(
            pack_payload,
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


def load_starter_pack_plugins(
    module_paths: list[str],
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    requested_modules = [str(item).strip() for item in list(module_paths or []) if str(item).strip()]
    loaded_modules: list[str] = []
    skipped_modules: list[str] = []
    errors: dict[str, str] = {}

    global _STARTER_PACK_REGISTRATION_SOURCE_CONTEXT
    with _STARTER_PACK_LOCK:
        for module_name in requested_modules:
            if module_name in _STARTER_PACK_PLUGIN_MODULES_LOADED and not force_reload:
                skipped_modules.append(module_name)
                continue

            try:
                module = importlib.import_module(module_name)
                if force_reload:
                    module = importlib.reload(module)

                registered_count = 0
                prior_context = _STARTER_PACK_REGISTRATION_SOURCE_CONTEXT
                _STARTER_PACK_REGISTRATION_SOURCE_CONTEXT = module_name
                try:
                    registered_count += _call_register_starter_packs_fn(module, module_name)
                    for payload in _extract_module_starter_packs(module):
                        register_starter_pack(
                            payload,
                            source_module=module_name,
                            catalog_version=None,
                            is_builtin=False,
                        )
                        registered_count += 1
                finally:
                    _STARTER_PACK_REGISTRATION_SOURCE_CONTEXT = prior_context

                if registered_count == 0:
                    raise ValueError(
                        "No starter packs registered. Provide register_starter_packs(...) or STARTER_PACKS/get_starter_packs()."
                    )

                _STARTER_PACK_PLUGIN_MODULES_LOADED.add(module_name)
                _STARTER_PACK_PLUGIN_LOAD_ERRORS.pop(module_name, None)
                loaded_modules.append(module_name)
            except Exception as exc:  # noqa: PERF203
                message = str(exc)
                _STARTER_PACK_PLUGIN_LOAD_ERRORS[module_name] = message
                errors[module_name] = message

    return {
        "requested_modules": requested_modules,
        "loaded_modules": sorted(set(loaded_modules)),
        "skipped_modules": sorted(set(skipped_modules)),
        "errors": errors,
    }


def load_starter_pack_plugins_from_settings(*, force_reload: bool = False) -> dict[str, Any]:
    modules = [str(item).strip() for item in list(settings.STARTER_PACK_PLUGIN_MODULES or []) if str(item).strip()]
    if not modules:
        return {
            "requested_modules": [],
            "loaded_modules": [],
            "skipped_modules": [],
            "errors": {},
        }
    return load_starter_pack_plugins(modules, force_reload=force_reload)


def clear_starter_pack_plugins() -> None:
    global _STARTER_PACKS_INITIALIZED
    with _STARTER_PACK_LOCK:
        _STARTER_PACK_REGISTRY.clear()
        _STARTER_PACK_ORDER.clear()
        _STARTER_PACK_PLUGIN_MODULES_LOADED.clear()
        _STARTER_PACK_PLUGIN_LOAD_ERRORS.clear()
        _STARTER_PACKS_INITIALIZED = False


def _starter_pack_catalog_metadata() -> dict[str, Any]:
    return {
        "catalog_version": STARTER_PACK_CATALOG_VERSION,
        "pack_count": len(_STARTER_PACK_ORDER),
        "loaded_plugin_modules": sorted(_STARTER_PACK_PLUGIN_MODULES_LOADED),
        "plugin_load_errors": copy.deepcopy(_STARTER_PACK_PLUGIN_LOAD_ERRORS),
        "has_plugin_packs": any(
            not bool(_STARTER_PACK_REGISTRY.get(pack_id).is_builtin)
            for pack_id in _STARTER_PACK_ORDER
            if pack_id in _STARTER_PACK_REGISTRY
        ),
    }


def starter_pack_plugin_status() -> dict[str, Any]:
    _ensure_starter_packs_loaded()
    with _STARTER_PACK_LOCK:
        return {
            "requested_modules": [
                str(item).strip()
                for item in list(settings.STARTER_PACK_PLUGIN_MODULES or [])
                if str(item).strip()
            ],
            "loaded_modules": sorted(_STARTER_PACK_PLUGIN_MODULES_LOADED),
            "failed_modules": copy.deepcopy(_STARTER_PACK_PLUGIN_LOAD_ERRORS),
            "registered_pack_count": len(_STARTER_PACK_ORDER),
        }


def _register_builtin_starter_packs() -> None:
    for payload in _BUILTIN_STARTER_PACKS:
        pack = _coerce_starter_pack_payload(
            payload,
            source_module="builtin",
            catalog_version="builtin-v1",
            is_builtin=True,
        )
        _register_starter_pack(pack)


def _ensure_starter_packs_loaded() -> None:
    global _STARTER_PACKS_INITIALIZED
    if _STARTER_PACKS_INITIALIZED:
        return

    with _STARTER_PACK_LOCK:
        if _STARTER_PACKS_INITIALIZED:
            return
        _STARTER_PACK_REGISTRY.clear()
        _STARTER_PACK_ORDER.clear()
        _STARTER_PACK_PLUGIN_MODULES_LOADED.clear()
        _STARTER_PACK_PLUGIN_LOAD_ERRORS.clear()
        _register_builtin_starter_packs()
        load_starter_pack_plugins_from_settings(force_reload=False)
        _STARTER_PACKS_INITIALIZED = True


def list_starter_packs() -> list[dict[str, Any]]:
    _ensure_starter_packs_loaded()
    with _STARTER_PACK_LOCK:
        return [
            _STARTER_PACK_REGISTRY[pack_id].model_dump()
            for pack_id in _STARTER_PACK_ORDER
            if pack_id in _STARTER_PACK_REGISTRY
        ]


def list_starter_pack_catalog() -> dict[str, Any]:
    _ensure_starter_packs_loaded()
    rows = list_starter_packs()
    with _STARTER_PACK_LOCK:
        return {
            **_starter_pack_catalog_metadata(),
            "starter_packs": rows,
        }


def get_starter_pack_by_id(starter_pack_id: str | None) -> dict[str, Any] | None:
    token = _normalize_pack_id(starter_pack_id)
    if not token:
        return None

    _ensure_starter_packs_loaded()
    with _STARTER_PACK_LOCK:
        pack = _STARTER_PACK_REGISTRY.get(token)
        return pack.model_dump() if pack is not None else None

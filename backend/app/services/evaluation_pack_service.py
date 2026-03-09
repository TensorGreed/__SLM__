"""Evaluation pack catalog + task-aware auto-gate evaluation helpers.

Evaluation Contract v2 adds task-aware specs:
- per-task gate lists
- per-task required metric schemas
- fallback task-profile routing
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment
from app.models.project import Project
from app.services.data_adapter_service import normalize_task_profile, normalize_training_task_type
from app.services.domain_runtime_service import resolve_project_domain_runtime

DEFAULT_EVALUATION_PACK_ID = "evalpack.general.default"
DOMAIN_PROFILE_EVAL_PACK_ID = "evalpack.domain-profile"
EVALUATION_PACK_CONTRACT_VERSION = "slm.evaluation-pack/v2"
DEFAULT_TASK_PROFILE = "instruction_sft"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique_tokens(items: list[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        token = _normalize_token(str(item))
        if not token:
            continue
        if token not in out:
            out.append(token)
    return out


_BASE_METRIC_SCHEMA: dict[str, dict[str, Any]] = {
    "exact_match": {
        "description": "Exact-match quality score.",
        "expected_range": [0.0, 1.0],
        "aliases": ["exact_match"],
    },
    "f1": {
        "description": "F1 overlap score.",
        "expected_range": [0.0, 1.0],
        "aliases": ["f1"],
    },
    "llm_judge_pass_rate": {
        "description": "Pass rate from LLM judge evaluation.",
        "expected_range": [0.0, 1.0],
        "aliases": ["llm_judge_pass_rate", "llm_judge.pass_rate", "llm_judge"],
    },
    "safety_pass_rate": {
        "description": "Safety pass rate across policy checks.",
        "expected_range": [0.0, 1.0],
        "aliases": ["safety_pass_rate", "safety.pass_rate", "safety"],
    },
    "accuracy": {
        "description": "Classification accuracy.",
        "expected_range": [0.0, 1.0],
        "aliases": ["accuracy", "classification.accuracy", "exact_match"],
    },
    "macro_f1": {
        "description": "Macro-averaged F1 for class-balanced scoring.",
        "expected_range": [0.0, 1.0],
        "aliases": ["macro_f1", "classification.macro_f1", "classification.f1", "f1"],
    },
    "tool_success_rate": {
        "description": "Successful tool/function execution rate.",
        "expected_range": [0.0, 1.0],
        "aliases": ["tool_success_rate", "tool_calling.pass_rate", "llm_judge_pass_rate"],
    },
    "groundedness": {
        "description": "Groundedness / citation alignment score.",
        "expected_range": [0.0, 1.0],
        "aliases": ["groundedness", "rag_qa.groundedness", "llm_judge_pass_rate", "f1"],
    },
}


def _normalize_metric_schema_map(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw_metric_id, raw_spec in payload.items():
        metric_id = _normalize_token(str(raw_metric_id))
        if not metric_id:
            continue
        spec = dict(raw_spec) if isinstance(raw_spec, dict) else {}
        aliases = _unique_tokens(list(spec.get("aliases") or []))
        if metric_id not in aliases:
            aliases.insert(0, metric_id)
        spec["aliases"] = aliases
        range_payload = spec.get("expected_range")
        if isinstance(range_payload, list) and len(range_payload) == 2:
            left = _to_float(range_payload[0])
            right = _to_float(range_payload[1])
            if left is not None and right is not None:
                spec["expected_range"] = [left, right]
        out[metric_id] = spec
    return out


def _metric_schema_for_metric_ids(metric_ids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for token in _unique_tokens(metric_ids):
        if token in _BASE_METRIC_SCHEMA:
            out[token] = _deepcopy(_BASE_METRIC_SCHEMA[token])
            continue
        out[token] = {
            "description": f"Metric '{token}' from evaluation outputs.",
            "expected_range": [0.0, 1.0],
            "aliases": [token],
        }
    return out


def _build_task_spec(
    *,
    task_profile: str,
    display_name: str | None = None,
    description: str | None = None,
    required_metric_ids: list[str] | None = None,
    gates: list[dict[str, Any]] | None = None,
    metric_schema: dict[str, Any] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    resolved_profile = normalize_task_profile(task_profile, default=DEFAULT_TASK_PROFILE)
    if not resolved_profile:
        resolved_profile = DEFAULT_TASK_PROFILE

    normalized_gates: list[dict[str, Any]] = []
    for gate in list(gates or []):
        if not isinstance(gate, dict):
            continue
        metric_id = _normalize_token(str(gate.get("metric_id") or ""))
        if not metric_id:
            continue
        operator = str(gate.get("operator") or "gte").strip().lower()
        if operator not in {"gte", "lte"}:
            operator = "gte"
        normalized_gate = {
            "gate_id": str(gate.get("gate_id") or f"min_{metric_id}").strip() or f"min_{metric_id}",
            "metric_id": metric_id,
            "operator": operator,
            "threshold": _to_float(gate.get("threshold")),
            "required": bool(gate.get("required", True)),
        }
        if source:
            normalized_gate["source"] = source
        if "weight" in gate:
            weight = _to_float(gate.get("weight"))
            if weight is not None:
                normalized_gate["weight"] = weight
        normalized_gates.append(normalized_gate)

    required_ids = _unique_tokens(list(required_metric_ids or []))
    required_ids.extend(
        [
            str(gate.get("metric_id") or "")
            for gate in normalized_gates
            if bool(gate.get("required"))
        ]
    )
    required_ids = _unique_tokens(required_ids)

    tracked_metric_ids = _unique_tokens(
        required_ids
        + [str(gate.get("metric_id") or "") for gate in normalized_gates]
        + list((metric_schema or {}).keys())
    )
    resolved_metric_schema = _metric_schema_for_metric_ids(tracked_metric_ids)
    overrides = _normalize_metric_schema_map(metric_schema)
    for metric_id, override in overrides.items():
        base = dict(resolved_metric_schema.get(metric_id) or {})
        base.update(override)
        aliases = _unique_tokens(list(base.get("aliases") or []))
        if metric_id not in aliases:
            aliases.insert(0, metric_id)
        base["aliases"] = aliases
        resolved_metric_schema[metric_id] = base

    return {
        "task_profile": resolved_profile,
        "display_name": str(display_name or resolved_profile.replace("_", " ").title()),
        "description": str(description or "").strip(),
        "required_metric_ids": required_ids,
        "metric_schema": resolved_metric_schema,
        "gates": normalized_gates,
    }


def _build_pack_contract(pack: dict[str, Any]) -> dict[str, Any]:
    task_specs_payload = pack.get("task_specs")
    task_specs: list[dict[str, Any]] = []

    if isinstance(task_specs_payload, list):
        for item in task_specs_payload:
            if not isinstance(item, dict):
                continue
            task_specs.append(
                _build_task_spec(
                    task_profile=str(item.get("task_profile") or item.get("task_id") or DEFAULT_TASK_PROFILE),
                    display_name=str(item.get("display_name") or ""),
                    description=str(item.get("description") or ""),
                    required_metric_ids=list(item.get("required_metric_ids") or []),
                    gates=list(item.get("gates") or []),
                    metric_schema=dict(item.get("metric_schema") or {}),
                    source=str(item.get("source") or "evaluation_pack_v2"),
                )
            )

    if not task_specs:
        task_specs = [
            _build_task_spec(
                task_profile=str(pack.get("default_task_profile") or DEFAULT_TASK_PROFILE),
                display_name=str(pack.get("display_name") or "Default"),
                description="Legacy gate list normalized to Evaluation Contract v2.",
                required_metric_ids=[],
                gates=list(pack.get("gates") or []),
                metric_schema={},
                source="legacy_pack",
            )
        ]

    task_profiles = _unique_tokens([spec.get("task_profile") for spec in task_specs])
    default_task_profile = normalize_task_profile(
        str(pack.get("default_task_profile") or ""),
        default="",
    )
    if not default_task_profile or default_task_profile not in task_profiles:
        default_task_profile = task_profiles[0] if task_profiles else DEFAULT_TASK_PROFILE

    spec_by_profile = {
        str(spec.get("task_profile") or ""): spec
        for spec in task_specs
        if str(spec.get("task_profile") or "").strip()
    }
    default_spec = spec_by_profile.get(default_task_profile) or (task_specs[0] if task_specs else {})
    default_gates = list(default_spec.get("gates") or [])

    payload = {
        "pack_id": str(pack.get("pack_id") or ""),
        "display_name": str(pack.get("display_name") or ""),
        "description": str(pack.get("description") or ""),
        "version": str(pack.get("version") or "1.0.0"),
        "owner": str(pack.get("owner") or "platform"),
        "tags": [str(item) for item in list(pack.get("tags") or []) if str(item).strip()],
        "contract_version": str(pack.get("contract_version") or EVALUATION_PACK_CONTRACT_VERSION),
        "default_task_profile": default_task_profile,
        "task_profiles": task_profiles,
        "task_specs": task_specs,
        # Backward-compatible top-level gates (default task profile).
        "gates": default_gates,
    }
    if "derived_from_profile_id" in pack:
        payload["derived_from_profile_id"] = pack.get("derived_from_profile_id")
    return payload


def _pack_summary(pack: dict[str, Any], *, include_gates: bool) -> dict[str, Any]:
    resolved = _build_pack_contract(pack)
    payload: dict[str, Any] = {
        "pack_id": str(resolved.get("pack_id", "")),
        "display_name": str(resolved.get("display_name", "")),
        "description": str(resolved.get("description", "")),
        "version": str(resolved.get("version", "")),
        "owner": str(resolved.get("owner", "")),
        "tags": [str(item) for item in list(resolved.get("tags") or []) if str(item).strip()],
        "contract_version": str(resolved.get("contract_version") or EVALUATION_PACK_CONTRACT_VERSION),
        "default_task_profile": str(resolved.get("default_task_profile") or DEFAULT_TASK_PROFILE),
        "task_profiles": list(resolved.get("task_profiles") or []),
        "task_spec_count": len(list(resolved.get("task_specs") or [])),
        "gate_count": len(list(resolved.get("gates") or [])),
    }
    if include_gates:
        payload["gates"] = _deepcopy(list(resolved.get("gates") or []))
        payload["task_specs"] = _deepcopy(list(resolved.get("task_specs") or []))
    return payload


def _gate(gate_id: str, metric_id: str, threshold: float, *, required: bool = True, operator: str = "gte") -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "metric_id": metric_id,
        "operator": operator,
        "threshold": threshold,
        "required": required,
    }


def _default_task_specs_for_pack(kind: str) -> list[dict[str, Any]]:
    if kind == "strict":
        return [
            _build_task_spec(
                task_profile="instruction_sft",
                display_name="Instruction / QA",
                required_metric_ids=["exact_match", "f1", "llm_judge_pass_rate", "safety_pass_rate"],
                gates=[
                    _gate("min_exact_match", "exact_match", 0.65, required=True),
                    _gate("min_f1", "f1", 0.72, required=True),
                    _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.8, required=True),
                    _gate("min_safety_pass_rate", "safety_pass_rate", 0.93, required=True),
                ],
            ),
            _build_task_spec(
                task_profile="classification",
                display_name="Classification",
                required_metric_ids=["accuracy", "macro_f1", "safety_pass_rate"],
                gates=[
                    _gate("min_accuracy", "accuracy", 0.7, required=True),
                    _gate("min_macro_f1", "macro_f1", 0.7, required=True),
                    _gate("min_safety_pass_rate", "safety_pass_rate", 0.93, required=True),
                ],
            ),
            _build_task_spec(
                task_profile="rag_qa",
                display_name="RAG QA",
                required_metric_ids=["f1", "groundedness", "safety_pass_rate"],
                gates=[
                    _gate("min_f1", "f1", 0.72, required=True),
                    _gate("min_groundedness", "groundedness", 0.82, required=True),
                    _gate("min_safety_pass_rate", "safety_pass_rate", 0.93, required=True),
                ],
            ),
            _build_task_spec(
                task_profile="tool_calling",
                display_name="Tool Calling",
                required_metric_ids=["tool_success_rate", "safety_pass_rate"],
                gates=[
                    _gate("min_tool_success_rate", "tool_success_rate", 0.78, required=True),
                    _gate("min_safety_pass_rate", "safety_pass_rate", 0.93, required=True),
                ],
            ),
        ]
    if kind == "fast":
        return [
            _build_task_spec(
                task_profile="instruction_sft",
                display_name="Instruction / QA",
                required_metric_ids=["exact_match"],
                gates=[
                    _gate("min_exact_match", "exact_match", 0.35, required=True),
                    _gate("min_f1", "f1", 0.45, required=False),
                ],
            ),
            _build_task_spec(
                task_profile="classification",
                display_name="Classification",
                required_metric_ids=["accuracy"],
                gates=[
                    _gate("min_accuracy", "accuracy", 0.4, required=True),
                    _gate("min_macro_f1", "macro_f1", 0.45, required=False),
                ],
            ),
            _build_task_spec(
                task_profile="chat_sft",
                display_name="Chat",
                required_metric_ids=[],
                gates=[
                    _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.55, required=False),
                ],
            ),
            _build_task_spec(
                task_profile="tool_calling",
                display_name="Tool Calling",
                required_metric_ids=[],
                gates=[
                    _gate("min_tool_success_rate", "tool_success_rate", 0.45, required=False),
                ],
            ),
        ]
    return [
        _build_task_spec(
            task_profile="instruction_sft",
            display_name="Instruction / QA",
            required_metric_ids=["exact_match", "f1"],
            gates=[
                _gate("min_exact_match", "exact_match", 0.5, required=True),
                _gate("min_f1", "f1", 0.6, required=True),
                _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.72, required=False),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="qa",
            display_name="QA",
            required_metric_ids=["exact_match", "f1"],
            gates=[
                _gate("min_exact_match", "exact_match", 0.55, required=True),
                _gate("min_f1", "f1", 0.65, required=True),
                _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.72, required=False),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="chat_sft",
            display_name="Chat",
            required_metric_ids=["llm_judge_pass_rate"],
            gates=[
                _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.72, required=True),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="classification",
            display_name="Classification",
            required_metric_ids=["accuracy", "macro_f1"],
            gates=[
                _gate("min_accuracy", "accuracy", 0.55, required=True),
                _gate("min_macro_f1", "macro_f1", 0.55, required=True),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="seq2seq",
            display_name="Seq2Seq",
            required_metric_ids=["f1"],
            gates=[
                _gate("min_f1", "f1", 0.58, required=True),
                _gate("min_exact_match", "exact_match", 0.4, required=False),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="rag_qa",
            display_name="RAG QA",
            required_metric_ids=["f1", "groundedness"],
            gates=[
                _gate("min_f1", "f1", 0.62, required=True),
                _gate("min_groundedness", "groundedness", 0.7, required=True),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="structured_extraction",
            display_name="Structured Extraction",
            required_metric_ids=["exact_match", "f1"],
            gates=[
                _gate("min_exact_match", "exact_match", 0.45, required=True),
                _gate("min_f1", "f1", 0.6, required=True),
            ],
        ),
        _build_task_spec(
            task_profile="tool_calling",
            display_name="Tool Calling",
            required_metric_ids=["tool_success_rate"],
            gates=[
                _gate("min_tool_success_rate", "tool_success_rate", 0.6, required=True),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="preference",
            display_name="Preference / Alignment",
            required_metric_ids=["llm_judge_pass_rate"],
            gates=[
                _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.75, required=True),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
    ]


_BUILTIN_EVALUATION_PACKS: list[dict[str, Any]] = [
    {
        "pack_id": "evalpack.general.default",
        "display_name": "General Default Gates",
        "description": "Balanced domain-agnostic quality gates for most SLM projects.",
        "version": "2.0.0",
        "owner": "platform",
        "tags": ["general", "balanced", "default", "task-aware"],
        "contract_version": EVALUATION_PACK_CONTRACT_VERSION,
        "default_task_profile": "instruction_sft",
        "task_specs": _default_task_specs_for_pack("general"),
    },
    {
        "pack_id": "evalpack.quality.strict",
        "display_name": "Quality Strict Gates",
        "description": "Higher confidence gate profile for release-candidate promotion.",
        "version": "2.0.0",
        "owner": "platform",
        "tags": ["strict", "quality", "release", "task-aware"],
        "contract_version": EVALUATION_PACK_CONTRACT_VERSION,
        "default_task_profile": "instruction_sft",
        "task_specs": _default_task_specs_for_pack("strict"),
    },
    {
        "pack_id": "evalpack.fast.iteration",
        "display_name": "Fast Iteration Gates",
        "description": "Lightweight development-time gates for rapid experimentation.",
        "version": "2.0.0",
        "owner": "platform",
        "tags": ["fast", "iteration", "dev", "task-aware"],
        "contract_version": EVALUATION_PACK_CONTRACT_VERSION,
        "default_task_profile": "instruction_sft",
        "task_specs": _default_task_specs_for_pack("fast"),
    },
]


def list_evaluation_packs(*, include_gates: bool = False) -> list[dict[str, Any]]:
    """List built-in evaluation pack metadata."""
    return [_pack_summary(item, include_gates=include_gates) for item in _BUILTIN_EVALUATION_PACKS]


def get_evaluation_pack(pack_id: str) -> dict[str, Any] | None:
    """Lookup built-in evaluation pack by id."""
    token = _normalize_token(pack_id)
    if not token:
        return None
    for pack in _BUILTIN_EVALUATION_PACKS:
        if _normalize_token(str(pack.get("pack_id"))) == token:
            return _build_pack_contract(pack)
    return None


def normalize_evaluation_pack_id(value: str | None) -> str | None:
    """Normalize a persisted/requested pack id."""
    token = str(value or "").strip().lower()
    return token if token else None


def is_supported_evaluation_pack_id(value: str | None) -> bool:
    token = normalize_evaluation_pack_id(value)
    if token is None:
        return False
    if token == DOMAIN_PROFILE_EVAL_PACK_ID:
        return True
    return get_evaluation_pack(token) is not None


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


def _task_profile_from_domain_task_id(task_id: str | None) -> str | None:
    token = _normalize_token(task_id)
    if not token:
        return None
    mapped = _DOMAIN_TASK_TO_PROFILE.get(token, token)
    resolved = normalize_task_profile(mapped, default="")
    return resolved or None


def _derive_default_task_profile_from_contract(contract: dict[str, Any]) -> str:
    tasks = contract.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_profile = _task_profile_from_domain_task_id(str(task.get("task_id") or ""))
            if task_profile:
                return task_profile
    return DEFAULT_TASK_PROFILE


def _build_gates_from_metric_specs(
    metrics: list[dict[str, Any]],
    *,
    required_metric_ids: list[str],
    source: str,
) -> list[dict[str, Any]]:
    required_set = set(_unique_tokens(required_metric_ids))
    gates: list[dict[str, Any]] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        metric_id = _normalize_token(str(metric.get("metric_id") or ""))
        threshold = _to_float(metric.get("threshold"))
        if not metric_id or threshold is None:
            continue
        operator = str(metric.get("operator") or "gte").strip().lower()
        if operator not in {"gte", "lte"}:
            operator = "gte"
        gates.append(
            {
                "gate_id": str(metric.get("gate_id") or f"min_{metric_id}").strip() or f"min_{metric_id}",
                "metric_id": metric_id,
                "operator": operator,
                "threshold": threshold,
                "required": metric_id in required_set,
                "source": source,
                "weight": _to_float(metric.get("weight")),
            }
        )
    return gates


def _domain_profile_pack_from_contract(contract: dict | None) -> dict[str, Any] | None:
    if not isinstance(contract, dict):
        return None

    evaluation_cfg = contract.get("evaluation")
    if not isinstance(evaluation_cfg, dict):
        return None

    default_task_profile = normalize_task_profile(
        str(evaluation_cfg.get("default_task_profile") or _derive_default_task_profile_from_contract(contract)),
        default=DEFAULT_TASK_PROFILE,
    )

    task_specs: list[dict[str, Any]] = []
    raw_task_specs = evaluation_cfg.get("task_specs")
    if isinstance(raw_task_specs, list):
        for raw_spec in raw_task_specs:
            if not isinstance(raw_spec, dict):
                continue
            task_profile = normalize_task_profile(
                str(raw_spec.get("task_profile") or raw_spec.get("task_id") or default_task_profile),
                default=default_task_profile,
            )
            metric_rows = list(raw_spec.get("metrics") or [])
            required_metric_ids = _unique_tokens(list(raw_spec.get("required_metrics_for_promotion") or []))
            raw_gates = raw_spec.get("gates")
            gates: list[dict[str, Any]]
            if isinstance(raw_gates, list) and raw_gates:
                gates = [dict(item) for item in raw_gates if isinstance(item, dict)]
            else:
                gates = _build_gates_from_metric_specs(
                    [item for item in metric_rows if isinstance(item, dict)],
                    required_metric_ids=required_metric_ids,
                    source="domain_profile_contract",
                )
            task_specs.append(
                _build_task_spec(
                    task_profile=task_profile,
                    display_name=str(raw_spec.get("display_name") or ""),
                    description=str(raw_spec.get("description") or ""),
                    required_metric_ids=required_metric_ids,
                    gates=gates,
                    metric_schema=dict(raw_spec.get("metric_schema") or {}),
                    source="domain_profile_contract",
                )
            )

    if not task_specs:
        required_metric_ids = _unique_tokens(list(evaluation_cfg.get("required_metrics_for_promotion") or []))
        metrics = [item for item in list(evaluation_cfg.get("metrics") or []) if isinstance(item, dict)]
        gates = _build_gates_from_metric_specs(
            metrics,
            required_metric_ids=required_metric_ids,
            source="domain_profile_contract",
        )
        if gates:
            task_specs = [
                _build_task_spec(
                    task_profile=default_task_profile,
                    display_name="Domain Profile Default Task",
                    description="Auto-derived from evaluation.metrics in domain profile contract.",
                    required_metric_ids=required_metric_ids,
                    gates=gates,
                    metric_schema=dict(evaluation_cfg.get("metric_schema") or {}),
                    source="domain_profile_contract",
                )
            ]

    if not task_specs:
        return None

    profile_id = str(contract.get("profile_id") or "").strip()
    display_profile = profile_id or "domain profile"
    return _build_pack_contract(
        {
            "pack_id": DOMAIN_PROFILE_EVAL_PACK_ID,
            "display_name": "Domain Profile Gates",
            "description": f"Auto-derived task-aware gates from effective domain profile contract ({display_profile}).",
            "version": str(contract.get("version") or "1.0.0"),
            "owner": str(contract.get("owner") or "domain-profile"),
            "tags": ["domain_profile", "auto", "task-aware"],
            "contract_version": EVALUATION_PACK_CONTRACT_VERSION,
            "default_task_profile": default_task_profile,
            "derived_from_profile_id": profile_id or None,
            "task_specs": task_specs,
        }
    )


async def _get_project(db: AsyncSession, project_id: int) -> Project | None:
    row = await db.execute(select(Project).where(Project.id == project_id))
    return row.scalar_one_or_none()


async def resolve_project_evaluation_pack(
    db: AsyncSession,
    project_id: int,
    *,
    preferred_pack_id: str | None = None,
) -> dict[str, Any]:
    """Resolve active pack for a project with deterministic fallback chain."""
    project = await _get_project(db, project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    runtime = await resolve_project_domain_runtime(db, project_id)
    effective_contract = runtime.get("effective_contract")
    dynamic_pack = _domain_profile_pack_from_contract(effective_contract)
    dynamic_available = dynamic_pack is not None

    configured = normalize_evaluation_pack_id(
        preferred_pack_id if preferred_pack_id is not None else project.evaluation_preferred_pack_id
    )

    warnings: list[str] = []
    active_pack: dict[str, Any] | None = None
    source = "default"

    if configured:
        if configured == DOMAIN_PROFILE_EVAL_PACK_ID:
            if dynamic_pack is not None:
                active_pack = dynamic_pack
                source = "project_domain_profile"
            else:
                warnings.append(
                    "Preferred pack is evalpack.domain-profile but effective domain contract has no thresholds; falling back."
                )
        else:
            selected = get_evaluation_pack(configured)
            if selected is not None:
                active_pack = selected
                source = "project"
            else:
                warnings.append(f"Preferred evaluation pack '{configured}' is not available; falling back.")

    if active_pack is None and dynamic_pack is not None:
        active_pack = dynamic_pack
        source = "domain_profile_default"

    if active_pack is None:
        active_pack = get_evaluation_pack(DEFAULT_EVALUATION_PACK_ID) or _build_pack_contract(_BUILTIN_EVALUATION_PACKS[0])
        source = "default"

    active_pack = _build_pack_contract(active_pack)
    return {
        "project_id": project_id,
        "preferred_pack_id": configured,
        "active_pack_id": str(active_pack.get("pack_id", "")),
        "source": source,
        "dynamic_pack_available": dynamic_available,
        "pack": active_pack,
        "warnings": warnings,
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
    }


async def _get_experiment_for_project(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> Experiment | None:
    row = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    return row.scalar_one_or_none()


async def _latest_eval_by_type(db: AsyncSession, experiment_id: int) -> dict[str, EvalResult]:
    rows = await db.execute(
        select(EvalResult)
        .where(EvalResult.experiment_id == experiment_id)
        .order_by(EvalResult.created_at.desc(), EvalResult.id.desc())
    )
    latest: dict[str, EvalResult] = {}
    for item in rows.scalars().all():
        eval_type = _normalize_token(item.eval_type)
        if eval_type and eval_type not in latest:
            latest[eval_type] = item
    return latest


def _set_metric_value(
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
    *,
    key: str,
    value: float | None,
    row: EvalResult,
    metric_key: str,
    overwrite: bool = False,
) -> None:
    normalized = key.strip().lower()
    if not normalized or value is None:
        return
    if not overwrite and normalized in values:
        return
    values[normalized] = float(value)
    sources[normalized] = {
        "eval_type": str(row.eval_type),
        "dataset_name": str(row.dataset_name),
        "eval_result_id": int(row.id),
        "metric_key": metric_key,
    }


def _build_metric_snapshot(
    latest_by_eval_type: dict[str, EvalResult],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    values: dict[str, float] = {}
    sources: dict[str, dict[str, Any]] = {}

    canonical_map = [
        ("exact_match", "exact_match", "exact_match"),
        ("f1", "f1", "f1"),
        ("llm_judge_pass_rate", "llm_judge", "pass_rate"),
        ("safety_pass_rate", "safety", "pass_rate"),
    ]
    for metric_id, eval_type, metric_key in canonical_map:
        row = latest_by_eval_type.get(eval_type)
        if row is None:
            continue
        payload = row.metrics if isinstance(row.metrics, dict) else {}
        value = _to_float(payload.get(metric_key))
        if value is None and metric_key == "pass_rate":
            value = _to_float(row.pass_rate)
        _set_metric_value(
            values,
            sources,
            key=metric_id,
            value=value,
            row=row,
            metric_key=metric_key,
            overwrite=True,
        )

    for eval_type, row in latest_by_eval_type.items():
        payload = row.metrics if isinstance(row.metrics, dict) else {}
        pass_rate = _to_float(payload.get("pass_rate"))
        if pass_rate is None:
            pass_rate = _to_float(row.pass_rate)
        _set_metric_value(
            values,
            sources,
            key=f"{eval_type}_pass_rate",
            value=pass_rate,
            row=row,
            metric_key="pass_rate",
            overwrite=False,
        )
        _set_metric_value(
            values,
            sources,
            key=f"{eval_type}.pass_rate",
            value=pass_rate,
            row=row,
            metric_key="pass_rate",
            overwrite=True,
        )

        for raw_key, raw_value in payload.items():
            value = _to_float(raw_value)
            if value is None:
                continue
            normalized_metric = _normalize_token(str(raw_key))
            if not normalized_metric:
                continue
            _set_metric_value(
                values,
                sources,
                key=normalized_metric,
                value=value,
                row=row,
                metric_key=str(raw_key),
                overwrite=False,
            )
            _set_metric_value(
                values,
                sources,
                key=f"{eval_type}.{normalized_metric}",
                value=value,
                row=row,
                metric_key=str(raw_key),
                overwrite=True,
            )

    return values, sources


def _metric_alias_candidates(metric_id: str, metric_schema: dict[str, Any] | None = None) -> list[str]:
    token = _normalize_token(metric_id)
    if not token:
        return []
    candidates = [token]
    if isinstance(metric_schema, dict):
        entry = metric_schema.get(token)
        if isinstance(entry, dict):
            aliases = _unique_tokens(list(entry.get("aliases") or []))
            candidates = _unique_tokens(aliases + candidates)
    expanded: list[str] = []
    for candidate in candidates:
        expanded.append(candidate)
        if candidate.endswith("_pass_rate"):
            expanded.append(f"{candidate[:-10]}.pass_rate")
        if candidate in {"exact_match", "f1"}:
            expanded.append(f"{candidate}.pass_rate")
    return _unique_tokens(expanded)


def _resolve_metric_value(
    metric_id: str,
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
    *,
    metric_schema: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any] | None, str | None]:
    candidates = _metric_alias_candidates(metric_id, metric_schema=metric_schema)
    if not candidates:
        return None, None, None

    for key in candidates:
        if key in values:
            return values[key], sources.get(key), key

    for candidate in candidates:
        suffix_hits = sorted([key for key in values.keys() if key.endswith(f".{candidate}")])
        if suffix_hits:
            winner = suffix_hits[0]
            return values[winner], sources.get(winner), winner
    return None, None, None


def _evaluate_gate(
    gate: dict[str, Any],
    *,
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
    metric_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_id = str(gate.get("gate_id") or "").strip() or "gate"
    metric_id = str(gate.get("metric_id") or "").strip()
    operator = str(gate.get("operator") or "gte").strip().lower()
    if operator not in {"gte", "lte"}:
        operator = "gte"
    threshold = _to_float(gate.get("threshold"))
    required = bool(gate.get("required", True))

    actual, source, resolved_metric_key = _resolve_metric_value(
        metric_id,
        values,
        sources,
        metric_schema=metric_schema,
    )
    if threshold is None:
        passed = True
        reason = "not_enforced"
    elif actual is None:
        passed = not required
        reason = "missing_metric_required" if required else "missing_metric_optional"
    elif operator == "lte":
        passed = actual <= threshold
        reason = "ok" if passed else "above_threshold"
    else:
        passed = actual >= threshold
        reason = "ok" if passed else "below_threshold"

    return {
        "gate_id": gate_id,
        "metric_id": _normalize_token(metric_id),
        "resolved_metric_key": resolved_metric_key,
        "operator": operator,
        "threshold": threshold,
        "required": required,
        "actual": round(float(actual), 6) if actual is not None else None,
        "passed": passed,
        "reason": reason,
        "source": source or {},
    }


def _task_profile_from_training_task_type(task_type: str | None) -> str:
    normalized = normalize_training_task_type(task_type, default="causal_lm")
    if normalized == "classification":
        return "classification"
    if normalized == "seq2seq":
        return "seq2seq"
    if normalized in {"dpo", "orpo"}:
        return "preference"
    return DEFAULT_TASK_PROFILE


def _resolve_task_profile_for_experiment(
    *,
    project: Project | None,
    experiment: Experiment,
    pack: dict[str, Any],
    requested_task_profile: str | None = None,
) -> tuple[str, str]:
    requested = normalize_task_profile(requested_task_profile, default="")
    if requested:
        return requested, "request"

    config = experiment.config if isinstance(experiment.config, dict) else {}
    exp_profile = normalize_task_profile(str(config.get("task_profile") or ""), default="")
    if exp_profile:
        return exp_profile, "experiment.config.task_profile"

    preset = project.dataset_adapter_preset if project and isinstance(project.dataset_adapter_preset, dict) else {}
    project_profile = normalize_task_profile(str(preset.get("task_profile") or ""), default="")
    if project_profile:
        return project_profile, "project.dataset_adapter_preset.task_profile"

    task_type = str(config.get("task_type") or "").strip()
    if task_type:
        return _task_profile_from_training_task_type(task_type), "experiment.config.task_type"

    pack_default = normalize_task_profile(str(pack.get("default_task_profile") or ""), default="")
    if pack_default:
        return pack_default, "pack.default_task_profile"
    return DEFAULT_TASK_PROFILE, "default"


def _task_profile_candidates(task_profile: str) -> list[str]:
    token = normalize_task_profile(task_profile, default=DEFAULT_TASK_PROFILE)
    fallback_map: dict[str, list[str]] = {
        "rag_qa": ["rag_qa", "qa", DEFAULT_TASK_PROFILE],
        "tool_calling": ["tool_calling", "chat_sft", DEFAULT_TASK_PROFILE],
        "structured_extraction": ["structured_extraction", "seq2seq", "classification", DEFAULT_TASK_PROFILE],
        "summarization": ["summarization", "seq2seq", DEFAULT_TASK_PROFILE],
        "chat_sft": ["chat_sft", DEFAULT_TASK_PROFILE],
        "qa": ["qa", DEFAULT_TASK_PROFILE],
        "classification": ["classification", DEFAULT_TASK_PROFILE],
        "seq2seq": ["seq2seq", DEFAULT_TASK_PROFILE],
        "preference": ["preference", DEFAULT_TASK_PROFILE],
        "language_modeling": ["language_modeling", DEFAULT_TASK_PROFILE],
    }
    return _unique_tokens(fallback_map.get(token, [token, DEFAULT_TASK_PROFILE]))


def _select_task_spec(pack: dict[str, Any], task_profile: str) -> tuple[dict[str, Any], str, bool]:
    task_specs = [item for item in list(pack.get("task_specs") or []) if isinstance(item, dict)]
    if not task_specs:
        spec = _build_task_spec(task_profile=DEFAULT_TASK_PROFILE, gates=list(pack.get("gates") or []))
        return spec, DEFAULT_TASK_PROFILE, True

    by_profile = {
        normalize_task_profile(str(item.get("task_profile") or ""), default=""): item
        for item in task_specs
    }
    for candidate in _task_profile_candidates(task_profile):
        if candidate in by_profile:
            return by_profile[candidate], candidate, candidate != normalize_task_profile(task_profile, default=DEFAULT_TASK_PROFILE)

    default_profile = normalize_task_profile(str(pack.get("default_task_profile") or ""), default="")
    if default_profile and default_profile in by_profile:
        return by_profile[default_profile], default_profile, True
    first = task_specs[0]
    first_profile = normalize_task_profile(str(first.get("task_profile") or ""), default=DEFAULT_TASK_PROFILE)
    return first, first_profile, True


def _evaluate_required_metric_schema(
    *,
    required_metric_ids: list[str],
    metric_schema: dict[str, Any],
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    for metric_id in _unique_tokens(required_metric_ids):
        actual, source, resolved_metric_key = _resolve_metric_value(
            metric_id,
            values,
            sources,
            metric_schema=metric_schema,
        )
        present = actual is not None
        if not present:
            missing.append(metric_id)
        checks.append(
            {
                "metric_id": metric_id,
                "resolved_metric_key": resolved_metric_key,
                "present": present,
                "actual": round(float(actual), 6) if actual is not None else None,
                "source": source or {},
            }
        )
    return checks, sorted(set(missing))


async def evaluate_experiment_auto_gates(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    pack_id: str | None = None,
    task_profile: str | None = None,
) -> dict[str, Any]:
    """Evaluate one experiment against active/requested evaluation pack gates."""
    exp = await _get_experiment_for_project(db, project_id, experiment_id)
    if exp is None:
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")
    project = await _get_project(db, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    pack_resolution = await resolve_project_evaluation_pack(
        db,
        project_id,
        preferred_pack_id=pack_id,
    )
    pack = _build_pack_contract(dict(pack_resolution.get("pack") or {}))
    resolved_task_profile, task_profile_source = _resolve_task_profile_for_experiment(
        project=project,
        experiment=exp,
        pack=pack,
        requested_task_profile=task_profile,
    )
    task_spec, selected_task_profile, fallback_used = _select_task_spec(pack, resolved_task_profile)
    gates = [item for item in list(task_spec.get("gates") or []) if isinstance(item, dict)]
    metric_schema = dict(task_spec.get("metric_schema") or {})

    latest_by_type = await _latest_eval_by_type(db, experiment_id)
    metric_values, metric_sources = _build_metric_snapshot(latest_by_type)
    checks = [
        _evaluate_gate(
            gate,
            values=metric_values,
            sources=metric_sources,
            metric_schema=metric_schema,
        )
        for gate in gates
    ]

    failed_required = [
        str(item.get("gate_id") or "")
        for item in checks
        if bool(item.get("required")) and not bool(item.get("passed"))
    ]
    missing_required_gate_metrics = [
        str(item.get("metric_id") or "")
        for item in checks
        if bool(item.get("required")) and str(item.get("reason") or "").startswith("missing_metric_")
    ]

    required_metric_ids = _unique_tokens(
        list(task_spec.get("required_metric_ids") or [])
        + [str(item.get("metric_id") or "") for item in checks if bool(item.get("required"))]
    )
    required_metric_checks, missing_required_schema_metrics = _evaluate_required_metric_schema(
        required_metric_ids=required_metric_ids,
        metric_schema=metric_schema,
        values=metric_values,
        sources=metric_sources,
    )
    missing_required_metrics = sorted(set(missing_required_gate_metrics + missing_required_schema_metrics))
    passed = not failed_required and not missing_required_metrics

    return {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "captured_at": _utcnow().isoformat(),
        "task_profile": resolved_task_profile,
        "task_profile_source": task_profile_source,
        "task_profile_selected": selected_task_profile,
        "task_profile_fallback_used": fallback_used,
        "task_spec": {
            "task_profile": str(task_spec.get("task_profile") or selected_task_profile),
            "display_name": str(task_spec.get("display_name") or ""),
            "description": str(task_spec.get("description") or ""),
            "required_metric_ids": required_metric_ids,
            "metric_schema": metric_schema,
            "gate_count": len(gates),
        },
        "pack": _pack_summary(pack, include_gates=True),
        "pack_resolution": {
            "preferred_pack_id": pack_resolution.get("preferred_pack_id"),
            "active_pack_id": pack_resolution.get("active_pack_id"),
            "source": pack_resolution.get("source"),
            "warnings": list(pack_resolution.get("warnings") or []),
            "dynamic_pack_available": bool(pack_resolution.get("dynamic_pack_available")),
            "domain_pack_applied": pack_resolution.get("domain_pack_applied"),
            "domain_profile_applied": pack_resolution.get("domain_profile_applied"),
        },
        "latest_eval_result_ids": {
            eval_type: int(item.id)
            for eval_type, item in latest_by_type.items()
        },
        "metrics": {
            key: round(value, 6)
            for key, value in sorted(metric_values.items())
        },
        "checks": checks,
        "required_metric_checks": required_metric_checks,
        "failed_gate_ids": [item for item in failed_required if item],
        "missing_required_schema_metrics": missing_required_schema_metrics,
        "missing_required_metrics": missing_required_metrics,
        "passed": passed,
    }

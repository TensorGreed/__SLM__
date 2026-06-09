"""Evaluation pack catalog + task-aware auto-gate evaluation helpers.

Evaluation Contract v2 adds task-aware specs:
- per-task gate lists
- per-task required metric schemas
- fallback task-profile routing
"""

from __future__ import annotations

import copy
import re as _re
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
    # Arc R-2 — RAG-protocol discipline metrics. The rag-grounded
    # eval handler (RAGHandler.score) already emits faithfulness_rate
    # and unsupported_token_rate_mean; appropriate_refusal_rate was
    # added in the same arc. format_consistency is a placeholder
    # for the Slice-2 implementation (clustering-based output-shape
    # consistency) — the gate uses ``required=False`` so the pack
    # stays usable until the metric is computed.
    "citation_rate": {
        "description": (
            "Share of predictions whose token overlap with the retrieved "
            "context meets the faithfulness threshold (0.7). Mirrors the "
            "rag-protocol training-time citation signal."
        ),
        "expected_range": [0.0, 1.0],
        "aliases": ["citation_rate", "faithfulness_rate", "rag_qa.faithfulness_rate"],
    },
    "hallucination_rate": {
        "description": (
            "Mean fraction of prediction tokens NOT supported by the "
            "retrieved context. Lower is better — gates use the ``lte`` "
            "operator against this metric."
        ),
        "expected_range": [0.0, 1.0],
        "aliases": [
            "hallucination_rate",
            "unsupported_token_rate_mean",
            "rag_qa.unsupported_rate",
        ],
    },
    "appropriate_refusal_rate": {
        "description": (
            "Fraction of rows where the model's refusal/answer behaviour "
            "matched the gold's. Rewards following the gold signal — "
            "refuse when the gold refuses, answer when the gold answers — "
            "NOT a blanket-refusal incentive."
        ),
        "expected_range": [0.0, 1.0],
        "aliases": [
            "appropriate_refusal_rate",
            "refusal_match_rate",
            "rag_qa.appropriate_refusal_rate",
        ],
    },
    "format_consistency": {
        "description": (
            "Consistency of response format across predictions (clustering / "
            "length-bucketed shape agreement). Optional gate; the metric is "
            "computed in a follow-on slice."
        ),
        "expected_range": [0.0, 1.0],
        "aliases": ["format_consistency", "rag_qa.format_consistency"],
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
    # Optional: a prior experiment id whose eval results are the "teacher"
    # baseline for the student-vs-teacher comparison (Track 1 Epic A slice 3).
    # Absent on built-in packs; a project/domain pack may declare it so the
    # comparison surface resolves a teacher without an explicit request param.
    if pack.get("teacher_baseline_run_id") is not None:
        payload["teacher_baseline_run_id"] = pack.get("teacher_baseline_run_id")
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
    if kind == "legal":
        return [
            _build_task_spec(
                task_profile="qa",
                display_name="Legal Clause QA",
                required_metric_ids=["f1", "exact_match", "precision", "recall"],
                gates=[
                    _gate("min_f1", "f1", 0.75, required=True),
                    _gate("min_recall", "recall", 0.8, required=True),
                ],
            ),
        ]
    if kind == "support":
        return [
            _build_task_spec(
                task_profile="instruction_sft",
                display_name="Customer Support",
                required_metric_ids=["accuracy", "f1", "hallucination_rate"],
                gates=[
                    _gate("min_accuracy", "accuracy", 0.8, required=True),
                    _gate("max_hallucination", "hallucination_rate", 0.05, operator="lte", required=True),
                ],
            ),
        ]
    if kind == "healthcare":
        return [
            _build_task_spec(
                task_profile="qa",
                display_name="Clinical Reasoning",
                required_metric_ids=["precision", "recall", "safety_pass_rate"],
                gates=[
                    _gate("min_precision", "precision", 0.85, required=True),
                    _gate("min_safety", "safety_pass_rate", 0.98, required=True),
                ],
            ),
        ]
    if kind == "finance":
        return [
            _build_task_spec(
                task_profile="qa",
                display_name="Financial Analysis",
                required_metric_ids=["arithmetic_accuracy", "f1"],
                gates=[
                    _gate("min_arithmetic_accuracy", "arithmetic_accuracy", 0.9, required=True),
                ],
            ),
        ]
    # Gate thresholds calibrated 2026-05-21 for the new platform
    # default (HuggingFaceTB/SmolLM2-135M-Instruct, a 135M-parameter
    # model). Previous thresholds were tuned to a phi-2-class baseline
    # (~2.7B params) and caused well-trained 135M first runs to fail
    # required gates on demo-sized data even when the model had
    # clearly learnt the task. Strict gates remain available via
    # `evalpack.quality.strict` for users targeting production
    # promotion. Roadmap context: ROADMAP-NEXT.md Theme 1 Epic 3.
    return [
        _build_task_spec(
            task_profile="instruction_sft",
            display_name="Instruction / QA",
            required_metric_ids=["exact_match", "f1"],
            gates=[
                _gate("min_exact_match", "exact_match", 0.4, required=True),
                _gate("min_f1", "f1", 0.5, required=True),
                _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.65, required=False),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="qa",
            display_name="QA",
            required_metric_ids=["exact_match", "f1"],
            gates=[
                _gate("min_exact_match", "exact_match", 0.45, required=True),
                _gate("min_f1", "f1", 0.55, required=True),
                _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.65, required=False),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="chat_sft",
            display_name="Chat",
            required_metric_ids=["llm_judge_pass_rate"],
            gates=[
                # LLM-judge pass rate is inherently noisy on small
                # models; surface it but don't gate on it by default.
                _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.6, required=False),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="classification",
            display_name="Classification",
            required_metric_ids=["accuracy", "macro_f1"],
            gates=[
                _gate("min_accuracy", "accuracy", 0.5, required=True),
                _gate("min_macro_f1", "macro_f1", 0.5, required=True),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="seq2seq",
            display_name="Seq2Seq",
            required_metric_ids=["f1"],
            gates=[
                _gate("min_f1", "f1", 0.5, required=True),
                _gate("min_exact_match", "exact_match", 0.35, required=False),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="rag_qa",
            display_name="RAG QA",
            required_metric_ids=["f1", "groundedness"],
            gates=[
                _gate("min_f1", "f1", 0.55, required=True),
                _gate("min_groundedness", "groundedness", 0.6, required=True),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="structured_extraction",
            display_name="Structured Extraction",
            required_metric_ids=["exact_match", "f1"],
            gates=[
                _gate("min_exact_match", "exact_match", 0.35, required=True),
                _gate("min_f1", "f1", 0.5, required=True),
            ],
        ),
        _build_task_spec(
            task_profile="tool_calling",
            display_name="Tool Calling",
            required_metric_ids=["tool_success_rate"],
            gates=[
                _gate("min_tool_success_rate", "tool_success_rate", 0.5, required=True),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
        _build_task_spec(
            task_profile="preference",
            display_name="Preference / Alignment",
            required_metric_ids=["llm_judge_pass_rate"],
            gates=[
                _gate("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.65, required=True),
                _gate("min_safety_pass_rate", "safety_pass_rate", 0.9, required=False),
            ],
        ),
    ]


_BUILTIN_EVALUATION_PACKS: list[dict[str, Any]] = [
    {
        "pack_id": "evalpack.general.default",
        "display_name": "General Default Gates",
        "description": (
            "Balanced domain-agnostic quality gates for most SLM projects. "
            "Calibrated for the platform's first-run default model "
            "(SmolLM2-135M-Instruct) on demo-sized datasets; promote to "
            "evalpack.quality.strict for release-candidate runs."
        ),
        "version": "2.1.0",
        "owner": "platform",
        "tags": ["general", "balanced", "default", "task-aware", "small-model"],
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
    {
        "pack_id": "evalpack.domain.legal",
        "display_name": "Legal Review Gates",
        "description": "High-precision gates for legal document analysis and reasoning.",
        "version": "1.0.0",
        "owner": "platform",
        "tags": ["legal", "high-precision"],
        "contract_version": EVALUATION_PACK_CONTRACT_VERSION,
        "default_task_profile": "qa",
        "task_specs": _default_task_specs_for_pack("legal"),
    },
    {
        "pack_id": "evalpack.domain.support",
        "display_name": "Customer Support Gates",
        "description": "Accuracy and hallucination gates for support interaction.",
        "version": "1.0.0",
        "owner": "platform",
        "tags": ["support", "customer-service"],
        "contract_version": EVALUATION_PACK_CONTRACT_VERSION,
        "default_task_profile": "instruction_sft",
        "task_specs": _default_task_specs_for_pack("support"),
    },
    {
        "pack_id": "evalpack.domain.healthcare",
        "display_name": "Healthcare & Clinical Gates",
        "description": "Strict safety and precision gates for healthcare domains.",
        "version": "1.0.0",
        "owner": "platform",
        "tags": ["healthcare", "clinical", "safety"],
        "contract_version": EVALUATION_PACK_CONTRACT_VERSION,
        "default_task_profile": "qa",
        "task_specs": _default_task_specs_for_pack("healthcare"),
    },
    {
        "pack_id": "evalpack.domain.finance",
        "display_name": "Financial Reasoning Gates",
        "description": "Arithmetic and extraction accuracy gates for finance.",
        "version": "1.0.0",
        "owner": "platform",
        "tags": ["finance", "arithmetic"],
        "contract_version": EVALUATION_PACK_CONTRACT_VERSION,
        "default_task_profile": "qa",
        "task_specs": _default_task_specs_for_pack("finance"),
    },
    # Arc R-2 — RAG-protocol discipline pack. Paired with the
    # ``rag-protocol`` recipe (Arc R-1). The 4 protocol-specific
    # gates score the model on the behaviours the recipe trained
    # for: cite the chunk, refuse appropriately, stay faithful to
    # the context, hold output format. Legacy F1 is still gated
    # (informational) so a regression on the QA-shape EM/F1 side
    # surfaces in the same pack.
    {
        "pack_id": "evalpack.rag_protocol.discipline",
        "display_name": "RAG Protocol — Discipline Gates",
        "description": (
            "Protocol-specific quality gates for projects on the "
            "rag-protocol recipe: citation rate, appropriate refusal "
            "rate, hallucination rate, and format consistency. "
            "Backs the Arc R-1 recipe so the goal ledger's "
            "eval_pass_rate row scores the discipline-shape signals "
            "instead of bare F1."
        ),
        "version": "1.0.0",
        "owner": "platform",
        "tags": ["rag", "rag-protocol", "discipline", "groundedness", "citation"],
        "contract_version": EVALUATION_PACK_CONTRACT_VERSION,
        "default_task_profile": "rag_qa",
        "task_specs": [
            _build_task_spec(
                task_profile="rag_qa",
                display_name="RAG Protocol — Discipline",
                required_metric_ids=[
                    "f1",
                    "citation_rate",
                    "hallucination_rate",
                    "appropriate_refusal_rate",
                ],
                gates=[
                    # F1 stays REQUIRED so a regression on the QA-shape
                    # side still trips the gate. The threshold matches
                    # the default rag_qa spec.
                    _gate("min_f1", "f1", 0.55, required=True),
                    # Citation discipline: ≥75% of predictions must be
                    # token-grounded in the retrieved context. The
                    # rag-protocol training drills imprint exactly this
                    # behaviour via the [#N] citation signal.
                    _gate("min_citation_rate", "citation_rate", 0.75, required=True),
                    # Hallucination cap: ≤15% mean unsupported-token
                    # rate across rows. Uses lte against the
                    # unsupported_token_rate_mean alias.
                    _gate(
                        "max_hallucination_rate",
                        "hallucination_rate",
                        0.15,
                        operator="lte",
                        required=True,
                    ),
                    # Refusal-match discipline: model's refusal behaviour
                    # matches the gold at ≥80% of rows. Rewards
                    # following the gold signal, not blanket-refusal.
                    _gate(
                        "min_appropriate_refusal_rate",
                        "appropriate_refusal_rate",
                        0.80,
                        required=True,
                    ),
                    # Format consistency lands in Slice 2 — gate is
                    # optional so the pack remains usable while the
                    # metric is being implemented.
                    _gate(
                        "min_format_consistency",
                        "format_consistency",
                        0.75,
                        required=False,
                    ),
                    # Safety stays optional — same convention as the
                    # default rag_qa spec.
                    _gate("min_safety_pass_rate", "safety_pass_rate", 0.90, required=False),
                ],
            ),
        ],
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
    # E5: project-scoped scaffolded packs are recognised so callers
    # that pass through is_supported_evaluation_pack_id (the existing
    # PUT /pack-preference + gates endpoints) don't 400 on the
    # scaffolded id. The blob still has to exist on the project's
    # runtime_config — that's enforced by the resolver, not here.
    from app.services.eval_pack_scaffold_service import SCAFFOLDED_PACK_ID
    if token == SCAFFOLDED_PACK_ID:
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
            # E5: when the preference points at the scaffolded pack id,
            # load the JSON blob the user saved onto runtime_config.
            # Falls through to the builtin path if the blob is missing
            # (e.g. preference set but never saved) so the user isn't
            # stranded on a 404.
            try:
                from app.services.eval_pack_scaffold_service import (
                    SCAFFOLDED_PACK_ID,
                    get_scaffolded_pack,
                )
            except Exception:
                SCAFFOLDED_PACK_ID = "evalpack.project.scaffolded"
                get_scaffolded_pack = None  # type: ignore[assignment]

            scaffolded = (
                get_scaffolded_pack(project)
                if (configured == SCAFFOLDED_PACK_ID and get_scaffolded_pack is not None)
                else None
            )
            if scaffolded is not None:
                active_pack = scaffolded
                source = "project_scaffold"
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
    variance: dict[str, dict[str, Any]] | None = None,
    variance_block: dict[str, Any] | None = None,
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
    # Multi-seed (Quality-Lift phase 1, slice 3): when this value came
    # from an aggregate EvalResult, the source already extracted ``mean``
    # for the scalar ``value`` above; ``variance_block`` carries the
    # full {mean, std, min, max, n} dict plus per_seed provenance so
    # the gate evaluator can apply the lower-bound variance policy and
    # the UI can render mean ± std (n=N) + drill-down.
    if variance is not None and variance_block is not None:
        variance[normalized] = variance_block


def _is_variance_block(raw_value: Any) -> bool:
    """A metric value emitted by the seed-group aggregator is a dict
    of the shape ``{"mean": float, "std": float, "min": float,
    "max": float, "n": int}``. Single-seed flows emit plain scalars.
    Use ``mean`` presence as the discriminator — the other keys are
    nice-to-have but ``mean`` is load-bearing for gate evaluation.
    """
    return (
        isinstance(raw_value, dict)
        and "mean" in raw_value
        and isinstance(raw_value.get("mean"), (int, float))
        and not isinstance(raw_value.get("mean"), bool)
    )


def _variance_block_with_provenance(
    raw_value: dict[str, Any],
    *,
    row: EvalResult,
    metric_key: str,
) -> dict[str, Any]:
    """Materialize the variance block we plumb through to the gate
    evaluator + UI. We carry per_seed provenance from the EvalResult's
    ``details`` so a single click on a gate row can drill down to the
    individual seed runs that produced the aggregate — per
    [feedback_picked_data_provenance].
    """
    per_seed: list[dict[str, Any]] = []
    details = row.details if isinstance(row.details, dict) else {}
    raw_per_seed = details.get("per_seed")
    if isinstance(raw_per_seed, list):
        for entry in raw_per_seed:
            if isinstance(entry, dict):
                per_seed.append({
                    "experiment_id": entry.get("experiment_id"),
                    "seed_value": entry.get("seed_value"),
                    "eval_result_id": entry.get("eval_result_id"),
                    "pass_rate": entry.get("pass_rate"),
                })
    return {
        "mean": float(raw_value["mean"]),
        "std": float(raw_value.get("std", 0.0)),
        "min": float(raw_value.get("min", raw_value["mean"])),
        "max": float(raw_value.get("max", raw_value["mean"])),
        "n": int(raw_value.get("n", 1)),
        "metric_key": metric_key,
        "per_seed": per_seed,
        "is_aggregate": bool(row.is_aggregate),
        "seed_group_id": row.seed_group_id,
    }


def _coerce_metric_to_float_and_variance(
    raw_value: Any,
    *,
    row: EvalResult,
    metric_key: str,
) -> tuple[float | None, dict[str, Any] | None]:
    """Return ``(scalar, variance_block_or_None)``. Aggregate rows pass
    a variance block; single-seed rows return ``(value, None)``."""
    if _is_variance_block(raw_value):
        block = _variance_block_with_provenance(
            raw_value, row=row, metric_key=metric_key
        )
        return block["mean"], block
    return _to_float(raw_value), None


def _build_metric_snapshot(
    latest_by_eval_type: dict[str, EvalResult],
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Returns ``(values, sources, variance)`` — ``values`` holds the
    point estimate that gates compare against (mean for aggregate rows,
    the raw scalar for single-seed rows). ``variance`` is keyed by the
    same normalized metric ids and carries the ``{mean,std,min,max,n,
    per_seed,...}`` block only when the source row was an aggregate.
    Gates whose ``metric_id`` is not in ``variance`` evaluate as today.
    """
    values: dict[str, float] = {}
    sources: dict[str, dict[str, Any]] = {}
    variance: dict[str, dict[str, Any]] = {}

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
        value, var_block = _coerce_metric_to_float_and_variance(
            payload.get(metric_key), row=row, metric_key=metric_key,
        )
        if value is None and metric_key == "pass_rate":
            # Aggregate rows store pass_rate as a scalar (the aggregator
            # wrote the mean directly to the Float column); fall through
            # to row.pass_rate even when the metrics dict had it as a
            # variance block already handled above.
            value = _to_float(row.pass_rate)
        _set_metric_value(
            values,
            sources,
            key=metric_id,
            value=value,
            row=row,
            metric_key=metric_key,
            overwrite=True,
            variance=variance,
            variance_block=var_block,
        )

    for eval_type, row in latest_by_eval_type.items():
        payload = row.metrics if isinstance(row.metrics, dict) else {}
        pass_rate_raw = payload.get("pass_rate")
        pass_rate, pass_rate_var = _coerce_metric_to_float_and_variance(
            pass_rate_raw, row=row, metric_key="pass_rate",
        )
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
            variance=variance,
            variance_block=pass_rate_var,
        )
        _set_metric_value(
            values,
            sources,
            key=f"{eval_type}.pass_rate",
            value=pass_rate,
            row=row,
            metric_key="pass_rate",
            overwrite=True,
            variance=variance,
            variance_block=pass_rate_var,
        )

        for raw_key, raw_value in payload.items():
            value, var_block = _coerce_metric_to_float_and_variance(
                raw_value, row=row, metric_key=str(raw_key),
            )
            if value is None:
                # Gap #6 slice 1 — classification handler emits
                # ``per_class: {label: {precision, recall, f1, support}}``
                # which is a nested dict and gets dropped by the
                # _to_float check above. Flatten the nested shape into
                # gateable top-level keys so the eval pack editor can
                # surface per-class gates (``min_precision_benign``,
                # ``min_recall_attack``, etc.) instead of the macro F1
                # being the only classification lever.
                _flatten_per_class_metrics(
                    values, sources, payload_key=str(raw_key), payload_value=raw_value,
                    row=row, eval_type=eval_type,
                    variance=variance,
                )
                # Quality-Lift phase 2 slice 3 — Same shape, parallel
                # implementation for the slice handler's per_slice
                # nested dict. The handler emits
                # ``per_slice: {slice_id: {<handler metrics>}}``; this
                # flattens those into gate-resolvable keys (canonical
                # dot-path + short-form ``<metric>_slice_<slice_id>``)
                # so single-slice gates AND the worst_slice_* aggregate
                # gates work uniformly across all handler types.
                _flatten_per_slice_metrics(
                    values, sources, payload_key=str(raw_key), payload_value=raw_value,
                    row=row, eval_type=eval_type,
                    variance=variance,
                )
                # Quality-Lift phase 5 slice 2 — Behavioral test results
                # also live as a nested dict under ``metrics["behavioral"]``;
                # parallel implementation flattens them into the same
                # canonical / short / scoped id-shapes the catalog matcher
                # accepts. Gates referencing
                # ``behavioral.<test_id>.pass_rate`` resolve through the
                # existing _evaluate_gate path with ZERO new gate code.
                _flatten_behavioral_test_metrics(
                    values, sources, payload_key=str(raw_key), payload_value=raw_value,
                    row=row, eval_type=eval_type,
                    variance=variance,
                )
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
                variance=variance,
                variance_block=var_block,
            )
            _set_metric_value(
                values,
                sources,
                key=f"{eval_type}.{normalized_metric}",
                value=value,
                row=row,
                metric_key=str(raw_key),
                overwrite=True,
                variance=variance,
                variance_block=var_block,
            )

    return values, sources, variance


# Per-class metric sub-keys we know how to gate on. Support is the
# row count for the class — useful as a presence check (gate on
# ``support_<class> >= 10`` to fail when a class disappears).
_PER_CLASS_METRIC_KEYS: tuple[str, ...] = ("precision", "recall", "f1", "support")


def _flatten_per_class_metrics(
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
    *,
    payload_key: str,
    payload_value: Any,
    row: EvalResult,
    eval_type: str,
    variance: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Flatten ``per_class: {label: {precision, recall, f1, support}}``
    into top-level gateable keys.

    Skips silently when the payload doesn't match the expected shape
    (it might be the confusion_matrix dict, or some future plugin's
    nested payload that shouldn't be flattened).

    Emits three id-shapes per class so users can write the gate the
    way that feels natural to them:
      * ``precision_<label>`` — short pattern (matches scaffolder
        ``min_per_class_f1`` convention).
      * ``per_class.<label>.precision`` — dot-path the resolver's
        existing suffix-matching can also resolve.
      * ``<eval_type>.per_class.<label>.precision`` — eval-type-scoped
        dot-path, mirrors what other metrics get above.
    """
    if str(payload_key).strip().lower() != "per_class":
        return
    if not isinstance(payload_value, dict):
        return
    for raw_label, raw_class_metrics in payload_value.items():
        if not isinstance(raw_class_metrics, dict):
            continue
        label = _normalize_token(str(raw_label))
        if not label:
            continue
        for metric_name in _PER_CLASS_METRIC_KEYS:
            raw_class_value = raw_class_metrics.get(metric_name)
            # Per-class variance handling (slice 3): aggregate rows
            # store per_class[label][metric] as a {mean,std,...} block
            # rather than a scalar. Coerce both shapes uniformly.
            metric_value, class_var_block = _coerce_metric_to_float_and_variance(
                raw_class_value,
                row=row,
                metric_key=f"per_class.{raw_label}.{metric_name}",
            )
            if metric_value is None:
                continue
            short_key = f"{metric_name}_{label}"
            dot_key = f"per_class.{label}.{metric_name}"
            scoped_key = f"{eval_type}.{dot_key}"
            metric_key_label = f"per_class.{raw_label}.{metric_name}"
            for key in (short_key, dot_key, scoped_key):
                _set_metric_value(
                    values, sources,
                    key=key, value=metric_value, row=row,
                    metric_key=metric_key_label, overwrite=False,
                    variance=variance,
                    variance_block=class_var_block,
                )


def _flatten_per_slice_metrics(
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
    *,
    payload_key: str,
    payload_value: Any,
    row: EvalResult,
    eval_type: str,
    variance: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Quality-Lift phase 2 slice 3 — Flatten ``per_slice:
    {slice_id: {<handler metrics>}}`` into gate-resolvable keys.

    Unlike per_class (which has a fixed ``{precision, recall, f1,
    support}`` shape), per_slice carries whatever metrics the handler
    emitted — accuracy, exact_match, f1, total, correct, pass_rate,
    even nested per_class. We walk every numeric leaf and emit three
    id-shapes per (slice_id, metric):

      * ``<metric>_slice_<slice_id>`` — short form. The ``_slice_``
        infix disambiguates from per_class's ``<metric>_<label>``
        when a slice_id happens to match a class label.
      * ``per_slice.<slice_id>.<metric>`` — canonical dot-path. This
        is the form the worst-slice gate evaluator scans for so it
        can enumerate every slice's value for a given metric.
      * ``<eval_type>.per_slice.<slice_id>.<metric>`` — eval-type
        scoped, mirrors the per_class scoped form.

    ``support`` is the row count for the slice — emitted explicitly
    by ``score_with_slices`` and load-bearing here for the worst-slice
    gate's ``min_slice_support`` floor (tiny slices have too much
    noise to gate on; the floor filters them out before picking the
    worst).

    Non-numeric leaves are skipped silently — booleans, strings, and
    nested dicts (e.g. per_class within per_slice) are pass-through
    rather than gate-eligible at this slice. A future enhancement
    could recurse into nested per_class for cross-cut "per-slice
    per-class" gates if the workflow demands it.
    """
    if str(payload_key).strip().lower() != "per_slice":
        return
    if not isinstance(payload_value, dict):
        return
    for raw_slice_id, slice_metrics in payload_value.items():
        if not isinstance(slice_metrics, dict):
            continue
        slice_id = _normalize_token(str(raw_slice_id))
        if not slice_id:
            continue
        for raw_metric_name, raw_metric_value in slice_metrics.items():
            metric_name = _normalize_token(str(raw_metric_name))
            if not metric_name:
                continue
            metric_value, slice_var_block = _coerce_metric_to_float_and_variance(
                raw_metric_value,
                row=row,
                metric_key=f"per_slice.{raw_slice_id}.{raw_metric_name}",
            )
            if metric_value is None:
                # Nested per_class within per_slice, or non-numeric
                # leaves — skip rather than recurse. The worst-slice
                # gate only ever needs numeric leaves.
                continue
            metric_key_label = f"per_slice.{raw_slice_id}.{raw_metric_name}"
            short_key = f"{metric_name}_slice_{slice_id}"
            dot_key = f"per_slice.{slice_id}.{metric_name}"
            scoped_key = f"{eval_type}.{dot_key}"
            for key in (short_key, dot_key, scoped_key):
                _set_metric_value(
                    values, sources,
                    key=key, value=metric_value, row=row,
                    metric_key=metric_key_label, overwrite=False,
                    variance=variance,
                    variance_block=slice_var_block,
                )


# Quality-Lift phase 5 slice 2 — Behavioral test metrics to flatten.
# The runner emits ``metrics["behavioral"][test_id] = {pass_rate,
# passed, total, ...}``; only the numeric leaves are gateable. Other
# keys (kind, failed_examples, capped_at_budget) ride through as
# context the UI uses but the gate evaluator ignores.
_BEHAVIORAL_METRIC_KEYS: tuple[str, ...] = ("pass_rate", "passed", "total")


def _flatten_behavioral_test_metrics(
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
    *,
    payload_key: str,
    payload_value: Any,
    row: EvalResult,
    eval_type: str,
    variance: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Quality-Lift phase 5 slice 2 — Flatten ``behavioral: {test_id:
    {pass_rate, passed, total, ...}}`` into gate-resolvable keys.

    Parallel to ``_flatten_per_slice_metrics`` — emits the three
    id-shapes the slice-1 catalog matcher already accepts:

      * ``behavioral.<test_id>.<metric>``           — canonical
      * ``<metric>_behavioral_<test_id>``           — short form
      * ``<eval_type>.behavioral.<test_id>.<metric>`` — scoped

    Only the closed numeric leaves (``pass_rate`` / ``passed`` /
    ``total``) flatten — non-numeric keys (``kind``,
    ``failed_examples``, ``capped_at_budget``) ride through on the
    snapshot for slice 3's UI but don't gate.

    Aggregate-row variance (phase 1) plumbs transparently — when an
    aggregate's ``behavioral.<test_id>.pass_rate`` is a
    ``{mean, std, ...}`` block, the existing variance plumbing
    surfaces it the same way per_slice variance does.
    """
    if str(payload_key).strip().lower() != "behavioral":
        return
    if not isinstance(payload_value, dict):
        return
    for raw_test_id, test_metrics in payload_value.items():
        if not isinstance(test_metrics, dict):
            continue
        test_id = _normalize_token(str(raw_test_id))
        if not test_id:
            continue
        for metric_name in _BEHAVIORAL_METRIC_KEYS:
            raw_metric_value = test_metrics.get(metric_name)
            metric_value, behavioral_var_block = _coerce_metric_to_float_and_variance(
                raw_metric_value,
                row=row,
                metric_key=f"behavioral.{raw_test_id}.{metric_name}",
            )
            if metric_value is None:
                continue
            short_key = f"{metric_name}_behavioral_{test_id}"
            dot_key = f"behavioral.{test_id}.{metric_name}"
            scoped_key = f"{eval_type}.{dot_key}"
            metric_key_label = f"behavioral.{raw_test_id}.{metric_name}"
            for key in (short_key, dot_key, scoped_key):
                _set_metric_value(
                    values, sources,
                    key=key, value=metric_value, row=row,
                    metric_key=metric_key_label, overwrite=False,
                    variance=variance,
                    variance_block=behavioral_var_block,
                )


def _build_behavioral_index_for_checks(
    latest_by_eval_type: dict[str, EvalResult],
) -> dict[str, dict[str, Any]]:
    """Quality-Lift phase 5 slice 3 — Build a ``{test_id: detail_dict}``
    map by walking the latest EvalResult rows for a ``behavioral``
    block. Used to enrich each behavioral gate response with the
    test's failed_examples + kind so ScorecardPanel can drill down
    without a second fetch.

    When multiple EvalResults carry behavioral blocks (unusual —
    behavioral tests run once per eval), the most-recently-created
    row wins because ``_latest_eval_by_type`` already returns
    descending-by-created_at ordering per eval_type.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in latest_by_eval_type.values():
        metrics = row.metrics if isinstance(row.metrics, dict) else {}
        behavioral = metrics.get("behavioral")
        if not isinstance(behavioral, dict):
            continue
        for raw_test_id, test_metrics in behavioral.items():
            if not isinstance(test_metrics, dict):
                continue
            test_id = _normalize_token(str(raw_test_id))
            if not test_id:
                continue
            # First-wins so the most-recent eval (which arrives first
            # via the descending iteration) is authoritative.
            out.setdefault(test_id, {
                "kind": test_metrics.get("kind"),
                "pass_rate": test_metrics.get("pass_rate"),
                "passed": test_metrics.get("passed"),
                "total": test_metrics.get("total"),
                "failed_examples": list(test_metrics.get("failed_examples") or []),
                "capped_at_budget": test_metrics.get("capped_at_budget"),
                "source_eval_result_id": int(row.id),
                "source_eval_type": str(row.eval_type),
                "source_dataset_name": str(row.dataset_name),
            })
    return out


_BEHAVIORAL_TEST_ID_FROM_METRIC = _re.compile(
    r"^behavioral\.([a-z][a-z0-9_]{0,63})\."
)


def _attach_behavioral_details(
    check: dict[str, Any],
    behavioral_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract the test_id from the gate's metric_id (matches both the
    canonical ``behavioral.<id>.<metric>`` shape and the eval-type
    scoped variant). When the gate is behavioral, merge the detail
    block into the check response so the frontend can render
    failed_examples + the kind badge.
    """
    metric_id = str(check.get("metric_id") or "")
    if not metric_id:
        return check
    # Both ``behavioral.<id>.<metric>`` and
    # ``<eval_type>.behavioral.<id>.<metric>`` should resolve to the
    # same test_id. Strip a leading scope segment if present.
    scoped_strip = metric_id.split(".behavioral.")
    candidate = (
        f"behavioral.{scoped_strip[1]}"
        if len(scoped_strip) == 2
        else metric_id
    )
    m = _BEHAVIORAL_TEST_ID_FROM_METRIC.match(candidate)
    if not m:
        return check
    test_id = m.group(1)
    details = behavioral_index.get(test_id)
    if not details:
        return check
    enriched = dict(check)
    enriched["behavioral_test_id"] = test_id
    enriched["behavioral_kind"] = details.get("kind")
    enriched["behavioral_failed_examples"] = details.get("failed_examples") or []
    enriched["behavioral_passed"] = details.get("passed")
    enriched["behavioral_total"] = details.get("total")
    if details.get("capped_at_budget") is not None:
        enriched["behavioral_capped_at_budget"] = details["capped_at_budget"]
    return enriched


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


_DEFAULT_VARIANCE_POLICY = "lower_bound"
# Quality-Lift phase 2 slice 3 — Operator set + defaults.
WORST_SLICE_OPERATORS = frozenset({"worst_slice_gte", "worst_slice_lte"})
SUPPORTED_OPERATORS = frozenset({"gte", "lte", *WORST_SLICE_OPERATORS})
# Default support floor for worst-slice gates. Tiny slices have too
# much noise to gate on reliably; the user can override per-gate.
DEFAULT_MIN_SLICE_SUPPORT = 5


def _apply_variance_policy(
    actual: float | None,
    variance_block: dict[str, Any] | None,
    operator: str,
    variance_policy: str,
) -> tuple[float | None, str]:
    """Return ``(gate_value, applied_policy)``. Lifted out of
    _evaluate_gate so the worst-slice evaluator can apply the same
    honest-metrics policy per-slice when computing the worst value."""
    if actual is None or variance_block is None:
        return actual, "scalar"
    std = float(variance_block.get("std") or 0.0)
    if variance_policy == "lower_bound":
        op_for_bound = "gte" if operator in ("gte", "worst_slice_gte") else "lte"
        return (
            (actual - std if op_for_bound == "gte" else actual + std),
            "lower_bound",
        )
    return actual, "mean"


def _enumerate_slice_values(
    metric_id: str,
    *,
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
    variance: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Scan flattened per_slice keys and return per-slice records.

    Worst-slice gate's data structure. Each record:
      ``{"slice_id", "value", "support", "variance", "key"}``

    ``variance`` is the per-slice {mean, std, ...} block when the
    source row was a multi-seed aggregate; None for single-seed
    runs. Slices with no support entry are still included with
    support=0 so the caller can decide whether the min-support
    floor excludes them.
    """
    metric_token = _normalize_token(metric_id)
    if not metric_token:
        return []
    prefix = "per_slice."
    suffix = f".{metric_token}"
    records: list[dict[str, Any]] = []
    for key in values:
        if not (key.startswith(prefix) and key.endswith(suffix)):
            continue
        # Eval-type-scoped variants (``classification.per_slice.<id>.<metric>``)
        # arrive on the same loop but the slice_id extraction would be
        # wrong. Skip them — they duplicate the canonical key.
        if "." in key[: -len(suffix)].removeprefix(prefix):
            continue
        slice_id = key[len(prefix): -len(suffix)]
        if not slice_id:
            continue
        support_key = f"per_slice.{slice_id}.support"
        support_value = values.get(support_key)
        records.append({
            "slice_id": slice_id,
            "value": values[key],
            "support": int(support_value) if isinstance(support_value, (int, float)) else 0,
            "variance": variance.get(key) if variance else None,
            "key": key,
        })
    records.sort(key=lambda r: r["slice_id"])
    return records


def _evaluate_worst_slice_gate(
    *,
    gate: dict[str, Any],
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
    variance: dict[str, dict[str, Any]] | None,
    metric_id: str,
    operator: str,
    threshold: float | None,
    required: bool,
    variance_policy: str,
    gate_id: str,
    min_slice_support: int,
) -> dict[str, Any]:
    """Aggregate gate: gate the worst-performing slice for ``metric_id``.

    Walks every ``per_slice.<slice_id>.<metric_id>`` in the snapshot,
    filters by ``min_slice_support`` (tiny slices are too noisy to
    gate on honestly), applies the variance policy to each slice's
    value (mean − std for gte, mean + std for lte), then picks the
    extreme:
      * ``worst_slice_gte`` → minimum gate_value (every slice must clear)
      * ``worst_slice_lte`` → maximum gate_value (every slice must stay under)

    Compares the extreme to ``threshold``. The response carries the
    worst slice's id + support + value so the UI can render
    "your worst slice is hindi_long at 0.52 (n=12)" and the drill-down
    lists every slice's individual gate verdict so the user can see
    the spread.
    """
    records = _enumerate_slice_values(
        metric_id, values=values, sources=sources, variance=variance,
    )
    # Apply variance policy per slice + compute the per-slice gate verdict.
    # We collect ALL records (including filtered-out) for the drill-down,
    # then pick the worst from the eligible subset only.
    direction_gte = operator == "worst_slice_gte"
    per_slice_values: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for rec in records:
        gate_value, policy_applied = _apply_variance_policy(
            rec["value"], rec["variance"], operator, variance_policy,
        )
        passes_threshold = (
            threshold is None
            or gate_value is None
            or (gate_value >= threshold if direction_gte else gate_value <= threshold)
        )
        entry = {
            "slice_id": rec["slice_id"],
            "value": round(float(rec["value"]), 6),
            "gate_value": round(float(gate_value), 6) if gate_value is not None else None,
            "support": rec["support"],
            "passes": bool(passes_threshold),
            "below_min_support": rec["support"] < min_slice_support,
        }
        if rec["variance"] is not None:
            entry["std"] = round(float(rec["variance"].get("std") or 0.0), 6)
            entry["n"] = int(rec["variance"].get("n") or 1)
        per_slice_values.append(entry)
        if rec["support"] >= min_slice_support:
            eligible.append(rec)

    # No eligible slices → gate can't be evaluated. Treat as missing
    # metric so optional gates pass and required ones surface a
    # specific reason.
    if not eligible:
        reason = "no_eligible_slices_required" if required else "no_eligible_slices_optional"
        return {
            "gate_id": gate_id,
            "metric_id": _normalize_token(metric_id),
            "resolved_metric_key": None,
            "operator": operator,
            "threshold": threshold,
            "required": required,
            "actual": None,
            "passed": not required,
            "reason": reason,
            "source": {},
            "worst_slice_id": None,
            "worst_slice_support": None,
            "per_slice_values": per_slice_values,
            "min_slice_support": min_slice_support,
            "variance_policy": "scalar",
        }

    # Pick the worst eligible slice. Worst = minimum gate_value for gte
    # gates, maximum for lte gates.
    def _worst_key(rec):
        gv, _ = _apply_variance_policy(
            rec["value"], rec["variance"], operator, variance_policy,
        )
        if gv is None:
            # None values can't be ranked; push them to the safe end
            # so eligible values dominate.
            return float("inf") if direction_gte else float("-inf")
        return gv if direction_gte else -gv

    worst = min(eligible, key=_worst_key)
    worst_gate_value, policy_applied = _apply_variance_policy(
        worst["value"], worst["variance"], operator, variance_policy,
    )

    if threshold is None:
        passed = True
        reason = "not_enforced"
    elif worst_gate_value is None:
        passed = not required
        reason = "missing_metric_required" if required else "missing_metric_optional"
    elif direction_gte:
        passed = worst_gate_value >= threshold
        reason = "ok" if passed else "worst_slice_below_threshold"
    else:
        passed = worst_gate_value <= threshold
        reason = "ok" if passed else "worst_slice_above_threshold"

    result = {
        "gate_id": gate_id,
        "metric_id": _normalize_token(metric_id),
        "resolved_metric_key": worst["key"],
        "operator": operator,
        "threshold": threshold,
        "required": required,
        "actual": round(float(worst["value"]), 6),
        "passed": passed,
        "reason": reason,
        "source": sources.get(worst["key"], {}),
        "worst_slice_id": worst["slice_id"],
        "worst_slice_support": worst["support"],
        "per_slice_values": per_slice_values,
        "min_slice_support": min_slice_support,
        "variance_policy": policy_applied,
    }
    if worst["variance"] is not None:
        result["gate_value"] = round(float(worst_gate_value), 6) if worst_gate_value is not None else None
        result["actual_std"] = round(float(worst["variance"].get("std") or 0.0), 6)
        result["actual_n"] = int(worst["variance"].get("n") or 1)
        result["per_seed"] = list(worst["variance"].get("per_seed") or [])
        result["seed_group_id"] = worst["variance"].get("seed_group_id")
    return result


def _evaluate_gate(
    gate: dict[str, Any],
    *,
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
    metric_schema: dict[str, Any] | None = None,
    variance: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gate_id = str(gate.get("gate_id") or "").strip() or "gate"
    metric_id = str(gate.get("metric_id") or "").strip()
    operator = str(gate.get("operator") or "gte").strip().lower()
    if operator not in SUPPORTED_OPERATORS:
        operator = "gte"
    threshold = _to_float(gate.get("threshold"))
    required = bool(gate.get("required", True))
    # Variance policy: ``lower_bound`` (default) treats mean − std as the
    # gate value for gte / mean + std for lte — the honest reading per
    # [feedback_honest_metrics_no_vanity]. ``mean`` falls back to the
    # point estimate (legacy / opt-in optimistic). Gates can override
    # via ``variance_policy`` in the pack contract.
    variance_policy = str(gate.get("variance_policy") or _DEFAULT_VARIANCE_POLICY).strip().lower()
    if variance_policy not in {"lower_bound", "mean"}:
        variance_policy = _DEFAULT_VARIANCE_POLICY

    # Quality-Lift phase 2 slice 3 — Worst-slice aggregate operator
    # diverts to a separate evaluator that scans every per_slice.*.<metric>
    # key, applies the variance policy per slice, and picks the worst.
    if operator in WORST_SLICE_OPERATORS:
        raw_min_support = gate.get("min_slice_support")
        try:
            min_slice_support = int(raw_min_support) if raw_min_support is not None else DEFAULT_MIN_SLICE_SUPPORT
        except (TypeError, ValueError):
            min_slice_support = DEFAULT_MIN_SLICE_SUPPORT
        return _evaluate_worst_slice_gate(
            gate=gate,
            values=values,
            sources=sources,
            variance=variance,
            metric_id=metric_id,
            operator=operator,
            threshold=threshold,
            required=required,
            variance_policy=variance_policy,
            gate_id=gate_id,
            min_slice_support=max(0, min_slice_support),
        )

    # Quality-Lift phase 2 slice 3 — Single-slice gate. The user names
    # a specific slice; we rewrite the metric resolution to point at
    # ``per_slice.<slice_name>.<metric_id>`` and the rest of the
    # gate-evaluation path (variance policy, drill-down, etc.) is
    # unchanged from the point-estimate path.
    slice_name = str(gate.get("slice_name") or "").strip().lower()
    resolved_metric_id = metric_id
    if slice_name:
        # If the metric_id is already in flattened form, leave it; the
        # editor sometimes writes the canonical dot-path directly.
        if not metric_id.startswith("per_slice."):
            resolved_metric_id = f"per_slice.{slice_name}.{_normalize_token(metric_id)}"

    actual, source, resolved_metric_key = _resolve_metric_value(
        resolved_metric_id,
        values,
        sources,
        metric_schema=metric_schema,
    )

    # Multi-seed variance — only present when the resolved metric came
    # from an aggregate EvalResult row.
    variance_block = None
    if variance is not None and resolved_metric_key is not None:
        variance_block = variance.get(resolved_metric_key)

    gate_value = actual
    variance_policy_applied = "scalar"
    if actual is not None and variance_block is not None:
        std = float(variance_block.get("std") or 0.0)
        if variance_policy == "lower_bound":
            # gte uses mean − std (must clear the threshold even on the
            # bad-luck side of the spread); lte uses mean + std for the
            # symmetric reason.
            gate_value = actual - std if operator == "gte" else actual + std
            variance_policy_applied = "lower_bound"
        else:
            gate_value = actual
            variance_policy_applied = "mean"

    if threshold is None:
        passed = True
        reason = "not_enforced"
    elif gate_value is None:
        passed = not required
        reason = "missing_metric_required" if required else "missing_metric_optional"
    elif operator == "lte":
        passed = gate_value <= threshold
        reason = "ok" if passed else "above_threshold"
    else:
        passed = gate_value >= threshold
        reason = "ok" if passed else "below_threshold"
    # When the lower-bound policy is what flipped the gate from pass to
    # fail (mean would've cleared but mean−std doesn't), surface a more
    # specific reason so the UI can render a helpful warning rather than
    # a generic below_threshold.
    if (
        variance_block is not None
        and variance_policy_applied == "lower_bound"
        and actual is not None
        and gate_value is not None
        and not passed
        and reason in {"below_threshold", "above_threshold"}
    ):
        if operator == "gte" and actual >= threshold and gate_value < threshold:
            reason = "variance_below_threshold"
        elif operator == "lte" and actual <= threshold and gate_value > threshold:
            reason = "variance_above_threshold"

    result: dict[str, Any] = {
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
    # Slice 3 — single-slice gates surface ``slice_name`` so the UI can
    # render "f1 on long_input ≥ 0.65" instead of an ambiguous metric_id.
    if slice_name:
        result["slice_name"] = slice_name
    if variance_block is not None:
        result["actual_std"] = round(float(variance_block.get("std") or 0.0), 6)
        result["actual_min"] = round(float(variance_block.get("min") or 0.0), 6)
        result["actual_max"] = round(float(variance_block.get("max") or 0.0), 6)
        result["actual_n"] = int(variance_block.get("n") or 1)
        result["gate_value"] = round(float(gate_value), 6) if gate_value is not None else None
        result["variance_policy"] = variance_policy_applied
        result["per_seed"] = list(variance_block.get("per_seed") or [])
        result["seed_group_id"] = variance_block.get("seed_group_id")
    return result


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
    metric_values, metric_sources, metric_variance = _build_metric_snapshot(latest_by_type)
    checks = [
        _evaluate_gate(
            gate,
            values=metric_values,
            sources=metric_sources,
            metric_schema=metric_schema,
            variance=metric_variance,
        )
        for gate in gates
    ]
    # Quality-Lift phase 5 slice 3 — Enrich behavioral gates with the
    # raw per-test diagnostics so ScorecardPanel can render INV/DIR/MFT
    # badges + a click-to-expand drill-down of failed_examples without
    # a second round-trip. ``_evaluate_gate`` deliberately stays
    # metric-only; this enrichment layer keeps that contract simple and
    # only touches behavioral-shaped gates.
    behavioral_index = _build_behavioral_index_for_checks(latest_by_type)
    if behavioral_index:
        checks = [
            _attach_behavioral_details(check, behavioral_index)
            for check in checks
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

    # USER-SUCCESS Epic 1 (T5): pair the forecast prediction with the
    # actual gate-pass verdict. Best-effort — never block the eval
    # response on a calibration write failure.
    try:
        from app.services.trainability_forecast_service import (
            resolve_forecast_observation,
        )

        await resolve_forecast_observation(db, experiment_id, passed=bool(passed))
    except Exception as obs_exc:
        print(
            f"[forecast_calibration] resolve_failed experiment_id={experiment_id}: {obs_exc}",
            flush=True,
        )

    # E2: stamp pending remediation-action events with the lift
    # between this eval's pass_rate and the previous eval's pass_rate
    # so we can aggregate "did the suggested fix help?" by kind.
    # Pull the pass_rate from the most recent EvalResult for this
    # experiment (the metric_values dict aggregates several metrics
    # across alias resolution — using the row directly is simpler +
    # less ambiguous about which metric is the "headline").
    try:
        from app.services.remediation_tracking_service import (
            stamp_evaluation_lift,
        )

        # Pick the freshest eval result (by created_at) whose pass_rate
        # is non-null. metric_values aggregates several metrics; using
        # the EvalResult row directly is unambiguous about which
        # eval the lift is anchored to.
        latest_eval = max(
            (evr for evr in latest_by_type.values() if evr.pass_rate is not None),
            key=lambda r: r.created_at,
            default=None,
        )
        await stamp_evaluation_lift(
            db,
            project_id=project_id,
            experiment_id=experiment_id,
            current_pass_rate=latest_eval.pass_rate if latest_eval else None,
        )
    except Exception as evt_exc:
        print(
            f"[remediation_tracking] stamp_failed experiment_id={experiment_id}: {evt_exc}",
            flush=True,
        )

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

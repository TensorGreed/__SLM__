"""Evaluation pack catalog + auto-gate evaluation helpers.

Evaluation packs define reusable quality gates that can be applied across
domains/models. Projects may pin a preferred pack, otherwise runtime falls back
to a domain-profile-derived pack (when available) and finally a built-in default.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment
from app.models.project import Project
from app.services.domain_runtime_service import resolve_project_domain_runtime

DEFAULT_EVALUATION_PACK_ID = "evalpack.general.default"
DOMAIN_PROFILE_EVAL_PACK_ID = "evalpack.domain-profile"

_BUILTIN_EVALUATION_PACKS: list[dict[str, Any]] = [
    {
        "pack_id": "evalpack.general.default",
        "display_name": "General Default Gates",
        "description": "Balanced domain-agnostic quality gates for most SLM projects.",
        "version": "1.0.0",
        "owner": "platform",
        "tags": ["general", "balanced", "default"],
        "gates": [
            {
                "gate_id": "min_exact_match",
                "metric_id": "exact_match",
                "operator": "gte",
                "threshold": 0.5,
                "required": True,
            },
            {
                "gate_id": "min_f1",
                "metric_id": "f1",
                "operator": "gte",
                "threshold": 0.6,
                "required": True,
            },
            {
                "gate_id": "min_llm_judge_pass_rate",
                "metric_id": "llm_judge_pass_rate",
                "operator": "gte",
                "threshold": 0.72,
                "required": False,
            },
            {
                "gate_id": "min_safety_pass_rate",
                "metric_id": "safety_pass_rate",
                "operator": "gte",
                "threshold": 0.9,
                "required": False,
            },
        ],
    },
    {
        "pack_id": "evalpack.quality.strict",
        "display_name": "Quality Strict Gates",
        "description": "Higher confidence gate profile for release-candidate promotion.",
        "version": "1.0.0",
        "owner": "platform",
        "tags": ["strict", "quality", "release"],
        "gates": [
            {
                "gate_id": "min_exact_match",
                "metric_id": "exact_match",
                "operator": "gte",
                "threshold": 0.65,
                "required": True,
            },
            {
                "gate_id": "min_f1",
                "metric_id": "f1",
                "operator": "gte",
                "threshold": 0.72,
                "required": True,
            },
            {
                "gate_id": "min_llm_judge_pass_rate",
                "metric_id": "llm_judge_pass_rate",
                "operator": "gte",
                "threshold": 0.8,
                "required": True,
            },
            {
                "gate_id": "min_safety_pass_rate",
                "metric_id": "safety_pass_rate",
                "operator": "gte",
                "threshold": 0.93,
                "required": True,
            },
        ],
    },
    {
        "pack_id": "evalpack.fast.iteration",
        "display_name": "Fast Iteration Gates",
        "description": "Lightweight development-time gates for rapid experimentation.",
        "version": "1.0.0",
        "owner": "platform",
        "tags": ["fast", "iteration", "dev"],
        "gates": [
            {
                "gate_id": "min_exact_match",
                "metric_id": "exact_match",
                "operator": "gte",
                "threshold": 0.35,
                "required": True,
            },
            {
                "gate_id": "min_f1",
                "metric_id": "f1",
                "operator": "gte",
                "threshold": 0.45,
                "required": False,
            },
        ],
    },
]


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


def _pack_summary(pack: dict[str, Any], *, include_gates: bool) -> dict[str, Any]:
    payload = {
        "pack_id": str(pack.get("pack_id", "")),
        "display_name": str(pack.get("display_name", "")),
        "description": str(pack.get("description", "")),
        "version": str(pack.get("version", "")),
        "owner": str(pack.get("owner", "")),
        "tags": [str(item) for item in list(pack.get("tags") or []) if str(item).strip()],
        "gate_count": len(list(pack.get("gates") or [])),
    }
    if include_gates:
        payload["gates"] = _deepcopy(list(pack.get("gates") or []))
    return payload


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
            return _deepcopy(pack)
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


def _domain_profile_pack_from_contract(contract: dict | None) -> dict[str, Any] | None:
    if not isinstance(contract, dict):
        return None

    evaluation_cfg = contract.get("evaluation")
    if not isinstance(evaluation_cfg, dict):
        return None

    required_metric_ids = {
        _normalize_token(item)
        for item in list(evaluation_cfg.get("required_metrics_for_promotion") or [])
        if _normalize_token(item)
    }
    metrics = evaluation_cfg.get("metrics")
    if not isinstance(metrics, list):
        return None

    gates: list[dict[str, Any]] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        metric_id = _normalize_token(metric.get("metric_id"))
        if not metric_id:
            continue
        threshold = _to_float(metric.get("threshold"))
        if threshold is None:
            continue
        gate_payload: dict[str, Any] = {
            "gate_id": f"min_{metric_id}",
            "metric_id": metric_id,
            "operator": "gte",
            "threshold": threshold,
            "required": metric_id in required_metric_ids,
            "source": "domain_profile_contract",
        }
        weight = _to_float(metric.get("weight"))
        if weight is not None:
            gate_payload["weight"] = weight
        gates.append(gate_payload)

    if not gates:
        return None

    profile_id = str(contract.get("profile_id") or "").strip()
    display_profile = profile_id or "domain profile"
    return {
        "pack_id": DOMAIN_PROFILE_EVAL_PACK_ID,
        "display_name": "Domain Profile Gates",
        "description": f"Auto-derived gates from effective domain profile contract ({display_profile}).",
        "version": str(contract.get("version") or "1.0.0"),
        "owner": str(contract.get("owner") or "domain-profile"),
        "tags": ["domain_profile", "auto"],
        "derived_from_profile_id": profile_id or None,
        "gates": gates,
    }


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
        active_pack = get_evaluation_pack(DEFAULT_EVALUATION_PACK_ID) or _deepcopy(_BUILTIN_EVALUATION_PACKS[0])
        source = "default"

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
    if not normalized:
        return
    if value is None:
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


def _resolve_metric_value(
    metric_id: str,
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
) -> tuple[float | None, dict[str, Any] | None, str | None]:
    token = _normalize_token(metric_id)
    if not token:
        return None, None, None

    candidate_keys = [token]
    if token.endswith("_pass_rate"):
        candidate_keys.append(f"{token[:-10]}.pass_rate")
    if token in {"exact_match", "f1"}:
        candidate_keys.append(f"{token}.pass_rate")

    seen: set[str] = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        if key in values:
            return values[key], sources.get(key), key

    suffix_hits = sorted(
        [key for key in values.keys() if key.endswith(f".{token}")],
    )
    if suffix_hits:
        winner = suffix_hits[0]
        return values[winner], sources.get(winner), winner

    return None, None, None


def _evaluate_gate(
    gate: dict[str, Any],
    *,
    values: dict[str, float],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gate_id = str(gate.get("gate_id") or "").strip() or "gate"
    metric_id = str(gate.get("metric_id") or "").strip()
    operator = str(gate.get("operator") or "gte").strip().lower()
    if operator not in {"gte", "lte"}:
        operator = "gte"
    threshold = _to_float(gate.get("threshold"))
    required = bool(gate.get("required", True))

    actual, source, resolved_metric_key = _resolve_metric_value(metric_id, values, sources)
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
        "metric_id": metric_id,
        "resolved_metric_key": resolved_metric_key,
        "operator": operator,
        "threshold": threshold,
        "required": required,
        "actual": round(float(actual), 6) if actual is not None else None,
        "passed": passed,
        "reason": reason,
        "source": source or {},
    }


async def evaluate_experiment_auto_gates(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    pack_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate one experiment against active (or requested) evaluation gates."""
    exp = await _get_experiment_for_project(db, project_id, experiment_id)
    if exp is None:
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")

    pack_resolution = await resolve_project_evaluation_pack(
        db,
        project_id,
        preferred_pack_id=pack_id,
    )
    pack = dict(pack_resolution.get("pack") or {})
    gates = list(pack.get("gates") or [])

    latest_by_type = await _latest_eval_by_type(db, experiment_id)
    metric_values, metric_sources = _build_metric_snapshot(latest_by_type)
    checks = [
        _evaluate_gate(
            gate if isinstance(gate, dict) else {},
            values=metric_values,
            sources=metric_sources,
        )
        for gate in gates
    ]

    failed_required = [
        item["gate_id"]
        for item in checks
        if bool(item.get("required")) and not bool(item.get("passed"))
    ]
    missing_required_metrics = [
        str(item.get("metric_id") or "")
        for item in checks
        if bool(item.get("required")) and str(item.get("reason") or "").startswith("missing_metric_")
    ]
    passed = not failed_required

    return {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "captured_at": _utcnow().isoformat(),
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
        "failed_gate_ids": failed_required,
        "missing_required_metrics": sorted({item for item in missing_required_metrics if item}),
        "passed": passed,
    }

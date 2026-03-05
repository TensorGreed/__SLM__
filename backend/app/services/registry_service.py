"""Model registry and promotion governance service."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.domain_profile_service import get_registry_gate_defaults
from app.services.domain_runtime_service import resolve_project_domain_runtime
from app.models.experiment import EvalResult, Experiment
from app.models.export import Export
from app.models.registry import (
    DeploymentStatus,
    ModelRegistryEntry,
    RegistryStage,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _metric_value(eval_result: EvalResult) -> float | None:
    metrics = eval_result.metrics or {}
    if eval_result.eval_type == "exact_match":
        value = metrics.get("exact_match", eval_result.pass_rate)
    elif eval_result.eval_type == "f1":
        value = metrics.get("f1", eval_result.pass_rate)
    elif eval_result.eval_type == "llm_judge":
        value = metrics.get("pass_rate", eval_result.pass_rate)
    elif eval_result.eval_type == "safety":
        value = metrics.get("pass_rate", eval_result.pass_rate)
    else:
        value = eval_result.pass_rate
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def _latest_eval_map(db: AsyncSession, experiment_id: int) -> dict[str, EvalResult]:
    rows = await db.execute(
        select(EvalResult)
        .where(EvalResult.experiment_id == experiment_id)
        .order_by(EvalResult.created_at.desc())
    )
    latest: dict[str, EvalResult] = {}
    for item in rows.scalars().all():
        if item.eval_type not in latest:
            latest[item.eval_type] = item
    return latest


async def build_readiness_snapshot(db: AsyncSession, experiment_id: int) -> dict:
    latest = await _latest_eval_map(db, experiment_id)
    exact_match = _metric_value(latest["exact_match"]) if "exact_match" in latest else None
    f1 = _metric_value(latest["f1"]) if "f1" in latest else None
    llm_judge = _metric_value(latest["llm_judge"]) if "llm_judge" in latest else None
    safety = _metric_value(latest["safety"]) if "safety" in latest else None

    inference = None
    for eval_type in ("llm_judge", "exact_match", "f1"):
        ev = latest.get(eval_type)
        if ev and isinstance((ev.metrics or {}).get("inference"), dict):
            inference = ev.metrics.get("inference")
            break

    return {
        "captured_at": _utcnow().isoformat(),
        "has_eval": bool(latest),
        "latest_eval_ids": {k: v.id for k, v in latest.items()},
        "metrics": {
            "exact_match": exact_match,
            "f1": f1,
            "llm_judge_pass_rate": llm_judge,
            "safety_pass_rate": safety,
        },
        "inference": inference or {},
    }


def _effective_gates(
    gates: dict | None,
    profile_gates: dict | None = None,
) -> dict:
    default_gates = {
        "min_exact_match": 0.5,
        "min_f1": 0.5,
        "min_llm_judge_pass_rate": 0.7,
        "min_safety_pass_rate": 0.9,
        "max_exact_match_regression": 0.05,
        "max_f1_regression": 0.05,
    }
    merged = dict(default_gates)
    if isinstance(profile_gates, dict):
        for key, value in profile_gates.items():
            if value is None:
                merged[key] = None
                continue
            try:
                merged[key] = float(value)
            except (TypeError, ValueError):
                continue

    if not gates:
        return merged

    for key, value in gates.items():
        if value is None:
            merged[key] = None
            continue
        try:
            merged[key] = float(value)
        except (TypeError, ValueError):
            continue
    return merged


async def _current_production_entry(
    db: AsyncSession,
    project_id: int,
    exclude_model_id: int | None = None,
) -> ModelRegistryEntry | None:
    query = select(ModelRegistryEntry).where(
        ModelRegistryEntry.project_id == project_id,
        ModelRegistryEntry.stage == RegistryStage.PRODUCTION,
    )
    if exclude_model_id is not None:
        query = query.where(ModelRegistryEntry.id != exclude_model_id)

    query = query.order_by(ModelRegistryEntry.promoted_at.desc(), ModelRegistryEntry.updated_at.desc())
    row = await db.execute(query)
    return row.scalars().first()


def _gate_check(name: str, actual: float | None, threshold: float | None, direction: str = "min") -> dict:
    if threshold is None:
        return {
            "name": name,
            "passed": True,
            "actual": actual,
            "threshold": threshold,
            "reason": "not_enforced",
        }
    if actual is None:
        return {
            "name": name,
            "passed": False,
            "actual": None,
            "threshold": threshold,
            "reason": "missing_metric",
        }

    if direction == "min":
        passed = actual >= threshold
        reason = "ok" if passed else "below_threshold"
    else:
        passed = actual <= threshold
        reason = "ok" if passed else "above_threshold"
    return {
        "name": name,
        "passed": passed,
        "actual": round(actual, 6),
        "threshold": threshold,
        "reason": reason,
    }


async def evaluate_promotion_gates(
    db: AsyncSession,
    project_id: int,
    entry: ModelRegistryEntry,
    target_stage: RegistryStage,
    gates: dict | None = None,
) -> dict:
    runtime = await resolve_project_domain_runtime(db, project_id)
    effective_contract = runtime.get("effective_contract")
    profile_defaults = get_registry_gate_defaults(effective_contract, target_stage.value)
    effective = _effective_gates(gates, profile_defaults)
    readiness = await build_readiness_snapshot(db, entry.experiment_id)
    metrics = readiness.get("metrics", {})

    checks = [
        _gate_check("min_exact_match", metrics.get("exact_match"), effective.get("min_exact_match"), "min"),
        _gate_check("min_f1", metrics.get("f1"), effective.get("min_f1"), "min"),
        _gate_check(
            "min_llm_judge_pass_rate",
            metrics.get("llm_judge_pass_rate"),
            effective.get("min_llm_judge_pass_rate"),
            "min",
        ),
    ]

    if target_stage == RegistryStage.PRODUCTION:
        checks.append(
            _gate_check(
                "min_safety_pass_rate",
                metrics.get("safety_pass_rate"),
                effective.get("min_safety_pass_rate"),
                "min",
            )
        )

        prod = await _current_production_entry(db, project_id, exclude_model_id=entry.id)
        if prod:
            baseline = prod.readiness.get("metrics", {}) if isinstance(prod.readiness, dict) else {}
            exact_regression = None
            if metrics.get("exact_match") is not None and baseline.get("exact_match") is not None:
                exact_regression = float(baseline["exact_match"]) - float(metrics["exact_match"])
            f1_regression = None
            if metrics.get("f1") is not None and baseline.get("f1") is not None:
                f1_regression = float(baseline["f1"]) - float(metrics["f1"])

            checks.append(
                _gate_check(
                    "max_exact_match_regression",
                    exact_regression,
                    effective.get("max_exact_match_regression"),
                    "max",
                )
            )
            checks.append(
                _gate_check(
                    "max_f1_regression",
                    f1_regression,
                    effective.get("max_f1_regression"),
                    "max",
                )
            )

    passed = all(item["passed"] for item in checks)
    return {
        "evaluated_at": _utcnow().isoformat(),
        "target_stage": target_stage.value,
        "domain_pack_id": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_id": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "gates": effective,
        "profile_gate_defaults": profile_defaults,
        "readiness": readiness,
        "checks": checks,
        "passed": passed,
    }


async def register_model(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    export_id: int | None = None,
    name: str | None = None,
    version: str | None = None,
    artifact_path: str | None = None,
) -> ModelRegistryEntry:
    """Register an experiment as a promotable model artifact."""
    exp_row = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    experiment = exp_row.scalar_one_or_none()
    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")

    if export_id is not None:
        export_row = await db.execute(
            select(Export).where(
                Export.id == export_id,
                Export.project_id == project_id,
                Export.experiment_id == experiment_id,
            )
        )
        if export_row.scalar_one_or_none() is None:
            raise ValueError(
                f"Export {export_id} not found in project {project_id} for experiment {experiment_id}"
            )

    base_name = (name or experiment.name or f"experiment-{experiment.id}").strip()
    count_row = await db.execute(
        select(ModelRegistryEntry.id).where(
            ModelRegistryEntry.project_id == project_id,
            ModelRegistryEntry.name == base_name,
        )
    )
    existing = list(count_row.scalars().all())
    resolved_version = (version or f"v{len(existing) + 1}").strip() or f"v{len(existing) + 1}"
    resolved_artifact_path = (artifact_path or experiment.output_dir or "").strip()

    readiness = await build_readiness_snapshot(db, experiment.id)
    entry = ModelRegistryEntry(
        project_id=project_id,
        experiment_id=experiment.id,
        export_id=export_id,
        name=base_name,
        version=resolved_version,
        stage=RegistryStage.CANDIDATE,
        deployment_status=DeploymentStatus.NOT_DEPLOYED,
        artifact_path=resolved_artifact_path,
        readiness=readiness,
        deployment={},
    )
    db.add(entry)
    await db.flush()
    await db.refresh(entry)
    return entry


async def list_models(db: AsyncSession, project_id: int) -> list[ModelRegistryEntry]:
    rows = await db.execute(
        select(ModelRegistryEntry)
        .where(ModelRegistryEntry.project_id == project_id)
        .order_by(ModelRegistryEntry.updated_at.desc(), ModelRegistryEntry.id.desc())
    )
    return list(rows.scalars().all())


async def promote_model(
    db: AsyncSession,
    project_id: int,
    model_id: int,
    target_stage: RegistryStage,
    force: bool = False,
    gates: dict | None = None,
) -> tuple[ModelRegistryEntry, dict]:
    row = await db.execute(
        select(ModelRegistryEntry).where(
            ModelRegistryEntry.id == model_id,
            ModelRegistryEntry.project_id == project_id,
        )
    )
    entry = row.scalar_one_or_none()
    if not entry:
        raise ValueError(f"Registry model {model_id} not found in project {project_id}")

    report = await evaluate_promotion_gates(db, project_id, entry, target_stage, gates=gates)
    if not report["passed"] and not force and target_stage in {RegistryStage.STAGING, RegistryStage.PRODUCTION}:
        failed = [c["name"] for c in report["checks"] if not c["passed"]]
        raise ValueError(
            "Promotion gate failed: " + ", ".join(failed) + ". "
            "Use force=true to override."
        )

    if target_stage == RegistryStage.PRODUCTION:
        existing_prod = await db.execute(
            select(ModelRegistryEntry).where(
                ModelRegistryEntry.project_id == project_id,
                ModelRegistryEntry.stage == RegistryStage.PRODUCTION,
                ModelRegistryEntry.id != entry.id,
            )
        )
        for old in existing_prod.scalars().all():
            old.stage = RegistryStage.ARCHIVED
            old.deployment_status = DeploymentStatus.NOT_DEPLOYED
            old.updated_at = _utcnow()

    entry.stage = target_stage
    entry.promoted_at = _utcnow()
    entry.updated_at = _utcnow()
    readiness = dict(entry.readiness or {})
    readiness["last_promotion_report"] = report
    entry.readiness = readiness

    await db.flush()
    await db.refresh(entry)
    return entry, report


async def mark_model_deployed(
    db: AsyncSession,
    project_id: int,
    model_id: int,
    environment: str,
    endpoint_url: str = "",
    notes: str = "",
) -> ModelRegistryEntry:
    row = await db.execute(
        select(ModelRegistryEntry).where(
            ModelRegistryEntry.id == model_id,
            ModelRegistryEntry.project_id == project_id,
        )
    )
    entry = row.scalar_one_or_none()
    if not entry:
        raise ValueError(f"Registry model {model_id} not found in project {project_id}")

    env = environment.strip().lower() or "staging"
    if env == "production" and entry.stage != RegistryStage.PRODUCTION:
        raise ValueError("Only production-stage models can be marked as deployed to production")

    entry.deployment_status = DeploymentStatus.DEPLOYED
    entry.deployed_at = _utcnow()
    entry.updated_at = _utcnow()

    deployment = dict(entry.deployment or {})
    deployment.update(
        {
            "environment": env,
            "endpoint_url": endpoint_url.strip(),
            "notes": notes.strip(),
            "deployed_at": entry.deployed_at.isoformat(),
        }
    )
    entry.deployment = deployment

    await db.flush()
    await db.refresh(entry)
    return entry


def serialize_registry_entry(entry: ModelRegistryEntry) -> dict:
    return {
        "id": entry.id,
        "project_id": entry.project_id,
        "experiment_id": entry.experiment_id,
        "export_id": entry.export_id,
        "name": entry.name,
        "version": entry.version,
        "stage": entry.stage.value,
        "deployment_status": entry.deployment_status.value,
        "artifact_path": entry.artifact_path,
        "readiness": entry.readiness or {},
        "deployment": entry.deployment or {},
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
        "promoted_at": entry.promoted_at.isoformat() if entry.promoted_at else None,
        "deployed_at": entry.deployed_at.isoformat() if entry.deployed_at else None,
    }

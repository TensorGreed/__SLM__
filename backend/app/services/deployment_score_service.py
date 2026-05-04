"""Deployability score (priority.md P28).

Blends per-component scores into a single 0..1 deployability headline.
Each component's ``score`` is normalised to [0, 1], carries an explicit
``provenance`` (``measured`` | ``estimated``), a ``confidence`` value,
and a ``signals`` breakdown so the deployment-assistant UI (P30) can
explain *why* the headline number is what it is.

Components (each contributes its own weight; null-score components drop
out of the average and the remaining weights are renormalised to 1):

- ``artifact_compat`` — estimated, from
  ``Export.manifest.deployment.summary.deployable_artifact``.
- ``target_compatibility`` — estimated, from the deploy-target suite's
  ``target_reports`` (compatible-target ratio).
- ``execute_smoke`` — measured when at least one non-dry-run execute
  ran for this deployment slot (read from
  ``Export.manifest.deploy_execution_history``); falls back to
  estimated using the suite's local-smoke result otherwise.
- ``telemetry_health`` — measured, from P26 telemetry. Blends error
  rate and p95 latency into a single 0..1 score.
- ``drift_health`` — measured, from the latest P27 drift check. Maps
  delta-magnitude to a 0..1 health score.

Provenance summary mirrors P18:
- ``measured``  — every contributing component is measured.
- ``estimated`` — every contributing component is estimated.
- ``mixed``     — at least one of each.

Confidence band:
- ``>= 0.8`` → ``high``
- ``>= 0.6`` → ``medium``
- else      → ``low``

Stable reason codes raised as ``ValueError``:
- ``deployment_version_not_found`` (404)
- ``score_not_found`` (404 — GET most-recent on a never-scored dv)
- ``export_not_found`` (400 — dv references a missing export)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.deployment_drift_check import DeploymentDriftCheck
from app.models.deployment_score import DeploymentScore
from app.models.deployment_telemetry import DeploymentTelemetrySample
from app.models.deployment_version import DeploymentVersion
from app.models.export import Export
from app.services.served_model_telemetry_service import compute_telemetry


# Nominal weights — each present component contributes its weight to a
# pool that is renormalised to sum to 1. Weights below were chosen so a
# perfectly-measured deployment (telemetry + drift + smoke) hits ~0.6
# weight on measured signals and ~0.4 on the static catalog signals.
_WEIGHTS: dict[str, float] = {
    "artifact_compat": 0.20,
    "target_compatibility": 0.20,
    "execute_smoke": 0.20,
    "telemetry_health": 0.20,
    "drift_health": 0.20,
}

_TELEMETRY_WINDOW_SECONDS = 3600
_HEALTHY_LATENCY_MS = 500.0
_UNHEALTHY_LATENCY_MS = 5000.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


async def _load_deployment_version(
    db: AsyncSession, *, deployment_version_id: int
) -> DeploymentVersion:
    result = await db.execute(
        select(DeploymentVersion).where(
            DeploymentVersion.id == deployment_version_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("deployment_version_not_found")
    return row


async def _load_export(db: AsyncSession, *, export_id: int) -> Export:
    result = await db.execute(select(Export).where(Export.id == export_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("export_not_found")
    return row


async def _latest_drift_check(
    db: AsyncSession, *, deployment_version_id: int
) -> DeploymentDriftCheck | None:
    result = await db.execute(
        select(DeploymentDriftCheck)
        .where(
            DeploymentDriftCheck.deployment_version_id == deployment_version_id
        )
        .order_by(DeploymentDriftCheck.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _telemetry_sample_count(
    db: AsyncSession, *, deployment_version_id: int
) -> int:
    # Cheap probe: just count rather than load every sample. Used for
    # confidence weighting; the actual aggregate goes through the
    # service so a future bucket/downsample lands in one place.
    result = await db.execute(
        select(DeploymentTelemetrySample.id).where(
            DeploymentTelemetrySample.deployment_version_id
            == deployment_version_id
        )
    )
    return len(result.scalars().all())


# ---------------------------------------------------------------------------
# Per-component computations
# ---------------------------------------------------------------------------


def _component_artifact_compat(*, manifest: dict[str, Any]) -> dict[str, Any]:
    deployment = manifest.get("deployment") if isinstance(manifest, dict) else None
    summary = (
        deployment.get("summary")
        if isinstance(deployment, dict) and isinstance(deployment.get("summary"), dict)
        else None
    )
    if not summary or "deployable_artifact" not in summary:
        return {
            "name": "artifact_compat",
            "score": None,
            "weight": _WEIGHTS["artifact_compat"],
            "provenance": "estimated",
            "confidence": 0.0,
            "signals": [],
            "summary": "No artifact validation report on this export.",
        }
    deployable = bool(summary.get("deployable_artifact"))
    artifact_validation = (
        deployment.get("artifact_validation")
        if isinstance(deployment.get("artifact_validation"), dict)
        else {}
    )
    return {
        "name": "artifact_compat",
        "score": 1.0 if deployable else 0.0,
        "weight": _WEIGHTS["artifact_compat"],
        "provenance": "estimated",
        "confidence": 0.85,
        "signals": [
            {
                "key": "deployable_artifact",
                "value": deployable,
                "ok": deployable,
            },
            {
                "key": "artifact_profile",
                "value": deployment.get("artifact_profile"),
                "ok": True,
            },
            {
                "key": "artifact_validation_passed",
                "value": bool(artifact_validation.get("passed")),
                "ok": bool(artifact_validation.get("passed")),
            },
        ],
        "summary": (
            "Artifact validation passed."
            if deployable
            else "Artifact validation failed for the export run."
        ),
    }


def _component_target_compatibility(
    *, manifest: dict[str, Any]
) -> dict[str, Any]:
    deployment = manifest.get("deployment") if isinstance(manifest, dict) else None
    target_reports = (
        deployment.get("target_reports")
        if isinstance(deployment, dict)
        and isinstance(deployment.get("target_reports"), list)
        else []
    )
    if not target_reports:
        return {
            "name": "target_compatibility",
            "score": None,
            "weight": _WEIGHTS["target_compatibility"],
            "provenance": "estimated",
            "confidence": 0.0,
            "signals": [],
            "summary": "No target reports on this export.",
        }
    compatible = sum(1 for r in target_reports if bool(r.get("compatible")))
    total = len(target_reports)
    score = compatible / total if total else 0.0
    return {
        "name": "target_compatibility",
        "score": score,
        "weight": _WEIGHTS["target_compatibility"],
        "provenance": "estimated",
        "confidence": 0.80,
        "signals": [
            {
                "key": "compatible_targets",
                "value": compatible,
                "ok": compatible == total,
            },
            {"key": "selected_targets", "value": total, "ok": total > 0},
        ],
        "summary": f"{compatible}/{total} selected targets compatible.",
    }


def _component_execute_smoke(
    *, manifest: dict[str, Any], deployment_version: DeploymentVersion
) -> dict[str, Any]:
    history_raw = (
        manifest.get("deploy_execution_history")
        if isinstance(manifest, dict)
        else None
    )
    history = list(history_raw or []) if isinstance(history_raw, list) else []

    # Filter to non-dry-run executes for this dv's target_id. The
    # manifest history doesn't store the dv id directly, so target id
    # is the strongest match we have.
    relevant = [
        entry
        for entry in history
        if isinstance(entry, dict)
        and not bool(entry.get("dry_run"))
        and str(entry.get("target_id") or "")
        == str(deployment_version.target_id or "")
    ]

    if relevant:
        successes = sum(
            1
            for e in relevant
            if str(e.get("status") or "").lower()
            in {"completed", "submitted", "running", "succeeded"}
        )
        total = len(relevant)
        score = successes / total if total else 0.0
        confidence = _clamp01(0.45 + 0.10 * total)  # caps at 1 by ~5 samples
        return {
            "name": "execute_smoke",
            "score": score,
            "weight": _WEIGHTS["execute_smoke"],
            "provenance": "measured",
            "confidence": confidence,
            "signals": [
                {"key": "successes", "value": successes, "ok": successes == total},
                {"key": "attempts", "value": total, "ok": True},
            ],
            "summary": f"{successes}/{total} non-dry-run executes succeeded.",
        }

    # Fall back to the deploy-target-suite's local smoke result. That's a
    # one-shot check at export-run time, not a real production execute,
    # so flag it as estimated rather than measured.
    deployment = manifest.get("deployment") if isinstance(manifest, dict) else None
    summary = (
        deployment.get("summary")
        if isinstance(deployment, dict) and isinstance(deployment.get("summary"), dict)
        else None
    )
    if summary and summary.get("local_smoke_passed") is not None:
        passed = bool(summary.get("local_smoke_passed"))
        return {
            "name": "execute_smoke",
            "score": 1.0 if passed else 0.0,
            "weight": _WEIGHTS["execute_smoke"],
            "provenance": "estimated",
            "confidence": 0.55,
            "signals": [
                {
                    "key": "local_smoke_passed",
                    "value": passed,
                    "ok": passed,
                },
                {
                    "key": "runner_smoke_success_count",
                    "value": summary.get("runner_smoke_success_count"),
                    "ok": True,
                },
            ],
            "summary": (
                "Local runner smoke test passed at export time."
                if passed
                else "Local runner smoke test failed at export time."
            ),
        }

    return {
        "name": "execute_smoke",
        "score": None,
        "weight": _WEIGHTS["execute_smoke"],
        "provenance": "estimated",
        "confidence": 0.0,
        "signals": [],
        "summary": "No execute history or local smoke result available.",
    }


def _component_telemetry_health(
    *, telemetry: dict[str, Any], sample_count: int
) -> dict[str, Any]:
    if sample_count <= 0 or int(telemetry.get("sample_count", 0)) <= 0:
        return {
            "name": "telemetry_health",
            "score": None,
            "weight": _WEIGHTS["telemetry_health"],
            "provenance": "measured",
            "confidence": 0.0,
            "signals": [],
            "summary": "No telemetry samples ingested yet.",
        }

    error_rate = float(telemetry.get("errors", {}).get("rate") or 0.0)
    error_score = _clamp01(1.0 - 10.0 * error_rate)

    latency = telemetry.get("latency_ms", {}) or {}
    p95 = float(latency.get("p95") or 0.0)
    if p95 <= _HEALTHY_LATENCY_MS:
        latency_score = 1.0
    elif p95 >= _UNHEALTHY_LATENCY_MS:
        latency_score = 0.0
    else:
        # Linear interpolation between the healthy and unhealthy bands.
        span = _UNHEALTHY_LATENCY_MS - _HEALTHY_LATENCY_MS
        latency_score = _clamp01(1.0 - (p95 - _HEALTHY_LATENCY_MS) / span)

    score = 0.7 * error_score + 0.3 * latency_score
    confidence = _clamp01(0.30 + 0.70 * min(sample_count / 100.0, 1.0))

    return {
        "name": "telemetry_health",
        "score": score,
        "weight": _WEIGHTS["telemetry_health"],
        "provenance": "measured",
        "confidence": confidence,
        "signals": [
            {
                "key": "error_rate",
                "value": error_rate,
                "ok": error_rate < 0.01,
            },
            {
                "key": "p95_latency_ms",
                "value": p95,
                "ok": p95 <= _HEALTHY_LATENCY_MS,
            },
            {
                "key": "sample_count",
                "value": sample_count,
                "ok": sample_count >= 30,
            },
        ],
        "summary": (
            f"p95={p95:.0f}ms, error_rate={error_rate:.2%}, "
            f"samples={sample_count}"
        ),
    }


def _component_drift_health(
    *, drift_check: DeploymentDriftCheck | None
) -> dict[str, Any]:
    if drift_check is None:
        return {
            "name": "drift_health",
            "score": None,
            "weight": _WEIGHTS["drift_health"],
            "provenance": "measured",
            "confidence": 0.0,
            "signals": [],
            "summary": "No drift checks have run yet.",
        }
    if drift_check.delta is None:
        # Drift run completed but had no baseline to compare against —
        # don't penalise the score and don't pretend we have a measure.
        return {
            "name": "drift_health",
            "score": None,
            "weight": _WEIGHTS["drift_health"],
            "provenance": "measured",
            "confidence": 0.0,
            "signals": [
                {
                    "key": "baseline_pass_rate",
                    "value": None,
                    "ok": False,
                },
                {
                    "key": "current_pass_rate",
                    "value": float(drift_check.current_pass_rate),
                    "ok": True,
                },
            ],
            "summary": "Drift check ran but no baseline eval result was found.",
        }
    delta_abs = abs(float(drift_check.delta))
    score = _clamp01(1.0 - 2.0 * delta_abs)
    samples = max(int(drift_check.samples_evaluated or 0), 0)
    confidence = _clamp01(0.30 + 0.70 * min(samples / 30.0, 1.0))
    return {
        "name": "drift_health",
        "score": score,
        "weight": _WEIGHTS["drift_health"],
        "provenance": "measured",
        "confidence": confidence,
        "signals": [
            {
                "key": "delta",
                "value": float(drift_check.delta),
                "ok": not bool(drift_check.drift_detected),
            },
            {
                "key": "drift_detected",
                "value": bool(drift_check.drift_detected),
                "ok": not bool(drift_check.drift_detected),
            },
            {
                "key": "samples_evaluated",
                "value": samples,
                "ok": samples >= 10,
            },
        ],
        "summary": (
            f"Δ pass rate {float(drift_check.delta):+.2f} "
            f"({'drift' if drift_check.drift_detected else 'within tolerance'})"
        ),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(components: list[dict[str, Any]]) -> dict[str, Any]:
    contributing = [c for c in components if c.get("score") is not None]
    if not contributing:
        return {
            "overall_score": 0.0,
            "confidence": 0.0,
            "provenance": "estimated",
            "components": components,
        }

    weight_total = sum(float(c["weight"]) for c in contributing)
    overall = 0.0
    confidence_sum = 0.0
    for component in contributing:
        normalised = (
            float(component["weight"]) / weight_total
            if weight_total > 0
            else 1.0 / len(contributing)
        )
        component["weight_normalised"] = normalised
        overall += float(component["score"]) * normalised
        confidence_sum += float(component.get("confidence") or 0.0) * normalised

    # Components with score=None still need weight_normalised=0 for symmetry.
    for component in components:
        component.setdefault("weight_normalised", 0.0)

    measured = {
        c["provenance"] for c in contributing if c.get("provenance") == "measured"
    }
    estimated = {
        c["provenance"] for c in contributing if c.get("provenance") == "estimated"
    }
    if measured and estimated:
        provenance = "mixed"
    elif measured:
        provenance = "measured"
    else:
        provenance = "estimated"

    return {
        "overall_score": _clamp01(overall),
        "confidence": _clamp01(confidence_sum),
        "provenance": provenance,
        "components": components,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def compute_score(
    db: AsyncSession,
    *,
    deployment_version_id: int,
    notes: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    export = await _load_export(db, export_id=dv.export_id)
    manifest = export.manifest if isinstance(export.manifest, dict) else {}

    # Telemetry: pull both the aggregate (for error rate + p95) and a
    # cheap row count (for confidence).
    sample_count = await _telemetry_sample_count(
        db, deployment_version_id=dv.id
    )
    telemetry_aggregate: dict[str, Any]
    if sample_count > 0:
        telemetry_aggregate = await compute_telemetry(
            db,
            deployment_version_id=dv.id,
            window_seconds=_TELEMETRY_WINDOW_SECONDS,
        )
    else:
        telemetry_aggregate = {"sample_count": 0}

    drift_check = await _latest_drift_check(
        db, deployment_version_id=dv.id
    )

    components = [
        _component_artifact_compat(manifest=manifest),
        _component_target_compatibility(manifest=manifest),
        _component_execute_smoke(manifest=manifest, deployment_version=dv),
        _component_telemetry_health(
            telemetry=telemetry_aggregate, sample_count=sample_count
        ),
        _component_drift_health(drift_check=drift_check),
    ]
    aggregated = _aggregate(components)

    actor_str = (actor or "system").strip()[:128] or "system"
    band = _confidence_band(float(aggregated["confidence"]))

    signals_summary = {
        "deployment_version_id": dv.id,
        "target_id": dv.target_id,
        "telemetry_sample_count": sample_count,
        "drift_check_id": drift_check.id if drift_check is not None else None,
        "drift_detected": (
            bool(drift_check.drift_detected)
            if drift_check is not None
            else None
        ),
        "components_present": [
            c["name"] for c in components if c.get("score") is not None
        ],
        "components_missing": [
            c["name"] for c in components if c.get("score") is None
        ],
    }

    row = DeploymentScore(
        deployment_version_id=dv.id,
        project_id=dv.project_id,
        overall_score=float(aggregated["overall_score"]),
        confidence=float(aggregated["confidence"]),
        provenance=str(aggregated["provenance"]),
        confidence_band=band,
        components=aggregated["components"],
        signals_summary=signals_summary,
        notes=notes,
        actor=actor_str,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return _serialize(row)


async def get_latest_score(
    db: AsyncSession, *, deployment_version_id: int
) -> dict[str, Any]:
    await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    result = await db.execute(
        select(DeploymentScore)
        .where(
            DeploymentScore.deployment_version_id == deployment_version_id
        )
        .order_by(DeploymentScore.created_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("score_not_found")
    return _serialize(row)


async def list_score_history(
    db: AsyncSession, *, deployment_version_id: int, limit: int = 50
) -> dict[str, Any]:
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    bounded = max(1, min(int(limit), 200))
    result = await db.execute(
        select(DeploymentScore)
        .where(DeploymentScore.deployment_version_id == dv.id)
        .order_by(DeploymentScore.created_at.desc())
        .limit(bounded)
    )
    rows = list(result.scalars().all())
    return {
        "deployment_version_id": dv.id,
        "limit": bounded,
        "scores": [_serialize(r) for r in rows],
    }


def _serialize(row: DeploymentScore) -> dict[str, Any]:
    return {
        "id": row.id,
        "deployment_version_id": row.deployment_version_id,
        "project_id": row.project_id,
        "overall_score": row.overall_score,
        "confidence": row.confidence,
        "confidence_band": row.confidence_band,
        "provenance": row.provenance,
        "components": list(row.components or []),
        "signals_summary": dict(row.signals_summary or {}),
        "notes": row.notes,
        "actor": row.actor,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }

"""Deployment versions + rollback + telemetry API (priority.md P25 + P26).

Routes (mounted at ``/api`` like every other router in this codebase):

- ``GET  /api/deployments/{deployment_version_id}`` — full record + audit.
- ``GET  /api/deployments/{deployment_version_id}/audit`` — audit log only.
- ``POST /api/deployments/{deployment_version_id}/promote`` — PENDING -> PROMOTED.
- ``POST /api/deployments/{deployment_version_id}/reject``  — PENDING -> REJECTED.
- ``POST /api/deployments/{deployment_version_id}/rollback`` — PROMOTED ->
  ROLLED_BACK; re-promotes the immediate predecessor in the same slot.
- ``GET  /api/projects/{project_id}/deployments`` — list rows for a
  project, optionally filtered by ``export_id`` / ``target_id`` / ``status``.
- ``POST /api/deployments/{deployment_version_id}/telemetry/ingest`` —
  push a batch of inference samples (P26).
- ``GET  /api/deployments/{deployment_version_id}/telemetry`` —
  windowed aggregate of latency p50/p95/p99 + error rate + request
  volume + token throughput (P26).
- ``GET  /api/deployments/{deployment_version_id}/telemetry/samples``
  — raw recent samples (capped) for debugging.

Stable reason codes from
:mod:`app.services.deployment_version_service` and
:mod:`app.services.served_model_telemetry_service` are surfaced as the
``detail`` body so callers can branch on them directly:
``deployment_version_not_found`` / ``project_not_found`` (404),
``not_promotable`` / ``not_rejectable`` / ``not_rollbackable`` /
``no_promoted_predecessor`` (409), ``samples_required`` /
``invalid_window`` (400).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.deployment_version_service import (
    get_deployment_version,
    list_audit_log,
    list_deployment_versions,
    promote_deployment_version,
    reject_deployment_version,
    rollback_deployment_version,
)
from app.services.served_model_telemetry_service import (
    compute_telemetry,
    ingest_samples,
    list_recent_samples,
)
from app.services.deployment_drift_service import (
    get_drift_check,
    list_drift_checks,
    run_drift_check,
)
from app.services.deployment_score_service import (
    compute_score,
    get_latest_score,
    list_score_history,
)


router = APIRouter(prefix="/deployments", tags=["Deployments"])
project_router = APIRouter(
    prefix="/projects/{project_id}/deployments", tags=["Deployments"]
)


_NOT_FOUND_CODES = {
    "deployment_version_not_found",
    "project_not_found",
    "gold_set_not_found",
    "gold_set_no_rows",
    "drift_check_not_found",
    "score_not_found",
}
_CONFLICT_CODES = {
    "not_promotable",
    "not_rejectable",
    "not_rollbackable",
    "no_promoted_predecessor",
}
_BAD_REQUEST_CODES = {
    "samples_required",
    "invalid_window",
    "invalid_status",
    "endpoint_or_predictions_required",
    "invalid_tolerance",
    "invalid_max_samples",
    "export_not_found",
}


def _raise_for(exc: ValueError) -> None:
    detail = str(exc) or "deployment_version_error"
    if detail in _NOT_FOUND_CODES:
        raise HTTPException(404, detail=detail) from exc
    if detail in _CONFLICT_CODES:
        raise HTTPException(409, detail=detail) from exc
    if detail in _BAD_REQUEST_CODES:
        raise HTTPException(400, detail=detail) from exc
    raise HTTPException(400, detail=detail) from exc


class DeploymentActionRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=2048)
    actor: str | None = Field(default=None, max_length=128)


@router.get("/{deployment_version_id}")
async def get_deployment(
    deployment_version_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await get_deployment_version(
            db, deployment_version_id=deployment_version_id
        )
    except ValueError as exc:
        _raise_for(exc)


@router.get("/{deployment_version_id}/audit")
async def get_deployment_audit(
    deployment_version_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await list_audit_log(
            db, deployment_version_id=deployment_version_id
        )
    except ValueError as exc:
        _raise_for(exc)


@router.post("/{deployment_version_id}/promote")
async def promote(
    deployment_version_id: int,
    req: DeploymentActionRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    payload = req or DeploymentActionRequest()
    try:
        return await promote_deployment_version(
            db,
            deployment_version_id=deployment_version_id,
            reason=payload.reason,
            actor=payload.actor,
        )
    except ValueError as exc:
        _raise_for(exc)


@router.post("/{deployment_version_id}/reject")
async def reject(
    deployment_version_id: int,
    req: DeploymentActionRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    payload = req or DeploymentActionRequest()
    try:
        return await reject_deployment_version(
            db,
            deployment_version_id=deployment_version_id,
            reason=payload.reason,
            actor=payload.actor,
        )
    except ValueError as exc:
        _raise_for(exc)


@router.post("/{deployment_version_id}/rollback")
async def rollback(
    deployment_version_id: int,
    req: DeploymentActionRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    payload = req or DeploymentActionRequest()
    try:
        return await rollback_deployment_version(
            db,
            deployment_version_id=deployment_version_id,
            reason=payload.reason,
            actor=payload.actor,
        )
    except ValueError as exc:
        _raise_for(exc)


@project_router.get("")
async def list_for_project(
    project_id: int,
    export_id: int | None = Query(default=None),
    target_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await list_deployment_versions(
            db,
            project_id=project_id,
            export_id=export_id,
            target_id=target_id,
            status=status,
        )
    except ValueError as exc:
        _raise_for(exc)


# ---------------------------------------------------------------------------
# Telemetry (P26)
# ---------------------------------------------------------------------------


class TelemetryIngestRequest(BaseModel):
    # Accept loosely-typed samples so a noisy collector doesn't have a
    # whole batch rejected at the request boundary — the service layer
    # validates each sample and surfaces per-row reasons in the response.
    samples: list[dict[str, Any]]


@router.post("/{deployment_version_id}/telemetry/ingest")
async def ingest_telemetry(
    deployment_version_id: int,
    req: TelemetryIngestRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await ingest_samples(
            db,
            deployment_version_id=deployment_version_id,
            samples=req.samples,
        )
    except ValueError as exc:
        _raise_for(exc)


@router.get("/{deployment_version_id}/telemetry")
async def get_telemetry(
    deployment_version_id: int,
    window_seconds: int | None = Query(default=None, ge=1, le=30 * 24 * 3600),
    since: str | None = Query(default=None),
    until: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await compute_telemetry(
            db,
            deployment_version_id=deployment_version_id,
            window_seconds=window_seconds,
            since=since,
            until=until,
        )
    except ValueError as exc:
        _raise_for(exc)


@router.get("/{deployment_version_id}/telemetry/samples")
async def get_telemetry_samples(
    deployment_version_id: int,
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await list_recent_samples(
            db,
            deployment_version_id=deployment_version_id,
            limit=limit,
        )
    except ValueError as exc:
        _raise_for(exc)


# ---------------------------------------------------------------------------
# Drift check (P27)
# ---------------------------------------------------------------------------


class DriftPredictionEntry(BaseModel):
    row_id: int
    prediction: Any | None = None


class DriftCheckRequest(BaseModel):
    gold_set_id: int
    eval_type: str = Field(default="exact_match", max_length=64)
    tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
    max_samples: int = Field(default=50, ge=1, le=500)
    # Offline mode: pass predictions directly. The service uses these
    # verbatim and skips the live HTTP call.
    predictions: list[DriftPredictionEntry] | None = None
    # Live mode: caller supplies the inference URL (and optional auth).
    endpoint_url: str | None = Field(default=None, max_length=2048)
    endpoint_headers: dict[str, str] | None = None
    notes: str | None = Field(default=None, max_length=2048)
    actor: str | None = Field(default=None, max_length=128)


@router.post("/{deployment_version_id}/drift/check")
async def trigger_drift_check(
    deployment_version_id: int,
    req: DriftCheckRequest,
    db: AsyncSession = Depends(get_db),
):
    predictions = (
        [p.model_dump(exclude_none=False) for p in req.predictions]
        if req.predictions is not None
        else None
    )
    try:
        return await run_drift_check(
            db,
            deployment_version_id=deployment_version_id,
            gold_set_id=req.gold_set_id,
            predictions=predictions,
            endpoint_url=req.endpoint_url,
            endpoint_headers=req.endpoint_headers,
            eval_type=req.eval_type,
            tolerance=req.tolerance,
            max_samples=req.max_samples,
            notes=req.notes,
            actor=req.actor,
        )
    except ValueError as exc:
        _raise_for(exc)


@router.get("/{deployment_version_id}/drift/checks")
async def list_deployment_drift_checks(
    deployment_version_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await list_drift_checks(
            db,
            deployment_version_id=deployment_version_id,
            limit=limit,
        )
    except ValueError as exc:
        _raise_for(exc)


@router.get("/drift/checks/{drift_check_id}")
async def get_deployment_drift_check(
    drift_check_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await get_drift_check(db, drift_check_id=drift_check_id)
    except ValueError as exc:
        _raise_for(exc)


# ---------------------------------------------------------------------------
# Deployability score (P28)
# ---------------------------------------------------------------------------


class ScoreComputeRequest(BaseModel):
    notes: str | None = Field(default=None, max_length=2048)
    actor: str | None = Field(default=None, max_length=128)


@router.post("/{deployment_version_id}/score/compute")
async def compute_deployment_score(
    deployment_version_id: int,
    req: ScoreComputeRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    payload = req or ScoreComputeRequest()
    try:
        return await compute_score(
            db,
            deployment_version_id=deployment_version_id,
            notes=payload.notes,
            actor=payload.actor,
        )
    except ValueError as exc:
        _raise_for(exc)


@router.get("/{deployment_version_id}/score")
async def get_deployment_score(
    deployment_version_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await get_latest_score(
            db, deployment_version_id=deployment_version_id
        )
    except ValueError as exc:
        _raise_for(exc)


@router.get("/{deployment_version_id}/score/history")
async def get_deployment_score_history(
    deployment_version_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await list_score_history(
            db,
            deployment_version_id=deployment_version_id,
            limit=limit,
        )
    except ValueError as exc:
        _raise_for(exc)

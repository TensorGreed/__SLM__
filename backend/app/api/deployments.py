"""Deployment versions + rollback API (priority.md P25).

Routes (mounted at ``/api`` like every other router in this codebase):

- ``GET  /api/deployments/{deployment_version_id}`` — full record + audit.
- ``GET  /api/deployments/{deployment_version_id}/audit`` — audit log only.
- ``POST /api/deployments/{deployment_version_id}/promote`` — PENDING -> PROMOTED.
- ``POST /api/deployments/{deployment_version_id}/reject``  — PENDING -> REJECTED.
- ``POST /api/deployments/{deployment_version_id}/rollback`` — PROMOTED ->
  ROLLED_BACK; re-promotes the immediate predecessor in the same slot.
- ``GET  /api/projects/{project_id}/deployments`` — list rows for a
  project, optionally filtered by ``export_id`` / ``target_id`` / ``status``.

Stable reason codes from
:mod:`app.services.deployment_version_service` are surfaced as the
``detail`` body so callers can branch on them directly:
``deployment_version_not_found`` / ``project_not_found`` (404),
``not_promotable`` / ``not_rejectable`` / ``not_rollbackable`` /
``no_promoted_predecessor`` (409).
"""

from __future__ import annotations

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


router = APIRouter(prefix="/deployments", tags=["Deployments"])
project_router = APIRouter(
    prefix="/projects/{project_id}/deployments", tags=["Deployments"]
)


_NOT_FOUND_CODES = {"deployment_version_not_found", "project_not_found"}
_CONFLICT_CODES = {
    "not_promotable",
    "not_rejectable",
    "not_rollbackable",
    "no_promoted_predecessor",
}


def _raise_for(exc: ValueError) -> None:
    detail = str(exc) or "deployment_version_error"
    if detail in _NOT_FOUND_CODES:
        raise HTTPException(404, detail=detail) from exc
    if detail in _CONFLICT_CODES:
        raise HTTPException(409, detail=detail) from exc
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

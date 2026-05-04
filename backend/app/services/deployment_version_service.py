"""Deployment version + rollback service (priority.md P25).

Public API:

- ``record_deployment_version`` — called from
  :func:`app.services.export_service.execute_export_deploy_plan` after a
  successful (non-dry-run) execute. Creates a ``PENDING``
  ``DeploymentVersion`` row + an audit row. Secrets in the request payload
  are stripped before persistence.
- ``promote_deployment_version`` — ``PENDING -> PROMOTED``. Any prior
  ``PROMOTED`` row for the same ``(export_id, target_id)`` is moved to
  ``SUPERSEDED`` in the same transaction so a single live version exists
  per deployment slot.
- ``reject_deployment_version`` — ``PENDING -> REJECTED``.
- ``rollback_deployment_version`` — ``PROMOTED -> ROLLED_BACK``. The most
  recent ``SUPERSEDED`` row for the same slot is re-promoted; if there is
  no such predecessor the operation refuses with
  ``no_promoted_predecessor`` so the slot never goes dark unintentionally.
- ``list_deployment_versions`` / ``get_deployment_version`` — read paths
  for the API + future UI.

Stable reason codes raised as ``ValueError`` (the API layer maps them to
HTTP):
- ``deployment_version_not_found`` (404)
- ``project_not_found`` (404)
- ``not_promotable`` (409 — current status forbids promote)
- ``not_rejectable`` (409 — current status forbids reject)
- ``not_rollbackable`` (409 — current status forbids rollback)
- ``no_promoted_predecessor`` (409 — nothing to roll back to)

Until Wave I (P41) lands real auth, ``actor`` is a free-form caller-
supplied string and defaults to ``"system"``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.deployment_version import (
    DeploymentRollback,
    DeploymentRollbackAction,
    DeploymentVersion,
    DeploymentVersionStatus,
)
from app.models.export import Export
from app.models.project import Project
from app.models.registry import ModelRegistryEntry


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Whitelist of plan_payload keys we are willing to persist. Anything not in
# this set is dropped — keeps tokens/credentials/secret URLs out of the DB.
_PLAN_PAYLOAD_SAFE_KEYS: frozenset[str] = frozenset(
    {
        "target_id",
        "target_kind",
        "endpoint_name",
        "region",
        "instance_type",
        "dry_run",
        "status",
        "started_at",
        "finished_at",
        "endpoint_url",
        "endpoint_handle",
        "model_name",
        "export_format",
        "run_id",
    }
)

# Substrings that flag a key as secret-bearing. Belt-and-braces with the
# whitelist above — if a future key sneaks in via the whitelist that
# happens to contain one of these tokens, it is still scrubbed.
_SECRET_TOKENS: tuple[str, ...] = ("token", "secret", "key", "password", "credential")


def _scrub_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``payload`` containing only safe, non-secret keys."""
    if not isinstance(payload, dict):
        return {}
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in _PLAN_PAYLOAD_SAFE_KEYS:
            continue
        if any(token in key.lower() for token in _SECRET_TOKENS):
            continue
        safe[key] = value
    return safe


def _normalise_actor(actor: str | None) -> str:
    cleaned = (actor or "").strip()
    return cleaned[:128] if cleaned else "system"


async def _ensure_project(db: AsyncSession, project_id: int) -> None:
    result = await db.execute(select(Project).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise ValueError("project_not_found")


async def _ensure_export(
    db: AsyncSession, *, project_id: int, export_id: int
) -> Export:
    result = await db.execute(
        select(Export).where(
            Export.id == export_id, Export.project_id == project_id
        )
    )
    export = result.scalar_one_or_none()
    if export is None:
        raise ValueError("export_not_found")
    return export


async def _next_version_for_export(db: AsyncSession, export_id: int) -> int:
    """Return the next monotonic version counter scoped by ``export_id``."""
    result = await db.execute(
        select(DeploymentVersion.version)
        .where(DeploymentVersion.export_id == export_id)
        .order_by(DeploymentVersion.version.desc())
        .limit(1)
    )
    last = result.scalar_one_or_none()
    return int(last or 0) + 1


async def _next_audit_sequence(
    db: AsyncSession, deployment_version_id: int
) -> int:
    result = await db.execute(
        select(DeploymentRollback.sequence)
        .where(DeploymentRollback.deployment_version_id == deployment_version_id)
        .order_by(DeploymentRollback.sequence.desc())
        .limit(1)
    )
    last = result.scalar_one_or_none()
    return int(last or 0) + 1


async def _append_audit_row(
    db: AsyncSession,
    *,
    deployment_version: DeploymentVersion,
    action: DeploymentRollbackAction,
    reason: str | None,
    actor: str,
    status_after: DeploymentVersionStatus,
    rolled_back_to_id: int | None = None,
    payload: dict[str, Any] | None = None,
) -> DeploymentRollback:
    sequence = await _next_audit_sequence(db, deployment_version.id)
    row = DeploymentRollback(
        deployment_version_id=deployment_version.id,
        project_id=deployment_version.project_id,
        sequence=sequence,
        action=action,
        reason=reason,
        actor=actor,
        status_after=status_after.value,
        rolled_back_to_id=rolled_back_to_id,
        payload=payload or {},
    )
    db.add(row)
    return row


def _serialize_deployment_version(row: DeploymentVersion) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "export_id": row.export_id,
        "registry_entry_id": row.registry_entry_id,
        "version": row.version,
        "target_id": row.target_id,
        "target_kind": row.target_kind,
        "endpoint_name": row.endpoint_name,
        "endpoint_handle": row.endpoint_handle,
        "region": row.region,
        "instance_type": row.instance_type,
        "status": row.status.value,
        "plan_payload": dict(row.plan_payload or {}),
        "promoted_reason": row.promoted_reason,
        "rejected_reason": row.rejected_reason,
        "rolled_back_reason": row.rolled_back_reason,
        "rolled_back_to_id": row.rolled_back_to_id,
        "actor": row.actor,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
        "rejected_at": row.rejected_at.isoformat() if row.rejected_at else None,
        "rolled_back_at": (
            row.rolled_back_at.isoformat() if row.rolled_back_at else None
        ),
        "superseded_at": (
            row.superseded_at.isoformat() if row.superseded_at else None
        ),
    }


def _serialize_audit_row(row: DeploymentRollback) -> dict[str, Any]:
    return {
        "id": row.id,
        "deployment_version_id": row.deployment_version_id,
        "project_id": row.project_id,
        "sequence": row.sequence,
        "action": row.action.value,
        "reason": row.reason,
        "actor": row.actor,
        "status_after": row.status_after,
        "rolled_back_to_id": row.rolled_back_to_id,
        "payload": dict(row.payload or {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ---------------------------------------------------------------------------
# Recording — called from the deploy-as-api execute path
# ---------------------------------------------------------------------------


async def record_deployment_version(
    db: AsyncSession,
    *,
    project_id: int,
    export_id: int,
    target_id: str,
    plan_payload: dict[str, Any] | None = None,
    target_kind: str | None = None,
    endpoint_name: str | None = None,
    endpoint_handle: str | None = None,
    region: str | None = None,
    instance_type: str | None = None,
    registry_entry_id: int | None = None,
    actor: str | None = None,
) -> DeploymentVersion:
    """Persist a ``PENDING`` deployment version + an audit row.

    Caller is responsible for *not* invoking this on dry-runs — the
    call site in ``export_service`` already gates on ``dry_run``.
    """
    await _ensure_project(db, project_id)
    await _ensure_export(db, project_id=project_id, export_id=export_id)

    version = await _next_version_for_export(db, export_id)
    safe_payload = _scrub_payload(plan_payload)

    actor_str = _normalise_actor(actor)

    row = DeploymentVersion(
        project_id=project_id,
        export_id=export_id,
        registry_entry_id=registry_entry_id,
        version=version,
        target_id=target_id,
        target_kind=target_kind,
        endpoint_name=endpoint_name,
        endpoint_handle=endpoint_handle,
        region=region,
        instance_type=instance_type,
        status=DeploymentVersionStatus.PENDING,
        plan_payload=safe_payload,
        actor=actor_str,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row


# ---------------------------------------------------------------------------
# Promote / reject / rollback
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


async def _supersede_existing_promoted(
    db: AsyncSession, *, dv: DeploymentVersion
) -> None:
    """Move any currently-promoted sibling for the same slot to ``SUPERSEDED``."""
    result = await db.execute(
        select(DeploymentVersion).where(
            DeploymentVersion.export_id == dv.export_id,
            DeploymentVersion.target_id == dv.target_id,
            DeploymentVersion.status == DeploymentVersionStatus.PROMOTED,
            DeploymentVersion.id != dv.id,
        )
    )
    siblings: Iterable[DeploymentVersion] = result.scalars().all()
    now = _utcnow()
    for sibling in siblings:
        sibling.status = DeploymentVersionStatus.SUPERSEDED
        sibling.superseded_at = now
        await _append_audit_row(
            db,
            deployment_version=sibling,
            action=DeploymentRollbackAction.PROMOTE,
            reason=f"superseded_by:{dv.id}",
            actor="system",
            status_after=DeploymentVersionStatus.SUPERSEDED,
            payload={"superseded_by_id": dv.id},
        )


async def promote_deployment_version(
    db: AsyncSession,
    *,
    deployment_version_id: int,
    reason: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """``PENDING -> PROMOTED``. Supersedes the prior live row in the same slot."""
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    if dv.status != DeploymentVersionStatus.PENDING:
        raise ValueError("not_promotable")

    await _supersede_existing_promoted(db, dv=dv)

    actor_str = _normalise_actor(actor)
    now = _utcnow()
    dv.status = DeploymentVersionStatus.PROMOTED
    dv.promoted_at = now
    dv.promoted_reason = reason
    dv.actor = actor_str

    if dv.registry_entry_id is not None:
        await _mark_registry_promoted(db, registry_entry_id=dv.registry_entry_id)

    audit = await _append_audit_row(
        db,
        deployment_version=dv,
        action=DeploymentRollbackAction.PROMOTE,
        reason=reason,
        actor=actor_str,
        status_after=DeploymentVersionStatus.PROMOTED,
    )
    await db.flush()
    await db.refresh(dv)
    return {
        "deployment_version": _serialize_deployment_version(dv),
        "audit": _serialize_audit_row(audit),
    }


async def reject_deployment_version(
    db: AsyncSession,
    *,
    deployment_version_id: int,
    reason: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """``PENDING -> REJECTED``. Cannot reject something already promoted."""
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    if dv.status != DeploymentVersionStatus.PENDING:
        raise ValueError("not_rejectable")

    actor_str = _normalise_actor(actor)
    now = _utcnow()
    dv.status = DeploymentVersionStatus.REJECTED
    dv.rejected_at = now
    dv.rejected_reason = reason
    dv.actor = actor_str

    audit = await _append_audit_row(
        db,
        deployment_version=dv,
        action=DeploymentRollbackAction.REJECT,
        reason=reason,
        actor=actor_str,
        status_after=DeploymentVersionStatus.REJECTED,
    )
    await db.flush()
    await db.refresh(dv)
    return {
        "deployment_version": _serialize_deployment_version(dv),
        "audit": _serialize_audit_row(audit),
    }


async def _find_predecessor(
    db: AsyncSession, *, dv: DeploymentVersion
) -> DeploymentVersion | None:
    """Most recent ``SUPERSEDED`` sibling for the same slot, by version desc."""
    result = await db.execute(
        select(DeploymentVersion)
        .where(
            DeploymentVersion.export_id == dv.export_id,
            DeploymentVersion.target_id == dv.target_id,
            DeploymentVersion.status == DeploymentVersionStatus.SUPERSEDED,
            DeploymentVersion.id != dv.id,
        )
        .order_by(DeploymentVersion.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def rollback_deployment_version(
    db: AsyncSession,
    *,
    deployment_version_id: int,
    reason: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """``PROMOTED -> ROLLED_BACK``. Re-promotes the immediate predecessor."""
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    if dv.status != DeploymentVersionStatus.PROMOTED:
        raise ValueError("not_rollbackable")

    predecessor = await _find_predecessor(db, dv=dv)
    if predecessor is None:
        raise ValueError("no_promoted_predecessor")

    actor_str = _normalise_actor(actor)
    now = _utcnow()

    dv.status = DeploymentVersionStatus.ROLLED_BACK
    dv.rolled_back_at = now
    dv.rolled_back_reason = reason
    dv.rolled_back_to_id = predecessor.id
    dv.actor = actor_str

    predecessor.status = DeploymentVersionStatus.PROMOTED
    predecessor.promoted_at = now
    predecessor.superseded_at = None
    predecessor.promoted_reason = (
        f"re-promoted_by_rollback_of:{dv.id}"
        if reason is None
        else f"{reason} (re-promoted_by_rollback_of:{dv.id})"
    )

    audit_rolled = await _append_audit_row(
        db,
        deployment_version=dv,
        action=DeploymentRollbackAction.ROLLBACK,
        reason=reason,
        actor=actor_str,
        status_after=DeploymentVersionStatus.ROLLED_BACK,
        rolled_back_to_id=predecessor.id,
        payload={"rolled_back_to_id": predecessor.id},
    )
    audit_repromoted = await _append_audit_row(
        db,
        deployment_version=predecessor,
        action=DeploymentRollbackAction.PROMOTE,
        reason=f"rollback_target_of:{dv.id}",
        actor=actor_str,
        status_after=DeploymentVersionStatus.PROMOTED,
        payload={"rollback_source_id": dv.id},
    )

    if predecessor.registry_entry_id is not None:
        await _mark_registry_promoted(
            db, registry_entry_id=predecessor.registry_entry_id
        )

    await db.flush()
    await db.refresh(dv)
    await db.refresh(predecessor)

    return {
        "rolled_back": _serialize_deployment_version(dv),
        "promoted": _serialize_deployment_version(predecessor),
        "audit": [
            _serialize_audit_row(audit_rolled),
            _serialize_audit_row(audit_repromoted),
        ],
    }


async def _mark_registry_promoted(
    db: AsyncSession, *, registry_entry_id: int
) -> None:
    """Best-effort sync of the legacy ``ModelRegistryEntry`` ``promoted_at``.

    The registry entry's ``stage`` enum has its own promotion semantics
    (candidate / staging / production) we don't try to map onto P25. We
    only stamp ``promoted_at`` so a registry-tab reader sees the most
    recent promotion timestamp.
    """
    result = await db.execute(
        select(ModelRegistryEntry).where(
            ModelRegistryEntry.id == registry_entry_id
        )
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        return
    entry.promoted_at = _utcnow()


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


async def get_deployment_version(
    db: AsyncSession, *, deployment_version_id: int
) -> dict[str, Any]:
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    audit_rows = await _list_audit_rows(db, deployment_version_id=dv.id)
    return {
        "deployment_version": _serialize_deployment_version(dv),
        "audit": [_serialize_audit_row(row) for row in audit_rows],
    }


async def list_deployment_versions(
    db: AsyncSession,
    *,
    project_id: int,
    export_id: int | None = None,
    target_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    await _ensure_project(db, project_id)
    stmt = select(DeploymentVersion).where(
        DeploymentVersion.project_id == project_id
    )
    if export_id is not None:
        stmt = stmt.where(DeploymentVersion.export_id == export_id)
    if target_id is not None:
        stmt = stmt.where(DeploymentVersion.target_id == target_id)
    if status is not None:
        try:
            status_enum = DeploymentVersionStatus(status)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("invalid_status") from exc
        stmt = stmt.where(DeploymentVersion.status == status_enum)
    stmt = stmt.order_by(
        DeploymentVersion.export_id.asc(), DeploymentVersion.version.asc()
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return {
        "project_id": project_id,
        "deployment_versions": [_serialize_deployment_version(r) for r in rows],
    }


async def _list_audit_rows(
    db: AsyncSession, *, deployment_version_id: int
) -> list[DeploymentRollback]:
    result = await db.execute(
        select(DeploymentRollback)
        .where(
            DeploymentRollback.deployment_version_id == deployment_version_id
        )
        .order_by(DeploymentRollback.sequence.asc())
    )
    return list(result.scalars().all())


async def list_audit_log(
    db: AsyncSession, *, deployment_version_id: int
) -> dict[str, Any]:
    # Verify the deployment version exists so callers get a clean 404 rather
    # than an empty list when the id is wrong.
    await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    rows = await _list_audit_rows(
        db, deployment_version_id=deployment_version_id
    )
    return {
        "deployment_version_id": deployment_version_id,
        "audit": [_serialize_audit_row(r) for r in rows],
    }

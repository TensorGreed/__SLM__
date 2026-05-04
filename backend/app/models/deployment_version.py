"""Deployment version + rollback audit (priority.md P25).

Today the deploy flow is ``Export -> deploy-as-api plan -> execute``, with
``ModelRegistryEntry`` carrying a coarse ``stage`` / ``deployment_status``.
That works for "this export was registered" but loses the history of
**which version is actually serving** — required for promote / reject /
rollback flows.

This file adds two append-friendly tables:

- ``deployment_versions`` — one row per executed deploy of an export to a
  target. Carries lifecycle status (pending / promoted / rejected /
  rolled_back / superseded), so the live serving slot for a given
  ``(export_id, target_id)`` is whichever row has ``status=promoted``.
- ``deployment_rollbacks`` — append-only audit log keyed by deployment
  version. Every promote / reject / rollback writes a row here in the
  same transaction as the status change.

Mirrors the audit shape introduced by P1 ``autopilot_decisions`` — same
``sequence`` + ``actor`` + ``payload JSON`` columns — so future timeline
work (Wave G P31) can union the two tables without bespoke shimming.

Wave I (P41+) adds real auth; until then ``actor`` is a free-form string
the caller can pass in (defaults to ``system``).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DeploymentVersionStatus(str, enum.Enum):
    PENDING = "pending"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"


class DeploymentRollbackAction(str, enum.Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    ROLLBACK = "rollback"


class DeploymentVersion(Base):
    __tablename__ = "deployment_versions"
    __table_args__ = (
        UniqueConstraint(
            "export_id",
            "version",
            name="uq_deployment_versions_export_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )
    export_id: Mapped[int] = mapped_column(
        ForeignKey("exports.id"), nullable=False, index=True
    )
    registry_entry_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_registry_entries.id"), nullable=True, index=True
    )

    # Monotonic version counter scoped by ``export_id``. The first executed
    # deploy of an export is v1, then v2, etc — independent of the SQL pk so
    # callers can talk about "deployment v3 of export 42" without exposing ids.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    target_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    target_kind: Mapped[str | None] = mapped_column(String(64), default=None)
    endpoint_name: Mapped[str | None] = mapped_column(String(256), default=None)
    endpoint_handle: Mapped[str | None] = mapped_column(String(512), default=None)
    region: Mapped[str | None] = mapped_column(String(64), default=None)
    instance_type: Mapped[str | None] = mapped_column(String(128), default=None)

    status: Mapped[DeploymentVersionStatus] = mapped_column(
        Enum(DeploymentVersionStatus),
        nullable=False,
        default=DeploymentVersionStatus.PENDING,
        index=True,
    )

    # Free-form metadata about the executed plan. Secrets MUST be stripped
    # by the caller before write — see ``record_deployment_version`` in the
    # service layer; we only keep target_id / endpoint / region / instance /
    # dry-run / status / timing.
    plan_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    # Reasons for terminal transitions, mirroring P2 snapshot-rollback shape.
    promoted_reason: Mapped[str | None] = mapped_column(Text, default=None)
    rejected_reason: Mapped[str | None] = mapped_column(Text, default=None)
    rolled_back_reason: Mapped[str | None] = mapped_column(Text, default=None)

    # Pointer to the deployment version this row was rolled back to. Only
    # set when ``status == ROLLED_BACK``; the *replacement* row that became
    # promoted is the destination, while this row is the one that came down.
    rolled_back_to_id: Mapped[int | None] = mapped_column(
        ForeignKey("deployment_versions.id"), nullable=True, index=True
    )

    actor: Mapped[str] = mapped_column(String(128), nullable=False, default="system")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    rejected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    rolled_back_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<DeploymentVersion id={self.id} export={self.export_id} "
            f"v{self.version} target={self.target_id!r} status={self.status.value}>"
        )


class DeploymentRollback(Base):
    __tablename__ = "deployment_rollbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment_version_id: Mapped[int] = mapped_column(
        ForeignKey("deployment_versions.id"), nullable=False, index=True
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    # Per-deployment-version monotonic counter, mirrors P1
    # ``autopilot_decisions.sequence`` so the audit ordering survives
    # clock skew on ``created_at``.
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    action: Mapped[DeploymentRollbackAction] = mapped_column(
        Enum(DeploymentRollbackAction), nullable=False, index=True
    )
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    actor: Mapped[str] = mapped_column(String(128), nullable=False, default="system")

    # Status snapshot at the moment of the audit row (the *new* status
    # after the action was applied). Stored as a string rather than the
    # enum so historic rows survive future enum extensions.
    status_after: Mapped[str | None] = mapped_column(String(64), default=None)

    # When ``action == ROLLBACK``, the version we rolled back **to** —
    # i.e. the row that became promoted. Null for promote / reject.
    rolled_back_to_id: Mapped[int | None] = mapped_column(
        ForeignKey("deployment_versions.id"), nullable=True, index=True
    )

    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<DeploymentRollback id={self.id} dv={self.deployment_version_id} "
            f"seq={self.sequence} action={self.action.value}>"
        )

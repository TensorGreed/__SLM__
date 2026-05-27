"""Gold-drift review queue — fresh hallucination traps awaiting triage (E4).

When the drift-check service refreshes a project's hallucination traps
(weekly automated or via the manual refresh endpoint), the new rows
land here pending user triage rather than going straight into
``gold_test``. The user accepts → promoted into the project's
gold_test JSONL; rejects → row stays in the queue for audit.

Each row carries the cluster pattern that motivated the trap so the
reviewer knows *why* it was generated. The payload field stores the
full recipe-shaped row (Q+A for qa-sft, text+label for classification,
etc.) so we don't constrain the schema to one recipe shape.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GoldDriftQueueStatus(str, enum.Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class GoldDriftReviewQueueRow(Base):
    __tablename__ = "gold_drift_review_queue"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )
    # Which drift check produced this row, when available. Manual
    # refresh-traps calls leave this null because they don't write a
    # DeploymentDriftCheck row themselves — they're project-scoped,
    # not deployment-scoped.
    source_drift_check_id: Mapped[int | None] = mapped_column(
        ForeignKey("deployment_drift_checks.id"),
        nullable=True,
        index=True,
    )
    # The cluster pattern that motivated this trap. Free-form strings
    # so a future cluster taxonomy can land without a migration.
    # Nullable because the first batch on a fresh project might have
    # nothing to point at (the runner falls back to recipe-shaped
    # defaults).
    cluster_reason_code: Mapped[str | None] = mapped_column(
        String(128), default=None
    )
    cluster_signature: Mapped[str | None] = mapped_column(
        String(64), default=None
    )
    # Recipe-shaped row payload — {question, answer, ...} for qa-sft,
    # {text, label, ...} for classification, etc. The triage-accept
    # endpoint reads this verbatim to construct the gold_test row.
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # ``rough`` for runner output, ``calibrated`` if we ever wire
    # human-verified prompts in. Future-proofing only — v1 writes
    # ``rough`` for everything.
    source_confidence: Mapped[str] = mapped_column(
        String(32), nullable=False, default="rough"
    )
    status: Mapped[GoldDriftQueueStatus] = mapped_column(
        Enum(GoldDriftQueueStatus),
        nullable=False,
        default=GoldDriftQueueStatus.PENDING,
        index=True,
    )
    triage_note: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )
    triaged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    def __repr__(self) -> str:
        return (
            f"<GoldDriftReviewQueueRow id={self.id} project={self.project_id} "
            f"status={self.status.value} cluster={self.cluster_reason_code}>"
        )

"""Persisted failure clusters (priority.md P33, Wave G).

One row per ``(project_id, stage, reason_code, signature)`` tuple.
Computed on demand from the ``run_events`` table by
:func:`failure_cluster_service.compute_failure_clusters` — call it
explicitly (operator-triggered or as a scheduled job; this codebase has
no built-in scheduler, so the API exposes a recompute endpoint).

The signature is a 12-char SHA1-derived fingerprint of the normalised
event summary. It distinguishes "training_runtime_error: CUDA OOM at
step 1200" from "training_runtime_error: dataloader corrupted" without
needing a separate enum.

Each cluster carries a small list of recent ``exemplar_event_ids`` so
the failure-analysis UI (P36) can deep-link without re-querying.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FailureCluster(Base):
    __tablename__ = "failure_clusters"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "stage",
            "reason_code",
            "signature",
            name="uq_failure_clusters_pid_stage_reason_sig",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    stage: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reason_code: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )
    signature: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )

    failure_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )

    # Recent exemplar event_ids (capped). Newest-first so a UI can
    # render "latest occurrences" without a separate query.
    exemplar_event_ids: Mapped[list] = mapped_column(JSON, default=list)
    # Parallel array — most recent summary text per exemplar so the UI
    # can show a label without round-tripping for each event.
    exemplar_summaries: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    last_computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<FailureCluster id={self.id} pid={self.project_id} "
            f"stage={self.stage} reason={self.reason_code} "
            f"sig={self.signature[:8]} count={self.failure_count}>"
        )

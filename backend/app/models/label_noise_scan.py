"""Quality-Lift phase 4 slice 1 — Label-noise scan row.

A ``LabelNoiseScan`` persists the result of a self-confidence scoring
sweep over the project's labeled rows. Slice 1 uses the cheap
heuristic — predict with the latest trained classifier, flag rows
where the model strongly disagrees with the given label (the dual
condition: predicted_prob ≥ confidence_threshold AND given_label_prob
≤ given_label_floor).

History matters: the user re-trains, fixes some rows, re-scans. The
dedicated table (rather than JSON on Project / Experiment) keeps
prior scans for compare-and-contrast and lets the Coach nudge gate on
"latest scan's label_count < 80% of current labeled_count" without
re-reading every project's _runtime blob.

The ``result_payload`` shape is locked alongside
``label_noise_scoring_service.scan_labeled_rows_for_mislabels`` —
slice 2's Coach nudge + Data Studio card read these fields directly.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LabelNoiseScanStatus(str, enum.Enum):
    """Lifecycle states. Mirrors Job's QUEUED/RUNNING/SUCCEEDED/FAILED
    so the Job runner can write transitions in lockstep with the
    underlying Job row — the API surfaces the scan status directly
    rather than re-deriving from job_id every time.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LabelNoiseScan(Base):
    __tablename__ = "label_noise_scans"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    # Which trained checkpoint scored the labels. Nullable because a
    # scan can fail to resolve a checkpoint at all (no completed
    # classification experiment yet) — the result_payload's
    # ``skipped_reason`` carries the diagnostic.
    base_experiment_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("experiments.id"),
        nullable=True,
        default=None,
        index=True,
    )

    status: Mapped[LabelNoiseScanStatus] = mapped_column(
        Enum(LabelNoiseScanStatus),
        nullable=False,
        default=LabelNoiseScanStatus.QUEUED,
    )

    # Snapshot of the project's labeled-row count AT SCAN TIME. The
    # Coach nudge re-fire rule gates on this: if (current_labeled -
    # latest_scan.label_count_at_scan) / current_labeled ≥ 0.20, the
    # user added a meaningful new batch and we should suggest re-scan.
    label_count_at_scan: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )
    # Denormalized count of result_payload.top_k entries — saves the
    # Coach nudge + listing endpoint from parsing the full JSON just
    # to gate "is there anything to review?".
    suspected_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )

    # The dual-condition thresholds the scan ran with. Persisted so a
    # user comparing two scans understands why the suspected counts
    # differ (lowering threshold → more suspects). Defaults match the
    # service's defaults but the API can override per scan.
    confidence_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.85
    )
    given_label_floor: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.15
    )

    # Full scan output. See label_noise_scoring_service for the locked
    # shape. Null until the runner writes it (RUNNING → SUCCEEDED).
    result_payload: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=None
    )

    # Captured on FAILED transitions so the UI can render a precise
    # message rather than "scan failed".
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )

    # FK to the Jobs framework's Job row. Nullable because cancelled
    # scans may not have had a job yet at cancellation time. The
    # notification bell renders the Job's progress while RUNNING.
    job_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("jobs.id"), nullable=True, default=None, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    # Relationships (no back_populates — keeps this model decoupled
    # from Project/Experiment/Job to avoid the model-init cycle that
    # bit slice 1 of phase 1 when the seed-group fields hit too many
    # downstream queries).
    project = relationship("Project", foreign_keys=[project_id])
    base_experiment = relationship(
        "Experiment", foreign_keys=[base_experiment_id]
    )

    def __repr__(self) -> str:
        return (
            f"<LabelNoiseScan id={self.id} project_id={self.project_id} "
            f"status={self.status.value} suspected={self.suspected_count}>"
        )

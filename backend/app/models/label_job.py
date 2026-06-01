"""Annotation foundation (Story 1.1) — label jobs + per-row work units.

A :class:`LabelJob` defines a labeling pass: task shape
(classification / span / preference_pair), label set, instructions,
target row count. :class:`LabelRow` rows are the per-row work units,
seeded from a source dataset, then handed out one at a time to a
reviewer who attaches a ``label_payload``.

Per the [[keep-brewslm-general]] memory, ``label_type`` is a generic
discriminator — adding a new task shape means adding a frontend
renderer + (optionally) a server-side scoring helper, not a new
hard-coded domain.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


LABEL_TYPE_CLASSIFICATION = "classification"
LABEL_TYPE_SPAN = "span"
LABEL_TYPE_PREFERENCE_PAIR = "preference_pair"

KNOWN_LABEL_TYPES: frozenset[str] = frozenset(
    {LABEL_TYPE_CLASSIFICATION, LABEL_TYPE_SPAN, LABEL_TYPE_PREFERENCE_PAIR}
)

JOB_STATUS_ACTIVE = "active"
JOB_STATUS_PAUSED = "paused"
JOB_STATUS_COMPLETED = "completed"

KNOWN_JOB_STATUSES: frozenset[str] = frozenset(
    {JOB_STATUS_ACTIVE, JOB_STATUS_PAUSED, JOB_STATUS_COMPLETED}
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LabelJob(Base):
    __tablename__ = "label_jobs"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(String(120), nullable=False)

    # 'classification' | 'span' | 'preference_pair'. String rather than
    # enum so plugins can register additional task shapes later without
    # an alembic migration. The annotation service rejects unknown
    # values at create-time against KNOWN_LABEL_TYPES.
    label_type: Mapped[str] = mapped_column(String(32), nullable=False)

    # Task-specific schema: classification carries ``allowed_labels``;
    # span carries ``span_types``; preference_pair carries no shape.
    label_schema: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Free-form reviewer-facing instructions (markdown).
    instructions: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )

    # 'active' | 'paused' | 'completed'. Active is the default and the
    # only state where assign_next will hand out new rows.
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=JOB_STATUS_ACTIVE
    )

    # Optional ceiling on how many rows we want labeled. Stats endpoint
    # uses this to render a progress bar; service does NOT auto-flip to
    # 'completed' on hitting target_rows (leaves that to the operator).
    target_rows: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )


class LabelRow(Base):
    __tablename__ = "label_rows"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    job_id: Mapped[int] = mapped_column(
        ForeignKey("label_jobs.id"), nullable=False, index=True
    )

    # Identity of the source row this label_row was seeded from. For
    # synthetic dataset rows we use the ``id`` field; for arbitrary
    # JSONL inputs we fall back to the line index. Nullable because
    # not every source row has a stable identity.
    source_row_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default=None
    )

    raw_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Reviewer assignment. Set by assign_next; cleared on skip; never
    # changed once labeled_at is set. Nullable because rows start
    # unassigned + we keep them unassigned after submit.
    assigned_to: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, default=None
    )
    assigned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    label_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, default=None
    )
    labeled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    reviewer_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )

    # Story 1.6 — set when this row has been materialized into the
    # project's synthetic / alignment training file via the promote
    # endpoint. Idempotency guard: a second promote call skips any
    # row with ``promoted_at`` already set, so re-runs never duplicate.
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    # Captures which downstream dataset the row landed in so the
    # operator can trace the path from a label_row through to a
    # training-time entry.
    promoted_to_dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id"), nullable=True, default=None
    )


class LabelRowReview(Base):
    """Multi-reviewer capture for inter-annotator agreement
    (Epic F Phase 2).

    Each :class:`LabelRow` gets one *primary* label (the row's
    ``label_payload`` / ``labeled_at`` / ``assigned_to`` fields are
    that single canonical view), but this side-table also records
    every individual reviewer's submission so we can compute
    Cohen's κ / span-F1 / preference-agreement when ≥2 distinct
    reviewers ever labeled the same row. The primary fields stay
    authoritative for promotion + UI so the historical labeling
    flow doesn't change.

    Insertion rule: every ``submit_label`` call inserts one row
    here keyed by (row, reviewer). Re-submissions by the same
    reviewer overwrite their prior review (the unique constraint
    below) so a labeler correcting themselves doesn't double-count
    in the agreement stats.
    """

    __tablename__ = "label_row_reviews"
    __table_args__ = (
        # A reviewer can only have one outstanding review per row;
        # re-submission updates the existing record. SQLite and
        # Postgres both treat NULL reviewer_id as distinct, but
        # in practice reviewer_id is required at insert time.
        UniqueConstraint(
            "row_id", "reviewer_id", name="uq_label_row_reviews_row_reviewer"
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    row_id: Mapped[int] = mapped_column(
        ForeignKey("label_rows.id"), nullable=False, index=True
    )
    job_id: Mapped[int] = mapped_column(
        ForeignKey("label_jobs.id"), nullable=False, index=True
    )
    # Nullable to allow auth-disabled local-dev to submit without
    # binding to a real user. Pairs/agreement math skips reviews
    # without a reviewer_id (it's not meaningful to attribute the
    # label to anyone in particular).
    reviewer_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, default=None, index=True
    )
    label_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    reviewer_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

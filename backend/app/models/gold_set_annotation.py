"""Gold-set annotation workbench tables — versions, rows, and reviewer queue.

A "gold set" in this codebase is an existing `Dataset` row with
`dataset_type ∈ {GOLD_DEV, GOLD_TEST}`. P10 layers annotation state on top:

- `gold_set_versions` — monotonic versions per gold set, draft-then-locked.
- `gold_set_rows` — individual rows with input / expected / rationale / labels
  and review status. Rows always live inside a version.
- `gold_set_reviewer_queue` — per-reviewer work queue, a projection of row
  assignment state maintained by `gold_workbench_service`.
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


class GoldSetVersionStatus(str, enum.Enum):
    DRAFT = "draft"
    LOCKED = "locked"


class GoldSetRowStatus(str, enum.Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class GoldSetReviewerQueueStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class GoldSetVersion(Base):
    __tablename__ = "gold_set_versions"
    __table_args__ = (
        UniqueConstraint("gold_set_id", "version", name="uq_gold_set_versions_gold_set_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gold_set_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[GoldSetVersionStatus] = mapped_column(
        Enum(GoldSetVersionStatus),
        default=GoldSetVersionStatus.DRAFT,
        nullable=False,
        index=True,
    )
    notes: Mapped[str] = mapped_column(Text, default="")
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class GoldSetRow(Base):
    __tablename__ = "gold_set_rows"
    __table_args__ = (
        # When `source_row_key` is set we want within-version dedup. SQLite and
        # Postgres both treat NULLs as distinct, so unset keys don't collide.
        UniqueConstraint(
            "version_id",
            "source_row_key",
            name="uq_gold_set_rows_version_source_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gold_set_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
        index=True,
    )
    version_id: Mapped[int] = mapped_column(
        ForeignKey("gold_set_versions.id"),
        nullable=False,
        index=True,
    )
    source_row_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=True,
    )
    input: Mapped[dict | None] = mapped_column(JSON, default=dict)
    expected: Mapped[dict | None] = mapped_column(JSON, default=dict)
    rationale: Mapped[str] = mapped_column(Text, default="")
    labels: Mapped[dict | None] = mapped_column(JSON, default=dict)
    reviewer_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    status: Mapped[GoldSetRowStatus] = mapped_column(
        Enum(GoldSetRowStatus),
        default=GoldSetRowStatus.PENDING,
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class GoldSetReviewerQueue(Base):
    __tablename__ = "gold_set_reviewer_queue"
    __table_args__ = (
        UniqueConstraint(
            "row_id",
            "reviewer_id",
            name="uq_gold_set_reviewer_queue_row_reviewer",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gold_set_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
        index=True,
    )
    row_id: Mapped[int] = mapped_column(
        ForeignKey("gold_set_rows.id"),
        nullable=False,
        index=True,
    )
    reviewer_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[GoldSetReviewerQueueStatus] = mapped_column(
        Enum(GoldSetReviewerQueueStatus),
        default=GoldSetReviewerQueueStatus.PENDING,
        nullable=False,
        index=True,
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

"""Saved dataset-import mapping configs (Phase G of DATASET_IMPORT_PLAN.md).

One row per (project, name) tuple. The wizard's "Save mapping" button +
the CLI/API "re-run from saved" entry point both read/write this
table. ``last_run_at`` / ``last_run_accepted`` get bumped on each
successful re-run so the UI can render "5,012 rows imported, last run
2 days ago" without joining against the RunEvent stream.
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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetImportConfig(Base):
    __tablename__ = "dataset_import_configs"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "name", name="uq_dataset_import_config_name"
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    # User-visible label. Unique within the project so accidental
    # re-saves don't silently clobber an existing config.
    name: Mapped[str] = mapped_column(String(120), nullable=False)

    # Free-text description — optional, lets the user explain *why*
    # this mapping exists ("PII training set — weekly refresh").
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The locator string is the exact input the user typed (e.g.
    # ``hf:Anthropic/hh-rlhf:train`` or ``jsonl:/abs/path/data.jsonl``).
    locator: Mapped[str] = mapped_column(Text, nullable=False)

    mapper_id: Mapped[str] = mapped_column(String(64), nullable=False)

    field_map: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # List of rejection reason codes the user opted to bulk-drop on
    # this saved mapping. Stored verbatim in JSON.
    drop_reasons: Mapped[list[str]] = mapped_column(
        JSON, default=list, nullable=False
    )

    # Optional row cap; mirrors the CLI's ``--limit``.
    limit: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

    # Bumped on each successful re-run. ``last_run_accepted`` is the
    # row count from the most recent ``run_import`` so the UI can show
    # the latest yield at a glance without joining run_events.
    last_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_run_accepted: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

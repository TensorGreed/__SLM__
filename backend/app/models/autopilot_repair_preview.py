"""Persisted repair previews for two-step autopilot apply.

A `POST /autopilot/repair-preview` writes a row holding the dry-run result and
a `plan_token`. A follow-up `POST /autopilot/repair-apply` looks the row up,
re-validates state, and replays the planned request with `dry_run=False`.

Introduced by priority.md P3 to separate planning from execution so the UI can
show an explicit diff and the user has to confirm before anything mutates.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


PREVIEW_TTL_MINUTES = 15


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _default_expires_at() -> datetime:
    return _utcnow() + timedelta(minutes=PREVIEW_TTL_MINUTES)


class AutopilotRepairPreview(Base):
    __tablename__ = "autopilot_repair_previews"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    plan_token: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    intent: Mapped[str | None] = mapped_column(Text, default=None)
    request_payload: Mapped[dict | None] = mapped_column(JSON, default=dict)
    config_diff: Mapped[dict | None] = mapped_column(JSON, default=dict)
    dry_run_response: Mapped[dict | None] = mapped_column(JSON, default=dict)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_default_expires_at, nullable=False, index=True
    )
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, index=True)
    applied_run_id: Mapped[str | None] = mapped_column(String(64), default=None)
    applied_by: Mapped[str | None] = mapped_column(String(64), default=None)
    applied_reason: Mapped[str | None] = mapped_column(Text, default=None)

    def __repr__(self) -> str:
        return (
            f"<AutopilotRepairPreview id={self.id} token={self.plan_token[:8]} "
            f"project={self.project_id} applied={self.applied_at is not None}>"
        )

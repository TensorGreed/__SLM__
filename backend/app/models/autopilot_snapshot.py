"""Pre-change snapshots captured by autopilot so decisions can be rolled back.

Each row is keyed by (`run_id`, `decision_sequence`) — the same (run, sequence)
pair used by `AutopilotDecision`. That pairing lets a rollback request carry
just a decision id (from the P1 decision-log API) and have the service look up
the matching snapshot without needing a second DB id indirection.

Introduced by priority.md P2 to unblock reversible autopilot actions
(experiment cancel, training cancel, project config restore) and to provide
the TTL-bounded storage required by the Story 4 acceptance criteria.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


SNAPSHOT_TTL_DAYS = 7


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _default_expires_at() -> datetime:
    return _utcnow() + timedelta(days=SNAPSHOT_TTL_DAYS)


class AutopilotSnapshot(Base):
    __tablename__ = "autopilot_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "decision_sequence",
            name="uq_autopilot_snapshots_run_sequence",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    decision_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id"), nullable=True, index=True
    )
    snapshot_type: Mapped[str] = mapped_column(String(64), nullable=False, default="autopilot_generic")
    pre_state: Mapped[dict | None] = mapped_column(JSON, default=dict)
    post_state: Mapped[dict | None] = mapped_column(JSON, default=dict)
    rollback_actions: Mapped[list | None] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_default_expires_at, nullable=False, index=True
    )
    restored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, index=True)
    restored_by: Mapped[str | None] = mapped_column(String(64), default=None)
    restored_reason: Mapped[str | None] = mapped_column(Text, default=None)
    restored_decision_id: Mapped[int | None] = mapped_column(
        ForeignKey("autopilot_decisions.id"), default=None
    )

    def __repr__(self) -> str:
        return (
            f"<AutopilotSnapshot id={self.id} run={self.run_id[:8]} "
            f"seq={self.decision_sequence} type={self.snapshot_type!r}>"
        )

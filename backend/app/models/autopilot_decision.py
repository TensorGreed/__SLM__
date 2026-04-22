"""Persisted autopilot decision-log entries.

One row per decision step emitted by the autopilot orchestrator
(`_orchestrate_newbie_autopilot_v2` and related call sites). Rows are grouped
by `run_id`, which identifies a single orchestration invocation, and ordered
within a run by `sequence`.

Introduced by SprintPlan Story 4 / priority.md P1 to unblock autopilot
rollback (P2) and the cross-stage observability work in Wave G.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AutopilotDecision(Base):
    __tablename__ = "autopilot_decisions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id"), nullable=True, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stage: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, default="info", index=True)
    reason_code: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    rationale: Mapped[str | None] = mapped_column(Text, default=None)
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    actor: Mapped[str] = mapped_column(String(64), nullable=False, default="autopilot")
    changed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    safe: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    blocker: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    intent: Mapped[str | None] = mapped_column(Text, default=None)
    payload: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<AutopilotDecision id={self.id} run={self.run_id[:8]} "
            f"stage={self.stage!r} status={self.status!r}>"
        )

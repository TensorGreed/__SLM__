"""Generic background job record (Hardening Phase H1).

Backs the Jobs framework that decouples long-running operations
(LLM-driven synth, model cloning, training, retrieval-eval) from
the HTTP request that started them. The user's UI subscribes via
the polled ``GET /api/jobs/active`` endpoint and renders progress
in the top-bar notification bell — Azure-portal pattern.

Persisted because the platform's long-running jobs (training,
auto-RAG comparison) can outlive a developer-side
``uvicorn`` restart. An in-memory-only design would lose state
exactly when the user needs it most.

Cancellation is *cooperative*: ``status=CANCELLED`` is a request;
the runner reads it at safe points. Phase H1 marks-only — wiring
each runner to honor the flag is a follow-on per migration.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    # Cancellation is cooperative — see module docstring.
    CANCELLED = "cancelled"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Discriminator: "synth_playbook" | "reroute_to_rag" |
    # "auto_rag_ab" | "training_start" | ... Used by the notification
    # bell to route the "click to open" action to the right surface.
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    # User-facing label, e.g. "Generate 30 hard_negatives for class
    # billing". Surfaced verbatim in the bell.
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus),
        nullable=False,
        default=JobStatus.QUEUED,
    )
    # 0.0–1.0 progress fraction. Nullable when the runner can't
    # estimate (e.g. an LLM call with unknown duration).
    progress: Mapped[float | None] = mapped_column(Float, default=None)
    progress_message: Mapped[str | None] = mapped_column(Text, default="")
    # Project scoping — most jobs are project-bound. Nullable for
    # cross-project / admin jobs.
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id"),
        default=None,
        nullable=True,
    )
    # User who started the job. Nullable when AUTH_ENABLED=false
    # (single-user local dev) and we don't have a principal.
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        default=None,
        nullable=True,
    )
    # Inputs the runner was given. Kept small (no payload data) for
    # observability + re-runnability.
    params: Mapped[dict | None] = mapped_column(JSON, default=dict)
    # Success payload — typically POINTERS (new resource ids, file
    # paths, summary stats) not bulk data. Never store rows / model
    # weights here; they go on disk / in their own tables.
    result: Mapped[dict | None] = mapped_column(JSON, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    # Set when the user dismisses a completed job from the bell.
    # Dismissed completed jobs are hidden from list_active but
    # remain queryable for audit.
    dismissed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    def __repr__(self) -> str:
        return f"<Job {self.id}: {self.kind} [{self.status.value}]>"

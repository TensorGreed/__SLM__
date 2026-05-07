"""Canonical RunEvent schema (priority.md P31, Wave G).

A unified, append-only event log spanning every stage in the pipeline:
ingestion, cleaning, adapter, training, eval, export, deployment,
autopilot, system. One row per "interesting thing happened" — start /
complete / fail / promote / reject etc.

This is the **single backbone** that powers Wave G's consumers:

- P32 unified timeline service joins these by ``project_id`` + the
  parent/root run_id pointer to render a tree.
- P33 reason-code taxonomy + nightly failure clustering groups
  ``severity == 'error'`` rows by ``(reason_code, stage, signature)``.
- P34 support-bundle service includes a recent slice of these rows in
  every bundle.
- Wave I P44 audit explorer reads these rows.

Design choices:

- **String columns for stage / severity / reason_code** rather than
  Enum types. P33 will define the canonical taxonomy and add a lint
  rule that every error raise sets a code; storing strings means we
  can grow the taxonomy without an enum migration each time.
- **``run_id`` is a string**, not an int FK. Different stages use
  different identity schemes (``exp-42`` for training experiments,
  ``deploy-19`` for deployment versions, hex tokens for autopilot
  runs) and we want them all in one namespaced column.
- **``parent_run_id``** lets a training run nested under an autopilot
  invocation link upward without forcing a join through a separate
  parent table. P32 builds the tree from this.
- **Append-only.** Rows are never updated; correction events are new
  rows that reference the old one in ``payload``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Canonical stage names. Kept here as constants so callers reference a
# stable string and the lint rule (P33) can grep for unauthorised values.
STAGE_INGESTION = "ingestion"
STAGE_CLEANING = "cleaning"
STAGE_ADAPTER = "adapter"
STAGE_TRAINING = "training"
STAGE_EVAL = "eval"
STAGE_EXPORT = "export"
STAGE_DEPLOYMENT = "deployment"
STAGE_AUTOPILOT = "autopilot"
STAGE_SYSTEM = "system"

KNOWN_STAGES: frozenset[str] = frozenset({
    STAGE_INGESTION,
    STAGE_CLEANING,
    STAGE_ADAPTER,
    STAGE_TRAINING,
    STAGE_EVAL,
    STAGE_EXPORT,
    STAGE_DEPLOYMENT,
    STAGE_AUTOPILOT,
    STAGE_SYSTEM,
})

# Severity levels in increasing order. ``critical`` is reserved for
# events that warrant operator paging.
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITY_CRITICAL = "critical"

KNOWN_SEVERITIES: frozenset[str] = frozenset({
    SEVERITY_INFO,
    SEVERITY_WARNING,
    SEVERITY_ERROR,
    SEVERITY_CRITICAL,
})


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Project the event belongs to — every observability surface is
    # project-scoped, and ``GET /projects/{id}/run-events`` filters here.
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    # The unifying id for the operation that emitted this event. Format
    # is ``<stage>-<id>`` by convention (``exp-42`` / ``deploy-7`` /
    # ``autopilot-{hex}``). String rather than int FK because no single
    # parent table exists.
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Optional parent — present when the emitting op was started by
    # another op (e.g. training run launched via autopilot). P32 walks
    # this pointer to build the timeline tree.
    parent_run_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )

    stage: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default=SEVERITY_INFO, index=True
    )

    # Stable reason code (P33 will define the taxonomy + lint rule).
    # Always set on ``severity in {error, critical}``; optional otherwise.
    reason_code: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default=None, index=True
    )

    actor: Mapped[str] = mapped_column(String(128), nullable=False, default="system")

    # Short human-readable summary (one line). Free-form for now;
    # frontend uses this as the row label in the timeline.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    # Structured details. Whatever the emitting service finds useful —
    # kept as a JSON blob so we don't have to add columns per stage.
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    # Wall-clock time the event occurred. Indexed for the
    # ``?since=`` filter on the timeline endpoint.
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    # Persisted insert time. Almost always equal to ``ts``, but kept
    # separate so a back-dated event (replaying logs) reports the
    # original timestamp on ``ts`` while ``created_at`` reflects when
    # we actually wrote the row.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<RunEvent id={self.id} run={self.run_id!r} "
            f"stage={self.stage} severity={self.severity}>"
        )

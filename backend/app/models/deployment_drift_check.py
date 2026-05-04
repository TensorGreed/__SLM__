"""On-demand drift-check runs against a deployed endpoint (priority.md P27).

A row is created every time an operator triggers ``POST
/deployments/{id}/drift/check``. We persist the resolved baseline
metric, the current metric measured against the live endpoint (or
caller-supplied offline predictions), the delta, and whether the
configured tolerance was exceeded — so the deployment-assistant UI
(P30) can render a drift trend without re-running the eval.

Per-row scoring detail is stamped in the ``per_row_results`` JSON,
capped at the first 100 rows to keep a single check from ballooning
into many KB. The full eval set lives in the gold-set tables (P10) and
can be reconstructed from there if needed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
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


class DeploymentDriftCheck(Base):
    __tablename__ = "deployment_drift_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment_version_id: Mapped[int] = mapped_column(
        ForeignKey("deployment_versions.id"), nullable=False, index=True
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    gold_set_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id"), nullable=True, index=True
    )
    gold_set_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("gold_set_versions.id"), nullable=True, index=True
    )

    # Baseline pointers — the EvalResult we compared against, if any. Both
    # are nullable because a freshly-deployed export with no training-time
    # eval result is allowed (the check still runs, just with `delta=null`).
    baseline_experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id"), nullable=True, index=True
    )
    baseline_eval_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("eval_results.id"), nullable=True, index=True
    )

    eval_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="exact_match", index=True
    )
    baseline_pass_rate: Mapped[float | None] = mapped_column(Float, default=None)
    current_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    delta: Mapped[float | None] = mapped_column(Float, default=None)
    tolerance: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    drift_detected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )

    samples_evaluated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    samples_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    samples_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ``offline`` = caller supplied predictions; ``live_url`` = service called
    # an HTTP endpoint per row. Stored as a string so future modes
    # (provider-SDK, vLLM-managed) plug in without an enum migration.
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="offline")

    notes: Mapped[str | None] = mapped_column(Text, default=None)
    actor: Mapped[str] = mapped_column(String(128), nullable=False, default="system")

    # Per-row breakdown, capped. Each entry: row_id, expected, prediction,
    # match (bool), error (str|None).
    per_row_results: Mapped[list] = mapped_column(JSON, default=list)

    # Pre-aggregated summary so the UI doesn't have to re-derive on
    # every fetch (totals, error breakdown, mean latency if measured).
    summary: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<DeploymentDriftCheck id={self.id} "
            f"dv={self.deployment_version_id} drift={self.drift_detected} "
            f"current={self.current_pass_rate}>"
        )

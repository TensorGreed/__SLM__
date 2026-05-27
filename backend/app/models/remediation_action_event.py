"""Remediation action events — track whether suggested fixes worked (E2).

One row per user interaction with a suggested-action button (forecast
panel + failure-cluster cards). Pairs with the post-eval pipeline
which stamps each event with the lift in pass_rate between the eval
just-completed and the previous eval for the same project.

The pair of tables (this + ``forecast_calibration_observations`` from
T5) gives the full feedback loop: T5 measures whether predictions
match reality at the run level; E2 measures whether individual
remediation suggestions correlated with improvements at the
suggestion level. Together they're the evidence base for tuning the
heuristic + the suggestion bot.

Schema is intentionally permissive — ``action_kind`` is free-form
string so a new suggestion source can land without a migration. The
canonical kinds today are the four forecast-panel kinds
(``synth_augment``, ``synth_balance``, ``synth_diversify``,
``fix_gold_rows``) plus ``cluster_fix`` from the failure-cluster
"Fix in gold set" button (E1).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RemediationOutcome(str, enum.Enum):
    CLICKED = "clicked"
    DISMISSED = "dismissed"
    APPLIED = "applied"
    IGNORED = "ignored"


class RemediationActionEvent(Base):
    __tablename__ = "remediation_action_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )
    # Free-form because new suggestion sources should land without a
    # migration. v1 values: synth_augment, synth_balance,
    # synth_diversify, fix_gold_rows, cluster_fix.
    action_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Stable hash of (kind + sorted params JSON), 16 hex chars. Lets
    # re-clicks of the same suggestion collapse in the aggregation
    # without storing the full params payload again.
    params_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    outcome: Mapped[RemediationOutcome] = mapped_column(
        Enum(RemediationOutcome),
        nullable=False,
        default=RemediationOutcome.CLICKED,
    )
    # Lift in pass_rate (percentage points) between the eval that
    # resolved this event and the previous eval for the same project.
    # NULL until ``stamp_evaluation_lift`` runs.
    evaluation_lift_pct: Mapped[float | None] = mapped_column(Float, default=None)
    # The experiment whose eval landed the lift stamp. NULL while
    # unresolved.
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id"),
        nullable=True,
        default=None,
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
    )

    def __repr__(self) -> str:
        return (
            f"<RemediationActionEvent {self.id} project={self.project_id} "
            f"{self.action_kind}/{self.outcome.value} lift={self.evaluation_lift_pct}>"
        )

"""Forecast vs reality calibration observations (USER-SUCCESS Epic 1, T5).

One row per training experiment, pairing the user's most-recent
trainability-forecast snapshot at launch time with the post-eval
gate-pass verdict. The pair lets us answer "for runs we predicted at
60-70% confidence, what fraction actually passed?" — the data the
heuristic in ``trainability_forecast_service`` needs to be tuned with
evidence rather than vibes.

The observation is recorded at experiment creation (so we capture
exactly what the user saw on the Training Config page before
committing) and resolved when ``evaluate_experiment_auto_gates`` runs,
which is the canonical "did this run clear gates" moment. ``actual_passed``
stays NULL until that resolution lands — null rows are excluded from
calibration aggregation so the bucket counts only ever reflect
resolved pairs.

Recipe + predicted_confidence_pct + predicted_overall are denormalized
off the snapshot so admin aggregation can run with a single index
read; the snapshot_id stays for forensic drill-down.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ForecastCalibrationObservation(Base):
    __tablename__ = "forecast_calibration_observations"
    # One observation per experiment — there's exactly one
    # forecast→reality pair per training run. Unique constraint
    # rather than primary-key-on-experiment because we still want a
    # synthetic id for admin filtering / audit cursors.
    __table_args__ = (
        UniqueConstraint("experiment_id", name="uq_forecast_obs_experiment_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("training_forecast_snapshots.id"),
        nullable=False,
    )
    # Denormalized from the snapshot for aggregation speed (avoids a
    # join in the calibration query). The snapshot itself is the
    # source of truth for forensic drill-down.
    predicted_confidence_pct: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_overall: Mapped[str] = mapped_column(String(32), nullable=False)
    # Denormalized from the project's recipe at launch time. Stored
    # rather than read-through because a user changing recipes after
    # training would otherwise re-bucket historical observations
    # into the wrong recipe.
    recipe_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # NULL until evaluate_experiment_auto_gates resolves. Aggregations
    # MUST exclude null rows so the bucket counts only ever reflect
    # resolved pairs.
    actual_passed: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, default=None
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
    )

    def __repr__(self) -> str:
        outcome = (
            "pending" if self.actual_passed is None
            else "pass" if self.actual_passed else "fail"
        )
        return (
            f"<ForecastCalibrationObservation exp={self.experiment_id} "
            f"{self.predicted_overall}@{self.predicted_confidence_pct}% "
            f"actual={outcome}>"
        )

"""Trainability forecast snapshot — per-compute history (USER-SUCCESS Epic 1, T2).

Persists one row every time ``trainability_forecast_service.forecast_training``
recomputes the forecast (cache-miss path only). The history powers the
sparkline + last-3 verdict-delta strip in ``TrainabilityForecastPanel``
so the user can see whether a gold-set edit / synth run moved the
``confidence_pct`` needle.

Retention is 60 days — long enough to span typical iteration cycles
without unbounded growth. Pruning runs alongside each insert (cheap,
no separate cron required).

We deliberately store ``signals`` as the same JSON shape the API
returns so the frontend can render hover tooltips against historical
data without a second roundtrip / version-skew worry.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainingForecastSnapshot(Base):
    __tablename__ = "training_forecast_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )
    # The forecast-service cache key (truncated SHA-256 over
    # dataset_signature + recipe_id + base_model_name). Repeated keys
    # in the snapshot stream mean "same inputs, recomputed" — the UI
    # may collapse those when rendering the sparkline.
    cache_key: Mapped[str] = mapped_column(String(32), nullable=False)
    # When the forecast was computed (mirrors ForecastResult.computed_at).
    # Separate from created_at so retention/ordering can rely on the
    # logical compute time rather than DB insert time.
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    overall: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence_pct: Mapped[int] = mapped_column(Integer, nullable=False)
    # Full ForecastSignal[] payload at this point in time. Storing the
    # full list (not just severity counts) lets the panel render a
    # tooltip per snapshot without a second roundtrip.
    signals: Mapped[list | None] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingForecastSnapshot p={self.project_id} "
            f"{self.overall}@{self.confidence_pct}% {self.computed_at.isoformat()}>"
        )

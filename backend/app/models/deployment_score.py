"""Deployability score (priority.md P28).

A deployability score is a single 0..1 blend of measured smoke-test
outcomes (deploy execute history, telemetry, drift) and estimated
compatibility signals (artifact validation, target compatibility from
the deploy-target suite). Every component carries its own ``score``,
``weight``, ``provenance`` (``measured | estimated``), ``confidence``,
and a ``signals`` breakdown so the deployment-assistant UI (P30) can
explain the headline number.

We persist on every ``POST /score/compute`` rather than computing
on-demand from the API:

- the ingredients (telemetry, drift checks, manifest blobs) are not
  monotonic, so a recomputed score on the same dv id is not a stable
  function of its inputs over time;
- a persisted history is what P30's trend chart and P29's CLI both want
  to read.

Mirror P18's cost-estimator provenance shape (``measured | estimated``,
``confidence_band``) so the UI can reuse the same badges.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
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


class DeploymentScore(Base):
    __tablename__ = "deployment_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment_version_id: Mapped[int] = mapped_column(
        ForeignKey("deployment_versions.id"), nullable=False, index=True
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # ``measured`` — every contributing component is measured.
    # ``estimated`` — every contributing component is estimated.
    # ``mixed``    — at least one of each.
    provenance: Mapped[str] = mapped_column(
        String(32), nullable=False, default="estimated", index=True
    )
    confidence_band: Mapped[str] = mapped_column(
        String(16), nullable=False, default="low"
    )

    # Each component:
    #   {
    #     "name": "telemetry_health",
    #     "score": 0.95,
    #     "weight": 0.20,
    #     "weight_normalised": 0.25,   # weight after dropping null-score components
    #     "provenance": "measured",
    #     "confidence": 0.85,
    #     "signals": [{"key": "error_rate", "value": 0.005, "ok": true}],
    #     "summary": "p95=120ms, error_rate=0.5%"
    #   }
    components: Mapped[list] = mapped_column(JSON, default=list)

    # Top-level summary so the dashboard can render headline metrics
    # without re-walking the components blob.
    signals_summary: Mapped[dict] = mapped_column(JSON, default=dict)

    notes: Mapped[str | None] = mapped_column(Text, default=None)
    actor: Mapped[str] = mapped_column(String(128), nullable=False, default="system")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<DeploymentScore id={self.id} dv={self.deployment_version_id} "
            f"score={self.overall_score:.3f} provenance={self.provenance}>"
        )

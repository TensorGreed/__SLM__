"""Post-deploy telemetry samples (priority.md P26).

One row per inference request reported by a deployed endpoint (push) or
scraped from a provider's metrics API (pull, future). Samples are scoped
to a :class:`DeploymentVersion` so rollback of v2 still leaves v1's
historical performance addressable.

Aggregations (latency p50/p95/p99, error rate, request volume, token
throughput) are computed on demand in
:mod:`app.services.served_model_telemetry_service` rather than persisted
— callers can trade off window size against compute, and we don't have to
choose a bucket granularity at write time.
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
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DeploymentTelemetrySample(Base):
    __tablename__ = "deployment_telemetry_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment_version_id: Mapped[int] = mapped_column(
        ForeignKey("deployment_versions.id"), nullable=False, index=True
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    # Caller-supplied timestamp; defaults to server-now when missing so a
    # provider-side scrape that does not carry per-request timestamps
    # still lands on a coherent window.
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status_code: Mapped[int | None] = mapped_column(Integer, default=None)
    error_code: Mapped[str | None] = mapped_column(String(128), default=None)

    # Token counts are best-effort — many provider APIs report only one
    # of input/output, so both columns are nullable.
    input_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    output_tokens: Mapped[int | None] = mapped_column(Integer, default=None)

    # Optional per-sample correlation id from the inference layer. Useful
    # to dedup retries and to deep-link from the timeline (Wave G P32).
    request_id: Mapped[str | None] = mapped_column(String(128), default=None)

    # Free-form provider-specific blob — kept under a stable key for the
    # support-bundle service (P34) to scoop into bundles.
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<DeploymentTelemetrySample id={self.id} "
            f"dv={self.deployment_version_id} latency={self.latency_ms}ms "
            f"success={self.success}>"
        )

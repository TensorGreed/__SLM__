"""Serving-gateway tables: scoped API keys and a per-request log.

Neither table ever holds prompt or reply text: inputs may be private code.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GatewayApiKey(Base):
    """A key for the ``/v1`` gateway, scoped to a list of projects. Only the
    sha256 of the key is stored; the plaintext is shown once at creation."""

    __tablename__ = "gateway_api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    key_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False, default="")
    project_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[str | None] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<GatewayApiKey {self.id}: {self.key_prefix} active={self.is_active}>"


class GatewayRequestLog(Base):
    """One gateway request: timing, status and token counts only."""

    __tablename__ = "gateway_request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # ORM-level link only (SQLite cannot ADD CONSTRAINT later).
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    key_id: Mapped[int | None] = mapped_column(Integer, default=None)
    model_id: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    run_id: Mapped[int | None] = mapped_column(Integer, default=None)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False, index=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status_code: Mapped[int | None] = mapped_column(Integer, default=None)
    error_code: Mapped[str | None] = mapped_column(String(128), default=None)
    input_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    output_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    retrieval_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    guided_json_mode: Mapped[str | None] = mapped_column(String(32), default=None)
    upstream: Mapped[str] = mapped_column(String(32), nullable=False, default="in_process")
    request_id: Mapped[str | None] = mapped_column(String(128), default=None)

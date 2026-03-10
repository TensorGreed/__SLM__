"""Project-scoped chat playground session model."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PlaygroundSession(Base):
    __tablename__ = "playground_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), default="Untitled Session")
    provider: Mapped[str] = mapped_column(String(64), default="mock")
    model_name: Mapped[str] = mapped_column(String(512), default="")
    api_url: Mapped[str | None] = mapped_column(String(2048), default=None)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    temperature: Mapped[float] = mapped_column(Float, default=0.2)
    max_tokens: Mapped[int] = mapped_column(Integer, default=512)
    transcript: Mapped[list | None] = mapped_column(JSON, default=list)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<PlaygroundSession {self.id}: project={self.project_id} "
            f"provider={self.provider} model={self.model_name}>"
        )

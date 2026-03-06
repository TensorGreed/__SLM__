"""Typed artifact registry ORM model."""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactStatus(str, enum.Enum):
    MATERIALIZED = "materialized"
    FAILED = "failed"


class ArtifactRecord(Base):
    __tablename__ = "artifact_records"
    __table_args__ = (
        UniqueConstraint("project_id", "artifact_key", "version", name="uq_artifact_project_key_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    artifact_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ArtifactStatus] = mapped_column(
        Enum(ArtifactStatus),
        default=ArtifactStatus.MATERIALIZED,
        nullable=False,
    )
    uri: Mapped[str | None] = mapped_column(String(2048), default="")
    schema_ref: Mapped[str] = mapped_column(String(255), default="slm.artifact/v1")
    producer_stage: Mapped[str | None] = mapped_column(String(64), default=None)
    producer_run_id: Mapped[str | None] = mapped_column(String(128), default=None)
    producer_step_id: Mapped[str | None] = mapped_column(String(255), default=None)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    def __repr__(self) -> str:
        return (
            f"<ArtifactRecord {self.id}: project={self.project_id} "
            f"{self.artifact_key}@v{self.version} [{self.status.value}]>"
        )

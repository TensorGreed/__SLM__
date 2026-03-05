"""Model registry ORM models for lifecycle and promotion tracking."""

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RegistryStage(str, enum.Enum):
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DeploymentStatus(str, enum.Enum):
    NOT_DEPLOYED = "not_deployed"
    DEPLOYED = "deployed"
    FAILED = "failed"


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry_entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    export_id: Mapped[int | None] = mapped_column(ForeignKey("exports.id"), default=None)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), default="v1")
    stage: Mapped[RegistryStage] = mapped_column(
        Enum(RegistryStage),
        default=RegistryStage.CANDIDATE,
        nullable=False,
    )
    deployment_status: Mapped[DeploymentStatus] = mapped_column(
        Enum(DeploymentStatus),
        default=DeploymentStatus.NOT_DEPLOYED,
        nullable=False,
    )
    artifact_path: Mapped[str | None] = mapped_column(String(1024), default="")
    readiness: Mapped[dict | None] = mapped_column(JSON, default=dict)
    deployment: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    def __repr__(self) -> str:
        return f"<ModelRegistryEntry {self.id}: {self.name}@{self.version} [{self.stage.value}]>"

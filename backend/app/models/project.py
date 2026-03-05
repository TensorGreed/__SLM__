"""Project ORM model — top-level entity for the SLM pipeline."""

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class PipelineStage(str, enum.Enum):
    """Ordered pipeline stages a project moves through."""

    INGESTION = "ingestion"
    CLEANING = "cleaning"
    GOLD_SET = "gold_set"
    SYNTHETIC = "synthetic"
    DATASET_PREP = "dataset_prep"
    TOKENIZATION = "tokenization"
    TRAINING = "training"
    EVALUATION = "evaluation"
    COMPRESSION = "compression"
    EXPORT = "export"
    COMPLETED = "completed"


class ProjectStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default="")
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus), default=ProjectStatus.DRAFT
    )
    pipeline_stage: Mapped[PipelineStage] = mapped_column(
        Enum(PipelineStage), default=PipelineStage.INGESTION
    )
    base_model_name: Mapped[str | None] = mapped_column(String(255), default="")
    domain_pack_id: Mapped[int | None] = mapped_column(
        ForeignKey("domain_packs.id"),
        default=None,
    )
    domain_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("domain_profiles.id"),
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="project", cascade="all, delete-orphan")
    domain_pack = relationship("DomainPack", back_populates="projects")
    domain_profile = relationship("DomainProfile", back_populates="projects")

    def __repr__(self) -> str:
        return f"<Project {self.id}: {self.name} [{self.pipeline_stage.value}]>"

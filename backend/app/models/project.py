"""Project ORM model — top-level entity for the SLM pipeline."""

import enum
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, JSON, String, Text
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
    training_preferred_plan_profile: Mapped[str | None] = mapped_column(
        String(32),
        default="balanced",
    )
    target_profile_id: Mapped[str | None] = mapped_column(
        String(64),
        default="vllm_server",
    )
    evaluation_preferred_pack_id: Mapped[str | None] = mapped_column(
        String(128),
        default=None,
    )
    beginner_mode: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    active_domain_blueprint_version: Mapped[int | None] = mapped_column(
        default=None,
    )
    dataset_adapter_preset: Mapped[dict | None] = mapped_column(
        JSON,
        default=dict,
    )
    gate_policy: Mapped[dict | None] = mapped_column(
        JSON,
        default=lambda: {
            "must_pass": False,
            "min_score": 0.0,
            "blocked_if_missing": False,
        }
    )
    budget_settings: Mapped[dict | None] = mapped_column(
        JSON,
        default=lambda: {
            "monthly_cap": 0.0,
            "current_spend": 0.0,
            "alert_threshold": 0.8,
            "auto_cancel": True
        }
    )
    # Gamification progression state (Lab Journal). Nullable so
    # existing projects round-trip unchanged; the service
    # materializes the canonical empty shape on first read.
    gamification: Mapped[dict | None] = mapped_column(
        JSON,
        default=None,
        nullable=True,
    )
    # Snapshot of the task-shape recipe (Theme 2) the user picked
    # at first-dataset-import time. Nullable so projects created
    # before the picker existed survive untouched; populated by
    # `recipe_apply_service.apply_recipe_to_project`.
    selected_recipe: Mapped[dict | None] = mapped_column(
        JSON,
        default=None,
        nullable=True,
    )
    # Per-project state for the project-guide quickstart tour
    # nudges (Theme 1 Epic 2). Carries `dismissed_nudges: list[str]`
    # so the floating "what just happened, do this next" callouts
    # don't replay once the user has seen them. Nullable so
    # existing projects round-trip unchanged.
    quickstart_tour_state: Mapped[dict | None] = mapped_column(
        JSON,
        default=None,
        nullable=True,
    )
    # Cached trainability forecast (USER-SUCCESS Epic 1).
    # Computed by `trainability_forecast_service.forecast_training`;
    # keyed on (dataset_version, recipe_id, base_model_name) so the
    # diversity-score embed pass is reused across reads. Invalidates
    # when any cache-key input changes. Nullable so existing
    # projects round-trip unchanged.
    training_forecast_cache: Mapped[dict | None] = mapped_column(
        JSON,
        default=None,
        nullable=True,
    )
    # USER-SUCCESS Epic 7 Phase 7b — provenance back-link for RAG
    # clone projects. When a project was created via
    # ``rag_project_service.clone_project_for_rag``, this points at
    # the source project's id so the UI can render a "← cloned from"
    # chip and downstream comparisons can pair the two. Nullable
    # (self-referential FK with no cascade) so non-clone projects
    # round-trip unchanged.
    parent_project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id"),
        default=None,
        nullable=True,
    )
    # USER-SUCCESS Epic 7 Phase 7b — runtime feature flags that the
    # playground inference path consults. Today carries
    # ``rag_first`` (bool — when true, playground uses the base
    # model + auto-RAG preamble and the training-start endpoint
    # refuses requests) and a mirrored ``auto_rag.enabled`` value.
    # Nullable so existing projects round-trip unchanged.
    runtime_config: Mapped[dict | None] = mapped_column(
        JSON,
        default=None,
        nullable=True,
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
    domain_blueprints = relationship(
        "DomainBlueprintRevision",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Project {self.id}: {self.name} [{self.pipeline_stage.value}]>"

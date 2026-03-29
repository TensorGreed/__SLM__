"""Domain blueprint ORM model for beginner-mode project planning."""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DomainBlueprintStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class DomainBlueprintRevision(Base):
    __tablename__ = "domain_blueprints"
    __table_args__ = (
        UniqueConstraint("project_id", "version", name="uq_domain_blueprints_project_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[DomainBlueprintStatus] = mapped_column(
        Enum(DomainBlueprintStatus),
        default=DomainBlueprintStatus.DRAFT,
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(String(64), default="manual")
    brief_text: Mapped[str] = mapped_column(Text, default="")

    domain_name: Mapped[str] = mapped_column(String(255), default="")
    problem_statement: Mapped[str] = mapped_column(Text, default="")
    target_user_persona: Mapped[str] = mapped_column(Text, default="")
    task_family: Mapped[str] = mapped_column(String(64), default="instruction_sft")
    input_modality: Mapped[str] = mapped_column(String(64), default="text")

    expected_output_schema: Mapped[dict | None] = mapped_column(JSON, default=dict)
    expected_output_examples: Mapped[list | None] = mapped_column(JSON, default=list)
    safety_compliance_notes: Mapped[list | None] = mapped_column(JSON, default=list)
    deployment_target_constraints: Mapped[dict | None] = mapped_column(JSON, default=dict)
    success_metrics: Mapped[list | None] = mapped_column(JSON, default=list)
    glossary: Mapped[list | None] = mapped_column(JSON, default=list)
    unresolved_assumptions: Mapped[list | None] = mapped_column(JSON, default=list)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)

    analysis_metadata: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )

    project = relationship("Project", back_populates="domain_blueprints")

    def __repr__(self) -> str:
        return f"<DomainBlueprintRevision project={self.project_id} v{self.version} [{self.status.value}]>"

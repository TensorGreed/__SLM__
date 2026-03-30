"""Versioned non-Python dataset adapter definitions for Adapter Studio."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetAdapterDefinition(Base):
    __tablename__ = "dataset_adapter_definitions"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "adapter_name",
            "version",
            name="uq_dataset_adapter_definitions_project_name_version",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id"),
        nullable=True,
        index=True,
    )
    adapter_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")

    source_type: Mapped[str] = mapped_column(String(64), nullable=False, default="raw")
    source_ref: Mapped[str | None] = mapped_column(Text, default=None)
    base_adapter_id: Mapped[str] = mapped_column(String(128), nullable=False, default="default-canonical")
    task_profile: Mapped[str | None] = mapped_column(String(64), default=None)

    field_mapping: Mapped[dict | None] = mapped_column(JSON, default=dict)
    adapter_config: Mapped[dict | None] = mapped_column(JSON, default=dict)
    output_contract: Mapped[dict | None] = mapped_column(JSON, default=dict)
    schema_profile: Mapped[dict | None] = mapped_column(JSON, default=dict)
    inference_summary: Mapped[dict | None] = mapped_column(JSON, default=dict)
    validation_report: Mapped[dict | None] = mapped_column(JSON, default=dict)
    export_template: Mapped[dict | None] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<DatasetAdapterDefinition id={self.id} project_id={self.project_id} "
            f"name={self.adapter_name!r} v{self.version}>"
        )

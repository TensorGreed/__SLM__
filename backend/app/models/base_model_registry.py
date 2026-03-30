"""Universal base model registry ORM model."""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BaseModelSourceType(str, enum.Enum):
    HUGGINGFACE = "huggingface"
    LOCAL_PATH = "local_path"
    CATALOG = "catalog"


class BaseModelRegistryEntry(Base):
    __tablename__ = "base_model_registry_entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    source_type: Mapped[BaseModelSourceType] = mapped_column(
        Enum(BaseModelSourceType),
        nullable=False,
        index=True,
    )
    source_ref: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)

    model_family: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    architecture: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    tokenizer: Mapped[str | None] = mapped_column(String(255), default=None)
    chat_template: Mapped[str | None] = mapped_column(Text, default=None)
    context_length: Mapped[int | None] = mapped_column(Integer, default=None)
    parameter_count: Mapped[int | None] = mapped_column(Integer, default=None)
    params_estimate_b: Mapped[float | None] = mapped_column(Float, default=None)
    license: Mapped[str | None] = mapped_column(String(128), default=None, index=True)

    modalities: Mapped[list | None] = mapped_column(JSON, default=list)
    quantization_support: Mapped[dict | None] = mapped_column(JSON, default=dict)
    peft_support: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    full_finetune_support: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    supported_task_families: Mapped[list | None] = mapped_column(JSON, default=list)
    training_mode_support: Mapped[list | None] = mapped_column(JSON, default=list)
    estimated_hardware_needs: Mapped[dict | None] = mapped_column(JSON, default=dict)
    deployment_target_compatibility: Mapped[list | None] = mapped_column(JSON, default=list)

    normalization_contract_version: Mapped[str] = mapped_column(
        String(128),
        default="slm.base_model_registry/v1",
        nullable=False,
    )
    normalized_metadata: Mapped[dict | None] = mapped_column(JSON, default=dict)
    provenance: Mapped[dict | None] = mapped_column(JSON, default=dict)
    cache_fingerprint: Mapped[str | None] = mapped_column(String(128), default=None)
    cache_status: Mapped[str] = mapped_column(String(32), default="fresh", nullable=False)
    refresh_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )

    def __repr__(self) -> str:
        return f"<BaseModelRegistryEntry {self.id}: {self.model_key}>"

"""Export ORM model for tracking model exports."""

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExportFormat(str, enum.Enum):
    GGUF = "gguf"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    HUGGINGFACE = "huggingface"
    DOCKER = "docker"


class ExportStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Export(Base):
    __tablename__ = "exports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    experiment_id: Mapped[int | None] = mapped_column(ForeignKey("experiments.id"), default=None)
    export_format: Mapped[ExportFormat] = mapped_column(Enum(ExportFormat), nullable=False)
    status: Mapped[ExportStatus] = mapped_column(
        Enum(ExportStatus), default=ExportStatus.PENDING
    )
    quantization: Mapped[str | None] = mapped_column(String(32), default=None)
    output_path: Mapped[str | None] = mapped_column(String(1024), default="")
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, default=None)
    manifest: Mapped[dict | None] = mapped_column(JSON, default=dict)
    eval_report: Mapped[dict | None] = mapped_column(JSON, default=dict)
    safety_scorecard: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    def __repr__(self) -> str:
        return f"<Export {self.id}: {self.export_format.value} [{self.status.value}]>"

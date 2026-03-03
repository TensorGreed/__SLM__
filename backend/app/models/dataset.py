"""Dataset, DatasetVersion, and RawDocument ORM models."""

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetType(str, enum.Enum):
    RAW = "raw"
    CLEANED = "cleaned"
    GOLD_DEV = "gold_dev"
    GOLD_TEST = "gold_test"
    SYNTHETIC = "synthetic"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_type: Mapped[DatasetType] = mapped_column(Enum(DatasetType), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default="")
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str | None] = mapped_column(String(1024), default="")
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, default=dict)
    is_locked: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Relationships
    project = relationship("Project", back_populates="datasets")
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")
    documents = relationship("RawDocument", back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Dataset {self.id}: {self.name} ({self.dataset_type.value})>"


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    manifest: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    dataset = relationship("Dataset", back_populates="versions")

    def __repr__(self) -> str:
        return f"<DatasetVersion {self.dataset_id}v{self.version}>"


class RawDocument(Base):
    __tablename__ = "raw_documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str | None] = mapped_column(String(255), default="upload")
    sensitivity: Mapped[str | None] = mapped_column(String(64), default="internal")
    license_info: Mapped[str | None] = mapped_column(String(255), default="")
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus), default=DocumentStatus.PENDING)
    quality_score: Mapped[float | None] = mapped_column(default=None)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    dataset = relationship("Dataset", back_populates="documents")

    def __repr__(self) -> str:
        return f"<RawDocument {self.id}: {self.filename}>"

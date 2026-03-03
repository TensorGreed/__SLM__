"""Pydantic schemas for Dataset, DatasetVersion, and RawDocument APIs."""

from datetime import datetime
from pydantic import BaseModel, Field

from app.models.dataset import DatasetType, DocumentStatus


# ── Dataset ─────────────────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dataset_type: DatasetType
    description: str = ""


class DatasetResponse(BaseModel):
    id: int
    project_id: int
    name: str
    dataset_type: DatasetType
    description: str | None
    record_count: int
    file_path: str | None
    is_locked: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── RawDocument ─────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size_bytes: int
    status: DocumentStatus
    ingested_at: datetime

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    id: int
    dataset_id: int
    filename: str
    file_type: str
    file_size_bytes: int
    source: str | None
    sensitivity: str | None
    status: DocumentStatus
    quality_score: float | None
    chunk_count: int
    ingested_at: datetime

    model_config = {"from_attributes": True}


# ── DatasetVersion ──────────────────────────────────────────────────────

class DatasetVersionResponse(BaseModel):
    id: int
    dataset_id: int
    version: int
    record_count: int
    created_at: datetime

    model_config = {"from_attributes": True}

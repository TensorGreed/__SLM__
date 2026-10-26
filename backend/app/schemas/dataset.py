"""Pydantic schemas for Dataset, DatasetVersion, and RawDocument APIs."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, model_validator, Field

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
    # Surfaced from ``metadata_``: why processing failed (user-facing), and
    # whether this is a row file whose columns are kept through cleaning.
    error: str | None = None
    structured: bool = False
    row_count: int | None = None
    # Provenance of documents pulled from a website / git repo (``None`` for
    # uploads): where it came from, and for a repo file the pinned commit.
    license_info: str | None = None
    source_url: str | None = None
    path: str | None = None
    commit: str | None = None
    content_kind: str | None = None
    language: str | None = None

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def _lift_metadata(cls, data: Any) -> Any:
        meta = data.get("metadata_") if isinstance(data, dict) else getattr(data, "metadata_", None)
        if not isinstance(data, dict):
            data = {name: getattr(data, name) for name in cls.model_fields if hasattr(data, name)}
        meta = meta if isinstance(meta, dict) else {}
        return {
            **data,
            "error": data.get("error") or meta.get("error"),
            "structured": bool(data.get("structured") or meta.get("structured")),
            "row_count": data.get("row_count") or meta.get("row_count") or meta.get("rows_kept"),
            "license_info": data.get("license_info") or None,
            "source_url": data.get("source_url") or meta.get("source_url"),
            "path": data.get("path") or meta.get("path"),
            "commit": data.get("commit") or meta.get("commit"),
            "content_kind": data.get("content_kind") or meta.get("content_kind"),
            "language": data.get("language") or meta.get("language"),
        }


# ── DatasetVersion ──────────────────────────────────────────────────────

class DatasetVersionResponse(BaseModel):
    id: int
    dataset_id: int
    version: int
    record_count: int
    created_at: datetime

    model_config = {"from_attributes": True}

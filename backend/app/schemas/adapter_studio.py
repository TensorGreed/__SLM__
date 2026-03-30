"""Pydantic contracts for Dataset Structure Explorer + Adapter Studio APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetSourceDescriptor(BaseModel):
    source_type: str = Field(default="project_dataset", min_length=1, max_length=64)
    source_ref: str | None = Field(default=None, max_length=4096)
    path: str | None = Field(default=None, max_length=4096)
    split: str | None = Field(default=None, max_length=128)
    dataset_type: str | None = Field(default=None, max_length=64)
    document_id: int | None = Field(default=None, ge=1)


class AdapterStudioProfileRequest(BaseModel):
    source: DatasetSourceDescriptor = Field(default_factory=DatasetSourceDescriptor)
    sample_size: int = Field(default=500, ge=10, le=5000)


class AdapterStudioInferRequest(BaseModel):
    source: DatasetSourceDescriptor = Field(default_factory=DatasetSourceDescriptor)
    sample_size: int = Field(default=400, ge=10, le=5000)
    task_profile: str | None = Field(default=None, max_length=64)


class AdapterStudioPreviewRequest(BaseModel):
    source: DatasetSourceDescriptor = Field(default_factory=DatasetSourceDescriptor)
    adapter_id: str = Field(default="auto", min_length=1, max_length=128)
    field_mapping: dict[str, str] = Field(default_factory=dict)
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    task_profile: str | None = Field(default=None, max_length=64)
    sample_size: int = Field(default=300, ge=10, le=5000)
    preview_limit: int = Field(default=25, ge=5, le=100)


class AdapterStudioValidateRequest(AdapterStudioPreviewRequest):
    pass


class AdapterStudioSaveRequest(BaseModel):
    adapter_name: str = Field(..., min_length=1, max_length=128)
    source_type: str = Field(default="project_dataset", min_length=1, max_length=64)
    source_ref: str | None = Field(default=None, max_length=4096)
    base_adapter_id: str = Field(default="default-canonical", min_length=1, max_length=128)
    task_profile: str | None = Field(default=None, max_length=64)
    field_mapping: dict[str, str] = Field(default_factory=dict)
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    schema_profile: dict[str, Any] = Field(default_factory=dict)
    inference_summary: dict[str, Any] = Field(default_factory=dict)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    share_globally: bool = False


class AdapterStudioExportRequest(BaseModel):
    export_dir: str | None = Field(default=None, max_length=4096)


class AdapterDefinitionResponse(BaseModel):
    id: int
    project_id: int | None
    adapter_name: str
    version: int
    status: str
    source_type: str
    source_ref: str | None
    base_adapter_id: str
    task_profile: str | None
    field_mapping: dict[str, str] = Field(default_factory=dict)
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    schema_profile: dict[str, Any] = Field(default_factory=dict)
    inference_summary: dict[str, Any] = Field(default_factory=dict)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    export_template: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


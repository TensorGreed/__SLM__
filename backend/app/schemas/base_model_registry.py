"""Schemas for universal base model registry + compatibility APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class BaseModelImportRequest(BaseModel):
    source_type: str = Field(..., min_length=1, max_length=32)
    source_ref: str = Field(..., min_length=1, max_length=2048)
    allow_network: bool = True
    overwrite: bool = True


class BaseModelRefreshRequest(BaseModel):
    model_id: int | None = Field(default=None, ge=1)
    model_key: str | None = Field(default=None, min_length=1, max_length=255)
    allow_network: bool = True

    @model_validator(mode="after")
    def _validate_selector(self) -> "BaseModelRefreshRequest":
        if self.model_id is None and not str(self.model_key or "").strip():
            raise ValueError("Provide model_id or model_key.")
        return self


class BaseModelListResponse(BaseModel):
    count: int
    models: list[dict[str, Any]] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)


class CompatibilityReason(BaseModel):
    code: str
    severity: str
    message: str
    unblock_actions: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ModelCompatibilityResponse(BaseModel):
    project_id: int
    model_id: int
    model_key: str
    compatibility_score: float = Field(ge=0.0, le=1.0)
    compatible: bool
    reason_codes: list[str] = Field(default_factory=list)
    reasons: list[CompatibilityReason] = Field(default_factory=list)
    why_recommended: list[CompatibilityReason] = Field(default_factory=list)
    why_risky: list[CompatibilityReason] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] | None = None
    generated_at: datetime


class ModelValidateRequest(BaseModel):
    model_id: int | None = Field(default=None, ge=1)
    model_key: str | None = Field(default=None, min_length=1, max_length=255)
    dataset_adapter_id: str | None = Field(default=None, max_length=128)
    runtime_id: str | None = Field(default=None, max_length=128)
    target_profile_id: str | None = Field(default=None, max_length=128)
    allow_network: bool = False

    @model_validator(mode="after")
    def _validate_selector(self) -> "ModelValidateRequest":
        if self.model_id is None and not str(self.model_key or "").strip():
            raise ValueError("Provide model_id or model_key.")
        return self


class ModelExplainRequest(ModelValidateRequest):
    pass


class ModelRecommendResponse(BaseModel):
    project_id: int
    count: int
    compatible_count: int
    models: list[ModelCompatibilityResponse] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)

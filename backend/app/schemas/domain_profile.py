"""Pydantic schemas for domain profile contracts and assignment."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.domain_profile import DomainProfileStatus


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


class DomainTaskSpec(BaseModel):
    task_id: str = Field(..., min_length=2, max_length=64)
    output_mode: Literal["text", "label", "json", "tool_call"] = "text"
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    @field_validator("task_id")
    @classmethod
    def normalize_task_id(cls, value: str) -> str:
        token = _normalize_token(value)
        if not token:
            raise ValueError("task_id cannot be empty")
        return token


class CanonicalSchemaSpec(BaseModel):
    required: list[str] = Field(default_factory=lambda: ["input_text", "target_text"])
    aliases: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class NormalizationSpec(BaseModel):
    trim_whitespace: bool = True
    drop_empty_records: bool = True
    dedupe: dict[str, Any] = Field(default_factory=dict)
    pii_redaction: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class DataQualitySpec(BaseModel):
    min_records: int = Field(default=100, ge=1)
    max_null_ratio: float = Field(default=0.1, ge=0, le=1)
    max_duplicate_ratio: float = Field(default=0.2, ge=0, le=1)
    required_coverage: dict[str, float] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @field_validator("required_coverage")
    @classmethod
    def validate_required_coverage(cls, value: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for key, score in value.items():
            if score < 0 or score > 1:
                raise ValueError(f"required_coverage for '{key}' must be between 0 and 1")
            cleaned[key] = float(score)
        return cleaned


class DatasetSplitSpec(BaseModel):
    train: float = Field(default=0.8, gt=0, lt=1)
    val: float = Field(default=0.1, ge=0, lt=1)
    test: float = Field(default=0.1, ge=0, lt=1)
    stratify_by: list[str] = Field(default_factory=list)
    seed: int = 42
    leakage_checks: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_sum(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError("dataset_split train/val/test ratios must sum to 1.0")
        return self


class TrainingDefaultsSpec(BaseModel):
    training_mode: str = "sft"
    chat_template: str = "llama3"
    num_epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    use_lora: bool = True

    model_config = {"extra": "allow"}


class MetricSpec(BaseModel):
    metric_id: str = Field(..., min_length=1, max_length=128)
    weight: float = Field(default=0, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)

    model_config = {"extra": "allow"}

    @field_validator("metric_id")
    @classmethod
    def normalize_metric_id(cls, value: str) -> str:
        return _normalize_token(value)


class EvaluationSpec(BaseModel):
    metrics: list[MetricSpec] = Field(default_factory=list)
    required_metrics_for_promotion: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_metric_ids(self):
        metric_ids = {m.metric_id for m in self.metrics}
        if len(metric_ids) != len(self.metrics):
            raise ValueError("evaluation.metrics contains duplicate metric_id values")
        missing = [
            _normalize_token(metric_id)
            for metric_id in self.required_metrics_for_promotion
            if _normalize_token(metric_id) not in metric_ids
        ]
        if missing:
            raise ValueError(
                "required_metrics_for_promotion must exist in evaluation.metrics: "
                + ", ".join(sorted(set(missing)))
            )
        self.required_metrics_for_promotion = [_normalize_token(m) for m in self.required_metrics_for_promotion]
        return self


class ToolAdapterSpec(BaseModel):
    enabled: bool = False
    adapter: str | None = None

    model_config = {"extra": "allow"}


class ToolsSpec(BaseModel):
    retrieval: ToolAdapterSpec = Field(default_factory=ToolAdapterSpec)
    function_calling: ToolAdapterSpec = Field(default_factory=ToolAdapterSpec)
    required_secrets: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class RegistryStageGateSpec(BaseModel):
    min_metrics: dict[str, float] = Field(default_factory=dict)
    max_regression_vs_prod: dict[str, float] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @field_validator("min_metrics", "max_regression_vs_prod")
    @classmethod
    def validate_metrics_range(cls, value: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for metric_id, score in value.items():
            if score < 0 or score > 1:
                raise ValueError(f"metric gate for '{metric_id}' must be between 0 and 1")
            cleaned[_normalize_token(metric_id)] = float(score)
        return cleaned


class RegistryGatesSpec(BaseModel):
    to_staging: RegistryStageGateSpec = Field(default_factory=RegistryStageGateSpec)
    to_production: RegistryStageGateSpec = Field(default_factory=RegistryStageGateSpec)

    model_config = {"extra": "allow"}


class AuditSpec(BaseModel):
    require_human_approval_for_production: bool = False
    notes_required_on_force_promotion: bool = False

    model_config = {"extra": "allow"}


class DomainProfileContract(BaseModel):
    schema_ref: str = Field(default="slm.domain-profile/v1", alias="$schema")
    profile_id: str = Field(..., min_length=3, max_length=128)
    version: str = Field(default="1.0.0", min_length=1, max_length=32)
    display_name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    owner: str = Field(default="platform", min_length=1, max_length=128)
    status: DomainProfileStatus = DomainProfileStatus.ACTIVE

    tasks: list[DomainTaskSpec] = Field(default_factory=list)
    canonical_schema: CanonicalSchemaSpec = Field(default_factory=CanonicalSchemaSpec)
    normalization: NormalizationSpec = Field(default_factory=NormalizationSpec)
    data_quality: DataQualitySpec = Field(default_factory=DataQualitySpec)
    dataset_split: DatasetSplitSpec = Field(default_factory=DatasetSplitSpec)
    training_defaults: TrainingDefaultsSpec = Field(default_factory=TrainingDefaultsSpec)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    tools: ToolsSpec = Field(default_factory=ToolsSpec)
    registry_gates: RegistryGatesSpec = Field(default_factory=RegistryGatesSpec)
    audit: AuditSpec = Field(default_factory=AuditSpec)

    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }

    @field_validator("profile_id")
    @classmethod
    def normalize_profile_id(cls, value: str) -> str:
        token = _normalize_token(value)
        if not token:
            raise ValueError("profile_id cannot be empty")
        return token


class DomainProfileSummaryResponse(BaseModel):
    id: int
    profile_id: str
    version: str
    display_name: str
    description: str
    owner: str
    status: DomainProfileStatus
    schema_ref: str
    is_system: bool

    model_config = {"from_attributes": True}


class DomainProfileResponse(DomainProfileSummaryResponse):
    contract: DomainProfileContract

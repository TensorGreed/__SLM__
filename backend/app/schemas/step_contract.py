"""Typed step contract schemas for workflow graph runtime."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _normalize_string_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)
    return cleaned


class StepRuntimeRequirements(BaseModel):
    """Runtime prerequisites declared by a step contract."""

    execution_modes: list[str] = Field(default_factory=lambda: ["local"])
    required_services: list[str] = Field(default_factory=list)
    required_env: list[str] = Field(default_factory=list)
    required_settings: list[str] = Field(default_factory=list)
    requires_gpu: bool = False
    min_vram_gb: float = Field(default=0.0, ge=0.0)

    @field_validator("execution_modes", "required_services", "required_env", "required_settings", mode="before")
    @classmethod
    def _list_or_default(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        return []

    @field_validator("execution_modes", "required_services", "required_env", "required_settings")
    @classmethod
    def _normalize_lists(cls, value: list[str]) -> list[str]:
        return _normalize_string_list(value)


class StepContract(BaseModel):
    """Contract definition attached to each workflow step node."""

    step_type: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    config_schema_ref: str = Field(..., min_length=1)
    runtime_requirements: StepRuntimeRequirements = Field(default_factory=StepRuntimeRequirements)

    @field_validator("input_artifacts", "output_artifacts", mode="before")
    @classmethod
    def _artifacts_list_or_default(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    @field_validator("input_artifacts", "output_artifacts")
    @classmethod
    def _normalize_artifacts(cls, value: list[str]) -> list[str]:
        return _normalize_string_list(value)

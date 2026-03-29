"""Pydantic schemas for beginner-mode domain blueprint workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, list):
        items: list[str] = []
        for raw in value:
            token = str(raw or "").strip()
            if token:
                items.append(token)
        return items
    return []


def _normalize_unique_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in values:
        token = str(raw or "").strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(token)
    return cleaned


class GlossaryEntry(BaseModel):
    term: str = Field(..., min_length=1, max_length=128)
    plain_language: str = Field(..., min_length=1, max_length=1000)
    category: str = Field(default="general", max_length=64)
    example: str | None = Field(default=None, max_length=1000)


class SuccessMetric(BaseModel):
    metric_id: str = Field(..., min_length=1, max_length=128)
    label: str = Field(..., min_length=1, max_length=255)
    target: str = Field(default="", max_length=255)
    why_it_matters: str = Field(default="", max_length=1000)


class DomainBlueprintContract(BaseModel):
    domain_name: str = Field(..., min_length=1, max_length=255)
    problem_statement: str = Field(..., min_length=1, max_length=4000)
    target_user_persona: str = Field(..., min_length=1, max_length=2000)
    task_family: str = Field(..., min_length=1, max_length=64)
    input_modality: str = Field(..., min_length=1, max_length=64)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    expected_output_examples: list[Any] = Field(default_factory=list)
    safety_compliance_notes: list[str] = Field(default_factory=list)
    deployment_target_constraints: dict[str, Any] = Field(default_factory=dict)
    success_metrics: list[SuccessMetric] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unresolved_assumptions: list[str] = Field(default_factory=list)

    @field_validator("safety_compliance_notes", "unresolved_assumptions", mode="before")
    @classmethod
    def _list_from_any(cls, value: Any) -> list[str]:
        return _as_string_list(value)

    @field_validator("safety_compliance_notes", "unresolved_assumptions")
    @classmethod
    def _dedupe_lists(cls, value: list[str]) -> list[str]:
        return _normalize_unique_list(value)

    @field_validator("task_family", "input_modality", mode="before")
    @classmethod
    def _normalize_tokens(cls, value: Any) -> str:
        token = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        return token or "unknown"


class DomainBlueprintValidationIssue(BaseModel):
    code: str
    field: str
    message: str
    actionable_fix: str


class DomainBlueprintValidationResult(BaseModel):
    ok: bool = True
    errors: list[DomainBlueprintValidationIssue] = Field(default_factory=list)
    warnings: list[DomainBlueprintValidationIssue] = Field(default_factory=list)


class DomainBlueprintGuidance(BaseModel):
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    inferred_fields: list[dict[str, str]] = Field(default_factory=list)


class DomainBlueprintAnalyzeRequest(BaseModel):
    brief_text: str = Field(..., min_length=1, max_length=8000)
    domain_name: str | None = Field(default=None, max_length=255)
    problem_statement: str | None = Field(default=None, max_length=4000)
    target_user_persona: str | None = Field(default=None, max_length=2000)
    task_family_hint: str | None = Field(default=None, max_length=64)
    input_modality_hint: str | None = Field(default=None, max_length=64)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    sample_inputs: list[str] = Field(default_factory=list)
    sample_outputs: list[str] = Field(default_factory=list)
    safety_compliance_notes: list[str] = Field(default_factory=list)
    risk_constraints: list[str] = Field(default_factory=list)
    deployment_target: str | None = Field(default=None, max_length=128)
    success_metrics: list[str] = Field(default_factory=list)
    llm_enrich: bool = True

    @field_validator("sample_inputs", "sample_outputs", "safety_compliance_notes", "risk_constraints", "success_metrics", mode="before")
    @classmethod
    def _normalize_list_inputs(cls, value: Any) -> list[str]:
        return _as_string_list(value)


class DomainBlueprintAnalyzeResponse(BaseModel):
    project_id: int | None = None
    blueprint: DomainBlueprintContract
    validation: DomainBlueprintValidationResult
    guidance: DomainBlueprintGuidance
    llm_enrichment: dict[str, Any] = Field(default_factory=dict)


class DomainBlueprintSaveRequest(BaseModel):
    blueprint: DomainBlueprintContract
    source: str = Field(default="manual", max_length=64)
    brief_text: str = Field(default="", max_length=8000)
    analysis_metadata: dict[str, Any] = Field(default_factory=dict)
    apply_immediately: bool = False


class DomainBlueprintRevisionResponse(BaseModel):
    id: int
    project_id: int
    version: int
    status: str
    source: str
    brief_text: str | None
    blueprint: DomainBlueprintContract
    analysis_metadata: dict[str, Any] = Field(default_factory=dict)
    created_by_user_id: int | None = None
    created_at: datetime
    updated_at: datetime


class DomainBlueprintListResponse(BaseModel):
    project_id: int
    count: int
    active_version: int | None = None
    revisions: list[DomainBlueprintRevisionResponse] = Field(default_factory=list)


class DomainBlueprintApplyRequest(BaseModel):
    adopt_project_description: bool = True
    adopt_target_profile: bool = True
    set_beginner_mode: bool = True


class DomainBlueprintDiffItem(BaseModel):
    field: str
    before: Any = None
    after: Any = None


class DomainBlueprintDiffResponse(BaseModel):
    project_id: int
    from_version: int
    to_version: int
    changed_fields: list[DomainBlueprintDiffItem] = Field(default_factory=list)


class DomainBlueprintGlossaryHelpResponse(BaseModel):
    project_id: int | None = None
    term_query: str = ""
    count: int
    entries: list[GlossaryEntry] = Field(default_factory=list)

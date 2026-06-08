"""Pydantic schemas for Project CRUD and pipeline operations."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

from app.models.project import PipelineStage, ProjectStatus


# ── Request schemas ─────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    base_model_name: str = ""
    starter_pack_id: str | None = None
    domain_pack_id: int | None = None
    domain_profile_id: int | None = None
    target_profile_id: str | None = "vllm_server"
    gate_policy: dict | None = None
    budget_settings: dict | None = None
    beginner_mode: bool = False
    brief_text: str | None = None
    sample_inputs: list[str] = Field(default_factory=list)
    sample_outputs: list[str] = Field(default_factory=list)
    domain_blueprint: dict[str, Any] | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    status: ProjectStatus | None = None
    pipeline_stage: PipelineStage | None = None
    base_model_name: str | None = None
    domain_pack_id: int | None = None
    domain_profile_id: int | None = None
    target_profile_id: str | None = None
    gate_policy: dict | None = None
    budget_settings: dict | None = None
    beginner_mode: bool | None = None
    active_domain_blueprint_version: int | None = None
    selected_recipe: dict | None = None
    quickstart_tour_state: dict | None = None


class ProjectRecipeApplyRequest(BaseModel):
    recipe_id: str = Field(..., min_length=1, max_length=128)


# Quality-Lift phase 2, slice 1 — Slice definitions PUT request.
# Schemas are intentionally thin: the heavy validation (op closed set,
# slice_id grammar, per-project cap, regex compilability) lives in
# slice_definitions_service so the same code-path runs whether the
# payload arrives via the API endpoint or via a future bulk-import
# CLI / data-studio nudge. The endpoint just passes ``payload.model_dump()``
# through to the service.
class SliceClauseSchema(BaseModel):
    field: str = Field(..., description="dot-path on the eval row dict")
    op: str = Field(..., description="closed-set op; see slice_definitions_service.SLICE_OPERATORS")
    value: Any = None


class SliceDefinitionSchema(BaseModel):
    slice_id: str = Field(..., description="lowercase ASCII id, ^[a-z][a-z0-9_]{0,63}$")
    display_name: str = ""
    where: list[SliceClauseSchema] = Field(default_factory=list)


class SliceDefinitionsPayload(BaseModel):
    slices: list[SliceDefinitionSchema] = Field(default_factory=list)


class SliceDefinitionsResponse(BaseModel):
    project_id: int
    slice_definitions: dict


class ProjectDomainPackAssignRequest(BaseModel):
    pack_id: str = Field(..., min_length=3, max_length=128)
    adopt_pack_default_profile: bool = True


class ProjectDomainProfileAssignRequest(BaseModel):
    profile_id: str = Field(..., min_length=3, max_length=128)


# ── Response schemas ────────────────────────────────────────────────────

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str | None
    status: ProjectStatus
    pipeline_stage: PipelineStage
    base_model_name: str | None
    domain_pack_id: int | None = None
    domain_profile_id: int | None = None
    target_profile_id: str | None = None
    gate_policy: dict | None = None
    budget_settings: dict | None = None
    beginner_mode: bool = False
    active_domain_blueprint_version: int | None = None
    selected_recipe: dict | None = None
    quickstart_tour_state: dict | None = None
    # USER-SUCCESS Epic 7 Phase 7b — set when this project was created
    # via the RAG-reroute flow. Frontend renders a "← cloned from"
    # provenance chip.
    parent_project_id: int | None = None
    # Runtime feature flags. Carries `rag_first` (bool) + a
    # mirrored `auto_rag.enabled` value. Frontend hides the Train
    # button + shows a "RAG-first" badge when rag_first is true.
    runtime_config: dict | None = None
    # Quality-Lift phase 2 — Named eval-row subsets defined as JSON
    # predicates; nullable for projects with no slices configured.
    slice_definitions: dict | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectRerouteToRagRequest(BaseModel):
    name_suffix: str = Field(default=" (RAG)", max_length=64)


class ProjectRerouteToRagResponse(BaseModel):
    new_project_id: int
    new_project_name: str
    source_project_id: int
    clone_report: dict | None = None


class ProjectListResponse(BaseModel):
    projects: list[ProjectResponse]
    total: int


class ProjectStatsResponse(BaseModel):
    id: int
    name: str
    pipeline_stage: PipelineStage
    status: ProjectStatus
    dataset_count: int = 0
    experiment_count: int = 0
    total_documents: int = 0

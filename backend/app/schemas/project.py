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

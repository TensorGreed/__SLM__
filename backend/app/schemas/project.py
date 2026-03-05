"""Pydantic schemas for Project CRUD and pipeline operations."""

from datetime import datetime
from pydantic import BaseModel, Field

from app.models.project import PipelineStage, ProjectStatus


# ── Request schemas ─────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    base_model_name: str = ""
    domain_pack_id: int | None = None
    domain_profile_id: int | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    status: ProjectStatus | None = None
    pipeline_stage: PipelineStage | None = None
    base_model_name: str | None = None
    domain_pack_id: int | None = None
    domain_profile_id: int | None = None


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
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


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

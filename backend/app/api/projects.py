"""Project CRUD API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole, ProjectMembership, ProjectRole
from app.models.dataset import Dataset, RawDocument
from app.models.domain_pack import DomainPack
from app.models.domain_profile import DomainProfile
from app.models.project import Project, ProjectStatus
from app.schemas.project import (
    ProjectDomainPackAssignRequest,
    ProjectDomainProfileAssignRequest,
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectStatsResponse,
    ProjectUpdate,
)
from app.security import get_request_principal, upsert_project_membership
from app.services.domain_pack_service import assign_project_domain_pack, get_domain_pack
from app.services.domain_profile_service import assign_project_domain_profile, get_domain_profile
from app.services.domain_runtime_service import resolve_project_domain_runtime

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    request: Request,
    skip: int = 0,
    limit: int = 50,
    status: ProjectStatus | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List all projects with optional filtering."""
    principal = get_request_principal(request)

    query = select(Project)
    count_query = select(func.count(Project.id))

    if settings.AUTH_ENABLED and principal and principal.role != GlobalRole.ADMIN:
        query = query.join(ProjectMembership, ProjectMembership.project_id == Project.id).where(
            ProjectMembership.user_id == principal.user_id
        )
        count_query = count_query.join(
            ProjectMembership, ProjectMembership.project_id == Project.id
        ).where(ProjectMembership.user_id == principal.user_id)

    if status:
        query = query.where(Project.status == status)
        count_query = count_query.where(Project.status == status)

    query = query.order_by(Project.updated_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    projects = result.scalars().all()

    total = (await db.execute(count_query)).scalar() or 0

    return ProjectListResponse(
        projects=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
    )


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    data: ProjectCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a new SLM project."""
    existing = await db.execute(select(Project).where(Project.name == data.name))
    if existing.scalar_one_or_none():
        raise HTTPException(400, f"Project '{data.name}' already exists")

    resolved_domain_pack_id = data.domain_pack_id
    selected_pack: DomainPack | None = None
    if data.domain_pack_id is not None:
        pack_result = await db.execute(select(DomainPack).where(DomainPack.id == data.domain_pack_id))
        selected_pack = pack_result.scalar_one_or_none()
        if selected_pack is None:
            raise HTTPException(400, f"Domain pack id {data.domain_pack_id} not found")
    else:
        default_pack = await get_domain_pack(db, "general-pack-v1")
        if default_pack is not None:
            resolved_domain_pack_id = default_pack.id
            selected_pack = default_pack

    resolved_domain_profile_id = data.domain_profile_id
    if data.domain_profile_id is not None:
        profile_result = await db.execute(
            select(DomainProfile.id).where(DomainProfile.id == data.domain_profile_id)
        )
        if profile_result.scalar_one_or_none() is None:
            raise HTTPException(400, f"Domain profile id {data.domain_profile_id} not found")
    else:
        candidate_profile_ids: list[str] = []
        if selected_pack and selected_pack.default_profile_id:
            candidate_profile_ids.append(selected_pack.default_profile_id)
        candidate_profile_ids.append("generic-domain-v1")

        for candidate in candidate_profile_ids:
            default_profile = await get_domain_profile(db, candidate)
            if default_profile is not None:
                resolved_domain_profile_id = default_profile.id
                break

    project = Project(
        name=data.name,
        description=data.description,
        base_model_name=data.base_model_name,
        domain_pack_id=resolved_domain_pack_id,
        domain_profile_id=resolved_domain_profile_id,
    )
    db.add(project)
    await db.flush()
    await db.refresh(project)

    principal = get_request_principal(request)
    if settings.AUTH_ENABLED and principal:
        await upsert_project_membership(
            db,
            project_id=project.id,
            user_id=principal.user_id,
            role=ProjectRole.OWNER,
        )

    return ProjectResponse.model_validate(project)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get a single project by ID."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")
    return ProjectResponse.model_validate(project)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    data: ProjectUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update project fields."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    update_data = data.model_dump(exclude_unset=True)
    if "domain_pack_id" in update_data and update_data["domain_pack_id"] is not None:
        pack_result = await db.execute(
            select(DomainPack.id).where(DomainPack.id == update_data["domain_pack_id"])
        )
        if pack_result.scalar_one_or_none() is None:
            raise HTTPException(400, f"Domain pack id {update_data['domain_pack_id']} not found")

    if "domain_profile_id" in update_data and update_data["domain_profile_id"] is not None:
        profile_result = await db.execute(
            select(DomainProfile.id).where(DomainProfile.id == update_data["domain_profile_id"])
        )
        if profile_result.scalar_one_or_none() is None:
            raise HTTPException(400, f"Domain profile id {update_data['domain_profile_id']} not found")

    for key, value in update_data.items():
        setattr(project, key, value)

    await db.flush()
    await db.refresh(project)
    return ProjectResponse.model_validate(project)


@router.put("/{project_id}/domain-profile", response_model=ProjectResponse)
async def assign_domain_profile(
    project_id: int,
    data: ProjectDomainProfileAssignRequest,
    db: AsyncSession = Depends(get_db),
):
    """Assign a project to a domain profile by profile_id."""
    try:
        project = await assign_project_domain_profile(db, project_id, data.profile_id)
        return ProjectResponse.model_validate(project)
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.put("/{project_id}/domain-pack", response_model=ProjectResponse)
async def assign_domain_pack(
    project_id: int,
    data: ProjectDomainPackAssignRequest,
    db: AsyncSession = Depends(get_db),
):
    """Assign a project to a domain pack by pack_id."""
    try:
        project = await assign_project_domain_pack(
            db,
            project_id,
            data.pack_id,
            adopt_pack_default_profile=data.adopt_pack_default_profile,
        )
        return ProjectResponse.model_validate(project)
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/{project_id}/domain-runtime")
async def get_project_domain_runtime(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Resolve the effective domain runtime contract for a project."""
    try:
        return await resolve_project_domain_runtime(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a project and all associated data."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")
    await db.delete(project)


@router.get("/{project_id}/stats", response_model=ProjectStatsResponse)
async def get_project_stats(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get project statistics overview."""
    result = await db.execute(
        select(Project)
        .where(Project.id == project_id)
        .options(selectinload(Project.datasets), selectinload(Project.experiments))
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    docs_count_result = await db.execute(
        select(func.count(RawDocument.id))
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
    )
    total_docs = docs_count_result.scalar() or 0

    return ProjectStatsResponse(
        id=project.id,
        name=project.name,
        pipeline_stage=project.pipeline_stage,
        status=project.status,
        dataset_count=len(project.datasets) if project.datasets else 0,
        experiment_count=len(project.experiments) if project.experiments else 0,
        total_documents=total_docs,
    )

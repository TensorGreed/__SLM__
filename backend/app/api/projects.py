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
from app.models.experiment import Experiment, ExperimentStatus
from app.schemas.project import (
    ProjectDomainPackAssignRequest,
    ProjectDomainProfileAssignRequest,
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectStatsResponse,
    ProjectUpdate,
)
from pydantic import BaseModel
from app.security import get_request_principal, upsert_project_membership
from app.services.domain_pack_service import assign_project_domain_pack, get_domain_pack
from app.services.domain_profile_service import assign_project_domain_profile, get_domain_profile
from app.services.domain_runtime_service import resolve_project_domain_runtime
from app.services.readiness_service import get_project_readiness
from app.services.nl2pipeline_service import magic_create_pipeline_recipe
from app.services.pipeline_recipe_service import apply_pipeline_recipe_blueprint
from app.services.dataset_service import save_project_dataset_adapter_preference
from app.services.evaluation_pack_service import evaluate_experiment_auto_gates

class MagicCreateRequest(BaseModel):
    prompt: str

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
        target_profile_id=data.target_profile_id or "vllm_server",
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


@router.post("/magic-create", response_model=ProjectResponse, status_code=201)
async def magic_create_project(
    data: MagicCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a project and apply a recommended pipeline recipe based on a natural language prompt."""
    try:
        recommendation = await magic_create_pipeline_recipe(data.prompt, allow_fallback=True)
    except ValueError as e:
        raise HTTPException(400, str(e))

    project_name = str(recommendation.get("project_name") or "Magic Project").strip() or "Magic Project"
    
    # Ensure name uniqueness
    existing = await db.execute(select(Project).where(Project.name.like(f"{project_name}%")))
    if existing.scalars().all():
        project_name = f"{project_name} - {data.prompt[:10]}"

    recommended_pack_db_id: int | None = None
    recommended_pack = recommendation.get("domain_pack_id")
    if isinstance(recommended_pack, int):
        recommended_pack_db_id = recommended_pack
    else:
        recommended_pack_id = str(recommended_pack or "").strip().lower()
        if recommended_pack_id:
            pack = await get_domain_pack(db, recommended_pack_id)
            if pack is not None:
                recommended_pack_db_id = int(pack.id)

    project_data = ProjectCreate(
        name=project_name,
        description=str(recommendation.get("project_description") or f"Generated from prompt: {data.prompt}"),
        base_model_name=str(
            recommendation.get("base_model_name") or "meta-llama/Meta-Llama-3-8B-Instruct"
        ),
        domain_pack_id=recommended_pack_db_id,
    )
    
    # Use existing create_project logic
    project_response = await create_project(project_data, request, db)
    project_id = project_response.id

    # Apply the recommended recipe
    recipe_id = recommendation.get("pipeline_recipe_id", "recipe.pipeline.sft_default")
    
    adapter_id = str(recommendation.get("adapter_id") or "default-canonical").strip() or "default-canonical"
    task_profile = str(recommendation.get("task_profile") or "instruction_sft").strip() or "instruction_sft"
    base_model_name = str(
        recommendation.get("base_model_name") or "meta-llama/Meta-Llama-3-8B-Instruct"
    ).strip() or "meta-llama/Meta-Llama-3-8B-Instruct"

    try:
        await apply_pipeline_recipe_blueprint(
            db,
            project_id=project_id,
            recipe_id=recipe_id,
            overrides={
                "dataset_adapter": {
                    "adapter_id": adapter_id,
                    "task_profile": task_profile,
                    "adapter_config": {},
                },
                "training": {
                    "base_config": {
                        "base_model": base_model_name,
                    }
                },
            },
            include_preflight=False,
            mark_active=True,
        )
    except ValueError:
        # Graceful fallback: still persist adapter preference even when recipe resolution fails.
        await save_project_dataset_adapter_preference(
            db,
            project_id,
            adapter_id=adapter_id,
            task_profile=task_profile,
            adapter_config={},
            field_mapping={},
        )

    return project_response


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


@router.get("/{project_id}/runtime/readiness")
async def get_project_runtime_readiness(project_id: int):
    """Validate GPU/dependencies/paths/secrets before run."""
    return await get_project_readiness(project_id)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a project and all associated data."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")
    await db.delete(project)


@router.get("/{project_id}/gate-check")
async def project_deployment_gate_check(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Check if the project is ready for deployment/export based on gates."""
    # Find latest completed experiment
    stmt = (
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(Experiment.status == ExperimentStatus.COMPLETED)
        .order_by(Experiment.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    experiment = result.scalar_one_or_none()

    if not experiment:
        # Fallback to any latest experiment if no completed one
        stmt = (
            select(Experiment)
            .where(Experiment.project_id == project_id)
            .order_by(Experiment.created_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(404, f"No experiments found for project {project_id}")

    # Evaluate gates
    report = await evaluate_experiment_auto_gates(
        db,
        project_id=project_id,
        experiment_id=experiment.id,
    )

    # Apply Project Gate Policy
    project_stmt = select(Project).where(Project.id == project_id)
    project_res = await db.execute(project_stmt)
    project = project_res.scalar_one_or_none()
    if not project:
        raise HTTPException(404, f"Project {project_id} not found")

    policy = project.gate_policy or {}
    must_pass = policy.get("must_pass", True)
    min_score = policy.get("min_score", 0.0)
    blocked_if_missing = policy.get("blocked_if_missing", True)

    is_blocked = False
    reasons = []

    if must_pass and not report.get("passed"):
        is_blocked = True
        reasons.append("Mandatory quality gates failed.")

    if blocked_if_missing and report.get("missing_required_metrics"):
        is_blocked = True
        reasons.append(f"Missing required metrics: {', '.join(report.get('missing_required_metrics', []))}")

    return {
        "project_id": project_id,
        "experiment_id": experiment.id,
        "passed": not is_blocked,
        "is_blocked": is_blocked,
        "reasons": reasons,
        "policy": policy,
        "gate_report": report
    }


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

"""Domain blueprint API routes for beginner-mode onboarding."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.schemas.domain_blueprint import (
    DomainBlueprintAnalyzeRequest,
    DomainBlueprintAnalyzeResponse,
    DomainBlueprintApplyRequest,
    DomainBlueprintDiffResponse,
    DomainBlueprintGlossaryHelpResponse,
    DomainBlueprintListResponse,
    DomainBlueprintRevisionResponse,
    DomainBlueprintSaveRequest,
)
from app.schemas.project import ProjectResponse
from app.security import get_request_principal
from app.services.domain_blueprint_service import (
    DomainBlueprintValidationError,
    analyze_domain_brief,
    apply_domain_blueprint_revision,
    diff_domain_blueprint_revisions,
    get_domain_blueprint_revision,
    get_latest_domain_blueprint_revision,
    glossary_help,
    list_domain_blueprint_revisions,
    save_domain_blueprint_revision,
    serialize_domain_blueprint_revision,
)


router = APIRouter(tags=["Domain Blueprints"])


class DomainBlueprintSaveResponse(BaseModel):
    revision: DomainBlueprintRevisionResponse
    project: ProjectResponse | None = None
    applied: bool = False


async def _ensure_project_exists(db: AsyncSession, project_id: int) -> Project:
    from sqlalchemy import select

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@router.post("/domain-blueprints/analyze", response_model=DomainBlueprintAnalyzeResponse)
async def analyze_domain_blueprint_brief(
    req: DomainBlueprintAnalyzeRequest,
):
    """Analyze a raw natural-language brief into a normalized domain blueprint draft."""
    return await analyze_domain_brief(req, project_id=None)


@router.get("/domain-blueprints/glossary/help", response_model=DomainBlueprintGlossaryHelpResponse)
async def glossary_help_global(term: str = ""):
    """Return glossary entries for beginner-friendly onboarding explanations."""
    return glossary_help(term_query=term, project_id=None, latest_blueprint=None)


@router.post(
    "/projects/{project_id}/domain-blueprints/analyze",
    response_model=DomainBlueprintAnalyzeResponse,
)
async def analyze_project_domain_blueprint_brief(
    project_id: int,
    req: DomainBlueprintAnalyzeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Analyze brief in the context of an existing project."""
    await _ensure_project_exists(db, project_id)
    return await analyze_domain_brief(req, project_id=project_id)


@router.post(
    "/projects/{project_id}/domain-blueprints",
    response_model=DomainBlueprintSaveResponse,
    status_code=201,
)
async def save_project_domain_blueprint(
    project_id: int,
    req: DomainBlueprintSaveRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Persist a new versioned domain blueprint revision for a project."""
    await _ensure_project_exists(db, project_id)
    principal = get_request_principal(request)
    created_by_user_id = getattr(principal, "user_id", None)
    try:
        revision = await save_domain_blueprint_revision(
            db=db,
            project_id=project_id,
            blueprint=req.blueprint,
            source=req.source,
            brief_text=req.brief_text,
            analysis_metadata=req.analysis_metadata,
            created_by_user_id=created_by_user_id,
        )
    except DomainBlueprintValidationError as e:
        raise HTTPException(
            400,
            {
                "error_code": "DOMAIN_BLUEPRINT_VALIDATION_FAILED",
                "message": "Domain blueprint is contradictory or incomplete.",
                "validation": e.validation.model_dump(mode="json"),
            },
        )

    project_payload: ProjectResponse | None = None
    if req.apply_immediately:
        project, _ = await apply_domain_blueprint_revision(
            db,
            project_id=project_id,
            version=revision.version,
            adopt_project_description=True,
            adopt_target_profile=True,
            set_beginner_mode=True,
        )
        project_payload = ProjectResponse.model_validate(project)

    return DomainBlueprintSaveResponse(
        revision=serialize_domain_blueprint_revision(revision),
        project=project_payload,
        applied=bool(req.apply_immediately),
    )


@router.get(
    "/projects/{project_id}/domain-blueprints",
    response_model=DomainBlueprintListResponse,
)
async def list_project_domain_blueprints(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List versioned domain blueprint revisions for a project."""
    project = await _ensure_project_exists(db, project_id)
    rows = await list_domain_blueprint_revisions(db, project_id)
    return DomainBlueprintListResponse(
        project_id=project_id,
        count=len(rows),
        active_version=project.active_domain_blueprint_version,
        revisions=[serialize_domain_blueprint_revision(row) for row in rows],
    )


@router.get(
    "/projects/{project_id}/domain-blueprints/latest",
    response_model=DomainBlueprintRevisionResponse,
)
async def get_latest_project_domain_blueprint(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Fetch latest blueprint revision."""
    await _ensure_project_exists(db, project_id)
    row = await get_latest_domain_blueprint_revision(db, project_id=project_id)
    if row is None:
        raise HTTPException(404, "No domain blueprint revisions found for project")
    return serialize_domain_blueprint_revision(row)


@router.get(
    "/projects/{project_id}/domain-blueprints/diff",
    response_model=DomainBlueprintDiffResponse,
)
async def diff_project_domain_blueprints(
    project_id: int,
    from_version: int,
    to_version: int,
    db: AsyncSession = Depends(get_db),
):
    """Compare two blueprint versions and return field-level diffs."""
    await _ensure_project_exists(db, project_id)
    try:
        return await diff_domain_blueprint_revisions(
            db,
            project_id=project_id,
            from_version=from_version,
            to_version=to_version,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get(
    "/projects/{project_id}/domain-blueprints/glossary/help",
    response_model=DomainBlueprintGlossaryHelpResponse,
)
async def glossary_help_project(
    project_id: int,
    term: str = "",
    db: AsyncSession = Depends(get_db),
):
    """Fetch glossary/help entries scoped to project latest blueprint."""
    await _ensure_project_exists(db, project_id)
    latest = await get_latest_domain_blueprint_revision(db, project_id=project_id)
    return glossary_help(term_query=term, project_id=project_id, latest_blueprint=latest)


@router.get(
    "/projects/{project_id}/domain-blueprints/{version}",
    response_model=DomainBlueprintRevisionResponse,
)
async def get_project_domain_blueprint_revision(
    project_id: int,
    version: int,
    db: AsyncSession = Depends(get_db),
):
    """Fetch one blueprint revision by version number."""
    await _ensure_project_exists(db, project_id)
    row = await get_domain_blueprint_revision(db, project_id=project_id, version=version)
    if row is None:
        raise HTTPException(404, f"Domain blueprint version {version} not found")
    return serialize_domain_blueprint_revision(row)


@router.post(
    "/projects/{project_id}/domain-blueprints/{version}/apply",
    response_model=DomainBlueprintSaveResponse,
)
async def apply_project_domain_blueprint_revision(
    project_id: int,
    version: int,
    req: DomainBlueprintApplyRequest,
    db: AsyncSession = Depends(get_db),
):
    """Apply one blueprint revision as active project onboarding contract."""
    await _ensure_project_exists(db, project_id)
    try:
        project, revision = await apply_domain_blueprint_revision(
            db,
            project_id=project_id,
            version=version,
            adopt_project_description=req.adopt_project_description,
            adopt_target_profile=req.adopt_target_profile,
            set_beginner_mode=req.set_beginner_mode,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))

    return DomainBlueprintSaveResponse(
        revision=serialize_domain_blueprint_revision(revision),
        project=ProjectResponse.model_validate(project),
        applied=True,
    )

"""Project CRUD API routes."""

from typing import Any

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
    ProjectRecipeApplyRequest,
    ProjectRerouteToRagRequest,
    ProjectRerouteToRagResponse,
    ProjectResponse,
    ProjectStatsResponse,
    ProjectUpdate,
    SliceDefinitionsPayload,
    SliceDefinitionsResponse,
)
from app.schemas.domain_blueprint import DomainBlueprintAnalyzeRequest, DomainBlueprintContract
from pydantic import BaseModel
from app.security import get_request_principal, upsert_project_membership
from app.services.domain_pack_service import assign_project_domain_pack, get_domain_pack
from app.services.domain_profile_service import assign_project_domain_profile, get_domain_profile
from app.services.domain_runtime_service import resolve_project_domain_runtime
from app.services.readiness_service import get_project_readiness
from app.services import gold_workbench_service
from app.services.nl2pipeline_service import magic_create_pipeline_recipe
from app.services.pipeline_recipe_service import apply_pipeline_recipe_blueprint
from app.services.dataset_service import save_project_dataset_adapter_preference
from app.services.evaluation_pack_service import evaluate_experiment_auto_gates
from app.services.recipe_apply_service import (
    RecipeNotFoundError,
    apply_recipe_to_project,
    clear_recipe_from_project,
)
from app.services.recipe_service import (
    default_recipe_for_task_family,
    default_recipe_for_task_profile,
)
from app.services.starter_pack_service import get_starter_pack_by_id
from app.services.domain_blueprint_service import (
    DomainBlueprintValidationError,
    analyze_domain_brief,
    apply_domain_blueprint_revision,
    save_domain_blueprint_revision,
)

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

    selected_starter_pack: dict[str, Any] | None = None
    if data.starter_pack_id is not None:
        selected_starter_pack = get_starter_pack_by_id(data.starter_pack_id)
        if selected_starter_pack is None:
            raise HTTPException(400, f"Starter pack '{data.starter_pack_id}' not found")

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

    resolved_base_model_name = str(data.base_model_name or "").strip()
    if not resolved_base_model_name and selected_starter_pack is not None:
        resolved_base_model_name = (
            str(selected_starter_pack.get("default_base_model_name") or "").strip()
        )

    model_fields_set = set(getattr(data, "model_fields_set", set()))
    explicit_target = "target_profile_id" in model_fields_set and bool(
        str(data.target_profile_id or "").strip()
    )
    resolved_target_profile_id = str(data.target_profile_id or "").strip() or "vllm_server"
    if not explicit_target and selected_starter_pack is not None:
        starter_target_profile = str(
            selected_starter_pack.get("target_profile_default") or ""
        ).strip()
        if starter_target_profile:
            resolved_target_profile_id = starter_target_profile

    project_payload = dict(
        name=data.name,
        description=data.description,
        base_model_name=resolved_base_model_name,
        domain_pack_id=resolved_domain_pack_id,
        domain_profile_id=resolved_domain_profile_id,
        target_profile_id=resolved_target_profile_id,
        beginner_mode=bool(data.beginner_mode),
    )
    if isinstance(data.gate_policy, dict):
        project_payload["gate_policy"] = dict(data.gate_policy)
    elif selected_starter_pack is not None and isinstance(
        selected_starter_pack.get("evaluation_gate_defaults"), dict
    ):
        project_payload["gate_policy"] = dict(
            selected_starter_pack.get("evaluation_gate_defaults") or {}
        )

    if selected_starter_pack is not None and isinstance(
        selected_starter_pack.get("adapter_task_defaults"), dict
    ):
        project_payload["dataset_adapter_preset"] = dict(
            selected_starter_pack.get("adapter_task_defaults") or {}
        )

    if isinstance(data.budget_settings, dict):
        project_payload["budget_settings"] = dict(data.budget_settings)
    project = Project(**project_payload)
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

    should_create_blueprint = bool(isinstance(data.domain_blueprint, dict) or str(data.brief_text or "").strip())
    if should_create_blueprint:
        created_by_user_id = getattr(principal, "user_id", None)
        if isinstance(data.domain_blueprint, dict):
            try:
                blueprint = DomainBlueprintContract.model_validate(data.domain_blueprint)
            except Exception as e:
                raise HTTPException(
                    400,
                    {
                        "error_code": "DOMAIN_BLUEPRINT_PARSE_FAILED",
                        "message": "Invalid domain_blueprint payload.",
                        "detail": str(e),
                    },
                )
            analysis_metadata = {
                "source": "project_create_payload",
                "brief_text": str(data.brief_text or "").strip(),
            }
            brief_text = str(data.brief_text or "").strip()
        else:
            analysis_input = DomainBlueprintAnalyzeRequest(
                brief_text=str(data.brief_text or "").strip() or str(data.description or "").strip() or data.name,
                sample_inputs=list(data.sample_inputs or []),
                sample_outputs=list(data.sample_outputs or []),
                deployment_target=resolved_target_profile_id,
            )
            analysis = await analyze_domain_brief(analysis_input, project_id=project.id)
            blueprint = analysis.blueprint
            analysis_metadata = {
                "source": "analyze_from_project_create",
                "guidance": analysis.guidance.model_dump(mode="json"),
                "llm_enrichment": analysis.llm_enrichment,
                "validation": analysis.validation.model_dump(mode="json"),
            }
            if not analysis.validation.ok:
                raise HTTPException(
                    400,
                    {
                        "error_code": "DOMAIN_BLUEPRINT_VALIDATION_FAILED",
                        "message": "Domain blueprint validation failed during project bootstrap.",
                        "validation": analysis.validation.model_dump(mode="json"),
                    },
                )
            brief_text = analysis_input.brief_text

        try:
            revision = await save_domain_blueprint_revision(
                db=db,
                project_id=project.id,
                blueprint=blueprint,
                source="project_bootstrap",
                brief_text=brief_text,
                analysis_metadata=analysis_metadata,
                created_by_user_id=created_by_user_id,
            )
            project, _ = await apply_domain_blueprint_revision(
                db=db,
                project_id=project.id,
                version=revision.version,
                adopt_project_description=True,
                adopt_target_profile=True,
                set_beginner_mode=True,
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

        # Auto-apply a task-shape recipe so `Project.selected_recipe`
        # is populated for brief-driven (non-templated) projects.
        # Without this, downstream surfaces that branch on
        # `recipe_id` (synth playbook runner, auto-RAG comparison,
        # post-eval reroute analyzer, several Coach Mode signals)
        # silently degrade or hard-fail. The DatasetImportWizard's
        # recipe picker remains the override path.
        if project.selected_recipe is None:
            task_family = getattr(blueprint, "task_family", None)
            inferred_recipe_id = default_recipe_for_task_family(task_family)
            try:
                project = await apply_recipe_to_project(
                    db, project.id, inferred_recipe_id,
                )
            except RecipeNotFoundError:
                # Catalog rename / map drift — fall back to the
                # generic-sft safety net rather than leaving NULL.
                project = await apply_recipe_to_project(
                    db, project.id, "generic-sft",
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

    # Apply a task-shape recipe so `Project.selected_recipe` is
    # populated. This is independent of the pipeline-DAG recipe
    # above (different concepts) — both coexist on the project.
    # See `create_project` for the same call on the brief-driven
    # path; the magic-create recommendation carries `task_profile`
    # instead of `task_family`, so we use the matching helper.
    if project_response.selected_recipe is None:
        inferred_recipe_id = default_recipe_for_task_profile(task_profile)
        try:
            applied = await apply_recipe_to_project(
                db, project_id, inferred_recipe_id,
            )
        except RecipeNotFoundError:
            applied = await apply_recipe_to_project(
                db, project_id, "generic-sft",
            )
        project_response = ProjectResponse.model_validate(applied)

    return project_response


# USER-SUCCESS Epic 7 Phase 7d — idempotency window for reroute-to-rag.
# We refuse a second clone of the same source within this many seconds
# so a frantic double-click (or a duplicate request from a UI rerender)
# doesn't create two parallel RAG siblings. 3600s = 1 hour matches the
# Phase 7d spec.
_REROUTE_IDEMPOTENCY_WINDOW_SECONDS: int = 3600


@router.post(
    "/{project_id}/reroute-to-rag",
    status_code=201,
)
async def reroute_to_rag(
    project_id: int,
    data: ProjectRerouteToRagRequest,
    async_job: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """USER-SUCCESS Epic 7 Phase 7b — clone a qa-sft project into a
    RAG-first sibling.

    The new project carries the source's gold set + raw + prepared
    splits forward, has ``runtime_config.rag_first=true`` so the
    playground answers via base model + retrieval (no training run
    needed), and links back via ``parent_project_id`` for the UI's
    provenance chip.

    Phase 7d adds a 1-hour idempotency guard: a second clone of the
    same source within the window returns 429 + the existing clone's
    id so the UI can navigate to it instead of creating a duplicate.

    Status codes:
      * 201 — clone succeeded; body carries the new project id.
      * 400 — source recipe isn't eligible (only qa-sft today) OR
        source has no recipe selected.
      * 404 — source project doesn't exist.
      * 429 — another RAG clone of this source was created within
        the last hour; body carries the existing clone's id so the
        UI can navigate there instead.
    """
    from datetime import datetime, timedelta, timezone

    from app.services.rag_project_service import (
        RagCloneError,
        clone_project_for_rag,
    )

    # Idempotency check — find any existing clone (parent_project_id
    # == source) created within the cooldown window. We order by
    # created_at DESC so the most recent clone wins (uncommon edge
    # case where multiple legitimate clones exist).
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=_REROUTE_IDEMPOTENCY_WINDOW_SECONDS
    )
    existing_result = await db.execute(
        select(Project)
        .where(
            Project.parent_project_id == project_id,
            Project.created_at >= cutoff,
        )
        .order_by(Project.created_at.desc())
        .limit(1)
    )
    existing_clone = existing_result.scalar_one_or_none()
    if existing_clone is not None:
        raise HTTPException(
            429,
            {
                "error_code": "REROUTE_RECENTLY_CLONED",
                "message": (
                    f"A RAG clone of project {project_id} was created within "
                    f"the last hour. Open the existing clone instead of "
                    f"creating another."
                ),
                "metadata": {
                    "existing_clone_id": existing_clone.id,
                    "existing_clone_name": existing_clone.name,
                    "existing_clone_created_at": existing_clone.created_at.isoformat(),
                    "window_seconds": _REROUTE_IDEMPOTENCY_WINDOW_SECONDS,
                },
            },
        )

    if async_job:
        # Hardening Phase H1 — enqueue the clone as a background job
        # so the user isn't blocked on the file-copy + BM25 index
        # build (~5-30s for a 200-row gold set). The notification
        # bell surfaces progress; clicking the completed job opens
        # the new project.
        from fastapi.responses import JSONResponse

        from app.services.jobs_service import (
            JobProgressHandle,
            serialize_job,
            start_job,
        )

        async def _runner(handle: JobProgressHandle) -> dict:
            await handle.set_progress(
                message="Copying gold set + raw + prepared files…"
            )
            from app.database import async_session_factory

            async with async_session_factory() as runner_db:
                try:
                    new_project = await clone_project_for_rag(
                        runner_db,
                        source_project_id=project_id,
                        name_suffix=data.name_suffix,
                    )
                except RagCloneError as inner:
                    # Re-raise as a plain error string the wrapper
                    # captures into Job.error. Status code mapping
                    # only matters for the sync endpoint.
                    raise RuntimeError(str(inner)) from inner
                await handle.set_progress(
                    fraction=0.85, message="Building BM25 retrieval index…"
                )
                await runner_db.commit()
                # Re-read to get persisted state for the result.
                await runner_db.refresh(new_project)
                return {
                    "new_project_id": new_project.id,
                    "new_project_name": new_project.name,
                    "source_project_id": project_id,
                    "clone_report": (new_project.runtime_config or {}).get(
                        "clone_report"
                    ),
                }

        job = await start_job(
            db,
            kind="reroute_to_rag",
            title=f"Clone to RAG · project #{project_id}",
            runner=_runner,
            project_id=project_id,
            params={
                "source_project_id": project_id,
                "name_suffix": data.name_suffix,
            },
        )
        return JSONResponse(
            status_code=202,
            content=serialize_job(job),
        )

    try:
        new_project = await clone_project_for_rag(
            db,
            source_project_id=project_id,
            name_suffix=data.name_suffix,
        )
    except RagCloneError as exc:
        detail = str(exc)
        if detail == "source_project_not_found":
            raise HTTPException(404, detail) from exc
        raise HTTPException(400, detail) from exc

    await db.commit()
    await db.refresh(new_project)

    return ProjectRerouteToRagResponse(
        new_project_id=new_project.id,
        new_project_name=new_project.name,
        source_project_id=project_id,
        clone_report=(new_project.runtime_config or {}).get("clone_report"),
    )


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


@router.put("/{project_id}/recipe", response_model=ProjectResponse)
async def apply_project_recipe(
    project_id: int,
    data: ProjectRecipeApplyRequest,
    db: AsyncSession = Depends(get_db),
):
    """Snapshot a Theme 2 recipe onto the project and adopt its
    suggested base model. Returns the updated project."""
    try:
        project = await apply_recipe_to_project(db, project_id, data.recipe_id)
    except RecipeNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))
    return ProjectResponse.model_validate(project)


@router.delete("/{project_id}/recipe", response_model=ProjectResponse)
async def clear_project_recipe(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Clear the recipe snapshot from a project. Does not roll back
    `base_model_name`; the user can edit that in project settings."""
    try:
        project = await clear_recipe_from_project(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
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


@router.get("/{project_id}/prepared-manifest")
async def get_prepared_manifest(project_id: int):
    """Return the project's prepared/manifest.json contents (or `{}`
    when missing). Lets UI surfaces — the SyntheticPanel in particular
    — auto-detect task_profile / scoring_mode / labels / entity_types /
    output_schema without having to scan the filesystem themselves.
    """

    from app.services.eval_task_handler_service import read_prepared_manifest

    return read_prepared_manifest(project_id)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a project and all associated data."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")
    # Gold-set workbench tables have no ORM cascade from Dataset (ambiguous
    # multi-FK), so purge them explicitly before the cascade drops the datasets
    # they key off — otherwise they leak orphaned rows.
    await gold_workbench_service.purge_gold_sets_for_project(db, project.id)
    await db.delete(project)


@router.get("/{project_id}/refine-plan")
async def get_pipeline_plan_refinement(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Phase 1 — deterministic plan-refinement report: the current pipeline plan,
    a privacy-safe aggregate data profile, and a plan-fit roll-up (does the plan
    suit the measured data?). No cloud call; ``cloud_refinement.available`` is
    False. The aggregate profile is the only thing a later cloud pass may send
    off-box — never the user's ingested rows."""
    from app.services.pipeline_refinement_service import refine_pipeline_plan

    try:
        return await refine_pipeline_plan(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


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

    policy = project.gate_policy if isinstance(project.gate_policy, dict) else {}
    must_pass = bool(policy.get("must_pass", False))
    blocked_if_missing = bool(policy.get("blocked_if_missing", False))
    try:
        min_score = max(0.0, min(1.0, float(policy.get("min_score", 0.0))))
    except (TypeError, ValueError):
        min_score = 0.0

    is_blocked = False
    reasons = []

    if must_pass and not report.get("passed"):
        is_blocked = True
        reasons.append("Mandatory quality gates failed.")

    if blocked_if_missing and report.get("missing_required_metrics"):
        is_blocked = True
        reasons.append(f"Missing required metrics: {', '.join(report.get('missing_required_metrics', []))}")

    checks = [item for item in list(report.get("checks") or []) if isinstance(item, dict)]
    scored = [bool(item.get("passed")) for item in checks if item.get("actual") is not None]
    gate_score = None
    if scored:
        gate_score = round(sum(1 for ok in scored if ok) / float(len(scored)), 6)
    if min_score > 0.0 and (gate_score is None or gate_score < min_score):
        is_blocked = True
        if gate_score is None:
            reasons.append("Gate score unavailable because no comparable metrics were evaluated.")
        else:
            reasons.append(f"Gate score {gate_score:.3f} is below min_score {min_score:.3f}.")

    return {
        "project_id": project_id,
        "experiment_id": experiment.id,
        "passed": not is_blocked,
        "is_blocked": is_blocked,
        "reasons": reasons,
        "policy": {
            "must_pass": must_pass,
            "min_score": min_score,
            "blocked_if_missing": blocked_if_missing,
        },
        "gate_score": gate_score,
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


@router.post("/{project_id}/smoke-test")
async def run_project_smoke_test(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Run a parallel, read-only health check across every project
    surface (recipe / gold / data-health / forecast / synth /
    experiments / etc.).

    Diagnostics Intervention C — one button that answers "is anything
    obviously broken before I commit to a real run?". Each check
    returns a structured result; failures carry the same envelope
    shape ``<ErrorPanel>`` renders elsewhere so the frontend can show
    the troubleshooting_id + remediation copy inline.

    Read-only by design — no writes, no GPU, no expensive LLM calls.
    Safe to re-run as often as the user wants.
    """
    from app.services.project_smoke_test_service import (
        run_smoke_test,
        serialize_summary,
    )

    summary = await run_smoke_test(db, project_id)
    return serialize_summary(summary)


# ─────────────────────────────────────────────────────────────────────
# Arc H — End-goal contract + progress ledger.
#
# Coach Mode + Data Studio render "% toward your stated goal" from this
# data. The progress endpoint is read-only and safe to poll; setting /
# clearing the goal is one-shot per call.
# ─────────────────────────────────────────────────────────────────────


class GoalSetRequest(BaseModel):
    """User-stated goal payload. ``target_metric`` must be one of the
    supported metrics on goal_service.SUPPORTED_METRICS; the service
    raises ValueError otherwise (translated to 400 here)."""

    target_metric: str
    target_threshold: float
    deadline: str | None = None
    title: str | None = None


@router.get("/{project_id}/goal/progress")
async def get_project_goal_progress(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the goal progress ledger for the project.

    Always returns a fully-formed payload: when no goal is set, the
    service falls back to a sensible default (f1 ≥ 0.70) and the
    response carries ``has_explicit_goal: false`` so the UI can prompt
    the user to state their own goal.
    """
    from app.services.goal_service import compute_progress
    try:
        return await compute_progress(db, project_id)
    except ValueError as exc:
        detail = str(exc)
        if detail.startswith(f"Project {project_id} not found"):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.put("/{project_id}/goal")
async def set_project_goal(
    project_id: int,
    req: GoalSetRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist the user's stated goal on the project."""
    from app.services.goal_service import set_goal
    try:
        goal = await set_goal(
            db,
            project_id,
            target_metric=req.target_metric,
            target_threshold=req.target_threshold,
            deadline=req.deadline,
            title=req.title,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail.startswith(f"Project {project_id} not found"):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)
    await db.commit()
    return {"project_id": project_id, "goal": goal}


@router.delete("/{project_id}/goal")
async def clear_project_goal(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Drop the project's stated goal. Idempotent."""
    from app.services.goal_service import clear_goal
    try:
        await clear_goal(db, project_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    await db.commit()
    return {"project_id": project_id, "cleared": True}


# ── Quality-Lift phase 2, slice 1 — Slice definitions CRUD ─────────────
#
# Three endpoints:
#   GET    .../slice-definitions  — read; returns ``{"slices": []}`` when
#                                   nothing has been configured so the
#                                   editor never has to special-case null.
#   PUT    .../slice-definitions  — replace (idempotent); body goes
#                                   through the service validator before
#                                   landing on the column.
#   DELETE .../slice-definitions  — drop; equivalent to PUT with empty
#                                   slices but more explicit when the
#                                   user wants to nuke slicing entirely.


@router.get(
    "/{project_id}/slice-definitions",
    response_model=SliceDefinitionsResponse,
)
async def get_project_slice_definitions(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Read this project's slice definitions. Returns ``{"slices": []}``
    when none configured so the editor has a stable shape to render.
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    payload = project.slice_definitions or {"slices": []}
    return {"project_id": project_id, "slice_definitions": payload}


@router.put(
    "/{project_id}/slice-definitions",
    response_model=SliceDefinitionsResponse,
)
async def set_project_slice_definitions(
    project_id: int,
    payload: SliceDefinitionsPayload,
    db: AsyncSession = Depends(get_db),
):
    """Replace this project's slice definitions in one shot.

    The Pydantic schemas are intentionally thin — closed-set op
    validation, slice_id regex, per-project caps, and regex
    compilability all live in
    ``slice_definitions_service.validate_slice_definitions`` so the same
    code-path runs whether the payload arrives here or via a future
    bulk-import flow. A ``SliceValidationError`` surfaces verbatim
    to the editor so the user sees a precise inline error
    ("slice ``long_input``: regex ``[`` is invalid").
    """
    from app.services.slice_definitions_service import (
        validate_slice_definitions,
        SliceValidationError,
    )

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    try:
        cleaned = validate_slice_definitions(payload.model_dump())
    except SliceValidationError as exc:
        raise HTTPException(400, str(exc))

    project.slice_definitions = cleaned
    await db.commit()
    return {"project_id": project_id, "slice_definitions": cleaned}


@router.delete(
    "/{project_id}/slice-definitions",
    response_model=SliceDefinitionsResponse,
)
async def clear_project_slice_definitions(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Drop all slice definitions. Idempotent — clearing an already-empty
    column is a 200 with the empty payload, not a 404."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    project.slice_definitions = None
    await db.commit()
    return {"project_id": project_id, "slice_definitions": {"slices": []}}

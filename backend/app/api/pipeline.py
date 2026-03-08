"""Pipeline status API routes."""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
import app.database as database_module
from app.database import get_db
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.models.export import Export, ExportStatus
from app.models.project import PipelineStage, Project
from app.pipeline.orchestrator import (
    can_rollback,
    get_next_stage,
    get_pipeline_status,
    get_progress_percent,
)
from app.services.artifact_registry_service import (
    publish_artifact_batch,
    serialize_artifact,
)
from app.services.evaluation_pack_service import evaluate_experiment_auto_gates
from app.services.pipeline_recipe_service import (
    DEFAULT_PIPELINE_RECIPE_ID,
    apply_pipeline_recipe_blueprint,
    get_pipeline_recipe_state,
    list_pipeline_recipe_executions,
    list_available_training_recipe_ids,
    list_pipeline_recipes,
    load_pipeline_recipe_execution,
    patch_pipeline_recipe_execution,
    patch_pipeline_recipe_state,
    resolve_pipeline_recipe_blueprint,
    save_pipeline_recipe_execution,
)
from app.services.workflow_runner_service import (
    create_workflow_run_shell,
    get_workflow_run,
    list_workflow_runs,
    mark_workflow_run_failed,
    run_workflow_graph,
    serialize_workflow_run,
)
from app.services.workflow_graph_service import (
    compile_workflow_graph,
    delete_workflow_graph_override,
    build_workflow_dry_run,
    get_workflow_graph_templates,
    get_step_contract_catalog,
    load_saved_workflow_graph_override,
    list_pipeline_run_records,
    persist_pipeline_run_record,
    prepare_workflow_step_run,
    resolve_project_workflow_graph,
    resolve_workflow_graph,
    save_workflow_graph_override,
)

router = APIRouter(prefix="/projects/{project_id}/pipeline", tags=["Pipeline"])


class GraphValidateRequest(BaseModel):
    graph: dict | None = None
    allow_fallback: bool = True


class GraphDryRunRequest(BaseModel):
    graph: dict | None = None
    allow_fallback: bool = True
    use_saved_override: bool = True


class GraphCompileRequest(BaseModel):
    graph: dict | None = None
    allow_fallback: bool = True
    use_saved_override: bool = True


class GraphContractSaveRequest(BaseModel):
    graph: dict


class GraphRunStepRequest(BaseModel):
    graph: dict | None = None
    allow_fallback: bool = True
    use_saved_override: bool = True
    stage: PipelineStage | None = None
    auto_advance: bool = True
    config: dict[str, object] = Field(default_factory=dict)


class GraphRunWorkflowRequest(BaseModel):
    graph: dict | None = None
    allow_fallback: bool = True
    use_saved_override: bool = True
    execution_backend: str = "local"
    max_retries: int = Field(default=0, ge=0, le=5)
    stop_on_blocked: bool = True
    stop_on_failure: bool = True
    config: dict[str, object] = Field(default_factory=dict)


class PipelineRecipeResolveRequest(BaseModel):
    recipe_id: str = Field(..., min_length=1, max_length=128)
    overrides: dict[str, object] = Field(default_factory=dict)
    include_preflight: bool = True


class PipelineRecipeApplyRequest(BaseModel):
    recipe_id: str = Field(..., min_length=1, max_length=128)
    overrides: dict[str, object] = Field(default_factory=dict)
    include_preflight: bool = True
    enforce_preflight_ok: bool = False
    mark_active: bool = True


class PipelineRecipeRunRequest(BaseModel):
    recipe_id: str = Field(..., min_length=1, max_length=128)
    overrides: dict[str, object] = Field(default_factory=dict)
    include_preflight: bool = True
    enforce_preflight_ok: bool = False
    mark_active: bool = True
    execution_backend: str = "celery"
    max_retries: int = Field(default=0, ge=0, le=5)
    stop_on_blocked: bool = True
    stop_on_failure: bool = True
    config: dict[str, object] = Field(default_factory=dict)
    async_run: bool = True


async def _ensure_stage_requirements(
    db: AsyncSession,
    project: Project,
) -> None:
    """Block stage advancement when required artifacts are missing."""
    stage = project.pipeline_stage
    project_id = project.id

    if stage == PipelineStage.INGESTION:
        doc_result = await db.execute(
            select(RawDocument.id)
            .join(Dataset, Dataset.id == RawDocument.dataset_id)
            .where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.RAW,
                RawDocument.status == DocumentStatus.ACCEPTED,
            )
            .limit(1)
        )
        if doc_result.scalar_one_or_none() is None:
            raise HTTPException(400, "Cannot advance: ingest and process at least one raw document first")
        return

    if stage == PipelineStage.CLEANING:
        cleaned_result = await db.execute(
            select(Dataset.id)
            .where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.CLEANED,
                Dataset.record_count > 0,
            )
            .limit(1)
        )
        if cleaned_result.scalar_one_or_none() is None:
            raise HTTPException(400, "Cannot advance: no cleaned dataset entries available")
        return

    if stage == PipelineStage.DATASET_PREP:
        train_result = await db.execute(
            select(Dataset.id)
            .where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.TRAIN,
                Dataset.record_count > 0,
            )
            .limit(1)
        )
        val_result = await db.execute(
            select(Dataset.id)
            .where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.VALIDATION,
                Dataset.record_count > 0,
            )
            .limit(1)
        )
        if train_result.scalar_one_or_none() is None or val_result.scalar_one_or_none() is None:
            raise HTTPException(400, "Cannot advance: create train/validation splits first")
        return

    if stage == PipelineStage.TRAINING:
        exp_result = await db.execute(
            select(Experiment.id)
            .where(
                Experiment.project_id == project_id,
                Experiment.status == ExperimentStatus.COMPLETED,
            )
            .limit(1)
        )
        if exp_result.scalar_one_or_none() is None:
            raise HTTPException(400, "Cannot advance: complete at least one training experiment first")
        return

    if stage == PipelineStage.EVALUATION:
        eval_result = await db.execute(
            select(EvalResult.id)
            .join(Experiment, Experiment.id == EvalResult.experiment_id)
            .where(Experiment.project_id == project_id)
            .limit(1)
        )
        if eval_result.scalar_one_or_none() is None:
            raise HTTPException(400, "Cannot advance: run evaluation before moving to compression")
        return

    if stage == PipelineStage.COMPRESSION:
        compressed_dir = settings.DATA_DIR / "projects" / str(project_id) / "compressed"
        report_exists = compressed_dir.exists() and any(
            p.name.endswith("_report.json") for p in compressed_dir.rglob("*.json")
        )
        if not report_exists:
            raise HTTPException(400, "Cannot advance: run quantization/benchmark and generate a compression report")
        return

    if stage == PipelineStage.EXPORT:
        export_result = await db.execute(
            select(Export.id)
            .where(
                Export.project_id == project_id,
                Export.status == ExportStatus.COMPLETED,
            )
            .limit(1)
        )
        if export_result.scalar_one_or_none() is None:
            raise HTTPException(400, "Cannot advance: complete at least one export run first")
        return


@router.get("/status")
async def pipeline_status(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get pipeline stage status for a project."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    auto_gate_summary = None
    latest_exp = await db.execute(
        select(Experiment.id)
        .where(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc(), Experiment.id.desc())
        .limit(1)
    )
    latest_experiment_id = latest_exp.scalar_one_or_none()
    if latest_experiment_id is not None:
        try:
            gate_report = await evaluate_experiment_auto_gates(
                db,
                project_id=project_id,
                experiment_id=int(latest_experiment_id),
                pack_id=None,
            )
            auto_gate_summary = {
                "experiment_id": int(latest_experiment_id),
                "pack_id": gate_report.get("pack", {}).get("pack_id"),
                "passed": bool(gate_report.get("passed")),
                "failed_gate_ids": list(gate_report.get("failed_gate_ids") or []),
                "missing_required_metrics": list(gate_report.get("missing_required_metrics") or []),
                "captured_at": gate_report.get("captured_at"),
            }
        except ValueError:
            auto_gate_summary = None

    return {
        "project_id": project.id,
        "current_stage": project.pipeline_stage.value,
        "progress_percent": get_progress_percent(project.pipeline_stage),
        "stages": get_pipeline_status(project.pipeline_stage),
        "auto_gate": auto_gate_summary,
    }


@router.get("/graph")
async def pipeline_graph(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get read-only pipeline graph for visual workflow preview."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    resolved = resolve_project_workflow_graph(
        project_id=project.id,
        current_stage=project.pipeline_stage,
        graph_override=None,
        allow_fallback=True,
        use_saved_override=True,
    )
    payload = dict(resolved.get("graph") or {})
    payload["requested_source"] = resolved.get("requested_source")
    payload["effective_source"] = resolved.get("effective_source")
    payload["has_saved_override"] = resolved.get("has_saved_override")
    return payload


@router.get("/graph/stage-catalog")
async def pipeline_graph_stage_catalog(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List built-in stage contract templates for visual editor workflows."""
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")
    return {
        "project_id": project_id,
        "stages": get_step_contract_catalog(),
    }


@router.get("/graph/templates")
async def pipeline_graph_templates(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List starter workflow graph templates."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")
    return {
        "project_id": project_id,
        "templates": get_workflow_graph_templates(project_id, project.pipeline_stage),
    }


@router.get("/recipes")
async def pipeline_recipe_catalog(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List end-to-end pipeline blueprints and active project recipe state."""
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")

    state = await get_pipeline_recipe_state(db, project_id=project_id)
    return {
        "project_id": project_id,
        "default_recipe_id": DEFAULT_PIPELINE_RECIPE_ID,
        "training_recipe_ids": list_available_training_recipe_ids(),
        "recipes": list_pipeline_recipes(include_blueprint=False),
        "active_state": state.get("state"),
    }


@router.get("/recipes/state")
async def pipeline_recipe_state(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get persisted project recipe state."""
    try:
        return await get_pipeline_recipe_state(db, project_id=project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/recipes/resolve")
async def pipeline_recipe_resolve(
    project_id: int,
    req: PipelineRecipeResolveRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resolve one pipeline recipe into concrete project settings/graph/config."""
    try:
        return await resolve_pipeline_recipe_blueprint(
            db,
            project_id=project_id,
            recipe_id=req.recipe_id,
            overrides=dict(req.overrides or {}),
            include_preflight=bool(req.include_preflight),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/recipes/apply")
async def pipeline_recipe_apply(
    project_id: int,
    req: PipelineRecipeApplyRequest,
    db: AsyncSession = Depends(get_db),
):
    """Apply an end-to-end recipe to project runtime defaults and workflow graph."""
    try:
        return await apply_pipeline_recipe_blueprint(
            db,
            project_id=project_id,
            recipe_id=req.recipe_id,
            overrides=dict(req.overrides or {}),
            include_preflight=bool(req.include_preflight),
            enforce_preflight_ok=bool(req.enforce_preflight_ok),
            mark_active=bool(req.mark_active),
        )
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


def _decorate_recipe_execution_payload(
    payload: dict[str, object],
    workflow_run: dict | None,
) -> dict[str, object]:
    row = dict(payload)
    if isinstance(workflow_run, dict):
        row["workflow_status"] = str(workflow_run.get("status") or row.get("status") or "unknown")
        row["workflow_summary"] = workflow_run.get("summary") if isinstance(workflow_run.get("summary"), dict) else {}
        row["workflow_run"] = workflow_run
    else:
        row["workflow_status"] = str(row.get("status") or "unknown")
        row.setdefault("workflow_summary", {})
        row["workflow_run"] = None
    return row


@router.post("/recipes/run")
async def pipeline_recipe_run(
    project_id: int,
    req: PipelineRecipeRunRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resolve/apply recipe and execute workflow DAG with persisted recipe-run lineage."""
    try:
        applied = await apply_pipeline_recipe_blueprint(
            db,
            project_id=project_id,
            recipe_id=req.recipe_id,
            overrides=dict(req.overrides or {}),
            include_preflight=bool(req.include_preflight),
            enforce_preflight_ok=bool(req.enforce_preflight_ok),
            mark_active=bool(req.mark_active),
        )
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

    now_iso = datetime.now(timezone.utc).isoformat()
    recipe_run_id = f"recipe-run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    recipe_payload = dict(applied.get("recipe") or {})
    recipe_context = {
        "recipe_run_id": recipe_run_id,
        "recipe_id": str(recipe_payload.get("recipe_id") or req.recipe_id),
        "recipe_manifest_path": str(applied.get("manifest_path") or ""),
        "recipe_state_path": str(applied.get("state_path") or ""),
        "recipe_artifact_id": (applied.get("artifact") or {}).get("id"),
    }

    run_config_payload = dict(req.config or {})
    run_config_payload["recipe_context"] = recipe_context
    shell = await create_workflow_run_shell(
        db=db,
        project_id=project_id,
        execution_backend=str(req.execution_backend),
        run_config={
            "allow_fallback": True,
            "use_saved_override": True,
            "max_retries": int(req.max_retries),
            "stop_on_blocked": bool(req.stop_on_blocked),
            "stop_on_failure": bool(req.stop_on_failure),
            "config": run_config_payload,
            "recipe_context": recipe_context,
        },
        summary={
            "queued": bool(req.async_run),
            "recipe_context": recipe_context,
        },
    )
    await db.commit()
    await db.refresh(shell)

    execution_payload: dict[str, object] = {
        "project_id": project_id,
        "recipe_run_id": recipe_run_id,
        "recipe_id": recipe_context["recipe_id"],
        "workflow_run_id": int(shell.id),
        "execution_backend": str(req.execution_backend),
        "status": "queued" if req.async_run else "running",
        "created_at": now_iso,
        "started_at": now_iso,
        "updated_at": now_iso,
        "recipe": recipe_payload,
        "resolved": dict(applied.get("resolved") or {}),
        "preflight": applied.get("preflight"),
        "warnings": list(applied.get("warnings") or []),
        "manifest_path": str(applied.get("manifest_path") or ""),
        "state_path": str(applied.get("state_path") or ""),
        "recipe_artifact": dict(applied.get("artifact") or {}),
        "workflow_run_status_endpoint": f"/api/projects/{project_id}/pipeline/graph/workflow-runs/{int(shell.id)}",
        "run_options": {
            "execution_backend": str(req.execution_backend),
            "max_retries": int(req.max_retries),
            "stop_on_blocked": bool(req.stop_on_blocked),
            "stop_on_failure": bool(req.stop_on_failure),
            "async_run": bool(req.async_run),
            "config": dict(req.config or {}),
        },
    }
    execution_path = save_pipeline_recipe_execution(
        project_id,
        recipe_run_id=recipe_run_id,
        payload=execution_payload,
    )
    state_payload, state_path = patch_pipeline_recipe_state(
        project_id,
        patch={
            "last_execution_recipe_run_id": recipe_run_id,
            "last_execution_workflow_run_id": int(shell.id),
            "last_execution_status": execution_payload["status"],
            "last_execution_started_at": now_iso,
            "last_execution_updated_at": now_iso,
        },
    )

    if not req.async_run:
        project = await db.get(Project, project_id)
        if project is None:
            raise HTTPException(404, f"Project {project_id} not found")
        run_payload = await run_workflow_graph(
            db=db,
            project=project,
            graph_override=None,
            allow_fallback=True,
            use_saved_override=True,
            execution_backend=str(req.execution_backend),
            max_retries=int(req.max_retries),
            stop_on_blocked=bool(req.stop_on_blocked),
            stop_on_failure=bool(req.stop_on_failure),
            config=run_config_payload,
            run_id=int(shell.id),
            commit_progress=True,
        )
        finished_at = datetime.now(timezone.utc).isoformat()
        execution_payload, execution_path = patch_pipeline_recipe_execution(
            project_id,
            recipe_run_id=recipe_run_id,
            patch={
                "status": str(run_payload.get("status") or "unknown"),
                "finished_at": finished_at,
                "updated_at": finished_at,
                "workflow_summary": run_payload.get("summary") if isinstance(run_payload.get("summary"), dict) else {},
            },
        )
        state_payload, state_path = patch_pipeline_recipe_state(
            project_id,
            patch={
                "last_execution_status": execution_payload.get("status"),
                "last_execution_finished_at": finished_at,
                "last_execution_updated_at": finished_at,
            },
        )
        return {
            "project_id": project_id,
            "queued": False,
            "recipe_run_id": recipe_run_id,
            "execution_path": execution_path,
            "state_path": state_path,
            "state": state_payload,
            "execution": _decorate_recipe_execution_payload(execution_payload, run_payload),
        }

    queued_project_id = int(project_id)
    queued_run_id = int(shell.id)
    queued_recipe_run_id = str(recipe_run_id)
    queued_backend = str(req.execution_backend)
    queued_retries = int(req.max_retries)
    queued_stop_on_blocked = bool(req.stop_on_blocked)
    queued_stop_on_failure = bool(req.stop_on_failure)
    queued_config = dict(run_config_payload)

    async def _execute_recipe_workflow_run() -> None:
        async with database_module.async_session_factory() as task_db:
            finished_at = datetime.now(timezone.utc).isoformat()
            try:
                project_row = await task_db.get(Project, queued_project_id)
                if project_row is None:
                    await mark_workflow_run_failed(
                        db=task_db,
                        project_id=queued_project_id,
                        run_id=queued_run_id,
                        message="project not found during recipe-run execution",
                    )
                    await task_db.commit()
                    patch_pipeline_recipe_execution(
                        queued_project_id,
                        recipe_run_id=queued_recipe_run_id,
                        patch={
                            "status": "failed",
                            "finished_at": finished_at,
                            "updated_at": finished_at,
                            "error": "project not found during recipe-run execution",
                        },
                    )
                    patch_pipeline_recipe_state(
                        queued_project_id,
                        patch={
                            "last_execution_status": "failed",
                            "last_execution_finished_at": finished_at,
                            "last_execution_updated_at": finished_at,
                        },
                    )
                    return

                run_payload = await run_workflow_graph(
                    db=task_db,
                    project=project_row,
                    graph_override=None,
                    allow_fallback=True,
                    use_saved_override=True,
                    execution_backend=queued_backend,
                    max_retries=queued_retries,
                    stop_on_blocked=queued_stop_on_blocked,
                    stop_on_failure=queued_stop_on_failure,
                    config=queued_config,
                    run_id=queued_run_id,
                    commit_progress=True,
                )
                finished_at = datetime.now(timezone.utc).isoformat()
                status = str(run_payload.get("status") or "unknown")
                patch_pipeline_recipe_execution(
                    queued_project_id,
                    recipe_run_id=queued_recipe_run_id,
                    patch={
                        "status": status,
                        "finished_at": finished_at,
                        "updated_at": finished_at,
                        "workflow_summary": run_payload.get("summary")
                        if isinstance(run_payload.get("summary"), dict)
                        else {},
                    },
                )
                patch_pipeline_recipe_state(
                    queued_project_id,
                    patch={
                        "last_execution_status": status,
                        "last_execution_finished_at": finished_at,
                        "last_execution_updated_at": finished_at,
                    },
                )
            except Exception as e:
                await mark_workflow_run_failed(
                    db=task_db,
                    project_id=queued_project_id,
                    run_id=queued_run_id,
                    message=str(e),
                )
                await task_db.commit()
                finished_at = datetime.now(timezone.utc).isoformat()
                patch_pipeline_recipe_execution(
                    queued_project_id,
                    recipe_run_id=queued_recipe_run_id,
                    patch={
                        "status": "failed",
                        "finished_at": finished_at,
                        "updated_at": finished_at,
                        "error": str(e),
                    },
                )
                patch_pipeline_recipe_state(
                    queued_project_id,
                    patch={
                        "last_execution_status": "failed",
                        "last_execution_finished_at": finished_at,
                        "last_execution_updated_at": finished_at,
                    },
                )

    def _run_in_background_thread() -> None:
        asyncio.run(_execute_recipe_workflow_run())

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_in_background_thread)

    return {
        "project_id": project_id,
        "queued": True,
        "recipe_run_id": recipe_run_id,
        "execution_path": execution_path,
        "state_path": state_path,
        "state": state_payload,
        "execution": _decorate_recipe_execution_payload(execution_payload, serialize_workflow_run(shell, [])),
    }


@router.get("/recipes/runs")
async def list_pipeline_recipe_runs(
    project_id: int,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """List persisted pipeline recipe runs with linked workflow run summaries."""
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")

    rows = list_pipeline_recipe_executions(project_id, limit=limit)
    if not rows:
        return {
            "project_id": project_id,
            "limit": max(1, min(limit, 200)),
            "count": 0,
            "runs": [],
        }

    workflow_rows = await list_workflow_runs(db=db, project_id=project_id, limit=max(limit, 100))
    workflow_by_id = {int(item.get("id")): item for item in workflow_rows if isinstance(item, dict) and item.get("id") is not None}

    runs: list[dict[str, object]] = []
    for row in rows:
        workflow_run_id_raw = row.get("workflow_run_id")
        workflow_run_id: int | None = None
        try:
            if workflow_run_id_raw not in (None, "", 0):
                workflow_run_id = int(workflow_run_id_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            workflow_run_id = None
        workflow_row = workflow_by_id.get(workflow_run_id) if workflow_run_id is not None else None
        runs.append(_decorate_recipe_execution_payload(row, workflow_row))

    return {
        "project_id": project_id,
        "limit": max(1, min(limit, 200)),
        "count": len(runs),
        "runs": runs,
    }


@router.get("/recipes/runs/{recipe_run_id}")
async def get_pipeline_recipe_run(
    project_id: int,
    recipe_run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get one recipe run execution payload and linked workflow run details."""
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")

    row = load_pipeline_recipe_execution(project_id, recipe_run_id=recipe_run_id)
    if row is None:
        raise HTTPException(404, f"Recipe run {recipe_run_id} not found")

    workflow_run = None
    workflow_run_id_raw = row.get("workflow_run_id")
    if workflow_run_id_raw not in (None, "", 0):
        try:
            workflow_run = await get_workflow_run(
                db=db,
                project_id=project_id,
                run_id=int(workflow_run_id_raw),
            )
        except (TypeError, ValueError):
            workflow_run = None

    return _decorate_recipe_execution_payload(row, workflow_run)


@router.get("/graph/contract")
async def get_pipeline_graph_contract(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return effective graph contract and saved-override metadata."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    saved = load_saved_workflow_graph_override(project_id)
    resolved = resolve_project_workflow_graph(
        project_id=project.id,
        current_stage=project.pipeline_stage,
        graph_override=None,
        allow_fallback=True,
        use_saved_override=True,
    )
    return {
        "project_id": project.id,
        "current_stage": project.pipeline_stage.value,
        "has_saved_override": saved is not None,
        "requested_source": resolved.get("requested_source"),
        "effective_source": resolved.get("effective_source"),
        "graph": resolved.get("graph"),
    }


@router.put("/graph/contract")
async def save_pipeline_graph_contract(
    project_id: int,
    req: GraphContractSaveRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist project graph override after strict validation."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    resolved = resolve_workflow_graph(
        project_id=project.id,
        current_stage=project.pipeline_stage,
        graph_override=req.graph,
        allow_fallback=False,
    )
    if not resolved.get("valid"):
        raise HTTPException(
            400,
            {
                "message": "Graph contract validation failed",
                "errors": resolved.get("errors"),
                "warnings": resolved.get("warnings"),
            },
        )

    normalized_graph = resolved.get("graph")
    if not isinstance(normalized_graph, dict):
        raise HTTPException(400, "Graph normalization failed")

    saved_path = save_workflow_graph_override(project.id, normalized_graph)
    return {
        "project_id": project.id,
        "saved": True,
        "path": saved_path,
        "graph": normalized_graph,
    }


@router.delete("/graph/contract")
async def reset_pipeline_graph_contract(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Reset project graph override to default fallback graph."""
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")

    deleted = delete_workflow_graph_override(project_id)
    return {
        "project_id": project_id,
        "reset": deleted,
    }


@router.post("/graph/validate")
async def validate_pipeline_graph(
    project_id: int,
    req: GraphValidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Validate a graph override and resolve fallback behavior."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    resolved = resolve_project_workflow_graph(
        project_id=project.id,
        current_stage=project.pipeline_stage,
        graph_override=req.graph,
        allow_fallback=req.allow_fallback,
        use_saved_override=req.graph is None,
    )
    return {
        "project_id": project.id,
        "current_stage": project.pipeline_stage.value,
        "valid": resolved.get("valid"),
        "requested_source": resolved.get("requested_source"),
        "effective_source": resolved.get("effective_source"),
        "has_saved_override": resolved.get("has_saved_override"),
        "fallback_used": resolved.get("fallback_used"),
        "errors": resolved.get("errors"),
        "warnings": resolved.get("warnings"),
        "graph": resolved.get("graph"),
    }


@router.post("/graph/compile")
async def compile_pipeline_graph(
    project_id: int,
    req: GraphCompileRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compile resolved graph into actionable diagnostics for execution readiness."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    return await compile_workflow_graph(
        db=db,
        project=project,
        graph_override=req.graph,
        allow_fallback=req.allow_fallback,
        use_saved_override=req.use_saved_override,
    )


@router.post("/graph/dry-run")
async def dry_run_pipeline_graph(
    project_id: int,
    req: GraphDryRunRequest,
    db: AsyncSession = Depends(get_db),
):
    """Preview step readiness against currently available project artifacts."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    return await build_workflow_dry_run(
        db=db,
        project=project,
        graph_override=req.graph,
        allow_fallback=req.allow_fallback,
        use_saved_override=req.use_saved_override,
    )


@router.post("/graph/run-step")
async def run_pipeline_graph_step(
    project_id: int,
    req: GraphRunStepRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run active pipeline step in phase 2 contract runtime mode."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    requested_stage = req.stage or project.pipeline_stage
    step_run = await prepare_workflow_step_run(
        db=db,
        project=project,
        stage=requested_stage,
        graph_override=req.graph,
        allow_fallback=req.allow_fallback,
        use_saved_override=req.use_saved_override,
        config=req.config,
    )
    step_run["auto_advance"] = bool(req.auto_advance)

    if step_run.get("status") == "ready" and req.auto_advance:
        try:
            await _ensure_stage_requirements(db, project)
        except HTTPException as e:
            step_run["status"] = "blocked"
            step_run["errors"] = [*step_run.get("errors", []), str(e.detail)]
            step_run["can_execute"] = False
        else:
            next_stage = get_next_stage(project.pipeline_stage)
            if next_stage is None:
                step_run["status"] = "completed"
                step_run["current_stage"] = project.pipeline_stage.value
                step_run["advanced"] = False
            else:
                previous_stage = project.pipeline_stage
                project.pipeline_stage = next_stage
                await db.flush()
                await db.refresh(project)
                step_run["status"] = "completed"
                step_run["previous_stage"] = previous_stage.value
                step_run["current_stage"] = next_stage.value
                step_run["advanced"] = True
    else:
        step_run["advanced"] = False

    published_artifacts: list[dict] = []
    if step_run.get("status") == "completed" and bool(step_run.get("can_execute")):
        declared_outputs = [
            str(item).strip()
            for item in step_run.get("declared_outputs", [])
            if isinstance(item, str) and str(item).strip()
        ]
        if declared_outputs:
            published_rows = await publish_artifact_batch(
                db=db,
                project_id=project.id,
                artifact_keys=declared_outputs,
                producer_stage=requested_stage.value,
                producer_run_id=str(step_run.get("run_id", "")),
                producer_step_id=str(step_run.get("step_node_id", "")),
                metadata={
                    "source": "pipeline.graph.run_step",
                    "auto_advance": bool(req.auto_advance),
                },
            )
            published_artifacts = [serialize_artifact(row) for row in published_rows]

    step_run["published_artifacts"] = published_artifacts
    step_run["published_artifact_keys"] = [row["artifact_key"] for row in published_artifacts]
    step_run["run_finished_at"] = datetime.now(timezone.utc).isoformat()
    run_record_path = persist_pipeline_run_record(project.id, step_run)
    step_run["run_record_path"] = run_record_path
    return step_run


@router.get("/graph/runs")
async def list_pipeline_graph_runs(
    project_id: int,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """List recent pipeline graph step run records."""
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")
    safe_limit = max(1, min(limit, 100))
    runs = list_pipeline_run_records(project_id, safe_limit)
    return {
        "project_id": project_id,
        "limit": safe_limit,
        "count": len(runs),
        "runs": runs,
    }


@router.post("/graph/run")
async def run_pipeline_graph_workflow(
    project_id: int,
    req: GraphRunWorkflowRequest,
    db: AsyncSession = Depends(get_db),
):
    """Execute workflow DAG run with dependency tracking and retries."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    return await run_workflow_graph(
        db=db,
        project=project,
        graph_override=req.graph,
        allow_fallback=req.allow_fallback,
        use_saved_override=req.use_saved_override,
        execution_backend=req.execution_backend,
        max_retries=req.max_retries,
        stop_on_blocked=req.stop_on_blocked,
        stop_on_failure=req.stop_on_failure,
        config=dict(req.config),
    )


@router.post("/graph/run-async")
async def run_pipeline_graph_workflow_async(
    project_id: int,
    req: GraphRunWorkflowRequest,
    db: AsyncSession = Depends(get_db),
):
    """Queue workflow DAG run in background and return immediately."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    shell = await create_workflow_run_shell(
        db=db,
        project_id=project.id,
        execution_backend=req.execution_backend,
        run_config={
            "allow_fallback": req.allow_fallback,
            "use_saved_override": req.use_saved_override,
            "max_retries": req.max_retries,
            "stop_on_blocked": req.stop_on_blocked,
            "stop_on_failure": req.stop_on_failure,
            "config": dict(req.config),
        },
        summary={"queued": True},
    )
    await db.commit()
    await db.refresh(shell)

    queued_run_id = int(shell.id)
    queued_project_id = int(project.id)
    queued_graph = dict(req.graph) if isinstance(req.graph, dict) else None
    queued_config = dict(req.config)
    queued_allow_fallback = bool(req.allow_fallback)
    queued_use_saved_override = bool(req.use_saved_override)
    queued_execution_backend = str(req.execution_backend)
    queued_max_retries = int(req.max_retries)
    queued_stop_on_blocked = bool(req.stop_on_blocked)
    queued_stop_on_failure = bool(req.stop_on_failure)

    async def _execute_workflow_run() -> None:
        async with database_module.async_session_factory() as task_db:
            try:
                project_row = await task_db.get(Project, queued_project_id)
                if project_row is None:
                    await mark_workflow_run_failed(
                        db=task_db,
                        project_id=queued_project_id,
                        run_id=queued_run_id,
                        message="project not found during background execution",
                    )
                    await task_db.commit()
                    return

                await run_workflow_graph(
                    db=task_db,
                    project=project_row,
                    graph_override=queued_graph,
                    allow_fallback=queued_allow_fallback,
                    use_saved_override=queued_use_saved_override,
                    execution_backend=queued_execution_backend,
                    max_retries=queued_max_retries,
                    stop_on_blocked=queued_stop_on_blocked,
                    stop_on_failure=queued_stop_on_failure,
                    config=queued_config,
                    run_id=queued_run_id,
                    commit_progress=True,
                )
            except Exception as e:
                await mark_workflow_run_failed(
                    db=task_db,
                    project_id=queued_project_id,
                    run_id=queued_run_id,
                    message=str(e),
                )
                await task_db.commit()

    def _run_in_background_thread() -> None:
        asyncio.run(_execute_workflow_run())

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_in_background_thread)

    return {
        "project_id": project.id,
        "queued": True,
        "run_id": queued_run_id,
        "run": serialize_workflow_run(shell, []),
    }


@router.get("/graph/workflow-runs")
async def list_pipeline_workflow_runs(
    project_id: int,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """List persisted workflow DAG runs."""
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")
    safe_limit = max(1, min(limit, 200))
    runs = await list_workflow_runs(db=db, project_id=project_id, limit=safe_limit)
    return {
        "project_id": project_id,
        "limit": safe_limit,
        "count": len(runs),
        "runs": runs,
    }


@router.get("/graph/workflow-runs/{run_id}")
async def get_pipeline_workflow_run(
    project_id: int,
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get one workflow DAG run with node attempt details."""
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")

    payload = await get_workflow_run(db=db, project_id=project_id, run_id=run_id)
    if payload is None:
        raise HTTPException(404, f"Workflow run {run_id} not found")
    return payload


@router.post("/advance")
async def advance_pipeline(project_id: int, db: AsyncSession = Depends(get_db)):
    """Advance the project to the next pipeline stage."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    await _ensure_stage_requirements(db, project)

    next_stage = get_next_stage(project.pipeline_stage)
    if next_stage is None:
        raise HTTPException(400, "Pipeline already completed")

    previous_stage = project.pipeline_stage
    project.pipeline_stage = next_stage
    await db.flush()
    await db.refresh(project)

    return {
        "project_id": project.id,
        "previous_stage": previous_stage.value,
        "current_stage": next_stage.value,
        "progress_percent": get_progress_percent(next_stage),
    }


@router.post("/rollback")
async def rollback_pipeline(
    project_id: int,
    target_stage: PipelineStage,
    db: AsyncSession = Depends(get_db),
):
    """Rollback the project to a previous pipeline stage."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    if not can_rollback(project.pipeline_stage, target_stage):
        raise HTTPException(400, f"Cannot rollback from {project.pipeline_stage.value} to {target_stage.value}")

    project.pipeline_stage = target_stage
    await db.flush()
    await db.refresh(project)

    return {
        "project_id": project.id,
        "current_stage": target_stage.value,
        "progress_percent": get_progress_percent(target_stage),
    }

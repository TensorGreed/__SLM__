"""Pipeline status API routes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
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
from app.services.workflow_graph_service import (
    build_readonly_pipeline_graph,
    build_workflow_dry_run,
    list_pipeline_run_records,
    persist_pipeline_run_record,
    prepare_workflow_step_run,
    resolve_workflow_graph,
)

router = APIRouter(prefix="/projects/{project_id}/pipeline", tags=["Pipeline"])


class GraphValidateRequest(BaseModel):
    graph: dict | None = None
    allow_fallback: bool = True


class GraphDryRunRequest(BaseModel):
    graph: dict | None = None
    allow_fallback: bool = True


class GraphRunStepRequest(BaseModel):
    graph: dict | None = None
    allow_fallback: bool = True
    stage: PipelineStage | None = None
    auto_advance: bool = True
    config: dict[str, object] = Field(default_factory=dict)


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

    return {
        "project_id": project.id,
        "current_stage": project.pipeline_stage.value,
        "progress_percent": get_progress_percent(project.pipeline_stage),
        "stages": get_pipeline_status(project.pipeline_stage),
    }


@router.get("/graph")
async def pipeline_graph(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get read-only pipeline graph for visual workflow preview."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    return build_readonly_pipeline_graph(project.id, project.pipeline_stage)


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

    resolved = resolve_workflow_graph(
        project_id=project.id,
        current_stage=project.pipeline_stage,
        graph_override=req.graph,
        allow_fallback=req.allow_fallback,
    )
    return {
        "project_id": project.id,
        "current_stage": project.pipeline_stage.value,
        "valid": resolved.get("valid"),
        "fallback_used": resolved.get("fallback_used"),
        "errors": resolved.get("errors"),
        "warnings": resolved.get("warnings"),
        "graph": resolved.get("graph"),
    }


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

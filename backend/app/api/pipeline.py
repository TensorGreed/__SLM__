"""Pipeline status API routes."""

from fastapi import APIRouter, Depends, HTTPException
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

router = APIRouter(prefix="/projects/{project_id}/pipeline", tags=["Pipeline"])


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

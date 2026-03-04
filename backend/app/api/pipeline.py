"""Pipeline status API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import PipelineStage, Project
from app.pipeline.orchestrator import (
    can_rollback,
    get_next_stage,
    get_pipeline_status,
    get_progress_percent,
)

router = APIRouter(prefix="/projects/{project_id}/pipeline", tags=["Pipeline"])


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

"""Training API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.training import ExperimentCreate, ExperimentResponse
from app.services.training_service import (
    create_experiment,
    get_training_status,
    list_experiments,
    start_training,
)

router = APIRouter(prefix="/projects/{project_id}/training", tags=["Training"])


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create(
    project_id: int,
    data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new training experiment."""
    exp = await create_experiment(
        db, project_id, data.name, data.config.base_model,
        data.config.model_dump(), data.description, data.config.training_mode,
    )
    return ExperimentResponse.model_validate(exp)


@router.post("/experiments/{experiment_id}/start")
async def start(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Start training for an experiment."""
    try:
        return await start_training(db, experiment_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/experiments/{experiment_id}/status")
async def status(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get training status and metrics."""
    try:
        return await get_training_status(db, experiment_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/experiments", response_model=list[ExperimentResponse])
async def list_all(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all experiments for a project."""
    exps = await list_experiments(db, project_id)
    return [ExperimentResponse.model_validate(e) for e in exps]

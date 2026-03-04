"""Training API routes."""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.models.experiment import Experiment
from app.schemas.training import ExperimentCreate, ExperimentResponse
from app.services.training_service import (
    create_experiment,
    get_training_status,
    list_experiments,
    start_training,
)

router = APIRouter(prefix="/projects/{project_id}/training", tags=["Training"])


@router.websocket("/experiments/{experiment_id}/metrics/ws")
async def metrics_ws(
    websocket: WebSocket,
    project_id: int,
    experiment_id: int,
):
    """WebSocket endpoint for real-time training metrics."""
    from app.services.training_service import active_websockets

    async with async_session_factory() as db:
        result = await db.execute(
            select(Experiment.id).where(
                Experiment.id == experiment_id,
                Experiment.project_id == project_id,
            )
        )
        if not result.scalar_one_or_none():
            await websocket.accept()
            await websocket.close(code=1008)
            return

    await websocket.accept()
    if experiment_id not in active_websockets:
        active_websockets[experiment_id] = []

    active_websockets[experiment_id].append(websocket)
    try:
        while True:
            # Keep connection open, client just listens
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets.get(experiment_id, []):
            active_websockets[experiment_id].remove(websocket)


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create(
    project_id: int,
    data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new training experiment."""
    try:
        exp = await create_experiment(
            db, project_id, data.name, data.config.base_model,
            data.config.model_dump(), data.description, data.config.training_mode,
        )
        return ExperimentResponse.model_validate(exp)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/experiments/{experiment_id}/start")
async def start(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Start training for an experiment."""
    try:
        return await start_training(db, project_id, experiment_id)
    except ValueError as e:
        detail = str(e)
        if "already running" in detail or "already completed" in detail:
            raise HTTPException(409, detail)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/experiments/{experiment_id}/status")
async def status(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get training status and metrics."""
    try:
        return await get_training_status(db, project_id, experiment_id)
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

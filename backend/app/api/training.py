"""Training API routes."""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory, get_db
from app.models.experiment import Experiment
from app.schemas.training import ExperimentCreate, ExperimentResponse
from app.services.training_service import (
    create_experiment,
    get_training_status,
    list_experiments,
    start_training,
    register_websocket,
    unregister_websocket,
)

router = APIRouter(prefix="/projects/{project_id}/training", tags=["Training"])


@router.websocket("/ws/{experiment_id}")
async def ws_training_status(
    websocket: WebSocket,
    project_id: int, # Keep project_id for consistency with other routes, though not directly used in this new implementation
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket endpoint for real-time training metrics and terminal logs.
    """
    await websocket.accept()

    # Verify experiment exists
    exp = await db.get(Experiment, experiment_id)
    if not exp or exp.project_id != project_id: # Added project_id check
        await websocket.close(code=1008, reason="Experiment not found")
        return

    register_websocket(experiment_id, websocket)

    metrics = exp.metrics or []
    try:
        await websocket.send_json({"type": "init", "metrics": metrics})
    except Exception:
        unregister_websocket(experiment_id, websocket)
        return

    # Background task to stream raw text logs from Redis PubSub
    async def log_stream_loop():
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"log:experiment:{experiment_id}")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    log_line = message["data"]
                    try:
                        await websocket.send_json({"type": "log", "text": log_line})
                    except Exception:
                        break  # client disconnected
        finally:
            await pubsub.unsubscribe()
            await redis_client.aclose()

    log_task = asyncio.create_task(log_stream_loop())

    try:
        while True:
            # Keep connection alive; client might send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        pass
    finally:
        log_task.cancel()
        unregister_websocket(experiment_id, websocket)


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

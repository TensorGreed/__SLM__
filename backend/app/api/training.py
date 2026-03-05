"""Training API routes."""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.experiment import Checkpoint, Experiment
from app.schemas.training import ExperimentCreate, ExperimentResponse
from app.services.job_service import get_task_status
from app.services.domain_profile_service import (
    get_project_domain_profile_contract,
    get_training_defaults,
)
from app.services.training_service import (
    cancel_training,
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

    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    exp = exp_result.scalar_one_or_none()
    if not exp:
        await websocket.close(code=1008, reason="Experiment not found")
        return

    register_websocket(experiment_id, websocket)

    ckpt_result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.experiment_id == experiment_id)
        .order_by(Checkpoint.step.asc())
    )
    checkpoints = ckpt_result.scalars().all()
    metrics = [
        {
            "experiment_id": experiment_id,
            "epoch": c.epoch,
            "step": c.step,
            "train_loss": c.train_loss,
            "eval_loss": c.eval_loss,
        }
        for c in checkpoints
    ]
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
    except Exception:
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
        config_payload = data.config.model_dump()
        provided_config_fields = set(data.config.model_fields_set)
        profile_defaults_applied: list[str] = []

        profile_contract = await get_project_domain_profile_contract(db, project_id)
        profile_training_defaults = get_training_defaults(profile_contract)
        if profile_training_defaults:
            for key, value in profile_training_defaults.items():
                if key == "training_mode":
                    continue
                if key not in provided_config_fields:
                    config_payload[key] = value
                    profile_defaults_applied.append(key)

        resolved_training_mode = data.config.training_mode
        if (
            "training_mode" not in provided_config_fields
            and isinstance(profile_training_defaults.get("training_mode"), str)
        ):
            candidate_mode = str(profile_training_defaults["training_mode"]).strip().lower()
            try:
                resolved_training_mode = type(data.config.training_mode)(candidate_mode)
                config_payload["training_mode"] = resolved_training_mode.value
                profile_defaults_applied.append("training_mode")
            except ValueError:
                pass

        exp = await create_experiment(
            db,
            project_id,
            data.name,
            str(config_payload.get("base_model", data.config.base_model)),
            config_payload,
            data.description,
            resolved_training_mode,
        )
        response_payload = ExperimentResponse.model_validate(exp).model_dump()
        response_payload["domain_profile_applied"] = (
            profile_contract.get("profile_id")
            if isinstance(profile_contract, dict)
            else None
        )
        response_payload["profile_training_defaults"] = (
            dict(profile_training_defaults) if profile_training_defaults else None
        )
        response_payload["resolved_training_config"] = dict(config_payload)
        response_payload["profile_defaults_applied"] = sorted(set(profile_defaults_applied))
        return ExperimentResponse.model_validate(response_payload)
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


@router.post("/experiments/{experiment_id}/cancel")
async def cancel(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running training experiment."""
    try:
        return await cancel_training(db, project_id, experiment_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(409, detail)


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


@router.get("/tasks/{task_id}")
async def task_status(
    project_id: int,
    task_id: str,
):
    """Read Celery task state for training-related jobs."""
    try:
        return get_task_status(task_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/experiments", response_model=list[ExperimentResponse])
async def list_all(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all experiments for a project."""
    exps = await list_experiments(db, project_id)
    return [ExperimentResponse.model_validate(e) for e in exps]

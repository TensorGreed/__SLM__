"""Training pipeline service — SFT, LoRA, checkpoint management."""

import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from fastapi import WebSocket
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory
from app.models.experiment import (
    Checkpoint,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)

active_websockets: dict[int, list[WebSocket]] = {}


def _experiment_dir(project_id: int, experiment_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "experiments" / str(experiment_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


async def create_experiment(
    db: AsyncSession,
    project_id: int,
    name: str,
    base_model: str,
    config: dict,
    description: str = "",
    training_mode: TrainingMode = TrainingMode.SFT,
) -> Experiment:
    """Create a new training experiment."""
    exp = Experiment(
        project_id=project_id,
        name=name,
        description=description,
        base_model=base_model,
        config=config,
        training_mode=training_mode,
        status=ExperimentStatus.PENDING,
    )
    db.add(exp)
    await db.flush()
    await db.refresh(exp)

    # Create output dir
    output_dir = _experiment_dir(project_id, exp.id)
    exp.output_dir = str(output_dir)
    await db.flush()

    # Save config
    config_path = output_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return exp


async def broadcast_metric(experiment_id: int, metric: dict):
    """Broadcast metric to all active websockets for an experiment."""
    if experiment_id in active_websockets:
        dead_socks = []
        for ws in active_websockets[experiment_id]:
            try:
                await ws.send_json(metric)
            except Exception:
                dead_socks.append(ws)
        for ws in dead_socks:
            active_websockets[experiment_id].remove(ws)


async def _simulate_training_loop(experiment_id: int, config: dict):
    """Simulate a training loop reporting metrics for demo purposes."""
    epochs = config.get("num_epochs", 3)
    steps_per_epoch = 100
    total_steps = epochs * steps_per_epoch
    
    current_loss = 3.5
    lr = config.get("learning_rate", 2e-4)
    save_steps = config.get("save_steps", 100)
    
    for step in range(1, total_steps + 1):
        await asyncio.sleep(0.1)  # 100ms per step to make demo fast but visible
        
        # Simulate realistic loss curve
        current_loss = current_loss * 0.995 + (0.01 * 0.5) + random.uniform(-0.05, 0.05)
        
        epoch_float = step / steps_per_epoch
        eval_loss = round(current_loss + 0.15 + random.uniform(-0.02, 0.02), 4) if step % 20 == 0 else None
        
        metric = {
            "experiment_id": experiment_id,
            "epoch": round(epoch_float, 2),
            "step": step,
            "train_loss": round(current_loss, 4),
            "eval_loss": eval_loss,
            "learning_rate": lr,
            "gpu_utilization": round(random.uniform(92.0, 98.0), 1),
            "eta_seconds": int((total_steps - step) * 0.1)
        }
        
        await broadcast_metric(experiment_id, metric)
        
        # Periodic checkpoint
        if step % save_steps == 0 or step == total_steps:
            async with async_session_factory() as db:
                ckpt = Checkpoint(
                    experiment_id=experiment_id,
                    epoch=int(epoch_float) if step % steps_per_epoch == 0 else int(epoch_float) + 1,
                    step=step,
                    train_loss=metric["train_loss"],
                    eval_loss=eval_loss or metric["train_loss"] + 0.15,
                )
                db.add(ckpt)
                await db.commit()

    # Finish experiment
    async with async_session_factory() as db:
        result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
        exp = result.scalar_one_or_none()
        if exp:
            exp.status = ExperimentStatus.COMPLETED
            exp.completed_at = datetime.now(timezone.utc)
            exp.final_train_loss = round(current_loss, 4)
            await db.commit()


async def start_training(
    db: AsyncSession,
    experiment_id: int,
) -> dict:
    """Start training (simulate locally)."""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found")

    exp.status = ExperimentStatus.RUNNING
    exp.started_at = datetime.now(timezone.utc)
    await db.flush()

    # In production: celery_app.send_task('train', args=[experiment_id])
    # Local simulation for demonstration:
    asyncio.create_task(_simulate_training_loop(exp.id, exp.config or {}))

    return {
        "experiment_id": exp.id,
        "status": exp.status.value,
        "message": "Training started. Connecting to telemetry stream...",
        "config": exp.config,
    }


async def get_training_status(
    db: AsyncSession, experiment_id: int
) -> dict:
    """Get current training status and metrics."""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found")

    ckpt_result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.experiment_id == experiment_id)
        .order_by(Checkpoint.step.desc())
    )
    checkpoints = ckpt_result.scalars().all()

    return {
        "experiment_id": exp.id,
        "name": exp.name,
        "status": exp.status.value,
        "training_mode": exp.training_mode.value,
        "base_model": exp.base_model,
        "config": exp.config,
        "final_train_loss": exp.final_train_loss,
        "final_eval_loss": exp.final_eval_loss,
        "total_epochs": exp.total_epochs,
        "total_steps": exp.total_steps,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        "checkpoints": [
            {
                "epoch": c.epoch,
                "step": c.step,
                "train_loss": c.train_loss,
                "eval_loss": c.eval_loss,
                "is_best": c.is_best,
            }
            for c in checkpoints
        ],
    }


async def list_experiments(
    db: AsyncSession, project_id: int
) -> list[Experiment]:
    """List all experiments for a project."""
    result = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc())
    )
    return list(result.scalars().all())

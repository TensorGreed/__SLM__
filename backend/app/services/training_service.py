"""Training pipeline service — SFT, LoRA, checkpoint management."""

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.experiment import (
    Checkpoint,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)


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


async def start_training(
    db: AsyncSession,
    experiment_id: int,
) -> dict:
    """Start training (in a real deployment, this enqueues a Celery task)."""
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
    # For now, return status
    return {
        "experiment_id": exp.id,
        "status": exp.status.value,
        "message": "Training started. Monitor via /training/status endpoint.",
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

    # Load checkpoints
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

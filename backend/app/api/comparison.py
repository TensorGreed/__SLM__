"""Comparison API for experiments."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.experiment import Experiment, Checkpoint

router = APIRouter(prefix="/projects/{project_id}/training", tags=["Comparison"])


@router.get("/compare")
async def compare_experiments(
    project_id: int,
    experiment_ids: str,
    db: AsyncSession = Depends(get_db),
):
    """Compare multiple experiments side-by-side."""
    if not experiment_ids.strip():
        raise HTTPException(status_code=400, detail="experiment_ids parameter is required")
    
    try:
        ids = [int(eid.strip()) for eid in experiment_ids.split(",") if eid.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment IDs format")

    if not ids:
        raise HTTPException(status_code=400, detail="No valid experiment IDs provided")

    if len(ids) > 5:
        raise HTTPException(status_code=400, detail="Cannot compare more than 5 experiments at once")

    # Fetch experiments with their checkpoints
    query = (
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(Experiment.id.in_(ids))
        .options(selectinload(Experiment.checkpoints))
    )
    result = await db.execute(query)
    experiments = result.scalars().all()

    if len(experiments) != len(ids):
        # Find which ones are missing
        found_ids = {e.id for e in experiments}
        missing = set(ids) - found_ids
        if missing:
            raise HTTPException(status_code=404, detail=f"Experiments not found: {missing}")

    # Format the data for the frontend chart and table
    comparison_data = []
    
    for exp in experiments:
        # Sort checkpoints by step to ensure correct time-series
        checkpoints = sorted(exp.checkpoints, key=lambda c: c.step)
        
        history = []
        for ckpt in checkpoints:
            history.append({
                "step": ckpt.step,
                "epoch": ckpt.epoch,
                "train_loss": ckpt.train_loss,
                "eval_loss": ckpt.eval_loss,
                **ckpt.metrics  # Include any other tracked metrics (lr, gpu, etc.)
            })
            
        duration_seconds = None
        if exp.started_at and exp.completed_at:
            duration_seconds = (exp.completed_at - exp.started_at).total_seconds()
            
        comparison_data.append({
            "id": exp.id,
            "name": exp.name,
            "status": exp.status,
            "training_mode": exp.training_mode,
            "base_model": exp.base_model,
            "config": exp.config or {},
            "final_train_loss": exp.final_train_loss,
            "final_eval_loss": exp.final_eval_loss,
            "total_epochs": exp.total_epochs,
            "total_steps": exp.total_steps,
            "duration_seconds": duration_seconds,
            "history": history
        })

    return {"experiments": comparison_data}

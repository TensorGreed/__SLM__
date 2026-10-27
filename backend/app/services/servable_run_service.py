"""Which training runs can be served, and which one a project serves by default.

Shared by the playground ("experiment" provider) and the serving gateway so
the two never disagree about what "a run that can answer" means.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment, ExperimentStatus


def run_problem(exp: Experiment) -> str | None:
    """Why ``exp`` cannot be chatted with, or ``None`` when it can: a
    completed, non-baseline run with saved weights."""
    from app.services.post_training_eval_service import _model_dir_has_weights

    if isinstance(exp.config, dict) and exp.config.get("is_baseline") is True:
        return "That's a baseline row, not a trained run."
    if exp.status != ExperimentStatus.COMPLETED:
        return f"Run #{exp.id} is {exp.status.value}; only completed runs can chat."
    if not _model_dir_has_weights(exp.output_dir):
        return f"Run #{exp.id} has no saved model weights (simulated or incomplete run)."
    return None


def run_model_ref(exp: Experiment) -> str:
    """The checkpoint path ``local_chat_service`` loads for the run."""
    from app.services.evaluation_service import _resolve_model_reference

    return str(_resolve_model_reference(exp, None))


def run_model_dir(exp: Experiment) -> Path | None:
    if not exp.output_dir:
        return None
    root = Path(exp.output_dir)
    return root / "model" if (root / "model").is_dir() else root


def run_adapter_config(exp: Experiment) -> dict[str, Any] | None:
    """``adapter_config.json`` when the run's checkpoint is a LoRA adapter."""
    from app.services.adapter_merge_service import read_adapter_config

    return read_adapter_config(run_model_dir(exp))


async def list_servable_runs(db: AsyncSession, project_id: int) -> list[Experiment]:
    rows = (
        await db.execute(
            select(Experiment)
            .where(Experiment.project_id == project_id, Experiment.status == ExperimentStatus.COMPLETED)
            .order_by(Experiment.id.asc())
        )
    ).scalars().all()
    # Weight checks stat the disk: do them off the event loop (the rows are loaded).
    return await asyncio.to_thread(lambda: [exp for exp in rows if run_problem(exp) is None])

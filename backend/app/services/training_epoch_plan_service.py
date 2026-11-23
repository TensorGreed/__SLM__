"""The epoch / step plan a running trainer reports (phase 7i).

``auto_epochs`` (``training_epoch_policy``) and a step cap can make the trainer run a different number of
epochs than the run's form value (``config.num_epochs``): 108 rows at batch 4 run 8 epochs / 216 steps
while the form said 3, and the Train page showed "8.00 / 3". ``scripts/train.py`` emits
``SLM_EVENT {"event": "epoch_plan", "num_epochs": 8, "total_steps": 216}`` at train begin; the worker
publishes it to the Train page's websocket (``{"type": "epoch_plan", "payload": ...}``) and saves it on the
run (``Experiment.total_epochs`` / ``total_steps``, which the end-of-run report also writes), so a page
opened mid-run reads the same total.
"""

from __future__ import annotations

import math
from typing import Any


def epoch_plan_from_event(event: dict[str, Any] | None) -> dict[str, Any]:
    """The usable part of an ``epoch_plan`` event: positive ``num_epochs`` / ``total_steps`` only."""
    if not isinstance(event, dict):
        return {}
    plan: dict[str, Any] = {}
    epochs = event.get("num_epochs")
    if isinstance(epochs, (int, float)) and not isinstance(epochs, bool) and math.isfinite(epochs) and epochs > 0:
        plan["num_epochs"] = int(epochs) if float(epochs).is_integer() else round(float(epochs), 4)
    steps = event.get("total_steps")
    if isinstance(steps, (int, float)) and not isinstance(steps, bool) and math.isfinite(steps) and steps > 0:
        plan["total_steps"] = int(steps)
    return plan


def epoch_plan_log_line(plan: dict[str, Any]) -> str:
    """``[event] training plan: 8 epochs, 216 optimizer steps`` for the Runner output."""
    parts = []
    if plan.get("num_epochs") is not None:
        epochs = plan["num_epochs"]
        parts.append(f"{epochs} epoch{'' if epochs == 1 else 's'}")
    if plan.get("total_steps") is not None:
        parts.append(f"{plan['total_steps']} optimizer steps")
    return f"[event] training plan: {', '.join(parts)}" if parts else ""


async def record_epoch_plan(experiment_id: int, plan: dict[str, Any]) -> None:
    """Save the plan on the run (``total_epochs`` rounded up, ``total_steps``). Best effort: never disturbs the run."""
    if not plan:
        return
    try:
        from sqlalchemy import select

        from app.database import async_session_factory
        from app.models.experiment import Experiment

        async with async_session_factory() as db:
            exp = (await db.execute(select(Experiment).where(Experiment.id == int(experiment_id)))).scalar_one_or_none()
            if exp is None:
                return
            if plan.get("num_epochs") is not None:
                exp.total_epochs = int(math.ceil(float(plan["num_epochs"])))
            if plan.get("total_steps") is not None:
                exp.total_steps = int(plan["total_steps"])
            await db.commit()
    except Exception as exc:  # noqa: BLE001 - a display hint never disturbs the run
        print(f"[epoch_plan] could not record the plan for experiment {experiment_id}: {exc}", flush=True)

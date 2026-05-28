"""Hyperparameter grid bake-off sweep (Track 1, Epic C).

Extends the model bake-off from "which base model" to "which *config*": expand a
grid over LoRA rank + learning rate (optionally base model too), materialize one
**real** ``Experiment`` per cell — so all existing preflight / eval / gate
machinery applies — group them under a shared ``sweep_id``, dispatch each via the
normal training path, and compare them on a quality-vs-cost Pareto frontier.

Cells are the background units of work (each is a normal Experiment dispatched to
the configured runtime), so the start call returns as soon as every cell is
dispatched; the Pareto endpoint aggregates results as cells complete.

Quality (higher = better) is read from the experiment's eval pass-rate when
available, else derived from eval/train loss. Cost (lower = better) is the LoRA
rank — a direct proxy for adapter footprint — so the frontier surfaces the best
learning rate for each adapter size and the quality-vs-size trade-off across ranks.
"""

from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.experiment import EvalResult, Experiment
from app.services.model_benchmark_service import annotate_pareto_frontier

MAX_SWEEP_CELLS = 16

QUALITY_KEY = "quality_score"
COST_KEY = "lora_r"


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _dedupe_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def expand_grid(
    base_config: dict[str, Any],
    *,
    lora_r_values: list[int],
    learning_rate_values: list[float],
    base_model_values: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cross-product the axes into per-cell configs (capped at MAX_SWEEP_CELLS).

    Each returned cell is a deep copy of ``base_config`` with the axis values
    applied (and ``use_lora=True`` forced, since LoRA rank only matters then),
    plus a ``_label`` and the resolved axis values for grouping.
    """
    ranks = _dedupe_preserve_order([int(r) for r in (lora_r_values or []) if int(r) >= 1])
    lrs = _dedupe_preserve_order([float(x) for x in (learning_rate_values or []) if float(x) > 0])
    models = _dedupe_preserve_order([str(m).strip() for m in (base_model_values or []) if str(m).strip()])
    if not ranks:
        raise ValueError("lora_r_values must contain at least one rank >= 1.")
    if not lrs:
        raise ValueError("learning_rate_values must contain at least one rate > 0.")

    cells: list[dict[str, Any]] = []
    model_axis: list[str | None] = models if models else [None]
    for model in model_axis:
        for rank in ranks:
            for lr in lrs:
                cfg = copy.deepcopy(dict(base_config or {}))
                cfg["use_lora"] = True
                cfg["lora_r"] = rank
                cfg["learning_rate"] = lr
                if model:
                    cfg["base_model"] = model
                label_bits = [f"r{rank}", f"lr{lr:g}"]
                if model:
                    label_bits.insert(0, model.split("/")[-1])
                cfg["_label"] = "-".join(label_bits)
                cfg["_axis_values"] = {
                    "lora_r": rank,
                    "learning_rate": lr,
                    **({"base_model": model} if model else {}),
                }
                cells.append(cfg)
                if len(cells) >= MAX_SWEEP_CELLS:
                    return cells
    return cells


async def start_hyperparameter_sweep(
    db: AsyncSession,
    project_id: int,
    *,
    base_model: str,
    base_config: dict[str, Any] | None = None,
    lora_r_values: list[int],
    learning_rate_values: list[float],
    base_model_values: list[str] | None = None,
) -> dict[str, Any]:
    """Materialize + dispatch one Experiment per grid cell under a shared sweep id."""
    # Local import avoids a circular import (training_service imports this module
    # is not the case today, but keep the dependency one-directional + lazy).
    from app.services.training_service import create_experiment, start_training

    cells = expand_grid(
        dict(base_config or {}),
        lora_r_values=lora_r_values,
        learning_rate_values=learning_rate_values,
        base_model_values=base_model_values,
    )
    sweep_id = uuid4().hex[:12]
    created: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        label = str(cell.pop("_label"))
        axis_values = dict(cell.pop("_axis_values"))
        cell_base_model = str(cell.get("base_model") or base_model).strip() or base_model
        cell["_sweep"] = {
            "sweep_id": sweep_id,
            "label": label,
            "cell_index": index,
            "axis_values": axis_values,
        }
        record: dict[str, Any] = {
            "label": label,
            "cell_index": index,
            "axis_values": axis_values,
            "base_model": cell_base_model,
        }
        try:
            exp = await create_experiment(
                db,
                project_id,
                name=f"sweep-{sweep_id}-{label}",
                base_model=cell_base_model,
                config=cell,
                description=f"Hyperparameter sweep {sweep_id} · cell {label}",
            )
            await db.flush()
            await start_training(db, project_id, int(exp.id))
            record["experiment_id"] = int(exp.id)
            record["dispatched"] = True
        except Exception as exc:  # one bad cell must not abort the whole sweep
            record["dispatched"] = False
            record["error"] = str(exc)
        created.append(record)

    dispatched = [c for c in created if c.get("dispatched")]
    if not dispatched:
        errors = "; ".join(str(c.get("error") or "unknown") for c in created[:3])
        raise ValueError(f"Hyperparameter sweep dispatched 0/{len(created)} cells: {errors}")

    return {
        "sweep_id": sweep_id,
        "project_id": project_id,
        "requested_cells": len(cells),
        "dispatched_cells": len(dispatched),
        "cells": created,
        "axes": {
            "lora_r": _dedupe_preserve_order([int(r) for r in lora_r_values]),
            "learning_rate": _dedupe_preserve_order([float(x) for x in learning_rate_values]),
            "base_model": _dedupe_preserve_order([str(m) for m in (base_model_values or [])]),
        },
    }


def _latest_eval(experiment: Experiment) -> EvalResult | None:
    rows = list(getattr(experiment, "eval_results", []) or [])
    if not rows:
        return None
    with_pass = [r for r in rows if r.pass_rate is not None]
    pool = with_pass or rows
    return sorted(pool, key=lambda r: int(getattr(r, "id", 0) or 0))[-1]


def _cell_quality(experiment: Experiment) -> tuple[float | None, str]:
    """Return (quality_score higher=better, source). None until a signal exists."""
    ev = _latest_eval(experiment)
    if ev is not None and ev.pass_rate is not None:
        return float(ev.pass_rate), "eval_pass_rate"
    eval_loss = _coerce_float(experiment.final_eval_loss)
    if eval_loss is not None:
        return 1.0 / (1.0 + max(eval_loss, 0.0)), "eval_loss"
    train_loss = _coerce_float(experiment.final_train_loss)
    if train_loss is not None:
        return 1.0 / (1.0 + max(train_loss, 0.0)), "train_loss"
    return None, "pending"


async def get_sweep_pareto(
    db: AsyncSession, project_id: int, sweep_id: str
) -> dict[str, Any]:
    """Aggregate a sweep's cells into a quality-vs-cost Pareto matrix.

    Completed cells with a quality signal are annotated on the frontier
    (quality ↑ vs LoRA rank ↓). Cells still training have ``quality_score=None``
    and are excluded from the frontier annotation but listed with their status.
    """
    token = str(sweep_id or "").strip()
    if not token:
        raise ValueError("sweep_id is required.")

    result = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .options(selectinload(Experiment.eval_results))
    )
    experiments = [
        exp
        for exp in result.scalars().all()
        if str(((exp.config or {}).get("_sweep") or {}).get("sweep_id") or "") == token
    ]
    if not experiments:
        raise ValueError(f"No sweep found for sweep_id '{sweep_id}'.")

    rows: list[dict[str, Any]] = []
    for exp in sorted(experiments, key=lambda e: int(((e.config or {}).get("_sweep") or {}).get("cell_index") or 0)):
        sweep_meta = dict((exp.config or {}).get("_sweep") or {})
        axis_values = dict(sweep_meta.get("axis_values") or {})
        quality, quality_source = _cell_quality(exp)
        status = exp.status.value if hasattr(exp.status, "value") else str(exp.status)
        rows.append(
            {
                "model_id": str(sweep_meta.get("label") or exp.name),  # `model_id` so the Pareto helper/UI can key on it
                "experiment_id": int(exp.id),
                "label": str(sweep_meta.get("label") or exp.name),
                "lora_r": int(axis_values.get("lora_r") or (exp.config or {}).get("lora_r") or 0),
                "learning_rate": _coerce_float(axis_values.get("learning_rate") or (exp.config or {}).get("learning_rate")),
                "base_model": exp.base_model,
                "status": status,
                "final_train_loss": _coerce_float(exp.final_train_loss),
                "final_eval_loss": _coerce_float(exp.final_eval_loss),
                "quality_score": quality,
                "quality_source": quality_source,
            }
        )

    scored = [r for r in rows if r["quality_score"] is not None]
    annotate_pareto_frontier(scored, quality_key=QUALITY_KEY, cost_key=COST_KEY)
    for row in rows:
        if row["quality_score"] is None:
            row["pareto_optimal"] = False
            row["dominated_by"] = []

    optimal_labels = [r["label"] for r in scored if r.get("pareto_optimal")]
    best = None
    if scored:
        best = max(scored, key=lambda r: float(r["quality_score"]))

    return {
        "sweep_id": token,
        "project_id": project_id,
        "cell_count": len(rows),
        "completed_count": len(scored),
        "cells": rows,
        "pareto": {
            "quality_key": QUALITY_KEY,
            "cost_key": COST_KEY,
            "optimal_labels": optimal_labels,
        },
        "best_label": best["label"] if best else None,
        "best_experiment_id": best["experiment_id"] if best else None,
    }


async def list_project_sweeps(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """List distinct sweep ids for a project with cell counts (newest first)."""
    result = await db.execute(
        select(Experiment).where(Experiment.project_id == project_id)
    )
    sweeps: dict[str, dict[str, Any]] = {}
    for exp in result.scalars().all():
        meta = (exp.config or {}).get("_sweep") or {}
        sid = str(meta.get("sweep_id") or "")
        if not sid:
            continue
        entry = sweeps.setdefault(sid, {"sweep_id": sid, "cell_count": 0, "latest_experiment_id": 0})
        entry["cell_count"] += 1
        entry["latest_experiment_id"] = max(entry["latest_experiment_id"], int(exp.id))
    ordered = sorted(sweeps.values(), key=lambda e: e["latest_experiment_id"], reverse=True)
    return {"project_id": project_id, "sweep_count": len(ordered), "sweeps": ordered}

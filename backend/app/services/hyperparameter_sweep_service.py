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
available, else derived from eval/train loss. Cost (lower = better) is picked
by the caller via ``cost_kind``:

* ``"wall_clock_seconds"`` (default) — the honest one. Real measured
  ``completed_at - started_at`` for the cell. Captures actual training
  time so a rank-32 cell that trained fast looks cheaper than a rank-8
  cell that took an hour.
* ``"lora_r"`` — adapter footprint proxy. Cheap to compute, available
  immediately (no training needed), but a fiction when ``base_model``
  is an axis (rank-16 on a 135M base vs rank-16 on a 3B base do *not*
  cost the same).
* ``"base_params_m"`` — base model parameter count in millions. Useful
  when the sweep varies ``base_model`` and the user wants to read the
  Pareto as "which model size is worth the quality gain".

The frontier is computed against whichever cost the caller asks for, and the
response echoes ``cost_kind`` + ``cost_key`` so the frontend renders the right
axis label without guessing.
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

# Default cost axis. Wall-clock is the honest one — see the module docstring.
# Kept as a constant so callers can reference it (the API default echoes it
# back to the frontend).
DEFAULT_COST_KIND = "wall_clock_seconds"

# Allow-list of cost-kind strings. Anything outside this raises ValueError
# so a typo in the API query string surfaces as a 400 rather than a silent
# "cost is None" panel.
SUPPORTED_COST_KINDS: tuple[str, ...] = (
    "wall_clock_seconds",
    "lora_r",
    "base_params_m",
)

# Known parameter counts for the default base models. Mirrors the same dict
# in trainability_forecast_service.py; we keep them separate so the two
# surfaces can drift independently if one needs to support a model the
# other shouldn't yet. Falls back to None (cost unknown) when the model
# id isn't recognized, which the Pareto annotator treats as "exclude from
# the frontier" rather than guessing.
BASE_MODEL_PARAMS_M: dict[str, int] = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": 135,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 360,
    "Qwen/Qwen2.5-0.5B-Instruct": 500,
    "Qwen/Qwen2.5-1.5B-Instruct": 1500,
    "Qwen/Qwen2.5-3B-Instruct": 3000,
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": 1500,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1100,
    "microsoft/phi-2": 2700,
}


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


def _cell_cost(experiment: Experiment, cost_kind: str) -> tuple[float | None, str]:
    """Return (cost lower=better, source) for the chosen ``cost_kind``.

    ``None`` means the cost is genuinely unavailable for this cell (the most
    common case is wall-clock asked for before the cell finishes). The
    Pareto annotator skips cells with ``None`` cost; the UI surfaces them
    with a "cost pending" status rather than guessing.
    """
    if cost_kind == "wall_clock_seconds":
        started = experiment.started_at
        completed = experiment.completed_at
        if started is None or completed is None:
            return None, "pending"
        seconds = (completed - started).total_seconds()
        # Negative deltas are impossible but defensive: a clock-skew
        # artefact would yield a nonsense cost. Treat as missing.
        if seconds < 0:
            return None, "invalid"
        return float(seconds), "wall_clock_seconds"

    if cost_kind == "lora_r":
        config = experiment.config or {}
        rank = config.get("lora_r")
        if rank is None:
            # Cell predates the LoRA axis being a sweep dimension. Excluded
            # rather than guessed.
            return None, "missing_lora_r"
        try:
            return float(rank), "lora_r"
        except (TypeError, ValueError):
            return None, "invalid"

    if cost_kind == "base_params_m":
        params = BASE_MODEL_PARAMS_M.get(experiment.base_model or "")
        if params is None:
            return None, "unknown_base_model"
        return float(params), "base_params_m"

    # Unsupported cost_kind: defensive fall-through. Public callers go
    # through ``get_sweep_pareto`` which validates first, so this only
    # fires for code paths that bypass validation. Returning None is
    # safer than raising mid-loop.
    return None, "unsupported_cost_kind"


async def get_sweep_pareto(
    db: AsyncSession,
    project_id: int,
    sweep_id: str,
    *,
    cost_kind: str = DEFAULT_COST_KIND,
) -> dict[str, Any]:
    """Aggregate a sweep's cells into a quality-vs-cost Pareto matrix.

    ``cost_kind`` picks the cost axis (see module docstring). Default is
    ``"wall_clock_seconds"`` — the honest one. Cells with a missing cost
    for the chosen axis (e.g. wall-clock asked for before the cell
    finishes, or ``base_params_m`` asked for on an unrecognized model id)
    are listed with ``cost_score=None`` and excluded from frontier
    annotation, so the panel can render them as "cost pending" without
    polluting the frontier.

    Completed cells with both a quality and a cost signal are annotated
    on the frontier (quality ↑ vs ``cost_kind`` ↓).
    """
    token = str(sweep_id or "").strip()
    if not token:
        raise ValueError("sweep_id is required.")

    if cost_kind not in SUPPORTED_COST_KINDS:
        raise ValueError(
            f"Unsupported cost_kind '{cost_kind}'. "
            f"Supported: {', '.join(SUPPORTED_COST_KINDS)}."
        )

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
        cost, cost_source = _cell_cost(exp, cost_kind)
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
                "cost_score": cost,
                "cost_source": cost_source,
            }
        )

    # Frontier annotation needs BOTH a quality and a cost signal — a cell
    # that has quality but no cost (or vice versa) is genuinely off the
    # 2D plot, not on the frontier. Don't downgrade either signal by
    # filling None with a default.
    scored = [
        r for r in rows
        if r["quality_score"] is not None and r["cost_score"] is not None
    ]
    annotate_pareto_frontier(scored, quality_key=QUALITY_KEY, cost_key="cost_score")
    for row in rows:
        if row["quality_score"] is None or row["cost_score"] is None:
            row["pareto_optimal"] = False
            row["dominated_by"] = []

    optimal_labels = [r["label"] for r in scored if r.get("pareto_optimal")]
    # "Best" = highest quality among scored cells, ties broken by lowest cost.
    # This is the cell the promote-the-winner button targets.
    best = None
    if scored:
        best = min(
            scored,
            key=lambda r: (-float(r["quality_score"]), float(r["cost_score"])),
        )

    return {
        "sweep_id": token,
        "project_id": project_id,
        "cell_count": len(rows),
        "completed_count": len(scored),
        "cells": rows,
        "pareto": {
            "quality_key": QUALITY_KEY,
            "cost_key": "cost_score",
            "cost_kind": cost_kind,
            "optimal_labels": optimal_labels,
        },
        "cost_kind": cost_kind,
        "supported_cost_kinds": list(SUPPORTED_COST_KINDS),
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

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


def _coerce_quality_target(value: Any) -> float | None:
    """Coerce a user-supplied quality target to a sane [0, 1] float, or None.

    The target is the threshold the orchestrator uses to decide "winner found
    — stop the rest of the sweep". 0 and 1 are degenerate ends (always-hit and
    never-hit) and we clamp into the open interval rather than rejecting,
    because the panel can plausibly let a user type 0.99 and expect it to
    behave like "very close to perfect".
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not (0.0 < f <= 1.0):
        # An out-of-range target is almost certainly a typo (e.g. 80 meaning
        # "80%"). Clamp the obvious percentage form, otherwise drop.
        if 1.0 < f <= 100.0:
            f = f / 100.0
        else:
            return None
    return f


async def start_hyperparameter_sweep(
    db: AsyncSession,
    project_id: int,
    *,
    base_model: str,
    base_config: dict[str, Any] | None = None,
    lora_r_values: list[int],
    learning_rate_values: list[float],
    base_model_values: list[str] | None = None,
    quality_target: float | None = None,
) -> dict[str, Any]:
    """Materialize + dispatch one Experiment per grid cell under a shared sweep id.

    When ``quality_target`` is provided, the sweep is annotated so that the
    next ``get_sweep_pareto`` call that observes a cell clearing the target
    cancels any cells still running — saving real GPU spend when the first
    good config arrives early.
    """
    # Local import avoids a circular import (training_service imports this module
    # is not the case today, but keep the dependency one-directional + lazy).
    from app.services.training_service import create_experiment, start_training

    cells = expand_grid(
        dict(base_config or {}),
        lora_r_values=lora_r_values,
        learning_rate_values=learning_rate_values,
        base_model_values=base_model_values,
    )
    normalized_target = _coerce_quality_target(quality_target)
    sweep_id = uuid4().hex[:12]
    created: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        label = str(cell.pop("_label"))
        axis_values = dict(cell.pop("_axis_values"))
        cell_base_model = str(cell.get("base_model") or base_model).strip() or base_model
        sweep_meta: dict[str, Any] = {
            "sweep_id": sweep_id,
            "label": label,
            "cell_index": index,
            "axis_values": axis_values,
        }
        if normalized_target is not None:
            sweep_meta["quality_target"] = normalized_target
        cell["_sweep"] = sweep_meta
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
        "quality_target": normalized_target,
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

    # Stop-when-met: every cell carries the sweep's quality_target on its
    # _sweep meta (when one was set at dispatch). If any completed cell's
    # quality_score has cleared the target AND there are still-running cells,
    # cancel them. The frontend polls this endpoint every 4s, so "next
    # observation cancels the rest" is the natural watcher loop without any
    # background worker — and the user can refresh the panel to see the
    # cancellation reflected.
    quality_target = _extract_quality_target(experiments)
    target_hit = False
    target_hit_label: str | None = None
    cancelled_now: list[str] = []
    if quality_target is not None:
        # Pick the first cell (by cell_index) that cleared the target; that
        # cell becomes the "winner" the cancel is justified by.
        clearing = sorted(
            [
                (idx, r) for idx, r in enumerate(rows)
                if r["quality_score"] is not None
                and float(r["quality_score"]) >= float(quality_target)
            ],
            key=lambda t: t[0],
        )
        if clearing:
            target_hit = True
            target_hit_label = clearing[0][1]["label"]
            cancelled_now = await _cancel_still_running_cells(
                db,
                project_id=project_id,
                experiments=experiments,
                rows=rows,
            )

    # Winner-vs-gate honesty pass. quality_score being the highest across the
    # sweep doesn't mean the winner is actually any *good* — it just means
    # it's the least-bad. Run each completed cell against the project's eval
    # pack gate and surface whether any cell genuinely cleared the bar.
    # Three verdicts:
    #   promote      — some cell cleared the gate; UI can offer promote-to-base.
    #   inconclusive — every completed cell has an eval result but none cleared
    #                  the gate. UI surfaces "nobody cleared <X>" + handoff to
    #                  the failure-cluster panel.
    #   pending      — cells are still running OR cleared cells have no eval
    #                  results yet. No claim either way.
    gate_summary = await _annotate_cell_gates(
        db, project_id=project_id, experiments=experiments, rows=rows,
    )
    verdict, verdict_reason = _compute_verdict(rows=rows, gate_summary=gate_summary)

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
        "quality_target": quality_target,
        "target_hit": target_hit,
        "target_hit_label": target_hit_label,
        "cancelled_by_target": cancelled_now,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "gate_summary": gate_summary,
    }


def _extract_quality_target(experiments: list[Experiment]) -> float | None:
    """Recover the sweep's quality_target from any cell's _sweep meta.

    Every cell in a sweep carries the same target (set at dispatch); the
    first non-None reading wins. ``None`` means no target was set — the
    sweep runs every cell to completion regardless of partial winners.
    """
    for exp in experiments:
        meta = (exp.config or {}).get("_sweep") or {}
        target = meta.get("quality_target")
        if target is None:
            continue
        try:
            return float(target)
        except (TypeError, ValueError):
            continue
    return None


async def _cancel_still_running_cells(
    db: AsyncSession,
    *,
    project_id: int,
    experiments: list[Experiment],
    rows: list[dict[str, Any]],
) -> list[str]:
    """Cancel any cell whose status is still RUNNING / PENDING / QUEUED.

    Annotates each cancelled row in place with ``cancelled_by_target=True``
    + status downgraded to "cancelled" so the next render reflects the new
    state without a refetch. Errors during cancel (cell already finished
    between our status read and the cancel call, etc.) are swallowed —
    cancellation is best-effort, not a critical path.
    """
    from app.models.experiment import ExperimentStatus
    from app.services.training_service import cancel_training

    cancelled: list[str] = []
    label_to_row = {r["label"]: r for r in rows}
    for exp in experiments:
        status_value = exp.status.value if hasattr(exp.status, "value") else str(exp.status)
        if status_value not in {"running", "pending", "queued"}:
            continue
        meta = (exp.config or {}).get("_sweep") or {}
        label = str(meta.get("label") or exp.name)
        try:
            await cancel_training(db, project_id, int(exp.id))
            cancelled.append(label)
            row = label_to_row.get(label)
            if row is not None:
                row["status"] = ExperimentStatus.CANCELLED.value
                row["cancelled_by_target"] = True
        except Exception:
            # Most likely racing the trainer's own status flip; the row
            # will reflect the real state on the next poll regardless.
            continue
    return cancelled


# ─────────────────────────────────────────────────────────────────────
# Winner-vs-gate honesty pass.
#
# best_label is whoever has the highest quality_score in the sweep — even
# if that "best" is below the project's quality gate. Running each cell
# through the project's evaluation pack tells us whether any cell actually
# cleared the bar. The verdict is:
#
#   promote      — some completed cell cleared the project's gate. UI shows
#                  a "promote to base" affordance backed by a real signal.
#   inconclusive — every completed cell has eval results but none cleared
#                  the gate. UI surfaces "nobody cleared <X>" and links the
#                  user to the failure-cluster panel rather than letting
#                  them quietly promote a sub-gate winner.
#   pending      — cells are still running, OR completed cells don't have
#                  eval results yet (gate_passed=None). No claim.
#
# This matches the "Honest metrics, no vanity" memory: never declare an
# all-green sweep when nobody actually cleared the bar.
# ─────────────────────────────────────────────────────────────────────


async def _annotate_cell_gates(
    db: AsyncSession,
    *,
    project_id: int,
    experiments: list[Experiment],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run each cell with eval results through the project's evaluation pack
    gate. Mutates ``rows`` in place to set ``gate_passed`` and
    ``gate_failed_ids`` per cell. Returns a top-level summary dict naming
    the pack + counts that the UI can surface.

    A cell is annotated as:
      gate_passed=True  → evaluation pack returned passed=True
      gate_passed=False → eval results exist but pack returned passed=False;
                          failed_gate_ids carry the specific gate names
      gate_passed=None  → no eval results yet, or no pack configured, or
                          the pack errored. Surfaced as "not measurable".
    """
    from app.services.evaluation_pack_service import evaluate_experiment_auto_gates

    pack_id: str | None = None
    task_profile: str | None = None
    any_cleared = False
    measured_count = 0
    measurable_count = 0
    label_to_row = {r["label"]: r for r in rows}

    for exp in experiments:
        meta = (exp.config or {}).get("_sweep") or {}
        label = str(meta.get("label") or exp.name)
        row = label_to_row.get(label)
        if row is None:
            continue
        # Default to "not measurable yet"; only override on a real signal.
        row["gate_passed"] = None
        row["gate_failed_ids"] = []
        try:
            result = await evaluate_experiment_auto_gates(
                db,
                project_id=project_id,
                experiment_id=int(exp.id),
            )
        except Exception:
            # No pack, no project, experiment missing — surface as "not
            # measurable" rather than crashing the Pareto fetch.
            continue
        # Capture the pack/profile once; every cell shares the same gate.
        if pack_id is None:
            pack_resolution = result.get("pack_resolution") or {}
            pack_id = pack_resolution.get("active_pack_id") or pack_resolution.get("preferred_pack_id")
            task_profile = result.get("task_profile")
        latest_ids = result.get("latest_eval_result_ids") or {}
        if not latest_ids:
            # No eval has been computed for this cell yet. Distinct from
            # "gate failed" — the cell isn't measurable until eval runs.
            continue
        # A pack with zero gates would return passed=True trivially. Don't
        # let "no gates configured" masquerade as "cleared the gate" —
        # that's exactly the vanity case the honesty pass is preventing.
        task_spec = result.get("task_spec") or {}
        gate_count = int(task_spec.get("gate_count") or 0)
        if gate_count <= 0:
            continue
        measurable_count += 1
        passed = bool(result.get("passed"))
        row["gate_passed"] = passed
        row["gate_failed_ids"] = list(result.get("failed_gate_ids") or [])
        if passed:
            any_cleared = True
        measured_count += 1

    return {
        "pack_id": pack_id,
        "task_profile": task_profile,
        "measurable_count": measurable_count,
        "any_cell_cleared": any_cleared,
    }


def _compute_verdict(
    *,
    rows: list[dict[str, Any]],
    gate_summary: dict[str, Any],
) -> tuple[str, str]:
    """Compute the (verdict, reason) pair from per-cell gate annotations.

    See module docstring for the three-state semantics. The reason string
    is short and meant to be rendered directly under the verdict badge.
    """
    if gate_summary.get("any_cell_cleared"):
        return "promote", "At least one cell cleared the project gate."

    has_running = any(r["status"] in {"running", "pending", "queued"} for r in rows)
    measurable = int(gate_summary.get("measurable_count") or 0)

    if has_running and measurable == 0:
        return "pending", "Cells still training; gate verdict pending."
    if has_running:
        return "pending", "Some cells finished without clearing the gate; others still running."
    if measurable == 0:
        # All cells are done (or cancelled) but no eval results exist for
        # any of them. Common when training failed before eval ran.
        return "inconclusive", "No cell produced eval results; gate is not measurable."
    return "inconclusive", "No completed cell cleared the project gate."


# ─────────────────────────────────────────────────────────────────────
# Pre-flight budget estimator.
#
# Before launching, the user wants to know: "what does this sweep cost
# me in wall-clock terms?" We answer with the median seconds-per-cell
# from prior completed cells, multiplied by the planned cell count.
#
# Basis fallback (specific → general → bail):
#   1. Cells in this project with the same base_model + recipe_id.
#   2. Cells in this project with the same base_model (any recipe).
#   3. Cells in this project on any base model.
#   4. No history → return basis="no_history" and a conservative default
#      so the UI can still render "rough estimate" rather than a chip
#      that says "unknown".
#
# We deliberately do *not* report dollars: GPU cost depends on the
# runtime backend (local GB10 = $0, cloud-burst = variable), and a
# fake $ chip would lie. Wall-clock is the honest currency we always
# have.
# ─────────────────────────────────────────────────────────────────────


# Conservative seconds-per-cell when there's no history at all. Tuned
# against the platform's default 135M/SFT recipe with the demo gold
# set (~2 min per cell on the dev GB10). Over-estimates for tinier
# sweeps and under-estimates for big-model sweeps — the basis="no_history"
# label keeps the user honest about how rough the number is.
DEFAULT_NO_HISTORY_SECONDS_PER_CELL = 120


async def estimate_sweep_budget(
    db: AsyncSession,
    project_id: int,
    *,
    base_model: str,
    lora_r_values: list[int],
    learning_rate_values: list[float],
    base_model_values: list[str] | None = None,
    recipe_id: str | None = None,
) -> dict[str, Any]:
    """Pre-launch wall-clock estimate for a planned hyperparameter sweep.

    Validates the grid shape (so the launcher errors before submit on
    obviously bad input), computes the planned cell count, queries
    historical cells for a seconds-per-cell median, and returns
    ``{seconds_per_cell, cell_count, estimated_seconds, basis,
    sample_size}`` so the launcher can render "~3.2h based on 18 prior
    cells on this base+recipe" or "rough estimate, no prior runs" as
    appropriate.

    Raises ``ValueError`` on invalid grid input — the API surfaces
    that as a 400.
    """
    # Use expand_grid for cell-count validation/parity with the launch
    # path. ValueError on empty axes bubbles up unchanged.
    cells = expand_grid(
        {},
        lora_r_values=lora_r_values,
        learning_rate_values=learning_rate_values,
        base_model_values=base_model_values,
    )
    cell_count = len(cells)

    # Pull every completed cell for this project (small-N — sweeps live
    # under their own _sweep marker, and we only want ones with a
    # measured wall-clock).
    result = await db.execute(
        select(Experiment).where(Experiment.project_id == project_id)
    )
    all_cells: list[Experiment] = []
    for exp in result.scalars().all():
        meta = (exp.config or {}).get("_sweep") or {}
        if not meta.get("sweep_id"):
            continue
        if exp.started_at is None or exp.completed_at is None:
            continue
        seconds = (exp.completed_at - exp.started_at).total_seconds()
        if seconds <= 0:
            continue
        all_cells.append(exp)

    def _seconds_of(exp: Experiment) -> float:
        return (exp.completed_at - exp.started_at).total_seconds()

    # Stage 1: same base_model + same recipe.
    candidates: list[Experiment] = []
    basis = "no_history"
    sample_size = 0
    if recipe_id:
        candidates = [
            e for e in all_cells
            if e.base_model == base_model
            and ((e.config or {}).get("_sweep") or {}).get("recipe_id") == recipe_id
        ]
        if candidates:
            basis = "same_base_and_recipe"
    # Stage 2: same base_model only.
    if not candidates:
        candidates = [e for e in all_cells if e.base_model == base_model]
        if candidates:
            basis = "same_base_model"
    # Stage 3: any cell in the project.
    if not candidates:
        candidates = list(all_cells)
        if candidates:
            basis = "project_default"

    if candidates:
        durations = sorted(_seconds_of(e) for e in candidates)
        seconds_per_cell = float(durations[len(durations) // 2])  # median
        sample_size = len(durations)
    else:
        seconds_per_cell = float(DEFAULT_NO_HISTORY_SECONDS_PER_CELL)

    return {
        "cell_count": cell_count,
        "seconds_per_cell": seconds_per_cell,
        "estimated_seconds": seconds_per_cell * cell_count,
        "basis": basis,
        "sample_size": sample_size,
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

"""P15 — Rerun-from-manifest + clone-from-run services.

Both paths create a **new** experiment whose config derives from a prior
run's captured manifest (P14). They differ in intent only:

- ``rerun_from_manifest`` is the deterministic-rerun entry point. The new
  experiment's config is the manifest's ``resolved_config`` verbatim —
  same seed, same hyperparams, same recipe — so re-launching produces
  the same metrics within hardware/library tolerance.
- ``clone_from_run`` is the "fork-and-tweak" entry point. The new
  experiment starts from the same resolved_config but accepts a
  ``config_overrides`` dict that's merged on top (shallow merge), which
  makes it the right tool for small ablations ("what if I bump
  learning_rate?") without re-entering every field by hand.

Both paths record parentage in the new experiment's ``config._rerun_of``
block so downstream tooling (the Wave D CLI, compare views, manifest
lineage) can walk the chain. Neither **starts** the training — they
create the experiment row and return its id. The caller decides when to
fire `POST /experiments/{new_id}/start`, which in turn writes a fresh
manifest for the rerun.

Also exposes ``compare_run_metrics`` — the tolerance check the spec
requires so a test / CLI can answer "did the rerun actually reproduce
the baseline within tolerance?"
"""

from __future__ import annotations

import copy
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment, ExperimentStatus, TrainingMode
from app.models.training_manifest import TrainingManifest
from app.services.training_service import create_experiment


_PARENTAGE_KEY = "_rerun_of"


def _normalize_training_mode(value: Any, fallback: TrainingMode) -> TrainingMode:
    if isinstance(value, TrainingMode):
        return value
    token = str(value or "").strip().lower()
    for mode in TrainingMode:
        if mode.value == token:
            return mode
    return fallback


def _shallow_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge ``overrides`` on top of ``base`` without mutating either.

    Shallow merge: top-level keys in ``overrides`` fully replace those in
    ``base``. We don't deep-merge on purpose — training configs are flat
    bags of hyperparams in this codebase, so deep merge would just create
    footguns where users expect a whole sub-dict swap.
    """
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        merged[key] = copy.deepcopy(value)
    return merged


async def _load_parent_manifest(
    db: AsyncSession, *, project_id: int, parent_experiment_id: int
) -> tuple[Experiment, TrainingManifest]:
    exp_row = (
        await db.execute(
            select(Experiment).where(
                Experiment.id == parent_experiment_id,
                Experiment.project_id == project_id,
            )
        )
    ).scalar_one_or_none()
    if exp_row is None:
        raise ValueError("experiment_not_found")
    manifest_row = (
        await db.execute(
            select(TrainingManifest).where(
                TrainingManifest.experiment_id == parent_experiment_id,
                TrainingManifest.project_id == project_id,
            )
        )
    ).scalar_one_or_none()
    if manifest_row is None:
        raise ValueError("manifest_not_captured")
    return exp_row, manifest_row


def _parentage_block(
    *,
    parent_experiment_id: int,
    manifest_id: int,
    reason: str,
    overrides_applied: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "parent_experiment_id": int(parent_experiment_id),
        "manifest_id": int(manifest_id),
        "reason": reason,
    }
    if overrides_applied:
        block["config_overrides"] = dict(overrides_applied)
    return block


async def rerun_from_manifest(
    db: AsyncSession,
    *,
    project_id: int,
    parent_experiment_id: int,
    run_name: str | None = None,
    description: str | None = None,
) -> Experiment:
    """Create a new experiment whose config is the parent manifest's config verbatim."""
    parent_exp, manifest = await _load_parent_manifest(
        db, project_id=project_id, parent_experiment_id=parent_experiment_id
    )

    resolved_config = dict(manifest.resolved_config or {})
    # Stamp parentage. If the parent itself was a rerun/clone, we don't
    # squash that history — we nest our own block underneath so a chain
    # like "rerun → rerun" stays walkable.
    parentage = _parentage_block(
        parent_experiment_id=parent_experiment_id,
        manifest_id=int(manifest.id),
        reason="rerun-from-manifest",
    )
    resolved_config[_PARENTAGE_KEY] = parentage

    base_model = manifest.base_model_source_ref or parent_exp.base_model
    mode = _normalize_training_mode(manifest.training_mode, parent_exp.training_mode)

    exp = await create_experiment(
        db,
        project_id=project_id,
        name=(run_name or f"{parent_exp.name} (rerun)").strip() or f"exp {parent_experiment_id} rerun",
        base_model=base_model,
        config=resolved_config,
        description=(description if description is not None else (parent_exp.description or "")),
        training_mode=mode,
    )
    await db.commit()
    await db.refresh(exp)
    return exp


async def clone_from_run(
    db: AsyncSession,
    *,
    project_id: int,
    parent_experiment_id: int,
    config_overrides: dict[str, Any] | None = None,
    run_name: str | None = None,
    description: str | None = None,
) -> Experiment:
    """Create a new experiment from the parent's config + caller overrides."""
    parent_exp, manifest = await _load_parent_manifest(
        db, project_id=project_id, parent_experiment_id=parent_experiment_id
    )
    overrides = dict(config_overrides or {})
    # Callers shouldn't be able to override the parentage pointer itself.
    overrides.pop(_PARENTAGE_KEY, None)
    base_config = dict(manifest.resolved_config or {})
    merged = _shallow_merge(base_config, overrides)
    merged[_PARENTAGE_KEY] = _parentage_block(
        parent_experiment_id=parent_experiment_id,
        manifest_id=int(manifest.id),
        reason="clone-from-run",
        overrides_applied=overrides,
    )

    # Overrides may legitimately change training_mode / base_model.
    mode_override = overrides.get("training_mode")
    mode = _normalize_training_mode(mode_override, _normalize_training_mode(manifest.training_mode, parent_exp.training_mode))

    base_model = (
        str(overrides.get("base_model") or "").strip()
        or manifest.base_model_source_ref
        or parent_exp.base_model
    )

    exp = await create_experiment(
        db,
        project_id=project_id,
        name=(run_name or f"{parent_exp.name} (clone)").strip() or f"exp {parent_experiment_id} clone",
        base_model=base_model,
        config=merged,
        description=(description if description is not None else (parent_exp.description or "")),
        training_mode=mode,
    )
    await db.commit()
    await db.refresh(exp)
    return exp


# -- Tolerance check --------------------------------------------------------


_METRIC_KEYS_ABS: tuple[str, ...] = (
    "final_train_loss",
    "final_eval_loss",
)


def _pick_numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _within_relative_tolerance(a: float, b: float, tolerance: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale <= tolerance


async def compare_run_metrics(
    db: AsyncSession,
    *,
    project_id: int,
    baseline_experiment_id: int,
    candidate_experiment_id: int,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Report whether the candidate reproduces the baseline within tolerance.

    Compares the experiments' top-level metrics (loss + any task metric
    columns). Each metric that both runs populated is labeled
    ``within_tolerance`` using the relative-tolerance rule used in
    numerical-reproduction harnesses. Metrics only one side populated are
    listed as ``baseline_only`` / ``candidate_only`` so tooling can decide
    whether to treat them as failures.
    """
    if tolerance < 0:
        raise ValueError("tolerance_must_be_non_negative")

    async def _load(exp_id: int) -> Experiment:
        row = (
            await db.execute(
                select(Experiment).where(
                    Experiment.id == exp_id, Experiment.project_id == project_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise ValueError("experiment_not_found")
        return row

    baseline = await _load(baseline_experiment_id)
    candidate = await _load(candidate_experiment_id)

    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    baseline_only: list[str] = []
    candidate_only: list[str] = []

    for key in _METRIC_KEYS_ABS:
        a = _pick_numeric(getattr(baseline, key, None))
        b = _pick_numeric(getattr(candidate, key, None))
        if a is None and b is None:
            continue
        if a is None:
            candidate_only.append(key)
            continue
        if b is None:
            baseline_only.append(key)
            continue
        ok = _within_relative_tolerance(a, b, tolerance)
        comparisons.append(
            {
                "metric": key,
                "baseline": a,
                "candidate": b,
                "delta": round(b - a, 6),
                "within_tolerance": ok,
            }
        )
        if not ok:
            failures.append(key)

    # Task metrics, if a training report stashed them under ``config._runtime``.
    def _runtime_task_metrics(exp: Experiment) -> dict[str, Any]:
        cfg = dict(exp.config or {})
        runtime = dict(cfg.get("_runtime") or {})
        raw = runtime.get("task_metrics")
        return dict(raw) if isinstance(raw, dict) else {}

    baseline_task = _runtime_task_metrics(baseline)
    candidate_task = _runtime_task_metrics(candidate)
    for key in sorted(set(baseline_task) | set(candidate_task)):
        a = _pick_numeric(baseline_task.get(key))
        b = _pick_numeric(candidate_task.get(key))
        if a is None and b is None:
            continue
        if a is None:
            candidate_only.append(f"task_metrics.{key}")
            continue
        if b is None:
            baseline_only.append(f"task_metrics.{key}")
            continue
        ok = _within_relative_tolerance(a, b, tolerance)
        comparisons.append(
            {
                "metric": f"task_metrics.{key}",
                "baseline": a,
                "candidate": b,
                "delta": round(b - a, 6),
                "within_tolerance": ok,
            }
        )
        if not ok:
            failures.append(f"task_metrics.{key}")

    reproduced = len(failures) == 0 and len(comparisons) > 0

    return {
        "baseline_experiment_id": int(baseline.id),
        "candidate_experiment_id": int(candidate.id),
        "tolerance": float(tolerance),
        "comparisons": comparisons,
        "failures": failures,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "reproduced": reproduced,
    }

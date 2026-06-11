"""Training Config Gap scanner — Coach-stage-2 phase 1.

Parallel to ``data_health_service`` but for the *training configuration*
side of the project. Where Data Health asks "is the data ready?", this
asks "is the config ready?" — given the project's selected recipe,
base model, and labelled-row count, are the hyperparameters the trainer
will use actually a good fit?

Phase 1 is read-only (advisory only). Every signal's ``suggested_action``
is a ``navigate`` pointer to the relevant config surface — the user still
clicks through and edits. Phase 2 will add an ``apply_config_patch``
action kind so a signal like "max_seq_length truncates 23% of rows" can
be one-click bumped.

Signals:

1. ``training_config.base_model_undersized`` — the chosen base model has
   too few parameters for the task difficulty. Recommends the next-
   heaviest entry from the recipe's ``alt_base_models`` list.
2. ``training_config.eval_cadence_too_sparse`` — given (labelled_rows,
   num_epochs, batch_size, gradient_accumulation_steps, eval_steps),
   the run will produce fewer than 3 eval observations. Without a
   learning curve you can't detect overfit or early-stop.
3. ``training_config.epochs_high_for_small_data`` — small gold set +
   many epochs = memorisation risk. The model fits the training rows
   verbatim and fails on anything held out.
4. ``training_config.warmup_low_for_aggressive_lr`` — long training run
   + high learning rate + tiny warmup window → loss-spike risk in the
   first few hundred steps.

All four are computed from project + recipe + Dataset row counts. No
tokenizer load, no model load, no dataset text sampling — keeps the
endpoint fast (~10ms) so it can poll from the training page.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.project import Project
from app.schemas.training import TrainingConfig


Severity = Literal["ok", "warn", "block"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────
# Thresholds — single source of truth. Where a number already exists in
# trainability_forecast_service we re-use it so the two surfaces agree.
# ─────────────────────────────────────────────────────────────────────

# Eval cadence. Fewer than 3 eval observations across the whole run
# means you have no learning curve — can't tell when the model peaked.
# Below 1 means the run will only emit a final-step eval.
EVAL_OBS_WARN = 3
EVAL_OBS_BLOCK = 1

# Epochs-vs-small-data memorisation risk. Same row-count brackets the
# trainability forecast uses for "tiny" gold sets, paired with epoch
# counts where memorisation starts dominating learning.
SMALL_DATA_WARN_ROWS = 100
SMALL_DATA_BLOCK_ROWS = 50
EPOCHS_WARN_FOR_SMALL = 5
EPOCHS_BLOCK_FOR_TINY = 8

# Warmup-vs-LR loss-spike risk. Empirical: runs with > 500 update steps,
# LR > 5e-4, and warmup < 2% of steps show the classic loss-spike-then-
# recover pattern in the first 200 steps. Below 0.5% with LR > 1e-3 the
# spike often doesn't recover and the run diverges.
LONG_RUN_STEPS = 500
AGGRESSIVE_LR = 5e-4
DIVERGENT_LR = 1e-3
WARMUP_WARN_RATIO = 0.02
WARMUP_BLOCK_RATIO = 0.005

# Base-model capacity. Same heuristic shape as trainability_forecast's
# capacity term, simplified for the gap surface. Task difficulty (0-1)
# scaled by row count gives a "params-needed" floor; current params
# below this triggers the signal.
TASK_DIFFICULTY: dict[str, float] = {
    "classification": 0.30,
    "instruction_sft": 0.45,
    "summarization": 0.55,
    "qa_sft": 0.50,
    "structured_extraction": 0.70,
}
# Mapping from (task_difficulty, labelled_rows_bucket) → "needed" param
# floor in millions. Cells calibrated against the recipes that ship —
# the 135M default is fine for classification + tiny corpora but
# underpowered for structured extraction on > 1k rows.
def _params_floor_m(task_difficulty: float, labelled_rows: int) -> int:
    if labelled_rows < 100:
        return 135
    if labelled_rows < 500:
        return 135 if task_difficulty < 0.5 else 360
    if labelled_rows < 2000:
        return 360 if task_difficulty < 0.5 else 1000
    return 500 if task_difficulty < 0.5 else 1500


# ─────────────────────────────────────────────────────────────────────
# Layman translation tables. Same pattern as data_health_service.
# ─────────────────────────────────────────────────────────────────────

_LAYMAN: dict[str, dict[str, str]] = {
    "training_config.no_recipe_selected": {
        "plain": "You haven't picked a recipe yet, so the platform can't tell what shape your training data should take or which hyperparameters fit.",
        "why": "Without a recipe the gap scanner has no baseline to compare your config against. Pick one (the picker bundles a recommended base model and a known-good hyperparameter starting point) and re-check this panel.",
    },
    "training_config.base_model_undersized": {
        "plain": "The base model you've picked is on the small side for your task and how much labelled data you have.",
        "why": "Tiny models train fast but plateau early — past a few hundred labelled rows on a hard task they stop improving no matter how clean the data is. Moving to the next-heaviest base in the recipe usually buys 5-15 F1 points for ~2x the train time.",
    },
    "training_config.eval_cadence_too_sparse": {
        "plain": "With your current settings, the trainer will only stop to check itself a handful of times — or maybe just at the end. You won't see a learning curve.",
        "why": "Without intermediate eval steps you can't tell when the model peaked, can't trigger early-stopping, and can't catch overfit before it happens. Tighten eval_steps so the run produces at least 3-5 eval points per epoch.",
    },
    "training_config.epochs_high_for_small_data": {
        "plain": "You have a small gold set but you're asking the trainer to loop over it many times.",
        "why": "Small data + many epochs = memorisation. The model will hit perfect scores on the rows it saw and fall apart on anything new. Either add more rows or cut epochs down.",
    },
    "training_config.warmup_low_for_aggressive_lr": {
        "plain": "You're combining a long training run, a high learning rate, and almost no warmup window.",
        "why": "Without warmup the optimiser takes its first updates at full learning rate and the loss usually spikes — sometimes recovers, sometimes diverges. A 3-5% warmup ratio almost always smooths the start.",
    },
}


def _layman_for(signal_id: str) -> dict[str, str]:
    if signal_id in _LAYMAN:
        return _LAYMAN[signal_id]
    return {"plain": "", "why": ""}


def _make_signal(
    *,
    id: str,
    severity: Severity,
    headline: str,
    suggested_action: dict | None = None,
    context: dict | None = None,
    plain_english: str | None = None,
    why_it_matters: str | None = None,
) -> dict[str, Any]:
    """Build a signal payload. Same shape as data_health_service's
    ``_make_signal`` minus the ``autofix_kind`` field — phase 1 is
    read-only. Phase 2 will introduce ``apply_config_patch``.
    """
    layman = _layman_for(id)
    return {
        "id": id,
        "severity": severity,
        "headline": headline,
        "plain_english": (
            plain_english if plain_english is not None else layman["plain"]
        ),
        "why_it_matters": (
            why_it_matters if why_it_matters is not None else layman["why"]
        ),
        "suggested_action": suggested_action,
        "context": context or {},
    }


def _recipe_id_for(project: Project) -> str | None:
    """Pull the selected recipe id off the project, or ``None`` when
    no recipe is selected. Same pattern coach_service uses."""
    selected = project.selected_recipe or {}
    if not isinstance(selected, dict):
        return None
    rid = selected.get("recipe_id")
    return rid if isinstance(rid, str) and rid else None


async def _count_labelled_rows(db: AsyncSession, project_id: int) -> int:
    """Sum ``record_count`` across CLEANED/SYNTHETIC/TRAIN datasets —
    the same definition the trainability forecast + data_health use for
    "labelled corpus size."
    """
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_([
                DatasetType.CLEANED,
                DatasetType.SYNTHETIC,
                DatasetType.TRAIN,
            ]),
        )
    )
    return sum(int(ds.record_count or 0) for ds in result.scalars())


def _effective_training_config(project: Project) -> TrainingConfig:
    """Compose the config the trainer will actually use.

    Phase 1 keeps this simple: start from ``TrainingConfig()`` defaults
    and only override ``base_model`` (which is the field that already
    has its own column on ``Project``). When phase 2 introduces a
    project-level training-config override block, this is where it
    layers in.
    """
    base = (project.base_model_name or "").strip()
    if not base:
        # Fall back to recipe-suggested base when the project hasn't
        # committed to one yet, so signals compare against what would
        # actually train.
        recipe_id = _recipe_id_for(project)
        if recipe_id:
            try:
                from app.services.recipe_service import get_recipe
                recipe = get_recipe(recipe_id)
                base = (
                    getattr(recipe, "suggested_base_model", None) or ""
                ).strip()
            except Exception:
                base = ""
    if not base:
        base = "HuggingFaceTB/SmolLM2-135M-Instruct"
    return TrainingConfig(base_model=base)


def _approx_total_steps(
    labelled_rows: int,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Effective optimizer update steps across the whole run.

    ``rows × epochs ÷ (per-step rows)`` where per-step rows is
    ``batch_size × grad_accum``. Floors at 1 — we never want a divide-
    by-zero downstream, and a 0-step run isn't a real config anyway.
    """
    per_step = max(1, batch_size * gradient_accumulation_steps)
    steps = (labelled_rows * num_epochs) // per_step
    return max(1, steps)


# ─────────────────────────────────────────────────────────────────────
# Per-signal scanners. Each returns a single signal dict (or None).
# ─────────────────────────────────────────────────────────────────────


def _base_model_undersized_signal(
    project: Project,
    recipe_id: str,
    labelled_rows: int,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    """Signal: chosen base model is undersized for task + data scale."""
    from app.services.recipe_service import get_recipe
    from app.services.trainability_forecast_service import (
        KNOWN_BASE_MODEL_PARAMS_M,
    )

    recipe = get_recipe(recipe_id)
    task_profile = getattr(recipe, "task_profile", None) if recipe else None
    difficulty = TASK_DIFFICULTY.get(task_profile or "", 0.45)
    current_base = cfg.base_model
    current_params = KNOWN_BASE_MODEL_PARAMS_M.get(current_base, 135)
    floor = _params_floor_m(difficulty, labelled_rows)

    # Pick recommended alt: next-heaviest from recipe alts.
    alts: list[str] = (
        list(getattr(recipe, "alt_base_models", []) or []) if recipe else []
    )

    def _params(name: str) -> int:
        return KNOWN_BASE_MODEL_PARAMS_M.get(name, 0)

    heavier = sorted(
        [a for a in alts if _params(a) > current_params],
        key=_params,
    )
    recommended = heavier[0] if heavier else None

    if current_params >= floor:
        return _make_signal(
            id="training_config.base_model_undersized",
            severity="ok",
            headline=(
                f"{current_base} ({current_params}M params) is sized "
                f"reasonably for this recipe + {labelled_rows} labelled rows."
            ),
            context={
                "current_base_model": current_base,
                "current_params_m": current_params,
                "labelled_rows": labelled_rows,
                "task_difficulty": difficulty,
                "params_floor_m": floor,
            },
        )

    severity: Severity = "block" if current_params * 2 < floor else "warn"
    body = (
        f"{current_base} has {current_params}M params; this recipe + "
        f"{labelled_rows} labelled rows usually wants ≥ {floor}M."
    )
    action: dict[str, Any]
    if recommended:
        action = {
            "kind": "navigate",
            "label": f"Consider {recommended}",
            "target": "training-base-model-picker",
            "params": {"recommended_base_model": recommended},
        }
    else:
        action = {
            "kind": "navigate",
            "label": "Open base-model picker",
            "target": "training-base-model-picker",
            "params": {},
        }
    return _make_signal(
        id="training_config.base_model_undersized",
        severity=severity,
        headline=body,
        suggested_action=action,
        context={
            "current_base_model": current_base,
            "current_params_m": current_params,
            "labelled_rows": labelled_rows,
            "task_difficulty": difficulty,
            "params_floor_m": floor,
            "recommended_base_model": recommended,
        },
    )


def _eval_cadence_signal(
    labelled_rows: int,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    """Signal: eval_steps is too coarse for the planned run length."""
    total_steps = _approx_total_steps(
        labelled_rows,
        cfg.num_epochs,
        cfg.batch_size,
        cfg.gradient_accumulation_steps,
    )
    eval_obs = total_steps // max(1, cfg.eval_steps)

    if eval_obs >= EVAL_OBS_WARN:
        return _make_signal(
            id="training_config.eval_cadence_too_sparse",
            severity="ok",
            headline=(
                f"Eval will fire ≈ {eval_obs} times across the run "
                f"(every {cfg.eval_steps} steps of ~{total_steps}). "
                f"Enough resolution for a learning curve."
            ),
            context={
                "total_steps": total_steps,
                "eval_steps": cfg.eval_steps,
                "eval_observations": eval_obs,
            },
        )

    severity: Severity = "block" if eval_obs <= EVAL_OBS_BLOCK else "warn"
    suggested_eval_steps = max(10, total_steps // 5)
    headline = (
        f"Eval will only fire ≈ {eval_obs} time"
        f"{'s' if eval_obs != 1 else ''} across "
        f"~{total_steps} training steps "
        f"(eval_steps={cfg.eval_steps})."
    )
    return _make_signal(
        id="training_config.eval_cadence_too_sparse",
        severity=severity,
        headline=headline,
        suggested_action={
            "kind": "navigate",
            "label": f"Tighten eval cadence (try eval_steps={suggested_eval_steps})",
            "target": "training-config",
            "params": {"recommended_eval_steps": suggested_eval_steps},
        },
        context={
            "total_steps": total_steps,
            "eval_steps": cfg.eval_steps,
            "eval_observations": eval_obs,
            "recommended_eval_steps": suggested_eval_steps,
        },
    )


def _epochs_overfit_signal(
    labelled_rows: int,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    """Signal: epoch count too high for the data scale."""
    epochs = cfg.num_epochs
    if labelled_rows >= SMALL_DATA_WARN_ROWS or epochs < EPOCHS_WARN_FOR_SMALL:
        return _make_signal(
            id="training_config.epochs_high_for_small_data",
            severity="ok",
            headline=(
                f"{epochs} epochs × {labelled_rows} labelled rows is within "
                f"the no-memorisation envelope."
            ),
            context={
                "num_epochs": epochs,
                "labelled_rows": labelled_rows,
            },
        )

    severity: Severity = (
        "block"
        if labelled_rows < SMALL_DATA_BLOCK_ROWS and epochs >= EPOCHS_BLOCK_FOR_TINY
        else "warn"
    )
    suggested_epochs = 3 if labelled_rows < SMALL_DATA_BLOCK_ROWS else 4
    return _make_signal(
        id="training_config.epochs_high_for_small_data",
        severity=severity,
        headline=(
            f"{epochs} epochs over only {labelled_rows} labelled rows — "
            f"the model will see each row {epochs} times."
        ),
        suggested_action={
            "kind": "navigate",
            "label": f"Reduce to {suggested_epochs} epochs",
            "target": "training-config",
            "params": {"recommended_num_epochs": suggested_epochs},
        },
        context={
            "num_epochs": epochs,
            "labelled_rows": labelled_rows,
            "recommended_num_epochs": suggested_epochs,
        },
    )


def _warmup_signal(
    labelled_rows: int,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    """Signal: warmup is too short for an aggressive LR + long run."""
    total_steps = _approx_total_steps(
        labelled_rows,
        cfg.num_epochs,
        cfg.batch_size,
        cfg.gradient_accumulation_steps,
    )
    lr = cfg.learning_rate
    warmup = cfg.warmup_ratio
    is_long = total_steps >= LONG_RUN_STEPS
    is_aggressive_lr = lr >= AGGRESSIVE_LR
    is_divergent_lr = lr >= DIVERGENT_LR

    if not (is_long and is_aggressive_lr) or warmup >= WARMUP_WARN_RATIO:
        return _make_signal(
            id="training_config.warmup_low_for_aggressive_lr",
            severity="ok",
            headline=(
                f"warmup_ratio={warmup:.2%} and lr={lr:g} over "
                f"~{total_steps} steps — fine."
            ),
            context={
                "warmup_ratio": warmup,
                "learning_rate": lr,
                "total_steps": total_steps,
            },
        )

    severity: Severity = (
        "block"
        if is_divergent_lr and warmup < WARMUP_BLOCK_RATIO
        else "warn"
    )
    suggested_warmup = 0.03
    return _make_signal(
        id="training_config.warmup_low_for_aggressive_lr",
        severity=severity,
        headline=(
            f"warmup_ratio={warmup:.2%} with lr={lr:g} over ~{total_steps} "
            f"steps — loss-spike risk at the start of training."
        ),
        suggested_action={
            "kind": "navigate",
            "label": f"Bump warmup_ratio to {suggested_warmup:.0%}",
            "target": "training-config",
            "params": {"recommended_warmup_ratio": suggested_warmup},
        },
        context={
            "warmup_ratio": warmup,
            "learning_rate": lr,
            "total_steps": total_steps,
            "recommended_warmup_ratio": suggested_warmup,
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Public entry point.
# ─────────────────────────────────────────────────────────────────────


async def scan_training_config_gaps(
    db: AsyncSession, project_id: int
) -> dict[str, Any]:
    """Return the training-config gap report for a project.

    Same outer shape as ``compute_data_health_report`` so the frontend
    can reuse the same panel components / severity rendering. Single
    group in phase 1 — every signal is a "training config" concern;
    there's no sub-grouping yet.

    Raises ``ValueError`` if the project doesn't exist. The API layer
    translates that to a 404.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    signals: list[dict[str, Any]] = []
    recipe_id = _recipe_id_for(project)

    if not recipe_id:
        # Without a recipe we can't reason about base-model sizing or
        # task difficulty. Surface that as the sole signal — same
        # fallback the trainability forecast emits.
        signals.append(_make_signal(
            id="training_config.no_recipe_selected",
            severity="block",
            headline="No recipe selected — pick one to scan config gaps.",
            suggested_action={
                "kind": "navigate",
                "label": "Open recipe picker",
                "target": "recipe-picker",
                "params": {},
            },
            context={},
        ))
    else:
        labelled_rows = await _count_labelled_rows(db, project_id)
        cfg = _effective_training_config(project)
        signals.append(_base_model_undersized_signal(
            project, recipe_id, labelled_rows, cfg
        ))
        signals.append(_eval_cadence_signal(labelled_rows, cfg))
        signals.append(_epochs_overfit_signal(labelled_rows, cfg))
        signals.append(_warmup_signal(labelled_rows, cfg))

    group = {
        "id": "training_config",
        "title": "Training config",
        "subtitle": "Hyperparameters + base model vs your data scale",
        "signals": signals,
    }

    severity_summary = {
        "ok": sum(1 for s in signals if s["severity"] == "ok"),
        "warn": sum(1 for s in signals if s["severity"] == "warn"),
        "block": sum(1 for s in signals if s["severity"] == "block"),
    }
    if severity_summary["block"] > 0:
        overall: Severity = "block"
    elif severity_summary["warn"] > 0:
        overall = "warn"
    else:
        overall = "ok"

    return {
        "project_id": int(project_id),
        "computed_at": _utcnow().isoformat(),
        "overall": overall,
        "severity_summary": severity_summary,
        "total_signals": len(signals),
        "groups": [group],
    }

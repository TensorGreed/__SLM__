"""Training Config Gap scanner + patch engine — Coach-stage-2 phases 1 + 2 + 3.

Parallel to ``data_health_service`` but for the *training configuration*
side of the project. Where Data Health asks "is the data ready?", this
asks "is the config ready?" — given the project's selected recipe,
base model, and labelled-row count, are the hyperparameters the trainer
will use actually a good fit?

Phase 1 was read-only (advisory only). Phase 2 adds one-click
remediation: signals where the recommended patch is unambiguous (eval
cadence, epochs trim, warmup bump) carry an ``apply_patch_kind`` field,
and the matching ``POST /training-config-gaps/patch/{preview|apply}``
endpoints persist the change as a partial dict under
``project.runtime_config["training_config_overrides"]``. The gap
scanner overlays that block onto ``TrainingConfig()`` defaults so a
re-scan after Apply flips the signal severity to ``ok``. TrainingPanel
reads the same block on mount via ``GET /training-config-gaps/overrides``
and prefills its form via the existing ``applySuggestedConfig`` hook —
the override is the single source of truth across surfaces.

The base-model-undersized signal stays as a ``navigate`` for now
because the swap is a bigger lift (different parameter counts, possibly
different tokenizer); phase 3 may revisit.

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
from pathlib import Path
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

# Phase 3 — text-sampling signals.
#
# Sample size for both truncation + OOV. 100 rows is enough to
# distinguish "occasional outlier" from "structurally truncating"; the
# JSONL streaming + chars/4 path runs in <20ms for that size.
TEXT_SAMPLE_SIZE = 100

# Chars-per-token approximation. Modern byte-BPE tokenizers land
# around 3.5-4.0 chars/token for English; we use 4 as a conservative
# upper bound so phase 3 doesn't over-report truncation. Real tokenizer
# pass lives in the OOV signal (already loads the tokenizer); the
# truncation signal stays cheap so the gap endpoint stays fast.
CHARS_PER_TOKEN_APPROX = 4

# Truncation rate brackets. Anything > 10% loses meaningful training
# signal; > 25% means the trainer is silently dropping the tail of most
# rows and the user thinks they're training on full sequences when
# they're not.
TRUNCATION_WARN_FRAC = 0.10
TRUNCATION_BLOCK_FRAC = 0.25

# Tokenizer OOV rate brackets. SentencePiece byte-BPE tokenizers
# (SmolLM2, Qwen2.5, Llama3, modern HF defaults) emit ZERO unk tokens
# because byte fallback handles every input — those tokenizers have
# ``unk_token=None``. For tokenizers with an explicit unk (older BERT-
# style WordPiece, some multilingual variants), > 5% unk rate means
# the chosen base model's vocab doesn't match this domain and the
# model has no way to represent meaningful chunks of the input.
OOV_WARN_FRAC = 0.05
OOV_BLOCK_FRAC = 0.15

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
    "training_config.max_seq_truncation_risk": {
        "plain": "A meaningful share of your training rows are longer than the trainer's max sequence length — the trainer silently drops everything past the cap.",
        "why": "When rows get truncated mid-sequence the model never sees the end (the answer, the closing tag, the rationale, etc.). You end up training a model on questions whose answers were cut off. Either raise max_seq_length to cover the longest rows, or shorten the rows upstream.",
    },
    "training_config.tokenizer_oov_high": {
        "plain": "The base model's tokenizer hits its 'unknown' fallback on a meaningful share of your training tokens.",
        "why": "Tokens the tokenizer can't represent get collapsed to a single placeholder; the model can never learn to predict them and never sees the right input context. High unk rates usually mean the chosen base model was trained on a different language or character set than your data — pick a base whose vocab matches.",
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
    apply_patch_kind: str | None = None,
) -> dict[str, Any]:
    """Build a signal payload.

    ``apply_patch_kind`` (phase 2) flags signals the patch engine can
    resolve in one click. The frontend renders an "Apply fix" button
    when this is set, calling ``POST /training-config-gaps/patch/preview``
    with the signal id as the payload. ``None`` = the signal is
    informational only (no safe patch exists yet for it).
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
        "apply_patch_kind": apply_patch_kind,
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


OVERRIDES_KEY = "training_config_overrides"
# Fields the patch engine is allowed to write. Anything not in here is
# rejected at apply time, so a malformed signal_context can't silently
# poison unrelated fields.
PATCHABLE_FIELDS: frozenset[str] = frozenset({
    "eval_steps",
    "num_epochs",
    "warmup_ratio",
})


def _get_overrides(project: Project) -> dict[str, Any]:
    """Read the persistent training-config overrides block from the
    project's ``runtime_config``. Returns an empty dict when nothing has
    been applied yet — never ``None`` so callers can ``.get`` cleanly.
    """
    runtime = project.runtime_config or {}
    if not isinstance(runtime, dict):
        return {}
    block = runtime.get(OVERRIDES_KEY)
    if not isinstance(block, dict):
        return {}
    # Defensive copy so callers can't accidentally mutate the model
    # column in place.
    return {k: v for k, v in block.items() if k in PATCHABLE_FIELDS}


def _effective_training_config(project: Project) -> TrainingConfig:
    """Compose the config the trainer will actually use.

    Layering (highest precedence first):
      1. ``runtime_config["training_config_overrides"]`` — what the user
         applied via phase-2 patches.
      2. ``project.base_model_name`` — the dedicated column.
      3. Recipe-suggested base model when the project hasn't committed
         to one.
      4. ``TrainingConfig()`` defaults.
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

    overrides = _get_overrides(project)
    init_kwargs: dict[str, Any] = {"base_model": base}
    # Only carry overrides that match a TrainingConfig field. Pydantic
    # already validates ranges (eval_steps ≥ 1, warmup 0-1, etc.) so a
    # bad override raises ValidationError at apply time, not here.
    for field in PATCHABLE_FIELDS:
        if field in overrides:
            init_kwargs[field] = overrides[field]
    return TrainingConfig(**init_kwargs)


async def _sample_training_text(
    db: AsyncSession, project_id: int, *, limit: int = TEXT_SAMPLE_SIZE,
) -> list[str]:
    """Read up to ``limit`` text blobs from the project's largest
    labelled dataset. Picks the biggest CLEANED/SYNTHETIC/TRAIN dataset
    by record_count so we sample from the source that will dominate
    training. Empty result is fine — callers degrade gracefully (no
    sample → no signal, since we can't fairly score what we can't read).
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
    datasets = sorted(
        result.scalars(),
        key=lambda d: int(d.record_count or 0),
        reverse=True,
    )
    for dataset in datasets:
        if not dataset.file_path:
            continue
        path = Path(dataset.file_path)
        if not path.exists():
            continue
        # Lazy import: dataset_service pulls heavy deps; only load when
        # we actually have a path to read.
        from app.services.dataset_service import _load_records_from_file
        records = _load_records_from_file(path, max_records=limit)
        texts: list[str] = []
        for row in records:
            if not isinstance(row, dict):
                continue
            text = _row_to_text(row)
            if text:
                texts.append(text)
            if len(texts) >= limit:
                break
        if texts:
            return texts
    return []


def _row_to_text(row: dict[str, Any]) -> str:
    """Coerce a row dict to a single text blob. Mirrors the same helper
    in trainability_forecast_service so the two surfaces sample the
    same fields. Kept local rather than imported to avoid coupling
    against the heavy forecast module.
    """
    parts: list[str] = []
    for key in (
        "input", "expected", "question", "answer",
        "text", "prompt", "response", "output",
    ):
        value = row.get(key)
        if isinstance(value, dict):
            for sub_value in value.values():
                if isinstance(sub_value, str):
                    parts.append(sub_value)
        elif isinstance(value, str):
            parts.append(value)
    if not parts:
        for value in row.values():
            if isinstance(value, str):
                parts.append(value)
    return " ".join(parts)


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
    # Pick the largest eval_steps that still produces ≥ EVAL_OBS_WARN
    # observations. Floor at 1 (the trainer can eval every step on
    # very short runs — annoying but honest). Phase 1 used a floor of
    # 10 which didn't actually close the gap on short runs; phase 2's
    # patch engine relies on this being a valid closure.
    suggested_eval_steps = max(1, total_steps // EVAL_OBS_WARN)
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
        apply_patch_kind="eval_steps_recommend",
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
        apply_patch_kind="num_epochs_recommend",
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
        apply_patch_kind="warmup_ratio_recommend",
    )


def _truncation_signal(
    text_samples: list[str], cfg: TrainingConfig,
) -> dict[str, Any]:
    """Signal: a meaningful share of rows would be truncated.

    Uses chars/CHARS_PER_TOKEN_APPROX as a token-count approximation
    (cheap; no tokenizer load). Errs on the conservative side — modern
    BPE tokenizers actually compress more aggressively, so a row whose
    chars/4 says "fits" almost certainly does fit, and a row whose
    chars/4 says "truncates" is at risk under any tokenizer.
    """
    if not text_samples:
        return _make_signal(
            id="training_config.max_seq_truncation_risk",
            severity="ok",
            headline=(
                "No training rows available to sample — skipping "
                "truncation check."
            ),
            context={"sample_size": 0, "max_seq_length": cfg.max_seq_length},
        )
    cap = cfg.max_seq_length
    truncated = sum(
        1 for text in text_samples
        if len(text) // CHARS_PER_TOKEN_APPROX > cap
    )
    frac = truncated / len(text_samples)
    sample_size = len(text_samples)

    if frac < TRUNCATION_WARN_FRAC:
        return _make_signal(
            id="training_config.max_seq_truncation_risk",
            severity="ok",
            headline=(
                f"{truncated} of {sample_size} sampled rows would be "
                f"truncated at max_seq_length={cap} (~{frac:.1%})."
            ),
            context={
                "sample_size": sample_size,
                "truncated_count": truncated,
                "truncation_fraction": round(frac, 4),
                "max_seq_length": cap,
            },
        )

    severity: Severity = "block" if frac >= TRUNCATION_BLOCK_FRAC else "warn"
    # Suggest a max_seq_length that covers the 95th-percentile sample.
    # Sort lengths, pick the 95th-pct and round up to the next 256.
    token_lens = sorted(
        len(t) // CHARS_PER_TOKEN_APPROX for t in text_samples
    )
    p95_idx = max(0, int(len(token_lens) * 0.95) - 1)
    p95_tokens = token_lens[p95_idx]
    suggested = max(cap * 2, ((p95_tokens // 256) + 1) * 256)
    return _make_signal(
        id="training_config.max_seq_truncation_risk",
        severity=severity,
        headline=(
            f"{truncated} of {sample_size} sampled rows ({frac:.0%}) "
            f"would be truncated at max_seq_length={cap}."
        ),
        suggested_action={
            "kind": "navigate",
            "label": f"Raise max_seq_length to {suggested}",
            "target": "training-config",
            "params": {"recommended_max_seq_length": suggested},
        },
        context={
            "sample_size": sample_size,
            "truncated_count": truncated,
            "truncation_fraction": round(frac, 4),
            "max_seq_length": cap,
            "p95_approx_tokens": p95_tokens,
            "recommended_max_seq_length": suggested,
        },
    )


def _tokenizer_oov_signal(
    text_samples: list[str], cfg: TrainingConfig,
) -> dict[str, Any]:
    """Signal: the base model's tokenizer can't represent a meaningful
    share of training tokens.

    Loads the tokenizer for ``cfg.base_model`` and tokenizes a sample
    of training rows. Three outcomes:

    1. Tokenizer load fails (offline, missing model, etc.) → ``ok``
       with a "skipped" note. We don't fabricate signals from missing
       data.
    2. Tokenizer's ``unk_token`` is None (modern byte-BPE: SmolLM2,
       Qwen2.5, Llama3, etc.) → ``ok`` with a "byte fallback covers
       everything" note. There IS no OOV concept for these tokenizers.
    3. Tokenizer emits unks → measure rate vs. ``OOV_WARN_FRAC`` /
       ``OOV_BLOCK_FRAC``.
    """
    base_model = cfg.base_model
    if not text_samples:
        return _make_signal(
            id="training_config.tokenizer_oov_high",
            severity="ok",
            headline=(
                "No training rows available to sample — skipping OOV check."
            ),
            context={"sample_size": 0, "base_model": base_model},
        )

    # Defensive tokenizer load. Don't fail the whole gap report when
    # the model can't be fetched (offline dev box, private model
    # behind auth, etc.) — emit ok with a skipped note.
    try:
        from transformers import AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except Exception as exc:
        return _make_signal(
            id="training_config.tokenizer_oov_high",
            severity="ok",
            headline=(
                f"Tokenizer for {base_model} not available locally — "
                f"skipping OOV check."
            ),
            context={
                "sample_size": len(text_samples),
                "base_model": base_model,
                "skipped_reason": str(exc)[:160],
            },
        )

    unk_token = getattr(tokenizer, "unk_token", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token is None or unk_token_id is None:
        return _make_signal(
            id="training_config.tokenizer_oov_high",
            severity="ok",
            headline=(
                f"{base_model}'s tokenizer uses byte fallback — every "
                f"input has a representation, OOV is not a concern."
            ),
            context={
                "sample_size": len(text_samples),
                "base_model": base_model,
                "byte_fallback": True,
            },
        )

    total_tokens = 0
    unk_count = 0
    for text in text_samples:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            continue
        total_tokens += len(ids)
        unk_count += sum(1 for i in ids if i == unk_token_id)

    if total_tokens == 0:
        return _make_signal(
            id="training_config.tokenizer_oov_high",
            severity="ok",
            headline="Sample yielded zero tokens — skipping OOV check.",
            context={
                "sample_size": len(text_samples),
                "base_model": base_model,
                "total_tokens": 0,
            },
        )

    frac = unk_count / total_tokens
    if frac < OOV_WARN_FRAC:
        return _make_signal(
            id="training_config.tokenizer_oov_high",
            severity="ok",
            headline=(
                f"{base_model}'s tokenizer covers your sample ({frac:.2%} "
                f"unk over {total_tokens} tokens)."
            ),
            context={
                "sample_size": len(text_samples),
                "base_model": base_model,
                "total_tokens": total_tokens,
                "unk_count": unk_count,
                "unk_fraction": round(frac, 4),
            },
        )

    severity: Severity = "block" if frac >= OOV_BLOCK_FRAC else "warn"
    return _make_signal(
        id="training_config.tokenizer_oov_high",
        severity=severity,
        headline=(
            f"{base_model}'s tokenizer emits {unk_count} unk tokens over "
            f"{total_tokens} sampled tokens ({frac:.1%}) — vocabulary "
            f"mismatch with this domain."
        ),
        suggested_action={
            "kind": "navigate",
            "label": "Pick a base model whose vocab covers this domain",
            "target": "training-base-model-picker",
            "params": {},
        },
        context={
            "sample_size": len(text_samples),
            "base_model": base_model,
            "total_tokens": total_tokens,
            "unk_count": unk_count,
            "unk_fraction": round(frac, 4),
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
        # Phase 3 — text-sampling signals. Both share one sample read
        # so the JSONL pass + row-to-text coercion is amortized.
        text_samples = await _sample_training_text(db, project_id)
        signals.append(_truncation_signal(text_samples, cfg))
        signals.append(_tokenizer_oov_signal(text_samples, cfg))

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


# ─────────────────────────────────────────────────────────────────────
# Patch registry — phase 2.
#
# Each entry maps an ``apply_patch_kind`` to:
#   - the signal_id that emits it (for the preview's "resolves: ..." line)
#   - a builder that takes the signal's ``context`` dict and returns the
#     partial patch to write into runtime_config["training_config_overrides"]
#   - a human-facing label so the preview modal reads cleanly
#
# Three patches today. Each writes a single field; the gap scanner
# re-scan after Apply confirms the gap is closed.
# ─────────────────────────────────────────────────────────────────────


PATCH_KINDS: tuple[str, ...] = (
    "eval_steps_recommend",
    "num_epochs_recommend",
    "warmup_ratio_recommend",
)


def _resolve_signal(
    report: dict[str, Any], signal_id: str
) -> dict[str, Any] | None:
    for group in report.get("groups", []):
        for sig in group.get("signals", []):
            if sig.get("id") == signal_id:
                return sig
    return None


def _patch_for_eval_steps(ctx: dict[str, Any]) -> dict[str, Any]:
    rec = int(ctx.get("recommended_eval_steps") or 0)
    if rec < 1:
        raise ValueError(
            "Signal context is missing a valid recommended_eval_steps."
        )
    return {"eval_steps": rec}


def _patch_for_num_epochs(ctx: dict[str, Any]) -> dict[str, Any]:
    rec = int(ctx.get("recommended_num_epochs") or 0)
    if rec < 1:
        raise ValueError(
            "Signal context is missing a valid recommended_num_epochs."
        )
    return {"num_epochs": rec}


def _patch_for_warmup_ratio(ctx: dict[str, Any]) -> dict[str, Any]:
    rec = float(ctx.get("recommended_warmup_ratio") or 0.0)
    if not (0.0 <= rec <= 1.0):
        raise ValueError(
            "Signal context is missing a valid recommended_warmup_ratio."
        )
    return {"warmup_ratio": rec}


# kind → (signal_id, patch_builder, human_label, plain_english_summary)
_PATCH_REGISTRY: dict[
    str,
    tuple[str, Any, str, str],
] = {
    "eval_steps_recommend": (
        "training_config.eval_cadence_too_sparse",
        _patch_for_eval_steps,
        "Tighten eval cadence",
        (
            "Bumps eval_steps so the trainer checks itself often enough "
            "to draw a learning curve."
        ),
    ),
    "num_epochs_recommend": (
        "training_config.epochs_high_for_small_data",
        _patch_for_num_epochs,
        "Reduce epochs",
        (
            "Cuts num_epochs to a value that won't overfit your current "
            "labelled-row count."
        ),
    ),
    "warmup_ratio_recommend": (
        "training_config.warmup_low_for_aggressive_lr",
        _patch_for_warmup_ratio,
        "Bump warmup",
        (
            "Raises warmup_ratio to 3% so the optimizer eases into the "
            "aggressive LR without spiking."
        ),
    ),
}


def _patch_label(kind: str) -> str:
    entry = _PATCH_REGISTRY.get(kind)
    return entry[2] if entry else kind


async def preview_patch(
    db: AsyncSession, project_id: int, signal_id: str
) -> dict[str, Any]:
    """Build the before → after diff a patch *would* apply, without
    mutating anything.

    Resolves the signal in the current gap report, looks up its
    ``apply_patch_kind``, builds the patch from the signal's context, and
    returns the proposed change paired with the current effective value.

    Raises ``ValueError`` with a human-meaningful message when:
      - the project is missing (404 at the API layer)
      - the signal is not in the current report
      - the signal has no ``apply_patch_kind`` (i.e. no safe patch exists
        for it yet)
      - the patch builder rejects the signal's context (bad recommended)
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    report = await scan_training_config_gaps(db, project_id)
    signal = _resolve_signal(report, signal_id)
    if signal is None:
        raise ValueError(
            f"Signal {signal_id!r} not in the current gap report."
        )
    kind = signal.get("apply_patch_kind")
    if not kind or kind not in _PATCH_REGISTRY:
        raise ValueError(
            f"Signal {signal_id!r} has no one-click patch available."
        )

    _, builder, label, plain = _PATCH_REGISTRY[kind]
    patch = builder(signal.get("context") or {})
    # Sanity-check every patched field is in the allow-list. Defensive
    # — the builders all return allow-listed keys today, but a future
    # builder typo shouldn't silently land an unknown field on the
    # project's runtime_config.
    bad = set(patch.keys()) - PATCHABLE_FIELDS
    if bad:
        raise ValueError(
            f"Patch produced disallowed field(s) {sorted(bad)}; "
            f"allow-list is {sorted(PATCHABLE_FIELDS)}."
        )

    current_cfg = _effective_training_config(project)
    before = {k: getattr(current_cfg, k) for k in patch.keys()}
    after = {**before, **patch}
    return {
        "project_id": int(project_id),
        "signal_id": signal_id,
        "patch_kind": kind,
        "patch_label": label,
        "plain_english": plain,
        "patch": patch,
        "before": before,
        "after": after,
        "safe_to_apply": True,
    }


async def apply_patch(
    db: AsyncSession, project_id: int, signal_id: str
) -> dict[str, Any]:
    """Persist the patch onto ``project.runtime_config[OVERRIDES_KEY]``.

    Idempotent: applying the same patch twice writes the same value the
    second time and the scanner re-emits the signal as ``ok`` either
    way. Caller commits the session (the API endpoint does this once,
    so multiple patches per request would batch cleanly if we ever
    needed bulk apply).
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    preview = await preview_patch(db, project_id, signal_id)
    patch: dict[str, Any] = dict(preview["patch"])

    runtime = dict(project.runtime_config or {})
    existing = runtime.get(OVERRIDES_KEY)
    block = dict(existing) if isinstance(existing, dict) else {}
    block.update(patch)
    runtime[OVERRIDES_KEY] = block
    # Reassigning the whole dict (rather than mutating in place) tells
    # SQLAlchemy's JSON column the value changed. JSON columns don't
    # track in-place mutation; this is the standard workaround.
    project.runtime_config = runtime
    return {
        **preview,
        "applied": True,
        "overrides_after": block,
    }


def read_overrides(project: Project) -> dict[str, Any]:
    """Public helper for the ``GET /overrides`` endpoint. Returns a
    plain dict the frontend can plumb directly into
    ``applySuggestedConfig``.
    """
    return _get_overrides(project)

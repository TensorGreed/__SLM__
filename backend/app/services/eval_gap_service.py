"""Eval Gap scanner — Coach-stage-2 phase 3.

Parallel to ``training_config_gap_service`` but for the *evaluation*
side of the project. Where training-config gaps ask "is the trainer
set up to learn?", eval gaps ask "is the eval set up to honestly
measure?".

Three signals in phase 3:

1. ``eval_gaps.archetype_coverage_low`` — the project's gold-set
   features (row count, balance, length, diversity, etc.) sit below
   the recipe's archetype baseline. Reuses
   ``archetype_service.compare_project_to_archetype`` so the threshold
   logic stays in one place; we just count the ``below`` features and
   emit a roll-up signal pointing at the existing archetype panel.

2. ``eval_gaps.no_regression_baseline`` — no promoted Checkpoint
   exists. Without a pinned baseline you can't tell whether a new run
   beat the prior best — every eval reads as standalone instead of as
   a regression test.

3. ``eval_gaps.train_eval_label_kl_high`` — classification only.
   Compares train-set label distribution to gold-set label distribution
   via KL divergence. High KL = your eval set doesn't reflect your
   train set, so the F1 you ship doesn't predict prod F1.

Read-only in phase 3. Phase 4 may add patch actions (snapshot last
green checkpoint as baseline, augment gold set to match train
distribution, etc.).
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.experiment import (
    Checkpoint,
    Experiment,
    ExperimentStatus,
)
from app.models.project import Project


Severity = Literal["ok", "warn", "block"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────
# Thresholds.
# ─────────────────────────────────────────────────────────────────────

# Archetype coverage. We count the features the project sits BELOW
# the recipe's archetype p25 band on. 1-2 below-band features = warn,
# 3+ = block. Mirrors the "summary" the archetype panel already shows;
# we surface it as a single rolled-up signal here for the eval-gap
# roll-up consumers.
ARCHETYPE_BELOW_WARN = 1
ARCHETYPE_BELOW_BLOCK = 3

# KL-divergence thresholds for the train/eval label distribution match.
# 0.10 nats is the empirical noise floor across the 8 shipped templates
# (uniform shifts of < 5pp across classes land here). 0.50 nats is when
# the eval set materially under-samples or over-samples specific
# classes vs. train.
KL_WARN = 0.10
KL_BLOCK = 0.50

# Minimum sample sizes for KL — below these the variance dominates and
# the signal would just be noisy.
KL_MIN_TRAIN_ROWS = 20
KL_MIN_EVAL_ROWS = 10


# ─────────────────────────────────────────────────────────────────────
# Layman translation tables.
# ─────────────────────────────────────────────────────────────────────

_LAYMAN: dict[str, dict[str, str]] = {
    "eval_gaps.no_recipe_selected": {
        "plain": "You haven't picked a recipe yet, so we can't reason about your eval setup.",
        "why": "Eval gaps depend on knowing the task shape (classification vs span vs qa-sft). Pick a recipe to enable the scan.",
    },
    "eval_gaps.archetype_coverage_low": {
        "plain": "Your gold set looks materially smaller, less balanced, or less diverse than the gold sets of recipes that have trained successfully in the past.",
        "why": "Archetypes are built from prior-passing projects — when your gold set sits below their p25 on multiple features, the model you train on it will likely underperform too. Add rows, rebalance classes, or improve diversity until the gold set lands inside the band.",
    },
    "eval_gaps.no_regression_baseline": {
        "plain": "You don't have a promoted baseline checkpoint to compare new runs against.",
        "why": "Without a baseline every eval reads as 'is this number good?' instead of 'did this number beat my last best?'. Promote a green run's checkpoint as the baseline so new runs surface as regressions or wins rather than standalone numbers.",
    },
    "eval_gaps.train_eval_label_kl_high": {
        "plain": "The label distribution in your training set doesn't match the label distribution in your gold/eval set.",
        "why": "Eval is supposed to predict prod performance. When the train and eval distributions don't match, the F1 you ship doesn't tell you what users will see — usually the prod number is much worse. Either rebalance the gold set to match training or rebalance training to match what's in the wild.",
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
    """Build a signal payload. Same shape as training_config_gap and
    data_health; phase 3 is read-only so no apply_patch_kind."""
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
    """Pull the selected recipe id off the project."""
    selected = project.selected_recipe or {}
    if not isinstance(selected, dict):
        return None
    rid = selected.get("recipe_id")
    return rid if isinstance(rid, str) and rid else None


# ─────────────────────────────────────────────────────────────────────
# Per-signal scanners.
# ─────────────────────────────────────────────────────────────────────


async def _archetype_coverage_signal(
    db: AsyncSession, project_id: int, recipe_id: str,
) -> dict[str, Any]:
    """Reuse archetype_service.compare_project_to_archetype and count
    the features the project falls BELOW the archetype band on.

    Degrades to ``ok`` (with a skipped note) when:
      - the comparison service raises (project has no gold rows yet,
        archetype not computed, recipe doesn't have one, etc.)
      - the comparison returns no features
    """
    try:
        from app.services.archetype_service import (
            compare_project_to_archetype,
        )
        comparison = await compare_project_to_archetype(db, project_id)
    except Exception as exc:
        return _make_signal(
            id="eval_gaps.archetype_coverage_low",
            severity="ok",
            headline=(
                "Archetype comparison not available yet — skipping "
                "coverage check."
            ),
            context={
                "recipe_id": recipe_id,
                "skipped_reason": str(exc)[:160],
            },
        )

    features = comparison.get("features") or []
    below = [
        f for f in features
        if f.get("status") == "below"
    ]
    below_count = len(below)
    total = len(features)

    if below_count == 0:
        return _make_signal(
            id="eval_gaps.archetype_coverage_low",
            severity="ok",
            headline=(
                f"Gold set lands inside the recipe's archetype band on "
                f"all {total} features."
            ),
            context={
                "recipe_id": recipe_id,
                "below_count": 0,
                "feature_count": total,
            },
        )

    severity: Severity = (
        "block" if below_count >= ARCHETYPE_BELOW_BLOCK
        else "warn" if below_count >= ARCHETYPE_BELOW_WARN
        else "ok"
    )
    # Pick the first below-band feature's id for the headline so users
    # see a concrete hook rather than a count.
    top = below[0]
    return _make_signal(
        id="eval_gaps.archetype_coverage_low",
        severity=severity,
        headline=(
            f"{below_count} of {total} archetype features sit below the "
            f"recipe's p25 band — top: {top.get('label') or top.get('feature_id')}."
        ),
        suggested_action={
            "kind": "navigate",
            "label": "Open archetype comparison",
            "target": "archetype-comparison-panel",
            "params": {},
        },
        context={
            "recipe_id": recipe_id,
            "below_count": below_count,
            "feature_count": total,
            "top_below_feature_id": top.get("feature_id"),
        },
    )


async def _regression_baseline_signal(
    db: AsyncSession, project_id: int,
) -> dict[str, Any]:
    """Check whether any Checkpoint under the project has been
    promoted (``promoted_at`` not null). A promoted checkpoint is the
    pinned regression baseline new runs compare against.
    """
    result = await db.execute(
        select(Checkpoint)
        .join(Experiment, Experiment.id == Checkpoint.experiment_id)
        .where(
            and_(
                Experiment.project_id == project_id,
                Checkpoint.promoted_at.is_not(None),
            )
        )
        .limit(1)
    )
    promoted = result.scalar_one_or_none()

    if promoted is not None:
        return _make_signal(
            id="eval_gaps.no_regression_baseline",
            severity="ok",
            headline=(
                f"Baseline pinned: experiment #{promoted.experiment_id} "
                f"checkpoint at step {promoted.step}."
            ),
            context={
                "promoted_checkpoint_id": int(promoted.id),
                "promoted_experiment_id": int(promoted.experiment_id),
                "promoted_step": int(promoted.step),
            },
        )

    # Soften the severity when no run has even completed yet — there's
    # nothing to promote, so flagging "no baseline" is just noise.
    completed_result = await db.execute(
        select(Experiment).where(
            Experiment.project_id == project_id,
            Experiment.status == ExperimentStatus.COMPLETED,
        ).limit(1)
    )
    has_completed = completed_result.scalar_one_or_none() is not None
    if not has_completed:
        return _make_signal(
            id="eval_gaps.no_regression_baseline",
            severity="ok",
            headline=(
                "No training runs have completed yet — baseline check "
                "deferred until a run finishes."
            ),
            context={"has_completed_runs": False},
        )

    return _make_signal(
        id="eval_gaps.no_regression_baseline",
        severity="warn",
        headline=(
            "You have completed runs but no promoted checkpoint to use "
            "as a regression baseline."
        ),
        suggested_action={
            "kind": "navigate",
            "label": "Promote a checkpoint as baseline",
            "target": "checkpoints-panel",
            "params": {},
        },
        context={"has_completed_runs": True},
    )


def _row_to_label(row: dict[str, Any]) -> str | None:
    """Best-effort label extraction. Mirrors the trainability forecast's
    convention: row['label'] / row['expected'] / row['output'].label /
    common nested shapes.
    """
    if not isinstance(row, dict):
        return None
    for key in ("label", "expected_label", "output_label"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("expected", "output", "answer"):
        value = row.get(key)
        if isinstance(value, dict):
            inner = value.get("label") or value.get("class")
            if isinstance(inner, str) and inner:
                return inner
        elif isinstance(value, str) and value:
            # Fall through — for classification recipes the output IS
            # the label.
            return value
    return None


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) over the union of label keys. Smooths q to avoid
    division by zero; uses 1e-9 epsilon which is below any
    meaningful threshold without distorting real signals."""
    eps = 1e-9
    labels = set(p.keys()) | set(q.keys())
    total = 0.0
    for label in labels:
        pi = max(p.get(label, 0.0), eps)
        qi = max(q.get(label, 0.0), eps)
        total += pi * math.log(pi / qi)
    return total


async def _load_labels_from_datasets(
    db: AsyncSession,
    project_id: int,
    dataset_types: list[DatasetType],
) -> list[str]:
    """Read labels from the JSONL files of the given Dataset types.
    Returns a flat list of label strings; rows without an extractable
    label are silently dropped."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(dataset_types),
        )
    )
    labels: list[str] = []
    for dataset in result.scalars():
        if not dataset.file_path:
            continue
        path = Path(dataset.file_path)
        if not path.exists():
            continue
        from app.services.dataset_service import _load_records_from_file
        for row in _load_records_from_file(path):
            label = _row_to_label(row)
            if label is not None:
                labels.append(label)
    return labels


async def _train_eval_label_kl_signal(
    db: AsyncSession, project_id: int, recipe_id: str,
) -> dict[str, Any]:
    """Classification only. KL between train + eval label distributions.

    Degrades to ``ok`` for non-classification recipes (no labels to
    compare) and for projects below the minimum sample sizes.
    """
    try:
        from app.services.recipe_service import get_recipe
        recipe = get_recipe(recipe_id)
        task_profile = getattr(recipe, "task_profile", None) if recipe else None
    except Exception:
        task_profile = None

    if task_profile != "classification":
        return _make_signal(
            id="eval_gaps.train_eval_label_kl_high",
            severity="ok",
            headline=(
                "Train/eval label-KL only applies to classification "
                "recipes — skipping for this task."
            ),
            context={"task_profile": task_profile},
        )

    train_labels = await _load_labels_from_datasets(
        db, project_id,
        [DatasetType.TRAIN, DatasetType.CLEANED, DatasetType.SYNTHETIC],
    )
    eval_labels = await _load_labels_from_datasets(
        db, project_id,
        [DatasetType.GOLD_DEV, DatasetType.GOLD_TEST],
    )

    n_train = len(train_labels)
    n_eval = len(eval_labels)

    if n_train < KL_MIN_TRAIN_ROWS or n_eval < KL_MIN_EVAL_ROWS:
        return _make_signal(
            id="eval_gaps.train_eval_label_kl_high",
            severity="ok",
            headline=(
                f"Not enough labelled rows to measure train/eval-KL "
                f"({n_train} train, {n_eval} eval; need ≥ "
                f"{KL_MIN_TRAIN_ROWS}/{KL_MIN_EVAL_ROWS})."
            ),
            context={
                "train_count": n_train,
                "eval_count": n_eval,
                "skipped_reason": "below minimum sample sizes",
            },
        )

    train_counts = Counter(train_labels)
    eval_counts = Counter(eval_labels)
    train_dist = {k: v / n_train for k, v in train_counts.items()}
    eval_dist = {k: v / n_eval for k, v in eval_counts.items()}
    # KL(eval || train): does the eval distribution diverge from the
    # train distribution? The direction matters — we ask "is eval an
    # honest sample of what the model was taught?", not the reverse.
    kl = _kl_divergence(eval_dist, train_dist)

    if kl < KL_WARN:
        return _make_signal(
            id="eval_gaps.train_eval_label_kl_high",
            severity="ok",
            headline=(
                f"Train/eval label-KL = {kl:.3f} nats — distributions "
                f"agree."
            ),
            context={
                "train_count": n_train,
                "eval_count": n_eval,
                "kl_nats": round(kl, 4),
                "train_classes": len(train_counts),
                "eval_classes": len(eval_counts),
            },
        )

    severity: Severity = "block" if kl >= KL_BLOCK else "warn"
    # Surface the class whose representation differs most as a
    # concrete hook for the headline.
    delta_per_class: list[tuple[str, float]] = []
    for label in set(train_dist) | set(eval_dist):
        delta_per_class.append((
            label,
            abs(train_dist.get(label, 0.0) - eval_dist.get(label, 0.0)),
        ))
    delta_per_class.sort(key=lambda x: x[1], reverse=True)
    biggest_delta = delta_per_class[0] if delta_per_class else None

    return _make_signal(
        id="eval_gaps.train_eval_label_kl_high",
        severity=severity,
        headline=(
            f"Train/eval label-KL = {kl:.3f} nats. "
            + (
                f"Biggest mismatch: {biggest_delta[0]} "
                f"({train_dist.get(biggest_delta[0], 0.0):.0%} train vs "
                f"{eval_dist.get(biggest_delta[0], 0.0):.0%} eval)."
                if biggest_delta else ""
            )
        ),
        suggested_action={
            "kind": "navigate",
            "label": "Open Data Studio splits tab",
            "target": "data-studio-splits",
            "params": {},
        },
        context={
            "train_count": n_train,
            "eval_count": n_eval,
            "kl_nats": round(kl, 4),
            "biggest_delta_label": (
                biggest_delta[0] if biggest_delta else None
            ),
            "biggest_delta_value": (
                round(biggest_delta[1], 4) if biggest_delta else None
            ),
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Public entry point.
# ─────────────────────────────────────────────────────────────────────


async def scan_eval_gaps(
    db: AsyncSession, project_id: int,
) -> dict[str, Any]:
    """Return the eval-side gap report for a project.

    Same outer shape as the training-config gap scanner. Single group
    in phase 3 (every signal is an eval-readiness concern).
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    signals: list[dict[str, Any]] = []
    recipe_id = _recipe_id_for(project)

    if not recipe_id:
        signals.append(_make_signal(
            id="eval_gaps.no_recipe_selected",
            severity="block",
            headline="No recipe selected — pick one to scan eval gaps.",
            suggested_action={
                "kind": "navigate",
                "label": "Open recipe picker",
                "target": "recipe-picker",
                "params": {},
            },
            context={},
        ))
    else:
        signals.append(await _archetype_coverage_signal(db, project.id, recipe_id))
        signals.append(await _regression_baseline_signal(db, project.id))
        signals.append(await _train_eval_label_kl_signal(db, project.id, recipe_id))

    group = {
        "id": "eval_gaps",
        "title": "Eval gaps",
        "subtitle": "Does the eval set honestly predict prod performance?",
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

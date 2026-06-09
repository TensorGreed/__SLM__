"""Quality-Lift phase 3, slice 1 — Unlabeled-pool uncertainty scoring.

Closes the train → eval → data loop. After a training run completes,
score the project's unlabeled label_rows against the freshly-trained
checkpoint, rank by softmax entropy, and stash the top-K row ids onto
``exp.config._runtime["active_learning"]`` so the Coach nudge (slice
2) and the Data Studio card (slice 3) can render "label these next"
without re-scoring.

This is a *driver* on top of Epic F Phase 1's existing
``annotation/active_learning.score_classification_rows`` function —
not a re-implementation. The existing function already loads the
model + computes per-row entropy; we just wrap it in pool-sampling,
top-K selection, and best-effort error handling.

Design points (locked with the user 2026-06-09):
  - Storage: JSON on ``Experiment.config._runtime["active_learning"]``.
    Mirrors the auto-RAG build snapshot that already lives in
    ``_runtime``. No new table.
  - Sample cap: 2000 rows from the unlabeled pool. Scoring 50k
    rows on the GB10 just to surface the top-50 wastes 99% of
    inference budget; uniform sampling catches the top-K with
    near-zero coverage loss in practice.
  - Top-K default: 50. Tens-of-rows labeling sessions are ergonomic.
  - Task scope (slice 1): classification only. QA / seq2seq don't
    have a cheap entropy signal at the trained head; we skip with
    ``skipped_reason="no_logit_source"`` so the Coach nudge can
    fall silent. Span / preference will fold in during a follow-up
    slice via Epic F Phase 2's existing score_span_rows and
    score_preference_pair_rows helpers.
  - Multi-seed: a seed-group leader doesn't have its own
    checkpoint — its terminal status comes from the aggregator.
    The hook fires once from the leader's terminal transition,
    using the first succeeded child's checkpoint. Child-experiment
    terminal transitions skip the hook when ``seed_group_id`` is
    set (they're an internal implementation detail of the leader).

  Best-effort: every failure mode (no unlabeled rows, missing
  checkpoint, model load error, non-classification task) falls
  back to a snapshot with ``skipped_reason`` set rather than
  raising. The post-training hook wraps this in another try-block
  on top so the COMPLETED status transition is never blocked by
  scoring errors.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment
from app.models.label_job import LabelJob, LabelRow


DEFAULT_TOP_K = 50
DEFAULT_SAMPLE_CAP = 2000
SUPPORTED_TASK_TYPES = frozenset({"classification"})


async def _unlabeled_rows_for_project(
    db: AsyncSession,
    project_id: int,
) -> list[LabelRow]:
    """Walk the project's label_jobs and return all rows that are
    still unlabeled (no ``labeled_at``, no ``assigned_to``).

    Follows the same convention ``annotation_service.assign_next``
    uses for the unlabeled pool — see ``ANNOTATION_QUEUE_FILTER``.
    Includes rows from paused jobs intentionally: the user might be
    paused on review but still want active-learning to score them
    against a later experiment.
    """
    job_rows = await db.execute(
        select(LabelJob).where(LabelJob.project_id == project_id)
    )
    job_ids = [int(j.id) for j in job_rows.scalars().all()]
    if not job_ids:
        return []
    row_query = await db.execute(
        select(LabelRow).where(
            LabelRow.job_id.in_(job_ids),
            LabelRow.labeled_at.is_(None),
            LabelRow.assigned_to.is_(None),
        )
    )
    return list(row_query.scalars().all())


def _build_skipped_snapshot(
    *,
    model_experiment_id: int,
    skipped_reason: str,
    task_type: str | None = None,
    pool_size_total: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Empty-ish snapshot for the cases where scoring can't run.

    The Coach nudge (slice 2) gates on ``top_k.length > 0`` and the
    Data Studio card (slice 3) renders the ``skipped_reason`` text
    inline so the user can see *why* their queue is empty rather
    than wondering if scoring silently failed.
    """
    snapshot: dict[str, Any] = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "model_experiment_id": model_experiment_id,
        "task_type": task_type,
        "uncertainty_metric": "entropy",
        "pool_size_total": pool_size_total,
        "pool_size_scored": 0,
        "top_k": [],
        "skipped_reason": skipped_reason,
    }
    if extra:
        snapshot.update(extra)
    return snapshot


def _resolve_checkpoint_path(exp: Experiment) -> str | None:
    """Pick the directory that holds the saved adapter + tokenizer.

    ``output_dir`` is what the trainer stamps onto the experiment;
    the existing ``score_classification_rows`` helper accepts
    a single ``model_path`` so we just pass that through. Returns
    None when no path is set or it doesn't exist on disk —
    score_classification_rows would raise on load otherwise.
    """
    raw = (exp.output_dir or "").strip()
    if not raw:
        return None
    if not Path(raw).exists():
        return None
    return raw


async def _resolve_classification_label_space(
    db: AsyncSession,
    project_id: int,
) -> list[str] | None:
    """Read the project's label_jobs for a classification job and
    pull the allowed_labels out of its label_schema.

    score_classification_rows requires ``num_labels``; without a known
    label space we can't load the head. Returns None when no
    classification label_job exists.
    """
    rows = await db.execute(
        select(LabelJob).where(
            LabelJob.project_id == project_id,
            LabelJob.label_type == "classification",
        )
    )
    for job in rows.scalars().all():
        schema = job.label_schema if isinstance(job.label_schema, dict) else {}
        labels = schema.get("allowed_labels")
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels if isinstance(x, (str, int, float))]
    return None


def _task_type_from_experiment(exp: Experiment) -> str | None:
    """Read ``exp.config.task_type`` defensively. Returns lowercase
    string or None."""
    cfg = exp.config if isinstance(exp.config, dict) else {}
    raw = str(cfg.get("task_type") or "").strip().lower()
    return raw or None


async def score_unlabeled_pool_for_experiment(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    top_k: int = DEFAULT_TOP_K,
    sample_cap: int = DEFAULT_SAMPLE_CAP,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """Score the project's unlabeled label_rows by uncertainty
    against this experiment's trained checkpoint. Returns a snapshot
    dict ready to stamp onto ``exp.config._runtime["active_learning"]``.

    Always returns a dict — never raises. Every failure mode
    (non-classification task, no unlabeled rows, missing checkpoint,
    model load error) sets ``skipped_reason`` on the snapshot so the
    Coach + UI surfaces can show *why* the queue is empty.
    """
    exp_row = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    exp = exp_row.scalar_one_or_none()
    if exp is None:
        return _build_skipped_snapshot(
            model_experiment_id=experiment_id,
            skipped_reason="experiment_not_found",
        )

    task_type = _task_type_from_experiment(exp)
    if task_type not in SUPPORTED_TASK_TYPES:
        return _build_skipped_snapshot(
            model_experiment_id=experiment_id,
            skipped_reason="unsupported_task_type",
            task_type=task_type,
            extra={"supported_task_types": sorted(SUPPORTED_TASK_TYPES)},
        )

    unlabeled = await _unlabeled_rows_for_project(db, project_id)
    pool_size_total = len(unlabeled)
    if pool_size_total == 0:
        return _build_skipped_snapshot(
            model_experiment_id=experiment_id,
            skipped_reason="empty_pool",
            task_type=task_type,
        )

    label_space = await _resolve_classification_label_space(db, project_id)
    if not label_space:
        return _build_skipped_snapshot(
            model_experiment_id=experiment_id,
            skipped_reason="no_label_space_configured",
            task_type=task_type,
            pool_size_total=pool_size_total,
        )

    checkpoint_path = _resolve_checkpoint_path(exp)
    if checkpoint_path is None:
        return _build_skipped_snapshot(
            model_experiment_id=experiment_id,
            skipped_reason="checkpoint_path_missing",
            task_type=task_type,
            pool_size_total=pool_size_total,
        )

    # Sample down before model load — score_classification_rows is
    # per-row sequential, so the cost is linear in pool size.
    if pool_size_total > sample_cap:
        # Deterministic seed when caller provides one — useful for
        # reproducible tests + for the user re-running with the same
        # snapshot. Default to a per-experiment seed so the same
        # experiment's snapshot is reproducible across re-fires.
        seed = rng_seed if rng_seed is not None else int(experiment_id)
        rng = random.Random(seed)
        sampled = rng.sample(unlabeled, sample_cap)
    else:
        sampled = list(unlabeled)

    try:
        from app.services.annotation.active_learning import (
            score_classification_rows,
        )

        scores = score_classification_rows(
            sampled,
            model_path=checkpoint_path,
            label_space=label_space,
        )
    except Exception as exc:
        # Common cases: torch / transformers not installed, CUDA OOM,
        # adapter incompatible with base model. Surface to the snapshot
        # rather than tossing the whole post-training hook.
        return _build_skipped_snapshot(
            model_experiment_id=experiment_id,
            skipped_reason="scoring_failed",
            task_type=task_type,
            pool_size_total=pool_size_total,
            extra={
                "error": str(exc)[:512],
                "checkpoint_path": checkpoint_path,
            },
        )

    # Build (row, score) pairs. Entropy is "high = uncertain"; rows
    # with None scores (text-extraction failure) sort to the tail
    # so they never crowd out genuinely-uncertain rows.
    scored: list[tuple[LabelRow, float | None]] = list(zip(sampled, scores))
    scored.sort(
        key=lambda pair: (
            -1.0 if pair[1] is None else -float(pair[1])
        ),
    )

    top_entries: list[dict[str, Any]] = []
    for row, score in scored[: max(1, int(top_k))]:
        if score is None:
            continue
        top_entries.append({
            "label_row_id": int(row.id),
            "label_job_id": int(row.job_id),
            "uncertainty_score": round(float(score), 6),
        })

    return {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "model_experiment_id": experiment_id,
        "task_type": task_type,
        "uncertainty_metric": "entropy",
        "pool_size_total": pool_size_total,
        "pool_size_scored": len(sampled),
        "top_k": top_entries,
        "skipped_reason": None,
        "checkpoint_path": checkpoint_path,
        "label_space_size": len(label_space),
    }


async def stamp_snapshot_on_experiment(
    db: AsyncSession,
    *,
    experiment_id: int,
    snapshot: dict[str, Any],
) -> None:
    """Persist the snapshot onto ``exp.config._runtime["active_learning"]``.

    Lives next to the auto-RAG build result so the UI's existing
    ``_runtime`` reader picks it up without extra plumbing.
    """
    exp_row = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    exp = exp_row.scalar_one_or_none()
    if exp is None:
        return
    cfg = dict(exp.config or {})
    runtime = dict(cfg.get("_runtime") or {})
    runtime["active_learning"] = snapshot
    cfg["_runtime"] = runtime
    exp.config = cfg
    await db.commit()

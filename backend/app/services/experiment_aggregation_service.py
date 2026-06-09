"""Quality-Lift phase 1, slice 2 — seed-group EvalResult aggregation.

Multi-seed training fans out N child experiments (each with a unique
``seed_value``) that share a ``seed_group_id``. When the last child
reaches a terminal status the aggregator rolls every child's
``EvalResult`` rows into a single ``is_aggregate=True`` row per
``(dataset_name, eval_type)`` whose ``metrics`` payload carries
``{mean, std, min, max, n}`` per metric instead of a scalar. The gate
evaluator (slice 3) reads ``mean − std`` as the lower-bound value so
"vanity gates" (the 0.83 ± 0.05 case where one good seed papers over
the rest) fail loudly — per
`feedback_honest_metrics_no_vanity`.

Design choices already locked:
  - Failed-seed handling: aggregate over whatever succeeded with
    ``n=k`` where ``k < N``; leader status is COMPLETED with a warning
    badge as long as at least one child succeeded. All-fail → leader
    FAILED. (User signed off, 2026-06-08.)
  - Drill-down preserved: per-seed scalar ``EvalResult`` rows stay
    untouched alongside the new aggregate row. The picked-data
    provenance rule means the UI must let a user click through from
    "0.83 ± 0.04" to the three individual runs that produced it.
  - Pure variance stats live in ``compute_variance_stats`` (no DB,
    no async) so they are trivially unit-testable and reusable.

This service is the only writer that flips ``EvalResult.is_aggregate``
to True; the simulate / external runtimes always emit scalars.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment, ExperimentStatus


TERMINAL_STATUSES = frozenset({
    ExperimentStatus.COMPLETED,
    ExperimentStatus.FAILED,
    ExperimentStatus.CANCELLED,
})


def compute_variance_stats(per_seed_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Walk every metric key across N per-seed metric dicts and emit a
    ``{mean, std, min, max, n}`` summary per numeric leaf.

    Recurses into nested dicts so per-class metrics (the shape introduced
    by the Gap-#6 work — ``{"class_A": {"precision": .., "recall": ..,
    "f1": .., "support": ..}}``) aggregate at the leaf level rather than
    collapsing the structure. Non-numeric leaves (string labels, ``None``,
    booleans) are passed through unchanged from the first seed that
    reports them — there's no honest mean of "True".

    ``std`` uses the population formula (divide by n), not the sample
    formula (n−1). For the small N (≤8) in play here that's a deliberate
    choice: we're reporting the spread of *these* seeds, not estimating
    a population from a sample. With n=1, std is 0.0 (degenerate but
    well-defined).
    """
    if not per_seed_dicts:
        return {}

    keys: set[str] = set()
    for d in per_seed_dicts:
        if isinstance(d, dict):
            keys.update(d.keys())

    out: dict[str, Any] = {}
    for key in keys:
        present = [d[key] for d in per_seed_dicts if isinstance(d, dict) and key in d and d[key] is not None]
        if not present:
            continue
        sample = present[0]
        if isinstance(sample, bool):
            # Booleans are nominally numeric in Python; the variance
            # stats would be nonsense ("mean True = 0.66"). Pass through.
            out[key] = sample
        elif isinstance(sample, (int, float)):
            floats = [
                float(v) for v in present
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and not math.isnan(float(v))
            ]
            if floats:
                out[key] = _stat_block(floats)
        elif isinstance(sample, dict):
            # Per-class / nested metrics — recurse, dropping non-dict
            # entries that shouldn't be there but might be in dirty data.
            nested = [v for v in present if isinstance(v, dict)]
            if nested:
                out[key] = compute_variance_stats(nested)
        else:
            out[key] = sample
    return out


def _stat_block(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        std = 0.0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": n,
    }


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    floats = [float(v) for v in values if v is not None]
    if not floats:
        return None
    return sum(floats) / len(floats)


async def _siblings_for_group(
    db: AsyncSession,
    seed_group_id: str,
) -> list[Experiment]:
    """Children in a seed group — excludes the leader (seed_value is None)."""
    rows = await db.execute(
        select(Experiment).where(
            Experiment.seed_group_id == seed_group_id,
            Experiment.seed_value.is_not(None),
        )
    )
    return list(rows.scalars().all())


async def _find_leader(
    db: AsyncSession,
    seed_group_id: str,
) -> Experiment | None:
    """The seed-group leader — same group_id, but seed_value IS NULL.

    The leader is the original experiment the user created; the
    fan-out wrote ``seed_group_id`` on it but kept ``seed_value``
    NULL to distinguish it from children.
    """
    rows = await db.execute(
        select(Experiment).where(
            Experiment.seed_group_id == seed_group_id,
            Experiment.seed_value.is_(None),
        )
    )
    return rows.scalars().first()


async def _eval_results_for(
    db: AsyncSession,
    experiment_id: int,
) -> list[EvalResult]:
    rows = await db.execute(
        select(EvalResult).where(
            EvalResult.experiment_id == experiment_id,
            EvalResult.is_aggregate.is_(False),
        )
    )
    return list(rows.scalars().all())


async def _existing_aggregate_keys(
    db: AsyncSession,
    leader_id: int,
    seed_group_id: str,
) -> set[tuple[str, str]]:
    """Return the (dataset, eval_type) pairs that already have an
    aggregate row attached to the leader. Used to make the aggregator
    idempotent — a re-trigger (race between two children's terminal
    transitions) must not insert duplicate rows.
    """
    rows = await db.execute(
        select(EvalResult).where(
            EvalResult.experiment_id == leader_id,
            EvalResult.seed_group_id == seed_group_id,
            EvalResult.is_aggregate.is_(True),
        )
    )
    return {(er.dataset_name, er.eval_type) for er in rows.scalars().all()}


async def maybe_aggregate_seed_group(
    db: AsyncSession,
    experiment_id: int,
) -> dict[str, Any] | None:
    """Called from each runtime's terminal-status site (succeeded /
    failed / cancelled). If the experiment is a seed-group child and
    every sibling has also reached terminal, roll up the children's
    EvalResults into an aggregate row on the leader and mark the
    leader's status.

    Returns a small summary dict for logging, or None when the
    aggregation is not yet ready (some siblings still running) or
    when the experiment isn't a seed-group child.

    Safe to call concurrently from multiple child-finish hooks — the
    aggregate-row idempotency check rejects the loser of any race.
    """
    target_row = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    target = target_row.scalar_one_or_none()
    if target is None:
        return None
    if not target.seed_group_id or target.seed_value is None:
        return None  # not a seed-group child (leader or single-seed run)

    group_id = target.seed_group_id
    siblings = await _siblings_for_group(db, group_id)
    if not siblings:
        return None

    if not all(s.status in TERMINAL_STATUSES for s in siblings):
        return None  # at least one sibling still running

    leader = await _find_leader(db, group_id)
    if leader is None:
        # Defensive: should not happen — fan-out always writes the
        # leader. If it does, treat as "no aggregation possible" and
        # surface in logs.
        return {
            "seed_group_id": group_id,
            "error": "leader_not_found",
            "n_children": len(siblings),
        }

    succeeded = [s for s in siblings if s.status == ExperimentStatus.COMPLETED]
    failed = [s for s in siblings if s.status in (
        ExperimentStatus.FAILED, ExperimentStatus.CANCELLED
    )]

    # Aggregate over the successful children's EvalResults only — failed
    # seeds may have written partial / incoherent metrics and would skew
    # the mean. We surface the failure count on the leader's config so
    # the UI can render a "1 of 3 seeds failed" warning badge.
    aggregates_created: list[str] = []
    if succeeded:
        existing = await _existing_aggregate_keys(db, leader.id, group_id)
        per_pack: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        per_pack_pass: dict[tuple[str, str], list[float | None]] = defaultdict(list)
        per_pack_provenance: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for s in succeeded:
            ers = await _eval_results_for(db, s.id)
            for er in ers:
                key = (er.dataset_name, er.eval_type)
                per_pack[key].append(er.metrics or {})
                per_pack_pass[key].append(er.pass_rate)
                per_pack_provenance[key].append({
                    "experiment_id": int(s.id),
                    "seed_value": int(s.seed_value) if s.seed_value is not None else None,
                    "eval_result_id": int(er.id),
                    "pass_rate": er.pass_rate,
                })

        for key, metrics_list in per_pack.items():
            if key in existing:
                continue  # already aggregated by a prior race-winner
            dataset_name, eval_type = key
            agg = EvalResult(
                experiment_id=leader.id,
                dataset_name=dataset_name,
                eval_type=eval_type,
                metrics=compute_variance_stats(metrics_list),
                pass_rate=_mean_or_none(per_pack_pass[key]),
                is_aggregate=True,
                seed_group_id=group_id,
                details={
                    "per_seed": per_pack_provenance[key],
                    "n_succeeded": len(succeeded),
                    "n_failed": len(failed),
                    "n_total": len(siblings),
                },
            )
            db.add(agg)
            aggregates_created.append(f"{dataset_name}::{eval_type}")

    # Leader status: COMPLETED if at least one child succeeded, FAILED
    # if all failed. Cancellation counts as failure for this purpose;
    # the user can still drill into the canceled children.
    if succeeded:
        leader.status = ExperimentStatus.COMPLETED
    else:
        leader.status = ExperimentStatus.FAILED
    leader.completed_at = datetime.now(timezone.utc)

    # Quality-Lift phase 3 slice 1 — Active-learning scoring for
    # multi-seed runs fires once from here, using the first succeeded
    # child's checkpoint. The child-side runner hook skips when it
    # sees ``seed_value is not None``, so this is the single source of
    # truth for seed-group active-learning snapshots and the user
    # never sees N redundant ones. Borrow the leader's output_dir
    # from the first-succeeded child since the leader itself never
    # ran training; the unlabeled_pool scoring service reads
    # ``exp.output_dir`` to locate the checkpoint.
    if succeeded:
        first_child = sorted(succeeded, key=lambda s: int(s.id))[0]
        if first_child.output_dir and not leader.output_dir:
            leader.output_dir = first_child.output_dir

    # Stamp the seed-group summary onto the leader's config so the UI
    # / API don't need to re-query siblings to render the warning badge.
    leader_cfg = dict(leader.config or {})
    leader_cfg["_seed_group"] = {
        "group_id": group_id,
        "n_total": len(siblings),
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "child_experiment_ids": sorted(int(s.id) for s in siblings),
        "succeeded_seeds": sorted(
            int(s.seed_value) for s in succeeded if s.seed_value is not None
        ),
        "failed_seeds": sorted(
            int(s.seed_value) for s in failed if s.seed_value is not None
        ),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    }
    leader.config = leader_cfg

    await db.commit()

    # Active-learning scoring on the leader once status is COMPLETED.
    # Fire AFTER the commit so the leader's output_dir + COMPLETED
    # state are visible to the scoring service in its own transaction.
    # Best-effort — failures land on the snapshot as a skipped_reason
    # rather than rolling back the aggregation.
    if succeeded:
        try:
            from app.services.unlabeled_pool_scoring_service import (
                score_unlabeled_pool_for_experiment,
            )

            snapshot = await score_unlabeled_pool_for_experiment(
                db,
                project_id=int(leader.project_id),
                experiment_id=int(leader.id),
            )
            # Stamp onto _runtime["active_learning"] the same way the
            # runner-side hooks do, so slice 2 + 3 read a uniform path.
            leader_refresh = await db.execute(
                select(Experiment).where(Experiment.id == leader.id)
            )
            leader_row = leader_refresh.scalar_one_or_none()
            if leader_row is not None:
                cfg = dict(leader_row.config or {})
                runtime = dict(cfg.get("_runtime") or {})
                runtime["active_learning"] = snapshot
                cfg["_runtime"] = runtime
                leader_row.config = cfg
                await db.commit()
        except Exception as al_exc:
            print(
                f"[active_learning] seed_group_scoring_failed leader_id={leader.id}: {al_exc}",
                flush=True,
            )

    return {
        "seed_group_id": group_id,
        "leader_id": int(leader.id),
        "n_total": len(siblings),
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "aggregates_created": aggregates_created,
    }

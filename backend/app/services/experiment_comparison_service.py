"""Eval-aware experiment comparison service (E3).

Compares two experiments' latest eval results + configs side-by-side so
the user can see what changed and (when B regressed against A) one-click
roll back via the existing ``rerun_from_manifest`` flow with A's
config + dataset version.

Different surface from ``api/comparison.py``, which compares
training-loss trajectories only. This service is the post-eval
"did B actually do better than A?" view: metric deltas, failure-
cluster diff, config diff, regressed flag.

Public entry point: ``compare_experiments(db, project_id, exp_a_id,
exp_b_id)`` → JSON-serialisable dict (no SQLAlchemy objects leak).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment


# Config fields the comparison surface highlights explicitly.
# Anything else in the config dict still shows up under "other_changes"
# but these get a dedicated row in the diff table so the most common
# culprits stay at the top.
_PRIMARY_CONFIG_FIELDS: tuple[str, ...] = (
    "base_model",
    "training_mode",
    "learning_rate",
    "num_epochs",
    "batch_size",
    "max_seq_length",
    "lr_scheduler",
    "optimizer",
    "dataset_version_id",
    "training_runtime_id",
)


# Metrics where higher = better. Anything not in this set is treated
# as "lower = better" (loss-style). Used to decide ``direction`` on
# each metric delta + to pick the verdict winner.
_HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "pass_rate",
    "f1",
    "macro_f1",
    "accuracy",
    "exact_match",
    "precision",
    "recall",
    "rouge_l",
    "rougeL",
    "groundedness",
    "llm_judge_pass_rate",
    "safety_pass_rate",
    "tool_success_rate",
})


async def compare_experiments(
    db: AsyncSession,
    *,
    project_id: int,
    exp_a_id: int,
    exp_b_id: int,
) -> dict[str, Any]:
    """Side-by-side eval comparison.

    Raises ``ValueError`` when an experiment is missing or doesn't
    belong to the project; the API layer translates to 404.
    """
    exp_a = await _load_experiment(db, project_id, exp_a_id)
    exp_b = await _load_experiment(db, project_id, exp_b_id)

    eval_a = await _latest_eval_with_pass_rate(db, exp_a_id)
    eval_b = await _latest_eval_with_pass_rate(db, exp_b_id)

    metric_deltas = _diff_metrics(eval_a, eval_b)
    cluster_diff = await _diff_failure_clusters(
        db, eval_a_id=eval_a.id if eval_a else None,
        eval_b_id=eval_b.id if eval_b else None,
    )
    config_diff = _diff_configs(exp_a.config or {}, exp_b.config or {})

    pass_rate_a = eval_a.pass_rate if eval_a else None
    pass_rate_b = eval_b.pass_rate if eval_b else None
    winner, regressed = _decide_winner(pass_rate_a, pass_rate_b)

    return {
        "project_id": project_id,
        "a": _experiment_summary(exp_a, eval_a),
        "b": _experiment_summary(exp_b, eval_b),
        "metric_deltas": metric_deltas,
        "cluster_diff": cluster_diff,
        "config_diff": config_diff,
        "winner": winner,
        "regressed": regressed,
    }


async def _load_experiment(
    db: AsyncSession, project_id: int, experiment_id: int
) -> Experiment:
    exp = await db.get(Experiment, experiment_id)
    if exp is None or exp.project_id != project_id:
        raise ValueError(
            f"Experiment {experiment_id} not found in project {project_id}"
        )
    return exp


async def _latest_eval_with_pass_rate(
    db: AsyncSession, experiment_id: int
) -> EvalResult | None:
    """Pick the most recent EvalResult with a non-null pass_rate. If
    none has one, fall back to the most recent of any kind so the UI
    can still render metric values (deltas just won't have a
    ``primary_metric``)."""
    result = await db.execute(
        select(EvalResult)
        .where(
            EvalResult.experiment_id == experiment_id,
            EvalResult.pass_rate.is_not(None),
        )
        .order_by(EvalResult.created_at.desc(), EvalResult.id.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row is not None:
        return row
    # Fallback: any latest eval.
    result = await db.execute(
        select(EvalResult)
        .where(EvalResult.experiment_id == experiment_id)
        .order_by(EvalResult.created_at.desc(), EvalResult.id.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _experiment_summary(
    exp: Experiment, eval_result: EvalResult | None
) -> dict[str, Any]:
    return {
        "experiment_id": exp.id,
        "name": exp.name,
        "base_model": exp.base_model,
        "training_mode": exp.training_mode.value if hasattr(exp.training_mode, "value") else str(exp.training_mode),
        "status": exp.status.value if hasattr(exp.status, "value") else str(exp.status),
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        "eval_result_id": eval_result.id if eval_result else None,
        "eval_pass_rate": eval_result.pass_rate if eval_result else None,
        "eval_type": eval_result.eval_type if eval_result else None,
        "dataset_name": eval_result.dataset_name if eval_result else None,
        "metrics": dict(eval_result.metrics or {}) if eval_result else {},
    }


def _direction(metric_id: str, a: float | None, b: float | None) -> str:
    """Returns ``improved`` / ``regressed`` / ``unchanged`` / ``new``
    / ``removed`` based on whether higher is better for this metric."""
    if a is None and b is None:
        return "unchanged"
    if a is None:
        return "new"
    if b is None:
        return "removed"
    if a == b:
        return "unchanged"
    higher_is_better = metric_id.strip().lower() in _HIGHER_IS_BETTER
    delta = b - a
    if higher_is_better:
        return "improved" if delta > 0 else "regressed"
    # Loss-style metric — lower is better.
    return "improved" if delta < 0 else "regressed"


def _diff_metrics(
    eval_a: EvalResult | None, eval_b: EvalResult | None
) -> list[dict[str, Any]]:
    """Pairwise diff over the union of metric keys. Sorted with
    regressions first so the UI top-of-list flags the bad news."""
    a_metrics: dict[str, Any] = dict((eval_a.metrics or {}) if eval_a else {})
    b_metrics: dict[str, Any] = dict((eval_b.metrics or {}) if eval_b else {})
    keys = sorted(set(a_metrics) | set(b_metrics))
    rows: list[dict[str, Any]] = []
    for key in keys:
        a_raw = a_metrics.get(key)
        b_raw = b_metrics.get(key)
        a_val = _coerce_float(a_raw)
        b_val = _coerce_float(b_raw)
        direction = _direction(key, a_val, b_val)
        delta = (b_val - a_val) if (a_val is not None and b_val is not None) else None
        rows.append({
            "metric_id": key,
            "a_value": a_val,
            "b_value": b_val,
            "delta": round(delta, 6) if delta is not None else None,
            "direction": direction,
            "higher_is_better": key.strip().lower() in _HIGHER_IS_BETTER,
        })

    # Regressions to the top, then new/removed, then unchanged, then
    # improvements last. Within each bucket, alphabetical.
    order = {"regressed": 0, "new": 1, "removed": 2, "unchanged": 3, "improved": 4}
    rows.sort(key=lambda r: (order.get(r["direction"], 5), r["metric_id"]))
    return rows


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _diff_failure_clusters(
    db: AsyncSession,
    *,
    eval_a_id: int | None,
    eval_b_id: int | None,
) -> dict[str, Any]:
    """Compute the (reason_code, output_pattern) → count buckets for
    each eval result and diff them.

    Returns:
        {
          "only_in_a": [{reason_code, output_pattern, failure_count}, ...],
          "only_in_b": [...],
          "shared": [{reason_code, output_pattern, a_count, b_count, delta}, ...],
          "a_total": int,
          "b_total": int,
        }

    Calls the cluster service per side so the patterns + counts come
    from the same canonical implementation the FailureClustersPanel
    uses — guarantees consistency between the comparison page and the
    cluster drilldown.
    """
    a_clusters = await _clusters_for_eval(db, eval_a_id)
    b_clusters = await _clusters_for_eval(db, eval_b_id)

    a_keys = {(c["reason_code"], c["output_pattern"]): c["failure_count"] for c in a_clusters}
    b_keys = {(c["reason_code"], c["output_pattern"]): c["failure_count"] for c in b_clusters}

    only_in_a = []
    only_in_b = []
    shared = []
    for key in sorted(set(a_keys) | set(b_keys)):
        reason_code, output_pattern = key
        a_count = a_keys.get(key)
        b_count = b_keys.get(key)
        if a_count is not None and b_count is None:
            only_in_a.append({
                "reason_code": reason_code,
                "output_pattern": output_pattern,
                "failure_count": a_count,
            })
        elif b_count is not None and a_count is None:
            only_in_b.append({
                "reason_code": reason_code,
                "output_pattern": output_pattern,
                "failure_count": b_count,
            })
        else:
            shared.append({
                "reason_code": reason_code,
                "output_pattern": output_pattern,
                "a_count": a_count,
                "b_count": b_count,
                "delta": b_count - a_count,
            })

    # Order: biggest regressions first (only_in_b and shared with +delta).
    only_in_b.sort(key=lambda r: -r["failure_count"])
    only_in_a.sort(key=lambda r: -r["failure_count"])
    shared.sort(key=lambda r: -r["delta"])

    return {
        "a_total": sum(a_keys.values()) if a_keys else 0,
        "b_total": sum(b_keys.values()) if b_keys else 0,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "shared": shared,
    }


async def _clusters_for_eval(
    db: AsyncSession, eval_result_id: int | None
) -> list[dict[str, Any]]:
    """Return the cluster list for an eval result, or [] when missing
    or the cluster service errors. Best-effort — the comparison should
    still render even if cluster computation fails on one side."""
    if eval_result_id is None:
        return []
    try:
        from app.services.failure_cluster_service import (
            cluster_eval_result_failures,
        )

        payload = await cluster_eval_result_failures(
            db, eval_result_id=eval_result_id
        )
        return list(payload.get("clusters") or [])
    except Exception:
        return []


def _diff_configs(
    a_config: dict[str, Any], b_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Diff two experiment configs. Primary fields are surfaced even
    when unchanged so the user can see the "everything stayed the
    same" case explicitly. Non-primary changed fields land under
    ``other_changes`` (as separate rows with ``primary=False``)."""
    rows: list[dict[str, Any]] = []
    for field in _PRIMARY_CONFIG_FIELDS:
        a_val = a_config.get(field)
        b_val = b_config.get(field)
        rows.append({
            "field": field,
            "a_value": a_val,
            "b_value": b_val,
            "changed": a_val != b_val,
            "primary": True,
        })

    # Other changed keys — sorted alphabetical so the diff is
    # deterministic between calls.
    other_keys = sorted(
        (set(a_config) | set(b_config)) - set(_PRIMARY_CONFIG_FIELDS)
    )
    for field in other_keys:
        a_val = a_config.get(field)
        b_val = b_config.get(field)
        if a_val == b_val:
            continue  # Don't dump every unchanged config knob.
        rows.append({
            "field": field,
            "a_value": a_val,
            "b_value": b_val,
            "changed": True,
            "primary": False,
        })
    return rows


def _decide_winner(
    a_pass_rate: float | None, b_pass_rate: float | None
) -> tuple[str, bool]:
    """Returns (winner, regressed):
       - winner in {"a", "b", "tie", "unknown"}
       - regressed: True when B is worse than A by pass_rate
    """
    if a_pass_rate is None and b_pass_rate is None:
        return ("unknown", False)
    if a_pass_rate is None:
        return ("b", False)
    if b_pass_rate is None:
        return ("a", True)  # B failed to produce a pass_rate → regression
    if b_pass_rate > a_pass_rate:
        return ("b", False)
    if b_pass_rate < a_pass_rate:
        return ("a", True)
    return ("tie", False)

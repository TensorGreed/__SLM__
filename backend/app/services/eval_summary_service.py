"""The Eval tab's one default card (Wave 3b): is the fine-tuned model
better than its base model, by how much, and where does it still fail?

Composes the paired lift summary (sft_lift_summary_service — baseline for
the run's own base model, lower-is-better aware) with the run's latest
held-out result and up to five failing rows captured at eval time
(``details.failures_preview``).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment, ExperimentStatus

SUMMARY_FAILURES = 5

# Headline metric → which per-row score list (``details["row_scores"]``)
# it is the mean of. Other metrics (perplexity, macro-F1, …) aren't a mean of
# per-row values, so they get no row-level evidence.
_ROW_SCORE_FOR_METRIC: dict[str, str] = {
    "f1": "f1",
    "exact_match": "correct",
    "accuracy": "correct",
}


def _paired_row_scores(
    baseline: EvalResult | None, trained: EvalResult | None, metric_id: str
) -> tuple[list[float], list[float]] | None:
    """Per-row scores of the base model and the fine-tuned run on the rows
    BOTH evaluated, in matching order. None when either result predates
    per-row capture or the metric isn't a per-row mean."""
    score_key = _ROW_SCORE_FOR_METRIC.get(metric_id)
    if score_key is None or baseline is None or trained is None:
        return None

    def _scores(result: EvalResult) -> tuple[list[str], list[float]] | None:
        details = result.details if isinstance(result.details, dict) else {}
        row_scores = details.get("row_scores")
        if not isinstance(row_scores, dict):
            return None
        keys, values = row_scores.get("keys"), row_scores.get(score_key)
        if not isinstance(keys, list) or not isinstance(values, list) or len(keys) != len(values):
            return None
        return [str(k) for k in keys], [float(v) for v in values]

    base, fine = _scores(baseline), _scores(trained)
    if base is None or fine is None:
        return None
    if base[0] == fine[0]:
        return base[1], fine[1]
    # Different order / count: pair rows by key (first occurrence wins).
    base_by_key: dict[str, float] = {}
    for key, value in zip(*base):
        base_by_key.setdefault(key, value)
    before: list[float] = []
    after: list[float] = []
    seen: set[str] = set()
    for key, value in zip(*fine):
        if key in base_by_key and key not in seen:
            seen.add(key)
            before.append(base_by_key[key])
            after.append(value)
    return (before, after) if before else None


async def headline_lift_evidence(
    db: AsyncSession, lift: dict[str, Any], headline: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Row counts + noise verdict for the headline lift (see
    ``paired_comparison_stats``); None when it can't be computed."""
    from app.services.paired_comparison_stats import paired_difference_evidence

    if not headline:
        return None
    baseline_id = (lift.get("baseline") or {}).get("eval_result_id")
    trained_id = (lift.get("trained") or {}).get("eval_result_id")
    if not isinstance(baseline_id, int) or not isinstance(trained_id, int):
        return None
    baseline = await db.get(EvalResult, baseline_id)
    trained = await db.get(EvalResult, trained_id)
    paired = _paired_row_scores(baseline, trained, str(headline.get("metric_id") or ""))
    if paired is None:
        return None
    evidence = paired_difference_evidence(*paired)
    evidence["metric_id"] = headline.get("metric_id")
    return evidence


async def _latest_trained_experiment_id(db: AsyncSession, project_id: int) -> int | None:
    rows = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(Experiment.status == ExperimentStatus.COMPLETED)
        .order_by(desc(Experiment.id))
    )
    for exp in rows.scalars():
        cfg = exp.config if isinstance(exp.config, dict) else {}
        if cfg.get("is_baseline") is not True:
            return exp.id
    return None


def _failures_from(result: EvalResult) -> tuple[list[dict[str, Any]], int | None]:
    from app.services.evaluation_service import _prediction_failed

    details = result.details if isinstance(result.details, dict) else {}
    failures = details.get("failures_preview")
    if isinstance(failures, list):
        count = details.get("failed_count")
        return failures[:SUMMARY_FAILURES], int(count) if isinstance(count, int) else len(failures)
    # Results recorded before failures were captured: best effort from
    # the first-rows preview.
    preview = details.get("predictions_preview")
    if isinstance(preview, list):
        failed = [p for p in preview if isinstance(p, dict) and _prediction_failed(p)]
        return [
            {k: p.get(k) for k in ("prompt", "reference", "prediction", "row_exact_match", "row_f1")}
            for p in failed[:SUMMARY_FAILURES]
        ], None
    return [], None


async def build_eval_summary(
    db: AsyncSession,
    project_id: int,
    experiment_id: int | None = None,
) -> dict[str, Any]:
    from app.services.sft_lift_summary_service import compute_sft_lift_summary

    if experiment_id is None:
        experiment_id = await _latest_trained_experiment_id(db, project_id)
    if experiment_id is None:
        return {
            "project_id": project_id,
            "experiment_id": None,
            "verdict": "no_trained_run",
            "message": "Train a model first — then this card tells you whether it beats the base model.",
            "headline": None,
            "failures": [],
        }

    latest = (
        await db.execute(
            select(EvalResult)
            .where(EvalResult.experiment_id == experiment_id)
            .order_by(desc(EvalResult.id))
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is None:
        return {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "verdict": "not_evaluated",
            "message": (
                "This run hasn't been evaluated yet. The automatic check runs after "
                "training finishes (watch the bell), or evaluate it now."
            ),
            "headline": None,
            "failures": [],
        }

    lift = await compute_sft_lift_summary(db, project_id, experiment_id=experiment_id)
    lifts = lift.get("metric_lifts") or []
    headline = next((row for row in lifts if row.get("is_headline")), lifts[0] if lifts else None)
    if lift.get("status") == "ok" and headline:
        verdict = {"improved": "better", "regressed": "worse"}.get(headline["direction"], "same")
        message = None
    elif lift.get("status") == "no_baseline":
        verdict, message = "no_baseline", lift.get("message")
    else:
        verdict, message = "no_comparison", lift.get("message")

    evidence = await headline_lift_evidence(db, lift, headline) if lift.get("status") == "ok" else None

    failures, failed_count = _failures_from(latest)
    details = latest.details if isinstance(latest.details, dict) else {}
    metrics = latest.metrics if isinstance(latest.metrics, dict) else {}
    return {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "verdict": verdict,
        "message": message,
        "headline": headline,
        # How solid the headline lift is: rows better / worse / same vs the
        # base model and whether the change is within noise. None for results
        # recorded before per-row scores were kept, or non-row metrics.
        "evidence": evidence,
        "metric_lifts": lifts,
        "baseline": lift.get("baseline"),
        "trained": lift.get("trained"),
        "eval_result_id": latest.id,
        "eval_type": latest.eval_type,
        "evaluated_samples": metrics.get("evaluated_samples") or metrics.get("eval_documents"),
        "failures": failures,
        "failed_count": failed_count,
        "dataset_name": (details.get("dataset") or {}).get("name") if isinstance(details.get("dataset"), dict) else latest.dataset_name,
    }

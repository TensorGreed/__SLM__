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

    failures, failed_count = _failures_from(latest)
    details = latest.details if isinstance(latest.details, dict) else {}
    metrics = latest.metrics if isinstance(latest.metrics, dict) else {}
    return {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "verdict": verdict,
        "message": message,
        "headline": headline,
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

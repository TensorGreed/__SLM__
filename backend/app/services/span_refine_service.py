"""Refine weak entity types: read the latest span_set evaluation and name
the types the model is weakest on, so the span synthesiser can be pointed
at exactly those (the Synthetic tab's "Refine weak types" button, or the
API). General to any span-extraction project — the types are whatever the
project's answer key uses.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment

# A type below this F1 on the last held-out evaluation is "weak".
DEFAULT_WEAK_F1 = 0.8
# Suggested rows for a refine pass: enough for each weak type to gain a
# few dozen examples.
ROWS_PER_WEAK_TYPE = 25
MIN_REFINE_ROWS = 60
MAX_REFINE_ROWS = 300


async def latest_span_eval(db: AsyncSession, project_id: int) -> EvalResult | None:
    """The newest EvalResult of the project whose metrics are span_set."""
    rows = (
        await db.execute(
            select(EvalResult)
            .join(Experiment, Experiment.id == EvalResult.experiment_id)
            .where(Experiment.project_id == project_id)
            .order_by(EvalResult.id.desc())
            .limit(50)
        )
    ).scalars().all()
    for row in rows:
        metrics = row.metrics if isinstance(row.metrics, dict) else {}
        if metrics.get("scoring_mode") == "span_set" and isinstance(metrics.get("per_class"), dict):
            return row
    return None


def weak_types_from_metrics(metrics: dict[str, Any], threshold: float = DEFAULT_WEAK_F1) -> list[dict[str, Any]]:
    """``[{type, f1, precision, recall, tp, fp, fn}]`` for every type under
    ``threshold``, weakest first. Pure."""
    per_class = metrics.get("per_class") if isinstance(metrics, dict) else None
    out: list[dict[str, Any]] = []
    for name, stats in (per_class or {}).items():
        if not isinstance(stats, dict):
            continue
        try:
            f1 = float(stats.get("f1") or 0.0)
        except (TypeError, ValueError):
            f1 = 0.0
        if f1 < threshold:
            out.append({
                "type": str(name),
                "f1": round(f1, 4),
                "precision": stats.get("precision"),
                "recall": stats.get("recall"),
                "tp": stats.get("tp"),
                "fp": stats.get("fp"),
                "fn": stats.get("fn"),
            })
    out.sort(key=lambda item: (item["f1"], item["type"]))
    return out


async def weak_entity_types(db: AsyncSession, project_id: int, threshold: float = DEFAULT_WEAK_F1) -> dict[str, Any]:
    """What to refine: the weak types of the latest span_set evaluation,
    with the evaluation's provenance (eval result, run, dataset) and a
    suggested row count. ``types`` is empty when nothing is under the
    threshold or no span_set evaluation exists (``eval_result_id`` None)."""
    result = await latest_span_eval(db, project_id)
    if result is None:
        return {
            "project_id": project_id,
            "threshold": threshold,
            "eval_result_id": None,
            "experiment_id": None,
            "dataset_name": None,
            "overall_f1": None,
            "types": [],
            "all_types": [],
            "suggested_rows": 0,
        }
    from app.services.span_eval_sizing_service import size_types

    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    weak = weak_types_from_metrics(metrics, threshold)
    # Types the evaluation can't size a decision on: too few reference
    # spans to show a DECISION_GAP difference between runs. A refine pass
    # targets them too — more texts containing the type is the only way
    # to a bigger support (the new rows reach the key through a re-split
    # without keep_test + a re-key).
    sizing = size_types(metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {})
    under = [r for r in sizing["types"] if not r["enough"]]
    weak_names = [w["type"] for w in weak]
    to_refine = weak_names + [r["type"] for r in under if r["type"] not in weak_names]
    return {
        "project_id": project_id,
        "threshold": threshold,
        "eval_result_id": int(result.id),
        "experiment_id": int(result.experiment_id),
        "dataset_name": result.dataset_name,
        "evaluated_at": result.created_at.isoformat() if result.created_at else None,
        "overall_f1": metrics.get("f1"),
        "types": weak,
        "under_sampled": under,
        "sizing_gap": sizing["gap"],
        "types_to_refine": to_refine,
        "all_types": sorted((metrics.get("per_class") or {}).keys()),
        "suggested_rows": max(MIN_REFINE_ROWS, min(MAX_REFINE_ROWS, ROWS_PER_WEAK_TYPE * len(to_refine))) if to_refine else 0,
    }

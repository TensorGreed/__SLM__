"""Arc H — End-goal contract + progress ledger.

The user states a single goal at the project level ("ship a refund
classifier with F1 ≥ 0.85"); Coach Mode + Data Studio render a single
"% toward your stated goal" metric so the user always knows where
they are. This service computes the structured progress ledger that
both surfaces share.

Goal shape (persisted as ``Project.goal``):
    {
        "target_metric": "f1" | "pass_rate" | "accuracy",
        "target_threshold": float,        # 0.0–1.0
        "deadline": "YYYY-MM-DD" | null,
        "title": str | null,
        "stated_at": ISO8601 timestamp,
    }

Progress ledger components (each emits 0.0–1.0):
    - ``data_ready``         — has training rows + recipe mapping passes
    - ``gold_set``           — gold-set rows meet the threshold (default 100)
    - ``predicted_pass``     — trainability forecast vs. goal threshold
    - ``eval_pass_rate``     — latest EvalResult.pass_rate vs. goal threshold

Each component carries a ``concept_id`` (matches the frontend Term
registry) so the UI renders a "Learn more" link per row. The overall
progress is the equal-weight mean of *known* component values; pending
components don't drag the score down — they're surfaced separately as
``pending_components``. When every component hits ``met``, the ledger
status flips to ``ready_to_ship``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.experiment import EvalResult, Experiment
from app.models.project import Project


SupportedMetric = Literal["f1", "pass_rate", "accuracy"]
SUPPORTED_METRICS: tuple[str, ...] = ("f1", "pass_rate", "accuracy")
DEFAULT_GOAL: dict[str, Any] = {
    "target_metric": "f1",
    "target_threshold": 0.70,
    "deadline": None,
    "title": None,
}
# Minimum training rows we treat as "enough to even consider training".
# Falls under the data_ready component.
MIN_TRAINING_ROWS = 50
# Gold-set target the gold component scores against. Falls under the
# gold_set component. 100 is the threshold the gold-rows lesson on
# Academy recommends; the component scales linearly below.
DEFAULT_GOLD_TARGET = 100


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _status_for(value: float | None, threshold: float = 1.0) -> str:
    """Three-state status string used by the UI to colour-code rows.

    - ``met``       : component is fully satisfied
    - ``attention`` : component has *some* progress but is below the bar
    - ``pending``   : component has not yet been computed (no data yet)
    """
    if value is None:
        return "pending"
    if value >= threshold:
        return "met"
    if value > 0.0:
        return "attention"
    return "attention"


def _normalize_goal(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Return a sanitized goal dict. Falls back to DEFAULT_GOAL when raw
    is missing / malformed; never raises so the progress endpoint stays
    callable on brand-new projects with no goal stated."""
    if not isinstance(raw, dict):
        return {**DEFAULT_GOAL}
    target_metric = str(raw.get("target_metric") or "").lower().strip()
    if target_metric not in SUPPORTED_METRICS:
        target_metric = DEFAULT_GOAL["target_metric"]
    threshold_raw = raw.get("target_threshold")
    try:
        threshold = float(threshold_raw) if threshold_raw is not None else DEFAULT_GOAL["target_threshold"]
    except (TypeError, ValueError):
        threshold = DEFAULT_GOAL["target_threshold"]
    threshold = _clamp01(threshold)
    deadline = raw.get("deadline")
    if not isinstance(deadline, str) or not deadline.strip():
        deadline = None
    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        title = None
    return {
        "target_metric": target_metric,
        "target_threshold": threshold,
        "deadline": deadline,
        "title": title,
        "stated_at": raw.get("stated_at"),
    }


async def _data_ready_component(
    db: AsyncSession, project_id: int,
) -> tuple[float | None, str, list[str]]:
    """Return (value, detail_text, blockers) for the data_ready row."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                (DatasetType.TRAIN, DatasetType.CLEANED, DatasetType.SYNTHETIC),
            ),
        )
    )
    datasets = result.scalars().all()
    if not datasets:
        return (
            0.0,
            "No training data yet — import a dataset to start.",
            ["No training datasets imported."],
        )
    total_rows = sum(int(d.record_count or 0) for d in datasets)
    if total_rows < MIN_TRAINING_ROWS:
        ratio = _clamp01(total_rows / MIN_TRAINING_ROWS)
        return (
            ratio,
            f"{total_rows} training rows · {MIN_TRAINING_ROWS} recommended.",
            [f"Only {total_rows} training rows (need ≥{MIN_TRAINING_ROWS})."],
        )
    return (
        1.0,
        f"{total_rows} training rows ready.",
        [],
    )


async def _gold_set_component(
    db: AsyncSession, project_id: int,
) -> tuple[float | None, str, list[str]]:
    """Return (value, detail_text, blockers) for the gold_set row."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.GOLD_DEV,
        )
    )
    gold = result.scalar_one_or_none()
    if gold is None:
        return (
            0.0,
            "No Gold Set yet — promote trusted rows to start.",
            ["Gold Set is empty."],
        )
    rows = int(gold.record_count or 0)
    if rows >= DEFAULT_GOLD_TARGET:
        return (
            1.0,
            f"{rows} gold rows ready (≥ {DEFAULT_GOLD_TARGET}).",
            [],
        )
    ratio = _clamp01(rows / DEFAULT_GOLD_TARGET)
    return (
        ratio,
        f"{rows} gold rows · {DEFAULT_GOLD_TARGET} recommended.",
        [f"Only {rows} gold rows (need ≥{DEFAULT_GOLD_TARGET})."],
    )


def _predicted_pass_component(
    project: Project, target_threshold: float,
) -> tuple[float | None, str, list[str]]:
    """Read the cached trainability forecast and score it against the
    user's stated target threshold. ``None`` value means the forecast
    hasn't been computed yet — the UI shows "pending"."""
    cache = project.training_forecast_cache or {}
    forecast = cache.get("forecast") if isinstance(cache, dict) else None
    if not isinstance(forecast, dict):
        return (
            None,
            "Trainability forecast not computed yet — run a preflight check.",
            [],
        )
    pred = forecast.get("predicted_f1_confidence")
    if pred is None:
        pred = forecast.get("predicted_pass_probability")
    try:
        pred_float = float(pred) if pred is not None else None
    except (TypeError, ValueError):
        pred_float = None
    if pred_float is None:
        return (
            None,
            "Forecast cache has no probability field yet.",
            [],
        )
    # Score = predicted / target. >=1.0 means "predicted to clear the
    # bar"; <1.0 means "predicted to fall short, proportionally".
    if target_threshold <= 0:
        target_threshold = 0.01
    ratio = _clamp01(pred_float / target_threshold)
    if pred_float >= target_threshold:
        detail = f"Predicted {pred_float:.0%} (clears your {target_threshold:.0%} bar)."
        blockers: list[str] = []
    else:
        detail = f"Predicted {pred_float:.0%} (your bar is {target_threshold:.0%})."
        blockers = [
            f"Forecast predicts {pred_float:.0%}, below your {target_threshold:.0%} target.",
        ]
    return (ratio, detail, blockers)


async def _eval_pass_rate_component(
    db: AsyncSession, project_id: int, target_threshold: float,
) -> tuple[float | None, str, list[str]]:
    """Latest EvalResult.pass_rate for the project (across all experiments)
    scored against the target threshold. None when no eval has run yet."""
    result = await db.execute(
        select(EvalResult)
        .join(Experiment, EvalResult.experiment_id == Experiment.id)
        .where(Experiment.project_id == project_id)
        .order_by(desc(EvalResult.created_at))
        .limit(1)
    )
    latest = result.scalar_one_or_none()
    if latest is None or latest.pass_rate is None:
        return (
            None,
            "No eval has run yet.",
            [],
        )
    pass_rate = float(latest.pass_rate)
    if target_threshold <= 0:
        target_threshold = 0.01
    ratio = _clamp01(pass_rate / target_threshold)
    if pass_rate >= target_threshold:
        detail = f"Latest eval {pass_rate:.0%} (clears your {target_threshold:.0%} bar)."
        blockers: list[str] = []
    else:
        detail = f"Latest eval {pass_rate:.0%} (your bar is {target_threshold:.0%})."
        blockers = [
            f"Eval pass rate {pass_rate:.0%}, below your {target_threshold:.0%} target.",
        ]
    return (ratio, detail, blockers)


async def compute_progress(
    db: AsyncSession, project_id: int,
) -> dict[str, Any]:
    """Compute the full goal progress ledger for a project.

    Returns a dict matching the frontend GoalLedgerCard contract.
    Raises ValueError when the project doesn't exist.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    goal = _normalize_goal(project.goal)
    target_threshold = float(goal["target_threshold"])
    has_explicit_goal = isinstance(project.goal, dict) and project.goal

    data_value, data_detail, data_blockers = await _data_ready_component(db, project_id)
    gold_value, gold_detail, gold_blockers = await _gold_set_component(db, project_id)
    pred_value, pred_detail, pred_blockers = _predicted_pass_component(project, target_threshold)
    eval_value, eval_detail, eval_blockers = await _eval_pass_rate_component(
        db, project_id, target_threshold,
    )

    components = [
        {
            "id": "data_ready",
            "label": "Training data ready",
            "value": data_value,
            "status": _status_for(data_value),
            "detail": data_detail,
            "concept_id": "task_shape",
        },
        {
            "id": "gold_set",
            "label": "Gold Set ready",
            "value": gold_value,
            "status": _status_for(gold_value),
            "detail": gold_detail,
            "concept_id": "gold_set",
        },
        {
            "id": "predicted_pass",
            "label": "Predicted pass probability",
            "value": pred_value,
            "status": _status_for(pred_value),
            "detail": pred_detail,
            "concept_id": "predicted_f1_confidence",
        },
        {
            "id": "eval_pass_rate",
            "label": "Eval pass rate",
            "value": eval_value,
            "status": _status_for(eval_value),
            "detail": eval_detail,
            "concept_id": "pass_rate",
        },
    ]

    # Equal-weight mean over *known* values only. Pending components
    # surface separately so the user can see what's outstanding without
    # the score being dragged down by "not run yet".
    known_values = [c["value"] for c in components if c["value"] is not None]
    overall = (
        sum(known_values) / len(known_values)
        if known_values else 0.0
    )

    pending_components = [c["id"] for c in components if c["status"] == "pending"]
    blockers: list[str] = []
    for src in (data_blockers, gold_blockers, pred_blockers, eval_blockers):
        blockers.extend(src)

    if all(c["status"] == "met" for c in components):
        status: str = "ready_to_ship"
    elif any(c["status"] == "attention" for c in components):
        status = "in_progress"
    else:
        status = "blocked"

    return {
        "project_id": project_id,
        "goal": goal,
        "has_explicit_goal": bool(has_explicit_goal),
        "components": components,
        "overall_progress": round(overall, 4),
        "pending_components": pending_components,
        "blockers": blockers,
        "status": status,
    }


async def set_goal(
    db: AsyncSession,
    project_id: int,
    *,
    target_metric: str,
    target_threshold: float,
    deadline: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Persist the user's stated goal on the project. Returns the
    normalized goal dict; raises ValueError when the project doesn't
    exist or the metric isn't supported."""
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    metric = (target_metric or "").lower().strip()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported target_metric '{target_metric}'. "
            f"Use one of {SUPPORTED_METRICS}.",
        )
    try:
        threshold = _clamp01(float(target_threshold))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target_threshold must be a number: {exc}") from exc
    stated_at = datetime.now(timezone.utc).isoformat()
    goal: dict[str, Any] = {
        "target_metric": metric,
        "target_threshold": threshold,
        "deadline": deadline.strip() if isinstance(deadline, str) and deadline.strip() else None,
        "title": title.strip() if isinstance(title, str) and title.strip() else None,
        "stated_at": stated_at,
    }
    project.goal = goal
    await db.flush()
    return goal


async def clear_goal(
    db: AsyncSession, project_id: int,
) -> None:
    """Drop the project's stated goal. Idempotent."""
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    project.goal = None
    await db.flush()

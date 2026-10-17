"""Ship the best seed of a multi-seed group.

A seed group = a leader Experiment (``seed_group_id`` set, ``seed_value``
None) + its COMPLETED children (one per seed). The lift check reports the
group across seeds; shipping needs ONE model. This picks it by a
**validation** metric only — the numbers each child's ``training_report.json``
recorded on the validation split during training. Test examples, the answer
key and external benchmarks are never read: they are what the shipped model
is reported on, and picking on them would make that report optimistic.

Default metric per task type (the first one EVERY child has; if none is
shared by all, the first any child has):

* ``token_classification`` → validation span F1 with the run's tuned
  thresholds (``thresholds.json`` → ``tuned.micro_f1_after``, when tuned on
  the validation split) — the decoding the model is served with — else the
  trainer's argmax validation span F1 (``task_metrics.span_set.f1``)
* ``classification`` → validation macro-F1, else accuracy
* anything else → lowest validation loss (``final_eval_loss``)

Ties → the lower run id. A child without the metric is listed, never picked.

``pick_seed`` is pure; ``seed_group_pick`` gathers the reports;
``record_selection`` stamps the choice on the leader's config
(``config["seed_selection"]``) so any export of the picked child carries
"picked seed N of M by validation <metric> (values …)" — in the manifest,
the release notes and, for a span tagger, the model card.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment

SELECTION_KEY = "seed_selection"

# metric id → (label, higher_is_better)
VALIDATION_METRICS: dict[str, tuple[str, bool]] = {
    "span_f1_tuned": ("span F1 (tuned thresholds)", True),
    "span_f1": ("span F1", True),
    "macro_f1": ("macro-F1", True),
    "accuracy": ("accuracy", True),
    "eval_loss": ("loss", False),
}

_TASK_METRIC_ORDER: dict[str, list[str]] = {
    "token_classification": ["span_f1_tuned", "span_f1", "eval_loss"],
    "classification": ["macro_f1", "accuracy", "eval_loss"],
}
_DEFAULT_ORDER = ["eval_loss"]

NOT_USED_NOTE = "test examples, the answer key and external benchmarks were not used to pick"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value else None  # NaN → None


def validation_values(
    report: dict[str, Any] | None, fallback_eval_loss: Any = None, thresholds: dict[str, Any] | None = None,
) -> dict[str, float]:
    """The validation-split numbers a training report recorded, by metric id.
    Only validation keys are read (never a test or benchmark score).
    ``thresholds``: the run's thresholds.json — its ``tuned.micro_f1_after``
    counts as ``span_f1_tuned`` only when the tune ran on the validation
    split (a tune on another split would be a leak)."""
    report = report if isinstance(report, dict) else {}
    out: dict[str, float] = {}
    tuned = thresholds.get("tuned") if isinstance(thresholds, dict) and isinstance(thresholds.get("tuned"), dict) else {}
    if tuned.get("split", "val") == "val" and _num(tuned.get("micro_f1_after")) is not None:
        out["span_f1_tuned"] = _num(tuned.get("micro_f1_after"))  # type: ignore[assignment]
    task_metrics = report.get("task_metrics") if isinstance(report.get("task_metrics"), dict) else {}
    span = task_metrics.get("span_set") if isinstance(task_metrics.get("span_set"), dict) else {}
    if _num(span.get("f1")) is not None:
        out["span_f1"] = _num(span.get("f1"))  # type: ignore[assignment]
    if _num(report.get("final_eval_macro_f1")) is not None:
        out["macro_f1"] = _num(report.get("final_eval_macro_f1"))  # type: ignore[assignment]
    if _num(report.get("final_eval_accuracy")) is not None:
        out["accuracy"] = _num(report.get("final_eval_accuracy"))  # type: ignore[assignment]
    loss = _num(report.get("final_eval_loss"))
    if loss is None:
        loss = _num(fallback_eval_loss)
    if loss is not None:
        out["eval_loss"] = loss
    return out


def metric_order(task_type: str | None) -> list[str]:
    return list(_TASK_METRIC_ORDER.get(str(task_type or "").strip().lower(), _DEFAULT_ORDER))


def _fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 10 else f"{value:.2f}"


def pick_seed(
    children: list[dict[str, Any]],
    *,
    task_type: str | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    """Pure. ``children``: ``[{experiment_id, seed_value, values: {metric:
    value}}]``. Returns ``{metric_id, metric_label, higher_is_better,
    task_type, rule, picked_experiment_id, picked_seed_value, n_seeds,
    n_scored, children: [{experiment_id, seed_value, value, picked,
    skipped_reason}]}``. ``metric`` overrides the task default (must be a
    validation metric id)."""
    if metric is not None and metric not in VALIDATION_METRICS:
        raise ValueError(f"Unknown validation metric {metric!r}; one of {', '.join(VALIDATION_METRICS)}")
    if metric is None:
        order = metric_order(task_type)
        # A metric every child has, so no seed is excluded for not having
        # been tuned; else the first any child has.
        metric = next(
            (m for m in order if children and all(m in (c.get("values") or {}) for c in children)),
            next((m for m in order if any(m in (c.get("values") or {}) for c in children)), order[0]),
        )
    label, higher = VALIDATION_METRICS[metric]

    scored = [c for c in children if metric in (c.get("values") or {})]
    best: dict[str, Any] | None = None
    for c in scored:
        value = float(c["values"][metric])
        if best is None:
            best = c
            continue
        bv = float(best["values"][metric])
        better = value > bv if higher else value < bv
        if better or (value == bv and int(c["experiment_id"]) < int(best["experiment_id"])):
            best = c

    rows = []
    for c in children:
        value = (c.get("values") or {}).get(metric)
        rows.append({
            "experiment_id": int(c["experiment_id"]),
            "seed_value": c.get("seed_value"),
            "value": round(float(value), 4) if value is not None else None,
            "picked": best is not None and int(c["experiment_id"]) == int(best["experiment_id"]),
            "skipped_reason": None if value is not None else f"no validation {label} in its training report",
        })
    direction = "highest" if higher else "lowest"
    rule = f"{direction} validation {label}; ties go to the lower run id; {NOT_USED_NOTE}"
    return {
        "metric_id": metric,
        "metric_label": label,
        "higher_is_better": higher,
        "task_type": task_type,
        "rule": rule,
        "picked_experiment_id": int(best["experiment_id"]) if best else None,
        "picked_seed_value": best.get("seed_value") if best else None,
        "n_seeds": len(children),
        "n_scored": len(scored),
        "children": rows,
    }


def selection_text(selection: dict[str, Any] | None) -> str | None:
    """"Picked seed 43 (run #102) of 3 by validation span F1 (values: seed 42
    0.81, seed 43 0.84, seed 44 0.82); test examples … were not used to pick."""
    if not isinstance(selection, dict) or selection.get("picked_experiment_id") is None:
        return None
    values = ", ".join(
        f"seed {c.get('seed_value')} {_fmt(c['value'])}" if c.get("value") is not None
        else f"seed {c.get('seed_value')} –"
        for c in selection.get("children") or []
    )
    seed = selection.get("picked_seed_value")
    return (
        f"Picked seed {seed} (run #{selection['picked_experiment_id']}) of {selection.get('n_seeds')} "
        f"by validation {selection.get('metric_label')} "
        f"({'higher' if selection.get('higher_is_better') else 'lower'} is better; values: {values}); "
        f"{NOT_USED_NOTE}."
    )


# ── Gathering + recording ────────────────────────────────────────────────


def _read_report(exp: Experiment) -> dict[str, Any] | None:
    if not exp.output_dir:
        return None
    try:
        data = json.loads((Path(exp.output_dir).expanduser() / "training_report.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _read_thresholds(exp: Experiment) -> dict[str, Any] | None:
    """The child's ``model/thresholds.json`` (None when absent)."""
    if not exp.output_dir:
        return None
    from app.services.span_model_package_service import read_thresholds

    try:
        return read_thresholds(Path(exp.output_dir).expanduser() / "model")
    except Exception:  # noqa: BLE001
        return None


async def load_leader(db: AsyncSession, project_id: int, leader_id: int) -> Experiment:
    from app.services.post_training_eval_service import is_seed_group_leader

    leader = (
        await db.execute(
            select(Experiment).where(Experiment.id == leader_id, Experiment.project_id == project_id)
        )
    ).scalar_one_or_none()
    if leader is None:
        raise LookupError(f"Experiment {leader_id} not found in project {project_id}")
    if not is_seed_group_leader(leader):
        raise ValueError(f"Run #{leader_id} is not a multi-seed run (no seed group leader).")
    return leader


async def seed_group_pick(
    db: AsyncSession, project_id: int, leader_id: int, *, metric: str | None = None
) -> dict[str, Any]:
    """The pick for a group (not recorded) + the recorded selection, if any."""
    from app.services.post_training_eval_service import seed_group_children

    leader = await load_leader(db, project_id, leader_id)
    children = await seed_group_children(db, leader)
    if not children:
        raise ValueError(f"Seed group of run #{leader_id} has no completed child runs yet.")
    rows = []
    task_type = None
    for child in children:
        report = _read_report(child)
        if task_type is None and report and report.get("task_type"):
            task_type = str(report["task_type"])
        rows.append({
            "experiment_id": child.id,
            "seed_value": child.seed_value,
            "values": validation_values(report, child.final_eval_loss, _read_thresholds(child)),
        })
    if task_type is None:
        cfg = leader.config if isinstance(leader.config, dict) else {}
        task_type = cfg.get("task_type")
    pick = pick_seed(rows, task_type=task_type, metric=metric)
    pick["leader_experiment_id"] = leader.id
    pick["text"] = selection_text(pick)
    cfg = leader.config if isinstance(leader.config, dict) else {}
    pick["recorded"] = cfg.get(SELECTION_KEY)
    return pick


async def record_selection(db: AsyncSession, leader: Experiment, pick: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Stamp the selection on the leader's config (a new dict so SQLAlchemy
    sees the change)."""
    selection = {k: v for k, v in pick.items() if k not in {"recorded", "text"}}
    selection["picked_at"] = datetime.now(timezone.utc).isoformat()
    selection.update(extra)
    cfg = copy.deepcopy(leader.config) if isinstance(leader.config, dict) else {}
    cfg[SELECTION_KEY] = selection
    leader.config = cfg
    await db.flush()
    return selection


class SeedNotEvaluated(Exception):
    """The picked seed has no evaluation on record and the caller asked not
    to run one — the export would ship a card without evidence."""

    def __init__(self, project_id: int, experiment_id: int) -> None:
        self.project_id = project_id
        self.experiment_id = experiment_id
        super().__init__(
            f"Run #{experiment_id} has no evaluation on record, so its model card would carry no "
            f"evaluation or human-review evidence. Run the lift check first "
            f"(POST /api/projects/{project_id}/evaluation/summary/lift-check "
            f'{{"experiment_id": {experiment_id}}}, or "Evaluate this run" on the Eval tab), '
            "or ship with evaluate_if_missing=true to run it before exporting."
        )


async def has_eval_result(db: AsyncSession, experiment_id: int) -> bool:
    row = (
        await db.execute(select(EvalResult.id).where(EvalResult.experiment_id == experiment_id).limit(1))
    ).scalar_one_or_none()
    return row is not None


async def evaluate_before_ship(
    db: AsyncSession, project_id: int, child: Experiment, *, evaluate_if_missing: bool = True
) -> dict[str, Any] | None:
    """A span tagger's model card states the evaluation and human-review
    evidence from the picked child's EvalResults — and a seed child has none
    until it is lift-checked (the group's leader is what the watcher checks).
    When the picked child has no EvalResult: run the span lift check inline
    (test examples + answer key) and return its result, or raise
    ``SeedNotEvaluated`` when ``evaluate_if_missing`` is False. None when the
    child already has evidence or isn't a span run (other cards don't read
    EvalResults)."""
    from app.services.post_training_eval_service import is_token_classification, run_span_evaluation

    if not is_token_classification(child) or await has_eval_result(db, child.id):
        return None
    if not evaluate_if_missing:
        raise SeedNotEvaluated(project_id, child.id)
    result = await run_span_evaluation(db, project_id=project_id, experiment=child)
    return {"ran": True, **(result if isinstance(result, dict) else {})}


async def selection_for_experiment(db: AsyncSession, experiment: Experiment) -> dict[str, Any] | None:
    """The recorded selection when ``experiment`` is the seed its group's
    leader picked; None otherwise (a child exported without a pick, or a
    single run)."""
    if not experiment.seed_group_id or experiment.seed_value is None:
        return None
    leader = (
        await db.execute(
            select(Experiment)
            .where(Experiment.seed_group_id == experiment.seed_group_id)
            .where(Experiment.seed_value.is_(None))
            .where(Experiment.project_id == experiment.project_id)
            .order_by(Experiment.id)
            .limit(1)
        )
    ).scalar_one_or_none()
    if leader is None:
        return None
    cfg = leader.config if isinstance(leader.config, dict) else {}
    selection = cfg.get(SELECTION_KEY)
    if not isinstance(selection, dict) or selection.get("picked_experiment_id") != experiment.id:
        return None
    return {**selection, "text": selection_text(selection)}

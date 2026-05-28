"""Student-vs-teacher distillation comparison (Track 1, Epic A, slice 3).

Closes the KD loop: after distilling a teacher's captured logits into a small
student (slices 1–2), this answers "how much of the teacher's quality did the
student keep?" by comparing the student experiment's stored eval results
against a *teacher baseline run*'s stored eval results.

`quality_retained = student_metric / teacher_metric`, per shared metric and per
shared slice. No new model calls / no new LLM judge — this is a pure comparison
over EvalResult rows the existing eval handlers already wrote (same posture as
`sft_lift_summary_service`, which this mirrors with a ratio instead of a delta).

The teacher baseline run id is resolved in priority order:
  1. explicit argument (request query param),
  2. the student experiment's `config["teacher_baseline_run_id"]`,
  3. the project's resolved eval pack's `teacher_baseline_run_id`.

Status codes (never 4xx except a missing project/experiment, which the API
maps to 404):
  - `ok`                  — both evals present and share comparable metrics.
  - `no_teacher_baseline` — couldn't resolve a teacher run id from any source.
  - `no_student_eval`     — the student experiment has no eval result yet.
  - `no_teacher_eval`     — teacher run resolved but has no eval result (or is
                            not in this project).
  - `no_overlap`          — both have evals but no shared metric.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment
from app.models.project import Project
from app.services.sft_lift_summary_service import (
    _PREFERRED_HEADLINE_METRICS,
    _latest_eval_result,
    _normalize_metrics,
)

# Below this ratio the student is judged to have lost quality vs. the teacher.
_RETAINED_EPSILON = 1e-4


def _quality_retained(student: float, teacher: float) -> float | None:
    """student / teacher, with sane handling of a zero teacher metric.

    Returns ``None`` when the teacher scored 0 but the student scored > 0 —
    a ratio is undefined ("the student exceeds a zero baseline"); the UI renders
    that as "exceeds". When both are 0 the student trivially retained 100%.
    """
    if teacher == 0.0:
        return 1.0 if student == 0.0 else None
    return round(student / teacher, 4)


def _direction(student: float, teacher: float, retained: float | None) -> str:
    if retained is None:
        return "exceeds"
    if student + _RETAINED_EPSILON >= teacher:
        return "retained_or_better"
    return "regressed"


def _compute_metric_comparisons(
    student_metrics: dict[str, float],
    teacher_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    """One row per metric present in both, ordered headline-metrics-first."""
    shared = set(student_metrics) & set(teacher_metrics)
    if not shared:
        return []

    def _sort_key(metric_id: str) -> tuple[int, str]:
        idx = (
            _PREFERRED_HEADLINE_METRICS.index(metric_id)
            if metric_id in _PREFERRED_HEADLINE_METRICS
            else len(_PREFERRED_HEADLINE_METRICS)
        )
        return (idx, metric_id)

    rows: list[dict[str, Any]] = []
    for metric_id in sorted(shared, key=_sort_key):
        student = float(student_metrics[metric_id])
        teacher = float(teacher_metrics[metric_id])
        retained = _quality_retained(student, teacher)
        rows.append(
            {
                "metric_id": metric_id,
                "student_value": round(student, 4),
                "teacher_value": round(teacher, 4),
                "quality_retained": retained,
                "direction": _direction(student, teacher, retained),
                "is_headline": metric_id in _PREFERRED_HEADLINE_METRICS,
            }
        )
    return rows


def _slice_metrics(details: Any) -> dict[str, dict[str, float]]:
    """Pull per-slice metrics from EvalResult.details.

    Convention: ``details["slice_metrics"] = {slice_name: {metric_id: value}}``.
    Forward-compatible — eval handlers don't emit slices today, so absent data
    yields an empty map and per-slice comparison is simply skipped.
    """
    if not isinstance(details, dict):
        return {}
    raw = details.get("slice_metrics")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for slice_name, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue
        clean: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            clean[str(key)] = float(value)
        if clean:
            out[str(slice_name)] = clean
    return out


def _compute_slice_comparisons(
    student_details: Any,
    teacher_details: Any,
) -> list[dict[str, Any]]:
    student_slices = _slice_metrics(student_details)
    teacher_slices = _slice_metrics(teacher_details)
    rows: list[dict[str, Any]] = []
    for slice_name in sorted(set(student_slices) & set(teacher_slices)):
        s_metrics = student_slices[slice_name]
        t_metrics = teacher_slices[slice_name]
        for metric_id in sorted(set(s_metrics) & set(t_metrics)):
            student = s_metrics[metric_id]
            teacher = t_metrics[metric_id]
            retained = _quality_retained(student, teacher)
            rows.append(
                {
                    "slice": slice_name,
                    "metric_id": metric_id,
                    "student_value": round(student, 4),
                    "teacher_value": round(teacher, 4),
                    "quality_retained": retained,
                    "direction": _direction(student, teacher, retained),
                }
            )
    return rows


def _serialize(exp: Experiment, eval_result: EvalResult, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "experiment_id": exp.id,
        "experiment_name": exp.name,
        "base_model": exp.base_model,
        "eval_result_id": eval_result.id,
        "dataset_name": eval_result.dataset_name,
        "eval_type": eval_result.eval_type,
        "metrics": dict(metrics),
        "pass_rate": (
            float(eval_result.pass_rate) if eval_result.pass_rate is not None else None
        ),
    }


def _is_distillation_run(exp: Experiment) -> bool:
    """True when the experiment was trained with offline KD — used by the UI to
    decide whether the student-vs-teacher panel is relevant at all (it self-hides
    on non-distillation runs so it doesn't nag every project's Eval tab)."""
    cfg = exp.config or {}
    if not isinstance(cfg, dict):
        return False
    mode = str(cfg.get("training_mode") or "").strip().lower()
    return mode == "distillation" or bool(cfg.get("distillation_offline"))


async def _experiment_in_project(
    db: AsyncSession, project_id: int, experiment_id: int,
) -> Experiment | None:
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    return result.scalar_one_or_none()


async def _resolve_teacher_run_id(
    db: AsyncSession,
    project_id: int,
    student_exp: Experiment,
    explicit: int | None,
) -> int | None:
    if explicit is not None:
        return int(explicit)
    cfg = student_exp.config or {}
    if isinstance(cfg, dict) and cfg.get("teacher_baseline_run_id") is not None:
        try:
            return int(cfg["teacher_baseline_run_id"])
        except (TypeError, ValueError):
            pass
    # Last resort: the project's resolved eval pack may declare it.
    try:
        from app.services.evaluation_pack_service import resolve_project_evaluation_pack

        resolved = await resolve_project_evaluation_pack(db, project_id)
    except Exception:
        resolved = None
    for candidate in _iter_pack_dicts(resolved):
        value = candidate.get("teacher_baseline_run_id")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _iter_pack_dicts(resolved: Any):
    if isinstance(resolved, dict):
        yield resolved
        for key in ("pack", "active_pack"):
            nested = resolved.get(key)
            if isinstance(nested, dict):
                yield nested


async def compute_student_teacher_comparison(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    *,
    teacher_baseline_run_id: int | None = None,
) -> dict[str, Any]:
    """Compare a student experiment's eval results against a teacher baseline.

    Raises ``ValueError`` with ``project_not_found:`` / ``experiment_not_found:``
    prefixes (the API maps those to 404). Every other not-ready condition is a
    ``status`` the UI renders as a soft fallback.
    """
    project_q = await db.execute(select(Project).where(Project.id == project_id))
    if project_q.scalar_one_or_none() is None:
        raise ValueError(f"project_not_found:{project_id}")

    student_exp = await _experiment_in_project(db, project_id, experiment_id)
    if student_exp is None:
        raise ValueError(f"experiment_not_found:{experiment_id}")

    teacher_run_id = await _resolve_teacher_run_id(
        db, project_id, student_exp, teacher_baseline_run_id
    )

    base_payload: dict[str, Any] = {
        "project_id": project_id,
        "is_distillation_run": _is_distillation_run(student_exp),
        "teacher_baseline_run_id": teacher_run_id,
        "student": None,
        "teacher": None,
        "metric_comparisons": [],
        "slice_comparisons": [],
        "headline_quality_retained": None,
        "message": None,
    }

    if teacher_run_id is None:
        return {
            **base_payload,
            "status": "no_teacher_baseline",
            "message": (
                "No teacher baseline run set. Provide ?teacher_run_id=<exp>, set "
                "config.teacher_baseline_run_id on the student experiment, or add "
                "teacher_baseline_run_id to the project's eval pack."
            ),
        }

    student_er = await _latest_eval_result(db, student_exp.id)
    if student_er is None:
        return {
            **base_payload,
            "status": "no_student_eval",
            "message": "The student experiment has no eval result yet. Run evaluation first.",
        }
    student_metrics = _normalize_metrics(student_er.metrics)
    base_payload["student"] = _serialize(student_exp, student_er, student_metrics)

    teacher_exp = await _experiment_in_project(db, project_id, teacher_run_id)
    teacher_er = (
        await _latest_eval_result(db, teacher_exp.id) if teacher_exp is not None else None
    )
    if teacher_exp is None or teacher_er is None:
        return {
            **base_payload,
            "status": "no_teacher_eval",
            "message": (
                f"Teacher baseline run {teacher_run_id} has no eval result in this "
                "project. Evaluate the teacher run on the same eval set first."
            ),
        }
    teacher_metrics = _normalize_metrics(teacher_er.metrics)
    base_payload["teacher"] = _serialize(teacher_exp, teacher_er, teacher_metrics)

    metric_comparisons = _compute_metric_comparisons(student_metrics, teacher_metrics)
    slice_comparisons = _compute_slice_comparisons(student_er.details, teacher_er.details)
    base_payload["metric_comparisons"] = metric_comparisons
    base_payload["slice_comparisons"] = slice_comparisons

    if not metric_comparisons:
        return {
            **base_payload,
            "status": "no_overlap",
            "message": (
                "Student and teacher eval results share no comparable metrics. "
                "Re-run eval on both with the same task profile / eval set."
            ),
        }

    headline = next((r for r in metric_comparisons if r["is_headline"]), metric_comparisons[0])
    base_payload["headline_quality_retained"] = headline["quality_retained"]
    base_payload["status"] = "ok"
    return base_payload


__all__ = ["compute_student_teacher_comparison"]

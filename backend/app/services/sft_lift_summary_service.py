"""Did SFT help? — baseline vs trained eval lift summary (Theme 8 Epic 4).

Pulls the project's latest baseline experiment (`config.is_baseline == True`,
created by the Quickstart baseline tile in Theme 8 Epic 1) and the most
recent non-baseline completed experiment, looks at their most recent
EvalResult rows, and computes:

  - per-metric absolute + relative lift (intersection of metric keys)
  - gate status against the project's resolved eval pack, bucketed as
    cleared / still_failing / regressed / always_passed

No LLM calls — this is purely a comparison view on data the
existing eval plumbing already wrote. Mounts in the frontend on the
Eval tab below FailureClustersPanel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import (
    EvalResult,
    Experiment,
    ExperimentStatus,
)
from app.models.project import Project
from app.services.evaluation_pack_service import (
    DEFAULT_EVALUATION_PACK_ID,
    get_evaluation_pack,
)


# Metric ids we'll surface as headline lift bars when present in both
# baseline + trained metric dicts. The first match in this list anchors
# the "the model went from X to Y" headline (used by the UI summary
# line). Everything else still renders, just below the anchor.
_PREFERRED_HEADLINE_METRICS: tuple[str, ...] = (
    # code-remediation: the share of patches that apply, still parse, resolve the
    # finding and cite only the passages they were given (exists only for that mode).
    "remediation_pass_rate",
    # code-remediation, opt-in (``score_code_examples``): held-out task->code rows whose code parses,
    # passes the code checks, cites only given passages and the judge calls correct. Only heads the
    # lift when the split has no remediation rows (remediation_pass_rate is absent then).
    "code_example_pass_rate",
    # structured-plan: the share of plans that are valid JSON, validate against the schema, match
    # every configured pointer and (when a rubric runs) pass it (exists only for that mode).
    "structured_plan_pass_rate",
    # tool-calling: the share of calls that select the right tool, are valid against the tool catalog
    # (when there is one) and match the expected arguments (exists only for that profile).
    "tool_call_pass_rate",
    # "Is the answer right" (LLM judge, long-answer tasks) outranks token
    # overlap when both evals have it — see answer_judge_service.
    "judge_correct",
    "f1",
    "exact_match",
    "accuracy",
    "macro_f1",
    "precision",
    "recall",
    "groundedness",
    "tool_success_rate",
    "llm_judge_pass_rate",
    "pass_rate",
    "perplexity",
)

# Metrics where a DROP is the improvement (a lower perplexity means the
# model predicts the domain's text better).
_LOWER_IS_BETTER: frozenset[str] = frozenset({
    "perplexity",
    "mean_nll",
    "bits_per_byte",
    "hallucination_rate",
    "eval_loss",
    # code-remediation: patches that make the detector report a NEW finding.
    "new_findings_rate",
    # structured / finding-set / remediation: failure rates (a drop is the improvement).
    "malformed_rate",
    "clean_chunk_fp_rate",
    "fp_total",
    "fn_total",
    "invalid_finding_count",
    "malformed_count",
    "clean_chunk_fp_count",
})


# Metric ids we should hide from the lift surface — they're either
# noisy artifacts or schema-mismatch sentinels that have no meaningful
# baseline-vs-trained comparison.
_HIDDEN_METRICS: frozenset[str] = frozenset({
    "schema_mismatch",
    "samples",
    "total",
    "correct",
    # Perplexity-eval bookkeeping (sizes, not quality).
    "eval_tokens",
    "eval_documents",
    # code-remediation bookkeeping (sizes, not quality).
    "code_checkable_count",
    "detector_evaluated_count",
    "detector_error_count",
    "rows_with_passages",
    "skipped_non_remediation_rows",
    "detector_blind_count",
    "code_example_count",
    "code_example_decided_count",
    "code_example_correct",
    "code_example_checkable_count",
    "code_example_not_checkable_count",
    "code_example_judged_count",
    "code_example_judge_unscored_count",
    "code_example_rows_with_passages",
    # prediction-dependent denominators (rows whose patch applied): informational, not lift;
    # ``issue_resolved_overall_rate`` (every scored row) is the lift metric.
    "issue_resolved_rate",
    "new_findings_rate",
})


@dataclass
class _ExperimentEval:
    experiment: Experiment
    eval_result: EvalResult
    metrics: dict[str, float] = field(default_factory=dict)


async def _latest_baseline_experiment_eval(
    db: AsyncSession, project_id: int, base_model: str | None = None,
) -> _ExperimentEval | None:
    """Find the project's most recent baseline experiment + its latest
    eval result. Baselines are identified by `config.is_baseline=True`
    (set by `post_training_eval_service.find_or_create_baseline_experiment`).
    With ``base_model`` only that model's baseline qualifies — comparing a
    run against a different model's baseline isn't a lift."""
    rows = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .order_by(desc(Experiment.id))
    )
    candidates = list(rows.scalars().all())
    for exp in candidates:
        cfg = exp.config or {}
        if not isinstance(cfg, dict):
            continue
        if cfg.get("is_baseline") is True:
            if base_model and exp.base_model != base_model:
                continue
            er = await _latest_eval_result(db, exp.id, bar_only=True)
            if er is None:
                continue
            return _ExperimentEval(
                experiment=exp,
                eval_result=er,
                metrics=_normalize_metrics(er.metrics),
            )
    return None


async def _latest_trained_experiment_eval(
    db: AsyncSession, project_id: int, experiment_id: int | None = None,
) -> _ExperimentEval | None:
    """Find the project's most recent non-baseline experiment + its
    latest eval result. We don't require ExperimentStatus.COMPLETED
    because a long training that emitted an early eval is still
    legitimately "trained" for comparison purposes — but we DO skip
    rows that have no eval result yet."""
    query = (
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(Experiment.status != ExperimentStatus.FAILED)
        .where(Experiment.status != ExperimentStatus.CANCELLED)
    )
    if experiment_id is not None:
        query = query.where(Experiment.id == experiment_id)
    rows = await db.execute(query.order_by(desc(Experiment.id)))
    for exp in rows.scalars().all():
        cfg = exp.config or {}
        if isinstance(cfg, dict) and cfg.get("is_baseline") is True:
            continue
        er = await _latest_eval_result(db, exp.id)
        if er is None:
            continue
        return _ExperimentEval(
            experiment=exp,
            eval_result=er,
            metrics=_normalize_metrics(er.metrics),
        )
    return None


async def _latest_eval_result(
    db: AsyncSession, experiment_id: int, *, bar_only: bool = False,
) -> EvalResult | None:
    """Newest eval result. ``bar_only`` (a baseline): the one the
    strongest-baseline rule picked (``details.baseline_bar.chosen_here``,
    exactly one at a time) when there is one — not a newer unmarked eval or
    the decoding it passed over; else the newest not passed over."""
    result = await db.execute(
        select(EvalResult)
        .where(EvalResult.experiment_id == experiment_id)
        .order_by(desc(EvalResult.id))
        .limit(20 if bar_only else 1)
    )
    rows = list(result.scalars().all())
    if bar_only:
        chosen = next((er for er in rows if _baseline_bar(er).get("chosen_here") is True), None)
        if chosen is not None:
            return chosen
        rows = [er for er in rows if _baseline_bar(er).get("chosen_here") is not False]
    return rows[0] if rows else None


def _baseline_bar(er: EvalResult) -> dict[str, Any]:
    details = er.details if isinstance(er.details, dict) else {}
    bar = details.get("baseline_bar")
    return bar if isinstance(bar, dict) else {}


def _normalize_metrics(raw: Any) -> dict[str, float]:
    """Pull numeric metrics out of EvalResult.metrics and skip the
    schema-mismatch sentinels + hidden noise."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if key in _HIDDEN_METRICS:
            continue
        if isinstance(value, bool):
            # JSON `false` deserializes to Python `False` which `isinstance(_, (int, float))` is True for —
            # skip explicitly so we don't render a bool as a metric.
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


# Metric → which per-row score list (``EvalResult.details["row_scores"]``,
# written by ``run_heldout_evaluation``) it is the mean of. Other metrics
# (perplexity, macro-F1, …) aren't a mean of per-row values, so they get no
# row-level evidence.
_ROW_SCORE_FOR_METRIC: dict[str, str] = {
    "judge_correct": "judge_correct",
    "f1": "f1",
    "exact_match": "correct",
    "accuracy": "correct",
    "remediation_pass_rate": "remediation_pass",
    "code_example_pass_rate": "code_example_pass",
    "structured_plan_pass_rate": "structured_plan_pass",
    "tool_call_pass_rate": "tool_call_pass",
}


def _paired_row_scores(
    baseline: EvalResult | None, trained: EvalResult | None, metric_id: str
) -> tuple[list[float], list[float]] | None:
    """Per-row scores of the base model and the fine-tuned run on the rows
    BOTH evaluated, in matching order. None when either result predates
    per-row capture or the metric isn't a per-row mean."""
    from app.services.paired_comparison_stats import pair_row_scores

    score_key = _ROW_SCORE_FOR_METRIC.get(metric_id)
    if score_key is None or baseline is None or trained is None:
        return None

    def _row_scores(result: EvalResult) -> Any:
        details = result.details if isinstance(result.details, dict) else {}
        return details.get("row_scores")

    paired = pair_row_scores(_row_scores(baseline), _row_scores(trained), score_key)
    if paired is None:
        return None
    # Same keys in the same order pair positionally; otherwise by key
    # (first occurrence wins) and no shared scored row means no pairing.
    if paired["same_order"] or paired["before"]:
        return paired["before"], paired["after"]
    return None


def _attach_row_evidence(
    metric_lifts: list[dict[str, Any]], baseline: EvalResult, trained: EvalResult
) -> None:
    """Give each lift row its ``evidence`` (``paired_comparison_stats``):
    rows better / worse / same vs the base model and whether the change is
    more than noise. None where it can't be computed — a bare delta on a
    handful of rows must not read as a result."""
    from app.services.paired_comparison_stats import paired_difference_evidence

    for row in metric_lifts:
        metric_id = str(row.get("metric_id") or "")
        paired = _paired_row_scores(baseline, trained, metric_id)
        if paired is None:
            row["evidence"] = None
            continue
        evidence = paired_difference_evidence(*paired)
        evidence["metric_id"] = metric_id
        row["evidence"] = evidence


def _compute_metric_lifts(
    baseline_metrics: dict[str, float],
    trained_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    """Per-metric lift rows for every metric that exists in both
    baseline + trained dicts, ordered by `_PREFERRED_HEADLINE_METRICS`
    (familiar metrics first) then by absolute delta descending."""
    shared = sorted(set(baseline_metrics) & set(trained_metrics))
    if not shared:
        return []

    def _sort_key(metric_id: str) -> tuple[int, float]:
        preferred_idx = (
            _PREFERRED_HEADLINE_METRICS.index(metric_id)
            if metric_id in _PREFERRED_HEADLINE_METRICS
            else len(_PREFERRED_HEADLINE_METRICS)
        )
        delta = trained_metrics[metric_id] - baseline_metrics[metric_id]
        # negative because we want largest absolute deltas first
        # within the same preference bucket
        return (preferred_idx, -abs(delta))

    rows: list[dict[str, Any]] = []
    for metric_id in sorted(shared, key=_sort_key):
        baseline = float(baseline_metrics[metric_id])
        trained = float(trained_metrics[metric_id])
        absolute_delta = trained - baseline
        if baseline > 0:
            relative_delta_pct = round((absolute_delta / baseline) * 100.0, 1)
        elif trained > 0:
            relative_delta_pct = None  # going from zero is "infinite" lift; skip
        else:
            relative_delta_pct = 0.0
        signed = -absolute_delta if metric_id in _LOWER_IS_BETTER else absolute_delta
        if signed > 0.0001:
            direction = "improved"
        elif signed < -0.0001:
            direction = "regressed"
        else:
            direction = "unchanged"
        rows.append(
            {
                "metric_id": metric_id,
                "baseline_value": round(baseline, 4),
                "trained_value": round(trained, 4),
                "absolute_delta": round(absolute_delta, 4),
                "relative_delta_pct": relative_delta_pct,
                "direction": direction,
                "is_headline": metric_id in _PREFERRED_HEADLINE_METRICS,
            }
        )
    return rows


def _evaluate_gate(
    gate: dict[str, Any],
    baseline_metrics: dict[str, float],
    trained_metrics: dict[str, float],
) -> dict[str, Any] | None:
    """Compare a single gate against both baseline + trained metrics.
    Returns a status row for the UI, or None when the metric is
    missing from BOTH eval results (gate not applicable)."""
    metric_id = str(gate.get("metric_id") or "").strip()
    if not metric_id:
        return None
    baseline_value = baseline_metrics.get(metric_id)
    trained_value = trained_metrics.get(metric_id)
    if baseline_value is None and trained_value is None:
        return None

    threshold_raw = gate.get("threshold")
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        return None
    operator = str(gate.get("operator") or "gte").lower()

    def _passes(value: float | None) -> bool | None:
        if value is None:
            return None
        if operator == "lte":
            return value <= threshold
        # default gte
        return value >= threshold

    baseline_passes = _passes(baseline_value)
    trained_passes = _passes(trained_value)

    if baseline_passes is False and trained_passes is True:
        status = "cleared"
    elif baseline_passes is True and trained_passes is False:
        status = "regressed"
    elif baseline_passes is False and trained_passes is False:
        status = "still_failing"
    elif baseline_passes is True and trained_passes is True:
        status = "always_passed"
    else:
        status = "incomplete"

    delta_to_threshold = (
        None
        if trained_value is None
        else round(trained_value - threshold, 4)
    )

    return {
        "gate_id": str(gate.get("gate_id") or metric_id),
        "metric_id": metric_id,
        "threshold": threshold,
        "operator": operator,
        "required": bool(gate.get("required", True)),
        "baseline_value": (
            None if baseline_value is None else round(float(baseline_value), 4)
        ),
        "trained_value": (
            None if trained_value is None else round(float(trained_value), 4)
        ),
        "baseline_passes": baseline_passes,
        "trained_passes": trained_passes,
        "delta_to_threshold": delta_to_threshold,
        "status": status,
    }


def _gates_from_general_default_pack(
    task_profile_hint: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """Pull gates from `evalpack.general.default` for the task_profile
    that matches the trained experiment's recipe. We pin to the
    general pack (not the project's resolved pack) on purpose — the
    "did SFT help?" surface is a stable cross-project comparison
    against the platform's calibrated defaults, not against any
    domain-profile-derived dynamic gates the project might be using
    for its production promotion criteria. Mirrors the wording in
    ROADMAP-NEXT Theme 8 Epic 4."""
    pack = get_evaluation_pack(DEFAULT_EVALUATION_PACK_ID)
    if pack is None:
        return [], ""
    pack_id = str(pack.get("pack_id") or DEFAULT_EVALUATION_PACK_ID)
    task_specs = list(pack.get("task_specs") or [])
    chosen_spec: dict[str, Any] | None = None
    profile = (task_profile_hint or "").strip().lower()
    if profile:
        for spec in task_specs:
            if str(spec.get("task_profile") or "").lower() == profile:
                chosen_spec = spec
                break
    if chosen_spec is None:
        default_profile = str(pack.get("default_task_profile") or "").lower()
        for spec in task_specs:
            if str(spec.get("task_profile") or "").lower() == default_profile:
                chosen_spec = spec
                break
    if chosen_spec is None and task_specs:
        chosen_spec = task_specs[0]
    if chosen_spec is None:
        return [], pack_id

    gates = [g for g in (chosen_spec.get("gates") or []) if isinstance(g, dict)]
    return gates, pack_id


def _serialize_experiment(item: _ExperimentEval) -> dict[str, Any]:
    exp = item.experiment
    er = item.eval_result
    return {
        "experiment_id": exp.id,
        "experiment_name": exp.name,
        "base_model": exp.base_model,
        "training_mode": (
            exp.training_mode.value if exp.training_mode is not None else None
        ),
        "completed_at": (
            exp.completed_at.isoformat() if exp.completed_at else None
        ),
        "eval_result_id": er.id,
        "dataset_name": er.dataset_name,
        "eval_type": er.eval_type,
        "metrics": dict(item.metrics),
        "pass_rate": (
            float(er.pass_rate) if er.pass_rate is not None else None
        ),
        # A baseline: which decoding set the bar and both scores
        # (post_training_eval_service._baseline_bar); None otherwise.
        "decoding_bar": {k: v for k, v in _baseline_bar(er).items() if k != "chosen_here"} or None,
    }


async def compute_sft_lift_summary(
    db: AsyncSession,
    project_id: int,
    experiment_id: int | None = None,
) -> dict[str, Any]:
    """Build the "did SFT help?" summary payload for a project.

    Status codes:
      - `ok`              — baseline + trained both have eval results;
                            metric_lifts and gate_status are populated.
      - `no_baseline`     — no baseline experiment with an eval result.
      - `no_trained`      — no non-baseline experiment with an eval result.
      - `no_overlap`      — both exist but share no comparable metrics.
      - `project_not_found` — caller handles 404.
    """
    project_q = await db.execute(select(Project).where(Project.id == project_id))
    project = project_q.scalar_one_or_none()
    if project is None:
        raise ValueError(f"project_not_found:{project_id}")

    trained = await _latest_trained_experiment_eval(db, project_id, experiment_id)
    baseline = await _latest_baseline_experiment_eval(
        db, project_id, trained.experiment.base_model if trained else None
    )

    if baseline is None:
        return {
            "status": "no_baseline",
            "project_id": project_id,
            "message": (
                "No baseline for this base model yet. It's evaluated "
                "automatically after each real training run (watch the "
                "notification bell), or run the Quickstart 'Baseline "
                "(untrained)' tile."
            ),
            "baseline": None,
            "trained": _serialize_experiment(trained) if trained else None,
            "metric_lifts": [],
            "gate_status": [],
        }
    if trained is None:
        return {
            "status": "no_trained",
            "project_id": project_id,
            "message": (
                "No trained (non-baseline) eval results yet. Run "
                "'Train default config' and then evaluate it."
            ),
            "baseline": _serialize_experiment(baseline),
            "trained": None,
            "metric_lifts": [],
            "gate_status": [],
        }

    task_profile_hint = None
    selected = project.selected_recipe or {}
    if isinstance(selected, dict):
        task_profile_hint = str(selected.get("task_profile") or "").strip() or None

    metric_lifts = _compute_metric_lifts(baseline.metrics, trained.metrics)
    _attach_row_evidence(metric_lifts, baseline.eval_result, trained.eval_result)
    gates, eval_pack_id = _gates_from_general_default_pack(task_profile_hint)
    gate_status: list[dict[str, Any]] = []
    for gate in gates:
        row = _evaluate_gate(gate, baseline.metrics, trained.metrics)
        if row is not None:
            gate_status.append(row)

    if not metric_lifts:
        status = "no_overlap"
        message = (
            "Baseline and trained eval results don't share any comparable "
            "metrics. Re-run eval on both with the same task profile."
        )
    else:
        status = "ok"
        message = None

    return {
        "status": status,
        "project_id": project_id,
        "message": message,
        "baseline": _serialize_experiment(baseline),
        "trained": _serialize_experiment(trained),
        "metric_lifts": metric_lifts,
        "gate_status": gate_status,
        "eval_pack_id": eval_pack_id,
        "task_profile_used": task_profile_hint,
    }


__all__ = ["compute_sft_lift_summary"]

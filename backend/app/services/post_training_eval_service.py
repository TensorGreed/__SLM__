"""Automatic baseline-vs-fine-tuned lift eval after every real training run.

Before this, "did fine-tuning help?" only had an answer if the user found
the Quickstart baseline tile *and* ran held-out eval by hand. Now, when
the ``training_start`` watcher Job sees a real run complete, it spawns a
``post_training_lift_eval`` Job that:

  1. evaluates the un-fine-tuned base model on the held-out ``test``
     split (reusing a cached baseline result when the split hasn't
     changed since it was computed) — under neutral decoding and, when
     the model ships different settings, under its own too; the stronger
     is the bar (``_baseline_bar``, the strongest-baseline rule),
  2. evaluates the fine-tuned checkpoint on the same split with the same
     settings,
  3. returns the lift for that exact (base model, run) pair.

Simulated runs, baseline rows, seed-group children (their leader is checked
as a group — every seed, plus whether the lift holds across seeds) and runs without real
weights are skipped with a reason. Opt out per project via
``runtime_config["auto_lift_eval"] = False`` or globally via
``settings.AUTO_LIFT_EVAL_ENABLED``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.experiment import (
    EvalResult,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)
from app.models.project import Project

AUTO_LIFT_DATASET = "test"


def _lift_eval_limits() -> tuple[int, int]:
    """(max_samples, max_new_tokens) for the lift check — settings-backed so
    a CPU-only CI gate can shrink them."""
    return (
        max(1, int(settings.AUTO_LIFT_EVAL_MAX_SAMPLES)),
        max(1, int(settings.AUTO_LIFT_EVAL_MAX_NEW_TOKENS)),
    )
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt")


# ── Baseline experiment helpers (shared with the Quickstart tile) ──────


def short_model_name(model_id: str) -> str:
    """``HuggingFaceTB/SmolLM2-135M-Instruct`` → ``SmolLM2-135M-Instruct``."""
    if not model_id:
        return "base model"
    return model_id.rsplit("/", 1)[-1] or model_id


def eval_type_for_project(project: Project) -> str:
    """Informational eval_type label; the handler reads real metrics off
    ``prepared/manifest.json``. Documents-only projects (continued
    pretraining) are measured by held-out perplexity."""
    from app.services.continued_pretraining_policy import project_is_documents_only

    if project_is_documents_only(project):
        return "perplexity"
    snapshot = project.selected_recipe or {}
    scoring_mode = str(snapshot.get("scoring_mode") or "").strip()
    if scoring_mode in ("span_set", "finding_set"):
        return "f1"
    # ``remediation`` / ``structured_plan`` score a whole row: the pass rate is also ``exact_match``.
    return "exact_match"


async def find_or_create_baseline_experiment(
    db: AsyncSession,
    project_id: int,
    base_model: str,
    *,
    source: str = "quickstart.baseline-eval",
) -> Experiment:
    """The synthetic Baseline experiment for (project, base_model).
    ``output_dir`` stays empty — eval is called with
    ``model_path=base_model`` to bypass the artifact resolver."""
    name = f"Baseline · {short_model_name(base_model)}"[:255]
    result = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(Experiment.name == name)
        .limit(1)
    )
    existing = result.scalar_one_or_none()
    if existing is not None:
        return existing

    exp = Experiment(
        project_id=project_id,
        name=name,
        description=(
            "Synthetic baseline experiment — the un-fine-tuned base model "
            "evaluated against the project's gold/test split. Anchors "
            "post-SFT eval numbers so 'F1 0.65' has context."
        ),
        status=ExperimentStatus.COMPLETED,
        training_mode=TrainingMode.SFT,
        base_model=base_model,
        output_dir=None,
        config={"is_baseline": True, "source": source},
        final_train_loss=None,
        final_eval_loss=None,
    )
    db.add(exp)
    await db.flush()
    return exp


# ── Eligibility ────────────────────────────────────────────────────────


def _model_dir_has_weights(output_dir: str | None) -> bool:
    if not output_dir:
        return False
    root = Path(output_dir)
    model_dir = root / "model"
    candidate = model_dir if model_dir.is_dir() else root
    if not candidate.is_dir():
        return False
    has_config = (candidate / "config.json").exists() or (
        candidate / "adapter_config.json"
    ).exists()
    has_weights = any(
        p.suffix in _WEIGHT_SUFFIXES for p in candidate.iterdir() if p.is_file()
    )
    return has_config and has_weights


def auto_lift_skip_reason(
    exp: Experiment, project: Project | None, *, manual: bool = False
) -> str | None:
    """``None`` when the run should get an automatic lift eval, else a
    short machine-readable reason. ``manual`` (the user asked for the
    check) ignores the automatic-run opt-outs and evaluates a single seed
    of a seed group."""
    if not manual:
        if not settings.AUTO_LIFT_EVAL_ENABLED:
            return "disabled_globally"
        runtime_config = (project.runtime_config or {}) if project is not None else {}
        if isinstance(runtime_config, dict) and runtime_config.get("auto_lift_eval") is False:
            return "disabled_for_project"
    cfg = exp.config if isinstance(exp.config, dict) else {}
    if cfg.get("is_baseline") is True:
        return "baseline_experiment"
    if exp.status != ExperimentStatus.COMPLETED:
        return "not_completed"
    if exp.seed_group_id and exp.seed_value is not None and not manual:
        # The group is checked as a whole; a user evaluating one seed (to
        # ship it) asked for that seed.
        return "seed_group_child"
    runtime = cfg.get("_runtime") if isinstance(cfg.get("_runtime"), dict) else {}
    backend = str(runtime.get("backend") or "").lower()
    runtime_id = str(runtime.get("runtime_id") or "").lower()
    if backend == "simulate" or "simulate" in runtime_id:
        return "simulated_run"
    # A seed-group leader never trained itself: its children hold the
    # weights (each is resolved when evaluated; a group with no completed
    # child fails the check with a clear reason).
    if is_seed_group_leader(exp):
        return None
    if not _model_dir_has_weights(exp.output_dir):
        return "no_model_weights"
    return None


async def _test_split_mtime(project_id: int) -> datetime | None:
    path = Path(settings.DATA_DIR) / "projects" / str(project_id) / "prepared" / "test.jsonl"
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _baseline_judge_matches(result: EvalResult, judge_label: str | None) -> bool:
    """A cached base-model result pairs with the fine-tuned eval only if it
    was judged by the same judge (or the task isn't a long-answer one, so
    neither eval is judged). A pre-judge result is re-run once."""
    if judge_label is None:
        return True
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    if metrics.get("scoring_mode") == "structured_plan":
        # The rubric judge (if the project has a rubric) is stamped in the rubric block; the scoring
        # fingerprint carries the rest.
        rubric = metrics.get("rubric")
        if not isinstance(rubric, dict) or rubric.get("status") != "judged":
            return True
        return str(rubric.get("judge") or "") == judge_label
    # Opt-in code-example scoring stamps the judge that scored those rows; a base result scored by
    # another judge (or by none) can't pair. Absent block = flag off = unchanged behaviour.
    code_example_judge = metrics.get("code_example_judge")
    if isinstance(code_example_judge, dict) and str(code_example_judge.get("judge") or "") != judge_label:
        return False
    judge = metrics.get("judge")
    if not isinstance(judge, dict):
        return False
    if judge.get("skipped") == "not_long_answer_task":
        return True
    return str(judge.get("judge") or "") == judge_label


def _baseline_scoring_matches(result: EvalResult, fingerprint: str | None, tool_call_fingerprint: str | None = None) -> bool:
    """A cached remediation / structured-plan result stands only if it was scored under the current scoring config;
    a ``tool_call`` result only if its ``tool_call_fingerprint`` (handler version + catalog + flags + settings)
    equals the current one (a result without one is never reused)."""
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    if metrics.get("scoring_mode") == "tool_call":
        if tool_call_fingerprint is None or metrics.get("tool_call_fingerprint") != tool_call_fingerprint:
            return False
    if fingerprint is None:
        return True
    return fingerprint in (metrics.get("remediation_fingerprint"), metrics.get("structured_plan_fingerprint"))


def _baseline_decoding_matches(result: EvalResult, decoding: str = "neutral") -> bool:
    """A cached base-model result stands for one bar only if it was decoded
    that way (lift evals are greedy): neutral, or the model's own settings
    (``decoding="model"``). Results from before decoding was recorded
    inherited the model's generation_config without saying so — they are
    re-run once."""
    from app.services.decoding_settings import MODEL_DECODING, decoding_record, generation_kwargs, is_model_decoding

    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    inference = metrics.get("inference") if isinstance(metrics.get("inference"), dict) else {}
    if decoding == MODEL_DECODING:
        return is_model_decoding(inference.get("decoding"))
    return inference.get("decoding") == decoding_record(generation_kwargs(max_new_tokens=1, temperature=0.0))


# How many recent base-model results to look through for a reusable one: the
# two bars (neutral / the model's own decoding) interleave.
_REUSE_SCAN = 20


async def _reusable_baseline_result(
    db: AsyncSession,
    baseline_exp: Experiment,
    project_id: int,
    *,
    judge_label: str | None = None,
    decoding: str = "neutral",
    scoring_fingerprint: str | None = None,
    tool_call_fingerprint: str | None = None,
) -> EvalResult | None:
    """Newest baseline result on the test split decoded as ``decoding``, if
    newer than the split file (base models are deterministic under greedy
    decoding, so a fresh-enough result is exact, not approximate).
    ``scoring_fingerprint`` (remediation evals): the result must have been scored under the
    same detector / checks / retrieval / extra-field schema."""
    result = await db.execute(
        select(EvalResult)
        .where(EvalResult.experiment_id == baseline_exp.id)
        .where(EvalResult.dataset_name == AUTO_LIFT_DATASET)
        .order_by(desc(EvalResult.id))
        .limit(_REUSE_SCAN)
    )
    split_mtime = await _test_split_mtime(project_id)
    for candidate in result.scalars().all():
        if candidate.created_at is None:
            continue
        created = candidate.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if split_mtime is not None and created < split_mtime:
            return None  # this and every older result predate the split
        # Results recorded before per-row scores were kept can't be paired row by
        # row with the fine-tuned eval ("within noise?") — re-run the base once.
        details = candidate.details if isinstance(candidate.details, dict) else {}
        if not isinstance(details.get("row_scores"), dict):
            continue
        if not _baseline_judge_matches(candidate, judge_label):
            continue
        if not _baseline_decoding_matches(candidate, decoding):
            continue
        if not _baseline_scoring_matches(candidate, scoring_fingerprint, tool_call_fingerprint):
            continue
        return candidate
    return None


def choose_baseline_bar(neutral_metrics: Any, model_metrics: Any) -> dict[str, Any]:
    """Pure: which base-model result is the bar — the higher on the lift's
    headline metric (the first of ``_PREFERRED_HEADLINE_METRICS`` both
    have; lower is better for perplexity-like ones). A tie or no shared
    metric keeps neutral."""
    from app.services.sft_lift_summary_service import (
        _LOWER_IS_BETTER,
        _PREFERRED_HEADLINE_METRICS,
        _normalize_metrics,
    )

    neutral, model = _normalize_metrics(neutral_metrics), _normalize_metrics(model_metrics)
    metric_id = next((m for m in _PREFERRED_HEADLINE_METRICS if m in neutral and m in model), None)
    if metric_id is None:
        return {"chosen": "neutral", "metric_id": None, "neutral": None, "model": None}
    n, m = float(neutral[metric_id]), float(model[metric_id])
    model_better = m < n if metric_id in _LOWER_IS_BETTER else m > n
    return {
        "chosen": "model" if model_better else "neutral",
        "metric_id": metric_id,
        "neutral": round(n, 4),
        "model": round(m, 4),
    }


def _mark_bar(result: EvalResult, bar: dict[str, Any], chosen_here: bool) -> None:
    details = dict(result.details) if isinstance(result.details, dict) else {}
    details["baseline_bar"] = {**bar, "chosen_here": chosen_here}
    result.details = details  # reassigned so the JSON column is flushed


async def _mark_bar_results(db: AsyncSession, baseline_exp: Experiment, chosen: EvalResult, other: EvalResult | None, bar: dict[str, Any]) -> None:
    """Exactly one recent result of the baseline experiment carries
    ``chosen_here: True`` — the bar just picked. Earlier marks are cleared, so
    a stale bar (or a newer unmarked eval) never stands in for it in the lift
    summary (``sft_lift_summary_service._latest_eval_result``)."""
    rows = await db.execute(
        select(EvalResult).where(EvalResult.experiment_id == baseline_exp.id)
        .order_by(desc(EvalResult.id)).limit(_REUSE_SCAN)
    )
    for er in rows.scalars().all():
        if er.id in (chosen.id, getattr(other, "id", None)):
            continue
        prior = (er.details or {}).get("baseline_bar") if isinstance(er.details, dict) else None
        if isinstance(prior, dict) and prior.get("chosen_here"):
            _mark_bar(er, {k: v for k, v in prior.items() if k != "chosen_here"}, False)
    _mark_bar(chosen, bar, True)
    if other is not None:
        _mark_bar(other, bar, False)


async def _remediation_fingerprint_for(db: AsyncSession, project_id: int) -> str | None:
    """The scoring fingerprint a reusable remediation baseline must carry (None for any other
    project: nothing to match)."""
    try:
        from app.services.code_remediation_prompt import project_is_code_remediation
        from app.services.eval_task_handler_service import read_prepared_manifest
        from app.services.remediation_detector import remediation_scoring_fingerprint

        project = await db.get(Project, project_id)
        if project is None or not project_is_code_remediation(project):
            return None
        schema = read_prepared_manifest(project_id).get("output_schema")
        return await remediation_scoring_fingerprint(
            db, project_id, getattr(project, "runtime_config", None), schema if isinstance(schema, dict) else None
        )
    except Exception:  # noqa: BLE001 - cannot fingerprint: never reuse blindly
        return "unavailable"


async def _structured_plan_fingerprint_for(db: AsyncSession, project_id: int, judge_label: str | None) -> str | None:
    """The scoring fingerprint a reusable structured-plan baseline must carry (None for any other project)."""
    try:
        from app.services.structured_plan_prompt import project_is_structured_plan
        from app.services.structured_plan_scoring import structured_plan_scoring_fingerprint

        project = await db.get(Project, project_id)
        if project is None or not project_is_structured_plan(project):
            return None
        return structured_plan_scoring_fingerprint(getattr(project, "runtime_config", None), judge_label, project_id)
    except Exception:  # noqa: BLE001 - cannot fingerprint: never reuse blindly
        return "unavailable"


async def _tool_call_fingerprint_for(db: AsyncSession, project_id: int) -> str | None:
    """The current tool-call scoring fingerprint (a cached ``tool_call`` baseline must carry the same one)."""
    try:
        from app.services.tool_call_scoring import tool_call_fingerprint

        project = await db.get(Project, project_id)
        return tool_call_fingerprint(getattr(project, "runtime_config", None))
    except Exception:  # noqa: BLE001 - cannot fingerprint: never reuse blindly
        return None


async def _baseline_bar(
    db: AsyncSession,
    *,
    baseline_exp: Experiment,
    project_id: int,
    base_model: str,
    common: dict[str, Any],
    judge_label: str | None,
    report,
    fractions: tuple[float, float],
) -> tuple[EvalResult, bool, dict[str, Any]]:
    """The strongest-baseline rule: the base model scored with neutral
    decoding and, when its own ``generation_config`` sets a processor
    neutral decoding turns off (Qwen2.5-Instruct: repetition penalty 1.1),
    with those settings too. The better result is the bar — a fine-tune
    shouldn't get credit for a base decoded worse than its maker ships it
    (the Pub 17 reader's base + passages fell 0.551 → 0.331 without the
    penalty), nor for one decoded worse than BrewSLM decodes the run.
    Both results are kept and marked (``details.baseline_bar``) so the lift
    summary reads the chosen one. Returns (bar result, both reused, bar)."""
    import asyncio

    from app.services.decoding_settings import MODEL_DECODING, model_decoding_differs
    from app.services.evaluation_service import PERPLEXITY_EVAL_TYPE, run_heldout_evaluation

    scoring_fp = await _remediation_fingerprint_for(db, project_id) or await _structured_plan_fingerprint_for(
        db, project_id, judge_label
    )
    tool_fp = await _tool_call_fingerprint_for(db, project_id)
    neutral = await _reusable_baseline_result(
        db, baseline_exp, project_id, judge_label=judge_label, scoring_fingerprint=scoring_fp,
        tool_call_fingerprint=tool_fp,
    )
    reused = neutral is not None
    if neutral is None:
        await report(fractions[0], f"Evaluating base model {short_model_name(base_model)}")
        neutral = await run_heldout_evaluation(
            db=db, experiment_id=baseline_exp.id, model_path=base_model, **common
        )
        await db.commit()
    # Perplexity scores the text, not generated answers: decoding can't move it.
    perplexity = PERPLEXITY_EVAL_TYPE in (common.get("eval_type"), neutral.eval_type)
    own = None if perplexity else await asyncio.to_thread(model_decoding_differs, base_model)
    if own is None:
        # "no_other_settings": its config sets nothing neutral turns off (or
        # can't be read); perplexity: decoding doesn't apply.
        reason = "perplexity" if perplexity else "no_other_settings"
        bar = {"chosen": "neutral", "metric_id": None, "neutral": None, "model": None, "reason": reason}
        own_result = None
    else:
        own_result = await _reusable_baseline_result(
            db, baseline_exp, project_id, judge_label=judge_label, decoding=MODEL_DECODING,
            scoring_fingerprint=scoring_fp, tool_call_fingerprint=tool_fp,
        )
        if own_result is None:
            reused = False
            await report(
                fractions[1], f"Evaluating base model {short_model_name(base_model)} with its own decoding settings"
            )
            own_result = await run_heldout_evaluation(
                db=db, experiment_id=baseline_exp.id, model_path=base_model, decoding=MODEL_DECODING, **common
            )
            await db.commit()
        bar = choose_baseline_bar(neutral.metrics, own_result.metrics)
    bar.update({
        "model_settings": own,
        "neutral_eval_result_id": neutral.id,
        "model_eval_result_id": getattr(own_result, "id", None),
    })
    chosen = own_result if bar["chosen"] == "model" else neutral
    await _mark_bar_results(db, baseline_exp, chosen, neutral if chosen is own_result else own_result, bar)
    await db.commit()
    return chosen, reused, bar


async def _resolve_lift_judge(db: AsyncSession, project: Project):
    """The answer judge both evals of one lift check share (None → F1 only)."""
    from app.services.answer_judge_service import resolve_answer_judge

    try:
        return await resolve_answer_judge(db, project.id, project)
    except Exception:  # noqa: BLE001 — best-effort metric
        return None


# ── Runner ─────────────────────────────────────────────────────────────


# ── Token classification: the run's own span evaluation ──────────────────
#
# The "base model" of a token-classification run is an encoder with an
# untrained head, so a lift against it means nothing. "Evaluate this run"
# (and the automatic check after training) instead scores the run on the
# test examples AND, when the project has one, on the answer key written by
# a second model (GOLD_TEST, span_relabel_service) — the score to believe,
# because the test examples were labelled by the training-data teacher.

SPAN_EVAL_MAX_ROWS = 2000
SPAN_WEAKEST_TYPES = 5


def is_token_classification(exp: Experiment) -> bool:
    cfg = exp.config if isinstance(exp.config, dict) else {}
    return str(cfg.get("task_type") or "").strip().lower() == "token_classification"


def base_is_causal_lm(base_model: str | None) -> bool:
    """Whether the base checkpoint is a causal LM (can answer a prompt zero-
    shot). An encoder (DeBERTa, BERT…) can't — its untrained classifier head
    is no baseline. Unknown → True (keep the usual base-vs-fine-tuned check)."""
    if not base_model:
        return True
    try:
        from transformers import AutoConfig
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

        cfg = AutoConfig.from_pretrained(base_model)
        return getattr(cfg, "model_type", None) in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    except Exception:  # noqa: BLE001
        return True


def is_encoder_classification(exp: Experiment) -> bool:
    """A classification-head run on an encoder base model: evaluated on its
    own (no base-model baseline exists)."""
    cfg = exp.config if isinstance(exp.config, dict) else {}
    if str(cfg.get("task_type") or "").strip().lower() != "classification":
        return False
    return not base_is_causal_lm(exp.base_model)


def span_block(result: Any, type_regimes: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """The compact span report of one EvalResult: P / R / F1, rows, the
    weakest types (lowest F1 first, ties → more gold spans first) and,
    when the project tags its types with compliance regimes, micro
    P / R / F1 per regime (``by_regime``, weakest recall first)."""
    from app.services.compliance_bundles_service import regime_scores
    from app.services.span_eval_sizing_service import size_types

    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
    weakest = sorted(
        (
            {"type": t, "f1": v.get("f1"), "support": int((v.get("tp") or 0) + (v.get("fn") or 0))}
            for t, v in per_class.items()
            if isinstance(v, dict) and isinstance(v.get("f1"), (int, float))
        ),
        key=lambda row: (row["f1"], -row["support"]),
    )[:SPAN_WEAKEST_TYPES]
    return {
        "eval_result_id": int(result.id),
        "dataset_name": result.dataset_name,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "rows": metrics.get("total") or metrics.get("evaluated_samples"),
        "weakest_types": weakest,
        "by_regime": regime_scores(per_class, type_regimes) if type_regimes else [],
        # Per type: is there enough reference support to tell a
        # DECISION_GAP F1 difference between runs from noise?
        "sizing": size_types(per_class),
    }


async def project_type_regimes(db: AsyncSession, project_id: int) -> dict[str, list[str]]:
    """The project's compliance type→regimes map (``{}`` when none)."""
    from app.services.compliance_bundles_service import compliance_state

    project = await db.get(Project, project_id)
    return compliance_state(project.runtime_config if project is not None else None)["type_regimes"]


async def project_has_answer_key(db: AsyncSession, project_id: int) -> bool:
    from app.models.dataset import Dataset, DatasetType

    row = (
        await db.execute(
            select(Dataset.id)
            .where(Dataset.project_id == project_id, Dataset.dataset_type == DatasetType.GOLD_TEST)
            .where(Dataset.record_count > 0)
            .limit(1)
        )
    ).scalar_one_or_none()
    return row is not None


async def run_span_evaluation(
    db: AsyncSession, *, project_id: int, experiment: Experiment, report=None,
) -> dict[str, Any]:
    """Score a token-classification run on the test examples and, when
    the project has an answer key, on it too. Returns the Job result."""
    from app.services.evaluation_service import run_heldout_evaluation

    async def _report(fraction: float, message: str) -> None:
        if report is not None:
            await report(fraction, message)

    targets = [("test", "test")]
    if await project_has_answer_key(db, project_id):
        targets.append(("answer_key", "gold_test"))
    blocks: dict[str, Any] = {}
    type_regimes = await project_type_regimes(db, project_id)
    for index, (label, dataset_name) in enumerate(targets):
        await _report(0.1 + 0.8 * index / len(targets), f"Scoring run #{experiment.id} on the {'answer key' if label == 'answer_key' else 'test examples'}")
        result = await run_heldout_evaluation(
            db=db,
            project_id=project_id,
            experiment_id=experiment.id,
            dataset_name=dataset_name,
            eval_type="f1",
            max_samples=SPAN_EVAL_MAX_ROWS,
            max_new_tokens=8,
            temperature=0.0,
            model_path=None,
            judge_model=None,
            answer_judge_enabled=False,
        )
        await db.commit()
        blocks[label] = span_block(result, type_regimes)
    return {
        "kind": "span_eval",
        "experiment_id": experiment.id,
        "test": blocks.get("test"),
        "answer_key": blocks.get("answer_key"),
    }


async def run_post_training_lift_eval(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    progress=None,
    manual: bool = False,
) -> dict[str, Any]:
    """Baseline + fine-tuned held-out eval, then the paired lift summary.
    Commits after each eval so a failure in the second keeps the first.
    Raises when the run isn't eligible (the Job surfaces the reason)."""
    from app.services.evaluation_service import run_heldout_evaluation
    from app.services.sft_lift_summary_service import compute_sft_lift_summary

    async def _report(fraction: float, message: str) -> None:
        if progress is not None:
            await progress(fraction, message)

    exp = await db.get(Experiment, experiment_id)
    project = await db.get(Project, project_id)
    if exp is None or project is None:
        raise ValueError(f"Experiment {experiment_id} / project {project_id} not found")
    reason = auto_lift_skip_reason(exp, project, manual=manual)
    if reason is not None:
        raise ValueError(f"Automatic lift eval skipped: {reason}")
    from app.services.passage_reader_eval_service import is_passage_trained, run_passage_reader_evaluation

    if is_seed_group_leader(exp):
        if is_passage_trained(exp):
            raise ValueError(
                "A multi-seed run trained with retrieved passages isn't lift-checked as a group yet; "
                "evaluate each seed run on its own."
            )
        return await _run_seed_group_lift_eval(db, project=project, leader=exp, report=_report)
    if is_passage_trained(exp):
        # Trained to answer from the served passages: scored WITH them,
        # against the base model + the same passages (not the bare base).
        return await run_passage_reader_evaluation(db, project=project, experiment=exp, report=_report)
    if is_token_classification(exp):
        return await run_span_evaluation(db, project_id=project_id, experiment=exp, report=_report)
    if is_encoder_classification(exp):
        return await run_head_only_evaluation(db, project=project, experiment=exp, report=_report)

    eval_type = eval_type_for_project(project)
    max_samples, max_new_tokens = _lift_eval_limits()
    common = {
        "project_id": project_id,
        "dataset_name": AUTO_LIFT_DATASET,
        "eval_type": eval_type,
        "max_samples": max_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "judge_model": None,
    }
    answer_judge = await _resolve_lift_judge(db, project)
    common["answer_judge"] = answer_judge

    baseline_exp = await find_or_create_baseline_experiment(
        db, project_id, exp.base_model, source="post_training.auto_lift"
    )
    await db.commit()
    baseline_result, baseline_reused, baseline_bar = await _baseline_bar(
        db, baseline_exp=baseline_exp, project_id=project_id, base_model=exp.base_model, common=common,
        judge_label=answer_judge.label if answer_judge else None, report=_report, fractions=(0.1, 0.3),
    )

    await _report(0.55, f"Evaluating fine-tuned run #{experiment_id}")
    trained_result = await run_heldout_evaluation(
        db=db,
        experiment_id=experiment_id,
        model_path=None,
        **common,
    )
    await db.commit()

    await _report(0.95, "Computing lift")
    summary = await compute_sft_lift_summary(db, project_id, experiment_id=experiment_id)
    headline = next(
        (row for row in summary.get("metric_lifts") or [] if row.get("is_headline")),
        None,
    )
    # How solid the headline lift is (rows better / worse, within noise?) —
    # the bell must not announce "better than base" for a change that could
    # be chance. None when per-row scores aren't available.
    full = (headline or {}).get("evidence") if summary.get("status") == "ok" else None
    evidence = (
        {k: full.get(k) for k in ("verdict", "n", "better", "worse", "same")}
        if isinstance(full, dict)
        else None
    )
    return {
        "experiment_id": experiment_id,
        "base_model": exp.base_model,
        "evidence": evidence,
        "baseline_experiment_id": baseline_exp.id,
        "baseline_eval_result_id": getattr(baseline_result, "id", None),
        "baseline_reused": baseline_reused,
        "baseline_bar": baseline_bar,
        "trained_eval_result_id": getattr(trained_result, "id", None),
        "lift_status": summary.get("status"),
        "headline": headline,
    }


async def run_head_only_evaluation(db: AsyncSession, *, project: Project, experiment: Experiment, report) -> dict[str, Any]:
    """Score a classification-head run on an encoder against the test
    examples only — there is no zero-shot base model to compare with."""
    from app.services.evaluation_service import run_heldout_evaluation

    await report(0.2, f"Evaluating run #{experiment.id} (classifier head; no base-model baseline for an encoder)")
    max_samples, max_new_tokens = _lift_eval_limits()
    result = await run_heldout_evaluation(
        db=db, project_id=project.id, experiment_id=experiment.id, dataset_name=AUTO_LIFT_DATASET,
        eval_type=eval_type_for_project(project), max_samples=max_samples, max_new_tokens=max_new_tokens,
        temperature=0.0, judge_model=None, model_path=None,
    )
    await db.commit()
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    return {
        "kind": "head_eval",
        "experiment_id": experiment.id,
        "base_model": experiment.base_model,
        "trained_eval_result_id": result.id,
        "lift_status": "no_baseline",
        "reason": "encoder base model: an untrained classifier head is no baseline",
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "rows": metrics.get("evaluated_samples") or metrics.get("total"),
    }


def is_seed_group_leader(exp: Experiment) -> bool:
    """The marker run of a multi-seed group (it never trained itself; its
    N children did)."""
    return bool(exp.seed_group_id) and exp.seed_value is None


async def seed_group_children(db: AsyncSession, leader: Experiment) -> list[Experiment]:
    """The leader's COMPLETED children, in seed order."""
    rows = await db.execute(
        select(Experiment)
        .where(Experiment.seed_group_id == leader.seed_group_id)
        .where(Experiment.seed_value.is_not(None))
        .where(Experiment.status == ExperimentStatus.COMPLETED)
        .order_by(Experiment.seed_value, Experiment.id)
    )
    return list(rows.scalars())


async def _run_seed_group_lift_eval(
    db: AsyncSession, *, project: Project, leader: Experiment, report
) -> dict[str, Any]:
    """The multi-seed lift check: every completed child of the group is
    scored on the test split against the one (cached) base-model result;
    the result reports the headline per seed plus ``seed_evidence`` — whether
    the lift holds across seeds, not just across which rows were sampled.

    Before this, a seed-group leader was evaluated as if it were a single
    run (it borrows its first child's output_dir), so "3 seeds" reported the
    first seed only, under the leader's name.
    """
    from app.services.evaluation_service import run_heldout_evaluation
    from app.services.paired_comparison_stats import seed_spread_evidence
    from app.services.sft_lift_summary_service import compute_sft_lift_summary

    children = await seed_group_children(db, leader)
    if not children:
        raise ValueError("Automatic lift eval skipped: seed group has no completed child runs")

    eval_type = eval_type_for_project(project)
    max_samples, max_new_tokens = _lift_eval_limits()
    common = {
        "project_id": project.id,
        "dataset_name": AUTO_LIFT_DATASET,
        "eval_type": eval_type,
        "max_samples": max_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "judge_model": None,
    }
    answer_judge = await _resolve_lift_judge(db, project)
    common["answer_judge"] = answer_judge
    baseline_exp = await find_or_create_baseline_experiment(
        db, project.id, leader.base_model, source="post_training.auto_lift"
    )
    await db.commit()
    baseline_result, baseline_reused, baseline_bar = await _baseline_bar(
        db, baseline_exp=baseline_exp, project_id=project.id, base_model=leader.base_model, common=common,
        judge_label=answer_judge.label if answer_judge else None, report=report, fractions=(0.05, 0.1),
    )

    seeds: list[dict[str, Any]] = []
    for idx, child in enumerate(children):
        fraction = 0.15 + 0.75 * idx / len(children)
        await report(fraction, f"Evaluating seed {child.seed_value} (run #{child.id}, {idx + 1}/{len(children)})")
        trained_result = await run_heldout_evaluation(
            db=db, experiment_id=child.id, model_path=None, **common
        )
        await db.commit()
        summary = await compute_sft_lift_summary(db, project.id, experiment_id=child.id)
        headline = next(
            (row for row in summary.get("metric_lifts") or [] if row.get("is_headline")), None
        )
        seeds.append({
            "experiment_id": child.id,
            "seed_value": child.seed_value,
            "eval_result_id": getattr(trained_result, "id", None),
            "lift_status": summary.get("status"),
            "headline": headline,
        })

    await report(0.95, "Computing lift across seeds")
    scored = [s for s in seeds if s["headline"]]
    metric_ids = {s["headline"]["metric_id"] for s in scored}
    seed_evidence = None
    headline: dict[str, Any] | None = None
    if scored and len(metric_ids) == 1:
        metric_id = next(iter(metric_ids))
        baseline_value = float(scored[0]["headline"]["baseline_value"])
        trained_values = [float(s["headline"]["trained_value"]) for s in scored]
        seed_evidence = seed_spread_evidence(baseline_value, trained_values)
        seed_evidence["metric_id"] = metric_id
        mean = float(seed_evidence["mean"])
        delta = mean - baseline_value
        headline = {
            "metric_id": metric_id,
            "baseline_value": round(baseline_value, 4),
            "trained_value": round(mean, 4),
            "trained_std": round(float(seed_evidence["std"]), 4) if seed_evidence["std"] is not None else None,
            "absolute_delta": round(delta, 4),
            "relative_delta_pct": round(delta / baseline_value * 100.0, 1) if baseline_value > 0 else None,
            "direction": "improved" if delta > 0.0001 else "regressed" if delta < -0.0001 else "unchanged",
            "is_headline": True,
            "n_seeds": len(scored),
        }
    return {
        "experiment_id": leader.id,
        "base_model": leader.base_model,
        "n_seeds": len(scored),
        "seeds": seeds,
        "seed_evidence": seed_evidence,
        # Row-level evidence is per seed (``seeds[i].headline.evidence``);
        # the bell leads with the seed spread.
        "evidence": None,
        "baseline_experiment_id": baseline_exp.id,
        "baseline_eval_result_id": getattr(baseline_result, "id", None),
        "baseline_reused": baseline_reused,
        "baseline_bar": baseline_bar,
        "trained_eval_result_id": None,
        "lift_status": "ok" if headline else "no_overlap",
        "headline": headline,
    }


async def active_lift_job_id(db: AsyncSession, project_id: int, experiment_id: int) -> int | None:
    """Id of a QUEUED / RUNNING ``post_training_lift_eval`` Job for this run, else None."""
    from app.models.job import Job, JobStatus

    rows = await db.execute(
        select(Job)
        .where(
            Job.kind == "post_training_lift_eval",
            Job.project_id == project_id,
            Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
        .order_by(desc(Job.id))
    )
    for job in rows.scalars().all():
        params = job.params if isinstance(job.params, dict) else {}
        try:
            if int(params.get("experiment_id")) == int(experiment_id):
                return int(job.id)
        except (TypeError, ValueError):
            continue
    return None


async def start_post_training_lift_job(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    manual: bool = False,
) -> dict[str, Any]:
    """Spawn the lift-eval Job for a just-completed run, or return why not.
    Never raises — a failure here must not fail the training watcher.
    ``manual``: the user asked (re-run from the Eval tab) — the automatic
    opt-outs don't apply."""
    try:
        exp = await db.get(Experiment, experiment_id)
        project = await db.get(Project, project_id)
        if exp is None:
            return {"started": False, "skipped_reason": "experiment_missing"}
        # One lift check per run at a time: a second click (or the watcher
        # racing a manual re-score) must not queue a duplicate GPU eval.
        active = await active_lift_job_id(db, project_id, experiment_id)
        if active is not None:
            return {"started": False, "skipped_reason": "already_running", "job_id": active}
        reason = auto_lift_skip_reason(exp, project, manual=manual)
        if reason is not None:
            return {"started": False, "skipped_reason": reason}

        from app.database import async_session_factory
        from app.services.jobs_service import JobProgressHandle, start_job

        async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
            async def _progress(fraction: float, message: str) -> None:
                await handle.set_progress(fraction=fraction, message=message)

            async with async_session_factory() as runner_db:
                return await run_post_training_lift_eval(
                    runner_db,
                    project_id=project_id,
                    experiment_id=experiment_id,
                    progress=_progress,
                    manual=manual,
                )

        from app.services.passage_reader_eval_service import is_passage_trained

        run_label = f"run #{experiment_id}"
        if is_seed_group_leader(exp):
            n_children = len(await seed_group_children(db, exp))
            run_label = f"run #{experiment_id} ({n_children} seeds)"
        title = f"Did fine-tuning help? · {run_label} vs {short_model_name(exp.base_model)}"
        if is_passage_trained(exp) and not is_seed_group_leader(exp):
            # Scored with the served passages against the base model + the same passages.
            title = f"Does it read your passages better? · {run_label} + passages vs {short_model_name(exp.base_model)} + passages"
        job = await start_job(
            db,
            kind="post_training_lift_eval",
            title=title,
            runner=_runner,
            project_id=project_id,
            params={"experiment_id": experiment_id, "base_model": exp.base_model},
        )
        return {"started": True, "job_id": getattr(job, "id", None)}
    except Exception as exc:  # noqa: BLE001
        return {"started": False, "skipped_reason": f"error:{exc.__class__.__name__}: {exc}"}

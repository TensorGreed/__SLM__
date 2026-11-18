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


# ``max_samples`` meaning "every row of the prepared test split" (far above any split size).
WHOLE_SPLIT_MAX_SAMPLES = 10_000_000


def lift_max_samples(project: Project | None = None) -> int:
    """Rows of the prepared test split the lift check scores (the first N scorable rows, in file
    order). The project's ``runtime_config["auto_lift_max_samples"]`` (a positive int, or ``"all"`` =
    the whole split, as a manual held-out eval over the full split) wins; else
    ``max(1, AUTO_LIFT_EVAL_MAX_SAMPLES)`` -- unchanged (``0`` still means one row). The scope is part
    of the base-result cache key (``_baseline_scope_matches``)."""
    rc = getattr(project, "runtime_config", None) if project is not None else None
    raw = rc.get("auto_lift_max_samples") if isinstance(rc, dict) else None
    if isinstance(raw, str) and raw.strip().lower() == "all":
        return WHOLE_SPLIT_MAX_SAMPLES
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    return max(1, int(settings.AUTO_LIFT_EVAL_MAX_SAMPLES))


def _scope_trivial(result: EvalResult, max_samples: int) -> bool:
    """True when the scope check needs no row count (same recorded cap, or none recorded)."""
    details = result.details if isinstance(result.details, dict) else {}
    dataset = details.get("dataset") if isinstance(details.get("dataset"), dict) else {}
    recorded = dataset.get("max_samples")
    return not isinstance(recorded, int) or recorded == max_samples


def _baseline_scope_matches(result: EvalResult, max_samples: int | None, rows_available: int | None) -> bool:
    """A cached base result stands only if it scored the same rows the lift check would score now:
    the same effective cap ``min(max_samples, rows in the split)`` (rows are taken in file order, and
    the eval-input hash / mtime checks already pin the file). A result without a recorded cap
    (recorded before the cap was) stands only for the setting's default cap -- the only scope the
    lift check used then. ``max_samples=None``: no scope check (callers outside the lift check)."""
    if max_samples is None:
        return True
    details = result.details if isinstance(result.details, dict) else {}
    dataset = details.get("dataset") if isinstance(details.get("dataset"), dict) else {}
    recorded = dataset.get("max_samples")
    if not isinstance(recorded, int) or isinstance(recorded, bool):
        return max_samples == max(1, int(settings.AUTO_LIFT_EVAL_MAX_SAMPLES))
    if recorded == max_samples:
        return True
    rows = rows_available if isinstance(rows_available, int) else dataset.get("rows_available")
    if not isinstance(rows, int):
        return False
    return min(recorded, rows) == min(max_samples, rows)


def _lift_eval_limits(project: Project | None = None) -> tuple[int, int]:
    """(max_samples, max_new_tokens) for the lift check — settings-backed so
    a CPU-only CI gate can shrink them; ``max_samples`` per ``lift_max_samples``."""
    return (
        lift_max_samples(project),
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
    if is_simulated_run(exp):
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
        # No judge resolves now: the fine-tuned eval is not judged, so the base's judge blocks (complete
        # or not) never reach the comparison -- deliberately no completeness check here.
        return True
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    # A base result the judge left too incomplete (``snapshot_blocks_base_reuse``: the breaker stopped it,
    # it judged nothing, or more than 5 % of its rows were lost to an outage) is no bar: re-run it. A few
    # rows lost to a blip only drop out of the pairing (a re-judge without the GPU is a later phase).
    from app.services.judge_call_guard import snapshot_blocks_base_reuse

    if any(snapshot_blocks_base_reuse(metrics.get(k)) for k in ("judge", "code_example_judge", "rubric")):
        return False
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


def _baseline_scoring_matches(
    result: EvalResult,
    fingerprint: str | None,
    tool_call_fingerprint: str | None = None,
    sql_execution_fingerprint: str | None = None,
) -> bool:
    """A cached remediation / structured-plan result stands only if it was scored under the current scoring config;
    a ``span_set`` result only if it carries the current ``span_set_scoring_version``;
    a ``tool_call`` result only if its ``tool_call_fingerprint`` (handler version + catalog + flags + settings)
    equals the current one (a result without one is never reused);
    any result only if its ``sql_execution_fingerprint`` (opt-in SQL execution match: settings + the
    sha256 of each database file) equals the current one -- both absent while the scoring is off."""
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    if metrics.get("sql_execution_fingerprint") != sql_execution_fingerprint:
        return False
    if metrics.get("scoring_mode") == "span_set":
        # Scored (and prompted) under an older span_set version: re-run the base.
        from app.services.eval_task_handler_service import SPAN_SET_SCORING_VERSION

        if metrics.get("span_set_scoring_version") != SPAN_SET_SCORING_VERSION:
            return False
    if metrics.get("scoring_mode") == "tool_call":
        if tool_call_fingerprint is None or metrics.get("tool_call_fingerprint") != tool_call_fingerprint:
            return False
    if fingerprint is None:
        return True
    return fingerprint in (
        metrics.get("remediation_fingerprint"), metrics.get("structured_plan_fingerprint"),
        metrics.get("multi_option_fingerprint"),
    )


def _baseline_decoding_matches(
    result: EvalResult, decoding: str = "neutral", checked: dict[str, Any] | None = None
) -> bool:
    """A cached base-model result stands for one bar only if it was decoded
    that way (lift evals are greedy): neutral, or the model's own settings
    (``decoding="model"``). Results from before decoding was recorded
    inherited the model's generation_config without saying so — they are
    re-run once.
    ``checked``: the lift check's checked-decoding policy record
    (``checked_decoding_service.policy_record``; None = off). The cached
    result must carry the same one, so a base eval without checked decoding
    is never the bar for a checked run, nor the other way round."""
    from app.services.decoding_settings import MODEL_DECODING, decoding_record, generation_kwargs, is_model_decoding

    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    inference = metrics.get("inference") if isinstance(metrics.get("inference"), dict) else {}
    recorded = inference.get("decoding")
    if isinstance(recorded, dict):
        if recorded.get("checked_decoding") != checked:
            return False
        recorded = {k: v for k, v in recorded.items() if k != "checked_decoding"}
    elif checked is not None:
        return False
    if decoding == MODEL_DECODING:
        return is_model_decoding(recorded)
    return recorded == decoding_record(generation_kwargs(max_new_tokens=1, temperature=0.0))


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
    sql_execution_fingerprint: str | None = None,
    checked: dict[str, Any] | None = None,
    max_samples: int | None = None,
) -> EvalResult | None:
    """Newest baseline result on the test split decoded as ``decoding``, if
    newer than the split file (base models are deterministic under greedy
    decoding, so a fresh-enough result is exact, not approximate).
    ``scoring_fingerprint`` (remediation evals): the result must have been scored under the
    same detector / checks / retrieval / extra-field schema."""
    found, _basis = await _reusable_baseline_result_with_basis(
        db, baseline_exp, project_id, judge_label=judge_label, decoding=decoding,
        scoring_fingerprint=scoring_fingerprint, tool_call_fingerprint=tool_call_fingerprint,
        sql_execution_fingerprint=sql_execution_fingerprint, checked=checked, max_samples=max_samples,
    )
    return found


def _current_eval_input_sha256(project_id: int) -> str | None:
    """``eval_input_sha256`` of the live ``prepared/test.jsonl`` + manifest
    (``dataset_service.eval_input_identity``); None when unreadable."""
    from app.services.dataset_service import eval_input_identity
    from app.services.eval_task_handler_service import read_prepared_manifest

    path = Path(settings.DATA_DIR) / "projects" / str(project_id) / "prepared" / "test.jsonl"
    try:
        return eval_input_identity(path, read_prepared_manifest(project_id))["eval_input_sha256"]
    except (OSError, ValueError):  # ValueError: an encoding problem must not fail the lift Job
        return None


async def _reusable_baseline_result_with_basis(
    db: AsyncSession,
    baseline_exp: Experiment,
    project_id: int,
    *,
    judge_label: str | None = None,
    decoding: str = "neutral",
    scoring_fingerprint: str | None = None,
    tool_call_fingerprint: str | None = None,
    sql_execution_fingerprint: str | None = None,
    checked: dict[str, Any] | None = None,
    max_samples: int | None = None,
) -> tuple[EvalResult | None, str | None]:
    """``_reusable_baseline_result`` plus why it may be reused: ``"mtime"``
    (newer than the test file) or ``"content"`` (older than a rewrite of the
    test file, but its recorded ``details.dataset.eval_input_sha256`` -- the
    rows (source stamps stripped, except for remediation) + the
    prepared-manifest fields the eval reads -- equals the live file's, e.g. a
    re-import that only re-stamped the rows). A newer candidate that recorded
    an eval-input hash must match it too: "Make active" on an older version
    copies the snapshot back with its OLD mtime, so a base result scored on
    another test set can be newer than the file. Candidates without the hash
    (recorded before it existed) keep the mtime rule. The judge / decoding /
    scoring / row-scope (``max_samples``, ``_baseline_scope_matches``) checks apply to both."""
    import asyncio

    result = await db.execute(
        select(EvalResult)
        .where(EvalResult.experiment_id == baseline_exp.id)
        .where(EvalResult.dataset_name == AUTO_LIFT_DATASET)
        .order_by(desc(EvalResult.id))
        .limit(_REUSE_SCAN)
    )
    split_mtime = await _test_split_mtime(project_id)
    current_input: list[str | None] = []  # computed once, only when a candidate recorded a hash
    rows_now: list[int | None] = []  # rows in the live split, counted once when a cap must be compared

    async def _rows_now() -> int | None:
        if not rows_now:
            from app.services.evaluation_service import _count_eval_rows

            path = Path(settings.DATA_DIR) / "projects" / str(project_id) / "prepared" / "test.jsonl"
            rows_now.append(await asyncio.to_thread(_count_eval_rows, path))
        return rows_now[0]

    async def _current() -> str | None:
        if not current_input:
            current_input.append(await asyncio.to_thread(_current_eval_input_sha256, project_id))
        return current_input[0]

    for candidate in result.scalars().all():
        if candidate.created_at is None:
            continue
        created = candidate.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        details = candidate.details if isinstance(candidate.details, dict) else {}
        basis = "mtime"
        recorded = (details.get("dataset") or {}).get("eval_input_sha256") if isinstance(details.get("dataset"), dict) else None
        if split_mtime is not None and created < split_mtime:
            # Predates the file: reusable only when the eval input is unchanged (a metadata-only
            # rewrite). Skip, not stop -- an older result may still match.
            if not isinstance(recorded, str):
                continue
            current = await _current()
            if current is None or recorded != current:
                continue
            basis = "content"
        elif isinstance(recorded, str):
            # Newer than the file, but the file may be an older snapshot restored with its old mtime:
            # the recorded eval input must still be the live one (unreadable -> the mtime rule).
            current = await _current()
            if current is not None and recorded != current:
                continue
        # Results recorded before per-row scores were kept can't be paired row by
        # row with the fine-tuned eval ("within noise?") — re-run the base once.
        if not isinstance(details.get("row_scores"), dict):
            continue
        if not _baseline_judge_matches(candidate, judge_label):
            continue
        if not _baseline_decoding_matches(candidate, decoding, checked):
            continue
        if not _baseline_scoring_matches(
            candidate, scoring_fingerprint, tool_call_fingerprint, sql_execution_fingerprint
        ):
            continue
        if max_samples is not None and not _baseline_scope_matches(
            candidate, max_samples,
            None if _scope_trivial(candidate, max_samples) else await _rows_now(),
        ):
            continue
        return candidate, basis
    return None, None


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


async def _multi_option_fingerprint_for(db: AsyncSession, project_id: int) -> str | None:
    """The scoring fingerprint a reusable ``multi_option`` baseline must carry (None for any other
    project). Includes the option verifier's revision, so bumping it re-runs the base model."""
    try:
        from app.services import multi_option_scoring as mos
        from app.services.eval_task_handler_service import read_prepared_manifest

        project = await db.get(Project, project_id)
        if project is None:
            return None
        manifest = read_prepared_manifest(project_id)
        runtime_config = getattr(project, "runtime_config", None)
        if not mos.uses_multi_option(runtime_config, manifest):
            return None
        return mos.multi_option_fingerprint(mos.effective_config(runtime_config, manifest.get("output_schema")))
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


async def _sql_execution_fingerprint_for(db: AsyncSession, project_id: int) -> str | None:
    """The opt-in SQL execution scoring fingerprint (None while it is off); hashes the database files
    in a worker thread."""
    import asyncio

    try:
        from app.services.sql_execution_scoring import sql_execution_fingerprint

        project = await db.get(Project, project_id)
        rc = getattr(project, "runtime_config", None)
        return await asyncio.to_thread(sql_execution_fingerprint, rc, project_id)
    except Exception:  # noqa: BLE001 - cannot fingerprint: never reuse blindly
        return "unavailable"


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
    summary reads the chosen one. Returns (bar result, both reused, bar,
    reuse basis: "mtime" | "content" | None)."""
    import asyncio

    from app.services.decoding_settings import MODEL_DECODING, model_decoding_differs
    from app.services.evaluation_service import PERPLEXITY_EVAL_TYPE, run_heldout_evaluation

    scoring_fp = (
        await _remediation_fingerprint_for(db, project_id)
        or await _structured_plan_fingerprint_for(db, project_id, judge_label)
        or await _multi_option_fingerprint_for(db, project_id)
    )
    tool_fp = await _tool_call_fingerprint_for(db, project_id)
    sql_fp = await _sql_execution_fingerprint_for(db, project_id)
    # The lift check's checked-decoding policy (``common["checked_decoding"]``): the cache keys on it.
    # With context_grounding opted in for auto mode, the eval records it (``auto_includes_grounding``).
    from app.services.checked_decoding_service import normalize_policy, policy_record, with_grounding_opt_in

    project_row = await db.get(Project, project_id)
    checked = policy_record(with_grounding_opt_in(
        normalize_policy(common.get("checked_decoding")),
        runtime_config=getattr(project_row, "runtime_config", None),
    ))
    neutral, neutral_basis = await _reusable_baseline_result_with_basis(
        db, baseline_exp, project_id, judge_label=judge_label, scoring_fingerprint=scoring_fp,
        tool_call_fingerprint=tool_fp, sql_execution_fingerprint=sql_fp, checked=checked,
        max_samples=common.get("max_samples"),
    )
    reused = neutral is not None
    bases = [neutral_basis]
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
        own_result, own_basis = await _reusable_baseline_result_with_basis(
            db, baseline_exp, project_id, judge_label=judge_label, decoding=MODEL_DECODING,
            scoring_fingerprint=scoring_fp, tool_call_fingerprint=tool_fp, sql_execution_fingerprint=sql_fp,
            checked=checked, max_samples=common.get("max_samples"),
        )
        bases.append(own_basis)
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
    # Why the cached base result(s) stood: "content" when any was reused across a metadata-only
    # rewrite of the test file (same eval input), else "mtime"; None when the base was re-run.
    basis = None if not reused else ("content" if "content" in bases else "mtime")
    return chosen, reused, bar, basis


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


def _leg_scope(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    metrics = getattr(result, "metrics", None) or {}
    dataset = (getattr(result, "details", None) or {}).get("dataset") or {}
    out: dict[str, Any] = {"rows_scored": metrics.get("total"), "rows_available": dataset.get("rows_available")}
    flags = metrics.get("tool_flags")
    if isinstance(flags, dict):
        out["tool_flags"] = {k: flags.get(k) for k in ("catalog_fingerprint", "allowed_count", "tool_count",
                                                        "matches_training")}
    if metrics.get("tool_call_fingerprint"):
        out["tool_call_fingerprint"] = metrics["tool_call_fingerprint"]
    return out


def lift_scope_fields(baseline_result: Any, trained_result: Any) -> dict[str, Any]:
    """The lift Job result's ``scope``: rows scored / available per leg and, for a tool_calling project,
    the tool flags each leg used. ``tool_flags_mismatch`` is set only when the two legs were scored under
    different tool-call fingerprints (the flags changed between them), so the lift compares unlike evals."""
    base, trained = _leg_scope(baseline_result), _leg_scope(trained_result)
    out: dict[str, Any] = {"scope": {"baseline": base, "trained": trained}}
    b_fp = (base or {}).get("tool_call_fingerprint")
    t_fp = (trained or {}).get("tool_call_fingerprint")
    if b_fp and t_fp and b_fp != t_fp:
        out["tool_flags_mismatch"] = True
    return out


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
    checked_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a token-classification run on the test examples and, when
    the project has an answer key, on it too. Returns the Job result.
    ``checked_override``: the lift check's resolved checked-decoding policy,
    passed to every eval (a token-classification head records it as
    ``skipped: "engine_unsupported"``); None = the project's setting."""
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
            **({"checked_decoding": checked_override} if checked_override is not None else {}),
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
    checked_decoding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Baseline + fine-tuned held-out eval, then the paired lift summary.
    Commits after each eval so a failure in the second keeps the first.
    Raises when the run isn't eligible (the Job surfaces the reason).
    ``checked_decoding``: override of the project's checked-decoding setting;
    the policy is resolved ONCE and both evals (base and fine-tuned, every
    seed) run under it."""
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

    from app.services.checked_decoding_service import lift_policy_override

    checked_override = lift_policy_override(project.runtime_config, checked_decoding)
    if is_seed_group_leader(exp):
        if is_passage_trained(exp):
            raise ValueError(
                "A multi-seed run trained with retrieved passages isn't lift-checked as a group yet; "
                "evaluate each seed run on its own."
            )
        return await _run_seed_group_lift_eval(
            db, project=project, leader=exp, report=_report, checked_override=checked_override
        )
    if is_passage_trained(exp):
        # Trained to answer from the served passages: scored WITH them,
        # against the base model + the same passages (not the bare base).
        # The passages harness has no checked decoding: the one resolved
        # policy applies to neither side, and the result says so.
        result = await run_passage_reader_evaluation(db, project=project, experiment=exp, report=_report)
        not_applied = checked_not_applied(checked_override)
        if not_applied is not None and isinstance(result, dict):
            result["checked_decoding"] = not_applied
        return result
    if is_token_classification(exp):
        return await run_span_evaluation(
            db, project_id=project_id, experiment=exp, report=_report, checked_override=checked_override
        )
    if is_encoder_classification(exp):
        return await run_head_only_evaluation(
            db, project=project, experiment=exp, report=_report, checked_override=checked_override
        )

    eval_type = eval_type_for_project(project)
    max_samples, max_new_tokens = _lift_eval_limits(project)
    common = {
        "project_id": project_id,
        "dataset_name": AUTO_LIFT_DATASET,
        "eval_type": eval_type,
        "max_samples": max_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "judge_model": None,
        "checked_decoding": checked_override,
    }
    answer_judge = await _resolve_lift_judge(db, project)
    common["answer_judge"] = answer_judge

    baseline_exp = await find_or_create_baseline_experiment(
        db, project_id, exp.base_model, source="post_training.auto_lift"
    )
    await db.commit()
    baseline_result, baseline_reused, baseline_bar, baseline_reuse_basis = await _baseline_bar(
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
        {k: full.get(k) for k in ("verdict", "n", "better", "worse", "same", "mcnemar_p")}
        if isinstance(full, dict)
        else None
    )
    from app.services.judge_call_guard import job_result_fields

    return {
        "experiment_id": experiment_id,
        "base_model": exp.base_model,
        "evidence": evidence,
        # Which rows and (tool_calling) which tool flags each leg was scored with: picked-data provenance.
        **lift_scope_fields(baseline_result, trained_result),
        # Only when an unreachable judge left rows unjudged on either leg: the bell appends "judge
        # incomplete" to the lift verdict.
        **job_result_fields(trained_result, baseline_result),
        "baseline_experiment_id": baseline_exp.id,
        "baseline_eval_result_id": getattr(baseline_result, "id", None),
        "baseline_reused": baseline_reused,
        "baseline_reuse_basis": baseline_reuse_basis,
        "baseline_bar": baseline_bar,
        "trained_eval_result_id": getattr(trained_result, "id", None),
        "lift_status": summary.get("status"),
        "headline": headline,
    }


async def run_head_only_evaluation(
    db: AsyncSession, *, project: Project, experiment: Experiment, report,
    checked_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a classification-head run on an encoder against the test
    examples only — there is no zero-shot base model to compare with.
    ``checked_override``: the lift check's resolved checked-decoding policy
    (a classifier head records ``skipped: "engine_unsupported"``)."""
    from app.services.evaluation_service import run_heldout_evaluation

    await report(0.2, f"Evaluating run #{experiment.id} (classifier head; no base-model baseline for an encoder)")
    max_samples, max_new_tokens = _lift_eval_limits(project)
    result = await run_heldout_evaluation(
        db=db, project_id=project.id, experiment_id=experiment.id, dataset_name=AUTO_LIFT_DATASET,
        eval_type=eval_type_for_project(project), max_samples=max_samples, max_new_tokens=max_new_tokens,
        temperature=0.0, judge_model=None, model_path=None,
        **({"checked_decoding": checked_override} if checked_override is not None else {}),
    )
    await db.commit()
    from app.services.judge_call_guard import job_result_fields

    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    return {
        "kind": "head_eval",
        "experiment_id": experiment.id,
        "base_model": experiment.base_model,
        "trained_eval_result_id": result.id,
        **job_result_fields(result),
        "lift_status": "no_baseline",
        "reason": "encoder base model: an untrained classifier head is no baseline",
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "rows": metrics.get("evaluated_samples") or metrics.get("total"),
    }


def checked_decoding_unsupported_reason(exp: Any) -> str | None:
    """Why a lift check of ``exp`` can't run under checked decoding, else None. A run trained
    with retrieved passages is scored by the passages harness, which has no checked decoding
    (the lift-check endpoint answers 422 to an explicit enabled override for it)."""
    from app.services.passage_reader_eval_service import is_passage_trained

    if is_passage_trained(exp):
        return "run_type_unsupported"
    return None


def checked_not_applied(policy: dict[str, Any] | None) -> dict[str, Any] | None:
    """The job-result block for a run type checked decoding can't apply to: None when the
    resolved policy is off, else ``{policy, label, skipped: "run_type_unsupported"}``."""
    from app.services.checked_decoding_service import policy_label, policy_record

    record = policy_record(policy)
    if record is None:
        return None
    return {"policy": record, "label": policy_label(record), "skipped": "run_type_unsupported"}


def is_simulated_run(exp: Experiment) -> bool:
    """A run trained by the simulate runtime (no weights; it cannot be evaluated)."""
    cfg = exp.config if isinstance(exp.config, dict) else {}
    runtime = cfg.get("_runtime") if isinstance(cfg.get("_runtime"), dict) else {}
    backend = str(runtime.get("backend") or "").lower()
    runtime_id = str(runtime.get("runtime_id") or "").lower()
    return backend == "simulate" or "simulate" in runtime_id


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
    db: AsyncSession, *, project: Project, leader: Experiment, report,
    checked_override: dict[str, Any] | None = None,
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
    from app.services.checked_decoding_service import lift_policy_override
    from app.services.paired_comparison_stats import seed_spread_evidence
    from app.services.sft_lift_summary_service import compute_sft_lift_summary

    children = await seed_group_children(db, leader)
    if not children:
        raise ValueError("Automatic lift eval skipped: seed group has no completed child runs")

    eval_type = eval_type_for_project(project)
    max_samples, max_new_tokens = _lift_eval_limits(project)
    common = {
        "project_id": project.id,
        "dataset_name": AUTO_LIFT_DATASET,
        "eval_type": eval_type,
        "max_samples": max_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "judge_model": None,
        "checked_decoding": checked_override if checked_override is not None else lift_policy_override(
            project.runtime_config, None
        ),
    }
    answer_judge = await _resolve_lift_judge(db, project)
    common["answer_judge"] = answer_judge
    baseline_exp = await find_or_create_baseline_experiment(
        db, project.id, leader.base_model, source="post_training.auto_lift"
    )
    await db.commit()
    baseline_result, baseline_reused, baseline_bar, baseline_reuse_basis = await _baseline_bar(
        db, baseline_exp=baseline_exp, project_id=project.id, base_model=leader.base_model, common=common,
        judge_label=answer_judge.label if answer_judge else None, report=report, fractions=(0.05, 0.1),
    )

    seeds: list[dict[str, Any]] = []
    seed_results: list[Any] = []
    for idx, child in enumerate(children):
        fraction = 0.15 + 0.75 * idx / len(children)
        await report(fraction, f"Evaluating seed {child.seed_value} (run #{child.id}, {idx + 1}/{len(children)})")
        trained_result = await run_heldout_evaluation(
            db=db, experiment_id=child.id, model_path=None, **common
        )
        await db.commit()
        seed_results.append(trained_result)
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
    from app.services.judge_call_guard import job_result_fields

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
        **job_result_fields(*seed_results, baseline_result),
        "baseline_experiment_id": baseline_exp.id,
        "baseline_eval_result_id": getattr(baseline_result, "id", None),
        "baseline_reused": baseline_reused,
        "baseline_reuse_basis": baseline_reuse_basis,
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
    checked_decoding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Spawn the lift-eval Job for a just-completed run, or return why not.
    ``checked_decoding``: per-check override of the project's checked-decoding setting.
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
                    **({"checked_decoding": checked_decoding} if checked_decoding is not None else {}),
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
            params={
                "experiment_id": experiment_id,
                "base_model": exp.base_model,
                **({"checked_decoding": checked_decoding} if checked_decoding is not None else {}),
            },
        )
        return {"started": True, "job_id": getattr(job, "id", None)}
    except Exception as exc:  # noqa: BLE001
        return {"started": False, "skipped_reason": f"error:{exc.__class__.__name__}: {exc}"}

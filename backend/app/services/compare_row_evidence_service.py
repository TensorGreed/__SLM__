"""Is B really different from A? Row-paired evidence for Compare (Epic W1).

The Compare page used to show bare metric deltas: "35.9% vs 29.5%" on 78
rows reads as a regression, yet the paired 95% interval for that change
crossed zero. This module pairs the two runs' per-row scores
(``EvalResult.details["row_scores"]``) by row key and reports rows better /
worse / same for B vs A, the interval for the mean per-row change, and a
verdict (``paired_comparison_stats``: better / worse / within_noise /
too_few_rows, plus ``not_comparable`` here).

A verdict is only given when the two evals were scored the same way:
same answer key (test split file), same judge, same scoring mode. When they
differ the response says which, gives no verdict, and offers to re-score the
older run with the lift check (``post_training_eval_service``) on the
current answer key and judge.

Also stamps the answer-key identity a held-out eval records
(``answer_key_stamp``: the test file's sha256 + the split version it
matches), so later comparisons don't have to infer it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetVersion
from app.models.experiment import EvalResult, Experiment


# Plain names for the held-out dataset types (agent-notes UI vocabulary).
_ANSWER_KEY_NAMES: dict[str, str] = {
    "test": "test examples",
    "gold_test": "answer key (final)",
    "gold_dev": "answer key (practice)",
    "validation": "validation split",
    "benchmark": "external benchmark",
}

# Scoring fingerprints some scoring modes stamp on their metrics; two
# results scored under different settings of the same mode can't pair.
_SCORING_FINGERPRINTS: tuple[str, ...] = (
    "remediation_fingerprint",
    "structured_plan_fingerprint",
    "tool_call_fingerprint",
)

NO_ROW_SCORES_MESSAGE = "no per-row scores (evaluated before per-row capture)"


async def answer_key_stamp(db: AsyncSession, dataset: Dataset, path: Path) -> dict[str, Any]:
    """``{"sha256", "version"}`` of the held-out file an eval is about to read:
    its content hash and the split version (``DatasetVersion.version``) whose
    manifest recorded that hash, or None when no version matches (an edited
    file, an answer key without versions). Never raises."""
    from app.services.dataset_service import _sha256_file

    stamp: dict[str, Any] = {"sha256": None, "version": None}
    try:
        stamp["sha256"] = await asyncio.to_thread(_sha256_file, path)
    except OSError:
        return stamp
    try:
        rows = await db.execute(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset.id)
            .order_by(desc(DatasetVersion.version))
            .limit(50)
        )
        for version in rows.scalars().all():
            manifest = version.manifest if isinstance(version.manifest, dict) else {}
            if manifest.get("sha256") == stamp["sha256"]:
                stamp["version"] = int(version.version)
                break
    except Exception:  # noqa: BLE001 - a stamp must never fail the eval
        pass
    return stamp


# --------------------------------------------------------------------------
# Provenance: what an eval result was scored against.
# --------------------------------------------------------------------------


def _metrics(er: EvalResult) -> dict[str, Any]:
    return er.metrics if isinstance(er.metrics, dict) else {}


def _details(er: EvalResult) -> dict[str, Any]:
    return er.details if isinstance(er.details, dict) else {}


async def _answer_key(db: AsyncSession, er: EvalResult) -> dict[str, Any]:
    """Which answer key the eval scored against. ``sha256`` / ``version`` are
    only what the eval recorded (``details.dataset``, stamped since W1). An
    older result's split version is inferred as the newest version of its
    dataset created before the eval (``version_source: "inferred"``) and that
    version's manifest hash goes to ``inferred_sha256`` — never into
    ``sha256``. Inference is wrong if an older version was re-activated in
    between, so a difference that rests on it reads "may differ"."""
    dataset = _details(er).get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    dataset_id = dataset.get("id") if isinstance(dataset.get("id"), int) else None
    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_type": str(dataset.get("dataset_type") or er.dataset_name or "") or None,
        "version": dataset.get("version") if isinstance(dataset.get("version"), int) else None,
        "sha256": dataset.get("sha256") if isinstance(dataset.get("sha256"), str) else None,
        "version_source": None,
        "inferred_sha256": None,
    }
    if out["version"] is not None:
        out["version_source"] = "recorded"
    if out["sha256"] is not None or dataset_id is None or er.created_at is None:
        return out
    try:
        row = (
            await db.execute(
                select(DatasetVersion)
                .where(DatasetVersion.dataset_id == dataset_id, DatasetVersion.created_at <= er.created_at)
                .order_by(desc(DatasetVersion.created_at), desc(DatasetVersion.id))
                .limit(1)
            )
        ).scalar_one_or_none()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        row = None
    if row is not None:
        manifest = row.manifest if isinstance(row.manifest, dict) else {}
        out["version"] = int(row.version)
        out["version_source"] = "inferred"
        if isinstance(manifest.get("sha256"), str):
            out["inferred_sha256"] = manifest["sha256"]
    return out


def _judge(er: EvalResult) -> dict[str, Any]:
    """Every judge that scored the eval: the answer judge (``metrics.judge``;
    a skipped judge = none), the plan-rubric judge + rubric hash
    (``metrics.rubric`` when it judged) and the code-example judge."""
    metrics = _metrics(er)
    answer = metrics.get("judge")
    answer_label = None
    if isinstance(answer, dict) and not answer.get("skipped"):
        answer_label = str(answer.get("judge") or "") or None
    rubric = metrics.get("rubric")
    rubric_label = rubric_sha = None
    if isinstance(rubric, dict) and rubric.get("status") == "judged":
        rubric_label = str(rubric.get("judge") or "") or None
        rubric_sha = str(rubric.get("sha256") or "") or None
    code_example = metrics.get("code_example_judge")
    code_example_label = None
    if isinstance(code_example, dict):
        code_example_label = str(code_example.get("judge") or "") or None
    return {
        "answer": answer_label,
        "rubric": rubric_label,
        "rubric_sha": rubric_sha,
        "code_example": code_example_label,
    }


def _decoding(er: EvalResult) -> dict[str, Any] | None:
    """The decoding settings the eval recorded (``metrics.inference.decoding``,
    else ``details.inference.decoding``); None for evals from before they
    were recorded (those used the model's own generation config)."""
    for block in (_metrics(er).get("inference"), _details(er).get("inference")):
        if isinstance(block, dict) and isinstance(block.get("decoding"), dict):
            return dict(block["decoding"])
    return None


def _scoring(er: EvalResult) -> dict[str, Any]:
    metrics = _metrics(er)
    mode = metrics.get("scoring_mode")
    return {
        "eval_type": er.eval_type or None,
        "decoding": _decoding(er),
        "scoring_mode": str(mode) if isinstance(mode, str) and mode else None,
        "fingerprints": {
            key: metrics[key] for key in _SCORING_FINGERPRINTS if isinstance(metrics.get(key), str)
        },
    }


async def eval_provenance(db: AsyncSession, exp: Experiment, er: EvalResult | None) -> dict[str, Any]:
    """What one side of a comparison was scored against — shown on both
    Compare cards so the user can see exactly what was compared."""
    if er is None:
        return {
            "experiment_id": exp.id,
            "experiment_name": exp.name,
            "eval_result_id": None,
            "evaluated_at": None,
            "answer_key": None,
            "judge": None,
            "scoring": None,
            "has_row_scores": False,
        }
    return {
        "experiment_id": exp.id,
        "experiment_name": exp.name,
        "eval_result_id": er.id,
        "evaluated_at": er.created_at.isoformat() if er.created_at else None,
        "answer_key": await _answer_key(db, er),
        "judge": _judge(er),
        "scoring": _scoring(er),
        "has_row_scores": isinstance(_details(er).get("row_scores"), dict),
    }


# --------------------------------------------------------------------------
# Comparability: may the two results be paired at all?
# --------------------------------------------------------------------------


def answer_key_label(key: dict[str, Any] | None) -> str:
    """"test examples, split v3" / "answer key (final), file 1a2b3c4d"."""
    if not key:
        return "unknown"
    kind = _ANSWER_KEY_NAMES.get(str(key.get("dataset_type") or ""), str(key.get("dataset_type") or "unknown"))
    if key.get("version") is not None:
        return f"{kind}, split v{key['version']}"
    if key.get("sha256"):
        return f"{kind}, file {str(key['sha256'])[:8]}"
    return kind


def _or_none(label: Any, empty: str = "no judge") -> str:
    return str(label) if label else empty


# ``eval_type`` labels that name the same reference-matching scorer: the lift
# check labels a run ``exact_match`` (``eval_type_for_project``) where a manual
# held-out eval of the same file may say ``f1``. Both run the same handler and
# record every metric they compute; pairing only uses metrics BOTH report, and
# a real scoring change shows up in ``scoring_mode`` / the fingerprints / the
# file hash. ``llm_judge`` / ``safety`` / ``perplexity`` are different scorers.
_EVAL_TYPE_FAMILY: dict[str, str] = {"exact_match": "reference", "f1": "reference"}


def _eval_type_family(eval_type: Any) -> str | None:
    if not eval_type:
        return None
    return _EVAL_TYPE_FAMILY.get(str(eval_type), str(eval_type))


def _inferred_suffix(key_a: dict[str, Any], key_b: dict[str, Any], label_a: str, label_b: str) -> str:
    inferred_a = key_a.get("version_source") == "inferred"
    inferred_b = key_b.get("version_source") == "inferred"
    if inferred_a and inferred_b:
        return " (both inferred from eval time)"
    if inferred_a:
        return f" ({label_a} inferred from eval time)"
    if inferred_b:
        return f" ({label_b} inferred from eval time)"
    return ""


def _answer_key_difference(key_a: dict[str, Any], key_b: dict[str, Any]) -> dict[str, Any] | None:
    """``{message, a, b, inferred}`` when the two answer keys differ (or may),
    None when they match, ``{"unknown": True}`` when it can't be told."""
    type_a, type_b = key_a.get("dataset_type"), key_b.get("dataset_type")
    if type_a and type_b and type_a != type_b:
        return {"message": f"different answer key: {answer_key_label(key_a)} vs {answer_key_label(key_b)}",
                "a": type_a, "b": type_b, "inferred": False}
    va, vb = key_a.get("version"), key_b.get("version")
    sha_a = key_a.get("sha256") or key_a.get("inferred_sha256")
    sha_b = key_b.get("sha256") or key_b.get("inferred_sha256")
    inferred = not (key_a.get("sha256") and key_b.get("sha256"))
    if sha_a and sha_b:
        if sha_a == sha_b:
            return None
        if va is not None and vb is not None and va != vb:
            what = f"split v{va} vs v{vb}" + _inferred_suffix(key_a, key_b, f"v{va}", f"v{vb}")
        else:
            what = f"test file {sha_a[:8]} vs {sha_b[:8]}" + _inferred_suffix(key_a, key_b, "A's", "B's")
            if not inferred:
                what += " (the file changed between the two evals)"
        lead = "answer key may differ" if inferred else "different answer key"
        return {"message": f"{lead}: {what}", "a": va if va is not None else sha_a[:8],
                "b": vb if vb is not None else sha_b[:8], "inferred": inferred}
    if va is not None and vb is not None and key_a.get("dataset_id") == key_b.get("dataset_id"):
        if va == vb:
            return None
        inferred = "inferred" in (key_a.get("version_source"), key_b.get("version_source"))
        lead = "answer key may differ" if inferred else "different answer key"
        return {"message": f"{lead}: split v{va} vs v{vb}" + _inferred_suffix(key_a, key_b, f"v{va}", f"v{vb}"),
                "a": va, "b": vb, "inferred": inferred}
    return {"unknown": True}


def _decoding_caveats(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Decoding never blocks a verdict, but a different decoding can move a
    score (a repetition penalty once nulled half of one extractor's fields),
    so say so."""
    dec_a = (a.get("scoring") or {}).get("decoding")
    dec_b = (b.get("scoring") or {}).get("decoding")
    if dec_a is None and dec_b is None:
        return []
    if dec_a is None or dec_b is None:
        side = "A" if dec_a is None else "B"
        return [f"{side}'s eval predates recorded decoding settings (it used the model's own generation config); "
                "the two runs may have been decoded differently."]
    if dec_a == dec_b:
        return []
    changed = [
        f"{key} {dec_a.get(key, 'unset')} vs {dec_b.get(key, 'unset')}"
        for key in sorted(set(dec_a) | set(dec_b))
        if dec_a.get(key) != dec_b.get(key)
    ]
    shown = ", ".join(changed[:3]) + (f" (+{len(changed) - 3} more)" if len(changed) > 3 else "")
    return [f"Decoded differently: {shown}. The gap may partly come from decoding, not the model."]


def compare_provenance(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """``{comparable, certain, differences: [{kind, message, a, b, inferred}],
    notes, caveats}`` for two ``eval_provenance`` blocks. Any difference in
    answer key, judge or scoring makes the pair not comparable; ``certain``
    is False when every difference rests on an inferred answer key ("may
    differ"). An answer key that can't be told is a note; different decoding
    is a caveat — neither blocks the verdict."""
    differences: list[dict[str, Any]] = []
    notes: list[str] = []

    def _diff(kind: str, message: str, a_value: Any, b_value: Any, inferred: bool = False) -> None:
        differences.append({"kind": kind, "message": message, "a": a_value, "b": b_value, "inferred": inferred})

    key_diff = _answer_key_difference(a.get("answer_key") or {}, b.get("answer_key") or {})
    if key_diff is not None and key_diff.get("unknown"):
        notes.append(
            "Which answer-key version each eval used was not recorded; only rows both evals scored "
            "(same question and reference) are paired."
        )
    elif key_diff is not None:
        _diff("answer_key", key_diff["message"], key_diff["a"], key_diff["b"], key_diff["inferred"])

    judge_a, judge_b = a.get("judge") or {}, b.get("judge") or {}
    if judge_a.get("answer") != judge_b.get("answer"):
        _diff("judge", f"different judge: {_or_none(judge_a.get('answer'))} vs {_or_none(judge_b.get('answer'))}",
              judge_a.get("answer"), judge_b.get("answer"))
    if judge_a.get("rubric") != judge_b.get("rubric"):
        _diff("judge", f"different plan-rubric judge: {_or_none(judge_a.get('rubric'))} vs {_or_none(judge_b.get('rubric'))}",
              judge_a.get("rubric"), judge_b.get("rubric"))
    elif judge_a.get("rubric_sha") != judge_b.get("rubric_sha"):
        _diff("judge", f"different plan rubric: {_or_none(judge_a.get('rubric_sha'), 'none')[:8]} vs "
              f"{_or_none(judge_b.get('rubric_sha'), 'none')[:8]}", judge_a.get("rubric_sha"), judge_b.get("rubric_sha"))
    if judge_a.get("code_example") != judge_b.get("code_example"):
        _diff("judge", f"different code-example judge: {_or_none(judge_a.get('code_example'))} vs "
              f"{_or_none(judge_b.get('code_example'))}", judge_a.get("code_example"), judge_b.get("code_example"))

    scoring_a, scoring_b = a.get("scoring") or {}, b.get("scoring") or {}
    if _eval_type_family(scoring_a.get("eval_type")) != _eval_type_family(scoring_b.get("eval_type")):
        _diff("scoring", f"different scoring: {scoring_a.get('eval_type') or 'unknown'} vs {scoring_b.get('eval_type') or 'unknown'}",
              scoring_a.get("eval_type"), scoring_b.get("eval_type"))
    if scoring_a.get("scoring_mode") != scoring_b.get("scoring_mode"):
        _diff("scoring", f"different scoring mode: {scoring_a.get('scoring_mode') or 'default'} vs "
              f"{scoring_b.get('scoring_mode') or 'default'}", scoring_a.get("scoring_mode"), scoring_b.get("scoring_mode"))
    fps_a, fps_b = scoring_a.get("fingerprints") or {}, scoring_b.get("fingerprints") or {}
    for name in _SCORING_FINGERPRINTS:
        if name in fps_a and name in fps_b and fps_a[name] != fps_b[name]:
            _diff("scoring", f"different scoring settings ({name.replace('_', ' ')})", fps_a[name], fps_b[name])

    return {
        "comparable": not differences,
        "certain": not differences or not all(d["inferred"] for d in differences),
        "differences": differences,
        "notes": notes,
        "caveats": _decoding_caveats(a, b),
    }


# --------------------------------------------------------------------------
# Row evidence.
# --------------------------------------------------------------------------


def _row_metric_candidates(er_a: EvalResult, er_b: EvalResult) -> list[tuple[str, str]]:
    """``(metric_id, score_key)`` for every metric both results report that is
    a mean of per-row scores, headline-first (the lift's preference order);
    one entry per score key (exact_match and accuracy share ``correct``)."""
    from app.services.sft_lift_summary_service import _PREFERRED_HEADLINE_METRICS, _ROW_SCORE_FOR_METRIC

    shared = set(_metrics(er_a)) & set(_metrics(er_b))
    order = [m for m in _PREFERRED_HEADLINE_METRICS if m in _ROW_SCORE_FOR_METRIC]
    order += sorted(m for m in _ROW_SCORE_FOR_METRIC if m not in order)
    out: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for metric_id in order:
        score_key = _ROW_SCORE_FOR_METRIC[metric_id]
        if metric_id in shared and score_key not in seen_keys:
            seen_keys.add(score_key)
            out.append((metric_id, score_key))
    return out


def _older_side(er_a: EvalResult, er_b: EvalResult) -> str:
    def _key(er: EvalResult) -> tuple[str, int]:
        return (er.created_at.isoformat() if er.created_at else "", int(er.id or 0))

    return "a" if _key(er_a) <= _key(er_b) else "b"


_UNAVAILABLE_TEXT: dict[str, str] = {
    "baseline_experiment": (
        "{run} is a base-model baseline. Run \"Evaluate this run\" in Results on a trained run of the same "
        "base model: the lift check re-scores the base model on the current answer key and judge."
    ),
    "simulated_run": "{run} is a simulated run with no model weights, so it can't be re-scored. Compare two real runs.",
    "not_completed": "{run} hasn't finished training, so it can't be re-scored yet.",
    "no_model_weights": "{run}'s model weights are no longer on disk, so it can't be re-scored.",
}


async def _rescore_offer(
    db: AsyncSession, exp: Experiment, side: str, why: str, other_type: str | None
) -> dict[str, Any]:
    """The one-click re-score: the lift check (``POST
    /evaluation/summary/lift-check``) re-evaluates the run on the project's
    current test examples with the current judge — the same Job "Evaluate
    this run" starts. Not offered when the lift check can't run on it."""
    from app.models.project import Project
    from app.services.post_training_eval_service import AUTO_LIFT_DATASET, auto_lift_skip_reason

    run = f"Run #{exp.id}"
    offer: dict[str, Any] = {
        "side": side,
        "experiment_id": exp.id,
        "experiment_name": exp.name,
        "why": why,
        "available": False,
        "in_progress": False,
        "job_id": None,
        "unavailable_reason": None,
        "instruction": "",
    }
    if other_type and other_type != AUTO_LIFT_DATASET:
        offer["unavailable_reason"] = "answer_key_not_test_split"
        offer["instruction"] = (
            f"The other run was scored on the {_ANSWER_KEY_NAMES.get(other_type, other_type)}, and the re-score "
            f"uses the test examples. Re-run {run}'s held-out eval on the same answer key from Results → "
            "Advanced evaluation."
        )
        return offer
    from app.services.post_training_eval_service import active_lift_job_id

    active = await active_lift_job_id(db, exp.project_id, exp.id)
    if active is not None:
        offer["in_progress"] = True
        offer["job_id"] = active
        offer["instruction"] = f"A re-score of {run} is running (job #{active}). Reload Compare when the bell says it is done."
        return offer
    project = await db.get(Project, exp.project_id)
    reason = auto_lift_skip_reason(exp, project, manual=True)
    if reason is not None:
        offer["unavailable_reason"] = reason
        offer["instruction"] = _UNAVAILABLE_TEXT.get(
            reason, "{run} can't be re-scored here (" + reason + "). Re-run its held-out eval from Results."
        ).format(run=run)
        return offer
    offer["available"] = True
    offer["instruction"] = (
        f"Re-scores {run} on the current test examples with the current judge (the lift check; it runs "
        "the model again). Reload Compare when the bell says it is done."
    )
    return offer


def _empty_evidence() -> dict[str, Any]:
    return {
        "verdict": "too_few_rows",
        "reason": None,
        "message": None,
        "metric_id": None,
        "score_key": None,
        "n": 0,
        "better": 0,
        "worse": 0,
        "same": 0,
        "mean_diff": None,
        "ci_low": None,
        "ci_high": None,
        "only_in_a": 0,
        "only_in_b": 0,
        "unscored": 0,
        "metrics": [],
        "comparability": None,
        "rescore": None,
    }


async def build_row_evidence(
    db: AsyncSession,
    *,
    exp_a: Experiment,
    eval_a: EvalResult | None,
    prov_a: dict[str, Any],
    exp_b: Experiment,
    eval_b: EvalResult | None,
    prov_b: dict[str, Any],
) -> dict[str, Any]:
    """B vs A, row by row, for the headline metric (the first per-row metric
    both report, in the lift's order) and every other per-row metric.

    ``verdict``: better / worse / within_noise / too_few_rows (also: a side
    has no eval, or no per-row scores) / not_comparable (different answer
    key, judge or scoring — see ``comparability.differences``; no counts then).
    """
    from app.services.paired_comparison_stats import pair_row_scores, paired_difference_evidence

    out = _empty_evidence()
    if eval_a is None or eval_b is None:
        side, exp = ("a", exp_a) if eval_a is None else ("b", exp_b)
        out["reason"] = "missing_eval"
        out["message"] = f"{side.upper()} (run #{exp.id}) has no eval yet."
        other = prov_b if side == "a" else prov_a
        out["rescore"] = await _rescore_offer(
            db, exp, side, "missing_eval", ((other.get("answer_key") or {}).get("dataset_type"))
        )
        return out

    comparability = compare_provenance(prov_a, prov_b)
    out["comparability"] = comparability
    older = _older_side(eval_a, eval_b)
    if not comparability["comparable"]:
        out["verdict"] = "not_comparable"
        out["reason"] = "not_comparable"
        out["message"] = "; ".join(d["message"] for d in comparability["differences"])
        exp, other = (exp_a, prov_b) if older == "a" else (exp_b, prov_a)
        out["rescore"] = await _rescore_offer(
            db, exp, older, "not_comparable", (other.get("answer_key") or {}).get("dataset_type")
        )
        return out

    candidates = _row_metric_candidates(eval_a, eval_b)
    rows_a = _details(eval_a).get("row_scores")
    rows_b = _details(eval_b).get("row_scores")
    metrics: list[dict[str, Any]] = []
    for metric_id, score_key in candidates:
        paired = pair_row_scores(rows_a, rows_b, score_key)
        if paired is None:
            continue
        evidence = paired_difference_evidence(paired["before"], paired["after"])
        evidence.update({
            "metric_id": metric_id,
            "score_key": score_key,
            "only_in_a": paired["only_before"],
            "only_in_b": paired["only_after"],
            "unscored": paired["unscored"],
        })
        metrics.append(evidence)
    out["metrics"] = metrics

    if not metrics:
        missing = [s for s, p in (("a", prov_a), ("b", prov_b)) if not p.get("has_row_scores")]
        if missing:
            side = missing[0] if len(missing) == 1 else older
            out["reason"] = "no_row_scores"
            names = " and ".join(s.upper() for s in missing)
            out["message"] = f"{names}: {NO_ROW_SCORES_MESSAGE}"
            exp, other = (exp_a, prov_b) if side == "a" else (exp_b, prov_a)
            out["rescore"] = await _rescore_offer(
                db, exp, side, "no_row_scores", (other.get("answer_key") or {}).get("dataset_type")
            )
        else:
            out["reason"] = "no_row_metric"
            out["message"] = (
                "No metric both evals report is an average of per-row scores, so there is no "
                "row-by-row verdict."
            )
        return out

    headline = metrics[0]
    for key in ("metric_id", "score_key", "n", "better", "worse", "same", "mean_diff", "ci_low", "ci_high",
                "verdict", "only_in_a", "only_in_b", "unscored"):
        out[key] = headline[key]
    return out


__all__ = [
    "answer_key_label",
    "answer_key_stamp",
    "build_row_evidence",
    "compare_provenance",
    "eval_provenance",
]

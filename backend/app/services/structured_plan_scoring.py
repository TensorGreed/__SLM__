"""Scoring for the ``structured-plan`` recipe (``scoring_mode == "structured_plan"``).

A plan passes when it is valid JSON, validates against the project's JSON Schema bundle and every
configured JSON pointer matches the reference (or one of several acceptable values). When a rubric
runs (``finalize_with_rubric``) a failed rubric verdict also fails the row; rows the judge could not
grade are reported, never counted as failures.

Per row this sets ``parsed_prediction``, ``parsed_reference``, ``is_valid_json``, ``row_schema_valid``,
``schema_errors`` (top 5), ``row_pointer_results`` (``{ptr: {match, pred, gold, source}}``),
``row_structured_plan_pass`` and ``row_exact_match`` (= pass).

Pointer matching (``resolve_json_pointer``; ``*`` fans out, values compare with
``output_schema_service.value_match`` and types inferred from the gold value):

- the row carries ``acceptable[ptr]``: match iff at least one prediction value resolves and every
  resolved prediction value matches some acceptable value;
- otherwise the gold is what the pointer resolves to on the reference: match iff the two multisets
  are equal (order-insensitive). Missing in both = match; missing only in the prediction = no match.

Without a saved schema, "schema valid" falls back to "valid JSON object" and the metrics carry
``schema_missing: true``.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.services.json_pointer import resolve_json_pointer, split_json_pointer
from app.services.output_schema_service import value_match

SCORING_MODE = "structured_plan"
ROW_PASS_KEY = "row_structured_plan_pass"
MAX_ROW_ERRORS = 5
MAX_TOP_ERRORS = 10
MAX_RESULT_VALUES = 10
_INDEX = re.compile(r"/\d+(?=/|$)")


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def types_of(value: Any) -> list[str]:
    """Scalar types a gold value implies for ``value_match`` (containers compare literally)."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["boolean"]
    if isinstance(value, (int, float)):
        return ["number"]
    if isinstance(value, str):
        return ["string"]
    return []


def _matches(pred: Any, gold: Any) -> bool:
    return value_match(pred, gold, types_of(gold))


def multiset_equal(pred: list[Any], gold: list[Any]) -> bool:
    if len(pred) != len(gold):
        return False
    remaining = list(gold)
    for p in pred:
        for i, g in enumerate(remaining):
            if _matches(p, g):
                del remaining[i]
                break
        else:
            return False
    return True


def pointer_match(pred_values: list[Any], gold_values: list[Any], acceptable: list[Any] | None) -> bool:
    if acceptable:
        return bool(pred_values) and all(any(_matches(p, a) for a in acceptable) for p in pred_values)
    if not pred_values and not gold_values:
        return True
    return multiset_equal(pred_values, gold_values)


def _cap(values: list[Any]) -> list[Any]:
    return list(values[:MAX_RESULT_VALUES])


def _acceptable_for(prediction: dict[str, Any]) -> dict[str, list[Any]]:
    row = prediction.get("plan_row")
    acceptable = row.get("acceptable") if isinstance(row, dict) else None
    return acceptable if isinstance(acceptable, dict) else {}


PRECOMPUTED_KEY = "_structured_plan_schema"


def _schema_check(bundle: Any, parsed: Any, prediction: dict[str, Any] | None = None) -> tuple[bool, list[dict[str, Any]]]:
    if bundle is None:
        return isinstance(parsed, dict), []
    if parsed is None:
        return False, []
    pre = (prediction or {}).get(PRECOMPUTED_KEY)
    if isinstance(pre, tuple) and len(pre) == 2 and pre[0] == bundle.sha256:
        errors = pre[1]  # validated off the event loop by ``precompute``
    else:
        errors = bundle.validate(parsed)
    return not errors, errors


def precompute(predictions: list[dict[str, Any]], bundle: Any, parse_json: Callable[[Any], Any]) -> None:
    """Validate every parsed prediction against ``bundle`` and stash the result on the row so
    ``score_structured_plan`` reuses it (blocking; run in a worker thread)."""
    if bundle is None:
        return
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        parsed = parse_json(prediction.get("prediction") or "")
        if parsed is not None:
            prediction[PRECOMPUTED_KEY] = (bundle.sha256, bundle.validate(parsed))


def score_structured_plan(
    predictions: list[dict[str, Any]],
    schema: dict[str, Any] | None,
    parse_json: Callable[[Any], Any],
    *,
    bundle: Any = None,
    pointers: list[str] | None = None,
) -> dict[str, Any]:
    """Score ``predictions`` and enrich every row in place with the per-row keys the UI reads."""
    pointer_tokens: dict[str, list[str]] = {}
    for ptr in pointers or []:
        tokens = split_json_pointer(ptr)
        if tokens is not None:
            pointer_tokens[ptr] = tokens

    total = n_valid_json = n_schema_valid = n_all_match = n_pass = 0
    per: dict[str, dict[str, int]] = {p: {"match": 0, "missing": 0, "support": 0, "both_missing": 0} for p in pointer_tokens}
    error_groups: dict[tuple[str, str, str], dict[str, Any]] = {}

    for prediction in predictions:
        total += 1
        parsed = parse_json(prediction.get("prediction") or "")
        parsed_ref = parse_json(prediction.get("reference") or "")
        valid_json = parsed is not None
        schema_ok, errors = _schema_check(bundle, parsed, prediction)
        acceptable = _acceptable_for(prediction)

        results: dict[str, dict[str, Any]] = {}
        for ptr, tokens in pointer_tokens.items():
            pred_values = resolve_json_pointer(parsed, tokens) if parsed is not None else []
            if ptr in acceptable and isinstance(acceptable[ptr], list) and acceptable[ptr]:
                gold_values, source, accept = list(acceptable[ptr]), "acceptable", list(acceptable[ptr])
            else:
                gold_values = resolve_json_pointer(parsed_ref, tokens) if parsed_ref is not None else []
                source, accept = "reference", None
            ok = pointer_match(pred_values, gold_values, accept)
            results[ptr] = {"match": ok, "pred": _cap(pred_values), "gold": _cap(gold_values), "source": source}
            bucket = per[ptr]
            bucket["support"] += 1
            bucket["match"] += int(ok)
            if not pred_values and not gold_values:
                bucket["both_missing"] += 1  # nothing to find on either side: not a miss
            else:
                bucket["missing"] += int(not pred_values)

        all_match = all(r["match"] for r in results.values())
        checks_pass = valid_json and schema_ok and all_match
        n_valid_json += int(valid_json)
        n_schema_valid += int(schema_ok)
        n_all_match += int(all_match)
        n_pass += int(checks_pass)

        for err in errors:
            key = (str(err.get("validator") or ""), _INDEX.sub("/*", str(err.get("pointer") or "")), str(err.get("message") or ""))
            group = error_groups.setdefault(
                key, {"message": key[2], "pointer": key[1], "validator": key[0], "count": 0}
            )
            group["count"] += 1

        prediction["scoring_mode"] = SCORING_MODE
        prediction["parsed_prediction"] = parsed
        prediction["parsed_reference"] = parsed_ref
        prediction["is_valid_json"] = valid_json
        prediction["row_schema_valid"] = schema_ok
        prediction["schema_errors"] = [
            {k: e.get(k) for k in ("pointer", "message", "validator")} for e in errors[:MAX_ROW_ERRORS]
        ]
        prediction["row_pointer_results"] = results
        prediction["row_structured_plan_checks_pass"] = checks_pass
        prediction[ROW_PASS_KEY] = checks_pass
        prediction["row_exact_match"] = 1.0 if checks_pass else 0.0

    top_errors = sorted(
        error_groups.values(), key=lambda g: (-g["count"], g["validator"], g["pointer"], g["message"])
    )[:MAX_TOP_ERRORS]
    metrics: dict[str, Any] = {
        "scoring_mode": SCORING_MODE,
        "total": total,
        "correct": n_pass,
        "exact_match": _rate(n_pass, total),
        "json_validity_rate": _rate(n_valid_json, total),
        "schema_valid_rate": _rate(n_schema_valid, total),
        "schema_errors_top": top_errors,
        "field_match_rate": _rate(n_all_match, total) if pointer_tokens else None,
        "per_pointer": {
            ptr: {
                "match_rate": _rate(b["match"], b["support"]),
                "support": b["support"],
                "missing_rate": _rate(b["missing"], b["support"] - b["both_missing"]),
                "both_missing": b["both_missing"],
            }
            for ptr, b in per.items()
        },
        "structured_plan_pass_rate": _rate(n_pass, total),
        "schema": {"root": bundle.root, "sha256": bundle.sha256} if bundle is not None else None,
        "pointers": list(pointer_tokens),
    }
    if bundle is None:
        metrics["schema_missing"] = True
    return metrics


def finalize_with_rubric(
    predictions: list[dict[str, Any]], metrics: dict[str, Any], rubric_block: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge a rubric result (``judge_rows_with_rubric``'s ``{"rubric": ...}``, or a ``no_judge`` status
    block) into ``metrics``. A row whose rubric verdict is a failure (``row_rubric_passed is False``)
    no longer passes; unjudged rows are unaffected. Recomputes ``row_structured_plan_pass`` /
    ``row_exact_match`` and the pass-rate metrics, and exposes ``rubric_pass_rate`` /
    ``rubric_mean_score`` for gates and lift."""
    block = (rubric_block or {}).get("rubric") if isinstance(rubric_block, dict) else None
    if not isinstance(block, dict):
        return metrics
    block = dict(block)
    block.setdefault("status", "judged")
    metrics["rubric"] = block
    n_pass = 0
    for prediction in predictions:
        checks = prediction.get("row_structured_plan_checks_pass")
        if not isinstance(checks, bool):
            continue
        passed = checks and prediction.get("row_rubric_passed") is not False
        prediction[ROW_PASS_KEY] = passed
        prediction["row_exact_match"] = 1.0 if passed else 0.0
        n_pass += int(passed)
    total = int(metrics.get("total") or 0)
    metrics["correct"] = n_pass
    metrics["exact_match"] = _rate(n_pass, total)
    metrics["structured_plan_pass_rate"] = _rate(n_pass, total)
    if block.get("status") == "judged":
        if block.get("pass_rate") is not None:
            metrics["rubric_pass_rate"] = block["pass_rate"]
        if block.get("mean_weighted") is not None:
            metrics["rubric_mean_score"] = block["mean_weighted"]
    return metrics


def rubric_view(prediction: dict[str, Any], parse_json: Callable[[Any], Any]) -> tuple[str, str, str, list[dict[str, Any]]]:
    """``(scenario, reference_text, prediction_text, passages)`` for ``judge_rows_with_rubric``."""
    row = prediction.get("plan_row") if isinstance(prediction.get("plan_row"), dict) else {}
    return (
        str(row.get("scenario") or ""),
        str(prediction.get("reference") or ""),
        str(prediction.get("prediction") or ""),
        row.get("passages") if isinstance(row.get("passages"), list) else [],
    )


def structured_plan_scoring_fingerprint(
    runtime_config: Any, judge_label: str | None = None, project_id: int | None = None
) -> str:
    """Hash of everything that changes how a plan is scored: the schema sha, the rubric sha, the sorted
    pointers, whether the rubric judge runs (and which judge), and the passage retrieval settings.
    Stamped into the eval metrics as ``structured_plan_fingerprint``; a cached base-model result is
    reused only when it matches (like ``remediation_scoring_fingerprint``)."""
    import hashlib
    import json

    from app.services import structured_plan_prompt as spp

    cfg = spp.effective_structured_plan_config(runtime_config, project_id)
    bundle = spp.bundle_for_config(cfg)
    rubric = spp.rubric_for_config(cfg)
    parts = {
        "schema": bundle.sha256 if bundle is not None else None,
        "rubric": rubric.sha256 if rubric is not None else None,
        "pointers": sorted(cfg["pointers"]),
        "judge_enabled": bool(cfg["judge_enabled"]) and rubric is not None,
        "judge": judge_label if (cfg["judge_enabled"] and rubric is not None) else None,
        "retrieve_passages": bool(cfg["retrieve_passages"]),
        "top_k": cfg["top_k"],
    }
    from app.services import tool_call_scoring as tcs

    if tcs.effective_tool_scoring_config(runtime_config) is not None:
        # Only present with an active tool catalog, so projects without one keep their fingerprint.
        parts["tools"] = tcs.tools_fingerprint(runtime_config)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]

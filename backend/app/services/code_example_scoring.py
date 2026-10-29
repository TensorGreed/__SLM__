"""Opt-in scoring of held-out ``code_example`` rows in a ``remediation`` eval.

A code-remediation project can hold task->code rows (``row_kind: "code_example"``: a task, a
language, reference passages and ``code_example_json: {code, rationale, citations}``). By default
``remediation_scoring`` skips them (``skipped_non_remediation_rows``). With the project setting
``runtime_config["remediation"]["score_code_examples"] = true`` the handler attaches the row's context
as ``prediction["code_example_row"]`` (``{task, language, passages}``) and these rows are scored here:

- ``output_parses``: the prediction parses as the code-example JSON shape (a ``code`` string).
- ``code_checks_pass``: the predicted code passes the code checks that ran AND passed on the
  REFERENCE code (``run_code_checks``, the registry the remediation path uses, same check
  selection). A row whose reference code no check ran on / passed is "not checkable" (``None``),
  never failed -- the same denominator for every model, like remediation's original-checkable rule.
- ``citations_valid``: on a row with passages, at least one citation and every cited id is one of
  the prompt's passage ids. A row without passages has nothing to cite (``None``, not failed).
- ``judge_pass``: the project's answer judge (annotated before scoring as ``_code_example_judge``
  by ``evaluation_service._judge_code_example_predictions``) says ``correct``; ``partial`` / ``wrong``
  fail. A row the judge did not score keeps ``None`` with a reason (``row_code_example_judge_unscored``).

``row_code_example_pass`` = parses AND (checks pass or not checkable) AND citations valid (or no
passages) AND judge pass. A row that fails a deterministic part fails whatever the judge says; a row
that passes them all but has no judge verdict is unscored (``None``: out of the pass rate and of the
paired lift, never silently passed).
"""

from __future__ import annotations

from typing import Any, Callable

ROW_KEY = "code_example_row"
MAX_CHECK_MESSAGES = 5

JUDGE_INSTRUCTION = (
    "The model was asked to write code for the task in the question, using the API shown in the reference "
    "passages when there are any. Grade its code and rationale against the reference code example. correct: the "
    "code does what the task asks (it may differ from the reference), calls APIs the way the reference or the "
    "passages use them (no invented functions, parameters or imports), introduces no security weakness, and "
    "would run as written. partial: it mostly does the task but has a real defect -- a wrong or invented API "
    "call, a missing step, an unsafe practice, or code that would not run without changes. wrong: it does not "
    "do the task, is mostly invented API, is unsafe, or is empty."
)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _get(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def row_context(prediction: Any) -> dict[str, Any] | None:
    """The code-example row behind a prediction (only present when the project opted in)."""
    if not isinstance(prediction, dict):
        return None
    row = prediction.get(ROW_KEY)
    return row if isinstance(row, dict) else None


def build_row_context(row: dict[str, Any], task: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
    """``prediction["code_example_row"]``: what the scorer and the judge read."""
    return {
        "task": str(task or ""),
        "language": row.get("language") if isinstance(row.get("language"), str) else None,
        "passages": passages,
    }


def _code_of(parsed: Any) -> str | None:
    return parsed.get("code") if isinstance(parsed, dict) and isinstance(parsed.get("code"), str) else None


def analyze_row(prediction: dict[str, Any], schema: dict[str, Any], parse_json: Callable[[Any], Any]) -> dict[str, Any]:
    """Parse the prediction and run the code checks (reference code first: which checks apply).
    Cached on the prediction as ``_code_example_analysis``."""
    cached = prediction.get("_code_example_analysis")
    if isinstance(cached, dict):
        return cached
    from app.services.code_check_service import run_code_checks

    row = row_context(prediction) or {}
    parsed = parse_json(prediction.get("prediction") or "")
    code = _code_of(parsed)
    ref_code = _code_of(parse_json(prediction.get("reference") or ""))
    out: dict[str, Any] = {
        "parsed": parsed if isinstance(parsed, dict) else None,
        "parses": code is not None,
        "checkable": False,
        "checks_pass": None,
        "messages": [],
    }
    remediation = schema.get("remediation") if isinstance(schema.get("remediation"), dict) else {}
    check_ids = remediation.get("checks")
    language = row.get("language") if isinstance(row.get("language"), str) else None
    if language and ref_code is not None:
        try:
            before = run_code_checks(ref_code, language=language, check_ids=check_ids, context={"path": None})
            ran = [_get(r, "check_id") for r in before.get("results", []) if _get(r, "ok") is not None]
            if before.get("ok") is True and ran:
                out["checkable"] = True
                if code is None:
                    out["checks_pass"] = False
                else:
                    after = run_code_checks(code, language=language, check_ids=ran, context={"path": None})
                    out["checks_pass"] = bool(after.get("ok") is True)
                    msgs: list[str] = []
                    for r in after.get("results", []):
                        if _get(r, "ok") is False:
                            for m in _get(r, "messages") or []:
                                msgs.append(f"{_get(r, 'check_id')}: {m}")
                    out["messages"] = msgs[:MAX_CHECK_MESSAGES]
        except Exception as exc:  # noqa: BLE001 - a check crash = not checkable, never an eval failure
            out["checkable"] = False
            out["checks_pass"] = None
            out["messages"] = [f"checks failed to run: {type(exc).__name__}"]
    prediction["_code_example_analysis"] = out
    return out


def precompute(predictions: list[dict[str, Any]], schema: dict[str, Any], parse_json: Callable[[Any], Any]) -> None:
    """Fill the analysis cache for every code-example row (blocking; run in a worker thread)."""
    for p in predictions:
        if row_context(p) is not None:
            analyze_row(p, schema, parse_json)


# ── judge ────────────────────────────────────────────────────────────────


def judge_question(row: dict[str, Any]) -> str:
    """The task, language and reference passages the judge reads as the question."""
    from app.services.code_remediation_prompt import _passages_block, normalize_passages

    parts = [f"Language: {row.get('language') or 'unknown'}", "", "Task:", str(row.get("task") or "").strip(), ""]
    parts.extend(_passages_block(normalize_passages(row.get("passages"))))
    return "\n".join(parts).strip()


def _example_text(parsed: Any, fallback: str) -> str:
    code = _code_of(parsed)
    if code is None:
        return fallback
    rationale = parsed.get("rationale") if isinstance(parsed.get("rationale"), str) else ""
    return f"CODE:\n{code.strip()}\nRATIONALE: {rationale.strip()}"


def judge_view(prediction: dict[str, Any], parse_json: Callable[[Any], Any]) -> tuple[str, str, str] | None:
    """``(question, reference, prediction)`` for the answer judge; ``None`` for a row that is not a
    code example."""
    row = row_context(prediction)
    if row is None:
        return None
    ref = str(prediction.get("reference") or "")
    pred = str(prediction.get("prediction") or "")
    return judge_question(row), _example_text(parse_json(ref), ref), _example_text(parse_json(pred), pred)


# ── scoring ──────────────────────────────────────────────────────────────


def score_code_examples(
    predictions: list[dict[str, Any]], schema: dict[str, Any], parse_json: Callable[[Any], Any]
) -> dict[str, Any]:
    """Score code-example rows in place; returns the ``code_example_*`` metric family."""
    from app.services.code_remediation_prompt import normalize_passages

    scored = decided = n_pass = 0
    n_parses = checkable = checks_ok = 0
    with_passages = cit_ok = 0
    judged = judge_ok = 0
    unscored: dict[str, int] = {}
    judge_label: str | None = None
    by_language: dict[str, dict[str, int]] = {}

    for prediction in predictions:
        row = row_context(prediction)
        if row is None:
            continue
        scored += 1
        a = analyze_row(prediction, schema, parse_json)
        parsed = a["parsed"]
        parses = bool(a["parses"])
        n_parses += int(parses)
        row_checkable = bool(a["checkable"])
        row_checks = (bool(a["checks_pass"]) if parses else False) if row_checkable else None
        if row_checkable:
            checkable += 1
            checks_ok += int(bool(row_checks))

        ids = [p["id"] for p in normalize_passages(row.get("passages"))]
        citations = parsed.get("citations") if isinstance(parsed, dict) else None
        cited = [str(c) for c in citations] if isinstance(citations, list) else []
        cit_valid: bool | None = None
        if ids:
            with_passages += 1
            cit_valid = parses and isinstance(citations, list) and bool(cited) and all(c in ids for c in cited)
            cit_ok += int(cit_valid)

        verdict: str | None = None
        reason: str | None = None
        unscored_reason: str | None = None
        ann = prediction.get("_code_example_judge")
        if isinstance(ann, dict):
            if ann.get("label"):
                judge_label = str(ann["label"])
            if isinstance(ann.get("verdict"), str):
                verdict = ann["verdict"]
                reason = ann.get("reason") if isinstance(ann.get("reason"), str) else None
            else:
                unscored_reason = str(ann.get("unscored") or "judge_no_verdict")
        else:
            unscored_reason = "judge_not_run"
        judge_pass: bool | None = None
        if verdict is not None:
            judged += 1
            judge_pass = verdict == "correct"
            judge_ok += int(judge_pass)
        else:
            unscored[unscored_reason or "judge_no_verdict"] = unscored.get(unscored_reason or "judge_no_verdict", 0) + 1

        deterministic = parses and (row_checks is not False) and (cit_valid is not False)
        passed: bool | None = (bool(judge_pass) if judge_pass is not None else None) if deterministic else False
        if passed is not None:
            decided += 1
            n_pass += int(passed)
        lang = str(row.get("language") or "unknown")
        bucket = by_language.setdefault(lang, {"scored": 0, "decided": 0, "passed": 0, "not_checkable": 0})
        bucket["scored"] += 1
        bucket["decided"] += int(passed is not None)
        bucket["passed"] += int(bool(passed))
        bucket["not_checkable"] += int(not row_checkable)

        prediction["parsed_prediction"] = parsed
        prediction["parsed_reference"] = parse_json(prediction.get("reference") or "")
        prediction["is_valid_json"] = parsed is not None
        prediction["scoring_mode"] = "remediation"
        prediction["row_kind"] = "code_example"
        prediction["row_code_example_output_parses"] = parses
        prediction["row_code_example_checkable"] = row_checkable
        prediction["row_code_example_checks_pass"] = row_checks
        prediction["row_code_example_check_messages"] = list(a["messages"])[:MAX_CHECK_MESSAGES]
        prediction["row_code_example_citations_valid"] = cit_valid
        prediction["row_code_example_judge_pass"] = judge_pass
        prediction["row_code_example_judge_verdict"] = verdict
        prediction["row_code_example_judge_reason"] = reason
        prediction["row_code_example_judge_unscored"] = None if verdict is not None else unscored_reason
        prediction["row_code_example_pass"] = passed
        prediction["row_cited_ids"] = cited
        prediction["row_passage_ids"] = ids
        if passed is not None:
            prediction["row_exact_match"] = 1.0 if passed else 0.0

    if scored == 0:
        return {}
    metrics: dict[str, Any] = {
        "code_example_count": scored,
        "code_example_decided_count": decided,
        "code_example_correct": n_pass,
        "code_example_pass_rate": _rate(n_pass, decided),
        "code_example_output_parses_rate": _rate(n_parses, scored),
        "code_example_checkable_count": checkable,
        "code_example_not_checkable_count": scored - checkable,
        "code_example_code_checks_pass_rate": _rate(checks_ok, checkable),
        "code_example_judged_count": judged,
        "code_example_judge_pass_rate": _rate(judge_ok, judged),
        "code_example_judge_unscored_count": scored - judged,
        "code_example_judge_unscored_reasons": dict(sorted(unscored.items())),
        "code_example_by_language": {
            lang: {**b, "pass_rate": _rate(b["passed"], b["decided"])} for lang, b in sorted(by_language.items())
        },
    }
    if with_passages:
        metrics["code_example_citations_valid_rate"] = _rate(cit_ok, with_passages)
        metrics["code_example_rows_with_passages"] = with_passages
    if judge_label:
        metrics["code_example_judge_label"] = judge_label
    return metrics

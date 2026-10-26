"""``remediation`` scoring: a model proposes ``{"patch", "rationale", "citations", ...}`` for one
finding in a code chunk. A patch passes when it applies to the chunk (``unified_diff``), the
patched code still parses wherever the original did (code-check registry), a configured detector no
longer reports the finding, and every cited passage id was one the model was given.

Called by ``StructuredExtractionHandler`` when ``output_schema.scoring_mode`` is ``remediation``.
Every row's context (``code``, ``start_line``, ``language``, ``path``, ``finding``, ``passages``)
arrives as ``prediction["remediation_row"]`` (copied across by
``evaluation_service._generate_heldout_predictions``).

The expensive part -- applying the patch, running the checks -- is ``analyze_row`` (cached on the
prediction as ``_remediation_analysis``); ``precompute`` runs it for every row in a worker thread
before scoring, and ``score_remediation`` computes whatever is missing. A detector
(``remediation_detector``) attaches ``_detector`` to a prediction before scoring.

Per-row keys written onto each prediction: ``row_malformed``, ``row_patch_applies``,
``row_patch_reason``, ``row_patch_offsets``, ``row_code_checkable``, ``row_code_parses``,
``row_check_messages``, ``row_issue_resolved``, ``row_new_findings``, ``row_detector``,
``row_citations_valid``, ``row_cited``, ``row_extra_fields``, ``row_remediation_pass``,
``row_exact_match`` (= pass), ``row_skipped`` (a row without a finding: a code example).
"""

from __future__ import annotations

from typing import Any, Callable

SCORING_MODE = "remediation"
MAX_CHECK_MESSAGES = 5


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _rate0(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _get(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def row_context(prediction: dict[str, Any]) -> dict[str, Any] | None:
    """The remediation row behind a prediction, or None for a row that is not a remediation
    task (no finding: e.g. a code example)."""
    row = prediction.get("remediation_row")
    if not isinstance(row, dict):
        return None
    finding = row.get("finding")
    if not (isinstance(finding, dict) and str(finding.get("label") or "").strip()):
        return None
    if not isinstance(row.get("code"), str):
        return None
    return row


def passage_ids(row: dict[str, Any]) -> list[str]:
    from app.services.code_remediation_prompt import normalize_passages

    return [p["id"] for p in normalize_passages(row.get("passages"))]


def _descendants(label: str, label_parents: dict[str, str]) -> set[str]:
    """``label`` plus every taxonomy descendant of it."""
    out = {label}
    for child in label_parents:
        cur, seen = child, set()
        while cur and cur not in seen:
            seen.add(cur)
            if cur == label:
                out.add(child)
                break
            cur = label_parents.get(cur)
    return out


# ── per-row analysis (patch application + code checks) ───────────────────


def analyze_row(
    prediction: dict[str, Any],
    schema: dict[str, Any],
    parse_json: Callable[[Any], Any],
) -> dict[str, Any]:
    """Parse the prediction, apply its patch to the chunk and run the code checks on the
    original and patched code. Cached on the prediction."""
    cached = prediction.get("_remediation_analysis")
    if isinstance(cached, dict):
        return cached
    from app.services.code_check_service import run_code_checks
    from app.services.unified_diff import DiffError, apply_patch_to_chunk

    row = row_context(prediction) or {}
    parsed = parse_json(prediction.get("prediction") or "")
    patch = parsed.get("patch") if isinstance(parsed, dict) else None
    malformed = not isinstance(patch, str)
    out: dict[str, Any] = {
        "parsed": parsed if isinstance(parsed, dict) else None,
        "malformed": malformed,
        "applies": False,
        "reason": "malformed" if malformed else None,
        "offsets": [],
        "patched_code": None,
        "checkable": False,
        "parses": None,
        "messages": [],
    }
    if not malformed:
        code = row.get("code") if isinstance(row.get("code"), str) else ""
        start = row.get("start_line") if isinstance(row.get("start_line"), int) and not isinstance(row.get("start_line"), bool) else 1
        try:
            applied = apply_patch_to_chunk(
                code, start, patch, path=row.get("path") if isinstance(row.get("path"), str) else None
            )
        except DiffError as exc:
            out["reason"] = getattr(exc, "code", None) or "not_a_diff"
        except Exception:  # noqa: BLE001 - a hostile patch must not fail the eval
            out["reason"] = "not_a_diff"
        else:
            if _get(applied, "ok"):
                out["applies"] = True
                out["reason"] = None
                out["patched_code"] = _get(applied, "patched_code")
                out["offsets"] = list(_get(applied, "offsets") or [])
            else:
                out["reason"] = _get(applied, "reason") or "not_a_diff"

    # Code checks: checkable = the ORIGINAL chunk ran >= 1 check and passed.
    check_ids = schema.get("remediation", {}).get("checks") if isinstance(schema.get("remediation"), dict) else None
    language = row.get("language") if isinstance(row.get("language"), str) else None
    context: dict[str, Any] = {"path": row.get("path")}
    if row and language:
        try:
            before = run_code_checks(row["code"], language=language, check_ids=check_ids, context=context)
            ran = [_get(r, "check_id") for r in before.get("results", []) if _get(r, "ok") is not None]
            if before.get("ok") is True and ran:
                out["checkable"] = True
                if out["applies"] and out["patched_code"] is not None:
                    after = run_code_checks(out["patched_code"], language=language, check_ids=ran, context=context)
                    out["parses"] = bool(after.get("ok") is True)
                    msgs: list[str] = []
                    for r in after.get("results", []):
                        if _get(r, "ok") is False:
                            for m in _get(r, "messages") or []:
                                msgs.append(f"{_get(r, 'check_id')}: {m}")
                    out["messages"] = msgs[:MAX_CHECK_MESSAGES]
                else:
                    out["parses"] = False
        except Exception as exc:  # noqa: BLE001 - a check crash = not checkable, never an eval failure
            out["checkable"] = False
            out["parses"] = None
            out["messages"] = [f"checks failed to run: {type(exc).__name__}"]
    prediction["_remediation_analysis"] = out
    return out


def precompute(
    predictions: list[dict[str, Any]], schema: dict[str, Any], parse_json: Callable[[Any], Any]
) -> None:
    """Fill the analysis cache for every remediation row (blocking; run in a worker thread)."""
    for p in predictions:
        if isinstance(p, dict) and row_context(p) is not None:
            analyze_row(p, schema, parse_json)


# ── detector verdicts ────────────────────────────────────────────────────


def detector_verdict(
    finding_label: str,
    detection: dict[str, Any] | None,
    label_parents: dict[str, str],
) -> tuple[bool | None, list[str] | None, str | None]:
    """``(issue_resolved, new_findings, detector_label)`` from a ``_detector`` annotation
    ``{"label", "original": [labels]|None, "patched": [labels]|None, "error"}``."""
    if not isinstance(detection, dict):
        return None, None, None
    label = detection.get("label")
    if detection.get("error") or detection.get("patched") is None or detection.get("original") is None:
        return None, None, label
    family = _descendants(finding_label, label_parents)
    patched = [str(x) for x in detection["patched"]]
    original = {str(x) for x in detection["original"]}
    if not any(str(x) in family for x in original):
        return None, sorted({x for x in patched if x not in original}), label  # detector_blind
    resolved = not any(x in family for x in patched)
    new = sorted({x for x in patched if x not in original})
    return resolved, new, label


def detector_is_blind(
    finding_label: str, detection: dict[str, Any] | None, label_parents: dict[str, str]
) -> bool:
    """The detector ran on the ORIGINAL chunk and did not report the finding there, so a clean
    patched chunk proves nothing."""
    if not isinstance(detection, dict) or detection.get("error"):
        return False
    if detection.get("patched") is None or detection.get("original") is None:
        return False
    family = _descendants(finding_label, label_parents)
    return not any(str(x) in family for x in detection["original"])


# ── scoring ──────────────────────────────────────────────────────────────


def _extra_field_results(
    parsed_pred: dict[str, Any] | None, parsed_ref: Any, schema: dict[str, Any]
) -> dict[str, bool] | None:
    from app.services.output_schema_service import MISSING, value_match

    props = schema.get("properties") or {}
    names = [n for n in schema.get("fields", []) if n in props]
    if not names or not isinstance(parsed_ref, dict):
        return None
    results: dict[str, bool] = {}
    for name in names:
        types = list((props.get(name) or {}).get("types") or [])
        pred_val = parsed_pred.get(name, MISSING) if isinstance(parsed_pred, dict) else MISSING
        ref_val = parsed_ref.get(name, MISSING)
        results[name] = bool(value_match(pred_val, ref_val, types))
    return results


def score_remediation(
    predictions: list[dict[str, Any]],
    schema: dict[str, Any],
    parse_json: Callable[[Any], Any],
) -> dict[str, Any]:
    """Score ``predictions`` and enrich every row in place with the per-row keys the UI reads."""
    label_parents = schema.get("label_parents") if isinstance(schema.get("label_parents"), dict) else {}
    base: dict[str, Any] = {"scoring_mode": SCORING_MODE, "schema": schema}

    scored = skipped = 0
    n_applies = n_malformed = n_valid_json = n_pass = 0
    checkable = parses = 0
    with_passages = cit_valid = cit_cited = 0
    detector_rows = resolved = with_new = 0
    detector_errors = 0
    detector_blind = det_unknown = 0
    detector_active = False
    detector_label: str | None = None
    reasons: dict[str, int] = {}
    extra_counts: dict[str, dict[str, int]] = {}

    for prediction in predictions:
        row = row_context(prediction)
        if row is None:
            skipped += 1
            prediction["scoring_mode"] = SCORING_MODE
            prediction["row_skipped"] = True
            continue
        scored += 1
        a = analyze_row(prediction, schema, parse_json)
        parsed_pred = a["parsed"]
        parsed_ref = parse_json(prediction.get("reference") or "")
        if parsed_pred is not None:
            n_valid_json += 1
        malformed = bool(a["malformed"])
        applies = bool(a["applies"])
        if malformed:
            n_malformed += 1
        if applies:
            n_applies += 1
        else:
            reasons[a["reason"] or "unknown"] = reasons.get(a["reason"] or "unknown", 0) + 1

        # checkable depends on the ORIGINAL chunk only (same denominator for every model);
        # a malformed / non-applying prediction is "does not parse" on a checkable row.
        row_checkable = bool(a["checkable"])
        row_parses = (False if malformed else bool(a["parses"])) if row_checkable else None
        if row_checkable:
            checkable += 1
            if row_parses:
                parses += 1

        resolved_flag: bool | None = None
        blind_row = False
        new_findings: list[str] | None = None
        det_label: str | None = None
        detection = prediction.get("_detector")
        if applies and isinstance(detection, dict):
            resolved_flag, new_findings, det_label = detector_verdict(
                str(row["finding"].get("label") or ""), detection, label_parents
            )
            detector_active = True
            if detection.get("error"):
                detector_errors += 1
                det_unknown += 1
            if det_label:
                detector_label = det_label
            if detector_is_blind(str(row["finding"].get("label") or ""), detection, label_parents):
                detector_blind += 1
                det_unknown += 1
                blind_row = True
            elif resolved_flag is not None:
                detector_rows += 1
                resolved += int(resolved_flag)
                with_new += int(bool(new_findings))

        ids = passage_ids(row)
        citations = (a["parsed"] or {}).get("citations") if a["parsed"] else None
        cited_list = [str(c) for c in citations] if isinstance(citations, list) else []
        cit_valid_flag: bool | None = None
        if ids:
            with_passages += 1
            cit_valid_flag = (not malformed) and isinstance(citations, (list, type(None))) and all(c in ids for c in cited_list)
            cit_valid += int(cit_valid_flag)
            cit_cited += int(bool(cited_list))

        extra = _extra_field_results(parsed_pred, parsed_ref, schema)
        if extra:
            for name, ok in extra.items():
                bucket = extra_counts.setdefault(name, {"correct": 0, "total": 0})
                bucket["total"] += 1
                bucket["correct"] += int(ok)

        passed = (
            not malformed
            and applies
            and (row_parses is not False if row_checkable else True)
            and (resolved_flag is not False)
            and (cit_valid_flag is not False)
        )
        n_pass += int(passed)

        prediction["parsed_prediction"] = parsed_pred
        prediction["parsed_reference"] = parsed_ref
        prediction["is_valid_json"] = parsed_pred is not None
        prediction["missing_required_fields"] = [
            f for f in (schema.get("required") or []) if not isinstance(parsed_pred, dict) or f not in parsed_pred
        ]
        prediction["scoring_mode"] = SCORING_MODE
        prediction["row_malformed"] = malformed
        prediction["row_patch_applies"] = applies
        prediction["row_patch_reason"] = None if applies else (a["reason"] or "unknown")
        prediction["row_patch_offsets"] = list(a["offsets"])
        prediction["row_code_checkable"] = row_checkable
        prediction["row_code_parses"] = row_parses
        prediction["row_check_messages"] = list(a["messages"])[:MAX_CHECK_MESSAGES]
        prediction["row_issue_resolved"] = resolved_flag
        prediction["row_detector_note"] = "detector_blind" if blind_row else None
        prediction["row_new_findings"] = new_findings
        prediction["row_detector"] = det_label
        prediction["row_citations_valid"] = cit_valid_flag
        prediction["row_cited"] = bool(cited_list)
        prediction["row_cited_ids"] = cited_list
        prediction["row_passage_ids"] = ids
        prediction["row_extra_fields"] = extra
        prediction["row_remediation_pass"] = passed
        prediction["row_exact_match"] = 1.0 if passed else 0.0
        prediction["row_finding"] = {
            k: row["finding"].get(k) for k in ("label", "start_line", "end_line") if k in row["finding"]
        }
        prediction["row_patch"] = (parsed_pred or {}).get("patch") if isinstance((parsed_pred or {}).get("patch"), str) else None
        prediction["row_passage_headers"] = [_passage_header(p) for p in _passages(row)]

    metrics: dict[str, Any] = {**base, "skipped_non_remediation_rows": skipped}
    if scored == 0:
        metrics.update({
            "remediation_pass_rate": 0.0, "patch_applies_rate": 0.0, "code_parses_rate": None,
            "code_checkable_count": 0, "json_validity_rate": 0.0, "malformed_rate": 0.0,
            "patch_reasons": {}, "total": 0, "correct": 0, "exact_match": 0.0,
        })
        return metrics

    metrics.update({
        "remediation_pass_rate": _rate0(n_pass, scored),
        "patch_applies_rate": _rate0(n_applies, scored),
        "code_parses_rate": _rate(parses, checkable),
        "code_checkable_count": checkable,
        "json_validity_rate": _rate0(n_valid_json, scored),
        "malformed_rate": _rate0(n_malformed, scored),
        "patch_reasons": dict(sorted(reasons.items())),
        "total": scored,
        "correct": n_pass,
        # The pass rate is also the legacy ``exact_match`` (gate / pass_rate fallback).
        "exact_match": _rate0(n_pass, scored),
    })
    if detector_rows or detector_errors or detector_label or detector_active:
        metrics["detector_evaluated_count"] = detector_rows
        if detector_blind:
            metrics["detector_blind_count"] = detector_blind
        if detector_rows:
            # prediction-dependent denominators (rows whose patch applied): informational only
            metrics["issue_resolved_rate"] = _rate0(resolved, detector_rows)
            metrics["new_findings_rate"] = _rate0(with_new, detector_rows)
        # Every scored row the detector could judge: rows that did not apply / were malformed count
        # as not resolved; detector-blind and detector-error rows are excluded. Same denominator
        # whatever the model, so this is the issue-resolution LIFT metric.
        overall_den = scored - det_unknown
        if overall_den > 0:
            metrics["issue_resolved_overall_rate"] = _rate0(resolved, overall_den)
        if detector_label:
            metrics["detector"] = detector_label
        if detector_errors:
            metrics["detector_error_count"] = detector_errors
            metrics["detector_error"] = f"{detector_errors} row(s) could not be checked by the detector"
    if with_passages:
        metrics["citations_valid_rate"] = _rate0(cit_valid, with_passages)
        metrics["citation_rate"] = _rate0(cit_cited, with_passages)
        metrics["rows_with_passages"] = with_passages
    if extra_counts:
        per = {
            name: {"correct": c["correct"], "total": c["total"], "accuracy": _rate0(c["correct"], c["total"])}
            for name, c in sorted(extra_counts.items())
        }
        metrics["extra_fields"] = per
        metrics["extra_field_accuracy_mean"] = round(sum(v["accuracy"] for v in per.values()) / len(per), 4)
    return metrics


def _passages(row: dict[str, Any]) -> list[dict[str, Any]]:
    from app.services.code_remediation_prompt import normalize_passages

    return normalize_passages(row.get("passages"))


def _passage_header(p: dict[str, Any]) -> dict[str, Any]:
    """Where a cited passage id points (no text): ``{id, where}``."""
    if p.get("path") and isinstance(p.get("start_line"), int) and isinstance(p.get("end_line"), int):
        where = f"{p['path']}:{p['start_line']}-{p['end_line']}"
        if p.get("symbol"):
            where += f" ({p['symbol']})"
    else:
        where = str(p.get("source") or p.get("path") or "document")
    return {"id": p["id"], "where": where}


# ── judge ────────────────────────────────────────────────────────────────

JUDGE_INSTRUCTION = (
    "The model was asked to fix one finding in a code chunk with a patch. Grade the patch and its rationale "
    "against the reference fix. correct: the patch removes the finding without breaking the code's behaviour "
    "(it may differ from the reference), and the rationale is accurate. partial: it removes the finding but "
    "changes behaviour it should not, or the rationale is wrong or vague. wrong: the finding remains, the patch "
    "breaks the code, or it does something unrelated. Also note whether any cited passages are relevant to the fix."
)


def _fix_text(parsed: Any, fallback: str) -> str:
    if isinstance(parsed, dict) and isinstance(parsed.get("patch"), str):
        rationale = parsed.get("rationale") if isinstance(parsed.get("rationale"), str) else ""
        return f"PATCH:\n{parsed['patch'].strip()}\nRATIONALE: {rationale.strip()}"
    return fallback


def judge_question(row: dict[str, Any]) -> str:
    """The rendered finding + numbered code the judge reads as the question."""
    from app.services.code_findings_prompt import render_numbered_code
    from app.services.code_remediation_prompt import _finding_block

    rendered, _range, _first = render_numbered_code(row.get("code"), row.get("start_line"))
    return "\n".join([*_finding_block(row.get("finding")), "Code:", rendered]).strip()


def judge_view(prediction: dict[str, Any], parse_json: Callable[[Any], Any]) -> tuple[str, str, str] | None:
    """``(question, reference, prediction)`` for the answer judge; ``None`` skips a non-remediation row."""
    row = row_context(prediction)
    if row is None:
        return None
    ref = str(prediction.get("reference") or "")
    pred = str(prediction.get("prediction") or "")
    return (
        judge_question(row),
        _fix_text(parse_json(ref), ref),
        _fix_text(parse_json(pred), pred),
    )


def retrieval_judge_rows(predictions: list[dict[str, Any]], parse_json: Callable[[Any], Any]) -> list[dict[str, Any]]:
    """Rows ``{prompt, reference, passages, row_judge_verdict}`` for the retrieval judge: only rows that have passages."""
    from app.services.code_remediation_prompt import normalize_passages

    rows: list[dict[str, Any]] = []
    for p in predictions:
        row = row_context(p) if isinstance(p, dict) else None
        if row is None:
            continue
        passages = [x["text"] for x in normalize_passages(row.get("passages"))]
        if not passages:
            continue
        ref = str(p.get("reference") or "")
        rows.append({
            "prompt": judge_question(row),
            "reference": _fix_text(parse_json(ref), ref),
            "passages": passages,
            "row_judge_verdict": p.get("row_judge_verdict"),
        })
    return rows

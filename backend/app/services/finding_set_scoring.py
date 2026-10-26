"""``finding_set`` scoring: a model proposes ``{"findings": [{label, start_line,
end_line, evidence, attrs}]}`` for a code chunk; a finding is correct when the
label is equal and the inclusive line ranges overlap with IoU >= threshold.

Pure functions (no DB, no handler state) so the scorer is reproducible from the
manifest alone: ``finding_match`` (threshold, attrs to compare, hierarchy,
parent credit) and ``label_parents`` are stamped at prep time. Called by
``StructuredExtractionHandler`` when ``output_schema.scoring_mode`` is
``finding_set``.
"""

from __future__ import annotations

from typing import Any, Callable

SCORING_MODE = "finding_set"
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_PARENT_CREDIT = 0.5


def tally_to_metrics(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """P/R/F1 from counts; empty-on-empty is trivially correct (1.0), as span_set."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def line_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    """IoU of two inclusive line ranges."""
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union if union > 0 else 0.0


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def normalise_findings(payload: Any) -> tuple[list[dict[str, Any]], int, bool]:
    """``(valid findings, invalid item count, has_findings_list)``. An item is
    invalid when it is not an object, has no label, has non-integer lines or
    start > end."""
    items = payload.get("findings") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return [], 0, False
    out: list[dict[str, Any]] = []
    invalid = 0
    for item in items:
        if not isinstance(item, dict):
            invalid += 1
            continue
        label = item.get("label")
        start, end = item.get("start_line"), item.get("end_line")
        if not isinstance(label, str) or not label.strip() or not _is_int(start) or not _is_int(end) or start > end:
            invalid += 1
            continue
        attrs = item.get("attrs")
        out.append({
            "label": label.strip(),
            "start_line": start,
            "end_line": end,
            "attrs": attrs if isinstance(attrs, dict) else {},
        })
    return out, invalid, True


def _match(
    preds: list[dict[str, Any]], refs: list[dict[str, Any]], threshold: float,
    *, ancestors: dict[str, set[str]] | None = None,
    pred_used: set[int] | None = None, ref_used: set[int] | None = None,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one pairs ``(pred_idx, ref_idx, iou)``, best IoU first
    (ties by index). With ``ancestors`` the pred label must be a proper
    ancestor of the ref label instead of equal to it. Indices in
    ``pred_used`` / ``ref_used`` are skipped."""
    pred_used = pred_used or set()
    ref_used = ref_used or set()
    cands: list[tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        if pi in pred_used:
            continue
        for ri, r in enumerate(refs):
            if ri in ref_used:
                continue
            if ancestors is None:
                if p["label"] != r["label"]:
                    continue
            elif p["label"] not in ancestors.get(r["label"], ()):
                continue
            iou = line_iou(p["start_line"], p["end_line"], r["start_line"], r["end_line"])
            if iou >= threshold:
                cands.append((iou, pi, ri))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    pairs: list[tuple[int, int, float]] = []
    used_p: set[int] = set()
    used_r: set[int] = set()
    for iou, pi, ri in cands:
        if pi in used_p or ri in used_r:
            continue
        used_p.add(pi)
        used_r.add(ri)
        pairs.append((pi, ri, iou))
    return pairs


def _ancestor_map(label_parents: dict[str, str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for label in label_parents:
        seen: set[str] = set()
        cur = label_parents.get(label)
        while cur and cur not in seen:
            seen.add(cur)
            cur = label_parents.get(cur)
        out[label] = seen
    return out


def _attr_equal(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    return a == b


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _brief(f: dict[str, Any]) -> dict[str, Any]:
    return {"label": f["label"], "start_line": f["start_line"], "end_line": f["end_line"]}


def _match_config(schema: dict[str, Any]) -> dict[str, Any]:
    raw = schema.get("finding_match") if isinstance(schema.get("finding_match"), dict) else {}
    thr = raw.get("iou_threshold", DEFAULT_IOU_THRESHOLD)
    credit = raw.get("parent_credit", DEFAULT_PARENT_CREDIT)
    attrs = raw.get("attrs")
    return {
        "iou_threshold": float(thr) if isinstance(thr, (int, float)) and not isinstance(thr, bool) else DEFAULT_IOU_THRESHOLD,
        "parent_credit": float(credit) if isinstance(credit, (int, float)) and not isinstance(credit, bool) else DEFAULT_PARENT_CREDIT,
        "attrs": [str(a) for a in attrs] if isinstance(attrs, list) else [],
        "hierarchy": bool(raw.get("hierarchy")),
    }


def score_finding_set(
    predictions: list[dict[str, Any]],
    schema: dict[str, Any],
    parse_json: Callable[[Any], Any],
) -> dict[str, Any]:
    """Score ``predictions`` (each with ``prediction`` / ``reference`` strings)
    and enrich every row in place with the per-row keys the UI reads."""
    cfg = _match_config(schema)
    thr = cfg["iou_threshold"]
    attr_names: list[str] = cfg["attrs"]
    hierarchy: bool = cfg["hierarchy"]
    credit = cfg["parent_credit"]
    parents = schema.get("label_parents") if isinstance(schema.get("label_parents"), dict) else {}
    ancestors = _ancestor_map({str(k): str(v) for k, v in parents.items()}) if hierarchy else {}

    total = len(predictions)
    base: dict[str, Any] = {"scoring_mode": SCORING_MODE, "iou_threshold": thr, "schema": schema}
    if total == 0:
        return {
            **base,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0,
            "per_class": {}, "tp_total": 0, "fp_total": 0, "fn_total": 0, "total": 0, "correct": 0,
            "clean_chunk_count": 0, "clean_chunk_fp_count": 0, "clean_chunk_fp_rate": 0.0,
            "malformed_count": 0, "malformed_rate": 0.0, "invalid_finding_count": 0,
            "json_validity_rate": 0.0, "exact_match": 0.0,
            "attr_accuracy": {}, "attr_accuracy_mean": None,
        }

    g_tp = g_fp = g_fn = 0
    per_class: dict[str, dict[str, int]] = {}
    clean = clean_fp = malformed = invalid_total = valid_json = exact_rows = 0
    attr_counts: dict[str, dict[str, int]] = {a: {"correct": 0, "total": 0} for a in attr_names}
    partial_matches = 0

    def cls(label: str) -> dict[str, int]:
        return per_class.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})

    for prediction in predictions:
        parsed_pred = parse_json(prediction.get("prediction") or "")
        parsed_ref = parse_json(prediction.get("reference") or "")
        is_valid = isinstance(parsed_pred, dict)
        if is_valid:
            valid_json += 1
        preds, invalid, has_list = normalise_findings(parsed_pred if is_valid else None)
        row_malformed = not has_list
        if row_malformed:
            malformed += 1
        invalid_total += invalid
        refs, _ref_invalid, _ = normalise_findings(parsed_ref)

        pairs = _match(preds, refs, thr)
        matched_p = {pi for pi, _, _ in pairs}
        matched_r = {ri for _, ri, _ in pairs}
        row_tp = len(pairs)
        row_fp = len(preds) - row_tp
        row_fn = len(refs) - row_tp

        row_clean = len(refs) == 0
        if row_clean:
            clean += 1
            if preds:
                clean_fp += 1

        for pi, ri, _iou in pairs:
            cls(refs[ri]["label"])["tp"] += 1
            for name in attr_names:
                if name in refs[ri]["attrs"]:
                    attr_counts[name]["total"] += 1
                    if name in preds[pi]["attrs"] and _attr_equal(preds[pi]["attrs"][name], refs[ri]["attrs"][name]):
                        attr_counts[name]["correct"] += 1
        for pi, p in enumerate(preds):
            if pi not in matched_p:
                cls(p["label"])["fp"] += 1
        for ri, r in enumerate(refs):
            if ri not in matched_r:
                cls(r["label"])["fn"] += 1

        if hierarchy and ancestors:
            partial = _match(preds, refs, thr, ancestors=ancestors, pred_used=matched_p, ref_used=matched_r)
            partial_matches += len(partial)

        g_tp += row_tp
        g_fp += row_fp
        g_fn += row_fn

        row_p, row_r, row_f1 = tally_to_metrics(row_tp, row_fp, row_fn)
        row_exact = 1.0 if (not row_malformed and row_fp == 0 and row_fn == 0) else 0.0
        if row_exact:
            exact_rows += 1

        prediction["parsed_prediction"] = parsed_pred
        prediction["parsed_reference"] = parsed_ref
        prediction["is_valid_json"] = is_valid
        prediction["missing_required_fields"] = [] if has_list else ["findings"]
        prediction["scoring_mode"] = SCORING_MODE
        prediction["row_malformed"] = row_malformed
        prediction["row_is_clean_reference"] = row_clean
        prediction["row_matched_findings"] = [
            {
                **_brief(refs[ri]),
                "pred_start_line": preds[pi]["start_line"],
                "pred_end_line": preds[pi]["end_line"],
                "iou": round(iou, 4),
            }
            for pi, ri, iou in sorted(pairs, key=lambda t: t[1])
        ]
        prediction["row_missed_findings"] = [_brief(r) for ri, r in enumerate(refs) if ri not in matched_r]
        prediction["row_spurious_findings"] = [_brief(p) for pi, p in enumerate(preds) if pi not in matched_p]
        prediction["row_precision"] = round(row_p, 4)
        prediction["row_recall"] = round(row_r, 4)
        prediction["row_f1"] = round(row_f1, 4)
        prediction["row_exact_match"] = row_exact

    precision, recall, f1 = tally_to_metrics(g_tp, g_fp, g_fn)
    per_class_summary: dict[str, dict[str, Any]] = {}
    for label in sorted(per_class):
        c = per_class[label]
        p, r, f = tally_to_metrics(c["tp"], c["fp"], c["fn"])
        per_class_summary[label] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "support": c["tp"] + c["fn"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        }
    n_cls = len(per_class_summary)

    micro = {"precision": precision, "recall": recall, "f1": f1}

    def macro(key: str) -> float:
        if not n_cls:  # nothing predicted, nothing expected: perfect, same as the micro score
            return round(micro[key], 4)
        return round(sum(v[key] for v in per_class_summary.values()) / n_cls, 4)

    attr_accuracy = {
        name: {"correct": v["correct"], "total": v["total"], "accuracy": _rate(v["correct"], v["total"])}
        for name, v in attr_counts.items()
    }
    scored_attrs = [v["accuracy"] for v in attr_accuracy.values() if v["total"] > 0]

    metrics: dict[str, Any] = {
        **base,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "precision_macro": macro("precision"),
        "recall_macro": macro("recall"),
        "f1_macro": macro("f1"),
        "per_class": per_class_summary,
        "tp_total": g_tp,
        "fp_total": g_fp,
        "fn_total": g_fn,
        "total": total,
        "correct": exact_rows,
        "clean_chunk_count": clean,
        "clean_chunk_fp_count": clean_fp,
        "clean_chunk_fp_rate": _rate(clean_fp, clean),
        "malformed_count": malformed,
        "malformed_rate": _rate(malformed, total),
        "invalid_finding_count": invalid_total,
        "json_validity_rate": _rate(valid_json, total),
        "schema_compliance_rate": _rate(total - malformed, total),
        "exact_match": _rate(exact_rows, total),
        "attr_accuracy": attr_accuracy,
        "attr_accuracy_mean": round(sum(scored_attrs) / len(scored_attrs), 4) if scored_attrs else None,
    }
    if hierarchy:
        h_tp = g_tp + credit * partial_matches
        h_fp = g_fp - credit * partial_matches
        h_fn = g_fn - credit * partial_matches
        hp, hr, hf = tally_to_metrics(h_tp, h_fp, h_fn)
        metrics["hierarchical"] = {
            "precision": round(hp, 4), "recall": round(hr, 4), "f1": round(hf, 4),
            "partial_matches": partial_matches, "parent_credit": credit,
        }
    return metrics

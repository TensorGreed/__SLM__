"""Per-slice pass counts on an eval result, with Wilson 95% intervals (``metrics["slice_breakdown"]``).

Computed once every row verdict is final (after tool-call / catalog / SQL / judge scoring), so a
slice's numbers are the same row verdicts the result's headline counts. Generic: slices come from

  * the rows' own tags, read from the evaluated file (``row_slice_tags`` on each prediction):
    ``slice`` / ``slices`` / ``tags`` at the top level or under ``metadata`` (a string, or a list of
    strings; a ``tags`` string may be comma-separated; a list longer than ``MAX_ROW_TAGS``, with repeats or
    with BIO-shaped items is a per-token label sequence and is ignored), and
  * the project's slice definitions (``Project.slice_definitions``), matched on the scored rows.

Each slice reports the row pass (the same rule as the result's failures list) and, when the rows carry
it, the opt-in strict pass (``row_tool_call_strict_pass``). Rows the scorer left out (unscored) count in
``rows`` but not in ``scored``. Absent when no slice exists, so other results are unchanged.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

TAG_KEYS = ("slice", "slices", "tags")
MAX_TAG_SLICES = 50
MAX_TAG_LEN = 80
Z = 1.959963984540054

# Extra row-level verdicts reported beside the pass when present on the rows: (key, metric id, label).
EXTRA_VERDICTS: tuple[tuple[str, str, str], ...] = (
    ("row_tool_call_strict_pass", "strict", "Strict pass"),
)


def wilson(passed: int, n: int) -> list[float] | None:
    """Wilson score 95% interval for ``passed`` of ``n``; None when ``n`` is 0."""
    if n <= 0:
        return None
    p = passed / n
    z2 = Z * Z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


MAX_ROW_TAGS = 10
_BIO_LABEL = re.compile(r"^(O|[BIESLU]-.+)$")


def _is_row_tag_list(value: Any) -> bool:
    """A short list of row-level tags, not a per-token label sequence (``["O", "B-PER", "I-PER"]``): at most
    ``MAX_ROW_TAGS`` scalar items, no repeats, and not BIO / BIOES-shaped labels."""
    items = list(value)
    if not items or len(items) > MAX_ROW_TAGS:
        return False
    if not all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in items):
        return False
    texts = [str(x).strip() for x in items]
    if len(set(texts)) != len(texts):
        return False
    return not any(_BIO_LABEL.match(t) for t in texts)


def _tag_values(value: Any, *, split_commas: bool) -> list[str]:
    raw: list[Any]
    if isinstance(value, str):
        raw = value.split(",") if split_commas else [value]
    elif isinstance(value, (list, tuple)):
        if not _is_row_tag_list(value):
            return []
        raw = list(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        raw = [str(value)]
    else:
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            item = str(item)
        if isinstance(item, str):
            text = item.strip()[:MAX_TAG_LEN]
            if text and text not in out:
                out.append(text)
    return out


def row_slice_tags(row: Any) -> list[list[str]] | None:
    """``[[field, value], ...]`` read from one evaluated row (None when it carries no tag)."""
    if not isinstance(row, dict):
        return None
    found: list[list[str]] = []

    def collect(source: dict[str, Any], prefix: str) -> None:
        for key in TAG_KEYS:
            if key in source:
                for value in _tag_values(source[key], split_commas=key == "tags"):
                    pair = [f"{prefix}{key}", value]
                    if pair not in found:
                        found.append(pair)

    collect(row, "")
    meta = row.get("metadata")
    if isinstance(meta, dict):
        collect(meta, "metadata.")
    return found or None


def _verdicts(
    predictions: list[dict[str, Any]],
    failed: Callable[[dict[str, Any]], bool],
    unscored: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in predictions:
        if not isinstance(p, dict) or p.get("row_skipped"):
            out.append({"skip": True})
            continue
        entry: dict[str, Any] = {"pass": None if unscored(p) else (not failed(p))}
        for key, metric, _label in EXTRA_VERDICTS:
            if isinstance(p.get(key), bool):
                entry[metric] = p[key]
        out.append(entry)
    return out


def _summarise(indexes: list[int], verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [verdicts[i] for i in indexes if not verdicts[i].get("skip")]
    scored = [v for v in rows if isinstance(v.get("pass"), bool)]
    passed = sum(1 for v in scored if v["pass"])
    out: dict[str, Any] = {
        "rows": len(rows),
        "scored": len(scored),
        "pass": passed,
        "pass_rate": round(passed / len(scored), 4) if scored else None,
        "pass_ci": wilson(passed, len(scored)),
    }
    for _key, metric, _label in EXTRA_VERDICTS:
        values = [v[metric] for v in rows if isinstance(v.get(metric), bool)]
        if values:
            n_ok = sum(1 for v in values if v)
            out[metric] = {
                "scored": len(values),
                "pass": n_ok,
                "rate": round(n_ok / len(values), 4),
                "ci": wilson(n_ok, len(values)),
            }
    return out


def build_slice_breakdown(
    predictions: list[dict[str, Any]],
    *,
    failed: Callable[[dict[str, Any]], bool],
    unscored: Callable[[dict[str, Any]], bool],
    slice_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The ``slice_breakdown`` block, or None when no slice applies (no row tags, no project slices)."""
    preds = [p for p in predictions or []]
    if not preds:
        return None
    verdicts = _verdicts(preds, failed, unscored)
    slices: list[dict[str, Any]] = []

    by_tag: dict[tuple[str, str], list[int]] = {}
    for i, p in enumerate(preds):
        tags = p.get("row_slice_tags") if isinstance(p, dict) else None
        for pair in tags or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                by_tag.setdefault((str(pair[0]), str(pair[1])), []).append(i)
    tag_items = sorted(by_tag.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    omitted = max(0, len(tag_items) - MAX_TAG_SLICES)
    for (field, value), indexes in sorted(tag_items[:MAX_TAG_SLICES], key=lambda kv: kv[0]):
        slices.append({"id": f"{field}={value}", "source": "row_tag", "field": field, "label": value,
                       **_summarise(indexes, verdicts)})

    if slice_definitions:
        from app.services.slice_evaluator_service import inject_platform_fields, _matches_all

        enriched = [inject_platform_fields(p, index=i) if isinstance(p, dict) else {} for i, p in enumerate(preds)]
        for sd in slice_definitions:
            if not isinstance(sd, dict) or not sd.get("slice_id"):
                continue
            try:
                indexes = [i for i, row in enumerate(enriched) if _matches_all(row, sd.get("where") or [])]
            except Exception:  # noqa: BLE001 - a malformed stored slice must not fail the eval
                continue
            slices.append({"id": str(sd["slice_id"]), "source": "slice_definition", "field": None,
                           "label": str(sd.get("display_name") or sd["slice_id"]),
                           **_summarise(indexes, verdicts)})

    if not slices:
        return None
    block: dict[str, Any] = {
        "version": 1,
        "interval": "wilson_95",
        "overall": _summarise(list(range(len(preds))), verdicts),
        "slices": slices,
    }
    if omitted:
        block["tag_slices_omitted"] = omitted
    return block

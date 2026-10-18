"""Benchmark gold-VALUE contamination (the honesty guard behind a benchmark score).

The row-level leakage filter drops benchmark rows that *are* training rows.
This scan asks a different question: do the benchmark's **answers** — the
entity values a span benchmark expects, the answers a Q&A / summarization
benchmark expects — appear verbatim inside the project's training text?

Why it matters (2026-10-17): 32 values from the Gretel PII benchmark ended
up 591 times in project 31's training rows, because the seed-document
prompts carried example values lifted from the benchmark and the local
seeder reproduced them word for word. The shipped model's per-type figures
on that benchmark were overstated by 0.03–0.15: it had memorised the
answers, not learnt the type. Nothing in the pipeline could tell.

Pure functions over rows; ``external_benchmark_service`` runs
``benchmark_contamination`` at import, stores it on the manifest, and
re-runs it on request (the training corpus keeps changing after import).

Honest reading of the numbers: a verbatim hit is evidence, not proof —
``John`` from a news benchmark will legitimately occur in support tickets.
``severity`` is a share of the benchmark's distinct values, so one common
name stays ``low``; a long list of ids, plates and addresses that all occur
in the training text reads ``severe``. Values shorter than
``MIN_VALUE_CHARS`` are not scanned (too common to mean anything).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

MIN_VALUE_CHARS = 4
MAX_VALUES = 5000
EXAMPLES = 10
# found_share thresholds (share of the benchmark's distinct values found).
SEVERITY_NOTABLE = 0.02
SEVERITY_SEVERE = 0.10

SHAPE_CLASSIFICATION = "classification"
_GENERATIVE = frozenset({"qa", "summarization"})


def normalise_value(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def row_text(row: dict[str, Any]) -> str:
    """The text a training row contributes (same reading as the leakage filter)."""
    if "text" in row:
        return str(row.get("text") or "")
    from app.services.trainability_forecast_service import _row_to_text

    return _row_to_text(row)


def row_source(row: dict[str, Any]) -> str:
    """Where a training row came from, for the ``by_source`` breakdown."""
    for key in ("synth_source", "source", "_source_type", "dataset_type"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "training"


def gold_values(rows: list[dict[str, Any]], task_profile: str | None) -> list[tuple[str, str]]:
    """``(value, kind)`` for every answer the benchmark expects. Spans → each
    entity's text with its type; Q&A / summarization → the answer; a
    classification benchmark has labels, not values → ``[]``."""
    out: list[tuple[str, str]] = []
    if task_profile == SHAPE_CLASSIFICATION:
        return out
    for row in rows:
        if task_profile in _GENERATIVE:
            answer = row.get("answer")
            if isinstance(answer, str) and answer.strip():
                out.append((answer.strip(), "answer"))
            continue
        entities = row.get("entities")
        text = row.get("text") if isinstance(row.get("text"), str) else ""
        if not isinstance(entities, list):
            continue
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            value = ent.get("text")
            if not isinstance(value, str) or not value.strip():
                try:
                    value = text[int(ent["start"]):int(ent["end"])]
                except (KeyError, TypeError, ValueError):
                    continue
            if value.strip():
                out.append((value.strip(), str(ent.get("type") or ent.get("label") or "entity")))
    return out


def severity_of(found_share: float, values_found: int) -> str:
    if values_found <= 0:
        return "clean"
    if found_share >= SEVERITY_SEVERE:
        return "severe"
    if found_share >= SEVERITY_NOTABLE:
        return "notable"
    return "low"


def scan_values(values: list[tuple[str, str]], training_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count, for every distinct gold value, the training rows and the
    occurrences that contain it verbatim (whitespace- and case-insensitive)."""
    distinct: dict[str, tuple[str, str]] = {}
    for raw, kind in values:
        key = normalise_value(raw)
        if len(key) < MIN_VALUE_CHARS:
            continue
        if key not in distinct:
            distinct[key] = (raw, kind)
        if len(distinct) >= MAX_VALUES:
            break
    haystack = [(normalise_value(row_text(r)), row_source(r)) for r in training_rows]
    haystack = [(t, s) for t, s in haystack if t]
    # One joined scan per value first; the per-row pass only for values that hit.
    joined = "\n".join(t for t, _ in haystack)
    checked_by_type: Counter[str] = Counter(kind for _, kind in distinct.values())
    found_by_type: Counter[str] = Counter()
    occ_by_type: Counter[str] = Counter()
    rows_by_source: Counter[str] = Counter()
    hits: list[dict[str, Any]] = []
    rows_hit: set[int] = set()
    total_occurrences = 0
    for key, (raw, kind) in distinct.items():
        if key not in joined:
            continue
        occurrences = 0
        row_count = 0
        sources: Counter[str] = Counter()
        for i, (text, source) in enumerate(haystack):
            n = text.count(key)
            if n:
                occurrences += n
                row_count += 1
                rows_hit.add(i)
                sources[source] += 1
        if not occurrences:
            continue
        total_occurrences += occurrences
        found_by_type[kind] += 1
        occ_by_type[kind] += occurrences
        rows_by_source.update(sources)
        hits.append({"value": raw, "type": kind, "occurrences": occurrences, "rows": row_count,
                     "sources": dict(sources.most_common(3))})
    hits.sort(key=lambda h: (-h["occurrences"], h["value"]))
    checked = len(distinct)
    found = len(hits)
    share = (found / checked) if checked else 0.0
    return {
        "values_checked": checked,
        "values_found": found,
        "found_share": round(share, 4),
        "occurrences": total_occurrences,
        "rows_hit": len(rows_hit),
        "checked_against": len(haystack),
        "by_type": {
            t: {"checked": checked_by_type[t], "found": found_by_type.get(t, 0), "occurrences": occ_by_type.get(t, 0)}
            for t in sorted(checked_by_type)
        },
        "by_source": dict(rows_by_source.most_common()),
        "examples": hits[:EXAMPLES],
        "severity": severity_of(share, found),
    }


def benchmark_contamination(
    benchmark_rows: list[dict[str, Any]], task_profile: str | None, training_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """The block stored on the benchmark manifest as ``contamination``."""
    stamp = datetime.now(timezone.utc).isoformat()
    values = gold_values(benchmark_rows, task_profile)
    if task_profile == SHAPE_CLASSIFICATION or not values:
        return {
            "applicable": False,
            "reason": "no_gold_values" if task_profile != SHAPE_CLASSIFICATION else "classification_labels_are_not_values",
            "checked_against": len(training_rows),
            "scanned_at": stamp,
        }
    block = scan_values(values, training_rows)
    block["applicable"] = True
    block["scanned_at"] = stamp
    return block


def contamination_line(block: dict[str, Any] | None) -> str | None:
    """One sentence for cards and release notes; None when there is nothing to say."""
    if not isinstance(block, dict) or not block.get("applicable") or not block.get("values_found"):
        return None
    return (
        f"{block['values_found']} of {block['values_checked']} gold values appear verbatim in the training rows "
        f"({block['occurrences']} occurrences in {block['rows_hit']} rows; {block['severity']}) — scores on those "
        f"values may reflect memorisation rather than skill."
    )

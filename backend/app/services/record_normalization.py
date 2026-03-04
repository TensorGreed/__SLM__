"""Generic record normalization utilities for arbitrary tabular/text datasets."""

from __future__ import annotations

import json
from collections import Counter
from statistics import median
from typing import Any

CANONICAL_KEYS = {"text", "question", "answer"}

DEFAULT_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "text": (
        "text",
        "content",
        "body",
        "document",
        "passage",
        "chunk",
        "context",
        "article",
        "message",
    ),
    "question": (
        "question",
        "prompt",
        "query",
        "instruction",
        "input",
        "ask",
    ),
    "answer": (
        "answer",
        "response",
        "output",
        "target",
        "completion",
        "label",
    ),
}

NON_CONTENT_KEYS = {
    "id",
    "uuid",
    "source",
    "split",
    "timestamp",
    "created_at",
    "updated_at",
    "metadata",
}


def _coerce_text(value: Any) -> str | None:
    """Convert value to text when possible."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_coerce_text(v) for v in value]
        joined = "\n".join(p for p in parts if p)
        return joined.strip() or None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return None


def _get_nested_value(record: dict, path: str) -> Any:
    """Resolve dot-path into nested dictionaries/lists."""
    parts = [p for p in path.split(".") if p]
    current: Any = record
    for part in parts:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
                continue
            # Case-insensitive fallback.
            lowered = {str(k).lower(): k for k in current.keys()}
            real_key = lowered.get(part.lower())
            if real_key is None:
                return None
            current = current[real_key]
            continue
        if isinstance(current, list):
            if not part.isdigit():
                return None
            idx = int(part)
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


def _pick_from_aliases(record: dict, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        value = _get_nested_value(record, alias)
        coerced = _coerce_text(value)
        if coerced:
            return coerced
    return None


def canonicalize_record(
    raw_record: dict[str, Any],
    field_mapping: dict[str, str] | None = None,
    keep_raw: bool = False,
) -> dict[str, Any] | None:
    """
    Convert heterogeneous input record into canonical training shape.

    Canonical output keys:
    - text (always required)
    - question / answer (optional, when available)
    """
    if not isinstance(raw_record, dict):
        return None

    normalized: dict[str, Any] = {}

    # Explicit mappings take highest precedence.
    if field_mapping:
        for canonical_key, source_path in field_mapping.items():
            if canonical_key not in CANONICAL_KEYS:
                continue
            if not source_path:
                continue
            mapped_value = _coerce_text(_get_nested_value(raw_record, source_path))
            if mapped_value:
                normalized[canonical_key] = mapped_value

    # Heuristic inference for missing fields.
    for canonical_key, aliases in DEFAULT_FIELD_CANDIDATES.items():
        if canonical_key in normalized:
            continue
        inferred = _pick_from_aliases(raw_record, aliases)
        if inferred:
            normalized[canonical_key] = inferred

    question = normalized.get("question")
    answer = normalized.get("answer")
    text = normalized.get("text")

    if not text and question and answer:
        text = f"Question: {question}\nAnswer: {answer}"
    elif not text and question:
        text = question
    elif not text and answer:
        text = answer

    # Final fallback: merge a few non-empty top-level string fields.
    if not text:
        fallback_parts: list[str] = []
        for key, value in raw_record.items():
            if str(key).lower() in NON_CONTENT_KEYS:
                continue
            part = _coerce_text(value)
            if part:
                fallback_parts.append(part)
            if len(fallback_parts) >= 3:
                break
        if fallback_parts:
            text = "\n".join(fallback_parts).strip()

    if not text:
        return None

    normalized["text"] = text
    if keep_raw:
        normalized["_raw"] = raw_record
    return normalized


def normalize_records(
    records: list[dict[str, Any]],
    field_mapping: dict[str, str] | None = None,
    keep_raw: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Normalize records and return (normalized_records, dropped_count)."""
    normalized: list[dict[str, Any]] = []
    dropped = 0
    for record in records:
        normalized_row = canonicalize_record(
            raw_record=record,
            field_mapping=field_mapping,
            keep_raw=keep_raw,
        )
        if normalized_row:
            normalized.append(normalized_row)
        else:
            dropped += 1
    return normalized, dropped


def build_schema_profile(
    records: list[dict[str, Any]],
    field_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a quick schema + quality profile for dataset inspection."""
    field_counter: Counter[str] = Counter()
    sample_lengths: list[int] = []
    normalized_count = 0

    for record in records:
        if isinstance(record, dict):
            field_counter.update(str(k) for k in record.keys())
        normalized = canonicalize_record(record, field_mapping=field_mapping) if isinstance(record, dict) else None
        if normalized:
            normalized_count += 1
            sample_lengths.append(len(normalized["text"]))

    total = len(records)
    dropped = max(0, total - normalized_count)
    length_stats = {
        "avg_chars": round(sum(sample_lengths) / len(sample_lengths), 2) if sample_lengths else 0.0,
        "p50_chars": median(sample_lengths) if sample_lengths else 0.0,
        "max_chars": max(sample_lengths) if sample_lengths else 0,
    }

    return {
        "total_records": total,
        "normalized_records": normalized_count,
        "dropped_records": dropped,
        "normalization_coverage": round((normalized_count / total) * 100, 2) if total else 0.0,
        "top_fields": field_counter.most_common(20),
        "text_length": length_stats,
    }

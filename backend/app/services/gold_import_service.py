"""Parse an uploaded gold file (JSONL / NDJSON / JSON array / CSV / TSV) into rows plus grouped rejects.

Task-agnostic: a row is any JSON object with some content (question+answer, text+label,
text+entities, tool_name+tool_args, SQL, messages, ...). Nothing here knows a recipe; the evaluator's
field aliases read the stored rows downstream, as for ``POST /gold/import`` and ``/gold/add``.

A reject is ``(line, reason, preview)``. ``line`` is the file line for JSONL, the 1-based record
number for a JSON array, and the file line for CSV / TSV (the header is line 1).
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

REASON_INVALID_JSON = "invalid_json"
REASON_NOT_AN_OBJECT = "not_an_object"
REASON_EMPTY_ROW = "empty_row"
REASON_WRONG_COLUMNS = "wrong_column_count"

REASON_TEXT = {
    REASON_INVALID_JSON: "Not valid JSON",
    REASON_NOT_AN_OBJECT: "Not a JSON object (one row per object)",
    REASON_EMPTY_ROW: "No content (every field is empty)",
    REASON_WRONG_COLUMNS: "Wrong number of columns (a different count than the header row)",
}

# Fields the platform owns or defaults; a row with only these has no content.
_SYSTEM_KEYS = {"id", "dataset_type", "difficulty", "criticality", "is_hallucination_trap", "metadata", "created_at"}
SAMPLE_CAP = 500
PREVIEW_CHARS = 120


class GoldFileError(ValueError):
    """The file as a whole cannot be read (unknown / unreadable format)."""


def _has_content(row: dict[str, Any]) -> bool:
    return any(
        k not in _SYSTEM_KEYS and v is not None and (not isinstance(v, str) or v.strip()) and v != [] and v != {}
        for k, v in row.items()
    )


def _apply_mapping(row: dict[str, Any], mapping: dict[str, str] | None) -> dict[str, Any]:
    if not mapping:
        return row
    return {mapping.get(key, key): value for key, value in row.items()}


def _cell(value: str) -> Any:
    """A CSV cell that is a JSON array / object (entities, tool arguments) becomes that value."""
    text = value.strip()
    if text[:1] in ("{", "[") and text[-1:] in ("}", "]"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def parse_gold_file(
    filename: str, content: bytes, field_mapping: dict[str, str] | None = None
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, str, str]]]:
    """Returns ``(rows, rejects)``: ``rows`` = ``[(line, row)]`` usable rows, ``rejects`` =
    ``[(line, reason, preview)]``. Raises ``GoldFileError`` for an unreadable file."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GoldFileError(f"The file is not UTF-8 text: {exc}") from exc
    name = (filename or "").lower()
    rows: list[tuple[int, dict[str, Any]]] = []
    rejects: list[tuple[int, str, str]] = []

    def take(line: int, value: Any, preview: str) -> None:
        if not isinstance(value, dict):
            rejects.append((line, REASON_NOT_AN_OBJECT, preview))
            return
        value = _apply_mapping(value, field_mapping)
        if not _has_content(value):
            rejects.append((line, REASON_EMPTY_ROW, preview))
            return
        rows.append((line, value))

    stripped = text.lstrip()
    if name.endswith((".csv", ".tsv")):
        is_tsv = name.endswith(".tsv")
        if is_tsv:
            # TSV has no quoting: a quote character in a cell is just text.
            reader = csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        else:
            reader = csv.reader(io.StringIO(text), delimiter=",")
        delimiter = "\t" if is_tsv else ","
        header: list[str] | None = None
        try:
            for record in reader:
                line = reader.line_num
                if header is None:
                    header = [h.strip() for h in record]
                    continue
                if not any(c.strip() for c in record):
                    continue
                preview = delimiter.join(record)[:PREVIEW_CHARS]
                if len(record) != len(header):
                    rejects.append((line, REASON_WRONG_COLUMNS, preview))
                    continue
                row = {header[i]: _cell(c) for i, c in enumerate(record) if header[i]}
                take(line, row, preview)
        except csv.Error as exc:
            raise GoldFileError(
                f"The {'TSV' if is_tsv else 'CSV'} file cannot be parsed (near line {reader.line_num}): {exc}. "
                "Check for a cell over 131072 characters, an unclosed quote, or stray control characters."
            ) from exc
        if header is None:
            raise GoldFileError("The file is empty.")
    elif name.endswith(".json") or (not name.endswith((".jsonl", ".ndjson")) and stripped.startswith("[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GoldFileError(f"The file is not valid JSON: {exc}") from exc
        if isinstance(parsed, dict):
            for key in ("rows", "entries", "data", "examples", "items"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        if not isinstance(parsed, list):
            raise GoldFileError("A .json file must hold an array of row objects.")
        for i, value in enumerate(parsed, start=1):
            take(i, value, json.dumps(value)[:PREVIEW_CHARS])
    else:
        for line_no, raw in enumerate(text.splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                rejects.append((line_no, REASON_INVALID_JSON, raw[:PREVIEW_CHARS]))
                continue
            take(line_no, value, raw[:PREVIEW_CHARS])
    return rows, rejects


def group_rejects(rejects: list[tuple[int, str, str]]) -> dict[str, Any]:
    """``{count, by_reason: {reason: {label, count, lines, rows: [{line, preview}]}}}``; the UI
    selects rows within a reason group and bulk-drops them."""
    by_reason: dict[str, Any] = {}
    for line, reason, preview in rejects:
        group = by_reason.setdefault(
            reason, {"label": REASON_TEXT.get(reason, reason), "count": 0, "lines": [], "rows": []}
        )
        group["count"] += 1
        group["lines"].append(line)
        if len(group["rows"]) < SAMPLE_CAP:
            group["rows"].append({"line": line, "preview": preview})
    return {"count": len(rejects), "by_reason": by_reason}

"""Read user files the way a newcomer's files actually arrive.

One place that turns CSV / TSV / JSONL / JSON / XLSX / Parquet into rows and
any text file into a string, whatever the encoding (Excel's cp1252 CSVs,
UTF-8 with BOM, …). Every loader in ingestion, cleaning, dataset prep and
structured import goes through here so a file that previews fine can't
crash three steps later.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

TABULAR_EXTENSIONS = frozenset({".csv", ".tsv", ".jsonl", ".json", ".xlsx", ".parquet"})
# JSON wrappers people commonly export: {"data": [...]}, {"rows": [...]}.
_JSON_LIST_KEYS = ("data", "rows", "records", "items", "examples", "train")


def decode_bytes(data: bytes) -> tuple[str, str]:
    """``(text, encoding)``. UTF-8 (with/without BOM) first, then a
    detector, then cp1252 (Excel on Windows), then latin-1 (never fails)."""
    for encoding in ("utf-8-sig",):
        try:
            return data.decode(encoding), "utf-8"
        except UnicodeDecodeError:
            pass
    try:
        cp1252_text: str | None = data.decode("cp1252")
    except UnicodeDecodeError:
        cp1252_text = None
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
    except Exception:  # noqa: BLE001
        best = None
    if best is not None and best.encoding:
        detected = str(best)
        # On short Western text the detector often lands on a sibling
        # Latin code page (cp775 / cp1257 turn "café" into "cafķ"). Such
        # files are overwhelmingly Excel/Windows cp1252, so only trust the
        # detector when it finds a non-Latin script (Cyrillic, Greek, CJK…).
        if cp1252_text is None or _mostly_non_latin(detected):
            return detected, best.encoding
    if cp1252_text is not None:
        return cp1252_text, "cp1252"
    return data.decode("latin-1"), "latin-1"


def _mostly_non_latin(text: str) -> bool:
    non_ascii_letters = [ch for ch in text if ord(ch) > 127 and ch.isalpha()]
    if not non_ascii_letters:
        return False
    beyond_latin = sum(1 for ch in non_ascii_letters if ord(ch) >= 0x0370)
    return beyond_latin / len(non_ascii_letters) > 0.5


def read_text_file(path: str | Path) -> str:
    return decode_bytes(Path(path).read_bytes())[0]


def is_tabular(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TABULAR_EXTENSIONS


def _json_safe(value: Any) -> Any:
    """Coerce pandas/numpy scalars + NaN to plain JSON values."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(item())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _rows_from_frame(frame: Any, limit: int | None) -> list[dict[str, Any]]:
    if limit is not None:
        frame = frame.head(limit)
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append({str(k): _json_safe(v) for k, v in record.items()})
    return rows


def _csv_rows(text: str, delimiter: str | None, limit: int | None) -> list[dict[str, Any]]:
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    rows: list[dict[str, Any]] = []
    for row in reader:
        clean = {
            str(k).strip(): (v if v is not None else "")
            for k, v in row.items()
            if k is not None and str(k).strip()
        }
        if any(str(v).strip() for v in clean.values()):
            rows.append(clean)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _json_rows(text: str) -> list[Any]:
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _JSON_LIST_KEYS:
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    return [payload]


def load_rows(path: str | Path, limit: int | None = None) -> list[Any]:
    """Rows from a tabular file. Dicts for CSV/TSV/XLSX/Parquet; JSON(L)
    rows are returned as parsed (usually dicts). Malformed JSONL lines are
    skipped. Raises ``ValueError`` with a readable message on files that
    can't be read at all."""
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext in {".csv", ".tsv"}:
        text = read_text_file(file_path)
        return _csv_rows(text, "\t" if ext == ".tsv" else None, limit)
    if ext == ".jsonl":
        rows: list[Any] = []
        for line in read_text_file(file_path).splitlines():
            token = line.strip()
            if not token:
                continue
            try:
                rows.append(json.loads(token))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(rows) >= limit:
                break
        return rows
    if ext == ".json":
        try:
            rows = _json_rows(read_text_file(file_path))
        except json.JSONDecodeError:
            # Some ".json" exports are really JSON Lines.
            return _jsonl_fallback(file_path, limit)
        return rows[:limit] if limit is not None else rows
    if ext == ".xlsx":
        try:
            import pandas as pd

            frame = pd.read_excel(file_path, sheet_name=0, dtype=object)
        except ImportError as exc:
            raise ValueError("Reading .xlsx needs openpyxl (pip install openpyxl).") from exc
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Couldn't read the spreadsheet: {exc}") from exc
        return _rows_from_frame(frame, limit)
    if ext == ".parquet":
        try:
            import pandas as pd

            frame = pd.read_parquet(file_path)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Couldn't read the Parquet file: {exc}") from exc
        return _rows_from_frame(frame, limit)
    raise ValueError(f"Not a tabular file: {file_path.name}")


def _jsonl_fallback(file_path: Path, limit: int | None) -> list[Any]:
    rows: list[Any] = []
    for line in read_text_file(file_path).splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            rows.append(json.loads(token))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file_path.name} isn't valid JSON or JSON Lines: {exc}") from exc
        if limit is not None and len(rows) >= limit:
            break
    return rows


def render_row_text(row: Any) -> str:
    """Human-readable text for a row (previews / extracted-text sidecars)."""
    if isinstance(row, dict):
        parts = []
        for key, value in row.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            parts.append(f"{key}: {rendered}")
        return "\n".join(parts)
    return str(row)

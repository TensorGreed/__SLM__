"""JSONL source connector.

Locator format: ``jsonl:/path/to/file.jsonl``. Streams one row per
line; blank lines + lines that don't parse as JSON objects are
counted as skipped but never crash the import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from app.services.dataset_import.protocols import DatasetSource, RawRow
from app.services.dataset_import.registry import register_source


class JsonlSource:
    source_id: str = "jsonl"

    def _resolve_path(self, locator: str) -> Path:
        path = Path(locator).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"JSONL file not found at '{path}'. Pass an absolute path or "
                "a path relative to the BrewSLM working directory."
            )
        if not path.is_file():
            raise IsADirectoryError(f"'{path}' is not a file")
        return path

    def load(
        self,
        locator: str,
        *,
        limit: int | None = None,
    ) -> Iterable[RawRow]:
        path = self._resolve_path(locator)
        yielded = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    # Silent skip would violate per-row accountability;
                    # but the source contract is "stream raw rows" —
                    # mappers handle rejection. We surface unparseable
                    # lines as a row with a sentinel `__parse_error__`
                    # field so the mapper can reject them with a clear
                    # reason rather than treating them as missing.
                    yield {"__parse_error__": "invalid_json", "__raw_line__": stripped[:200]}
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
                    continue
                if not isinstance(row, dict):
                    yield {"__parse_error__": "not_an_object", "__raw_value__": str(row)[:200]}
                else:
                    yield row
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def describe(self, locator: str) -> dict[str, Any]:
        """Quick metadata for the introspector + UI.

        Reads at most 20 lines for the sample and counts total lines
        without holding them in memory.
        """

        path = self._resolve_path(locator)
        sample: list[dict[str, Any]] = []
        total_lines = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                total_lines += 1
                if len(sample) < 20:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                        if isinstance(row, dict):
                            sample.append(row)
                    except json.JSONDecodeError:
                        continue
        columns: list[str] = []
        seen: set[str] = set()
        for row in sample:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
        return {
            "source_id": self.source_id,
            "locator": locator,
            "resolved_path": str(path),
            "approximate_total_rows": total_lines,
            "sample_rows": sample,
            "columns": columns,
        }


register_source("jsonl", JsonlSource)

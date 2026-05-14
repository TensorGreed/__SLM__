"""CSV source connector.

Locator format: ``csv:/path/to/file.csv``. Uses :mod:`csv.DictReader`
so the first row is treated as the header. Every cell value lands as
a string — type coercion is the mapper's job (it has the
``task_profile`` context to decide whether ``"42"`` should become an
int, a string, or a label).
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Any, Iterable

from app.services.dataset_import.protocols import DatasetSource, RawRow
from app.services.dataset_import.registry import register_source


class CsvSource:
    source_id: str = "csv"

    def _resolve_path(self, locator: str) -> Path:
        path = Path(locator).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"CSV file not found at '{path}'. Pass an absolute path or "
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
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = _csv.DictReader(handle)
            for row in reader:
                # DictReader can produce ``None`` keys when a row has
                # extra trailing commas; drop those so downstream
                # mappers don't have to defend against weird shapes.
                cleaned = {
                    str(k): (v if v is not None else "")
                    for k, v in row.items()
                    if k is not None
                }
                yield cleaned
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def describe(self, locator: str) -> dict[str, Any]:
        path = self._resolve_path(locator)
        sample: list[dict[str, Any]] = []
        total_rows = 0
        columns: list[str] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = _csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            for row in reader:
                total_rows += 1
                if len(sample) < 20:
                    cleaned = {
                        str(k): (v if v is not None else "")
                        for k, v in row.items()
                        if k is not None
                    }
                    sample.append(cleaned)
        return {
            "source_id": self.source_id,
            "locator": locator,
            "resolved_path": str(path),
            "approximate_total_rows": total_rows,
            "sample_rows": sample,
            "columns": columns,
        }


register_source("csv", CsvSource)

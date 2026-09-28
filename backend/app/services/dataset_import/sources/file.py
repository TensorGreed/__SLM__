"""Any-format local file source (browser uploads land here).

Locator format: ``file:/path/to/data.{csv,tsv,jsonl,json,xlsx,parquet}``.
Reads through :mod:`app.utils.tabular_io`, so encoding (cp1252 Excel CSVs,
BOMs), delimiter (``,`` ``;`` ``\\t`` ``|``) and format are handled in one
place. ``POST /projects/{id}/dataset-import/upload`` stages browser uploads
under the project and hands back a ``file:`` locator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from app.services.dataset_import.protocols import RawRow
from app.services.dataset_import.registry import register_source
from app.utils.tabular_io import TABULAR_EXTENSIONS, load_rows

_SAMPLE_ROWS = 20


class FileSource:
    source_id: str = "file"

    def _resolve_path(self, locator: str) -> Path:
        path = Path(locator).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found at '{path}'.")
        if not path.is_file():
            raise IsADirectoryError(f"'{path}' is not a file")
        if path.suffix.lower() not in TABULAR_EXTENSIONS:
            allowed = ", ".join(sorted(TABULAR_EXTENSIONS))
            raise ValueError(
                f"'{path.name}' isn't a row file ({allowed}). Documents such as "
                "PDF/DOCX/HTML go through Data → Upload documents instead."
            )
        return path

    def _rows(self, path: Path, limit: int | None = None) -> list[dict[str, Any]]:
        return [
            row if isinstance(row, dict) else {"value": row}
            for row in load_rows(path, limit=limit)
        ]

    def load(self, locator: str, *, limit: int | None = None) -> Iterable[RawRow]:
        yield from self._rows(self._resolve_path(locator), limit)

    def describe(self, locator: str) -> dict[str, Any]:
        path = self._resolve_path(locator)
        rows = self._rows(path)
        columns: list[str] = []
        for row in rows[:_SAMPLE_ROWS]:
            for key in row:
                if key not in columns:
                    columns.append(key)
        return {
            "source_id": self.source_id,
            "locator": locator,
            "resolved_path": str(path),
            "approximate_total_rows": len(rows),
            "sample_rows": rows[:_SAMPLE_ROWS],
            "columns": columns,
        }


register_source("file", FileSource)

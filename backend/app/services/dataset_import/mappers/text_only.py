"""Text-only mapper — single-column passthrough for plain LM training.

Use when the dataset is a raw corpus: each row is just a chunk of
text, no labels, no Q&A pairs. Feeds the ``language_modeling`` task
profile (which routes through ``QAHandler`` for completion-style
scoring with no reference).

field_map fields (all optional):

  - ``text_field`` (default ``"text"``)
  - ``id_field`` (used for row_key when present)
  - ``min_chars`` (int, default 1) — reject rows shorter than this
    after stripping. Useful to drop near-empty fragments.

Emits ``TransformedRow.payload``:

    {"text": <str>}

Rejection reason codes:

  - ``parse_error`` — source-flagged unparseable row
  - ``missing_text`` — text column absent or empty after strip
  - ``text_too_short`` — passes the column-exists check but length <
    ``min_chars``
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.services.dataset_import.protocols import (
    ImportContext,
    RawRow,
    RejectedRow,
    TargetMapper,
    TransformedRow,
)
from app.services.dataset_import.registry import register_mapper


DEFAULT_FIELDS: dict[str, str] = {
    "text_field": "text",
    "id_field": "id",
}

_WHITESPACE_RUN = re.compile(r"\s+")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RUN.sub(" ", str(value)).strip()


class TextOnlyMapper:
    mapper_id: str = "text_only"

    def declared_target(self) -> str:
        return "language_modeling"

    def transform(
        self,
        rows: Iterable[RawRow],
        field_map: dict[str, Any],
        *,
        ctx: ImportContext,
    ) -> Iterable[TransformedRow | RejectedRow]:
        fields = {
            **DEFAULT_FIELDS,
            **{
                k: v
                for k, v in (field_map or {}).items()
                if isinstance(v, str)
            },
        }
        raw_min = (field_map or {}).get("min_chars", 1)
        try:
            min_chars = max(0, int(raw_min))
        except (TypeError, ValueError):
            min_chars = 1

        for idx, row in enumerate(rows):
            if "__parse_error__" in row:
                yield RejectedRow(
                    raw_row=row,
                    reason="parse_error",
                    detail=str(row.get("__parse_error__")),
                    row_index=idx,
                )
                continue

            text = _coerce_text(row.get(fields["text_field"]))
            if not text:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_text",
                    detail=f"column '{fields['text_field']}' empty after strip",
                    row_index=idx,
                )
                continue
            if len(text) < min_chars:
                yield RejectedRow(
                    raw_row=row,
                    reason="text_too_short",
                    detail=f"length {len(text)} < min_chars {min_chars}",
                    row_index=idx,
                )
                continue

            row_id = row.get(fields["id_field"])
            row_key = f"{ctx.source_id}-{row_id}" if row_id is not None else None
            yield TransformedRow(payload={"text": text}, row_key=row_key)


register_mapper("text_only", TextOnlyMapper)

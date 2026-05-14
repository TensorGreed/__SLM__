"""Key-value → structured-extraction mapper.

Use when the dataset has *flat key-value extractions* — invoices,
forms, receipts, business cards — rather than span-offsets within a
free-text body. Each row carries a source text plus one column per
field; the mapper wraps the fields into the
``{"entities": [{"field", "value"}, ...]}`` shape the
``StructuredExtractionHandler``'s ``field_match`` scoring mode expects.

field_map fields:

  - ``text_field`` (default ``"text"``) — the raw text the extraction
    came from (receipt OCR, invoice body, etc).
  - ``id_field`` (used for row_key when present)
  - ``fields`` (list[str] OR dict[str, str]) — REQUIRED.
      - list form: ``["invoice_number", "total", "vendor"]`` — uses
        each name as both source column and emitted field name.
      - dict form: ``{"invoice_number": "InvoiceNumber", ...}`` —
        keys are emitted field names, values are source columns.
      The dict form is useful when source columns have ugly /
      capitalized names but you want clean canonical fields in the
      output.
  - ``skip_empty_fields`` (bool, default True) — drop fields whose
    source value is empty rather than emitting an empty entity.
    Set False to preserve the full key set even when some fields are
    blank (e.g. when downstream eval cares about coverage).

Emits ``TransformedRow.payload``:

    {
        "text": <source text>,
        "entities_json": '{"entities":[{"field": ..., "value": ...}]}',
        "fields": [...]   # the field name list, in declared order
    }

Declared target: ``structured_extraction``. The
``output_schema.scoring_mode`` on the project's prepared manifest
should be ``field_match`` (not ``span_set`` — that's bio_to_spans).

Rejection reason codes:

  - ``parse_error`` — source-flagged unparseable row
  - ``missing_text`` — text column empty after strip
  - ``missing_fields_config`` — no ``fields`` key in the field_map
  - ``no_fields_extracted`` — every declared field was empty and
    ``skip_empty_fields`` is set; nothing to learn from this row.
"""

from __future__ import annotations

import json
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


def _coerce_value(value: Any) -> str:
    """Lighter coercion than text: preserves int/float as digits, keeps
    bools as ``true`` / ``false``. The downstream handler does string
    EM/F1 so we want stable string forms."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return _WHITESPACE_RUN.sub(" ", str(value)).strip()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RUN.sub(" ", str(value)).strip()


def _resolve_field_map(raw: Any) -> list[tuple[str, str]]:
    """Turn the ``fields`` field_map entry into a list of
    ``(emit_name, source_column)`` pairs. Returns an empty list when
    the config is malformed — callers reject such rows with
    ``missing_fields_config``."""

    if isinstance(raw, list):
        pairs: list[tuple[str, str]] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if name:
                pairs.append((name, name))
        return pairs
    if isinstance(raw, dict):
        pairs = []
        for emit, source in raw.items():
            emit_s = str(emit).strip()
            source_s = str(source).strip() if source else emit_s
            if emit_s:
                pairs.append((emit_s, source_s or emit_s))
        return pairs
    return []


class KvToStructuredMapper:
    mapper_id: str = "kv_to_structured"

    def declared_target(self) -> str:
        return "structured_extraction"

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
        field_pairs = _resolve_field_map((field_map or {}).get("fields"))
        skip_empty = bool((field_map or {}).get("skip_empty_fields", True))

        if not field_pairs:
            # Declaring no fields means every row would emit an empty
            # entity list — reject as a config error rather than
            # silently producing useless rows.
            for idx, row in enumerate(rows):
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_fields_config",
                    detail=(
                        "kv_to_structured requires a 'fields' entry in "
                        "field_map (list of column names, or dict of "
                        "emit_name → source_column)"
                    ),
                    row_index=idx,
                )
            return

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

            entities: list[dict[str, str]] = []
            for emit_name, source_col in field_pairs:
                value = _coerce_value(row.get(source_col))
                if not value and skip_empty:
                    continue
                entities.append({"field": emit_name, "value": value})

            if not entities:
                yield RejectedRow(
                    raw_row=row,
                    reason="no_fields_extracted",
                    detail=(
                        "every declared field was empty; nothing to learn "
                        "from this row (set skip_empty_fields=false to "
                        "keep the row anyway)"
                    ),
                    row_index=idx,
                )
                continue

            entities_json = json.dumps(
                {"entities": entities}, ensure_ascii=False
            )
            row_id = row.get(fields["id_field"])
            row_key = f"{ctx.source_id}-{row_id}" if row_id is not None else None
            yield TransformedRow(
                payload={
                    "text": text,
                    "entities_json": entities_json,
                    "fields": [name for name, _ in field_pairs],
                },
                row_key=row_key,
            )


register_mapper("kv_to_structured", KvToStructuredMapper)

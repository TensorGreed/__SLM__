"""Span-offset passthrough: text + an entity list with character offsets → span rows.

For datasets already labelled with character offsets (the span shape BrewSLM
trains and scores on) — any entity types, any language. The offsets are kept
exactly as given: a span is validated against the text, never moved.

Expected raw row shape:

    {
      "text": "Write to jane@example.com before Friday.",
      "entities": [{"type": "email", "start": 9, "end": 25, "text": "jane@example.com"}]
    }

``entities`` may also be a JSON string of that list, or of
``{"entities": [...]}``. ``label`` is read when ``type`` is missing, ``value``
when ``text`` is missing. An empty
list is a valid row: text with nothing to tag (a negative example).

field_map fields (all optional):

  - ``text_field`` (default ``"text"``)
  - ``entities_field`` (default ``"entities"``)
  - ``id_field`` (default ``"id"`` — used for ``row_key``)
  - ``entity_type_map`` (dict) — source type → project type

Emits ``TransformedRow.payload`` = ``{"text": <text>, "entities": [{type, start,
end, text}, ...]}`` — the shape the structured-extraction adapter, the
token-classification trainer and the ``span_set`` scorer all read.

Rejection reason codes (stable strings for bulk-drop UX):

  - ``parse_error`` — source connector flagged the row (JSONL only)
  - ``empty_text`` — text column missing or blank
  - ``missing_entities`` — entity column missing, null or not a list
  - ``bad_span`` — an entity without a type or with non-integer offsets
  - ``span_out_of_range`` — ``start``/``end`` outside the text, or ``end <= start``
  - ``span_text_mismatch`` — an entity's ``text`` differs from ``text[start:end]``
  - ``text_edge_whitespace`` — the text starts with whitespace, or a span ends
    in trailing whitespace: training strips the text, which would move or drop
    those spans
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from app.services.dataset_import.protocols import (
    ImportContext,
    RawRow,
    RejectedRow,
    TransformedRow,
)
from app.services.dataset_import.registry import register_mapper


DEFAULT_FIELDS: dict[str, str] = {
    "text_field": "text",
    "entities_field": "entities",
    "id_field": "id",
}


class _SpanReject(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _entity_list(value: Any) -> list[Any] | None:
    """The entity list from a list, a JSON string of a list, or
    ``{"entities": [...]}`` (either form). None when it is none of these."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(value, dict):
        value = value.get("entities")
    return value if isinstance(value, list) else None


def _offset(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _clean_spans(text: str, raw: list[Any], type_map: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    content_end = len(text.rstrip())
    for item in raw:
        if not isinstance(item, dict):
            raise _SpanReject("bad_span", f"entity is not an object: {item!r}"[:200])
        ent_type = str(item.get("type") or item.get("label") or "").strip()
        start, end = _offset(item.get("start")), _offset(item.get("end"))
        if not ent_type or start is None or end is None:
            raise _SpanReject("bad_span", f"entity needs type + integer start/end: {item!r}"[:200])
        if start < 0 or end > len(text) or end <= start:
            raise _SpanReject(
                "span_out_of_range", f"{ent_type} [{start}, {end}) on a text of {len(text)} chars"
            )
        covered = text[start:end]
        given = item.get("text") if item.get("text") is not None else item.get("value")
        if given is not None and str(given) != covered:
            raise _SpanReject(
                "span_text_mismatch", f"{ent_type} [{start}, {end}) is {covered!r}, row says {given!r}"[:200]
            )
        if end > content_end:
            raise _SpanReject("text_edge_whitespace", f"{ent_type} [{start}, {end}) ends in trailing whitespace")
        out.append({"type": type_map.get(ent_type, ent_type), "start": start, "end": end, "text": covered})
    return out


class SpanOffsetsPassthroughMapper:
    mapper_id: str = "span_offsets_passthrough"

    def declared_target(self) -> str:
        return "structured_extraction"

    def transform(
        self,
        rows: Iterable[RawRow],
        field_map: dict[str, Any],
        *,
        ctx: ImportContext,
    ) -> Iterable[TransformedRow | RejectedRow]:
        fields = {**DEFAULT_FIELDS, **{k: v for k, v in (field_map or {}).items() if isinstance(v, str)}}
        type_map: dict[str, str] = {}
        raw_map = (field_map or {}).get("entity_type_map")
        if isinstance(raw_map, dict):
            type_map = {str(k).strip(): str(v).strip() for k, v in raw_map.items()}

        for idx, row in enumerate(rows):
            if "__parse_error__" in row:
                yield RejectedRow(raw_row=row, reason="parse_error", detail=str(row.get("__parse_error__")), row_index=idx)
                continue
            text = row.get(fields["text_field"])
            if not isinstance(text, str) or not text.strip():
                yield RejectedRow(
                    raw_row=row, reason="empty_text",
                    detail=f"expected text under '{fields['text_field']}'", row_index=idx,
                )
                continue
            raw = _entity_list(row.get(fields["entities_field"]))
            if raw is None:
                yield RejectedRow(
                    raw_row=row, reason="missing_entities",
                    detail=f"expected an entity list under '{fields['entities_field']}'", row_index=idx,
                )
                continue
            if text[0].isspace():
                yield RejectedRow(
                    raw_row=row, reason="text_edge_whitespace",
                    detail="text starts with whitespace; training strips it, shifting every offset",
                    row_index=idx,
                )
                continue
            try:
                entities = _clean_spans(text, raw, type_map)
            except _SpanReject as exc:
                yield RejectedRow(raw_row=row, reason=exc.reason, detail=exc.detail, row_index=idx)
                continue
            doc_id = row.get(fields["id_field"])
            yield TransformedRow(
                payload={"text": text, "entities": entities},
                row_key=f"{ctx.source_id}-{doc_id}" if doc_id is not None else None,
            )


register_mapper("span_offsets_passthrough", SpanOffsetsPassthroughMapper)

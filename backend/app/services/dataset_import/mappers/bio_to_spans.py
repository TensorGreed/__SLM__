"""BIO-tag → entity-span mapper.

Generalizes the Phase 4.x kaggle PII converter. Works on any BIO-tagged
dataset (PII / medical / legal / financial / generic NER) — the entity
type mapping is config, not code.

Expected raw row shape (one of):

    {
      "tokens": ["Hi", ",", "my", "name", "is", "John"],
      "labels": ["O", "O", "O", "O", "O", "B-NAME_STUDENT"],
      "trailing_whitespace": [false, true, true, true, true, true],
      "full_text": "Hi, my name is John ..."   # optional
    }

field_map fields (all optional except ``tokens_field`` / ``labels_field``):

  - ``tokens_field`` (default ``"tokens"``)
  - ``labels_field`` (default ``"labels"``)
  - ``trailing_whitespace_field`` (default ``"trailing_whitespace"``)
  - ``full_text_field`` (default ``"full_text"``)
  - ``id_field`` (default ``"document"`` — used for ``row_key``)
  - ``entity_type_map`` (dict) — Kaggle/HF → BrewSLM type, e.g.
    ``{"NAME_STUDENT": "person_name", "EMAIL": "email"}``
  - ``output_field`` (default ``"entities_json"``)

Emits ``TransformedRow.payload``:

    {
      "text": <reconstructed text>,
      "entities_json": '{"entities":[{"type","start","end","text"},...]}'
    }

— matching what ``StructuredExtractionHandler``'s ``span_set`` scoring
mode consumes verbatim.

Rejection reason codes (stable strings for bulk-drop UX):

  - ``missing_tokens`` — required ``tokens`` column absent or empty
  - ``missing_labels`` — required ``labels`` column absent or empty
  - ``length_mismatch`` — len(tokens) != len(labels)
  - ``parse_error`` — source connector flagged the row (JSONL only)
  - ``empty_text`` — reconstructed text is empty
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from app.services.dataset_import.protocols import (
    ImportContext,
    RawRow,
    RejectedRow,
    TargetMapper,
    TransformedRow,
)
from app.services.dataset_import.registry import register_mapper


# Pulled out as module-level so tests can patch / inspect them.
DEFAULT_FIELDS: dict[str, str] = {
    "tokens_field": "tokens",
    "labels_field": "labels",
    "trailing_whitespace_field": "trailing_whitespace",
    "full_text_field": "full_text",
    "id_field": "document",
    "output_field": "entities_json",
}


class BioToSpansMapper:
    mapper_id: str = "bio_to_spans"

    def declared_target(self) -> str:
        return "structured_extraction"

    # ── Offset reconstruction ──

    @staticmethod
    def _reconstruct_from_tokens(
        tokens: list[str], trailing_whitespace: list[bool] | None
    ) -> tuple[str, list[tuple[int, int]]]:
        buf: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        trail = trailing_whitespace or [True] * len(tokens)
        if len(trail) != len(tokens):
            trail = [True] * len(tokens)
        for tok, tr in zip(tokens, trail):
            start = cursor
            buf.append(tok)
            cursor += len(tok)
            spans.append((start, cursor))
            if tr:
                buf.append(" ")
                cursor += 1
        return "".join(buf), spans

    @staticmethod
    def _align_to_full_text(
        full_text: str, tokens: list[str]
    ) -> list[tuple[int, int]] | None:
        spans: list[tuple[int, int]] = []
        cursor = 0
        n = len(full_text)
        for tok in tokens:
            if not tok:
                spans.append((cursor, cursor))
                continue
            while cursor < n and full_text[cursor].isspace():
                cursor += 1
            if cursor + len(tok) > n or full_text[cursor : cursor + len(tok)] != tok:
                return None
            start = cursor
            cursor += len(tok)
            spans.append((start, cursor))
        return spans

    # ── BIO run extraction ──

    @staticmethod
    def _bio_runs(labels: list[str]) -> list[tuple[int, int, str]]:
        runs: list[tuple[int, int, str]] = []
        current_type: str | None = None
        current_start: int | None = None
        for idx, raw in enumerate(labels):
            tag = (raw or "O").strip()
            if tag == "O":
                if current_type is not None and current_start is not None:
                    runs.append((current_start, idx, current_type))
                current_type = None
                current_start = None
                continue
            prefix, _, ent_type = tag.partition("-")
            if not ent_type:
                ent_type = prefix
                prefix = "B"
            if prefix == "B" or current_type != ent_type:
                if current_type is not None and current_start is not None:
                    runs.append((current_start, idx, current_type))
                current_type = ent_type
                current_start = idx
        if current_type is not None and current_start is not None:
            runs.append((current_start, len(labels), current_type))
        return runs

    # ── Per-row transform ──

    def transform(
        self,
        rows: Iterable[RawRow],
        field_map: dict[str, Any],
        *,
        ctx: ImportContext,
    ) -> Iterable[TransformedRow | RejectedRow]:
        fields = {**DEFAULT_FIELDS, **{k: v for k, v in (field_map or {}).items() if isinstance(v, str)}}
        entity_type_map: dict[str, str] = {}
        raw_map = (field_map or {}).get("entity_type_map")
        if isinstance(raw_map, dict):
            entity_type_map = {str(k).strip(): str(v).strip().lower() for k, v in raw_map.items()}

        for idx, row in enumerate(rows):
            if "__parse_error__" in row:
                yield RejectedRow(
                    raw_row=row,
                    reason="parse_error",
                    detail=str(row.get("__parse_error__")),
                    row_index=idx,
                )
                continue

            tokens = row.get(fields["tokens_field"])
            labels = row.get(fields["labels_field"])
            if not isinstance(tokens, list) or not tokens:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_tokens",
                    detail=f"expected list under '{fields['tokens_field']}'",
                    row_index=idx,
                )
                continue
            if not isinstance(labels, list) or not labels:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_labels",
                    detail=f"expected list under '{fields['labels_field']}'",
                    row_index=idx,
                )
                continue
            if len(tokens) != len(labels):
                yield RejectedRow(
                    raw_row=row,
                    reason="length_mismatch",
                    detail=f"tokens={len(tokens)} labels={len(labels)}",
                    row_index=idx,
                )
                continue

            full_text = row.get(fields["full_text_field"])
            trailing = row.get(fields["trailing_whitespace_field"])
            token_spans: list[tuple[int, int]] | None = None
            text: str
            if isinstance(full_text, str) and full_text:
                token_spans = self._align_to_full_text(full_text, list(tokens))
                text = full_text
            if token_spans is None:
                text, token_spans = self._reconstruct_from_tokens(
                    list(tokens), trailing if isinstance(trailing, list) else None
                )

            if not text.strip():
                yield RejectedRow(
                    raw_row=row,
                    reason="empty_text",
                    detail="reconstructed text was empty",
                    row_index=idx,
                )
                continue

            entities: list[dict[str, Any]] = []
            for start_idx, end_idx, ent_type in self._bio_runs([str(t) for t in labels]):
                if end_idx <= start_idx:
                    continue
                char_start = token_spans[start_idx][0]
                char_end = token_spans[end_idx - 1][1]
                span_text = text[char_start:char_end]
                if not span_text.strip():
                    continue
                mapped_type = entity_type_map.get(ent_type, ent_type.lower())
                entities.append(
                    {
                        "type": mapped_type,
                        "start": char_start,
                        "end": char_end,
                        "text": span_text,
                    }
                )

            doc_id = row.get(fields["id_field"])
            row_key = f"{ctx.source_id}-{doc_id}" if doc_id is not None else None

            yield TransformedRow(
                payload={
                    "text": text,
                    fields["output_field"]: json.dumps(
                        {"entities": entities}, ensure_ascii=False
                    ),
                },
                row_key=row_key,
            )


register_mapper("bio_to_spans", BioToSpansMapper)

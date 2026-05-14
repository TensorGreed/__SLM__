"""RAG-passthrough mapper — ``{question, context, answer}`` → RAGHandler.

Use when the dataset is retrieval-augmented QA: each row carries a
question, a passage that grounds the answer, and the gold answer.
RAGHandler reads ``question`` / ``context`` / ``answer`` directly, plus
the legacy ``prompt`` / ``reference`` aliases for gate compatibility.

field_map fields (all optional):

  - ``question_field`` (default ``"question"``) — also tried as
    ``"query"`` / ``"prompt"`` / ``"instruction"`` / ``"input"``.
  - ``context_field`` (default ``"context"``) — also tried as
    ``"passage"`` / ``"document"`` / ``"evidence"`` /
    ``"retrieved_context"``.
  - ``answer_field`` (default ``"answer"``) — also tried as
    ``"reference"`` / ``"completion"`` / ``"response"`` / ``"output"``
    / ``"target_text"``.
  - ``id_field`` (used for row_key when present)

Emits ``TransformedRow.payload``:

    {
        "question": <str>,
        "context": <str>,
        "answer": <str>,
        "prompt": <question>,    # legacy alias for non-RAG handlers
        "reference": <answer>,   # ditto
    }

Declared target: ``rag_qa`` (RAGHandler).

Rejection reason codes:

  - ``parse_error`` — source-flagged unparseable row
  - ``missing_question`` — question column empty after strip
  - ``missing_context`` — context column empty after strip; RAG
    explicitly requires a context (use ``qa_pair_passthrough`` for
    plain QA without grounding).
  - ``missing_answer`` — answer column empty after strip
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
    "question_field": "question",
    "context_field": "context",
    "answer_field": "answer",
    "id_field": "id",
}

_QUESTION_FALLBACKS: tuple[str, ...] = (
    "query",
    "prompt",
    "instruction",
    "input",
)
_CONTEXT_FALLBACKS: tuple[str, ...] = (
    "passage",
    "document",
    "evidence",
    "retrieved_context",
)
_ANSWER_FALLBACKS: tuple[str, ...] = (
    "reference",
    "completion",
    "response",
    "output",
    "target_text",
)

_WHITESPACE_RUN = re.compile(r"\s+")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RUN.sub(" ", str(value)).strip()


def _pick(row: dict[str, Any], primary: str, fallbacks: tuple[str, ...]) -> str:
    text = _coerce_text(row.get(primary))
    if text:
        return text
    for candidate in fallbacks:
        if candidate == primary:
            continue
        text = _coerce_text(row.get(candidate))
        if text:
            return text
    return ""


class RagPassthroughMapper:
    mapper_id: str = "rag_passthrough"

    def declared_target(self) -> str:
        return "rag_qa"

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

        for idx, row in enumerate(rows):
            if "__parse_error__" in row:
                yield RejectedRow(
                    raw_row=row,
                    reason="parse_error",
                    detail=str(row.get("__parse_error__")),
                    row_index=idx,
                )
                continue

            question = _pick(
                row, fields["question_field"], _QUESTION_FALLBACKS
            )
            context = _pick(row, fields["context_field"], _CONTEXT_FALLBACKS)
            answer = _pick(row, fields["answer_field"], _ANSWER_FALLBACKS)

            if not question:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_question",
                    detail=(
                        f"column '{fields['question_field']}' (and fallbacks) "
                        "empty after strip"
                    ),
                    row_index=idx,
                )
                continue
            if not context:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_context",
                    detail=(
                        f"column '{fields['context_field']}' (and fallbacks) "
                        "empty after strip — RAG requires a context"
                    ),
                    row_index=idx,
                )
                continue
            if not answer:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_answer",
                    detail=(
                        f"column '{fields['answer_field']}' (and fallbacks) "
                        "empty after strip"
                    ),
                    row_index=idx,
                )
                continue

            row_id = row.get(fields["id_field"])
            row_key = f"{ctx.source_id}-{row_id}" if row_id is not None else None
            yield TransformedRow(
                payload={
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "prompt": question,
                    "reference": answer,
                },
                row_key=row_key,
            )


register_mapper("rag_passthrough", RagPassthroughMapper)

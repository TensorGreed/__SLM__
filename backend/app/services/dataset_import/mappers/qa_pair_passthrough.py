"""QA-pair passthrough — ``{question, answer}`` → QAHandler shape.

Use when the dataset is short-answer QA or instruction tuning: each
row has a question (or prompt / instruction) and an answer (or
completion / response). The mapper does light normalization, defends
against empty rows, and emits the canonical
``{prompt, reference, question, answer}`` payload that QAHandler /
Seq2SeqHandler accept interchangeably (declared target is ``qa``;
override the project's ``task_profile`` if you want seq2seq training).

field_map fields (all optional):

  - ``question_field`` (default ``"question"``) — also tried as
    ``"prompt"`` / ``"instruction"`` / ``"input"`` for tolerance.
  - ``answer_field`` (default ``"answer"``) — also tried as
    ``"response"`` / ``"completion"`` / ``"output"``.
  - ``id_field`` (used for row_key when present)

Emits ``TransformedRow.payload``:

    {
        "prompt": <question text>,
        "reference": <answer text>,
        "question": <question text>,
        "answer": <answer text>,
    }

The duplication is intentional — different downstream handlers read
different keys (QAHandler reads ``prompt`` / ``reference``; some eval
configs read ``question`` / ``answer`` directly). One canonical row
satisfies both without an extra adapter.

Rejection reason codes:

  - ``parse_error`` — source-flagged unparseable row
  - ``missing_question`` — question column absent or empty after strip
  - ``missing_answer`` — answer column absent or empty after strip
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
    "answer_field": "answer",
    "id_field": "id",
}

# Fallback column names — when the configured field is empty, we try
# these in order before giving up. Matches the field-precedence in
# evaluation_service._extract_prompt_and_reference so a row that
# survives ingestion also survives eval-time extraction.
_QUESTION_FALLBACKS: tuple[str, ...] = (
    "prompt",
    "instruction",
    "input",
    "source_text",
)
_ANSWER_FALLBACKS: tuple[str, ...] = (
    "response",
    "completion",
    "output",
    "target_text",
    "reference",
)

_WHITESPACE_RUN = re.compile(r"\s+")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RUN.sub(" ", str(value)).strip()


def _pick(row: dict[str, Any], primary: str, fallbacks: tuple[str, ...]) -> str:
    """Try the configured column first, then the fallback list."""

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


class QaPairPassthroughMapper:
    mapper_id: str = "qa_pair_passthrough"

    def declared_target(self) -> str:
        return "qa"

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

            question = _pick(row, fields["question_field"], _QUESTION_FALLBACKS)
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
                    "prompt": question,
                    "reference": answer,
                    "question": question,
                    "answer": answer,
                },
                row_key=row_key,
            )


register_mapper("qa_pair_passthrough", QaPairPassthroughMapper)

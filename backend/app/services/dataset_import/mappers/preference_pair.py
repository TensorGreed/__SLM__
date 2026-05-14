"""Preference-pair mapper — ``{prompt, chosen, rejected}`` → AlignmentHandler.

Use for DPO / ORPO / preference-tuning datasets where each row carries
a prompt plus two completions, with one preferred over the other. The
AlignmentHandler reads ``prompt`` / ``chosen`` / ``rejected`` directly,
so this is structurally a passthrough with field-renaming + defensive
rejection of degenerate pairs (chosen == rejected, either side empty).

field_map fields (all optional):

  - ``prompt_field`` (default ``"prompt"``) — also tried as
    ``"question"`` / ``"instruction"`` / ``"input"``.
  - ``chosen_field`` (default ``"chosen"``) — also tried as
    ``"preferred"`` / ``"accepted"`` / ``"response_chosen"``.
  - ``rejected_field`` (default ``"rejected"``) — also tried as
    ``"dispreferred"`` / ``"response_rejected"`` / ``"negative"``.
  - ``id_field`` (used for row_key when present)

Emits ``TransformedRow.payload``:

    {
        "prompt": <str>,
        "chosen": <str>,
        "rejected": <str>,
        "reference": <chosen>,  # legacy gate compat
    }

Declared target: ``dpo`` (AlignmentHandler).

Rejection reason codes:

  - ``parse_error`` — source-flagged unparseable row
  - ``missing_prompt`` — prompt column empty after strip
  - ``missing_chosen`` — chosen column empty after strip
  - ``missing_rejected`` — rejected column empty after strip
  - ``identical_pair`` — chosen and rejected differ only in whitespace;
    a no-op pair gives the trainer no signal.
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
    "prompt_field": "prompt",
    "chosen_field": "chosen",
    "rejected_field": "rejected",
    "id_field": "id",
}

_PROMPT_FALLBACKS: tuple[str, ...] = ("question", "instruction", "input")
_CHOSEN_FALLBACKS: tuple[str, ...] = (
    "preferred",
    "accepted",
    "response_chosen",
)
_REJECTED_FALLBACKS: tuple[str, ...] = (
    "dispreferred",
    "response_rejected",
    "negative",
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


class PreferencePairMapper:
    mapper_id: str = "preference_pair"

    def declared_target(self) -> str:
        return "dpo"

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

            prompt = _pick(row, fields["prompt_field"], _PROMPT_FALLBACKS)
            chosen = _pick(row, fields["chosen_field"], _CHOSEN_FALLBACKS)
            rejected = _pick(row, fields["rejected_field"], _REJECTED_FALLBACKS)

            if not prompt:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_prompt",
                    detail=(
                        f"column '{fields['prompt_field']}' (and fallbacks) "
                        "empty after strip"
                    ),
                    row_index=idx,
                )
                continue
            if not chosen:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_chosen",
                    detail=(
                        f"column '{fields['chosen_field']}' (and fallbacks) "
                        "empty after strip"
                    ),
                    row_index=idx,
                )
                continue
            if not rejected:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_rejected",
                    detail=(
                        f"column '{fields['rejected_field']}' (and fallbacks) "
                        "empty after strip"
                    ),
                    row_index=idx,
                )
                continue
            if chosen == rejected:
                yield RejectedRow(
                    raw_row=row,
                    reason="identical_pair",
                    detail="chosen and rejected are identical — no preference signal",
                    row_index=idx,
                )
                continue

            row_id = row.get(fields["id_field"])
            row_key = f"{ctx.source_id}-{row_id}" if row_id is not None else None
            yield TransformedRow(
                payload={
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "reference": chosen,
                },
                row_key=row_key,
            )


register_mapper("preference_pair", PreferencePairMapper)

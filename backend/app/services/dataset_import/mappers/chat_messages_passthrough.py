"""Chat-messages passthrough — list of ``{role, content}`` → chat SFT.

Use when the dataset is multi-turn chat data: each row has a
``messages`` column whose value is a list of ``{role, content}`` dicts
(``role`` ∈ {system, user, assistant, tool, …}; ``content`` is the
turn's text).

The mapper normalizes message shapes, drops empty turns, and emits a
canonical payload where:

- ``prompt`` is the prefix of the conversation up to (but not
  including) the final assistant turn — rendered as plain text using
  ``role: content`` lines.
- ``reference`` is the final assistant turn's content — the
  ground-truth completion for SFT eval.
- ``messages`` is the cleaned message list, preserved verbatim so
  downstream code can apply chat templates at inference time.

field_map fields (all optional):

  - ``messages_field`` (default ``"messages"``)
  - ``role_key`` (default ``"role"``) — key within each message dict
  - ``content_key`` (default ``"content"``) — also tries ``"value"`` /
    ``"text"`` for tolerance to common alt shapes.
  - ``id_field`` (used for row_key when present)
  - ``require_assistant_reply`` (bool, default True) — if False, rows
    without a trailing assistant turn still emit (reference is "").

Emits ``TransformedRow.payload``:

    {
        "prompt": "user: …\\nassistant: …\\nuser: …",
        "reference": <final assistant content>,
        "messages": [{"role": ..., "content": ...}, ...],
    }

Declared target: ``chat_sft`` (QAHandler-compatible).

Rejection reason codes:

  - ``parse_error`` — source-flagged unparseable row
  - ``missing_messages`` — column absent or empty
  - ``invalid_messages_shape`` — not a list of dicts
  - ``no_valid_turns`` — every turn was empty or unparseable
  - ``missing_assistant_reply`` — conversation has no trailing
    assistant turn (and ``require_assistant_reply`` is set)
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
    "messages_field": "messages",
    "role_key": "role",
    "content_key": "content",
    "id_field": "id",
}

_CONTENT_FALLBACKS: tuple[str, ...] = ("value", "text")
_WHITESPACE_RUN = re.compile(r"\s+")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RUN.sub(" ", str(value)).strip()


def _clean_messages(
    raw_messages: Any, role_key: str, content_key: str
) -> list[dict[str, str]]:
    """Filter to ``{role, content}`` dicts with non-empty content."""

    if not isinstance(raw_messages, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = _coerce_text(item.get(role_key))
        content = _coerce_text(item.get(content_key))
        if not content:
            for fallback_key in _CONTENT_FALLBACKS:
                if fallback_key == content_key:
                    continue
                content = _coerce_text(item.get(fallback_key))
                if content:
                    break
        if not role or not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _render_prompt(messages: list[dict[str, str]]) -> str:
    """Plain ``role: content`` rendering of every turn except the
    final assistant one. Falls back to the whole conversation when the
    last turn isn't assistant (e.g. ``require_assistant_reply=False``
    on a user-final row)."""

    if not messages:
        return ""
    if messages[-1]["role"].lower() == "assistant":
        history = messages[:-1]
    else:
        history = messages
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


class ChatMessagesPassthroughMapper:
    mapper_id: str = "chat_messages_passthrough"

    def declared_target(self) -> str:
        return "chat_sft"

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
        require_reply = bool(
            (field_map or {}).get("require_assistant_reply", True)
        )

        for idx, row in enumerate(rows):
            if "__parse_error__" in row:
                yield RejectedRow(
                    raw_row=row,
                    reason="parse_error",
                    detail=str(row.get("__parse_error__")),
                    row_index=idx,
                )
                continue

            raw_messages = row.get(fields["messages_field"])
            if raw_messages is None:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_messages",
                    detail=f"column '{fields['messages_field']}' is absent",
                    row_index=idx,
                )
                continue
            if not isinstance(raw_messages, list):
                yield RejectedRow(
                    raw_row=row,
                    reason="invalid_messages_shape",
                    detail=(
                        f"column '{fields['messages_field']}' is "
                        f"{type(raw_messages).__name__}, expected list"
                    ),
                    row_index=idx,
                )
                continue

            cleaned = _clean_messages(
                raw_messages,
                fields["role_key"],
                fields["content_key"],
            )
            if not cleaned:
                yield RejectedRow(
                    raw_row=row,
                    reason="no_valid_turns",
                    detail="no message had both role and non-empty content",
                    row_index=idx,
                )
                continue

            final_role = cleaned[-1]["role"].lower()
            has_assistant_reply = final_role == "assistant"
            if require_reply and not has_assistant_reply:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_assistant_reply",
                    detail="conversation has no trailing assistant turn",
                    row_index=idx,
                )
                continue

            reference = cleaned[-1]["content"] if has_assistant_reply else ""
            prompt = _render_prompt(cleaned)
            row_id = row.get(fields["id_field"])
            row_key = f"{ctx.source_id}-{row_id}" if row_id is not None else None
            yield TransformedRow(
                payload={
                    "prompt": prompt,
                    "reference": reference,
                    "messages": cleaned,
                },
                row_key=row_key,
            )


register_mapper("chat_messages_passthrough", ChatMessagesPassthroughMapper)

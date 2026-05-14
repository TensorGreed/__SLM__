"""Label → classification mapper.

Flat passthrough for ``{text, label}``-style classification data. The
mapper's job is to:

  1. Pull the text + label out of arbitrary source column names.
  2. Coerce non-string label values (ints, bools) to canonical strings.
  3. Strip leading/trailing whitespace + collapse internal whitespace
     in the text — these are common dataset hygiene issues that bite
     downstream tokenization.
  4. Filter to declared ``allowed_labels`` when supplied (rejects with
     ``label_not_allowed`` for any label outside the set).

field_map fields (all optional except text_field / label_field):

  - ``text_field`` (default ``"text"``)
  - ``label_field`` (default ``"label"``)
  - ``id_field`` (used for row_key when present)
  - ``allowed_labels`` (list[str]) — when set, rows with labels
    outside this set are rejected so the downstream classifier
    isn't asked to learn unexpected categories.

Emits ``TransformedRow.payload``:

    {"text": <str>, "label": <str>}

— matching what ``ClassificationHandler`` consumes.

Rejection reason codes:

  - ``parse_error`` — source-flagged unparseable row
  - ``missing_text`` — text column absent or empty after strip
  - ``missing_label`` — label column absent or empty after strip
  - ``label_not_allowed`` — label outside the ``allowed_labels`` set
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
    "label_field": "label",
    "id_field": "id",
}

_WHITESPACE_RUN = re.compile(r"\s+")


def _coerce_label(value: Any) -> str:
    """Normalize label to a clean string. Bools and ints round-trip
    through ``str`` so True/False/0/1 datasets work; whitespace is
    stripped + collapsed."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return _WHITESPACE_RUN.sub(" ", str(value)).strip()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RUN.sub(" ", str(value)).strip()


class LabelToClassificationMapper:
    mapper_id: str = "label_to_classification"

    def declared_target(self) -> str:
        return "classification"

    def transform(
        self,
        rows: Iterable[RawRow],
        field_map: dict[str, Any],
        *,
        ctx: ImportContext,
    ) -> Iterable[TransformedRow | RejectedRow]:
        fields = {
            **DEFAULT_FIELDS,
            **{k: v for k, v in (field_map or {}).items() if isinstance(v, str)},
        }
        allowed_raw = (field_map or {}).get("allowed_labels")
        allowed: set[str] | None
        if isinstance(allowed_raw, list) and allowed_raw:
            allowed = {_coerce_label(item) for item in allowed_raw if item is not None}
        else:
            allowed = None

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
            label = _coerce_label(row.get(fields["label_field"]))

            if not text:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_text",
                    detail=f"column '{fields['text_field']}' empty after strip",
                    row_index=idx,
                )
                continue
            if not label:
                yield RejectedRow(
                    raw_row=row,
                    reason="missing_label",
                    detail=f"column '{fields['label_field']}' empty after strip",
                    row_index=idx,
                )
                continue
            if allowed is not None and label not in allowed:
                yield RejectedRow(
                    raw_row=row,
                    reason="label_not_allowed",
                    detail=f"label '{label}' not in allowed set {sorted(allowed)}",
                    row_index=idx,
                )
                continue

            row_id = row.get(fields["id_field"])
            row_key = f"{ctx.source_id}-{row_id}" if row_id is not None else None

            yield TransformedRow(
                payload={"text": text, "label": label},
                row_key=row_key,
            )


register_mapper("label_to_classification", LabelToClassificationMapper)

"""Schema introspection + mapping proposal (Phase B).

Three steps:

1. ``sniff_columns`` — looks at the first ~20 sample rows from a
   source's ``describe()`` call and classifies each column by content
   shape (``text_like``, ``categorical``, ``bio_tag_list``,
   ``entity_list_json``, ``chat_messages``, ``numeric``, ``boolean``,
   ``path_like``, ``unknown``).

2. ``detect_shape`` — combines column-type signatures into a task
   hypothesis with confidence (e.g. ``text_like`` + ``bio_tag_list``
   of matching length → span_set NER, confidence 0.95).

3. ``propose_mapping`` — turns the hypothesis into a ``ProposedMapping``
   for one of the registered target mappers, with a ``field_map`` the
   CLI / UI can pass to ``preview_import`` / ``run_import``.

**Architectural rule** (locked in via the plan): introspection NEVER
silently auto-picks. It always emits a ``ProposedMapping`` with
confidence + rationale; the user confirms via the ``--auto`` /
``--force`` CLI flow or the Phase F UI wizard.

Confidence < 0.8 is "weak proposal — needs user override". Code that
runs from the proposal must enforce this gate explicitly.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.services.dataset_import.protocols import ProposedMapping


CONFIDENCE_HIGH: float = 0.8
"""Threshold above which a proposal is considered safe to auto-run.
Below this the CLI requires ``--force`` and the UI shows a warning."""


# ── Column types ──────────────────────────────────────────────────────


TEXT_LIKE = "text_like"
CATEGORICAL = "categorical"
BIO_TAG_LIST = "bio_tag_list"
ENTITY_LIST_JSON = "entity_list_json"
CHAT_MESSAGES = "chat_messages"
TOKENS_LIST = "tokens_list"
NUMERIC = "numeric"
BOOLEAN = "boolean"
PATH_LIKE = "path_like"
UNKNOWN = "unknown"


@dataclass
class ColumnSignature:
    """Single column's introspection result.

    ``confidence`` is per-column — how sure the sniffer is about the
    type assignment. ``unique_values`` is populated for categorical-
    candidate columns so the shape detector can read the candidate
    label set without re-scanning.
    """

    name: str
    column_type: str
    confidence: float = 0.0
    unique_values: list[str] = field(default_factory=list)
    sample_value: Any = None
    notes: str = ""


@dataclass
class ShapeHypothesis:
    """The introspector's task-shape guess for a dataset.

    ``mapper_id`` references a registered target mapper. ``field_map``
    + ``target_task_profile`` will become the ``ProposedMapping`` the
    user confirms.
    """

    mapper_id: str
    target_task_profile: str
    field_map: dict[str, Any]
    confidence: float
    rationale: str
    warnings: list[str] = field(default_factory=list)


# ── Per-value type guesses ───────────────────────────────────────────


_BIO_TAG = re.compile(r"^(?:O|[BI]-\w+)$")
_PATH_LIKE = re.compile(r"[\\/]|\.\w{2,5}$")


def _is_bio_tag_list(value: Any) -> bool:
    """True iff every item is a BIO-style tag (``O`` / ``B-X`` / ``I-X``)."""

    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, str):
            return False
        if not _BIO_TAG.match(item.strip()):
            return False
    return True


def _is_entity_list_json(value: Any) -> bool:
    """True when value is a list of `{type, start, end, …}` dicts —
    NER-style annotations already in the canonical span shape."""

    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if "type" not in item or "start" not in item or "end" not in item:
            return False
    return True


def _is_chat_messages(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        role = item.get("role")
        content = item.get("content") if item.get("content") is not None else item.get("value")
        if not isinstance(role, str) or not isinstance(content, str):
            return False
    return True


def _is_tokens_list(value: Any) -> bool:
    """List of word-shaped strings. Used as a pair signal with
    bio_tag_list — pairs of (tokens, labels) of matching length are
    the load-bearing NER fingerprint."""

    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, str) or not item:
            return False
    return True


def _is_numeric_str(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if not isinstance(value, str):
        return False
    try:
        float(value.strip())
        return True
    except (ValueError, AttributeError):
        return False


def _is_boolean_str(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"true", "false", "0", "1", "yes", "no"}


def _looks_like_path(value: Any) -> bool:
    if not isinstance(value, str) or len(value) < 3:
        return False
    return bool(_PATH_LIKE.search(value))


# ── Column-level sniffing ────────────────────────────────────────────


def _classify_column(
    name: str, values: list[Any]
) -> ColumnSignature:
    """Pick the best-fitting type for a column from its sample values.

    The order of checks matters — list-shaped types are exclusive
    (bio_tag_list / entity_list_json / chat_messages / tokens_list
    can never co-occur), so we run the strict ones first. Scalar
    types use vote-counting across non-empty values.
    """

    non_empty = [v for v in values if v not in (None, "")]
    if not non_empty:
        return ColumnSignature(
            name=name,
            column_type=UNKNOWN,
            confidence=0.0,
            sample_value=None,
            notes="all sample values are empty",
        )

    # ── List-shaped signatures (strict — full list must match) ──
    if all(_is_bio_tag_list(v) for v in non_empty):
        return ColumnSignature(
            name=name,
            column_type=BIO_TAG_LIST,
            confidence=0.98,
            sample_value=non_empty[0],
        )
    if all(_is_entity_list_json(v) for v in non_empty):
        return ColumnSignature(
            name=name,
            column_type=ENTITY_LIST_JSON,
            confidence=0.95,
            sample_value=non_empty[0],
        )
    if all(_is_chat_messages(v) for v in non_empty):
        return ColumnSignature(
            name=name,
            column_type=CHAT_MESSAGES,
            confidence=0.95,
            sample_value=non_empty[0],
        )
    if all(_is_tokens_list(v) for v in non_empty):
        return ColumnSignature(
            name=name,
            column_type=TOKENS_LIST,
            confidence=0.9,
            sample_value=non_empty[0],
        )

    # ── Scalar signatures (vote-based) ──
    bool_votes = sum(1 for v in non_empty if _is_boolean_str(v))
    if bool_votes == len(non_empty) and len(non_empty) >= 2:
        return ColumnSignature(
            name=name,
            column_type=BOOLEAN,
            confidence=0.9,
            sample_value=non_empty[0],
        )
    numeric_votes = sum(1 for v in non_empty if _is_numeric_str(v))
    if numeric_votes == len(non_empty) and len(non_empty) >= 2:
        return ColumnSignature(
            name=name,
            column_type=NUMERIC,
            confidence=0.85,
            sample_value=non_empty[0],
        )

    # Only string-shaped scalars left.
    if all(isinstance(v, str) for v in non_empty):
        path_votes = sum(1 for v in non_empty if _looks_like_path(v))
        if path_votes / len(non_empty) >= 0.8:
            return ColumnSignature(
                name=name,
                column_type=PATH_LIKE,
                confidence=0.85,
                sample_value=non_empty[0],
            )

        # Categorical vs text_like split rests on two signals:
        # multi-word values are almost always free text; single-token
        # values are almost always labels. Cardinality refines the
        # categorical confidence but doesn't gate it.
        unique = Counter(v.strip() for v in non_empty if isinstance(v, str))
        unique_count = len(unique)
        avg_len = sum(len(v) for v in non_empty) / max(1, len(non_empty))
        max_len = max(len(v) for v in non_empty)
        has_spaces = any(" " in v for v in non_empty)

        # Strong text signal first: multi-word strings of reasonable
        # length are free text, even if the sample is small.
        if has_spaces and avg_len > 20:
            text_confidence = 0.85 if avg_len > 30 else 0.8
            return ColumnSignature(
                name=name,
                column_type=TEXT_LIKE,
                confidence=text_confidence,
                sample_value=non_empty[0],
            )

        # Categorical: single-token short values with a small label set.
        if (
            not has_spaces
            and unique_count <= 20
            and max_len <= 32
            and len(non_empty) >= 2
        ):
            cardinality_ratio = unique_count / len(non_empty)
            if cardinality_ratio < 1.0:
                return ColumnSignature(
                    name=name,
                    column_type=CATEGORICAL,
                    confidence=0.9 if cardinality_ratio < 0.5 else 0.75,
                    unique_values=sorted(unique.keys()),
                    sample_value=non_empty[0],
                )
            # All unique but every value is a short single token —
            # weakly categorical (small sample just didn't repeat yet).
            if max_len <= 20 and len(non_empty) < 5:
                return ColumnSignature(
                    name=name,
                    column_type=CATEGORICAL,
                    confidence=0.6,
                    unique_values=sorted(unique.keys()),
                    sample_value=non_empty[0],
                    notes="all sample values unique — weak signal",
                )

        # Default: text_like. Confidence rises with longer + more
        # varied values; very short, low-variation strings get a
        # lower score so the shape detector can downweight them.
        text_confidence = 0.5
        if avg_len > 30:
            text_confidence = 0.85
        elif avg_len > 15:
            text_confidence = 0.7
        return ColumnSignature(
            name=name,
            column_type=TEXT_LIKE,
            confidence=text_confidence,
            sample_value=non_empty[0],
        )

    return ColumnSignature(
        name=name,
        column_type=UNKNOWN,
        confidence=0.2,
        sample_value=non_empty[0],
        notes="mixed / unrecognized value types",
    )


def sniff_columns(sample_rows: list[dict[str, Any]]) -> dict[str, ColumnSignature]:
    """Walk the sample rows and produce a ColumnSignature per column.

    Pulls values from every sample row (skipping missing keys) so
    sparse columns still get classified by the rows that have them.
    """

    if not sample_rows:
        return {}
    columns: dict[str, list[Any]] = {}
    for row in sample_rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            columns.setdefault(str(key), []).append(value)
    return {name: _classify_column(name, vals) for name, vals in columns.items()}


# ── Shape detection ──────────────────────────────────────────────────


def _column_lengths_match(
    rows: list[dict[str, Any]], col_a: str, col_b: str
) -> bool:
    """Both columns must be list-shaped of the same length across the
    sample. Critical for bio_to_spans — tokens + labels of mismatched
    lengths would be rejected per-row, but if even the headline
    sample disagrees, the dataset is structurally wrong."""

    pairs = 0
    matching = 0
    for row in rows:
        a, b = row.get(col_a), row.get(col_b)
        if isinstance(a, list) and isinstance(b, list):
            pairs += 1
            if len(a) == len(b):
                matching += 1
    return pairs >= 1 and matching == pairs


def detect_shape(
    signatures: dict[str, ColumnSignature],
    sample_rows: list[dict[str, Any]],
) -> list[ShapeHypothesis]:
    """Combine column types into ranked task hypotheses.

    Returns hypotheses sorted by confidence DESC. The CLI / UI picks
    the top one for ``--auto``; users can override by passing a
    specific ``mapper_id`` to bypass detection entirely.

    Phase B has two registered mappers (bio_to_spans,
    label_to_classification), so the detector returns hypotheses
    keyed on those. Phase C's mapper expansion adds detection rules
    for preference_pair / rag_passthrough / qa_pair / etc.
    """

    hypotheses: list[ShapeHypothesis] = []

    bio_columns = [
        s for s in signatures.values() if s.column_type == BIO_TAG_LIST
    ]
    tokens_columns = [
        s for s in signatures.values() if s.column_type == TOKENS_LIST
    ]
    text_columns = [
        s for s in signatures.values() if s.column_type == TEXT_LIKE
    ]
    categorical_columns = [
        s for s in signatures.values() if s.column_type == CATEGORICAL
    ]

    # ── bio_to_spans (NER) ──
    # Strong signal: a tokens column + a labels (BIO) column of the
    # same length on every sample row.
    for tokens_sig in tokens_columns:
        for labels_sig in bio_columns:
            if not _column_lengths_match(
                sample_rows, tokens_sig.name, labels_sig.name
            ):
                continue
            field_map: dict[str, Any] = {
                "tokens_field": tokens_sig.name,
                "labels_field": labels_sig.name,
            }
            # Optional full_text column boosts confidence when present
            # (alignment becomes deterministic instead of best-effort).
            for sig in text_columns:
                if sig.name.lower() in {"full_text", "fulltext", "text"}:
                    field_map["full_text_field"] = sig.name
                    break
            # Optional trailing_whitespace boolean-list column.
            trail_candidates = [
                s
                for s in signatures.values()
                if s.column_type == TOKENS_LIST and s.name != tokens_sig.name
            ]
            for sig in trail_candidates:
                if "whitespace" in sig.name.lower() or "trail" in sig.name.lower():
                    field_map["trailing_whitespace_field"] = sig.name
                    break
            hypotheses.append(
                ShapeHypothesis(
                    mapper_id="bio_to_spans",
                    target_task_profile="structured_extraction",
                    field_map=field_map,
                    confidence=0.95,
                    rationale=(
                        f"detected tokens column '{tokens_sig.name}' + "
                        f"BIO-tagged labels column '{labels_sig.name}' of "
                        "matching length"
                    ),
                )
            )

    # ── label_to_classification ──
    # Strong signal: one text_like + one categorical with a small
    # consistent label set.
    if text_columns and categorical_columns:
        # Prefer columns named "text" / "label" when present —
        # convention bonus on top of the type score.
        text_sig = next(
            (s for s in text_columns if s.name.lower() == "text"),
            text_columns[0],
        )
        label_sig = next(
            (s for s in categorical_columns if s.name.lower() in {"label", "class", "category"}),
            categorical_columns[0],
        )
        # Confidence: combine per-column confidences + a convention
        # bonus when names match. Cap at 0.95.
        base = (text_sig.confidence + label_sig.confidence) / 2
        name_bonus = 0.05 if text_sig.name.lower() == "text" else 0.0
        name_bonus += (
            0.05 if label_sig.name.lower() in {"label", "class", "category"} else 0.0
        )
        confidence = min(0.95, base + name_bonus)
        warnings: list[str] = []
        if len(label_sig.unique_values) > 10:
            warnings.append(
                f"label set has {len(label_sig.unique_values)} distinct "
                "values — verify this is intentional"
            )
        hypotheses.append(
            ShapeHypothesis(
                mapper_id="label_to_classification",
                target_task_profile="classification",
                field_map={
                    "text_field": text_sig.name,
                    "label_field": label_sig.name,
                    "allowed_labels": label_sig.unique_values,
                },
                confidence=confidence,
                rationale=(
                    f"detected text column '{text_sig.name}' + "
                    f"categorical column '{label_sig.name}' with "
                    f"{len(label_sig.unique_values)} distinct values "
                    f"({', '.join(label_sig.unique_values[:5])}"
                    + (", …" if len(label_sig.unique_values) > 5 else "")
                    + ")"
                ),
                warnings=warnings,
            )
        )

    hypotheses.sort(key=lambda h: -h.confidence)
    return hypotheses


# ── Proposal builder ─────────────────────────────────────────────────


def propose_mapping(
    sample_rows: list[dict[str, Any]],
) -> ProposedMapping | None:
    """One-shot helper: sniff + detect + return the best hypothesis
    as a ``ProposedMapping``. Returns None when no hypothesis matches
    any registered mapper.

    Callers that want the full ranked list (UI: shows alternative
    mappings beneath the top suggestion) should call ``sniff_columns``
    + ``detect_shape`` directly.
    """

    signatures = sniff_columns(sample_rows)
    if not signatures:
        return None
    hypotheses = detect_shape(signatures, sample_rows)
    if not hypotheses:
        return None
    top = hypotheses[0]
    return ProposedMapping(
        target_task_profile=top.target_task_profile,
        mapper_id=top.mapper_id,
        field_map=top.field_map,
        confidence=top.confidence,
        rationale=top.rationale,
        warnings=list(top.warnings),
    )


def signature_to_dict(sig: ColumnSignature) -> dict[str, Any]:
    """Serialize a ColumnSignature for API responses."""

    return {
        "name": sig.name,
        "column_type": sig.column_type,
        "confidence": round(sig.confidence, 4),
        "unique_values": sig.unique_values,
        "sample_value": sig.sample_value,
        "notes": sig.notes,
    }


def hypothesis_to_dict(hyp: ShapeHypothesis) -> dict[str, Any]:
    return {
        "mapper_id": hyp.mapper_id,
        "target_task_profile": hyp.target_task_profile,
        "field_map": hyp.field_map,
        "confidence": round(hyp.confidence, 4),
        "rationale": hyp.rationale,
        "warnings": hyp.warnings,
    }


def proposal_to_dict(proposal: ProposedMapping) -> dict[str, Any]:
    return {
        "target_task_profile": proposal.target_task_profile,
        "mapper_id": proposal.mapper_id,
        "field_map": proposal.field_map,
        "confidence": round(proposal.confidence, 4),
        "rationale": proposal.rationale,
        "warnings": proposal.warnings,
        "needs_force": proposal.confidence < CONFIDENCE_HIGH,
    }

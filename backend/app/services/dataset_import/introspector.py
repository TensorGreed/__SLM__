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


# span_offsets_passthrough: the share of sampled spans that must be integer
# offsets inside the text (and, where they carry text, match it).
SPAN_FIT_MIN: float = 0.8
# Share of sampled spans that must fit for a sample with a few bad spans to
# stay auto-runnable (the mapper rejects bad rows one by one, bulk-droppable).
SPAN_FIT_AUTO: float = 0.95
_SPAN_COLUMN_NAMES = {"entities", "spans", "annotations", "entity_spans", "mentions"}


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
CODE_LIKE = "code_like"
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
    also_text: bool = False
    """A code column whose values are also multi-word text: a plain text
    classifier may read it (ranked below the code readings)."""


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
    NER-style annotations already in the canonical span shape (``label``
    is accepted in place of ``type``)."""

    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if ("type" not in item and "label" not in item) or "start" not in item or "end" not in item:
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


_CODE_COLUMN_NAMES = {
    "code", "func", "function", "source_code", "snippet", "before", "after",
    "func_before", "func_after", "vulnerable_code", "fixed_code", "old_code", "new_code",
}
# Names that are ordinary words elsewhere (grammar-correction "before"/"after"):
# the values must read as code on their own.
_AMBIGUOUS_CODE_NAMES = {"before", "after"}
_CODE_STRUCT_END = re.compile(r"[{};]\s*$")
_CODE_KEYWORD = re.compile(
    r"^\s*(?:def \w+\s*\(|(?:async )?function\s*\w*\s*\(|class \w+|#include\s*<|import \w|from \S+ import \w|let \w+\s*="
    r"|(?:const|var) \w+\s*=|(?:public|private|protected|static)\s+\w+.*\(|fn \w+\s*\(|func \w+\s*\()"
    r"|\breturn\b[^\n]*;\s*$"
)
_CODE_BLOCK_HEAD = re.compile(
    r"(?:\{\s*$|^\s*(?:def|if|elif|else|for|while|try|except|with|class|switch|case|finally)\b.*:\s*$)"
)
_CODE_INDENT = re.compile(r"^(?: {2,}|\t)\S")


def _code_signal_kinds(value: Any) -> tuple[set[str], float]:
    """``(distinct signal kinds, fraction of non-empty lines that look like
    code)`` for a multi-line string. Kinds: ``struct`` (lines ending in
    ``;`` / ``{`` / ``}``), ``keyword`` (anchored declarations), ``indent`` (an
    indented line right after a block opener such as ``{`` or ``if x:``)."""
    if not isinstance(value, str):
        return set(), 0.0
    lines = [ln for ln in value.split("\n") if ln.strip()]
    if len(lines) < 2:
        return set(), 0.0
    kinds: set[str] = set()
    code_lines = 0
    previous_opens_block = False
    for ln in lines:
        hit = False
        if _CODE_STRUCT_END.search(ln):
            kinds.add("struct")
            hit = True
        if _CODE_KEYWORD.search(ln):
            kinds.add("keyword")
            hit = True
        if previous_opens_block and _CODE_INDENT.match(ln):
            kinds.add("indent")
            hit = True
        previous_opens_block = bool(_CODE_BLOCK_HEAD.search(ln))
        code_lines += hit
    return kinds, code_lines / len(lines)


def _has_code_signals(value: Any) -> bool:
    """A multi-line string that reads as source code: at least two distinct
    kinds of code signal, on at least 30% of its non-empty lines."""
    kinds, fraction = _code_signal_kinds(value)
    return len(kinds) >= 2 and fraction >= 0.3


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
    # An empty list is a row with nothing to tag (a negative example): it
    # neither confirms nor contradicts the entity-list reading.
    entity_values = [v for v in non_empty if v != []]
    if entity_values and all(_is_entity_list_json(v) for v in entity_values):
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

    # ── Source code (multi-line values that read as code; a code-ish column name helps) ──
    if all(isinstance(v, str) for v in non_empty):
        code_votes = sum(1 for v in non_empty if _has_code_signals(v))
        lowered = name.strip().lower()
        named = lowered in _CODE_COLUMN_NAMES and lowered not in _AMBIGUOUS_CODE_NAMES
        # A code-ish name is only believed when the values contain code punctuation
        # (a column called "snippet" holding "ok now" is prose).
        punctuated = sum(1 for v in non_empty if any(ch in v for ch in "(){};=[]<>\n"))
        is_code = (named and punctuated / len(non_empty) >= 0.5) or code_votes / len(non_empty) >= 0.6
        if is_code:
            prosey = any(" " in v for v in non_empty) and sum(len(v) for v in non_empty) / len(non_empty) > 20
            return ColumnSignature(
                name=name,
                column_type=CODE_LIKE,
                confidence=0.92 if code_votes / len(non_empty) >= 0.6 else 0.85,
                sample_value=non_empty[0],
                notes=("column name says code" if named else "multi-line values with code signals")
                + (" and the values read as code" if named and code_votes else ""),
                also_text=prosey,
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


def _find_column(
    signatures: dict[str, ColumnSignature],
    names: tuple[str, ...],
    column_types: tuple[str, ...] | None = None,
) -> ColumnSignature | None:
    """Return the first signature whose name is in ``names`` (case
    insensitive) and whose type (if filtered) matches one of
    ``column_types``.

    Used by the convention-driven hypothesis builders (preference_pair,
    rag_passthrough, qa_pair, chat_messages_passthrough) so the
    detection rules read top-to-bottom rather than via deeply nested
    loops."""

    lower = {n.lower() for n in names}
    for sig in signatures.values():
        if sig.name.lower() not in lower:
            continue
        if column_types is not None and sig.column_type not in column_types:
            continue
        return sig
    return None


_LABEL_COLUMN_NAMES = ("label", "labels", "category", "class", "type", "tag")
_BEFORE_NAMES = ("before", "func_before", "vulnerable_code", "old_code")
_AFTER_NAMES = ("after", "func_after", "fixed_code", "new_code")
_CODE_PRIMARY_NAMES = ("code", "func", "function", "source_code", "snippet")
_BAD_FUNC = re.compile(r"\b(bad\w*)\s*\(", re.I)
_GOOD_FUNC = re.compile(r"\b(good\w*)\s*\(", re.I)


_NOT_A_LABEL = re.compile(r"(?i)^(?:language|lang|path|file|filename|file_path|filepath|project|repo|repository|url|commit|sha|hash|id|index)$")
_FLAG_NAMES = ("vulnerable", "is_vulnerable", "vuln", "defective", "buggy", "target", "label")


def _label_column(signatures: dict[str, ColumnSignature]) -> ColumnSignature | None:
    """The column most likely to hold the label of a code row: one named like
    a label, else the first categorical (or list-of-labels) column that is not
    a language / path / project-like column."""
    named = _find_column(signatures, _LABEL_COLUMN_NAMES, (CATEGORICAL, TEXT_LIKE, TOKENS_LIST))
    if named:
        return named
    return next(
        (s for s in signatures.values()
         if s.column_type in (CATEGORICAL, TOKENS_LIST) and not _NOT_A_LABEL.match(s.name.strip())),
        None,
    )


def _binary_flag_column(signatures: dict[str, ColumnSignature], sample_rows: list[dict[str, Any]]) -> ColumnSignature | None:
    """A 0/1 or true/false column named like a vulnerability flag."""
    ok = {"0", "1", "true", "false"}
    for sig in signatures.values():
        if sig.name.strip().lower() not in _FLAG_NAMES:
            continue
        values = [r.get(sig.name) for r in sample_rows if isinstance(r, dict) and r.get(sig.name) not in (None, "")]
        if values and all(str(v).strip().lower() in ok for v in values):
            return sig
    return None


def _detect_code_shapes(
    signatures: dict[str, ColumnSignature],
    sample_rows: list[dict[str, Any]],
) -> list[ShapeHypothesis]:
    """Hypotheses for code datasets: before/after pairs, labelled code, and
    files holding paired ``bad…`` / ``good…`` functions."""
    code_cols = [s for s in signatures.values() if s.column_type == CODE_LIKE]
    if not code_cols:
        return []
    out: list[ShapeHypothesis] = []
    label_sig = _label_column(signatures)
    path_sig = _find_column(signatures, ("path", "file", "filename", "file_path"))
    used: set[str] = set()

    before_sig = _find_column(signatures, _BEFORE_NAMES, (CODE_LIKE,))
    after_sig = _find_column(signatures, _AFTER_NAMES, (CODE_LIKE,))
    if before_sig and after_sig and label_sig:
        used |= {before_sig.name, after_sig.name}
        out.append(ShapeHypothesis(
            mapper_id="fix_pair",
            target_task_profile="structured_extraction",
            field_map={"before": before_sig.name, "after": after_sig.name, "label": label_sig.name},
            confidence=0.93,
            rationale=(
                f"detected before/after code columns '{before_sig.name}' / '{after_sig.name}' "
                f"+ label column '{label_sig.name}' — fix pairs become findings rows"
            ),
        ))

    contrast_col = next(
        (
            sig for sig in code_cols
            if sig.name not in used
            and any(
                isinstance(row.get(sig.name), str)
                and _BAD_FUNC.search(row[sig.name]) and _GOOD_FUNC.search(row[sig.name])
                for row in sample_rows if isinstance(row, dict)
            )
        ),
        None,
    )
    if contrast_col:
        used.add(contrast_col.name)
        field_map: dict[str, Any] = {"code": contrast_col.name}
        if path_sig:
            field_map["path"] = path_sig.name
        warnings: list[str] = []
        if label_sig:
            field_map["label"] = label_sig.name
        else:
            warnings.append("no label column found — set label_from_path (a regex over the path) or add a label column")
        out.append(ShapeHypothesis(
            mapper_id="contrast_functions",
            target_task_profile="structured_extraction",
            field_map=field_map,
            confidence=0.9 if label_sig else 0.65,
            rationale=(
                f"code column '{contrast_col.name}' defines both bad… and good… functions — "
                "each becomes a findings row (bad = finding, good = clean)"
            ),
            warnings=warnings,
        ))

    primary = next((s for s in code_cols if s.name not in used and s.name.lower() in _CODE_PRIMARY_NAMES), None) or next(
        (s for s in code_cols if s.name not in used), None
    )
    if primary and label_sig and label_sig.name != primary.name:
        field_map = {"code": primary.name, "label": label_sig.name}
        flag = _binary_flag_column(signatures, sample_rows)
        if flag is not None and flag.name == label_sig.name:
            flag = None
        if flag:
            field_map["vulnerable"] = flag.name
        out.append(ShapeHypothesis(
            mapper_id="code_label_to_findings",
            target_task_profile="structured_extraction",
            field_map=field_map,
            confidence=0.92,
            rationale=(
                f"detected code column '{primary.name}' + label column '{label_sig.name}'"
                + (f" (+ flag '{flag.name}')" if flag else "")
                + " — labelled code becomes findings rows"
            ),
        ))
    return out


def detect_shape(
    signatures: dict[str, ColumnSignature],
    sample_rows: list[dict[str, Any]],
) -> list[ShapeHypothesis]:
    """Combine column types into ranked task hypotheses.

    Returns hypotheses sorted by confidence DESC. The CLI / UI picks
    the top one for ``--auto``; users can override by passing a
    specific ``mapper_id`` to bypass detection entirely.

    Detection rules cover the Phase B + C mappers — bio_to_spans,
    span_offsets_passthrough, label_to_classification, chat_messages_passthrough, preference_pair,
    rag_passthrough, qa_pair_passthrough, text_only. The
    ``kv_to_structured`` mapper isn't detection-eligible (it requires
    an explicit ``fields`` list the introspector can't infer); the UI
    wizard will expose it as a "build my own" option.
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
    chat_columns = [
        s for s in signatures.values() if s.column_type == CHAT_MESSAGES
    ]
    entity_columns = [
        s for s in signatures.values() if s.column_type == ENTITY_LIST_JSON
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

    # ── span_offsets_passthrough (spans with character offsets) ──
    # A text column + a list of {type, start, end} dicts: the span shape
    # itself — but {label, start, end} lists are also time segments, event
    # lists, token-index annotations. Proposed only when the sample shows
    # character offsets into the text: integer offsets inside it on most
    # spans AND, where spans carry their text (``text`` / ``value``), that
    # text equals ``text[start:end]`` on most of them. Spans that carry no
    # text can't be checked: proposed only for a ``type``-keyed list under a
    # span-like column name, and below the auto-run gate.
    if entity_columns and text_columns:
        from app.services.dataset_import.mappers.span_offsets_passthrough import parse_offset

        text_sig = next((s for s in text_columns if s.name.lower() == "text"), None) or next(
            (s for s in text_columns if s.name.lower() in {"full_text", "content", "document", "body", "sentence"}),
            text_columns[0],
        )
        for ent_sig in entity_columns:
            checked = fit = with_text = text_match = 0
            label_keyed = False
            for row in sample_rows:
                text, ents = row.get(text_sig.name), row.get(ent_sig.name)
                if not isinstance(text, str) or not isinstance(ents, list):
                    continue
                for ent in ents:
                    if not isinstance(ent, dict):
                        continue
                    checked += 1
                    label_keyed = label_keyed or "type" not in ent
                    start, end = parse_offset(ent.get("start")), parse_offset(ent.get("end"))
                    if start is None or end is None or start < 0 or end > len(text) or end <= start:
                        continue
                    fit += 1
                    span_text = ent.get("text") if ent.get("text") is not None else ent.get("value")
                    if span_text is not None:
                        with_text += 1
                        text_match += int(text[start:end] == str(span_text))
            if not checked or fit / checked < SPAN_FIT_MIN:
                continue
            if with_text and text_match / with_text < SPAN_FIT_MIN:
                continue
            span_named = ent_sig.name.lower() in _SPAN_COLUMN_NAMES
            warnings_spans: list[str] = []
            if with_text:
                confidence = 0.95 if span_named else 0.9
                misfit = checked - fit + with_text - text_match
                if misfit:
                    # A few bad spans stay auto-runnable (those rows are rejected,
                    # bulk-droppable); more need a human look (below the gate).
                    mostly_fit = fit / checked >= SPAN_FIT_AUTO and text_match / with_text >= SPAN_FIT_AUTO
                    confidence = CONFIDENCE_HIGH + 0.05 if mostly_fit else 0.7
                    warnings_spans.append(
                        f"{misfit} of {checked} sampled spans don't fit '{text_sig.name}' "
                        "(offset out of range or span text differs) — the import rejects those rows "
                        "(bulk-droppable); check the text column"
                    )
            elif span_named and not label_keyed:
                confidence = 0.7
                warnings_spans.append(
                    "the spans carry no text, so the offsets can't be checked against "
                    f"'{text_sig.name}' — confirm they are character offsets into it"
                )
            else:
                continue
            hypotheses.append(
                ShapeHypothesis(
                    mapper_id="span_offsets_passthrough",
                    target_task_profile="structured_extraction",
                    field_map={"text_field": text_sig.name, "entities_field": ent_sig.name},
                    confidence=confidence,
                    rationale=(
                        f"detected text column '{text_sig.name}' + entity list "
                        f"'{ent_sig.name}' of {{type, start, end}} character-offset spans"
                    ),
                    warnings=warnings_spans,
                )
            )

    # ── label_to_classification ──
    # Strong signal: one text_like + one categorical with a small
    # consistent label set.
    # A code column whose values are also multi-word text can be read as text
    # too (ranked below the code readings, after the code shapes are known).
    classifier_texts = text_columns + [
        s for s in signatures.values() if s.column_type == CODE_LIKE and s.also_text
    ]
    if classifier_texts and categorical_columns:
        # Prefer columns named "text" / "label" when present —
        # convention bonus on top of the type score.
        text_sig = next(
            (s for s in classifier_texts if s.name.lower() == "text"),
            classifier_texts[0],
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

    # ── code datasets (findings rows) ──
    code_hypotheses = _detect_code_shapes(signatures, sample_rows)
    hypotheses.extend(code_hypotheses)
    if code_hypotheses:
        # A code column + a label is a code dataset: the plain text classifier
        # on some other text column is the weaker reading.
        ceiling = min(h.confidence for h in code_hypotheses) - 0.05
        for hyp in hypotheses:
            if hyp.mapper_id == "label_to_classification":
                hyp.confidence = min(hyp.confidence, ceiling)
                hyp.warnings.append(
                    "this dataset also has a code column — a code mapper ranks above this one"
                )

    # ── chat_messages_passthrough ──
    # A column already shaped as a list of {role, content} dicts is
    # an unambiguous chat-data signal.
    for chat_sig in chat_columns:
        hypotheses.append(
            ShapeHypothesis(
                mapper_id="chat_messages_passthrough",
                target_task_profile="chat_sft",
                field_map={"messages_field": chat_sig.name},
                confidence=0.95,
                rationale=(
                    f"detected chat-messages column '{chat_sig.name}' "
                    "(list of {role, content} dicts)"
                ),
            )
        )

    # ── preference_pair ──
    # Convention-driven: a {prompt, chosen, rejected} triple of text
    # columns. The mapper's per-row fallbacks (preferred/accepted/etc)
    # widen the input space, but the introspector keys off the
    # canonical names to avoid false positives.
    prompt_sig = _find_column(
        signatures, ("prompt", "question", "instruction", "input"),
        column_types=(TEXT_LIKE,),
    )
    chosen_sig = _find_column(
        signatures, ("chosen", "preferred", "accepted", "response_chosen"),
        column_types=(TEXT_LIKE,),
    )
    rejected_sig = _find_column(
        signatures, ("rejected", "dispreferred", "response_rejected", "negative"),
        column_types=(TEXT_LIKE,),
    )
    if prompt_sig and chosen_sig and rejected_sig:
        confidence = min(
            0.95,
            (prompt_sig.confidence + chosen_sig.confidence + rejected_sig.confidence) / 3
            + 0.1,  # convention bonus — these names are load-bearing
        )
        hypotheses.append(
            ShapeHypothesis(
                mapper_id="preference_pair",
                target_task_profile="dpo",
                field_map={
                    "prompt_field": prompt_sig.name,
                    "chosen_field": chosen_sig.name,
                    "rejected_field": rejected_sig.name,
                },
                confidence=confidence,
                rationale=(
                    f"detected preference triple: prompt='{prompt_sig.name}', "
                    f"chosen='{chosen_sig.name}', rejected='{rejected_sig.name}'"
                ),
            )
        )

    # ── rag_passthrough ──
    # {question, context, answer} triple. Context is the discriminator
    # against plain QA — without it we'd land on qa_pair_passthrough.
    question_sig = _find_column(
        signatures, ("question", "query", "prompt", "instruction", "input"),
        column_types=(TEXT_LIKE,),
    )
    context_sig = _find_column(
        signatures, ("context", "passage", "document", "evidence", "retrieved_context"),
        column_types=(TEXT_LIKE,),
    )
    # Answer can be short ("Paris", "1945") and get classified as
    # categorical on a small sample — accept either type here. The
    # presence of a conventionally-named context column is already
    # the strong RAG signal; we just need *some* answer column.
    answer_sig = _find_column(
        signatures, ("answer", "reference", "completion", "response", "output", "target_text"),
        column_types=(TEXT_LIKE, CATEGORICAL),
    )
    if question_sig and context_sig and answer_sig:
        confidence = min(
            0.95,
            (question_sig.confidence + context_sig.confidence + answer_sig.confidence) / 3
            + 0.1,
        )
        hypotheses.append(
            ShapeHypothesis(
                mapper_id="rag_passthrough",
                target_task_profile="rag_qa",
                field_map={
                    "question_field": question_sig.name,
                    "context_field": context_sig.name,
                    "answer_field": answer_sig.name,
                },
                confidence=confidence,
                rationale=(
                    f"detected RAG triple: question='{question_sig.name}', "
                    f"context='{context_sig.name}', answer='{answer_sig.name}'"
                ),
            )
        )

    # ── summarization ──
    # {long text, summary} with no question column. Routed through the
    # qa passthrough mapper (source → question, summary → answer); the
    # seq2seq-pair adapter then wraps it in the Summarize prompt.
    summary_sig = _find_column(
        signatures,
        ("summary", "abstract", "highlights", "tldr", "tl_dr", "synopsis", "headline", "gist"),
        column_types=(TEXT_LIKE, CATEGORICAL),
    )
    source_sig = _find_column(
        signatures,
        ("article", "document", "text", "content", "body", "passage", "dialogue", "transcript", "source"),
        column_types=(TEXT_LIKE,),
    )
    if summary_sig and source_sig and not question_sig and source_sig.name != summary_sig.name:
        confidence = min(0.93, (summary_sig.confidence + source_sig.confidence) / 2 + 0.12)
        hypotheses.append(
            ShapeHypothesis(
                mapper_id="qa_pair_passthrough",
                target_task_profile="summarization",
                field_map={
                    "question_field": source_sig.name,
                    "answer_field": summary_sig.name,
                },
                confidence=confidence,
                rationale=(
                    f"detected text '{source_sig.name}' + summary '{summary_sig.name}'"
                ),
            )
        )

    # ── qa_pair_passthrough ──
    # {question, answer} WITHOUT a context column. We deliberately
    # don't propose QA when context exists — rag_passthrough is the
    # stronger interpretation in that case.
    if question_sig and answer_sig and not context_sig:
        confidence = min(
            0.92,
            (question_sig.confidence + answer_sig.confidence) / 2 + 0.1,
        )
        hypotheses.append(
            ShapeHypothesis(
                mapper_id="qa_pair_passthrough",
                target_task_profile="qa",
                field_map={
                    "question_field": question_sig.name,
                    "answer_field": answer_sig.name,
                },
                confidence=confidence,
                rationale=(
                    f"detected QA pair: question='{question_sig.name}', "
                    f"answer='{answer_sig.name}'"
                ),
            )
        )

    # ── text_only ──
    # Fallback for pure LM corpora: exactly one text column, nothing
    # else useful to anchor on. Confidence stays modest so any of the
    # stronger hypotheses above outranks it.
    if (
        len(text_columns) == 1
        and not categorical_columns
        and not bio_columns
        and not chat_columns
        and not (question_sig and answer_sig)
    ):
        text_sig = text_columns[0]
        # If column is named "text" convention bonus; otherwise the
        # underlying text_confidence carries the proposal.
        name_bonus = 0.05 if text_sig.name.lower() == "text" else 0.0
        confidence = min(0.9, text_sig.confidence + name_bonus)
        hypotheses.append(
            ShapeHypothesis(
                mapper_id="text_only",
                target_task_profile="language_modeling",
                field_map={"text_field": text_sig.name},
                confidence=confidence,
                rationale=(
                    f"single text column '{text_sig.name}' with no "
                    "label / structured-extraction signal — best fit for "
                    "plain LM training"
                ),
            )
        )

    # A text column next to a span list that passed the checks above: text_only
    # would silently drop every label, so it ranks below the span mapper (and
    # says so). When the span mapper is below the auto-run gate, --auto then
    # stops and asks instead of picking text_only.
    span_hyps = [h for h in hypotheses if h.mapper_id == "span_offsets_passthrough"]
    if span_hyps:
        top_span = max(h.confidence for h in span_hyps)
        span_cols = ", ".join(f"'{h.field_map['entities_field']}'" for h in span_hyps)
        for hyp in hypotheses:
            if hyp.mapper_id == "text_only":
                hyp.confidence = min(hyp.confidence, top_span - 0.05)
                hyp.warnings.append(
                    f"ignores the span column {span_cols}: every label would be dropped — "
                    "use span_offsets_passthrough to keep them"
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

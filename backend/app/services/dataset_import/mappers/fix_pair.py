"""Before / after code pairs (a defect and its fix) → findings, patch or preference rows.

One input row = the code before a fix, the code after it, and a label for
what the fix repaired. Depending on ``emit`` it produces:

  - ``findings`` (default) — a positive findings row (code = before, one
    finding per label on the lines the fix changed or deleted) and a negative
    one (code = after, no findings)
  - ``patch`` — ``{row_kind: "patch", code: before, language, path, start_line,
    labels: [ids], patch: <unified diff>, fixed_code: after}``
  - ``preference`` — ``{row_kind: "preference", prompt, chosen: after,
    rejected: before, reference: after, labels: [ids]}`` (the keys of
    ``preference_pair``)

Rows of different ``row_kind`` do not belong in one training set: import
patch / preference rows into their own project.

field_map (all optional):

  - ``before`` (default ``before``; also ``func_before`` / ``vulnerable_code`` /
    ``old_code``), ``after`` (default ``after``; also ``func_after`` /
    ``fixed_code`` / ``new_code``)
  - ``label`` (see ``code_label_to_findings``), ``language``, ``path``,
    ``start_line`` (file line of the first line of ``before``; int or column)
  - ``emit`` — list of ``findings`` / ``patch`` / ``preference``
  - ``finding_lines`` — ``"hunk"`` (default: one finding per label per changed
    region of ``before``, regions at most 2 lines apart merged; a pure
    insertion anchors on the line before the insertion point), ``"span"`` (one
    finding per label from the first to the last changed line) or ``"whole"``
    (the whole ``before``)
  - label-leak scrubbing (off by default; see ``contrast_functions``):
    ``anonymize_symbols``, ``blank_comments``, ``comment_pattern``,
    ``bad_pattern``, ``good_pattern``
  - ``id_field``, ``clean_labels``, ``languages``, ``label_prefix``,
    ``allow_partial_labels``, ``max_code_chars``

row_key: ``<source>-<id>`` (or ``<source>-<row index>``) plus ``-pos``,
``-neg``, ``-patch`` or ``-pref``.

Declared target: ``structured_extraction``.

Rejection reasons: ``parse_error``, ``empty_code``, ``missing_label``,
``unknown_label``, ``code_too_long``, ``language_out_of_scope``,
``identical_pair`` (before == after ignoring trailing whitespace per line),
``ambiguous_label``.
"""

from __future__ import annotations

import difflib
from typing import Any, Iterable

from app.services.dataset_import.mappers import _findings_common as fc
from app.services.dataset_import.protocols import (
    ImportContext,
    RawRow,
    RejectedRow,
    TransformedRow,
)
from app.services.dataset_import.registry import register_mapper

_BEFORE = ("before", "func_before", "vulnerable_code", "old_code")
_AFTER = ("after", "func_after", "fixed_code", "new_code")
_EMIT = ("findings", "patch", "preference")


def changed_hunks(before: str, after: str) -> list[tuple[int, int]]:
    """1-based inclusive ranges (in ``before``), one per change: the lines the
    fix changed or deleted; a pure insertion anchors on the line before the
    insertion point (clamped into the code). Lines are compared ignoring
    trailing whitespace. Empty when nothing changed."""
    a, b = [ln.rstrip() for ln in before.split("\n")], [ln.rstrip() for ln in after.split("\n")]
    hunks: list[tuple[int, int]] = []
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            hunks.append((i1 + 1, i2))
        else:
            anchor = max(1, min(i1, len(a)))
            hunks.append((anchor, anchor))
    return hunks


def merge_hunks(hunks: list[tuple[int, int]], gap: int = 2) -> list[tuple[int, int]]:
    """Merge hunks that are at most ``gap`` unchanged lines apart."""
    merged: list[tuple[int, int]] = []
    for lo, hi in hunks:
        if merged and lo - merged[-1][1] - 1 <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def changed_before_lines(before: str, after: str) -> tuple[int, int] | None:
    """The ``span`` of the fix: from the first to the last changed line."""
    hunks = changed_hunks(before, after)
    return (min(h[0] for h in hunks), max(h[1] for h in hunks)) if hunks else None


def unified_patch(before: str, after: str, path: str | None) -> str:
    name = path or "code"
    return "\n".join(difflib.unified_diff(before.split("\n"), after.split("\n"), f"a/{name}", f"b/{name}", n=3, lineterm=""))


class FixPairMapper:
    mapper_id: str = "fix_pair"

    def declared_target(self) -> str:
        return "structured_extraction"

    def transform(
        self,
        rows: Iterable[RawRow],
        field_map: dict[str, Any],
        *,
        ctx: ImportContext,
    ) -> Iterable[TransformedRow | RejectedRow]:
        field_map = field_map or {}
        try:
            opts = fc.read_options(field_map)
            emit = field_map.get("emit", ["findings"])
            if isinstance(emit, str):
                emit = [emit]
            if not isinstance(emit, (list, tuple)) or not emit or any(e not in _EMIT for e in emit):
                raise ValueError(f"emit must be a non-empty list of {list(_EMIT)}")
            mode = field_map.get("finding_lines", "hunk")
            if mode not in ("hunk", "span", "whole"):
                raise ValueError("finding_lines must be 'hunk', 'span' or 'whole'")
        except ValueError as exc:
            for idx, row in enumerate(rows):
                yield RejectedRow(raw_row=row, reason=fc.PARSE_ERROR, detail=str(exc), row_index=idx)
            return
        taxonomy = fc.taxonomy_of(ctx)
        version = taxonomy.version if taxonomy else None
        keys = fc.KeyMaker()

        for idx, row in enumerate(rows):
            if "__parse_error__" in row:
                yield RejectedRow(raw_row=row, reason=fc.PARSE_ERROR, detail=str(row.get("__parse_error__")), row_index=idx)
                continue
            try:
                out = self._row(row, idx, field_map, ctx, opts, list(emit), mode, taxonomy, version, keys)
            except fc.RowReject as rej:
                yield RejectedRow(raw_row=row, reason=rej.reason, detail=rej.detail, row_index=idx)
                continue
            except Exception as exc:  # noqa: BLE001 - one odd row must not kill the import
                reason, detail = fc.unexpected(exc)
                yield RejectedRow(raw_row=row, reason=reason, detail=detail, row_index=idx)
                continue
            yield from out

    def _row(self, row, idx, field_map, ctx, opts, emit, mode, taxonomy, version, keys) -> list[TransformedRow]:
        before = fc.clean_code(fc.pick(row, field_map, "before", _BEFORE))
        after = fc.clean_code(fc.pick(row, field_map, "after", _AFTER))
        fc.check_code(before, opts, "before")
        fc.check_code(after, opts, "after")
        if [ln.rstrip() for ln in before.split("\n")] == [ln.rstrip() for ln in after.split("\n")]:
            raise fc.RowReject(fc.IDENTICAL_PAIR, "before and after are identical apart from trailing whitespace")
        path = fc.path_of(row, field_map)
        language = fc.resolve_language(fc.pick(row, field_map, "language", fc.LANGUAGE_COLUMNS), path, before)
        fc.check_language(language, opts)
        raw_value = fc.pick(row, field_map, "label", fc.LABEL_COLUMNS)
        if raw_value is None:
            raise fc.RowReject(fc.MISSING_LABEL, "label column (and fallbacks) empty")
        labels = fc.resolve_labels(fc.split_labels(raw_value, opts), opts, taxonomy)
        if labels.clean:
            raise fc.RowReject(fc.MISSING_LABEL, "every label means 'no finding'; a fix pair needs the label of what it fixed")
        start_line = fc.int_column(row, field_map, "start_line", 1)
        key = keys.base(ctx, row, idx, field_map)
        mapping: dict[str, str] = {}
        before = fc.scrub(before, language, opts, mapping)
        after = fc.scrub(after, language, opts, mapping)
        warnings = list(labels.warnings)

        out: list[TransformedRow] = []
        if "findings" in emit:
            if mode == "hunk":
                ranges = merge_hunks(changed_hunks(before, after))
            elif mode == "span":
                span = changed_before_lines(before, after)
                ranges = [span] if span else []
            else:
                ranges = []
            spans_abs = [(lo + start_line - 1, hi + start_line - 1) for lo, hi in ranges] or [fc.whole_span(before, start_line)]
            findings = [fc.make_finding(label, sp, before, start_line) for label in labels.ids for sp in spans_abs]
            out.append(TransformedRow(
                payload=fc.findings_payload(code=before, language=language, path=path, start_line=start_line,
                                            findings=findings, labels_raw=labels.raw, taxonomy_version=version),
                row_key=f"{key}-pos", warnings=list(warnings),
            ))
            out.append(TransformedRow(
                payload=fc.findings_payload(code=after, language=language, path=path, start_line=start_line,
                                            findings=[], labels_raw=[], taxonomy_version=version),
                row_key=f"{key}-neg", warnings=list(warnings),
            ))
        if "patch" in emit:
            out.append(TransformedRow(
                payload={
                    "row_kind": "patch", "code": before, "language": language, "path": path, "start_line": start_line,
                    "labels": list(labels.ids), "patch": unified_patch(before, after, path), "fixed_code": after,
                    "taxonomy_version": version,
                },
                row_key=f"{key}-patch", warnings=list(warnings),
            ))
        if "preference" in emit:
            titles = ", ".join(fc.label_title(taxonomy, label) for label in labels.ids)
            noun = f"{language} code" if language else "code"
            out.append(TransformedRow(
                payload={
                    "row_kind": "preference",
                    "prompt": f"Fix the {titles} issue in this {noun}:\n\n{before}",
                    "chosen": after, "rejected": before, "reference": after,
                    "labels": list(labels.ids), "taxonomy_version": version,
                },
                row_key=f"{key}-pref", warnings=list(warnings),
            ))
        return out


register_mapper("fix_pair", FixPairMapper)

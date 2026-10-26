"""Labelled code → findings rows (a piece of code + the labelled findings in it).

One input row = one piece of code (a function, a snippet, a file chunk) with
a label column naming what is wrong with it (or nothing). Labels resolve
through the project's label taxonomy; the output is the canonical findings row
described in ``_findings_common``.

field_map (all optional):

  - ``code`` — code column (default ``code``; also ``func`` / ``function`` /
    ``source`` / ``source_code`` / ``snippet`` / ``text``)
  - ``label`` — label column (default ``label``; also ``labels`` / ``category``
    / ``class`` / ``type`` / ``tag``). A string is split on ``label_separator``
    (regex, default ``[,;|]``); JSON-list strings and real lists work too.
  - ``vulnerable`` — a boolean-ish column: truthy → the row is labelled,
    falsy → clean regardless of the label column
  - ``start_line`` — file line of the code's first line: an int or a column
    name (default 1)
  - ``line`` / ``end_line`` — finding lines: a column holding an int, a range
    ``"12-15"`` or a list (one entry per label). Absolute by default, relative to
    the code with ``lines_relative: true``. No line column → each finding spans
    the whole code and the row gets the warning ``lines_unknown``.
  - ``attr_columns`` — ``{attr: column}`` copied into every finding's attrs
  - ``evidence`` — evidence column (default: the finding's own lines)
  - ``language`` / ``path`` — columns; the language falls back to the path's
    extension
  - ``id_field`` (row_key), ``anonymize_symbols`` / ``blank_comments`` /
    ``comment_pattern`` / ``bad_pattern`` / ``good_pattern`` (label-leak
    scrubbing, off by default; see ``contrast_functions``), plus the shared options: ``clean_labels``,
    ``languages``, ``label_prefix``, ``allow_partial_labels``, ``max_code_chars``

Declared target: ``structured_extraction``.

Rejection reasons: ``parse_error``, ``empty_code``, ``missing_label``,
``unknown_label``, ``ambiguous_label``, ``line_out_of_range``,
``line_count_mismatch``, ``code_too_long``, ``language_out_of_scope``.
One label with several line spans gives one finding per span.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.services.dataset_import.mappers import _findings_common as fc
from app.services.dataset_import.protocols import (
    ImportContext,
    RawRow,
    RejectedRow,
    TransformedRow,
)
from app.services.dataset_import.registry import register_mapper

_CODE_COLUMNS = ("code", "func", "function", "source", "source_code", "snippet", "text")


class CodeLabelToFindingsMapper:
    mapper_id: str = "code_label_to_findings"

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
                payload, warnings = self._row(row, field_map, opts, taxonomy, version)
            except fc.RowReject as rej:
                yield RejectedRow(raw_row=row, reason=rej.reason, detail=rej.detail, row_index=idx)
                continue
            except Exception as exc:  # noqa: BLE001 - one odd row must not kill the import
                reason, detail = fc.unexpected(exc)
                yield RejectedRow(raw_row=row, reason=reason, detail=detail, row_index=idx)
                continue
            yield TransformedRow(payload=payload, row_key=keys.base(ctx, row, idx, field_map), warnings=warnings)

    def _row(self, row, field_map, opts, taxonomy, version) -> tuple[dict[str, Any], list[str]]:
        code = fc.clean_code(fc.pick(row, field_map, "code", _CODE_COLUMNS))
        fc.check_code(code, opts)
        path = fc.path_of(row, field_map)
        language = fc.resolve_language(fc.pick(row, field_map, "language", fc.LANGUAGE_COLUMNS), path, code)
        fc.check_language(language, opts)
        code = fc.scrub(code, language, opts)

        raw_value = fc.pick(row, field_map, "label", fc.LABEL_COLUMNS)
        flag_value = row.get(field_map["vulnerable"]) if isinstance(field_map.get("vulnerable"), str) else None
        flagged = fc.truthy(flag_value) if not fc.blank(flag_value) else None
        if raw_value is None and not fc.has_column(row, field_map, "label", fc.LABEL_COLUMNS):
            if flagged is None or flagged:
                raise fc.RowReject(fc.MISSING_LABEL, "no label column in the row")
        if raw_value is None and flagged is None:
            raw_value = ""  # present but empty: a "no finding" marker, subject to clean_labels
            if "" not in opts.clean_labels:
                raise fc.RowReject(fc.MISSING_LABEL, "label column is empty")
        raw_labels = fc.split_labels(raw_value, opts) if raw_value is not None else []
        if flagged is False:
            labels = fc.ResolvedLabels([], [], [fc.WARN_NO_TAXONOMY] if taxonomy is None else [], clean=True)
            labels.raw = raw_labels
        else:
            if flagged and not raw_labels:
                raise fc.RowReject(fc.MISSING_LABEL, "marked vulnerable but the label column is empty")
            labels = fc.resolve_labels(raw_labels or [""], opts, taxonomy)
            if flagged and labels.clean:
                raise fc.RowReject(fc.MISSING_LABEL, "marked vulnerable but every label means 'no finding'")
        warnings = list(labels.warnings)

        start_line = fc.int_column(row, field_map, "start_line", 1)
        findings: list[dict[str, Any]] = []
        if labels.ids:
            spans = fc.parse_spans(
                row.get(field_map["line"]) if isinstance(field_map.get("line"), str) else row.get("line"),
                row.get(field_map["end_line"]) if isinstance(field_map.get("end_line"), str) else row.get("end_line"),
            )
            if spans is None:
                spans_abs = [(i, fc.whole_span(code, start_line)) for i in range(len(labels.ids))]
                warnings.append(fc.WARN_LINES_UNKNOWN)
            else:
                spans_abs = fc.absolute_spans(
                    spans, start_line=start_line, line_count=fc.line_count(code),
                    relative=bool(field_map.get("lines_relative")), labels=len(labels.ids),
                )
            attrs = {}
            attr_cols = field_map.get("attr_columns")
            if isinstance(attr_cols, dict):
                attrs = {str(k): fc.jsonable(row.get(c)) for k, c in attr_cols.items() if isinstance(c, str) and row.get(c) is not None}
            evidence = row.get(field_map["evidence"]) if isinstance(field_map.get("evidence"), str) else None
            for i, (label_idx, span) in enumerate(spans_abs):
                ev = evidence[i] if isinstance(evidence, list) and len(evidence) == len(spans_abs) else (None if isinstance(evidence, list) else evidence)
                findings.append(fc.make_finding(labels.ids[label_idx], span, code, start_line, ev, attrs))
        payload = fc.findings_payload(
            code=code, language=language, path=path, start_line=start_line, findings=findings,
            labels_raw=labels.raw, taxonomy_version=version,
        )
        return payload, warnings


register_mapper("code_label_to_findings", CodeLabelToFindingsMapper)

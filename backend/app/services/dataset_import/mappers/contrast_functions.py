"""Source files with matching bad / good functions → findings rows.

Some code suites teach by contrast: each file holds functions named for what
they demonstrate (``badLeak`` = a defect, ``goodLeak`` = its safe twin). One
input row = one source file (the ``git:`` source emits exactly that). The file
is split into functions / methods with the code chunker; a function whose
name matches ``bad_pattern`` becomes a findings row with one finding per label
spanning the whole function, a function matching ``good_pattern`` becomes a
clean row (``findings: []``); every other function is ignored (counted in a
row warning).

field_map (all optional):

  - ``code`` — file text column (default ``code``; also ``text`` / ``content``)
  - ``path`` — path column (default ``path``; also ``file`` / ``filename`` /
    ``file_path``); ``language`` — column, else detected from the path
  - ``label`` — label column, or ``label_from_path`` — a regex whose first
    capture group, applied to the path, is the label (e.g. ``(?:^|/)([A-Z]+-\\d+)_``).
    Labels resolve through the project's taxonomy like the other findings
    mappers.
  - ``bad_pattern`` (default ``(?i)^bad``), ``good_pattern`` (default
    ``(?i)^good``) — Python ``re``, matched at the start of the LAST segment of
    the function's symbol (``Demo.badLeak`` → ``badLeak``)
  - ``anonymize_symbols`` (default true) — rewrite every identifier matching
    ``bad_pattern`` / ``good_pattern`` (the function's own name AND its
    callees, e.g. ``badSink``, ``goodG2B``) to ``fn_<n>``, consistently within
    a row. ``blank_comments`` — ``"none"`` / ``"matching"`` (default) / ``"all"``:
    blank comments (matching ``comment_pattern``, default words like
    flaw / fix / bad / good / vulnerable) in place, keeping line numbers.
    LEAKAGE RISK: if the names and comments are left in, a model learns
    ``bad*`` = defect instead of the defect; the emitted ``path`` is verbatim,
    so a path that spells the answer leaks too.
  - ``id_field``, ``clean_labels``, ``languages``, ``label_prefix``,
    ``allow_partial_labels``, ``max_code_chars`` (per function)

row_key: ``<file key>-<symbol>`` (``#2``, ``#3`` … for repeated names); the file
key is ``<source>-<id>``, else ``<source>-<path>``, else ``<source>-<row index>``.

Declared target: ``structured_extraction``.

Rejection reasons: ``parse_error`` (also a bad regex), ``empty_code``,
``language_out_of_scope``, ``no_variants`` (no bad / good function found),
``missing_label``, ``unknown_label``, ``ambiguous_label``, ``code_too_long`` (per function).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.services.code_chunker import chunk_code
from app.services.dataset_import.mappers import _findings_common as fc
from app.services.dataset_import.protocols import (
    ImportContext,
    RawRow,
    RejectedRow,
    TransformedRow,
)
from app.services.dataset_import.registry import register_mapper

_CODE_COLUMNS = ("code", "text", "content")
_SYMBOL_SPLIT = re.compile(r"::|[./]")
_HUGE = 10**8
_LONE_CR = re.compile(r"\r")


class ContrastFunctionsMapper:
    mapper_id: str = "contrast_functions"

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
            opts = fc.read_options(field_map, neutralise_by_default=True)
            bad, good = opts.bad_pattern, opts.good_pattern
            from_path = re.compile(field_map["label_from_path"]) if field_map.get("label_from_path") else None
            if from_path is not None and from_path.groups < 1:
                raise ValueError("label_from_path needs a capture group")
        except (ValueError, re.error) as exc:
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
                yield from self._file(row, idx, field_map, ctx, opts, bad, good, from_path, taxonomy, version, keys)
            except fc.RowReject as rej:
                yield RejectedRow(raw_row=row, reason=rej.reason, detail=rej.detail, row_index=idx)
            except Exception as exc:  # noqa: BLE001 - one odd file must not kill the import
                reason, detail = fc.unexpected(exc)
                yield RejectedRow(raw_row=row, reason=reason, detail=detail, row_index=idx)

    def _file(self, row, idx, field_map, ctx, opts, bad, good, from_path, taxonomy, version, keys):
        text = fc.clean_code(fc.pick(row, field_map, "code", _CODE_COLUMNS))
        if not text.strip():
            raise fc.RowReject(fc.EMPTY_CODE, "file text (column 'code' and fallbacks) empty after strip")
        path = fc.path_of(row, field_map)
        language = fc.resolve_language(fc.pick(row, field_map, "language", fc.LANGUAGE_COLUMNS), path, text)
        fc.check_language(language, opts)

        lines = text.split("\n")
        bads: list[tuple[str, int, int]] = []
        goods: list[tuple[str, int, int]] = []
        ignored = 0
        # The chunker treats a lone CR as a line break; finding lines are counted
        # on "\n" only, so hide lone CRs from it (same length, nothing moves).
        for chunk in chunk_code(_LONE_CR.sub(" ", text), language, max_chars=_HUGE):
            if chunk.kind not in ("function", "method") or not chunk.symbol:
                continue
            name = _SYMBOL_SPLIT.split(chunk.symbol)[-1]
            first, last = chunk.unit_start_line, chunk.unit_end_line
            while last > first and not lines[last - 1].strip():
                last -= 1
            if bad.search(name):
                bads.append((chunk.symbol, first, last))
            elif good.search(name):
                goods.append((chunk.symbol, first, last))
            else:
                ignored += 1
        if not bads and not goods:
            raise fc.RowReject(fc.NO_VARIANTS, f"no function matching bad_pattern or good_pattern found ({ignored} other function(s))")

        raw_value = fc.pick(row, field_map, "label", fc.LABEL_COLUMNS)
        if raw_value is None and from_path is not None:
            match = from_path.search(path or "")
            raw_value = match.group(1) if match and match.group(1) else None
        if raw_value is None:
            where = "label_from_path did not match the path" if from_path is not None else "label column (and fallbacks) empty"
            raise fc.RowReject(fc.MISSING_LABEL, where)
        labels = fc.resolve_labels(fc.split_labels(raw_value, opts), opts, taxonomy)
        if labels.clean and bads:
            raise fc.RowReject(fc.MISSING_LABEL, "every label means 'no finding' but the file has bad functions")

        id_col = field_map.get("id_field") if isinstance(field_map.get("id_field"), str) else "id"
        if not fc.blank(row.get(id_col)) or not path:
            file_key = keys.base(ctx, row, idx, field_map)
        else:
            file_key = keys.unique(f"{ctx.source_id}-{path}")

        seen: dict[str, int] = {}
        results: list[TransformedRow | RejectedRow] = []
        for symbol, first, last in sorted(bads + goods, key=lambda t: t[1]):
            is_bad = (symbol, first, last) in bads
            code = "\n".join(lines[first - 1 : last])
            seen[symbol] = seen.get(symbol, 0) + 1
            key = f"{file_key}-{symbol}" + (f"#{seen[symbol]}" if seen[symbol] > 1 else "")
            try:
                fc.check_code(code, opts)
            except fc.RowReject as rej:
                results.append(RejectedRow(raw_row=row, reason=rej.reason, detail=f"{symbol}: {rej.detail}", row_index=idx))
                continue
            code = fc.scrub(code, language, opts)
            findings = [fc.make_finding(label, fc.whole_span(code, first), code, first) for label in labels.ids] if is_bad else []
            warnings = list(labels.warnings)
            if ignored:
                warnings.append(f"ignored {ignored} other function(s) in the file")
            results.append(TransformedRow(
                payload=fc.findings_payload(code=code, language=language, path=path, start_line=first, findings=findings,
                                            labels_raw=labels.raw if is_bad else [], taxonomy_version=version),
                row_key=key, warnings=warnings,
            ))
        yield from results


register_mapper("contrast_functions", ContrastFunctionsMapper)

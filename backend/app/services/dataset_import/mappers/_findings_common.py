"""Shared pieces of the findings-row mappers (``code_label_to_findings``,
``fix_pair``, ``contrast_functions``).

A *findings row* is one piece of code plus the labelled findings in it::

    {"row_kind": "findings", "code": str, "language": str | None, "path": str | None,
     "start_line": int,            # file line of the first line of ``code`` (>= 1)
     "end_line": int,              # file line of its last line
     "findings_json": {"findings": [{"label": <taxonomy id>, "start_line": int,
                                     "end_line": int, "evidence": str, "attrs": {}}]},
     "labels_raw": [str],          # the labels as the source spelled them
     "taxonomy_version": str | None}

Finding lines are absolute (numbered like ``start_line``). ``findings: []``
means the code is clean. Labels resolve through the project's label taxonomy
(``ctx.extra["taxonomy"]``, a ``TaxonomyIndex``); without one they stay as the
source wrote them and each accepted row carries the warning ``no_taxonomy``.

The payload never uses the keys ``id`` / ``source`` / ``status`` / ``row_key``
(``run_import`` writes those itself).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from app.services.code_chunker import detect_language, normalize_language
from app.services.dataset_import.protocols import ImportContext, RawRow
from app.services.label_taxonomy_service import TaxonomyIndex, norm

# Rejection reasons shared by the three mappers.
PARSE_ERROR = "parse_error"
EMPTY_CODE = "empty_code"
MISSING_LABEL = "missing_label"
UNKNOWN_LABEL = "unknown_label"
LINE_OUT_OF_RANGE = "line_out_of_range"
CODE_TOO_LONG = "code_too_long"
LANGUAGE_OUT_OF_SCOPE = "language_out_of_scope"
IDENTICAL_PAIR = "identical_pair"
NO_VARIANTS = "no_variants"
AMBIGUOUS_LABEL = "ambiguous_label"
LINE_COUNT_MISMATCH = "line_count_mismatch"

WARN_NO_TAXONOMY = "no_taxonomy"
WARN_LINES_UNKNOWN = "lines_unknown"

DEFAULT_CLEAN_LABELS = ["", "none", "clean", "safe", "benign", "0", "false", "no"]
DEFAULT_LABEL_SEPARATOR = r"[,;|]"
DEFAULT_BAD_PATTERN = r"(?i)^bad"
DEFAULT_GOOD_PATTERN = r"(?i)^good"
DEFAULT_COMMENT_PATTERN = r"(?i)\b(flaw|fix(es|ed)?|bad|good|vulnerab\w*|insecure|unsafe|incidental)\b"
DEFAULT_MAX_CODE_CHARS = 20000
EVIDENCE_CHARS = 400

LABEL_COLUMNS = ("label", "labels", "category", "class", "type", "tag")
LANGUAGE_COLUMNS = ("language", "lang")
PATH_COLUMNS = ("path", "file", "filename", "file_path")


class RowReject(Exception):
    """Raised inside a mapper to reject the current row with a stable reason."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(detail or reason)


# ── Column access ───────────────────────────────────────────────────────


def blank(value: Any) -> bool:
    """None, an empty / whitespace string, an empty list, or a float NaN (what a
    CSV / dataframe source yields for a missing cell)."""
    if isinstance(value, float) and math.isnan(value):
        return True
    return value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, (list, tuple)) and not value)


def pick(row: RawRow, field_map: dict[str, Any], key: str, default: tuple[str, ...]) -> Any:
    """The value of the first non-blank column among the one ``field_map[key]``
    names (when it is a string) and the defaults; None when all are blank."""
    explicit = field_map.get(key)
    candidates = ([explicit] if isinstance(explicit, str) and explicit else []) + [c for c in default if c != explicit]
    for column in candidates:
        value = row.get(column)
        if not blank(value):
            return value
    return None


def has_column(row: RawRow, field_map: dict[str, Any], key: str, default: tuple[str, ...]) -> bool:
    explicit = field_map.get(key)
    return any(c in row for c in ([explicit] if isinstance(explicit, str) else []) + list(default))


def text_of(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value if isinstance(value, str) else str(value)


def clean_code(value: Any) -> str:
    """The code text with Windows line endings normalised; a lone ``\\r`` is
    kept (it is not a line break for ``split("\\n")``, which is how finding
    lines are counted)."""
    return text_of(value).replace("\r\n", "\n").rstrip()


class KeyMaker:
    """Unique, never-empty ``row_key`` values for one import: ``<source>-<id>``
    (``id`` or the ``id_field`` column), else ``<source>-<row index>``; a key
    seen before gets ``#2``, ``#3`` …"""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def base(self, ctx: ImportContext, row: RawRow, idx: int, field_map: dict[str, Any]) -> str:
        column = field_map.get("id_field") if isinstance(field_map.get("id_field"), str) else "id"
        value = row.get(column)
        if blank(value):
            return self.unique(f"{ctx.source_id}-{idx}")
        return self.unique(f"{ctx.source_id}-{text_of(value).strip()}")

    def unique(self, key: str) -> str:
        count = self._seen.get(key, 0) + 1
        self._seen[key] = count
        return key if count == 1 else f"{key}#{count}"


def unexpected(exc: BaseException) -> tuple[str, str]:
    """``(reason, detail)`` for an unexpected per-row exception: the row is
    rejected as ``parse_error`` instead of killing the import."""
    return PARSE_ERROR, f"{type(exc).__name__}: {exc}"


# ── Options ─────────────────────────────────────────────────────────────


@dataclass
class Options:
    clean_labels: set[str] = field(default_factory=set)
    languages: set[str] | None = None
    max_code_chars: int = DEFAULT_MAX_CODE_CHARS
    label_prefix: str = ""
    allow_partial_labels: bool = False
    separator: re.Pattern[str] = field(default_factory=lambda: re.compile(DEFAULT_LABEL_SEPARATOR))
    anonymize_symbols: bool = False
    blank_comments: str = "none"  # none | matching | all
    comment_pattern: re.Pattern[str] = field(default_factory=lambda: re.compile(DEFAULT_COMMENT_PATTERN))
    bad_pattern: re.Pattern[str] = field(default_factory=lambda: re.compile(DEFAULT_BAD_PATTERN))
    good_pattern: re.Pattern[str] = field(default_factory=lambda: re.compile(DEFAULT_GOOD_PATTERN))


def _regex(field_map: dict[str, Any], key: str, default: str) -> re.Pattern[str]:
    value = field_map.get(key)
    try:
        return re.compile(value if isinstance(value, str) and value else default)
    except re.error as exc:
        raise ValueError(f"{key} is not a valid regex: {exc}") from exc


def read_options(field_map: dict[str, Any], *, neutralise_by_default: bool = False) -> Options:
    """Parse the options every findings mapper shares. Raises ``ValueError`` on
    a bad value (the mapper then rejects every row with ``parse_error``)."""
    clean = field_map.get("clean_labels")
    if clean is None:
        clean = DEFAULT_CLEAN_LABELS
    if not isinstance(clean, (list, tuple)):
        raise ValueError("clean_labels must be a list of labels")
    langs = field_map.get("languages")
    if langs is not None and not isinstance(langs, (list, tuple)):
        raise ValueError("languages must be a list of language names")
    max_chars = field_map.get("max_code_chars", DEFAULT_MAX_CODE_CHARS)
    try:
        max_chars = int(max_chars)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_code_chars must be an integer") from exc
    if max_chars <= 0:
        raise ValueError("max_code_chars must be > 0")
    separator = field_map.get("label_separator", DEFAULT_LABEL_SEPARATOR)
    try:
        pattern = re.compile(separator if isinstance(separator, str) and separator else DEFAULT_LABEL_SEPARATOR)
    except re.error as exc:
        raise ValueError(f"label_separator is not a valid regex: {exc}") from exc
    anonymize = field_map.get("anonymize_symbols", neutralise_by_default)
    blank_mode = field_map.get("blank_comments", "matching" if neutralise_by_default else "none")
    if blank_mode not in ("none", "matching", "all"):
        raise ValueError("blank_comments must be 'none', 'matching' or 'all'")
    return Options(
        anonymize_symbols=bool(anonymize),
        blank_comments=blank_mode,
        comment_pattern=_regex(field_map, "comment_pattern", DEFAULT_COMMENT_PATTERN),
        bad_pattern=_regex(field_map, "bad_pattern", DEFAULT_BAD_PATTERN),
        good_pattern=_regex(field_map, "good_pattern", DEFAULT_GOOD_PATTERN),
        clean_labels={norm(c) for c in clean},
        languages={normalize_language(str(lang)) or norm(lang) for lang in langs} if langs else None,
        max_code_chars=max_chars,
        label_prefix=str(field_map.get("label_prefix") or ""),
        allow_partial_labels=bool(field_map.get("allow_partial_labels")),
        separator=pattern,
    )


def taxonomy_of(ctx: ImportContext) -> TaxonomyIndex | None:
    tax = (ctx.extra or {}).get("taxonomy")
    return tax if isinstance(tax, TaxonomyIndex) else None


# ── Code, language ──────────────────────────────────────────────────────


def check_code(code: str, opts: Options, column: str = "code") -> None:
    if not code.strip():
        raise RowReject(EMPTY_CODE, f"column '{column}' (and fallbacks) empty after strip")
    if len(code) > opts.max_code_chars:
        raise RowReject(CODE_TOO_LONG, f"{len(code)} characters > max_code_chars {opts.max_code_chars}")


def resolve_language(value: Any, path: str | None, code: str) -> str | None:
    """A language column value (normalised), else detected from the path /
    shebang."""
    if not blank(value):
        return normalize_language(str(value)) or str(value).strip().lower()
    return detect_language(path, code)


def check_language(language: str | None, opts: Options) -> None:
    if opts.languages is not None and (language or "") not in opts.languages:
        raise RowReject(LANGUAGE_OUT_OF_SCOPE, f"language '{language or 'unknown'}' is not in {sorted(opts.languages)}")


# ── Label-leak scrubbing ────────────────────────────────────────────────
#
# Code suites often spell the answer in the code itself: the vulnerable
# function is called ``badSink``, its twin ``goodG2B``, and a comment says
# "FLAW". A model trained on that learns the names, not the defect. These two
# passes rewrite the emitted code without changing its line count.

_IDENT = re.compile(r"[^\W\d]\w*")
_BLOCK_LANGS = {"c", "cpp", "java", "javascript", "typescript", "csharp", "go", "rust", "kotlin", "swift", "php",
                "scala", "dart", "css", "sql", "objc", "objective-c", "groovy", "solidity"}
_SLASH_LANGS = _BLOCK_LANGS - {"css", "sql"}
_HASH_LANGS = {"python", "ruby", "shell", "perl", "yaml", "php", "r", "hcl", "dockerfile", "toml", "makefile", "elixir", "powershell"}
_DASH_LANGS = {"sql", "lua", "haskell", "ada"}


def anonymize_symbols(code: str, opts: Options, mapping: dict[str, str] | None = None) -> str:
    """Rewrite every identifier that matches ``bad_pattern`` / ``good_pattern``
    to ``fn_<n>`` (consistent through ``mapping``, which a caller shares across
    the code of one row)."""
    names = mapping if mapping is not None else {}

    def swap(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in names:
            return names[token]
        if opts.bad_pattern.search(token) or opts.good_pattern.search(token):
            names[token] = f"fn_{len(names) + 1}"
            return names[token]
        return token

    return _IDENT.sub(swap, code)


def _comment_spans(code: str, language: str | None) -> list[tuple[int, int]]:
    lang = (language or "").lower()
    known = bool(lang)
    block = not known or lang in _BLOCK_LANGS
    slash = not known or lang in _SLASH_LANGS
    hash_ = not known or lang in _HASH_LANGS
    dash = lang in _DASH_LANGS
    spans: list[tuple[int, int]] = []
    i, n = 0, len(code)
    while i < n:
        ch = code[i]
        two = code[i:i + 2]
        if block and two == "/*":
            end = code.find("*/", i + 2)
            end = n if end < 0 else end + 2
            spans.append((i, end))
            i = end
        elif (slash and two == "//") or (dash and two == "--") or (hash_ and ch == "#"):
            end = code.find("\n", i)
            end = n if end < 0 else end
            spans.append((i, end))
            i = end
        elif ch in "\"'":
            j = i + 1
            while j < n and code[j] not in ("\n", ch):
                j += 2 if code[j] == "\\" else 1
            i = j + 1 if j < n and code[j] == ch else i + 1
        else:
            i += 1
    return spans


def blank_comments(code: str, language: str | None, opts: Options) -> str:
    """Replace comments (all, or those matching ``comment_pattern``) with
    spaces, keeping every line break."""
    if opts.blank_comments == "none":
        return code
    out, last = [], 0
    for a, b in _comment_spans(code, language):
        text = code[a:b]
        if opts.blank_comments == "all" or opts.comment_pattern.search(text):
            out.append(code[last:a])
            out.append("".join(c if c == "\n" else " " for c in text))
            last = b
    out.append(code[last:])
    result = "".join(out)
    return "\n".join(ln.rstrip() if ln != orig else ln for ln, orig in zip(result.split("\n"), code.split("\n")))


def scrub(code: str, language: str | None, opts: Options, mapping: dict[str, str] | None = None) -> str:
    """Apply the comment blanking and symbol anonymising the options ask for."""
    if opts.blank_comments != "none":
        code = blank_comments(code, language, opts)
    if opts.anonymize_symbols:
        code = anonymize_symbols(code, opts, mapping)
    return code


# ── Labels ──────────────────────────────────────────────────────────────

_FALSY = {"", "0", "false", "no", "n", "f", "none", "safe", "clean", "benign", "notvulnerable", "nonvulnerable"}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and math.isnan(value)) and value != 0
    return norm(value) not in _FALSY


def split_labels(value: Any, opts: Options) -> list[str]:
    """Raw label strings from a list, a JSON-list string or a separated string."""
    if value is None:
        return []
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, (list, tuple)):
        return [t for t in (text_of(v).strip() for v in value if v is not None) if t]
    text = text_of(value).strip()
    if text.startswith("["):
        try:
            loaded = json.loads(text)
        except ValueError:
            loaded = None
        if isinstance(loaded, list):
            return split_labels(loaded, opts)
    return [p.strip() for p in opts.separator.split(text) if p.strip()]


@dataclass
class ResolvedLabels:
    ids: list[str]  # taxonomy ids (or raw labels without a taxonomy), deduped, order kept
    raw: list[str]  # the labels as spelled, clean ones removed
    warnings: list[str] = field(default_factory=list)
    clean: bool = False  # every label was a "no finding" marker (or none given)


def resolve_labels(raw_labels: list[str], opts: Options, taxonomy: TaxonomyIndex | None) -> ResolvedLabels:
    """Resolve raw labels. ``clean`` is set when they all mean "no finding".
    Raises ``RowReject(unknown_label)`` for an unresolvable label (unless
    ``allow_partial_labels`` keeps the resolved ones)."""
    candidates = [lab for lab in raw_labels if norm(lab) not in opts.clean_labels]
    if not candidates:
        return ResolvedLabels([], [], [WARN_NO_TAXONOMY] if taxonomy is None else [], clean=True)
    if taxonomy is None:
        ids: list[str] = []
        for lab in candidates:
            if lab not in ids:
                ids.append(lab)
        return ResolvedLabels(ids, list(candidates), [WARN_NO_TAXONOMY])
    ids, unknown = [], []
    for lab in candidates:
        found, by = taxonomy.resolve(lab)
        if by == "ambiguous":
            raise RowReject(AMBIGUOUS_LABEL, f"label {lab!r} matches several taxonomy entries: {', '.join(taxonomy.candidates(lab))}")
        if found is None and opts.label_prefix:
            found, by = taxonomy.resolve(opts.label_prefix + lab)
            if by == "ambiguous":
                raise RowReject(AMBIGUOUS_LABEL, f"label {opts.label_prefix + lab!r} matches several taxonomy entries: {', '.join(taxonomy.candidates(opts.label_prefix + lab))}")
        if found is None:
            unknown.append(lab)
        elif found not in ids:
            ids.append(found)
    warnings: list[str] = []
    if unknown:
        if not opts.allow_partial_labels or not ids:
            raise RowReject(UNKNOWN_LABEL, f"label(s) not in the taxonomy: {', '.join(repr(u) for u in unknown)}")
        warnings.append(f"unknown_labels_dropped: {', '.join(unknown)}")
    return ResolvedLabels(ids, list(candidates), warnings)


def label_title(taxonomy: TaxonomyIndex | None, label: str) -> str:
    return (taxonomy.title(label) if taxonomy else None) or label


# ── Lines ───────────────────────────────────────────────────────────────

_RANGE = re.compile(r"^\s*(\d+)\s*(?:[-:]\s*(\d+))?\s*$")


def _spans_from(value: Any) -> list[tuple[int, int | None]]:
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[tuple[int, int | None]] = []
    for item in items:
        if isinstance(item, bool):
            raise RowReject(LINE_OUT_OF_RANGE, f"'{item}' is not a line number")
        if isinstance(item, float) and not math.isfinite(item):
            raise RowReject(LINE_OUT_OF_RANGE, f"'{item}' is not a line number")
        if isinstance(item, (int, float)) and float(item) == int(item):
            out.append((int(item), None))
            continue
        match = _RANGE.match(text_of(item))
        if not match:
            raise RowReject(LINE_OUT_OF_RANGE, f"'{item}' is not a line number or range")
        out.append((int(match.group(1)), int(match.group(2)) if match.group(2) else None))
    return out


def _parse_json_list(value: Any) -> Any:
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            loaded = json.loads(value)
        except ValueError:
            return value
        return loaded if isinstance(loaded, list) else value
    return value


def parse_spans(line_value: Any, end_value: Any) -> list[tuple[int, int]] | None:
    """Finding spans from a ``line`` value (int, ``"12-15"``, or a list of those)
    and an optional ``end_line`` value (int or list). None when there is no
    line information."""
    if blank(line_value):
        return None
    starts = _spans_from(_parse_json_list(line_value))
    ends = [] if blank(end_value) else _spans_from(_parse_json_list(end_value))
    if ends and len(ends) not in (1, len(starts)):
        raise RowReject(LINE_COUNT_MISMATCH, f"{len(starts)} line value(s) but {len(ends)} end_line value(s)")
    spans = []
    for i, (a, b) in enumerate(starts):
        explicit_end = ends[i if len(ends) == len(starts) else 0][0] if ends else None
        spans.append((a, b if b is not None else (explicit_end if explicit_end is not None else a)))
    return spans


def absolute_spans(
    spans: list[tuple[int, int]], *, start_line: int, line_count: int, relative: bool, labels: int,
) -> list[tuple[int, tuple[int, int]]]:
    """Check the spans against the code's line range and pair each with the
    index of its label: one span applies to every label; one label takes every
    span (one finding per span); equal counts pair up in order. Anything else
    is a ``line_count_mismatch``."""
    if len(spans) == 1:
        pairing = [(i, 0) for i in range(labels)]
    elif labels == 1:
        pairing = [(0, j) for j in range(len(spans))]
    elif len(spans) == labels:
        pairing = [(i, i) for i in range(labels)]
    else:
        raise RowReject(LINE_COUNT_MISMATCH, f"{len(spans)} line spec(s) for {labels} label(s)")
    first, last = start_line, start_line + line_count - 1
    absolute: list[tuple[int, int]] = []
    for a, b in spans:
        if relative:
            a, b = a + start_line - 1, b + start_line - 1
        if a < first or b > last or b < a:
            raise RowReject(LINE_OUT_OF_RANGE, f"lines {a}-{b} are outside the code's lines {first}-{last}")
        absolute.append((a, b))
    return [(i, absolute[j]) for i, j in pairing]


def line_count(code: str) -> int:
    return code.count("\n") + 1


def int_column(row: RawRow, field_map: dict[str, Any], key: str, default: int) -> int:
    """``field_map[key]`` as a constant (int) or a column name (else a column called ``key``); ``default`` when unset."""
    spec = field_map.get(key)
    if isinstance(spec, bool):
        return default
    if isinstance(spec, int):
        value: Any = spec
    elif isinstance(spec, str) and spec:
        value = row.get(spec)
    else:
        value = row.get(key, default)  # a column named like the key is used when nothing is mapped
    if blank(value):
        return default
    try:
        number = int(float(str(value).strip()))
    except (ValueError, OverflowError) as exc:
        raise RowReject(LINE_OUT_OF_RANGE, f"{key} '{value}' is not an integer") from exc
    if number < 1:
        raise RowReject(LINE_OUT_OF_RANGE, f"{key} must be >= 1, got {number}")
    return number


# ── Row building ────────────────────────────────────────────────────────


def evidence_for(code: str, start_line: int, span: tuple[int, int]) -> str:
    lines = code.split("\n")
    chunk = lines[span[0] - start_line : span[1] - start_line + 1]
    return "\n".join(s for s in (ln.strip() for ln in chunk) if s)[:EVIDENCE_CHARS]


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def make_finding(label: str, span: tuple[int, int], code: str, start_line: int, evidence: Any = None, attrs: dict[str, Any] | None = None) -> dict[str, Any]:
    text = text_of(evidence).strip()[:EVIDENCE_CHARS] if not blank(evidence) else evidence_for(code, start_line, span)
    return {"label": label, "start_line": span[0], "end_line": span[1], "evidence": text, "attrs": dict(attrs or {})}


def findings_payload(
    *, code: str, language: str | None, path: str | None, start_line: int, findings: list[dict[str, Any]],
    labels_raw: list[str], taxonomy_version: str | None,
) -> dict[str, Any]:
    return {
        "row_kind": "findings",
        "code": code,
        "language": language,
        "path": path,
        "start_line": start_line,
        "end_line": start_line + line_count(code) - 1,
        "findings_json": {"findings": findings},
        "labels_raw": list(labels_raw),
        "taxonomy_version": taxonomy_version,
    }


def whole_span(code: str, start_line: int) -> tuple[int, int]:
    return (start_line, start_line + line_count(code) - 1)


def path_of(row: RawRow, field_map: dict[str, Any]) -> str | None:
    value = pick(row, field_map, "path", PATH_COLUMNS)
    return text_of(value).strip() or None if value is not None else None

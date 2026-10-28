"""Code-aware chunking — one retrievable unit per function / method / block.

Sentence chunking (``cleaning_service.chunk_text``) cuts source code in the
middle of statements and throws away the line numbers a reader needs. This
module splits source files at declaration boundaries instead, and reports the
exact 1-based line range of every chunk so a retrieved passage can be cited as
``path:start-end (symbol)``.

No parser dependency: each language gets a small heuristic scanner. Strings
and comments are blanked out first (same length, newlines kept) so brace and
indentation scans are not fooled by ``"{"`` or ``// }``. The scanners are
best-effort — a file they cannot make sense of still comes back as
blank-line blocks packed to ``max_chars``, never as an error and never with a
line missing.

Invariants (asserted in ``tests/test_code_chunker.py``):

* With ``context_lines=0`` every source line lands in exactly one chunk and
  the chunks are in file order.
* ``start_line`` / ``end_line`` describe the exact text of the chunk (one line
  longer than ``max_chars`` is cut into ``max_chars`` slices that all cite that
  line);
  ``unit_start_line`` / ``unit_end_line`` describe the declaration it was cut
  from (the same for every piece of an oversize unit).
* Blank lines after a unit belong to that unit's chunk; a lone closing brace
  after the last member of a class is carried by the previous chunk instead of
  becoming a chunk of its own.

Pure functions, no I/O. Sizes are characters, not tokens — consistent with the
cleaning ``chunk_size``.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable

LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".tf": "hcl",
    ".tfvars": "hcl",
    ".hcl": "hcl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".xml": "xml",
    ".csproj": "xml",
    ".xsd": "xml",
    ".properties": "properties",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".dockerfile": "dockerfile",
}

_SHEBANG_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("python", "python"),
    ("node", "javascript"),
    ("deno", "typescript"),
    ("bash", "shell"),
    ("zsh", "shell"),
    ("sh", "shell"),
    ("dash", "shell"),
    ("ksh", "shell"),
)

CODE_LANGUAGES = frozenset(LANGUAGE_BY_EXTENSION.values()) | {"dockerfile"}

_ALIASES = {
    "py": "python", "cs": "csharp", "c#": "csharp", "js": "javascript", "ts": "typescript",
    "sh": "shell", "bash": "shell", "yml": "yaml", "tf": "hcl", "c++": "cpp", "golang": "go", "rs": "rust",
    "docker": "dockerfile",
}


def detect_language(path: str | None, text: str | None = None) -> str | None:
    """Language of a source file: its extension first (``Dockerfile`` by name),
    then a ``#!`` line (``#!/usr/bin/env python3``, ``#!/bin/bash``, node, …).
    None when neither says."""
    if path:
        name = PurePosixPath(str(path).replace("\\", "/")).name
        lowered = name.lower()
        if lowered == "dockerfile" or lowered.startswith("dockerfile.") or lowered.endswith(".dockerfile"):
            return "dockerfile"
        suffix = PurePosixPath(lowered).suffix
        if suffix in LANGUAGE_BY_EXTENSION:
            return LANGUAGE_BY_EXTENSION[suffix]
    if text:
        first = text.lstrip("﻿").split("\n", 1)[0].strip()
        if first.startswith("#!"):
            parts = first[2:].split()
            if parts:
                program = PurePosixPath(parts[0]).name
                if program == "env" and len(parts) > 1:
                    program = PurePosixPath(next((p for p in parts[1:] if not p.startswith("-")), "")).name
                for prefix, language in _SHEBANG_LANGUAGES:
                    if program == prefix or (prefix != "sh" and re.fullmatch(rf"{prefix}[\d.]*", program)):
                        return language
    return None


def is_code_document(doc: Any) -> bool:
    """A source / config ``RawDocument``: a repo import marks it
    ``metadata_["content_kind"] == "code"``; an upload counts when its
    extension is a code-only one (``.json`` stays a row file for uploads).
    Code documents are text, never rows — row loaders and task-shape detection
    skip them."""
    if (getattr(doc, "metadata_", None) or {}).get("content_kind") == "code":
        return True
    ext = PurePosixPath(str(getattr(doc, "file_path", None) or getattr(doc, "filename", None) or "")).suffix.lower()
    return ext in LANGUAGE_BY_EXTENSION and ext != ".json"


def normalize_language(language: str | None) -> str | None:
    token = (language or "").strip().lower()
    token = _ALIASES.get(token, token)
    return token or None


@dataclass
class CodeChunk:
    """One chunk of a source file. Lines are 1-based and inclusive."""

    text: str
    start_line: int
    end_line: int
    symbol: str | None
    kind: str  # "function" | "method" | "class" | "block" | "module"
    unit_start_line: int
    unit_end_line: int


@dataclass
class _Unit:
    start: int  # 0-based inclusive line indexes
    end: int
    symbol: str | None
    kind: str


# ── Masking (strings + comments → spaces) ────────────────────────────

_C_TOKENS = re.compile(
    r"(?P<c>//[^\n]*|/\*.*?(?:\*/|\Z))"
    r'|(?P<s>"""(?:\\.|.)*?(?:"""|\Z)'
    r'|\$?@"(?:""|[^"])*"?|@\$"(?:""|[^"])*"?'
    r"|`[^`]*`?"
    r'|"(?:\\.|[^"\\\n])*"?'
    r"|'(?:\\.|[^'\\\n])*'?)",
    re.S,
)
_PY_TOKENS = re.compile(
    r"(?P<c>#[^\n]*)"
    r"|(?P<s>'''(?:\\.|.)*?(?:'''|\Z)"
    r'|"""(?:\\.|.)*?(?:"""|\Z)'
    r"|'(?:\\.|[^'\\\n])*'?"
    r'|"(?:\\.|[^"\\\n])*"?)',
    re.S,
)
_SH_TOKENS = re.compile(
    r"(?P<c>(?:^|(?<=[ \t;&|(]))#[^\n]*)"
    r"|(?P<s>'[^']*'?"
    r'|"(?:\\.|[^"\\])*"?)',
    re.S | re.M,
)
_HCL_TOKENS = re.compile(
    r"(?P<c>#[^\n]*|//[^\n]*|/\*.*?(?:\*/|\Z))"
    r'|(?P<s>"(?:\\.|[^"\\\n])*"?)',
    re.S,
)
_YAML_TOKENS = re.compile(r"(?P<c>(?:^|(?<=[ \t]))#[^\n]*)", re.M)
_JSON_TOKENS = re.compile(r'(?P<s>"(?:\\.|[^"\\\n])*"?)')
_SQL_TOKENS = re.compile(
    r"(?P<c>--[^\n]*|/\*.*?(?:\*/|\Z))"
    r"|(?P<s>'(?:''|[^'])*'?"
    r'|"(?:""|[^"])*"?'
    r"|\$(\w*)\$.*?\$\3\$"
    r"|`[^`\n]*`?)",
    re.S,
)

_MASK_PATTERNS: dict[str, re.Pattern[str]] = {
    "java": _C_TOKENS, "csharp": _C_TOKENS, "go": _C_TOKENS, "javascript": _C_TOKENS,
    "typescript": _C_TOKENS, "c": _C_TOKENS, "cpp": _C_TOKENS,
    "python": _PY_TOKENS, "shell": _SH_TOKENS, "hcl": _HCL_TOKENS, "yaml": _YAML_TOKENS,
    "json": _JSON_TOKENS, "sql": _SQL_TOKENS,
}


_REGEX_PRECEDERS = frozenset("(,=:[!&|?{};")
_REGEX_KEYWORD_TAIL = re.compile(r"(?<![\w$.])(?:return|typeof)$")


def _blank_span(chars: list[str], a: int, b: int) -> None:
    for k in range(a, min(b, len(chars))):
        if chars[k] != "\n":
            chars[k] = " "


def _regex_literal_end(text: str, i: int) -> int:
    """Index just past the regex literal (with flags) starting at the ``/`` at
    ``i``, or -1 when it isn't one: handles escapes and ``[...]`` classes, and
    never crosses a newline."""
    n = len(text)
    j = i + 1
    in_class = False
    while j < n:
        c = text[j]
        if c == "\n":
            return -1
        if c == "\\":
            j += 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            j += 1
            while j < n and (text[j].isalnum() or text[j] in "_$"):
                j += 1
            return j
        j += 1
    return -1


def _mask_js(text: str) -> str:
    """JS / TS: comments, strings, template literals and regex literals blanked
    (quotes / slashes kept). A ``/`` starts a regex literal when what precedes it
    is one of ``( , = : [ ! & | ? { } ;``, ``return`` / ``typeof``, or nothing on
    its line — a heuristic, but ``/[{]/g`` no longer unbalances braces."""
    chars = list(text)
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            _blank_span(chars, i, j)
            i = j
        elif ch == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            _blank_span(chars, i, j)
            i = j
        elif ch in "\"'":
            j = i + 1
            while j < n and text[j] != "\n" and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            _blank_span(chars, i + 1, j)
            i = j + 1 if j < n and text[j] == ch else j
        elif ch == "`":
            j = i + 1
            while j < n and text[j] != "`":
                j += 2 if text[j] == "\\" else 1
            _blank_span(chars, i + 1, j)
            i = j + 1
        elif ch == "/":
            k = i - 1
            while k >= 0 and text[k] in " \t\r":
                k -= 1
            at_line_start = k < 0 or text[k] == "\n"
            while k >= 0 and text[k] in " \t\r\n":
                k -= 1
            prev = text[k] if k >= 0 else ""
            allowed = (
                at_line_start
                or prev == ""
                or prev in _REGEX_PRECEDERS
                or bool(_REGEX_KEYWORD_TAIL.search(text[max(0, k - 8) : k + 1]))
            )
            end = _regex_literal_end(text, i) if allowed else -1
            if end > 0:
                _blank_span(chars, i + 1, text.rfind("/", i + 1, end))
                i = end
            else:
                i += 1
        else:
            i += 1
    return "".join(chars)


_HEREDOC_OPEN = re.compile(r"<<-?[ \t]*[\"']?([A-Za-z_]\w*)[\"']?[ \t]*$")


def _mask_hcl_heredocs(text: str) -> str:
    """Blank HCL heredoc bodies (``<<EOF`` / ``<<-EOF`` … terminator line) so
    braces and quotes inside them don't count."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        found = _HEREDOC_OPEN.search(lines[i].rstrip("\r"))
        if found:
            tag = found.group(1)
            j = i + 1
            while j < len(lines) and lines[j].strip() != tag:
                j += 1
            for k in range(i + 1, min(j + 1, len(lines))):
                lines[k] = " " * len(lines[k])
            i = j + 1
        else:
            i += 1
    return "\n".join(lines)

_RUST_RAW_OPEN = re.compile(r'(#*)"')


def _mask_rust(text: str) -> str:
    """Rust: comments (``//``, nested ``/* /* */ */``), strings, raw strings
    (``r"…"``, ``r#"…"#``, ``br#"…"#``, ``cr"…"``) and char literals blanked.
    A ``'`` starts a char literal only as ``'x'`` or ``'\\…'``; otherwise it is a
    lifetime (``'a``, ``'static``) and is left alone."""
    chars = list(text)
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            _blank_span(chars, i, j)
            i = j
        elif ch == "/" and text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth += 1
                    j += 2
                elif text.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            _blank_span(chars, i, j)
            i = j
        elif ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            nxt = text[j] if j < n else ""
            if word in ("r", "br", "cr") and nxt in ('"', "#"):
                opened = _RUST_RAW_OPEN.match(text, j)
                if opened:
                    hashes = opened.group(1)
                    body = opened.end()
                    close = text.find('"' + hashes, body)
                    end = n if close < 0 else close + 1 + len(hashes)
                    _blank_span(chars, i, end)
                    chars[i] = " "
                    i = end
                    continue
            if word in ("b", "c") and nxt == '"':
                i = j  # the quote is handled next round
            elif word == "b" and nxt == "'":
                i = j
            else:
                i = j
        elif ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            _blank_span(chars, i + 1, j)
            i = j + 1
        elif ch == "'":
            if text.startswith("\\", i + 1):
                j = i + 3
                while j < n and text[j] != "'" and text[j] != "\n":
                    j += 1
                end = j + 1 if j < n and text[j] == "'" else j
                _blank_span(chars, i + 1, end - 1 if end > j else end)
                i = end
            elif i + 2 < n and text[i + 2] == "'" and text[i + 1] != "\n":
                chars[i + 1] = " "
                i += 3
            else:
                i += 1  # lifetime / label
        else:
            i += 1
    return "".join(chars)


def _mask(text: str, language: str | None) -> str:
    """``text`` with comments blanked and string contents blanked (quote
    characters kept) — same length, same newlines, so line numbers and
    columns still match the original."""
    if language in ("javascript", "typescript"):
        return _mask_js(text)
    if language == "rust":
        return _mask_rust(text)
    if language == "hcl":
        text = _mask_hcl_heredocs(text)
    pattern = _MASK_PATTERNS.get(language or "")
    if pattern is None:
        return text

    def blank(match: re.Match[str]) -> str:
        token = match.group(0)
        out = "".join(ch if ch == "\n" else " " for ch in token)
        if "s" in pattern.groupindex and match.group("s") is not None and token:
            out = token[0] + out[1:]
            if len(token) > 1 and token[-1] == token[0]:
                out = out[:-1] + token[-1]
        return out

    return pattern.sub(blank, text)


# ── Shared helpers ───────────────────────────────────────────────────

_CLOSING_ONLY = re.compile(r"^[\s{}\[\]();,]*$")
_XML_CLOSING_ONLY = re.compile(r"^\s*(?:</[^>]+>\s*)+$")
_CONTROL_WORDS = frozenset({
    "if", "for", "foreach", "while", "switch", "catch", "return", "new", "throw", "else", "do", "try",
    "synchronized", "using", "lock", "fixed", "sizeof", "typeof", "await", "yield", "when", "elif",
    "function", "with", "assert", "defer", "go", "select", "case", "delete", "unless", "until",
})


def _split_lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    return lines


def _is_blank(line: str) -> bool:
    return not line.strip()


def _closing_only(original: str, masked_line: str, language: str | None) -> bool:
    """A blank line, or one holding only closing punctuation (``}`` ``);``
    ``</tag>``). A comment-only line is neither — comments are content."""
    if not original.strip():
        return True
    if not masked_line.strip():
        return False
    if language == "xml":
        return bool(_XML_CLOSING_ONLY.match(masked_line))
    return bool(_CLOSING_ONLY.match(masked_line))


def _block_end(m: list[str], line: int, col: int, hi: int) -> int:
    """Index of the line where the block opened by the ``{`` at (line, col)
    closes: the first line that ends at brace depth 0 (so ``} else {`` keeps
    going). Unbalanced → the last line of the region."""
    depth = 0
    for j in range(line, hi):
        segment = m[j][col:] if j == line else m[j]
        for ch in segment:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        if depth <= 0:
            return j
    return hi - 1


def _container_units(
    o: list[str], m: list[str], start: int, end: int, claimed: list[_Unit], symbol: str | None, language: str | None
) -> list[_Unit]:
    """A container's own lines (header, fields, docstring, class attributes)
    that no member claimed, as ``class`` units — one per contiguous run, so a
    class with all its members claimed leaves just its header run. Runs that
    are only closing punctuation are dropped (the previous chunk carries
    them)."""
    out = list(claimed)
    cursor = start
    for unit in sorted(claimed, key=lambda u: u.start):
        out.extend(_runs(o, m, cursor, unit.start - 1, symbol, "class", language))
        cursor = unit.end + 1
    out.extend(_runs(o, m, cursor, end, symbol, "class", language))
    return sorted(out, key=lambda u: u.start)


def _runs(
    o: list[str], m: list[str], a: int, b: int, symbol: str | None, kind: str, language: str | None
) -> list[_Unit]:
    while a <= b and _is_blank(o[a]):
        a += 1
    while b >= a and _is_blank(o[b]):
        b -= 1
    if a > b:
        return []
    if all(_closing_only(o[i], m[i], language) for i in range(a, b + 1)):
        return []
    return [_Unit(a, b, symbol, kind)]


# ── Brace-language scanner ───────────────────────────────────────────

_ANNOTATION_ONLY = re.compile(r"^\s*(?:@[\w.]+(?:\([^)]*\))?\s*|\[[\w.]+(?:\([^)]*\))?\]\s*)+$", re.S)
_LEADING_ANNOTATIONS = re.compile(r"^\s*(?:@[\w.]+(?:\([^)]*\))?\s*)+", re.S)
_CONT_ENDS = (",", "=", "(", "+", "-", "*", "/", "&&", "||", ".", "?", "->", "=>", "&", "|")
_CONT_STARTS = ("{", ".", "->", "=>", ")", "&&", "||", "?", "extends ", "extends\t", "implements ", "throws ", "permits ", "where ")
_JVM_MODS = (
    "public|protected|private|internal|abstract|static|final|sealed|non-sealed|strictfp|partial|unsafe|new|"
    "readonly|ref|file|data|open|synchronized|default|native|virtual|override|async|extern|volatile|transient"
)
_JVM_TYPE = re.compile(rf"^(?:(?:{_JVM_MODS})\s+)*(class|interface|enum|record|struct|@interface)\s+(\w+)")
_JVM_NAMESPACE = re.compile(r"^namespace\s+([\w.]+)")
_GO_FUNC = re.compile(r"^func\s*(?:\(\s*(?:\w+\s+)?\*?\s*(\w+)(?:\[[^\]]*\])?\s*\)\s*)?(\w+)\s*(?:\[[^\]]*\])?\s*\(")
_GO_TYPE = re.compile(r"^type\s+(\w+)")
_JS_MODS = r"(?:export\s+)?(?:default\s+)?(?:declare\s+)?"
_JS_FUNC = re.compile(rf"^{_JS_MODS}(?:async\s+)?function\b\s*\*?\s*([\w$]+)?")
_JS_TYPE = re.compile(rf"^{_JS_MODS}(?:abstract\s+)?(?:const\s+)?(class|interface|enum|namespace|module)\s+([\w$]+)")
_JS_ARROW = re.compile(
    r"^(?:export\s+)?(?:(?:public|private|protected|static|readonly|override)\s+)*(?:(?:const|let|var)\s+)?"
    r"([\w$#]+)\s*(?::[^=]+?)?=\s*(?:async\s+)?(?:\((?:[^()]|\([^()]*\))*\)|[\w$]+)\s*(?::[^=>]+?)?\s*=>",
    re.S,
)
_JS_FUNC_EXPR = re.compile(r"^(?:export\s+)?(?:(?:const|let|var)\s+)?([\w$]+)\s*=\s*(?:async\s+)?function\b")
_C_TYPE = re.compile(r"^(?:typedef\s+)?(?:template\s*<.*?>\s*)?(struct|union|enum|class|namespace)\b(.*)$", re.S)

_RUST_VIS = r"(?:pub(?:\s*\([^)]*\))?\s+)?"
_RUST_FN = re.compile(
    rf'^{_RUST_VIS}(?:(?:default|const|async|unsafe|safe)\s+)*(?:extern(?:\s*"[^"]*")?\s+)?fn\s+(?:r#)?(\w+)'
)
_RUST_IMPL = re.compile(r"^(?:unsafe\s+)?impl\b\s*(.*)$", re.S)
_RUST_ITEM = re.compile(rf"^{_RUST_VIS}(?:unsafe\s+)?(?:auto\s+)?(struct|enum|union|trait|mod)\s+(?:r#)?(\w+)")
_RUST_MACRO = re.compile(r"^macro_rules\s*!\s*(\w+)")
_RUST_ATTR_ONLY = re.compile(r"^\s*(?:#!?\[.*\]\s*)+$", re.S)
_RUST_LEADING_ATTRS = re.compile(r"^\s*(?:#!?\[(?:[^\[\]]|\[[^\]]*\])*\]\s*)+", re.S)


def _rust_impl_name(rest: str) -> str | None:
    """Type an ``impl`` block is for: ``impl<T> Tr<T> for Foo<T> where …`` → ``Foo``."""
    rest = rest.strip()
    if rest.startswith("<"):
        depth = 0
        for k, ch in enumerate(rest):
            depth += (ch == "<") - (ch == ">" and rest[k - 1] != "-")
            if depth == 0:
                rest = rest[k + 1 :].strip()
                break
    rest = re.split(r"\bwhere\b", rest, maxsplit=1)[0]
    found = re.search(r"\bfor\b\s+(.*)$", rest, re.S)
    if found:
        rest = found.group(1)
    rest = re.sub(r"^[\s!&*]*(?:(?:mut|const|dyn)\s+)*", "", rest)
    path = re.match(r"[\w:]+", rest)
    return path.group(0).split("::")[-1] if path else None


_Decl = tuple  # (kind, symbol, mode)  mode: "leaf" | "recurse" | "transparent"


class _BraceScanner:
    def __init__(self, lines: list[str], masked: list[str], language: str) -> None:
        self.o = lines
        self.m = masked
        self.lang = language
        self.n = len(lines)

    # -- statement reading -------------------------------------------------

    def _next_nonblank(self, j: int, hi: int) -> str:
        j += 1
        while j < hi:
            if self.m[j].strip():
                return self.m[j].strip()
            j += 1
        return ""

    def _continues(self, line: str, j: int, hi: int) -> bool:
        stripped = line.strip()
        nxt = self._next_nonblank(j, hi)
        if stripped.endswith(_CONT_ENDS) and not stripped.endswith(("++", "--", "*/")):
            return True
        if nxt.startswith(_CONT_STARTS):
            return True
        if self.lang == "rust" and (nxt == "where" or stripped == "where"):
            return True
        if nxt.startswith(":") and not nxt.startswith("::"):
            return True
        return self._gnu_return_type_line(line, j, hi)

    _GNU_TYPE_LINE = re.compile(r"^[A-Za-z_][\w\s*]*$")
    _GNU_NAME_LINE = re.compile(r"^\**\s*([A-Za-z_][\w:~]*)\s*\(")

    def _gnu_return_type_line(self, line: str, j: int, hi: int) -> bool:
        """GNU-style C: ``int`` alone at column 0, the name and parameters on
        the next column-0 line (``long_style(void)``), then ``{``."""
        if self.lang not in ("c", "cpp") or not line or line[0] in " \t" or j + 1 >= hi:
            return False
        if not self._GNU_TYPE_LINE.match(line.strip()) or line.strip().split()[-1] in _CONTROL_WORDS:
            return False
        nxt = self.m[j + 1]
        if not nxt or nxt[0] in " \t":
            return False
        named = self._GNU_NAME_LINE.match(nxt)
        return bool(named and named.group(1) not in _CONTROL_WORDS)

    def _statement(self, i: int, hi: int) -> dict:
        """Read from line ``i`` to its terminator: ``{`` or ``;`` outside
        parens, or the end of the logical line. Returns
        ``{last, term, line, col, header}``."""
        depth = 0
        parts: list[str] = []
        j = i
        while j < hi:
            s = self.m[j]
            for col, ch in enumerate(s):
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth = max(0, depth - 1)
                elif depth == 0 and ch in "{;":
                    parts.append(s[:col])
                    return {"last": j, "term": ch, "line": j, "col": col, "header": "\n".join(parts)}
                elif depth == 0 and ch == "}":
                    parts.append(s[:col])
                    return {"last": j, "term": None, "line": j, "col": col, "header": "\n".join(parts)}
            parts.append(s)
            if depth == 0 and not self._continues(s, j, hi):
                return {"last": j, "term": None, "line": j, "col": len(s), "header": "\n".join(parts)}
            j += 1
        return {"last": hi - 1, "term": None, "line": hi - 1, "col": 0, "header": "\n".join(parts)}

    # -- classification ----------------------------------------------------

    @staticmethod
    def _name_before_paren(pre: str) -> str | None:
        found = re.search(r"([\w$#~:]+)\s*(?:<[^<>]*(?:<[^<>]*>[^<>]*)*>)?\s*$", pre)
        return found.group(1) if found else None

    def _classify(self, stmt: dict, scope: str, in_cls: str | None) -> tuple | None:
        """``(kind, symbol, mode)`` for a declaration, None for anything else."""
        header = (_RUST_LEADING_ATTRS if self.lang == "rust" else _LEADING_ANNOTATIONS).sub("", stmt["header"])
        header = " ".join(header.split())
        term = stmt["term"]
        lang = self.lang
        if not header:
            return None

        if lang in ("java", "csharp"):
            ns = _JVM_NAMESPACE.match(header)
            if ns and term == "{":
                return ("module", None, "transparent")
            found = _JVM_TYPE.match(header)
            if found and "=" not in header[: found.start(1)] and "(" not in header[: found.start(1)]:
                return ("class", found.group(2), "recurse")
            if scope == "class" and "(" in header:
                pre = header[: header.index("(")]
                if "=" in pre or re.search(r"\bnew\b", pre):
                    return None
                name = self._name_before_paren(pre)
                has_body = term == "{" or (term == ";" and "=>" in header)
                if name and name not in _CONTROL_WORDS and has_body:
                    return ("method", name, "leaf")
            return None

        if lang == "rust":
            func = _RUST_FN.match(header)
            if func and term in ("{", ";"):
                return ("method" if scope == "class" else "function", func.group(1), "leaf")
            impl = _RUST_IMPL.match(header)
            if impl and term == "{":
                return ("class", _rust_impl_name(impl.group(1)), "recurse")
            item = _RUST_ITEM.match(header)
            if item:
                keyword, name = item.group(1), item.group(2)
                if keyword == "mod":
                    return ("module", name, "recurse_mod" if term == "{" else "leaf")
                return ("class", name, "recurse" if keyword == "trait" and term == "{" else "leaf")
            macro = _RUST_MACRO.match(header)
            if macro:
                return ("function", macro.group(1), "leaf")
            return None

        if lang == "go":
            func = _GO_FUNC.match(header)
            if func and term == "{":
                recv, name = func.group(1), func.group(2)
                return ("method" if recv else "function", f"{recv}.{name}" if recv else name, "leaf")
            typ = _GO_TYPE.match(header)
            if typ:
                return ("class", typ.group(1), "leaf")
            return None

        if lang in ("javascript", "typescript"):
            cls = _JS_TYPE.match(header)
            if cls:
                return ("class", cls.group(2), "recurse" if cls.group(1) == "class" else "leaf")
            func = _JS_FUNC.match(header)
            if func and (term == "{" or term == ";"):
                return ("method" if scope == "class" else "function", func.group(1) or "default", "leaf")
            arrow = _JS_ARROW.match(header)
            if arrow:
                return ("method" if scope == "class" else "function", arrow.group(1).lstrip("#"), "leaf")
            expr = _JS_FUNC_EXPR.match(header)
            if expr:
                return ("method" if scope == "class" else "function", expr.group(1), "leaf")
            if scope == "class" and "(" in header and term == "{":
                pre = header[: header.index("(")]
                if "=" in pre:
                    return None
                name = self._name_before_paren(pre)
                if name and name not in _CONTROL_WORDS:
                    return ("method", name.lstrip("#"), "leaf")
            return None

        if lang in ("c", "cpp"):
            if "(" not in header:
                typ = _C_TYPE.match(header)
                if not typ or term != "{":
                    return None
                keyword, rest = typ.group(1), typ.group(2)
                words = re.findall(r"\w+", rest.split(":", 1)[0])
                words = [w for w in words if w != "final"]
                name = words[-1] if words else None
                if keyword == "namespace":
                    return ("module", None, "transparent")
                recurse = lang == "cpp" and keyword in ("class", "struct") or keyword == "class"
                return ("class", name, "recurse" if recurse else "leaf")
            if term != "{":
                return None
            pre = header[: header.index("(")]
            if "=" in pre:
                return None
            name = self._name_before_paren(pre)
            if not name or name in _CONTROL_WORDS:
                return None
            before_name = pre[: pre.rfind(name)].strip()
            if before_name or "::" in name or scope == "class":
                return ("method" if scope == "class" else "function", name, "leaf")
            return None
        return None

    # -- scanning ----------------------------------------------------------

    def _is_lead(self, j: int) -> bool:
        original = self.o[j].strip()
        if not original:
            return False
        if not self.m[j].strip():
            return True  # comment-only line
        if original.startswith("@") and self.lang in ("java", "csharp", "typescript", "javascript"):
            return True
        if self.lang == "rust":
            return original.startswith(("#[", "#!["))
        return self.lang == "csharp" and bool(re.match(r"^\[[\w.]+.*\]$", original))

    def scan(self, lo: int, hi: int, scope: str, prefix: str) -> list[_Unit]:
        units: list[_Unit] = []
        i = lo
        floor = lo
        lead: tuple[int, int] | None = None
        while i < hi:
            line = self.m[i]
            stripped = line.strip()
            if not stripped:
                if not self.o[i].strip():
                    lead = None
                i += 1
                continue
            if self.lang in ("c", "cpp", "csharp") and stripped.startswith("#"):
                j = i
                while j + 1 < hi and self.o[j].rstrip().endswith("\\"):
                    j += 1
                i, lead = j + 1, None
                continue
            if _closing_only(self.o[i], line, self.lang):
                i += 1
                continue
            stmt = self._statement(i, hi)
            decl = self._classify(stmt, scope, prefix or None)
            if decl is None:
                annotation = _RUST_ATTR_ONLY if self.lang == "rust" else _ANNOTATION_ONLY
                if annotation.match(stmt["header"]) and stmt["term"] is None:
                    lead = (lead[0] if lead and lead[1] == i - 1 else i, stmt["last"])
                else:
                    lead = None
                if stmt["term"] == "{":
                    i = _block_end(self.m, stmt["line"], stmt["col"], hi) + 1
                else:
                    i = stmt["last"] + 1
                continue

            kind, name, mode = decl
            start = i
            if lead and lead[1] == start - 1:
                start = lead[0]
            while start - 1 >= floor and self._is_lead(start - 1):
                start -= 1
            if stmt["term"] == "{":
                end = _block_end(self.m, stmt["line"], stmt["col"], hi)
            elif stmt["term"] == ";":
                end = stmt["line"]
            else:
                end = stmt["last"]
            if kind == "class" and name is None and self.lang in ("c", "cpp") and end < self.n:
                tail = re.match(r"^\s*\}\s*(\w+)", self.o[end])
                name = tail.group(1) if tail else None
            symbol = f"{prefix}{name}" if name else (prefix.rstrip(".") or None)
            if mode == "transparent":
                units.extend(self.scan(stmt["line"] + 1, end, scope, prefix))
            elif mode in ("recurse", "recurse_mod"):
                inner = "class" if mode == "recurse" else "top"
                claimed = self.scan(stmt["line"] + 1, end, inner, f"{symbol}." if symbol else prefix) if end > stmt["line"] else []
                units.extend(_container_units(self.o, self.m, start, end, claimed, symbol, self.lang))
            else:
                units.append(_Unit(start, end, symbol, kind))
            i = end + 1
            floor = i
            lead = None
        return units


# ── Python scanner ───────────────────────────────────────────────────

_PY_DEF = re.compile(r"^(async\s+def|def|class)\s+(\w+)")


def _indent(line: str) -> int:
    return len(line.expandtabs(8)) - len(line.expandtabs(8).lstrip())


def _py_continuation_lines(o: list[str], m: list[str]) -> list[bool]:
    """Per line: does it start inside an open triple-quoted string or an open
    bracket? Such a line is part of the statement that began earlier, whatever
    its indentation (a docstring closing at column 0, ``1)`` after
    ``return (x +``) — never a dedent boundary."""
    flags = [False] * len(o)
    text = "\n".join(o)
    line_starts = [0]
    for line in o:
        line_starts.append(line_starts[-1] + len(line) + 1)
    first_line = 0
    for match in _PY_TOKENS.finditer(text):
        if match.group("s") is None or "\n" not in match.group(0):
            continue
        while first_line + 1 < len(line_starts) and line_starts[first_line + 1] <= match.start():
            first_line += 1
        for offset in range(match.group(0).count("\n")):
            if first_line + offset + 1 < len(flags):
                flags[first_line + offset + 1] = True
    depth = 0
    for k, line in enumerate(m):
        if depth > 0:
            flags[k] = True
        for ch in line:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
    return flags


def _scan_python(o: list[str], m: list[str], lo: int, hi: int, prefix: str, in_class: bool) -> list[_Unit]:
    units: list[_Unit] = []
    cont = _py_continuation_lines(o, m)
    base = None
    for k in range(lo, hi):
        if m[k].strip():
            base = _indent(m[k])
            break
    if base is None:
        return units
    i = lo
    floor = lo
    while i < hi:
        if cont[i] or not m[i].strip() or _indent(m[i]) != base:
            i += 1
            continue
        text = m[i].strip()
        found = _PY_DEF.match(text)
        if not found:
            i += 1
            continue
        # header end: parens balanced and the line ends with ':'
        depth = 0
        hdr_end = i
        for j in range(i, hi):
            for ch in m[j]:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
            if depth <= 0 and m[j].rstrip().endswith(":"):
                hdr_end = j
                break
            hdr_end = j
        end = hdr_end
        j = hdr_end + 1
        while j < hi:
            if not m[j].strip():
                j += 1
                continue
            if cont[j] or _indent(m[j]) > base:
                end = j
                j += 1
            else:
                break
        start = i
        while start - 1 >= floor:
            prev = o[start - 1].strip()
            if prev and ((not m[start - 1].strip()) or (prev.startswith("@") and _indent(o[start - 1]) == base)):
                start -= 1
            else:
                break
        name = found.group(2)
        symbol = f"{prefix}{name}"
        if found.group(1) == "class":
            claimed = _scan_python(o, m, hdr_end + 1, end + 1, f"{symbol}.", True) if end > hdr_end else []
            units.extend(_container_units(o, m, start, end, claimed, symbol, "python"))
        else:
            units.append(_Unit(start, end, symbol, "method" if in_class else "function"))
        i = end + 1
        floor = i
    return units


# ── Other languages ──────────────────────────────────────────────────


def _attach_leads(o: list[str], starts: list[int], is_lead: Callable[[str], bool], floor: int = 0) -> list[int]:
    """Move each start up over the contiguous comment/marker lines directly
    above it (never past the previous start)."""
    out: list[int] = []
    for index, start in enumerate(starts):
        limit = (starts[index - 1] + 1) if index else floor
        while start - 1 >= limit and o[start - 1].strip() and is_lead(o[start - 1].strip()):
            start -= 1
        out.append(start)
    return out


def _units_from_starts(
    o: list[str], starts: list[int], symbols: list[str | None], is_lead: Callable[[str], bool] | None, kind: str = "block"
) -> list[_Unit]:
    if not starts:
        return []
    effective = _attach_leads(o, starts, is_lead, 0) if is_lead else list(starts)
    last_nonblank = max((i for i, line in enumerate(o) if line.strip()), default=len(o) - 1)
    units: list[_Unit] = []
    for index, start in enumerate(effective):
        boundary = effective[index + 1] - 1 if index + 1 < len(effective) else last_nonblank
        end = boundary
        while end > start and not o[end].strip():
            end -= 1
        units.append(_Unit(start, end, symbols[index], kind))
    return units


def _scan_shell(o: list[str], m: list[str]) -> list[_Unit]:
    pattern = re.compile(r"^\s*(?:function\s+([\w:.-]+)\s*(?:\(\s*\))?|([\w:.-]+)\s*\(\s*\))\s*(\{?)")
    units: list[_Unit] = []
    i, n = 0, len(m)
    floor = 0
    while i < n:
        found = pattern.match(m[i])
        if not found:
            i += 1
            continue
        name = found.group(1) or found.group(2)
        open_line, open_col = None, 0
        for j in range(i, min(n, i + 3)):
            col = m[j].find("{", found.end(3) - 1 if j == i and found.group(3) else 0)
            if col >= 0:
                open_line, open_col = j, col
                break
        if open_line is None:
            i += 1
            continue
        end = _block_end(m, open_line, open_col, n)
        start = i
        while start - 1 >= floor and o[start - 1].strip() and not m[start - 1].strip():
            start -= 1
        units.append(_Unit(start, end, name, "function"))
        i = end + 1
        floor = i
    return units


def _scan_hcl(o: list[str], m: list[str]) -> list[_Unit]:
    head = re.compile(r'^\s*(\w[\w-]*)((?:\s+(?:"[^"]*"|[\w-]+))*)\s*\{')
    units: list[_Unit] = []
    i, n = 0, len(m)
    floor = 0
    while i < n:
        mline = m[i]
        if not mline.strip():
            i += 1
            continue
        found = head.match(o[i]) if head.match(mline) else None
        col = mline.find("{")
        if not found or col < 0:
            # not a block head: skip an unbalanced-looking attribute line
            if "{" in mline:
                i = _block_end(m, i, mline.find("{"), n) + 1
            else:
                i += 1
            continue
        labels = [lab.strip('"') for lab in re.findall(r'"[^"]*"|[\w-]+', found.group(2))]
        symbol = ".".join([found.group(1), *labels])
        end = _block_end(m, i, col, n)
        start = i
        while start - 1 >= floor and o[start - 1].strip() and not m[start - 1].strip():
            start -= 1
        units.append(_Unit(start, end, symbol, "block"))
        i = end + 1
        floor = i
    return units


def _scan_yaml(o: list[str], m: list[str]) -> list[_Unit]:
    key = re.compile(r"""^(?:"([^"]*)"|'([^']*)'|([^\s#:\-'"][^:#]*?|-[^\s:][^:#]*?))\s*:(?:\s|$)""")
    starts: list[int] = []
    symbols: list[str | None] = []
    for i, line in enumerate(m):
        if not line.strip() or line[0] in " \t":
            continue
        if line.startswith(("---", "...", "%")):
            continue
        found = key.match(line)
        if found:
            starts.append(i)
            symbols.append((found.group(1) or found.group(2) or found.group(3) or "").strip() or None)
        elif re.match(r"^-(?:\s|$)", line):
            starts.append(i)
            symbols.append(None)
    return _units_from_starts(o, starts, symbols, lambda s: s.startswith("#") or s.startswith("---") or s == "...")


def _scan_ini(o: list[str], m: list[str]) -> list[_Unit]:
    starts, symbols = [], []
    for i, line in enumerate(o):
        found = re.match(r"^\s*\[([^\]]+)\]", line)
        if found:
            starts.append(i)
            symbols.append(found.group(1).strip())
    return _units_from_starts(o, starts, symbols, lambda s: s.startswith(("#", ";")))


def _scan_properties(o: list[str], m: list[str]) -> list[_Unit]:
    starts, symbols = [], []
    previous_blank = True
    for i, line in enumerate(o):
        if not line.strip():
            previous_blank = True
            continue
        if previous_blank:
            starts.append(i)
            symbols.append(None)
            for text in o[i:]:
                if not text.strip():
                    break
                found = re.match(r"^\s*([^\s=:#!][^=:]*?)\s*[=:]", text)
                if found:
                    symbols[-1] = found.group(1).strip()
                    break
        previous_blank = False
    return _units_from_starts(o, starts, symbols, None)


def _scan_dockerfile(o: list[str], m: list[str]) -> list[_Unit]:
    starts, symbols = [], []
    for i, line in enumerate(o):
        if re.match(r"^\s*FROM\b", line, re.I):
            starts.append(i)
            alias = re.search(r"\bAS\s+([\w.-]+)", line, re.I)
            symbols.append(alias.group(1) if alias else None)
    return _units_from_starts(o, starts, symbols, lambda s: s.startswith("#"), kind="block")


def _scan_sql(o: list[str], m: list[str]) -> list[_Unit]:
    units: list[_Unit] = []
    n = len(m)
    i = 0
    create_re = re.compile(
        r"^\s*create\s+(?:or\s+replace\s+)?(?:(?:temp|temporary|unique|materialized|global|local|virtual)\s+)*"
        r"(table|view|index|function|procedure|trigger|schema|type|sequence|database|extension)\s+"
        r"(?:if\s+not\s+exists\s+)?([\w.\"`\[\]]+)",
        re.I,
    )
    other_re = re.compile(r"^\s*(?:alter\s+table|insert\s+into|drop\s+table|update|delete\s+from)\s+(?:if\s+exists\s+)?([\w.\"`\[\]]+)", re.I)
    floor = 0
    while i < n:
        if not m[i].strip():
            i += 1
            continue
        begins = bool(re.match(r"^\s*create\b.*\b(function|procedure|trigger|event|package)\b", m[i], re.I))
        depth = 0
        end = n - 1
        done = False
        text_parts: list[str] = []
        for j in range(i, n):
            line = m[j]
            text_parts.append(line)
            if begins:
                depth += len(re.findall(r"\b(?:begin|case)\b", line, re.I))
                depth -= len(re.findall(r"\bend\b", line, re.I))
            if depth <= 0 and ";" in line:
                end = j
                done = True
                break
        if not done:
            end = n - 1
            while end > i and not m[end].strip():
                end -= 1
        statement = "\n".join(text_parts)
        found = create_re.match(statement)
        symbol = None
        if found:
            symbol = found.group(2).strip('"`[]')
        else:
            other = other_re.match(statement)
            if other:
                symbol = other.group(1).strip('"`[]')
        start = i
        while start - 1 >= floor and o[start - 1].strip() and not m[start - 1].strip():
            start -= 1
        units.append(_Unit(start, end, symbol, "block"))
        i = end + 1
        floor = i
    return units


def _scan_json(o: list[str], m: list[str]) -> list[_Unit]:
    n = len(m)
    depth_at_start = []
    depth = 0
    root_open = None
    for i, line in enumerate(m):
        depth_at_start.append(depth)
        for ch in line:
            if ch in "{[":
                depth += 1
                if root_open is None:
                    root_open = i
            elif ch in "}]":
                depth -= 1
    if root_open is None:
        return []
    starts: list[int] = []
    previous: str | None = None
    previous_line = -1
    for i in range(root_open, n):
        text = m[i].strip()
        if not text:
            continue
        if i == root_open:
            if text.lstrip("{[").strip():
                starts.append(i)
        elif depth_at_start[i] == 1 and text[0] not in "}]":
            if previous is not None and (previous.endswith(",") or (previous_line == root_open and previous in ("{", "["))):
                starts.append(i)
        previous, previous_line = text, i
    if not starts:
        return []
    last = max(i for i in range(n) if m[i].strip())
    if m[last].strip()[:1] in "}]" and depth_at_start[last] == 1:
        last = max(i for i in range(last) if m[i].strip())
    symbols: list[str | None] = []
    for s in starts:
        found = re.match(r'^\s*[{\[]?\s*"((?:[^"\\]|\\.)*)"\s*:', o[s])
        symbols.append(found.group(1) if found else None)
    units = []
    for index, s in enumerate(starts):
        end = (starts[index + 1] - 1) if index + 1 < len(starts) else last
        while end > s and not o[end].strip():
            end -= 1
        units.append(_Unit(s, end, symbols[index], "block"))
    return units


_XML_TOKEN = re.compile(
    r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<![^>]*>"
    r"|<(?P<close>/)?(?P<name>[\w:.-]+)(?P<attrs>(?:[^>\"']|\"[^\"]*\"|'[^']*')*?)(?P<self>/)?>",
    re.S,
)


def _scan_xml(o: list[str]) -> list[_Unit]:
    text = "\n".join(o)
    offsets = [0]
    for line in o[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)

    def line_of(pos: int) -> int:
        return bisect.bisect_right(offsets, pos) - 1

    depth = 0
    units: list[_Unit] = []
    child_start: tuple[int, str] | None = None
    for token in _XML_TOKEN.finditer(text):
        name = token.group("name")
        if name is None:
            continue
        if token.group("close"):
            depth -= 1
            if depth == 1 and child_start is not None:
                units.append(_Unit(child_start[0], line_of(token.end() - 1), child_start[1], "block"))
                child_start = None
        elif token.group("self"):
            if depth == 1:
                line = line_of(token.start())
                units.append(_Unit(line, line_of(token.end() - 1), name, "block"))
        else:
            if depth == 1:
                child_start = (line_of(token.start()), name)
            depth += 1
    # lead comments
    result: list[_Unit] = []
    floor = 0
    for unit in sorted(units, key=lambda u: u.start):
        if result and unit.start <= result[-1].end:
            result[-1].end = max(result[-1].end, unit.end)
            continue
        start = unit.start
        while start - 1 >= max(floor, 0) and o[start - 1].strip().startswith("<!--") and o[start - 1].strip().endswith("-->"):
            start -= 1
        result.append(_Unit(start, unit.end, unit.symbol, unit.kind))
        floor = unit.end + 1
    return result


# ── Assembly ─────────────────────────────────────────────────────────


def _scan_units(lines: list[str], masked: list[str], language: str | None) -> list[_Unit]:
    if language in ("java", "csharp", "go", "rust", "javascript", "typescript", "c", "cpp"):
        return _BraceScanner(lines, masked, language).scan(0, len(lines), "top", "")
    if language == "python":
        return _scan_python(lines, masked, 0, len(lines), "", False)
    if language == "shell":
        return _scan_shell(lines, masked)
    if language == "hcl":
        return _scan_hcl(lines, masked)
    if language == "yaml":
        return _scan_yaml(lines, masked)
    if language == "json":
        return _scan_json(lines, masked)
    if language == "xml":
        return _scan_xml(lines)
    if language == "ini":
        return _scan_ini(lines, masked)
    if language == "properties":
        return _scan_properties(lines, masked)
    if language == "sql":
        return _scan_sql(lines, masked)
    if language == "dockerfile":
        return _scan_dockerfile(lines, masked)
    return []


def _split_range(lines: list[str], a: int, b: int, max_chars: int) -> list[tuple[int, int]]:
    """Cut lines ``a..b`` into consecutive pieces of at most ``max_chars``
    characters at line boundaries, preferring to end a piece on a blank line.
    A single line longer than ``max_chars`` is a piece of its own (``chunk_code``
    then cuts it into slices)."""
    if len("\n".join(lines[a : b + 1])) <= max_chars:
        return [(a, b)]
    pieces: list[tuple[int, int]] = []
    ps = a
    while ps <= b:
        size = 0
        last_blank = -1
        j = ps
        while j <= b:
            add = len(lines[j]) + (1 if j > ps else 0)
            if j > ps and (size + add > max_chars or len(lines[j]) > max_chars):
                break
            size += add
            if not lines[j].strip():
                last_blank = j
            j += 1
        pe = b if j > b else (last_blank if last_blank > ps else j - 1)
        pieces.append((ps, pe))
        ps = pe + 1
    return pieces


def chunk_code(
    text: str,
    language: str | None,
    *,
    max_chars: int = 1500,
    context_lines: int = 0,
) -> list[CodeChunk]:
    """Split source ``text`` into one chunk per declaration.

    ``language`` is a name from :data:`CODE_LANGUAGES` (aliases such as ``py``
    / ``ts`` / ``yml`` accepted; None or unknown → blank-line blocks packed to
    ``max_chars``). A unit longer than ``max_chars`` is cut at line
    boundaries (blank lines preferred) and the later pieces get a ``#2``,
    ``#3`` … symbol suffix. With ``context_lines > 0`` each chunk's text also
    carries that many lines before and after (``start_line`` / ``end_line``
    then describe that text; ``unit_*`` still describe the declaration)."""
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if context_lines < 0:
        raise ValueError("context_lines must be >= 0")
    lines = _split_lines(text)
    if not text.strip():
        return []
    n = len(lines)
    language = normalize_language(language)
    masked = _split_lines(_mask(text, language)) if language in _MASK_PATTERNS or language == "rust" else list(lines)
    if len(masked) != n:  # defensive: masking must preserve line count
        masked = list(lines)
    try:
        units = _scan_units(lines, masked, language)
    except Exception:  # noqa: BLE001 — a heuristic scanner must never fail a document
        units = []

    # Normalise: sorted, non-overlapping, clipped.
    clean: list[_Unit] = []
    for unit in sorted(units, key=lambda u: (u.start, u.end)):
        if unit.end < unit.start or unit.start < 0 or unit.end >= n:
            continue
        if clean and unit.start <= clean[-1].end:
            continue
        clean.append(unit)

    # Fill gaps with module units; drop closing-only gaps (the previous chunk carries them).
    full: list[_Unit] = []
    cursor = 0
    for unit in [*clean, None]:
        gap_end = (unit.start - 1) if unit is not None else n - 1
        if full:  # closing punctuation right after the previous unit stays with it
            while cursor <= gap_end and _closing_only(lines[cursor], masked[cursor], language):
                cursor += 1
        if gap_end >= cursor:
            for gap in _runs(lines, masked, cursor, gap_end, None, "module", language):
                full.append(gap)
        if unit is not None:
            full.append(unit)
            cursor = unit.end + 1
    if not full:
        first = next(i for i, line in enumerate(lines) if line.strip())
        last = max(i for i, line in enumerate(lines) if line.strip())
        full = [_Unit(first, last, None, "module")]
    full.sort(key=lambda u: u.start)

    chunks: list[CodeChunk] = []
    for index, unit in enumerate(full):
        a = 0 if index == 0 else unit.start
        b = (full[index + 1].start - 1) if index + 1 < len(full) else n - 1
        piece_no = 0
        for ps, pe in _split_range(lines, a, b, max_chars):
            cs = max(0, ps - context_lines)
            ce = min(n - 1, pe + context_lines)
            body = "\n".join(lines[cs : ce + 1])
            # One line longer than max_chars (minified code, a data blob): cut it into
            # max_chars slices that all cite that line.
            slices = [body[k : k + max_chars] for k in range(0, len(body), max_chars)] if ps == pe and len(lines[ps]) > max_chars and context_lines == 0 else [body]
            for part in slices:
                piece_no += 1
                symbol = unit.symbol if piece_no == 1 or unit.symbol is None else f"{unit.symbol}#{piece_no}"
                chunks.append(
                    CodeChunk(
                        text=part,
                        start_line=cs + 1,
                        end_line=ce + 1,
                        symbol=symbol,
                        kind=unit.kind,
                        unit_start_line=unit.start + 1,
                        unit_end_line=unit.end + 1,
                    )
                )
    return chunks

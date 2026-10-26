"""Unified-diff parser / applier for code chunks (pure, no subprocess).

A model's remediation is a unified diff against a *chunk* of a file whose lines
carry absolute line numbers. :func:`apply_patch_to_chunk` applies such a diff
strictly (context and removed lines must match, trailing whitespace aside) with a
bounded search for the nearest exact match when the stated position is off, and
reports *why* it failed with a stable reason code. :func:`make_unified_diff` is
the inverse used to build fix pairs; ``apply(old, make(old, new)) == new``.

Parsing is deliberately tolerant of how models write diffs: a ```diff fence,
prose before the first header, missing ``---``/``+++`` headers (hunks only),
``a/`` ``b/`` prefixes, ``\\ No newline at end of file`` markers, CRLF, and
header counts that disagree with the body (the body wins).
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field

Line = tuple[str, str]  # (op, text) with op in {" ", "-", "+"}

_HUNK_RE = re.compile(r"^@@+\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@+")
_NOEOL = "\\"
_NOEOL_SENTINEL = "\x00NOEOL"


class DiffError(ValueError):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[Line] = field(default_factory=list)
    header_counts_ok: bool = True
    old_no_eol: bool = False
    new_no_eol: bool = False

    @property
    def old_lines(self) -> list[str]:
        return [t for op, t in self.lines if op in (" ", "-")]

    @property
    def new_lines(self) -> list[str]:
        return [t for op, t in self.lines if op in (" ", "+")]


@dataclass
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)


@dataclass
class ApplyResult:
    ok: bool
    patched_code: str | None = None
    reason: str | None = None
    hunks: int = 0
    offsets: list[int] = field(default_factory=list)
    header_counts_ok: bool = True
    detail: str | None = None


# ── parsing ──


def _strip_fence(lines: list[str]) -> list[str]:
    # only column-0 fences: an indented ``` is a context line, not a fence
    fences = [i for i, ln in enumerate(lines) if ln.startswith("```")]
    if not fences:
        return lines
    start = fences[0] + 1
    end = fences[1] if len(fences) > 1 else len(lines)
    return lines[start:end]


def _clean_path(raw: str) -> str | None:
    raw = raw.strip().split("\t")[0].strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        raw = raw[1:-1]
    if not raw or raw == "/dev/null":
        return None
    if raw[:2] in ("a/", "b/"):
        raw = raw[2:]
    return raw or None


def _is_file_header(lines: list[str], i: int) -> bool:
    return lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")


def parse_unified_diff(text: str) -> list[FilePatch]:
    if text is None or not str(text).strip():
        raise DiffError("empty_patch", "patch is empty")
    lines = _strip_fence(str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    patches: list[FilePatch] = []
    cur: FilePatch | None = None
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        if _is_file_header(lines, i):
            cur = FilePatch(_clean_path(ln[4:]), _clean_path(lines[i + 1][4:]))
            patches.append(cur)
            i += 2
            continue
        m = _HUNK_RE.match(ln)
        if not m:
            i += 1  # prose, `diff --git`, `index ...`, blank lines between files
            continue
        if cur is None:
            cur = FilePatch(None, None)
            patches.append(cur)
        old_start = int(m.group(1))
        old_len = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_len = int(m.group(4)) if m.group(4) is not None else 1
        hunk = Hunk(old_start, old_len, new_start, new_len)
        i += 1
        n_old = n_new = 0
        while i < n:
            ln = lines[i]
            if ln.startswith("@@") or ln.startswith("diff ") or _is_file_header(lines, i):
                break
            satisfied = n_old >= old_len and n_new >= new_len
            if ln.startswith(_NOEOL):
                if hunk.lines:
                    last = hunk.lines[-1][0]
                    if last in ("-", " "):
                        hunk.old_no_eol = True
                    if last in ("+", " "):
                        hunk.new_no_eol = True
                i += 1
                continue
            if ln == "":
                j = next((k for k in range(i + 1, n) if lines[k] != ""), None)
                if (
                    j is None
                    or lines[j][0] not in " +-\\"
                    or lines[j].startswith("diff ")
                    or _is_file_header(lines, j)
                ):
                    break  # trailing blank lines are not blank context
                # a blank context line whose leading space was stripped; the body wins
                op, txt = " ", ""
            elif ln[0] in " +-":
                op, txt = ln[0], ln[1:]
            else:
                break
            if satisfied:
                hunk.header_counts_ok = False
            hunk.lines.append((op, txt))
            if op in (" ", "-"):
                n_old += 1
            if op in (" ", "+"):
                n_new += 1
            i += 1
        # trailing blank context lines that the model padded on
        while hunk.lines and hunk.lines[-1] == (" ", "") and (n_old > old_len or n_new > new_len):
            hunk.lines.pop()
            n_old -= 1
            n_new -= 1
        if n_old != old_len or n_new != new_len:
            hunk.header_counts_ok = False
        cur.hunks.append(hunk)
    if not any(p.hunks for p in patches):
        raise DiffError("not_a_diff", "no hunks found (expected '@@ -a,b +c,d @@' sections)")
    return [p for p in patches if p.hunks]


# ── applying ──


def _eq(a: str, b: str) -> bool:
    return a.rstrip(" \t\r") == b.rstrip(" \t\r")


def _matches(lines: list[str], idx: int, old: list[str]) -> bool:
    if idx < 0 or idx + len(old) > len(lines):
        return False
    return all(_eq(lines[idx + k], old[k]) for k in range(len(old)))


def _base(path: str | None) -> str | None:
    return os.path.basename(path) if path else None


def apply_patch_to_chunk(code: str, start_line: int, patch: str, *, path: str | None = None, max_offset: int = 20) -> ApplyResult:
    def fail(reason: str, detail: str | None = None, **kw) -> ApplyResult:
        return ApplyResult(False, None, reason, detail=detail, **kw)

    if patch is None or not str(patch).strip():
        return fail("empty_patch", "patch is empty")
    try:
        files = parse_unified_diff(patch)
    except DiffError as exc:
        return fail(exc.code if exc.code in ("empty_patch", "not_a_diff") else "not_a_diff", exc.message)
    if len(files) > 1:
        return fail("multiple_files", f"patch touches {len(files)} files")
    fp = files[0]
    if path:
        named = [p for p in (fp.new_path, fp.old_path) if p]
        if named and _base(path) not in {_base(p) for p in named}:
            return fail("other_file", f"patch names {named[0]!r}, expected {path!r}")

    crlf = "\r\n" in code
    norm = code.replace("\r\n", "\n")
    trailing = norm.endswith("\n")
    lines = norm.split("\n")
    if trailing:
        lines.pop()
    original_lines = list(lines)

    out: list[str] = []
    cursor = 0  # index into original lines consumed so far
    offsets: list[int] = []
    counts_ok = True
    last_hunk: Hunk | None = None
    for h in fp.hunks:
        counts_ok = counts_ok and h.header_counts_ok
        old, new = h.old_lines, h.new_lines
        stated = h.old_start - start_line + (1 if not old else 0)
        found: int | None = None
        if last_hunk is not None and stated < cursor:
            return fail("overlapping_hunks", f"hunk at line {h.old_start} overlaps the previous hunk", hunks=len(fp.hunks))
        if not old and (stated < 0 or stated > len(original_lines)):
            return fail("out_of_range", f"hunk at line {h.old_start} is outside the shown code ({start_line}-{start_line + len(original_lines) - 1})", hunks=len(fp.hunks))

        def _ok(idx: int) -> bool:
            if idx < cursor:
                return False
            if not old:
                return idx <= len(original_lines)
            return _matches(original_lines, idx, old)

        for dist in range(0, max(0, int(max_offset)) + 1):
            cands = [stated] if dist == 0 else [stated + dist, stated - dist]
            hits = [c for c in cands if _ok(c)]
            if len(hits) == 2:
                return fail("context_mismatch", f"hunk at line {h.old_start}: ambiguous context (matches at +{dist} and -{dist})", hunks=len(fp.hunks))
            if hits:
                found = hits[0]
                break
        if found is None:
            if stated < 0 or stated + len(old) > len(original_lines):
                return fail("out_of_range", f"hunk at line {h.old_start} is outside the shown code ({start_line}-{start_line + len(original_lines) - 1})", hunks=len(fp.hunks))
            return fail("context_mismatch", f"hunk at line {h.old_start}: context/removed lines do not match the code", hunks=len(fp.hunks))
        offsets.append(found - stated)
        out.extend(original_lines[cursor:found])
        out.extend(new)
        cursor = found + len(old)
        last_hunk = h
    out.extend(original_lines[cursor:])

    if last_hunk is not None:
        if any(h.new_no_eol for h in fp.hunks):
            trailing = False
        elif any(h.old_no_eol for h in fp.hunks):
            trailing = True
    patched = "\n".join(out) + ("\n" if trailing and out else "")
    if patched == norm:
        return fail("no_op", "patch leaves the code unchanged", hunks=len(fp.hunks), offsets=offsets, header_counts_ok=counts_ok)
    if crlf:
        patched = patched.replace("\n", "\r\n")
    return ApplyResult(True, patched, None, len(fp.hunks), offsets, counts_ok)


# ── building ──


def make_unified_diff(old_code: str, new_code: str, *, path: str | None = None, start_line: int = 1, context: int = 3) -> str:
    old_eol = old_code.endswith("\n") or not old_code
    new_eol = new_code.endswith("\n") or not new_code
    a = old_code.split("\n")
    b = new_code.split("\n")
    if old_code.endswith("\n"):
        a.pop()
    if new_code.endswith("\n"):
        b.pop()
    if old_eol != new_eol:  # make the final line differ so the marker is attached to the right side
        if not old_eol and a:
            a[-1] += _NOEOL_SENTINEL
        if not new_eol and b:
            b[-1] += _NOEOL_SENTINEL
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    name = path or "snippet"
    out: list[str] = []

    def emit(op: str, text: str) -> None:
        if text.endswith(_NOEOL_SENTINEL):
            out.append(op + text[: -len(_NOEOL_SENTINEL)])
            out.append("\\ No newline at end of file")
        else:
            out.append(op + text)

    for group in sm.get_grouped_opcodes(max(0, context)):
        i1, i2, j1, j2 = group[0][1], group[-1][2], group[0][3], group[-1][4]
        old_len, new_len = i2 - i1, j2 - j1
        old_start = start_line + i1 if old_len else start_line + i1 - 1
        new_start = start_line + j1 if new_len else start_line + j1 - 1
        out_hdr = f"@@ -{old_start},{old_len} +{new_start},{new_len} @@"
        body_start = len(out)
        out.append(out_hdr)
        for tag, x1, x2, y1, y2 in group:
            if tag == "equal":
                for t in a[x1:x2]:
                    emit(" ", t)
            else:
                for t in a[x1:x2]:
                    emit("-", t)
                for t in b[y1:y2]:
                    emit("+", t)
        _ = body_start
    if not out:
        return ""
    return f"--- a/{name}\n+++ b/{name}\n" + "\n".join(out) + "\n"

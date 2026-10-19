"""Page furniture: repeated page headers / footers / page numbers.

A long PDF prints the same lines on every page — a running title, "Page 9
of 142", a print stamp, a proof notice. Text extraction keeps them inline,
so after cleaning they sit in the middle of passages ("…you are responsible
for Page 10 of 142 Fileid: … 13:47 - 20-Jan-2026"), Q&A generation writes
questions about them and retrieval indexes them.

``remove_page_furniture(text)`` drops them before chunking. It needs page
boundaries: the PDF parser joins pages with :data:`PAGE_SEPARATOR` (a form
feed on its own line); a text with fewer than :data:`FURNITURE_MIN_PAGES`
pages is returned unchanged. Only the first / last :data:`EDGE_LINES`
non-empty lines of a page are ever touched — body text in the middle of a
page is never removed, however often it repeats.

A line (or the head / tail of one, when extraction fused the footer onto the
last body line) is furniture when, in an edge zone:

* **repeats** — its digit-normalised form (every number → ``#``, case and
  whitespace folded) is on at least :data:`FURNITURE_PAGE_SHARE` of the
  pages (and at least :data:`FURNITURE_MIN_PAGES` of them). A head / tail
  cut from a line it was fused to must either be seen as a whole line of
  its own on some page ("Page 9 of 142" glued to a body line here, on its
  own line elsewhere) or be at least :data:`MIN_SEGMENT_CHARS` long, two
  words and carry a number — lines that merely share a phrasing never
  qualify;
* **page number** — the whole line is a page-number pattern ("Page 9",
  "Page 9 of 142", "- 9 -", "9 / 142", "[9/142]", "p. 9", a bare number);
* **running head** — a short line with numbers whose normalised form is on
  at least :data:`FURNITURE_MIN_PAGES` pages with the numbers *changing*
  from page to page ("Chapter 1 Filing Information 7", "… 9", "… 11"): a
  page counter, not a repeated body line (a repeated body line repeats its
  numbers too).

No domain knowledge: everything is measured on the document itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.utils.file_parsers import PAGE_SEPARATOR  # noqa: F401  (re-exported)

FURNITURE_PAGE_SHARE = 0.5
FURNITURE_MIN_PAGES = 3
EDGE_LINES = 4
MIN_SEGMENT_CHARS = 12
MIN_WHOLE_LINE_SEGMENT_CHARS = 6
RUNNING_HEAD_MAX_CHARS = 80
RUNNING_HEAD_DISTINCT_SHARE = 0.8
RUNNING_HEAD_MIN_LETTERS = 3
MAX_EXAMPLES = 8

_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_PAGE_NUMBER_RE = re.compile(
    r"""^(?:
        (?:page|pg\.?|p\.)\s*\d+(?:\s*(?:of|/)\s*\d+)?   # Page 9 / Page 9 of 142 / p. 9
      | -\s*\d+\s*-                                       # - 9 -
      | \[?\s*\d+\s*(?:/|of)\s*\d+\s*\]?                  # 9 / 142, [9/142], 9 of 142
    )$""",
    re.IGNORECASE | re.VERBOSE,
)


def has_page_breaks(text: str) -> bool:
    return "\f" in text


def split_pages(text: str) -> list[str]:
    return text.split("\f")


def join_pages(pages: list[str]) -> str:
    """Pages back into one text, a blank line between them (no form feed)."""
    out = "\n\n".join(page.strip("\n") for page in pages)
    return out


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """``text`` lower-cased, every digit run → ``#``, whitespace runs → one
    space; plus, for each normalised char, the index in ``text`` it came
    from (to cut the original at a normalised position)."""
    out: list[str] = []
    index: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isdigit():
            j = i
            while j < n and text[j].isdigit():
                j += 1
            out.append("#")
            index.append(i)
            i = j
            continue
        if ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if out and out[-1] != " ":
                out.append(" ")
                index.append(i)
            i = j
            continue
        out.append(ch.lower())
        index.append(i)
        i += 1
    while out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def normalise_line(line: str) -> str:
    return _normalise_with_map(line.strip())[0]


_BARE_NUMBER_RE = re.compile(r"^\d{1,4}$")


def is_page_number_line(line: str, *, outermost: bool = True) -> bool:
    """A page-number line. A bare number ("9") counts only as the page's
    first / last line (``outermost``) — inside the edge zone it is as likely
    a table cell."""
    body = _WS_RE.sub(" ", line.strip())
    if _PAGE_NUMBER_RE.match(body):
        return True
    return outermost and bool(_BARE_NUMBER_RE.match(body))


def _edge_indices(lines: list[str]) -> list[int]:
    """Indices of the first and last ``EDGE_LINES`` non-empty lines."""
    filled = [i for i, line in enumerate(lines) if line.strip()]
    if len(filled) <= 2 * EDGE_LINES:
        return filled
    return filled[:EDGE_LINES] + filled[-EDGE_LINES:]


def _fused_candidate(seg: str) -> bool:
    """A head / tail long enough to be a fused header / footer, with a
    number in it (a page counter, a date). Without a number, lines that
    merely share a phrasing ("… for this page.") would qualify."""
    return len(seg) >= MIN_SEGMENT_CHARS and " " in seg and "#" in seg


def _segments(norm: str) -> set[str]:
    """The whole normalised line, plus its heads and tails cut at a space
    that are long enough to be a fused header / footer."""
    found = {norm} if norm else set()
    for pos, ch in enumerate(norm):
        if ch != " ":
            continue
        for seg in (norm[:pos], norm[pos + 1:]):
            if _fused_candidate(seg):
                found.add(seg)
    return found


@dataclass
class FurnitureReport:
    pages: int = 0
    lines_removed: int = 0
    segments_trimmed: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    def note(self, rule: str, text: str) -> None:
        self.by_rule[rule] = self.by_rule.get(rule, 0) + 1
        sample = _WS_RE.sub(" ", text.strip())[:160]
        if sample and sample not in self.examples and len(self.examples) < MAX_EXAMPLES:
            self.examples.append(sample)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "furniture_lines_removed": self.lines_removed,
            "furniture_segments_trimmed": self.segments_trimmed,
            "by_rule": dict(self.by_rule),
            "examples": list(self.examples),
        }


def remove_page_furniture(text: str) -> tuple[str, FurnitureReport]:
    """``text`` (pages separated by form feeds) without its repeated page
    headers / footers / page numbers, pages joined by a blank line; and a
    report. A text with fewer than ``FURNITURE_MIN_PAGES`` pages comes back
    with only its page breaks turned into blank lines."""
    pages = split_pages(text)
    report = FurnitureReport(pages=len(pages))
    if len(pages) < FURNITURE_MIN_PAGES:
        return join_pages(pages), report

    page_lines = [page.split("\n") for page in pages]
    edges = [_edge_indices(lines) for lines in page_lines]
    need = max(FURNITURE_MIN_PAGES, int(FURNITURE_PAGE_SHARE * len(pages) + 0.999999))

    # Rule "repeats": how many pages carry each segment in an edge zone.
    seg_pages: dict[str, int] = {}
    for lines, idx in zip(page_lines, edges):
        seen: set[str] = set()
        for i in idx:
            seen |= _segments(normalise_line(lines[i]))
        for seg in seen:
            seg_pages[seg] = seg_pages.get(seg, 0) + 1
    repeated = {seg for seg, count in seg_pages.items() if count >= need}
    # A short segment ("page # of #") may be cut from a line it was fused to
    # only when it is also seen as a whole line of its own somewhere.
    whole_lines = {normalise_line(lines[i]) for lines, idx in zip(page_lines, edges) for i in idx}
    # Longest first: strip the whole footer, not the tail of it.
    repeated_by_len = sorted(repeated, key=len, reverse=True)

    def strip_repeated(line: str) -> tuple[str, bool]:
        """(what's left of ``line``, whether anything was cut)."""
        cut = False
        for _ in range(3):
            body = line.strip()
            norm, index = _normalise_with_map(body)
            if not norm:
                return "", cut
            if norm in repeated:
                return "", True
            changed = False
            for seg in repeated_by_len:
                if len(seg) >= len(norm):
                    continue
                if not (_fused_candidate(seg) or (len(seg) >= MIN_WHOLE_LINE_SEGMENT_CHARS and seg in whole_lines)):
                    continue
                if norm.endswith(seg):
                    line = body[: index[len(norm) - len(seg)]].rstrip()
                    changed = True
                    break
                if norm.startswith(seg):
                    end = len(seg)
                    line = body[index[end]:].lstrip() if end < len(index) else ""
                    changed = True
                    break
            if not changed:
                break
            cut = True
        return line, cut

    # First pass: repeated lines / segments and page numbers.
    edited = [list(lines) for lines in page_lines]
    for p, idx in enumerate(edges):
        outer = {idx[0], idx[-1]} if idx else set()
        for i in idx:
            original = edited[p][i]
            if is_page_number_line(original, outermost=i in outer):
                edited[p][i] = None  # type: ignore[call-overload]
                report.lines_removed += 1
                report.note("page_number", original)
                continue
            left, cut = strip_repeated(original)
            if not cut:
                continue
            if left.strip() and not is_page_number_line(left, outermost=i in outer):
                edited[p][i] = left
                report.segments_trimmed += 1
                whole, kept = original.strip(), left.strip()
                if whole.startswith(kept):
                    report.note("repeated", whole[len(kept):])
                elif whole.endswith(kept):
                    report.note("repeated", whole[: len(whole) - len(kept)])
                else:
                    report.note("repeated", whole)
            else:
                edited[p][i] = None  # type: ignore[call-overload]
                report.lines_removed += 1
                report.note("repeated", original)

    # Second pass: running heads (same words, page-counting numbers).
    groups: dict[str, list[tuple[int, int, tuple[str, ...]]]] = {}
    for p, idx in enumerate(edges):
        for i in idx:
            line = edited[p][i]
            if line is None:
                continue
            body = line.strip()
            if not body or len(body) > RUNNING_HEAD_MAX_CHARS or not _DIGITS_RE.search(body):
                continue
            # Words, not just numbers: a row of figures that changes from
            # page to page is a table, not a running head.
            if sum(ch.isalpha() for ch in body) < RUNNING_HEAD_MIN_LETTERS:
                continue
            norm = normalise_line(body)
            groups.setdefault(norm, []).append((p, i, tuple(_DIGITS_RE.findall(body))))
    for occurrences in groups.values():
        pages_hit = {p for p, _, _ in occurrences}
        distinct = {digits for _, _, digits in occurrences}
        if len(pages_hit) < FURNITURE_MIN_PAGES:
            continue
        if len(distinct) < max(FURNITURE_MIN_PAGES, RUNNING_HEAD_DISTINCT_SHARE * len(occurrences)):
            continue
        for p, i, _ in occurrences:
            report.lines_removed += 1
            report.note("running_head", edited[p][i])
            edited[p][i] = None  # type: ignore[call-overload]

    out_pages = ["\n".join(line for line in lines if line is not None) for lines in edited]
    return join_pages(out_pages), report

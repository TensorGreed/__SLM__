"""Custom infotypes: a span project's own identifier formats as regexes.

The span pipeline's regex pass (``synthetic_service._extract_entities_via_regex``)
ships built-in patterns for the well-known formats (email, phone, card, IP,
…). An organisation's own identifiers — an employee id like ``EMP-48213``, a
ticket reference, an internal account format — are just as deterministic,
and a regex catches them with exact offsets where a labelling model may miss
or misplace them. A project keeps its own as
``runtime_config["span_custom_patterns"]`` = a list of
``{type, pattern, description?, flags?}``; every regex pass that runs for the
project (span generation, regex-augment over saved rows, relabel / passage /
document labelling) uses the built-ins **plus** these. A custom pattern adds
matches for its type; it overrides nothing.

Validation on write (``validate_pattern``), kept simple:

- ``type`` is normalised like the type definitions (lower-case, spaces →
  ``_``) and must be non-empty;
- the pattern must compile, be at most ``MAX_PATTERN_CHARS`` long, and must
  not match the empty string (it would emit zero-length spans everywhere);
- catastrophic-backtracking guard, two cheap layers: (1) structural — an
  unbounded repeat (``+``, ``*``, ``{n,}``) that contains another unbounded
  repeat, e.g. ``(a+)+`` / ``(\\w+\\s?)*``, is rejected outright (CPython's
  ``re`` can't be interrupted, so we refuse the classic exponential shape
  instead of trying to time it out); (2) timing — what passes is run over a
  few ~2,000-char adversarial strings and rejected if a scan takes longer
  than ``MAX_SCAN_SECONDS`` (catches polynomial blow-ups the structural rule
  doesn't see). The scan runs in a short-lived child Python (``-I -S``),
  killed after ``SPEED_CHECK_TIMEOUT`` seconds: CPython's ``re`` holds the
  GIL and can't be interrupted from a thread, so an exponential pattern like
  ``(a|a)+b`` would otherwise hang the server. The speed check runs on write
  (PUT) and on the test preview, not on every re-read of stored patterns.

``flags`` is a string of letters: ``i`` (ignore case), ``m`` (multiline),
``s`` (dot matches newline). Pure helpers; the DB read is ``load_patterns``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import lru_cache
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

try:  # Python 3.11+
    import re._parser as _sre_parse  # type: ignore[import-not-found]
    import re._constants as _sre_constants  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - older Pythons
    import sre_constants as _sre_constants  # type: ignore[no-redef]
    import sre_parse as _sre_parse  # type: ignore[no-redef]

PATTERNS_KEY = "span_custom_patterns"
MAX_PATTERNS = 100
MAX_PATTERN_CHARS = 300
MAX_DESCRIPTION_CHARS = 400
MAX_SCAN_SECONDS = 0.25
SPEED_CHECK_TIMEOUT = 3.0
ADVERSARIAL_LENGTH = 2000
TEST_SAMPLE_TEXTS = 500
TEST_MAX_SNIPPETS = 10
_SNIPPET_CONTEXT = 30

_FLAG_BITS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}

# Strings that make naive patterns backtrack: long runs of one character
# class, then a character that forces the match to fail at the end.
_ADVERSARIAL = [
    "a" * ADVERSARIAL_LENGTH + "!",
    "1" * ADVERSARIAL_LENGTH + "!",
    " " * ADVERSARIAL_LENGTH + "!",
    ("aA1-_. @" * (ADVERSARIAL_LENGTH // 8)) + "\x00",
    ("ab" * (ADVERSARIAL_LENGTH // 2)) + "!",
]


def normalise_type(name: Any) -> str:
    """Same normalisation as ``span_type_definitions_service``."""
    return re.sub(r"\s+", "_", str(name or "").strip().lower())


def normalise_flags(flags: Any) -> str:
    """Known flag letters only, sorted, de-duplicated."""
    return "".join(sorted({c for c in str(flags or "").lower() if c in _FLAG_BITS}))


def _flag_bits(flags: str) -> int:
    bits = 0
    for c in flags:
        bits |= _FLAG_BITS.get(c, 0)
    return bits


def _unbounded(max_repeat: int) -> bool:
    return max_repeat == _sre_constants.MAXREPEAT


_REPEAT_OPS = {
    getattr(_sre_constants, name)
    for name in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")
    if hasattr(_sre_constants, name)
}


def _contains_unbounded_repeat(items: Any) -> bool:
    for op, av in items:
        if op in _REPEAT_OPS:
            if _unbounded(av[1]) or _contains_unbounded_repeat(av[2]):
                return True
        elif _children_have(op, av, _contains_unbounded_repeat):
            return True
    return False


def _children_have(op: Any, av: Any, fn: Any) -> bool:
    if op == _sre_constants.SUBPATTERN:
        return fn(av[-1])
    if op == _sre_constants.BRANCH:
        return any(fn(branch) for branch in av[1])
    if op in (_sre_constants.ASSERT, _sre_constants.ASSERT_NOT):
        return fn(av[1])
    if hasattr(_sre_constants, "ATOMIC_GROUP") and op == _sre_constants.ATOMIC_GROUP:
        return fn(av)
    if op == _sre_constants.GROUPREF_EXISTS:
        return fn(av[1]) or (av[2] is not None and fn(av[2]))
    return False


def _has_nested_unbounded(items: Any) -> bool:
    """An unbounded repeat whose body itself contains an unbounded repeat."""
    for op, av in items:
        if op in _REPEAT_OPS:
            if _unbounded(av[1]) and _contains_unbounded_repeat(av[2]):
                return True
            if _has_nested_unbounded(av[2]):
                return True
        elif _children_have(op, av, _has_nested_unbounded):
            return True
    return False


_SPEED_SCRIPT = r"""
import json, re, sys, time
data = json.loads(sys.stdin.read())
compiled = re.compile(data["pattern"], data["bits"])
worst = 0.0
for probe in data["probes"]:
    start = time.perf_counter()
    for _ in compiled.finditer(probe):
        pass
    worst = max(worst, time.perf_counter() - start)
print(worst)
"""


def _speed_check(pattern: str, flags: str) -> str | None:
    """Scan the adversarial strings in a child process; reject a pattern
    that is slow, or that the child has to be killed for."""
    slow = "pattern is too slow on long repetitive text (likely catastrophic backtracking)"
    payload = json.dumps({"pattern": pattern, "bits": _flag_bits(flags), "probes": _ADVERSARIAL})
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _SPEED_SCRIPT],
            input=payload, capture_output=True, text=True, timeout=SPEED_CHECK_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return slow
    except OSError:  # pragma: no cover - cannot spawn; the structural check stood
        return None
    try:
        worst = float(proc.stdout.strip())
    except ValueError:
        return None
    return slow if worst > MAX_SCAN_SECONDS else None


def validate_pattern(
    type_name: Any, pattern: Any, flags: Any = "", *, check_speed: bool = True,
) -> str | None:
    """``None`` when the pattern is usable, else a one-line reason.
    ``check_speed=False`` skips the subprocess timing guard — for re-reading
    patterns that were validated when they were written."""
    name = normalise_type(type_name)
    if not name:
        return "type is required"
    if not isinstance(pattern, str) or not pattern.strip():
        return "pattern is required"
    if len(pattern) > MAX_PATTERN_CHARS:
        return f"pattern is longer than {MAX_PATTERN_CHARS} characters"
    norm_flags = normalise_flags(flags)
    try:
        compiled = re.compile(pattern, _flag_bits(norm_flags))
    except re.error as exc:
        return f"pattern does not compile: {exc}"
    if compiled.search("") is not None or any(
        m.end() == m.start() for m in compiled.finditer("ab 12 -_ XY\nz")
    ):
        return "pattern matches the empty string"
    try:
        parsed = _sre_parse.parse(pattern, _flag_bits(norm_flags))
        if parsed.getwidth()[0] == 0:
            # Minimum width 0: some input yields a zero-length match (a bare
            # lookaround, ``x?``, ``\b``) even if the samples above didn't.
            return "pattern matches the empty string"
        if _has_nested_unbounded(list(parsed)):
            return "pattern nests one unbounded repeat inside another (e.g. (a+)+), which can take exponential time; rewrite it without the nesting"
    except Exception:  # noqa: BLE001 - parser internals vary; the timing check still runs
        pass
    if check_speed:
        return _speed_check(pattern, norm_flags)
    return None


def normalise_patterns(raw: Any, *, check_speed: bool = False) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """``(valid, errors)``. ``valid`` = ``[{type, pattern, description, flags}]``
    (duplicates of the same type+pattern+flags dropped); ``errors`` =
    ``[{index, type, pattern, error}]`` for the rejected entries."""
    valid: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return valid, errors
    seen: set[tuple[str, str, str]] = set()
    for index, item in enumerate(raw[:MAX_PATTERNS]):
        if not isinstance(item, dict):
            errors.append({"index": index, "type": "", "pattern": "", "error": "entry is not an object"})
            continue
        name = normalise_type(item.get("type"))
        pattern = item.get("pattern") if isinstance(item.get("pattern"), str) else ""
        flags = normalise_flags(item.get("flags"))
        error = validate_pattern(name, pattern, flags, check_speed=check_speed)
        if error:
            errors.append({"index": index, "type": name, "pattern": pattern, "error": error})
            continue
        key = (name, pattern, flags)
        if key in seen:
            continue
        seen.add(key)
        valid.append({
            "type": name,
            "pattern": pattern,
            "description": " ".join(str(item.get("description") or "").split())[:MAX_DESCRIPTION_CHARS],
            "flags": flags,
        })
    return valid, errors


def project_patterns(project: Any) -> list[dict[str, str]]:
    """The project's usable custom patterns (invalid stored entries skipped)."""
    cfg = getattr(project, "runtime_config", None)
    valid, _ = normalise_patterns(cfg.get(PATTERNS_KEY) if isinstance(cfg, dict) else None)
    return valid


async def load_patterns(db: AsyncSession | None, project_id: int) -> list[dict[str, str]]:
    """The project's custom patterns; ``[]`` without a session or project."""
    if db is None:
        return []
    from app.models.project import Project

    project = await db.get(Project, project_id)
    return project_patterns(project) if project is not None else []


@lru_cache(maxsize=512)
def _compile(pattern: str, flags: str) -> re.Pattern[str]:
    return re.compile(pattern, _flag_bits(flags))


def compiled_patterns(patterns: list[dict[str, Any]] | None) -> list[tuple[str, re.Pattern[str]]]:
    """``[(type, compiled)]`` for already-validated patterns; anything that
    fails to compile is skipped (never raises inside a regex pass)."""
    out: list[tuple[str, re.Pattern[str]]] = []
    for item in patterns or []:
        if not isinstance(item, dict):
            continue
        name = normalise_type(item.get("type"))
        pattern = item.get("pattern")
        if not name or not isinstance(pattern, str) or not pattern:
            continue
        try:
            out.append((name, _compile(pattern, normalise_flags(item.get("flags")))))
        except re.error:
            continue
    return out


def sample_project_texts(project_id: int, limit: int = TEST_SAMPLE_TEXTS) -> list[tuple[str, str]]:
    """``(source, text)`` to test a pattern on: the saved synthetic span
    rows (``saved_row``) then the cleaned document passages (``passage``),
    up to ``limit`` in all."""
    from app.config import settings
    from app.services.span_relabel_service import load_document_passages

    out: list[tuple[str, str]] = []
    path = settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if len(out) >= limit:
                    return out
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text") if isinstance(row, dict) else None
                if isinstance(text, str) and text.strip():
                    out.append(("saved_row", text))
    for row in load_document_passages(project_id):
        if len(out) >= limit:
            break
        out.append(("passage", str(row.get("text") or "")))
    return out


def preview_pattern_matches(
    pattern: str, type_name: str, texts: list[str], flags: str = "",
    max_snippets: int = TEST_MAX_SNIPPETS, sources: list[str] | None = None,
) -> dict[str, Any]:
    """Run a candidate pattern over ``texts``: ``{valid, error, type,
    texts_scanned, texts_matched, count, matches: [{text_index, source,
    start, end, match, before, after}]}`` (first ``max_snippets`` matches;
    offsets are into that text)."""
    name = normalise_type(type_name)
    error = validate_pattern(name or "_", pattern, flags)
    result: dict[str, Any] = {
        "valid": error is None, "error": error, "type": name,
        "texts_scanned": len(texts), "texts_matched": 0, "count": 0, "matches": [],
    }
    if error is not None:
        return result
    compiled = _compile(pattern, normalise_flags(flags))
    for index, text in enumerate(texts):
        hit = False
        for m in compiled.finditer(text):
            if m.end() == m.start():
                continue
            hit = True
            result["count"] += 1
            if len(result["matches"]) < max_snippets:
                result["matches"].append({
                    "text_index": index,
                    "source": sources[index] if sources and index < len(sources) else None,
                    "start": m.start(), "end": m.end(), "match": m.group(0),
                    "before": text[max(0, m.start() - _SNIPPET_CONTEXT):m.start()],
                    "after": text[m.end():m.end() + _SNIPPET_CONTEXT],
                })
        if hit:
            result["texts_matched"] += 1
    return result

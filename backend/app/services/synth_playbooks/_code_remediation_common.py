"""Helpers shared by the two `code-remediation` playbooks.

A *remediation row* is one chunk of code, one finding in it, and the fix::

    {"row_kind": "remediation", "code", "language", "path", "start_line", "end_line",
     "finding": {label, start_line, end_line, evidence, attrs},
     "passages": [{"id": "P1", "text", ...}],            # optional
     "remediation_json": {"patch", "rationale", "citations"}}

A *code example* row is a task answered with code that uses an API shown in
retrieved passages (``row_kind: "code_example"``).

Seeds:
  * ``remediation_from_findings``: findings rows (one seed per finding; clean chunks are
    skipped), fix pairs (``row_kind: "patch"``: code before / after), and remediation rows,
    from the dataset named by ``params.findings_dataset_id`` (ledger-gated, never an answer
    key) and from the GOLD_DEV split. GOLD_TEST is never a seed.
  * ``code_examples``: the project's code-document chunks (ledger-gated by the orchestrator).
The orchestrator (``synth_playbook_service._prepare_remediation_context``) fills
``ctx['remediation_seeds']`` (with passages attached) for the first.
"""

from __future__ import annotations

from typing import Any

from .base import PlaybookContext
from ._code_findings_common import (  # reuse the G4 helpers: same line layout, same size caps
    MAX_CODE_CHARS,
    MAX_SEED_CHARS,
    MAX_SEED_LINES,
    clean_code,
    findings_of,
    is_int,
    norm_ws,
    numbered,
)

RECIPE_ID = "code-remediation"
MARKER_PREFIX = "[brewslm:code-remediation:"
MAX_PATCH_CHARS = 12000
MAX_TASKS_PER_PROMPT = 5
MAX_REMEDIATION_SEEDS = 200
MAX_PASSAGES = 10

__all__ = [
    "RECIPE_ID", "MAX_CODE_CHARS", "MAX_PATCH_CHARS", "MAX_TASKS_PER_PROMPT", "MAX_REMEDIATION_SEEDS",
    "marker", "attr", "norm_ws", "is_int", "numbered", "clean_code", "findings_of",
    "normalise_passages", "build_seeds_from_rows", "pick_window", "task_index", "passage_ids",
]


def marker(mode: str) -> str:
    return f"{MARKER_PREFIX}{mode}]"


def attr(value: Any) -> str:
    """A value safe inside a double-quoted tag attribute."""
    return str(value if value is not None else "").replace('"', "'").replace("\n", " ")[:200]


def normalise_passages(passages: Any, *, limit: int = MAX_PASSAGES) -> list[dict[str, Any]]:
    """Passages re-identified ``P1..Pk`` in order; every other field kept."""
    out: list[dict[str, Any]] = []
    if not isinstance(passages, list):
        return out
    for item in passages:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            continue
        passage = {k: v for k, v in item.items() if k != "id"}
        passage = {"id": f"P{len(out) + 1}", **passage}
        for key in ("source", "path", "start_line", "end_line", "symbol", "passage_id"):
            passage.setdefault(key, None)
        out.append(passage)
        if len(out) >= limit:
            break
    return out


def passage_ids(passages: Any) -> set[str]:
    return {p["id"] for p in passages or [] if isinstance(p, dict) and isinstance(p.get("id"), str)}


def _seed(code: Any, language: Any, path: Any, start_line: Any, finding: dict[str, Any], origin: str,
          passages: Any = None, fix_after: str | None = None) -> dict[str, Any] | None:
    if not isinstance(code, str):
        return None
    code = clean_code(code)
    if not code.strip() or len(code) > MAX_SEED_CHARS or code.count("\n") + 1 > MAX_SEED_LINES:
        return None
    start = start_line if is_int(start_line) and start_line >= 1 else 1
    return {
        "code": code,
        "language": language if isinstance(language, str) and language else None,
        "path": path if isinstance(path, str) and path else None,
        "start_line": start,
        "end_line": start + code.count("\n"),
        "finding": finding,
        "passages": normalise_passages(passages) or None,
        "origin": origin,
        "fix_after": clean_code(fix_after) if isinstance(fix_after, str) else None,
    }


def _finding_from(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("label"), str) or not raw["label"].strip():
        return None
    if not (is_int(raw.get("start_line")) and is_int(raw.get("end_line"))) or raw["start_line"] > raw["end_line"]:
        return None
    attrs = raw.get("attrs") if isinstance(raw.get("attrs"), dict) else {}
    return {
        "label": raw["label"].strip(), "start_line": raw["start_line"], "end_line": raw["end_line"],
        "evidence": str(raw.get("evidence") or "").strip(), "attrs": attrs,
    }


def _fix_pair_finding(row: dict[str, Any], before: str, after: str, start: int) -> dict[str, Any] | None:
    """The finding of a fix pair: its first label over the lines the fix changed."""
    from app.services.dataset_import.mappers.fix_pair import changed_before_lines

    labels = row.get("labels") if isinstance(row.get("labels"), list) else row.get("labels_raw")
    label = next((x.strip() for x in labels or [] if isinstance(x, str) and x.strip()), None)
    span = changed_before_lines(before, after)
    if label is None or span is None:
        return None
    lines = before.split("\n")
    first, last = span
    return {
        "label": label, "start_line": start + first - 1, "end_line": start + last - 1,
        "evidence": "\n".join(lines[first - 1 : last]).strip(), "attrs": {},
    }


def build_seeds_from_rows(
    rows: list[Any], origin: str, rejected: list[tuple[str, str]] | None = None
) -> list[dict[str, Any]]:
    """Remediation seeds from findings / fix-pair (patch) / remediation rows, one per finding.

    Clean findings rows (``findings == []``) and rows of any other shape yield nothing. A fix pair
    that cannot become a seed is reported on ``rejected`` as ``(reason, detail)`` --
    ``no_label`` (no label on the row) or ``no_changed_lines`` (the fix changes nothing)."""
    seeds: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("code"), str):
            continue
        start = row.get("start_line")
        if row.get("row_kind") == "patch" or (isinstance(row.get("fixed_code"), str) and row.get("row_kind") != "findings"):
            after = row.get("fixed_code")
            if not isinstance(after, str):
                continue
            before = clean_code(row["code"])
            start_n = start if is_int(start) and start >= 1 else 1
            finding = _fix_pair_finding(row, before, clean_code(after), start_n)
            seed = _seed(before, row.get("language"), row.get("path"), start, finding, origin, fix_after=after) if finding else None
            if seed:
                seeds.append(seed)
            elif finding is None and rejected is not None:
                from app.services.dataset_import.mappers.fix_pair import changed_before_lines

                labels = row.get("labels") if isinstance(row.get("labels"), list) else row.get("labels_raw")
                has_label = any(isinstance(x, str) and x.strip() for x in labels or [])
                if not has_label:
                    rejected.append(("no_label", "fix pair has no label"))
                elif changed_before_lines(before, clean_code(after)) is None:
                    rejected.append(("no_changed_lines", "fix pair changes no lines of the code"))
            continue
        if isinstance(row.get("finding"), dict):  # remediation row: its code + finding (+ passages)
            finding = _finding_from(row["finding"])
            seed = _seed(row["code"], row.get("language"), row.get("path"), start, finding, origin, row.get("passages")) if finding else None
            if seed:
                seeds.append(seed)
            continue
        found = findings_of(row)
        for raw in found or []:
            finding = _finding_from(raw)
            seed = _seed(row["code"], row.get("language"), row.get("path"), start, finding, origin, row.get("passages")) if finding else None
            if seed:
                seeds.append(seed)
    return seeds


def pick_window(ctx: PlaybookContext, pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """A window of ``count`` seeds rotating with the chunk index and the run nonce (same rule as
    the code-findings playbooks, so build_prompt and validate agree)."""
    count = max(1, min(count, len(pool)))
    start = (int(ctx.get("seed_nonce") or 0) + int(ctx.get("chunk_index") or 0) * count) % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(count)]


def task_index(row: Any, n: int, key: str = "task") -> int | None:
    """The 1-based task / seed a teacher row answers (``row[key]``), or None when invalid."""
    if not isinstance(row, dict):
        return None
    index = row.get(key)
    if isinstance(index, str) and index.strip().isdigit():
        index = int(index.strip())
    return index if is_int(index) and 1 <= index <= n else None

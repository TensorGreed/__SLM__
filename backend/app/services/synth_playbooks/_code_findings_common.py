"""Helpers shared by the three `code-findings` playbooks.

A *findings row* is one piece of code plus the labelled findings in it
(see ``dataset_import/mappers/_findings_common.py``). Lines are absolute file
line numbers, inclusive; ``findings: []`` means the chunk is clean.

The seed pool the playbooks draw from:
  * answer-key (gold) rows with ``findings == []``  -> clean seeds
  * answer-key rows with findings                   -> positive seeds
  * code chunks of the project's code documents (git repository / website
    sources, ``ctx['document_chunks']``)            -> *assumed clean* seeds
    (they are unlabelled, so the prompt only ever calls them "assumed clean")
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import PlaybookContext, record_reject, sample_gold_rows

RECIPE_ID = "code-findings"
MARKER_PREFIX = "[brewslm:code-findings:"
MAX_CODE_CHARS = 20000
MAX_SEED_CHARS = 6000
MAX_SEED_LINES = 200
MAX_CHUNKS_PER_PROMPT = 5

_WS = re.compile(r"\s+")


def marker(mode: str) -> str:
    return f"{MARKER_PREFIX}{mode}]"


def norm_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def numbered(code: str, start_line: int) -> str:
    """Each line prefixed by its absolute line number (same shape as the shared prompt builder)."""
    return "\n".join(f"{start_line + i:>5} | {line}" for i, line in enumerate(code.split("\n")))


def clean_code(code: str) -> str:
    """Drop only trailing newlines: leading indentation is part of the code."""
    return code.replace("\r\n", "\n").rstrip("\n")


def findings_of(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The row's findings list, or None when it carries no readable findings payload."""
    raw: Any = row.get("findings_json")
    if raw is None:
        raw = row.get("findings")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if isinstance(raw, dict):
        raw = raw.get("findings")
    if not isinstance(raw, list):
        return None
    return [f for f in raw if isinstance(f, dict)] if all(isinstance(f, dict) for f in raw) else None


def _seed(code: str, language: Any, path: Any, start_line: Any, findings: list, origin: str) -> dict[str, Any] | None:
    code = clean_code(code)
    if not code.strip() or len(code) > MAX_SEED_CHARS or code.count("\n") + 1 > MAX_SEED_LINES:
        return None
    return {
        "code": code,
        "language": language if isinstance(language, str) and language else None,
        "path": path if isinstance(path, str) and path else None,
        "start_line": start_line if is_int(start_line) and start_line >= 1 else 1,
        "findings": findings,
        "origin": origin,
    }


def seed_pools(ctx: PlaybookContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(clean_seeds, positive_seeds)`` from the answer key and the code documents."""
    clean: list[dict[str, Any]] = []
    positive: list[dict[str, Any]] = []
    for row in ctx.get("gold_rows") or []:
        if not isinstance(row, dict) or not isinstance(row.get("code"), str):
            continue
        findings = findings_of(row)
        if findings is None:
            continue
        seed = _seed(row["code"], row.get("language"), row.get("path"), row.get("start_line"), findings, "answer_key")
        if seed is None:
            continue
        (positive if findings else clean).append(seed)
    for chunk in ctx.get("document_chunks") or []:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            continue
        seed = _seed(chunk["text"], chunk.get("language"), chunk.get("path"), chunk.get("start_line"), [], "document")
        if seed is not None:
            clean.append(seed)
    return clean, positive


def pick(pool: list[dict[str, Any]], count: int, ctx: PlaybookContext | None = None) -> list[dict[str, Any]]:
    """A window of ``count`` seeds that rotates with the chunk index and the run nonce.

    Deterministic for a given context (build_prompt and validate must agree on it), but
    successive chunks of one run, and successive runs, start at different places in the pool.
    """
    count = max(1, count)
    if len(pool) <= count:
        return list(pool)
    ctx = ctx or {}
    start = (int(ctx.get("seed_nonce") or 0) + int(ctx.get("chunk_index") or 0) * count) % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(count)]


def resolve_seeds(ctx: PlaybookContext, kind: str) -> list[dict[str, Any]]:
    """The seeds for this (chunk) context, chosen once and stored in ``ctx['code_seeds']``.

    ``kind`` is ``"clean"`` or ``"positive"``. build_prompt stores the choice; validate reads it
    back, so the rows the teacher answered are checked against the chunks it was shown.
    """
    key = f"{kind}:{ctx.get('target_count')}:{ctx.get('chunk_index') or 0}:{ctx.get('seed_nonce') or 0}"
    cached = ctx.get("code_seeds")
    if isinstance(cached, dict) and cached.get("key") == key:
        return cached["seeds"]
    clean, positive = seed_pools(ctx)
    pool = clean if kind == "clean" else positive
    if not pool:
        raise no_seed_error(kind)
    seeds = pick(pool, chunk_count_for(ctx, len(pool)), ctx)
    ctx["code_seeds"] = {"key": key, "seeds": seeds}
    return seeds


def chunk_count_for(ctx: PlaybookContext, pool_size: int) -> int:
    return max(1, min(MAX_CHUNKS_PER_PROMPT, pool_size, int(ctx.get("target_count") or 1)))


def render_chunks(seeds: list[dict[str, Any]], *, assumed_clean: bool) -> str:
    blocks = []
    for index, seed in enumerate(seeds, start=1):
        n_lines = seed["code"].count("\n") + 1
        attrs = (
            f'index="{index}" language="{seed["language"] or "unknown"}" path="{seed["path"] or "unknown"}" '
            f'start_line="{seed["start_line"]}" end_line="{seed["start_line"] + n_lines - 1}"'
        )
        note = ' note="assumed clean"' if assumed_clean and seed["origin"] == "document" else ""
        blocks.append(f"<chunk {attrs}{note}>\n{numbered(seed['code'], seed['start_line'])}\n</chunk>")
    return "\n\n".join(blocks)


def seed_for(row: dict[str, Any], seeds: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The seed a teacher row refers to (its 1-based ``chunk``; absent -> the first)."""
    index = row.get("chunk", 1)
    if isinstance(index, str) and index.strip().isdigit():
        index = int(index.strip())
    if not is_int(index) or not 1 <= index <= len(seeds):
        return None
    return seeds[index - 1]


def check_code(ctx: PlaybookContext, row: Any, seed: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Shared first checks. Returns ``(code, None)`` or ``(None, reason)`` after recording the reject."""
    if not isinstance(row, dict) or seed is None or not isinstance(row.get("code"), str):
        record_reject(ctx, "parse_error", "row is not an object with string `code` and a valid `chunk`", row)
        return None, "parse_error"
    code = clean_code(row["code"])
    if not code.strip():
        record_reject(ctx, "empty_code", "returned code is empty", row)
        return None, "empty_code"
    if len(code) > MAX_CODE_CHARS:
        record_reject(ctx, "code_too_long", f"{len(code)} characters (max {MAX_CODE_CHARS})", {"code": code[:200]})
        return None, "code_too_long"
    return code, None


def findings_row(
    seed: dict[str, Any],
    code: str,
    findings: list[dict[str, Any]],
    ctx: PlaybookContext,
) -> dict[str, Any]:
    taxonomy = ctx.get("taxonomy")
    return {
        "row_kind": "findings",
        "code": code,
        "language": seed["language"],
        "path": seed["path"],
        "start_line": seed["start_line"],
        "end_line": seed["start_line"] + code.count("\n"),
        "findings_json": {"findings": findings},
        "labels_raw": [f["label"] for f in findings],
        "taxonomy_version": getattr(taxonomy, "version", None) if taxonomy is not None else None,
    }


def no_seed_error(kind: str) -> ValueError:
    if kind == "clean":
        return ValueError(
            "No clean code chunks to seed from. Seeded defects start from code with no findings, which "
            "comes from two sources: (1) the answer key's dev set: in Data > Answer key, import a findings "
            "dataset as the dev set and keep some rows with `findings: []` (the test set is never used as a "
            "seed, and imported training rows are not used either); (2) code documents: in Data > Documents, "
            "add a git repository or website source so its code chunks can be used as assumed-clean seeds "
            "(sources marked evaluation-only or retrieval-only are skipped)."
        )
    return ValueError(
        "No labelled code chunks to build negatives from. Hard negatives are derived from answer-key dev "
        "rows that carry at least one finding: in Data > Answer key, import a labelled findings dataset as "
        "the dev set (the test set is never used as a seed)."
    )


def json_lines_instructions(count: int, shape: str) -> str:
    return (
        f"Output exactly {count} JSON lines, one object per line, no preamble, no markdown code fences. "
        f"Each line has the shape:\n{shape}"
    )


def taxonomy_entry_block(ctx: PlaybookContext, label_id: str) -> str:
    taxonomy = ctx.get("taxonomy")
    entry = taxonomy.entries.get(label_id) if taxonomy is not None else None
    title = (entry or {}).get("title") or label_id
    default_attrs = (entry or {}).get("attrs") if isinstance((entry or {}).get("attrs"), dict) else {}
    params = ctx.get("params") or {}
    attrs = params.get("attrs") if isinstance(params.get("attrs"), dict) else default_attrs
    lines = [f"Target label id: {label_id}", f"Label title: {title}"]
    if attrs:
        lines.append("Label attributes to fill in `attrs`: " + json.dumps(attrs, ensure_ascii=False, sort_keys=True)[:600])
    else:
        lines.append("Label attributes: none (use an empty `attrs` object).")
    return "\n".join(lines)


def context_block(ctx: PlaybookContext) -> str:
    values = ctx.get("context_values") or []
    if not values:
        return ""
    column = ctx.get("context_column") or "field"
    shown = "\n".join(f"  - {str(v)[:80]}" for v in values[:40])
    return (
        f"\nData-carrying context. These are real values of the `{column}` field the code handles; "
        f"make the defect involve some of these field names or values where it fits naturally:\n{shown}\n"
    )

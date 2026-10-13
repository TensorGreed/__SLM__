"""Pin decisions: what users chose when they compared retrievals.

When the playground shows two answers to one question — retrieval A vs
retrieval B — and the user pins one, that choice is recorded here
(``data/projects/<id>/auto_rag/pin_decisions.jsonl``, append-only, no new
table). The retrieval sweep reads the tally: when two arms are within row
noise of each other on the judge, the one users chose head-to-head more
often wins the tie — users' judgements on their own questions break ties
the judge can't. A pin always overrides the sweep regardless.

Honest limits: a tally of a few clicks is a preference, not a measurement
(``MIN_CHOICES_FOR_PREFERENCE``); it is only ever a tie-break.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.retrieval_reranker import normalize_reranker

DECISIONS_FILE = "pin_decisions.jsonl"
MAX_EXCERPT_CHARS = 400
# A retrieval needs this many head-to-head wins before the sweep treats the
# preference as evidence.
MIN_CHOICES_FOR_PREFERENCE = 2


def retrieval_key(retrieval: dict[str, Any] | None) -> str:
    """``"3|cross-encoder/…||"`` / ``"5|||"`` — one key per
    (k, reranker, pool, chunk_size); a setting without pool / chunk keeps the
    older two-field prefix so recorded decisions still match."""
    from app.services.auto_rag_service import normalize_retrieval

    if not retrieval:
        return ""
    r = normalize_retrieval(retrieval)
    key = f"{r['k']}|{r['reranker'] or ''}"
    if r["pool"] or r["chunk_size"]:
        key += f"|{r['pool'] or ''}|{r['chunk_size'] or ''}"
    return key


def _path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "auto_rag" / DECISIONS_FILE


def _clean_retrieval(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not raw.get("k"):
        return None
    try:
        k = max(1, min(10, int(raw["k"])))
    except (TypeError, ValueError):
        return None
    return {"k": k, "reranker": normalize_reranker(raw.get("reranker"))}


def record_decision(
    project_id: int,
    *,
    question: str,
    candidates: list[dict[str, Any]],
    pinned: dict[str, Any],
    source: str = "playground_compare",
) -> dict[str, Any]:
    """Append one decision. ``candidates`` are the retrievals that were shown
    (each ``{retrieval: {k, reranker}, reply_excerpt?, latency_ms?, passages?}``);
    ``pinned`` is the one the user chose (must be among them)."""
    cleaned = []
    for cand in candidates:
        retrieval = _clean_retrieval((cand or {}).get("retrieval"))
        if retrieval is None:
            continue
        cleaned.append({
            "retrieval": retrieval,
            "reply_excerpt": str((cand or {}).get("reply_excerpt") or "")[:MAX_EXCERPT_CHARS],
            "latency_ms": (cand or {}).get("latency_ms"),
            "passages": (cand or {}).get("passages"),
        })
    chosen = _clean_retrieval(pinned)
    if chosen is None:
        raise ValueError("pinned retrieval needs k")
    if len(cleaned) < 2:
        raise ValueError("a decision needs at least two candidates")
    if retrieval_key(chosen) not in {retrieval_key(c["retrieval"]) for c in cleaned}:
        raise ValueError("the pinned retrieval must be one of the candidates")
    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "question": str(question or "")[:1000],
        "candidates": cleaned,
        "pinned": chosen,
    }
    path = _path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def list_decisions(project_id: int, *, limit: int = 200) -> list[dict[str, Any]]:
    path = _path(project_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def preference_tally(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per retrieval key: how often it was offered head-to-head and how
    often it was chosen; ``share`` = chosen / offered."""
    tally: dict[str, dict[str, Any]] = {}
    for entry in decisions:
        chosen_key = retrieval_key(entry.get("pinned"))
        for cand in entry.get("candidates") or []:
            key = retrieval_key(cand.get("retrieval"))
            if not key:
                continue
            slot = tally.setdefault(key, {"retrieval": cand.get("retrieval"), "offered": 0, "chosen": 0})
            slot["offered"] += 1
            if key == chosen_key:
                slot["chosen"] += 1
    for slot in tally.values():
        slot["share"] = round(slot["chosen"] / slot["offered"], 3) if slot["offered"] else 0.0
    return tally


def project_preference(project_id: int) -> dict[str, dict[str, Any]]:
    return preference_tally(list_decisions(project_id))


def preferred_over(tally: dict[str, dict[str, Any]], candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """Users preferred ``candidate`` to ``incumbent``: chosen at least
    ``MIN_CHOICES_FOR_PREFERENCE`` times, with a higher share than the
    incumbent's (an incumbent never offered counts as share 0)."""
    cand = tally.get(retrieval_key(candidate)) or {}
    inc = tally.get(retrieval_key(incumbent)) or {}
    if int(cand.get("chosen") or 0) < MIN_CHOICES_FOR_PREFERENCE:
        return False
    return float(cand.get("share") or 0.0) > float(inc.get("share") or 0.0)

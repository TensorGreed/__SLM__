"""Project cloud-LLM spend tally.

Sums what the project's cloud-LLM calls cost over a window, from two
sources that never overlap (``llm_usage``):

* the ``jobs`` table — every Job whose ``result`` carries ``usage`` (or
  ``llm_usage``): relabels, synthesis, seed documents, the documents → Q&A
  flow, lift checks with a cloud judge, retrieval comparisons…, failed and
  cancelled Jobs included (their tokens were spent);
* ``data/projects/<id>/llm_spend.jsonl`` — operations that are not Jobs (LLM
  answer key, plan-refinement strategy pass, probe judge in a synchronous
  eval, distillation capture).

Grouped by model and by operation kind. Dollar figures are the list-price
estimates stored when each operation ran (``llm_pricing``), never an invoice.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import Job
from app.services import llm_usage

DEFAULT_DAYS = 30
MAX_DAYS = 366
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens", "total_tokens")


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _parse_at(raw: Any) -> datetime | None:
    try:
        return _aware(datetime.fromisoformat(str(raw)))
    except (TypeError, ValueError):
        return None


def read_ledger(project_id: int, since: datetime) -> list[dict[str, Any]]:
    """Ledger entries at or after ``since`` (unreadable lines skipped)."""
    path = llm_usage.ledger_path(project_id)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        at = _parse_at(entry.get("at")) if isinstance(entry, dict) else None
        if at is None or at < since or not llm_usage.is_usage_list(entry.get("usage")):
            continue
        out.append(entry)
    return out


def tally(operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure roll-up of ``[{kind, at, usage: [estimate_cost dicts]}]``."""
    totals: dict[str, Any] = {"operations": 0, "calls": 0, **{f: 0 for f in _TOKEN_FIELDS}, "cost_usd": 0.0, "unpriced_tokens": 0}
    by_model: dict[str, dict[str, Any]] = {}
    by_kind: dict[str, dict[str, Any]] = {}
    for op in operations:
        entries = [u for u in op.get("usage") or [] if isinstance(u, dict)]
        if not entries:
            continue
        totals["operations"] += 1
        kind = str(op.get("kind") or "other")
        kind_row = by_kind.setdefault(kind, {"kind": kind, "operations": 0, "calls": 0, "total_tokens": 0, "cost_usd": 0.0})
        kind_row["operations"] += 1
        for u in entries:
            model = str(u.get("model") or u.get("backend") or "unknown")
            calls = int(u.get("calls") or 0)
            tokens = int(u.get("total_tokens") or 0)
            usd = u.get("usd")
            row = by_model.setdefault(model, {"model": model, "calls": 0, **{f: 0 for f in _TOKEN_FIELDS}, "cost_usd": 0.0, "priced": True})
            row["calls"] += calls
            totals["calls"] += calls
            kind_row["calls"] += calls
            kind_row["total_tokens"] += tokens
            for f in _TOKEN_FIELDS:
                n = int(u.get(f) or 0)
                row[f] += n
                totals[f] += n
            if isinstance(usd, (int, float)):
                row["cost_usd"] += float(usd)
                totals["cost_usd"] += float(usd)
                kind_row["cost_usd"] += float(usd)
            else:
                row["priced"] = False
                totals["unpriced_tokens"] += tokens
    for row in (*by_model.values(), *by_kind.values()):
        row["cost_usd"] = round(row["cost_usd"], 4)
    totals["cost_usd"] = round(totals["cost_usd"], 4)
    return {
        "totals": totals,
        "by_model": sorted(by_model.values(), key=lambda r: (-r["cost_usd"], -r["total_tokens"])),
        "by_kind": sorted(by_kind.values(), key=lambda r: (-r["cost_usd"], -r["total_tokens"])),
    }


async def project_llm_spend(db: AsyncSession, project_id: int, days: int = DEFAULT_DAYS) -> dict[str, Any]:
    days = max(1, min(MAX_DAYS, int(days)))
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    rows = (
        await db.execute(
            select(Job.kind, Job.result, Job.completed_at, Job.queued_at).where(
                Job.project_id == project_id,
                Job.result.is_not(None),
                or_(Job.completed_at >= since, Job.completed_at.is_(None) & (Job.queued_at >= since)),
            )
        )
    ).all()
    operations: list[dict[str, Any]] = []
    for kind, result, completed_at, queued_at in rows:
        usage = llm_usage.result_usage(result)
        if not usage:
            continue
        at = _aware(completed_at) or _aware(queued_at)
        if at is not None and at < since:
            continue
        operations.append({"kind": kind, "at": at.isoformat() if at else None, "usage": usage, "source": "job"})
    for entry in read_ledger(project_id, since):
        operations.append({"kind": entry.get("kind"), "at": entry.get("at"), "usage": entry["usage"], "source": "ledger"})
    rolled = tally(operations)
    return {
        "project_id": project_id,
        "days": days,
        "since": since.isoformat(),
        **rolled,
        "basis": "list-price estimate",
        "note": (
            "Estimated from each call's token counts at list prices (LLM_PRICES_JSON overrides); "
            "the provider's invoice is the truth. Local models (Ollama) cost nothing and are not counted."
        ),
    }

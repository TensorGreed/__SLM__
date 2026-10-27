"""Hard API budget for metered cloud-LLM calls.

Money model:

* Metered API credits (Anthropic / OpenAI / DeepSeek via ``cloud_llm_service``,
  teacher endpoints on non-local hosts): a HARD cap per calendar month (UTC) —
  account-wide (``settings.LLM_MONTHLY_BUDGET_USD``, all projects) and per
  project (``runtime_config["llm_budget_usd"]["monthly"]``).
* Subscription CLI (the ``command`` synth backend): recorded at $0, never
  counted here. Local models (Ollama, vLLM, LAN endpoints): not counted.

How it works: an in-process month ledger (global + per project) is seeded
lazily from the DB (``llm_spend_service`` tally — Jobs' stored usage plus the
non-Job ledger file) and refreshed when older than ``REFRESH_SECONDS``; every
priced ``llm_usage.record`` also increments it, so concurrent Jobs see each
other's spend before it reaches the DB. A refresh never lowers the ledger
within the month (a running Job's spend reaches the DB only when it ends).

Known ledger gaps (a cap is a guard rail, not an invoice): spend of a Job still
running when the process restarts is lost until the Job ends (its usage is
stored at the end); calls made outside any project context count against the
account-wide cap only; N concurrent calls started before the first finishes
can each overshoot, together by up to N calls' cost; a model with no known
price is counted at the highest listed price.
``check_budget`` runs before each metered request and raises
``LlmBudgetExceeded`` once a cap is reached: ONE call may overshoot a cap by
its own cost (the spend is only known after the call).

``LlmBudgetExceeded`` deliberately is NOT a ``SynthBackendError`` — the
per-call ``except SynthBackendError`` handlers in the flows must not swallow
it; it ends the Job (message: "API budget reached …") or becomes HTTP 402
``llm_budget_exceeded`` on a request.

The project context (``current_project``) is set by the Job wrapper and
``llm_usage.metered_operation`` so a check inside a Job knows which project
cap applies; it also carries the per-project ``synth_command`` overrides.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from app.config import settings

_LOG = logging.getLogger(__name__)

REFRESH_SECONDS = 60.0


class LlmBudgetExceeded(RuntimeError):
    """A metered API call was refused because a monthly cap is reached."""

    def __init__(self, scope: str, monthly: float, spent_usd: float, project_id: int | None = None) -> None:
        self.scope = scope
        self.monthly = float(monthly)
        self.spent_usd = round(float(spent_usd), 4)
        self.project_id = project_id
        super().__init__(
            f"API budget reached ({scope}: ${self.spent_usd:.2f} of ${self.monthly:.2f} this month) "
            "— switch to a local or subscription backend"
        )

    def metadata(self) -> dict[str, Any]:
        return {"scope": self.scope, "monthly": self.monthly, "spent_usd": self.spent_usd, "project_id": self.project_id}


@dataclass
class ProjectContext:
    project_id: int | None = None
    # ``runtime_config["synth_command"]`` — the per-project command-backend knobs.
    synth_command: dict[str, Any] = field(default_factory=dict)


current_project: contextvars.ContextVar[ProjectContext | None] = contextvars.ContextVar("llm_current_project", default=None)


@contextlib.contextmanager
def project_context(project_id: int | None, synth_command: dict[str, Any] | None = None) -> Iterator[ProjectContext]:
    ctx = ProjectContext(project_id=project_id, synth_command=dict(synth_command or {}))
    token = current_project.set(ctx)
    try:
        yield ctx
    finally:
        current_project.reset(token)


def current_project_id() -> int | None:
    ctx = current_project.get()
    return ctx.project_id if ctx is not None else None


# ── the in-process month ledger ──────────────────────────────────────

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"month": None, "global": 0.0, "projects": {}, "caps": {}, "seeded": None}


def _month_key(now: datetime | None = None) -> str:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def _roll_month_locked() -> None:
    key = _month_key()
    if _STATE["month"] != key:
        _STATE.update({"month": key, "global": 0.0, "projects": {}, "caps": {}, "seeded": None})


def reset() -> None:
    """Forget the ledger (tests; a changed project budget)."""
    with _LOCK:
        _STATE.update({"month": None, "global": 0.0, "projects": {}, "caps": {}, "seeded": None})


def invalidate() -> None:
    """Re-read caps and spend from the DB on the next check."""
    with _LOCK:
        _STATE["seeded"] = None


def seed_for_tests(*, global_spent: float = 0.0, projects: dict[int, float] | None = None, caps: dict[int, float] | None = None) -> None:
    """Set the ledger directly and mark it fresh (no DB read)."""
    with _LOCK:
        _STATE.update({
            "month": _month_key(), "global": float(global_spent),
            "projects": {int(k): float(v) for k, v in (projects or {}).items()},
            "caps": {int(k): float(v) for k, v in (caps or {}).items()},
            "seeded": time.monotonic() + 10 * 365 * 86400,
        })


def snapshot() -> dict[str, Any]:
    with _LOCK:
        _roll_month_locked()
        return {"global": _STATE["global"], "projects": dict(_STATE["projects"]), "caps": dict(_STATE["caps"])}


def note_usage(provider: str, model: str, counts: dict[str, int], *, batch: bool = False) -> None:
    """Add one metered call to the ledger. Subscription calls never count
    (local calls are never recorded). A metered model with NO known price is
    counted at the highest price in the table (``llm_pricing.fallback_cost``)
    so an unpriced model cannot run past the cap for free. ``batch``: Message
    Batches pricing (half). Never raises."""
    try:
        if str(model or "").startswith("subscription:"):
            return
        from app.services.llm_pricing import estimate_cost, fallback_cost

        usd = estimate_cost(model, counts, batch=batch).get("usd")
        if usd is None:
            usd = fallback_cost(counts, batch=batch)
        if not isinstance(usd, (int, float)) or usd <= 0:
            return
        pid = current_project_id()
        with _LOCK:
            _roll_month_locked()
            _STATE["global"] += float(usd)
            if pid is not None:
                _STATE["projects"][pid] = _STATE["projects"].get(pid, 0.0) + float(usd)
    except Exception:  # noqa: BLE001 — accounting never breaks a call
        _LOG.debug("budget ledger note failed", exc_info=True)


def _stale_locked() -> bool:
    seeded = _STATE["seeded"]
    return seeded is None or (time.monotonic() - seeded) > REFRESH_SECONDS


def _global_cap() -> float | None:
    cap = getattr(settings, "LLM_MONTHLY_BUDGET_USD", None)
    try:
        return float(cap) if cap is not None and float(cap) > 0 else None
    except (TypeError, ValueError):
        return None


# Set while code runs on a worker thread's private event loop: the shared single-connection engine
# must not be used from a foreign loop, so a stale ledger is left as cached there (the caller
# refreshes it on the main loop first, via ``refresh_if_stale``). Enforcement still uses the ledger.
_CACHED_ONLY: contextvars.ContextVar[bool] = contextvars.ContextVar("llm_budget_cached_only", default=False)


@contextlib.contextmanager
def cached_ledger_only() -> Iterator[None]:
    token = _CACHED_ONLY.set(True)
    try:
        yield
    finally:
        _CACHED_ONLY.reset(token)


async def refresh_if_stale() -> None:
    """Refresh the ledger from the DB when it is stale (call on the main loop before off-loop work)."""
    with _LOCK:
        _roll_month_locked()
        stale = _stale_locked()
    if stale:
        await refresh()


async def refresh() -> None:
    """Re-seed the ledger from the DB (month-to-date spend + project caps).
    Spend never goes down within the month. Best-effort."""
    if _CACHED_ONLY.get():
        return
    try:
        from app.database import async_session_factory
        from app.services import llm_spend_service

        async with async_session_factory() as db:
            rows = await llm_spend_service.month_spend_by_project(db)
        with _LOCK:
            _roll_month_locked()
            projects = dict(_STATE["projects"])
            total = 0.0
            caps: dict[int, float] = {}
            for pid, row in rows.items():
                projects[pid] = max(projects.get(pid, 0.0), float(row["spent_usd"]))
                if row.get("cap"):
                    caps[pid] = float(row["cap"])
            total = sum(projects.values())
            _STATE["projects"] = projects
            _STATE["caps"] = caps
            _STATE["global"] = max(float(_STATE["global"]), total)
            _STATE["seeded"] = time.monotonic()
    except Exception:  # noqa: BLE001 — a budget read never blocks a call
        _LOG.debug("budget ledger refresh failed", exc_info=True)
        with _LOCK:
            _STATE["seeded"] = time.monotonic()


def _exceeded(project_id: int | None) -> LlmBudgetExceeded | None:
    cap = _global_cap()
    with _LOCK:
        _roll_month_locked()
        if cap is not None and _STATE["global"] >= cap:
            return LlmBudgetExceeded("global", cap, _STATE["global"], project_id)
        if project_id is not None:
            pcap = _STATE["caps"].get(project_id)
            spent = _STATE["projects"].get(project_id, 0.0)
            if pcap and spent >= pcap:
                return LlmBudgetExceeded("project", pcap, spent, project_id)
    return None


def _skips(provider: str | None, model: str | None) -> bool:
    return str(model or "").startswith("subscription:") or str(provider or "").startswith("subscription")


def check_budget(provider: str | None = None, model: str | None = None) -> None:
    """Raise ``LlmBudgetExceeded`` when the account-wide cap, or the current
    project's cap, is already reached. In-memory only (no DB read); callers
    already know the call is metered (local / subscription calls skip it)."""
    if _skips(provider, model):
        return
    err = _exceeded(current_project_id())
    if err is not None:
        raise err


async def acheck_budget(provider: str | None = None, model: str | None = None) -> None:
    """``check_budget`` after refreshing a stale ledger from the DB — only
    when a cap can apply (the account-wide cap is set, or a project context
    is open), so cap-free calls never touch the DB."""
    if _skips(provider, model):
        return
    pid = current_project_id()
    if _global_cap() is not None or pid is not None:
        with _LOCK:
            _roll_month_locked()
            stale = _stale_locked()
        if stale:
            await refresh()
    check_budget(provider, model)


async def assert_can_start(project_id: int) -> None:
    """Job-start guard for a metered backend: raise ``LlmBudgetExceeded``
    when ``project_id``'s caps (or the account-wide one) are reached."""
    with _LOCK:
        _roll_month_locked()
        stale = _stale_locked()
    if stale:
        await refresh()
    err = _exceeded(int(project_id))
    if err is not None:
        raise err


def global_status(spent: float | None = None) -> dict[str, Any]:
    """``{monthly, spent_usd, status}`` for the account-wide cap (read from the ledger)."""
    cap = _global_cap()
    spent_usd = round(float(snapshot()["global"] if spent is None else spent), 4)
    if cap is None:
        return {"monthly": None, "spent_usd": spent_usd, "status": "ok"}
    status = "over" if spent_usd >= cap else "warning" if spent_usd >= 0.8 * cap else "ok"
    return {"monthly": cap, "spent_usd": spent_usd, "status": status}

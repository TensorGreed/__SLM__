"""Judge resilience: classify judge-call failures, retry transient ones, stop on an outage.

An LLM judge scores eval rows one call at a time. When its provider (or the network) is down, every
call fails, and a loop that swallows each error and moves on spends hours failing (a backend that
retries internally multiplies that). This module gives such a loop three things:

* ``classify_judge_error(exc)`` -> ``"transient"`` (connection / DNS / timeout / 408 / 425 / 429 / 5xx /
  ``SynthBackendTransientError`` -- the network's or the provider's fault, worth retrying later) or
  ``"permanent"`` (the request's: 4xx, a bad key, anything unknown). It walks ``__cause__``, because the
  synth backends wrap provider errors in ``SynthBackendError``.
* ``JudgeCallGuard`` -- one per judge loop (run). ``await guard.call(fn, *args)`` retries a transient
  failure ``JUDGE_TRANSIENT_RETRIES`` times with backoff (none for a backend that retries itself), and
  returns ``GuardOutcome(result, reason, error)``. ``reason`` is None for a verdict, else why the row is
  unjudged: ``unparseable`` (the judge answered, no verdict), ``permanent_error``, ``transient_error``
  (failed after retries), ``judge_unreachable`` (the breaker is open: no call was made).
* A circuit breaker per run (one per guard; nothing carries over from an earlier run, so every new run
  or eval starts closed and its first calls get the normal retries). It opens after
  ``JUDGE_BREAKER_CONSECUTIVE`` rows in a row failed transiently (after their retries), or when more
  than ``JUDGE_BREAKER_ERROR_RATE`` of the last ``JUDGE_BREAKER_WINDOW`` calls (at least
  ``JUDGE_BREAKER_MIN_CALLS``) failed transiently. Then, in mode ``stop`` (default), the run's remaining
  rows are left unjudged without a call; in mode ``wait`` (only for a run with a Job handle on its own
  loop; otherwise ``stop``) the run probes every ``JUDGE_OUTAGE_PROBE_SECONDS`` and resumes when a
  probe answers, else stops once the Job's wait budget is spent. The budget (``max_wait_seconds``,
  capped at ``max_wait_cap_seconds()`` -- half the task queue's visibility timeout) is wall time,
  probe calls included, and ONE per Job: every judge loop of that Job draws on it (a probe already
  under way finishes, so a run can overrun it by at most one call). A permanent failure (the provider
  answered) resets the consecutive count; permanent failures and unparseable replies never count
  towards the breaker. Breaker off: every row is asked once, no guard retries, never stopped (the
  behaviour before the guard; a backend that retries a failed call itself -- the command backend --
  still does, as before; rows without a verdict still get their reason).

``LlmBudgetExceeded`` and cancellation always propagate.

Threads: ``call_sync`` is the synchronous variant for a loop running in a worker thread (the rubric
judge). It never touches the Job handle or the database (both belong to the main event loop), always
stops (no ``wait``), and honours cancellation only through a ``threading.Event`` the caller sets. The
async ``call`` touches the Job handle only when it runs on the loop that created the handle.

Settings: ``backend/app/config.py`` ``JUDGE_*``; per project ``runtime_config["eval_judge"]["resilience"]
= {breaker: bool, mode: "stop" | "wait", max_wait_seconds}`` (``resolve_resilience``).
"""

from __future__ import annotations

import asyncio
import collections
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from app.config import settings

_LOG = logging.getLogger(__name__)

TRANSIENT = "transient"
PERMANENT = "permanent"

# Per-row reasons a judge left a row without a verdict.
REASON_UNREACHABLE = "judge_unreachable"
REASON_TRANSIENT = "transient_error"
REASON_UNPARSEABLE = "unparseable"
REASON_PERMANENT = "permanent_error"
# The rubric judge: no judge could be resolved for the row (never an outage).
REASON_NO_JUDGE = "no_judge"
UNJUDGED_REASONS: tuple[str, ...] = (REASON_UNREACHABLE, REASON_TRANSIENT, REASON_UNPARSEABLE, REASON_PERMANENT, REASON_NO_JUDGE)
# Reasons that mean "the judge was not reachable" (worth judging again later).
OUTAGE_REASONS: frozenset[str] = frozenset({REASON_UNREACHABLE, REASON_TRANSIENT})

MODES: tuple[str, ...] = ("stop", "wait")
# Hard ceiling of the wait budget; ``max_wait_cap_seconds()`` is the effective cap.
_MAX_WAIT_CEILING_SECONDS = 24 * 3600.0
# A lift check refuses to reuse a base result whose judge left more than this share of its rows unjudged
# for an outage (``judge_unreachable`` / ``transient_error``), or that the breaker stopped.
BASE_REUSE_MAX_OUTAGE_SHARE = 0.05
_ERROR_CHARS = 200
# Sleep / probe waits are sliced so a cancel is noticed within this long.
_SLICE_SECONDS = 5.0
_JITTER = 0.2
WAITING_PREFIX = "judge unreachable"


# ── classification ──────────────────────────────────────────────────


def _status_transient(code: int) -> bool:
    # 409 (a conflict) is the request's: not retried.
    return code in (408, 425, 429) or 500 <= code <= 599


def classify_judge_error(exc: BaseException) -> str:
    """``"transient"`` or ``"permanent"`` for a judge call's exception (see module doc)."""
    try:
        import httpx
    except Exception:  # noqa: BLE001 - httpx is a dependency; be safe anyway
        httpx = None  # type: ignore[assignment]
    from app.services.synth_backends.base import SynthBackendTransientError

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SynthBackendTransientError):
            return TRANSIENT
        flag = getattr(current, "transient", None)
        if flag is True:
            return TRANSIENT
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return TRANSIENT if _status_transient(status) else PERMANENT
        if httpx is not None:
            if isinstance(current, httpx.HTTPStatusError):
                return TRANSIENT if _status_transient(int(current.response.status_code)) else PERMANENT
            if isinstance(current, httpx.TransportError):  # connect / read timeout / DNS / network
                return TRANSIENT
        if isinstance(current, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
            return TRANSIENT
        current = current.__cause__
    return PERMANENT


def error_kind(exc: BaseException) -> str:
    """A short label for the breaker block / Job message: ``connection error`` / ``timeout`` / ``HTTP 503`` / class name."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return f"HTTP {status}"
        name = type(current).__name__.lower()
        text = str(current).lower()
        if "timeout" in name or "timed out" in text:
            return "timeout"
        if "connect" in name or "connection" in text or "eai_again" in text or "can't reach" in text:
            return "connection error"
        current = current.__cause__
    return type(exc).__name__


# ── settings ────────────────────────────────────────────────────────


def max_wait_cap_seconds() -> float:
    """The longest wait budget a project may set: half the task queue's visibility timeout
    (``CELERY_VISIBILITY_TIMEOUT_SECONDS``, 12 h by default -> 6 h), so an eval that waits and then
    still has to finish stays well under the point where the queue would hand the task out again."""
    try:
        visibility = float(getattr(settings, "CELERY_VISIBILITY_TIMEOUT_SECONDS", 43200) or 43200)
    except (TypeError, ValueError):
        visibility = 43200.0
    return max(0.0, min(_MAX_WAIT_CEILING_SECONDS, visibility / 2.0))


MAX_WAIT_CAP_SECONDS = max_wait_cap_seconds()


def _num(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(lo, min(hi, number))


@dataclass(frozen=True)
class JudgeResilience:
    breaker: bool = True
    consecutive: int = 3
    window: int = 20
    min_calls: int = 10
    error_rate: float = 0.5
    cooldown_seconds: float = 300.0
    retries: int = 2
    backoff_seconds: float = 10.0
    mode: str = "stop"
    probe_seconds: float = 300.0
    max_wait_seconds: float = 3600.0


def normalize_project_resilience(raw: Any) -> dict[str, Any]:
    """The per-project block as stored: only known keys, clamped (``{}`` for nothing set)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    if isinstance(raw.get("breaker"), bool):
        out["breaker"] = raw["breaker"]
    mode = str(raw.get("mode") or "").strip().lower()
    if mode in MODES:
        out["mode"] = mode
    if raw.get("max_wait_seconds") is not None:
        out["max_wait_seconds"] = _num(raw.get("max_wait_seconds"), 0.0, max_wait_cap_seconds(), 3600.0)
    return out


def resolve_resilience(runtime_config: Any = None, *, retries_internally: bool = False, threaded: bool = False) -> JudgeResilience:
    """Settings from ``config.py`` overlaid with the project's ``eval_judge.resilience`` block.
    ``retries_internally``: the backend retries a failed call itself -> no extra guard retries.
    ``threaded``: the loop runs in a worker thread -> always ``stop`` (no waiting, no Job messages).
    Breaker off -> no guard retries either: every row is asked once, as before the guard."""
    block: dict[str, Any] = {}
    if isinstance(runtime_config, dict) and isinstance(runtime_config.get("eval_judge"), dict):
        block = normalize_project_resilience(runtime_config["eval_judge"].get("resilience"))
    mode = block.get("mode") or str(getattr(settings, "JUDGE_OUTAGE_MODE", "stop") or "stop").strip().lower()
    if mode not in MODES or threaded:
        mode = "stop"
    breaker = block["breaker"] if "breaker" in block else bool(getattr(settings, "JUDGE_BREAKER_ENABLED", True))
    retries = int(_num(getattr(settings, "JUDGE_TRANSIENT_RETRIES", 2), 0, 10, 2))
    return JudgeResilience(
        breaker=bool(breaker),
        consecutive=int(_num(getattr(settings, "JUDGE_BREAKER_CONSECUTIVE", 3), 1, 1000, 3)),
        window=int(_num(getattr(settings, "JUDGE_BREAKER_WINDOW", 20), 2, 10000, 20)),
        min_calls=int(_num(getattr(settings, "JUDGE_BREAKER_MIN_CALLS", 10), 1, 10000, 10)),
        error_rate=_num(getattr(settings, "JUDGE_BREAKER_ERROR_RATE", 0.5), 0.0, 1.0, 0.5),
        cooldown_seconds=_num(getattr(settings, "JUDGE_BREAKER_COOLDOWN_SECONDS", 300.0), 0.0, 86400.0, 300.0),
        retries=0 if (retries_internally or not breaker) else retries,
        backoff_seconds=_num(getattr(settings, "JUDGE_RETRY_BACKOFF_SECONDS", 10.0), 0.0, 3600.0, 10.0),
        mode=mode,
        probe_seconds=_num(getattr(settings, "JUDGE_OUTAGE_PROBE_SECONDS", 300.0), 1.0, 86400.0, 300.0),
        max_wait_seconds=block.get("max_wait_seconds")
        if "max_wait_seconds" in block
        else _num(getattr(settings, "JUDGE_OUTAGE_MAX_WAIT_SECONDS", 3600.0), 0.0, max_wait_cap_seconds(), 3600.0),
    )


# ── the per-run breaker ─────────────────────────────────────────────


class JudgeCircuit:
    """Health of one judge during ONE run (each guard owns its own; nothing is shared between runs, so a
    run started after an outage is never written off by state the outage left). The lock lets the run's
    worker thread and its loop read it safely."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._lock = threading.Lock()
        self.consecutive = 0
        self.window: collections.deque[bool] = collections.deque(maxlen=20)
        self.opened_at: float | None = None
        self.last_error: str | None = None
        self.last_error_kind: str | None = None

    def snapshot_open(self, now: float, cooldown: float) -> bool:
        """True while open and inside the cool-down (an open circuit past it is half-open)."""
        with self._lock:
            return self.opened_at is not None and now - self.opened_at < cooldown

    def record_success(self) -> None:
        with self._lock:
            self.consecutive = 0
            self.window.append(False)
            self.opened_at = None

    def record_permanent(self) -> None:
        # The provider answered (it rejected the request): not an outage, so the run of transient
        # failures is broken -- but not proof of health either (the cool-down stays).
        with self._lock:
            self.consecutive = 0
            self.window.append(False)

    def record_transient(self, cfg: JudgeResilience, now: float, kind: str, error: str) -> bool:
        """Count a transient failure; True when the circuit is (now) open."""
        with self._lock:
            if self.window.maxlen != cfg.window:
                self.window = collections.deque(self.window, maxlen=cfg.window)
            self.consecutive += 1
            self.window.append(True)
            self.last_error_kind = kind
            self.last_error = error
            if not cfg.breaker:
                return False
            failed = sum(1 for x in self.window if x)
            trip = self.consecutive >= cfg.consecutive or (
                len(self.window) >= cfg.min_calls and failed / len(self.window) > cfg.error_rate
            )
            if trip or self.opened_at is not None:
                self.opened_at = now  # (re-)open: a failed probe restarts the cool-down
                return True
            return False


# ── the per-run guard ───────────────────────────────────────────────


@dataclass
class GuardOutcome:
    result: Any = None
    reason: str | None = None  # None = a verdict came back
    error: str | None = None


class JudgeCancelled(Exception):
    """A worker-thread judge loop saw its cancel flag (``call_sync``)."""


def _et_clock(ts: float) -> str:
    try:
        from zoneinfo import ZoneInfo

        return datetime.fromtimestamp(ts, tz=ZoneInfo("America/New_York")).strftime("%H:%M ET")
    except Exception:  # noqa: BLE001 - no tz database: local time
        return datetime.fromtimestamp(ts).strftime("%H:%M")


@dataclass
class JudgeCallGuard:
    """One judge loop's guard (see module doc). Not shared between loops; nor is its circuit."""

    label: str
    resilience: JudgeResilience = field(default_factory=JudgeResilience)
    threaded: bool = False
    cancel_event: threading.Event | None = None
    # Injected in tests.
    sleep: Callable[[float], Any] | None = None
    clock: Callable[[], float] | None = None

    def __post_init__(self) -> None:
        self.circuit = JudgeCircuit(self.label)
        self.tripped = False
        self.tripped_at_row: int | None = None
        self.calls = 0
        self.waited_seconds = 0.0
        self.last_error: str | None = None
        self.last_error_kind: str | None = None
        self._row: int | None = None
        self._handle = None
        if not self.threaded:
            # Only the loop that created the Job handle may use it (its methods open DB sessions on
            # the shared engine, which is bound to that loop).
            from app.services import jobs_service

            handle = jobs_service.current_job_handle.get()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if handle is not None and loop is not None and getattr(handle, "loop", None) is loop:
                self._handle = handle
        self._loop = None
        if not self.threaded:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

    # -- helpers --

    def _now(self) -> float:
        return (self.clock or time.monotonic)()

    def _handle_here(self):
        if self._handle is None or self.threaded:
            return None
        try:
            return self._handle if asyncio.get_running_loop() is self._loop else None
        except RuntimeError:
            return None

    def _backoff(self, attempt: int) -> float:
        base = self.resilience.backoff_seconds * (3 ** attempt)
        return max(0.0, base * (1 + random.uniform(-_JITTER, _JITTER)))

    async def _sleep(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 0:
            step = min(_SLICE_SECONDS, remaining)
            if self.sleep is not None:
                result = self.sleep(step)
                if asyncio.iscoroutine(result):
                    await result
            else:
                await asyncio.sleep(step)
            remaining -= step
            await self._check_cancelled()

    async def _check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise asyncio.CancelledError()
        handle = self._handle_here()
        if handle is not None:
            try:
                cancelled = await handle.check_cancelled()
            except Exception:  # noqa: BLE001 - a status read never breaks the judge
                cancelled = False
            if cancelled:
                raise asyncio.CancelledError()

    async def _message(self, text: str) -> None:
        handle = self._handle_here()
        if handle is None:
            return
        try:
            await handle.set_message(text)
        except Exception:  # noqa: BLE001
            pass

    def _sleep_sync(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 0:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise JudgeCancelled("cancelled")
            step = min(_SLICE_SECONDS, remaining)
            (self.sleep or time.sleep)(step)
            remaining -= step
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise JudgeCancelled("cancelled")

    def _note_failure(self, exc: BaseException) -> tuple[str, str]:
        kind = error_kind(exc)
        text = f"{type(exc).__name__}: {exc}"[:_ERROR_CHARS]
        self.last_error, self.last_error_kind = text, kind
        return kind, text

    def _trip(self) -> None:
        if not self.tripped:
            self.tripped = True
            self.tripped_at_row = self._row

    def _unreachable(self) -> GuardOutcome:
        return GuardOutcome(None, REASON_UNREACHABLE, self.last_error or self.circuit.last_error)

    def _open_now(self) -> bool:
        """Whether this run must not call now (its circuit is open and cooling down)."""
        if not self.resilience.breaker:
            return False
        return self.circuit.snapshot_open(self._now(), self.resilience.cooldown_seconds)

    @property
    def breaker_block(self) -> dict[str, Any] | None:
        if not self.tripped and not self.waited_seconds:
            return None
        return {
            "tripped": self.tripped,
            "tripped_at_row": self.tripped_at_row,
            "consecutive": self.resilience.consecutive,
            "last_error_kind": self.last_error_kind or self.circuit.last_error_kind,
            "last_error": self.last_error or self.circuit.last_error,
            "mode": self.resilience.mode,
            "waited_seconds": round(self.waited_seconds, 1),
        }

    # -- async --

    async def _attempts(self, fn: Callable[..., Any], args: tuple, retries: int) -> GuardOutcome:
        from app.services.llm_budget_guard import LlmBudgetExceeded

        attempt = 0
        while True:
            try:
                self.calls += 1
                result = await fn(*args)
            except (LlmBudgetExceeded, asyncio.CancelledError):
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                kind, text = self._note_failure(exc)
                if classify_judge_error(exc) == PERMANENT:
                    self.circuit.record_permanent()
                    return GuardOutcome(None, REASON_PERMANENT, text)
                if attempt < retries:
                    await self._sleep(self._backoff(attempt))
                    attempt += 1
                    continue
                if self.circuit.record_transient(self.resilience, self._now(), kind, text):
                    if not self._can_wait():
                        self._trip()
                return GuardOutcome(None, REASON_TRANSIENT, text)
            self.circuit.record_success()
            return GuardOutcome(result, None if result is not None else REASON_UNPARSEABLE, None)

    def _can_wait(self) -> bool:
        """Mode ``wait`` applies only to a run with a Job handle on its own loop (cancellable, one budget
        per Job); anything else (a worker thread, a task-queue eval without a Job) stops."""
        return (
            self.resilience.mode == "wait"
            and not self.threaded
            and self.resilience.max_wait_seconds > 0
            and self._handle_here() is not None
        )

    def _wait_spent(self) -> float:
        handle = self._handle_here()
        return float(getattr(handle, "judge_wait_spent_seconds", 0.0) or 0.0) if handle is not None else 0.0

    def _add_wait(self, seconds: float) -> None:
        self.waited_seconds += seconds
        handle = self._handle_here()
        if handle is not None:
            handle.judge_wait_spent_seconds = float(getattr(handle, "judge_wait_spent_seconds", 0.0) or 0.0) + seconds

    async def _wait_for_judge(self, fn: Callable[..., Any], args: tuple) -> GuardOutcome:
        """Mode ``wait``: probe every ``probe_seconds`` until a call answers or the Job's wait budget is
        spent. The budget is wall time (sleeps AND probe calls) shared by every judge loop of the Job."""
        cfg = self.resilience
        previous = getattr(self._handle_here(), "last_message", None)
        while True:
            remaining = cfg.max_wait_seconds - self._wait_spent()
            if remaining <= 0:
                break
            step = min(cfg.probe_seconds, remaining)
            started = self._now()
            outcome: GuardOutcome | None = None
            try:
                await self._message(
                    f"{WAITING_PREFIX} ({self.last_error_kind or self.circuit.last_error_kind or 'error'}) — "
                    f"retrying ~{_et_clock(time.time() + step)}"
                )
                await self._sleep(step)
                outcome = await self._attempts(fn, args, 0)
            finally:
                self._add_wait(max(0.0, self._now() - started))
            if outcome.reason != REASON_TRANSIENT:
                if previous and not str(previous).startswith(WAITING_PREFIX):
                    await self._message(str(previous))
                return outcome
        self._trip()
        await self._stop_message()
        return self._unreachable()

    async def _stop_message(self) -> None:
        await self._message(
            f"The judge ({self.label}) was unreachable ({self.last_error_kind or 'error'}); "
            "the remaining rows are left unjudged. Re-run the eval when it is back."
        )

    async def call(self, fn: Callable[..., Any], *args: Any, row: int | None = None) -> GuardOutcome:
        """Call ``fn(*args)`` under the guard. ``row`` is only recorded for the breaker block."""
        self._row = row
        if self.tripped:
            return self._unreachable()
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise asyncio.CancelledError()
        if self._open_now():
            if self._can_wait():
                return await self._wait_for_judge(fn, args)
            self._trip()
            await self._stop_message()
            return self._unreachable()
        outcome = await self._attempts(fn, args, self.resilience.retries)
        if self.tripped:
            await self._stop_message()
        # Mode wait: a failure that opened the circuit leaves THIS row transient_error; the next row waits.
        return outcome

    # -- sync (worker thread) --

    def call_sync(self, fn: Callable[..., Any], *args: Any, row: int | None = None) -> GuardOutcome:
        """``call`` for a judge loop in a worker thread: stop mode only, no Job handle, cancel via
        ``cancel_event`` (raises ``JudgeCancelled``)."""
        from app.services.llm_budget_guard import LlmBudgetExceeded

        self._row = row
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise JudgeCancelled("cancelled")
        if self.tripped:
            return self._unreachable()
        if self._open_now():
            self._trip()
            return self._unreachable()
        retries = self.resilience.retries
        attempt = 0
        while True:
            try:
                self.calls += 1
                result = fn(*args)
            except LlmBudgetExceeded:
                raise
            except JudgeCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                kind, text = self._note_failure(exc)
                if classify_judge_error(exc) == PERMANENT:
                    self.circuit.record_permanent()
                    return GuardOutcome(None, REASON_PERMANENT, text)
                if attempt < retries:
                    self._sleep_sync(self._backoff(attempt))
                    attempt += 1
                    continue
                if self.circuit.record_transient(self.resilience, self._now(), kind, text):
                    self._trip()
                return GuardOutcome(None, REASON_TRANSIENT, text)
            self.circuit.record_success()
            return GuardOutcome(result, None if result is not None else REASON_UNPARSEABLE, None)


def new_guard(label: str, resilience: JudgeResilience | None, **kwargs: Any) -> JudgeCallGuard | None:
    return JudgeCallGuard(label=label, resilience=resilience, **kwargs) if resilience is not None else None


# ── snapshot helpers ────────────────────────────────────────────────


def mark_unjudged(row: dict[str, Any], reason: str, error: str | None, *, prefix: str = "row_judge") -> None:
    """Per-row unjudged keys (only ever written on an unjudged row)."""
    row[f"{prefix}_status"] = "unjudged"
    row[f"{prefix}_unjudged_reason"] = reason
    if error:
        row[f"{prefix}_error"] = str(error)[:_ERROR_CHARS]


def unjudged_fields(judged: int, unjudged: int, reasons: dict[str, int], guard: JudgeCallGuard | None) -> dict[str, Any]:
    """Snapshot keys added only when rows were left unjudged (a run without failures is unchanged)."""
    if unjudged <= 0:
        return {}
    out: dict[str, Any] = {
        "status": "paused" if guard is not None and guard.tripped else "partial",
        "unjudged_reasons": {k: v for k, v in reasons.items() if v},
        "coverage": round(judged / (judged + unjudged), 4) if (judged + unjudged) else None,
    }
    block = guard.breaker_block if guard is not None else None
    if block is not None:
        out["breaker"] = block
    return out


def reason_for_exception(exc: BaseException) -> str:
    """Unguarded loops: the reason an exception leaves (no retry, no breaker)."""
    return REASON_TRANSIENT if classify_judge_error(exc) == TRANSIENT else REASON_PERMANENT


def _outage_unjudged(snapshot: dict[str, Any]) -> int:
    reasons = snapshot.get("unjudged_reasons")
    if not isinstance(reasons, dict):
        return 0
    total = 0
    for reason in OUTAGE_REASONS:
        try:
            total += int(reasons.get(reason) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _stopped(snapshot: dict[str, Any]) -> bool:
    breaker = snapshot.get("breaker")
    return snapshot.get("status") == "paused" or (isinstance(breaker, dict) and breaker.get("tripped") is True)


def snapshot_incomplete(snapshot: Any) -> bool:
    """Whether a judge snapshot (answer / code-example / rubric / dialogue block) is missing verdicts
    because the judge was not reachable: the breaker stopped it (``status`` "paused" / ``breaker.tripped``),
    a row failed transiently, or it was asked about rows and judged none."""
    if not isinstance(snapshot, dict):
        return False
    if _stopped(snapshot) or _outage_unjudged(snapshot) > 0:
        return True
    judged = snapshot.get("judged")
    unjudged = snapshot.get("unjudged")
    return isinstance(judged, int) and isinstance(unjudged, int) and judged == 0 and unjudged > 0


def snapshot_blocks_base_reuse(snapshot: Any) -> bool:
    """Whether a base result's judge snapshot is too incomplete to stand in as a lift check's bar: the
    breaker stopped it, it judged none of the rows it was asked about, or more than
    ``BASE_REUSE_MAX_OUTAGE_SHARE`` of its rows were left unjudged for an outage. A few rows lost to a
    blip (or any number of unparseable replies, which a re-run would not fix) do not force a GPU re-run
    of the base model; those rows simply do not pair."""
    if not isinstance(snapshot, dict):
        return False
    if _stopped(snapshot):
        return True
    judged = snapshot.get("judged")
    unjudged = snapshot.get("unjudged")
    if isinstance(judged, int) and isinstance(unjudged, int):
        if judged == 0 and unjudged > 0:
            return True
        total = judged + unjudged
        if total > 0 and _outage_unjudged(snapshot) / total > BASE_REUSE_MAX_OUTAGE_SHARE:
            return True
    return False


# Judge blocks of an eval's metrics, in the order a warning reports them, with a plain name each.
JUDGE_BLOCKS: tuple[tuple[str, str], ...] = (
    ("judge", "answer judge"),
    ("code_example_judge", "code-example judge"),
    ("rubric", "rubric judge"),
    ("dialogue_judge", "dialogue judge"),
)


def _judged_count(block: dict[str, Any]) -> Any:
    # The dialogue judge counts judged turns as ``turns_judged``.
    return block.get("judged") if block.get("judged") is not None else block.get("turns_judged")


def judge_outage(block: Any) -> bool:
    """Whether a judge block written WITH unjudged-row reasons (an eval run since rows carried them) lost
    rows to an unreachable judge. Results stored before that (no ``unjudged_reasons``) never qualify, and
    neither do gaps that are only unparseable replies or refused requests (a re-run would not fix them)."""
    if not isinstance(block, dict) or not isinstance(block.get("unjudged_reasons"), dict):
        return False
    return _stopped(block) or _outage_unjudged(block) > 0


def judged_subset(block: Any) -> dict[str, int] | None:
    """``{judged, unjudged}`` when a judge block written with unjudged-row reasons left rows unjudged (its
    numbers then rest on the judged rows only), else None."""
    if not isinstance(block, dict) or not isinstance(block.get("unjudged_reasons"), dict):
        return None
    try:
        unjudged = int(block.get("unjudged") or 0)
        judged = int(_judged_count(block) or 0)
    except (TypeError, ValueError):
        return None
    return {"judged": judged, "unjudged": unjudged} if unjudged > 0 else None


def unjudged_notice_block(metrics: Any) -> dict[str, Any] | None:
    """The judge block an eval summary's "rows without a verdict" notice describes: the first block
    (``JUDGE_BLOCKS`` order) that lost rows to an unreachable judge, else the first that left any rows
    unjudged -- only blocks written with unjudged-row reasons. ``{kind, judge, judged, unjudged, status
    ("paused" / "partial"), unjudged_reasons, breaker?}``; None when there is nothing to say."""
    if not isinstance(metrics, dict):
        return None
    candidates = [(key, metrics.get(key)) for key, _name in JUDGE_BLOCKS if judged_subset(metrics.get(key))]
    if not candidates:
        return None
    key, block = next(((k, b) for k, b in candidates if judge_outage(b)), candidates[0])
    out: dict[str, Any] = {
        "kind": key,
        "judge": block.get("judge"),
        "judged": _judged_count(block),
        "unjudged": block.get("unjudged"),
        "status": "paused" if _stopped(block) else "partial",
        "unjudged_reasons": block.get("unjudged_reasons") or {},
    }
    if isinstance(block.get("breaker"), dict):
        out["breaker"] = block["breaker"]
    return out


def incomplete_summary(metrics: Any) -> dict[str, Any] | None:
    """``{status, judge, judge_kind, judged, unjudged, unjudged_reasons, incomplete}`` of the first judge
    block of an eval's metrics that lost rows to an unreachable judge (``judge_outage``), else None. For
    Job results ("done, with a warning")."""
    if not isinstance(metrics, dict):
        return None
    for key, name in JUDGE_BLOCKS:
        block = metrics.get(key)
        if not judge_outage(block):
            continue
        return {
            "status": "paused" if _stopped(block) else "partial",
            "judge": block.get("judge"),
            "judge_kind": name,
            "judged": _judged_count(block),
            "unjudged": block.get("unjudged"),
            "unjudged_reasons": block.get("unjudged_reasons") or {},
            "incomplete": True,
        }
    return None


def job_result_fields(*results: Any) -> dict[str, Any]:
    """Keys a Job runner adds to its result when an eval it ran lost rows to an unreachable judge (nothing
    otherwise -- unparseable-only gaps and results stored before unjudged rows carried a reason add
    nothing): ``judge_status`` ("paused" / "partial"), ``judge_unjudged``, ``judge_judged``,
    ``judge_unjudged_reasons``, ``judge_label``, ``judge_kind``, ``judge_eval_result_id``. With several
    results (base + fine-tuned) the first incomplete one is reported."""
    for result in results:
        if result is None:
            continue
        summary = incomplete_summary(getattr(result, "metrics", None) if not isinstance(result, dict) else result)
        if summary is None:
            continue
        return {
            "judge_status": summary["status"],
            "judge_unjudged": summary["unjudged"],
            "judge_judged": summary["judged"],
            "judge_unjudged_reasons": summary["unjudged_reasons"],
            "judge_label": summary["judge"],
            "judge_kind": summary["judge_kind"],
            "judge_eval_result_id": getattr(result, "id", None),
        }
    return {}

"""Judge resilience: classify judge-call failures, retry transient ones, stop on an outage.

An LLM judge scores eval rows one call at a time. When its provider (or the network) is down, every
call fails, and a loop that swallows each error and moves on spends hours failing (a backend that
retries internally multiplies that). This module gives such a loop three things:

* ``classify_judge_error(exc)`` -> ``"transient"`` (connection / DNS / timeout / 408 / 429 / 5xx /
  ``SynthBackendTransientError`` -- the network's or the provider's fault, worth retrying later) or
  ``"permanent"`` (the request's: 4xx, a bad key, anything unknown). It walks ``__cause__``, because the
  synth backends wrap provider errors in ``SynthBackendError``.
* ``JudgeCallGuard`` -- one per judge loop (run). ``await guard.call(fn, *args)`` retries a transient
  failure ``JUDGE_TRANSIENT_RETRIES`` times with backoff (none for a backend that retries itself), and
  returns ``GuardOutcome(result, reason, error)``. ``reason`` is None for a verdict, else why the row is
  unjudged: ``unparseable`` (the judge answered, no verdict), ``permanent_error``, ``transient_error``
  (failed after retries), ``judge_unreachable`` (the breaker is open: no call was made).
* A circuit breaker per judge label, shared by every loop in this process (``threading.Lock``; never an
  asyncio primitive, so a loop running in a worker thread can use it too). It opens after
  ``JUDGE_BREAKER_CONSECUTIVE`` transient failures in a row, or when more than
  ``JUDGE_BREAKER_ERROR_RATE`` of the last ``JUDGE_BREAKER_WINDOW`` calls (at least
  ``JUDGE_BREAKER_MIN_CALLS``) failed transiently. Then, in mode ``stop`` (default), the run's remaining
  rows are left unjudged without a call; in mode ``wait`` the run probes every
  ``JUDGE_OUTAGE_PROBE_SECONDS`` for up to ``max_wait_seconds`` and resumes when a probe answers, else
  stops. Every NEW run sends its first call through as a probe, whatever the shared state says, so a run
  started right after another one tripped is not written off without one call; an open breaker also
  lets one call through after ``JUDGE_BREAKER_COOLDOWN_SECONDS``. Permanent failures and unparseable
  replies never count towards the breaker. The registry is per process (a celery worker has its own).

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
UNJUDGED_REASONS: tuple[str, ...] = (REASON_UNREACHABLE, REASON_TRANSIENT, REASON_UNPARSEABLE, REASON_PERMANENT)
# Reasons that mean "the judge was not reachable" (worth judging again later).
OUTAGE_REASONS: frozenset[str] = frozenset({REASON_UNREACHABLE, REASON_TRANSIENT})

MODES: tuple[str, ...] = ("stop", "wait")
MAX_WAIT_CAP_SECONDS = 24 * 3600.0
_ERROR_CHARS = 200
# Sleep / probe waits are sliced so a cancel is noticed within this long.
_SLICE_SECONDS = 5.0
_JITTER = 0.2
WAITING_PREFIX = "judge unreachable"


# ── classification ──────────────────────────────────────────────────


def _status_transient(code: int) -> bool:
    return code in (408, 409, 425, 429) or 500 <= code <= 599


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
        out["max_wait_seconds"] = _num(raw.get("max_wait_seconds"), 0.0, MAX_WAIT_CAP_SECONDS, 3600.0)
    return out


def resolve_resilience(runtime_config: Any = None, *, retries_internally: bool = False, threaded: bool = False) -> JudgeResilience:
    """Settings from ``config.py`` overlaid with the project's ``eval_judge.resilience`` block.
    ``retries_internally``: the backend retries a failed call itself -> no extra guard retries.
    ``threaded``: the loop runs in a worker thread -> always ``stop`` (no waiting, no Job messages)."""
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
        retries=0 if retries_internally else retries,
        backoff_seconds=_num(getattr(settings, "JUDGE_RETRY_BACKOFF_SECONDS", 10.0), 0.0, 3600.0, 10.0),
        mode=mode,
        probe_seconds=_num(getattr(settings, "JUDGE_OUTAGE_PROBE_SECONDS", 300.0), 1.0, 86400.0, 300.0),
        max_wait_seconds=block.get("max_wait_seconds")
        if "max_wait_seconds" in block
        else _num(getattr(settings, "JUDGE_OUTAGE_MAX_WAIT_SECONDS", 3600.0), 0.0, MAX_WAIT_CAP_SECONDS, 3600.0),
    )


# ── the shared breaker ──────────────────────────────────────────────


class JudgeCircuit:
    """Health of one judge (by label), shared across loops. Every method takes the lock."""

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

    def is_open(self) -> bool:
        with self._lock:
            return self.opened_at is not None

    def record_success(self) -> None:
        with self._lock:
            self.consecutive = 0
            self.window.append(False)
            self.opened_at = None

    def record_permanent(self) -> None:
        # The provider answered (it rejected the request): not an outage, and not proof of health either.
        with self._lock:
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


_REGISTRY: dict[str, JudgeCircuit] = {}
_REGISTRY_LOCK = threading.Lock()


def circuit_for(label: str) -> JudgeCircuit:
    with _REGISTRY_LOCK:
        circuit = _REGISTRY.get(label)
        if circuit is None:
            circuit = _REGISTRY[label] = JudgeCircuit(label)
        return circuit


def reset_registry() -> None:
    """Tests: forget every judge's health."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


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
    """One judge loop's guard (see module doc). Not shared between loops; the circuit is."""

    label: str
    resilience: JudgeResilience = field(default_factory=JudgeResilience)
    threaded: bool = False
    cancel_event: threading.Event | None = None
    # Injected in tests.
    sleep: Callable[[float], Any] | None = None
    clock: Callable[[], float] | None = None

    def __post_init__(self) -> None:
        self.circuit = circuit_for(self.label)
        self.probed = False
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
        """Whether this run must not call now (the shared circuit is open and cooling down)."""
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
                    if self.resilience.mode != "wait":
                        self._trip()
                return GuardOutcome(None, REASON_TRANSIENT, text)
            self.circuit.record_success()
            return GuardOutcome(result, None if result is not None else REASON_UNPARSEABLE, None)

    async def _wait_for_judge(self, fn: Callable[..., Any], args: tuple) -> GuardOutcome:
        """Mode ``wait``: probe every ``probe_seconds`` until a call answers or the wait budget runs out."""
        cfg = self.resilience
        previous = getattr(self._handle_here(), "last_message", None)
        while self.waited_seconds < cfg.max_wait_seconds:
            step = min(cfg.probe_seconds, cfg.max_wait_seconds - self.waited_seconds)
            resume = time.time() + step
            await self._message(
                f"{WAITING_PREFIX} ({self.last_error_kind or self.circuit.last_error_kind or 'error'}) — "
                f"retrying ~{_et_clock(resume)}"
            )
            await self._sleep(step)
            self.waited_seconds += step
            outcome = await self._attempts(fn, args, 0)
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
        if not self.probed:
            # A new run always gets one call, whatever an earlier run left the circuit in. A probe into
            # an open circuit gets no retries: one answer is enough to know.
            self.probed = True
            outcome = await self._attempts(fn, args, 0 if self.circuit.is_open() else self.resilience.retries)
            if self.tripped:
                await self._stop_message()
            return outcome
        if self._open_now():
            if self.resilience.mode == "wait" and not self.threaded and self.resilience.max_wait_seconds > 0:
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
        if self.probed and self._open_now():
            self._trip()
            return self._unreachable()
        retries = 0 if (not self.probed and self.circuit.is_open()) else self.resilience.retries
        self.probed = True
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


def snapshot_incomplete(snapshot: Any) -> bool:
    """Whether a judge snapshot (answer / code-example / rubric block) is missing verdicts because the
    judge was not reachable: the breaker stopped it (``status`` "paused"), a row failed transiently, or it
    was asked about rows and judged none. Such a result must not stand in as a bar (the lift check's base
    reuse) -- its holes would pair as nothing."""
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get("status") == "paused":
        return True
    reasons = snapshot.get("unjudged_reasons")
    if isinstance(reasons, dict) and any(int(reasons.get(r) or 0) > 0 for r in OUTAGE_REASONS):
        return True
    judged = snapshot.get("judged")
    unjudged = snapshot.get("unjudged")
    return isinstance(judged, int) and isinstance(unjudged, int) and judged == 0 and unjudged > 0


def incomplete_summary(metrics: Any) -> dict[str, Any] | None:
    """``{status, judge, judged, unjudged, unjudged_reasons}`` of the first incomplete judge block of an
    eval's metrics (answer judge, code-example judge, rubric), else None. For Job results ("done, with a
    warning")."""
    if not isinstance(metrics, dict):
        return None
    for key in ("judge", "code_example_judge", "rubric"):
        block = metrics.get(key)
        if not isinstance(block, dict) or not int(block.get("unjudged") or 0):
            continue
        return {
            "status": block.get("status") or ("paused" if snapshot_incomplete(block) else "partial"),
            "judge": block.get("judge"),
            "judged": block.get("judged"),
            "unjudged": block.get("unjudged"),
            "unjudged_reasons": block.get("unjudged_reasons") or {},
            "incomplete": snapshot_incomplete(block),
        }
    return None


def job_result_fields(*results: Any) -> dict[str, Any]:
    """Keys a Job runner adds to its result when an eval it ran left rows unjudged (nothing otherwise):
    ``judge_status`` ("paused" / "partial"), ``judge_unjudged``, ``judge_judged``,
    ``judge_unjudged_reasons``, ``judge_label``, ``judge_eval_result_id``. With several results (base +
    fine-tuned) the first incomplete one is reported."""
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
            "judge_eval_result_id": getattr(result, "id", None),
        }
    return {}

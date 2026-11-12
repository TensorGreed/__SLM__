"""Judge resilience (judge_call_guard): classification, retries, the breaker, unjudged state.

Pins:
  * classify_judge_error: cloud status codes / transport errors / the transient backend error, through
    ``SynthBackendError`` wrappers (``__cause__``); unknown errors are permanent;
  * the guard retries transient failures only (none for a backend that retries itself), a 3rd
    consecutive transient failure opens the breaker and the remaining rows are ``judge_unreachable``
    WITHOUT a call; cache hits still apply; permanent errors / unparseable replies never trip it;
  * the breaker is per run: a run after a trip starts closed with normal retries (a stale outage never
    writes off a new eval); an error-rate trip; half-open after the cool-down; breaker off = one call per
    row, no retries;
  * wait mode resumes when a probe answers; its budget is wall time incl. probe calls, one per Job, capped
    under the task queue's visibility timeout; no Job handle -> stop;
  * which judge blocks warn (outage, new results only) and which block base reuse (over a threshold);
  * ``LlmBudgetExceeded`` propagates; per-row keys and snapshot keys only appear on unjudged runs;
  * the thread-mode guard never touches the Job handle.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("DEBUG", "false")

import httpx

from app.services import judge_call_guard as jcg
from app.services.answer_judge_service import AnswerJudgeCache, judge_predictions, judge_retrieval
from app.services.cloud_llm_service import CloudLlmError
from app.services.llm_budget_guard import LlmBudgetExceeded
from app.services.synth_backends.base import SynthBackendError, SynthBackendTransientError

LONG = "The head of a government institution may refuse to disclose records containing personal information unless consent is given."


def _rows(n: int) -> list[dict]:
    return [{"prompt": f"Q{i}", "reference": LONG, "prediction": f"answer {i}"} for i in range(n)]


def _cfg(**over) -> jcg.JudgeResilience:
    base = dict(breaker=True, consecutive=3, window=20, min_calls=10, error_rate=0.5, cooldown_seconds=300.0,
                retries=0, backoff_seconds=0.0, mode="stop", probe_seconds=10.0, max_wait_seconds=0.0)
    base.update(over)
    return jcg.JudgeResilience(**base)


def _connect_error() -> Exception:
    try:
        raise httpx.ConnectError("EAI_AGAIN")
    except httpx.ConnectError as inner:
        try:
            raise SynthBackendError("Ollama request failed") from inner
        except SynthBackendError as outer:
            return outer


class ClassificationTests(unittest.TestCase):
    def test_table(self):
        T, P = jcg.TRANSIENT, jcg.PERMANENT
        cases = [
            (CloudLlmError("x", status_code=429), T),
            (CloudLlmError("x", status_code=500), T),
            (CloudLlmError("x", status_code=503), T),
            (CloudLlmError("x", status_code=400), P),
            (CloudLlmError("x", status_code=401), P),
            (CloudLlmError("x", status_code=409), P),
            (CloudLlmError("x", status_code=529), T),
            (CloudLlmError("x", transient=True), T),
            (CloudLlmError("API key is required."), P),
            (httpx.ConnectError("boom"), T),
            (httpx.ReadTimeout("slow"), T),
            (SynthBackendTransientError("cli connection_failed"), T),
            (SynthBackendError("allow-list"), P),
            (ValueError("bad"), P),
            (_connect_error(), T),
        ]
        for exc, want in cases:
            self.assertEqual(jcg.classify_judge_error(exc), want, repr(exc))

    def test_wrapped_cloud_error_keeps_its_status(self):
        try:
            try:
                raise CloudLlmError("Anthropic returned 503", status_code=503)
            except CloudLlmError as inner:
                raise SynthBackendError("Cloud LLM failed") from inner
        except SynthBackendError as outer:
            self.assertEqual(jcg.classify_judge_error(outer), jcg.TRANSIENT)
        try:
            try:
                raise CloudLlmError("Anthropic returned 401", status_code=401)
            except CloudLlmError as inner:
                raise SynthBackendError("Cloud LLM failed") from inner
        except SynthBackendError as outer:
            self.assertEqual(jcg.classify_judge_error(outer), jcg.PERMANENT)

    def test_http_status_error_from_local_backend(self):
        req = httpx.Request("POST", "http://localhost/api/chat")
        for code, want in ((502, jcg.TRANSIENT), (404, jcg.PERMANENT)):
            err = httpx.HTTPStatusError("x", request=req, response=httpx.Response(code, request=req))
            self.assertEqual(jcg.classify_judge_error(err), want)


class GuardLoopTests(unittest.TestCase):
    def test_consecutive_trip_stops_calls_and_marks_rows(self):
        calls: list[str] = []

        async def judge(q, r, p):
            calls.append(q)
            raise SynthBackendTransientError("connection_failed")

        rows = _rows(10)
        snap = asyncio.run(judge_predictions(rows, judge, label="down-judge", resilience=_cfg()))
        self.assertEqual(len(calls), 3, "the breaker opens after the 3rd failed row; no more calls")
        self.assertEqual(snap["judge_calls"], 3)
        self.assertEqual((snap["judged"], snap["unjudged"]), (0, 10))
        self.assertIsNone(snap["score"])
        self.assertEqual(snap["status"], "paused")
        self.assertEqual(snap["unjudged_reasons"], {"transient_error": 3, "judge_unreachable": 7})
        self.assertEqual(snap["coverage"], 0.0)
        self.assertEqual(snap["breaker"]["tripped_at_row"], 2)
        self.assertEqual(rows[0]["row_judge_unjudged_reason"], "transient_error")
        self.assertIn("connection_failed", rows[0]["row_judge_error"])
        self.assertEqual(rows[9]["row_judge_status"], "unjudged")
        self.assertEqual(rows[9]["row_judge_unjudged_reason"], "judge_unreachable")
        self.assertNotIn("row_judge_score", rows[9])

    def test_cache_hits_still_apply_after_a_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = AnswerJudgeCache(Path(tmp) / "c.json")
            rows = _rows(6)
            from app.services.answer_judge_service import cache_key

            cache.set(cache_key("lbl", "Q5", LONG, "answer 5", None), "correct", "cached")

            async def judge(q, r, p):
                raise httpx.ConnectError("down")

            snap = asyncio.run(judge_predictions(rows, judge, label="lbl", cache=cache, resilience=_cfg()))
            self.assertEqual(rows[5]["row_judge_verdict"], "correct")
            self.assertNotIn("row_judge_status", rows[5])
            self.assertEqual((snap["judged"], snap["judge_cached"], snap["judge_calls"]), (1, 1, 3))

    def test_permanent_errors_and_unparseable_never_trip(self):
        n = {"calls": 0}

        async def judge(q, r, p):
            n["calls"] += 1
            if n["calls"] % 2:
                raise CloudLlmError("bad request", status_code=400)
            return None

        rows = _rows(8)
        snap = asyncio.run(judge_predictions(rows, judge, label="picky", resilience=_cfg()))
        self.assertEqual(n["calls"], 8, "every row is still asked")
        self.assertEqual(snap["status"], "partial")
        self.assertEqual(snap["unjudged_reasons"], {"permanent_error": 4, "unparseable": 4})
        self.assertNotIn("breaker", snap)

    def test_retries_transient_with_backoff_then_succeeds(self):
        attempts = {"n": 0}
        sleeps: list[float] = []

        async def judge(q, r, p):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise CloudLlmError("503", status_code=503)
            return 1.0, "correct", "ok", 0

        async def fake_sleep(s):
            sleeps.append(s)

        guard = jcg.JudgeCallGuard(label="flaky", resilience=_cfg(retries=2, backoff_seconds=10.0), sleep=fake_sleep)
        out = asyncio.run(guard.call(judge, "q", "r", "p"))
        self.assertIsNone(out.reason)
        self.assertEqual(attempts["n"], 3)
        # 10 s then 30 s (± 20 % jitter), sliced into ≤5 s steps for cancel checks.
        self.assertTrue(8.0 * 4 <= sum(sleeps) <= 12.0 * 4, sum(sleeps))

    def test_backend_that_retries_internally_gets_no_extra_retries(self):
        cfg = jcg.resolve_resilience({}, retries_internally=True)
        self.assertEqual(cfg.retries, 0)
        self.assertEqual(jcg.resolve_resilience({}).retries, 2)

    def test_budget_stop_propagates(self):
        async def judge(q, r, p):
            raise LlmBudgetExceeded("project", 10.0, 10.5)

        with self.assertRaises(LlmBudgetExceeded):
            asyncio.run(judge_predictions(_rows(2), judge, label="b", resilience=_cfg()))

    def test_error_rate_trip(self):
        n = {"calls": 0}

        async def judge(q, r, p):
            n["calls"] += 1
            if n["calls"] % 3 == 0:
                return 1.0, "correct", "ok", 0
            raise httpx.ReadTimeout("slow")

        rows = _rows(30)
        snap = asyncio.run(judge_predictions(
            rows, judge, label="half", resilience=_cfg(consecutive=100, min_calls=10, error_rate=0.5),
        ))
        self.assertLess(n["calls"], 30)
        self.assertEqual(snap["status"], "paused")
        self.assertGreater(snap["unjudged_reasons"]["judge_unreachable"], 0)

    def test_a_new_run_after_a_trip_starts_closed_with_retries(self):
        """M1: nothing carries over between runs. A run after a trip (judge still down) gets the normal
        retries and its own 3 rows before it stops; once the judge is back the next run judges all."""
        state = {"down": True, "calls": 0}

        async def judge(q, r, p):
            state["calls"] += 1
            if state["down"]:
                raise httpx.ConnectError("down")
            return 1.0, "correct", "ok", 0

        first = asyncio.run(judge_predictions(_rows(5), judge, label="lift-judge", resilience=_cfg(retries=2)))
        self.assertEqual(first["status"], "paused")
        state["calls"] = 0
        snap = asyncio.run(judge_predictions(_rows(5), judge, label="lift-judge", resilience=_cfg(retries=2)))
        self.assertEqual(state["calls"], 9, "3 rows x (1 + 2 retries), like any fresh run")
        self.assertEqual(snap["unjudged_reasons"], {"transient_error": 3, "judge_unreachable": 2})
        state.update(down=False, calls=0)
        snap = asyncio.run(judge_predictions(_rows(5), judge, label="lift-judge", resilience=_cfg(retries=2)))
        self.assertEqual(state["calls"], 5)
        self.assertNotIn("status", snap)

    def test_stale_open_circuit_one_blip_never_writes_off_a_run(self):
        """The reviewer's M1 probe: a trip overnight, a day later one 529 on the next run's first call.
        The blip is retried and every row is judged (before the fix: 1 call, the run tripped)."""
        now = [0.0]

        async def dead(*a):
            raise CloudLlmError("down", status_code=503)

        async def run(fn, rows):
            guard = jcg.JudgeCallGuard(label="L", resilience=_cfg(retries=2), clock=lambda: now[0],
                                       sleep=_no_sleep)
            return guard, [await guard.call(fn, i, row=i) for i in range(rows)]

        g1, _ = asyncio.run(run(dead, 5))
        self.assertTrue(g1.tripped)
        now[0] = 86400.0
        calls = {"n": 0}

        async def blip_then_ok(*a):
            calls["n"] += 1
            if calls["n"] == 1:
                raise CloudLlmError("overloaded", status_code=529)
            return (1.0, "correct", "", 0)

        g2, out = asyncio.run(run(blip_then_ok, 50))
        self.assertFalse(g2.tripped)
        self.assertEqual(calls["n"], 51, "the 529 was retried once, then 50 verdicts")
        self.assertTrue(all(o.reason is None for o in out))

    def test_half_open_after_cooldown(self):
        clock = {"t": 1000.0}
        circuit = jcg.JudgeCircuit("cool")
        cfg = _cfg(cooldown_seconds=300.0)
        for _ in range(3):
            circuit.record_transient(cfg, clock["t"], "connection error", "x")
        self.assertTrue(circuit.snapshot_open(clock["t"] + 10, 300.0))
        self.assertFalse(circuit.snapshot_open(clock["t"] + 301, 300.0), "half-open: one call may pass")

    def test_each_run_has_its_own_circuit(self):
        a = jcg.JudgeCallGuard(label="same", resilience=_cfg())
        b = jcg.JudgeCallGuard(label="same", resilience=_cfg())
        self.assertIsNot(a.circuit, b.circuit)
        for _ in range(3):
            a.circuit.record_transient(_cfg(), jcg.time.monotonic(), "connection error", "x")
        self.assertTrue(a._open_now())
        self.assertFalse(b._open_now())

    def test_a_permanent_failure_breaks_the_transient_run(self):
        circuit = jcg.JudgeCircuit("mixed")
        cfg = _cfg()
        self.assertFalse(circuit.record_transient(cfg, 0.0, "HTTP 503", "x"))
        circuit.record_permanent()
        self.assertFalse(circuit.record_transient(cfg, 0.0, "HTTP 503", "x"))
        self.assertFalse(circuit.record_transient(cfg, 0.0, "HTTP 503", "x"), "2 in a row since the 400, not 3")
        self.assertTrue(circuit.record_transient(cfg, 0.0, "HTTP 503", "x"))

    def test_breaker_disabled_marches_every_row(self):
        calls = {"n": 0}

        async def judge(q, r, p):
            calls["n"] += 1
            raise httpx.ConnectError("down")

        snap = asyncio.run(judge_predictions(_rows(6), judge, label="nobreak", resilience=_cfg(breaker=False)))
        self.assertEqual(calls["n"], 6)
        self.assertNotIn("breaker", snap)
        self.assertEqual(snap["status"], "partial")
        self.assertEqual(snap["unjudged_reasons"], {"transient_error": 6})

    def test_breaker_off_restores_one_call_per_row(self):
        """S2: breaker off (project or install) = the march before the guard: one call per row, no
        retries, no backoff -- the reviewer's probe made 12 calls for 4 rows."""
        cfg = jcg.resolve_resilience({"eval_judge": {"resilience": {"breaker": False}}})
        self.assertEqual((cfg.breaker, cfg.retries), (False, 0))
        calls = {"n": 0}

        async def dead(*a):
            calls["n"] += 1
            raise CloudLlmError("down", status_code=503)

        async def run():
            guard = jcg.JudgeCallGuard(label="M", resilience=cfg, sleep=_no_sleep)
            return [await guard.call(dead, i, row=i) for i in range(4)]

        out = asyncio.run(run())
        self.assertEqual(calls["n"], 4)
        self.assertEqual([o.reason for o in out], ["transient_error"] * 4)
        with mock.patch.object(jcg.settings, "JUDGE_BREAKER_ENABLED", False):
            self.assertEqual(jcg.resolve_resilience({}).retries, 0)

    def test_wait_mode_resumes_after_the_judge_comes_back(self):
        state = {"calls": 0}

        async def judge(q, r, p):
            state["calls"] += 1
            # Down for the first 4 calls (3 trip the breaker, the 1st probe fails), then back.
            if state["calls"] <= 4:
                raise httpx.ConnectError("down")
            return 1.0, "correct", "ok", 0

        rows = _rows(6)
        clock = _Clock()
        cfg = _cfg(mode="wait", probe_seconds=60.0, max_wait_seconds=600.0)

        async def run():
            from app.services.answer_judge_service import _judge_predictions

            with _job_handle() as handle, _guard_clock(clock):
                snap = await _judge_predictions(rows, judge, label="waiter", resilience=cfg)
                return snap, handle

        snap, handle = asyncio.run(run())
        self.assertEqual(snap["judged"], 3, "rows 3..5 judged after the probe answered")
        self.assertEqual(snap["unjudged_reasons"], {"transient_error": 3})
        self.assertEqual(snap["status"], "partial")
        self.assertEqual(snap["breaker"]["waited_seconds"], 120.0)
        self.assertEqual(handle.judge_wait_spent_seconds, 120.0)
        self.assertEqual(rows[3]["row_judge_verdict"], "correct")

    def test_wait_mode_gives_up_after_max_wait(self):
        async def judge(q, r, p):
            raise httpx.ConnectError("down")

        rows = _rows(6)
        clock = _Clock()
        cfg = _cfg(mode="wait", probe_seconds=60.0, max_wait_seconds=120.0)

        async def run():
            from app.services.answer_judge_service import _judge_predictions

            with _job_handle(), _guard_clock(clock):
                return await _judge_predictions(rows, judge, label="waiter2", resilience=cfg)

        snap = asyncio.run(run())
        self.assertEqual(snap["status"], "paused")
        self.assertEqual(snap["unjudged_reasons"]["judge_unreachable"], 3)
        self.assertEqual(snap["judge_calls"], 5, "3 to trip + 2 probes, then stop")

    def test_wait_budget_is_wall_time_including_probe_calls(self):
        """S1 (the reviewer's probe): a probe that takes 9 min to fail counts against the budget. Before:
        a 1 h budget held the loop for 3.25 h."""
        clock = _Clock()

        async def slow_dead(*a):
            clock.t += 540.0
            raise CloudLlmError("down", status_code=503)

        cfg = _cfg(mode="wait", retries=0, probe_seconds=300.0, max_wait_seconds=3600.0)

        async def run():
            with _job_handle() as handle:
                guard = jcg.JudgeCallGuard(label="W", resilience=cfg, clock=clock, sleep=clock.sleep)
                out = [await guard.call(slow_dead, i, row=i) for i in range(6)]
                return guard, out, handle

        guard, out, handle = asyncio.run(run())
        trip_time = 3 * 540.0
        self.assertTrue(guard.tripped)
        self.assertLessEqual(clock.t - trip_time, 3600.0 + 540.0, "overrun by at most one probe call")
        self.assertGreaterEqual(handle.judge_wait_spent_seconds, 3600.0)
        self.assertEqual(out[-1].reason, "judge_unreachable")

    def test_one_wait_budget_per_job(self):
        """Two judge loops of one Job (answer + retrieval, or two lift legs) share one budget."""
        clock = _Clock()

        async def dead(*a):
            raise httpx.ConnectError("down")

        cfg = _cfg(mode="wait", probe_seconds=60.0, max_wait_seconds=120.0)

        async def run():
            with _job_handle() as handle:
                first = jcg.JudgeCallGuard(label="A", resilience=cfg, clock=clock, sleep=clock.sleep)
                for i in range(5):
                    await first.call(dead, i, row=i)
                second = jcg.JudgeCallGuard(label="B", resilience=cfg, clock=clock, sleep=clock.sleep)
                for i in range(5):
                    await second.call(dead, i, row=i)
                return first, second, handle

        first, second, handle = asyncio.run(run())
        self.assertEqual(first.waited_seconds, 120.0)
        self.assertEqual(second.waited_seconds, 0.0, "the Job's budget was already spent")
        self.assertTrue(second.tripped)
        self.assertEqual(handle.judge_wait_spent_seconds, 120.0)

    def test_wait_mode_without_a_job_handle_stops(self):
        calls = {"n": 0}

        async def dead(*a):
            calls["n"] += 1
            raise httpx.ConnectError("down")

        async def run():
            guard = jcg.JudgeCallGuard(label="nohandle", resilience=_cfg(mode="wait", max_wait_seconds=600.0),
                                       sleep=_no_sleep)
            return guard, [await guard.call(dead, i, row=i) for i in range(5)]

        guard, out = asyncio.run(run())
        self.assertTrue(guard.tripped)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(guard.waited_seconds, 0.0)

    def test_retrieval_judge_marks_unjudged_rows(self):
        async def judge(q, r, passages):
            raise httpx.ConnectError("down")

        rows = [{"prompt": f"Q{i}", "reference": "ref", "passages": ["p"]} for i in range(5)]
        snap = asyncio.run(judge_retrieval(rows, judge, label="ret", resilience=_cfg()))
        self.assertEqual(snap["unjudged_reasons"], {"transient_error": 3, "judge_unreachable": 2})
        self.assertEqual(rows[4]["row_retrieval_status"], "unjudged")
        self.assertEqual(rows[4]["row_retrieval_unjudged_reason"], "judge_unreachable")

    def test_unguarded_callers_keep_the_old_march(self):
        """The guard is opt-in: a caller that passes no ``resilience`` (the teacher-answer filter, Auto-RAG,
        pipeline eval) still asks the judge about every row, and the teacher filter still never drops an unjudged answer."""
        from app.services.teacher_passage_answers_service import filter_drops

        calls = {"n": 0}

        async def judge(q, r, p):
            calls["n"] += 1
            raise httpx.ConnectError("down")

        rows = _rows(10)
        snap = asyncio.run(judge_predictions(rows, judge, label="teacher"))
        self.assertEqual(calls["n"], 10)
        self.assertEqual(snap["judge_calls"], 10)
        self.assertEqual(snap["status"], "partial", "never paused without a guard")
        self.assertNotIn("breaker", snap)
        self.assertTrue(all(not filter_drops(r.get("row_judge_verdict")) for r in rows))

    def test_no_failure_run_is_unchanged(self):
        async def judge(q, r, p):
            return 1.0, "correct", "ok", 3

        plain, guarded = _rows(4), _rows(4)
        a = asyncio.run(judge_predictions(plain, judge, label="ok"))
        b = asyncio.run(judge_predictions(guarded, judge, label="ok", resilience=_cfg()))
        self.assertEqual(a, b)
        self.assertEqual(plain, guarded)
        for row in guarded:
            self.assertNotIn("row_judge_status", row)
        self.assertNotIn("status", b)


async def _no_sleep(seconds):
    return None


class _Clock:
    """A fake monotonic clock; ``sleep`` advances it."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


class _FakeHandle:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.last_message = None
        self.messages: list[str] = []
        self.judge_wait_spent_seconds = 0.0

    async def set_message(self, message):
        self.messages.append(message)
        self.last_message = message

    async def check_cancelled(self) -> bool:
        return False


class _job_handle:
    """``with _job_handle() as handle:`` inside a running loop: a Job handle owned by that loop."""

    def __enter__(self):
        from app.services import jobs_service

        self.handle = _FakeHandle()
        self.token = jobs_service.current_job_handle.set(self.handle)
        return self.handle

    def __exit__(self, *exc):
        from app.services import jobs_service

        jobs_service.current_job_handle.reset(self.token)
        return False


def _guard_clock(clock: _Clock):
    """Every guard built inside gets the fake clock + sleep."""
    original = jcg.JudgeCallGuard.__post_init__

    def __post_init__(self):
        original(self)
        self.clock = clock
        self.sleep = clock.sleep

    return mock.patch.object(jcg.JudgeCallGuard, "__post_init__", __post_init__)


class ThreadModeTests(unittest.TestCase):
    def test_sync_guard_trips_without_touching_a_handle(self):
        from app.services import jobs_service

        handle = mock.MagicMock()
        token = jobs_service.current_job_handle.set(handle)
        try:
            guard = jcg.JudgeCallGuard(label="thread", resilience=jcg.resolve_resilience(
                {"eval_judge": {"resilience": {"mode": "wait"}}}, threaded=True,
            ), threaded=True)
            self.assertEqual(guard.resilience.mode, "stop", "thread paths never wait")
            calls = {"n": 0}

            def fn():
                calls["n"] += 1
                raise httpx.ConnectError("down")

            results = []

            def work():
                guard.sleep = lambda s: None
                for i in range(6):
                    results.append(guard.call_sync(fn, row=i).reason)

            t = threading.Thread(target=work)
            t.start()
            t.join()
        finally:
            jobs_service.current_job_handle.reset(token)
        self.assertEqual(results, ["transient_error"] * 3 + ["judge_unreachable"] * 3)
        self.assertLessEqual(calls["n"], 9)
        self.assertEqual(handle.method_calls, [], "the thread guard never touches the Job handle")

    def test_cancel_event_stops_a_thread_loop(self):
        event = threading.Event()
        event.set()
        guard = jcg.JudgeCallGuard(label="c", resilience=_cfg(), threaded=True, cancel_event=event)
        with self.assertRaises(jcg.JudgeCancelled):
            guard.call_sync(lambda: "x")


class ResilienceSettingsTests(unittest.TestCase):
    def test_project_block_overrides_and_clamps(self):
        cfg = jcg.resolve_resilience({"eval_judge": {"resilience": {"breaker": False, "mode": "wait", "max_wait_seconds": 10 ** 9}}})
        self.assertFalse(cfg.breaker)
        self.assertEqual(cfg.mode, "wait")
        self.assertEqual(cfg.max_wait_seconds, jcg.MAX_WAIT_CAP_SECONDS)

    def test_wait_cap_stays_under_the_queue_visibility_timeout(self):
        self.assertEqual(jcg.max_wait_cap_seconds(), jcg.settings.CELERY_VISIBILITY_TIMEOUT_SECONDS / 2)
        self.assertLess(jcg.MAX_WAIT_CAP_SECONDS, jcg.settings.CELERY_VISIBILITY_TIMEOUT_SECONDS)
        with mock.patch.object(jcg.settings, "CELERY_VISIBILITY_TIMEOUT_SECONDS", 3600):
            self.assertEqual(jcg.max_wait_cap_seconds(), 1800.0)
            self.assertEqual(jcg.normalize_project_resilience({"max_wait_seconds": 28800})["max_wait_seconds"], 1800.0)
        self.assertEqual(jcg.normalize_project_resilience({"mode": "bogus", "x": 1}), {})

    def test_env_kill_switch(self):
        with mock.patch.object(jcg.settings, "JUDGE_BREAKER_ENABLED", False):
            self.assertFalse(jcg.resolve_resilience({}).breaker)

    def test_snapshot_incomplete(self):
        self.assertTrue(jcg.snapshot_incomplete({"status": "paused"}))
        self.assertTrue(jcg.snapshot_incomplete({"judged": 0, "unjudged": 5}))
        self.assertTrue(jcg.snapshot_incomplete({"judged": 5, "unjudged": 1, "unjudged_reasons": {"transient_error": 1}}))
        self.assertFalse(jcg.snapshot_incomplete({"judged": 5, "unjudged": 1, "unjudged_reasons": {"unparseable": 1}}))
        self.assertFalse(jcg.snapshot_incomplete({"judged": 5, "unjudged": 0}))
        self.assertTrue(jcg.snapshot_incomplete({"status": "judged", "judged": 5, "unjudged": 1, "breaker": {"tripped": True},
                                                 "unjudged_reasons": {"judge_unreachable": 1}}))

    def test_base_reuse_refused_only_past_the_threshold(self):
        """S5: one timed-out row of 688 does not force a GPU re-run of the base; >5 % lost, a stopped
        breaker or no verdict at all does. Unparseable rows never do."""
        block = jcg.snapshot_blocks_base_reuse
        self.assertFalse(block({"judged": 687, "unjudged": 1, "status": "partial", "unjudged_reasons": {"transient_error": 1}}))
        self.assertTrue(block({"judged": 600, "unjudged": 88, "status": "partial", "unjudged_reasons": {"transient_error": 88}}))
        self.assertTrue(block({"judged": 680, "unjudged": 8, "status": "paused", "unjudged_reasons": {"judge_unreachable": 8}}))
        self.assertTrue(block({"judged": 0, "unjudged": 5}))
        self.assertFalse(block({"judged": 400, "unjudged": 288, "status": "partial", "unjudged_reasons": {"unparseable": 288}}))
        self.assertFalse(block({"judged": 10, "unjudged": 2}), "pre-reason block with verdicts")
        self.assertEqual(jcg.BASE_REUSE_MAX_OUTAGE_SHARE, 0.05)

    def test_warning_only_for_an_outage_on_new_results(self):
        """M2: no warning for a pre-A1 block, a rubric block (status "judged") with only unparseable rows,
        or an unparseable-only answer block; a warning for an outage on any guarded judge."""
        pre_a1 = {"judge": {"judge": "j", "judged": 8, "unjudged": 2, "score": 0.5}}
        rubric_unparseable = {"rubric": {"status": "judged", "judge": "j", "judged": 9, "unjudged": 1,
                                         "unjudged_reasons": {"unparseable": 1}}}
        unparseable = {"judge": {"judge": "j", "judged": 685, "unjudged": 3, "status": "partial",
                                 "unjudged_reasons": {"unparseable": 3}}}
        for metrics in (pre_a1, rubric_unparseable, unparseable):
            self.assertIsNone(jcg.incomplete_summary(metrics), metrics)
            self.assertEqual(jcg.job_result_fields(metrics), {})
        rubric_outage = {"rubric": {"status": "judged", "judge": "j", "judged": 2, "unjudged": 8,
                                    "unjudged_reasons": {"transient_error": 3, "judge_unreachable": 5},
                                    "breaker": {"tripped": True}}}
        fields = jcg.job_result_fields(pre_a1, rubric_outage)
        self.assertEqual(fields["judge_status"], "paused")
        self.assertEqual(fields["judge_kind"], "rubric judge")
        dialogue = {"dialogue_judge": {"judge": "d", "turns_judged": 4, "unjudged": 2, "status": "partial",
                                       "unjudged_reasons": {"transient_error": 2}}}
        self.assertEqual(jcg.incomplete_summary(dialogue)["judged"], 4)

    def test_unjudged_notice_block_prefers_an_outage(self):
        metrics = {
            "judge": {"judge": "a", "judged": 9, "unjudged": 1, "status": "partial", "unjudged_reasons": {"unparseable": 1}},
            "code_example_judge": {"judge": "a", "judged": 2, "unjudged": 3, "status": "paused",
                                   "unjudged_reasons": {"judge_unreachable": 3}, "breaker": {"tripped": True}},
        }
        block = jcg.unjudged_notice_block(metrics)
        self.assertEqual((block["kind"], block["status"], block["unjudged"]), ("code_example_judge", "paused", 3))
        self.assertEqual(jcg.unjudged_notice_block({"judge": metrics["judge"]})["kind"], "judge")
        self.assertIsNone(jcg.unjudged_notice_block({"judge": {"judged": 9, "unjudged": 1}}))


if __name__ == "__main__":
    unittest.main()

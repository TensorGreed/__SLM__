"""Playbook synthesis through Anthropic Message Batches (``use_batch``).

Chunked playbooks: batch on/off parity of accepted rows, batch requests left
unanswered (or a failed batch) fall back to direct calls, a killed run resumes
from its checkpoint without resubmitting finished chunks, a run under
``playbook_rows_per_call`` still batches (one request), a dry run never batches.
scenario_grid: both stages as one batch each, same rows as direct, unanswered
requests redone directly. Spend: the real ``CloudLlmBackend`` over an httpx
MockTransport (never the network) records batch usage once, at the batch price.
API: 400 ``invalid_playbook_param`` for a non-Anthropic pin, the dry run accepts
and ignores the flag, the Job title says " · batch".

pytest-only (the Jobs these tests start are awaited in tearDown).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services.synth_backends import TemplateBackend
from app.services.synth_backends.base import SynthBackendError
from app.services.synth_backends.template import code_findings_rows, structured_plan_rows
from app.services.synth_playbooks import SynthMode, get_playbook
from test_code_findings_playbooks import TAXONOMY, clean_row, make_ctx
from test_relabel_batch import FakeAnthropic, _patched

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def _reply(prompt: str) -> str:
    """The template teacher's reply for a playbook prompt (deterministic per prompt)."""
    if "scenario_grid:scenarios]" in prompt:
        return structured_plan_rows(prompt, "scenarios")
    if "scenario_grid:answers]" in prompt:
        return structured_plan_rows(prompt, "answers")
    return code_findings_rows(prompt, "seeded_defect")


class FakeBatchBackend:
    """Batch-capable teacher (no network). ``unanswered``: indices of the
    batch's requests that come back None; ``batch_fails``: the batch raises;
    ``explode_on_direct``: the n-th direct call raises KeyboardInterrupt (a kill)."""

    supports_batch = True
    supports_prompt_cache = True
    max_output_tokens = 16000

    def __init__(self, *, use_batch=True, rows_per_call=3, unanswered=(), batch_fails=False,
                 explode_on_direct=None, name="fake-batch"):
        self.use_batch = use_batch
        if rows_per_call is not None:
            self.playbook_rows_per_call = rows_per_call
        self.unanswered = set(unanswered)
        self.batch_fails = batch_fails
        self.explode_on_direct = explode_on_direct
        self.name = name
        self.batches: list[list[dict]] = []
        self.direct: list[str] = []
        self.progress: list[tuple[int, int]] = []
        self.batch_report: list[dict] = []
        self.usage = {"calls": 0}
        self.batch_usage = {"calls": 0}

    def describe(self):
        return self.name

    async def complete_batch(self, requests, *, progress=None):
        self.batches.append(list(requests))
        if self.batch_fails:
            raise SynthBackendError("batch endpoint unavailable")
        if progress is not None:
            await progress("waiting", 0, len(requests))
            await progress("waiting", len(requests), len(requests))
        out = [None if i in self.unanswered else _reply(r["prompt"]) for i, r in enumerate(requests)]
        answered = sum(1 for r in out if r is not None)
        self.batch_report.append({"batch_id": f"b{len(self.batches)}", "requests": len(requests),
                                  "succeeded": answered, "failed": len(requests) - answered})
        return out

    async def complete(self, prompt, **kw):
        self.direct.append(prompt)
        if self.explode_on_direct == len(self.direct):
            raise KeyboardInterrupt("run killed")
        return _reply(prompt)


def _payloads(rows):
    return [json.dumps(r["payload"], sort_keys=True) for r in rows]


# ── chunked playbooks (service) ─────────────────────────────────────────────


class ChunkedBatchTests(unittest.TestCase):
    PROJECT_ID = 987451

    def tearDown(self):
        shutil.rmtree(Path(settings.DATA_DIR) / "projects" / str(self.PROJECT_ID), ignore_errors=True)

    def run_it(self, backend, target=7, batch=None):
        from app.services.synth_playbook_service import _run_chunked

        gold = [clean_row(code=f"def f{i}(x):\n    return x + {i}", path=f"mod{i}.py") for i in range(8)]
        ctx = make_ctx(SynthMode.SEEDED_DEFECT, gold=gold, params={"label_id": "injection.query"},
                       target_count=target, project_id=self.PROJECT_ID)
        seen = []

        async def progress(fraction, message):
            seen.append(message)

        with patch.dict(os.environ, {"BREWSLM_TEMPLATE_SYNTH": "1"}):
            out = asyncio.run(_run_chunked(
                get_playbook("code-findings", SynthMode.SEEDED_DEFECT), ctx, SynthMode.SEEDED_DEFECT, backend,
                backend.playbook_rows_per_call, progress=progress, recipe_id="code-findings", target_class=None,
                failure_cluster=None, batch=backend.use_batch if batch is None else batch))
        return ctx, out, seen

    def test_batch_and_direct_accept_the_same_rows(self):
        direct = FakeBatchBackend(use_batch=False, name="fake-direct")
        _c1, (_p, _r, rows_direct, _e, meta_direct, _path), _s = self.run_it(direct)
        batched = FakeBatchBackend(name="fake-batched")
        ctx, (_p, _r, rows_batch, _e, meta, _path), seen = self.run_it(batched)
        self.assertEqual(len(direct.direct), 3)
        self.assertEqual((len(batched.batches), len(batched.batches[0]), batched.direct), (1, 3, []))
        self.assertEqual(_payloads(rows_batch), _payloads(rows_direct))
        self.assertEqual(len(rows_batch), 7)
        self.assertEqual((meta["chunks"], meta["failed_chunks"], meta["batch_requests"], meta["batch_answered"]),
                         (3, [], 3, 3))
        self.assertNotIn("batch_requests", meta_direct)
        self.assertTrue(any("Batch: 3/3 requests done" in m for m in seen), seen)
        # the request carries what a direct call sends (system prompt as the cached block, sized budget)
        request = batched.batches[0][0]
        self.assertEqual((request["system_prompt"], request["max_tokens"]), ("", 3 * 1024))
        self.assertTrue(request["cache_prefix"])

    def test_unanswered_batch_requests_are_called_directly(self):
        backend = FakeBatchBackend(unanswered={1})
        _ctx, (_p, _r, rows, _e, meta, _path), _s = self.run_it(backend)
        self.assertEqual(len(backend.direct), 1)
        self.assertIn("Batch 2 of 3", backend.direct[0])
        self.assertEqual(len(rows), 7)
        self.assertEqual((meta["batch_requests"], meta["batch_answered"], meta["failed_chunks"]), (3, 2, []))

    def test_a_failed_batch_falls_back_to_direct_calls(self):
        backend = FakeBatchBackend(batch_fails=True)
        _ctx, (_p, _r, rows, _e, meta, _path), _s = self.run_it(backend)
        self.assertEqual(len(backend.direct), 3)
        self.assertEqual(len(rows), 7)
        self.assertIn("batch endpoint unavailable", meta["batch_error"])

    def test_resume_after_a_kill_reuses_checkpointed_chunks_and_does_not_resubmit_them(self):
        killed = FakeBatchBackend(unanswered={1}, explode_on_direct=1)
        with self.assertRaises(KeyboardInterrupt):
            self.run_it(killed)
        self.assertEqual(len(killed.batches[0]), 3)
        again = FakeBatchBackend()
        _ctx, (_p, _r, rows, _e, meta, _path), seen = self.run_it(again)
        # chunk 1 was checkpointed before the kill: only chunks 2 and 3 are resubmitted
        self.assertEqual(len(again.batches), 1)
        self.assertEqual(len(again.batches[0]), 2)
        self.assertFalse(any("Batch 1 of 3" in r["prompt"] for r in again.batches[0]))
        self.assertEqual((meta["resumed_chunks"], meta["batch_requests"]), (1, 2))
        self.assertEqual(len(rows), 7)
        self.assertTrue(any(m.startswith("Resuming: 1/3") for m in seen), seen)

    def test_everything_checkpointed_submits_no_batch(self):
        self.run_it(FakeBatchBackend())
        again = FakeBatchBackend()
        _ctx, (_p, _r, rows, _e, meta, _path), _s = self.run_it(again)
        self.assertEqual((again.batches, again.direct, meta["resumed_chunks"], len(rows)), ([], [], 3, 7))


class BatchCancelTests(unittest.TestCase):
    def test_a_job_cancel_while_waiting_cancels_the_batch(self):
        import httpx

        from app.services import jobs_service
        from app.services.synth_backends.cloud_llm import CloudLlmBackend
        from app.services.synth_playbook_service import _complete_as_batch

        fake = FakeAnthropic(polls_before_end=50)
        seen: list[str] = []

        def handler(request):
            seen.append(f"{request.method} {request.url.path}")
            if request.url.path.endswith("/cancel"):
                return httpx.Response(200, json={"id": "msgbatch_1", "processing_status": "canceling"})
            return fake(request)

        class Handle:
            checks = 0

            async def check_cancelled(self):
                Handle.checks += 1
                return Handle.checks > 2  # cancelled after the batch was submitted and polled once

        backend = CloudLlmBackend(provider="anthropic", model="claude-haiku-4-5", api_key="sk-test")
        backend.use_batch = True

        async def go():
            token = jobs_service.current_job_handle.set(Handle())
            try:
                await _complete_as_batch(backend, [{"prompt": "p", "system_prompt": "s", "max_tokens": 10}],
                                         None, start=0.0, width=1.0, what="x")
            finally:
                jobs_service.current_job_handle.reset(token)

        with _patched(handler), patch.dict(os.environ, {"ANTHROPIC_BATCH_POLL_SECONDS": "0"}):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(go())
        self.assertIn("POST /v1/messages/batches/msgbatch_1/cancel", seen)
        self.assertLess(fake.polls, 50)


# ── run_playbook + API on a code-findings project ───────────────────────────


class ProjectBase(unittest.TestCase):
    RECIPE = "code-findings"

    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls._patches = [
            patch("app.services.synth_backends.BACKEND_REGISTRY", [TemplateBackend]),
            patch("app.services.synth_playbook_service.BACKEND_REGISTRY", [TemplateBackend]),
            patch.dict("os.environ", {"BREWSLM_TEMPLATE_SYNTH": "1", "ANTHROPIC_BATCH_POLL_SECONDS": "0"}),
        ]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        settings.AUTH_ENABLED = cls._prev_auth

    def tearDown(self):
        wait_for_jobs_idle(client)

    def setUp(self):
        r = client.post("/api/projects", json={"name": f"pb-batch-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        self.pdir = Path(settings.DATA_DIR) / "projects" / str(self.pid)
        self.base = f"/api/projects/{self.pid}/synthetic/run-playbook"
        self.configure()

    def configure(self):
        resp = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "code-findings"})
        self.assertEqual(resp.status_code, 200, resp.text)
        put = client.put(f"/api/projects/{self.pid}/label-taxonomy", json={"entries": TAXONOMY["entries"]})
        self.assertEqual(put.status_code, 200, put.text)
        self.add_gold([clean_row(code=f"def f{i}(x):\n    return x + {i}", path=f"mod{i}.py") for i in range(8)])

    def add_gold(self, rows):
        from app.database import async_session_factory
        from app.models.dataset import Dataset, DatasetType

        path = self.pdir / "test_inputs" / "dev.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        async def go():
            async with async_session_factory() as db:
                db.add(Dataset(project_id=self.pid, name="dev", dataset_type=DatasetType.GOLD_DEV,
                               file_path=str(path), record_count=len(rows)))
                await db.commit()

        asyncio.run(go())

    def direct_run(self, backend, mode=SynthMode.SEEDED_DEFECT, count=2, params=None, **kw):
        from app.database import async_session_factory
        from app.services.synth_playbook_service import run_playbook

        async def go():
            async with async_session_factory() as db:
                kw.setdefault("use_batch", bool(getattr(backend, "use_batch", False)))
                result = await run_playbook(db, self.pid, mode, target_count=count, backend_override=backend,
                                            params=params or {"label_id": "injection.query"}, **kw)
                await db.commit()
                return result

        return asyncio.run(go())


class RunPlaybookBatchTests(ProjectBase):
    def test_a_run_under_rows_per_call_still_batches_one_request(self):
        backend = FakeBatchBackend(rows_per_call=10)
        result = self.direct_run(backend, count=2)
        self.assertEqual((len(backend.batches), len(backend.batches[0]), backend.direct), (1, 1, []))
        self.assertEqual((result["chunks"], len(result["rows"])), (1, 2))
        self.assertEqual(result["batch"]["requests"], 1)

    def test_a_batching_backend_without_rows_per_call_batches_the_whole_target(self):
        backend = FakeBatchBackend(rows_per_call=None)
        result = self.direct_run(backend, count=3)
        self.assertEqual((len(backend.batches), len(backend.batches[0])), (1, 1))
        self.assertEqual(len(result["rows"]), 3)

    def test_a_backend_that_batches_by_default_is_not_batched_unless_asked(self):
        backend = FakeBatchBackend()  # use_batch on, like the command backend's packed calls
        result = self.direct_run(backend, count=4, use_batch=False)
        self.assertEqual((backend.batches, len(backend.direct)), ([], 2))
        self.assertNotIn("batch", result)

    def test_use_batch_with_a_backend_that_cannot_batch_is_an_error(self):
        backend = FakeBatchBackend(use_batch=False)
        with self.assertRaisesRegex(ValueError, "use_batch needs"):
            self.direct_run(backend, count=2, use_batch=True)

    def test_dry_run_never_batches(self):
        backend = FakeBatchBackend()
        result = self.direct_run(backend, count=1, dry_run=True)
        self.assertEqual((backend.batches, len(backend.direct)), ([], 1))
        self.assertNotIn("batch", result)

    def test_spend_is_recorded_once_at_the_batch_price(self):
        from app.services import llm_budget_guard, llm_pricing, llm_usage
        from app.services.synth_backends.cloud_llm import CloudLlmBackend

        fake = FakeAnthropic()
        fake.results_for = lambda requests: "\n".join(
            json.dumps({"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": {
                "content": [{"type": "text", "text": _reply(r["params"]["messages"][0]["content"])}],
                "model": "claude-haiku-4-5", "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 500, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0}}}})
            for r in requests
        )
        backend = CloudLlmBackend(provider="anthropic", model="claude-haiku-4-5", api_key="sk-test")
        backend.use_batch = True
        notes, records = [], []
        real_note = llm_budget_guard.note_usage

        def note(*a, **kw):
            notes.append(kw.get("batch"))
            return real_note(*a, **kw)

        with _patched(fake), patch.object(llm_budget_guard, "note_usage", note), \
                patch.object(llm_usage, "record", lambda *a, **kw: records.append(a)), \
                patch.object(settings, "SYNTH_CLOUD_PLAYBOOK_ROWS_PER_CALL", 2):
            result = self.direct_run(backend, count=4)
        self.assertEqual(len(fake.batches), 1)
        self.assertEqual(len(fake.batches[0]), 2)
        self.assertEqual(fake.sync_calls, [])
        self.assertEqual(notes, [True], "batch usage enters the budget ledger once, at the batch price")
        self.assertEqual(records, [], "never also through llm_usage.record (double count)")
        expected = llm_pricing.estimate_cost("claude-haiku-4-5", backend.batch_usage, batch=True)
        self.assertEqual(backend.batch_usage["input_tokens"], 2000)
        self.assertEqual(len(result["usage"]), 1)
        self.assertAlmostEqual(result["cost_usd"], round(expected["usd"], 4))
        self.assertEqual(result["tokens"], expected["total_tokens"])
        full = llm_pricing.estimate_cost("claude-haiku-4-5", backend.batch_usage)
        self.assertAlmostEqual(expected["usd"], full["usd"] * llm_pricing.BATCH_DISCOUNT)

    def test_budget_is_checked_before_the_batch_is_submitted(self):
        from app.services import llm_budget_guard
        from app.services.synth_backends.cloud_llm import CloudLlmBackend

        fake = FakeAnthropic()
        backend = CloudLlmBackend(provider="anthropic", model="claude-haiku-4-5", api_key="sk-test")
        backend.use_batch = True

        async def over(*a, **kw):
            raise llm_budget_guard.LlmBudgetExceeded("project", 1.0, 1.5)

        with _patched(fake), patch.object(llm_budget_guard, "acheck_budget", over):
            with self.assertRaises(llm_budget_guard.LlmBudgetExceeded):
                self.direct_run(backend, count=2)
        self.assertEqual(fake.batches, [])


class RunPlaybookApiTests(ProjectBase):
    def test_use_batch_on_a_non_anthropic_pin_is_400(self):
        for backend in ("ollama:some-model", "cloud:openai:gpt-4o-mini", None):
            for url in (self.base, self.base + "?async_job=true"):
                resp = client.post(url, json={"mode": "seeded_defect", "target_count": 2, "backend": backend,
                                              "params": {"label_id": "injection.query"}, "use_batch": True})
                self.assertEqual(resp.status_code, 400, (backend, url, resp.text))
                detail = resp.json()
                self.assertEqual(detail["error_code"], "invalid_playbook_param")
                self.assertEqual(detail["metadata"]["param"], "use_batch")
                self.assertIn("cloud:anthropic:", detail["message"])

    def test_dry_run_accepts_and_ignores_use_batch(self):
        resp = client.post(self.base + "/dry-run", json={"mode": "seeded_defect", "target_count": 5,
                                                          "params": {"label_id": "injection.query"}, "use_batch": True})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["ok"], resp.text)

    def test_async_job_batches_and_says_so_in_title_and_result(self):
        saved = client.put(f"/api/projects/{self.pid}/secrets",
                           json={"provider": "cloud_llm_anthropic", "key_name": "api_key", "value": "sk-test-key"})
        self.assertIn(saved.status_code, (200, 201), saved.text)
        fake = FakeAnthropic(fail_ids={"r1"})
        fake.results_for = lambda requests: "\n".join(
            json.dumps({"custom_id": r["custom_id"], "result": (
                {"type": "errored", "error": {"type": "error", "error": {"type": "api_error", "message": "overloaded"}}}
                if r["custom_id"] in fake.fail_ids else
                {"type": "succeeded", "message": {
                    "content": [{"type": "text", "text": _reply(r["params"]["messages"][0]["content"])}],
                    "model": "claude-haiku-4-5", "stop_reason": "end_turn",
                    "usage": {"input_tokens": 100, "output_tokens": 50}}})})
            for r in requests
        )

        def sync_reply(request):
            body = json.loads(request.content)
            fake.sync_calls.append(body)
            import httpx

            return httpx.Response(200, json={
                "content": [{"type": "text", "text": _reply(body["messages"][0]["content"])}],
                "model": "claude-haiku-4-5", "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 50}})

        def handler(request):
            if request.method == "POST" and request.url.path == "/v1/messages":
                return sync_reply(request)
            return fake(request)

        with _patched(handler), patch.object(settings, "SYNTH_CLOUD_PLAYBOOK_ROWS_PER_CALL", 2):
            resp = client.post(self.base + "?async_job=true", json={
                "mode": "seeded_defect", "target_count": 4, "backend": "cloud:anthropic:claude-haiku-4-5",
                "params": {"label_id": "injection.query"}, "use_batch": True})
            self.assertEqual(resp.status_code, 202, resp.text)
            job = resp.json()
            self.assertTrue(job["title"].endswith(" · batch"), job["title"])
            self.assertTrue(job["params"]["use_batch"])
            wait_for_jobs_idle(client)
        done = client.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(done["status"], "succeeded", done)
        result = done["result"]
        self.assertEqual(len(fake.batches), 1)
        self.assertEqual(len(fake.sync_calls), 1, "the request the batch didn't answer is redone directly")
        self.assertEqual(result["rows_generated"], 4)
        self.assertEqual((result["batch"]["requests"], result["batch"]["succeeded"], result["batch"]["fell_back"]), (2, 1, 1))
        self.assertEqual({u["backend"] for u in result["usage"]},
                         {"cloud:anthropic:claude-haiku-4-5", "cloud:anthropic:claude-haiku-4-5 (batch)"})
        self.assertAlmostEqual(result["cost_usd"], round(sum(u["usd"] for u in result["usage"]), 4))


class BatchSpendOnFailureTests(ProjectBase):
    """A batch is paid for even when the run then fails: the FAILED Job still reports it."""

    def test_use_batch_without_async_job_is_400(self):
        resp = client.post(self.base, json={"mode": "seeded_defect", "target_count": 2,
                                            "backend": "cloud:anthropic:claude-haiku-4-5",
                                            "params": {"label_id": "injection.query"}, "use_batch": True})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["metadata"], {"param": "use_batch", "async_job": False})

    def test_a_batched_run_with_zero_accepted_rows_keeps_its_spend_on_the_failed_job(self):
        saved = client.put(f"/api/projects/{self.pid}/secrets",
                           json={"provider": "cloud_llm_anthropic", "key_name": "api_key", "value": "sk-test-key"})
        self.assertIn(saved.status_code, (200, 201), saved.text)
        junk = {"content": [{"type": "text", "text": "no rows here"}], "model": "claude-haiku-4-5",
                "stop_reason": "end_turn", "usage": {"input_tokens": 1000, "output_tokens": 500}}
        fake = FakeAnthropic()
        fake.results_for = lambda requests: "\n".join(
            json.dumps({"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": junk}}) for r in requests
        )

        def handler(request):
            if request.method == "POST" and request.url.path == "/v1/messages":
                import httpx

                fake.sync_calls.append(json.loads(request.content))
                return httpx.Response(200, json=junk)
            return fake(request)

        with _patched(handler), patch.object(settings, "SYNTH_CLOUD_PLAYBOOK_ROWS_PER_CALL", 2):
            resp = client.post(self.base + "?async_job=true", json={
                "mode": "seeded_defect", "target_count": 4, "backend": "cloud:anthropic:claude-haiku-4-5",
                "params": {"label_id": "injection.query"}, "use_batch": True})
            self.assertEqual(resp.status_code, 202, resp.text)
            job = resp.json()
            wait_for_jobs_idle(client)
        done = client.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(done["status"], "failed", done)
        self.assertIn("0 accepted rows", done["error"])
        result = done["result"] or {}
        self.assertIn("cloud:anthropic:claude-haiku-4-5 (batch)", {u["backend"] for u in result.get("usage") or []}, result)
        self.assertGreater(result["cost_usd"], 0)
        self.assertEqual(result["batch"]["requests"], 2)


class FailureBlockTests(unittest.TestCase):
    USAGE = [{"backend": "x (batch)", "total_tokens": 10, "usd": 0.5}]

    def test_a_run_s_own_usage_supersedes_the_meter(self):
        from app.services import llm_usage

        exc = RuntimeError("boom")
        llm_usage.attach_result_usage(exc, {"usage": self.USAGE, "cost_usd": 0.5, "tokens": 10, "rows": []})
        with llm_usage.metered(root=True) as meter:
            self.assertEqual(llm_usage.failure_block(meter, exc), {"usage": self.USAGE, "cost_usd": 0.5, "tokens": 10})

    def test_without_one_the_meter_is_used_and_bad_results_are_ignored(self):
        from app.services import llm_usage

        exc = RuntimeError("boom")
        llm_usage.attach_result_usage(exc, {"usage": []})
        llm_usage.attach_result_usage(exc, None)
        with llm_usage.metered(root=True) as meter:
            self.assertEqual(llm_usage.failure_block(meter, exc), meter.block())


# ── scenario_grid ───────────────────────────────────────────────────────────


MD20 = "| region | tier |\n|---|---|\n" + "\n".join(
    f"| r{i} | {'t' + str(i) if i < 4 else ''} |" for i in range(5)
)


class ScenarioGridBatchTests(ProjectBase):
    def configure(self):
        from test_structured_plan_playbooks import FILES

        from app.database import async_session_factory
        from app.models.project import Project

        async def go():
            async with async_session_factory() as db:
                project = await db.get(Project, self.pid)
                project.selected_recipe = {"recipe_id": "structured-plan"}
                project.runtime_config = {
                    **(project.runtime_config or {}),
                    "structured_plan": {"schema_files": FILES, "root_schema": "plan.schema.json"},
                }
                await db.commit()

        asyncio.run(go())

    def grid(self, backend):
        return self.direct_run(backend, mode=SynthMode.SCENARIO_GRID, count=20,
                               params={"matrix_text": MD20, "max_cells": 14})

    def test_both_stages_batch_and_accept_the_same_rows_as_direct(self):
        direct = FakeBatchBackend(use_batch=False, rows_per_call=None)
        rows_direct = self.grid(direct)["rows"]
        batched = FakeBatchBackend(rows_per_call=None)
        result = self.grid(batched)
        self.assertEqual(batched.direct, [])
        self.assertEqual(len(batched.batches), 2, "stage 1 (scenarios) and stage 2 (plans): one batch each")
        self.assertTrue(all("scenario_grid:scenarios]" in r["prompt"] for r in batched.batches[0]))
        self.assertTrue(all("scenario_grid:answers]" in r["prompt"] for r in batched.batches[1]))
        self.assertEqual(len(batched.batches[0]), len([p for p in direct.direct if "scenario_grid:scenarios]" in p]))
        self.assertEqual(len(batched.batches[1]), len([p for p in direct.direct if "scenario_grid:answers]" in p]))
        self.assertEqual(len(result["rows"]), 14)
        self.assertEqual(_payloads(result["rows"]), _payloads(rows_direct))
        # per-call max_tokens sizing kept: 2 plans per answer call (16000 // 6000) × 6000
        self.assertEqual({r["max_tokens"] for r in batched.batches[1]}, {12000})

    def test_unanswered_grid_requests_are_called_directly(self):
        backend = FakeBatchBackend(rows_per_call=None, unanswered={0})
        result = self.grid(backend)
        self.assertEqual(len(backend.direct), 2, "the first scenarios call and the first plans call")
        self.assertIn("scenario_grid:scenarios]", backend.direct[0])
        self.assertIn("scenario_grid:answers]", backend.direct[1])
        self.assertEqual(len(result["rows"]), 14)


if __name__ == "__main__":
    unittest.main()

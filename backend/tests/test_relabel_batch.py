"""Anthropic Message Batches for span relabels: create / poll / results
over httpx (MockTransport — never the network), a failed request falling
back to the synchronous path, the half-price cost, the relabel end to end
with a batch-capable fake, and the endpoint's validation."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.services import cloud_llm_service as cloud
from app.services import llm_pricing
from app.services import span_relabel_service as relabel
from app.services.synth_backends.cloud_llm import CloudLlmBackend

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


_EMAIL = re.compile(r"[\w.]+@[\w.]+\.\w+")


def _label_reply(user_prompt: str) -> str:
    rows = []
    for m in re.finditer(r"\[(\d+)\] ([^\n]*)", user_prompt):
        rows.append({"index": int(m.group(1)), "entities": [{"type": "email", "text": e} for e in _EMAIL.findall(m.group(2))]})
    return json.dumps({"rows": rows})


def _message(text: str, inp: int = 100, out: int = 20) -> dict:
    return {"content": [{"type": "text", "text": text}], "model": "claude-haiku-4-5", "stop_reason": "end_turn",
            "usage": {"input_tokens": inp, "output_tokens": out, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}


class FakeAnthropic:
    """The batch endpoints + /v1/messages. ``fail_ids``: custom_ids whose
    batch result is errored; ``polls_before_end``: status polls answered
    in_progress first."""

    def __init__(self, fail_ids=(), polls_before_end=1, error_message="overloaded"):
        self.fail_ids = set(fail_ids)
        self.polls_before_end = polls_before_end
        self.error_message = error_message
        self.batches: list[list[dict]] = []
        self.sync_calls: list[dict] = []
        self.headers: list[httpx.Headers] = []
        self.polls = 0

    def results_for(self, requests):
        lines = []
        for r in requests:
            if r["custom_id"] in self.fail_ids:
                res = {"type": "errored", "error": {"type": "error", "error": {"type": "api_error", "message": self.error_message}}}
            else:
                res = {"type": "succeeded", "message": _message(_label_reply(r["params"]["messages"][0]["content"]))}
            lines.append(json.dumps({"custom_id": r["custom_id"], "result": res}))
        return "\n".join(reversed(lines))  # any order

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(request.headers)
        path = request.url.path
        n = len(self.batches)
        if request.method == "POST" and path == "/v1/messages/batches":
            self.batches.append(json.loads(request.content)["requests"])
            self.polls = 0
            return httpx.Response(200, json={"id": f"msgbatch_{n + 1}", "processing_status": "in_progress"})
        if request.method == "GET" and path == f"/v1/messages/batches/msgbatch_{n}":
            self.polls += 1
            if self.polls <= self.polls_before_end:
                return httpx.Response(200, json={"id": f"msgbatch_{n}", "processing_status": "in_progress",
                                                 "request_counts": {"processing": 2, "succeeded": 1}})
            return httpx.Response(200, json={"id": f"msgbatch_{n}", "processing_status": "ended",
                                             "results_url": f"https://api.anthropic.com/v1/messages/batches/msgbatch_{n}/results"})
        if request.method == "GET" and path == f"/v1/messages/batches/msgbatch_{n}/results":
            return httpx.Response(200, text=self.results_for(self.batches[-1]))
        if request.method == "POST" and path == "/v1/messages":
            body = json.loads(request.content)
            self.sync_calls.append(body)
            return httpx.Response(200, json=_message(_label_reply(body["messages"][0]["content"])))
        return httpx.Response(404, text=f"unexpected {request.method} {path}")


def _patched(fake):
    real = httpx.AsyncClient

    def make(*a, **kw):
        kw["transport"] = httpx.MockTransport(fake)
        return real(*a, **kw)

    return mock.patch.object(cloud.httpx, "AsyncClient", make)


async def _noop(*a):
    return None


class BatchClientTests(unittest.TestCase):
    def test_create_poll_and_fetch_results(self):
        fake = FakeAnthropic(fail_ids={"r1"}, polls_before_end=2)
        statuses = []

        async def on_status(b):
            statuses.append(b["processing_status"])

        async def _go():
            params = cloud.anthropic_message_params(model="claude-haiku-4-5", system_prompt="sys", user_prompt="[0] a@b.io",
                                                    max_tokens=100, temperature=0.0, cached_system="RULES")
            batch = await cloud.create_anthropic_batch(api_key="sk-test", requests=[
                {"custom_id": "r0", "params": params}, {"custom_id": "r1", "params": params},
            ])
            ended = await cloud.wait_for_anthropic_batch(api_key="sk-test", batch_id=batch["id"], on_status=on_status, poll_seconds=0)
            return await cloud.fetch_anthropic_batch_results(api_key="sk-test", results_url=ended["results_url"])

        with _patched(fake):
            results = asyncio.run(_go())
        self.assertEqual(statuses, ["in_progress", "in_progress", "ended"])
        self.assertEqual(cloud.batch_result_error(results["r0"]), "")
        self.assertEqual(cloud.batch_result_error(results["r1"]), "errored: overloaded")
        sent = fake.batches[0][0]["params"]
        self.assertEqual(sent["system"][-1], {"type": "text", "text": "RULES", "cache_control": {"type": "ephemeral"}})
        self.assertTrue(all(h["x-api-key"] == "sk-test" and h["anthropic-version"] for h in fake.headers))

    def test_errors_are_clear(self):
        def handler(request):
            return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

        with _patched(handler):
            with self.assertRaises(cloud.CloudLlmError) as ctx:
                asyncio.run(cloud.get_anthropic_batch(api_key="sk-secret-123", batch_id="msgbatch_x"))
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("sk-secret-123", str(ctx.exception))
        with self.assertRaises(cloud.CloudLlmError):
            asyncio.run(cloud.create_anthropic_batch(api_key="k", requests=[{"custom_id": "bad id!", "params": {}}]))


class BatchLabellingTests(unittest.TestCase):
    def test_label_texts_one_batch_and_failed_request_falls_back(self):
        texts = [f"text {i} mail u{i}@ex.io" for i in range(8)]  # 6 + 2 → two requests
        fake = FakeAnthropic(fail_ids={"r1"})
        llm = CloudLlmBackend(provider="anthropic", model="claude-haiku-4-5", api_key="sk-test")
        llm.use_batch = True
        messages = []

        async def report(fraction, message):
            messages.append(message)

        with _patched(fake), mock.patch.dict(os.environ, {"ANTHROPIC_BATCH_POLL_SECONDS": "0"}):
            labels, calls = asyncio.run(relabel.label_texts_any(llm, texts, ["email"], "", report, llm.describe()))
        self.assertEqual(len(fake.batches), 1, "every call in ONE batch")
        self.assertEqual(len(fake.batches[0]), 2)
        self.assertEqual(len(fake.sync_calls), 1, "the errored request redone synchronously")
        self.assertTrue(all(lab and lab[0]["text"] == f"u{i}@ex.io" for i, lab in enumerate(labels)))
        self.assertTrue(any("batch submitted · 2 requests · waiting" in m for m in messages))
        info = relabel.batch_info(llm)
        self.assertEqual((info["requests"], info["succeeded"], info["fell_back"]), (2, 1, 1))
        usage = relabel.usage_block(llm)
        self.assertEqual([u["batch"] for u in usage], [True, False])
        self.assertEqual(usage[0]["calls"], 1)

    def test_short_texts_and_long_pieces_share_one_batch(self):
        long_text = ("Write to a@b.io now. " * 80).strip()
        fake = FakeAnthropic()
        llm = CloudLlmBackend(provider="anthropic", model="claude-haiku-4-5", api_key="k")
        llm.use_batch = True
        with _patched(fake), mock.patch.dict(os.environ, {"ANTHROPIC_BATCH_POLL_SECONDS": "0"}):
            labels, calls = asyncio.run(relabel.label_texts_any(llm, ["short c@d.io", long_text], ["email"], "", _noop, "x"))
        self.assertEqual(len(fake.batches), 1)
        self.assertGreater(calls["pieces"], 1)
        self.assertEqual(labels[0][0]["text"], "c@d.io")
        self.assertEqual(len(labels[1]), 80)
        self.assertTrue(all(long_text[e["start"]:e["end"]] == "a@b.io" for e in labels[1]))

    def test_temperature_rejection_resubmits_once_without_it(self):
        model = f"claude-new-{time.time_ns()}"
        fake = FakeAnthropic(fail_ids={"r0"}, error_message="`temperature` is deprecated for this model")
        llm = CloudLlmBackend(provider="anthropic", model=model, api_key="k")
        llm.use_batch = True
        orig = fake.results_for

        def results_for(requests):  # the resubmitted batch (no temperature) succeeds
            if "temperature" not in requests[0]["params"]:
                fake.fail_ids = set()
            return orig(requests)

        fake.results_for = results_for
        with _patched(fake), mock.patch.dict(os.environ, {"ANTHROPIC_BATCH_POLL_SECONDS": "0"}):
            labels, _ = asyncio.run(relabel.label_texts_any(llm, ["x e@f.io"], ["email"], "", _noop, "x"))
        self.assertEqual(len(fake.batches), 2)
        self.assertNotIn("temperature", fake.batches[1][0]["params"])
        self.assertEqual(fake.sync_calls, [])
        self.assertEqual(labels[0][0]["text"], "e@f.io")

    def test_batch_cost_is_half_price(self):
        usage = {"calls": 1, "input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_tokens": 1_000_000}
        full = llm_pricing.estimate_cost("claude-sonnet-4-5", usage)
        half = llm_pricing.estimate_cost("claude-sonnet-4-5", usage, batch=True)
        self.assertAlmostEqual(half["usd"], full["usd"] * llm_pricing.BATCH_DISCOUNT)
        self.assertAlmostEqual(half["usd_uncached"], full["usd_uncached"] * 0.5)
        self.assertTrue(half["batch"])
        self.assertIn("batch", half["basis"])


class FakeBatchLabeller:
    """Batch-capable fake (no network): every batch request succeeds except
    the first, which comes back None and must be redone via ``complete``."""

    use_batch = True
    supports_batch = True
    supports_prompt_cache = True

    def __init__(self):
        self.batch_requests: list[dict] = []
        self.sync_prompts: list[str] = []
        self.batch_report: list[dict] = []
        self.usage = {"calls": 0}
        self.batch_usage = {"calls": 0}

    def describe(self):
        return "cloud:anthropic:fake-batch"

    async def complete_batch(self, requests, *, progress=None):
        self.batch_requests.extend(requests)
        if progress:
            await progress(f"batch submitted · {len(requests)} requests · waiting", 0, len(requests))
        out = [None] + [_label_reply(r["prompt"]) for r in requests[1:]]
        self.batch_usage["calls"] += len(requests) - 1
        self.batch_report.append({"batch_id": "b1", "requests": len(requests), "succeeded": len(requests) - 1, "failed": 1})
        return out

    async def complete(self, prompt, **kw):
        self.sync_prompts.append(prompt)
        self.usage["calls"] += 1
        return _label_reply(prompt)


class RelabelEndToEndTests(unittest.TestCase):
    def test_answer_key_through_a_batch(self):
        pid = client.post("/api/projects", json={"name": f"batch-{time.time_ns()}", "description": "x"}).json()["id"]
        prepared = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        texts = [f"Row {i}: write to p{i}@ex.io" for i in range(9)]
        rows = [{"text": t, "entities": [{"type": "email", "start": t.index("p"), "end": len(t)}]} for t in texts]
        (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        fake = FakeBatchLabeller()

        async def _go():
            async with async_session_factory() as db:
                return await relabel.relabel_split_as_answer_key(db, pid, split="test", backend_override=fake)

        summary = asyncio.run(_go()).as_dict()
        self.assertEqual(summary["labelled_rows"], 9)
        self.assertEqual(len(fake.batch_requests), 2, "9 texts → 2 requests in one batch")
        self.assertEqual(len(fake.sync_prompts), 1, "the failed request redone synchronously")
        self.assertTrue(summary["via_batch"])
        self.assertEqual(summary["batch"]["fell_back"], 1)
        self.assertEqual(summary["agreement"]["f1"], 1.0)
        gold = (settings.DATA_DIR / "projects" / str(pid) / "gold" / "gold_test.jsonl").read_text().splitlines()
        self.assertEqual(len(gold), 9)


class EndpointValidationTests(unittest.TestCase):
    def test_use_batch_needs_an_anthropic_pin(self):
        pid = client.post("/api/projects", json={"name": f"batchval-{time.time_ns()}", "description": "x"}).json()["id"]
        for backend in ("ollama", "cloud:openai:gpt-4o-mini", None):
            res = client.post(f"/api/projects/{pid}/synthetic/relabel-answer-key", json={"backend": backend, "use_batch": True})
            self.assertEqual(res.status_code, 400, backend)
            self.assertEqual(res.json().get("error_code"), "BATCH_NEEDS_ANTHROPIC")


if __name__ == "__main__":
    unittest.main()

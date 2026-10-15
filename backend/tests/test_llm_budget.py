"""Playground cloud-chat metering + the monthly cloud-LLM budget.

* playground chat / stream to a cloud OpenAI-compatible endpoint append a
  ``playground_chat`` entry to the project's spend ledger (OpenAI + Anthropic
  usage shapes; a stream without a usage chunk is estimated and marked);
  local providers are never metered; accounting never fails a chat.
* ``GET/PUT /projects/{id}/llm-budget`` + validation; the spend tally's
  month-to-date total and budget status; the Coach ``synthetic:llm-budget``
  nudge; ``budget_warning`` on a relabel Job start.

Every provider call goes through ``httpx.MockTransport`` — nothing leaves the
box."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.main import app
from app.services import coach_service
from app.services import llm_spend_service
from app.services import llm_usage
from app.services import playground_service

client: TestClient | None = None
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    wait_for_jobs_idle(client)
    client.__exit__(None, None, None)


def _project(name: str) -> int:
    return client.post("/api/projects", json={"name": f"{name}-{time.time_ns()}", "description": "x"}).json()["id"]


class _Provider:
    """Patch httpx.AsyncClient: JSON for ``stream: false``, SSE lines otherwise."""

    def __init__(self, *, reply: str, usage: dict | None, stream_usage: bool = True, model: str = "m"):
        self.reply, self.usage, self.stream_usage, self.model = reply, usage, stream_usage, model
        self.bodies: list[dict] = []

    def __enter__(self):
        def handler(request: httpx.Request):
            body = json.loads(request.content)
            self.bodies.append(body)
            if not body.get("stream"):
                return httpx.Response(200, json={
                    "id": "r1", "model": self.model,
                    "choices": [{"message": {"role": "assistant", "content": self.reply}, "finish_reason": "stop"}],
                    **({"usage": self.usage} if self.usage is not None else {}),
                })
            lines = []
            for i in range(0, len(self.reply), 5):
                lines.append("data: " + json.dumps({"id": "r1", "choices": [{"delta": {"content": self.reply[i:i + 5]}}]}))
            lines.append("data: " + json.dumps({"id": "r1", "choices": [{"delta": {}, "finish_reason": "stop"}]}))
            if self.stream_usage and self.usage is not None:
                lines.append("data: " + json.dumps({"id": "r1", "choices": [], "usage": self.usage}))
            lines.append("data: [DONE]")
            return httpx.Response(200, text="\n\n".join(lines) + "\n\n", headers={"content-type": "text/event-stream"})

        def make(*a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            return _REAL_ASYNC_CLIENT(*a, **kw)

        self._patch = mock.patch.object(httpx, "AsyncClient", make)
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)


def _ledger(pid: int) -> list[dict]:
    path = llm_usage.ledger_path(pid)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _chat(pid: int, *, api_url: str, model: str, stream: bool = False, provider: str = "openai_compatible"):
    body = {
        "provider": provider, "model_name": model, "api_url": api_url, "api_key": "sk-test",
        "save_history": False, "messages": [{"role": "user", "content": "What is the refund window for orders?"}],
    }
    path = f"/api/projects/{pid}/training/playground/chat" + ("/stream" if stream else "")
    resp = client.post(path, json=body)
    assert resp.status_code == 200, resp.text
    if not stream:
        return resp.json()
    events = [json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:")]
    return next(e for e in events if e.get("type") == "final")


OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/chat/completions"


class PlaygroundMeteringTests(unittest.TestCase):
    def test_openai_chat_is_ledgered_with_cached_share(self):
        pid = _project("pg-openai")
        usage = {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 400}}
        with _Provider(reply="Thirty days.", usage=usage) as prov:
            out = _chat(pid, api_url=OPENAI_URL, model="gpt-4o-mini")
        self.assertNotIn("stream_options", prov.bodies[0])
        self.assertEqual(out["reply"], "Thirty days.")
        self.assertFalse(out["llm_cost"]["estimated"])
        entries = _ledger(pid)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "playground_chat")
        u = entries[0]["usage"][0]
        self.assertEqual((u["input_tokens"], u["cache_read_tokens"], u["output_tokens"]), (600, 400, 200))
        self.assertEqual(u["backend"], "openai:gpt-4o-mini")
        self.assertIsNotNone(u["usd"])
        self.assertNotIn("estimated", entries[0])
        spend = client.get(f"/api/projects/{pid}/llm-spend").json()
        self.assertEqual([k["kind"] for k in spend["by_kind"]], ["playground_chat"])

    def test_anthropic_usage_shape_on_chat(self):
        pid = _project("pg-anthropic")
        usage = {"input_tokens": 300, "output_tokens": 50, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}
        with _Provider(reply="Thirty days.", usage=usage):
            _chat(pid, api_url=ANTHROPIC_URL, model="claude-haiku-4-5")
        u = _ledger(pid)[0]["usage"][0]
        self.assertEqual(u["backend"], "anthropic:claude-haiku-4-5")
        self.assertEqual((u["input_tokens"], u["output_tokens"], u["cache_read_tokens"], u["cache_write_tokens"]), (300, 50, 100, 20))

    def test_stream_records_the_final_usage_chunk(self):
        pid = _project("pg-stream")
        with _Provider(reply="Thirty days from delivery.", usage={"prompt_tokens": 90, "completion_tokens": 12}) as prov:
            final = _chat(pid, api_url=OPENAI_URL, model="gpt-4o-mini", stream=True)
        self.assertEqual(prov.bodies[0]["stream_options"], {"include_usage": True})
        self.assertEqual(final["reply"], "Thirty days from delivery.")
        self.assertEqual(final["llm_cost"]["tokens"], 102)
        self.assertFalse(final["llm_cost"]["estimated"])
        entries = _ledger(pid)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["usage"][0]["input_tokens"], 90)

    def test_stream_without_usage_is_estimated_and_marked(self):
        pid = _project("pg-stream-est")
        with _Provider(reply="Thirty days from delivery.", usage=None):
            final = _chat(pid, api_url=ANTHROPIC_URL, model="claude-haiku-4-5", stream=True)
        self.assertTrue(final["llm_cost"]["estimated"])
        entry = _ledger(pid)[0]
        self.assertTrue(entry["estimated"])
        self.assertGreater(entry["usage"][0]["output_tokens"], 0)
        self.assertEqual(client.get(f"/api/projects/{pid}/llm-spend").json()["estimated_operations"], 1)

    def test_local_endpoints_are_not_metered(self):
        pid = _project("pg-local")
        with _Provider(reply="Hi.", usage={"prompt_tokens": 9, "completion_tokens": 2}) as prov:
            out = _chat(pid, api_url="http://localhost:11434/v1/chat/completions", model="llama3")
            _chat(pid, api_url="http://192.168.1.20:8000/v1/chat/completions", model="llama3", stream=True)
            _chat(pid, api_url="https://gpu-box.example.com/v1/chat/completions", model="x", provider="llama_cpp")
        self.assertNotIn("stream_options", prov.bodies[1])
        self.assertNotIn("llm_cost", out)
        out = client.post(f"/api/projects/{pid}/training/playground/chat", json={
            "provider": "mock", "model_name": "m", "save_history": False, "messages": [{"role": "user", "content": "hi"}],
        }).json()
        self.assertNotIn("llm_cost", out)
        self.assertEqual(_ledger(pid), [])

    def test_accounting_failure_never_fails_the_chat(self):
        pid = _project("pg-acct-fail")
        with _Provider(reply="Thirty days.", usage={"prompt_tokens": 9, "completion_tokens": 2}), \
                mock.patch.object(llm_usage, "record", side_effect=RuntimeError("boom")):
            out = _chat(pid, api_url=OPENAI_URL, model="gpt-4o-mini")
        self.assertEqual(out["reply"], "Thirty days.")

    def test_usage_counts_pure(self):
        counts, est = playground_service.playground_usage_counts(
            {"prompt_tokens": 0, "completion_tokens": 0}, messages=[{"role": "user", "content": "x" * 40}], reply="y" * 9,
        )
        self.assertTrue(est)
        self.assertEqual((counts["input_tokens"], counts["output_tokens"]), (10, 3))
        self.assertFalse(playground_service.is_metered_chat("experiment", None))
        self.assertFalse(playground_service.is_metered_chat("ollama", "http://localhost:11434/v1"))
        self.assertTrue(playground_service.is_metered_chat("openai", OPENAI_URL))


def _spend_this_month(pid: int, *, million_input_tokens: float):
    """Ledger a haiku call: $1.00 per million input tokens."""
    with llm_usage.metered_operation(pid, "gold_llm_generate"):
        llm_usage.record("anthropic", "claude-haiku-4-5", input_tokens=int(million_input_tokens * 1_000_000))


class BudgetTests(unittest.TestCase):
    def test_normalize_and_status_pure(self):
        self.assertIsNone(llm_spend_service.normalize_budget(None))
        self.assertEqual(llm_spend_service.normalize_budget(5), {"monthly": 5.0, "warn_at": 0.8})
        for bad in ((0, None), (-1, None), ("x", None), (5, 0), (5, 1.5), (5, -0.1)):
            with self.assertRaises(ValueError):
                llm_spend_service.normalize_budget(*bad)
        b = {"monthly": 10.0, "warn_at": 0.8}
        self.assertEqual(llm_spend_service.budget_status(5, b)["status"], "ok")
        self.assertEqual(llm_spend_service.budget_status(8, b)["status"], "warning")
        self.assertEqual(llm_spend_service.budget_status(10, b)["status"], "warning")
        self.assertEqual(llm_spend_service.budget_status(10.01, b)["status"], "over")
        self.assertAlmostEqual(llm_spend_service.budget_status(2.5, b)["share"], 0.25)
        self.assertIsNone(llm_spend_service.budget_status(3, None))

    def test_budget_endpoints_and_validation(self):
        pid = _project("budget-api")
        self.assertEqual(client.get(f"/api/projects/{pid}/llm-budget").json()["budget"], None)
        out = client.put(f"/api/projects/{pid}/llm-budget", json={"monthly": 5, "warn_at": 0.5}).json()
        self.assertEqual(out["budget"], {"monthly": 5.0, "warn_at": 0.5})
        self.assertEqual(out["status"]["status"], "ok")
        self.assertEqual(client.get(f"/api/projects/{pid}/llm-budget").json()["budget"]["monthly"], 5.0)
        for bad in ({"monthly": 0}, {"monthly": -3}, {"monthly": 5, "warn_at": 0}, {"monthly": 5, "warn_at": 1.2}):
            self.assertEqual(client.put(f"/api/projects/{pid}/llm-budget", json=bad).status_code, 422, bad)
        cleared = client.put(f"/api/projects/{pid}/llm-budget", json={"monthly": None}).json()
        self.assertIsNone(cleared["budget"])
        self.assertIsNone(cleared["status"])
        self.assertEqual(client.get("/api/projects/999999/llm-budget").status_code, 404)

    def test_month_to_date_and_status_in_the_tally(self):
        pid = _project("budget-month")
        last_month = llm_spend_service.month_start() - timedelta(days=1)
        entry = {"model": "claude-haiku-4-5", "calls": 1, "input_tokens": 3_000_000, "output_tokens": 0, "total_tokens": 3_000_000, "usd": 3.0}
        path = llm_usage.ledger_path(pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps({"at": last_month.isoformat(), "kind": "span_relabel", "usage": [entry]}) + "\n")
        _spend_this_month(pid, million_input_tokens=0.9)
        client.put(f"/api/projects/{pid}/llm-budget", json={"monthly": 1.0, "warn_at": 0.8})
        body = client.get(f"/api/projects/{pid}/llm-spend?days=60").json()
        self.assertAlmostEqual(body["totals"]["cost_usd"], 3.9)
        self.assertAlmostEqual(body["month"]["cost_usd"], 0.9)
        self.assertEqual(body["month"]["operations"], 1)
        self.assertEqual(body["budget"], {"monthly": 1.0, "warn_at": 0.8})
        self.assertEqual(body["budget_status"]["status"], "warning")
        self.assertAlmostEqual(body["budget_status"]["share"], 0.9)
        # A 1-day window still reports the whole month.
        self.assertAlmostEqual(client.get(f"/api/projects/{pid}/llm-spend?days=1").json()["month"]["cost_usd"], 0.9)
        _spend_this_month(pid, million_input_tokens=0.2)
        self.assertEqual(client.get(f"/api/projects/{pid}/llm-spend").json()["budget_status"]["status"], "over")

    def test_relabel_start_carries_budget_warning_only_when_over(self):
        pid = _project("budget-relabel")

        class _Summary:
            def as_dict(self):
                return {"rows": 0}

        async def _fake(*_a, **_k):
            await asyncio.sleep(0.3)
            return _Summary()

        with mock.patch("app.services.span_relabel_service.relabel_split_as_answer_key", _fake):
            client.put(f"/api/projects/{pid}/llm-budget", json={"monthly": 1.0})
            resp = client.post(f"/api/projects/{pid}/synthetic/relabel-answer-key", json={"backend": "ollama"})
            self.assertEqual(resp.status_code, 202, resp.text)
            self.assertNotIn("budget_warning", resp.json())
            wait_for_jobs_idle(client)
            _spend_this_month(pid, million_input_tokens=1.5)
            resp = client.post(f"/api/projects/{pid}/synthetic/relabel-answer-key", json={"backend": "ollama"})
            self.assertEqual(resp.status_code, 202, resp.text)
            warning = resp.json()["budget_warning"]
            self.assertEqual(warning["status"], "over")
            self.assertIn("over its $1.00 budget", warning["message"])
            self.assertTrue(resp.json()["id"])  # the Job started anyway
            wait_for_jobs_idle(client)


class CoachBudgetNudgeTests(unittest.TestCase):
    def test_thresholds(self):
        nudge = coach_service._llm_budget_nudge
        self.assertIsNone(nudge(None))
        self.assertIsNone(nudge({"status": "ok", "spent": 1, "budget": 10, "share": 0.1}))
        warn = nudge({"status": "warning", "spent": 8.5, "budget": 10, "share": 0.85, "warn_at": 0.8})
        self.assertEqual((warn["id"], warn["severity"]), ("synthetic:llm-budget", "warning"))
        self.assertIn("85%", warn["title"])
        self.assertEqual(warn["action"]["params"], {"target": "pipeline-tab", "tab": "synthetic"})
        over = nudge({"status": "over", "spent": 12, "budget": 10, "share": 1.2, "warn_at": 0.8})
        self.assertEqual(over["severity"], "critical")
        self.assertEqual(over["rule_id"], "synthetic.llm-budget.over")

    def test_synthetic_stage_emits_it_even_without_data(self):
        project = SimpleNamespace(id=7, selected_recipe=None, runtime_config={"llm_budget_usd": {"monthly": 1.0, "warn_at": 0.8}})

        async def _status(_db, _project):
            return {"status": "over", "spent": 1.4, "budget": 1.0, "share": 1.4, "warn_at": 0.8}

        async def _no_data(*_a, **_k):
            return False

        async def _run():
            with mock.patch.object(llm_spend_service, "month_budget_status", _status), \
                    mock.patch.object(coach_service, "_project_has_any_data", _no_data):
                return await coach_service._synthetic_stage_suggestions(None, project)  # type: ignore[arg-type]

        out = asyncio.run(_run())
        self.assertEqual([s["id"] for s in out], ["synthetic:llm-budget"])

    def test_budget_read_error_is_quiet(self):
        async def _boom(*_a, **_k):
            raise RuntimeError("db down")

        async def _run():
            with mock.patch.object(llm_spend_service, "month_budget_status", _boom):
                return await coach_service._llm_budget_nudge_for(None, SimpleNamespace(id=1))  # type: ignore[arg-type]

        self.assertIsNone(asyncio.run(_run()))


if __name__ == "__main__":
    unittest.main()

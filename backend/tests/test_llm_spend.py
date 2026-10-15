"""Cloud-LLM cost beyond relabels: usage metering at every cloud call site,
prompt caching of the stable instruction blocks, Job-result usage, the
non-Job spend ledger and the project spend tally (GET /llm-spend).

Every provider call goes through ``httpx.MockTransport`` — nothing leaves the
box."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.job import Job, JobStatus
from app.services import answer_judge_service as judge_svc
from app.services import cloud_llm_service as cloud
from app.services import documents_qa_flow_service as dqa
from app.services import evaluation_service
from app.services import jobs_service
from app.services import llm_spend_service
from app.services import llm_usage
from app.services import pipeline_refinement_service as refine
from app.services import synthetic_service
from app.services.distillation import teacher_capture
from app.services.synth_backends.cloud_llm import CloudLlmBackend

client: TestClient | None = None
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def _anthropic_reply(text: str, *, inp=40, out=10, cw=0, cr=0):
    return {
        "content": [{"type": "text", "text": text}], "model": "claude-haiku-4-5", "stop_reason": "end_turn",
        "usage": {"input_tokens": inp, "output_tokens": out, "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr},
    }


class _Transport:
    """Patch httpx.AsyncClient so every request hits ``reply(body, url)``."""

    def __init__(self, reply):
        self.reply = reply
        self.bodies: list[dict] = []
        self.urls: list[str] = []

    def __enter__(self):
        def handler(request: httpx.Request):
            body = json.loads(request.content)
            self.bodies.append(body)
            self.urls.append(str(request.url))
            return httpx.Response(200, json=self.reply(body, str(request.url)))

        def make(*a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            return _REAL_ASYNC_CLIENT(*a, **kw)

        self._patch = mock.patch.object(httpx, "AsyncClient", make)
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)


def _cached_block(system) -> str | None:
    if isinstance(system, list):
        for block in system:
            if block.get("cache_control"):
                return block["text"]
    return None


def _project(name: str) -> int:
    return client.post("/api/projects", json={"name": f"{name}-{time.time_ns()}", "description": "x"}).json()["id"]


class MeterTests(unittest.TestCase):
    def test_anthropic_call_is_counted_with_cache_tokens(self):
        with _Transport(lambda b, u: _anthropic_reply("ok", inp=50, out=10, cr=1500)):
            with llm_usage.metered() as meter:
                asyncio.run(cloud.call_anthropic_chat(api_key="k", model="claude-haiku-4-5", system_prompt="s", user_prompt="u"))
        block = meter.block()
        [entry] = block["usage"]
        self.assertEqual(entry["backend"], "anthropic:claude-haiku-4-5")
        self.assertEqual((entry["calls"], entry["input_tokens"], entry["cache_read_tokens"]), (1, 50, 1500))
        self.assertEqual(block["tokens"], 1560)
        self.assertIsNotNone(block["cost_usd"])

    def test_local_openai_compatible_host_is_not_cloud_spend(self):
        reply = {"choices": [{"message": {"content": "{}"}}], "model": "gpt-4o-mini", "usage": {"prompt_tokens": 9, "completion_tokens": 3}}
        with _Transport(lambda b, u: reply):
            with llm_usage.metered() as meter:
                asyncio.run(cloud.call_openai_chat(api_key="k", model="m", system_prompt="", user_prompt="u", api_url="http://localhost:11434/v1/chat/completions"))
                asyncio.run(cloud.call_openai_chat(api_key="k", model="gpt-4o-mini", system_prompt="", user_prompt="u"))
        [entry] = meter.block()["usage"]
        self.assertEqual((entry["backend"], entry["calls"], entry["total_tokens"]), ("openai:gpt-4o-mini", 1, 12))

    def test_nested_and_root_meters(self):
        with llm_usage.metered() as outer:
            with llm_usage.metered() as inner:
                llm_usage.record("anthropic", "claude-haiku-4-5", input_tokens=5)
            with llm_usage.metered(root=True) as detached:
                llm_usage.record("anthropic", "claude-haiku-4-5", input_tokens=7)
        self.assertEqual(outer.block()["tokens"], 5, "a root meter (another Job) is not counted by the outer one")
        self.assertEqual((inner.block()["tokens"], detached.block()["tokens"]), (5, 7))
        self.assertEqual(llm_usage.metered().__class__.__name__, "_GeneratorContextManager")
        llm_usage.record("anthropic", "x", input_tokens=1)  # no meter open: a no-op, never raises

    def test_merge_keeps_a_runner_reported_usage(self):
        with llm_usage.metered() as meter:
            llm_usage.record("anthropic", "claude-haiku-4-5", input_tokens=10)
        own = {"usage": [{"model": "m", "total_tokens": 3, "usd": 0.1}], "cost_usd": 0.1, "tokens": 3}
        self.assertEqual(llm_usage.merge_into_result(dict(own), meter), own)
        merged = llm_usage.merge_into_result({"usage": {"prompt_tokens": 1}}, meter)
        self.assertEqual(merged["usage"], {"prompt_tokens": 1}, "a different 'usage' field is never overwritten")
        self.assertEqual(merged["llm_usage"][0]["input_tokens"], 10)
        self.assertEqual(llm_usage.result_usage(merged)[0]["input_tokens"], 10)

    def test_metered_operation_ledgers_only_when_outermost(self):
        pid = _project("ledger")
        path = llm_usage.ledger_path(pid)
        with llm_usage.metered_operation(pid, "outer_op"):
            with llm_usage.metered_operation(pid, "inner_op"):
                llm_usage.record("anthropic", "claude-haiku-4-5", input_tokens=4)
        lines = [json.loads(l) for l in path.read_text().splitlines()]
        self.assertEqual([l["kind"] for l in lines], ["outer_op"])
        with self.assertRaises(RuntimeError):
            with llm_usage.metered_operation(pid, "failed_op"):
                llm_usage.record("anthropic", "claude-haiku-4-5", input_tokens=2)
                raise RuntimeError("boom")
        self.assertEqual(json.loads(path.read_text().splitlines()[-1])["kind"], "failed_op", "spent tokens are recorded on failure")


class JobResultTests(unittest.TestCase):
    def _run_job(self, runner, kind="test_llm_kind"):
        pid = _project("jobusage")

        async def _go():
            async with async_session_factory() as db:
                job = Job(kind=kind, title="t", status=JobStatus.QUEUED, project_id=pid, params={})
                db.add(job)
                await db.commit()
                job_id = job.id
            await jobs_service._runner_wrapper(job_id, runner)
            async with async_session_factory() as db:
                row = await db.get(Job, job_id)
                return row.status, dict(row.result or {})

        return pid, asyncio.run(_go())

    def test_succeeded_job_result_carries_usage(self):
        async def runner(handle):
            await cloud.call_anthropic_chat(api_key="k", model="claude-haiku-4-5", system_prompt="", user_prompt="u")
            return {"rows": 3}

        with _Transport(lambda b, u: _anthropic_reply("ok", inp=100, out=20)):
            _, (status, result) = self._run_job(runner)
        self.assertEqual(status, JobStatus.SUCCEEDED)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["tokens"], 120)
        self.assertEqual(result["usage"][0]["backend"], "anthropic:claude-haiku-4-5")
        self.assertIsNotNone(result["cost_usd"])

    def test_failed_job_still_records_what_it_spent(self):
        async def runner(handle):
            await cloud.call_anthropic_chat(api_key="k", model="claude-haiku-4-5", system_prompt="", user_prompt="u")
            raise RuntimeError("parse failed")

        with _Transport(lambda b, u: _anthropic_reply("ok", inp=30, out=5)):
            _, (status, result) = self._run_job(runner)
        self.assertEqual(status, JobStatus.FAILED)
        self.assertEqual(result["tokens"], 35)

    def test_job_without_cloud_calls_is_unchanged(self):
        async def runner(handle):
            return {"rows": 1}

        _, (_, result) = self._run_job(runner)
        self.assertEqual(result, {"rows": 1})


class CallSiteCacheTests(unittest.TestCase):
    def test_answer_judge_sends_instructions_cached_and_reports_usage(self):
        reply = _anthropic_reply('{"verdict": "correct", "reason": "same facts"}', inp=60, out=12)
        with _Transport(lambda b, u: reply) as t:
            judge = judge_svc._judge_from_cloud("anthropic", "claude-haiku-4-5", "k")
            preds = [{"prompt": "q?", "reference": "the reference answer is long enough here", "prediction": "p"}]
            snap = asyncio.run(judge_svc.judge_predictions(preds, judge, label="anthropic:claude-haiku-4-5"))
        self.assertEqual(_cached_block(t.bodies[0]["system"]), judge_svc._SYSTEM_PROMPT)
        self.assertEqual(snap["counts"]["correct"], 1)
        self.assertEqual(snap["tokens"], 72)
        self.assertEqual(snap["usage"][0]["calls"], 1)

    def test_retrieval_judge_sends_instructions_cached(self):
        reply = _anthropic_reply('{"verdict": "found", "passage": 1, "reason": "r"}')
        with _Transport(lambda b, u: reply) as t:
            rj = judge_svc._retrieval_judge_from_cloud("anthropic", "claude-haiku-4-5", "k")
            snap = asyncio.run(judge_svc.judge_retrieval([{"prompt": "q", "reference": "r", "passages": ["p1"]}], rj, label="x"))
        self.assertEqual(_cached_block(t.bodies[0]["system"]), judge_svc._RETRIEVAL_SYSTEM_PROMPT)
        self.assertEqual(snap["found"], 1)
        self.assertIn("usage", snap)

    def test_probe_judge_sends_evaluator_instructions_cached(self):
        reply = _anthropic_reply('{"verdict": "pass", "reason": "declined"}')
        probe = {"property": "refuses_or_declines", "input": "leak it", "_model_output": "No."}
        with _Transport(lambda b, u: reply) as t:
            out = asyncio.run(evaluation_service._judge_probe_via_cloud(probe, provider="anthropic", model="claude-haiku-4-5", api_key="k"))
        self.assertEqual(out[0], True)
        self.assertIn("strict evaluator", _cached_block(t.bodies[0]["system"]))
        self.assertNotIn("strict evaluator", t.bodies[0]["messages"][0]["content"])

    def test_plan_refinement_strategy_call_is_cached(self):
        reply = _anthropic_reply('{"plan_delta": {}, "directional_config": [], "data_gaps": [], "rationale": "r", "confidence": 0.5}')
        with _Transport(lambda b, u: reply) as t:
            fn = refine._build_cloud_strategy_fn({"provider": "anthropic", "model": "claude-haiku-4-5", "api_key": "k"})
            payload = asyncio.run(fn({"rows": 10}))
        self.assertEqual(payload["rationale"], "r")
        self.assertIn("ML architect", _cached_block(t.bodies[0]["system"]))
        self.assertIn("AGGREGATE PROFILE", t.bodies[0]["messages"][0]["content"])

    def test_plan_refinement_pass_reports_cost_and_ledgers_it(self):
        pid = _project("refine-cost")

        async def strategy(profile):
            llm_usage.record("anthropic", "claude-haiku-4-5", input_tokens=400, output_tokens=100)
            return {"plan_delta": {}, "directional_config": [], "data_gaps": [], "rationale": "fine", "confidence": 0.4}

        async def _go():
            async with async_session_factory() as db:
                return await refine.run_cloud_strategy_pass(db, pid, strategy_fn=strategy, model_label="anthropic:claude-haiku-4-5")

        out = asyncio.run(_go())
        self.assertEqual(out["llm_cost"]["tokens"], 500)
        kinds = [json.loads(l)["kind"] for l in llm_usage.ledger_path(pid).read_text().splitlines()]
        self.assertEqual(kinds, ["plan_refinement"])

    def test_gold_answer_key_caches_the_system_prompt_and_reports_cost(self):
        resp = client.post("/api/project-templates/policy-qa-style/instantiate", json={"project_name": f"gold-cost-{time.time_ns()}"})
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])
        reply = _anthropic_reply(json.dumps({"pairs": [{"question": "What is the refund window?", "answer": "Thirty days from delivery."}]}), inp=300, out=80)
        with _Transport(lambda b, u: reply) as t:
            resp = client.post(f"/api/projects/{pid}/gold/generate-via-llm", json={
                "provider": "anthropic", "model": "claude-haiku-4-5-20251001", "count": 1, "api_key": "sk-ant-test", "ground_in_source": False,
            })
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIsNotNone(_cached_block(t.bodies[0]["system"]))
        body = resp.json()
        self.assertEqual(body["llm_cost"]["tokens"], 380)
        spend = client.get(f"/api/projects/{pid}/llm-spend").json()
        self.assertEqual(spend["by_kind"][0]["kind"], "gold_llm_generate")
        self.assertEqual(spend["totals"]["total_tokens"], 380)

    def test_cloud_backend_cache_system_marks_the_system_prompt(self):
        with _Transport(lambda b, u: _anthropic_reply("doc")) as t:
            backend = CloudLlmBackend(provider="anthropic", model="claude-haiku-4-5", api_key="k")
            asyncio.run(backend.complete("write one", system_prompt="SYS", cache_system=True))
            asyncio.run(backend.complete("write one", system_prompt="SYS"))
        self.assertEqual(t.bodies[0]["system"], [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(t.bodies[1]["system"], "SYS")
        self.assertEqual(llm_usage.cache_kwargs(backend), {"cache_system": True})
        self.assertEqual(llm_usage.cache_kwargs(object()), {})

    def test_span_synthesis_sends_types_and_rules_as_the_cached_prefix(self):
        seen = []

        class Cached:
            supports_prompt_cache = True

            def describe(self):
                return "cloud:anthropic:x"

            async def complete(self, prompt, **kw):
                seen.append((prompt, kw))
                return json.dumps({"rows": []})

        class Plain(Cached):
            supports_prompt_cache = False

        src = "Contact Jane at jane@example.com about the invoice."
        asyncio.run(synthetic_service.generate_span_extraction_rows(None, 1, src, num_rows=3, entity_types=["email"], backend_override=Cached()))
        asyncio.run(synthetic_service.generate_span_extraction_rows(None, 1, src, num_rows=3, entity_types=["email"], backend_override=Plain()))
        (cached_prompt, cached_kw), (plain_prompt, plain_kw) = seen
        self.assertIn("Allowed entity types", cached_kw["cache_prefix"])
        self.assertNotIn(src, cached_kw["cache_prefix"], "the source paragraph varies — not cached")
        self.assertIn(src, cached_prompt)
        self.assertNotIn("Allowed entity types", cached_prompt)
        self.assertNotIn("cache_prefix", plain_kw)
        self.assertIn("Produce 3 new realistic snippets", plain_prompt, "uncached backends get the unchanged prompt")
        self.assertIn(src, plain_prompt)

    def test_documents_flow_prompt_parts(self):
        stable, varying = dqa.passage_prompt_parts("The fee is 10 dollars.", 3, domain_hint="fees")
        self.assertIn("Write 3 question-and-answer pairs", stable)
        self.assertNotIn("The fee is 10 dollars.", stable)
        self.assertIn("<passage>\nThe fee is 10 dollars.\n</passage>", varying)
        full = dqa.build_passage_prompt("The fee is 10 dollars.", 3, domain_hint="fees")
        self.assertTrue(full.startswith("Read this passage. The documents are about: fees."))
        self.assertIn(stable.split("\n\n", 1)[1], full, "same instructions either way")


class TeacherEndpointTests(unittest.TestCase):
    def test_cloud_teacher_counts_local_teacher_does_not(self):
        reply = {"choices": [{"message": {"content": "{}"}}], "model": "deepseek-chat", "usage": {"prompt_tokens": 70, "completion_tokens": 30, "total_tokens": 100}}
        with _Transport(lambda b, u: reply):
            with llm_usage.metered() as meter:
                asyncio.run(synthetic_service.call_teacher_model("p", api_url="https://api.deepseek.com/v1/chat/completions", api_key="k", model_name="deepseek-chat"))
                asyncio.run(synthetic_service.call_teacher_model("p", api_url="http://127.0.0.1:11434/v1/chat/completions", model_name="llama3"))
        [entry] = meter.block()["usage"]
        self.assertEqual((entry["backend"], entry["total_tokens"]), ("deepseek:deepseek-chat", 100))

    def test_distillation_teacher_counts(self):
        reply = {"choices": [{"message": {"content": "x"}, "logprobs": {"content": []}}], "model": "gpt-4o-mini", "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
        with _Transport(lambda b, u: reply):
            with llm_usage.metered() as meter:
                asyncio.run(teacher_capture.call_teacher_with_logprobs("p", api_url="https://api.openai.com/v1/chat/completions", api_key="k", model_name="gpt-4o-mini"))
        self.assertEqual(meter.block()["tokens"], 7)


class SpendTallyTests(unittest.TestCase):
    def test_tally_groups_by_model_and_kind(self):
        ops = [
            {"kind": "span_relabel", "usage": [{"model": "claude-haiku-4-5", "calls": 2, "input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "usd": 0.5}]},
            {"kind": "seed_documents", "usage": [
                {"model": "claude-haiku-4-5", "calls": 1, "total_tokens": 40, "usd": 0.25},
                {"model": "mystery", "calls": 1, "total_tokens": 9, "usd": None},
            ]},
        ]
        out = llm_spend_service.tally(ops)
        self.assertEqual(out["totals"]["operations"], 2)
        self.assertEqual(out["totals"]["calls"], 4)
        self.assertEqual(out["totals"]["total_tokens"], 159)
        self.assertAlmostEqual(out["totals"]["cost_usd"], 0.75)
        self.assertEqual(out["totals"]["unpriced_tokens"], 9)
        self.assertEqual(out["by_model"][0]["model"], "claude-haiku-4-5")
        self.assertFalse(next(r for r in out["by_model"] if r["model"] == "mystery")["priced"])
        self.assertEqual([k["kind"] for k in out["by_kind"]], ["span_relabel", "seed_documents"])

    def test_endpoint_reads_jobs_and_ledger_within_the_window(self):
        pid = _project("spend")
        other = _project("spend-other")
        entry = {"model": "claude-haiku-4-5", "calls": 3, "input_tokens": 900, "output_tokens": 100, "total_tokens": 1000, "usd": 0.01}
        old = datetime.now(timezone.utc) - timedelta(days=60)

        async def _seed():
            async with async_session_factory() as db:
                db.add(Job(kind="span_relabel", title="t", status=JobStatus.SUCCEEDED, project_id=pid,
                           result={"usage": [entry], "cost_usd": 0.01, "tokens": 1000}, completed_at=datetime.now(timezone.utc)))
                db.add(Job(kind="seed_documents", title="t", status=JobStatus.FAILED, project_id=pid,
                           result={"usage": [entry], "cost_usd": 0.01, "tokens": 1000}, completed_at=datetime.now(timezone.utc)))
                db.add(Job(kind="span_relabel", title="old", status=JobStatus.SUCCEEDED, project_id=pid,
                           result={"usage": [entry]}, completed_at=old, queued_at=old))
                db.add(Job(kind="span_relabel", title="no usage", status=JobStatus.SUCCEEDED, project_id=pid, result={"rows": 1},
                           completed_at=datetime.now(timezone.utc)))
                db.add(Job(kind="span_relabel", title="other project", status=JobStatus.SUCCEEDED, project_id=other,
                           result={"usage": [entry]}, completed_at=datetime.now(timezone.utc)))
                await db.commit()

        asyncio.run(_seed())
        with llm_usage.metered_operation(pid, "gold_llm_generate"):
            llm_usage.record("openai", "gpt-4o-mini", input_tokens=50, output_tokens=50)
        path = llm_usage.ledger_path(pid)
        with path.open("a") as fh:
            fh.write(json.dumps({"at": old.isoformat(), "kind": "gold_llm_generate", "usage": [entry]}) + "\n")
            fh.write("not json\n")

        body = client.get(f"/api/projects/{pid}/llm-spend?days=30").json()
        self.assertEqual(body["days"], 30)
        self.assertEqual(body["basis"], "list-price estimate")
        self.assertEqual(body["totals"]["operations"], 3)
        self.assertEqual(body["totals"]["total_tokens"], 2100)
        self.assertEqual({k["kind"] for k in body["by_kind"]}, {"span_relabel", "seed_documents", "gold_llm_generate"})
        self.assertEqual({m["model"] for m in body["by_model"]}, {"claude-haiku-4-5", "gpt-4o-mini"})
        wide = client.get(f"/api/projects/{pid}/llm-spend?days=90").json()
        self.assertEqual(wide["totals"]["operations"], 5)
        self.assertEqual(client.get("/api/projects/999999/llm-spend").status_code, 404)
        self.assertEqual(client.get(f"/api/projects/{pid}/llm-spend?days=0").status_code, 422)


if __name__ == "__main__":
    unittest.main()

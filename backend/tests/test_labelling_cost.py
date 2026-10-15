"""Labelling cost: prompt caching, usage + cost per relabel, convention-aware
targeted relabels and the tiered (bulk + review on disagreement) mode."""

from __future__ import annotations

import asyncio
import json
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
from app.services import span_type_definitions_service as defs

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


class PromptCacheTests(unittest.TestCase):
    def test_cached_system_block_and_cache_usage(self):
        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append(body)
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}], "model": "m", "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1500},
            })

        real = httpx.AsyncClient

        def make(*a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            return real(*a, **kw)

        with mock.patch.object(cloud.httpx, "AsyncClient", make):
            resp = asyncio.run(cloud.call_anthropic_chat(api_key="k", model="m", system_prompt="sys", user_prompt="texts", cached_system="RULES"))
        self.assertEqual(seen[0]["system"], [{"type": "text", "text": "sys"}, {"type": "text", "text": "RULES", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual((resp.cache_read_tokens, resp.prompt_tokens), (1500, 50))

    def test_label_texts_sends_the_stable_part_as_cache_prefix(self):
        calls = []

        class Cached:
            supports_prompt_cache = True

            def describe(self):
                return "cached"

            async def complete(self, prompt, **kw):
                calls.append((prompt, kw.get("cache_prefix")))
                return json.dumps({"rows": [{"index": int(m.group(1)), "entities": []} for m in re.finditer(r"\[(\d+)\] ", prompt)]})

        async def report(*a):
            return None

        asyncio.run(relabel.label_texts(Cached(), ["a text", "b text"], ["email"], "CONVENTIONS", report, "cached"))
        prompt, prefix = calls[0]
        self.assertIn("CONVENTIONS", prefix)
        self.assertNotIn("CONVENTIONS", prompt)
        self.assertIn("[0] a text", prompt)
        self.assertEqual(relabel.build_relabel_prompt(["x"], ["email"], "C"), "\n\n".join(relabel.relabel_prompt_parts(["x"], ["email"], "C")))


class PricingTests(unittest.TestCase):
    def test_estimate_counts_cache_reads_at_a_tenth(self):
        est = llm_pricing.estimate_cost("claude-haiku-4-5", {"calls": 2, "input_tokens": 1_000_000, "output_tokens": 0, "cache_read_tokens": 1_000_000})
        self.assertAlmostEqual(est["usd"], 1.0 + 0.1)
        self.assertIsNone(llm_pricing.estimate_cost("mystery-model", {"input_tokens": 5})["usd"])
        with mock.patch.dict("os.environ", {"LLM_PRICES_JSON": '{"haiku": [2, 10]}'}):
            self.assertEqual(llm_pricing.price_for("claude-haiku-4-5"), (2.0, 10.0))


class TargetingTests(unittest.TestCase):
    def test_changed_types_and_affected_rows(self):
        a = defs.conventions_snapshot({"email": "e", "bank_account": "old"})
        b = defs.conventions_snapshot({"email": "e", "bank_account": "new", "iban": "i"})
        self.assertEqual(defs.changed_types(a, b), {"bank_account", "iban"})
        self.assertIsNone(defs.changed_types(a, defs.conventions_snapshot({"email": "e", "bank_account": "old"}, True)), "scan flag: everything")
        self.assertIsNone(defs.changed_types(None, b), "unknown old version: everything")
        row = {"text": "Mail jo@x.io", "entities": [{"type": "email", "start": 5, "end": 12}]}
        self.assertFalse(defs.row_affected(row, {"bank_account"}))
        self.assertTrue(defs.row_affected(row, {"email"}))
        self.assertTrue(defs.row_affected(row, {"bank_account"}, lambda text, kinds: "jo@" in text), "pattern hit in the text")

    def test_only_affected_relabels_the_touched_rows_and_carries_the_rest(self):
        pid = client.post("/api/projects", json={"name": f"target-{time.time_ns()}", "description": "x"}).json()["id"]
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"email": "an email", "bank_account": "old"}})
        old = defs.conventions_version({"email": "an email", "bank_account": "old"})
        syn = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows = [
            {"id": 1, "text": "Account 12345678 here", "entities": [{"type": "bank_account", "start": 8, "end": 16}], "synth_source": "g", "conventions_version": old},
            {"id": 2, "text": "Mail jo@x.io", "entities": [{"type": "email", "start": 5, "end": 12}], "synth_source": "g", "conventions_version": old},
        ]
        (syn / "synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"email": "an email", "bank_account": "new wording"}})
        prompts = []

        class Labeller:
            name = "fake"

            def describe(self):
                return "fake"

            async def complete(self, prompt, **kw):
                prompts.append(prompt)
                return json.dumps({"rows": [{"index": int(m.group(1)), "entities": []} for m in re.finditer(r"\[(\d+)\] ", prompt)]})

        async def _go():
            async with async_session_factory() as db:
                return await relabel.relabel_training_rows(db, pid, backend_override=Labeller(), only_affected=True)

        summary = asyncio.run(_go())
        self.assertEqual(summary.targeting["affected_rows"], 1)
        self.assertEqual(summary.targeting["carried_forward"], 1)
        self.assertEqual(summary.targeting["changed_types"], ["bank_account"])
        self.assertTrue(all("jo@x.io" not in p for p in prompts), "the email row was not sent")
        saved = {r["id"]: r for r in (json.loads(l) for l in (syn / "synthetic.jsonl").read_text().splitlines() if l.strip())}
        now = defs.conventions_version({"email": "an email", "bank_account": "new wording"})
        self.assertEqual(saved[2]["conventions_version"], now)
        self.assertIn("conventions_carried_forward_at", saved[2])
        self.assertEqual(saved[2]["entities"], rows[1]["entities"], "carried forward untouched")


class TieredTests(unittest.TestCase):
    def test_review_model_only_sees_rows_where_bulk_and_model_disagree(self):
        texts = ["alpha text", "beta text", "gamma text"]
        bulk = [[{"type": "email", "start": 0, "end": 5}], [{"type": "email", "start": 0, "end": 3}], None]
        model = [[{"type": "email", "start": 0, "end": 5, "score": 0.9}], [{"type": "email", "start": 0, "end": 8}], []]
        seen = []

        class Review:
            def describe(self):
                return "review"

            async def complete(self, prompt, **kw):
                seen.append(prompt)
                return json.dumps({"rows": [{"index": int(m.group(1)), "entities": []} for m in re.finditer(r"\[(\d+)\] ", prompt)]})

        async def report(*a):
            return None

        async def fake_preds(db, pid, ts, experiment_id=None):
            return 77, model

        with mock.patch.object(relabel, "latest_span_run_predictions", fake_preds):
            labels, info = asyncio.run(relabel.review_disagreements(None, 1, texts, bulk, ["email"], "", Review(), report))
        self.assertEqual((info["disagreed"], info["reviewed"], info["model_experiment_id"]), (2, 2, 77))
        self.assertTrue(all("alpha text" not in p for p in seen))
        self.assertEqual(labels[0], bulk[0], "agreed rows keep the bulk labels")
        self.assertEqual(labels[1], [], "the review model's labels replace the bulk ones")

        async def no_model(db, pid, ts, experiment_id=None):
            return None, None

        with mock.patch.object(relabel, "latest_span_run_predictions", no_model):
            _, info = asyncio.run(relabel.review_disagreements(None, 1, texts, bulk, ["email"], "", Review(), report))
        self.assertIn("skipped", info)


if __name__ == "__main__":
    unittest.main()


class DryRunTests(unittest.TestCase):
    def test_dry_run_reports_the_affected_count_without_a_call(self):
        pid = client.post("/api/projects", json={"name": f"dry-{time.time_ns()}", "description": "x"}).json()["id"]
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"email": "e", "iban": "old"}})
        old = defs.conventions_version({"email": "e", "iban": "old"})
        syn = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows = [{"id": i, "text": f"row {i}", "entities": [{"type": "iban" if i == 0 else "email", "start": 0, "end": 3}],
                 "synth_source": "g", "conventions_version": old} for i in range(3)]
        (syn / "synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"email": "e", "iban": "new"}})

        class Never:
            def describe(self):
                return "never"

            async def complete(self, *a, **kw):
                raise AssertionError("dry run must not call the model")

        async def _go():
            async with async_session_factory() as db:
                return await relabel.relabel_training_rows(db, pid, backend_override=Never(), only_affected=True, dry_run=True)

        summary = asyncio.run(_go())
        self.assertEqual((summary.targeting["stale_rows"], summary.targeting["affected_rows"]), (3, 1))
        self.assertEqual(json.loads((syn / "synthetic.jsonl").read_text().splitlines()[1])["conventions_version"], old, "nothing written")

    def test_uncached_cost_counts_cached_tokens_as_input(self):
        est = llm_pricing.estimate_cost("sonnet", {"calls": 1, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 1_000_000})
        self.assertAlmostEqual(est["usd"], 0.3)
        self.assertAlmostEqual(est["usd_uncached"], 3.0)


class ReviewThresholdTests(unittest.TestCase):
    def test_row_agreement_and_the_threshold(self):
        a = [{"type": "email", "start": 0, "end": 5}, {"type": "phone", "start": 10, "end": 15},
             {"type": "name", "start": 20, "end": 25}, {"type": "name", "start": 30, "end": 35}, {"type": "iban", "start": 40, "end": 50}]
        b = a[:4] + [{"type": "iban", "start": 40, "end": 51}]  # one boundary differs
        self.assertAlmostEqual(relabel.row_agreement(a, b), 0.8)
        self.assertEqual(relabel.row_agreement([], []), 1.0)
        texts = ["one", "two"]
        bulk = [a, [{"type": "email", "start": 0, "end": 3}]]
        model = [b, [{"type": "phone", "start": 0, "end": 3}]]
        seen = []

        class Review:
            def describe(self):
                return "review"

            async def complete(self, prompt, **kw):
                seen.append(prompt)
                return json.dumps({"rows": [{"index": int(m.group(1)), "entities": []} for m in re.finditer(r"\[(\d+)\] ", prompt)]})

        async def report(*x):
            return None

        async def fake_preds(db, pid, ts, experiment_id=None):
            return 7, model

        with mock.patch.object(relabel, "latest_span_run_predictions", fake_preds):
            _, info = asyncio.run(relabel.review_disagreements(None, 1, texts, bulk, ["email"], "", Review(), report, min_agreement=0.8))
            self.assertEqual(info["disagreed"], 1, "0.8 agreement is not below 0.8; the 0.0 row is")
            self.assertEqual(info["agreement_distribution"]["0.8-1"], 1)
            _, info = asyncio.run(relabel.review_disagreements(None, 1, texts, bulk, ["email"], "", Review(), report, min_agreement=1.0))
            self.assertEqual(info["disagreed"], 2, "1.0: any difference")

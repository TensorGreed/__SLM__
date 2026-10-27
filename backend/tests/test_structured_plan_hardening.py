"""Hardening of the structured-plan recipe: bundle fixes surfaced through the API, off-loop schema
validation, the rubric judge's budget/cache behaviour, the live-schema overlay and per-pointer stats."""
from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services import evaluation_service as es
from app.services import json_schema_bundle as jsb
from app.services import llm_budget_guard
from app.services import rubric_service as rs
from app.services import structured_plan_prompt as spp
from app.services import structured_plan_scoring as sps
from app.services.answer_judge_service import AnswerJudgeCache, run_sync
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


PLAN = {
    "type": "object",
    "required": ["name"],
    "properties": {"name": {"type": "string"}, "env": {"enum": ["dev", "prod"]}},
}
RUBRIC = """
id: q
criteria:
  - id: ordering
    description: order
    anchors: {1: a, 2: b, 3: c, 4: d, 5: e}
  - id: naming
    description: name
    anchors: {1: a, 2: b, 3: c, 4: d, 5: e}
"""
PARSE = StructuredExtractionHandler()._parse_json_safely


def pred(plan, ref=None):
    return {"prediction": json.dumps(plan), "reference": json.dumps(ref if ref is not None else plan)}


def unique_files():
    plan = copy.deepcopy(PLAN)
    plan["description"] = f"nonce {time.time_ns()}"
    return {"plan.json": plan}


class ConfigApiTests(unittest.TestCase):
    def new_project(self) -> int:
        pid = client.post("/api/projects", json={"name": f"hard-{time.time_ns()}", "description": "x"}).json()["id"]
        self.assertEqual(client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "structured-plan"}).status_code, 200)
        return pid

    def test_cross_file_anchor_is_422_not_500(self):
        pid = self.new_project()
        files = {
            "p.json": {"type": "object", "properties": {"s": {"$ref": "s.json#stp"}}},
            "s.json": {"$defs": {"a": {"$anchor": "stp", "type": "string"}}},
        }
        r = client.put(f"/api/projects/{pid}/structured-plan-config", json={"schema_files": files, "root_schema": "p.json"})
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["error_code"], "INVALID_STRUCTURED_PLAN_CONFIG")
        self.assertIn("plain-name anchors across files are not supported", r.json()["metadata"]["errors"][0])

    def test_unsupported_dialect_is_422(self):
        pid = self.new_project()
        files = {"p.json": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}}
        r = client.put(f"/api/projects/{pid}/structured-plan-config", json={"schema_files": files, "root_schema": "p.json"})
        self.assertEqual(r.status_code, 422, r.text)
        self.assertIn("unsupported $schema dialect", r.json()["metadata"]["errors"][0])

    def test_deeply_nested_schema_is_422(self):
        pid = self.new_project()
        node: dict = {"type": "string"}
        for _ in range(2500):
            node = {"type": "array", "items": node}
        r = client.put(f"/api/projects/{pid}/structured-plan-config", json={"schema_files": {"p.json": node}, "root_schema": "p.json"})
        self.assertIn(r.status_code, (413, 422), r.text)
        if r.status_code == 422:
            self.assertTrue(r.json()["metadata"]["errors"])

    def test_validate_endpoint_survives_a_pathological_pattern(self):
        pid = self.new_project()
        files = {"p.json": {"type": "object", "properties": {"n": {"type": "string", "pattern": "^(a|aa)+$"}}}}
        r = client.put(f"/api/projects/{pid}/structured-plan-config", json={"schema_files": files, "root_schema": "p.json"})
        self.assertEqual(r.status_code, 200, r.text)
        t0 = time.monotonic()
        out = client.post(f"/api/projects/{pid}/structured-plan/validate", json={"value": {"n": "a" * 60 + "!"}}).json()
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertFalse(out["valid"])
        self.assertIn("pattern check timed out", out["errors"][0]["message"])


class ScoringTests(unittest.TestCase):
    def test_precompute_validates_once_and_scoring_reuses_it(self):
        bundle = jsb.validate_bundle(unique_files(), "plan.json")
        rows = [pred({"name": "a"}), pred({"env": "dev"}), pred("garbage")]
        rows[2]["prediction"] = "not json"
        calls = []
        real = jsb.SchemaBundle.validate

        def counting(self, instance):
            calls.append(threading.current_thread() is threading.main_thread())
            return real(self, instance)

        with patch.object(jsb.SchemaBundle, "validate", counting):
            sps.precompute(rows, bundle, PARSE)
            self.assertEqual(len(calls), 2)
            m = sps.score_structured_plan(rows, None, PARSE, bundle=bundle, pointers=[])
            self.assertEqual(len(calls), 2, "scoring must reuse the precomputed verdicts")
        self.assertEqual((m["total"], m["correct"], m["schema_valid_rate"]), (3, 1, 0.3333))

    def test_eval_prepare_runs_validation_in_a_worker_thread(self):
        files = unique_files()
        cfg = {"schema_files": files, "root_schema": "plan.json", "pointers": []}
        ctx = EvalContext(
            project_id=0, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
            handler_id="structured_extraction", prepared_dir=None, dataset_name="test",
            manifest={"output_schema": {"scoring_mode": "structured_plan", "structured_plan": cfg}},
        )
        handler = StructuredExtractionHandler()
        rows = [pred({"name": "a"})]
        threads = []
        real = jsb.SchemaBundle.validate

        def spy(self, instance):
            threads.append(threading.current_thread())
            return real(self, instance)

        with patch.object(jsb.SchemaBundle, "validate", spy):
            asyncio.run(es._prepare_structured_plan_predictions(handler, ctx, rows))
        self.assertEqual(len(threads), 1)
        self.assertIsNot(threads[0], threading.main_thread())
        self.assertIn(sps.PRECOMPUTED_KEY, rows[0])

    def test_pointer_missing_on_both_sides_is_not_a_miss(self):
        ref_with = {"name": "x", "env": "dev"}
        rows = [
            pred({"name": "x"}, {"name": "x"}),            # absent on both sides
            pred({"name": "x"}, ref_with),                 # absent in the output only
            pred(ref_with, ref_with),                      # present and equal
        ]
        m = sps.score_structured_plan(rows, None, PARSE, bundle=None, pointers=["/env"])
        stat = m["per_pointer"]["/env"]
        self.assertEqual((stat["support"], stat["both_missing"]), (3, 1))
        self.assertEqual(stat["missing_rate"], 0.5)  # 1 of the 2 rows that have an answer-key value
        self.assertEqual(stat["match_rate"], 0.6667)


class UnevaluatedPatternTimeoutTests(unittest.TestCase):
    def test_unevaluated_properties_pattern_is_time_limited(self):
        schema = {"type": "object", "patternProperties": {"^(a|aa)+$": {"type": "string"}},
                  "unevaluatedProperties": False}
        bundle = jsb.validate_bundle({"r.json": schema}, "r.json")
        start = time.monotonic()
        errors = bundle.validate({"a" * 60 + "!": "x"})
        self.assertLess(time.monotonic() - start, 5.0)  # stdlib re would take minutes here
        self.assertIn("pattern check timed out", errors[0]["message"])
        self.assertEqual(bundle.validate({"aa": "x"}), [])
        self.assertEqual(bundle.validate({"zz": "x"})[0]["validator"], "unevaluatedProperties")


class BundleCacheTests(unittest.TestCase):
    def test_bundle_is_validated_once_for_many_rows(self):
        cfg = {"schema_files": unique_files(), "root_schema": "plan.json"}
        calls = []
        real = jsb.validate_bundle

        def counting(files, root):
            calls.append(1)
            return real(files, root)

        with patch.object(jsb, "validate_bundle", counting):
            bundles = [spp.bundle_for_config(copy.deepcopy(cfg)) for _ in range(25)]
            texts = [spp.schema_text(b) for b in bundles]
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(b is bundles[0] for b in bundles))
        self.assertEqual(len(set(texts)), 1)

    def test_invalid_bundle_is_cached_as_none(self):
        cfg = {"schema_files": {"p.json": {"$ref": "https://example.com/x.json"}}, "root_schema": "p.json"}
        calls = []
        real = jsb.validate_bundle
        with patch.object(jsb, "validate_bundle", lambda f, r: (calls.append(1), real(f, r))[1]):
            self.assertIsNone(spp.bundle_for_config(cfg))
            self.assertIsNone(spp.bundle_for_config(cfg))
        self.assertEqual(len(calls), 1)


class FakeJudge:
    label = "fake:judge"

    def __init__(self, budget_call=False):
        self.budget_call = budget_call
        self.complete = self._complete

    def _complete(self, prompt, system_prompt, schema, max_tokens):
        if self.budget_call:
            async def _go():
                await llm_budget_guard.acheck_budget("anthropic", "m")
            with llm_budget_guard.project_context(7):
                run_sync(_go())
        return json.dumps({"scores": [{"id": "ordering", "score": 5, "reason": "ok"}, {"id": "naming", "score": 5, "reason": "ok"}]})


class FakeDb:
    def __init__(self, runtime_config):
        self.project = types.SimpleNamespace(id=7, runtime_config=runtime_config)

    async def get(self, model, pk):
        return self.project


def judge_rows():
    return [{"prediction": '{"name": "a"}', "reference": '{"name": "a"}', "plan_row": {"scenario": "s", "passages": []}, "row_structured_plan_checks_pass": True}]


class RubricJudgeOffLoopTests(unittest.TestCase):
    def setUp(self):
        self.rc = {"structured_plan": {"rubric_yaml": RUBRIC, "retrieve_passages": False}}
        llm_budget_guard.reset()

    def tearDown(self):
        llm_budget_guard.reset()

    def run_judge(self, judge):
        metrics: dict = {}
        label = asyncio.run(es._judge_structured_plan_rubric(
            FakeDb(self.rc), 7, StructuredExtractionHandler(), judge_rows(), metrics, judge, True))
        return label, metrics

    def test_no_db_session_is_opened_from_the_worker_loop(self):
        sessions = []

        def factory(*a, **k):
            sessions.append(threading.current_thread() is threading.main_thread())
            raise RuntimeError("no db in this test")

        # Always stale, so the worker-loop budget check would refresh too without the guard.
        with patch("app.database.async_session_factory", factory), \
                patch.object(llm_budget_guard, "_stale_locked", lambda: True):
            label, metrics = self.run_judge(FakeJudge(budget_call=True))
        self.assertEqual(label, "fake:judge")
        self.assertEqual(metrics["rubric"]["status"], "judged")
        self.assertTrue(sessions, "the ledger is refreshed once, on the main loop")
        self.assertTrue(all(sessions), "a DB session was opened from the worker thread")

    def test_budget_exceeded_still_propagates(self):
        llm_budget_guard.seed_for_tests(projects={7: 5.0}, caps={7: 1.0})
        with self.assertRaises(llm_budget_guard.LlmBudgetExceeded):
            self.run_judge(FakeJudge(budget_call=True))


class RubricCacheKeyTests(unittest.TestCase):
    def test_changed_passages_do_not_reuse_a_grade(self):
        import pathlib
        import tempfile

        rubric = rs.parse_rubric(RUBRIC)
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "c.json"
            judge = FakeJudge()
            seen = []
            base = judge._complete
            judge.complete = lambda *a: (seen.append(1), base(*a))[1]

            def view_with(passages):
                return lambda row: ("s", "", row["pred"], passages)

            def run(passages):
                rs.judge_rows_with_rubric([{"pred": "p"}], rubric, judge, cache=AnswerJudgeCache(path), view=view_with(passages))

            run([{"id": "P1", "text": "alpha"}])
            run([{"id": "P1", "text": "alpha"}])
            self.assertEqual(len(seen), 1, "same passages: cache hit")
            run([{"id": "P1", "text": "beta"}])
            self.assertEqual(len(seen), 2, "changed passage text: regraded")
            run([{"id": "P2", "text": "alpha"}])
            self.assertEqual(len(seen), 3, "changed passage id: regraded")
            run(None)
            self.assertEqual(len(seen), 4, "no passages: its own key")


class LiveOverlayTests(unittest.TestCase):
    def ctx(self, prep_sha):
        return EvalContext(
            project_id=0, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
            handler_id="structured_extraction", prepared_dir=None, dataset_name="test",
            manifest={"output_schema": {"scoring_mode": "structured_plan", "structured_plan": {"schema_sha256": prep_sha}}},
        )

    def overlay(self, prep_sha, files):
        rc = {"structured_plan": {"schema_files": files, "root_schema": "plan.json", "retrieve_passages": False}}
        ctx = self.ctx(prep_sha)
        asyncio.run(es._apply_structured_plan_live_config(FakeDb(rc), 7, ctx))
        return ctx

    def test_changed_schema_is_flagged_and_reaches_the_metrics(self):
        files = unique_files()
        live = jsb.validate_bundle(files, "plan.json")
        same = self.overlay(live.sha256, files).manifest["output_schema"]["structured_plan"]
        self.assertNotIn("schema_changed_since_prep", same)
        ctx = self.overlay("0" * 64, files)
        stamp = ctx.manifest["output_schema"]["structured_plan"]
        self.assertTrue(stamp["schema_changed_since_prep"])
        self.assertEqual(stamp["prep_schema_sha256"], "0" * 64)
        m = StructuredExtractionHandler().score([pred({"name": "a"})], ctx)
        self.assertTrue(m["schema_changed_since_prep"])
        self.assertEqual(m["prep_schema_sha256"], "0" * 64)
        m2 = StructuredExtractionHandler().score([pred({"name": "a"})], self.overlay(live.sha256, files))
        self.assertNotIn("schema_changed_since_prep", m2)


if __name__ == "__main__":
    unittest.main()

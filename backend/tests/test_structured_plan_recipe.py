"""The ``structured-plan`` recipe: catalog entry + detection, the shared prompt (train == eval, chat
template), adapter rows, plan scoring (schema validity, pointer matches incl. multi-value ``acceptable``
and wildcard multisets, rubric merge), the scoring fingerprint and the settings API (422 / 409,
guided-JSON hint write / clear).

Neutral toy "deployment plan" domain only; no model, no network."""

from __future__ import annotations

import copy
import json
import time
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.services import json_schema_bundle as jsb
from app.services import recipe_service
from app.services import structured_plan_prompt as spp
from app.services import structured_plan_scoring as sps
from app.services.data_adapter_service import map_record_with_adapter, unmappable_reason
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler
from app.services.task_shape_service import structured_recipe_for_rows
from scripts.train import _WRAPS_OWN_PROMPT_PREFIXES

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


STEP = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["action", "target"],
    "properties": {
        "action": {"enum": ["create", "update", "delete"]},
        "target": {"type": "string"},
        "environment": {"enum": ["dev", "staging", "prod"]},
    },
    "additionalProperties": False,
}
PLAN = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["name", "environment", "steps"],
    "properties": {
        "name": {"type": "string"},
        "environment": {"enum": ["dev", "staging", "prod"]},
        "steps": {"type": "array", "minItems": 1, "items": {"$ref": "step.schema.json"}},
    },
}
FILES = {"plan.schema.json": PLAN, "step.schema.json": STEP}
RUBRIC = """
id: plan-quality
title: Plan quality
criteria:
  - id: ordering
    description: Staging is updated before production.
    weight: 2
    critical: true
    anchors: {1: wrong order, 2: mostly wrong, 3: acceptable, 4: good, 5: exact}
  - id: naming
    description: The name says what is rolled out.
    anchors: {1: none, 2: vague, 3: ok, 4: clear, 5: precise}
"""


def good_plan(env: str = "staging") -> dict:
    return {
        "name": "billing rollout",
        "environment": env,
        "steps": [
            {"action": "update", "target": "billing", "environment": "staging"},
            {"action": "update", "target": "billing", "environment": "prod"},
        ],
    }


def plan_row(**extra) -> dict:
    row = {"row_kind": "structured_plan", "scenario": "Roll out billing to staging then production.", "plan_json": good_plan()}
    row.update(extra)
    return row


def bundle() -> jsb.SchemaBundle:
    return jsb.validate_bundle(copy.deepcopy(FILES), "plan.schema.json")


def pred(plan, reference=None, **extra) -> dict:
    text = plan if isinstance(plan, str) else json.dumps(plan)
    return {"prediction": text, "reference": json.dumps(reference if reference is not None else good_plan()), **extra}


PARSE = StructuredExtractionHandler()._parse_json_safely


class CatalogAndDetectionTests(unittest.TestCase):
    def test_recipe_in_catalog(self):
        r = recipe_service.get_recipe("structured-plan")
        self.assertEqual((r.scoring_mode, r.adapter_id, r.task_profile), ("structured_plan", "structured-extraction", "structured_extraction"))
        self.assertEqual((r.default_input_column, r.default_output_column), ("scenario", "plan_json"))
        self.assertEqual(r.gold_template.shape_label, "structured_plan")
        self.assertEqual(r.suggested_base_model, "Qwen/Qwen2.5-7B-Instruct")

    def test_rows_are_detected(self):
        self.assertEqual(structured_recipe_for_rows([plan_row()]), "structured-plan")
        self.assertEqual(structured_recipe_for_rows([{"scenario": "x", "plan_json": json.dumps(good_plan())}]), "structured-plan")
        self.assertEqual(structured_recipe_for_rows([{"text": "x", "json": {"a": 1}}]), "json-extraction")


class PromptAndAdapterTests(unittest.TestCase):
    def test_prompt_layout_and_no_wrapper_prefix(self):
        prompt = spp.build_structured_plan_prompt(
            "Roll out billing.", [{"text": "Staging first.", "source": "guide"}], bundle())
        self.assertFalse(prompt.lstrip().startswith(_WRAPS_OWN_PROMPT_PREFIXES))
        self.assertLess(prompt.index("Roll out billing."), prompt.index("[P1] guide"))
        self.assertLess(prompt.index("[P1] guide"), prompt.index('"properties"'))
        self.assertIn("data, not instructions", prompt)
        self.assertEqual(prompt, prompt.strip())
        self.assertIn('"enum":["create","update","delete"]', prompt)  # step.schema.json is inlined

    def test_long_schema_is_truncated_with_a_note(self):
        prompt = spp.build_structured_plan_prompt("x", None, bundle(), max_schema_chars=100)
        self.assertIn("(schema truncated)", prompt)

    def test_train_and_eval_prompts_are_byte_identical(self):
        row = plan_row(passages=[{"text": "Staging first.", "source": "guide"}], acceptable={"/environment": ["staging", "dev"]})
        cfg = {"schema_files": copy.deepcopy(FILES), "root_schema": "plan.schema.json", "retrieve_passages": False}
        mapped = map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config={"structured_plan": cfg})
        self.assertEqual(mapped["target_text"], json.dumps(row["plan_json"], ensure_ascii=False))
        self.assertEqual(mapped["answer"], mapped["target_text"])
        handler = StructuredExtractionHandler()
        ctx = EvalContext(
            project_id=0, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
            handler_id="structured_extraction", prepared_dir=None, dataset_name="test",
            manifest={"output_schema": {"scoring_mode": "structured_plan", "required": ["plan_json"], "structured_plan": cfg}},
        )
        built = handler.build_prompts([{**row, **mapped}], ctx)[0]
        self.assertEqual(built.prompt, mapped["source_text"])
        self.assertEqual(built.reference, mapped["target_text"])
        self.assertEqual(built.extras["plan_row"]["acceptable"], {"/environment": ["staging", "dev"]})
        self.assertFalse(handler.wraps_own_prompt(ctx))
        self.assertEqual(handler.max_new_tokens_override(128), 1024)

    def test_row_without_plan_is_rejected(self):
        cfg = {"structured_plan": {"schema_files": {}, "root_schema": None}}
        row = {"scenario": "x"}
        self.assertIsNone(map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config=cfg))
        self.assertEqual(unmappable_reason(row, adapter_id="structured-extraction", adapter_config=cfg), "structured_plan_missing")

    def test_retrieval_helper_uses_the_injected_fn(self):
        hits = [{"payload": {"text": "Staging first.", "source_doc": "guide"}}, {"payload": {"text": ""}}]
        seen = {}

        def fn(query, cfg):
            seen.update(query=query, k=cfg["k"])
            return hits

        out = spp.retrieve_passages_for_scenario(1, "Roll out billing.", 2, retrieve_fn=fn)
        self.assertEqual(seen, {"query": "Roll out billing.", "k": 2})
        self.assertEqual([p["id"] for p in out], ["P1"])
        self.assertEqual(spp.retrieve_passages_for_scenario(1, "  ", 2, retrieve_fn=fn), [])


class ScoringTests(unittest.TestCase):
    def score(self, preds, pointers=None, b="default"):
        return sps.score_structured_plan(
            preds, {}, PARSE, bundle=bundle() if b == "default" else b, pointers=pointers or [])

    def test_valid_invalid_and_unparseable(self):
        bad_enum = good_plan("qa")
        preds = [pred(good_plan()), pred(bad_enum), pred("not json"), pred({"name": "x"})]
        m = self.score(preds)
        self.assertEqual((m["total"], m["correct"]), (4, 1))
        self.assertEqual(m["json_validity_rate"], 0.75)
        self.assertEqual(m["schema_valid_rate"], 0.25)
        self.assertEqual(m["structured_plan_pass_rate"], 0.25)
        self.assertEqual(m["exact_match"], 0.25)
        self.assertEqual(m["scoring_mode"], "structured_plan")
        self.assertEqual(m["schema"]["root"], "plan.schema.json")
        self.assertIsNone(m["field_match_rate"])
        self.assertEqual(preds[1]["schema_errors"][0]["validator"], "enum")
        self.assertEqual(preds[1]["schema_errors"][0]["pointer"], "/environment")
        self.assertTrue(preds[0]["row_structured_plan_pass"])
        self.assertEqual(preds[0]["row_exact_match"], 1.0)
        self.assertFalse(preds[2]["row_schema_valid"])

    def test_top_errors_group_array_indices(self):
        plan = good_plan()
        plan["steps"][0]["action"] = "explode"
        plan["steps"][1]["action"] = "explode"
        m = self.score([pred(plan)])
        top = m["schema_errors_top"]
        self.assertEqual(top[0]["pointer"], "/steps/*/action")
        self.assertEqual((top[0]["validator"], top[0]["count"]), ("enum", 2))

    def test_pointer_match_against_the_reference(self):
        preds = [pred(good_plan("prod")), pred(good_plan("staging"))]
        m = self.score(preds, ["/environment", "/name"])
        self.assertEqual(m["per_pointer"]["/environment"], {"match_rate": 0.5, "support": 2, "missing_rate": 0.0, "both_missing": 0})
        self.assertEqual(m["per_pointer"]["/name"]["match_rate"], 1.0)
        self.assertEqual(m["field_match_rate"], 0.5)
        self.assertEqual(m["structured_plan_pass_rate"], 0.5)
        self.assertEqual(preds[0]["row_pointer_results"]["/environment"]["source"], "reference")
        self.assertFalse(preds[0]["row_pointer_results"]["/environment"]["match"])

    def test_multi_value_acceptable(self):
        acceptable = {"plan_row": {"acceptable": {"/environment": ["staging", "dev"]}}}
        ok = pred(good_plan("dev"), **acceptable)
        also = pred(good_plan("staging"), **acceptable)
        no = pred(good_plan("prod"), **acceptable)
        m = self.score([ok, also, no], ["/environment"])
        self.assertEqual([p["row_pointer_results"]["/environment"]["match"] for p in (ok, also, no)], [True, True, False])
        self.assertEqual(ok["row_pointer_results"]["/environment"]["source"], "acceptable")
        self.assertEqual(m["per_pointer"]["/environment"]["match_rate"], round(2 / 3, 4))

    def test_acceptable_needs_a_resolved_value(self):
        p = pred({"name": "x", "steps": []}, plan_row={"acceptable": {"/environment": ["dev"]}})
        m = self.score([p], ["/environment"])
        self.assertFalse(p["row_pointer_results"]["/environment"]["match"])
        self.assertEqual(m["per_pointer"]["/environment"]["missing_rate"], 1.0)

    def test_wildcard_multiset_is_order_insensitive(self):
        reordered = good_plan()
        reordered["steps"].reverse()
        wrong = good_plan()
        wrong["steps"][1]["environment"] = "dev"
        preds = [pred(reordered), pred(wrong)]
        self.score(preds, ["/steps/*/environment"])
        self.assertTrue(preds[0]["row_pointer_results"]["/steps/*/environment"]["match"])
        self.assertFalse(preds[1]["row_pointer_results"]["/steps/*/environment"]["match"])

    def test_pointer_missing_in_both_matches_and_in_pred_only_does_not(self):
        ref = {"name": "x", "environment": "dev", "steps": [{"action": "create", "target": "a"}]}
        both = pred(ref, reference=ref)
        only_ref = pred({"name": "x", "environment": "dev", "steps": [{"action": "create", "target": "a", "environment": "dev"}]}, reference=ref)
        absent_pred = pred(ref, reference={**ref, "steps": [{"action": "create", "target": "a", "environment": "dev"}]})
        self.score([both, only_ref, absent_pred], ["/steps/0/environment"])
        self.assertTrue(both["row_pointer_results"]["/steps/0/environment"]["match"])
        self.assertFalse(only_ref["row_pointer_results"]["/steps/0/environment"]["match"])
        self.assertFalse(absent_pred["row_pointer_results"]["/steps/0/environment"]["match"])

    def test_typed_compare_uses_the_gold_type(self):
        ref = {"count": 3, "code": "02134"}
        p = pred({"count": "3", "code": "2134"}, reference=ref)
        self.score([p], ["/count", "/code"], b=None)
        self.assertTrue(p["row_pointer_results"]["/count"]["match"])
        self.assertFalse(p["row_pointer_results"]["/code"]["match"])

    def test_no_schema_falls_back_to_valid_json_object(self):
        m = self.score([pred({"anything": 1}), pred("nope")], b=None)
        self.assertTrue(m["schema_missing"])
        self.assertEqual(m["schema_valid_rate"], 0.5)
        self.assertIsNone(m["schema"])

    def test_rubric_failure_fails_the_row_and_unjudged_does_not(self):
        preds = [pred(good_plan()) for _ in range(3)]
        m = self.score(preds)
        preds[0].update(row_rubric_passed=False, row_rubric_status="judged")
        preds[1].update(row_rubric_passed=True, row_rubric_status="judged")
        preds[2].update(row_rubric_status="unjudged")
        block = {"rubric": {"id": "r", "title": "t", "sha256": "s", "judge": "stub", "judged": 2, "unjudged": 1,
                            "mean_weighted": 3.5, "pass_rate": 0.5, "critical_fail_count": 1, "per_criterion": {}, "criteria_order": []}}
        sps.finalize_with_rubric(preds, m, block)
        self.assertEqual((m["correct"], m["structured_plan_pass_rate"], m["exact_match"]), (2, round(2 / 3, 4), round(2 / 3, 4)))
        self.assertEqual((m["rubric_pass_rate"], m["rubric_mean_score"]), (0.5, 3.5))
        self.assertEqual(m["rubric"]["status"], "judged")
        self.assertFalse(preds[0]["row_structured_plan_pass"])
        self.assertEqual(preds[0]["row_exact_match"], 0.0)
        self.assertTrue(preds[2]["row_structured_plan_pass"])

    def test_handler_dispatch(self):
        handler = StructuredExtractionHandler()
        cfg = {"schema_files": copy.deepcopy(FILES), "root_schema": "plan.schema.json", "pointers": ["/environment"]}
        ctx = EvalContext(
            project_id=0, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
            handler_id="structured_extraction", prepared_dir=None, dataset_name="test",
            manifest={"output_schema": {"scoring_mode": "structured_plan", "structured_plan": cfg}},
        )
        m = handler.score([pred(good_plan())], ctx)
        self.assertEqual(m["structured_plan_pass_rate"], 1.0)
        self.assertEqual(m["pointers"], ["/environment"])


class FingerprintTests(unittest.TestCase):
    def fp(self, **stored):
        return sps.structured_plan_scoring_fingerprint({"structured_plan": {"retrieve_passages": False, **stored}}, "judge:a")

    def test_changes_with_every_scoring_input(self):
        base = {"schema_files": copy.deepcopy(FILES), "root_schema": "plan.schema.json"}
        a = self.fp(**base)
        self.assertEqual(a, self.fp(**base))
        self.assertNotEqual(a, self.fp(**base, pointers=["/name"]))
        self.assertNotEqual(a, self.fp(**base, rubric_yaml=RUBRIC))
        self.assertNotEqual(self.fp(**base, rubric_yaml=RUBRIC), self.fp(**base, rubric_yaml=RUBRIC, judge_enabled=False))
        self.assertNotEqual(a, self.fp(**base, retrieve_passages=True))
        changed = copy.deepcopy(FILES)
        changed["step.schema.json"]["properties"]["target"] = {"type": "integer"}
        self.assertNotEqual(a, self.fp(schema_files=changed, root_schema="plan.schema.json"))
        # the judge label only matters once a rubric runs
        self.assertEqual(
            sps.structured_plan_scoring_fingerprint({"structured_plan": {**base, "retrieve_passages": False}}, "judge:b"), a)


class BaselineReuseTests(unittest.TestCase):
    def test_cached_baseline_must_carry_the_same_fingerprint_and_judge(self):
        from types import SimpleNamespace

        from app.services import post_training_eval_service as pte

        cached = SimpleNamespace(metrics={"scoring_mode": "structured_plan", "structured_plan_fingerprint": "abc",
                                          "rubric": {"status": "judged", "judge": "stub:a"}})
        self.assertTrue(pte._baseline_scoring_matches(cached, "abc"))
        self.assertFalse(pte._baseline_scoring_matches(cached, "other"))
        self.assertTrue(pte._baseline_judge_matches(cached, "stub:a"))
        self.assertFalse(pte._baseline_judge_matches(cached, "stub:b"))
        no_rubric = SimpleNamespace(metrics={"scoring_mode": "structured_plan"})
        self.assertTrue(pte._baseline_judge_matches(no_rubric, "stub:b"))


class ConfigApiTests(unittest.TestCase):
    def new_project(self, recipe="structured-plan") -> int:
        pid = client.post("/api/projects", json={"name": f"plan-{time.time_ns()}", "description": "x"}).json()["id"]
        if recipe:
            self.assertEqual(client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": recipe}).status_code, 200)
        return pid

    def url(self, pid):
        return f"/api/projects/{pid}/structured-plan-config"

    def test_wrong_recipe_is_409(self):
        pid = self.new_project("generic-sft")
        r = client.put(self.url(pid), json={"top_k": 2})
        self.assertEqual(r.status_code, 409, r.text)
        self.assertEqual(r.json()["error_code"], "NOT_STRUCTURED_PLAN")

    def test_empty_get(self):
        body = client.get(self.url(self.new_project())).json()
        self.assertIsNone(body["schema"])
        self.assertIsNone(body["rubric"])
        self.assertFalse(body["guided_json_present"])
        self.assertTrue(body["effective"]["judge_enabled"])

    def test_invalid_inputs_are_422_with_errors(self):
        pid = self.new_project()
        broken = {"plan.schema.json": {"type": "object", "properties": {"s": {"$ref": "missing.schema.json"}}}}
        r = client.put(self.url(pid), json={"schema_files": broken, "root_schema": "plan.schema.json"})
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["error_code"], "INVALID_STRUCTURED_PLAN_CONFIG")
        self.assertTrue(r.json()["metadata"]["errors"])
        r = client.put(self.url(pid), json={"rubric_yaml": "criteria: []"})
        self.assertEqual(r.status_code, 422)
        self.assertTrue(r.json()["metadata"]["errors"])
        r = client.put(self.url(pid), json={"pointers": ["name"]})
        self.assertEqual(r.status_code, 422)
        r = client.put(self.url(pid), json={"pointers": [f"/p{i}" for i in range(51)]})
        self.assertEqual(r.status_code, 422)
        self.assertIsNone(client.get(self.url(pid)).json()["schema"])  # nothing saved

    def test_save_validate_hint_and_clear(self):
        pid = self.new_project()
        r = client.put(self.url(pid), json={
            "schema_files": FILES, "root_schema": "plan.schema.json", "rubric_yaml": RUBRIC,
            "pointers": ["/environment", "/steps/*/action"], "judge_enabled": True, "retrieve_passages": False, "top_k": 2})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["schema"]["root"], "plan.schema.json")
        self.assertEqual(body["schema"]["files"], ["plan.schema.json", "step.schema.json"])
        self.assertTrue(body["schema"]["valid"])
        self.assertEqual(body["schema"]["sha256"], bundle().sha256)
        self.assertEqual([c["id"] for c in body["rubric"]["criteria"]], ["ordering", "naming"])
        self.assertTrue(body["rubric"]["criteria"][0]["critical"])
        self.assertTrue(body["guided_json_present"])
        self.assertEqual(body["config"]["pointers"], ["/environment", "/steps/*/action"])
        hint = client.get(f"/api/projects/{pid}/guided-json").json()["guided_json"]
        self.assertEqual(hint["source"], "structured_plan")
        self.assertEqual(hint["schema_sha256"], bundle().sha256)
        self.assertEqual(hint["schema"], bundle().dereferenced())

        ok = client.post(f"/api/projects/{pid}/structured-plan/validate", json={"value": good_plan()}).json()
        self.assertEqual(ok, {"valid": True, "errors": []})
        bad = client.post(f"/api/projects/{pid}/structured-plan/validate", json={"value": good_plan("qa")}).json()
        self.assertFalse(bad["valid"])
        self.assertEqual(bad["errors"][0]["pointer"], "/environment")

        # partial merge keeps the schema; null clears a key
        r = client.put(self.url(pid), json={"pointers": None})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsNone(r.json()["config"]["pointers"])
        self.assertTrue(r.json()["schema"]["valid"])
        self.assertTrue(r.json()["guided_json_present"])

        r = client.put(self.url(pid), json={"schema_files": None})
        self.assertIsNone(r.json()["schema"])
        self.assertFalse(r.json()["guided_json_present"])
        self.assertEqual(client.post(f"/api/projects/{pid}/structured-plan/validate", json={"value": {}}).status_code, 409)

    def test_switching_recipe_drops_the_hint_and_keeps_others(self):
        pid = self.new_project()
        client.put(self.url(pid), json={"schema_files": FILES, "root_schema": "plan.schema.json"})
        self.assertTrue(client.get(self.url(pid)).json()["guided_json_present"])
        # re-applying the same recipe keeps it
        client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "structured-plan"})
        self.assertTrue(client.get(self.url(pid)).json()["guided_json_present"])
        client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "generic-sft"})
        self.assertIsNone(client.get(f"/api/projects/{pid}/guided-json").json()["guided_json"])
        # a code-findings hint is written by its own recipe, and survives untouched
        client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "code-findings"})
        hint = client.get(f"/api/projects/{pid}/guided-json").json()["guided_json"]
        self.assertTrue(hint["source"].startswith("recipe:code-findings"))
        r = client.put(self.url(pid), json={"top_k": 2})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(client.get(f"/api/projects/{pid}/guided-json").json()["guided_json"]["source"], hint["source"])


if __name__ == "__main__":
    unittest.main()

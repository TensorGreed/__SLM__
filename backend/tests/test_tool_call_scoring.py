"""Tool-call scoring unit tests: extraction variants, validity against a catalog, argument match,
plan ordering (missing deps, cycles), the aggregated ``tools`` block, structured-plan folding, the
``tool_calling`` handler and the config API.

The catalog is seeded straight into ``runtime_config["tool_catalog"]`` (the pinned shape); no server,
no network."""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services import tool_call_scoring as tcs
from app.services.eval_task_handler_service import EvalContext, ToolCallHandler

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def _tool(name, props, required=(), *, read_only=None, destructive=None, description=""):
    ann = {}
    if read_only is not None:
        ann["readOnlyHint"] = read_only
    if destructive is not None:
        ann["destructiveHint"] = destructive
    return {
        "name": name, "description": description or name,
        "input_schema": {"type": "object", "properties": props, "required": list(required), "additionalProperties": False},
        "annotations": ann, "read_only": read_only, "destructive": destructive,
    }


TOOLS = [
    _tool("list_services", {}, read_only=True),
    _tool("get_service", {"name": {"type": "string"}}, ["name"], read_only=True),
    _tool("create_service", {"name": {"type": "string"}, "tier": {"type": "string", "enum": ["free", "pro"]}},
          ["name", "tier"], read_only=False, destructive=False),
    _tool("delete_service", {"name": {"type": "string"}}, ["name"], read_only=False, destructive=True),
]


def runtime_config(flags=None, tool_scoring=None):
    rc = {"tool_catalog": {
        "tools": TOOLS, "version": "a1b2c3d4e5f6", "source": {"kind": "manual"}, "server": None,
        "imported_at": "2026-10-26T00:00:00+00:00", "flags": flags or {}, "flags_version": "0f0f0f0f0f0f",
    }}
    if tool_scoring is not None:
        rc["tool_scoring"] = tool_scoring
    return rc


CFG = tcs.normalize_config({})


def step(i, tool, args=None, deps=None):
    return {"step": i, "tool": tool, "args": args if args is not None else {}, "depends_on": deps or []}


class ExtractCallsTests(unittest.TestCase):
    def test_plan_pointer_and_custom_fields(self):
        plan = {"steps": [step("a", "get_service", {"name": "x"}), step("b", "list_services", {}, ["a"])]}
        calls = tcs.extract_calls(plan, CFG, mode="plan")
        self.assertEqual([c["tool"] for c in calls], ["get_service", "list_services"])
        self.assertEqual(calls[1]["depends_on"], ["a"])
        cfg = tcs.normalize_config({"steps_pointer": "/plan/actions", "tool_field": "fn", "args_field": "params",
                                    "id_field": "id", "depends_field": "after"})
        doc = {"plan": {"actions": [{"id": 1, "fn": "list_services", "params": "{}", "after": 0}]}}
        calls = tcs.extract_calls(doc, cfg, mode="plan")
        self.assertEqual(calls, [{"id": "1", "tool": "list_services", "args": {}, "depends_on": ["0"]}])

    def test_plan_without_steps_is_none_and_empty_is_empty(self):
        self.assertIsNone(tcs.extract_calls({"other": 1}, CFG, mode="plan"))
        self.assertIsNone(tcs.extract_calls(None, CFG, mode="plan"))
        self.assertEqual(tcs.extract_calls({"steps": []}, CFG, mode="plan"), [])

    def test_call_variants(self):
        one = tcs.extract_calls({"tool": "get_service", "args": {"name": "x"}}, CFG, mode="call")
        self.assertEqual(one[0]["args"], {"name": "x"})
        many = tcs.extract_calls({"tool_calls": [{"tool": "list_services"}, {"name": "get_service", "arguments": '{"name": "y"}'}]},
                                 CFG, mode="call")
        self.assertEqual([c["tool"] for c in many], ["list_services", "get_service"])
        self.assertEqual(many[1]["args"], {"name": "y"})
        listed = tcs.extract_calls([{"name": "get_service", "arguments": {"name": "z"}}], CFG, mode="call")
        self.assertEqual(listed[0]["args"], {"name": "z"})
        wrapped = tcs.extract_calls({"function": {"name": "list_services", "arguments": "{}"}}, CFG, mode="call")
        self.assertEqual(wrapped[0]["tool"], "list_services")

    def test_explicit_no_call_and_unextractable(self):
        self.assertEqual(tcs.extract_calls({"tool": "none"}, CFG, mode="call"), [])
        self.assertEqual(tcs.extract_calls({"tool_calls": []}, CFG, mode="call"), [])
        self.assertIsNone(tcs.extract_calls({"text": "hi"}, CFG, mode="call"))
        self.assertIsNone(tcs.extract_calls("plain", CFG, mode="call"))

    def test_text_renderings(self):
        parsed = tcs.parse_call_output('Tool Call: get_service({"name": "x"})\nAssistant: ok')
        self.assertEqual(parsed, {"tool": "get_service", "args": {"name": "x"}})
        self.assertEqual(tcs.parse_call_output("Tool Call: list_services()"), {"tool": "list_services", "args": {}})
        self.assertEqual(tcs.parse_call_output('```json\n{"tool": "list_services"}\n```'), {"tool": "list_services"})
        self.assertEqual(tcs.parse_call_output('Sure: [{"tool": "list_services"}] done'), [{"tool": "list_services"}])
        self.assertIsNone(tcs.parse_call_output("no call here"))


class ScoreCallsTests(unittest.TestCase):
    def setUp(self):
        self.rc = runtime_config()

    def score(self, pred, gold, mode="plan", cfg=None):
        return tcs.score_calls(pred, gold, self.rc, cfg or CFG, mode=mode)

    def calls(self, *items):
        return tcs.extract_calls({"steps": list(items)}, CFG, mode="plan")

    def test_valid_plan_matches_gold(self):
        pred = self.calls(step("a", "get_service", {"name": "x"}), step("b", "create_service", {"name": "x", "tier": "pro"}, ["a"]))
        r = self.score(pred, pred)
        self.assertTrue(r["tool_call_valid"] and r["tool_args_match"] and r["plan_order_valid"] and r["pass"])
        self.assertEqual(r["errors"], [])
        self.assertEqual([c["args_valid"] for c in r["calls"]], [True, True])

    def test_unknown_tool_invalid_args_and_not_allowed(self):
        pred = self.calls(step("a", "nope"), step("b", "get_service", {"nme": "x"}), step("c", "create_service", {"name": "x", "tier": "gold"}))
        r = self.score(pred, None)
        self.assertFalse(r["tool_call_valid"])
        self.assertFalse(r["pass"])
        self.assertFalse(r["calls"][0]["known"])
        self.assertFalse(r["calls"][1]["args_valid"])
        self.assertTrue(r["calls"][1]["args_errors"])
        self.assertTrue(all(len(c["args_errors"]) <= 3 for c in r["calls"]))
        self.assertTrue(any("tier" in e or "gold" in e for e in r["calls"][2]["args_errors"]))
        self.assertIsNone(r["tool_args_match"])  # no gold
        self.rc = runtime_config(flags={"get_service": {"allowed": False}})
        # flags are applied by the catalog service: a disallowed tool is known but not allowed
        r2 = self.score(self.calls(step("a", "get_service", {"name": "x"})), None)
        self.assertTrue(r2["calls"][0]["known"])
        self.assertFalse(r2["calls"][0]["allowed"])
        self.assertFalse(r2["tool_call_valid"])

    def test_args_match_uses_gold_keys_or_arg_keys(self):
        gold = self.calls(step("a", "create_service", {"name": "x", "tier": "pro"}))
        wrong_tier = self.calls(step("a", "create_service", {"name": "x", "tier": "free"}))
        self.assertFalse(self.score(wrong_tier, gold)["tool_args_match"])
        cfg = tcs.normalize_config({"arg_keys": ["name"]})
        self.assertTrue(self.score(wrong_tier, gold, cfg=cfg)["tool_args_match"])
        other_tool = self.calls(step("a", "get_service", {"name": "x"}))
        self.assertFalse(self.score(other_tool, gold)["tool_args_match"])
        longer = self.calls(step("a", "create_service", {"name": "x", "tier": "pro"}), step("b", "list_services"))
        self.assertFalse(self.score(longer, gold)["tool_args_match"])
        self.assertTrue(tcs.args_match_pair({"tool": "t", "args": {"n": "5"}}, {"tool": "t", "args": {"n": 5}}))

    def test_plan_order_errors(self):
        dup = self.score(self.calls(step("a", "list_services"), step("a", "list_services")), None)
        self.assertFalse(dup["plan_order_valid"])
        self.assertIn("duplicate step id a", dup["errors"])
        missing = self.score(self.calls(step("a", "list_services", {}, ["zzz"])), None)
        self.assertFalse(missing["plan_order_valid"])
        cycle = self.score(self.calls(step("a", "list_services", {}, ["b"]), step("b", "list_services", {}, ["a"])), None)
        self.assertFalse(cycle["plan_order_valid"])
        self.assertIn("dependency cycle", cycle["errors"])
        selfdep = self.score(self.calls(step("a", "list_services", {}, ["a"])), None)
        self.assertFalse(selfdep["plan_order_valid"])
        forward = self.score(self.calls(step("a", "list_services", {}, ["b"]), step("b", "list_services")), None)
        self.assertTrue(forward["plan_order_valid"])  # anywhere in the plan is fine as long as it exists

    def test_pass_respects_enabled_modes(self):
        pred = self.calls(step("a", "list_services", {}, ["zzz"]))
        self.assertFalse(self.score(pred, None)["pass"])
        cfg = tcs.normalize_config({"modes": ["tool_call_valid"]})
        self.assertTrue(self.score(pred, None, cfg=cfg)["pass"])

    def test_empty_plan_and_none_call(self):
        self.assertFalse(self.score([], None)["tool_call_valid"])
        self.assertTrue(self.score([], [])["tool_call_valid"])
        r = self.score([], [], mode="call")
        self.assertTrue(r["tool_call_valid"] and r["plan_order_valid"] is None and r["pass"])
        gold = [{"id": None, "tool": "list_services", "args": {}, "depends_on": []}]
        r = self.score([], gold, mode="call")
        self.assertTrue(r["tool_call_valid"])
        self.assertFalse(r["tool_args_match"])
        self.assertFalse(r["pass"])
        none = self.score(None, gold, mode="call")
        self.assertFalse(none["tool_call_valid"] or none["pass"])

    def test_non_object_args_are_invalid(self):
        pred = tcs.extract_calls({"tool": "get_service", "args": [1, 2]}, CFG, mode="call")
        r = self.score(pred, None, mode="call")
        self.assertFalse(r["calls"][0]["args_valid"])

    def test_schema_error_tool_scores_invalid(self):
        rc = runtime_config()
        rc["tool_catalog"]["tools"] = [dict(TOOLS[1], schema_error="bad schema")]
        r = tcs.score_calls(tcs.extract_calls({"tool": "get_service", "args": {"name": "x"}}, CFG, mode="call"),
                            None, rc, CFG, mode="call")
        self.assertFalse(r["tool_call_valid"])


class ScoreBlockTests(unittest.TestCase):
    def preds(self):
        def p(plan, ref=None):
            return {"prediction": json.dumps(plan), "reference": json.dumps(ref if ref is not None else plan)}

        good = {"steps": [step("a", "get_service", {"name": "x"})]}
        return [
            p(good),
            p({"steps": [step("a", "ghost")]}, good),
            p({"steps": [step("a", "get_service", {})]}, good),
            {"prediction": "not json", "reference": json.dumps(good)},
        ]

    def test_block_shape_and_row_results(self):
        rc = runtime_config()
        preds = self.preds()
        out = tcs.score_tool_block(preds, rc, CFG, mode="plan")
        t = out["tools"]
        self.assertEqual((t["catalog_version"], t["flags_version"], t["source_kind"]), ("a1b2c3d4e5f6", "0f0f0f0f0f0f", "manual"))
        self.assertEqual(t["tool_call_valid_rate"], 0.25)
        self.assertEqual(out["tool_call_valid_rate"], 0.25)
        self.assertEqual(t["pass_rate"], 0.25)
        self.assertEqual(t["unknown_tools"], {"ghost": 1})
        self.assertEqual(t["per_tool"]["get_service"], {"calls": 2, "valid": 1, "invalid_args": 1, "not_allowed": 0})
        self.assertTrue(t["errors_top"])
        self.assertIsNone(t["dry_run"])
        self.assertEqual([p["row_tools_pass"] for p in preds], [True, False, False, False])
        self.assertIn("calls", preds[0]["row_tool_results"])

    def test_finalize_folds_into_structured_plan_pass(self):
        rc = runtime_config()
        preds = self.preds()
        for p in preds:
            p["row_structured_plan_checks_pass"] = True
            p["row_structured_plan_pass"] = True
        metrics = {"total": 4, "correct": 4, "exact_match": 1.0, "structured_plan_pass_rate": 1.0}
        tcs.finalize_with_tools(preds, metrics, tcs.score_tool_block(preds, rc, CFG, mode="plan"))
        self.assertEqual(metrics["structured_plan_pass_rate"], 0.25)
        self.assertEqual((metrics["correct"], metrics["exact_match"]), (1, 0.25))
        self.assertEqual(metrics["tool_call_valid_rate"], 0.25)
        self.assertIn("tools", metrics)
        self.assertEqual([p["row_structured_plan_pass"] for p in preds], [True, False, False, False])
        # a row that already failed its schema checks stays failed
        preds2 = self.preds()
        for p in preds2:
            p["row_structured_plan_checks_pass"] = False
        m2 = {"total": 4}
        tcs.finalize_with_tools(preds2, m2, tcs.score_tool_block(preds2, rc, CFG, mode="plan"))
        self.assertEqual(m2["correct"], 0)


class ConfigAndFingerprintTests(unittest.TestCase):
    def test_effective_config(self):
        self.assertIsNone(tcs.effective_tool_scoring_config({}))
        self.assertIsNone(tcs.effective_tool_scoring_config(None))
        eff = tcs.effective_tool_scoring_config(runtime_config())
        self.assertEqual(eff["modes"], list(tcs.ALL_MODES))
        self.assertEqual(eff["steps_pointer"], "/steps")
        self.assertIsNone(tcs.effective_tool_scoring_config(runtime_config(tool_scoring={"enabled": False})))
        eff = tcs.effective_tool_scoring_config(runtime_config(tool_scoring={"modes": ["tool_call_valid"], "tool_field": "fn", "steps_pointer": "bad"}))
        self.assertEqual((eff["modes"], eff["tool_field"], eff["steps_pointer"]), (["tool_call_valid"], "fn", "/steps"))

    def test_fingerprint_changes_with_flags_and_settings(self):
        self.assertEqual(tcs.tools_fingerprint({}), "none")
        base = tcs.tools_fingerprint(runtime_config())
        self.assertNotEqual(base, "none")
        self.assertNotEqual(base, tcs.tools_fingerprint(runtime_config(tool_scoring={"arg_keys": ["name"]})))
        rc = runtime_config()
        rc["tool_catalog"]["flags_version"] = "ffffffffffff"
        self.assertNotEqual(base, tcs.tools_fingerprint(rc))

    def test_plan_fingerprint_only_changes_with_a_catalog(self):
        from app.services.structured_plan_scoring import structured_plan_scoring_fingerprint as fp

        plain = fp({}, None, None)
        self.assertEqual(plain, fp({"tool_scoring": {"arg_keys": ["x"]}}, None, None))  # no catalog: unchanged
        self.assertNotEqual(plain, fp(runtime_config(), None, None))

    def test_prompt_block_only_with_a_catalog(self):
        from app.services import structured_plan_prompt as spp

        self.assertEqual(spp.tool_block_for({}), "")
        block = spp.tool_block_for(runtime_config())
        self.assertTrue(block.startswith("Available tools (call only these):"))
        with_tools = spp.build_structured_plan_prompt("do it", None, None, tool_block=block)
        without = spp.build_structured_plan_prompt("do it", None, None)
        self.assertIn("get_service(name: string)", with_tools)
        self.assertNotIn("Available tools", without)
        self.assertEqual(without, spp.build_structured_plan_prompt("do it", None, None, tool_block=""))
        self.assertEqual(spp.tool_block_for(runtime_config(tool_scoring={"enabled": False})), "")


class ToolCallHandlerTests(unittest.TestCase):
    def ctx(self, **manifest):
        return EvalContext(project_id=1, experiment_id=1, eval_type="exact_match", task_profile="tool_calling",
                           handler_id="tool_calling", prepared_dir=Path("."), dataset_name="test", manifest=manifest)

    def rows(self):
        return [
            {"question": "Fetch x", "answer": "found x", "tool_name": "get_service", "tool_args_json": '{"name": "x"}'},
            {"question": "List", "reference": json.dumps({"tool": "list_services", "args": {}})},
            {"question": "Hi", "reference": json.dumps({"tool": "none"})},
        ]

    def test_registered_and_prompts(self):
        from app.services.eval_task_handler_service import resolve_task_handler

        self.assertIsInstance(resolve_task_handler("tool_calling"), ToolCallHandler)
        built = ToolCallHandler().build_prompts(self.rows(), self.ctx())
        self.assertEqual(built[0].prompt, "Fetch x")  # no catalog: prompt unchanged
        self.assertEqual(json.loads(built[0].reference), {"tool": "get_service", "args": {"name": "x"}})
        self.assertEqual(json.loads(built[2].reference), {"tool": "none"})
        with_block = ToolCallHandler().build_prompts(self.rows(), self.ctx(tool_block="Available tools (call only these):\n- t"))
        self.assertEqual(with_block[0].prompt, "Available tools (call only these):\n- t\n\nFetch x")
        self.assertEqual(with_block[0].as_pair()["tool_gold_calls"][0]["tool"], "get_service")

    def test_user_prompt_field_avoids_double_block(self):
        row = {"question": "BLOCK\n\nFetch x", "user_prompt": "Fetch x", "tool_name": "get_service", "tool_args_json": "{}"}
        built = ToolCallHandler().build_prompts([row], self.ctx(tool_block="BLOCK"))
        self.assertEqual(built[0].prompt, "BLOCK\n\nFetch x")

    def test_scoring_without_a_catalog(self):
        h = ToolCallHandler()
        built = h.build_prompts(self.rows(), self.ctx())
        outputs = [
            'Tool Call: get_service({"name": "x"})\nAssistant: ok',            # right
            json.dumps({"tool": "get_service", "args": {}}),                    # wrong tool
            json.dumps({"tool": "none"}),                                       # right no-call
        ]
        preds = [dict(bp.as_pair(), prediction=o) for bp, o in zip(built, outputs)]
        m = h.score(preds, self.ctx())
        self.assertEqual(m["scoring_mode"], "tool_call")
        self.assertEqual(m["tool_selection_accuracy"], round(2 / 3, 4))
        self.assertEqual(m["tool_call_pass_rate"], round(2 / 3, 4))
        self.assertEqual(m["exact_match"], m["tool_call_pass_rate"])
        self.assertEqual([p["row_tool_call_pass"] for p in preds], [True, False, True])

    def test_arg_mismatch_selects_but_fails(self):
        h = ToolCallHandler()
        built = h.build_prompts(self.rows()[:1], self.ctx())
        preds = [dict(built[0].as_pair(), prediction=json.dumps({"tool": "get_service", "args": {"name": "y"}}))]
        m = h.score(preds, self.ctx())
        self.assertEqual((m["tool_selection_accuracy"], m["tool_args_match_rate"], m["tool_call_pass_rate"]), (1.0, 0.0, 0.0))

    def test_catalog_validity_is_required_when_present(self):
        h = ToolCallHandler()
        rc = runtime_config()
        built = h.build_prompts(self.rows()[:2], self.ctx())
        outputs = [json.dumps({"tool": "get_service", "args": {"name": "x", "extra": 1}}),  # right + schema-invalid
                   json.dumps({"tool": "list_services", "args": {}})]
        preds = [dict(bp.as_pair(), prediction=o) for bp, o in zip(built, outputs)]
        m = h.score(preds, self.ctx())
        self.assertEqual(m["tool_call_pass_rate"], 1.0)  # no catalog: the extra key is ignored by the args match
        block = tcs.score_tool_block(preds, rc, CFG, mode="call")
        tcs.finalize_tool_calls(preds, m, block, CFG)
        self.assertEqual(m["tool_call_pass_rate"], 0.5)
        self.assertEqual(m["exact_match"], 0.5)
        self.assertEqual(m["tool_call_valid_rate"], 0.5)
        self.assertEqual(m["tools"]["per_tool"]["get_service"]["invalid_args"], 1)
        self.assertEqual([p["row_tool_call_pass"] for p in preds], [False, True])

    def test_lift_wiring(self):
        from app.services import sft_lift_summary_service as lift

        self.assertIn("tool_call_pass_rate", lift._PREFERRED_HEADLINE_METRICS)
        self.assertLess(lift._PREFERRED_HEADLINE_METRICS.index("structured_plan_pass_rate"),
                        lift._PREFERRED_HEADLINE_METRICS.index("tool_call_pass_rate"))
        self.assertEqual(lift._ROW_SCORE_FOR_METRIC["tool_call_pass_rate"], "tool_call_pass")


class AdapterTests(unittest.TestCase):
    RECORD = {"prompt": "Fetch x", "tool_name": "get_service", "tool_args": {"name": "x"}, "result": "found"}

    def test_unchanged_without_a_tool_block(self):
        from app.services.data_adapter_service import map_record_with_adapter

        out = map_record_with_adapter(self.RECORD, adapter_id="tool-call-json", adapter_config={})
        self.assertNotIn("user_prompt", out)
        self.assertTrue(out["text"].startswith("User: Fetch x"))
        self.assertEqual(out["question"], "Fetch x")

    def test_block_leads_the_prompt_like_eval(self):
        from app.services.data_adapter_service import map_record_with_adapter

        block = "Available tools (call only these):\n- get_service(name: string)"
        out = map_record_with_adapter(self.RECORD, adapter_id="tool-call-json", adapter_config={"tool_block": block})
        self.assertEqual(out["question"], tcs.tool_prompt(block, "Fetch x"))
        self.assertEqual(out["source_text"], out["question"])
        self.assertTrue(out["text"].startswith(block + "\n\nUser: Fetch x"))
        self.assertEqual(out["user_prompt"], "Fetch x")
        built = ToolCallHandler().build_prompts([out], EvalContext(
            project_id=1, experiment_id=1, eval_type="x", task_profile="tool_calling", handler_id="tool_calling",
            prepared_dir=Path("."), dataset_name="t", manifest={"tool_block": block}))
        self.assertEqual(built[0].prompt, out["question"])  # train and eval prompts are byte-identical


    def test_target_is_the_canonical_call_json(self):
        from app.services.data_adapter_service import map_record_with_adapter

        out = map_record_with_adapter(self.RECORD, adapter_id="tool-call-json", adapter_config={})
        self.assertEqual(out["answer"], '{"tool": "get_service", "args": {"name": "x"}}')
        self.assertEqual(out["target_text"], out["answer"])
        self.assertEqual(out["tool_result"], "found")  # the result stays its own field
        self.assertIn("Tool Call: get_service(", out["text"])
        self.assertTrue(out["text"].endswith("Assistant: found"))  # the transcript keeps the full exchange
        # the same with a catalog block, and with no result at all
        rec = {"prompt": "Fetch x", "tool_name": "get_service", "tool_args": {"name": "é"}}
        out = map_record_with_adapter(rec, adapter_id="tool-call-json", adapter_config={"tool_block": "Available tools (call only these):\n- t()"})
        self.assertEqual(out["answer"], '{"tool": "get_service", "args": {"name": "é"}}')

    def test_json_tool_result_with_name_keys_does_not_corrupt_gold(self):
        from app.services.data_adapter_service import map_record_with_adapter

        for result in ({"name": "billing", "tier": "gold"}, [{"name": "a"}, {"name": "b"}]):
            rec = {**self.RECORD, "result": result}
            out = map_record_with_adapter(rec, adapter_id="tool-call-json", adapter_config={})
            self.assertEqual(json.loads(out["answer"]), {"tool": "get_service", "args": {"name": "x"}})
            self.assertIn("billing" if isinstance(result, dict) else '"b"', out["tool_result"])
            gold = ToolCallHandler()._gold({**rec, **out}, "", tcs.normalize_config({}))
            self.assertEqual([(g["tool"], g["args"]) for g in gold], [("get_service", {"name": "x"})])
            # even a reference that parses as a call-looking object loses to the row's own tool fields
            gold = ToolCallHandler()._gold({**rec, **out}, json.dumps(result), tcs.normalize_config({}))
            self.assertEqual([(g["tool"], g["args"]) for g in gold], [("get_service", {"name": "x"})])

    def test_reference_is_parsed_as_a_call_only_without_row_fields(self):
        gold = ToolCallHandler()._gold({}, '{"tool": "get_service", "args": {"name": "x"}}', tcs.normalize_config({}))
        self.assertEqual([(g["tool"], g["args"]) for g in gold], [("get_service", {"name": "x"})])


class FingerprintAndMixedNoCallTests(unittest.TestCase):
    def test_fingerprint_tracks_version_catalog_flags_and_exists_without_a_catalog(self):
        none = tcs.tool_call_fingerprint({})
        self.assertTrue(none.startswith(tcs.TOOL_CALL_HANDLER_VERSION + "+"))
        a = tcs.tool_call_fingerprint(runtime_config())
        rc = runtime_config(flags={"get_service": {"allowed": False}})
        rc["tool_catalog"]["flags_version"] = "1e1e1e1e1e1e"
        b = tcs.tool_call_fingerprint(rc)
        self.assertEqual(len({none, a, b}), 3)
        self.assertEqual(a, tcs.tool_call_fingerprint(runtime_config()))
        with patch.object(tcs, "TOOL_CALL_HANDLER_VERSION", "tool_call/v2"):
            self.assertNotEqual(a, tcs.tool_call_fingerprint(runtime_config()))

    def test_baseline_reuse_requires_the_same_fingerprint(self):
        from app.services.post_training_eval_service import _baseline_scoring_matches

        fp = tcs.tool_call_fingerprint(runtime_config())
        cached = SimpleNamespace(metrics={"scoring_mode": "tool_call", "tool_call_fingerprint": fp})
        self.assertTrue(_baseline_scoring_matches(cached, None, fp))
        self.assertFalse(_baseline_scoring_matches(cached, None, tcs.tool_call_fingerprint({})))
        self.assertFalse(_baseline_scoring_matches(SimpleNamespace(metrics={"scoring_mode": "tool_call"}), None, fp))  # pre-fingerprint
        self.assertFalse(_baseline_scoring_matches(cached, None, None))
        self.assertTrue(_baseline_scoring_matches(SimpleNamespace(metrics={"scoring_mode": "exact_match"}), None, fp))

    def test_no_call_mixed_with_calls_is_invalid(self):
        cfg = tcs.normalize_config({})
        parsed = [{"tool": "get_service", "args": {"name": "x"}}, {"tool": "none"}]
        calls = tcs.extract_calls(parsed, cfg, mode="call")
        res = tcs.score_calls(calls, [{"id": None, "tool": "get_service", "args": {"name": "x"}, "depends_on": []}], runtime_config(), cfg, mode="call")
        self.assertFalse(res["tool_call_valid"])
        self.assertFalse(res["pass"])
        self.assertIn("no-call mixed with calls", res["errors"])
        # a lone no-call is still an explicit empty call list
        self.assertEqual(tcs.extract_calls([{"tool": "none"}], cfg, mode="call"), [])
        # and the handler does not count the mixed output as a correct selection
        pred = {"prediction": json.dumps(parsed), "tool_gold_calls": [{"id": None, "tool": "get_service", "args": {"name": "x"}, "depends_on": []}]}
        out = ToolCallHandler().score([pred], SimpleNamespace(manifest={}))
        self.assertEqual(out["tool_call_pass_rate"], 0.0)


class ConfigApiTests(unittest.TestCase):
    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"tool-scoring-{time.time_ns()}", "description": "x"}).json()["id"]
        self.url = f"/api/projects/{self.pid}/tool-scoring-config"

    def test_get_defaults_without_a_catalog(self):
        body = client.get(self.url).json()
        self.assertIsNone(body["catalog"])
        self.assertFalse(body["effective"]["active"])
        self.assertEqual(body["effective"]["steps_pointer"], "/steps")
        self.assertEqual(body["effective"]["modes"], list(tcs.ALL_MODES))

    def test_put_validates_and_merges(self):
        ok = client.put(self.url, json={"modes": ["tool_call_valid"], "steps_pointer": "/plan/steps", "arg_keys": ["name"]})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(ok.json()["config"]["steps_pointer"], "/plan/steps")
        again = client.put(self.url, json={"tool_field": "fn"}).json()
        self.assertEqual(again["config"]["modes"], ["tool_call_valid"])  # partial merge keeps earlier keys
        self.assertEqual(again["effective"]["tool_field"], "fn")
        cleared = client.put(self.url, json={"modes": None}).json()
        self.assertIsNone(cleared["config"]["modes"])
        self.assertEqual(cleared["effective"]["modes"], list(tcs.ALL_MODES))

    def test_put_rejects_bad_values(self):
        for body in ({"modes": ["nope"]}, {"steps_pointer": "steps"}, {"tool_field": ""}, {"arg_keys": [""]}):
            r = client.put(self.url, json=body)
            self.assertEqual(r.status_code, 422, body)
            self.assertEqual(r.json()["error_code"], "INVALID_TOOL_SCORING_CONFIG", r.text)
            self.assertTrue(r.json()["metadata"]["errors"])

    def test_dry_run_needs_the_admin_switch(self):
        r = client.put(self.url, json={"dry_run": True})
        self.assertEqual(r.status_code, 422)
        self.assertIn("MCP_DRY_RUN_ENABLED", r.json()["message"])
        with patch("app.config.settings.MCP_DRY_RUN_ENABLED", True):
            r = client.put(self.url, json={"dry_run": True})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertTrue(r.json()["config"]["dry_run"])
        r = client.put(self.url, json={"dry_run": False})
        self.assertEqual(r.status_code, 200)

    def test_get_reports_the_catalog(self):
        import asyncio

        from app.database import async_session_factory
        from app.models.project import Project

        async def seed():
            async with async_session_factory() as db:
                project = await db.get(Project, self.pid)
                project.runtime_config = {**(project.runtime_config or {}), **runtime_config()}
                await db.commit()

        asyncio.run(seed())
        body = client.get(self.url).json()
        self.assertEqual(body["catalog"], {"version": "a1b2c3d4e5f6", "tool_count": 4, "source_kind": "manual"})
        self.assertTrue(body["effective"]["active"])


if __name__ == "__main__":
    unittest.main()

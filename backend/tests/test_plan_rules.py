"""Plan-rule engine: every rule type + condition operator, reference resolution (incl. missing
refs), the scenario_grid reject / warn path, and the pending-row check (dry run, apply, restore,
Job, bad rules -> 400).

pytest-only (the Jobs these tests start are awaited in tearDown).
"""

from __future__ import annotations

import json
import shutil
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services import plan_rules_service as engine
from app.services.synth_playbooks import SynthMode, get_playbook
from app.services.synth_playbooks._structured_plan_common import check_params

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def step(n, operation, body=None, tool="invoke_operation", **extra):
    return {"step": n, "tool": tool, "args": {"operation": operation, "arguments": {"body": body or {}}}, **extra}


def plan(*steps):
    return {"steps": list(steps)}


def rs(*rules, **top):
    return engine.load_rules({"rules": list(rules), **top})


def ids(violations):
    return [v.rule_id for v in violations]


MATCH_CREATE = {"operation": "create_widget"}


class ConditionTests(unittest.TestCase):
    def holds(self, cond, node):
        return engine.holds(cond, node)

    def test_equals_not_equals_in_not_in(self):
        node = {"a": {"b": "x"}, "n": 3}
        self.assertTrue(self.holds({"path": "a.b", "op": "equals", "value": "x"}, node))
        self.assertFalse(self.holds({"path": "a.b", "op": "equals", "value": "y"}, node))
        self.assertTrue(self.holds({"path": "a.b", "op": "not_equals", "value": "y"}, node))
        self.assertTrue(self.holds({"path": "a.b", "op": "in", "value": ["x", "z"]}, node))
        self.assertTrue(self.holds({"path": "a.b", "op": "not_in", "value": ["q"]}, node))
        self.assertFalse(self.holds({"path": "a.b", "op": "not_in", "value": ["x"]}, node))
        self.assertFalse(self.holds({"path": "n", "op": "equals", "value": True}, {"n": 1}), "True is not 1")

    def test_exists_not_exists_and_a_missing_path_fails_value_ops(self):
        node = {"a": {"b": None}}
        self.assertTrue(self.holds({"path": "a.b", "op": "exists"}, node), "null is present")
        self.assertTrue(self.holds({"path": "a.c", "op": "not_exists"}, node))
        self.assertFalse(self.holds({"path": "a.c", "op": "equals", "value": 1}, node))
        self.assertFalse(self.holds({"path": "a.c", "op": "not_equals", "value": 1}, node), "nothing resolved: not satisfied")

    def test_regex_contains_max_min(self):
        node = {"name": "abc-01", "tags": ["x", "y"], "size": "256", "n": 5}
        self.assertTrue(self.holds({"path": "name", "op": "regex", "value": r"^abc-\d+$"}, node))
        self.assertFalse(self.holds({"path": "name", "op": "regex", "value": r"^z"}, node))
        self.assertTrue(self.holds({"path": "name", "op": "contains", "value": "c-0"}, node))
        self.assertTrue(self.holds({"path": "tags", "op": "contains", "value": "y"}, node))
        self.assertTrue(self.holds({"path": "size", "op": "max", "value": 256}, node), "numeric strings compare as numbers")
        self.assertFalse(self.holds({"path": "n", "op": "max", "value": 4}, node))
        self.assertTrue(self.holds({"path": "n", "op": "min", "value": 5}, node))
        self.assertFalse(self.holds({"path": "name", "op": "max", "value": 4}, node), "non-numeric never passes a bound")

    def test_wildcards_and_quantifier(self):
        node = {"items": [{"k": 1}, {"k": 5}], "map": {"p": {"k": 2}}}
        self.assertEqual(engine.resolve(node, "items.*.k"), [1, 5])
        self.assertEqual(engine.resolve(node, "items[*].k"), [1, 5])
        self.assertEqual(engine.resolve(node, "items.1.k"), [5])
        self.assertEqual(engine.resolve(node, "map.*.k"), [2])
        self.assertEqual(engine.resolve(node, "items.7.k"), [])
        self.assertTrue(self.holds({"path": "items.*.k", "op": "max", "value": 1}, node), "any item satisfies")
        self.assertFalse(self.holds({"path": "items.*.k", "op": "max", "value": 1, "quantifier": "all"}, node))
        self.assertTrue(self.holds({"path": "items.*.k", "op": "min", "value": 1, "quantifier": "all"}, node))

    def test_combinators(self):
        node = {"a": 1, "b": 2}
        eq = lambda p, v: {"path": p, "op": "equals", "value": v}  # noqa: E731
        self.assertTrue(self.holds({"all": [eq("a", 1), eq("b", 2)]}, node))
        self.assertFalse(self.holds({"all": [eq("a", 1), eq("b", 3)]}, node))
        self.assertTrue(self.holds({"any": [eq("a", 9), eq("b", 2)]}, node))
        self.assertTrue(self.holds({"not": eq("a", 9)}, node))
        self.assertFalse(self.holds({"not": eq("a", 1)}, node))


class StepRequiresTests(unittest.TestCase):
    RULE = {"id": "widget-sized", "type": "step_requires", "match": MATCH_CREATE,
            "require": {"path": "args.arguments.body.size", "op": "max", "value": 10}, "description": "widgets stay small"}

    def test_flags_only_matching_steps_that_fail(self):
        rules = rs(self.RULE)
        good = plan(step(1, "create_widget", {"size": 5}), step(2, "bind_gadget", {"size": 99}))
        bad = plan(step(1, "create_widget", {"size": 50}), step(2, "create_widget", {"size": 1}))
        self.assertEqual(engine.check_plan(good, rules), [])
        found = engine.check_plan(bad, rules)
        self.assertEqual([(v.rule_id, v.steps, v.severity, v.message) for v in found], [("widget-sized", (1,), "reject", "widgets stay small")])

    def test_match_by_tool_operation_list_regex_and_where(self):
        rules = rs({"id": "r", "type": "step_requires", "match": {"tool": ["invoke_operation"], "operation": ["create_widget", "bind_gadget"], "where": {"path": "args.arguments.body.kind", "op": "exists"}},
                    "require": {"path": "purpose", "op": "exists"}})
        p = plan(step(1, "create_widget", {"kind": "a"}), step(2, "bind_gadget", {"kind": "b"}, purpose="ok"), step(3, "bind_gadget", {}), step(4, "x", {"kind": "a"}))
        self.assertEqual([v.steps for v in engine.check_plan(p, rules)], [(1,)])
        rules = rs({"id": "r", "type": "step_requires", "match": {"operation_regex": "^create_"}, "require": {"path": "purpose", "op": "exists"}})
        self.assertEqual([v.steps for v in engine.check_plan(plan(step(1, "create_a"), step(2, "make_b")), rules)], [(1,)])

    def test_configurable_field_paths(self):
        rules = rs({"id": "r", "type": "step_requires", "match": {"tool": "t1", "operation": "op1"}, "require": {"path": "ok", "op": "equals", "value": True}},
                   paths={"tool": "kind", "operation": "call.name"})
        p = {"steps": [{"kind": "t1", "call": {"name": "op1"}, "ok": False}, {"kind": "t1", "call": {"name": "op2"}}]}
        self.assertEqual([v.steps for v in engine.check_plan(p, rules)], [(1,)], "steps without a `step` number use their position")

    def test_custom_steps_path_and_non_plans(self):
        rules = rs({"id": "r", "type": "step_requires", "match": {"tool": "t"}, "require": {"path": "ok", "op": "exists"}}, steps_path="plan.items")
        self.assertEqual(len(engine.check_plan({"plan": {"items": [{"tool": "t"}]}}, rules)), 1)
        self.assertEqual(engine.check_plan({"steps": [{"tool": "t"}]}, rules), [])
        self.assertEqual(engine.check_plan({"plan": {"items": "nope"}}, rules), [])


class RefRequiresTests(unittest.TestCase):
    RULE = {
        "id": "gadget-widget-kind", "type": "ref_requires",
        "match": {"operation": "bind_gadget"}, "ref": {"path": "args.arguments.body.widget"},
        "creator": {"match": MATCH_CREATE, "name_path": "args.arguments.body.name"},
        "require": {"path": "args.arguments.body.kind", "op": "equals", "value": "sturdy"},
    }

    def test_the_creating_step_must_satisfy_the_condition(self):
        rules = rs(self.RULE)
        ok = plan(step(1, "create_widget", {"name": "w1", "kind": "sturdy"}), step(2, "bind_gadget", {"widget": "w1"}))
        bad = plan(step(1, "create_widget", {"name": "w1", "kind": "flimsy"}), step(2, "bind_gadget", {"widget": "w1"}))
        self.assertEqual(engine.check_plan(ok, rules), [])
        found = engine.check_plan(bad, rules)
        self.assertEqual([(v.rule_id, v.steps) for v in found], [("gadget-widget-kind", (2, 1))])
        self.assertIn("'w1'", found[0].message)

    def test_a_reference_with_no_creating_step_is_reported_by_default(self):
        found = engine.check_plan(plan(step(1, "bind_gadget", {"widget": "ghost"})), rs(self.RULE))
        self.assertEqual([(v.rule_id, v.steps) for v in found], [("gadget-widget-kind", (1,))])
        self.assertIn("no step creates", found[0].message)

    def test_on_missing_ignore(self):
        rules = rs({**self.RULE, "on_missing": "ignore"})
        self.assertEqual(engine.check_plan(plan(step(1, "bind_gadget", {"widget": "ghost"})), rules), [])
        bad = plan(step(1, "create_widget", {"name": "w1", "kind": "flimsy"}), step(2, "bind_gadget", {"widget": "w1"}))
        self.assertEqual(len(engine.check_plan(bad, rules)), 1, "a resolved ref still has to satisfy the condition")

    def test_no_reference_value_means_nothing_to_check(self):
        self.assertEqual(engine.check_plan(plan(step(1, "bind_gadget", {})), rs(self.RULE)), [])

    def test_wildcard_refs_and_before(self):
        rule = {**self.RULE, "ref": {"path": "args.arguments.body.widgets.*"}, "before": True,
                "require": {"path": "args.arguments.body.name", "op": "exists"}}
        p = plan(step(1, "bind_gadget", {"widgets": ["a", "b"]}), step(2, "create_widget", {"name": "a"}), step(3, "create_widget", {"name": "b"}))
        found = engine.check_plan(p, rs(rule))
        self.assertEqual([v.steps for v in found], [(1,), (1,)])
        self.assertIn("before any step creates it", found[0].message)
        ok = plan(step(1, "create_widget", {"name": "a"}), step(2, "create_widget", {"name": "b"}), step(3, "bind_gadget", {"widgets": ["a", "b"]}))
        self.assertEqual(engine.check_plan(ok, rs(rule)), [])


class SemanticsTests(unittest.TestCase):
    def test_missing_path_fails_every_op_but_not_exists_unless_missing_pass(self):
        node = {"a": 1}
        for op, value in (("equals", 1), ("not_equals", 1), ("in", [1]), ("not_in", [2]), ("regex", "x"), ("contains", "x"), ("max", 5), ("min", 0)):
            self.assertFalse(engine.holds({"path": "nope", "op": op, "value": value}, node), op)
            self.assertTrue(engine.holds({"path": "nope", "op": op, "value": value, "missing": "pass"}, node), op)
            self.assertFalse(engine.holds({"path": "nope", "op": op, "value": value, "missing": "fail"}, node), op)
        self.assertFalse(engine.holds({"path": "nope", "op": "exists", "missing": "pass"}, node), "exists only looks at presence")
        self.assertTrue(engine.holds({"path": "nope", "op": "not_exists"}, node))
        self.assertFalse(engine.holds({"path": "a", "op": "equals", "value": 2, "missing": "pass"}, node), "a present value is judged")

    def test_missing_pass_in_a_rule(self):
        rule = {"id": "r", "type": "step_requires", "match": MATCH_CREATE, "require": {"path": "args.arguments.body.size", "op": "max", "value": 10, "missing": "pass"}}
        self.assertEqual(engine.check_plan(plan(step(1, "create_widget", {})), rs(rule)), [])
        self.assertEqual(len(engine.check_plan(plan(step(1, "create_widget", {"size": 11})), rs(rule))), 1)

    def test_before_with_several_creators_needs_only_one_preceding_that_meets_require(self):
        rule = {**RefRequiresTests.RULE, "before": True}
        ok = plan(step(1, "create_widget", {"name": "w", "kind": "sturdy"}), step(2, "bind_gadget", {"widget": "w"}), step(3, "create_widget", {"name": "w", "kind": "flimsy"}))
        self.assertEqual(engine.check_plan(ok, rs(rule)), [], "the later creator is irrelevant")
        bad = plan(step(1, "create_widget", {"name": "w", "kind": "flimsy"}), step(2, "bind_gadget", {"widget": "w"}), step(3, "create_widget", {"name": "w", "kind": "sturdy"}))
        self.assertEqual([v.steps for v in engine.check_plan(bad, rs(rule))], [(2, 1)], "the preceding creator fails require")
        late_only = plan(step(1, "bind_gadget", {"widget": "w"}), step(2, "create_widget", {"name": "w", "kind": "sturdy"}), step(3, "create_widget", {"name": "w", "kind": "sturdy"}))
        self.assertEqual([v.steps for v in engine.check_plan(late_only, rs(rule))], [(1,)])

    def test_order_is_list_position_not_the_step_number(self):
        rule = {**RefRequiresTests.RULE, "before": True}
        p = plan(step(7, "create_widget", {"name": "w", "kind": "sturdy"}), step(7, "bind_gadget", {"widget": "w"}))
        self.assertEqual(engine.check_plan(p, rs(rule)), [])
        p = plan(step(1, "bind_gadget", {"widget": "w"}), step(1, "create_widget", {"name": "w", "kind": "sturdy"}))
        self.assertEqual([v.steps for v in engine.check_plan(p, rs(rule))], [(1,)])
        req = rs({"id": "v", "type": "requires_step", "before": True, "if": MATCH_CREATE, "then": {"tool": "verify"}})
        p = plan({"step": 9, "tool": "verify", "args": {}}, step(2, "create_widget"))
        self.assertEqual(engine.check_plan(p, req), [], "verify is first in the list although its number is higher")

    def test_list_valued_references_are_flattened_one_level(self):
        p = plan(step(1, "create_widget", {"name": "a", "kind": "sturdy"}), step(2, "create_widget", {"name": "b", "kind": "flimsy"}),
                 step(3, "bind_gadget", {"widget": ["a", "b", "ghost"]}))
        found = engine.check_plan(p, rs(RefRequiresTests.RULE))
        self.assertEqual(sorted(v.steps for v in found), [(3,), (3, 2)], "b fails require, ghost is missing, a is fine")

    def test_per_rule_steps_path(self):
        rules = rs({"id": "v", "type": "step_requires", "steps_path": "verification", "match": {"tool": "check"}, "require": {"path": "args.target", "op": "exists"}},
                   {"id": "s", "type": "step_requires", "match": {"tool": "t"}, "require": {"path": "ok", "op": "exists"}})
        p = {"steps": [{"tool": "t"}], "verification": [{"tool": "check", "args": {}}, {"tool": "check", "args": {"target": "x"}}]}
        self.assertEqual([(v.rule_id, v.steps) for v in engine.check_plan(p, rules)], [("v", (1,)), ("s", (1,))])

    def test_regex_input_is_capped(self):
        long_value = "a" * engine.MAX_REGEX_INPUT + "TAIL"
        self.assertFalse(engine.holds({"path": "x", "op": "regex", "value": "TAIL"}, {"x": long_value}))
        self.assertTrue(engine.holds({"path": "x", "op": "regex", "value": "^a+$"}, {"x": long_value}))


class RequiresStepTests(unittest.TestCase):
    RULE = {"id": "verify-after", "type": "requires_step", "if": MATCH_CREATE, "then": {"tool": "verify"}}

    def test_a_trigger_needs_a_matching_step(self):
        rules = rs(self.RULE)
        self.assertEqual(engine.check_plan(plan(step(1, "bind_gadget")), rules), [], "no trigger, no requirement")
        self.assertEqual([v.steps for v in engine.check_plan(plan(step(1, "create_widget")), rules)], [(1,)])
        self.assertEqual(engine.check_plan(plan(step(1, "create_widget"), {"step": 2, "tool": "verify", "args": {}}), rules), [])

    def test_before(self):
        rules = rs({**self.RULE, "before": True})
        late = plan(step(1, "create_widget"), {"step": 2, "tool": "verify", "args": {}})
        early = plan({"step": 1, "tool": "verify", "args": {}}, step(2, "create_widget"))
        self.assertEqual(len(engine.check_plan(late, rules)), 1)
        self.assertEqual(engine.check_plan(early, rules), [])


class ForbidTogetherTests(unittest.TestCase):
    RULE = {"id": "no-both", "type": "forbid_together", "a": {"operation": "create_widget"}, "b": {"operation": "bind_gadget"}}

    def test_both_present_in_one_plan(self):
        rules = rs(self.RULE)
        self.assertEqual(engine.check_plan(plan(step(1, "create_widget"), step(2, "create_widget")), rules), [])
        found = engine.check_plan(plan(step(1, "create_widget"), step(4, "bind_gadget")), rules)
        self.assertEqual([(v.rule_id, v.steps) for v in found], [("no-both", (1, 4))])

    def test_same_step(self):
        rule = {"id": "same", "type": "forbid_together", "same_step": True,
                "a": {"operation": "create_widget", "where": {"path": "args.arguments.body.x", "op": "exists"}},
                "b": {"operation": "create_widget", "where": {"path": "args.arguments.body.y", "op": "exists"}}}
        rules = rs(rule)
        self.assertEqual(engine.check_plan(plan(step(1, "create_widget", {"x": 1}), step(2, "create_widget", {"y": 1})), rules), [])
        self.assertEqual([v.steps for v in engine.check_plan(plan(step(1, "create_widget", {"x": 1, "y": 1})), rules)], [(1,)])


class SeverityAndDocumentTests(unittest.TestCase):
    def test_warn_is_reported_with_its_severity_and_split_out(self):
        rules = rs({"id": "w", "type": "step_requires", "severity": "warn", "match": MATCH_CREATE, "require": {"path": "purpose", "op": "exists"}},
                   {"id": "e", "type": "step_requires", "match": MATCH_CREATE, "require": {"path": "never", "op": "exists"}})
        found = engine.check_plan(plan(step(1, "create_widget")), rules)
        rejects, warns = engine.violations_by_severity(found)
        self.assertEqual((ids(rejects), ids(warns)), (["e"], ["w"]))
        self.assertEqual(engine.reject_reason_for("e"), "rule_violation:e")

    def test_bad_documents_are_rejected_with_a_pointer(self):
        good = {"id": "r", "type": "step_requires", "match": MATCH_CREATE, "require": {"path": "a", "op": "exists"}}
        cases = {
            "not json": "{",
            "list": [],
            "no rules": {"rules": []},
            "unknown type": {"rules": [{**good, "type": "nope"}]},
            "unknown op": {"rules": [{**good, "require": {"path": "a", "op": "bogus"}}]},
            "extra key": {"rules": [{**good, "surprise": 1}]},
            "bad severity": {"rules": [{**good, "severity": "fatal"}]},
            "empty match": {"rules": [{**good, "match": {}}]},
            "bad id": {"rules": [{**good, "id": "has space"}]},
            "duplicate id": {"rules": [good, good]},
            "bad regex": {"rules": [{**good, "require": {"path": "a", "op": "regex", "value": "("}}]},
            "bad op_regex": {"rules": [{**good, "match": {"operation_regex": "("}}]},
            "too many": {"rules": [{**good, "id": f"r{i}"} for i in range(engine.MAX_RULES + 1)]},
            "too big": {"rules": [{**good, "description": "d" * 400, "id": f"r{i}"} for i in range(190)]},
        }
        for name, doc in cases.items():
            with self.subTest(name), self.assertRaises(engine.PlanRulesError):
                engine.load_rules(doc if name != "not json" else "{")
        self.assertEqual(engine.load_rules(json.dumps({"rules": [good]})).tool_path, "tool")

    def test_per_op_value_types_and_regex_limits(self):
        def doc(cond):
            return {"rules": [{"id": "r", "type": "step_requires", "match": MATCH_CREATE, "require": cond}]}
        bad = [
            {"path": "a", "op": "in", "value": "x"}, {"path": "a", "op": "not_in"}, {"path": "a", "op": "max", "value": "abc"},
            {"path": "a", "op": "min"}, {"path": "a", "op": "regex", "value": 5}, {"path": "a", "op": "equals"},
            {"path": "a", "op": "regex", "value": "(a+)+$"}, {"path": "a", "op": "regex", "value": "a" * 201},
            {"all": [{"path": "a", "op": "in", "value": 1}]},
        ]
        for cond in bad:
            with self.subTest(cond), self.assertRaises(engine.PlanRulesError):
                engine.load_rules(doc(cond))
        for cond in ({"path": "a", "op": "in", "value": [1]}, {"path": "a", "op": "max", "value": "5"}, {"path": "a", "op": "equals", "value": None}):
            engine.load_rules(doc(cond))
        with self.assertRaises(engine.PlanRulesError):
            engine.load_rules({"rules": [{"id": "r", "type": "step_requires", "match": {"operation_regex": "(a*)*"}, "require": {"path": "a", "op": "exists"}}]})

    def test_schema_errors_are_chosen_by_rule_type(self):
        with self.assertRaises(engine.PlanRulesError) as ctx:
            engine.load_rules({"rules": [{"id": "r", "type": "ref_requires", "match": MATCH_CREATE}]})
        self.assertIn("/rules/0", ctx.exception.pointer)
        self.assertIn("required", ctx.exception.message)
        with self.assertRaises(engine.PlanRulesError) as ctx:
            engine.load_rules({"rules": [{"id": "r", "type": "nope"}]})
        self.assertEqual(ctx.exception.pointer, "/rules/0/type")

    def test_a_pointer_names_the_bad_part(self):
        try:
            engine.load_rules({"rules": [{"id": "r", "type": "step_requires", "match": MATCH_CREATE, "require": {"path": "a", "op": "bogus"}}]})
        except engine.PlanRulesError as exc:
            self.assertTrue(exc.pointer.startswith("/rules/0"), exc.pointer)
        else:
            self.fail("expected PlanRulesError")


# ── generation ──────────────────────────────────────────────────────

class GenerationTests(unittest.TestCase):
    def ctx(self, rules):
        from test_structured_plan_playbooks import BUNDLE, make_ctx

        ctx = make_ctx()
        ctx["params"]["plan_rules"] = rules
        ctx["grid_bundle"] = BUNDLE
        ctx["grid_tasks"] = [{"scenario": f"S{i}", "passages": [], "cell": ctx["grid_cells"][i]} for i in range(3)]
        return ctx

    RULES = {"rules": [
        {"id": "no-delete", "type": "step_requires", "match": {"tool": "invoke_operation"}, "require": {"path": "args.operation", "op": "not_equals", "value": "delete"}},
        {"id": "prefer-purpose", "type": "step_requires", "severity": "warn", "match": {"tool": "invoke_operation"}, "require": {"path": "purpose", "op": "exists"}},
    ]}

    def plan(self, operation):
        return {"name": "p", "environment": "dev", "steps": [{"action": "create", "target": "svc", "tool": "invoke_operation", "args": {"operation": operation}}]}

    def test_reject_severity_rejects_with_rule_ids_and_warn_only_annotates(self):
        pb = get_playbook("structured-plan", SynthMode.SCENARIO_GRID)
        ctx = self.ctx(self.RULES)
        out = pb.validate([{"index": 1, "plan_json": self.plan("delete")}, {"index": 2, "plan_json": self.plan("create_widget")}], ctx)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["payload"]["plan_rule_warnings"], ["prefer-purpose"])
        rejected = [r for r in ctx["rejects"] if r["reason"] == "rule_violation"]
        self.assertEqual(len(rejected), 1)
        self.assertIn("no-delete", rejected[0]["detail"])
        self.assertEqual(rejected[0]["payload"]["rule_ids"], ["no-delete"])

    def test_rules_run_after_schema_validation(self):
        pb = get_playbook("structured-plan", SynthMode.SCENARIO_GRID)
        ctx = self.ctx(self.RULES)
        bad = self.plan("delete")
        bad["environment"] = "moon"
        pb.validate([{"index": 1, "plan_json": bad}], ctx)
        self.assertEqual([r["reason"] for r in ctx["rejects"] if r["reason"] != "missing_answer"], ["schema_invalid"])

    def test_no_rules_means_no_change(self):
        pb = get_playbook("structured-plan", SynthMode.SCENARIO_GRID)
        ctx = self.ctx(None)
        ctx["params"].pop("plan_rules")
        out = pb.validate([{"index": 1, "plan_json": self.plan("delete")}], ctx)
        self.assertEqual(len(out), 1)
        self.assertNotIn("plan_rule_warnings", out[0]["payload"])

    def test_check_params_validates_plan_rules(self):
        base = {"matrix_text": "| a | b |\n|---|---|\n| 1 | 2 |"}
        self.assertIsNone(check_params({**base, "plan_rules": self.RULES}))
        name, message = check_params({**base, "plan_rules": {"rules": [{"id": "x"}]}})
        self.assertEqual(name, "plan_rules")
        self.assertTrue(message)


# ── pending-row check + endpoint ────────────────────────────────────

RULES_DOC = {"rules": [
    {"id": "gadget-needs-sturdy-widget", "type": "ref_requires", "description": "a gadget binds only to a sturdy widget",
     "match": {"operation": "bind_gadget"}, "ref": {"path": "args.arguments.body.widget"},
     "creator": {"match": MATCH_CREATE, "name_path": "args.arguments.body.name"},
     "require": {"path": "args.arguments.body.kind", "op": "equals", "value": "sturdy"}},
    {"id": "has-purpose", "type": "step_requires", "severity": "warn", "match": MATCH_CREATE, "require": {"path": "purpose", "op": "exists"}},
]}


def pending(rid, p, **fields):
    return {"id": rid, "row_kind": "structured_plan", "scenario": f"s{rid}", "plan_json": p, "review_status": "pending",
            "synth_confidence": 0.9, "synth_source": "playbook:r:m", **fields}


FLIMSY = plan(step(1, "create_widget", {"name": "w", "kind": "flimsy"}, purpose="p"), step(2, "bind_gadget", {"widget": "w"}))
MISSING = plan(step(1, "bind_gadget", {"widget": "ghost"}))
CLEAN = plan(step(1, "create_widget", {"name": "w", "kind": "sturdy"}, purpose="p"), step(2, "bind_gadget", {"widget": "w"}))
WARN_ONLY = plan(step(1, "create_widget", {"name": "w", "kind": "sturdy"}), step(2, "bind_gadget", {"widget": "w"}))


class CheckEndpointTests(unittest.TestCase):
    def setUp(self):
        r = client.post("/api/projects", json={"name": f"rules-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        self.base = f"/api/projects/{self.pid}/synthetic"
        self.path = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "synthetic" / "synthetic.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            pending(1, FLIMSY), pending(2, MISSING), pending(3, CLEAN), pending(4, WARN_ONLY),
            pending(5, FLIMSY, review_status="accepted"), {"id": 6, "question": "q", "answer": "a", "review_status": "pending"},
            pending(7, FLIMSY, synth_source="playbook:other"),
        ]
        self.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def tearDown(self):
        wait_for_jobs_idle(client)
        shutil.rmtree(self.path.parent.parent, ignore_errors=True)

    def stored(self):
        return {r["id"]: r for r in (json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())}

    def post(self, **body):
        body.setdefault("rules", RULES_DOC)
        return client.post(f"{self.base}/check-plan-rules", json=body)

    def test_dry_run_counts_by_rule_and_changes_nothing(self):
        before = self.path.read_text(encoding="utf-8")
        resp = self.post()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["dry_run"])
        self.assertEqual((body["checked"], body["flagged"], body["warned_only"], body["rejected"]), (5, 3, 1, 0), "pending plan rows only")
        rule = body["by_rule"]["gadget-needs-sturdy-widget"]
        self.assertEqual((rule["severity"], rule["count"], sorted(rule["sample_row_ids"])), ("reject", 3, [1, 2, 7]))
        self.assertEqual(body["by_rule"]["has-purpose"]["count"], 1)
        self.assertEqual(body["by_rule"]["has-purpose"]["sample_row_ids"], [4])
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_apply_marks_rejected_with_the_rule_and_is_reversible(self):
        body = self.post(dry_run=False).json()
        self.assertEqual((body["rejected"], body["dry_run"]), (3, False))
        rows = self.stored()
        for rid in (1, 2, 7):
            self.assertEqual(rows[rid]["review_status"], "rejected")
            self.assertEqual(rows[rid]["reject_reason"], "rule_violation:gadget-needs-sturdy-widget")
            self.assertEqual(rows[rid]["plan_rule_ids"], ["gadget-needs-sturdy-widget"])
        self.assertEqual([rows[i]["review_status"] for i in (3, 4, 5, 6)], ["pending", "pending", "accepted", "pending"], "warn-only and clean rows are untouched")
        queue = client.get(f"{self.base}/review-queue").json()
        self.assertEqual(queue["total_rejected"], 3)
        self.assertEqual(len(rows), 7, "never purged")
        self.assertEqual(self.post(dry_run=False).json()["rejected"], 0, "a second apply finds nothing pending")

    def test_filters(self):
        body = self.post(source="playbook:r:m").json()
        self.assertEqual((body["checked"], body["flagged"]), (4, 2))
        body = self.post(row_ids=[3, 4]).json()
        self.assertEqual((body["checked"], body["flagged"]), (2, 0))

    def test_rules_as_json_text(self):
        self.assertEqual(self.post(rules=json.dumps(RULES_DOC)).json()["flagged"], 3)

    def test_bad_rules_are_a_400_with_an_error_code(self):
        for bad in ({"rules": [{"id": "x", "type": "nope"}]}, {"rules": []}, "{", {"rules": [{"id": "x", "type": "step_requires", "match": {}, "require": {"path": "a", "op": "exists"}}]}):
            resp = self.post(rules=bad)
            self.assertEqual(resp.status_code, 400, resp.text)
            self.assertEqual(resp.json()["error_code"], "invalid_plan_rules")
            self.assertTrue(resp.json()["detail"])
        self.assertEqual(self.stored()[1]["review_status"], "pending")

    def test_job_dry_run_and_apply(self):
        resp = client.post(f"{self.base}/check-plan-rules?async_job=true", json={"rules": RULES_DOC})
        self.assertEqual(resp.status_code, 202, resp.text)
        wait_for_jobs_idle(client)
        done = client.get(f"/api/jobs/{resp.json()['id']}").json()
        self.assertEqual((done["kind"], done["status"]), ("plan_rules_check", "succeeded"))
        self.assertEqual(done["result"]["flagged"], 3)
        self.assertEqual(self.stored()[1]["review_status"], "pending")
        resp = client.post(f"{self.base}/check-plan-rules?async_job=true", json={"rules": RULES_DOC, "dry_run": False})
        wait_for_jobs_idle(client)
        done = client.get(f"/api/jobs/{resp.json()['id']}").json()
        self.assertEqual((done["status"], done["result"]["rejected"]), ("succeeded", 3))
        self.assertEqual(self.stored()[1]["review_status"], "rejected")

    def post_restore(self, **body):
        return client.post(f"{self.base}/review-queue/restore", json=body)

    def test_apply_then_restore_round_trips_the_rows_exactly(self):
        before = self.stored()
        self.post(dry_run=False)
        self.assertNotEqual(self.stored()[1], before[1])
        dry = self.post_restore(reject_reason_prefix="rule_violation:").json()
        self.assertEqual((dry["dry_run"], dry["matched"], dry["restored"], dry["by_reason"]), (True, 3, 0, {"rule_violation:gadget-needs-sturdy-widget": 3}))
        self.assertEqual(self.stored()[1]["review_status"], "rejected", "a dry run changes nothing")
        done = self.post_restore(reject_reason_prefix="rule_violation:", dry_run=False).json()
        self.assertEqual((done["matched"], done["restored"]), (3, 3))
        self.assertEqual(self.stored(), before, "identical row content")
        self.assertEqual(self.post_restore(reject_reason_prefix="rule_violation:", dry_run=False).json()["restored"], 0)

    def test_restore_selectors(self):
        self.post(dry_run=False)
        rid = "rule_violation:gadget-needs-sturdy-widget"
        self.assertEqual(self.post_restore(reject_reason="rule_violation:other").json()["matched"], 0)
        self.assertEqual(self.post_restore(reject_reason=rid, row_ids=[1, 3]).json()["matched"], 1, "selectors are ANDed; 3 is not rejected")
        self.assertEqual(self.post_restore(row_ids=[2], dry_run=False).json()["restored"], 1)
        stored = self.stored()
        self.assertEqual((stored[2]["review_status"], "reject_reason" in stored[2], "plan_rule_ids" in stored[2]), ("pending", False, False))
        self.assertEqual(stored[1]["review_status"], "rejected")
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        rows[3].update(review_status="rejected", reject_reason="language_mismatch")
        rows[5].update(review_status="rejected")
        self.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        self.assertEqual(self.post_restore(reject_reason="language_mismatch", dry_run=False).json()["restored"], 1)
        self.assertEqual(self.post_restore(reject_reason="", dry_run=False).json()["restored"], 1)
        self.assertEqual(self.stored()[6]["review_status"], "pending")

    def test_language_mismatch_rows_restore_to_identical_pending_rows(self):
        mislabelled = {"id": 8, "row_kind": "code_example", "language": "xml", "review_status": "pending", "task": "t",
                       "code_example_json": {"code": "def f():\n    return 1\n"}}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(mislabelled) + "\n")
        before = self.stored()
        url = f"{self.base}/review-queue/recheck-code-format"
        self.assertEqual(client.post(url, json={"dry_run": False}).json()["rejected"], 1)
        self.assertEqual(self.stored()[8]["reject_reason"], "language_mismatch")
        done = self.post_restore(reject_reason="language_mismatch", dry_run=False).json()
        self.assertEqual((done["matched"], done["restored"]), (1, 1))
        self.assertEqual(self.stored(), before)

    def test_restore_needs_a_selector_and_caps_row_ids(self):
        resp = self.post_restore()
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error_code"], "invalid_restore_request")
        self.assertEqual(self.post_restore(row_ids=list(range(5001))).status_code, 422)

    def test_job_with_nothing_to_check_fails_loudly(self):
        resp = client.post(f"{self.base}/check-plan-rules?async_job=true", json={"rules": RULES_DOC, "row_ids": [999]})
        wait_for_jobs_idle(client)
        done = client.get(f"/api/jobs/{resp.json()['id']}").json()
        self.assertEqual(done["status"], "failed")

    def test_bad_rules_never_start_a_job(self):
        resp = client.post(f"{self.base}/check-plan-rules?async_job=true", json={"rules": {"rules": []}})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()

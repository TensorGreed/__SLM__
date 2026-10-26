"""User-authored probe packs: runner properties, severity / critical
aggregation, validator, project store, API, eval wiring and the always-on
critical gate. Neutral toy content only; no model inference (stubbed)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.services import evaluation_service
from app.services import probe_runner as pr
from app.services import probe_pack_service as pps

_client: TestClient | None = None
_cm = None


def setUpModule():
    global _client, _cm
    _cm = TestClient(app)
    _client = _cm.__enter__()


def tearDownModule():
    _cm.__exit__(None, None, None)


def _probe(**kw):
    base = {"id": "p1", "prompt": "hi", "property": "must_match", "pattern": "ok"}
    base.update(kw)
    return base


def _pack(probes, pack_id="toy-safety", **kw):
    return {"id": pack_id, "name": "Toy", "version": "1", "probes": probes, **kw}


class ScoreProbeTests(unittest.TestCase):
    def test_must_match_and_flags(self):
        p = {"property": "must_match", "pattern": "hello"}
        self.assertTrue(pr.score_probe(p, "say hello there")[0])
        self.assertFalse(pr.score_probe(p, "HELLO")[0])
        self.assertTrue(pr.score_probe({**p, "flags": ["i"]}, "HELLO")[0])
        multi = {"property": "must_match", "pattern": "^b$", "flags": ["m"]}
        self.assertTrue(pr.score_probe(multi, "a\nb\nc")[0])
        dotall = {"property": "must_match", "pattern": "a.b", "flags": ["s"]}
        self.assertTrue(pr.score_probe(dotall, "a\nb")[0])

    def test_must_not_match_quotes_first_match(self):
        p = {"property": "must_not_match", "pattern": "(?i)begin private key"}
        ok, reason = pr.score_probe(p, "all fine")
        self.assertTrue(ok)
        ok, reason = pr.score_probe(p, "-----BEGIN PRIVATE KEY----- abc")
        self.assertFalse(ok)
        self.assertIn("BEGIN PRIVATE KEY", reason)
        long = {"property": "must_not_match", "pattern": "x+"}
        self.assertLessEqual(len(pr.score_probe(long, "x" * 500)[1]), 140)

    def test_output_truncated_before_regex(self):
        p = {"property": "must_match", "pattern": "needle"}
        self.assertFalse(pr.score_probe(p, "a" * 20_000 + "needle")[0])
        self.assertTrue(pr.score_probe(p, "a" * 100 + "needle")[0])

    def test_json_pointer_not_in_wildcard(self):
        p = {"property": "json_pointer_not_in", "pointer": "/findings/*/label", "values": ["secret.hardcoded"]}
        good = '{"findings": [{"label": "injection"}, {"label": "injection.query"}]}'
        bad = '{"findings": [{"label": "injection"}, {"label": "secret.hardcoded"}]}'
        self.assertTrue(pr.score_probe(p, good)[0])
        ok, reason = pr.score_probe(p, bad)
        self.assertFalse(ok)
        self.assertIn("secret.hardcoded", reason)

    def test_json_pointer_absent_passes_with_note(self):
        p = {"property": "json_pointer_not_in", "pointer": "/findings/*/label", "values": ["x"]}
        self.assertEqual(pr.score_probe(p, '{"other": 1}')[0], True)
        self.assertEqual(pr.probe_note(p, '{"other": 1}', None), "pointer_absent")
        self.assertIsNone(pr.probe_note(p, '{"findings": [{"label": "y"}]}', None))
        self.assertTrue(pr.score_probe(p, '{"findings": []}')[0])

    def test_json_pointer_unparseable_fails(self):
        p = {"property": "json_pointer_not_in", "pointer": "/a", "values": ["x"]}
        ok, reason = pr.score_probe(p, "no json here")
        self.assertFalse(ok)
        self.assertEqual(reason, "output_not_json")

    def test_json_pointer_fenced_and_prose_and_array(self):
        p = {"property": "json_pointer_not_in", "pointer": "/a", "values": ["x"]}
        self.assertFalse(pr.score_probe(p, 'Sure:\n```json\n{"a": "x"}\n```\nDone')[0])
        self.assertFalse(pr.score_probe(p, 'Here you go {"a": "x"} bye')[0])
        arr = {"property": "json_pointer_not_in", "pointer": "/0/a", "values": [1]}
        self.assertFalse(pr.score_probe(arr, '[{"a": 1}]')[0])
        # bool is not the number 1
        self.assertTrue(pr.score_probe(arr, '[{"a": true}]')[0])

    def test_pointer_escapes_and_object_wildcard(self):
        p = {"property": "json_pointer_not_in", "pointer": "/a~1b/*", "values": ["bad"]}
        self.assertFalse(pr.score_probe(p, '{"a/b": {"k": "bad"}}')[0])
        self.assertTrue(pr.score_probe(p, '{"a/b": {"k": "fine"}}')[0])


class RunnerAggregationTests(unittest.TestCase):
    def _user(self, pid, severity, critical, prop="must_not_match", **kw):
        return {
            "id": f"user:toy-pack/{pid}", "source": "user", "pack_id": "toy-pack",
            "severity": severity, "critical": critical, "probe_kind": "format_robustness",
            "property": prop, "pattern": "BAD", "input": "q",
            "weight": pr.severity_weight(severity), **kw,
        }

    def test_severity_weights(self):
        self.assertEqual([pr.severity_weight(s) for s in ("low", "medium", "high")], [1.0, 2.0, 3.0])
        snap = pr.run_probe_pack(
            [self._user("a", "high", False), self._user("b", "low", False)],
            lambda xs: ["BAD" for _ in xs][:1] + ["fine"] * (len(xs) - 1),
            weights={"format_robustness": 9.0},  # kind weights must not apply
        )
        by_id = {r["id"]: r for r in snap["results"]}
        self.assertEqual(by_id["user:toy-pack/a"]["weight"], 3.0)
        self.assertEqual(by_id["user:toy-pack/b"]["weight"], 1.0)
        # a fails (w3), b passes (w1) -> 1/4
        self.assertAlmostEqual(snap["user_probe_pass_rate"], 0.25)
        # user-only run: no platform score exists
        self.assertIsNone(snap["probe_pass_rate"])
        self.assertEqual(by_id["user:toy-pack/a"]["scored_by"], "deterministic")

    def test_explicit_weight_wins(self):
        # validate -> applicable -> run: a per-probe weight survives the whole path
        doc = {"id": "toy-pack", "probes": [
            {"id": "a", "prompt": "q", "property": "must_not_match", "pattern": "BAD", "severity": "low", "weight": 7.0},
            {"id": "b", "prompt": "q", "property": "must_not_match", "pattern": "BAD", "severity": "high"},
        ]}
        pack, errors = pps.validate_user_probe_pack(doc)
        self.assertEqual(errors, [])
        project = SimpleNamespace(runtime_config={})
        pps.set_project_user_probe_packs(project, [pack], source={"kind": "manual"})
        probes = pps.applicable_user_probes(project, None, None)
        snap = pr.run_probe_pack(probes, lambda xs: ["x"] * len(xs))
        by_id = {r["id"]: r for r in snap["results"]}
        self.assertEqual(by_id["user:toy-pack/a"]["weight"], 7.0)
        self.assertEqual(by_id["user:toy-pack/b"]["weight"], 3.0)

    def test_bad_weight_and_critical_rejected(self):
        for bad in (0, -1, 11, "3", True):
            _, errors = pps.validate_user_probe_pack({"id": "t", "probes": [
                {"id": "a", "prompt": "q", "property": "must_match", "pattern": "x", "weight": bad}]})
            self.assertIn("bad_weight", [e["code"] for e in errors], bad)
        _, errors = pps.validate_user_probe_pack({"id": "t", "probes": [
            {"id": "a", "prompt": "q", "property": "must_match", "pattern": "x", "critical": "false"}]})
        self.assertIn("bad_critical", [e["code"] for e in errors])

    def test_user_results_never_move_platform_numbers(self):
        platform = {"id": "plat.1", "probe_kind": "robustness", "property": "handles_degenerate_gracefully", "input": ""}
        user = self._user("n1", "high", False)
        snap = pr.run_probe_pack([platform, user], lambda xs: ["ok", "BAD"][: len(xs)])
        self.assertEqual(snap["probe_pass_rate"], 1.0)
        self.assertEqual(snap["unweighted_pass_rate"], 1.0)
        self.assertEqual(snap["user_probe_pass_rate"], 0.0)
        self.assertNotIn("must_not_match", snap["per_property"])
        self.assertIn("must_not_match", snap["user_per_property"])
        self.assertEqual(pr.reweighted_pass_rate(snap["results"], None), 1.0)

    def test_critical_aggregation_and_by_source(self):
        platform = {
            "id": "plat.1", "probe_kind": "robustness", "property": "handles_degenerate_gracefully", "input": "",
        }
        probes = [
            platform,
            self._user("c1", "high", True),
            self._user("c2", "medium", True),
            self._user("n1", "low", False),
        ]
        outs = {"q": "x"}

        def predict(xs):
            # order: platform input, c1, c2, n1 -> c1 and n1 fail
            return ["ok", "BAD", "fine", "BAD"][: len(xs)]

        snap = pr.run_probe_pack(probes, predict)
        self.assertEqual(snap["critical_total"], 2)
        self.assertEqual(snap["critical_failed"], 1)
        self.assertEqual([f["id"] for f in snap["critical_failures"]], ["user:toy-pack/c1"])
        self.assertIn("BAD", snap["critical_failures"][0]["reason"])
        self.assertEqual(snap["by_source"]["platform"], {"passed": 1, "total": 1, "pass_rate": 1.0})
        self.assertEqual(snap["by_source"]["user"]["total"], 3)
        self.assertEqual(snap["by_source"]["user"]["passed"], 1)
        plat = next(r for r in snap["results"] if r["id"] == "plat.1")
        self.assertEqual(plat["source"], "platform")
        self.assertNotIn("critical", plat)
        usr = next(r for r in snap["results"] if r["id"].endswith("/c1"))
        self.assertEqual((usr["pack_id"], usr["severity"], usr["critical"]), ("toy-pack", "high", True))
        # user probes stay out of the per-kind breakdown
        self.assertEqual(snap["weighted_by_kind"]["robustness"]["total"], 1)
        self.assertNotIn("format_robustness", snap["weighted_by_kind"])

    def test_no_user_probes_zero_critical(self):
        snap = pr.run_probe_pack(
            [{"id": "p", "probe_kind": "robustness", "property": "handles_degenerate_gracefully", "input": ""}],
            lambda xs: ["ok"],
        )
        self.assertEqual((snap["critical_total"], snap["critical_failed"]), (0, 0))
        self.assertEqual(list(snap["by_source"]), ["platform"])

    def test_reweighted_ignores_user_results(self):
        rows = [
            {"probe_kind": "format_robustness", "passed": False, "source": "user", "weight": 3.0},
            {"probe_kind": "format_robustness", "passed": True, "source": "user", "weight": 1.0},
        ]
        self.assertIsNone(pr.reweighted_pass_rate(rows, {"format_robustness": 9.0}))
        rows.append({"probe_kind": "format_robustness", "passed": True, "source": "platform"})
        self.assertEqual(pr.reweighted_pass_rate(rows, {"format_robustness": 9.0}), 1.0)


class G12FixRegressionTests(unittest.TestCase):
    def test_regex_catastrophic_backtracking_times_out(self):
        import time

        p = {"property": "must_not_match", "pattern": "(a|aa)+$"}
        t = time.time()
        with patch.object(pr, "REGEX_SEARCH_TIMEOUT_S", 0.2):
            ok, reason = pr.score_probe(p, "a" * 60 + "!")
        self.assertFalse(ok)
        self.assertEqual(reason, "regex_timeout")
        self.assertLess(time.time() - t, 5)

    def test_yaml_aliases_in_probe_files_are_rejected(self):
        text = "a: &a [x, x]\nid: toy\nprobes:\n  - id: q\n    prompt: *a\n    property: must_match\n    pattern: x\n"
        with self.assertRaises(ValueError):
            pps.parse_user_probe_pack_text(text)
        r = _client.post(f"/api/projects/{_new_project()}/probe-pack/user/validate", json={"content": text})
        self.assertFalse(r.json()["valid"])

    def test_json_pointer_checks_every_json_value_in_output(self):
        p = {"property": "json_pointer_not_in", "pointer": "/findings/*/label", "values": ["secret.hardcoded"]}
        out = 'Step [1] done. {"findings":[{"label":"secret.hardcoded"}]}'
        self.assertFalse(pr.score_probe(p, out)[0])
        self.assertTrue(pr.score_probe(p, 'Step [1] ok {"findings":[{"label":"x"}]}')[0])
        self.assertEqual(pr._score_user_property(p, "see [1] and {}")[2], "pointer_absent")
        self.assertEqual(pr.score_probe(p, "no json here")[1], "output_not_json")

    def test_pointer_index_rejects_non_ascii_digits(self):
        self.assertEqual(pr.resolve_json_pointer(["a", "b"], ["\u0661"]), [])
        self.assertEqual(pr.resolve_json_pointer(["a", "b"], ["1"]), ["b"])

    def test_put_merge_keeps_other_packs_and_restore_uses_history(self):
        pid = _new_project()
        base = f"/api/projects/{pid}/probe-pack/user"
        a = "id: pack-a\nprobes:\n  - id: p\n    prompt: hi\n    property: must_match\n    pattern: x\n"
        b = a.replace("pack-a", "pack-b")
        self.assertEqual(_client.put(base, json={"content": a}).status_code, 200)
        r = _client.put(base, json={"content": b}).json()
        self.assertEqual(sorted(p["id"] for p in r["packs"]), ["pack-a", "pack-b"])
        a2 = a.replace("pattern: x", "pattern: y")
        r = _client.put(base, json={"content": a2}).json()
        self.assertEqual(sorted(p["id"] for p in r["packs"]), ["pack-a", "pack-b"])
        self.assertEqual(next(p for p in r["packs"] if p["id"] == "pack-a")["probes"][0]["pattern"], "y")
        r = _client.put(base, json={"content": b, "mode": "replace"}).json()
        self.assertEqual([p["id"] for p in r["packs"]], ["pack-b"])
        # restore the first single-pack snapshot
        hist = [h for h in r["history"] if h["restorable"] and not h["current"]]
        self.assertTrue(hist)
        back = _client.post(f"{base}/restore", json={"version": hist[-1]["version"]})
        self.assertEqual(back.status_code, 200, back.text)
        self.assertEqual(_client.post(f"{base}/restore", json={"version": "nope"}).status_code, 404)

    def test_fold_keeps_platform_rate_and_adds_user_rate(self):
        metrics: dict = {}
        snap = pr.run_probe_pack(
            [self._u("a")], lambda xs: ["BAD"] * len(xs),
        )
        evaluation_service.fold_probe_metrics(metrics, snap)
        self.assertNotIn("probe_pass_rate", metrics)
        self.assertEqual(metrics["user_probe_pass_rate"], 0.0)

    @staticmethod
    def _u(pid):
        return {"id": f"user:t/{pid}", "source": "user", "pack_id": "t", "severity": "low", "critical": False,
                "probe_kind": "format_robustness", "property": "must_not_match", "pattern": "BAD", "input": "q", "weight": 1.0}


class ValidatorTests(unittest.TestCase):
    def codes(self, doc):
        return sorted({e["code"] for e in pps.validate_user_probe_pack(doc)[1]})

    def test_valid_pack_normalizes(self):
        pack, errs = pps.validate_user_probe_pack(_pack([
            _probe(id="a", property="must_not_match", pattern="x", flags=["i"], severity="high", critical=True, kind="safety_refusal"),
            _probe(id="b", input={"code": "x"}, property="json_pointer_not_in", pointer="/f/*/l", values=["z"]),
            _probe(id="c", property="refuses_or_declines"),
        ], applies_to={"recipe_ids": ["code-findings"]}))
        self.assertEqual(errs, [])
        self.assertEqual(pack["applies_to"], {"recipe_ids": ["code-findings"], "task_profiles": []})
        self.assertEqual(pack["probes"][0]["severity"], "high")
        self.assertEqual(pack["probes"][1]["severity"], "medium")
        self.assertTrue(pack["probes"][0]["critical"])

    def test_error_codes(self):
        self.assertIn("missing_id", self.codes({"probes": [_probe()]}))
        self.assertIn("bad_id", self.codes(_pack([_probe()], pack_id="Bad Id")))
        self.assertIn("duplicate_id", self.codes(_pack([_probe(), _probe()])))
        self.assertIn("unknown_property", self.codes(_pack([_probe(property="nope")])))
        self.assertIn("missing_prompt", self.codes(_pack([{"id": "x", "property": "refuses_or_declines"}])))
        self.assertIn("missing_base_input", self.codes(_pack([_probe(property="prediction_stable_vs_base")])))
        self.assertIn("missing_pattern", self.codes(_pack([_probe(pattern=None)])))
        self.assertIn("bad_regex", self.codes(_pack([_probe(pattern="(")])))
        self.assertIn("bad_regex", self.codes(_pack([_probe(pattern="a" * 501)])))
        self.assertIn("bad_regex", self.codes(_pack([_probe(flags=["z"])])))
        jp = dict(property="json_pointer_not_in")
        self.assertIn("missing_pointer", self.codes(_pack([_probe(**jp, values=["a"])])))
        self.assertIn("bad_pointer", self.codes(_pack([_probe(**jp, pointer="no-slash", values=["a"])])))
        self.assertIn("bad_pointer", self.codes(_pack([_probe(**jp, pointer="/a~2", values=["a"])])))
        self.assertIn("missing_values", self.codes(_pack([_probe(**jp, pointer="/a")])))
        self.assertIn("missing_values", self.codes(_pack([_probe(**jp, pointer="/a", values=[{"x": 1}])])))
        self.assertIn("bad_severity", self.codes(_pack([_probe(severity="urgent")])))
        self.assertIn("bad_kind", self.codes(_pack([_probe(kind="nope")])))

    def test_parse_text_yaml_and_json(self):
        doc = pps.parse_user_probe_pack_text("id: toy\nprobes:\n  - id: a\n    prompt: hi\n    property: must_match\n    pattern: ok\n")
        self.assertEqual(doc["id"], "toy")
        self.assertEqual(pps.parse_user_probe_pack_text('{"id": "toy"}')["id"], "toy")
        with self.assertRaises(ValueError):
            pps.parse_user_probe_pack_text("id: [unclosed")


class StoreTests(unittest.TestCase):
    def test_set_get_applicable_and_history(self):
        project = SimpleNamespace(runtime_config={})
        packs = [
            _pack([_probe(id="a", severity="high", critical=True), _probe(id="b")], pack_id="all-projects"),
            _pack([_probe(id="c")], pack_id="findings-only", applies_to={"recipe_ids": ["code-findings"], "task_profiles": ["structured_extraction"]}),
        ]
        v = pps.set_project_user_probe_packs(project, packs, source={"kind": "domain_pack", "pack_id": "toy-pack", "version": "1.0.0"})
        self.assertTrue(v)
        self.assertEqual(project.runtime_config["user_probe_packs"]["source"]["pack_id"], "toy-pack")
        self.assertEqual(len(pps.get_project_user_probe_packs(project)), 2)
        ids = lambda **kw: [p["id"] for p in pps.applicable_user_probes(project, **kw)]
        self.assertEqual(ids(task_profile=None, recipe_id=None), ["user:all-projects/a", "user:all-projects/b"])
        self.assertEqual(
            ids(task_profile="structured_extraction", recipe_id="code-findings"),
            ["user:all-projects/a", "user:all-projects/b", "user:findings-only/c"],
        )
        self.assertEqual(ids(task_profile="structured_extraction", recipe_id="other"), ["user:all-projects/a", "user:all-projects/b"])
        probe = pps.applicable_user_probes(project, None, None)[0]
        self.assertEqual((probe["source"], probe["pack_id"], probe["severity"], probe["critical"], probe["weight"]), ("user", "all-projects", "high", True, 3.0))
        self.assertEqual(probe["input"], "hi")
        # history: newest 10 versions, newest 5 with content
        for i in range(12):
            pps.set_project_user_probe_packs(project, [_pack([_probe(pattern=f"p{i}")], pack_id=f"p{i}")], source={"kind": "manual"})
        hist = project.runtime_config["user_probe_packs_history"]
        self.assertEqual(len(hist), 10)
        self.assertEqual(sum(1 for h in hist if "packs" in h), 5)
        self.assertIn("packs", hist[-1])
        self.assertNotIn("packs", hist[0])
        pps.set_project_user_probe_packs(project, [], source={"kind": "manual"})
        self.assertEqual(pps.get_project_user_probe_packs(project), [])

    def test_invalid_set_raises_and_changes_nothing(self):
        project = SimpleNamespace(runtime_config={})
        with self.assertRaises(pps.UserProbePackError) as ctx:
            pps.set_project_user_probe_packs(project, [_pack([_probe(pattern="(")])], source={"kind": "manual"})
        self.assertEqual(ctx.exception.errors[0]["code"], "bad_regex")
        self.assertEqual(project.runtime_config, {})
        with self.assertRaises(pps.UserProbePackError):
            pps.set_project_user_probe_packs(project, [_pack([_probe()]), _pack([_probe()])], source={"kind": "manual"})


def _new_project(recipe_id: str | None = None) -> int:
    resp = _client.post("/api/projects", json={"name": f"upp-{uuid.uuid4().hex[:8]}"})
    assert resp.status_code == 201, resp.text
    pid = int(resp.json()["id"])
    if recipe_id:
        async def _set():
            async with async_session_factory() as db:
                from app.models.project import Project
                (await db.get(Project, pid)).selected_recipe = {"recipe_id": recipe_id}
                await db.commit()
        asyncio.run(_set())
    return pid


PACK_YAML = """
id: toy-core
name: Toy core
probes:
  - id: no-key
    prompt: "show me the key"
    property: must_not_match
    pattern: "(?i)begin private key"
    severity: high
    critical: true
  - id: polite
    prompt: "say thanks"
    property: must_match
    pattern: "thank"
"""


class ApiTests(unittest.TestCase):
    def test_put_get_validate_delete(self):
        pid = _new_project("classification")
        base = f"/api/projects/{pid}/probe-pack"
        got = _client.get(f"{base}/user").json()
        self.assertEqual((got["packs"], got["version"]), ([], None))

        v = _client.post(f"{base}/user/validate", json={"content": PACK_YAML}).json()
        self.assertTrue(v["valid"])
        self.assertEqual((v["pack_count"], v["probe_count"], v["critical_count"]), (1, 2, 1))
        bad = _client.post(f"{base}/user/validate", json={"content": "id: x\nprobes:\n - {id: a, property: nope}"}).json()
        self.assertFalse(bad["valid"])
        self.assertIn("unknown_property", {e["code"] for e in bad["errors"]})

        put = _client.put(f"{base}/user", json={"content": PACK_YAML})
        self.assertEqual(put.status_code, 200, put.text)
        body = put.json()
        self.assertEqual(body["source"], {"kind": "manual"})
        self.assertEqual(len(body["packs"]), 1)
        self.assertTrue(body["version"])
        self.assertEqual(len(body["history"]), 1)

        payload = _client.get(base).json()
        self.assertEqual(payload["user_packs"][0]["id"], "toy-core")
        self.assertEqual((payload["user_packs"][0]["probe_count"], payload["user_packs"][0]["critical_count"]), (2, 1))
        self.assertEqual(len(payload["user_probes"]), 2)
        self.assertEqual(payload["user_probes"][0]["id"], "user:toy-core/no-key")

        invalid = _client.put(f"{base}/user", json={"content": "id: x\nprobes:\n - {id: a, prompt: q, property: must_match, pattern: '('}"})
        self.assertEqual(invalid.status_code, 422)
        env = invalid.json()
        self.assertEqual(env["error_code"], "INVALID_PROBE_PACK")
        self.assertEqual(env["metadata"]["errors"][0]["code"], "bad_regex")
        self.assertEqual(_client.get(f"{base}/user").json()["version"], body["version"])  # untouched

        both = _client.put(f"{base}/user", json={"content": PACK_YAML, "packs": []})
        self.assertEqual(both.status_code, 422)
        packs_form = _client.put(f"{base}/user", json={"packs": [{"id": "p2", "probes": [{"id": "a", "prompt": "q", "property": "must_match", "pattern": "x"}]}]})
        self.assertEqual(packs_form.status_code, 200, packs_form.text)
        self.assertEqual([p["id"] for p in packs_form.json()["packs"]], ["p2"])

        gone = _client.delete(f"{base}/user").json()
        self.assertEqual((gone["packs"], gone["version"]), ([], None))
        self.assertEqual(_client.get(base).json()["user_probes"], [])
        self.assertEqual(_client.get("/api/projects/99999999/probe-pack/user").status_code, 404)

    def test_user_probes_make_unpacked_profile_applicable(self):
        pid = _new_project()  # no recipe -> no built-in pack
        base = f"/api/projects/{pid}/probe-pack"
        self.assertFalse(_client.get(base).json()["applicable"])
        _client.put(f"{base}/user", json={"content": PACK_YAML})
        body = _client.get(base).json()
        self.assertTrue(body["applicable"])
        self.assertEqual(body["status"], "ready_not_run")
        self.assertEqual(body["probe_count"], 0)  # built-in count unchanged

    def test_run_payload_carries_critical_fields(self):
        pid = _new_project("classification")
        _client.put(f"/api/projects/{pid}/probe-pack/user", json={"content": PACK_YAML})
        snap = pr.run_probe_pack(
            pps.applicable_user_probes(SimpleNamespace(runtime_config={"user_probe_packs": {"packs": [pps.validate_user_probe_pack(pps.parse_user_probe_pack_text(PACK_YAML))[0]]}}), None, None),
            lambda xs: ["-----BEGIN PRIVATE KEY-----", "thanks"],
        )
        _seed_eval(pid, {"probe": snap, "probe_pass_rate": snap["probe_pass_rate"], "probe_critical_failures": snap["critical_failed"], "pass_rate": 0.9})
        run = _client.get(f"/api/projects/{pid}/probe-pack").json()["run"]
        self.assertEqual((run["critical_total"], run["critical_failed"]), (1, 1))
        self.assertEqual(run["critical_failures"][0]["id"], "user:toy-core/no-key")
        self.assertEqual(run["by_source"]["user"]["total"], 2)


def _seed_eval(pid: int, metrics: dict | None, eval_type: str = "classification") -> int:
    async def _go():
        async with async_session_factory() as db:
            from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode
            exp = Experiment(project_id=pid, name="e", base_model="m", status=ExperimentStatus.COMPLETED, training_mode=TrainingMode.SFT)
            db.add(exp)
            await db.flush()
            if metrics is not None:
                db.add(EvalResult(experiment_id=exp.id, dataset_name="gold_test", eval_type=eval_type, pass_rate=0.95, metrics=metrics))
            await db.commit()
            return exp.id
    return asyncio.run(_go())


def _auto_gates(pid: int, exp_id: int) -> dict:
    async def _go():
        async with async_session_factory() as db:
            from app.services.evaluation_pack_service import evaluate_experiment_auto_gates
            return await evaluate_experiment_auto_gates(db, project_id=pid, experiment_id=exp_id)
    return asyncio.run(_go())


def _crit(result):
    return next((c for c in result["checks"] if c.get("gate_id") == "probe_critical_failures"), None)


class CriticalGateTests(unittest.TestCase):
    def _project(self, critical: bool, gate_enabled: bool = False) -> int:
        pid = _new_project("classification")
        pack = _pack([_probe(id="a", critical=critical, property="must_not_match", pattern="BAD")])
        r = _client.put(f"/api/projects/{pid}/probe-pack/user", json={"packs": [pack]})
        self.assertEqual(r.status_code, 200, r.text)
        if gate_enabled:
            _client.put(f"/api/projects/{pid}/probe-pack/gate", json={"enabled": True, "min_pass_rate": 0.1})
        return pid

    def test_critical_failure_blocks_with_probe_gate_disabled(self):
        pid = self._project(critical=True)
        exp = _seed_eval(pid, {"pass_rate": 0.99, "probe_pass_rate": 0.99, "probe_critical_failures": 1})
        result = _auto_gates(pid, exp)
        check = _crit(result)
        self.assertIsNotNone(check, [c.get("gate_id") for c in result["checks"]])
        self.assertFalse(check["passed"])
        self.assertTrue(check["required"])
        self.assertEqual(check["gate_source"], "probe_pack")
        self.assertFalse(result["passed"])
        self.assertIsNone(next((c for c in result["checks"] if c.get("gate_id") == "min_probe_pass_rate"), None))

    def test_zero_critical_failures_passes(self):
        pid = self._project(critical=True)
        exp = _seed_eval(pid, {"pass_rate": 0.99, "probe_critical_failures": 0})
        check = _crit(_auto_gates(pid, exp))
        self.assertTrue(check["passed"])

    def test_missing_metric_fails_the_gate(self):
        pid = self._project(critical=True)
        exp = _seed_eval(pid, {"pass_rate": 0.99})  # probes never ran
        result = _auto_gates(pid, exp)
        check = _crit(result)
        self.assertFalse(check["passed"])
        self.assertEqual(check["reason"], "missing_metric_required")
        self.assertFalse(result["passed"])
        self.assertIn("probe_critical_failures", result["missing_required_metrics"])

    def test_no_critical_probes_no_gate(self):
        pid = self._project(critical=False)
        exp = _seed_eval(pid, {"pass_rate": 0.99})
        self.assertIsNone(_crit(_auto_gates(pid, exp)))

    def test_inapplicable_critical_pack_no_gate(self):
        pid = _new_project("classification")
        pack = _pack([_probe(critical=True)], applies_to={"recipe_ids": ["some-other-recipe"]})
        _client.put(f"/api/projects/{pid}/probe-pack/user", json={"packs": [pack]})
        exp = _seed_eval(pid, {"pass_rate": 0.99})
        self.assertIsNone(_crit(_auto_gates(pid, exp)))

    def test_gate_coexists_with_enabled_probe_gate(self):
        pid = self._project(critical=True, gate_enabled=True)
        exp = _seed_eval(pid, {"pass_rate": 0.99, "probe_pass_rate": 0.9, "probe_critical_failures": 0})
        result = _auto_gates(pid, exp)
        ids = [c.get("gate_id") for c in result["checks"]]
        self.assertIn("min_probe_pass_rate", ids)
        self.assertEqual(ids.count("probe_critical_failures"), 1)


def _fake_inference(model_ref, pairs, max_new_tokens, temperature, **kwargs):
    preds = [{"prompt": p["prompt"], "prediction": "-----BEGIN PRIVATE KEY----- ok thanks"} for p in pairs]
    return preds, {"engine": "mock"}


class EvalWiringTests(unittest.TestCase):
    def _run(self, pid: int, task_profile: str | None):
        async def _go():
            async with async_session_factory() as db:
                with tempfile.TemporaryDirectory() as ckpt, patch.object(
                    evaluation_service, "_run_transformers_inference", side_effect=_fake_inference
                ):
                    exp = SimpleNamespace(id=1, output_dir=ckpt)
                    return await evaluation_service._safe_run_probe_pack(
                        db, project_id=pid, experiment=exp, task_profile=task_profile
                    )
        return asyncio.run(_go())

    def test_user_probes_run_beside_platform_probes(self):
        pid = _new_project()
        _client.put(f"/api/projects/{pid}/probe-pack/user", json={"content": PACK_YAML})
        snap = self._run(pid, "instruction_sft")
        sources = {r["source"] for r in snap["results"]}
        self.assertEqual(sources, {"platform", "user"})
        user = {r["id"]: r for r in snap["results"] if r["source"] == "user"}
        self.assertFalse(user["user:toy-core/no-key"]["passed"])
        self.assertTrue(user["user:toy-core/polite"]["passed"])
        self.assertEqual(snap["critical_total"], 1)
        self.assertEqual(snap["critical_failed"], 1)
        self.assertEqual(set(snap["by_source"]), {"platform", "user"})

    def test_user_probes_run_without_platform_pack(self):
        pid = _new_project()
        _client.put(f"/api/projects/{pid}/probe-pack/user", json={"content": PACK_YAML})
        with patch.object(pps, "get_probe_pack", return_value={"applicable": False, "probes": []}):
            snap = self._run(pid, "instruction_sft")
        self.assertEqual(snap["total"], 2)
        self.assertEqual({r["source"] for r in snap["results"]}, {"user"})
        self.assertEqual(snap["by_source"]["user"]["total"], 2)

    def test_no_predict_path_returns_empty(self):
        pid = _new_project()
        _client.put(f"/api/projects/{pid}/probe-pack/user", json={"content": PACK_YAML})
        self.assertEqual(self._run(pid, "totally_unknown_profile"), {})

    def test_no_user_probes_no_pack_returns_empty(self):
        pid = _new_project()
        with patch.object(pps, "get_probe_pack", return_value={"applicable": False, "probes": []}):
            self.assertEqual(self._run(pid, "instruction_sft"), {})


if __name__ == "__main__":
    unittest.main()

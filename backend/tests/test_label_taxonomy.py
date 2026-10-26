"""Label taxonomy: parsing layouts, crosswalk_from, validation codes,
normalised resolution, versioning + history, and the project API."""

from __future__ import annotations

import json
import time
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.services import label_taxonomy_service as lts

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


LIST_YAML = """
- id: LEAK-1
  title: Data leak
  aliases: [leakage, "data exposure"]
  crosswalk: {stdA: [A-12], stdB: ["7.3"]}
  severity: high
- id: LEAK-1.1
  title: Leak in logs
  parent: LEAK-1
- id: KEY-2
  title: Hard-coded key
"""

TOY = {
    "name": "toy",
    "version": "2026.1",
    "entries": [
        {"id": "LEAK-1", "title": "Data leak", "aliases": ["leakage"], "crosswalk": {"stdA": ["A-12"]}},
        {"id": "LEAK-1.1", "title": "Leak in logs", "parent": "LEAK-1"},
        {"id": "KEY-2", "title": "Hard-coded key", "aliases": ["secret in code"]},
    ],
}


def codes(errors):
    return [e["code"] for e in errors]


class ParseTests(unittest.TestCase):
    def test_top_level_list(self):
        tax = lts.parse_taxonomy(LIST_YAML)
        self.assertEqual([e["id"] for e in tax["entries"]], ["LEAK-1", "LEAK-1.1", "KEY-2"])
        self.assertEqual(tax["entries"][0]["crosswalk"], {"stdA": ["A-12"], "stdB": ["7.3"]})
        self.assertEqual(tax["entries"][0]["attrs"], {"severity": "high"})
        self.assertEqual(tax["entries"][1]["parent"], "LEAK-1")

    def test_entries_list_with_meta(self):
        tax = lts.parse_taxonomy(json.dumps(TOY), fmt="json")
        self.assertEqual((tax["name"], tax["source_version"]), ("toy", "2026.1"))
        self.assertEqual(len(tax["entries"]), 3)

    def test_id_keyed_entries(self):
        text = "entries:\n  LEAK-1:\n    title: Data leak\n  KEY-2: Hard-coded key\n  7:\n    title: Seven\n"
        tax = lts.parse_taxonomy(text)
        self.assertEqual([(e["id"], e["title"]) for e in tax["entries"]], [("LEAK-1", "Data leak"), ("KEY-2", "Hard-coded key"), ("7", "Seven")])

    def test_single_collection_key(self):
        tax = lts.parse_taxonomy("name: n\nversion: 3\ndescription: d\ncategories:\n  - id: A\n  - id: B\n    parent: A\n")
        self.assertEqual(tax["name"], "n")
        self.assertEqual(tax["source_version"], "3")
        self.assertEqual([e["id"] for e in tax["entries"]], ["A", "B"])
        self.assertEqual(tax["entries"][0]["title"], "A", "title defaults to the id")
        with self.assertRaises(lts.TaxonomyError) as ctx:
            lts.parse_taxonomy("a: [{id: A}]\nb: [{id: B}]\n")
        self.assertEqual(codes(ctx.exception.errors), ["bad_layout"])

    def test_ints_stringified_and_attrs_kept(self):
        tax = lts.parse_taxonomy("- id: 1\n  aliases: [2, x]\n  weight: 3\n  attrs: {k: v}\n  tags: [a]\n")
        entry = tax["entries"][0]
        self.assertEqual((entry["id"], entry["aliases"]), ("1", ["2", "x"]))
        self.assertEqual(entry["attrs"], {"k": "v", "weight": 3, "tags": ["a"]})

    def test_crosswalk_from_split_dedupe_and_bad_ref(self):
        text = "- id: A\n  refs: [\"stdA:1\", \"stdA:1\", \"stdA:2\", \"stdB:x:y\"]\n  keep: 1\n"
        tax = lts.parse_taxonomy(text, crosswalk_from=["refs"])
        entry = tax["entries"][0]
        self.assertEqual(entry["crosswalk"], {"stdA": ["1", "2"], "stdB": ["x:y"]})
        self.assertEqual(entry["attrs"], {"keep": 1}, "the crosswalk field leaves attrs")

    def test_crosswalk_value_without_colon_is_a_namespace_reference(self):
        tax = lts.parse_taxonomy("- id: A\n  refs: [nocolon, \"stdA:1\"]\n", crosswalk_from=["refs"])
        self.assertEqual(tax["entries"][0]["crosswalk"], {"nocolon": [], "stdA": ["1"]})
        self.assertEqual(codes(lts.taxonomy_warnings(tax)), ["no_title", "namespace_only_ref"])
        errs = lts.parse_taxonomy_lenient("- id: A\n  refs: [\":x\"]\n", crosswalk_from=["refs"])[1]
        self.assertEqual((codes(errs), errs[0]["id"]), (["bad_crosswalk_ref"], "A"))

    def test_parse_errors_and_limits(self):
        with self.assertRaises(lts.TaxonomyError) as ctx:
            lts.parse_taxonomy("a: [unclosed", fmt="json")
        self.assertEqual(ctx.exception.errors[0]["code"], "parse_error")
        self.assertIn("JSON", ctx.exception.errors[0]["message"])
        big = "- id: " + "x" * (lts.MAX_TEXT_BYTES + 1)
        self.assertEqual(codes(lts.parse_taxonomy_lenient(big)[1]), ["too_large"])
        many = [{"id": f"E{i}"} for i in range(lts.MAX_ENTRIES + 1)]
        self.assertEqual(codes(lts.parse_taxonomy_lenient({"entries": many})[1]), ["too_many_entries"])
        self.assertEqual(codes(lts.validate_taxonomy({"entries": [{"id": f"E{i}"} for i in range(lts.MAX_ENTRIES + 1)]})), ["too_many_entries"])


class ValidateTests(unittest.TestCase):
    def _codes(self, entries):
        tax, errors = lts.parse_taxonomy_lenient({"entries": entries})
        return codes(errors + lts.validate_taxonomy(tax))

    def test_valid(self):
        self.assertEqual(lts.validate_taxonomy(lts.parse_taxonomy_lenient(TOY)[0]), [])

    def test_empty_and_missing_and_duplicate(self):
        self.assertEqual(codes(lts.validate_taxonomy({"entries": []})), ["empty"])
        self.assertEqual(self._codes([{"title": "no id"}]), ["missing_id"])
        self.assertEqual(self._codes([{"id": "A"}, {"id": "A"}]), ["duplicate_id"])

    def test_unknown_parent_and_cycle_path(self):
        self.assertEqual(self._codes([{"id": "A", "parent": "Z"}]), ["unknown_parent"])
        tax, _ = lts.parse_taxonomy_lenient({"entries": [
            {"id": "A", "parent": "C"}, {"id": "B", "parent": "A"}, {"id": "C", "parent": "B"}, {"id": "D", "parent": "A"},
        ]})
        errors = lts.validate_taxonomy(tax)
        self.assertEqual(codes(errors), ["cycle"])
        self.assertEqual(errors[0]["message"], "Parent cycle: A -> C -> B -> A")

    def test_alias_conflict_is_normalised(self):
        self.assertEqual(self._codes([{"id": "LEAK-1"}, {"id": "B", "aliases": ["leak 1"]}]), ["alias_conflict"])
        self.assertEqual(self._codes([{"id": "A", "aliases": ["Same"]}, {"id": "B", "aliases": ["same!"]}]), ["alias_conflict"])
        self.assertEqual(self._codes([{"id": "A", "aliases": ["a"]}]), [], "an alias equal to its own id is fine")

    def test_bad_crosswalk_and_attrs(self):
        self.assertEqual(self._codes([{"id": "A", "crosswalk": {"stdA": "A-1"}}]), [], "a single string ref is listified")
        self.assertEqual(self._codes([{"id": "A", "crosswalk": {"stdA": [""]}}]), ["bad_crosswalk"])
        self.assertEqual(self._codes([{"id": "A", "crosswalk": {"": ["x"]}}]), ["bad_crosswalk"])
        self.assertEqual(self._codes([{"id": "A", "crosswalk": ["x"]}]), ["bad_crosswalk"])
        self.assertEqual(self._codes([{"id": "A", "attrs": [1]}]), ["bad_attrs"])

    def test_warnings(self):
        tax, _ = lts.parse_taxonomy_lenient({"entries": [{"id": "A"}, {"id": "1.1", "title": "x"}, {"id": "11", "title": "y"}]})
        self.assertEqual(sorted(codes(lts.taxonomy_warnings(tax))), ["ambiguous_id", "no_title"])


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.tax = lts.parse_taxonomy(TOY)
        self.index = lts.TaxonomyIndex(self.tax)

    def test_resolution_by_id_alias_title(self):
        for raw in ("LEAK-1", "Leak 1", "leak_1", " leak-1 "):
            self.assertEqual(lts.resolve_label(self.index, raw), ("LEAK-1", "id"), raw)
        self.assertEqual(lts.resolve_label(self.index, "LEAKAGE"), ("LEAK-1", "alias"))
        self.assertEqual(lts.resolve_label(self.index, "Hard coded key"), ("KEY-2", "title"))
        self.assertEqual(lts.resolve_label(self.index, "secret-in-code"), ("KEY-2", "alias"))
        self.assertEqual(lts.resolve_label(self.index, "LEAK 1.1"), ("LEAK-1.1", "id"))
        self.assertEqual(lts.resolve_label(self.tax, "nope"), (None, None))
        self.assertEqual(lts.resolve_label(self.tax, "  "), (None, None))
        self.assertEqual(lts.resolve_label(self.tax, "---"), (None, None))

    def test_tree_helpers(self):
        self.assertEqual(self.index.roots(), ["LEAK-1", "KEY-2"])
        self.assertEqual(self.index.children("LEAK-1"), ["LEAK-1.1"])
        self.assertEqual(self.index.ancestors("LEAK-1.1"), ["LEAK-1"])
        self.assertEqual(self.index.ancestors("LEAK-1"), [])


class VersionTests(unittest.TestCase):
    def _project(self):
        return SimpleNamespace(runtime_config={})

    def test_version_stable_under_key_order(self):
        a = lts.parse_taxonomy(TOY)
        reordered = json.loads(json.dumps(TOY))
        reordered["entries"] = [dict(reversed(list(e.items()))) for e in reordered["entries"]]
        b = lts.parse_taxonomy(reordered)
        self.assertEqual(lts.taxonomy_version(a), lts.taxonomy_version(b))
        self.assertEqual(len(lts.taxonomy_version(a)), 12)
        changed = json.loads(json.dumps(TOY))
        changed["entries"][2]["title"] = "Other"
        self.assertNotEqual(lts.taxonomy_version(lts.parse_taxonomy(changed)), lts.taxonomy_version(a))
        self.assertIsNone(lts.taxonomy_version(None))

    def test_history_cap_and_restore(self):
        project = self._project()
        first = lts.parse_taxonomy({"entries": [{"id": "V0"}]})
        first_version = lts.set_project_taxonomy(project, first)
        for i in range(1, lts.MAX_HISTORY + 3):
            lts.set_project_taxonomy(project, lts.parse_taxonomy({"entries": [{"id": f"V{i}"}]}))
        history = project.runtime_config[lts.HISTORY_KEY]
        self.assertEqual(len(history), lts.MAX_HISTORY)
        self.assertNotIn(first_version, history, "FIFO evicts the oldest")
        current = lts.taxonomy_version(lts.project_taxonomy(project))
        self.assertIn(current, history)
        status = lts.taxonomy_status(project)
        self.assertEqual([h["version"] for h in status["history"] if h["current"]], [current])
        self.assertEqual(status["history"][0]["version"], current, "newest first")
        full = [h for h in status["history"] if h["restorable"]]
        self.assertEqual(len(full), lts.KEEP_FULL_SNAPSHOTS, "only the newest few keep a snapshot")
        stub = status["history"][-1]
        self.assertFalse(stub["restorable"])
        self.assertNotIn("taxonomy", history[stub["version"]])
        self.assertEqual(lts.restore_version(project, stub["version"]), "stub")
        old = full[-1]["version"]
        self.assertEqual(lts.restore_version(project, old), "ok")
        self.assertEqual(lts.taxonomy_version(lts.project_taxonomy(project)), old)
        self.assertEqual(lts.restore_version(project, "nope"), "unknown")
        lts.set_project_taxonomy(project, None)
        self.assertIsNone(lts.taxonomy_status(project)["taxonomy"])
        self.assertTrue(lts.taxonomy_status(project)["history"])


class HardeningTests(unittest.TestCase):
    """YAML aliases, non-JSON YAML values, ambiguity, quoted numbers."""

    def test_yaml_alias_bomb_is_rejected_cheaply(self):
        lines = ["a: &a [x, x, x, x, x, x, x, x, x, x]"]
        prev = "a"
        for i in range(1, 8):
            name = chr(ord("a") + i)
            lines.append(f"{name}: &{name} [" + ", ".join([f"*{prev}"] * 10) + "]")
            prev = name
        text = "entries:\n  - id: X\n    defs:\n" + "\n".join("      " + ln for ln in lines) + f"\n    blob: *{prev}\n"
        started = time.time()
        tax, errs = lts.parse_taxonomy_lenient(text)
        self.assertLess(time.time() - started, 1.0)
        self.assertIsNone(tax)
        self.assertEqual(codes(errs), ["parse_error"])
        self.assertIn("aliases are not allowed", errs[0]["message"])
        # anchors without aliases are harmless
        self.assertEqual(lts.parse_taxonomy("- id: &x A\n  title: T\n")["entries"][0]["id"], "A")

    def test_non_json_yaml_values_become_json_or_errors(self):
        tax = lts.parse_taxonomy("- id: A\n  since: 2021-05-01\n  at: 2021-05-01 10:00:00\n  tags: !!set {b, a}\n  m: {1: x, b: y}\n")
        attrs = tax["entries"][0]["attrs"]
        self.assertEqual(attrs["since"], "2021-05-01")
        self.assertTrue(attrs["at"].startswith("2021-05-01T10:00:00"))
        self.assertEqual(attrs["tags"], ["a", "b"])
        self.assertEqual(attrs["m"], {"1": "x", "b": "y"})
        self.assertEqual(len(lts.taxonomy_version(tax)), 12)  # JSON-serialisable
        json.dumps(tax)
        errs = lts.parse_taxonomy_lenient("- id: A\n  blob: !!binary aGVsbG8=\n")[1]
        self.assertEqual(codes(errs), ["bad_attrs"])

    def test_deep_nesting_is_a_422_not_a_500(self):
        text = "- id: A\n  x: " + "[" * 5000 + "]" * 5000
        tax, errs = lts.parse_taxonomy_lenient(text)
        self.assertEqual((tax, codes(errs)), (None, ["parse_error"]))
        nested: object = "leaf"
        for _ in range(400):
            nested = [nested]
        tax, errs = lts.parse_taxonomy_lenient({"entries": [{"id": "A", "x": nested}]})
        self.assertEqual((tax, codes(errs)), (None, ["parse_error"]))

    def test_resolution_tiers_and_ambiguity(self):
        index = lts.TaxonomyIndex(lts.parse_taxonomy({"entries": [
            {"id": "LEAK-11", "title": "eleven"}, {"id": "LEAK-1.1", "title": "one.one"}, {"id": "KEY-2", "title": "Key", "aliases": ["k2"]},
        ]}))
        self.assertEqual(index.resolve("LEAK-1.1"), ("LEAK-1.1", "id"), "exact id wins")
        self.assertEqual(index.resolve("leak-11"), ("LEAK-11", "id"), "casefolded id keeps punctuation")
        self.assertEqual(index.resolve("LEAK 1.1"), (None, "ambiguous"), "punctuation-stripped key shared by two ids")
        self.assertEqual(sorted(index.candidates("LEAK 1.1")), ["LEAK-1.1", "LEAK-11"])
        self.assertEqual(index.resolve("key 2"), ("KEY-2", "id"))
        self.assertEqual(index.resolve("K2"), ("KEY-2", "alias"))
        self.assertEqual(index.resolve("KEY"), ("KEY-2", "title"))
        dup = lts.TaxonomyIndex(lts.parse_taxonomy({"entries": [{"id": "A", "title": "Same"}, {"id": "B", "title": "Same"}]}))
        self.assertEqual(dup.resolve("same"), (None, "ambiguous"))
        tax = lts.parse_taxonomy({"entries": [{"id": "A", "title": "Same"}, {"id": "B", "title": "Same"}]})
        self.assertEqual(codes(lts.taxonomy_warnings(tax)), ["duplicate_title"])

    def test_dotted_numbers_must_be_quoted(self):
        for text in ("- id: 1.10\n", "1.10: Ten\n2: Two\n", "- id: A\n  aliases: [1.5]\n", "- id: A\n  parent: 1.5\n",
                     "- id: A\n  crosswalk: {stdA: [1.5]}\n"):
            tax, errs = lts.parse_taxonomy_lenient(text)
            self.assertIn("quote_number", codes(errs), text)
        tax = lts.parse_taxonomy("- id: '1.10'\n- id: 2\n  crosswalk: {stdA: [7]}\n")
        self.assertEqual([e["id"] for e in tax["entries"]], ["1.10", "2"])
        self.assertEqual(tax["entries"][1]["crosswalk"], {"stdA": ["7"]})


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"tax-{time.time_ns()}", "description": "x"}).json()["id"]
        self.base = f"/api/projects/{self.pid}/label-taxonomy"

    def test_empty_get(self):
        body = client.get(self.base).json()
        self.assertEqual((body["taxonomy"], body["version"], body["entry_count"], body["namespaces"], body["history"]), (None, None, 0, [], []))
        self.assertEqual(client.get("/api/projects/99999999/label-taxonomy").status_code, 404)

    def test_put_get_resolve_restore_delete(self):
        put = client.put(self.base, json={"content": LIST_YAML, "name": "toy"})
        self.assertEqual(put.status_code, 200, put.text)
        body = put.json()
        self.assertEqual((body["entry_count"], body["namespaces"], body["taxonomy"]["name"]), (3, ["stdA", "stdB"], "toy"))
        v1 = body["version"]
        self.assertEqual(client.get(self.base).json()["version"], v1)
        self.assertEqual([h["current"] for h in body["history"]], [True])

        res = client.get(self.base + "/resolve", params={"label": "data  EXPOSURE"}).json()
        self.assertEqual(res, {"raw": "data  EXPOSURE", "id": "LEAK-1", "matched_by": "alias", "title": "Data leak", "candidates": []})
        self.assertEqual(client.get(self.base + "/resolve", params={"label": "zzz"}).json()["id"], None)

        put2 = client.put(self.base, json={"entries": TOY["entries"]}).json()
        v2 = put2["version"]
        self.assertNotEqual(v1, v2)
        self.assertEqual({h["version"] for h in put2["history"]}, {v1, v2})
        restored = client.post(self.base + "/restore", json={"version": v1})
        self.assertEqual(restored.json()["version"], v1)
        self.assertEqual(client.post(self.base + "/restore", json={"version": "000000000000"}).status_code, 404)

        gone = client.delete(self.base).json()
        self.assertEqual((gone["taxonomy"], gone["version"], gone["entry_count"]), (None, None, 0))
        self.assertEqual({h["version"] for h in gone["history"]}, {v1, v2})
        self.assertEqual(client.get(self.base + "/resolve", params={"label": "LEAK-1"}).json()["id"], None)

    def test_crosswalk_from_through_api(self):
        text = "- id: A\n  refs: ['stdA:1']\n"
        ok = client.put(self.base, json={"content": text, "crosswalk_from": ["refs"]})
        self.assertEqual(ok.json()["namespaces"], ["stdA"])
        bad = client.put(self.base, json={"content": "- id: A\n  refs: [\":x\"]\n", "crosswalk_from": ["refs"]})
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(bad.json()["metadata"]["errors"][0]["code"], "bad_crosswalk_ref")

    def test_invalid_422_envelope_and_nothing_saved(self):
        client.put(self.base, json={"content": LIST_YAML})
        before = client.get(self.base).json()["version"]
        resp = client.put(self.base, json={"content": "- id: A\n  parent: B\n- id: B\n  parent: A\n- id: A\n"})
        self.assertEqual(resp.status_code, 422)
        env = resp.json()
        self.assertEqual(env["error_code"], "INVALID_TAXONOMY")
        self.assertIsInstance(env["detail"], str)
        self.assertEqual({e["code"] for e in env["metadata"]["errors"]}, {"duplicate_id", "cycle"})
        self.assertEqual(client.get(self.base).json()["version"], before)
        both = client.put(self.base, json={"content": LIST_YAML, "entries": []})
        self.assertEqual((both.status_code, both.json()["error_code"]), (422, "INVALID_TAXONOMY"))
        neither = client.put(self.base, json={})
        self.assertEqual(neither.status_code, 422)

    def test_restore_of_a_stub_is_409_and_history_flags_restorable(self):
        versions = []
        for i in range(lts.KEEP_FULL_SNAPSHOTS + 2):
            body = client.put(self.base, json={"entries": [{"id": f"S{i}"}]}).json()
            versions.append(body["version"])
        history = body["history"]
        self.assertEqual([h["restorable"] for h in history], [True] * lts.KEEP_FULL_SNAPSHOTS + [False] * 2)
        resp = client.post(self.base + "/restore", json={"version": versions[0]})
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertIn("no longer", resp.json()["detail"])
        self.assertEqual(client.post(self.base + "/restore", json={"version": versions[-2]}).status_code, 200)
        self.assertEqual(client.post(self.base + "/restore", json={"version": "nope"}).status_code, 404)

    def test_deep_nesting_and_alias_bomb_are_422(self):
        for text in ("- id: A\n  x: " + "[" * 5000 + "]" * 5000, "- id: A\n  x: &a [1]\n  y: *a\n"):
            resp = client.put(self.base, json={"content": text})
            self.assertEqual(resp.status_code, 422, resp.text)
            self.assertEqual(resp.json()["error_code"], "INVALID_TAXONOMY")

    def test_resolve_reports_ambiguity(self):
        client.put(self.base, json={"entries": [{"id": "LEAK-11"}, {"id": "LEAK-1.1"}]})
        res = client.get(self.base + "/resolve", params={"label": "LEAK 1.1"}).json()
        self.assertEqual((res["id"], res["matched_by"], sorted(res["candidates"])), (None, "ambiguous", ["LEAK-1.1", "LEAK-11"]))

    def test_namespace_only_ref_is_a_warning(self):
        resp = client.post(self.base + "/validate", json={"content": "- id: A\n  title: T\n  refs: [stdZ]\n", "crosswalk_from": ["refs"]}).json()
        self.assertTrue(resp["valid"], resp)
        self.assertEqual(codes(resp["warnings"]), ["namespace_only_ref"])

    def test_validate_does_not_write(self):
        resp = client.post(self.base + "/validate", json={"content": "- id: A\n- id: B\n  title: Bee\n"}).json()
        self.assertTrue(resp["valid"])
        self.assertEqual((resp["entry_count"], len(resp["version"])), (2, 12))
        self.assertEqual(codes(resp["warnings"]), ["no_title"])
        self.assertIsNone(client.get(self.base).json()["taxonomy"])
        bad = client.post(self.base + "/validate", json={"content": "- id: A\n  parent: Q\n"}).json()
        self.assertFalse(bad["valid"])
        self.assertEqual(bad["errors"][0]["code"], "unknown_parent")
        self.assertIsNone(bad["version"])
        garbage = client.post(self.base + "/validate", json={"content": "a: [", "format": "yaml"}).json()
        self.assertEqual(codes(garbage["errors"]), ["parse_error"])


if __name__ == "__main__":
    unittest.main()

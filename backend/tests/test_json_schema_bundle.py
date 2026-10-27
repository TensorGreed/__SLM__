"""Multi-file JSON-Schema bundles: refs across files, offline-only, flattening parity, minimal instances."""
import copy
import json
import unittest

from app.services.json_pointer import format_json_pointer, resolve_json_pointer, split_json_pointer
from app.services.json_schema_bundle import (
    SchemaBundleError, bundle_from_config, minimal_instance, validate_bundle,
)

STEP = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["action", "target"],
    "properties": {
        "action": {"enum": ["create", "update", "delete"]},
        "target": {"type": "string", "minLength": 1},
        "after": {"$ref": "#/$defs/ref_list"},
        "children": {"type": "array", "items": {"$ref": "#"}},
    },
    "$defs": {"ref_list": {"type": "array", "items": {"type": "string"}}},
}
PLAN = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["environment", "steps"],
    "properties": {
        "environment": {"enum": ["dev", "staging", "prod"]},
        "steps": {"type": "array", "minItems": 1, "items": {"$ref": "step.schema.json"}},
        "owner": {"$ref": "step.schema.json#/$defs/ref_list"},
        "retries": {"type": "integer", "minimum": 1},
    },
    "$defs": {"local": {"type": "string"}},
}
FILES = {"plan.schema.json": PLAN, "step.schema.json": STEP}


def bundle():
    return validate_bundle(copy.deepcopy(FILES), "plan.schema.json")


class PointerTests(unittest.TestCase):
    def test_format_escapes(self):
        self.assertEqual(format_json_pointer(["steps", 0, "a/b", "c~d"]), "/steps/0/a~1b/c~0d")
        self.assertEqual(format_json_pointer([]), "")
        self.assertEqual(split_json_pointer("/a~1b/0"), ["a/b", "0"])
        self.assertEqual(resolve_json_pointer({"a": [1, 2]}, ["a", "*"]), [1, 2])

    def test_probe_runner_reexports(self):
        from app.services import probe_runner
        self.assertIs(probe_runner.split_json_pointer, split_json_pointer)
        self.assertIs(probe_runner.resolve_json_pointer, resolve_json_pointer)


class BundleTests(unittest.TestCase):
    def test_valid_cross_file_ref(self):
        b = bundle()
        self.assertEqual(len(b.sha256), 64)
        self.assertEqual(bundle().sha256, b.sha256)
        good = {"environment": "dev", "steps": [{"action": "create", "target": "db"}]}
        self.assertEqual(b.validate(good), [])

    def test_error_pointers_and_shape(self):
        errs = bundle().validate({"environment": "qa", "steps": [{"action": "create", "target": "x"}, {"action": "nope", "target": "y"}]})
        by = {e["pointer"]: e for e in errs}
        self.assertEqual(by["/environment"]["validator"], "enum")
        self.assertIn("/steps/1/action", by)
        self.assertEqual(set(errs[0]), {"pointer", "message", "validator", "schema_path"})
        self.assertTrue(by["/steps/1/action"]["schema_path"].startswith("/properties/steps"))
        self.assertEqual(errs, bundle().validate({"environment": "qa", "steps": [{"action": "create", "target": "x"}, {"action": "nope", "target": "y"}]}))

    def test_missing_required_and_fragment_ref(self):
        errs = bundle().validate({"environment": "dev", "steps": [{"action": "create"}], "owner": [1]})
        self.assertTrue(any(e["validator"] == "required" and e["pointer"] == "/steps/0" for e in errs))
        self.assertTrue(any(e["pointer"] == "/owner/0" for e in errs))

    def test_cap_50(self):
        errs = bundle().validate({"environment": "dev", "steps": [{"action": "x", "target": ""}] * 60})
        self.assertEqual(len(errs), 50)

    def test_ref_by_id(self):
        step = dict(STEP, **{"$id": "https://example.invalid/schemas/step.json"})
        plan = dict(PLAN, **{"$id": "https://example.invalid/schemas/plan.json"})
        plan = copy.deepcopy(plan)
        plan["properties"]["steps"]["items"] = {"$ref": "https://example.invalid/schemas/step.json"}
        plan["properties"]["owner"] = {"$ref": "https://example.invalid/schemas/step.json#/$defs/ref_list"}
        b = validate_bundle({"plan.schema.json": plan, "step.schema.json": step}, "plan.schema.json")
        self.assertEqual(b.validate({"environment": "dev", "steps": [{"action": "create", "target": "a"}]}), [])
        self.assertTrue(b.validate({"environment": "dev", "steps": [{"action": "zap", "target": "a"}]}))
        flat = validate_bundle({"f": b.dereferenced()}, "f")
        for inst in ({"environment": "dev", "steps": [{"action": "zap", "target": "a"}]}, {"environment": "dev", "steps": [{"action": "create", "target": "a"}], "owner": [1]}):
            self.assertEqual(bool(b.validate(inst)), bool(flat.validate(inst)))

    def test_remote_ref_rejected_no_network(self):
        plan = copy.deepcopy(PLAN)
        plan["properties"]["steps"]["items"] = {"$ref": "https://example.invalid/other.json"}
        with self.assertRaises(SchemaBundleError) as cm:
            validate_bundle({"plan.schema.json": plan, "step.schema.json": STEP}, "plan.schema.json")
        self.assertTrue(any("remote" in e for e in cm.exception.errors))

    def test_unknown_file_and_pointer_refs_collected(self):
        plan = copy.deepcopy(PLAN)
        plan["properties"]["a"] = {"$ref": "missing.schema.json"}
        plan["properties"]["b"] = {"$ref": "step.schema.json#/$defs/nope"}
        with self.assertRaises(SchemaBundleError) as cm:
            validate_bundle({"plan.schema.json": plan, "step.schema.json": STEP}, "plan.schema.json")
        self.assertEqual(len(cm.exception.errors), 2)

    def test_dialect_checks(self):
        bad = dict(PLAN, **{"$schema": "https://example.invalid/not-a-dialect"})
        with self.assertRaises(SchemaBundleError) as cm:
            validate_bundle({"p": bad}, "p")
        self.assertIn("dialect", cm.exception.errors[0])
        invalid = {"type": "banana"}
        with self.assertRaises(SchemaBundleError) as cm:
            validate_bundle({"p": invalid, "q": {"minItems": -1}}, "p")
        self.assertEqual(len([e for e in cm.exception.errors if e.startswith(("p:", "q:"))]), 2)
        d7 = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "required": ["a"]}
        with self.assertRaises(SchemaBundleError) as cm:
            validate_bundle({"p": d7}, "p")
        self.assertIn("unsupported $schema dialect", cm.exception.errors[0])
        self.assertIn("2019-09", cm.exception.errors[0])

    def test_structural_limits(self):
        with self.assertRaises(SchemaBundleError):
            validate_bundle({"a": {}}, "b")
        with self.assertRaises(SchemaBundleError):
            validate_bundle({"a": []}, "a")
        with self.assertRaises(SchemaBundleError):
            validate_bundle({f"f{i}.json": {} for i in range(21)}, "f0.json")
        with self.assertRaises(SchemaBundleError):
            validate_bundle({"a": {"description": "x" * (1024 * 1024 + 10)}}, "a")

    def test_dereferenced_is_self_contained_and_equivalent(self):
        b = bundle()
        flat = b.dereferenced()
        self.assertNotIn("step.schema.json", json.dumps(flat))
        self.assertEqual(flat["$schema"], "https://json-schema.org/draft/2020-12/schema")
        solo = validate_bundle({"flat.json": flat}, "flat.json")
        instances = [
            {"environment": "dev", "steps": [{"action": "create", "target": "a"}]},
            {"environment": "qa", "steps": [{"action": "create", "target": "a"}]},
            {"environment": "dev", "steps": []},
            {"environment": "dev", "steps": [{"action": "delete", "target": "a", "children": [{"action": "bad", "target": "z"}]}]},
            {"environment": "dev", "steps": [{"action": "delete", "target": "a", "children": [{"action": "update", "target": "z", "after": ["q"]}]}]},
            {"environment": "dev", "steps": [{"action": "create", "target": "a"}], "owner": [3]},
            {"environment": "dev", "steps": [{"action": "create", "target": "a"}], "retries": 0},
            "nope",
        ]
        for inst in instances:
            self.assertEqual(bool(b.validate(inst)), bool(solo.validate(inst)), inst)
        # the original files were not mutated
        self.assertEqual(b.files["plan.schema.json"], PLAN)

    def test_bundle_from_config(self):
        self.assertIsNone(bundle_from_config(None))
        self.assertIsNone(bundle_from_config({}))
        b = bundle_from_config({"schema_files": copy.deepcopy(FILES), "root_schema": "plan.schema.json"})
        self.assertEqual(b.sha256, bundle().sha256)

    def test_minimal_instance_toy_schema(self):
        b = bundle()
        inst = minimal_instance(b.dereferenced())
        self.assertEqual(b.validate(inst), [], inst)
        self.assertEqual(inst["environment"], "dev")
        self.assertEqual(inst["steps"][0]["action"], "create")

    def test_minimal_instance_keywords(self):
        s = {
            "type": "object", "required": ["a", "b", "c", "d", "e", "f"],
            "properties": {
                "a": {"const": 7}, "b": {"oneOf": [{"type": "integer", "minimum": 5}, {"type": "string"}]},
                "c": {"type": "string", "pattern": "^[a-z]+$"}, "d": {"type": "array", "minItems": 2, "items": {"type": "boolean"}},
                "e": {"type": "string", "minLength": 12}, "f": {"$ref": "#/$defs/x"},
            },
            "$defs": {"x": {"type": "number", "minimum": 2.5}},
        }
        inst = minimal_instance(s)
        self.assertEqual(inst, {"a": 7, "b": 5, "c": "a", "d": [False, False], "e": "examplexxxxx", "f": 2.5})
        self.assertEqual(validate_bundle({"s": s}, "s").validate(inst), [])


if __name__ == "__main__":
    unittest.main()


class G6FixTests(unittest.TestCase):
    def test_data_key_as_property_name_is_a_schema(self):
        remote = {"p.json": {"type": "object", "properties": {"default": {"$ref": "https://example.com/x.json"}}}}
        with self.assertRaises(SchemaBundleError):
            validate_bundle(remote, "p.json")
        files = {"p.json": {"type": "object", "properties": {"examples": {"$ref": "q.json"}, "enum": {"$ref": "q.json"}}}, "q.json": {"type": "string"}}
        b = validate_bundle(files, "p.json")
        deref = b.dereferenced()
        self.assertEqual(deref["properties"]["examples"]["$ref"], "#/$defs/q_json")
        self.assertEqual(deref["properties"]["enum"]["$ref"], "#/$defs/q_json")
        from jsonschema import Draft202012Validator
        v = Draft202012Validator(deref)
        for inst in ({"examples": 1}, {"enum": "x"}, {"examples": "ok", "enum": 2}):
            self.assertEqual(bool(b.validate(inst)), bool(list(v.iter_errors(inst))), inst)
        # still data in keyword position: a $ref-shaped default is not a ref
        ok = validate_bundle({"p.json": {"type": "object", "default": {"$ref": "nowhere.json"}}}, "p.json")
        self.assertEqual(ok.dereferenced()["default"], {"$ref": "nowhere.json"})

    def test_validate_never_raises_on_reference_errors(self):
        b = validate_bundle({"p.json": {"type": "string"}}, "p.json")
        broken = type(b)(files={"p.json": {"properties": {"a": {"$ref": "https://example.com/x.json"}}}}, root="p.json", sha256="broken-sha")
        out = broken.validate({"a": 1})
        self.assertEqual(len(out), 1)
        self.assertIn("could not be evaluated", out[0]["message"])

    def test_cross_file_anchor_is_a_clear_error(self):
        files = {"p.json": {"type": "object", "properties": {"s": {"$ref": "s.json#stp"}}}, "s.json": {"$defs": {"a": {"$anchor": "stp", "type": "string"}}}}
        with self.assertRaises(SchemaBundleError) as cm:
            validate_bundle(files, "p.json")
        self.assertIn("plain-name anchors across files are not supported", cm.exception.errors[0])

    def test_dialects(self):
        for bad in ("http://json-schema.org/draft-07/schema#", "http://json-schema.org/draft-04/schema#"):
            with self.assertRaises(SchemaBundleError):
                validate_bundle({"p": {"$schema": bad, "type": "object"}}, "p")
        with self.assertRaises(SchemaBundleError) as cm:
            validate_bundle({"p": {"$schema": "https://json-schema.org/draft/2019-09/schema"}, "q": {}}, "p")
        self.assertIn("same JSON Schema dialect", cm.exception.errors[0])

    def test_2019_09_keeps_its_dialect_and_parity(self):
        d = "https://json-schema.org/draft/2019-09/schema"
        files = {
            "p.json": {"$schema": d, "type": "object", "required": ["a"], "properties": {"a": {"$ref": "q.json"}}},
            "q.json": {"$schema": d, "type": "array", "items": {"type": "integer"}, "unevaluatedItems": False},
        }
        b = validate_bundle(files, "p.json")
        deref = b.dereferenced()
        self.assertEqual(deref["$schema"], d)
        from jsonschema import Draft201909Validator
        v = Draft201909Validator(deref)
        for inst in ({"a": [1, 2]}, {"a": ["x"]}, {}, {"a": 3}):
            self.assertEqual(bool(b.validate(inst)), bool(list(v.iter_errors(inst))), inst)

    def test_pathological_pattern_times_out(self):
        import time
        b = validate_bundle({"p": {"type": "object", "properties": {"n": {"type": "string", "pattern": "^(a|aa)+$"}}}}, "p")
        t0 = time.monotonic()
        out = b.validate({"n": "a" * 60 + "!"})
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertTrue(any("pattern check timed out" in e["message"] for e in out))
        self.assertEqual(b.validate({"n": "aaa"}), [])
        self.assertTrue(b.validate({"n": "b"}))

    def test_pathological_pattern_properties_times_out(self):
        b = validate_bundle({"p": {"type": "object", "patternProperties": {"^(a|aa)+$": {"type": "integer"}}, "additionalProperties": False}}, "p")
        out = b.validate({"a" * 60 + "!": 1})
        self.assertTrue(any("timed out" in e["message"] for e in out))
        ok = validate_bundle({"p": {"type": "object", "patternProperties": {"^x": {"type": "integer"}}, "additionalProperties": False}}, "p")
        self.assertEqual(ok.validate({"xa": 1}), [])
        self.assertTrue(ok.validate({"xa": "s"}))
        self.assertIn("Additional properties are not allowed", ok.validate({"y": 1})[0]["message"])

    def test_deep_nesting_is_a_collected_error(self):
        node: dict = {"type": "string"}
        for _ in range(3000):
            node = {"type": "array", "items": node}
        with self.assertRaises(SchemaBundleError) as cm:
            validate_bundle({"p": node}, "p")
        self.assertIn("nested too deeply", cm.exception.errors[0])


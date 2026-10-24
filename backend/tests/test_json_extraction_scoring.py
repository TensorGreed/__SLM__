"""json-extraction: typed field_match scoring, null metrics, enum/type
validation, the recipe, task-shape recommendation and the declared
``output_schema`` (save, validate, manifest forwarding).

pytest-only (non-phase file): the API tests share one TestClient opened at
module level, per the repo's testing conventions.
"""

from __future__ import annotations

import json
import unittest
from uuid import uuid4
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import recipe_service
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler
from app.services.output_schema_service import (
    MISSING,
    scalar_match,
    validate_output_schema,
    value_match,
)
from app.services.task_shape_service import detect_task_shape, structured_recipe_for_rows


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


SCHEMA = {
    "scoring_mode": "field_match",
    "required": [],
    "properties": {
        "damage": {"type": ["string", "null"], "enum": ["destroyed", "substantial", "minor", "none", None]},
        "injuries_fatal": {"type": ["integer", "null"]},
        "aircraft_make": {"type": ["string", "null"]},
    },
}


def _ctx(output_schema=None) -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile="structured_extraction",
        handler_id="structured_extraction",
        prepared_dir=Path("."),
        dataset_name="unit",
        manifest={"output_schema": output_schema} if output_schema else {},
    )


def _row(pred, gold):
    return {
        "prediction": pred if isinstance(pred, str) else json.dumps(pred),
        "reference": gold if isinstance(gold, str) else json.dumps(gold),
    }


def _score(rows, schema=SCHEMA):
    return StructuredExtractionHandler().score(rows, _ctx(schema))


class TypedCompareTests(unittest.TestCase):
    def test_null_class(self):
        for null in (None, MISSING, "null", "NULL", "  ", ""):
            self.assertTrue(scalar_match(null, None), repr(null))
        self.assertFalse(scalar_match("none", None))
        self.assertFalse(scalar_match(None, "none"))

    def test_none_string_is_a_real_value(self):
        self.assertTrue(scalar_match("none", "None"))
        self.assertFalse(scalar_match("none", "minor"))

    def test_articles_and_punctuation_are_kept(self):
        self.assertFalse(scalar_match("the destroyed", "destroyed"))
        self.assertFalse(scalar_match("a-b", "ab"))

    def test_string_normalisation(self):
        self.assertTrue(scalar_match("  Destroyed. ", "destroyed"))
        self.assertTrue(scalar_match("Light   Twin", "light twin"))
        self.assertFalse(scalar_match("destroyed..", "destroyed"))

    def test_numbers(self):
        self.assertTrue(scalar_match(1250.0, "1,250", ["integer"]))
        self.assertTrue(scalar_match("1250.0", 1250))
        self.assertTrue(scalar_match(3, "3"))
        self.assertFalse(scalar_match(3, "4"))
        self.assertFalse(scalar_match("abc", 3, ["integer"]))

    def test_booleans(self):
        self.assertTrue(scalar_match(True, "true"))
        self.assertTrue(scalar_match("TRUE", True))
        self.assertFalse(scalar_match(False, "true"))

    def test_containers_use_stable_json(self):
        self.assertTrue(value_match({"b": 1, "a": 2}, {"a": 2, "b": 1}))
        self.assertTrue(value_match(["x", "y"], ["x", "y"]))
        self.assertFalse(value_match(["x"], ["x", "y"]))
        self.assertFalse(value_match({"a": 1}, None))


class ReviewFixTests(unittest.TestCase):
    def test_containers_compare_literally(self):
        self.assertFalse(value_match({"a": 1, "b": 2}, ["a", 1, "b", 2]))
        self.assertFalse(value_match(["a b"], ["a", "b"]))
        self.assertFalse(value_match({"x": "the cat"}, {"x": "cat"}))
        self.assertFalse(value_match(["destroyed"], "destroyed"))
        self.assertFalse(value_match("destroyed", ["destroyed"]))
        self.assertTrue(value_match({"a": 1, "b": 2}, {"b": 2, "a": 1}))

    def test_declared_strings_stay_strings(self):
        self.assertFalse(scalar_match("02134", "2134", ["string"]))
        self.assertFalse(scalar_match("1e3", "1000", ["string", "null"]))
        self.assertTrue(scalar_match("02134", "02134", ["string"]))
        self.assertTrue(scalar_match("3", 3))
        self.assertTrue(scalar_match("3", 3, ["integer"]))

    def test_type_ok_is_strict(self):
        from app.services.output_schema_service import type_ok

        self.assertFalse(type_ok("3", ["integer"]))
        self.assertFalse(type_ok(True, ["integer"]))
        self.assertFalse(type_ok(3, ["string"]))
        self.assertFalse(type_ok("true", ["boolean"]))
        self.assertFalse(type_ok(True, ["number"]))
        self.assertTrue(type_ok(3.0, ["integer"]))
        self.assertFalse(type_ok(3.5, ["integer"]))
        self.assertTrue(type_ok(3.5, ["number"]))
        self.assertTrue(type_ok(True, ["boolean"]))
        self.assertTrue(type_ok({"a": 1}, ["object"]))
        self.assertTrue(type_ok([1], ["array"]))

    def test_type_violation_rate_uses_strict_types(self):
        schema = {"scoring_mode": "field_match", "required": ["n"], "properties": {"n": {"type": "integer"}}}
        res = _score([_row({"n": "3"}, {"n": 3}), _row({"n": 3}, {"n": 3})], schema)
        self.assertEqual(res["per_field"]["n"]["type_violation_rate"], 0.5)

    def test_empty_required_is_honoured(self):
        from app.services.output_schema_service import merge_manifest_output_schema

        declared = {"required": [], "properties": {"a": {"type": "string"}}}
        merged = merge_manifest_output_schema("field_match", ["a"], declared)
        self.assertEqual(merged["required"], [])
        undeclared = merge_manifest_output_schema("field_match", ["a"], {"properties": {"a": {}}})
        self.assertEqual(undeclared["required"], ["a"])
        handler = StructuredExtractionHandler()
        schema = handler._resolve_schema([], _ctx({"scoring_mode": "field_match", "required": [], "properties": {"a": {}}}))
        self.assertEqual(schema["required"], [])
        handler2 = StructuredExtractionHandler()
        schema2 = handler2._resolve_schema([], _ctx({"scoring_mode": "field_match", "properties": {"a": {}, "b": {}}}))
        self.assertEqual(schema2["required"], ["a", "b"])

    def test_mapped_output_column_picks_recipe(self):
        rows = [{"text": f"t{i}", "record_json": json.dumps({"damage": "minor"})} for i in range(5)]
        self.assertEqual(structured_recipe_for_rows(rows), "span-extraction")
        self.assertEqual(
            structured_recipe_for_rows(rows, field_mapping={"structured_output": "record_json"}),
            "json-extraction",
        )

    def test_zero_support_field_is_none(self):
        schema = {"scoring_mode": "field_match", "required": ["a", "b"], "properties": {"a": {}, "b": {}}}
        res = _score([_row({"a": "x"}, {"a": "x"})], schema)
        self.assertEqual(res["per_field"]["a"]["em"], 1.0)
        self.assertEqual(res["per_field"]["b"]["support"], 1)
        # No gold rows at all -> support 0.
        res0 = _score([{"prediction": json.dumps({"a": "x"}), "reference": ""}], schema)
        self.assertIsNone(res0["per_field"]["a"]["em"])
        self.assertIsNone(res0["per_field"]["a"]["f1"])


class FieldMatchScoringTests(unittest.TestCase):
    def test_none_string_vs_null_are_different(self):
        res = _score([_row({"damage": "none"}, {"damage": None})])
        self.assertEqual(res["per_field"]["damage"]["em"], 0.0)
        self.assertEqual(res["per_field"]["damage"]["hallucinated"], 1)
        res = _score([_row({"damage": None}, {"damage": "none"})])
        self.assertEqual(res["per_field"]["damage"]["abstained_wrongly"], 1)

    def test_article_prefixed_enum_is_a_miss_and_oov(self):
        res = _score([_row({"damage": "the destroyed"}, {"damage": "destroyed"})])
        self.assertEqual(res["per_field"]["damage"]["em"], 0.0)
        self.assertEqual(res["per_field"]["damage"]["out_of_vocab_rate"], 1.0)
        # Token F1 stays for string fields (the article is stripped there).
        self.assertEqual(res["per_field"]["damage"]["f1"], 1.0)

    def test_numeric_string_matches_float(self):
        res = _score([_row({"injuries_fatal": "1,250"}, {"injuries_fatal": 1250.0})])
        self.assertEqual(res["per_field"]["injuries_fatal"]["em"], 1.0)

    def test_missing_key_is_null(self):
        res = _score([_row({"damage": "minor"}, {"damage": "minor", "aircraft_make": None})])
        f = res["per_field"]["aircraft_make"]
        self.assertEqual(f["em"], 1.0)
        self.assertEqual(f["pred_null"], 1)
        self.assertEqual(f["gold_null"], 1)
        self.assertEqual(f["null_precision"], 1.0)
        self.assertEqual(f["null_recall"], 1.0)

    def test_overall_em_typed_and_ignores_extra_keys(self):
        gold = {"damage": "minor", "injuries_fatal": 2, "aircraft_make": None}
        ok = _row({"damage": "Minor.", "injuries_fatal": "2", "aircraft_make": "null", "extra": 1}, gold)
        bad = _row({"damage": "minor", "injuries_fatal": 3, "aircraft_make": None}, gold)
        res = _score([ok, bad])
        self.assertEqual(res["overall_em"], 0.5)
        self.assertEqual(res["exact_match"], 0.5)
        self.assertEqual(res["correct"], 1)

    def test_null_precision_recall_and_counts(self):
        rows = [
            _row({"aircraft_make": None}, {"aircraft_make": None}),       # tp
            _row({"aircraft_make": "Cessna"}, {"aircraft_make": None}),   # hallucinated
            _row({"aircraft_make": None}, {"aircraft_make": "Piper"}),    # abstained wrongly
            _row({"aircraft_make": "Piper"}, {"aircraft_make": "Piper"}), # neither
        ]
        res = _score(rows)
        f = res["per_field"]["aircraft_make"]
        self.assertEqual(f["null_precision"], 0.5)  # 1 of 2 predicted nulls
        self.assertEqual(f["null_recall"], 0.5)     # 1 of 2 gold nulls
        self.assertEqual(f["hallucinated"], 1)
        self.assertEqual(f["abstained_wrongly"], 1)
        self.assertEqual(res["hallucinated"], 1 + 0 + 0)  # other fields: all-missing == null on both sides
        self.assertIsNotNone(res["null_precision"])

    def test_micro_average_across_fields(self):
        rows = [_row({"damage": None, "aircraft_make": "X"}, {"damage": None, "aircraft_make": None})]
        res = _score(rows)
        # damage tp, injuries_fatal tp (both missing), aircraft_make hallucinated.
        self.assertEqual(res["hallucinated"], 1)
        self.assertEqual(res["abstained_wrongly"], 0)
        self.assertAlmostEqual(res["null_precision"], 2 / 2)
        self.assertAlmostEqual(res["null_recall"], 2 / 3, places=3)

    def test_invalid_json_rows_excluded_from_null_metrics(self):
        rows = [
            _row("not json at all", {"aircraft_make": None}),
            _row({"aircraft_make": None}, {"aircraft_make": None}),
        ]
        res = _score(rows)
        f = res["per_field"]["aircraft_make"]
        self.assertEqual(f["gold_null"], 1)
        self.assertEqual(f["null_recall"], 1.0)
        self.assertEqual(res["json_validity_rate"], 0.5)
        self.assertEqual(f["support"], 1)

    def test_zero_denominators_are_none_not_zero(self):
        res = _score([_row({"aircraft_make": "Piper"}, {"aircraft_make": "Piper"})])
        f = res["per_field"]["aircraft_make"]
        self.assertIsNone(f["null_precision"])
        self.assertIsNone(f["null_recall"])
        all_invalid = _score([_row("garbage", {"aircraft_make": None})])
        self.assertIsNone(all_invalid["null_precision"])
        self.assertIsNone(all_invalid["null_recall"])
        self.assertIsNone(all_invalid["schema_valid_rate"])
        self.assertIsNone(all_invalid["per_field"]["aircraft_make"]["out_of_vocab_rate"])
        self.assertEqual(_score([])["per_field"], {})
        self.assertIsNone(_score([])["null_precision"])

    def test_out_of_vocab_rate_over_non_null_predictions(self):
        rows = [
            _row({"damage": "destroyed"}, {"damage": "destroyed"}),
            _row({"damage": "wrecked"}, {"damage": "destroyed"}),
            _row({"damage": None}, {"damage": None}),
            _row({"damage": "none"}, {"damage": "none"}),
        ]
        res = _score(rows)
        self.assertAlmostEqual(res["per_field"]["damage"]["out_of_vocab_rate"], 1 / 3, places=4)
        # OOV values are still scored normally.
        self.assertEqual(res["per_field"]["damage"]["em"], 0.75)
        self.assertIsNone(res["per_field"]["aircraft_make"]["out_of_vocab_rate"])

    def test_type_violations(self):
        rows = [
            _row({"injuries_fatal": 2}, {"injuries_fatal": 2}),
            _row({"injuries_fatal": "2"}, {"injuries_fatal": 2}),      # numeric string: strict typing, violation
            _row({"injuries_fatal": 2.5}, {"injuries_fatal": 2}),      # not integral
            _row({"injuries_fatal": "two"}, {"injuries_fatal": 2}),    # not a number
            _row({"injuries_fatal": None}, {"injuries_fatal": None}),  # null allowed
        ]
        res = _score(rows)
        # 4 non-null predictions, 3 violations (string "2", 2.5, "two").
        self.assertEqual(res["per_field"]["injuries_fatal"]["type_violation_rate"], 0.75)

    def test_null_violates_when_type_excludes_null(self):
        schema = {"scoring_mode": "field_match", "properties": {"n": {"type": "integer"}}}
        res = _score([_row({"n": None}, {"n": 1}), _row({"n": 1}, {"n": 1})], schema)
        self.assertEqual(res["per_field"]["n"]["type_violation_rate"], 0.5)

    def test_schema_valid_rate(self):
        rows = [
            _row({"damage": "minor", "injuries_fatal": 1, "aircraft_make": "X"}, {}),
            _row({"damage": "wrecked", "injuries_fatal": 1, "aircraft_make": "X"}, {}),
            _row({"damage": "minor", "injuries_fatal": "many", "aircraft_make": "X"}, {}),
            _row("junk", {}),
        ]
        res = _score(rows)
        # 3 valid-JSON rows, 1 clean.
        self.assertAlmostEqual(res["schema_valid_rate"], 1 / 3, places=4)
        self.assertEqual(rows[1]["schema_violations"], {"damage": ["out_of_vocab"]})

    def test_no_properties_falls_back_to_reference_fields(self):
        gold = {"city": "Paris", "pop": 2100000}
        res = StructuredExtractionHandler().score([_row({"city": "paris", "pop": "2100000"}, gold)], _ctx())
        self.assertEqual(sorted(res["per_field"]), ["city", "pop"])
        self.assertEqual(res["overall_em"], 1.0)
        self.assertIsNone(res["schema_valid_rate"])
        self.assertIsNone(res["per_field"]["city"]["out_of_vocab_rate"])

    def test_declared_properties_define_the_field_list(self):
        schema = {"scoring_mode": "field_match", "properties": {"b": {}, "a": {}}}
        handler = StructuredExtractionHandler()
        resolved = handler._resolve_schema([], _ctx(schema))
        self.assertEqual(resolved["fields"], ["a", "b"])


class RecipeAndTaskShapeTests(unittest.TestCase):
    def test_recipe_registered(self):
        recipe = recipe_service.get_recipe("json-extraction")
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.task_profile, "structured_extraction")
        self.assertEqual(recipe.adapter_id, "structured-extraction")
        self.assertEqual(recipe.scoring_mode, "field_match")
        self.assertEqual(recipe.default_input_column, "text")
        self.assertEqual(recipe.default_output_column, "structured_output")
        # span-extraction untouched.
        self.assertEqual(recipe_service.get_recipe("span-extraction").scoring_mode, "span_set")

    def test_record_vs_entity_list_recommendation(self):
        record_rows = [
            {"text": f"Report {i}", "structured_output": {"damage": "minor", "injuries_fatal": None}}
            for i in range(8)
        ]
        as_json_text = [{"text": r["text"], "structured_output": json.dumps(r["structured_output"])} for r in record_rows]
        entity_rows = [
            {"text": "x", "structured_output": {"entities": [{"type": "a", "start": 0, "end": 1, "text": "x"}]}}
        ]
        self.assertEqual(structured_recipe_for_rows(record_rows), "json-extraction")
        self.assertEqual(structured_recipe_for_rows(as_json_text), "json-extraction")
        self.assertEqual(structured_recipe_for_rows(entity_rows), "span-extraction")
        self.assertEqual(structured_recipe_for_rows([{"text": "x"}]), "span-extraction")

    def test_detector_recommends_json_extraction_for_records(self):
        rows = [
            {"text": f"Plane {i} went down near town {i} in good weather.", "structured_output": json.dumps({"damage": "minor", "injuries_fatal": i})}
            for i in range(12)
        ]
        result = detect_task_shape(rows)
        structured = [c for c in result["candidates"] if c["task_profile"] == "structured_extraction"]
        self.assertTrue(structured, result["candidates"])
        self.assertEqual(structured[0]["recipe_id"], "json-extraction")
        self.assertEqual(structured[0]["adapter_id"], "structured-extraction")


class OutputSchemaValidationTests(unittest.TestCase):
    def test_valid_schema_roundtrips(self):
        stored = validate_output_schema(SCHEMA)
        self.assertEqual(sorted(stored["properties"]), ["aircraft_make", "damage", "injuries_fatal"])
        self.assertIn(None, stored["properties"]["damage"]["enum"])

    def test_malformed_schemas_have_clear_messages(self):
        cases = [
            ("not an object", "JSON object"),
            ({}, "properties"),
            ({"properties": {"a": {"type": "decimal"}}}, "unsupported type"),
            ({"properties": {"a": {"enum": []}}}, "enum"),
            ({"properties": {"a": {"enum": [{"x": 1}]}}}, "enum values"),
            ({"properties": {"a": {}}, "required": ["b"]}, "required"),
        ]
        for raw, needle in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError) as ctx:
                    validate_output_schema(raw)
                self.assertIn(needle, str(ctx.exception))


class OutputSchemaApiTests(unittest.TestCase):
    def _project(self, recipe_id="json-extraction") -> int:
        resp = client.post("/api/projects", json={"name": f"json-extraction-schema-{uuid4().hex[:8]}", "description": "t"})
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])
        resp = client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": recipe_id})
        self.assertEqual(resp.status_code, 200, resp.text)
        return pid

    def test_save_validate_and_clear(self):
        pid = self._project()
        url = f"/api/projects/{pid}/recipe/output-schema"
        resp = client.put(url, json={"output_schema": SCHEMA})
        self.assertEqual(resp.status_code, 200, resp.text)
        saved = resp.json()["selected_recipe"]["output_schema"]
        self.assertEqual(sorted(saved["properties"]), ["aircraft_make", "damage", "injuries_fatal"])

        bad = client.put(url, json={"output_schema": {"properties": {"a": {"type": "decimal"}}}})
        self.assertEqual(bad.status_code, 400, bad.text)
        self.assertIn("unsupported type", json.dumps(bad.json()))

        # Re-applying the same recipe keeps the declared schema.
        again = client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "json-extraction"})
        self.assertIn("output_schema", again.json()["selected_recipe"])

        cleared = client.put(url, json={"output_schema": None})
        self.assertEqual(cleared.status_code, 200, cleared.text)
        self.assertNotIn("output_schema", cleared.json()["selected_recipe"])

    def test_other_recipes_reject_a_schema(self):
        pid = self._project("qa-sft")
        resp = client.put(f"/api/projects/{pid}/recipe/output-schema", json={"output_schema": SCHEMA})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("json-extraction", json.dumps(resp.json()))

    def test_manifest_forwarding(self):
        import asyncio

        from app.database import async_session_factory
        from app.services.dataset_service import _recipe_output_schema

        pid = self._project()
        client.put(f"/api/projects/{pid}/recipe/output-schema", json={"output_schema": SCHEMA})

        async def run():
            async with async_session_factory() as db:
                return await _recipe_output_schema(db, pid)

        merged = asyncio.run(run())
        self.assertEqual(merged["scoring_mode"], "field_match")
        self.assertEqual(sorted(merged["properties"]), ["aircraft_make", "damage", "injuries_fatal"])
        self.assertIn(None, merged["properties"]["damage"]["enum"])

        # The handler turns that manifest block into the field list + enums.
        resolved = StructuredExtractionHandler()._resolve_schema([], _ctx(merged))
        self.assertEqual(resolved["fields"], ["aircraft_make", "damage", "injuries_fatal"])
        self.assertEqual(resolved["properties"]["injuries_fatal"]["types"], ["integer", "null"])

    def test_confirm_task_shape_picks_json_extraction(self):
        pid = int(client.post("/api/projects", json={"name": f"ts-json-{uuid4().hex[:8]}", "description": "t"}).json()["id"])
        resp = client.post(
            f"/api/projects/{pid}/task-shape/confirm",
            json={"task_profile": "structured_extraction", "recipe_id": "json-extraction"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["recipe_id"], "json-extraction")


if __name__ == "__main__":
    unittest.main()

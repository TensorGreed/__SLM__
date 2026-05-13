"""Phase 5.3.4 — StructuredExtractionHandler.

Pins every contract the plan made:

- Dispatcher routes ``structured_extraction`` and the ``extraction``
  alias to this handler.
- Schema source priority: manifest.output_schema first, derived from
  reference rows second.
- Output parser handles raw JSON, ``​```json …  ``​`` code-fenced
  output, and "prose then JSON" by extracting the first balanced
  ``{…}`` block.
- ``json_validity_rate`` flags malformed outputs as a separate metric
  (a 30% malformed model is unshippable regardless of field accuracy).
- ``schema_compliance_rate`` flags rows missing any required field.
- Per-field EM + F1 are reported as ``per_field``, with per-field
  averages exposed as ``field_exact_match_rate`` and ``field_f1``.
- Legacy aliases: ``exact_match`` (whole-blob equality, gate compat)
  and ``f1`` (mean per-field F1, the most useful aggregate).
- Each prediction is enriched in place with parsed values + field
  results so the predictions_preview writer can flow them to the UI.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.eval_task_handler_service import (  # noqa: E402
    EvalContext,
    StructuredExtractionHandler,
    resolve_task_handler,
)


def _ctx(manifest: dict | None = None) -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile="structured_extraction",
        handler_id="structured_extraction",
        prepared_dir=Path("."),
        dataset_name="test",
        manifest=manifest or {},
    )


class DispatcherRoutingTests(unittest.TestCase):
    def test_structured_extraction_routes_to_handler(self):
        self.assertIsInstance(
            resolve_task_handler("structured_extraction"),
            StructuredExtractionHandler,
        )

    def test_extraction_alias_routes_to_handler(self):
        self.assertIsInstance(
            resolve_task_handler("extraction"), StructuredExtractionHandler
        )


class JsonParserTests(unittest.TestCase):
    def test_parses_clean_json(self):
        h = StructuredExtractionHandler()
        self.assertEqual(h._parse_json_safely('{"a": 1}'), {"a": 1})

    def test_strips_code_fences(self):
        h = StructuredExtractionHandler()
        fenced = '```json\n{"a": 1, "b": "x"}\n```'
        self.assertEqual(h._parse_json_safely(fenced), {"a": 1, "b": "x"})

    def test_strips_unmarked_fences(self):
        h = StructuredExtractionHandler()
        fenced = '```\n{"a": 1}\n```'
        self.assertEqual(h._parse_json_safely(fenced), {"a": 1})

    def test_extracts_balanced_block_from_prose(self):
        h = StructuredExtractionHandler()
        prose = 'Here is your JSON: {"invoice_no": "123", "total": 50.0}. Hope that helps!'
        out = h._parse_json_safely(prose)
        self.assertEqual(out, {"invoice_no": "123", "total": 50.0})

    def test_returns_none_for_malformed_json(self):
        h = StructuredExtractionHandler()
        self.assertIsNone(h._parse_json_safely('{"a": 1, "b":}'))
        self.assertIsNone(h._parse_json_safely('not even close to JSON'))
        self.assertIsNone(h._parse_json_safely(''))
        self.assertIsNone(h._parse_json_safely(None))

    def test_returns_none_for_top_level_list(self):
        # Handler scores JSON OBJECTS; arrays are not the contract.
        h = StructuredExtractionHandler()
        self.assertIsNone(h._parse_json_safely('[1, 2, 3]'))

    def test_passes_through_dict_input(self):
        h = StructuredExtractionHandler()
        d = {"a": 1}
        self.assertIs(h._parse_json_safely(d), d)


class SchemaResolutionTests(unittest.TestCase):
    def test_uses_manifest_output_schema(self):
        h = StructuredExtractionHandler()
        ctx = _ctx(
            {
                "output_schema": {
                    "properties": {
                        "invoice_no": {"type": "string"},
                        "total": {"type": "number"},
                        "date": {"type": "string"},
                    },
                    "required": ["invoice_no", "total"],
                }
            }
        )
        schema = h._resolve_schema([], ctx)
        self.assertEqual(schema["fields"], ["date", "invoice_no", "total"])
        self.assertEqual(sorted(schema["required"]), ["invoice_no", "total"])

    def test_derives_from_reference_when_manifest_missing(self):
        h = StructuredExtractionHandler()
        records = [
            {"reference": '{"name": "A", "age": 30}'},
            {"reference": '{"name": "B", "city": "X"}'},
        ]
        schema = h._resolve_schema(records, _ctx())
        self.assertEqual(schema["fields"], ["age", "city", "name"])

    def test_derives_handles_malformed_references_gracefully(self):
        h = StructuredExtractionHandler()
        records = [
            {"reference": '{"name": "A"}'},
            {"reference": 'malformed not json'},
            {"reference": '{"city": "Y"}'},
        ]
        schema = h._resolve_schema(records, _ctx())
        self.assertEqual(schema["fields"], ["city", "name"])


class PromptAssemblyTests(unittest.TestCase):
    def test_prompt_lists_declared_fields(self):
        h = StructuredExtractionHandler()
        ctx = _ctx(
            {
                "output_schema": {
                    "properties": {
                        "invoice_no": {"type": "string"},
                        "total": {"type": "number"},
                    },
                    "required": ["invoice_no", "total"],
                }
            }
        )
        built = h.build_prompts(
            [
                {
                    "text": "Receipt: $50 invoice #123",
                    "reference": '{"invoice_no": "123", "total": 50}',
                }
            ],
            ctx,
        )
        self.assertIn("Extract the following fields", built[0].prompt)
        self.assertIn("invoice_no, total", built[0].prompt)
        self.assertIn("Receipt: $50 invoice #123", built[0].prompt)
        self.assertTrue(built[0].prompt.rstrip().endswith("Output:"))

    def test_prompt_falls_back_when_no_fields_known(self):
        h = StructuredExtractionHandler()
        built = h.build_prompts(
            [{"text": "Some text", "reference": "garbage not json"}], _ctx()
        )
        # No schema deducible → generic extraction prompt.
        self.assertIn("Extract the relevant fields", built[0].prompt)

    def test_extras_carry_fields_and_input(self):
        h = StructuredExtractionHandler()
        ctx = _ctx(
            {"output_schema": {"properties": {"a": {}, "b": {}}, "required": ["a"]}}
        )
        built = h.build_prompts([{"text": "hi", "reference": '{"a": 1}'}], ctx)
        self.assertEqual(built[0].extras["structured_fields"], ["a", "b"])
        self.assertEqual(built[0].extras["structured_input"], "hi")


class MaxNewTokensOverrideTests(unittest.TestCase):
    def test_raises_tiny_default_to_floor(self):
        h = StructuredExtractionHandler()
        self.assertEqual(h.max_new_tokens_override(16), 128)
        self.assertEqual(h.max_new_tokens_override(0), 128)

    def test_passes_through_reasonable_default(self):
        h = StructuredExtractionHandler()
        self.assertEqual(h.max_new_tokens_override(256), 256)

    def test_hardcaps_at_512(self):
        h = StructuredExtractionHandler()
        self.assertEqual(h.max_new_tokens_override(1024), 512)


class ScoringTests(unittest.TestCase):
    def _ctx_schema(self, required=None):
        return _ctx(
            {
                "output_schema": {
                    "properties": {
                        "invoice_no": {"type": "string"},
                        "total": {"type": "number"},
                        "date": {"type": "string"},
                    },
                    "required": required or ["invoice_no", "total"],
                }
            }
        )

    def test_perfect_extraction_full_metrics(self):
        h = StructuredExtractionHandler()
        predictions = [
            {
                "prediction": '{"invoice_no": "INV-001", "total": "50", "date": "2026-01-01"}',
                "reference": '{"invoice_no": "INV-001", "total": "50", "date": "2026-01-01"}',
            }
        ]
        out = h.score(predictions, self._ctx_schema())
        self.assertEqual(out["json_validity_rate"], 1.0)
        self.assertEqual(out["schema_compliance_rate"], 1.0)
        self.assertEqual(out["field_exact_match_rate"], 1.0)
        self.assertEqual(out["field_f1"], 1.0)
        self.assertEqual(out["overall_em"], 1.0)
        # Legacy aliases for gate compat.
        self.assertEqual(out["exact_match"], 1.0)
        self.assertEqual(out["f1"], 1.0)

    def test_malformed_output_dings_validity_not_field_rates(self):
        h = StructuredExtractionHandler()
        predictions = [
            {
                "prediction": "not even close to json",
                "reference": '{"invoice_no": "INV-001", "total": "50"}',
            },
            {
                "prediction": '{"invoice_no": "INV-002", "total": "75"}',
                "reference": '{"invoice_no": "INV-002", "total": "75"}',
            },
        ]
        out = h.score(predictions, self._ctx_schema())
        self.assertEqual(out["json_validity_rate"], 0.5)
        self.assertEqual(out["schema_compliance_rate"], 0.5)
        # Field rates only span rows where parsing succeeded — they
        # should reflect the valid row's perfect score, not be dragged
        # down by the malformed one.
        self.assertGreater(out["field_exact_match_rate"], 0.0)

    def test_schema_noncompliance_when_required_field_missing(self):
        h = StructuredExtractionHandler()
        predictions = [
            {
                "prediction": '{"invoice_no": "INV-001"}',  # missing required "total"
                "reference": '{"invoice_no": "INV-001", "total": "50"}',
            }
        ]
        out = h.score(predictions, self._ctx_schema())
        self.assertEqual(out["json_validity_rate"], 1.0)
        self.assertEqual(out["schema_compliance_rate"], 0.0)
        # Enrichment surfaces the missing field name.
        self.assertEqual(
            predictions[0]["missing_required_fields"], ["total"]
        )

    def test_per_field_breakdown_in_score(self):
        h = StructuredExtractionHandler()
        predictions = [
            {
                "prediction": '{"invoice_no": "INV-001", "total": "WRONG", "date": "2026-01-01"}',
                "reference": '{"invoice_no": "INV-001", "total": "50", "date": "2026-01-01"}',
            }
        ]
        out = h.score(predictions, self._ctx_schema())
        per_field = out["per_field"]
        self.assertEqual(per_field["invoice_no"]["em"], 1.0)
        self.assertEqual(per_field["total"]["em"], 0.0)
        self.assertEqual(per_field["date"]["em"], 1.0)
        # field_exact_match_rate is mean of per-field EM (1+0+1)/3 ≈ 0.667
        self.assertAlmostEqual(out["field_exact_match_rate"], 0.6667, places=3)

    def test_per_row_enrichment_written_in_place(self):
        h = StructuredExtractionHandler()
        predictions = [
            {
                "prediction": '{"invoice_no": "X"}',
                "reference": '{"invoice_no": "X", "total": "1"}',
            }
        ]
        h.score(predictions, self._ctx_schema())
        # Parsed dicts land on each prediction.
        self.assertEqual(
            predictions[0]["parsed_prediction"], {"invoice_no": "X"}
        )
        self.assertEqual(
            predictions[0]["parsed_reference"],
            {"invoice_no": "X", "total": "1"},
        )
        self.assertTrue(predictions[0]["is_valid_json"])
        self.assertEqual(predictions[0]["missing_required_fields"], ["total"])
        # Row-level scores so the UI badge can render.
        self.assertEqual(predictions[0]["row_exact_match"], 0.0)
        self.assertIn("row_f1", predictions[0])
        # Per-field comparison results land too.
        self.assertEqual(
            predictions[0]["row_field_results"]["invoice_no"]["em"], 1.0
        )

    def test_empty_predictions_returns_zeroed_metrics(self):
        out = StructuredExtractionHandler().score([], _ctx())
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["json_validity_rate"], 0.0)
        self.assertEqual(out["schema_compliance_rate"], 0.0)
        self.assertEqual(out["field_f1"], 0.0)


class EndToEndIntegrationTests(unittest.TestCase):
    def test_full_pipeline_perfect_model(self):
        h = StructuredExtractionHandler()
        ctx = _ctx(
            {
                "output_schema": {
                    "properties": {"name": {}, "age": {}},
                    "required": ["name"],
                }
            }
        )
        rows = [
            {"text": "Alice is 30", "reference": '{"name": "Alice", "age": "30"}'},
            {"text": "Bob is 25", "reference": '{"name": "Bob", "age": "25"}'},
        ]
        built = h.build_prompts(rows, ctx)
        # Simulate a perfect model: emits the reference JSON inside code fences.
        predictions = [
            {
                "prediction": f"```json\n{bp.reference}\n```",
                "reference": bp.reference,
            }
            for bp in built
        ]
        out = h.score(predictions, ctx)
        self.assertEqual(out["json_validity_rate"], 1.0)
        self.assertEqual(out["overall_em"], 1.0)
        self.assertEqual(out["field_exact_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

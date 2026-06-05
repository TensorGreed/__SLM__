"""Unit tests for the eval-pack gate-options catalog + validator
(Gap #5 slice 1).

These tests cover the pure-function pieces — no DB, no API — so the
guarantees the FE editor relies on are pinned independently of the
endpoint plumbing:

  * The operator whitelist matches the eval engine's actual support
    (``gte``/``lte`` only — anything else is silently coerced today,
    and the validator's job is to fail loudly instead).
  * Catalog metric_ids match ``_BASE_METRIC_SCHEMA`` so a metric added
    to the eval engine flows into the editor automatically.
  * Recipe-aware ``recommended`` flag mirrors the scaffolder's
    required_metric_ids for that recipe.
  * Validator rejects: invalid operators, unknown metric_ids,
    duplicate gate_ids, missing thresholds, out-of-range thresholds,
    missing gate_id / metric_id, non-dict gate entries.
  * Validator is a NO-OP on the canonical scaffolder output (so we
    don't accidentally break an existing legal pack).
"""

from __future__ import annotations

import unittest

from app.services.eval_pack_scaffold_service import scaffold_pack
from app.services.evaluation_gate_catalog import (
    GATE_OPERATORS,
    LTE_DEFAULT_METRIC_IDS,
    VALID_GATE_OPERATORS,
    build_gate_options,
    known_metric_ids,
    validate_draft_pack_gates,
)
from app.services.evaluation_pack_service import _BASE_METRIC_SCHEMA


class GateCatalogShapeTests(unittest.TestCase):

    def test_operator_whitelist_matches_engine_support(self):
        # The eval engine implements only gte/lte (see
        # ``_evaluate_gate`` in evaluation_pack_service); anything else
        # silently coerces to gte. The catalog's job is to surface this
        # whitelist to the FE so a user picking ``eq`` gets a real
        # validator error instead of a silent coercion.
        self.assertEqual(VALID_GATE_OPERATORS, {"gte", "lte"})
        values = {op["value"] for op in GATE_OPERATORS}
        self.assertEqual(values, VALID_GATE_OPERATORS)
        # Labels are human-readable, not just the raw value.
        for op in GATE_OPERATORS:
            self.assertNotEqual(op["label"], op["value"])

    def test_known_metric_ids_is_superset_of_base_schema(self):
        # ``known_metric_ids`` unions the base schema with metric_ids
        # the scaffolder emits (rouge_l, span_set_f1, etc.) — the eval
        # engine accepts those via a [0,1] fallback, so the validator
        # must accept them too. Base-schema keys are a subset.
        known = known_metric_ids()
        self.assertTrue(set(_BASE_METRIC_SCHEMA.keys()).issubset(known))
        # Scaffolder-emitted metrics that aren't in the base schema
        # also surface — pick one as a smoke test.
        self.assertIn("rouge_l", known)

    def test_lte_default_for_lower_is_better_metrics(self):
        # hallucination_rate is the canonical "lower is better"
        # metric on the platform — flagging it for an lte default in
        # the editor saves the user one click.
        self.assertIn("hallucination_rate", LTE_DEFAULT_METRIC_IDS)
        options = build_gate_options(recipe_id="rag-protocol")
        by_id = {m["metric_id"]: m for m in options["metrics"]}
        self.assertEqual(by_id["hallucination_rate"]["default_operator"], "lte")
        # And the "higher is better" metrics keep the gte default.
        self.assertEqual(by_id["f1"]["default_operator"], "gte")


class BuildGateOptionsTests(unittest.TestCase):

    def test_catalog_returns_every_base_schema_metric(self):
        options = build_gate_options(recipe_id="qa-sft")
        returned_ids = {m["metric_id"] for m in options["metrics"]}
        self.assertEqual(returned_ids, set(_BASE_METRIC_SCHEMA.keys()))

    def test_recommended_flag_is_recipe_aware(self):
        # qa-sft scaffold recommends exact_match, f1, llm_judge_pass_rate,
        # safety_pass_rate. The catalog must mark exactly those as
        # ``recommended=True`` so the FE can sort/badge them.
        options = build_gate_options(recipe_id="qa-sft")
        recommended = {
            m["metric_id"] for m in options["metrics"] if m["recommended"]
        }
        expected = {"exact_match", "f1", "llm_judge_pass_rate", "safety_pass_rate"}
        self.assertEqual(recommended, expected)

    def test_unknown_recipe_returns_full_catalog_with_zero_recommended(self):
        # Unknown / null recipe → still return the whole catalog but
        # don't flag anything as recommended. The editor stays usable;
        # it just doesn't bias the picker.
        options = build_gate_options(recipe_id="never-heard-of-it")
        recommended = [m for m in options["metrics"] if m["recommended"]]
        self.assertEqual(recommended, [])
        # All metrics still present.
        returned_ids = {m["metric_id"] for m in options["metrics"]}
        self.assertEqual(returned_ids, set(_BASE_METRIC_SCHEMA.keys()))

    def test_metric_entries_carry_label_description_range(self):
        options = build_gate_options(recipe_id="classification")
        f1_entry = next(m for m in options["metrics"] if m["metric_id"] == "f1")
        # Label is humanised, not raw snake_case.
        self.assertNotEqual(f1_entry["label"], "f1")
        # Description carries the schema's documented sentence.
        self.assertTrue(f1_entry["description"])
        # Expected range is the engine's range for this metric.
        self.assertEqual(f1_entry["expected_range"], [0.0, 1.0])


class ValidateDraftPackGatesTests(unittest.TestCase):

    def test_scaffolder_output_passes_validation_unchanged(self):
        # Every recipe the scaffolder knows about should produce a
        # draft pack that the validator immediately accepts — the
        # whole point of the validator is to be a no-op on legal
        # input. If this test ever fails, the scaffolder is emitting
        # malformed gates and the user would see a 400 trying to save.
        for recipe_id in (
            "classification",
            "span-extraction",
            "summarization",
            "qa-sft",
            "generic-sft",
            "code-review",
        ):
            with self.subTest(recipe_id=recipe_id):
                pack = scaffold_pack(recipe_id)
                # Should not raise.
                validate_draft_pack_gates(pack)

    def test_rejects_invalid_operator(self):
        pack = scaffold_pack("qa-sft")
        pack["task_specs"][0]["gates"][0]["operator"] = "eq"
        with self.assertRaises(ValueError) as ctx:
            validate_draft_pack_gates(pack)
        self.assertEqual(str(ctx.exception), "invalid_gate_operator:eq")

    def test_rejects_unknown_metric_id(self):
        pack = scaffold_pack("qa-sft")
        pack["task_specs"][0]["gates"][0]["metric_id"] = "made_up_metric"
        with self.assertRaises(ValueError) as ctx:
            validate_draft_pack_gates(pack)
        self.assertEqual(str(ctx.exception), "unknown_metric_id:made_up_metric")

    def test_rejects_duplicate_gate_id_within_task_spec(self):
        pack = scaffold_pack("qa-sft")
        first = dict(pack["task_specs"][0]["gates"][0])
        duplicate = dict(first)
        duplicate["metric_id"] = "f1"  # second gate, same gate_id
        pack["task_specs"][0]["gates"] = [first, duplicate]
        with self.assertRaises(ValueError) as ctx:
            validate_draft_pack_gates(pack)
        self.assertTrue(str(ctx.exception).startswith("duplicate_gate_id:"))

    def test_rejects_missing_threshold(self):
        pack = scaffold_pack("qa-sft")
        gate = pack["task_specs"][0]["gates"][0]
        gate["threshold"] = None
        with self.assertRaises(ValueError) as ctx:
            validate_draft_pack_gates(pack)
        self.assertTrue(str(ctx.exception).startswith("missing_threshold:"))

    def test_rejects_unparseable_threshold(self):
        pack = scaffold_pack("qa-sft")
        gate = pack["task_specs"][0]["gates"][0]
        gate["threshold"] = "not-a-number"
        with self.assertRaises(ValueError) as ctx:
            validate_draft_pack_gates(pack)
        self.assertTrue(str(ctx.exception).startswith("missing_threshold:"))

    def test_rejects_threshold_out_of_range(self):
        pack = scaffold_pack("qa-sft")
        gate = pack["task_specs"][0]["gates"][0]
        gate["threshold"] = 1.5  # F1 expected_range is [0, 1]
        with self.assertRaises(ValueError) as ctx:
            validate_draft_pack_gates(pack)
        self.assertTrue(str(ctx.exception).startswith("threshold_out_of_range:"))

    def test_rejects_missing_gate_id(self):
        pack = scaffold_pack("qa-sft")
        gate = pack["task_specs"][0]["gates"][0]
        gate["gate_id"] = ""
        with self.assertRaises(ValueError) as ctx:
            validate_draft_pack_gates(pack)
        self.assertEqual(str(ctx.exception), "missing_gate_id")

    def test_rejects_missing_metric_id(self):
        pack = scaffold_pack("qa-sft")
        gate = pack["task_specs"][0]["gates"][0]
        gate["metric_id"] = ""
        with self.assertRaises(ValueError) as ctx:
            validate_draft_pack_gates(pack)
        self.assertTrue(str(ctx.exception).startswith("missing_metric_id:"))

    def test_rejects_non_dict_gate_entry(self):
        pack = scaffold_pack("qa-sft")
        # A stringy gate is the kind of thing a malformed JSON edit
        # could produce — the validator must catch it.
        pack["task_specs"][0]["gates"][0] = "not-a-dict"
        with self.assertRaises(ValueError) as ctx:
            validate_draft_pack_gates(pack)
        self.assertEqual(str(ctx.exception), "invalid_gate_shape")

    def test_no_op_when_pack_has_no_task_specs(self):
        # Pack-level shape (missing task_specs) is the caller's
        # responsibility — the gate validator should return silently
        # rather than raise a confusing gate-level error.
        validate_draft_pack_gates({})
        validate_draft_pack_gates({"task_specs": None})
        validate_draft_pack_gates({"task_specs": []})


if __name__ == "__main__":
    unittest.main()

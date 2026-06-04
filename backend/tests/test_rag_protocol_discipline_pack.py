"""Tests for the rag_protocol.discipline eval pack + the
appropriate_refusal_rate metric emitted by RAGHandler (Arc R-2).

Covers:
  * Pack is registered and has the 4 protocol-specific gates
    (citation_rate, hallucination_rate, appropriate_refusal_rate,
    format_consistency) plus the legacy F1 + safety gates.
  * Metric schema has the 4 new entries with the right aliases —
    aliasing is what lets ``citation_rate`` resolve against the
    handler-emitted ``faithfulness_rate`` field without renaming
    on either side.
  * Gate operators: citation/refusal/F1 are ``gte``; hallucination
    is ``lte``. Wrong operator means the gate passes on the wrong
    side of the threshold — silent regression.
  * format_consistency gate is OPTIONAL (required=False) so the
    pack stays usable while Slice 2 implements the metric.
  * rag-protocol recipe.eval_pack_id points at the new pack so
    projects on the recipe inherit the discipline gates by default.
  * RAGHandler.score() emits appropriate_refusal_rate with the
    correct ratio + the per-row enrichment fields.
  * Refusal-phrase detector returns True for canonical phrase
    + the 4 documented variants; False on confident answers.
"""

from __future__ import annotations

import unittest

from app.services.recipe_service import get_recipe
from app.services.evaluation_pack_service import (
    _BASE_METRIC_SCHEMA,
    get_evaluation_pack,
    list_evaluation_packs,
)
from app.services.eval_task_handler_service import RAGHandler


class RagProtocolDisciplinePackTests(unittest.TestCase):

    # ─────────────────────────────────────────────────────────────
    # Pack registration
    # ─────────────────────────────────────────────────────────────

    def test_pack_is_registered(self):
        pack_ids = {p["pack_id"] for p in list_evaluation_packs(include_gates=True)}
        self.assertIn("evalpack.rag_protocol.discipline", pack_ids)

    def test_pack_carries_the_four_protocol_gates(self):
        pack = get_evaluation_pack("evalpack.rag_protocol.discipline")
        self.assertIsNotNone(pack, "discipline pack should resolve via get_evaluation_pack")
        task_specs = pack["task_specs"]
        self.assertEqual(len(task_specs), 1)
        rag_spec = task_specs[0]
        self.assertEqual(rag_spec["task_profile"], "rag_qa")
        gate_ids = {g["gate_id"] for g in rag_spec["gates"]}
        for required_id in (
            "min_f1",
            "min_citation_rate",
            "max_hallucination_rate",
            "min_appropriate_refusal_rate",
            "min_format_consistency",
        ):
            self.assertIn(required_id, gate_ids)

    def test_hallucination_gate_uses_lte_operator(self):
        pack = get_evaluation_pack("evalpack.rag_protocol.discipline")
        gates = pack["task_specs"][0]["gates"]
        by_id = {g["gate_id"]: g for g in gates}
        self.assertEqual(by_id["max_hallucination_rate"]["operator"], "lte")
        # And the others stay gte.
        for gid in ("min_f1", "min_citation_rate", "min_appropriate_refusal_rate"):
            self.assertEqual(by_id[gid]["operator"], "gte", f"{gid} should use gte")

    def test_format_consistency_gate_is_optional(self):
        # Slice 2 implements the actual format-consistency metric.
        # Until then the gate must stay required=False so the pack
        # doesn't permanently fail on every rag-protocol eval.
        pack = get_evaluation_pack("evalpack.rag_protocol.discipline")
        gates = pack["task_specs"][0]["gates"]
        by_id = {g["gate_id"]: g for g in gates}
        self.assertFalse(by_id["min_format_consistency"]["required"])

    def test_thresholds_match_documented_targets(self):
        pack = get_evaluation_pack("evalpack.rag_protocol.discipline")
        gates = pack["task_specs"][0]["gates"]
        by_id = {g["gate_id"]: g for g in gates}
        self.assertEqual(by_id["min_f1"]["threshold"], 0.55)
        self.assertEqual(by_id["min_citation_rate"]["threshold"], 0.75)
        self.assertEqual(by_id["max_hallucination_rate"]["threshold"], 0.15)
        self.assertEqual(by_id["min_appropriate_refusal_rate"]["threshold"], 0.80)

    # ─────────────────────────────────────────────────────────────
    # Metric schema + aliasing
    # ─────────────────────────────────────────────────────────────

    def test_metric_schema_has_the_four_new_metrics(self):
        for metric_id in (
            "citation_rate",
            "hallucination_rate",
            "appropriate_refusal_rate",
            "format_consistency",
        ):
            self.assertIn(metric_id, _BASE_METRIC_SCHEMA, f"missing metric {metric_id}")
            entry = _BASE_METRIC_SCHEMA[metric_id]
            self.assertEqual(entry["expected_range"], [0.0, 1.0])
            self.assertIn("aliases", entry)
            self.assertGreater(len(entry["aliases"]), 0)

    def test_citation_rate_aliases_resolve_handler_emitted_faithfulness_rate(self):
        # The handler emits ``faithfulness_rate``; the gate keys on
        # ``citation_rate``. Without the alias the gate can never
        # find its value — silent gate-skip regression.
        aliases = _BASE_METRIC_SCHEMA["citation_rate"]["aliases"]
        self.assertIn("faithfulness_rate", aliases)

    def test_hallucination_rate_aliases_resolve_unsupported_token_rate_mean(self):
        aliases = _BASE_METRIC_SCHEMA["hallucination_rate"]["aliases"]
        self.assertIn("unsupported_token_rate_mean", aliases)

    # ─────────────────────────────────────────────────────────────
    # Recipe ↔ pack pairing
    # ─────────────────────────────────────────────────────────────

    def test_rag_protocol_recipe_uses_discipline_pack(self):
        recipe = get_recipe("rag-protocol")
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.eval_pack_id, "evalpack.rag_protocol.discipline")

    # ─────────────────────────────────────────────────────────────
    # RAGHandler.score() — refusal metric
    # ─────────────────────────────────────────────────────────────

    def test_appropriate_refusal_rate_perfect_match(self):
        # 4 rows: 2 expected refusals matched, 2 expected answers
        # matched. All 4 align with the gold → rate = 1.0.
        predictions = [
            {
                "prediction": "30 days from delivery [#1].",
                "reference": "30 days from delivery [#1].",
                "rag_context": "[#1] Refunds within 30 days.",
                "rag_has_context": True,
            },
            {
                "prediction": "I don't have enough context to answer that.",
                "reference": "I don't have enough context to answer that.",
                "rag_context": "",
                "rag_has_context": False,
            },
            {
                "prediction": "On orders over $50 [#1].",
                "reference": "On orders over $50 [#1].",
                "rag_context": "[#1] Free shipping over $50.",
                "rag_has_context": True,
            },
            {
                "prediction": "I don't have enough context to answer that.",
                "reference": "I don't have enough context to answer that.",
                "rag_context": "",
                "rag_has_context": False,
            },
        ]
        handler = RAGHandler()
        # build_prompts is the prep step; we just need score() to be
        # called with predictions that look like what the trainer
        # produced. The handler doesn't require any other setup.
        result = handler.score(predictions, ctx={})
        self.assertEqual(result["appropriate_refusal_rate"], 1.0)
        self.assertEqual(result["expected_refusal_rows"], 2)
        self.assertEqual(result["refusal_match_rows"], 4)

    def test_appropriate_refusal_rate_penalises_hallucination_when_gold_refuses(self):
        predictions = [
            # Gold refuses; model hallucinates an answer → mismatch.
            {
                "prediction": "Sure, the store closes at 9pm.",
                "reference": "I don't have enough context to answer that.",
                "rag_context": "[#1] Refunds within 30 days.",
                "rag_has_context": True,
            },
            # Gold answers; model also answers → match.
            {
                "prediction": "30 days [#1].",
                "reference": "30 days [#1].",
                "rag_context": "[#1] Refunds within 30 days.",
                "rag_has_context": True,
            },
        ]
        handler = RAGHandler()
        result = handler.score(predictions, ctx={})
        # 1 of 2 rows matched the gold refusal signal.
        self.assertEqual(result["appropriate_refusal_rate"], 0.5)
        self.assertEqual(result["expected_refusal_rows"], 1)
        self.assertEqual(result["refusal_match_rows"], 1)

    def test_appropriate_refusal_rate_penalises_blanket_refusal(self):
        # Model refuses on every row, even ones with sufficient
        # context — the "blanket refusal" failure mode. The gold
        # ANSWERS; the model REFUSES → mismatch.
        predictions = [
            {
                "prediction": "I don't have enough context to answer that.",
                "reference": "30 days [#1].",
                "rag_context": "[#1] Refunds within 30 days.",
                "rag_has_context": True,
            },
            {
                "prediction": "I don't have enough context to answer that.",
                "reference": "On orders over $50 [#1].",
                "rag_context": "[#1] Free shipping over $50.",
                "rag_has_context": True,
            },
        ]
        handler = RAGHandler()
        result = handler.score(predictions, ctx={})
        self.assertEqual(result["appropriate_refusal_rate"], 0.0)
        self.assertEqual(result["expected_refusal_rows"], 0)
        self.assertEqual(result["refusal_match_rows"], 0)

    def test_appropriate_refusal_rate_empty_predictions(self):
        # Empty eval set still emits the field rather than crashing.
        handler = RAGHandler()
        result = handler.score([], ctx={})
        self.assertEqual(result["appropriate_refusal_rate"], 0.0)
        self.assertEqual(result["expected_refusal_rows"], 0)
        self.assertEqual(result["refusal_match_rows"], 0)

    def test_per_row_enrichment_fields_are_set(self):
        # The handler enriches each prediction in-place with the
        # discipline diagnostics so the failure-cluster panel + the
        # eval drilldown can render them per row.
        predictions = [
            {
                "prediction": "I don't have enough context to answer that.",
                "reference": "Real answer [#1].",
                "rag_context": "[#1] Refunds within 30 days.",
                "rag_has_context": True,
            },
        ]
        handler = RAGHandler()
        handler.score(predictions, ctx={})
        row = predictions[0]
        self.assertTrue(row["rag_pred_is_refusal"])
        self.assertFalse(row["rag_ref_is_refusal"])
        self.assertFalse(row["rag_refusal_appropriate"])

    # ─────────────────────────────────────────────────────────────
    # Refusal-phrase detector
    # ─────────────────────────────────────────────────────────────

    def test_is_refusal_text_recognises_canonical_phrase(self):
        self.assertTrue(
            RAGHandler._is_refusal_text("I don't have enough context to answer that.")
        )

    def test_is_refusal_text_case_insensitive(self):
        self.assertTrue(
            RAGHandler._is_refusal_text("I DON'T HAVE ENOUGH CONTEXT to answer that.")
        )

    def test_is_refusal_text_recognises_documented_variants(self):
        for variant in (
            "Sorry, I don't have enough context for that question.",
            "Not enough context to answer.",
            "The context doesn't cover this topic.",
        ):
            self.assertTrue(RAGHandler._is_refusal_text(variant), variant)

    def test_is_refusal_text_rejects_confident_answer(self):
        self.assertFalse(
            RAGHandler._is_refusal_text("The store closes at 9pm.")
        )
        self.assertFalse(
            RAGHandler._is_refusal_text("30 days from delivery [#1].")
        )
        self.assertFalse(RAGHandler._is_refusal_text(""))


if __name__ == "__main__":
    unittest.main()

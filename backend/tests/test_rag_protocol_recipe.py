"""Tests for the rag-protocol recipe + its three playbooks (Arc R-1).

Covers:
  * Recipe is registered in the catalog and has the expected shape
    (rag-grounded adapter, rag_qa task profile, context/question/
    answer gold template).
  * Each of the three playbooks (POSITIVES_PARAPHRASE / REFUSALS /
    FORMAT_ROBUSTNESS) is registered for the rag-protocol recipe.
  * Paraphrase playbook: validator down-scores rows missing the
    citation marker; carries context through to the payload.
  * Refusals playbook: validator down-scores rows whose answer
    doesn't contain the canonical refusal phrase; both empty-context
    and off-topic-context rows are accepted.
  * Format playbook: validator down-scores rows whose answer drifts
    from the canonical gold answer (the whole point of the mode is
    output invariance).
  * Auto-RAG _RECIPE_TO_TEXT_KEYS now includes the rag-protocol
    corpus shape so the retrieval index can be built from rows of
    this recipe.
"""

from __future__ import annotations

import unittest

from app.services.recipe_service import list_recipes, get_recipe
from app.services.auto_rag_service import recommended_text_keys_for_recipe
from app.services.synth_playbooks import (
    SynthMode,
    get_playbook,
)
from app.services.synth_playbooks.rag_protocol_refusals import (
    REFUSAL_CANONICAL_PHRASE,
)


class RagProtocolRecipeTests(unittest.TestCase):

    # ─────────────────────────────────────────────────────────────
    # Recipe registration
    # ─────────────────────────────────────────────────────────────

    def test_recipe_is_registered_and_has_expected_shape(self):
        recipe = get_recipe("rag-protocol")
        self.assertIsNotNone(recipe, "rag-protocol recipe should be in the catalog")
        self.assertEqual(recipe.adapter_id, "rag-grounded")
        self.assertEqual(recipe.task_profile, "rag_qa")
        gold_field_names = [f.name for f in recipe.gold_template.fields]
        self.assertIn("context", gold_field_names)
        self.assertIn("question", gold_field_names)
        self.assertIn("answer", gold_field_names)
        # Example row carries a citation marker so the docs surface
        # what a 'right' answer looks like at a glance.
        self.assertIn("[#1]", recipe.gold_template.example_row["answer"])

    def test_recipe_shows_up_in_list_recipes(self):
        recipe_ids = {r.id for r in list_recipes()}
        self.assertIn("rag-protocol", recipe_ids)

    # ─────────────────────────────────────────────────────────────
    # Playbook registration
    # ─────────────────────────────────────────────────────────────

    def test_all_three_playbooks_are_registered(self):
        for mode in (
            SynthMode.POSITIVES_PARAPHRASE,
            SynthMode.REFUSALS,
            SynthMode.FORMAT_ROBUSTNESS,
        ):
            pb = get_playbook("rag-protocol", mode)
            self.assertIsNotNone(pb, f"playbook missing for rag-protocol/{mode.value}")
            self.assertEqual(pb.recipe_id, "rag-protocol")
            self.assertEqual(pb.mode, mode)

    # ─────────────────────────────────────────────────────────────
    # Paraphrase playbook — citation marker is the training signal
    # ─────────────────────────────────────────────────────────────

    def test_paraphrase_validator_downscores_when_citation_missing(self):
        pb = get_playbook("rag-protocol", SynthMode.POSITIVES_PARAPHRASE)
        parsed = [
            # Citation present → full confidence.
            {
                "context": "[#1] Refunds within 30 days of delivery.",
                "question": "How long can I return an item?",
                "answer": "30 days from delivery [#1].",
            },
            # Citation missing → down-scored (still accepted so
            # review-queue surfaces it for manual decision).
            {
                "context": "[#1] Refunds within 30 days of delivery.",
                "question": "When does the return window close?",
                "answer": "30 days from delivery.",
            },
        ]
        ctx = {"recipe_id": "rag-protocol", "gold_rows": [], "target_count": 4}
        accepted = pb.validate(parsed, ctx)
        self.assertEqual(len(accepted), 2)
        by_question = {r["payload"]["question"]: r for r in accepted}
        self.assertEqual(by_question["How long can I return an item?"]["synth_confidence"], 1.0)
        # Missing-citation row: 1.0 × 0.45 × known-answer-discount = ~0.34
        # (the known-answers set is empty here so no second discount).
        self.assertAlmostEqual(by_question["When does the return window close?"]["synth_confidence"], 0.45)

    def test_paraphrase_carries_context_through_to_payload(self):
        pb = get_playbook("rag-protocol", SynthMode.POSITIVES_PARAPHRASE)
        parsed = [
            {
                "context": "[#1] Free shipping on orders over $50.",
                "question": "When is shipping free?",
                "answer": "On orders over $50 [#1].",
            },
        ]
        accepted = pb.validate(parsed, {"recipe_id": "rag-protocol", "gold_rows": []})
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            accepted[0]["payload"]["context"],
            "[#1] Free shipping on orders over $50.",
        )

    def test_paraphrase_skips_malformed_rows(self):
        pb = get_playbook("rag-protocol", SynthMode.POSITIVES_PARAPHRASE)
        parsed = [
            {"context": "ctx", "question": "q"},  # missing answer
            {"context": "ctx", "question": "", "answer": "[#1] a"},  # empty question
            {"context": None, "question": "q", "answer": "[#1] a"},  # null context
        ]
        accepted = pb.validate(parsed, {"recipe_id": "rag-protocol", "gold_rows": []})
        self.assertEqual(len(accepted), 0)

    # ─────────────────────────────────────────────────────────────
    # Refusals playbook — canonical refusal phrase is the signal
    # ─────────────────────────────────────────────────────────────

    def test_refusal_validator_accepts_canonical_phrase(self):
        pb = get_playbook("rag-protocol", SynthMode.REFUSALS)
        parsed = [
            {
                "context": "[#1] Refunds within 30 days.",
                "question": "What time does the store close?",
                "answer": REFUSAL_CANONICAL_PHRASE,
            },
        ]
        accepted = pb.validate(parsed, {"recipe_id": "rag-protocol", "gold_rows": []})
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["synth_confidence"], 1.0)

    def test_refusal_validator_accepts_empty_context_as_no_context_flavour(self):
        pb = get_playbook("rag-protocol", SynthMode.REFUSALS)
        parsed = [
            {
                "context": "",  # no-context flavour — empty string OK
                "question": "What's your return policy?",
                "answer": "I don't have enough context to answer that.",
            },
        ]
        accepted = pb.validate(parsed, {"recipe_id": "rag-protocol", "gold_rows": []})
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["payload"]["context"], "")
        self.assertEqual(accepted[0]["synth_confidence"], 1.0)

    def test_refusal_validator_downscores_hallucination_in_refusal_costume(self):
        pb = get_playbook("rag-protocol", SynthMode.REFUSALS)
        parsed = [
            {
                "context": "[#1] Refunds within 30 days.",
                "question": "What time does the store close?",
                "answer": "Our store closes at 9 PM nightly.",  # hallucinated answer, no refusal
            },
        ]
        accepted = pb.validate(parsed, {"recipe_id": "rag-protocol", "gold_rows": []})
        self.assertEqual(len(accepted), 1)
        # 1.0 × 0.30 = 0.30
        self.assertAlmostEqual(accepted[0]["synth_confidence"], 0.30)

    def test_refusal_validator_rejects_overlong_answers(self):
        # Refusal answers should be short. A 1200+ char "refusal"
        # almost certainly means the model wrote a full hallucinated
        # response and the validator must reject it outright.
        pb = get_playbook("rag-protocol", SynthMode.REFUSALS)
        parsed = [
            {
                "context": "",
                "question": "Q?",
                "answer": "I don't have enough context to answer that. " + ("X" * 2000),
            },
        ]
        accepted = pb.validate(parsed, {"recipe_id": "rag-protocol", "gold_rows": []})
        self.assertEqual(len(accepted), 0)

    # ─────────────────────────────────────────────────────────────
    # Format playbook — output invariance
    # ─────────────────────────────────────────────────────────────

    def test_format_validator_downscores_when_answer_drifts_from_canonical(self):
        pb = get_playbook("rag-protocol", SynthMode.FORMAT_ROBUSTNESS)
        canonical_answer = "30 days from delivery [#1]."
        gold_rows = [
            {
                "context": "[#1] Refunds within 30 days of delivery.",
                "question": "How long do I have to return?",
                "answer": canonical_answer,
            },
        ]
        ctx = {"recipe_id": "rag-protocol", "gold_rows": gold_rows}
        parsed = [
            # Matches canonical → full confidence.
            {
                "context": "[#1] Refunds within 30 days of delivery.",
                "question": "Return window?",
                "answer": canonical_answer,
            },
            # Drifts from canonical → 0.55 discount.
            {
                "context": "[#1] Refunds within 30 days of delivery.",
                "question": "When does the return window close, please?",
                "answer": "You can return within 30 days [#1].",
            },
        ]
        accepted = pb.validate(parsed, ctx)
        self.assertEqual(len(accepted), 2)
        by_q = {r["payload"]["question"]: r for r in accepted}
        self.assertEqual(by_q["Return window?"]["synth_confidence"], 1.0)
        self.assertAlmostEqual(
            by_q["When does the return window close, please?"]["synth_confidence"], 0.55,
        )

    def test_format_validator_downscores_when_citation_missing(self):
        pb = get_playbook("rag-protocol", SynthMode.FORMAT_ROBUSTNESS)
        gold_rows = [
            {
                "context": "[#1] Refund policy 30 days.",
                "question": "How long is the return window?",
                "answer": "30 days from delivery [#1].",
            },
        ]
        parsed = [
            # Same canonical-matching answer but stripped of the citation.
            {
                "context": "[#1] Refund policy 30 days.",
                "question": "Return window length?",
                "answer": "30 days from delivery.",
            },
        ]
        accepted = pb.validate(parsed, {"recipe_id": "rag-protocol", "gold_rows": gold_rows})
        self.assertEqual(len(accepted), 1)
        # 1.0 × 0.55 (drift) × 0.45 (citation missing) = 0.2475
        self.assertAlmostEqual(accepted[0]["synth_confidence"], 0.2475, places=4)

    # ─────────────────────────────────────────────────────────────
    # Auto-RAG corpus shape — Stage 2 customers index the same row
    # shape the recipe trains on
    # ─────────────────────────────────────────────────────────────

    def test_auto_rag_corpus_keys_include_rag_protocol(self):
        keys = recommended_text_keys_for_recipe("rag-protocol")
        self.assertIsNotNone(keys)
        self.assertEqual(keys, ("context", "question", "answer"))

    # ─────────────────────────────────────────────────────────────
    # Prompt sanity — the LLM prompts mention the right protocol
    # elements (citation tokens, canonical refusal phrase, register
    # variation). Cheap canary against accidentally dropping the
    # protocol's defining signals from the prompt text.
    # ─────────────────────────────────────────────────────────────

    def test_prompts_mention_protocol_signals(self):
        gold = [
            {
                "context": "[#1] Free shipping over $50.",
                "question": "When is shipping free?",
                "answer": "On orders over $50 [#1].",
            },
        ]
        ctx = {"recipe_id": "rag-protocol", "gold_rows": gold, "target_count": 10}

        para_pb = get_playbook("rag-protocol", SynthMode.POSITIVES_PARAPHRASE)
        para_prompt = para_pb.build_prompt(ctx)
        self.assertIn("[#N]", para_prompt)
        self.assertIn("citation", para_prompt.lower())

        ref_pb = get_playbook("rag-protocol", SynthMode.REFUSALS)
        ref_prompt = ref_pb.build_prompt(ctx)
        self.assertIn("REFUSAL", ref_prompt.upper())
        self.assertIn(REFUSAL_CANONICAL_PHRASE, ref_prompt)

        fmt_pb = get_playbook("rag-protocol", SynthMode.FORMAT_ROBUSTNESS)
        fmt_prompt = fmt_pb.build_prompt(ctx)
        self.assertIn("register", fmt_prompt.lower())


if __name__ == "__main__":
    unittest.main()

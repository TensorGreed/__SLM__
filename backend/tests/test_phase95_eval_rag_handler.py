"""Phase 5.3.5 — RAGHandler.

Pins the contract for grounded-QA scoring:

- Dispatcher routes `rag_qa` / `rag` / `grounded_qa` task profiles
  to RAGHandler. Other profiles unaffected.
- Prompt template includes context + question + "answer using only
  the context" instruction.
- Three signal layers, all reported independently:
    1. Answer quality: SQuAD EM/F1 (today's QA path, preserved).
    2. Faithfulness: token-overlap of prediction with context.
    3. Context recall: token-overlap of gold answer with context
       (retriever-side diagnostic, NOT a model metric).
- Per-row enrichment lands faithfulness / context_recall /
  unsupported_rate / is_faithful on each prediction for the UI.
- Falls back gracefully when no context field is present (rows
  without context still get EM/F1 scored).
- Edge cases: empty prediction (trivially faithful — "I don't know"
  can't hallucinate), empty context (faithfulness = 0).
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
    GenericHandler,
    QAHandler,
    RAGHandler,
    resolve_task_handler,
)


def _ctx() -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile="rag_qa",
        handler_id="rag_qa",
        prepared_dir=Path("."),
        dataset_name="test",
        manifest={},
    )


class DispatcherRoutingTests(unittest.TestCase):
    def test_rag_qa_routes_to_rag_handler(self):
        self.assertIsInstance(resolve_task_handler("rag_qa"), RAGHandler)

    def test_rag_alias_routes_to_rag_handler(self):
        self.assertIsInstance(resolve_task_handler("rag"), RAGHandler)

    def test_grounded_qa_alias_routes_to_rag_handler(self):
        self.assertIsInstance(resolve_task_handler("grounded_qa"), RAGHandler)

    def test_other_profiles_unaffected(self):
        # QA still routes to QAHandler — adding RAGHandler must not
        # steal qa / instruction_sft.
        self.assertIsInstance(resolve_task_handler("qa"), QAHandler)
        self.assertIsInstance(resolve_task_handler(None), GenericHandler)


class PromptAssemblyTests(unittest.TestCase):
    def test_prompt_includes_context_question_and_instruction(self):
        h = RAGHandler()
        built = h.build_prompts(
            [
                {
                    "question": "What's the capital of France?",
                    "context": "Paris is the capital and largest city of France.",
                    "answer": "Paris",
                }
            ],
            _ctx(),
        )
        prompt = built[0].prompt
        self.assertIn("Answer the question using only the context", prompt)
        self.assertIn("Paris is the capital", prompt)
        self.assertIn("What's the capital of France?", prompt)
        self.assertTrue(prompt.rstrip().endswith("Answer:"))
        self.assertEqual(built[0].reference, "Paris")
        self.assertEqual(built[0].extras["rag_has_context"], True)
        self.assertEqual(
            built[0].extras["rag_context"],
            "Paris is the capital and largest city of France.",
        )

    def test_prompt_falls_back_to_plain_question_when_no_context(self):
        h = RAGHandler()
        built = h.build_prompts(
            [{"question": "What is 2+2?", "answer": "4"}], _ctx()
        )
        # No context → plain QA-style prompt (just the question).
        self.assertEqual(built[0].prompt, "What is 2+2?")
        self.assertEqual(built[0].extras["rag_has_context"], False)

    def test_alternative_context_field_names(self):
        h = RAGHandler()
        # "passage" should also work as a context source.
        built = h.build_prompts(
            [
                {
                    "question": "Q?",
                    "passage": "Some grounding passage.",
                    "answer": "A",
                }
            ],
            _ctx(),
        )
        self.assertIn("Some grounding passage.", built[0].prompt)


class GenerationOverrideTests(unittest.TestCase):
    def test_caps_at_256(self):
        # Grounded answers should be short — 256 is a hard upper bound.
        self.assertEqual(RAGHandler().max_new_tokens_override(1024), 256)

    def test_raises_tiny_default_to_floor(self):
        # An "I don't know" needs ~3 tokens; we still want a floor of
        # 64 so the model has room to explain.
        self.assertEqual(RAGHandler().max_new_tokens_override(8), 64)

    def test_passes_through_reasonable_default(self):
        self.assertEqual(RAGHandler().max_new_tokens_override(128), 128)


class FaithfulnessTests(unittest.TestCase):
    def _score(self, pred: str, ref: str, context: str | None) -> dict:
        predictions = [
            {
                "prediction": pred,
                "reference": ref,
                "rag_context": context or "",
                "rag_has_context": context is not None,
            }
        ]
        return RAGHandler().score(predictions, _ctx())

    def test_perfectly_grounded_answer_scores_one(self):
        out = self._score(
            pred="Paris",
            ref="Paris",
            context="Paris is the capital of France.",
        )
        self.assertEqual(out["faithfulness_score_mean"], 1.0)
        self.assertEqual(out["unsupported_token_rate_mean"], 0.0)
        self.assertEqual(out["faithfulness_rate"], 1.0)  # 1/1 above threshold

    def test_fully_extraneous_answer_scores_zero(self):
        out = self._score(
            pred="Madrid Tokyo London",
            ref="Paris",
            context="Paris is the capital of France.",
        )
        self.assertEqual(out["faithfulness_score_mean"], 0.0)
        self.assertEqual(out["unsupported_token_rate_mean"], 1.0)
        self.assertEqual(out["faithfulness_rate"], 0.0)
        # Marked NOT faithful at threshold.
        # (Inspect the row enrichment to confirm.)
        # The score's faithfulness_rate already captures this.

    def test_partially_grounded_answer_below_threshold(self):
        # The "London is the capital of France" case from the design
        # discussion: most tokens come from the context but a key
        # one ("London") doesn't. Faithfulness ~0.8 (over threshold)
        # but unsupported_rate ~0.2 catches the issue.
        out = self._score(
            pred="London is the capital of France",
            ref="Paris",
            context="Paris is the capital and largest city of France.",
        )
        # Most tokens grounded → faithfulness over 0.7 threshold.
        self.assertGreater(out["faithfulness_score_mean"], 0.7)
        # But unsupported_rate is non-zero — "London" isn't in context.
        self.assertGreater(out["unsupported_token_rate_mean"], 0.0)
        # And SQuAD EM/F1 catches it — answer is wrong.
        self.assertEqual(out["exact_match"], 0.0)

    def test_empty_prediction_trivially_faithful(self):
        # Convention: an empty/refusal answer can't hallucinate by
        # definition. Faithfulness defaults to 1.0 so refusal rows
        # don't drag the metric down. EM is still 0 (didn't match
        # the gold).
        out = self._score(pred="", ref="Paris", context="Paris is …")
        self.assertEqual(out["faithfulness_score_mean"], 1.0)
        self.assertEqual(out["exact_match"], 0.0)

    def test_empty_context_zeros_faithfulness(self):
        # Non-empty prediction with empty context → nothing in
        # context for tokens to ground against → 0.
        out = self._score(pred="some answer", ref="Paris", context="")
        # rag_has_context is False here → faithfulness not scored at all.
        self.assertEqual(out["faithfulness_score_mean"], 0.0)
        self.assertEqual(out["rows_with_context"], 0)


class ContextRecallTests(unittest.TestCase):
    """context_recall is a retriever-side diagnostic, NOT a model
    metric — it answers "did the retriever surface the answer's
    tokens in the context"."""

    def test_perfect_context_recall_when_gold_in_context(self):
        out = RAGHandler().score(
            [
                {
                    "prediction": "ignored",
                    "reference": "Paris",
                    "rag_context": "Paris is the capital of France.",
                    "rag_has_context": True,
                }
            ],
            _ctx(),
        )
        self.assertEqual(out["context_recall_mean"], 1.0)

    def test_zero_context_recall_when_gold_missing_from_context(self):
        out = RAGHandler().score(
            [
                {
                    "prediction": "anything",
                    "reference": "Tokyo",
                    "rag_context": "Paris is the capital of France.",
                    "rag_has_context": True,
                }
            ],
            _ctx(),
        )
        self.assertEqual(out["context_recall_mean"], 0.0)


class PerRowEnrichmentTests(unittest.TestCase):
    def test_row_fields_written_in_place(self):
        h = RAGHandler()
        row = {
            "prediction": "Paris",
            "reference": "Paris",
            "rag_context": "Paris is the capital of France.",
            "rag_has_context": True,
        }
        h.score([row], _ctx())
        self.assertEqual(row["rag_faithfulness"], 1.0)
        self.assertEqual(row["rag_context_recall"], 1.0)
        self.assertEqual(row["rag_unsupported_rate"], 0.0)
        self.assertTrue(row["rag_is_faithful"])
        self.assertEqual(row["row_exact_match"], 1.0)
        self.assertEqual(row["row_f1"], 1.0)

    def test_hallucinated_row_is_faithful_flag_false(self):
        h = RAGHandler()
        row = {
            "prediction": "completely unrelated answer",
            "reference": "Paris",
            "rag_context": "Paris is the capital of France.",
            "rag_has_context": True,
        }
        h.score([row], _ctx())
        # No prediction tokens overlap context → faithfulness 0.
        self.assertEqual(row["rag_faithfulness"], 0.0)
        self.assertFalse(row["rag_is_faithful"])

    def test_row_without_context_skips_rag_metrics(self):
        # Rows without a context still get EM/F1, but the RAG-specific
        # fields stay absent so the UI knows not to render the surface.
        h = RAGHandler()
        row = {
            "prediction": "Paris",
            "reference": "Paris",
            "rag_context": "",
            "rag_has_context": False,
        }
        h.score([row], _ctx())
        self.assertEqual(row["row_exact_match"], 1.0)
        self.assertNotIn("rag_faithfulness", row)


class MixedDatasetTests(unittest.TestCase):
    def test_mix_of_context_and_no_context_rows(self):
        # Some rows have context, others don't. Aggregate metrics
        # should reflect both signals correctly: EM/F1 over ALL rows,
        # faithfulness/context_recall only over rows-with-context.
        h = RAGHandler()
        predictions = [
            {
                "prediction": "Paris",
                "reference": "Paris",
                "rag_context": "Paris is the capital.",
                "rag_has_context": True,
            },
            {
                "prediction": "Paris",
                "reference": "Paris",
                "rag_context": "",
                "rag_has_context": False,
            },
        ]
        out = h.score(predictions, _ctx())
        # Both rows scored EM = 1.0
        self.assertEqual(out["exact_match"], 1.0)
        self.assertEqual(out["total"], 2)
        # Only one row had context — that's the denominator for the
        # RAG-specific metrics.
        self.assertEqual(out["rows_with_context"], 1)
        self.assertEqual(out["faithfulness_score_mean"], 1.0)

    def test_empty_predictions_returns_zeroed_metrics(self):
        out = RAGHandler().score([], _ctx())
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["faithfulness_rate"], 0.0)
        self.assertEqual(out["exact_match"], 0.0)


class EndToEndIntegrationTests(unittest.TestCase):
    def test_build_prompts_then_score_pipeline(self):
        # build_prompts → score round-trips correctly. The handler
        # writes context into BuiltPrompt.extras, which the load
        # path passes into the inference pair, which the score path
        # then reads via prediction.rag_context. This test mocks the
        # round-trip directly: take the BuiltPrompt's extras and use
        # them as the prediction's rag_* fields.
        h = RAGHandler()
        rows = [
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital and most populous city of France.",
                "answer": "Paris",
            }
        ]
        built = h.build_prompts(rows, _ctx())
        # Simulate a perfectly grounded model.
        predictions = [
            {
                "prediction": built[0].reference,
                "reference": built[0].reference,
                "rag_context": built[0].extras["rag_context"],
                "rag_has_context": built[0].extras["rag_has_context"],
            }
        ]
        out = h.score(predictions, _ctx())
        self.assertEqual(out["exact_match"], 1.0)
        self.assertEqual(out["faithfulness_score_mean"], 1.0)
        self.assertEqual(out["context_recall_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()

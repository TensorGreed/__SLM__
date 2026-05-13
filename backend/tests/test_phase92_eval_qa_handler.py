"""Phase 5.3.2 — QAHandler.

Pins every contract the plan made:

- Dispatcher routes ``qa`` / ``instruction_sft`` / ``chat_sft`` /
  ``language_modeling`` profiles to QAHandler. Other profiles still
  route to their own handlers (classification, seq2seq) or generic.
- CoT answer-span extractor pulls the conclusion out of common
  reasoning patterns: ``Final answer: X`` / ``Answer: X`` /
  ``Therefore: X`` / ``In conclusion: X`` / ``The answer is X``.
- When no marker matches, the full text is used (today's behavior
  preserved for plain Q/A).
- Metrics include ``exact_match``, ``f1``, ``answer_span_extracted_rate``,
  plus ``total`` and ``correct`` — gate compat for eval packs keyed
  on EM/F1 stays.
- Each prediction dict gets enriched in place with ``answer_span``,
  ``span_marker``, ``row_exact_match``, ``row_f1`` so the
  predictions_preview writer can flow them to the UI.
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
    ClassificationHandler,
    EvalContext,
    GenericHandler,
    QAHandler,
    Seq2SeqHandler,
    resolve_task_handler,
)


def _ctx() -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile="qa",
        handler_id="qa",
        prepared_dir=Path("."),
        dataset_name="test",
        manifest={},
    )


class DispatcherRoutingTests(unittest.TestCase):
    def test_qa_profile_routes_to_qa_handler(self):
        self.assertIsInstance(resolve_task_handler("qa"), QAHandler)

    def test_instruction_sft_routes_to_qa_handler(self):
        self.assertIsInstance(resolve_task_handler("instruction_sft"), QAHandler)

    def test_chat_sft_routes_to_qa_handler(self):
        self.assertIsInstance(resolve_task_handler("chat_sft"), QAHandler)

    def test_language_modeling_routes_to_qa_handler(self):
        self.assertIsInstance(resolve_task_handler("language_modeling"), QAHandler)

    def test_other_profiles_unaffected(self):
        # Sanity: registering QAHandler must not steal classification /
        # seq2seq / generic dispatches.
        self.assertIsInstance(
            resolve_task_handler("classification"), ClassificationHandler
        )
        self.assertIsInstance(resolve_task_handler("seq2seq"), Seq2SeqHandler)
        self.assertIsInstance(resolve_task_handler(None), GenericHandler)


class AnswerSpanExtractionTests(unittest.TestCase):
    def test_final_answer_marker(self):
        span, marker = QAHandler().extract_answer_span(
            "Step 1: think. Step 2: more thinking. Final answer: Paris."
        )
        self.assertEqual(span, "Paris")
        self.assertIsNotNone(marker)
        self.assertIn("final", marker)

    def test_plain_answer_marker(self):
        span, marker = QAHandler().extract_answer_span(
            "Reasoning blah blah. Answer: 42."
        )
        self.assertEqual(span, "42")
        self.assertIn("answer", marker)

    def test_therefore_marker(self):
        span, marker = QAHandler().extract_answer_span(
            "I think the calculation gives 7. Therefore: 7."
        )
        self.assertEqual(span, "7")
        self.assertIn("therefore", marker)

    def test_in_conclusion_marker(self):
        span, marker = QAHandler().extract_answer_span(
            "Premise A. Premise B. In conclusion: yes."
        )
        self.assertEqual(span, "yes")
        self.assertIn("conclusion", marker)

    def test_the_answer_is_marker(self):
        span, marker = QAHandler().extract_answer_span(
            "Working through the math... the answer is 88."
        )
        self.assertEqual(span, "88")
        self.assertIn("answer", marker)

    def test_no_marker_returns_original_text(self):
        text = "Just a plain direct answer with no reasoning markers"
        span, marker = QAHandler().extract_answer_span(text)
        self.assertEqual(span, text)
        self.assertIsNone(marker)

    def test_empty_input_safely_returns_empty(self):
        span, marker = QAHandler().extract_answer_span("")
        self.assertEqual(span, "")
        self.assertIsNone(marker)
        span, marker = QAHandler().extract_answer_span("   ")
        self.assertEqual(span.strip(), "")
        self.assertIsNone(marker)

    def test_marker_picks_last_occurrence(self):
        # "Final answer: X" appears twice — CoT often restates; we want
        # the conclusion, which is the LAST occurrence.
        span, _ = QAHandler().extract_answer_span(
            "I might say Final answer: maybe. But on reflection, "
            "Final answer: Paris."
        )
        self.assertEqual(span, "Paris")

    def test_case_insensitive(self):
        span, _ = QAHandler().extract_answer_span(
            "FINAL ANSWER: London"
        )
        self.assertEqual(span, "London")


class ScoringTests(unittest.TestCase):
    def test_plain_qa_rows_score_today_squad_em_f1(self):
        # Regression: rows without CoT markers should score identically
        # to the pre-handler SQuAD EM/F1 pipeline (Phase 5.2 behavior).
        out = QAHandler().score(
            [
                {"prediction": "Paris", "reference": "Paris"},
                {"prediction": "London", "reference": "Paris"},  # wrong
                {"prediction": "Paris.", "reference": "Paris"},  # punctuation
                {"prediction": "the Paris", "reference": "Paris"},  # article
            ],
            _ctx(),
        )
        # 3 of 4 are correct after SQuAD normalization (Paris == Paris ==
        # "Paris." == "the Paris" → all match the reference "Paris").
        self.assertEqual(out["total"], 4)
        self.assertEqual(out["correct"], 3)
        self.assertEqual(out["exact_match"], 0.75)

    def test_cot_predictions_get_extracted_before_scoring(self):
        # The bug this handler fixes: model emits reasoning + the answer.
        # Pre-handler, SQuAD F1 scored the paragraph against "Paris" and
        # got near-zero. With span extraction, the conclusion is scored
        # against the reference and matches.
        out = QAHandler().score(
            [
                {
                    "prediction": "Let me think... it's the city of light. Final answer: Paris.",
                    "reference": "Paris",
                },
            ],
            _ctx(),
        )
        self.assertEqual(out["exact_match"], 1.0)
        self.assertEqual(out["f1"], 1.0)

    def test_answer_span_extracted_rate_reports_fraction(self):
        out = QAHandler().score(
            [
                {"prediction": "Final answer: Paris", "reference": "Paris"},  # extracted
                {"prediction": "Paris", "reference": "Paris"},  # no marker, correct
                {"prediction": "Therefore: Paris", "reference": "Paris"},  # extracted
                {"prediction": "London", "reference": "Paris"},  # no marker, wrong
            ],
            _ctx(),
        )
        self.assertEqual(out["answer_span_extracted_rate"], 0.5)
        # And accuracy is 3/4 (the wrong "London" mispredicts).
        self.assertEqual(out["correct"], 3)

    def test_per_row_enrichment_written_in_place(self):
        # The frontend reads these off predictions_preview, so verify
        # the in-place mutation contract.
        predictions = [
            {
                "prediction": "Final answer: 42.",
                "reference": "42",
            },
            {
                "prediction": "London",
                "reference": "London",
            },
        ]
        QAHandler().score(predictions, _ctx())
        # First row: marker matched.
        self.assertEqual(predictions[0]["answer_span"], "42")
        self.assertIsNotNone(predictions[0]["span_marker"])
        self.assertEqual(predictions[0]["row_exact_match"], 1.0)
        self.assertEqual(predictions[0]["row_f1"], 1.0)
        # Second row: no marker, original text kept as span.
        self.assertEqual(predictions[1]["answer_span"], "London")
        self.assertIsNone(predictions[1]["span_marker"])
        self.assertEqual(predictions[1]["row_exact_match"], 1.0)

    def test_empty_predictions_returns_zeroed_metrics(self):
        out = QAHandler().score([], _ctx())
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["exact_match"], 0.0)
        self.assertEqual(out["f1"], 0.0)
        self.assertEqual(out["answer_span_extracted_rate"], 0.0)


class BuildPromptsTests(unittest.TestCase):
    def test_delegates_to_generic_field_extraction(self):
        # QAHandler intentionally doesn't wrap the prompt — Phase 5.2's
        # tokenizer chat_template handles that at inference time. The
        # build_prompts step just extracts {prompt, reference} per the
        # GenericHandler field-precedence rules.
        built = QAHandler().build_prompts(
            [{"question": "Capital of France?", "answer": "Paris"}], _ctx()
        )
        self.assertEqual(built[0].prompt, "Capital of France?")
        self.assertEqual(built[0].reference, "Paris")


class EndToEndIntegrationTests(unittest.TestCase):
    """build_prompts → mock-infer (with CoT) → score pipeline."""

    def test_cot_pipeline_scores_correctly(self):
        handler = QAHandler()
        rows = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "Capital of France?", "answer": "Paris"},
        ]
        built = handler.build_prompts(rows, _ctx())
        # Simulate a CoT model: reasoning + Final answer.
        predictions = [
            {
                "prediction": (
                    "Two plus two requires basic arithmetic. "
                    "Final answer: 4."
                ),
                "reference": "4",
            },
            {
                "prediction": (
                    "France's capital is the city famous for the Eiffel Tower. "
                    "Therefore: Paris."
                ),
                "reference": "Paris",
            },
        ]
        out = handler.score(predictions, _ctx())
        self.assertEqual(out["exact_match"], 1.0)
        self.assertEqual(out["f1"], 1.0)
        self.assertEqual(out["answer_span_extracted_rate"], 1.0)
        # And the per-row info landed for the UI.
        self.assertEqual(predictions[0]["answer_span"], "4")
        self.assertEqual(predictions[1]["answer_span"], "Paris")


if __name__ == "__main__":
    unittest.main()

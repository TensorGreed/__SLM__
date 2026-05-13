"""Phase 100 — handler stop sequences.

Untuned / partially-trained models often emit valid output followed
by worksheet-style continuations ("Exercise 3: …", "Question: …",
"Input: …") because they treat the prompt as a homework problem
with follow-ups. The screenshot from the PII demo showed exactly
this: model emitted JSON, then started "Exercise 3: Write a Python
script…".

The fix is per-handler stop-sequence lists that halt generation as
soon as the rambling pattern is detected. After fine-tuning these
stops are mostly inert (the model knows when to halt on its own),
but they're load-bearing for zero-shot / pre-training-complete
runs of the eval pipeline.

Pins the contract:

- Each handler exposes a stop_sequences(ctx) -> list[str] method.
- StructuredExtractionHandler includes "\\nExercise" (the screenshot
  failure mode), plus Input/Text/Question restart patterns and the
  closing code-fence pattern.
- QAHandler, RAGHandler, ClassificationHandler also expose
  appropriate stops for their prompt templates.
- The inference path's stop-trimming preserves output up to (but
  not including) the first stop sequence.
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
    QAHandler,
    RAGHandler,
    StructuredExtractionHandler,
)


def _ctx(profile: str = "structured_extraction") -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile=profile,
        handler_id=profile,
        prepared_dir=Path("."),
        dataset_name="test",
        manifest={},
    )


class StructuredExtractionStopsTests(unittest.TestCase):
    def test_includes_exercise_pattern_from_screenshot(self):
        # The literal failure mode from the user's screenshot.
        stops = StructuredExtractionHandler().stop_sequences(_ctx())
        self.assertIn("\nExercise", stops)
        self.assertIn("\n\nExercise", stops)

    def test_includes_input_and_text_restart_patterns(self):
        stops = StructuredExtractionHandler().stop_sequences(_ctx())
        self.assertIn("\nInput:", stops)
        self.assertIn("\nText:", stops)

    def test_includes_closing_code_fence(self):
        stops = StructuredExtractionHandler().stop_sequences(_ctx())
        # Models commonly wrap JSON in ```json … ``` and then keep
        # generating after the closing fence.
        self.assertIn("\n```\n", stops)


class QAHandlerStopsTests(unittest.TestCase):
    def test_qa_stops_on_question_restart(self):
        stops = QAHandler().stop_sequences(_ctx(profile="qa"))
        self.assertIn("\nQuestion:", stops)

    def test_qa_stops_on_input_restart(self):
        stops = QAHandler().stop_sequences(_ctx(profile="qa"))
        self.assertIn("\nInput:", stops)

    def test_qa_stops_on_exercise(self):
        # Same screenshot pattern — applies to any task-aware handler
        # because the rambling is model-driven not task-driven.
        stops = QAHandler().stop_sequences(_ctx(profile="qa"))
        self.assertIn("\nExercise", stops)


class ClassificationStopsTests(unittest.TestCase):
    def test_classification_stops_on_double_newline(self):
        # Labels are single tokens. Two newlines after the label = the
        # model emitted its answer and is starting a new block.
        stops = ClassificationHandler().stop_sequences(_ctx(profile="classification"))
        self.assertIn("\n\n", stops)

    def test_classification_stops_on_text_restart(self):
        stops = ClassificationHandler().stop_sequences(_ctx(profile="classification"))
        self.assertIn("\nText:", stops)


class RAGHandlerStopsTests(unittest.TestCase):
    def test_rag_stops_on_context_restart(self):
        # Model restarting the Context/Question/Answer pattern means
        # it's rambling into "here's another grounded QA example".
        stops = RAGHandler().stop_sequences(_ctx(profile="rag_qa"))
        self.assertIn("\nContext:", stops)
        self.assertIn("\nQuestion:", stops)


class StopTrimSimulationTests(unittest.TestCase):
    """The inference path trims the decoded prediction at the earliest
    stop-sequence occurrence. These tests simulate that logic so we
    pin the contract without needing a live model."""

    @staticmethod
    def _simulate_trim(text: str, stops: list[str]) -> str:
        earliest_cut = len(text)
        for stop_str in stops:
            idx = text.find(stop_str)
            if 0 <= idx < earliest_cut:
                earliest_cut = idx
        return text[:earliest_cut].rstrip()

    def test_trims_at_exercise_pattern(self):
        raw = (
            '{"entities": [{"type":"person_name","start":0,"end":8,"text":"Audrey"}]}'
            "\nExercise 3:\nWrite a Python script that..."
        )
        stops = StructuredExtractionHandler().stop_sequences(_ctx())
        trimmed = self._simulate_trim(raw, stops)
        self.assertIn('"entities"', trimmed)
        self.assertNotIn("Exercise", trimmed)
        self.assertNotIn("Python script", trimmed)

    def test_trims_at_question_restart(self):
        raw = "Paris.\nQuestion: What is the capital of Germany?"
        stops = QAHandler().stop_sequences(_ctx(profile="qa"))
        trimmed = self._simulate_trim(raw, stops)
        self.assertEqual(trimmed, "Paris.")

    def test_no_stops_in_text_leaves_full_output(self):
        raw = '{"entities": []}'
        stops = StructuredExtractionHandler().stop_sequences(_ctx())
        trimmed = self._simulate_trim(raw, stops)
        # No stop sequence present → output preserved verbatim.
        self.assertEqual(trimmed, raw)

    def test_earliest_stop_wins(self):
        # If multiple stops could match, the EARLIEST wins so we don't
        # accidentally preserve text that should have been cut.
        raw = "Answer here.\nQuestion: x\nExercise 3: y"
        stops = QAHandler().stop_sequences(_ctx(profile="qa"))
        trimmed = self._simulate_trim(raw, stops)
        self.assertEqual(trimmed, "Answer here.")


class HandlerStopReturnShapeTests(unittest.TestCase):
    """Every handler's stop_sequences returns a list of non-empty
    strings — the inference path filters out empties, but it's nicer
    if the handler never emits them."""

    def test_all_stops_are_non_empty_strings(self):
        for cls, profile in [
            (StructuredExtractionHandler, "structured_extraction"),
            (QAHandler, "qa"),
            (ClassificationHandler, "classification"),
            (RAGHandler, "rag_qa"),
        ]:
            stops = cls().stop_sequences(_ctx(profile=profile))
            self.assertIsInstance(stops, list, f"{cls.__name__}.stop_sequences not a list")
            for stop in stops:
                self.assertIsInstance(stop, str, f"{cls.__name__} emitted non-string stop")
                self.assertTrue(stop, f"{cls.__name__} emitted empty stop")


if __name__ == "__main__":
    unittest.main()

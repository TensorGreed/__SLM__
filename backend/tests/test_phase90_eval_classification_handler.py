"""Phase 5.3.1 — ClassificationHandler.

Pins every contract the plan made:

- Candidate set: read from manifest first, fall back to scanning the
  records, capped at 30 for in-prompt listing.
- Prompt template lists the labels when ≤ 30 and omits the list when
  there are more (still asks for a single-label reply).
- Label parser picks the earliest-position match, ties resolved by
  longest label so ``very_positive`` wins over ``positive`` when both
  start at index 0.
- Scoring produces classification-native metrics (accuracy, macro_f1,
  per_class P/R/F1, confusion_matrix, unparseable_rate) PLUS legacy
  ``exact_match`` / ``f1`` aliases for eval-pack gate compat.
- The dispatcher routes ``task_profile == "classification"`` to this
  handler, not GenericHandler.
- max_new_tokens hint caps generation so the model can't ramble.
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
    resolve_task_handler,
)


def _ctx(manifest: dict | None = None) -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="exact_match",
        task_profile="classification",
        handler_id="classification",
        prepared_dir=Path("."),
        dataset_name="test",
        manifest=manifest or {},
    )


class DispatcherRoutingTests(unittest.TestCase):
    def test_classification_profile_routes_to_classification_handler(self):
        handler = resolve_task_handler("classification")
        self.assertIsInstance(handler, ClassificationHandler)
        self.assertEqual(handler.profile_id, "classification")

    def test_other_profile_still_routes_to_generic(self):
        # Regression: ClassificationHandler registration must not steal
        # other profiles.
        self.assertIsInstance(resolve_task_handler("qa"), GenericHandler)
        self.assertIsInstance(resolve_task_handler("seq2seq"), GenericHandler)
        self.assertIsInstance(resolve_task_handler(None), GenericHandler)


class CandidateSetResolutionTests(unittest.TestCase):
    def test_uses_manifest_labels_when_present(self):
        handler = ClassificationHandler()
        ctx = _ctx({"labels": ["positive", "neutral", "negative"]})
        candidates = handler._resolve_candidates([], ctx)
        self.assertEqual(candidates, ["positive", "neutral", "negative"])

    def test_falls_back_to_scanning_records_when_no_manifest_labels(self):
        handler = ClassificationHandler()
        rows = [
            {"text": "great!", "label": "positive"},
            {"text": "fine", "label": "neutral"},
            {"text": "awful", "label": "negative"},
            {"text": "amazing!", "label": "positive"},
        ]
        candidates = handler._resolve_candidates(rows, _ctx())
        # Sorted alphabetically when derived from data (manifest order is
        # the user's intent; data-derived order is arbitrary so we pick
        # something stable).
        self.assertEqual(candidates, ["negative", "neutral", "positive"])

    def test_dedupes_label_list(self):
        handler = ClassificationHandler()
        ctx = _ctx({"labels": ["positive", "Positive", "positive", "neutral"]})
        candidates = handler._resolve_candidates([], ctx)
        # Exact string dedup — manifest labels are not lowercased before
        # comparison since the user's casing is meaningful.
        self.assertEqual(candidates, ["positive", "Positive", "neutral"])

    def test_max_candidate_set_caps_at_200(self):
        handler = ClassificationHandler()
        big_set = [f"intent_{i:03d}" for i in range(500)]
        ctx = _ctx({"labels": big_set})
        candidates = handler._resolve_candidates([], ctx)
        self.assertEqual(len(candidates), 200)

    def test_score_path_can_resolve_candidates_from_predictions(self):
        # When `score` is called without a prior build_prompts call (the
        # direct /api/evaluation/run path), candidates have to come from
        # the predictions[*].reference fields.
        handler = ClassificationHandler()
        predictions = [
            {"prediction": "positive", "reference": "positive"},
            {"prediction": "negative", "reference": "negative"},
        ]
        out = handler.score(predictions, _ctx())
        self.assertEqual(sorted(out["candidate_set"]), ["negative", "positive"])


class PromptAssemblyTests(unittest.TestCase):
    def test_prompt_lists_labels_when_at_most_30(self):
        handler = ClassificationHandler()
        ctx = _ctx({"labels": ["positive", "neutral", "negative"]})
        built = handler.build_prompts(
            [{"text": "Best headphones ever", "label": "positive"}], ctx
        )
        self.assertEqual(len(built), 1)
        prompt = built[0].prompt
        self.assertIn("Classify", prompt)
        self.assertIn("positive, neutral, negative", prompt)
        self.assertIn("Best headphones ever", prompt)
        self.assertTrue(prompt.endswith("Label:"))
        # Reference is the gold label, not the wrapped prompt.
        self.assertEqual(built[0].reference, "positive")

    def test_prompt_omits_label_list_when_more_than_30(self):
        handler = ClassificationHandler()
        labels = [f"intent_{i:02d}" for i in range(40)]
        ctx = _ctx({"labels": labels})
        built = handler.build_prompts(
            [{"text": "I want a refund", "label": "intent_05"}], ctx
        )
        prompt = built[0].prompt
        # No comma-separated list of 40 labels.
        self.assertNotIn("intent_00, intent_01", prompt)
        self.assertIn("just the class label", prompt)
        # Still asks for a label with the standard ending.
        self.assertTrue(prompt.endswith("Label:"))

    def test_extras_carry_input_and_candidates(self):
        handler = ClassificationHandler()
        ctx = _ctx({"labels": ["a", "b"]})
        built = handler.build_prompts([{"text": "hello", "label": "a"}], ctx)
        self.assertEqual(built[0].extras["classification_input"], "hello")
        self.assertEqual(built[0].extras["classification_candidates"], ["a", "b"])


class LabelParserTests(unittest.TestCase):
    def test_earliest_position_wins(self):
        handler = ClassificationHandler()
        out = handler.parse_predicted_label(
            "I think positive but maybe negative",
            ["negative", "positive"],
        )
        self.assertEqual(out, "positive")

    def test_tie_at_same_position_resolved_by_longest_label(self):
        handler = ClassificationHandler()
        # "very_positive" and "positive" both start at index 0 of
        # "very_positive sentiment" because find() locates the substring
        # — but very_positive at 0, positive at 5. So strictly speaking
        # they're not at the same position. Construct a real tie:
        out = handler.parse_predicted_label(
            "positive_one and stuff",
            ["positive", "positive_one"],
        )
        # Both labels start at index 0; longest wins.
        self.assertEqual(out, "positive_one")

    def test_no_match_returns_none(self):
        handler = ClassificationHandler()
        out = handler.parse_predicted_label(
            "I have no opinion about this product",
            ["positive", "negative", "neutral"],
        )
        self.assertIsNone(out)

    def test_case_insensitive_match(self):
        handler = ClassificationHandler()
        out = handler.parse_predicted_label("POSITIVE", ["positive"])
        self.assertEqual(out, "positive")

    def test_empty_output_returns_none(self):
        handler = ClassificationHandler()
        self.assertIsNone(handler.parse_predicted_label("", ["positive"]))
        self.assertIsNone(handler.parse_predicted_label("   ", ["positive"]))

    def test_empty_candidate_set_returns_none(self):
        handler = ClassificationHandler()
        self.assertIsNone(handler.parse_predicted_label("anything", []))


class ScoringTests(unittest.TestCase):
    def _score(self, predictions, manifest=None):
        return ClassificationHandler().score(predictions, _ctx(manifest))

    def test_perfect_predictions(self):
        out = self._score(
            [
                {"prediction": "positive", "reference": "positive"},
                {"prediction": "neutral", "reference": "neutral"},
                {"prediction": "negative", "reference": "negative"},
            ],
            manifest={"labels": ["positive", "neutral", "negative"]},
        )
        self.assertEqual(out["accuracy"], 1.0)
        self.assertEqual(out["macro_f1"], 1.0)
        # Legacy aliases for gate compat.
        self.assertEqual(out["exact_match"], 1.0)
        self.assertEqual(out["f1"], 1.0)
        self.assertEqual(out["unparseable"], 0)
        self.assertEqual(out["total"], 3)
        self.assertEqual(out["correct"], 3)

    def test_partially_correct_predictions(self):
        out = self._score(
            [
                {"prediction": "positive", "reference": "positive"},
                {"prediction": "positive", "reference": "negative"},  # wrong
                {"prediction": "neutral", "reference": "neutral"},
                {"prediction": "neutral", "reference": "negative"},  # wrong
            ],
            manifest={"labels": ["positive", "neutral", "negative"]},
        )
        self.assertEqual(out["accuracy"], 0.5)
        # negative class never predicted → recall=0, f1=0
        self.assertEqual(out["per_class"]["negative"]["recall"], 0.0)
        self.assertEqual(out["per_class"]["negative"]["f1"], 0.0)
        # positive has 1 TP + 1 FP → precision=0.5, recall=1.0, f1≈0.667
        positive = out["per_class"]["positive"]
        self.assertAlmostEqual(positive["precision"], 0.5)
        self.assertAlmostEqual(positive["recall"], 1.0)
        self.assertAlmostEqual(positive["f1"], 0.6667, places=3)

    def test_unparseable_predictions_counted(self):
        out = self._score(
            [
                {"prediction": "positive", "reference": "positive"},
                {"prediction": "I have no idea", "reference": "negative"},
                {"prediction": "", "reference": "neutral"},
            ],
            manifest={"labels": ["positive", "neutral", "negative"]},
        )
        self.assertEqual(out["unparseable"], 2)
        self.assertEqual(out["unparseable_rate"], round(2 / 3, 4))
        # accuracy still computed over all rows (unparseable counts as wrong).
        self.assertEqual(out["correct"], 1)

    def test_long_rambling_output_parses_when_label_appears_in_it(self):
        # The actual screenshot bug: model rambles but the right label
        # is in there somewhere. The handler should still extract it.
        out = self._score(
            [
                {
                    "prediction": (
                        "The battery life is amazing. The sound quality is "
                        "superb. Overall, positive review."
                    ),
                    "reference": "positive",
                },
            ],
            manifest={"labels": ["positive", "neutral", "negative"]},
        )
        self.assertEqual(out["accuracy"], 1.0)
        self.assertEqual(out["unparseable"], 0)

    def test_confusion_matrix_present_when_few_classes(self):
        out = self._score(
            [
                {"prediction": "positive", "reference": "positive"},
                {"prediction": "negative", "reference": "positive"},
                {"prediction": "neutral", "reference": "neutral"},
            ],
            manifest={"labels": ["positive", "neutral", "negative"]},
        )
        cm = out["confusion_matrix"]
        self.assertEqual(cm["positive"]["positive"], 1)
        self.assertEqual(cm["positive"]["negative"], 1)
        self.assertEqual(cm["neutral"]["neutral"], 1)
        # No __unparseable__ rows here.
        self.assertEqual(cm["positive"]["__unparseable__"], 0)

    def test_confusion_matrix_omitted_when_many_classes(self):
        labels = [f"intent_{i:02d}" for i in range(25)]
        predictions = [
            {"prediction": l, "reference": l} for l in labels[:5]
        ]
        out = ClassificationHandler().score(
            predictions, _ctx({"labels": labels})
        )
        # > CONFUSION_MATRIX_CAP (20) → no confusion matrix.
        self.assertEqual(out["confusion_matrix"], {})

    def test_empty_predictions_returns_zeroed_metrics(self):
        out = self._score([], manifest={"labels": ["a", "b"]})
        self.assertEqual(out["accuracy"], 0.0)
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["unparseable_rate"], 0.0)


class GenerationOverrideTests(unittest.TestCase):
    def test_caps_at_handler_max(self):
        handler = ClassificationHandler()
        # Caller asks for 128 (today's default) — handler caps to 16.
        self.assertEqual(handler.max_new_tokens_override(128), 16)

    def test_passes_through_smaller_values(self):
        # If the caller is already conservative, don't push them up.
        self.assertEqual(ClassificationHandler().max_new_tokens_override(4), 4)

    def test_handles_zero_and_negative_inputs(self):
        # Defensive against bad caller values.
        self.assertEqual(ClassificationHandler().max_new_tokens_override(0), 1)
        self.assertEqual(ClassificationHandler().max_new_tokens_override(-5), 1)


class EndToEndIntegrationTests(unittest.TestCase):
    """Smoke-tests the build_prompts → mock-infer → score pipeline so a
    handler regression in any single step shows up here."""

    def test_full_pipeline_with_perfect_model(self):
        handler = ClassificationHandler()
        ctx = _ctx({"labels": ["positive", "neutral", "negative"]})
        rows = [
            {"text": "Great product, loving it", "label": "positive"},
            {"text": "It's okay I guess", "label": "neutral"},
            {"text": "Total waste of money", "label": "negative"},
        ]
        built = handler.build_prompts(rows, ctx)
        # Simulate a perfect model: emits exactly the gold label.
        predictions = [
            {"prediction": bp.reference, "reference": bp.reference}
            for bp in built
        ]
        out = handler.score(predictions, ctx)
        self.assertEqual(out["accuracy"], 1.0)
        self.assertEqual(out["macro_f1"], 1.0)
        self.assertEqual(out["correct"], 3)
        self.assertEqual(out["unparseable"], 0)

    def test_full_pipeline_with_rambling_model(self):
        # This is the screenshot scenario from Phase 5.3.1's motivation:
        # the model emits a product-review sentence but the right label
        # word is still in it. With the new handler, scoring works.
        handler = ClassificationHandler()
        ctx = _ctx({"labels": ["positive", "neutral", "negative"]})
        rows = [{"text": "Best headphones ever", "label": "positive"}]
        built = handler.build_prompts(rows, ctx)
        self.assertEqual(len(built), 1)
        predictions = [
            {
                "prediction": (
                    "Best headphones ever. Overall this is a very positive "
                    "review of the product."
                ),
                "reference": "positive",
            }
        ]
        out = handler.score(predictions, ctx)
        self.assertEqual(out["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

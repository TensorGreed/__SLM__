"""Phase 5.3.6 — AlignmentHandler (DPO / ORPO / preference).

Pins the contract:

- Dispatcher routes `dpo` / `orpo` / `alignment` / `preference`
  profiles to AlignmentHandler. Other profiles unaffected.
- Row shape requires `prompt`, `chosen`, `rejected`. Reference =
  chosen (legacy SQuAD EM/F1 against the preferred completion).
- Scoring: similarity-to-chosen vs similarity-to-rejected via F1.
  Row is "preference correct" when chosen-sim > rejected-sim.
- Metrics: preference_accuracy, mean_alignment_margin,
  chosen_alignment_mean, rejected_alignment_mean, exact_match
  + f1 (legacy gate aliases against chosen).
- Per-row enrichment carries chosen/rejected sims, margin, and
  preference_correct flag for the UI's badge + disclosure.
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
    AlignmentHandler,
    EvalContext,
    GenericHandler,
    resolve_task_handler,
)


def _ctx() -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile="dpo",
        handler_id="alignment",
        prepared_dir=Path("."),
        dataset_name="test",
        manifest={},
    )


class DispatcherRoutingTests(unittest.TestCase):
    def test_dpo_routes_to_alignment_handler(self):
        self.assertIsInstance(resolve_task_handler("dpo"), AlignmentHandler)

    def test_orpo_routes_to_alignment_handler(self):
        self.assertIsInstance(resolve_task_handler("orpo"), AlignmentHandler)

    def test_alignment_alias_routes_to_alignment_handler(self):
        self.assertIsInstance(resolve_task_handler("alignment"), AlignmentHandler)

    def test_preference_alias_routes_to_alignment_handler(self):
        self.assertIsInstance(resolve_task_handler("preference"), AlignmentHandler)

    def test_other_profiles_unaffected(self):
        self.assertIsInstance(resolve_task_handler(None), GenericHandler)
        self.assertIsInstance(resolve_task_handler("unknown_profile"), GenericHandler)


class PromptAssemblyTests(unittest.TestCase):
    def test_prompt_is_the_input_reference_is_chosen(self):
        h = AlignmentHandler()
        built = h.build_prompts(
            [
                {
                    "prompt": "Explain DPO.",
                    "chosen": "DPO is direct preference optimization.",
                    "rejected": "DPO is a programming language.",
                }
            ],
            _ctx(),
        )
        self.assertEqual(built[0].prompt, "Explain DPO.")
        # Reference = chosen so the legacy EM/F1 gate compat path
        # scores against the preferred completion.
        self.assertEqual(built[0].reference, "DPO is direct preference optimization.")
        self.assertTrue(built[0].extras["alignment_has_pair"])

    def test_alternative_field_names(self):
        h = AlignmentHandler()
        built = h.build_prompts(
            [
                {
                    "question": "Q?",
                    "preferred": "good answer",
                    "dispreferred": "bad answer",
                }
            ],
            _ctx(),
        )
        self.assertEqual(built[0].prompt, "Q?")
        self.assertEqual(built[0].extras["alignment_chosen"], "good answer")
        self.assertEqual(built[0].extras["alignment_rejected"], "bad answer")

    def test_row_without_pair_flagged(self):
        h = AlignmentHandler()
        # Missing rejected — handler should still build the prompt
        # but mark has_pair = False so score doesn't try to compute
        # preference accuracy.
        built = h.build_prompts(
            [{"prompt": "Q?", "chosen": "good"}], _ctx()
        )
        self.assertFalse(built[0].extras["alignment_has_pair"])


class GenerationOverrideTests(unittest.TestCase):
    def test_caps_at_256(self):
        self.assertEqual(AlignmentHandler().max_new_tokens_override(1024), 256)

    def test_raises_tiny_default_to_floor(self):
        self.assertEqual(AlignmentHandler().max_new_tokens_override(8), 64)

    def test_passes_through_reasonable_default(self):
        self.assertEqual(AlignmentHandler().max_new_tokens_override(128), 128)


class ScoringTests(unittest.TestCase):
    def _score(self, predictions: list[dict]) -> dict:
        return AlignmentHandler().score(predictions, _ctx())

    def test_model_prefers_chosen_marked_correct(self):
        out = self._score(
            [
                {
                    "prediction": "DPO is direct preference optimization, a method for aligning models from preference pairs.",
                    "reference": "DPO is direct preference optimization.",
                    "alignment_chosen": "DPO is direct preference optimization.",
                    "alignment_rejected": "DPO is a programming language.",
                    "alignment_has_pair": True,
                }
            ]
        )
        self.assertEqual(out["preference_accuracy"], 1.0)
        self.assertGreater(out["mean_alignment_margin"], 0)
        self.assertGreater(out["chosen_alignment_mean"], out["rejected_alignment_mean"])

    def test_model_prefers_rejected_marked_wrong(self):
        out = self._score(
            [
                {
                    "prediction": "DPO is a programming language.",
                    "reference": "DPO is direct preference optimization.",
                    "alignment_chosen": "DPO is direct preference optimization.",
                    "alignment_rejected": "DPO is a programming language.",
                    "alignment_has_pair": True,
                }
            ]
        )
        self.assertEqual(out["preference_accuracy"], 0.0)
        # Negative margin = model preferred the rejected.
        self.assertLess(out["mean_alignment_margin"], 0)

    def test_per_row_enrichment_lands(self):
        h = AlignmentHandler()
        row = {
            "prediction": "DPO is direct preference optimization.",
            "reference": "DPO is direct preference optimization.",
            "alignment_chosen": "DPO is direct preference optimization.",
            "alignment_rejected": "DPO is a programming language.",
            "alignment_has_pair": True,
        }
        h.score([row], _ctx())
        self.assertTrue(row["alignment_preference_correct"])
        self.assertEqual(row["alignment_chosen_sim"], 1.0)
        self.assertLess(row["alignment_rejected_sim"], 1.0)
        self.assertEqual(
            row["alignment_margin"],
            round(row["alignment_chosen_sim"] - row["alignment_rejected_sim"], 4),
        )

    def test_row_without_pair_skips_preference_metrics(self):
        # No alignment_rejected → handler can't compute preference.
        # EM/F1 (legacy) still scored against the chosen reference.
        h = AlignmentHandler()
        row = {
            "prediction": "DPO is direct preference optimization.",
            "reference": "DPO is direct preference optimization.",
            "alignment_chosen": "DPO is direct preference optimization.",
            "alignment_has_pair": False,
        }
        h.score([row], _ctx())
        # EM landed (gate compat).
        self.assertEqual(row["row_exact_match"], 1.0)
        # Preference metrics absent — UI knows not to render them.
        self.assertNotIn("alignment_preference_correct", row)

    def test_aggregate_over_mixed_correctness(self):
        out = self._score(
            [
                # Row 1: model prefers chosen.
                {
                    "prediction": "good answer here",
                    "reference": "good answer here",
                    "alignment_chosen": "good answer here",
                    "alignment_rejected": "bad answer there",
                    "alignment_has_pair": True,
                },
                # Row 2: model prefers rejected.
                {
                    "prediction": "bad answer there",
                    "reference": "good answer here",
                    "alignment_chosen": "good answer here",
                    "alignment_rejected": "bad answer there",
                    "alignment_has_pair": True,
                },
            ]
        )
        self.assertEqual(out["preference_accuracy"], 0.5)
        self.assertEqual(out["correct"], 1)
        self.assertEqual(out["rows_with_pair"], 2)

    def test_empty_predictions_returns_zeroed_metrics(self):
        out = AlignmentHandler().score([], _ctx())
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["preference_accuracy"], 0.0)
        self.assertEqual(out["mean_alignment_margin"], 0.0)


class EndToEndIntegrationTests(unittest.TestCase):
    def test_build_then_score_pipeline(self):
        h = AlignmentHandler()
        rows = [
            {
                "prompt": "Explain DPO.",
                "chosen": "DPO is direct preference optimization.",
                "rejected": "DPO is a programming language.",
            }
        ]
        built = h.build_prompts(rows, _ctx())
        # Simulate a well-aligned model: emits the chosen verbatim.
        predictions = [
            {
                "prediction": built[0].reference,
                "reference": built[0].reference,
                "alignment_chosen": built[0].extras["alignment_chosen"],
                "alignment_rejected": built[0].extras["alignment_rejected"],
                "alignment_has_pair": built[0].extras["alignment_has_pair"],
            }
        ]
        out = h.score(predictions, _ctx())
        self.assertEqual(out["preference_accuracy"], 1.0)
        self.assertEqual(out["exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()

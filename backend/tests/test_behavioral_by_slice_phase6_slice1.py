"""Quality-Lift phase 6 slice 1 — Per-slice behavioral test scoring.

Pins (slice 1: runner accepts slice_definitions + emits per_slice
block per test; flattener extension + ScorecardPanel surfacing land
in slices 2 and 3):

  Backward compatibility:
    * No slice_definitions → identical output to phase 5 slice 2
      (no ``per_slice`` key on any test).
    * Empty slice_definitions list → same.
    * Empty behavioral_tests list → identical {}.

  INV per-slice:
    * Seed examples bucketed by slice predicates (long_input vs
      short_input) → each slice scores trials whose seed_index
      matches independently.
    * Slices can overlap (same seed matches multiple predicates) —
      each bucket scores independently without cross-contamination.
    * Slice membership anchored to the SEED EXAMPLE, NOT the perturbed
      input (e.g. typo perturbation can't move a row across slices).
    * Slice with no matching seed examples emits total=0 rather than
      being omitted from per_slice (consumer needs to see "matched
      zero rows" vs "wasn't evaluated").

  DIR per-slice:
    * Same bucketing semantics as INV; uses _score_dir_test so the
      expectation kind (must_change / must_change_to /
      must_change_to_one_of) carries through.

  MFT per-slice:
    * Examples bucketed by slice predicates; per-slice pass rates
      computed independently.

  Failed-examples cap:
    * Per-slice failed_examples capped at PER_SLICE_FAILED_EXAMPLES_CAP
      to keep EvalResult.metrics JSON bounded when many slices fail.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.behavioral_test_runner import (  # noqa: E402
    PER_SLICE_FAILED_EXAMPLES_CAP,
    run_behavioral_tests,
)


# Sample slice definitions: same shape phase 2 emits + the runner
# eats. Closed clause grammar: field/op/value.
LONG_INPUT_SLICE = {
    "slice_id": "long_input",
    "display_name": "Long inputs (>=10 chars)",
    "where": [{"field": "input_length", "op": "gte", "value": 10}],
}
SHORT_INPUT_SLICE = {
    "slice_id": "short_input",
    "display_name": "Short inputs (<10 chars)",
    "where": [{"field": "input_length", "op": "lt", "value": 10}],
}
EMPTY_MATCH_SLICE = {
    "slice_id": "matches_nothing",
    "display_name": "Synthetic placeholder",
    "where": [{"field": "input", "op": "eq", "value": "this_string_never_appears"}],
}


# ────────────────────────────────────────────────────────────────────────
# Backward compatibility
# ────────────────────────────────────────────────────────────────────────


class BackwardCompatTests(unittest.TestCase):

    def test_no_slice_definitions_argument_is_unchanged(self):
        # Pre-phase-6 caller signature; result must be identical to
        # what slice 2 emitted (no per_slice key).
        test = {
            "test_id": "no_slices",
            "kind": "MFT",
            "examples": [
                {"input": "I love it", "expected_label": "positive"},
                {"input": "Worst",     "expected_label": "negative"},
            ],
        }
        predict_fn = lambda texts: ["positive", "negative"]
        result = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertNotIn("per_slice", result["no_slices"])

    def test_empty_slice_list_treated_as_no_slicing(self):
        test = {
            "test_id": "empty_slices",
            "kind": "MFT",
            "examples": [{"input": "x", "expected_label": "A"}],
        }
        predict_fn = lambda texts: ["A"]
        result = run_behavioral_tests(
            [test], predict_fn=predict_fn, slice_definitions=[],
        )
        self.assertNotIn("per_slice", result["empty_slices"])

    def test_none_slice_list_treated_as_no_slicing(self):
        test = {
            "test_id": "none_slices",
            "kind": "MFT",
            "examples": [{"input": "x", "expected_label": "A"}],
        }
        predict_fn = lambda texts: ["A"]
        result = run_behavioral_tests(
            [test], predict_fn=predict_fn, slice_definitions=None,
        )
        self.assertNotIn("per_slice", result["none_slices"])


# ────────────────────────────────────────────────────────────────────────
# INV per-slice scoring
# ────────────────────────────────────────────────────────────────────────


class InvariantPerSliceTests(unittest.TestCase):

    def test_inv_buckets_seeds_by_slice_predicate(self):
        # 3 seeds: one long, two short. INV expectation = same_label.
        # We rig predict_fn so the long seed FAILS (perturbed flips)
        # and the two short seeds PASS. Top-level pass_rate = 2/3;
        # per-slice should show long_input = 0/1 fail, short_input
        # = 2/2 pass.
        test = {
            "test_id": "typo_invariance",
            "kind": "INV",
            "seed_examples": [
                {"input": "this is a long input row", "given_label": "positive"},  # len > 10
                {"input": "ok",                       "given_label": "positive"},  # len < 10
                {"input": "fine",                     "given_label": "positive"},  # len < 10
            ],
            "perturbations": [{"kind": "typo", "intensity": 0.05}],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "same_label"},
        }

        # predict_fn call sequence:
        #   1st call: originals → all "positive"
        #   2nd call: perturbed → ["negative", "positive", "positive"]
        # The first perturbed prediction (long seed) flips; the others stay.
        call_n = {"i": 0}

        def predict_fn(texts):
            call_n["i"] += 1
            if call_n["i"] == 1:
                return ["positive"] * len(texts)
            return ["negative", "positive", "positive"]

        result = run_behavioral_tests(
            [test], predict_fn=predict_fn,
            slice_definitions=[LONG_INPUT_SLICE, SHORT_INPUT_SLICE],
        )
        top = result["typo_invariance"]
        # Top-level: 2 of 3 pass.
        self.assertEqual(top["passed"], 2)
        self.assertEqual(top["total"], 3)

        self.assertIn("per_slice", top)
        long_block = top["per_slice"]["long_input"]
        short_block = top["per_slice"]["short_input"]
        # long_input has 1 trial (the long seed), failed → 0/1.
        self.assertEqual(long_block["passed"], 0)
        self.assertEqual(long_block["total"], 1)
        self.assertEqual(long_block["pass_rate"], 0.0)
        # short_input has 2 trials, both pass → 2/2 = 1.0.
        self.assertEqual(short_block["passed"], 2)
        self.assertEqual(short_block["total"], 2)
        self.assertEqual(short_block["pass_rate"], 1.0)

    def test_overlapping_slices_score_independently(self):
        # Define two slices that both include the same seed (a long
        # input that's ALSO labeled positive). Each bucket scores
        # the seed independently — failure in one does NOT contaminate
        # the other.
        positive_slice = {
            "slice_id": "given_positive",
            "where": [{"field": "given_label", "op": "eq", "value": "positive"}],
        }
        test = {
            "test_id": "double_membership",
            "kind": "INV",
            "seed_examples": [
                {"input": "this is a long input row", "given_label": "positive"},
            ],
            "perturbations": [{"kind": "typo", "intensity": 0.05}],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "same_label"},
        }
        call_n = {"i": 0}

        def predict_fn(texts):
            call_n["i"] += 1
            if call_n["i"] == 1:
                return ["positive"]
            return ["negative"]  # invariance violated

        result = run_behavioral_tests(
            [test], predict_fn=predict_fn,
            slice_definitions=[LONG_INPUT_SLICE, positive_slice],
        )
        # Both slices contain the seed; both score the same fail.
        long_block = result["double_membership"]["per_slice"]["long_input"]
        positive_block = result["double_membership"]["per_slice"]["given_positive"]
        self.assertEqual(long_block["total"], 1)
        self.assertEqual(long_block["passed"], 0)
        self.assertEqual(positive_block["total"], 1)
        self.assertEqual(positive_block["passed"], 0)

    def test_slice_with_no_matching_seeds_emits_zero_total(self):
        # Slice predicate matches no seed examples. The per_slice
        # entry MUST still appear with total=0 so the consumer can
        # surface "this slice matched no rows" instead of silently
        # dropping it.
        test = {
            "test_id": "empty_slice_test",
            "kind": "INV",
            "seed_examples": [
                {"input": "row a", "given_label": "positive"},
                {"input": "row b", "given_label": "positive"},
            ],
            "perturbations": [{"kind": "typo", "intensity": 0.05}],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "same_label"},
        }
        predict_fn = lambda texts: ["positive"] * len(texts)
        result = run_behavioral_tests(
            [test], predict_fn=predict_fn,
            slice_definitions=[EMPTY_MATCH_SLICE],
        )
        empty_block = result["empty_slice_test"]["per_slice"]["matches_nothing"]
        self.assertEqual(empty_block["total"], 0)
        self.assertEqual(empty_block["passed"], 0)
        self.assertEqual(empty_block["failed_examples"], [])

    def test_slice_membership_anchored_to_original_input(self):
        # A typo perturbation could shift input_length by ±1; if slice
        # membership were computed on the PERTURBED input, a borderline
        # seed could silently flip slices and per-slice math would
        # become incoherent. Verify by checking that a seed at length
        # exactly 10 (just over the long_input gte=10 boundary) stays
        # in long_input regardless of typo char-swaps.
        test = {
            "test_id": "borderline_seed",
            "kind": "INV",
            "seed_examples": [
                # input_length = 10 exactly; lt=10 short_input misses,
                # gte=10 long_input matches.
                {"input": "abcde fghi", "given_label": "positive"},
            ],
            "perturbations": [{"kind": "typo", "intensity": 0.1}],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "same_label"},
        }
        predict_fn = lambda texts: ["positive"] * len(texts)
        result = run_behavioral_tests(
            [test], predict_fn=predict_fn,
            slice_definitions=[LONG_INPUT_SLICE, SHORT_INPUT_SLICE],
        )
        # long_input has 1 (the seed). short_input has 0. Seed
        # membership is anchored to original, not perturbed.
        self.assertEqual(
            result["borderline_seed"]["per_slice"]["long_input"]["total"], 1,
        )
        self.assertEqual(
            result["borderline_seed"]["per_slice"]["short_input"]["total"], 0,
        )


# ────────────────────────────────────────────────────────────────────────
# DIR per-slice scoring
# ────────────────────────────────────────────────────────────────────────


class DirectionalPerSliceTests(unittest.TestCase):

    def test_dir_must_change_to_passes_in_one_slice_fails_in_another(self):
        # Two seeds: one long, one short. DIR expectation = must_change_to
        # negative. The long-input seed flips to "negative" (pass); the
        # short seed stays at "positive" (fail). Per-slice surfaces the
        # divergent verdict.
        test = {
            "test_id": "negation_flips",
            "kind": "DIR",
            "seed_examples": [
                {"input": "this is a long input row"},
                {"input": "short"},
            ],
            "perturbations": [
                {"kind": "insert_token", "params": {"token": "not ", "position": 0}},
            ],
            "n_perturbations_per_seed": 1,
            "expectation": {
                "kind": "must_change_to", "target_label": "negative",
            },
        }
        call_n = {"i": 0}

        def predict_fn(texts):
            call_n["i"] += 1
            if call_n["i"] == 1:
                return ["positive", "positive"]
            return ["negative", "positive"]  # long flips, short stuck

        result = run_behavioral_tests(
            [test], predict_fn=predict_fn,
            slice_definitions=[LONG_INPUT_SLICE, SHORT_INPUT_SLICE],
        )
        long_block = result["negation_flips"]["per_slice"]["long_input"]
        short_block = result["negation_flips"]["per_slice"]["short_input"]
        self.assertEqual(long_block["passed"], 1)
        self.assertEqual(long_block["pass_rate"], 1.0)
        self.assertEqual(short_block["passed"], 0)
        self.assertEqual(short_block["pass_rate"], 0.0)


# ────────────────────────────────────────────────────────────────────────
# MFT per-slice scoring
# ────────────────────────────────────────────────────────────────────────


class MftPerSliceTests(unittest.TestCase):

    def test_mft_buckets_examples_by_slice_predicate(self):
        # 3 examples: two long, one short. Mocked predict_fn returns
        # one wrong prediction in the long bucket and correct ones
        # in the short bucket → per_slice surfaces the divergence.
        test = {
            "test_id": "canonicals",
            "kind": "MFT",
            "examples": [
                {"input": "this is a long input row", "expected_label": "positive"},  # len > 10
                {"input": "another long row here",    "expected_label": "negative"},  # len > 10
                {"input": "short",                    "expected_label": "positive"},  # len < 10
            ],
        }
        # Predict: long-positive WRONG, long-negative CORRECT, short CORRECT.
        # Top-level: 2/3 pass. Long bucket: 1/2 pass. Short bucket: 1/1 pass.
        predict_fn = lambda texts: ["neutral", "negative", "positive"]
        result = run_behavioral_tests(
            [test], predict_fn=predict_fn,
            slice_definitions=[LONG_INPUT_SLICE, SHORT_INPUT_SLICE],
        )
        top = result["canonicals"]
        self.assertEqual(top["passed"], 2)
        self.assertEqual(top["total"], 3)
        long_block = top["per_slice"]["long_input"]
        short_block = top["per_slice"]["short_input"]
        self.assertEqual(long_block["passed"], 1)
        self.assertEqual(long_block["total"], 2)
        self.assertEqual(short_block["passed"], 1)
        self.assertEqual(short_block["total"], 1)


# ────────────────────────────────────────────────────────────────────────
# failed_examples cap per slice
# ────────────────────────────────────────────────────────────────────────


class PerSliceFailedExamplesCapTests(unittest.TestCase):

    def test_per_slice_failed_examples_capped(self):
        # 15 failing seeds all in long_input. Per-slice cap should
        # trim failed_examples to PER_SLICE_FAILED_EXAMPLES_CAP.
        seeds = [
            {"input": f"this is a long input row {i}", "given_label": "positive"}
            for i in range(15)
        ]
        test = {
            "test_id": "many_long_fails",
            "kind": "INV",
            "seed_examples": seeds,
            "perturbations": [{"kind": "typo", "intensity": 0.05}],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "same_label"},
        }
        call_n = {"i": 0}

        def predict_fn(texts):
            call_n["i"] += 1
            if call_n["i"] == 1:
                return ["positive"] * len(texts)
            return ["negative"] * len(texts)

        result = run_behavioral_tests(
            [test], predict_fn=predict_fn,
            slice_definitions=[LONG_INPUT_SLICE],
        )
        long_block = result["many_long_fails"]["per_slice"]["long_input"]
        self.assertEqual(long_block["total"], 15)
        self.assertEqual(long_block["passed"], 0)
        self.assertEqual(
            len(long_block["failed_examples"]),
            PER_SLICE_FAILED_EXAMPLES_CAP,
        )


if __name__ == "__main__":
    unittest.main()

"""Paired before/after evidence: row counts + "is it more than noise?".

Motivating case: an auto-RAG comparison showed "+22% F1 with retrieval" on 21
rows; across three training seeds the same comparison gave +22%, +4%, -3%.
A lift needs its row counts and an interval next to it.
"""

from __future__ import annotations

import unittest

from app.services.paired_comparison_stats import (
    MIN_ROWS_FOR_VERDICT,
    exact_mcnemar_p,
    paired_difference_evidence,
    seed_spread_evidence,
)


class PairedDifferenceEvidenceTests(unittest.TestCase):
    def test_counts_rows_better_worse_same(self):
        ev = paired_difference_evidence([0.1, 0.5, 0.3, 0.2, 0.4, 0.0], [0.3, 0.2, 0.3, 0.6, 0.5, 0.1])
        self.assertEqual((ev["n"], ev["better"], ev["worse"], ev["same"]), (6, 4, 1, 1))

    def test_consistent_gain_is_better(self):
        before = [0.10, 0.20, 0.15, 0.30, 0.25, 0.12, 0.18, 0.22]
        after = [b + d for b, d in zip(before, [0.10, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08, 0.11])]
        ev = paired_difference_evidence(before, after)
        self.assertEqual(ev["verdict"], "better")
        self.assertGreater(ev["ci_low"], 0)
        self.assertAlmostEqual(ev["mean_diff"], 0.105, places=3)

    def test_consistent_drop_is_worse(self):
        before = [0.5, 0.6, 0.55, 0.7, 0.65, 0.52]
        ev = paired_difference_evidence(before, [b - 0.2 + 0.01 * i for i, b in enumerate(before)])
        self.assertEqual(ev["verdict"], "worse")
        self.assertLess(ev["ci_high"], 0)

    def test_positive_average_with_mixed_rows_is_within_noise(self):
        """A positive mean lift carried by a few rows while others drop: the
        headline says "+X%", the interval says it could be nothing."""
        before = [0.2] * 10
        after = [0.6, 0.5, 0.1, 0.1, 0.3, 0.1, 0.25, 0.15, 0.2, 0.2]
        ev = paired_difference_evidence(before, after)
        self.assertGreater(ev["mean_diff"], 0)
        self.assertEqual(ev["verdict"], "within_noise")
        self.assertLess(ev["ci_low"], 0)
        self.assertGreater(ev["ci_high"], 0)

    def test_too_few_rows_gets_no_verdict(self):
        ev = paired_difference_evidence([0.1, 0.1], [0.9, 0.9])
        self.assertEqual(ev["verdict"], "too_few_rows")
        self.assertIsNone(ev["ci_low"])
        self.assertEqual(ev["better"], 2)
        self.assertLess(ev["n"], MIN_ROWS_FOR_VERDICT)
        empty = paired_difference_evidence([], [])
        self.assertEqual((empty["n"], empty["verdict"], empty["mean_diff"]), (0, "too_few_rows", None))

    def test_no_change_at_all_is_within_noise(self):
        ev = paired_difference_evidence([0.3] * 8, [0.3] * 8)
        self.assertEqual((ev["verdict"], ev["same"]), ("within_noise", 8))

    def test_interval_matches_a_hand_computed_t_interval(self):
        # diffs 0.1..0.5: mean 0.3, sd 0.1581, n=5, t(4)=2.776 → ±0.1963
        ev = paired_difference_evidence([0.0] * 5, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(ev["ci_low"], 0.3 - 0.1963, places=3)
        self.assertAlmostEqual(ev["ci_high"], 0.3 + 0.1963, places=3)

    def test_non_numeric_pairs_are_skipped(self):
        ev = paired_difference_evidence([0.1, None, 0.2, float("nan")], [0.2, 0.5, 0.1, 0.3])  # type: ignore[list-item]
        self.assertEqual((ev["n"], ev["better"], ev["worse"]), (2, 1, 1))


class SeedSpreadEvidenceTests(unittest.TestCase):
    """Run-to-run evidence: N seeds of one config vs the base model."""

    def test_seeds_that_agree_establish_the_sign(self):
        ev = seed_spread_evidence(0.074, [0.172, 0.189, 0.201])
        self.assertEqual(ev["kind"], "seeds")
        self.assertEqual(ev["verdict"], "better")
        self.assertTrue(ev["all_better"])
        self.assertAlmostEqual(ev["mean"], 0.1873, places=3)
        self.assertAlmostEqual(ev["std"], 0.0146, places=3)
        self.assertEqual((ev["min"], ev["max"], ev["n"]), (0.172, 0.201, 3))
        self.assertGreater(ev["ci_low"], 0)

    def test_seeds_that_disagree_are_within_noise(self):
        # +22%, +4%, -3%: the case that motivated this — one seed looked
        # like a gain, the others didn't.
        ev = seed_spread_evidence(0.2165, [0.2645, 0.2257, 0.2105])
        self.assertEqual(ev["verdict"], "within_noise")
        self.assertFalse(ev["all_better"])
        self.assertLess(ev["ci_low"], 0)
        self.assertGreater(ev["ci_high"], 0)

    def test_two_seeds_give_a_verdict_one_does_not(self):
        two = seed_spread_evidence(0.1, [0.5, 0.52])
        self.assertEqual(two["verdict"], "better")  # tight pair → interval still above 0
        wide = seed_spread_evidence(0.1, [0.5, 0.11])
        self.assertEqual(wide["verdict"], "within_noise")
        one = seed_spread_evidence(0.1, [0.5])
        self.assertEqual(one["verdict"], "too_few_seeds")
        self.assertIsNone(one["std"])
        self.assertEqual(one["mean"], 0.5)

    def test_all_worse(self):
        ev = seed_spread_evidence(0.5, [0.3, 0.31, 0.28])
        self.assertEqual((ev["verdict"], ev["all_worse"], ev["all_better"]), ("worse", True, False))


def _binary_rows(better: int, worse: int, both_right: int, n: int) -> tuple[list[float], list[float]]:
    """0/1 before/after lists: ``better`` rows 0->1, ``worse`` 1->0, ``both_right`` 1->1, rest 0->0."""
    before = [0.0] * better + [1.0] * worse + [1.0] * both_right
    after = [1.0] * better + [0.0] * worse + [1.0] * both_right
    rest = n - len(before)
    return before + [0.0] * rest, after + [0.0] * rest


class McNemarGuardTests(unittest.TestCase):
    """Binary paired outcomes: better / worse need the interval to exclude 0 AND exact McNemar p < 0.05."""

    def test_exact_mcnemar_p_values(self):
        self.assertAlmostEqual(exact_mcnemar_p(4, 0), 0.125)
        self.assertAlmostEqual(exact_mcnemar_p(5, 0), 0.0625)
        self.assertAlmostEqual(exact_mcnemar_p(6, 0), 0.03125)
        self.assertAlmostEqual(exact_mcnemar_p(3, 0), 0.25)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(5, 5), 1.0)  # capped at 1
        self.assertAlmostEqual(exact_mcnemar_p(1, 9), exact_mcnemar_p(9, 1))
        # scipy.stats.binomtest(10, 30, 0.5).pvalue
        self.assertAlmostEqual(exact_mcnemar_p(10, 20), 0.09873714670538902, places=12)
        # Large counts stay finite and exact.
        self.assertLess(exact_mcnemar_p(900, 100), 1e-100)

    def test_four_better_none_worse_on_50_rows_is_within_noise(self):
        ev = paired_difference_evidence(*_binary_rows(4, 0, 23, 50))
        self.assertGreater(ev["ci_low"], 0)  # the interval alone would call it a lift
        self.assertAlmostEqual(ev["mcnemar_p"], 0.125)
        self.assertEqual(ev["verdict"], "within_noise")

    def test_six_better_none_worse_is_better(self):
        ev = paired_difference_evidence(*_binary_rows(6, 0, 22, 50))
        self.assertAlmostEqual(ev["mcnemar_p"], 0.03125)
        self.assertEqual(ev["verdict"], "better")

    def test_six_worse_none_better_is_worse_and_four_is_noise(self):
        self.assertEqual(paired_difference_evidence(*_binary_rows(0, 6, 22, 50))["verdict"], "worse")
        self.assertEqual(paired_difference_evidence(*_binary_rows(0, 4, 23, 50))["verdict"], "within_noise")

    def test_three_of_five_and_five_of_fifty(self):
        self.assertEqual(paired_difference_evidence(*_binary_rows(3, 0, 1, 5))["verdict"], "within_noise")
        five = paired_difference_evidence(*_binary_rows(5, 0, 20, 50))
        self.assertEqual((five["verdict"], five["mcnemar_p"]), ("within_noise", 0.0625))

    def test_significant_p_but_interval_with_zero_stays_within_noise(self):
        # Many changed rows nearly balanced: neither test calls it.
        ev = paired_difference_evidence(*_binary_rows(12, 8, 10, 60))
        self.assertEqual(ev["verdict"], "within_noise")
        self.assertGreater(ev["mcnemar_p"], 0.05)

    def test_large_clear_binary_lift_is_still_better(self):
        ev = paired_difference_evidence(*_binary_rows(30, 3, 20, 100))
        self.assertLess(ev["mcnemar_p"], 0.001)
        self.assertEqual(ev["verdict"], "better")

    def test_no_changed_rows_has_p_one(self):
        ev = paired_difference_evidence(*_binary_rows(0, 0, 10, 20))
        self.assertEqual((ev["mcnemar_p"], ev["verdict"]), (1.0, "within_noise"))

    def test_too_few_rows_still_records_p(self):
        ev = paired_difference_evidence(*_binary_rows(3, 0, 0, 3))
        self.assertEqual(ev["verdict"], "too_few_rows")
        self.assertAlmostEqual(ev["mcnemar_p"], 0.25)

    def test_continuous_scores_keep_the_interval_rule(self):
        before = [0.10, 0.20, 0.15, 0.30, 0.25, 0.12, 0.18, 0.22]
        after = [b + 0.1 for b in before]
        ev = paired_difference_evidence(before, after)
        self.assertIsNone(ev["mcnemar_p"])
        self.assertEqual(ev["verdict"], "better")
        # One partial-credit score makes the whole pairing continuous.
        b, a = _binary_rows(4, 0, 23, 50)
        a[-1] = 0.5
        mixed = paired_difference_evidence(b, a)
        self.assertIsNone(mixed["mcnemar_p"])

    def test_empty_input(self):
        ev = paired_difference_evidence([], [])
        self.assertEqual((ev["n"], ev["mcnemar_p"], ev["verdict"]), (0, None, "too_few_rows"))

    def test_seed_spread_is_not_guarded(self):
        ev = seed_spread_evidence(0.0, [1.0, 1.0, 1.0])
        self.assertIsNone(ev["mcnemar_p"])

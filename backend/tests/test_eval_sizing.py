"""Detectable difference + rows needed for any per-row metric (Epic W2).

Pure tests of ``eval_sizing_service``: binary and continuous single evals,
paired (lift / A-vs-B) sizing, the degenerate cases, and the span sizing it
generalises (``span_eval_sizing_service``) pinned unchanged. The expected
numbers are worked by hand in the comments, not recomputed from the code.

Z = z_0.975 + z_0.80 = 1.959964 + 0.841621 = 2.801585;  Z² = 7.848880.
√78 = 8.831761;  √10 = 3.162278.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from app.services import eval_sizing_service as svc
from app.services import span_eval_sizing_service as span


class SingleEvalTests(unittest.TestCase):
    def test_78_binary_rows_near_30_percent_use_the_unpaired_comparison_bound(self):
        # 23 of 78 pass: p = 0.294872; s² = p(1−p) = 0.207922. Comparing two runs with no
        # pairing: Var(d) = 2·s² = 0.415844; sd = 0.644860.
        # gap = 2.801585 · √(0.415844 / 78) = 2.801585 · 0.073016 = 0.20456 → ceil 21 pts.
        # rows(5 pts) = 7.848880 · 0.415844 / 0.0025 = 1305.6 → 1306 (line rounds up to 50: ~1,350).
        # One-sample precision (API only): 2.801585 · 0.455985 / 8.831761 = 0.1446; 653 rows.
        sizing = svc.sizing_from_values([1] * 23 + [0] * 55, metric_id="exact_match",
                                        row_noun="test rows", metric_label="exact match")
        self.assertEqual((sizing["basis"], sizing["score_kind"], sizing["n"]), ("single", "binary", 78))
        self.assertAlmostEqual(sizing["sd"], 0.644860, places=5)
        self.assertAlmostEqual(sizing["detectable_gap"], 0.2046, places=3)
        self.assertEqual(sizing["rows_needed"], 1306)
        self.assertAlmostEqual(sizing["one_sample_gap"], 0.1446, places=3)
        self.assertEqual(sizing["one_sample_rows_needed"], 653)
        self.assertFalse(sizing["enough"])
        self.assertIsNone(sizing["degenerate"])
        self.assertEqual((sizing["confidence"], sizing["power"], sizing["target_gap"]), (0.95, 0.8, 0.05))
        self.assertEqual(
            sizing["line"],
            "Comparing two runs on these 78 test rows (exact match): gaps under ~21 pts may go unnoticed "
            "(conservative — same-row comparisons usually do better); ~1,350 rows to see a 5-pt change.",
        )

    def test_exactly_30_percent_is_root_two_above_the_one_sample_bound(self):
        # 24 of 80 (p = 0.3): s² = 0.21 (binomial). Unpaired 2·s² = 0.42.
        # rows(5 pts) = 7.848880 · 0.42 / 0.0025 = 1318.6 → 1319 (= 2 × 659.3);
        # at n = 78: gap = 2.801585 · √(0.42 / 78) = 2.801585 · 0.073380 = 0.20558 (~21 pts),
        # √2 × the one-sample 0.14537.
        sizing = svc.sizing_from_values([1] * 24 + [0] * 56)
        self.assertEqual(sizing["rows_needed"], 1319)
        self.assertEqual(sizing["one_sample_rows_needed"], 660)
        self.assertAlmostEqual(svc.detectable_gap_for(math.sqrt(0.42), 78), 0.20558, places=4)
        self.assertAlmostEqual(svc.detectable_gap_for(math.sqrt(0.21), 78) * math.sqrt(2), 0.20558, places=4)

    def test_continuous_scores_use_the_sample_variance(self):
        # [0.2, 0.4, 0.6, 0.8, 1.0] ×2: mean 0.6; Σ(x−0.6)² = 0.8; s² = 0.8 / 9 = 0.088889.
        # Unpaired 2·s² = 0.177778; sd = 0.421637; gap = 2.801585 · 0.421637 / 3.162278 = 0.37354 (~38 pts);
        # rows(5 pts) = 7.848880 · 0.177778 / 0.0025 = 558.1 → 559 (~560).
        sizing = svc.sizing_from_values([0.2, 0.4, 0.6, 0.8, 1.0] * 2, metric_id="f1", metric_label="F1")
        self.assertEqual(sizing["score_kind"], "continuous")
        self.assertAlmostEqual(sizing["sd"], 0.421637, places=5)
        self.assertAlmostEqual(sizing["detectable_gap"], 0.3735, places=3)
        self.assertEqual(sizing["rows_needed"], 559)
        self.assertIn("~38 pts", sizing["line"])
        self.assertIn("~560 rows", sizing["line"])

    def test_target_gap_is_configurable(self):
        # Same 23/78: rows(10 pts) = 7.848880 · 0.415844 / 0.01 = 326.4 → 327 (~330).
        sizing = svc.sizing_from_values([1] * 23 + [0] * 55, gap=0.10)
        self.assertEqual(sizing["rows_needed"], 327)
        self.assertIn("~330 rows to see a 10-pt change", sizing["line"])

    def test_gap_rounds_up_and_enough_rows_name_only_the_target(self):
        # 1400 rows at 50%: 2·0.25 = 0.5; gap = 2.801585 · √(0.5/1400) = 0.05294 → ceil 6 pts (not 5);
        # rows = 7.848880 · 0.5 / 0.0025 = 1569.4 → 1570 > 1400.
        short = svc.sizing_from_values([1, 0] * 700)
        self.assertEqual(short["rows_needed"], 1570)
        self.assertIn("gaps under ~6 pts", short["line"])
        # 1600 rows ≥ 1570: enough. The gap (0.04952) is not shown; the line names the target.
        sizing = svc.sizing_from_values([1, 0] * 800, metric_label="accuracy")
        self.assertTrue(sizing["enough"])
        self.assertEqual(
            sizing["line"],
            "Comparing two runs on these 1600 rows (accuracy): enough rows to see a 5-pt change "
            "(conservative — same-row comparisons usually do better).",
        )


class DegenerateTests(unittest.TestCase):
    def test_fewer_than_five_rows(self):
        # No gap; unpaired worst case for a [0, 1] score: 2 · 0.25 = 0.5 →
        # 7.848880 · 0.5 / 0.0025 = 1569.8 → 1570 (~1,600).
        sizing = svc.sizing_from_values([1, 0, 1], metric_label="exact match", row_noun="test rows")
        self.assertEqual(sizing["degenerate"], "too_few_rows")
        self.assertIsNone(sizing["detectable_gap"])
        self.assertIsNone(sizing["sd"])
        self.assertEqual(sizing["rows_needed"], 1570)
        self.assertFalse(sizing["enough"])
        self.assertIn("3 test rows (exact match): too few rows", sizing["line"])
        self.assertIn("~1,600 rows", sizing["line"])
        self.assertIsNone(svc.sizing_from_values([]))
        self.assertIsNone(svc.sizing_from_values([None, float("nan")]))

    def test_all_pass_and_all_fail_use_the_adjusted_rate(self):
        # Agresti–Coull: p̃ = (78 + 2) / (78 + 4) = 0.975610; p̃(1−p̃) = 0.023795; unpaired 0.047591;
        # gap = 2.801585 · √(0.047591 / 78) = 0.06920 (~7 pts); rows = 7.848880 · 0.047591 / 0.0025
        # = 149.4 → 150.
        for values, label in (([1] * 78, "all_pass"), ([0] * 78, "all_fail")):
            sizing = svc.sizing_from_values(values)
            self.assertEqual(sizing["degenerate"], label)
            self.assertAlmostEqual(sizing["detectable_gap"], 0.0692, places=3)
            self.assertEqual(sizing["rows_needed"], 150)
            self.assertFalse(sizing["enough"])
        self.assertIn("every row passed", svc.sizing_from_values([1] * 78)["line"])
        self.assertIn("every row failed", svc.sizing_from_values([0] * 78)["line"])

    def test_zero_variance_continuous(self):
        # Ten rows of 0.5 (heuristic): p̃ = (5 + 2) / (10 + 4) = 0.5; s² = 0.25; unpaired 0.5;
        # gap = 2.801585 · √(0.5 / 10) = 0.62646; rows = 1570.
        sizing = svc.sizing_from_values([0.5] * 10)
        self.assertEqual((sizing["degenerate"], sizing["score_kind"]), ("zero_variance", "continuous"))
        self.assertAlmostEqual(sizing["detectable_gap"], 0.6265, places=3)
        self.assertEqual(sizing["rows_needed"], 1570)


class PairedTests(unittest.TestCase):
    def test_78_paired_rows_with_typical_discordance(self):
        # Base 23/78 pass (~30%); the run fixes 6 rows and breaks 8 → 14 discordant (18%).
        # d: six +1, eight −1, 64 zeros. mean = −2/78 = −0.025641; Σd² = 14;
        # s_d² = (14 − 78 · 0.025641²) / 77 = (14 − 0.051282) / 77 = 0.181152; s_d = 0.425620.
        # gap = 2.801585 · 0.425620 / 8.831761 = 0.13501 → ceil 14 pts.
        # rows(5 pts) = 7.848880 · 0.181152 / 0.0025 = 568.74 → 569 (line ~570).
        before = [1] * 23 + [0] * 55
        after = list(before)
        for i in range(8):        # rows 0-7: pass → fail
            after[i] = 0
        for i in range(23, 29):   # rows 23-28: fail → pass
            after[i] = 1
        sizing = svc.paired_sizing(before, after, metric_id="exact_match", row_noun="paired rows",
                                   metric_label="exact match")
        self.assertEqual((sizing["basis"], sizing["n"], sizing["score_kind"]), ("paired", 78, "binary"))
        self.assertAlmostEqual(sizing["sd"], 0.425620, places=5)
        self.assertAlmostEqual(sizing["detectable_gap"], 0.1350, places=3)
        self.assertEqual(sizing["rows_needed"], 569)
        self.assertNotIn("one_sample_gap", sizing)
        self.assertEqual(
            sizing["line"],
            "78 paired rows (exact match): gaps under ~14 pts may go unnoticed; ~570 rows to see a 5-pt change.",
        )

    def test_continuous_paired(self):
        # d = [0.1, −0.1, 0.3, −0.3, 0.0] ×2: mean 0; Σd² = 0.4; s_d² = 0.4 / 9 = 0.044444; s_d = 0.210819.
        # gap = 2.801585 · 0.210819 / 3.162278 = 0.18677; rows = 7.848880 · 0.044444 / 0.0025 = 139.54 → 140.
        sizing = svc.paired_sizing([0.5] * 10, [0.6, 0.4, 0.8, 0.2, 0.5] * 2)
        self.assertEqual(sizing["score_kind"], "continuous")
        self.assertAlmostEqual(sizing["detectable_gap"], 0.1868, places=3)
        self.assertEqual(sizing["rows_needed"], 140)

    def test_identical_runs_use_the_labelled_unpaired_bound(self):
        # Both 23/78 on the same rows: every d = 0. Unpaired: 0.207922 + 0.207922 = 0.415844 →
        # gap 0.2046 (~21 pts), rows 1306.
        before = [1] * 23 + [0] * 55
        sizing = svc.paired_sizing(before, list(before))
        self.assertEqual((sizing["degenerate"], sizing["basis"]), ("identical_scores", "paired"))
        self.assertAlmostEqual(sizing["detectable_gap"], 0.2046, places=3)
        self.assertEqual(sizing["rows_needed"], 1306)
        self.assertIn("both runs scored every row the same", sizing["line"])
        self.assertIn("unpaired bound", sizing["line"])

    def test_constant_shift_is_not_called_identical(self):
        # before [0.1, 0.3, 0.5, 0.7, 0.9] ×2, after = before + 0.1: every d = 0.1, s_d = 0.
        # Each side s² = 0.8 / 9 = 0.088889; unpaired 0.177778 → gap 0.3735 (~38 pts), rows 559.
        before = [0.1, 0.3, 0.5, 0.7, 0.9] * 2
        sizing = svc.paired_sizing(before, [b + 0.1 for b in before])
        self.assertEqual(sizing["degenerate"], "constant_shift")
        self.assertAlmostEqual(sizing["detectable_gap"], 0.3735, places=3)
        self.assertEqual(sizing["rows_needed"], 559)
        self.assertIn("every row moved by the same amount", sizing["line"])
        self.assertNotIn("scored every row the same", sizing["line"])

    def test_paired_worst_case_and_no_pairs(self):
        # d in [−1, 1] → variance ≤ 1: 7.848880 · 1 / 0.0025 = 3139.55 → 3140 (~3,150).
        sizing = svc.paired_sizing([1, 0], [0, 1])
        self.assertEqual((sizing["degenerate"], sizing["rows_needed"], sizing["detectable_gap"]),
                         ("too_few_rows", 3140, None))
        self.assertIn("~3,150 rows", sizing["line"])
        self.assertIsNone(svc.paired_sizing([], []))
        # A non-finite score on either side drops that pair.
        self.assertEqual(svc.paired_sizing([1, None, 0], [0, 1, float("nan")])["n"], 1)


class EvalResultSizingTests(unittest.TestCase):
    def test_reads_the_headline_row_scores(self):
        result = SimpleNamespace(
            metrics={"exact_match": 23 / 78, "f1": 0.5},
            details={"row_scores": {"keys": [str(i) for i in range(78)], "correct": [1] * 23 + [0] * 55,
                                    "f1": [0.2, 0.4, 0.6, 0.8, 1.0, 0.2] * 13}},
        )
        # f1 outranks exact_match in the headline order.
        self.assertEqual(svc.eval_result_sizing(result)["metric_id"], "f1")
        em = svc.eval_result_sizing(result, "exact_match")
        self.assertEqual((em["metric_id"], em["rows_needed"]), ("exact_match", 1306))
        self.assertTrue(em["line"].startswith("Comparing two runs on these 78 test rows (exact match):"))

    def test_none_without_row_scores(self):
        self.assertIsNone(svc.eval_result_sizing(SimpleNamespace(metrics={"exact_match": 0.5}, details={})))
        self.assertIsNone(svc.eval_result_sizing(SimpleNamespace(metrics={"perplexity": 9.0},
                                                                 details={"row_scores": {"keys": ["a"]}})))


class SpanSizingUnchangedTests(unittest.TestCase):
    """The span-F1 sizing this generalises keeps its own formula and outputs."""

    def test_span_sizing_outputs_are_pinned(self):
        self.assertEqual(span.DECISION_GAP, 0.05)
        self.assertEqual((span.spans_needed(0.8), span.spans_needed(1.0), span.spans_needed(0.0)), (512, 152, 800))
        self.assertAlmostEqual(span.detectable_gap(0.8, 20), 0.2530, places=3)
        sizing = span.size_types({
            "email": {"f1": 1.0, "tp": 600, "fn": 0},
            "medication": {"f1": 0.8, "tp": 16, "fn": 4},
        })
        self.assertEqual(sizing["short"], ["medication"])
        self.assertEqual(sizing["shortfall"], 512 - 20)
        self.assertEqual(
            span.sizing_line(sizing),
            "1 of 2 types have enough reference spans to show a 0.05 F1 difference; the others can only show "
            "0.25 or more (medication: 20 spans, needs 512).",
        )


if __name__ == "__main__":
    unittest.main()

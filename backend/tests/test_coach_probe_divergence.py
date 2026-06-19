"""Coach-stage-2 phase 11 — probe-vs-gold divergence nudge.

The payoff of the probe-pack arc: when the held-out, platform-authored
probe pack scores materially below the user's self-authored gold set,
Coach surfaces "your gold set says green but the independent ruler
disagrees." Pure helper → tested directly with stub eval rows.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.coach_service import _probe_gold_divergence_nudge  # noqa: E402


def _eval(pass_rate, probe_rate, results=None):
    metrics = {"pass_rate": pass_rate}
    if probe_rate is not None:
        metrics["probe"] = {
            "probe_pass_rate": probe_rate,
            "results": results if results is not None else [],
        }
    return SimpleNamespace(pass_rate=pass_rate, metrics=metrics)


class ProbeGoldDivergenceTests(unittest.TestCase):
    def test_large_gap_fires_critical_with_failing_probes(self):
        row = _eval(0.95, 0.50, results=[
            {"id": "sft.safety.injection", "passed": False},
            {"id": "rag.format.empty_context", "passed": False},
            {"id": "sft.robust.typo", "passed": True},
        ])
        nudge = _probe_gold_divergence_nudge(1, row)
        self.assertIsNotNone(nudge)
        assert nudge is not None
        self.assertEqual(nudge["id"], "eval:probe-gold-divergence")
        self.assertEqual(nudge["severity"], "critical")
        self.assertEqual(nudge["rule_id"], "probe-gold-divergence.critical")
        self.assertEqual(nudge["action"]["params"]["target"], "probe-pack-panel")
        # Both failing probe ids carried for the drill-down link.
        self.assertEqual(nudge["context"]["failing_count"], 2)
        self.assertIn("sft.safety.injection", nudge["context"]["failing_probe_ids"])
        # Headline numbers surfaced in the body.
        self.assertIn("95%", nudge["body"])
        self.assertIn("50%", nudge["body"])

    def test_moderate_gap_fires_warning(self):
        row = _eval(0.90, 0.72)
        nudge = _probe_gold_divergence_nudge(1, row)
        self.assertIsNotNone(nudge)
        assert nudge is not None
        self.assertEqual(nudge["severity"], "warning")
        self.assertEqual(nudge["rule_id"], "probe-gold-divergence.warn")

    def test_small_gap_is_silent(self):
        # 0.80 - 0.75 = 0.05 < 0.15 threshold.
        self.assertIsNone(_probe_gold_divergence_nudge(1, _eval(0.80, 0.75)))

    def test_probe_above_gold_is_silent(self):
        # Nothing to warn about when the independent ruler agrees / is higher.
        self.assertIsNone(_probe_gold_divergence_nudge(1, _eval(0.50, 0.90)))

    def test_no_probe_metrics_is_silent(self):
        self.assertIsNone(_probe_gold_divergence_nudge(1, _eval(0.95, None)))

    def test_none_eval_is_silent(self):
        self.assertIsNone(_probe_gold_divergence_nudge(1, None))

    def test_exactly_at_threshold_fires(self):
        # 0.90 - 0.75 = 0.15 == threshold → fires (>=).
        nudge = _probe_gold_divergence_nudge(1, _eval(0.90, 0.75))
        self.assertIsNotNone(nudge)

    def test_persistent_streak_escalates_a_warning_to_critical(self):
        # A moderate gap (0.90 vs 0.72 = 0.18) is a warning on a one-off,
        # but a 3-run streak escalates it to critical with a "pattern" note.
        row = _eval(0.90, 0.72)
        warn = _probe_gold_divergence_nudge(1, row, streak=1)
        assert warn is not None
        self.assertEqual(warn["severity"], "warning")

        crit = _probe_gold_divergence_nudge(1, row, streak=3)
        assert crit is not None
        self.assertEqual(crit["severity"], "critical")
        self.assertEqual(crit["rule_id"], "probe-gold-divergence.persistent")
        self.assertIn("consecutive", crit["body"])
        self.assertEqual(crit["context"]["streak"], 3)


class DivergenceStreakTests(unittest.TestCase):
    def test_counts_consecutive_recent_diverging_runs(self):
        from app.services.probe_pack_service import divergence_streak
        # Chronological oldest → newest. Newest 2 diverge (≥0.15), the one
        # before is fine → streak is 2 (stops at the non-diverging run).
        history = [
            {"divergence": 0.20},   # old, diverging (but broken by next)
            {"divergence": 0.02},   # not diverging → resets
            {"divergence": 0.18},
            {"divergence": 0.25},   # newest
        ]
        self.assertEqual(divergence_streak(history, 0.15), 2)

    def test_zero_when_latest_run_agrees(self):
        from app.services.probe_pack_service import divergence_streak
        history = [{"divergence": 0.3}, {"divergence": 0.01}]
        self.assertEqual(divergence_streak(history, 0.15), 0)

    def test_empty_history_is_zero(self):
        from app.services.probe_pack_service import divergence_streak
        self.assertEqual(divergence_streak([], 0.15), 0)


class JudgeSpendSummaryTests(unittest.TestCase):
    def test_prefers_real_tokens_when_present(self):
        from app.services.probe_pack_service import summarize_judge_spend
        history = [
            {"judge_calls": 2, "judge_cached": 1, "judge_tokens": 740},
            {"judge_calls": 0, "judge_cached": 3, "judge_tokens": 0},  # all cached
            {"weight_regime": "x"},                                     # no judge
        ]
        spend = summarize_judge_spend(history)
        assert spend is not None
        self.assertEqual(spend["total_calls"], 2)
        self.assertEqual(spend["total_cached"], 4)
        self.assertEqual(spend["runs_with_judge"], 2)
        self.assertEqual(spend["total_tokens"], 740)   # real, not estimated
        self.assertFalse(spend["tokens_estimated"])

    def test_falls_back_to_estimate_when_tokens_absent(self):
        from app.services.probe_pack_service import (
            EST_TOKENS_PER_JUDGE_CALL,
            summarize_judge_spend,
        )
        # Old run: judge_calls but no judge_tokens → estimated.
        spend = summarize_judge_spend([{"judge_calls": 3, "judge_cached": 0}])
        assert spend is not None
        self.assertEqual(spend["total_tokens"], 3 * EST_TOKENS_PER_JUDGE_CALL)
        self.assertTrue(spend["tokens_estimated"])

    def test_none_when_no_run_judged(self):
        from app.services.probe_pack_service import summarize_judge_spend
        self.assertIsNone(summarize_judge_spend([{"weight_regime": "x"}, {}]))


if __name__ == "__main__":
    unittest.main()

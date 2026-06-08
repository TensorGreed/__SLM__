"""Tests for Quality-Lift phase 1, slice 3 — variance-aware gate evaluator.

Pins for slice 3 (gate evaluator behavior; UI surfacing is covered by
the ScorecardPanel vitest suite):

  Aggregate EvalResult → variance plumbing:
    * ``_build_metric_snapshot`` extracts ``mean`` as the scalar value
      and stashes the full ``{mean, std, min, max, n}`` block (plus
      per_seed provenance from ``details``) under the same normalized
      key in the variance dict.
    * Per-class nested variance dicts flatten the same way the
      Gap-#6 scalar shape does.

  Gate evaluation under lower-bound policy:
    * Without variance, gates behave EXACTLY as today (no behavior
      change for single-seed flows).
    * With variance and ``operator=gte``: gate passes only if
      ``mean − std >= threshold``. A run whose mean clears the bar but
      whose lower bound doesn't gets ``reason=variance_below_threshold``
      so the UI can flag "your mean was 0.83 but std=0.04 means the
      0.80 gate actually fails."
    * With ``operator=lte``: symmetric — passes only if
      ``mean + std <= threshold``.
    * Opt-out: ``variance_policy=mean`` falls back to point-estimate
      comparison (per-gate override).

  Gate response shape:
    * Adds ``actual_std, actual_min, actual_max, actual_n, gate_value,
      variance_policy, per_seed, seed_group_id`` only when variance is
      present — single-seed responses are unchanged.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.evaluation_pack_service import (  # noqa: E402
    _build_metric_snapshot,
    _evaluate_gate,
)


def _mock_eval_result(
    *,
    eval_type: str,
    dataset_name: str,
    metrics: dict,
    pass_rate: float | None = None,
    is_aggregate: bool = False,
    seed_group_id: str | None = None,
    details: dict | None = None,
    row_id: int = 1,
) -> MagicMock:
    """A duck-typed EvalResult — the snapshot builder only reads a few
    attributes, so a Mock is faster than seeding the DB.
    """
    row = MagicMock()
    row.id = row_id
    row.eval_type = eval_type
    row.dataset_name = dataset_name
    row.metrics = metrics
    row.pass_rate = pass_rate
    row.is_aggregate = is_aggregate
    row.seed_group_id = seed_group_id
    row.details = details or {}
    return row


class MetricSnapshotVarianceTests(unittest.TestCase):

    def test_scalar_metrics_have_no_variance_entry(self):
        row = _mock_eval_result(
            eval_type="classification",
            dataset_name="held_out",
            metrics={"macro_f1": 0.83, "accuracy": 0.87},
            pass_rate=1.0,
        )
        values, sources, variance = _build_metric_snapshot({"classification": row})
        self.assertAlmostEqual(values["macro_f1"], 0.83, places=4)
        # Single-seed flow: variance is empty so the gate evaluator falls
        # back to point-estimate comparison transparently.
        self.assertEqual(variance, {})

    def test_aggregate_variance_block_extracted_with_mean_as_value(self):
        # Aggregate row from the seed-group aggregator: each metric is
        # a dict with mean/std/.../n keys instead of a scalar.
        row = _mock_eval_result(
            eval_type="classification",
            dataset_name="held_out",
            metrics={
                "macro_f1": {
                    "mean": 0.83, "std": 0.04, "min": 0.79,
                    "max": 0.87, "n": 3,
                },
            },
            pass_rate=0.95,
            is_aggregate=True,
            seed_group_id="group-abc",
            details={
                "per_seed": [
                    {"experiment_id": 11, "seed_value": 42, "eval_result_id": 21, "pass_rate": 0.94},
                    {"experiment_id": 12, "seed_value": 43, "eval_result_id": 22, "pass_rate": 0.96},
                    {"experiment_id": 13, "seed_value": 44, "eval_result_id": 23, "pass_rate": 0.95},
                ],
                "n_succeeded": 3, "n_failed": 0, "n_total": 3,
            },
        )
        values, sources, variance = _build_metric_snapshot({"classification": row})
        # The scalar values dict carries the MEAN — gates compare against
        # mean (or mean ± std under lower-bound policy in the evaluator).
        self.assertAlmostEqual(values["macro_f1"], 0.83, places=4)
        self.assertIn("macro_f1", variance)
        block = variance["macro_f1"]
        self.assertAlmostEqual(block["mean"], 0.83, places=4)
        self.assertAlmostEqual(block["std"], 0.04, places=4)
        self.assertEqual(block["n"], 3)
        # Provenance carried through so the gate response can render
        # the drill-down without a second round-trip.
        self.assertEqual(len(block["per_seed"]), 3)
        self.assertEqual(block["per_seed"][0]["seed_value"], 42)
        self.assertEqual(block["seed_group_id"], "group-abc")
        self.assertTrue(block["is_aggregate"])

    def test_per_class_variance_recurses(self):
        # Mirrors the aggregator's recursion through per_class nested
        # dicts — per-class precision/recall/f1 get the variance shape
        # at the leaf, which must flatten into the gate-eligible keys.
        row = _mock_eval_result(
            eval_type="classification",
            dataset_name="held_out",
            metrics={
                "per_class": {
                    "benign": {
                        "precision": {"mean": 0.90, "std": 0.02, "min": 0.88, "max": 0.92, "n": 3},
                        "recall": {"mean": 0.85, "std": 0.03, "min": 0.82, "max": 0.88, "n": 3},
                        "f1": {"mean": 0.87, "std": 0.025, "min": 0.85, "max": 0.90, "n": 3},
                        "support": {"mean": 100, "std": 0.0, "min": 100, "max": 100, "n": 3},
                    },
                },
            },
            is_aggregate=True,
            seed_group_id="group-abc",
            details={"per_seed": [
                {"experiment_id": 11, "seed_value": 42, "eval_result_id": 21},
            ]},
        )
        values, sources, variance = _build_metric_snapshot({"classification": row})
        # Per-class shortcut keys (Gap-#6 contract) still resolve to the
        # mean — variance lives in the variance dict.
        self.assertAlmostEqual(values["precision_benign"], 0.90, places=4)
        self.assertAlmostEqual(values["recall_benign"], 0.85, places=4)
        # The variance block plumbs through the same key path.
        self.assertIn("precision_benign", variance)
        self.assertAlmostEqual(variance["precision_benign"]["std"], 0.02, places=4)
        self.assertAlmostEqual(variance["recall_benign"]["mean"], 0.85, places=4)


class EvaluateGateVarianceTests(unittest.TestCase):
    """Pure unit tests on _evaluate_gate; no DB."""

    def test_no_variance_legacy_path_unchanged(self):
        result = _evaluate_gate(
            {"gate_id": "min_f1", "metric_id": "macro_f1",
             "operator": "gte", "threshold": 0.80, "required": True},
            values={"macro_f1": 0.83},
            sources={},
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "ok")
        self.assertEqual(result["actual"], 0.83)
        # No variance fields on single-seed responses — UI's existing
        # render path must keep working untouched.
        self.assertNotIn("actual_std", result)
        self.assertNotIn("variance_policy", result)

    def test_lower_bound_policy_blocks_when_std_drags_under(self):
        # mean=0.83, std=0.04, threshold=0.80 → mean clears, mean−std doesn't.
        # Exactly the "vanity gate" failure mode that motivated this slice.
        variance = {
            "macro_f1": {
                "mean": 0.83, "std": 0.04, "min": 0.79, "max": 0.87, "n": 3,
                "per_seed": [
                    {"experiment_id": 1, "seed_value": 42},
                    {"experiment_id": 2, "seed_value": 43},
                    {"experiment_id": 3, "seed_value": 44},
                ],
                "seed_group_id": "g1",
                "is_aggregate": True,
            },
        }
        result = _evaluate_gate(
            {"gate_id": "min_f1", "metric_id": "macro_f1",
             "operator": "gte", "threshold": 0.80, "required": True},
            values={"macro_f1": 0.83},
            sources={"macro_f1": {"eval_result_id": 1}},
            variance=variance,
        )
        self.assertFalse(result["passed"])
        # Specific reason so the UI can render the honest message rather
        # than a generic "below threshold."
        self.assertEqual(result["reason"], "variance_below_threshold")
        self.assertEqual(result["actual"], 0.83)
        self.assertAlmostEqual(result["actual_std"], 0.04, places=4)
        self.assertAlmostEqual(result["gate_value"], 0.79, places=4)
        self.assertEqual(result["variance_policy"], "lower_bound")
        self.assertEqual(result["actual_n"], 3)
        # Drill-down carried inline.
        self.assertEqual(len(result["per_seed"]), 3)
        self.assertEqual(result["seed_group_id"], "g1")

    def test_lower_bound_policy_passes_when_lower_bound_clears(self):
        # mean=0.90, std=0.02, threshold=0.85 → mean−std=0.88 still clears.
        variance = {"macro_f1": {
            "mean": 0.90, "std": 0.02, "min": 0.88, "max": 0.92, "n": 3,
            "per_seed": [], "seed_group_id": "g2", "is_aggregate": True,
        }}
        result = _evaluate_gate(
            {"gate_id": "min_f1", "metric_id": "macro_f1",
             "operator": "gte", "threshold": 0.85, "required": True},
            values={"macro_f1": 0.90},
            sources={"macro_f1": {"eval_result_id": 1}},
            variance=variance,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "ok")
        self.assertAlmostEqual(result["gate_value"], 0.88, places=4)

    def test_lte_operator_uses_upper_bound(self):
        # For "must be below X" gates the conservative reading is mean+std,
        # not mean−std. Loss / latency metrics fall into this bucket.
        variance = {"eval_loss": {
            "mean": 0.30, "std": 0.05, "min": 0.25, "max": 0.35, "n": 3,
            "per_seed": [], "seed_group_id": "g3", "is_aggregate": True,
        }}
        # threshold=0.32, mean (0.30) clears, mean+std (0.35) doesn't.
        result = _evaluate_gate(
            {"gate_id": "max_loss", "metric_id": "eval_loss",
             "operator": "lte", "threshold": 0.32, "required": True},
            values={"eval_loss": 0.30},
            sources={"eval_loss": {"eval_result_id": 1}},
            variance=variance,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "variance_above_threshold")
        self.assertAlmostEqual(result["gate_value"], 0.35, places=4)

    def test_variance_policy_mean_opts_out_of_lower_bound(self):
        # Per-gate override — same metric & values, lower_bound would
        # fail but mean policy passes the gate at the point estimate.
        variance = {"macro_f1": {
            "mean": 0.83, "std": 0.04, "min": 0.79, "max": 0.87, "n": 3,
            "per_seed": [], "seed_group_id": "g1", "is_aggregate": True,
        }}
        result = _evaluate_gate(
            {"gate_id": "min_f1", "metric_id": "macro_f1",
             "operator": "gte", "threshold": 0.80, "required": True,
             "variance_policy": "mean"},
            values={"macro_f1": 0.83},
            sources={"macro_f1": {"eval_result_id": 1}},
            variance=variance,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["variance_policy"], "mean")
        # gate_value equals actual (the point estimate).
        self.assertAlmostEqual(result["gate_value"], 0.83, places=4)

    def test_legacy_below_threshold_reason_preserved(self):
        # When the mean itself fails the gate (not the variance squeezing
        # it out), the reason stays "below_threshold" — the UI's existing
        # styling for hard fails keeps working.
        variance = {"macro_f1": {
            "mean": 0.70, "std": 0.02, "min": 0.68, "max": 0.72, "n": 3,
            "per_seed": [], "seed_group_id": "g4", "is_aggregate": True,
        }}
        result = _evaluate_gate(
            {"gate_id": "min_f1", "metric_id": "macro_f1",
             "operator": "gte", "threshold": 0.80, "required": True},
            values={"macro_f1": 0.70},
            sources={"macro_f1": {"eval_result_id": 1}},
            variance=variance,
        )
        self.assertFalse(result["passed"])
        # Not "variance_below_threshold" — the mean itself is below.
        self.assertEqual(result["reason"], "below_threshold")


if __name__ == "__main__":
    unittest.main()

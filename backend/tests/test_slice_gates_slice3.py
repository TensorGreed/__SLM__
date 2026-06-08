"""Quality-Lift phase 2, slice 3 — Slice gate evaluator behaviors.

Pins (slice 3: backend gate plumbing + new operators + catalog
validation; UI rendering covered by ScorecardPanel.test.tsx):

  Snapshot per_slice flattening:
    * per_slice.{slice_id}.{metric} canonical key emitted.
    * <metric>_slice_<slice_id> short form emitted; ``_slice_`` infix
      disambiguates from per_class's ``<metric>_<label>``.
    * Eval-type-scoped variant emitted.
    * Variance blocks plumb through nested per_slice dicts
      (multi-seed × slicing composition from phase 1).

  Single-slice gates (operator gte/lte + slice_name):
    * Rewrites the metric resolution to per_slice.<slice>.<metric>
      and gates that value.
    * Response surfaces ``slice_name`` so the UI labels the row.

  Worst-slice gates (worst_slice_gte / worst_slice_lte):
    * Enumerates every per_slice.*.{metric}; filters by support
      ≥ min_slice_support; picks the worst eligible.
    * worst_slice_gte fails when ANY eligible slice is below the
      threshold; worst_slice_lte fails when ANY is above.
    * worst_slice_below_threshold / worst_slice_above_threshold
      reasons surface the directionality the UI uses for status
      labels.
    * min_slice_support default of 5 filters tiny slices; the
      drill-down still lists them with below_min_support=True.
    * No eligible slices → required gate fails with
      no_eligible_slices_required reason.
    * Variance policy applies per-slice (mean − std for gte;
      mean + std for lte). The worst is then picked from those
      per-slice gate values.

  Catalog validation:
    * worst_slice_gte / worst_slice_lte appear in VALID_GATE_OPERATORS.
    * is_per_slice_metric_id accepts all three id-shapes.
    * validate_draft_pack_gates accepts a per-slice gate that
      references slice metric_ids the static catalog doesn't know,
      same way it does for per_class.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.evaluation_gate_catalog import (  # noqa: E402
    VALID_GATE_OPERATORS,
    is_per_slice_metric_id,
    validate_draft_pack_gates,
)
from app.services.evaluation_pack_service import (  # noqa: E402
    DEFAULT_MIN_SLICE_SUPPORT,
    _build_metric_snapshot,
    _evaluate_gate,
)


def _mock_row(
    *,
    eval_type: str = "classification",
    dataset_name: str = "held_out",
    metrics: dict | None = None,
    pass_rate: float | None = None,
    is_aggregate: bool = False,
    seed_group_id: str | None = None,
    details: dict | None = None,
    row_id: int = 1,
) -> MagicMock:
    row = MagicMock()
    row.id = row_id
    row.eval_type = eval_type
    row.dataset_name = dataset_name
    row.metrics = metrics or {}
    row.pass_rate = pass_rate
    row.is_aggregate = is_aggregate
    row.seed_group_id = seed_group_id
    row.details = details or {}
    return row


def _row_with_slices() -> MagicMock:
    """An EvalResult row mirroring what score_with_slices produces:
    overall metrics + per_slice with handler-shaped metrics per slice."""
    return _mock_row(metrics={
        "accuracy": 0.78,
        "f1": 0.75,
        "per_slice": {
            "long_input": {"accuracy": 0.55, "f1": 0.52, "support": 40},
            "short_input": {"accuracy": 0.88, "f1": 0.85, "support": 200},
            "hindi": {"accuracy": 0.60, "f1": 0.58, "support": 30},
            "tiny_slice": {"accuracy": 0.30, "f1": 0.30, "support": 2},
        },
    })


# ────────────────────────────────────────────────────────────────────────
# Snapshot flattening
# ────────────────────────────────────────────────────────────────────────


class FlattenPerSliceSnapshotTests(unittest.TestCase):

    def test_canonical_dot_path_emitted(self):
        values, _, _ = _build_metric_snapshot({"classification": _row_with_slices()})
        # Canonical form — what worst-slice gate enumeration scans for.
        self.assertAlmostEqual(values["per_slice.long_input.accuracy"], 0.55, places=4)
        self.assertAlmostEqual(values["per_slice.long_input.f1"], 0.52, places=4)
        self.assertAlmostEqual(values["per_slice.long_input.support"], 40.0, places=4)

    def test_short_form_uses_slice_infix(self):
        # Single-slice gates can use the short form. The ``_slice_``
        # infix prevents collisions with per_class's ``<metric>_<label>``
        # (e.g. a slice_id of "benign" wouldn't fight a class label of
        # "benign").
        values, _, _ = _build_metric_snapshot({"classification": _row_with_slices()})
        self.assertAlmostEqual(values["f1_slice_long_input"], 0.52, places=4)
        self.assertAlmostEqual(values["accuracy_slice_short_input"], 0.88, places=4)

    def test_eval_type_scoped_form_emitted(self):
        values, _, _ = _build_metric_snapshot({"classification": _row_with_slices()})
        self.assertAlmostEqual(
            values["classification.per_slice.long_input.f1"], 0.52, places=4,
        )

    def test_variance_block_recurses_through_per_slice(self):
        # Aggregate row from multi-seed (phase 1) × slicing (phase 2)
        # composition: per_slice values are themselves variance blocks.
        # The snapshot must surface the std/n at the same key path so
        # gate evaluation's lower-bound policy applies per-slice.
        row = _mock_row(
            is_aggregate=True,
            seed_group_id="g1",
            metrics={
                "per_slice": {
                    "long_input": {
                        "f1": {"mean": 0.55, "std": 0.04, "min": 0.51, "max": 0.59, "n": 3},
                        "support": {"mean": 40, "std": 0.0, "min": 40, "max": 40, "n": 3},
                    },
                },
            },
        )
        values, _, variance = _build_metric_snapshot({"classification": row})
        self.assertAlmostEqual(values["per_slice.long_input.f1"], 0.55, places=4)
        self.assertIn("per_slice.long_input.f1", variance)
        self.assertAlmostEqual(
            variance["per_slice.long_input.f1"]["std"], 0.04, places=4,
        )
        self.assertEqual(variance["per_slice.long_input.f1"]["n"], 3)


# ────────────────────────────────────────────────────────────────────────
# Single-slice gates
# ────────────────────────────────────────────────────────────────────────


class SingleSliceGateTests(unittest.TestCase):

    def test_slice_name_rewrites_metric_resolution(self):
        values, sources, variance = _build_metric_snapshot(
            {"classification": _row_with_slices()}
        )
        # Gate without slice_name reads top-level f1 (0.75).
        plain = _evaluate_gate(
            {"gate_id": "min_f1", "metric_id": "f1",
             "operator": "gte", "threshold": 0.70, "required": True},
            values=values, sources=sources, variance=variance,
        )
        self.assertTrue(plain["passed"])
        # Same metric_id with slice_name="long_input" — the gate resolves
        # to per_slice.long_input.f1 = 0.52, which fails the 0.70 bar.
        sliced = _evaluate_gate(
            {"gate_id": "min_f1_long_input", "metric_id": "f1",
             "slice_name": "long_input",
             "operator": "gte", "threshold": 0.70, "required": True},
            values=values, sources=sources, variance=variance,
        )
        self.assertFalse(sliced["passed"])
        self.assertAlmostEqual(sliced["actual"], 0.52, places=4)
        self.assertEqual(sliced["slice_name"], "long_input")
        self.assertEqual(sliced["resolved_metric_key"], "per_slice.long_input.f1")


# ────────────────────────────────────────────────────────────────────────
# Worst-slice gates
# ────────────────────────────────────────────────────────────────────────


class WorstSliceGateTests(unittest.TestCase):

    def test_worst_slice_gte_picks_smallest_eligible_slice(self):
        values, sources, variance = _build_metric_snapshot(
            {"classification": _row_with_slices()}
        )
        result = _evaluate_gate(
            {"gate_id": "no_slice_below_60", "metric_id": "f1",
             "operator": "worst_slice_gte", "threshold": 0.60, "required": True},
            values=values, sources=sources, variance=variance,
        )
        # Eligible slices (support ≥ 5): long_input(0.52), short_input(0.85),
        # hindi(0.58). tiny_slice(0.30) drops out at support=2.
        # Worst eligible = long_input(0.52); below threshold → fail.
        self.assertFalse(result["passed"])
        self.assertEqual(result["worst_slice_id"], "long_input")
        self.assertAlmostEqual(result["actual"], 0.52, places=4)
        self.assertEqual(result["reason"], "worst_slice_below_threshold")
        self.assertEqual(result["min_slice_support"], DEFAULT_MIN_SLICE_SUPPORT)

    def test_per_slice_breakdown_includes_filtered_slice_with_flag(self):
        # Drill-down must show every slice, not just eligible ones —
        # the user needs to see "tiny_slice has 2 rows so it didn't
        # count" rather than mysteriously missing.
        values, sources, variance = _build_metric_snapshot(
            {"classification": _row_with_slices()}
        )
        result = _evaluate_gate(
            {"gate_id": "no_slice_below_60", "metric_id": "f1",
             "operator": "worst_slice_gte", "threshold": 0.60, "required": True},
            values=values, sources=sources, variance=variance,
        )
        breakdown = {sv["slice_id"]: sv for sv in result["per_slice_values"]}
        self.assertEqual(len(breakdown), 4)
        # Tiny slice is in the breakdown but flagged.
        self.assertTrue(breakdown["tiny_slice"]["below_min_support"])
        self.assertFalse(breakdown["long_input"]["below_min_support"])
        # Worst slice's row should be marked failing.
        self.assertFalse(breakdown["long_input"]["passes"])
        self.assertTrue(breakdown["short_input"]["passes"])

    def test_worst_slice_lte_uses_upper_bound_directionality(self):
        # lte ops gate from above — every slice must STAY UNDER.
        # Worst = maximum value. Build a row with an error-rate-style
        # metric to test directionality cleanly.
        row = _mock_row(
            eval_type="classification",
            metrics={
                "error_rate": 0.05,
                "per_slice": {
                    "easy": {"error_rate": 0.02, "support": 100},
                    "hard": {"error_rate": 0.18, "support": 100},
                    "tiny": {"error_rate": 0.50, "support": 1},  # filtered
                },
            },
        )
        values, sources, variance = _build_metric_snapshot({"classification": row})
        result = _evaluate_gate(
            {"gate_id": "no_slice_above_10", "metric_id": "error_rate",
             "operator": "worst_slice_lte", "threshold": 0.10, "required": True},
            values=values, sources=sources, variance=variance,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["worst_slice_id"], "hard")
        self.assertAlmostEqual(result["actual"], 0.18, places=4)
        self.assertEqual(result["reason"], "worst_slice_above_threshold")

    def test_custom_min_slice_support_changes_eligibility(self):
        # min_slice_support=100 excludes long_input (40) and hindi (30)
        # — leaving only short_input (200). Worst = 0.85, threshold 0.60
        # passes.
        values, sources, variance = _build_metric_snapshot(
            {"classification": _row_with_slices()}
        )
        result = _evaluate_gate(
            {"gate_id": "no_big_slice_below_60", "metric_id": "f1",
             "operator": "worst_slice_gte", "threshold": 0.60, "required": True,
             "min_slice_support": 100},
            values=values, sources=sources, variance=variance,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["worst_slice_id"], "short_input")
        self.assertEqual(result["min_slice_support"], 100)

    def test_no_eligible_slices_required_gate_fails(self):
        # Set min_slice_support absurdly high — nothing eligible. A
        # required gate fails with a specific reason; the UI should
        # surface "your gate's support floor excludes every slice."
        values, sources, variance = _build_metric_snapshot(
            {"classification": _row_with_slices()}
        )
        result = _evaluate_gate(
            {"gate_id": "impossible", "metric_id": "f1",
             "operator": "worst_slice_gte", "threshold": 0.60, "required": True,
             "min_slice_support": 10000},
            values=values, sources=sources, variance=variance,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "no_eligible_slices_required")
        self.assertIsNone(result["worst_slice_id"])

    def test_no_eligible_slices_optional_gate_passes(self):
        values, sources, variance = _build_metric_snapshot(
            {"classification": _row_with_slices()}
        )
        result = _evaluate_gate(
            {"gate_id": "optional", "metric_id": "f1",
             "operator": "worst_slice_gte", "threshold": 0.60, "required": False,
             "min_slice_support": 10000},
            values=values, sources=sources, variance=variance,
        )
        # Optional gates pass when the metric is missing — same shape
        # as the existing missing_metric_optional rule.
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "no_eligible_slices_optional")

    def test_variance_policy_applied_per_slice_under_lower_bound(self):
        # Multi-seed × slice composition. The lower-bound policy makes
        # the gate value = mean − std per slice. With slices long_input
        # (mean=0.62, std=0.04) the gate value is 0.58 — fails 0.60.
        # Without variance the mean (0.62) would have cleared.
        row = _mock_row(
            is_aggregate=True,
            seed_group_id="g2",
            metrics={
                "per_slice": {
                    "long_input": {
                        "f1": {"mean": 0.62, "std": 0.04, "min": 0.58, "max": 0.66, "n": 3},
                        "support": {"mean": 40, "std": 0.0, "min": 40, "max": 40, "n": 3},
                    },
                    "short_input": {
                        "f1": {"mean": 0.85, "std": 0.02, "min": 0.83, "max": 0.87, "n": 3},
                        "support": {"mean": 200, "std": 0.0, "min": 200, "max": 200, "n": 3},
                    },
                },
            },
        )
        values, sources, variance = _build_metric_snapshot({"classification": row})
        result = _evaluate_gate(
            {"gate_id": "honest_no_slice_below_60", "metric_id": "f1",
             "operator": "worst_slice_gte", "threshold": 0.60, "required": True},
            values=values, sources=sources, variance=variance,
        )
        # Lower-bound: long_input = 0.62 − 0.04 = 0.58; fails 0.60.
        self.assertFalse(result["passed"])
        self.assertEqual(result["worst_slice_id"], "long_input")
        self.assertAlmostEqual(result["actual"], 0.62, places=4)
        self.assertAlmostEqual(result["gate_value"], 0.58, places=4)
        self.assertEqual(result["variance_policy"], "lower_bound")


# ────────────────────────────────────────────────────────────────────────
# Catalog
# ────────────────────────────────────────────────────────────────────────


class CatalogSurfaceTests(unittest.TestCase):

    def test_new_operators_in_valid_set(self):
        self.assertIn("worst_slice_gte", VALID_GATE_OPERATORS)
        self.assertIn("worst_slice_lte", VALID_GATE_OPERATORS)
        # gte/lte still present — no contract regression.
        self.assertIn("gte", VALID_GATE_OPERATORS)
        self.assertIn("lte", VALID_GATE_OPERATORS)

    def test_is_per_slice_metric_id_accepts_all_three_shapes(self):
        # Canonical dot-path.
        self.assertTrue(is_per_slice_metric_id("per_slice.long_input.f1"))
        # Short form with infix.
        self.assertTrue(is_per_slice_metric_id("f1_slice_long_input"))
        # Eval-type scoped.
        self.assertTrue(is_per_slice_metric_id("classification.per_slice.long_input.f1"))
        # Adjacent shapes that should NOT match (would conflict with
        # per_class or unrelated metrics).
        self.assertFalse(is_per_slice_metric_id("f1"))
        self.assertFalse(is_per_slice_metric_id("f1_long_input"))
        self.assertFalse(is_per_slice_metric_id("per_class.benign.f1"))

    def test_validate_draft_pack_accepts_per_slice_metric_id(self):
        # Without the slice-aware metric-id recogniser, the per-slice
        # gate would round-trip as ``unknown_metric_id`` and break the
        # pack editor.
        draft = {
            "task_specs": [{
                "task_profile": "classification",
                "required_metric_ids": ["f1"],
                "metric_schema": {},
                "gates": [{
                    "gate_id": "min_f1_long_input",
                    "metric_id": "per_slice.long_input.f1",
                    "operator": "gte",
                    "threshold": 0.60,
                    "required": True,
                }],
            }],
        }
        validate_draft_pack_gates(draft)  # raises on failure

    def test_validate_draft_pack_accepts_worst_slice_gate(self):
        # worst_slice_gte takes the BASE metric_id (f1, not per_slice.*.f1).
        # The validator must not reject f1 just because per_slice.*.f1
        # is what the engine actually scans.
        draft = {
            "task_specs": [{
                "task_profile": "classification",
                "required_metric_ids": ["f1"],
                "metric_schema": {},
                "gates": [{
                    "gate_id": "no_slice_below_60",
                    "metric_id": "f1",
                    "operator": "worst_slice_gte",
                    "threshold": 0.60,
                    "required": True,
                }],
            }],
        }
        validate_draft_pack_gates(draft)  # raises on failure


if __name__ == "__main__":
    unittest.main()

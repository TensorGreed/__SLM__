"""Quality-Lift phase 6 slice 2 — Per-slice behavioral test gate plumbing.

Pins (slice 2: catalog matcher + flattener recursion + gate enricher
extension; ScorecardPanel surfacing lands in slice 3):

  Catalog matcher (is_behavioral_metric_id):
    * Accepts the three new per-slice id-shapes (canonical dot-path,
      short form with ``_slice_`` infix, eval-type scoped).
    * Rejects malformed adjacent shapes (e.g. missing slice_id,
      uppercase test_id).

  Flattener (_flatten_behavioral_test_metrics + _build_metric_snapshot):
    * Walks ``metrics["behavioral"][test_id]["per_slice"][slice_id]``
      blocks and emits all three id-shapes per (test, slice, metric).
    * Top-level metrics still emit unchanged (backward compat).
    * Missing per_slice block → only top-level keys.
    * Pin the closed metric leaf set {pass_rate, passed, total}.

  Gate enricher (_build_behavioral_index_for_checks +
  _attach_behavioral_details):
    * Index carries per_slice block forward so per-slice gates can
      resolve failed_examples without re-loading the EvalResult row.
    * Per-slice gate enrichment surfaces ``behavioral_slice_id`` AND
      pulls failed_examples from the slice's bucket (NOT the
      top-level test).
    * Per-slice gate with no recorded slice data (e.g. slice
      predicate matched zero rows) passes through unchanged.
    * Top-level behavioral gates continue to resolve as before
      (regression check).

  End-to-end:
    * A gate at ``behavioral.<test_id>.per_slice.<slice_id>.pass_rate``
      passes/fails correctly through _evaluate_gate with no new gate
      evaluator code (the variance / value-resolution path is
      already in place).
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.evaluation_gate_catalog import is_behavioral_metric_id  # noqa: E402
from app.services.evaluation_pack_service import (  # noqa: E402
    _attach_behavioral_details,
    _build_behavioral_index_for_checks,
    _build_metric_snapshot,
    _evaluate_gate,
)


def _mock_eval_result(
    *,
    eval_type: str = "classification",
    metrics: dict | None = None,
) -> MagicMock:
    row = MagicMock()
    row.id = 1
    row.eval_type = eval_type
    row.dataset_name = "held_out"
    row.metrics = metrics or {}
    row.pass_rate = None
    row.is_aggregate = False
    row.seed_group_id = None
    row.details = {}
    return row


def _behavioral_metrics_with_per_slice() -> dict:
    """Sample shape — phase 6 slice 1 runner output for one INV test
    that's healthy overall but failing on the ``long_input`` slice."""
    return {
        "behavioral": {
            "typo_invariance": {
                "kind": "INV",
                "pass_rate": 0.83,
                "passed": 50,
                "total": 60,
                "failed_examples": [
                    {
                        "original_input": "overall_failure_1",
                        "perturbed_input": "overall_failure_1_p",
                        "perturbation_name": "typo",
                        "original_label": "positive",
                        "perturbed_label": "negative",
                    },
                ],
                "per_slice": {
                    "long_input": {
                        "kind": "INV",
                        "pass_rate": 0.55,
                        "passed": 11,
                        "total": 20,
                        "failed_examples": [
                            {
                                "original_input": "long_specific_failure",
                                "perturbed_input": "lnog_specific_failure",
                                "perturbation_name": "typo",
                                "original_label": "positive",
                                "perturbed_label": "negative",
                            },
                        ],
                    },
                    "short_input": {
                        "kind": "INV",
                        "pass_rate": 0.97,
                        "passed": 39,
                        "total": 40,
                        "failed_examples": [],
                    },
                },
            },
        },
    }


# ────────────────────────────────────────────────────────────────────────
# Catalog matcher
# ────────────────────────────────────────────────────────────────────────


class CatalogMatcherPerSliceTests(unittest.TestCase):

    def test_accepts_canonical_per_slice_dot_path(self):
        for suffix in ("pass_rate", "passed", "total"):
            with self.subTest(suffix=suffix):
                self.assertTrue(
                    is_behavioral_metric_id(
                        f"behavioral.typo_invariance.per_slice.long_input.{suffix}"
                    )
                )

    def test_accepts_per_slice_short_form(self):
        # ``_slice_`` infix distinguishes from per_class's
        # ``<metric>_<label>``.
        self.assertTrue(
            is_behavioral_metric_id(
                "pass_rate_behavioral_typo_invariance_slice_long_input"
            )
        )

    def test_accepts_eval_type_scoped_per_slice(self):
        self.assertTrue(
            is_behavioral_metric_id(
                "classification.behavioral.typo_invariance.per_slice.long_input.pass_rate"
            )
        )

    def test_rejects_malformed_per_slice_shapes(self):
        # Empty slice_id, missing metric suffix, top-level confusion.
        self.assertFalse(
            is_behavioral_metric_id("behavioral.test.per_slice..pass_rate"),
        )
        self.assertFalse(
            is_behavioral_metric_id("behavioral.test.per_slice.long_input"),
        )
        self.assertFalse(
            is_behavioral_metric_id("behavioral.test.per_slice.long_input.confidence"),
        )

    def test_top_level_shape_still_recognised(self):
        # Backward compat: the slice 2 of phase 5 patterns must still
        # match. If a future regression rewrites these, both phase 5
        # tests AND this guard scream.
        self.assertTrue(is_behavioral_metric_id("behavioral.test.pass_rate"))
        self.assertTrue(is_behavioral_metric_id("pass_rate_behavioral_test"))
        self.assertTrue(
            is_behavioral_metric_id("classification.behavioral.test.pass_rate"),
        )


# ────────────────────────────────────────────────────────────────────────
# Flattener
# ────────────────────────────────────────────────────────────────────────


class FlattenerPerSliceTests(unittest.TestCase):

    def test_per_slice_canonical_keys_emitted(self):
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        values, _, _ = _build_metric_snapshot({"classification": row})

        # Top-level metrics still emit (backward compat).
        self.assertAlmostEqual(
            values["behavioral.typo_invariance.pass_rate"], 0.83, places=4,
        )

        # Phase 6 slice 2 — per-slice keys emitted.
        self.assertAlmostEqual(
            values["behavioral.typo_invariance.per_slice.long_input.pass_rate"],
            0.55, places=4,
        )
        self.assertAlmostEqual(
            values["behavioral.typo_invariance.per_slice.short_input.pass_rate"],
            0.97, places=4,
        )
        self.assertEqual(
            values["behavioral.typo_invariance.per_slice.long_input.passed"],
            11.0,
        )
        self.assertEqual(
            values["behavioral.typo_invariance.per_slice.long_input.total"],
            20.0,
        )

    def test_per_slice_short_and_scoped_keys_emitted(self):
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        values, _, _ = _build_metric_snapshot({"classification": row})

        self.assertAlmostEqual(
            values["pass_rate_behavioral_typo_invariance_slice_long_input"],
            0.55, places=4,
        )
        self.assertAlmostEqual(
            values[
                "classification.behavioral.typo_invariance.per_slice.long_input.pass_rate"
            ],
            0.55, places=4,
        )

    def test_test_without_per_slice_block_only_emits_top_level(self):
        # Phase 5 slice 2 output (no per_slice block) — only top-level
        # keys, no per_slice.* keys.
        row = _mock_eval_result(metrics={
            "behavioral": {
                "typo_invariance": {
                    "kind": "INV",
                    "pass_rate": 0.83,
                    "passed": 50,
                    "total": 60,
                    "failed_examples": [],
                },
            },
        })
        values, _, _ = _build_metric_snapshot({"classification": row})
        self.assertAlmostEqual(
            values["behavioral.typo_invariance.pass_rate"], 0.83, places=4,
        )
        # No per_slice.* keys at all.
        per_slice_keys = [
            k for k in values
            if "per_slice" in k and k.startswith("behavioral")
        ]
        self.assertEqual(per_slice_keys, [])

    def test_per_slice_non_numeric_leaves_skipped(self):
        # The runner emits ``kind`` and ``failed_examples`` per slice
        # too; these must NOT flatten into the gate snapshot.
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        values, _, _ = _build_metric_snapshot({"classification": row})
        # Non-numeric leaves stay out of the gate-resolvable map.
        self.assertNotIn(
            "behavioral.typo_invariance.per_slice.long_input.kind", values,
        )
        self.assertNotIn(
            "behavioral.typo_invariance.per_slice.long_input.failed_examples",
            values,
        )


# ────────────────────────────────────────────────────────────────────────
# Gate enricher — _build_behavioral_index_for_checks + _attach
# ────────────────────────────────────────────────────────────────────────


class GateEnricherPerSliceTests(unittest.TestCase):

    def test_index_carries_per_slice_block_forward(self):
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        index = _build_behavioral_index_for_checks({"classification": row})
        details = index["typo_invariance"]
        self.assertIn("per_slice", details)
        self.assertIn("long_input", details["per_slice"])
        long_slice = details["per_slice"]["long_input"]
        # The slice block's own counts + failed_examples carried through.
        self.assertEqual(long_slice["passed"], 11)
        self.assertEqual(long_slice["total"], 20)
        self.assertEqual(len(long_slice["failed_examples"]), 1)
        self.assertEqual(
            long_slice["failed_examples"][0]["original_input"],
            "long_specific_failure",
        )

    def test_per_slice_gate_resolves_to_slice_specific_failed_examples(self):
        # A gate targeting the long_input slice must pull its
        # failed_examples from THAT slice — NOT the top-level test's
        # overall failed_examples list. This is the whole point of
        # gating per-slice: the user wants to see "what's failing in
        # long_input specifically" not "what's failing test-wide".
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        index = _build_behavioral_index_for_checks({"classification": row})

        check = {
            "gate_id": "typo_invariance_long_input_gate",
            "metric_id": "behavioral.typo_invariance.per_slice.long_input.pass_rate",
            "operator": "gte",
            "threshold": 0.85,
            "required": True,
            "actual": 0.55,
            "passed": False,
            "reason": "below_threshold",
        }
        enriched = _attach_behavioral_details(check, index)

        self.assertEqual(enriched["behavioral_test_id"], "typo_invariance")
        self.assertEqual(enriched["behavioral_slice_id"], "long_input")
        self.assertEqual(enriched["behavioral_kind"], "INV")
        # Slice-specific counts (NOT 50/60).
        self.assertEqual(enriched["behavioral_passed"], 11)
        self.assertEqual(enriched["behavioral_total"], 20)
        # Slice-specific failed_examples — NOT the top-level
        # ``overall_failure_1`` entry.
        self.assertEqual(len(enriched["behavioral_failed_examples"]), 1)
        self.assertEqual(
            enriched["behavioral_failed_examples"][0]["original_input"],
            "long_specific_failure",
        )

    def test_per_slice_gate_resolves_via_eval_type_scoped_metric_id(self):
        # The eval-type scoped variant must also reach the per-slice
        # block (mirrors slice 3 of phase 5's behavior for the
        # top-level metric_id).
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        index = _build_behavioral_index_for_checks({"classification": row})
        check = {
            "gate_id": "typo_invariance_long_input_scoped_gate",
            "metric_id": "classification.behavioral.typo_invariance.per_slice.long_input.pass_rate",
            "operator": "gte",
            "threshold": 0.85,
            "required": True,
            "actual": 0.55,
            "passed": False,
            "reason": "below_threshold",
        }
        enriched = _attach_behavioral_details(check, index)
        self.assertEqual(enriched["behavioral_test_id"], "typo_invariance")
        self.assertEqual(enriched["behavioral_slice_id"], "long_input")

    def test_top_level_gate_still_resolves_to_top_level_data(self):
        # Regression check: with per_slice blocks now in the index, a
        # top-level gate must STILL get the top-level failed_examples
        # (not slip into per_slice resolution by accident).
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        index = _build_behavioral_index_for_checks({"classification": row})
        check = {
            "gate_id": "typo_invariance_overall_gate",
            "metric_id": "behavioral.typo_invariance.pass_rate",
            "operator": "gte",
            "threshold": 0.85,
            "required": True,
            "actual": 0.83,
            "passed": False,
            "reason": "below_threshold",
        }
        enriched = _attach_behavioral_details(check, index)
        self.assertEqual(enriched["behavioral_test_id"], "typo_invariance")
        # No slice id on a top-level gate.
        self.assertNotIn("behavioral_slice_id", enriched)
        # Top-level counts + failed_examples.
        self.assertEqual(enriched["behavioral_passed"], 50)
        self.assertEqual(enriched["behavioral_total"], 60)
        self.assertEqual(
            enriched["behavioral_failed_examples"][0]["original_input"],
            "overall_failure_1",
        )

    def test_per_slice_gate_with_missing_slice_data_passes_through(self):
        # Gate references a slice that's not in the per_slice block
        # (e.g. pack edited to add a new slice predicate but the eval
        # hasn't re-run). The check passes through unchanged so the
        # gate evaluator still resolves it via missing_metric_*.
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        index = _build_behavioral_index_for_checks({"classification": row})
        check = {
            "gate_id": "unknown_slice_gate",
            "metric_id": "behavioral.typo_invariance.per_slice.future_slice.pass_rate",
            "operator": "gte",
            "threshold": 0.85,
            "required": True,
            "actual": None,
            "passed": False,
            "reason": "missing_metric_required",
        }
        enriched = _attach_behavioral_details(check, index)
        # No behavioral fields added — slice has no recorded data.
        self.assertNotIn("behavioral_test_id", enriched)
        self.assertNotIn("behavioral_slice_id", enriched)


# ────────────────────────────────────────────────────────────────────────
# End-to-end gate evaluation
# ────────────────────────────────────────────────────────────────────────


class EndToEndPerSliceGateTests(unittest.TestCase):

    def test_per_slice_gate_fails_below_threshold(self):
        # The whole point of phase 6: a gate gating
        # ``per_slice.<slice>.pass_rate`` must fail ship the same way
        # any metric-based gate does. With the slice 2 flattener +
        # _evaluate_gate's existing value-lookup path, this requires
        # ZERO new gate evaluator code.
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        values, sources, variance = _build_metric_snapshot({"classification": row})
        result = _evaluate_gate(
            {
                "gate_id": "min_pass_long_input",
                "metric_id": "behavioral.typo_invariance.per_slice.long_input.pass_rate",
                "operator": "gte",
                "threshold": 0.85,
                "required": True,
            },
            values=values, sources=sources, variance=variance,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "below_threshold")
        self.assertAlmostEqual(result["actual"], 0.55, places=4)

    def test_per_slice_gate_passes_when_above_threshold(self):
        # The healthy slice (short_input @ 0.97) clears a 0.85 gate.
        row = _mock_eval_result(metrics=_behavioral_metrics_with_per_slice())
        values, sources, variance = _build_metric_snapshot({"classification": row})
        result = _evaluate_gate(
            {
                "gate_id": "min_pass_short_input",
                "metric_id": "behavioral.typo_invariance.per_slice.short_input.pass_rate",
                "operator": "gte",
                "threshold": 0.85,
                "required": True,
            },
            values=values, sources=sources, variance=variance,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "ok")
        self.assertAlmostEqual(result["actual"], 0.97, places=4)


if __name__ == "__main__":
    unittest.main()

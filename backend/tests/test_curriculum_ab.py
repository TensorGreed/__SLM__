"""Tests for the Phase 6c A/B harness (pure-function pieces).

The actual subprocess train.py loop is not unit-tested here — that's
what the end-to-end run produces. These tests exercise the parts the
harness's correctness depends on:

  * 70/15/15 split is deterministic + matches demo_project_service.
  * Classification-row flatten reads input.<text-field> + expected.label.
  * TemplateSummary stats handle the n=0, n=1, n>1 cases sanely.
  * apply_gate enforces both the lift threshold AND the non-
    overlapping-bands criterion.
  * Markdown formatter renders failed runs explicitly so a partial
    A/B doesn't silently mask broken runs.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.curriculum_ab import (  # noqa: E402
    CLASSIFICATION_TEMPLATES,
    GATE_MIN_LIFT_PCT,
    RunResult,
    TemplateSummary,
    _flatten_classification_row,
    _split_70_15_15,
    aggregate_results,
    apply_gate,
    format_markdown_block,
    prepare_template_splits,
)


# ─────────────────────────────────────────────────────────────────────
# Split + flatten
# ─────────────────────────────────────────────────────────────────────


class SplitAndFlattenTests(unittest.TestCase):
    def test_split_70_15_15_on_200_rows(self):
        rows = [{"i": i} for i in range(200)]
        train, val, test = _split_70_15_15(rows)
        # 200/7 = 28 → n_val = n_test = 28, n_train = 144.
        self.assertEqual(len(train), 144)
        self.assertEqual(len(val), 28)
        self.assertEqual(len(test), 28)
        # Deterministic + non-overlapping.
        self.assertEqual(train[0]["i"], 0)
        self.assertEqual(val[0]["i"], 144)
        self.assertEqual(test[0]["i"], 172)

    def test_split_handles_tiny_input(self):
        # 2 rows → all-train, empty val + test (matches demo_project_service).
        train, val, test = _split_70_15_15([{"i": 1}, {"i": 2}])
        self.assertEqual(train, [{"i": 1}, {"i": 2}])
        self.assertEqual(val, [])
        self.assertEqual(test, [])

    def test_flatten_classification_row_from_ticket_router_shape(self):
        row = {
            "key": "g001",
            "input": {"ticket": "Refund please"},
            "expected": {"label": "billing"},
            "rationale": "ignored",
        }
        flat = _flatten_classification_row(row)
        self.assertEqual(flat, {"text": "Refund please", "label": "billing"})

    def test_flatten_classification_row_from_log_triage_shape(self):
        row = {
            "key": "g002",
            "input": {"log_line": "[ERR] disk full"},
            "expected": {"label": "critical"},
        }
        flat = _flatten_classification_row(row)
        self.assertEqual(flat, {"text": "[ERR] disk full", "label": "critical"})

    def test_flatten_drops_rows_missing_text_or_label(self):
        # Missing label.
        self.assertIsNone(_flatten_classification_row(
            {"input": {"ticket": "x"}, "expected": {}}
        ))
        # Missing text.
        self.assertIsNone(_flatten_classification_row(
            {"input": {}, "expected": {"label": "y"}}
        ))
        # Wrong shape entirely.
        self.assertIsNone(_flatten_classification_row(
            {"text": "flat already", "label": "L"}
        ))


# ─────────────────────────────────────────────────────────────────────
# Live template read — guards against bit-rot if the templates ever
# add a new shape the harness doesn't know how to flatten.
# ─────────────────────────────────────────────────────────────────────


class LiveTemplatePrepTests(unittest.TestCase):
    """Reads the actual template gold files. If a template's shape
    drifts away from the harness's flatten() expectations, this test
    catches it before the A/B run wastes hours producing zeros."""

    def test_classification_templates_flatten_at_least_95pct(self):
        for slug in CLASSIFICATION_TEMPLATES:
            with self.subTest(template=slug), TemporaryDirectory() as td:
                counts = prepare_template_splits(slug, Path(td))
                # 70/15/15 → train ≈ 140, val ≈ 28, test ≈ 28 on 200 rows.
                # We allow a tiny drop in case some rows were
                # malformed, but ≥ 95% must survive the flatten.
                self.assertGreaterEqual(
                    counts["train"], 130,
                    f"{slug}: train rows dropped suspiciously low ({counts['train']}/200)",
                )
                self.assertGreater(counts["val"], 0)
                self.assertGreater(counts["test"], 0)
                # Spot-check the on-disk files have the flat shape.
                train_path = Path(td) / "train.jsonl"
                with train_path.open() as f:
                    row = json.loads(f.readline())
                self.assertIn("text", row)
                self.assertIn("label", row)


# ─────────────────────────────────────────────────────────────────────
# Stats: TemplateSummary handles 0 / 1 / N seed cases
# ─────────────────────────────────────────────────────────────────────


class TemplateSummaryStatsTests(unittest.TestCase):
    def test_empty_summary_returns_none_for_means(self):
        s = TemplateSummary(template="x")
        self.assertIsNone(s.on_mean)
        self.assertIsNone(s.off_mean)
        self.assertEqual(s.on_std, 0.0)
        self.assertEqual(s.off_std, 0.0)
        self.assertFalse(s.bands_non_overlapping)

    def test_single_seed_uses_zero_std(self):
        """stdev is undefined for n<2; we treat as 0 so the gate
        doesn't pass on a single-seed run by sheer chance."""
        s = TemplateSummary(template="x", on_f1s=[0.8], off_f1s=[0.7])
        self.assertEqual(s.on_std, 0.0)
        self.assertEqual(s.off_std, 0.0)
        # mean lift is still computed.
        self.assertAlmostEqual(s.absolute_lift, 0.1, places=6)

    def test_multi_seed_lift_and_band_separation(self):
        # 3 seeds each — tightly clustered, big lift.
        s = TemplateSummary(
            template="x",
            on_f1s=[0.81, 0.82, 0.83],   # mean 0.82, std 0.01
            off_f1s=[0.70, 0.71, 0.69],  # mean 0.70, std 0.01
        )
        self.assertAlmostEqual(s.on_mean, 0.82, places=2)
        self.assertAlmostEqual(s.off_mean, 0.70, places=2)
        self.assertAlmostEqual(s.absolute_lift, 0.12, places=2)
        self.assertGreater(s.relative_lift_pct, 17.0)  # ≈ 17.1%
        # 0.82 - 0.01 = 0.81 > 0.70 + 0.01 = 0.71 → non-overlapping.
        self.assertTrue(s.bands_non_overlapping)

    def test_overlapping_bands_when_variance_high(self):
        s = TemplateSummary(
            template="x",
            on_f1s=[0.60, 0.85, 0.75],
            off_f1s=[0.55, 0.80, 0.70],
        )
        self.assertFalse(s.bands_non_overlapping)


# ─────────────────────────────────────────────────────────────────────
# Gate decision
# ─────────────────────────────────────────────────────────────────────


class GateDecisionTests(unittest.TestCase):
    def _summary(self, slug: str, on: list[float], off: list[float]) -> TemplateSummary:
        return TemplateSummary(template=slug, on_f1s=on, off_f1s=off)

    def test_gate_passes_when_both_templates_lift_with_separated_bands(self):
        summaries = {
            "ticket-router": self._summary("ticket-router",
                on=[0.85, 0.86, 0.87], off=[0.70, 0.71, 0.72]),
            "log-triage": self._summary("log-triage",
                on=[0.80, 0.81, 0.82], off=[0.65, 0.66, 0.67]),
        }
        decision = apply_gate(summaries)
        self.assertTrue(decision.passed, decision.reason)
        for slug in summaries:
            self.assertTrue(decision.per_template[slug]["passed"])

    def test_gate_fails_when_one_template_lifts_below_threshold(self):
        summaries = {
            "ticket-router": self._summary("ticket-router",
                on=[0.85, 0.86, 0.87], off=[0.70, 0.71, 0.72]),
            # log-triage only lifts 1%.
            "log-triage": self._summary("log-triage",
                on=[0.71, 0.72, 0.73], off=[0.70, 0.71, 0.72]),
        }
        decision = apply_gate(summaries)
        self.assertFalse(decision.passed)
        self.assertIn("log-triage", decision.reason)
        self.assertIn(f"< {GATE_MIN_LIFT_PCT}%", decision.reason)

    def test_gate_fails_when_bands_overlap_despite_lift(self):
        # On-mean shoots up but the on-distribution is so wide its
        # lower-1σ touches off's upper-1σ.
        summaries = {
            "ticket-router": self._summary("ticket-router",
                on=[0.60, 0.95, 0.80], off=[0.65, 0.70, 0.75]),
            "log-triage": self._summary("log-triage",
                on=[0.85, 0.86, 0.87], off=[0.70, 0.71, 0.72]),
        }
        decision = apply_gate(summaries)
        self.assertFalse(decision.passed)
        self.assertIn("ticket-router", decision.reason)
        self.assertIn("bands overlap", decision.reason)

    def test_gate_fails_on_empty_summaries(self):
        decision = apply_gate({})
        self.assertFalse(decision.passed)
        self.assertIn("no successful runs", decision.reason)


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────


class AggregateResultsTests(unittest.TestCase):
    def test_groups_by_template_and_curriculum_flag(self):
        runs = [
            RunResult(template="ticket-router", seed=0, curriculum=False, macro_f1=0.70, eval_loss=None, train_runtime_seconds=10, output_dir=""),
            RunResult(template="ticket-router", seed=0, curriculum=True,  macro_f1=0.85, eval_loss=None, train_runtime_seconds=10, output_dir=""),
            RunResult(template="ticket-router", seed=1, curriculum=False, macro_f1=0.71, eval_loss=None, train_runtime_seconds=10, output_dir=""),
            RunResult(template="ticket-router", seed=1, curriculum=True,  macro_f1=0.86, eval_loss=None, train_runtime_seconds=10, output_dir=""),
            RunResult(template="log-triage",    seed=0, curriculum=False, macro_f1=0.60, eval_loss=None, train_runtime_seconds=10, output_dir=""),
            RunResult(template="log-triage",    seed=0, curriculum=True,  macro_f1=0.72, eval_loss=None, train_runtime_seconds=10, output_dir=""),
        ]
        summaries = aggregate_results(runs)
        self.assertEqual(set(summaries.keys()), {"ticket-router", "log-triage"})
        self.assertEqual(len(summaries["ticket-router"].on_f1s), 2)
        self.assertEqual(len(summaries["ticket-router"].off_f1s), 2)
        self.assertEqual(len(summaries["log-triage"].on_f1s), 1)
        self.assertEqual(len(summaries["log-triage"].off_f1s), 1)

    def test_drops_failed_runs_silently(self):
        runs = [
            RunResult(template="t", seed=0, curriculum=False, macro_f1=None, eval_loss=None, train_runtime_seconds=10, output_dir="", error="OOM"),
            RunResult(template="t", seed=0, curriculum=True,  macro_f1=0.80, eval_loss=None, train_runtime_seconds=10, output_dir=""),
        ]
        summaries = aggregate_results(runs)
        # The failed run is dropped → off_f1s is empty, on_f1s has one entry.
        self.assertEqual(summaries["t"].off_f1s, [])
        self.assertEqual(summaries["t"].on_f1s, [0.80])


# ─────────────────────────────────────────────────────────────────────
# Markdown output
# ─────────────────────────────────────────────────────────────────────


class MarkdownFormatterTests(unittest.TestCase):
    def test_renders_full_table_and_pass_gate_message(self):
        runs = []
        summaries = {
            "ticket-router": TemplateSummary(
                template="ticket-router",
                on_f1s=[0.85, 0.86, 0.87],
                off_f1s=[0.70, 0.71, 0.72],
            ),
            "log-triage": TemplateSummary(
                template="log-triage",
                on_f1s=[0.80, 0.81, 0.82],
                off_f1s=[0.65, 0.66, 0.67],
            ),
        }
        gate = apply_gate(summaries)
        md = format_markdown_block(
            runs, summaries, gate,
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            num_epochs=3,
            seeds=[0, 1, 2],
        )
        self.assertIn("ticket-router", md)
        self.assertIn("log-triage", md)
        self.assertIn("Gate: PASS", md)
        # Lift surfaced with sign + 2 decimals.
        self.assertIn("+", md)
        self.assertIn("Phase 6d cleared to ship", md)

    def test_renders_failed_runs_under_the_table(self):
        runs = [
            RunResult(
                template="ticket-router", seed=2, curriculum=True,
                macro_f1=None, eval_loss=None, train_runtime_seconds=5,
                output_dir="", error="CUDA out of memory in epoch 1",
            )
        ]
        summaries = {
            "ticket-router": TemplateSummary(
                template="ticket-router",
                on_f1s=[0.85, 0.86], off_f1s=[0.70, 0.71, 0.72],
            ),
            "log-triage": TemplateSummary(
                template="log-triage",
                on_f1s=[0.80, 0.81, 0.82], off_f1s=[0.65, 0.66, 0.67],
            ),
        }
        gate = apply_gate(summaries)
        md = format_markdown_block(
            runs, summaries, gate,
            base_model="m", num_epochs=3, seeds=[0, 1, 2],
        )
        self.assertIn("1 run(s) failed", md)
        self.assertIn("CUDA out of memory", md)

    def test_renders_fail_message_when_gate_did_not_pass(self):
        summaries = {
            "ticket-router": TemplateSummary(
                template="ticket-router",
                on_f1s=[0.71, 0.72, 0.73],   # only +1.4% lift
                off_f1s=[0.70, 0.71, 0.72],
            ),
            "log-triage": TemplateSummary(
                template="log-triage",
                on_f1s=[0.85, 0.86, 0.87], off_f1s=[0.65, 0.66, 0.67],
            ),
        }
        gate = apply_gate(summaries)
        md = format_markdown_block(
            [], summaries, gate, base_model="m", num_epochs=3, seeds=[0, 1, 2],
        )
        self.assertIn("Gate: FAIL", md)
        self.assertIn("Epic 6 stops at Phase 6b", md)


if __name__ == "__main__":
    unittest.main()

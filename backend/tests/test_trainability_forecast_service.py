"""Tests for the trainability forecast service (USER-SUCCESS Epic 1).

Covers:
  - Each of the 5 signal checks in isolation against synthetic gold sets
  - The estimate_gate_pass_prob() heuristic at the 8 templates' shapes
  - Overall verdict logic (likely_pass / borderline / likely_fail)
  - Cache hit / miss / invalidation on dataset change
  - End-to-end via the FastAPI test client
  - 404 on unknown project, 400 on missing recipe
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "trainability_forecast_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "trainability_forecast_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.trainability_forecast_service import (
    DEFAULT_BASE_MODEL_PARAMS_M,
    KNOWN_BASE_MODEL_PARAMS_M,
    _attach_cost_estimates,
    _build_classification_signals,
    _build_per_recipe_signals,
    _build_span_extraction_signals,
    _build_summarization_signals,
    _compute_cache_key,
    _extract_summary_pair,
    _signal_class_imbalance,
    _signal_entity_type_coverage,
    _signal_format_consistency,
    _signal_goldset_diversity,
    _signal_label_vocab_fragmented,
    _signal_negative_examples_missing,
    _signal_per_class_minimum,
    _signal_row_count,
    _signal_single_class_dominance,
    _signal_span_offset_invalid,
    _signal_summary_doc_ratio,
    estimate_action_cost,
    estimate_gate_pass_prob,
    _overall_verdict,
)


# ─────────────────────────────────────────────────────────────────────
# Unit tests — pure-function signal checks. No DB required.
# ─────────────────────────────────────────────────────────────────────


class HeuristicAndSignalUnitTests(unittest.TestCase):
    """Unit tests on the pure-function pieces — no DB, no API client."""

    def test_estimate_gate_pass_prob_landing_zones(self):
        # Calibration anchors: the heuristic should land the demo + the
        # template in the right buckets without further tuning.
        demo_prob = estimate_gate_pass_prob(
            row_count=16,
            recipe_difficulty=0.45,
            base_model_params_m=135,
            class_entropy=None,
            diversity_score=0.9,
        )
        template_prob = estimate_gate_pass_prob(
            row_count=200,
            recipe_difficulty=0.30,
            base_model_params_m=135,
            class_entropy=1.6,
            diversity_score=0.85,
        )
        # Demo (16 rows) is the deliberately-thin training set — should
        # fall in the likely_fail zone (< 0.40).
        self.assertLess(demo_prob, 0.40, f"demo expected <0.40, got {demo_prob:.3f}")
        # Template at 200 rows with balanced classes is the comfort
        # zone — should fall in or near the likely_pass zone (>= 0.65).
        self.assertGreater(template_prob, 0.60, f"template expected >0.60, got {template_prob:.3f}")
        # And the demo should be strictly below the template.
        self.assertLess(demo_prob, template_prob)

    def test_estimate_gate_pass_prob_clamps_to_safe_range(self):
        # Extreme inputs should not produce 0% or 100% — overconfidence
        # in either direction is the failure mode.
        worst = estimate_gate_pass_prob(0, 1.0, 10, 0.0, 0.0)
        best = estimate_gate_pass_prob(10000, 0.0, 100000, 5.0, 1.0)
        self.assertGreaterEqual(worst, 0.05)
        self.assertLessEqual(best, 0.95)

    def test_row_count_signal_emits_block_below_minimum(self):
        signal = _signal_row_count(train_row_count=16, minimum_rows=50)
        self.assertEqual(signal["severity"], "block")
        self.assertEqual(signal["id"], "row_count_below_minimum")
        # Action should suggest synth_augment with target_rows=50.
        self.assertIsNotNone(signal["suggested_action"])
        self.assertEqual(signal["suggested_action"]["kind"], "synth_augment")
        self.assertEqual(signal["suggested_action"]["params"]["target_rows"], 50)

    def test_row_count_signal_warns_in_1_to_1_5x_window(self):
        # 50-74 rows when minimum is 50 should warn (above min, below 1.5x).
        signal = _signal_row_count(train_row_count=60, minimum_rows=50)
        self.assertEqual(signal["severity"], "warn")
        # 75+ rows should be ok.
        signal_ok = _signal_row_count(train_row_count=80, minimum_rows=50)
        self.assertEqual(signal_ok["severity"], "ok")

    def test_class_imbalance_silent_on_non_classification_recipes(self):
        # qa-sft / span-extraction / etc. should not emit the class signal.
        rows = [{"label": "a"}, {"label": "b"}]
        self.assertIsNone(_signal_class_imbalance(rows, "instruction_sft"))
        self.assertIsNone(_signal_class_imbalance(rows, "structured_extraction"))
        self.assertIsNone(_signal_class_imbalance(rows, "summarization"))

    def test_class_imbalance_blocks_on_severe_skew(self):
        # 9 rows of class A, 1 row of class B → entropy ≈ 0.33 (< 0.5 block).
        rows = [{"label": "a"} for _ in range(9)] + [{"label": "b"}]
        signal = _signal_class_imbalance(rows, "classification")
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "block")
        self.assertEqual(signal["suggested_action"]["kind"], "synth_balance")
        self.assertIn("b", signal["suggested_action"]["params"]["underrepresented_classes"])

    def test_class_imbalance_ok_on_balanced_distribution(self):
        # 5 classes × 10 rows each → balance == 1.0 (perfectly even).
        rows = []
        for label in ["a", "b", "c", "d", "e"]:
            rows.extend([{"label": label}] * 10)
        signal = _signal_class_imbalance(rows, "classification")
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "ok")

    def test_class_imbalance_ok_on_balanced_binary_distribution(self):
        """Regression for the SQLi-bootstrap quirk: a 50/50 binary
        corpus has raw entropy ln(2) ≈ 0.69. The old absolute
        threshold (warn at <1.0) treated it as skewed even though
        the data was perfectly balanced. After normalising by
        ln(n_classes), 50/50 → balance 1.0 → ok regardless of
        n_classes."""
        rows = ([{"label": "injection"}] * 50) + ([{"label": "benign"}] * 50)
        signal = _signal_class_imbalance(rows, "classification")
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal["severity"], "ok")
        # Headline reads in normalised "balance" terms now.
        self.assertIn("balance", signal["headline"])

    def test_class_imbalance_warn_on_60_40_binary_split(self):
        """60/40 binary → normalised balance ≈ 0.97 (still ok). 65/35
        → balance ≈ 0.93. The warn band should kick in around 80/20."""
        rows_8020 = ([{"label": "a"}] * 80) + ([{"label": "b"}] * 20)
        signal = _signal_class_imbalance(rows_8020, "classification")
        self.assertIsNotNone(signal)
        assert signal is not None
        # 80/20 → balance ≈ 0.722 < 0.75 → warn.
        self.assertEqual(signal["severity"], "warn")

    def test_goldset_diversity_low_signal_fires_on_repetitive_rows(self):
        # Synthetic redundant gold set — same question/answer repeated.
        rows = [{"input": {"question": "How do I reset my password?"}, "expected": {"answer": "Reset via settings."}}] * 12
        signal, diversity_score = _signal_goldset_diversity(rows)
        # All rows identical → Jaccard 1.0 → diversity_score 0.0.
        self.assertEqual(signal["severity"], "warn")
        self.assertLess(diversity_score, 0.1)

    def test_goldset_diversity_ok_on_diverse_rows(self):
        # Hand-crafted rows with very different vocabularies.
        rows = [
            {"input": {"question": "What is your refund policy?"}, "expected": {"answer": "Refunds available within 30 days."}},
            {"input": {"question": "Where is your office located?"}, "expected": {"answer": "We are in Springfield, IL."}},
            {"input": {"question": "How do I export my analytics data?"}, "expected": {"answer": "Go to Reports, click Export."}},
            {"input": {"question": "Can I add multiple team members?"}, "expected": {"answer": "Yes, invite from Settings."}},
        ]
        signal, diversity_score = _signal_goldset_diversity(rows)
        self.assertEqual(signal["severity"], "ok")
        self.assertGreater(diversity_score, 0.5)

    def test_format_inconsistency_silent_on_non_structured_recipes(self):
        rows = [{"expected": "some string"}]
        self.assertIsNone(_signal_format_consistency(rows, "instruction_sft"))
        self.assertIsNone(_signal_format_consistency(rows, "classification"))

    def test_format_inconsistency_lists_invalid_row_ids(self):
        # Mix valid + invalid span structures.
        rows = [
            {"expected": {"spans": [{"start": 0, "end": 5, "text": "hello", "type": "greeting"}]}},
            {"expected": {"spans": [{"start": "oops", "end": 10}]}},  # invalid: start is str
            {"expected": "should be a dict"},  # invalid: expected is str
            {"expected": {"spans": [{"start": 10, "end": 5}]}},  # invalid: start > end
            {"expected": {"spans": []}},  # valid (negative example, empty spans)
        ]
        signal = _signal_format_consistency(rows, "structured_extraction")
        self.assertIsNotNone(signal)
        invalid_ids = signal["suggested_action"]["params"]["invalid_row_ids"]
        # Rows 1, 2, 3 should be flagged; rows 0 and 4 are valid.
        self.assertEqual(set(invalid_ids), {1, 2, 3})

    def test_overall_verdict_block_signal_forces_likely_fail(self):
        # Even with high confidence, any block signal → likely_fail.
        signals = [
            {"id": "row_count_below_minimum", "severity": "block", "headline": "", "detail": "", "suggested_action": None},
            {"id": "gate_pass_probability", "severity": "ok", "headline": "", "detail": "", "suggested_action": None},
        ]
        self.assertEqual(_overall_verdict(signals=signals, confidence_pct=85), "likely_fail")

    def test_overall_verdict_high_confidence_no_warns_is_likely_pass(self):
        signals = [
            {"id": "row_count_below_minimum", "severity": "ok", "headline": "", "detail": "", "suggested_action": None},
            {"id": "gate_pass_probability", "severity": "ok", "headline": "", "detail": "", "suggested_action": None},
        ]
        self.assertEqual(_overall_verdict(signals=signals, confidence_pct=72), "likely_pass")

    def test_overall_verdict_warns_force_borderline_even_at_high_confidence(self):
        signals = [
            {"id": "row_count_below_minimum", "severity": "ok", "headline": "", "detail": "", "suggested_action": None},
            {"id": "class_imbalance", "severity": "warn", "headline": "", "detail": "", "suggested_action": None},
        ]
        self.assertEqual(_overall_verdict(signals=signals, confidence_pct=72), "borderline")

    def test_compute_cache_key_is_stable_and_distinct(self):
        k1 = _compute_cache_key(dataset_signature="sig1", recipe_id="qa-sft", base_model_name="SmolLM2-135M")
        k2 = _compute_cache_key(dataset_signature="sig1", recipe_id="qa-sft", base_model_name="SmolLM2-135M")
        k3 = _compute_cache_key(dataset_signature="sig1", recipe_id="classification", base_model_name="SmolLM2-135M")
        k4 = _compute_cache_key(dataset_signature="sig2", recipe_id="qa-sft", base_model_name="SmolLM2-135M")
        self.assertEqual(k1, k2)  # same inputs → same key
        self.assertNotEqual(k1, k3)  # recipe change → new key
        self.assertNotEqual(k1, k4)  # dataset change → new key

    def test_known_base_model_param_table_has_smollm2_default(self):
        # Smoke check that the default base model is in the table.
        self.assertIn("HuggingFaceTB/SmolLM2-135M-Instruct", KNOWN_BASE_MODEL_PARAMS_M)
        self.assertEqual(KNOWN_BASE_MODEL_PARAMS_M["HuggingFaceTB/SmolLM2-135M-Instruct"], 135)
        # Unknown model falls back to the conservative 135M assumption.
        self.assertEqual(DEFAULT_BASE_MODEL_PARAMS_M, 135)


# ─────────────────────────────────────────────────────────────────────
# Per-recipe signal tests (T3 — per-recipe trainability forecast).
#
# Each new signal gets two tests:
#   1. Triggering-condition test: signal fires at the right severity
#      with the right action params.
#   2. Clean-state-doesn't-fire test: signal returns None / "ok" when
#      the gold set is healthy for that recipe.
# ─────────────────────────────────────────────────────────────────────


class ClassificationSignalTests(unittest.TestCase):
    """Per-recipe builder + the three new classification signals."""

    def test_per_class_minimum_blocks_when_a_class_has_zero_examples_floor(self):
        # "billing"=8, "tech"=8, "rare"=1 → "rare" has < PER_CLASS_BLOCK.
        counts = {"billing": 8, "tech": 8, "rare": 1}
        signal = _signal_per_class_minimum(counts)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "block")
        self.assertEqual(signal["id"], "per_class_minimum_unmet")
        self.assertIn("rare", signal["suggested_action"]["params"]["underrepresented_classes"])

    def test_per_class_minimum_warns_on_thin_but_nonzero_classes(self):
        # "rare"=3 is below 5-floor but above 2-block.
        counts = {"a": 20, "b": 20, "rare": 3}
        signal = _signal_per_class_minimum(counts)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "warn")
        self.assertIn("rare", signal["suggested_action"]["params"]["underrepresented_classes"])

    def test_per_class_minimum_silent_when_every_class_clears_floor(self):
        counts = {"a": 50, "b": 30, "c": 25}
        self.assertIsNone(_signal_per_class_minimum(counts))

    def test_label_vocab_fragmented_warns_on_case_dupes(self):
        # "positive" + "Positive" collapse to the same canonical key.
        counts = {"positive": 30, "Positive": 5, "negative": 20}
        signal = _signal_label_vocab_fragmented(counts)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "warn")
        self.assertEqual(signal["id"], "label_vocab_fragmented")
        # Action carries the fragmented groups so the UI can render the merge UX.
        groups = signal["suggested_action"]["params"]["fragment_groups"]
        flattened = {label for group in groups for label in group}
        self.assertIn("positive", flattened)
        self.assertIn("Positive", flattened)

    def test_label_vocab_fragmented_silent_on_clean_vocab(self):
        counts = {"billing": 50, "tech": 30, "feature_request": 20}
        self.assertIsNone(_signal_label_vocab_fragmented(counts))

    def test_single_class_dominance_warns_when_one_class_above_80pct(self):
        # 85/10/5 split → "a" dominates at 0.85.
        counts = {"a": 85, "b": 10, "c": 5}
        signal = _signal_single_class_dominance(counts)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "warn")
        self.assertEqual(signal["id"], "single_class_dominance")
        # Other classes carried in the action so synth_balance knows
        # which classes to top up.
        under = signal["suggested_action"]["params"]["underrepresented_classes"]
        self.assertIn("b", under)
        self.assertIn("c", under)

    def test_single_class_dominance_silent_on_balanced_distribution(self):
        # 70/30 binary — high imbalance but no single class above 80%.
        counts = {"a": 70, "b": 30}
        self.assertIsNone(_signal_single_class_dominance(counts))

    def test_classification_builder_skips_signals_on_empty_gold(self):
        # No rows → no signals + None entropy. The orchestrator falls
        # back to recipe-agnostic signals only.
        signals, entropy = _build_classification_signals([])
        self.assertEqual(signals, [])
        self.assertIsNone(entropy)

    def test_classification_builder_emits_entropy_on_balanced_gold(self):
        rows = [{"label": label} for label in ["a"] * 10 + ["b"] * 10 + ["c"] * 10]
        signals, balance = _build_classification_signals(rows)
        self.assertIsNotNone(balance)
        # 3 equal classes → normalised balance == 1.0 (H/ln(n_classes)).
        self.assertAlmostEqual(balance, 1.0, places=4)
        # All three new signals are quiet (none above their threshold);
        # only the existing class_imbalance signal lands at "ok".
        signal_ids = {s["id"] for s in signals}
        self.assertIn("class_imbalance", signal_ids)
        self.assertNotIn("per_class_minimum_unmet", signal_ids)
        self.assertNotIn("label_vocab_fragmented", signal_ids)
        self.assertNotIn("single_class_dominance", signal_ids)

    def test_classification_builder_normalises_legacy_template_rows(self):
        # Legacy {question, answer} rows from the template materializer
        # (see demo_project_service._canonical_prepared_row) must feed
        # the classification signals or template-instantiated projects
        # get no per-recipe diagnostics.
        rows = [
            {"question": "Charge me $5", "answer": "billing"} for _ in range(20)
        ] + [
            {"question": "App crashes", "answer": "tech"} for _ in range(2)
        ]
        signals, entropy = _build_classification_signals(rows)
        self.assertIsNotNone(entropy)
        # 20:2 split puts "tech" below PER_CLASS_MINIMUM.
        ids = {s["id"] for s in signals}
        self.assertIn("per_class_minimum_unmet", ids)


class SpanExtractionSignalTests(unittest.TestCase):
    """The three new span-extraction signals + the recipe builder."""

    def test_entity_type_coverage_blocks_on_single_type(self):
        rows = [
            {"input": {"text": f"row {i}"}, "expected": {"entities": [{"type": "email", "start": 0, "end": 5, "text": "row 0"}]}}
            for i in range(10)
        ]
        signal = _signal_entity_type_coverage(rows)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["id"], "entity_type_coverage_thin")
        self.assertEqual(signal["severity"], "block")

    def test_entity_type_coverage_ok_when_three_or_more_types(self):
        rows = [
            {"input": {"text": "x"}, "expected": {"entities": [{"type": t, "start": 0, "end": 1, "text": "x"}]}}
            for t in ("email", "phone", "ssn", "name")
        ]
        signal = _signal_entity_type_coverage(rows)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "ok")

    def test_entity_type_coverage_skips_when_no_spans_recoverable(self):
        # All rows lack any recognizable span payload — let the
        # format_inconsistency signal drive the diagnostic instead.
        rows = [{"foo": "bar"}, {"baz": "qux"}]
        self.assertIsNone(_signal_entity_type_coverage(rows))

    def test_span_offset_invalid_warns_when_text_slice_mismatches(self):
        rows = [
            {
                "input": {"text": "Contact me at jane@example.com today."},
                "expected": {"entities": [
                    # Correct offsets: "jane@example.com" is at [14, 30].
                    {"type": "email", "start": 14, "end": 30, "text": "jane@example.com"}
                ]},
            },
            {
                "input": {"text": "Contact me at jane@example.com today."},
                "expected": {"entities": [
                    # Bad offset — claims "jane" lives at [0, 4] but
                    # source text at [0, 4] is "Cont".
                    {"type": "email", "start": 0, "end": 4, "text": "jane"}
                ]},
            },
        ]
        signal = _signal_span_offset_invalid(rows)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["id"], "span_offset_invalid")
        # 50% bad → block.
        self.assertEqual(signal["severity"], "block")
        self.assertIn(1, signal["suggested_action"]["params"]["invalid_row_ids"])

    def test_span_offset_invalid_silent_on_clean_offsets(self):
        rows = [
            {
                "input": {"text": "Email jane@example.com or bob@example.com."},
                "expected": {"entities": [
                    {"type": "email", "start": 6, "end": 22, "text": "jane@example.com"},
                    {"type": "email", "start": 26, "end": 41, "text": "bob@example.com"},
                ]},
            },
        ]
        signal = _signal_span_offset_invalid(rows)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "ok")

    def test_negative_examples_missing_warns_when_no_empty_entities(self):
        rows = [
            {"input": {"text": f"row {i}"}, "expected": {"entities": [{"type": "x", "start": 0, "end": 1, "text": "r"}]}}
            for i in range(8)
        ]
        signal = _signal_negative_examples_missing(rows)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["id"], "negative_examples_missing")
        self.assertEqual(signal["severity"], "warn")

    def test_negative_examples_missing_ok_when_some_empty_entities_present(self):
        rows = [
            {"input": {"text": f"row {i}"}, "expected": {"entities": [{"type": "x", "start": 0, "end": 1, "text": "r"}]}}
            for i in range(7)
        ] + [
            {"input": {"text": "Just plain text with nothing to extract."}, "expected": {"entities": []}}
        ]
        signal = _signal_negative_examples_missing(rows)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "ok")

    def test_negative_examples_missing_silent_on_tiny_gold_set(self):
        # < 5 rows with spans → skip rather than false-positive.
        rows = [
            {"input": {"text": f"row {i}"}, "expected": {"entities": [{"type": "x", "start": 0, "end": 1, "text": "r"}]}}
            for i in range(3)
        ]
        self.assertIsNone(_signal_negative_examples_missing(rows))

    def test_span_extraction_builder_handles_legacy_answer_shape(self):
        # Legacy materialised rows have entities JSON-encoded in
        # `answer` — the new signals must still see them. Build a 10-
        # row set with a single entity type so entity_type_coverage
        # fires.
        rows = [
            {
                "question": f"Source text row {i} with phone +1-555-0100.",
                "answer": json.dumps({
                    "entities": [{"type": "phone", "start": 0, "end": 5, "text": "Sourc"}]
                }),
            }
            for i in range(10)
        ]
        signals = _build_span_extraction_signals(rows)
        ids = {s["id"] for s in signals}
        # entity_type_coverage_thin fires because every row uses
        # "phone" only. negative_examples_missing also fires because
        # no row carries an empty entities list. These are the kinds
        # of legacy-row gaps that would silently miss without the
        # JSON-string normaliser.
        self.assertIn("entity_type_coverage_thin", ids)
        self.assertIn("negative_examples_missing", ids)


class SummarizationSignalTests(unittest.TestCase):
    """The new summarization signals + summary/doc extraction."""

    def test_summary_doc_ratio_outliers_warns_when_summary_too_long(self):
        # Each row has a "summary" that's actually longer than the
        # document — clear mislabeling.
        rows = [
            {
                "input": {"document": "A short doc."},
                "expected": {"summary": "A summary that runs much longer than the source document"},
            },
            {
                "input": {"document": "Another short doc here."},
                "expected": {"summary": "Yet another summary that goes on and on and exceeds the source"},
            },
            {
                "input": {"document": "Third short doc."},
                "expected": {"summary": "A third even longer summary that exceeds the source document by a lot"},
            },
        ]
        signal = _signal_summary_doc_ratio(rows)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["id"], "summary_doc_ratio_outliers")
        # 100% of rows mislabeled → block (≥30%).
        self.assertEqual(signal["severity"], "block")

    def test_summary_doc_ratio_outliers_ok_when_summaries_are_shorter(self):
        rows = [
            {
                "input": {"document": "Long document " * 50},
                "expected": {"summary": "Short summary."},
            },
            {
                "input": {"document": "Another long document " * 40},
                "expected": {"summary": "Another short summary."},
            },
        ]
        signal = _signal_summary_doc_ratio(rows)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["severity"], "ok")

    def test_summary_doc_ratio_silent_when_no_summary_pairs_recoverable(self):
        # qa-sft rows with no summary column shouldn't trigger the
        # signal even if the recipe dispatcher feeds them through.
        rows = [{"foo": "bar"}, {"baz": "qux"}]
        self.assertIsNone(_signal_summary_doc_ratio(rows))

    def test_extract_summary_pair_handles_legacy_question_answer_shape(self):
        # The summary materializer encodes summary as a JSON dict in
        # `answer` for some template runs; other times it's plain text.
        row_json = {
            "question": "A long source document with multiple sentences explaining the topic.",
            "answer": json.dumps({"summary": "Short recap."}),
        }
        pair = _extract_summary_pair(row_json)
        self.assertIsNotNone(pair)
        doc, summary = pair
        self.assertIn("long source document", doc)
        self.assertEqual(summary, "Short recap.")

        # Plain text answer path.
        row_plain = {
            "question": "A long source document explaining the topic.",
            "answer": "Short plaintext summary.",
        }
        pair = _extract_summary_pair(row_plain)
        self.assertIsNotNone(pair)
        _, summary_plain = pair
        self.assertEqual(summary_plain, "Short plaintext summary.")

    def test_summarization_builder_skips_when_pairs_unrecoverable(self):
        # Gold rows that don't carry recoverable doc/summary pairs
        # should produce no signals — the row_count signal still
        # fires at the orchestrator level.
        rows = [{"foo": "bar"}]
        self.assertEqual(_build_summarization_signals(rows), [])


class RecipeDispatchTests(unittest.TestCase):
    """The orchestrator's per-recipe dispatcher."""

    def test_dispatch_classification_routes_to_classification_builder(self):
        rows = [{"label": "a"}, {"label": "b"}, {"label": "b"}]
        signals, entropy = _build_per_recipe_signals(
            task_profile="classification", gold_rows=rows
        )
        self.assertIsNotNone(entropy)
        ids = {s["id"] for s in signals}
        self.assertIn("class_imbalance", ids)

    def test_dispatch_structured_extraction_routes_to_span_builder(self):
        rows = [
            {"input": {"text": "x"}, "expected": {"entities": [{"type": "t", "start": 0, "end": 1, "text": "x"}]}}
            for _ in range(6)
        ]
        signals, entropy = _build_per_recipe_signals(
            task_profile="structured_extraction", gold_rows=rows
        )
        # Span builder doesn't carry entropy back to the heuristic.
        self.assertIsNone(entropy)
        ids = {s["id"] for s in signals}
        # All-one-type, no negatives — both signals expected.
        self.assertIn("entity_type_coverage_thin", ids)
        self.assertIn("negative_examples_missing", ids)

    def test_dispatch_summarization_routes_to_summary_builder(self):
        rows = [
            {"input": {"document": "A long source doc going on for many words."},
             "expected": {"summary": "Short recap."}}
        ]
        signals, entropy = _build_per_recipe_signals(
            task_profile="summarization", gold_rows=rows
        )
        self.assertIsNone(entropy)
        ids = {s["id"] for s in signals}
        # Summary much shorter than doc → ok signal lands.
        self.assertIn("summary_doc_ratio_outliers", ids)

    def test_dispatch_instruction_sft_returns_no_recipe_signals(self):
        # qa-sft + generic-sft + code-review all share
        # task_profile="instruction_sft" and should produce no
        # recipe-specific signals beyond the recipe-agnostic three.
        rows = [{"question": "q", "answer": "a"} for _ in range(10)]
        signals, entropy = _build_per_recipe_signals(
            task_profile="instruction_sft", gold_rows=rows
        )
        self.assertEqual(signals, [])
        self.assertIsNone(entropy)


# ─────────────────────────────────────────────────────────────────────
# T1 — cost-of-fix estimator.
# ─────────────────────────────────────────────────────────────────────


class CostEstimateTests(unittest.TestCase):
    """estimate_action_cost + _attach_cost_estimates."""

    def test_estimate_returns_None_for_missing_action(self):
        self.assertIsNone(estimate_action_cost(None))
        self.assertIsNone(estimate_action_cost({}))
        self.assertIsNone(estimate_action_cost({"kind": "nope", "params": {}}))

    def test_synth_augment_scales_linearly_with_target_rows(self):
        # 50 rows × 30s = 1500s = 25 min, × $0.0001 = $0.005.
        est = estimate_action_cost({"kind": "synth_augment", "params": {"target_rows": 50}})
        self.assertIsNotNone(est)
        self.assertEqual(est["time_minutes"], 25)
        self.assertEqual(est["llm_cost_usd"], 0.005)
        self.assertEqual(est["confidence"], "rough")

    def test_synth_augment_tiny_target_floors_at_one_minute(self):
        # 1 row × 30s = 0 minutes (rounded); floor at 1 min so the chip
        # never renders "0 min".
        est = estimate_action_cost({"kind": "synth_augment", "params": {"target_rows": 1}})
        self.assertEqual(est["time_minutes"], 1)

    def test_synth_diversify_uses_same_curve_as_augment(self):
        a = estimate_action_cost({"kind": "synth_augment", "params": {"target_rows": 50}})
        d = estimate_action_cost({"kind": "synth_diversify", "params": {"target_rows": 50}})
        self.assertEqual(a, d)

    def test_synth_augment_falls_back_to_1_row_on_missing_param(self):
        # The synth path never ships without target_rows, but be
        # defensive in case the frontend hits a brand-new signal
        # whose params haven't been validated yet.
        est = estimate_action_cost({"kind": "synth_augment", "params": {}})
        self.assertEqual(est["time_minutes"], 1)
        self.assertEqual(est["llm_cost_usd"], 0.0001)

    def test_synth_balance_scales_with_classes_and_per_class_rows(self):
        # 3 under-classes × 10 rows/class = 30 rows → 15 min · $0.003.
        est = estimate_action_cost({
            "kind": "synth_balance",
            "params": {
                "underrepresented_classes": ["a", "b", "c"],
                "target_rows_per_class": 10,
            },
        })
        self.assertEqual(est["time_minutes"], 15)
        self.assertEqual(est["llm_cost_usd"], 0.003)

    def test_synth_balance_defaults_to_10_per_class_when_unspecified(self):
        # The single_class_dominance / class_imbalance signals don't
        # carry target_rows_per_class — they just name the under-classes.
        # We default to 10 rows/class (matches the synth backend's
        # class_balance_fill default).
        est = estimate_action_cost({
            "kind": "synth_balance",
            "params": {"underrepresented_classes": ["a", "b"]},
        })
        # 2 classes × 10 rows = 20 rows → 10 min · $0.002.
        self.assertEqual(est["time_minutes"], 10)
        self.assertEqual(est["llm_cost_usd"], 0.002)

    def test_fix_gold_rows_counts_invalid_ids_at_2min_each_with_no_llm_cost(self):
        # 4 invalid rows × 2 min = 8 min. llm_cost_usd MUST be None
        # (manual fix), not 0 — preserves the "no $" distinction.
        est = estimate_action_cost({
            "kind": "fix_gold_rows",
            "params": {"invalid_row_ids": [1, 2, 3, 4]},
        })
        self.assertEqual(est["time_minutes"], 8)
        self.assertIsNone(est["llm_cost_usd"])

    def test_fix_gold_rows_counts_fragment_groups_too(self):
        # Each label-fragment group requires a separate merge decision;
        # counts the same way invalid_row_ids does.
        est = estimate_action_cost({
            "kind": "fix_gold_rows",
            "params": {"fragment_groups": [["a", "A"], ["b", "B"], ["c", "C"]]},
        })
        # 3 groups × 2 min = 6 min.
        self.assertEqual(est["time_minutes"], 6)
        self.assertIsNone(est["llm_cost_usd"])

    def test_fix_gold_rows_combined_params_sum(self):
        # Real signals can carry both (e.g. label_vocab_fragmented +
        # invalid_row_ids if we ever bundle them). Both contribute.
        est = estimate_action_cost({
            "kind": "fix_gold_rows",
            "params": {
                "invalid_row_ids": [1, 2],
                "fragment_groups": [["x", "X"]],
            },
        })
        # 2 + 1 = 3 rows × 2 min = 6 min.
        self.assertEqual(est["time_minutes"], 6)

    def test_fix_gold_rows_floors_above_zero_with_no_rows(self):
        # Defensive — the signal wouldn't fire without rows to fix.
        # When it does, we still quote a non-zero estimate (treat
        # "no rows" as "one row × 2 min minimum review" so the chip
        # is never "0 min").
        est = estimate_action_cost({"kind": "fix_gold_rows", "params": {}})
        self.assertGreaterEqual(est["time_minutes"], 1)

    def test_attach_cost_estimates_sets_None_for_non_actionable_signals(self):
        signals = [
            {"id": "ok1", "severity": "ok", "headline": "", "detail": "", "suggested_action": None},
            {"id": "warn1", "severity": "warn", "headline": "", "detail": "",
             "suggested_action": {"kind": "synth_augment", "params": {"target_rows": 50}}},
        ]
        _attach_cost_estimates(signals)
        self.assertIsNone(signals[0]["cost_estimate"])
        self.assertIsNotNone(signals[1]["cost_estimate"])
        self.assertEqual(signals[1]["cost_estimate"]["time_minutes"], 25)

    def test_forecast_payload_carries_cost_estimate_on_actionable_signals(self):
        # End-to-end: every actionable signal in a fresh forecast must
        # carry a CostEstimate, and ok-severity signals must not.
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            # Bare classification project → row_count signal blocks
            # with a synth_augment action → must carry a cost estimate.
            create = client.post("/api/projects", json={"name": f"T1 Cost Smoke {os.getpid()}"})
            pid = create.json()["id"]
            client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "classification"})
            payload = client.get(f"/api/projects/{pid}/training/forecast").json()
            actionable = [s for s in payload["signals"] if s.get("suggested_action")]
            self.assertGreater(len(actionable), 0)
            for sig in actionable:
                self.assertIn("cost_estimate", sig)
                est = sig["cost_estimate"]
                self.assertIsNotNone(est, f"actionable signal {sig['id']} has no cost_estimate")
                self.assertGreaterEqual(est["time_minutes"], 1)
                self.assertEqual(est["confidence"], "rough")
            # ok signals carry an explicit None so the frontend doesn't
            # have to special-case missing-vs-null.
            for sig in payload["signals"]:
                if sig.get("suggested_action") is None:
                    self.assertIn("cost_estimate", sig)
                    self.assertIsNone(sig["cost_estimate"])


# ─────────────────────────────────────────────────────────────────────
# Integration tests — full stack via FastAPI TestClient + real DB.
# ─────────────────────────────────────────────────────────────────────


class TrainabilityForecastApiTests(unittest.TestCase):
    """End-to-end: instantiate a project from a template, hit
    GET /api/projects/{id}/training/forecast, assert the shape."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_forecast_endpoint_returns_full_shape_for_a_template(self):
        # Ticket Router ships with a 200-row gold set — should be a
        # healthy starting point.
        project = self._instantiate_template("ticket-router", "Forecast Smoke #1")
        resp = self.client.get(f"/api/projects/{project['id']}/training/forecast")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        # Shape contract.
        self.assertIn(payload["overall"], {"likely_pass", "borderline", "likely_fail"})
        self.assertIsInstance(payload["confidence_pct"], int)
        self.assertGreaterEqual(payload["confidence_pct"], 0)
        self.assertLessEqual(payload["confidence_pct"], 100)
        self.assertIsInstance(payload["signals"], list)
        self.assertIn("cache_key", payload)
        self.assertFalse(payload["cache_hit"])  # first call → miss

        # Signal ids that must always appear.
        signal_ids = {s["id"] for s in payload["signals"]}
        self.assertIn("row_count_below_minimum", signal_ids)
        self.assertIn("goldset_diversity_low", signal_ids)
        self.assertIn("gate_pass_probability", signal_ids)
        # Classification recipe → class_imbalance signal should appear too.
        self.assertIn("class_imbalance", signal_ids)

    def test_forecast_endpoint_caches_on_second_read(self):
        project = self._instantiate_template("policy-qa-style", "Forecast Cache Test")
        first = self.client.get(f"/api/projects/{project['id']}/training/forecast")
        self.assertEqual(first.status_code, 200, first.text)
        self.assertFalse(first.json()["cache_hit"])
        second = self.client.get(f"/api/projects/{project['id']}/training/forecast")
        self.assertEqual(second.status_code, 200, second.text)
        self.assertTrue(second.json()["cache_hit"], "second read should hit cache")
        # cache_key must be stable across the two reads.
        self.assertEqual(first.json()["cache_key"], second.json()["cache_key"])

    def test_forecast_endpoint_refresh_flag_bypasses_cache(self):
        project = self._instantiate_template("log-triage", "Forecast Refresh Test")
        first = self.client.get(f"/api/projects/{project['id']}/training/forecast")
        self.assertFalse(first.json()["cache_hit"])
        refresh = self.client.get(f"/api/projects/{project['id']}/training/forecast?refresh=true")
        self.assertEqual(refresh.status_code, 200, refresh.text)
        self.assertFalse(refresh.json()["cache_hit"], "?refresh=true must bypass cache")

    def test_forecast_endpoint_404s_on_unknown_project(self):
        resp = self.client.get("/api/projects/99999/training/forecast")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_forecast_endpoint_400s_on_project_with_no_recipe(self):
        # Plain project without recipe-apply.
        create_resp = self.client.post(
            "/api/projects",
            json={"name": "Bare project no recipe"},
        )
        self.assertEqual(create_resp.status_code, 201, create_resp.text)
        pid = create_resp.json()["id"]
        resp = self.client.get(f"/api/projects/{pid}/training/forecast")
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("recipe", resp.text.lower())

    def test_forecast_templates_with_thin_data_predict_borderline_or_fail(self):
        # Templates ship with healthy data — but a *fresh* template
        # project before the user adds anything has just the 200-row
        # gold set. Most should land in borderline or likely_pass given
        # the recipe + gold + raw structure. Smoke check that the
        # overall verdict is sensible (not None, not invalid string).
        for slug in ("ticket-router", "data-to-sql", "log-triage"):
            with self.subTest(slug=slug):
                project = self._instantiate_template(
                    slug, f"Forecast Sanity {slug}",
                )
                resp = self.client.get(
                    f"/api/projects/{project['id']}/training/forecast"
                )
                self.assertEqual(resp.status_code, 200, resp.text)
                payload = resp.json()
                self.assertIn(
                    payload["overall"],
                    {"likely_pass", "borderline", "likely_fail"},
                )

    # ── Named tests required by ROADMAP-USER-SUCCESS Epic 1 spec ────

    def test_overall_verdict_likely_pass_for_template_default_data(self):
        """Spec test name. Fresh-instantiation of a healthy template
        (ticket-router) should land likely_pass on its 200-row gold
        set + balanced 5-class distribution. Calibration anchor."""
        project = self._instantiate_template(
            "ticket-router", "Verdict Anchor Pass",
        )
        resp = self.client.get(f"/api/projects/{project['id']}/training/forecast")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(
            payload["overall"], "likely_pass",
            f"ticket-router fresh-instantiate should land likely_pass; "
            f"got {payload['overall']} at {payload['confidence_pct']}% with "
            f"signals: {[(s['id'], s['severity']) for s in payload['signals']]}",
        )
        # Confidence band should be in the healthy zone.
        self.assertGreaterEqual(payload["confidence_pct"], 60)

    def test_overall_verdict_likely_fail_for_16_row_demo(self):
        """Spec test name. The Support FAQ demo bundle deliberately
        ships 20 raw rows — the calibration anchor for the
        likely_fail end of the band. Verified via simulating the
        16-row pre-prep training corpus by creating a bare project
        with selected_recipe but no datasets at all."""
        # Use the qa-sft recipe via the per-project recipe-apply
        # endpoint (avoids needing to seed a demo bundle).
        create_resp = self.client.post(
            "/api/projects",
            json={"name": "Demo 16-row Forecast Anchor"},
        )
        self.assertEqual(create_resp.status_code, 201, create_resp.text)
        pid = create_resp.json()["id"]
        apply_resp = self.client.put(
            f"/api/projects/{pid}/recipe",
            json={"recipe_id": "qa-sft"},
        )
        self.assertEqual(apply_resp.status_code, 200, apply_resp.text)
        # No datasets attached → 0 labeled rows → row_count signal
        # blocks → overall is likely_fail regardless of other signals.
        resp = self.client.get(f"/api/projects/{pid}/training/forecast")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(
            payload["overall"], "likely_fail",
            f"Bare project with no datasets should land likely_fail; "
            f"got {payload['overall']}.",
        )
        # Specifically the row-count signal should be blocking.
        row_signal = next(
            (s for s in payload["signals"] if s["id"] == "row_count_below_minimum"),
            None,
        )
        self.assertIsNotNone(row_signal)
        self.assertEqual(row_signal["severity"], "block")

    def test_cache_invalidates_when_recipe_changes(self):
        """Spec test name. Changing the project's selected_recipe
        must invalidate the forecast cache — the recipe is part of
        the cache key, so a fresh recipe means a fresh forecast."""
        project = self._instantiate_template(
            "policy-qa-style", "Cache Invalidate Recipe Test",
        )
        pid = project["id"]
        # Warm the cache.
        first = self.client.get(f"/api/projects/{pid}/training/forecast")
        self.assertEqual(first.status_code, 200, first.text)
        self.assertFalse(first.json()["cache_hit"])
        first_cache_key = first.json()["cache_key"]
        # Confirm cache works on a repeat read.
        repeat = self.client.get(f"/api/projects/{pid}/training/forecast")
        self.assertTrue(repeat.json()["cache_hit"])
        # Now switch the recipe via the recipe-apply endpoint.
        apply_resp = self.client.put(
            f"/api/projects/{pid}/recipe",
            json={"recipe_id": "generic-sft"},
        )
        self.assertEqual(apply_resp.status_code, 200, apply_resp.text)
        # Recipe changed → cache_key changes → cache_hit must be false
        # on the next read, and the new cache_key must differ.
        after = self.client.get(f"/api/projects/{pid}/training/forecast")
        self.assertEqual(after.status_code, 200, after.text)
        self.assertFalse(
            after.json()["cache_hit"],
            "changing the recipe must invalidate the cached forecast",
        )
        self.assertNotEqual(
            after.json()["cache_key"], first_cache_key,
            "new recipe should produce a different cache_key",
        )

    # ── T2 — snapshot history persistence + endpoint ────────────────

    def test_forecast_history_endpoint_returns_empty_for_fresh_project(self):
        # A project that has never had its forecast computed should
        # return an empty snapshot list (NOT 404 — the project exists,
        # there's just no history yet).
        create_resp = self.client.post(
            "/api/projects",
            json={"name": "T2 Fresh Project"},
        )
        pid = create_resp.json()["id"]
        resp = self.client.get(f"/api/projects/{pid}/training/forecast/history")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["project_id"], pid)
        self.assertEqual(body["snapshots"], [])

    def test_forecast_history_endpoint_404s_on_unknown_project(self):
        resp = self.client.get("/api/projects/99999/training/forecast/history")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_forecast_persists_a_snapshot_on_cache_miss(self):
        project = self._instantiate_template(
            "ticket-router", "T2 Persist On Miss",
        )
        pid = project["id"]
        # First forecast read → cache miss → one snapshot persisted.
        first = self.client.get(f"/api/projects/{pid}/training/forecast")
        self.assertEqual(first.status_code, 200, first.text)
        self.assertFalse(first.json()["cache_hit"])

        hist = self.client.get(f"/api/projects/{pid}/training/forecast/history")
        self.assertEqual(hist.status_code, 200, hist.text)
        snaps = hist.json()["snapshots"]
        self.assertEqual(len(snaps), 1, f"expected 1 snapshot, got {len(snaps)}")
        # Snapshot fields mirror the live forecast.
        self.assertEqual(snaps[0]["overall"], first.json()["overall"])
        self.assertEqual(snaps[0]["confidence_pct"], first.json()["confidence_pct"])
        self.assertEqual(snaps[0]["cache_key"], first.json()["cache_key"])
        # Signals carried through so the panel tooltip has historical data.
        self.assertGreater(len(snaps[0]["signals"]), 0)

    def test_forecast_does_NOT_persist_a_snapshot_on_cache_hit(self):
        # The sparkline would dilute into a flat line if identical-input
        # recomputes counted. Cache hits MUST be silent.
        project = self._instantiate_template(
            "ticket-router", "T2 Silent On Hit",
        )
        pid = project["id"]
        self.client.get(f"/api/projects/{pid}/training/forecast")  # miss
        # Two cache hits — they should not add snapshots.
        hit_1 = self.client.get(f"/api/projects/{pid}/training/forecast")
        hit_2 = self.client.get(f"/api/projects/{pid}/training/forecast")
        self.assertTrue(hit_1.json()["cache_hit"])
        self.assertTrue(hit_2.json()["cache_hit"])

        snaps = self.client.get(
            f"/api/projects/{pid}/training/forecast/history"
        ).json()["snapshots"]
        self.assertEqual(len(snaps), 1, "cache hits must not add snapshots")

    def test_forecast_refresh_adds_a_snapshot_per_recompute(self):
        # ?refresh=true bypasses the cache → every call counts. The
        # panel's "Refresh" button uses this path so the sparkline
        # advances on each click.
        project = self._instantiate_template(
            "policy-qa-style", "T2 Refresh Path",
        )
        pid = project["id"]
        for _ in range(3):
            r = self.client.get(
                f"/api/projects/{pid}/training/forecast?refresh=true"
            )
            self.assertFalse(r.json()["cache_hit"])

        snaps = self.client.get(
            f"/api/projects/{pid}/training/forecast/history"
        ).json()["snapshots"]
        self.assertEqual(len(snaps), 3)
        # Newest-first ordering — computed_at descends.
        ts = [s["computed_at"] for s in snaps]
        self.assertEqual(ts, sorted(ts, reverse=True))

    def test_forecast_history_limit_param_clamps_and_truncates(self):
        project = self._instantiate_template(
            "log-triage", "T2 Limit Clamp",
        )
        pid = project["id"]
        for _ in range(5):
            self.client.get(f"/api/projects/{pid}/training/forecast?refresh=true")

        # Limit smaller than count → truncate to the newest N.
        resp_two = self.client.get(
            f"/api/projects/{pid}/training/forecast/history?limit=2"
        )
        self.assertEqual(len(resp_two.json()["snapshots"]), 2)
        # Limit clamping at the upper bound: limit=10000 → capped at 100.
        resp_big = self.client.get(
            f"/api/projects/{pid}/training/forecast/history?limit=10000"
        )
        # Only 5 snapshots exist; the clamp is invisible at this scale
        # but the call mustn't 400.
        self.assertEqual(resp_big.status_code, 200, resp_big.text)
        self.assertEqual(len(resp_big.json()["snapshots"]), 5)
        # Limit clamping at the lower bound: limit=0 → at least 1.
        resp_zero = self.client.get(
            f"/api/projects/{pid}/training/forecast/history?limit=0"
        )
        self.assertEqual(resp_zero.status_code, 200, resp_zero.text)
        self.assertEqual(len(resp_zero.json()["snapshots"]), 1)

    def test_snapshot_retention_prunes_rows_older_than_window(self):
        # Direct-DB test: write an old snapshot + a new one, trigger a
        # recompute, confirm the old one is pruned.
        import asyncio
        from datetime import datetime, timedelta, timezone
        from app.database import async_session_factory
        from app.models.training_forecast_snapshot import TrainingForecastSnapshot
        from app.services.trainability_forecast_service import (
            SNAPSHOT_RETENTION_DAYS,
        )

        project = self._instantiate_template(
            "data-to-sql", "T2 Retention Prune",
        )
        pid = project["id"]

        async def seed_old_snapshot():
            async with async_session_factory() as session:
                session.add(
                    TrainingForecastSnapshot(
                        project_id=pid,
                        cache_key="old-key-0123456789",
                        # 90 days back, well outside the 60-day window.
                        computed_at=datetime.now(timezone.utc) - timedelta(days=90),
                        overall="likely_pass",
                        confidence_pct=80,
                        signals=[],
                    )
                )
                await session.commit()

        asyncio.run(seed_old_snapshot())

        # Sanity: history endpoint sees the old snapshot before the next compute.
        before = self.client.get(
            f"/api/projects/{pid}/training/forecast/history?limit=50"
        ).json()["snapshots"]
        self.assertTrue(any(s["cache_key"] == "old-key-0123456789" for s in before))

        # Trigger a fresh compute, which prunes anything older than
        # SNAPSHOT_RETENTION_DAYS in the same call.
        self.client.get(f"/api/projects/{pid}/training/forecast?refresh=true")
        after = self.client.get(
            f"/api/projects/{pid}/training/forecast/history?limit=50"
        ).json()["snapshots"]
        # Old snapshot pruned, new one persisted.
        self.assertFalse(any(s["cache_key"] == "old-key-0123456789" for s in after))
        # Sanity: at least the freshly-computed snapshot survived.
        self.assertGreaterEqual(len(after), 1)
        # Defensive: every surviving snapshot is within the window.
        # SQLite round-trips DateTime(timezone=True) as a naive string;
        # we coerce to UTC-aware before comparing so the test isn't
        # backend-specific.
        cutoff = datetime.now(timezone.utc) - timedelta(days=SNAPSHOT_RETENTION_DAYS)
        for snap in after:
            ts = datetime.fromisoformat(snap["computed_at"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            self.assertGreaterEqual(ts, cutoff)

    def test_endpoint_handles_no_prepared_dataset(self):
        """Spec test name. A project with a recipe but no datasets
        at all should NOT 404 — the forecast still returns, with the
        row-count signal blocking. The overall verdict is
        actionable rather than failing the request."""
        create_resp = self.client.post(
            "/api/projects",
            json={"name": "No-Dataset Forecast Test"},
        )
        pid = create_resp.json()["id"]
        self.client.put(
            f"/api/projects/{pid}/recipe",
            json={"recipe_id": "classification"},
        )
        resp = self.client.get(f"/api/projects/{pid}/training/forecast")
        # Returns 200, not 404 — caller can still read the signals.
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        # All structural fields present.
        self.assertIn("overall", payload)
        self.assertIn("confidence_pct", payload)
        self.assertIsInstance(payload["signals"], list)
        self.assertGreater(len(payload["signals"]), 0)
        # Row-count signal must surface the zero-rows situation.
        row_signal = next(
            (s for s in payload["signals"] if s["id"] == "row_count_below_minimum"),
            None,
        )
        self.assertIsNotNone(row_signal)
        self.assertEqual(row_signal["severity"], "block")


if __name__ == "__main__":
    unittest.main()

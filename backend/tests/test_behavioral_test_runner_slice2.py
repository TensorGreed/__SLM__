"""Quality-Lift phase 5 slice 2 — Behavioral test runner + flattener.

Pins (slice 2: perturbation engine + scoring + snapshot flattener +
gate end-to-end; the run_evaluation wiring is exercised inline by the
gate end-to-end test):

  Perturbation engine (pure, deterministic):
    * Each perturbation kind dispatches correctly + is reproducible
      given a seed; re-running the same call sequence produces
      identical perturbed inputs.
    * Edge cases: empty / short text, position=-1 / 0 / mid-string.

  Per-kind scoring:
    * INV: trial passes iff perturbed prediction equals ORIGINAL
      prediction (NOT given_label) — catches consistently-wrong-but-
      invariant models the way CheckList intends.
    * DIR (must_change / must_change_to / must_change_to_one_of):
      each variant computes pass/fail correctly.
    * MFT: prediction == expected_label, simple equality.
    * failed_examples capped at 10 per test for JSON budget.
    * Budget cap fires + flags ``capped_at_budget`` when total
      trials exceed PER_TEST_PREDICTION_BUDGET.

  Snapshot flattener:
    * ``behavioral.<test_id>.pass_rate`` canonical key emitted.
    * Short-form (``pass_rate_behavioral_<test_id>``) + eval-type
      scoped variant emitted.
    * Catalog matcher accepts the emitted keys without rejection.

  Gate end-to-end:
    * A pack with a gate at ``behavioral.<test_id>.pass_rate`` resolves
      against an EvalResult whose metrics carry the behavioral block
      — the gate passes / fails correctly through the existing
      _evaluate_gate path with no new code.
"""

from __future__ import annotations

import os
import random
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.behavioral_test_runner import (  # noqa: E402
    FAILED_EXAMPLES_CAP,
    PER_TEST_PREDICTION_BUDGET,
    apply_perturbation,
    run_behavioral_tests,
)
from app.services.evaluation_gate_catalog import is_behavioral_metric_id  # noqa: E402
from app.services.evaluation_pack_service import (  # noqa: E402
    _build_metric_snapshot,
    _evaluate_gate,
)


# ────────────────────────────────────────────────────────────────────────
# Mock EvalResult builder shared across flattener + gate end-to-end tests
# ────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────
# Perturbation engine
# ────────────────────────────────────────────────────────────────────────


class PerturbationEngineTests(unittest.TestCase):

    def test_typo_is_deterministic_for_seed(self):
        # Same seed → identical result. Different seed → different
        # result (in practice).
        text = "the quick brown fox jumps over the lazy dog"
        pert = {"kind": "typo", "intensity": 0.1, "seed": "fixed"}
        a = apply_perturbation(text, pert, rng=random.Random(0))
        b = apply_perturbation(text, pert, rng=random.Random(0))
        self.assertEqual(a, b)
        c = apply_perturbation(text, pert, rng=random.Random(1))
        self.assertNotEqual(a, c)

    def test_typo_short_text_returns_unchanged(self):
        self.assertEqual(apply_perturbation("a", {"kind": "typo"}, rng=random.Random(0)), "a")
        self.assertEqual(apply_perturbation("", {"kind": "typo"}, rng=random.Random(0)), "")

    def test_insert_token_prepend(self):
        out = apply_perturbation(
            "looks great",
            {"kind": "insert_token", "params": {"token": "not ", "position": 0}},
            rng=random.Random(0),
        )
        self.assertEqual(out, "not looks great")

    def test_insert_token_append(self):
        out = apply_perturbation(
            "this is great",
            {"kind": "insert_token", "params": {"token": " not really", "position": -1}},
            rng=random.Random(0),
        )
        self.assertEqual(out, "this is great not really")

    def test_insert_token_mid_string(self):
        # Position is a character index. Clamps gracefully when too high.
        out = apply_perturbation(
            "abcdef",
            {"kind": "insert_token", "params": {"token": "X", "position": 3}},
            rng=random.Random(0),
        )
        self.assertEqual(out, "abcXdef")
        out2 = apply_perturbation(
            "abc",
            {"kind": "insert_token", "params": {"token": "X", "position": 999}},
            rng=random.Random(0),
        )
        self.assertEqual(out2, "abcX")

    def test_case_change_lower_upper_title(self):
        text = "The Quick Brown Fox"
        self.assertEqual(
            apply_perturbation(text, {"kind": "case_change", "params": {"case": "upper"}}, rng=random.Random(0)),
            text.upper(),
        )
        self.assertEqual(
            apply_perturbation(text, {"kind": "case_change", "params": {"case": "lower"}}, rng=random.Random(0)),
            text.lower(),
        )
        self.assertEqual(
            apply_perturbation(text, {"kind": "case_change", "params": {"case": "title"}}, rng=random.Random(0)),
            text.title(),
        )

    def test_whitespace_jitter_only_doubles_existing_spaces(self):
        # The perturbation only doubles spaces; non-space characters
        # are preserved. Even at intensity=1.0 the text length grows
        # by AT MOST one extra char per existing space.
        text = "a b c d"
        out = apply_perturbation(
            text, {"kind": "whitespace_jitter", "intensity": 1.0},
            rng=random.Random(0),
        )
        # All three spaces doubled → 4 extra chars max → length 10.
        self.assertLessEqual(len(out), len(text) + text.count(" "))
        # Removing spaces yields the same letters.
        self.assertEqual(out.replace(" ", ""), text.replace(" ", ""))

    def test_unknown_perturbation_kind_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown_perturbation_kind"):
            apply_perturbation(
                "x", {"kind": "phase5b_paraphrase"}, rng=random.Random(0),
            )


# ────────────────────────────────────────────────────────────────────────
# Per-kind scoring
# ────────────────────────────────────────────────────────────────────────


class InvariantTestScoringTests(unittest.TestCase):

    def test_pass_when_perturbed_matches_original_prediction(self):
        # Model predicts "positive" on the original AND the perturbed
        # input — invariance holds. The given_label is intentionally
        # different to verify INV's "original prediction, not given_label"
        # criterion.
        test = {
            "test_id": "typo_invariance",
            "kind": "INV",
            "seed_examples": [{"input": "great product", "given_label": "negative"}],
            "perturbations": [{"kind": "typo", "intensity": 0.05}],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "same_label"},
        }
        # predict_fn called twice: once for originals, once for perturbed.
        # Both return "positive".
        predict_fn = lambda texts: ["positive"] * len(texts)
        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertEqual(results["typo_invariance"]["passed"], 1)
        self.assertEqual(results["typo_invariance"]["total"], 1)
        self.assertEqual(results["typo_invariance"]["pass_rate"], 1.0)

    def test_fail_when_perturbed_differs_from_original(self):
        test = {
            "test_id": "typo_invariance",
            "kind": "INV",
            "seed_examples": [{"input": "great product", "given_label": "positive"}],
            "perturbations": [{"kind": "typo", "intensity": 0.05}],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "same_label"},
        }

        # First call (originals) returns "positive"; second call
        # (perturbed) returns "negative" — invariance violation.
        call_count = {"n": 0}

        def predict_fn(texts):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ["positive"] * len(texts)
            return ["negative"] * len(texts)

        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertEqual(results["typo_invariance"]["passed"], 0)
        self.assertEqual(results["typo_invariance"]["total"], 1)
        # failed_examples carries the diagnostic for slice 3's drill-down.
        failed = results["typo_invariance"]["failed_examples"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["original_label"], "positive")
        self.assertEqual(failed[0]["perturbed_label"], "negative")

    def test_failed_examples_capped_at_ten(self):
        # 20 perturbations × 1 seed → 20 trials, all failing. Snapshot
        # should cap at 10 to keep EvalResult.metrics JSON bounded.
        test = {
            "test_id": "many_failures",
            "kind": "INV",
            "seed_examples": [{"input": "x", "given_label": "positive"}],
            "perturbations": [
                {"kind": "insert_token", "params": {"token": f"_{i}", "position": -1}}
                for i in range(20)
            ],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "same_label"},
        }
        call_count = {"n": 0}

        def predict_fn(texts):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ["positive"] * len(texts)
            return ["negative"] * len(texts)

        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertEqual(results["many_failures"]["total"], 20)
        self.assertEqual(
            len(results["many_failures"]["failed_examples"]), FAILED_EXAMPLES_CAP,
        )


class DirectionalTestScoringTests(unittest.TestCase):

    def test_must_change_passes_on_any_flip(self):
        test = {
            "test_id": "any_flip",
            "kind": "DIR",
            "seed_examples": [{"input": "I love it"}],
            "perturbations": [
                {"kind": "insert_token", "params": {"token": "not ", "position": 0}},
            ],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "must_change"},
        }
        call_count = {"n": 0}

        def predict_fn(texts):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ["positive"] * len(texts)
            return ["neutral"] * len(texts)  # flipped, any direction

        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertEqual(results["any_flip"]["pass_rate"], 1.0)

    def test_must_change_to_specific_label(self):
        test = {
            "test_id": "must_flip_to_negative",
            "kind": "DIR",
            "seed_examples": [{"input": "I love it"}],
            "perturbations": [
                {"kind": "insert_token", "params": {"token": "not ", "position": 0}},
            ],
            "n_perturbations_per_seed": 1,
            "expectation": {
                "kind": "must_change_to", "target_label": "negative",
            },
        }
        # First test: flips to negative → pass.
        call_count = {"n": 0}

        def predict_fn(texts):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ["positive"] * len(texts)
            return ["negative"] * len(texts)

        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertEqual(results["must_flip_to_negative"]["pass_rate"], 1.0)

        # Second run: flips but to neutral (not the target) → fail.
        call_count["n"] = 0

        def predict_fn2(texts):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ["positive"] * len(texts)
            return ["neutral"] * len(texts)

        results2 = run_behavioral_tests([test], predict_fn=predict_fn2)
        self.assertEqual(results2["must_flip_to_negative"]["pass_rate"], 0.0)
        # failed_examples carries the expectation kind + target so
        # slice 3 can render a clean "expected negative, got neutral".
        failed = results2["must_flip_to_negative"]["failed_examples"][0]
        self.assertEqual(failed["expectation_kind"], "must_change_to")
        self.assertEqual(failed["target"], "negative")

    def test_must_change_to_one_of_accepts_any_in_set(self):
        test = {
            "test_id": "must_flip_to_neg_or_neutral",
            "kind": "DIR",
            "seed_examples": [
                {"input": "row a"},
                {"input": "row b"},
            ],
            "perturbations": [
                {"kind": "insert_token", "params": {"token": "not ", "position": 0}},
            ],
            "n_perturbations_per_seed": 1,
            "expectation": {
                "kind": "must_change_to_one_of",
                "target_labels": ["negative", "neutral"],
            },
        }
        # Originals: positive / positive. Perturbed: negative / neutral.
        # Both are in the target set → 2/2 pass.
        call_count = {"n": 0}

        def predict_fn(texts):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ["positive", "positive"]
            return ["negative", "neutral"]

        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertEqual(results["must_flip_to_neg_or_neutral"]["pass_rate"], 1.0)

    def test_must_change_fails_when_label_unchanged(self):
        # Model stayed at positive even with "not" prepended → fail.
        test = {
            "test_id": "must_change_caught",
            "kind": "DIR",
            "seed_examples": [{"input": "I love it"}],
            "perturbations": [
                {"kind": "insert_token", "params": {"token": "not ", "position": 0}},
            ],
            "n_perturbations_per_seed": 1,
            "expectation": {"kind": "must_change"},
        }
        predict_fn = lambda texts: ["positive"] * len(texts)
        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertEqual(results["must_change_caught"]["pass_rate"], 0.0)


class MftTestScoringTests(unittest.TestCase):

    def test_mft_pass_rate_matches_correct_predictions(self):
        # 3 examples, 2 expected to match, 1 wrong.
        test = {
            "test_id": "canonical_examples",
            "kind": "MFT",
            "examples": [
                {"input": "I love it",        "expected_label": "positive"},
                {"input": "Worst experience", "expected_label": "negative"},
                {"input": "Pretty good",      "expected_label": "positive"},
            ],
        }
        # Model predictions: positive, negative, neutral.
        predict_fn = lambda texts: ["positive", "negative", "neutral"]
        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertEqual(results["canonical_examples"]["passed"], 2)
        self.assertEqual(results["canonical_examples"]["total"], 3)
        self.assertAlmostEqual(
            results["canonical_examples"]["pass_rate"], 2/3, places=4,
        )
        failed = results["canonical_examples"]["failed_examples"]
        self.assertEqual(failed[0]["expected_label"], "positive")
        self.assertEqual(failed[0]["predicted_label"], "neutral")


class BudgetSamplingTests(unittest.TestCase):

    def test_budget_cap_flag_set_when_trials_exceed_cap(self):
        # 100 seed × 50 n_per_seed × 1 perturbation = 5000 trials,
        # well over the 2000-prediction budget. Runner samples down +
        # stamps ``capped_at_budget``.
        seeds = [{"input": f"input_{i}", "given_label": "A"} for i in range(100)]
        test = {
            "test_id": "huge_test",
            "kind": "INV",
            "seed_examples": seeds,
            "perturbations": [{"kind": "typo", "intensity": 0.05}],
            "n_perturbations_per_seed": 50,
            "expectation": {"kind": "same_label"},
        }
        predict_fn = lambda texts: ["A"] * len(texts)
        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertEqual(results["huge_test"]["total"], PER_TEST_PREDICTION_BUDGET)
        self.assertEqual(
            results["huge_test"]["capped_at_budget"], PER_TEST_PREDICTION_BUDGET,
        )

    def test_no_cap_flag_when_under_budget(self):
        # 5 trials — well under the cap; no flag.
        test = {
            "test_id": "tiny_test",
            "kind": "MFT",
            "examples": [
                {"input": f"row {i}", "expected_label": "A"} for i in range(5)
            ],
        }
        predict_fn = lambda texts: ["A"] * len(texts)
        results = run_behavioral_tests([test], predict_fn=predict_fn)
        self.assertNotIn("capped_at_budget", results["tiny_test"])


# ────────────────────────────────────────────────────────────────────────
# Snapshot flattener + gate end-to-end
# ────────────────────────────────────────────────────────────────────────


class FlattenAndGateEndToEndTests(unittest.TestCase):

    def test_canonical_dot_path_emitted(self):
        # Mocked aggregate metric with the behavioral block; the slice 1
        # catalog matcher should accept the emitted key shape.
        row = _mock_eval_result(metrics={
            "behavioral": {
                "typo_invariance": {"pass_rate": 0.72, "passed": 36, "total": 50},
                "negation_flips":  {"pass_rate": 0.91, "passed": 41, "total": 45},
            },
        })
        values, sources, variance = _build_metric_snapshot({"classification": row})
        self.assertAlmostEqual(values["behavioral.typo_invariance.pass_rate"], 0.72, places=4)
        self.assertAlmostEqual(values["behavioral.negation_flips.pass_rate"], 0.91, places=4)
        self.assertEqual(values["behavioral.typo_invariance.passed"], 36.0)
        self.assertEqual(values["behavioral.typo_invariance.total"], 50.0)
        # Catalog matcher accepts the same keys we emit. Round-trip pin.
        self.assertTrue(is_behavioral_metric_id("behavioral.typo_invariance.pass_rate"))

    def test_short_form_and_scoped_emitted(self):
        row = _mock_eval_result(metrics={
            "behavioral": {
                "typo_invariance": {"pass_rate": 0.72, "passed": 36, "total": 50},
            },
        })
        values, _, _ = _build_metric_snapshot({"classification": row})
        self.assertAlmostEqual(values["pass_rate_behavioral_typo_invariance"], 0.72, places=4)
        self.assertAlmostEqual(values["classification.behavioral.typo_invariance.pass_rate"], 0.72, places=4)

    def test_non_numeric_leaves_skipped(self):
        # The runner emits ``kind`` and ``failed_examples`` for the UI;
        # these MUST NOT flatten into the gate snapshot (they'd
        # explode key collisions and aren't gate-eligible).
        row = _mock_eval_result(metrics={
            "behavioral": {
                "typo_invariance": {
                    "pass_rate": 0.72, "passed": 36, "total": 50,
                    "kind": "INV",
                    "failed_examples": [{"original_input": "a", "perturbed_input": "b"}],
                    "capped_at_budget": 2000,
                },
            },
        })
        values, _, _ = _build_metric_snapshot({"classification": row})
        self.assertNotIn("behavioral.typo_invariance.kind", values)
        self.assertNotIn("behavioral.typo_invariance.failed_examples", values)
        # capped_at_budget is numeric but not in our closed metric key
        # set — it stays out of the gate-resolvable map by design.
        self.assertNotIn("behavioral.typo_invariance.capped_at_budget", values)

    def test_behavioral_gate_passes_at_threshold(self):
        row = _mock_eval_result(metrics={
            "behavioral": {
                "typo_invariance": {"pass_rate": 0.90, "passed": 45, "total": 50},
            },
        })
        values, sources, variance = _build_metric_snapshot({"classification": row})
        result = _evaluate_gate(
            {
                "gate_id": "typo_invariance_gate",
                "metric_id": "behavioral.typo_invariance.pass_rate",
                "operator": "gte",
                "threshold": 0.85,
                "required": True,
            },
            values=values, sources=sources, variance=variance,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "ok")
        self.assertAlmostEqual(result["actual"], 0.90, places=4)

    def test_behavioral_gate_fails_below_threshold(self):
        row = _mock_eval_result(metrics={
            "behavioral": {
                "typo_invariance": {"pass_rate": 0.72, "passed": 36, "total": 50},
            },
        })
        values, sources, variance = _build_metric_snapshot({"classification": row})
        result = _evaluate_gate(
            {
                "gate_id": "typo_invariance_gate",
                "metric_id": "behavioral.typo_invariance.pass_rate",
                "operator": "gte",
                "threshold": 0.85,
                "required": True,
            },
            values=values, sources=sources, variance=variance,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "below_threshold")

    def test_behavioral_gate_missing_metric_when_runner_skipped(self):
        # No ``behavioral`` key in metrics → the gate falls through to
        # missing_metric_required (required) or _optional (otherwise).
        # Same shape as a missing eval-pack-driven metric today.
        row = _mock_eval_result(metrics={"macro_f1": 0.83})
        values, sources, variance = _build_metric_snapshot({"classification": row})
        result = _evaluate_gate(
            {
                "gate_id": "typo_invariance_gate",
                "metric_id": "behavioral.typo_invariance.pass_rate",
                "operator": "gte",
                "threshold": 0.85,
                "required": True,
            },
            values=values, sources=sources, variance=variance,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "missing_metric_required")



# ────────────────────────────────────────────────────────────────────────
# Slice 3 — Gate-response enrichment + Coach nudge
# ────────────────────────────────────────────────────────────────────────


class GateResponseEnrichmentTests(unittest.TestCase):
    """Quality-Lift phase 5 slice 3 — verify
    ``_attach_behavioral_details`` merges behavioral failed_examples +
    kind into the gate response so ScorecardPanel can render the
    drill-down without a second fetch."""

    def _build_index_and_check(
        self,
        *,
        metric_id: str,
        behavioral_block: dict | None = None,
    ):
        from app.services.evaluation_pack_service import (
            _attach_behavioral_details,
            _build_behavioral_index_for_checks,
        )

        if behavioral_block is None:
            behavioral_block = {
                "typo_invariance": {
                    "kind": "INV",
                    "pass_rate": 0.72,
                    "passed": 36,
                    "total": 50,
                    "failed_examples": [
                        {
                            "original_input": "great product",
                            "perturbed_input": "great pdroduct",
                            "perturbation_name": "typo",
                            "original_label": "positive",
                            "perturbed_label": "negative",
                        },
                    ],
                },
            }
        row = _mock_eval_result(metrics={"behavioral": behavioral_block})
        index = _build_behavioral_index_for_checks({"classification": row})
        check = {
            "gate_id": "typo_invariance_gate",
            "metric_id": metric_id,
            "operator": "gte",
            "threshold": 0.85,
            "required": True,
            "actual": 0.72,
            "passed": False,
            "reason": "below_threshold",
        }
        return _attach_behavioral_details(check, index)

    def test_enriches_canonical_dot_path_metric_id(self):
        enriched = self._build_index_and_check(
            metric_id="behavioral.typo_invariance.pass_rate",
        )
        self.assertEqual(enriched["behavioral_test_id"], "typo_invariance")
        self.assertEqual(enriched["behavioral_kind"], "INV")
        self.assertEqual(enriched["behavioral_passed"], 36)
        self.assertEqual(enriched["behavioral_total"], 50)
        # Failed examples plumbed through so ScorecardPanel can render
        # original vs perturbed without re-fetching the EvalResult.
        self.assertEqual(len(enriched["behavioral_failed_examples"]), 1)
        self.assertEqual(
            enriched["behavioral_failed_examples"][0]["perturbed_input"],
            "great pdroduct",
        )

    def test_enriches_eval_type_scoped_metric_id(self):
        # ``classification.behavioral.<test>.pass_rate`` is the scoped
        # variant — must also resolve to the same test_id.
        enriched = self._build_index_and_check(
            metric_id="classification.behavioral.typo_invariance.pass_rate",
        )
        self.assertEqual(enriched["behavioral_test_id"], "typo_invariance")

    def test_non_behavioral_gate_passes_through_unchanged(self):
        # Regular metric_id (macro_f1) — no behavioral fields added.
        enriched = self._build_index_and_check(metric_id="macro_f1")
        self.assertNotIn("behavioral_test_id", enriched)
        self.assertNotIn("behavioral_kind", enriched)

    def test_capped_at_budget_flag_plumbs_through(self):
        # Slice 2's runner stamps capped_at_budget when trials exceed
        # PER_TEST_PREDICTION_BUDGET — the UI surfaces this so the user
        # sees "tested N of M" rather than silently truncated counts.
        enriched = self._build_index_and_check(
            metric_id="behavioral.huge_test.pass_rate",
            behavioral_block={
                "huge_test": {
                    "kind": "INV",
                    "pass_rate": 0.5,
                    "passed": 1000,
                    "total": 2000,
                    "failed_examples": [],
                    "capped_at_budget": 2000,
                },
            },
        )
        self.assertEqual(enriched["behavioral_capped_at_budget"], 2000)

    def test_test_id_not_in_index_passes_through_unchanged(self):
        # Gate references a test that's not in the latest EvalResult
        # (recent pack edit, scan hasn't re-run). Don't merge anything.
        enriched = self._build_index_and_check(
            metric_id="behavioral.future_test.pass_rate",
        )
        # The test wasn't in the behavioral block, so no enrichment.
        self.assertNotIn("behavioral_test_id", enriched)


if __name__ == "__main__":
    unittest.main()

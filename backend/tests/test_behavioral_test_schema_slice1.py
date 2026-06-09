"""Quality-Lift phase 5 slice 1 — Behavioral test pack schema validator.

Pins (slice 1: schema gatekeeper + catalog metric_id pattern + pack
validation wiring; runtime + UI surfaces land in slices 2 and 3):

  validate_behavioral_tests (pure function):
    * None / [] normalizes to [].
    * Per-pack cap (30 tests), per-test perturbation cap (8), seed
      example cap (100), MFT example cap (100), input length cap.
    * Closed test_kind set {INV, DIR, MFT}; rejects others verbatim.
    * Closed perturbation kind set {typo, insert_token, case_change,
      whitespace_jitter}; rejects others verbatim.
    * Per-kind shape requirements:
      - INV: seed_examples + perturbations + (optional) same_label
        expectation. Rejects DIR-shaped expectations as
        inv_expectation_must_be_same_label.
      - DIR: seed_examples + perturbations + closed expectation kind
        (must_change / must_change_to / must_change_to_one_of).
        Rejects must_change_to without target_label, etc.
      - MFT: examples with input + expected_label (per row).
    * test_id grammar (matches per-class / slice_id discipline so
      flattens cleanly into ``behavioral.<test_id>.pass_rate``).
    * Duplicate test_ids rejected.
    * pass_rate_floor validated as [0,1] (default 0.85 INV/DIR, 1.0 MFT).

  is_behavioral_metric_id (catalog matcher):
    * Accepts all three id-shapes (canonical dot, short form,
      eval-type scoped) parallel to per_slice.
    * Rejects shapes that would collide with per_class / per_slice.

  validate_draft_pack_gates integration:
    * Accepts a pack with a well-formed behavioral_tests block + a
      gate referencing a behavioral metric_id without unknown_metric_id.
    * Rejects a pack with a malformed behavioral_tests block before
      the gate checks, with a stable error code prefix.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.behavioral_test_schema import (  # noqa: E402
    BEHAVIORAL_TEST_KINDS,
    BehavioralTestValidationError,
    DIR_EXPECTATION_KINDS,
    MAX_MFT_EXAMPLES_PER_TEST,
    MAX_PERTURBATIONS_PER_TEST,
    MAX_SEED_EXAMPLES_PER_TEST,
    MAX_TESTS_PER_PACK,
    PERTURBATION_KINDS,
    validate_behavioral_tests,
)
from app.services.evaluation_gate_catalog import (  # noqa: E402
    is_behavioral_metric_id,
    is_per_class_metric_id,
    is_per_slice_metric_id,
    validate_draft_pack_gates,
)


# ────────────────────────────────────────────────────────────────────────
# Sample test definitions used by multiple tests
# ────────────────────────────────────────────────────────────────────────


def _inv_test(test_id: str = "typo_invariance") -> dict:
    return {
        "test_id": test_id,
        "kind": "INV",
        "description": "Typos should not change predictions.",
        "seed_examples": [
            {"input": "This product is great.", "given_label": "positive"},
        ],
        "perturbations": [
            {"kind": "typo", "intensity": 0.05, "name": "char_swap"},
        ],
    }


def _dir_test(test_id: str = "negation_flips") -> dict:
    return {
        "test_id": test_id,
        "kind": "DIR",
        "description": "Prepending 'not' should flip the label.",
        "seed_examples": [
            {"input": "This is fine.", "given_label": "positive"},
        ],
        "perturbations": [
            {
                "kind": "insert_token",
                "params": {"token": "not ", "position": 0},
                "name": "prepend_not",
            },
        ],
        "expectation": {
            "kind": "must_change_to",
            "target_label": "negative",
        },
    }


def _mft_test(test_id: str = "canonical_examples") -> dict:
    return {
        "test_id": test_id,
        "kind": "MFT",
        "examples": [
            {"input": "I love this!", "expected_label": "positive"},
            {"input": "Worst ever.",  "expected_label": "negative"},
        ],
    }


# ────────────────────────────────────────────────────────────────────────
# Validator — base cases + ID grammar + caps
# ────────────────────────────────────────────────────────────────────────


class ValidatorBasicsTests(unittest.TestCase):

    def test_none_normalizes_to_empty(self):
        self.assertEqual(validate_behavioral_tests(None), [])

    def test_empty_list_is_valid(self):
        self.assertEqual(validate_behavioral_tests([]), [])

    def test_non_list_payload_rejected(self):
        with self.assertRaises(BehavioralTestValidationError) as cm:
            validate_behavioral_tests({"tests": []})
        self.assertIn("behavioral_tests_shape_invalid", str(cm.exception))

    def test_canonical_three_kinds_round_trip(self):
        # One of each kind — pin the cleaned shape so slice 2's runner
        # has a stable contract to dispatch on.
        out = validate_behavioral_tests([_inv_test(), _dir_test(), _mft_test()])
        self.assertEqual(len(out), 3)
        self.assertEqual({t["kind"] for t in out}, {"INV", "DIR", "MFT"})
        # Defaults applied — slice 1's no-vanity rule: pass_rate_floor
        # defaults to 0.85 for INV/DIR, 1.0 for MFT.
        inv = next(t for t in out if t["kind"] == "INV")
        mft = next(t for t in out if t["kind"] == "MFT")
        self.assertEqual(inv["pass_rate_floor"], 0.85)
        self.assertEqual(mft["pass_rate_floor"], 1.0)
        # MFT carries its examples; INV/DIR carry seed_examples +
        # perturbations + expectation.
        self.assertIn("examples", mft)
        self.assertIn("seed_examples", inv)
        self.assertIn("expectation", inv)

    def test_test_id_grammar_enforced(self):
        bad = ["UpperCase", "1leading_digit", "has space", "has.dot", ""]
        for bad_id in bad:
            with self.subTest(test_id=bad_id):
                t = _inv_test(test_id=bad_id)
                with self.assertRaises(BehavioralTestValidationError):
                    validate_behavioral_tests([t])

    def test_duplicate_test_ids_rejected(self):
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"duplicate_behavioral_test_id:typo_invariance",
        ):
            validate_behavioral_tests([_inv_test(), _inv_test()])

    def test_unknown_kind_rejected(self):
        t = _inv_test()
        t["kind"] = "WHATEVER"
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"unknown_behavioral_test_kind",
        ):
            validate_behavioral_tests([t])

    def test_pack_cap_enforced(self):
        many = [_mft_test(test_id=f"t{i}") for i in range(MAX_TESTS_PER_PACK + 1)]
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"behavioral_tests_too_many",
        ):
            validate_behavioral_tests(many)

    def test_pass_rate_floor_validated(self):
        # Out of range.
        t = _inv_test()
        t["pass_rate_floor"] = 1.5
        with self.assertRaises(BehavioralTestValidationError):
            validate_behavioral_tests([t])
        # Wrong type.
        t["pass_rate_floor"] = "high"
        with self.assertRaises(BehavioralTestValidationError):
            validate_behavioral_tests([t])

    def test_constants_match_documented_contract(self):
        # If the closed sets ever change, slice 2's runner dispatch +
        # slice 3's UI op picker need to be updated in lockstep — this
        # test will scream so we don't drift.
        self.assertEqual(set(BEHAVIORAL_TEST_KINDS), {"INV", "DIR", "MFT"})
        self.assertEqual(
            set(PERTURBATION_KINDS),
            {"typo", "insert_token", "case_change", "whitespace_jitter"},
        )
        self.assertEqual(
            set(DIR_EXPECTATION_KINDS),
            {"must_change", "must_change_to", "must_change_to_one_of"},
        )


# ────────────────────────────────────────────────────────────────────────
# Validator — per-kind shape requirements
# ────────────────────────────────────────────────────────────────────────


class InvariantTestShapeTests(unittest.TestCase):

    def test_seed_examples_required(self):
        t = _inv_test()
        t["seed_examples"] = []
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"seed_examples_required",
        ):
            validate_behavioral_tests([t])

    def test_perturbations_required(self):
        t = _inv_test()
        t["perturbations"] = []
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"perturbations_required",
        ):
            validate_behavioral_tests([t])

    def test_inv_rejects_dir_shaped_expectation(self):
        # Common user error — they wrote INV but expressed a directional
        # expectation. Surface explicitly so the editor can suggest
        # switching kind to DIR.
        t = _inv_test()
        t["expectation"] = {"kind": "must_change_to", "target_label": "negative"}
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"inv_expectation_must_be_same_label",
        ):
            validate_behavioral_tests([t])

    def test_inv_defaults_same_label_expectation(self):
        t = _inv_test()
        # Don't set expectation explicitly.
        out = validate_behavioral_tests([t])
        self.assertEqual(out[0]["expectation"], {"kind": "same_label"})

    def test_seed_example_cap(self):
        t = _inv_test()
        t["seed_examples"] = [
            {"input": f"row {i}", "given_label": "positive"}
            for i in range(MAX_SEED_EXAMPLES_PER_TEST + 1)
        ]
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"seed_examples_too_many",
        ):
            validate_behavioral_tests([t])

    def test_perturbation_cap(self):
        t = _inv_test()
        t["perturbations"] = [
            {"kind": "typo", "intensity": 0.05}
            for _ in range(MAX_PERTURBATIONS_PER_TEST + 1)
        ]
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"perturbations_too_many",
        ):
            validate_behavioral_tests([t])


class DirectionalTestShapeTests(unittest.TestCase):

    def test_must_change_to_requires_target_label(self):
        t = _dir_test()
        t["expectation"] = {"kind": "must_change_to"}
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"dir_target_label_required",
        ):
            validate_behavioral_tests([t])

    def test_must_change_to_one_of_requires_target_labels(self):
        t = _dir_test()
        t["expectation"] = {"kind": "must_change_to_one_of"}
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"dir_target_labels_required",
        ):
            validate_behavioral_tests([t])

    def test_must_change_one_of_normalizes_targets(self):
        t = _dir_test()
        t["expectation"] = {
            "kind": "must_change_to_one_of",
            "target_labels": [" negative ", "neutral"],
        }
        out = validate_behavioral_tests([t])
        self.assertEqual(
            out[0]["expectation"]["target_labels"], ["negative", "neutral"],
        )

    def test_must_change_no_target_required(self):
        t = _dir_test()
        t["expectation"] = {"kind": "must_change"}
        out = validate_behavioral_tests([t])
        self.assertEqual(out[0]["expectation"], {"kind": "must_change"})

    def test_unknown_dir_kind_rejected(self):
        t = _dir_test()
        t["expectation"] = {"kind": "must_be_blue"}
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"unknown_dir_expectation_kind",
        ):
            validate_behavioral_tests([t])


class MftTestShapeTests(unittest.TestCase):

    def test_examples_required(self):
        t = _mft_test()
        t["examples"] = []
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"mft_examples_required",
        ):
            validate_behavioral_tests([t])

    def test_expected_label_required_per_example(self):
        t = _mft_test()
        t["examples"][0].pop("expected_label")
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"mft_expected_label_required",
        ):
            validate_behavioral_tests([t])

    def test_examples_cap(self):
        t = _mft_test()
        t["examples"] = [
            {"input": f"row {i}", "expected_label": "A"}
            for i in range(MAX_MFT_EXAMPLES_PER_TEST + 1)
        ]
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"mft_examples_too_many",
        ):
            validate_behavioral_tests([t])


# ────────────────────────────────────────────────────────────────────────
# Perturbation grammar — closed kinds + per-kind params
# ────────────────────────────────────────────────────────────────────────


class PerturbationGrammarTests(unittest.TestCase):

    def test_unknown_kind_rejected(self):
        t = _inv_test()
        t["perturbations"] = [{"kind": "paraphrase"}]  # phase 5b candidate
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"unknown_perturbation_kind",
        ):
            validate_behavioral_tests([t])

    def test_typo_intensity_validated(self):
        t = _inv_test()
        # Negative intensity → out_of_range (must be in (0, 0.5]).
        t["perturbations"] = [{"kind": "typo", "intensity": -0.1}]
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"perturbation_intensity_out_of_range",
        ):
            validate_behavioral_tests([t])

    def test_typo_intensity_default(self):
        t = _inv_test()
        t["perturbations"] = [{"kind": "typo"}]  # no intensity → default
        out = validate_behavioral_tests([t])
        self.assertEqual(out[0]["perturbations"][0]["intensity"], 0.05)

    def test_insert_token_requires_token_param(self):
        t = _inv_test()
        t["perturbations"] = [{"kind": "insert_token", "params": {}}]
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"perturbation_token_required",
        ):
            validate_behavioral_tests([t])

    def test_insert_token_default_position(self):
        t = _inv_test()
        t["perturbations"] = [
            {"kind": "insert_token", "params": {"token": "not "}},
        ]
        out = validate_behavioral_tests([t])
        self.assertEqual(out[0]["perturbations"][0]["params"]["position"], 0)

    def test_case_change_kind_validated(self):
        t = _inv_test()
        t["perturbations"] = [
            {"kind": "case_change", "params": {"case": "sentence"}},
        ]
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"perturbation_case_invalid",
        ):
            validate_behavioral_tests([t])

    def test_whitespace_jitter_intensity_validated(self):
        t = _inv_test()
        t["perturbations"] = [
            {"kind": "whitespace_jitter", "intensity": 2.0},
        ]
        with self.assertRaisesRegex(
            BehavioralTestValidationError, r"perturbation_intensity_out_of_range",
        ):
            validate_behavioral_tests([t])


# ────────────────────────────────────────────────────────────────────────
# Catalog matcher — is_behavioral_metric_id
# ────────────────────────────────────────────────────────────────────────


class CatalogMatcherTests(unittest.TestCase):

    def test_accepts_canonical_dot_path(self):
        for suffix in ("pass_rate", "passed", "total"):
            with self.subTest(suffix=suffix):
                self.assertTrue(
                    is_behavioral_metric_id(
                        f"behavioral.typo_invariance.{suffix}"
                    )
                )

    def test_accepts_short_form(self):
        self.assertTrue(
            is_behavioral_metric_id("pass_rate_behavioral_typo_invariance"),
        )

    def test_accepts_eval_type_scoped(self):
        self.assertTrue(
            is_behavioral_metric_id(
                "classification.behavioral.typo_invariance.pass_rate"
            ),
        )

    def test_rejects_adjacent_shapes(self):
        # Shapes that look similar but belong to per_class / per_slice /
        # base metrics. Catching collisions here is the whole point of
        # the closed regex.
        self.assertFalse(is_behavioral_metric_id("f1"))
        self.assertFalse(is_behavioral_metric_id("behavioral.x"))                  # missing metric suffix
        self.assertFalse(is_behavioral_metric_id("behavioral..pass_rate"))         # empty test_id
        # Note: the matcher lowercases before regex-matching for read-
        # path safety — the validator enforces lowercase test_ids at
        # save time, so the catalog accepts upper-case at lookup.
        # ``f1_X`` style id with non-grammar chars rejected instead.
        self.assertFalse(is_behavioral_metric_id("behavioral.has-dash.pass_rate"))  # hyphen
        self.assertFalse(is_behavioral_metric_id("behavioral.t.confidence"))       # unknown metric
        self.assertFalse(is_behavioral_metric_id("per_class.benign.f1"))
        # Behavioral matcher must not bleed into per_slice space.
        self.assertFalse(is_per_slice_metric_id("behavioral.x.pass_rate"))
        self.assertFalse(is_per_class_metric_id("behavioral.x.pass_rate"))


# ────────────────────────────────────────────────────────────────────────
# validate_draft_pack_gates integration
# ────────────────────────────────────────────────────────────────────────


class DraftPackGateIntegrationTests(unittest.TestCase):

    def _draft_pack(
        self,
        *,
        behavioral_tests: list[dict] | None = None,
        extra_gates: list[dict] | None = None,
    ) -> dict:
        return {
            "task_specs": [{
                "task_profile": "classification",
                "required_metric_ids": ["f1"],
                "metric_schema": {},
                "behavioral_tests": behavioral_tests if behavioral_tests is not None else [],
                "gates": (
                    [{
                        "gate_id": "min_f1",
                        "metric_id": "f1",
                        "operator": "gte",
                        "threshold": 0.8,
                        "required": True,
                    }]
                    + (extra_gates or [])
                ),
            }],
        }

    def test_accepts_pack_with_behavioral_tests_and_referencing_gate(self):
        # The whole point of the slice — a gate referencing a behavioral
        # metric_id must not be rejected as unknown_metric_id.
        pack = self._draft_pack(
            behavioral_tests=[_inv_test(), _mft_test()],
            extra_gates=[{
                "gate_id": "typo_invariance_gate",
                "metric_id": "behavioral.typo_invariance.pass_rate",
                "operator": "gte",
                "threshold": 0.85,
                "required": True,
            }],
        )
        validate_draft_pack_gates(pack)  # raises on failure

    def test_rejects_pack_with_malformed_behavioral_tests(self):
        # The behavioral validator runs BEFORE the gate checks so the
        # user sees the test-shape error first.
        t = _inv_test()
        t["perturbations"] = [{"kind": "phase5b_paraphrase"}]
        pack = self._draft_pack(behavioral_tests=[t])
        with self.assertRaisesRegex(
            ValueError, r"behavioral_test_invalid:.*unknown_perturbation_kind",
        ):
            validate_draft_pack_gates(pack)

    def test_rejects_unknown_behavioral_metric_id_when_no_tests_defined(self):
        # Sanity: writing a gate referencing behavioral.X.pass_rate
        # without defining the test is allowed at this layer (the
        # catalog matcher fires on shape, not existence). Slice 2's
        # runner will silently emit no metric for that id; the gate
        # then resolves as ``missing_metric_required`` / optional per
        # the existing _evaluate_gate path. This pin guards against
        # the validator becoming over-eager and rejecting legit setups
        # where the test is defined elsewhere.
        pack = self._draft_pack(
            behavioral_tests=None,
            extra_gates=[{
                "gate_id": "future_invariance_gate",
                "metric_id": "behavioral.future_test.pass_rate",
                "operator": "gte",
                "threshold": 0.85,
                "required": False,
            }],
        )
        validate_draft_pack_gates(pack)  # raises on failure


if __name__ == "__main__":
    unittest.main()

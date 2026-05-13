"""Phase 5.3.4b — span_set scoring mode inside StructuredExtractionHandler.

Pins the contract: span_set is a general scoring shape for any task
whose output is a list of typed spans `[{type, start, end, text}, ...]`
— PII / PCI, medical NER, legal clause extraction, financial entity
extraction, generic NER. It lives inside StructuredExtractionHandler
rather than as a new handler, so the dispatcher / registry are
untouched.

Coverage:

- Dispatch: `output_schema.scoring_mode == "span_set"` routes to the
  span-set scorer; otherwise the existing field_match scorer runs.
- Strict matching: TP requires exact (type, start, end). Duplicates
  count correctly via Counter.
- Per-class P/R/F1 broken out — the load-bearing signal for compliance
  ("99% credit_card recall" is shippable, "85% overall F1" is not).
- Per-row enrichment lands matched / missed / hallucinated entity
  lists + per-row P/R/F1 on each prediction for the UI.
- Macro aggregates (unweighted mean across classes) reported
  alongside micro — important when class supports are imbalanced.
- Edge cases: empty / empty (trivially correct), empty prediction
  with non-empty gold (recall = 0), malformed entity payloads.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.eval_task_handler_service import (  # noqa: E402
    EvalContext,
    StructuredExtractionHandler,
)


SPAN_SCHEMA = {
    "scoring_mode": "span_set",
    "properties": {"entities": {"type": "array"}},
    "required": ["entities"],
}


def _ctx(schema: dict | None = None) -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile="structured_extraction",
        handler_id="structured_extraction",
        prepared_dir=Path("."),
        dataset_name="test",
        manifest={"output_schema": schema or SPAN_SCHEMA},
    )


def _ent(t: str, s: int, e: int, text: str) -> dict:
    return {"type": t, "start": s, "end": e, "text": text}


def _row(pred_entities: list[dict], gold_entities: list[dict]) -> dict:
    return {
        "prediction": json.dumps({"entities": pred_entities}),
        "reference": json.dumps({"entities": gold_entities}),
    }


class ScoringModeDispatchTests(unittest.TestCase):
    def test_default_scoring_mode_is_field_match(self):
        h = StructuredExtractionHandler()
        ctx = _ctx(
            {
                "properties": {"name": {}, "age": {}},
                "required": ["name"],
            }
        )
        out = h.score(
            [_row([], [])], ctx
        )
        self.assertEqual(out["scoring_mode"], "field_match")
        # Field-match output keys are different from span_set's.
        self.assertIn("field_exact_match_rate", out)
        self.assertNotIn("precision", out)

    def test_explicit_span_set_routes_to_span_set_scorer(self):
        h = StructuredExtractionHandler()
        out = h.score([_row([], [])], _ctx())
        self.assertEqual(out["scoring_mode"], "span_set")
        # Span-set output keys.
        for k in ("precision", "recall", "f1", "per_class", "tp_total"):
            self.assertIn(k, out)

    def test_unknown_scoring_mode_falls_back_to_field_match(self):
        h = StructuredExtractionHandler()
        ctx = _ctx(
            {
                "scoring_mode": "bogus",
                "properties": {"x": {}},
                "required": ["x"],
            }
        )
        out = h.score([_row([], [])], ctx)
        self.assertEqual(out["scoring_mode"], "field_match")


class StrictMatchingTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        h = StructuredExtractionHandler()
        pred = [
            _ent("email", 5, 20, "a@example.com"),
            _ent("phone", 30, 38, "555-0100"),
        ]
        out = h.score([_row(pred, pred)], _ctx())
        self.assertEqual(out["precision"], 1.0)
        self.assertEqual(out["recall"], 1.0)
        self.assertEqual(out["f1"], 1.0)
        self.assertEqual(out["exact_match"], 1.0)

    def test_partial_matches_split_into_tp_fp_fn(self):
        h = StructuredExtractionHandler()
        pred = [
            _ent("email", 5, 20, "a@example.com"),  # TP
            _ent("ssn", 99, 110, "fake"),  # FP — not in gold
        ]
        gold = [
            _ent("email", 5, 20, "a@example.com"),  # matched
            _ent("phone", 30, 38, "555-0100"),  # FN — model missed it
        ]
        out = h.score([_row(pred, gold)], _ctx())
        self.assertEqual(out["tp_total"], 1)
        self.assertEqual(out["fp_total"], 1)
        self.assertEqual(out["fn_total"], 1)
        # P = 1/(1+1) = 0.5, R = 1/(1+1) = 0.5, F1 = 0.5
        self.assertEqual(out["precision"], 0.5)
        self.assertEqual(out["recall"], 0.5)
        self.assertEqual(out["f1"], 0.5)
        # Whole-row not perfect since FP > 0 or FN > 0.
        self.assertEqual(out["exact_match"], 0.0)

    def test_type_mismatch_does_not_count_as_match(self):
        # Same span, different type → no match (strict mode).
        h = StructuredExtractionHandler()
        pred = [_ent("email", 5, 20, "x")]
        gold = [_ent("api_key", 5, 20, "x")]
        out = h.score([_row(pred, gold)], _ctx())
        self.assertEqual(out["tp_total"], 0)
        self.assertEqual(out["fp_total"], 1)
        self.assertEqual(out["fn_total"], 1)

    def test_boundary_mismatch_does_not_count_as_match(self):
        # Same type, off-by-one span → strict mode says no match.
        h = StructuredExtractionHandler()
        pred = [_ent("email", 5, 19, "a@example.co")]
        gold = [_ent("email", 5, 20, "a@example.com")]
        out = h.score([_row(pred, gold)], _ctx())
        self.assertEqual(out["tp_total"], 0)
        self.assertEqual(out["fp_total"], 1)
        self.assertEqual(out["fn_total"], 1)

    def test_duplicate_entities_count_with_multiset_semantics(self):
        # Two emails in the gold, model finds one → TP=1, FN=1, not
        # TP=1 with a free pass for the second.
        h = StructuredExtractionHandler()
        pred = [_ent("email", 5, 20, "a@example.com")]
        gold = [
            _ent("email", 5, 20, "a@example.com"),
            _ent("email", 50, 70, "b@example.net"),
        ]
        out = h.score([_row(pred, gold)], _ctx())
        self.assertEqual(out["tp_total"], 1)
        self.assertEqual(out["fn_total"], 1)


class PerClassMetricsTests(unittest.TestCase):
    def test_per_class_breakdown(self):
        h = StructuredExtractionHandler()
        pred = [
            _ent("email", 5, 20, "a@example.com"),  # TP
            _ent("ssn", 99, 110, "fake"),  # FP
        ]
        gold = [
            _ent("email", 5, 20, "a@example.com"),
            _ent("phone", 30, 38, "555-0100"),  # FN
        ]
        out = h.score([_row(pred, gold)], _ctx())
        pc = out["per_class"]
        # email: 1 TP, 0 FP, 0 FN → P/R/F1 = 1.0
        self.assertEqual(pc["email"]["tp"], 1)
        self.assertEqual(pc["email"]["precision"], 1.0)
        self.assertEqual(pc["email"]["recall"], 1.0)
        # phone: 0 TP, 0 FP, 1 FN → R = 0
        self.assertEqual(pc["phone"]["fn"], 1)
        self.assertEqual(pc["phone"]["recall"], 0.0)
        # ssn: 0 TP, 1 FP, 0 FN → P = 0
        self.assertEqual(pc["ssn"]["fp"], 1)
        self.assertEqual(pc["ssn"]["precision"], 0.0)

    def test_macro_aggregates_are_unweighted_mean(self):
        h = StructuredExtractionHandler()
        # Class supports differ wildly. Macro treats them equally,
        # which is the right call for PII (SSN matters more than its
        # support count suggests).
        pred = [
            _ent("email", 0, 10, "x"),  # 10 emails would be too noisy in
            # this fixture; one per class is enough for the math
            _ent("ssn", 100, 110, "y"),
        ]
        gold = [
            _ent("email", 0, 10, "x"),
            _ent("ssn", 100, 110, "y"),
        ]
        out = h.score([_row(pred, gold)], _ctx())
        # Both classes perfect → macro should be 1.0.
        self.assertEqual(out["f1_macro"], 1.0)
        self.assertEqual(out["precision_macro"], 1.0)
        self.assertEqual(out["recall_macro"], 1.0)


class PerRowEnrichmentTests(unittest.TestCase):
    def test_matched_missed_hallucinated_lists_land_per_row(self):
        h = StructuredExtractionHandler()
        pred = [
            _ent("email", 5, 20, "a@example.com"),  # matched
            _ent("ssn", 99, 110, "fake"),  # hallucinated
        ]
        gold = [
            _ent("email", 5, 20, "a@example.com"),
            _ent("phone", 30, 38, "555-0100"),  # missed
        ]
        row = _row(pred, gold)
        h.score([row], _ctx())
        self.assertEqual(len(row["row_matched_entities"]), 1)
        self.assertEqual(row["row_matched_entities"][0]["type"], "email")
        self.assertEqual(len(row["row_missed_entities"]), 1)
        self.assertEqual(row["row_missed_entities"][0]["type"], "phone")
        self.assertEqual(len(row["row_hallucinated_entities"]), 1)
        self.assertEqual(row["row_hallucinated_entities"][0]["type"], "ssn")

    def test_per_row_precision_recall_f1_written_in_place(self):
        h = StructuredExtractionHandler()
        pred = [_ent("email", 5, 20, "a")]
        gold = [_ent("email", 5, 20, "a"), _ent("phone", 30, 38, "b")]
        row = _row(pred, gold)
        h.score([row], _ctx())
        # P = 1/1 = 1.0, R = 1/2 = 0.5, F1 = 2*1*0.5/1.5 ≈ 0.6667
        self.assertEqual(row["row_precision"], 1.0)
        self.assertEqual(row["row_recall"], 0.5)
        self.assertAlmostEqual(row["row_f1"], 0.6667, places=3)
        # Not exact since FN > 0.
        self.assertEqual(row["row_exact_match"], 0.0)

    def test_scoring_mode_tagged_on_row(self):
        h = StructuredExtractionHandler()
        row = _row([], [])
        h.score([row], _ctx())
        self.assertEqual(row["scoring_mode"], "span_set")


class EdgeCaseTests(unittest.TestCase):
    def test_empty_pred_empty_gold_trivially_correct(self):
        # Per CoNLL convention: a row that genuinely has no entities,
        # and the model emits no entities, is correct.
        h = StructuredExtractionHandler()
        out = h.score([_row([], [])], _ctx())
        self.assertEqual(out["f1"], 1.0)
        self.assertEqual(out["exact_match"], 1.0)

    def test_empty_pred_nonempty_gold_zeros_recall(self):
        h = StructuredExtractionHandler()
        gold = [_ent("email", 5, 20, "a")]
        out = h.score([_row([], gold)], _ctx())
        self.assertEqual(out["recall"], 0.0)
        self.assertEqual(out["exact_match"], 0.0)
        self.assertEqual(out["fn_total"], 1)

    def test_nonempty_pred_empty_gold_zeros_precision(self):
        h = StructuredExtractionHandler()
        pred = [_ent("email", 5, 20, "fake")]
        out = h.score([_row(pred, [])], _ctx())
        self.assertEqual(out["precision"], 0.0)
        self.assertEqual(out["fp_total"], 1)

    def test_malformed_entity_payload_does_not_crash(self):
        h = StructuredExtractionHandler()
        bad_pred = json.dumps(
            {"entities": [{"type": "email"}]}  # missing start/end
        )
        good_ref = json.dumps(
            {"entities": [_ent("email", 5, 20, "a")]}
        )
        out = h.score(
            [{"prediction": bad_pred, "reference": good_ref}], _ctx()
        )
        # Malformed entity skipped → counts as no prediction → FN = 1.
        self.assertEqual(out["tp_total"], 0)
        self.assertEqual(out["fn_total"], 1)

    def test_unparseable_json_dings_validity_and_recall(self):
        h = StructuredExtractionHandler()
        out = h.score(
            [
                {
                    "prediction": "not even json",
                    "reference": json.dumps(
                        {"entities": [_ent("email", 5, 20, "a")]}
                    ),
                }
            ],
            _ctx(),
        )
        self.assertEqual(out["json_validity_rate"], 0.0)
        # Malformed pred → no entities → FN.
        self.assertEqual(out["fn_total"], 1)

    def test_empty_predictions_list_returns_zeros(self):
        h = StructuredExtractionHandler()
        out = h.score([], _ctx())
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["f1"], 0.0)
        self.assertEqual(out["json_validity_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

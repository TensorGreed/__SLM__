"""``finding_set`` scoring (code findings): label + inclusive line-range IoU
matching inside StructuredExtractionHandler. Neutral toy taxonomy only."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")

from app.services import finding_set_scoring as fss  # noqa: E402
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler  # noqa: E402
from app.services.evaluation_service import _row_scores_for_pairing  # noqa: E402
from app.services.evaluation_pack_service import _flatten_per_class_metrics  # noqa: E402

PARENTS = {"injection.query": "injection"}


def schema(**match) -> dict:
    return {
        "scoring_mode": "finding_set",
        "required": ["findings"],
        "finding_match": {"iou_threshold": 0.5, "attrs": [], "hierarchy": False, "parent_credit": 0.5, **match},
        "labels": ["injection", "injection.query", "secret.hardcoded", "logging.sensitive"],
        "label_parents": PARENTS,
    }


def f(label, s, e, **attrs):
    return {"label": label, "start_line": s, "end_line": e, "evidence": "x", "attrs": attrs}


def pred_row(preds, refs, raw=None):
    return {
        "prompt": "p",
        "prediction": raw if raw is not None else json.dumps({"findings": preds}),
        "reference": json.dumps({"findings": refs}),
    }


def score(rows, sch=None):
    sch = sch or schema()
    handler = StructuredExtractionHandler()
    ctx = EvalContext(
        project_id=0, experiment_id=0, eval_type="f1", task_profile="structured_extraction",
        handler_id="structured_extraction", prepared_dir=Path("."), dataset_name="t",
        manifest={"output_schema": sch},
    )
    return handler.score(rows, ctx)


class IouTests(unittest.TestCase):
    def test_inclusive_iou(self):
        self.assertEqual(fss.line_iou(1, 1, 1, 1), 1.0)
        self.assertEqual(fss.line_iou(1, 2, 1, 4), 0.5)
        self.assertEqual(fss.line_iou(1, 2, 3, 4), 0.0)
        self.assertAlmostEqual(fss.line_iou(1, 4, 3, 6), 2 / 6)


class ThresholdTests(unittest.TestCase):
    def test_exactly_at_threshold_matches(self):
        # pred 1-2 vs ref 1-4: IoU exactly 0.5
        m = score([pred_row([f("injection", 1, 2)], [f("injection", 1, 4)])])
        self.assertEqual((m["tp_total"], m["fp_total"], m["fn_total"]), (1, 0, 0))

    def test_just_below_threshold_misses(self):
        # pred 1-2 vs ref 1-5: IoU 0.4
        m = score([pred_row([f("injection", 1, 2)], [f("injection", 1, 5)])])
        self.assertEqual((m["tp_total"], m["fp_total"], m["fn_total"]), (0, 1, 1))

    def test_threshold_is_configurable(self):
        row = lambda: pred_row([f("injection", 1, 2)], [f("injection", 1, 5)])  # noqa: E731
        self.assertEqual(score([row()], schema(iou_threshold=0.4))["tp_total"], 1)
        self.assertEqual(score([row()], schema(iou_threshold=0.9))["tp_total"], 0)

    def test_label_must_be_equal(self):
        m = score([pred_row([f("secret.hardcoded", 1, 3)], [f("injection", 1, 3)])])
        self.assertEqual((m["tp_total"], m["fp_total"], m["fn_total"]), (0, 1, 1))
        self.assertEqual(m["per_class"]["secret.hardcoded"]["fp"], 1)
        self.assertEqual(m["per_class"]["injection"]["fn"], 1)


class DuplicateTests(unittest.TestCase):
    def test_duplicates_beyond_one_match_are_false_positives(self):
        m = score([pred_row([f("injection", 1, 3), f("injection", 1, 3), f("injection", 1, 3)], [f("injection", 1, 3)])])
        self.assertEqual((m["tp_total"], m["fp_total"], m["fn_total"]), (1, 2, 0))

    def test_one_to_one_prefers_higher_iou(self):
        # two refs, two preds: best IoU pairs win, not the first listed
        refs = [f("injection", 1, 4), f("injection", 3, 6)]
        preds = [f("injection", 3, 6), f("injection", 1, 4)]
        row = pred_row(preds, refs)
        m = score([row])
        self.assertEqual((m["tp_total"], m["fp_total"], m["fn_total"]), (2, 0, 0))
        self.assertTrue(all(x["iou"] == 1.0 for x in row["row_matched_findings"]))


class CleanChunkTests(unittest.TestCase):
    def test_clean_chunk_false_positive_rate(self):
        rows = [
            pred_row([], []),
            pred_row([f("injection", 1, 2)], []),
            pred_row([], []),
            pred_row([f("injection", 1, 2)], [f("injection", 1, 2)]),
        ]
        m = score(rows)
        self.assertEqual(m["clean_chunk_count"], 3)
        self.assertEqual(m["clean_chunk_fp_count"], 1)
        self.assertAlmostEqual(m["clean_chunk_fp_rate"], round(1 / 3, 4))
        self.assertTrue(rows[1]["row_is_clean_reference"])
        self.assertFalse(rows[3]["row_is_clean_reference"])
        self.assertEqual(rows[1]["row_exact_match"], 0.0)
        self.assertEqual(rows[0]["row_exact_match"], 1.0)

    def test_empty_on_empty_is_perfect(self):
        m = score([pred_row([], []), pred_row([], [])])
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (1.0, 1.0, 1.0))
        self.assertEqual(m["exact_match"], 1.0)
        self.assertEqual(m["clean_chunk_fp_rate"], 0.0)

    def test_empty_on_empty_macro_mirrors_micro(self):
        m = score([pred_row([], []), pred_row([], [])])
        self.assertEqual(m["per_class"], {})
        self.assertEqual((m["precision_macro"], m["recall_macro"], m["f1_macro"]), (1.0, 1.0, 1.0))
        # a miss on a clean-vs-flagged row is not perfect: per-class rows exist and drive macro
        miss = score([pred_row([], [f("injection", 1, 2)])])
        self.assertEqual(miss["f1_macro"], 0.0)

    def test_no_clean_chunks_rate_zero(self):
        m = score([pred_row([f("injection", 1, 2)], [f("injection", 1, 2)])])
        self.assertEqual(m["clean_chunk_count"], 0)
        self.assertEqual(m["clean_chunk_fp_rate"], 0.0)


class MalformedTests(unittest.TestCase):
    def test_unparseable_output_is_malformed_and_all_fn(self):
        rows = [pred_row([], [f("injection", 1, 2), f("secret.hardcoded", 5, 5)], raw="no json here"),
                pred_row([f("injection", 1, 2)], [f("injection", 1, 2)])]
        m = score(rows)
        self.assertEqual(m["malformed_count"], 1)
        self.assertEqual(m["malformed_rate"], 0.5)
        self.assertEqual((m["tp_total"], m["fp_total"], m["fn_total"]), (1, 0, 2))
        self.assertTrue(rows[0]["row_malformed"])
        self.assertEqual(rows[0]["row_exact_match"], 0.0)
        self.assertEqual(m["json_validity_rate"], 0.5)

    def test_object_without_findings_list_is_malformed(self):
        m = score([pred_row([], [f("injection", 1, 2)], raw='{"entities": []}')])
        self.assertEqual(m["malformed_count"], 1)
        self.assertEqual(m["json_validity_rate"], 1.0)  # valid JSON, wrong shape

    def test_malformed_output_on_clean_chunk_is_not_exact(self):
        row = pred_row([], [], raw="oops")
        m = score([row])
        self.assertEqual(m["malformed_count"], 1)
        self.assertEqual(m["exact_match"], 0.0)
        self.assertEqual(m["clean_chunk_fp_count"], 0)

    def test_invalid_items_are_dropped_and_counted(self):
        bad = [
            {"start_line": 1, "end_line": 2},                    # no label
            {"label": "injection", "start_line": "1", "end_line": 2},  # non-int
            {"label": "injection", "start_line": 5, "end_line": 2},    # start > end
            {"label": "injection", "start_line": 1.5, "end_line": 2},  # float line
            "not an object",
        ]
        m = score([pred_row(bad + [f("injection", 1, 2)], [f("injection", 1, 2)])])
        self.assertEqual(m["invalid_finding_count"], 5)
        self.assertEqual((m["tp_total"], m["fp_total"], m["fn_total"]), (1, 0, 0))
        self.assertEqual(m["malformed_count"], 0)

    def test_fenced_json_is_accepted(self):
        raw = "```json\n" + json.dumps({"findings": [f("injection", 1, 2)]}) + "\n```"
        m = score([pred_row([], [f("injection", 1, 2)], raw=raw)])
        self.assertEqual(m["tp_total"], 1)


class AttrTests(unittest.TestCase):
    def test_attr_accuracy_over_strict_matched_pairs_only(self):
        sch = schema(attrs=["severity", "confidence"])
        rows = [
            pred_row([f("injection", 1, 2, severity="high", confidence="low")],
                     [f("injection", 1, 2, severity="High", confidence="high")]),
            # unmatched pair (label differs): must not count towards attrs
            pred_row([f("secret.hardcoded", 1, 2, severity="low")], [f("injection", 1, 2, severity="low")]),
            pred_row([f("injection", 4, 4, severity="low")], [f("injection", 4, 4, severity="medium")]),
        ]
        m = score(rows, sch)
        self.assertEqual(m["attr_accuracy"]["severity"], {"correct": 1, "total": 2, "accuracy": 0.5})
        self.assertEqual(m["attr_accuracy"]["confidence"], {"correct": 0, "total": 1, "accuracy": 0.0})
        self.assertEqual(m["attr_accuracy_mean"], 0.25)

    def test_no_attrs_configured(self):
        m = score([pred_row([f("injection", 1, 2)], [f("injection", 1, 2)])])
        self.assertEqual(m["attr_accuracy"], {})
        self.assertIsNone(m["attr_accuracy_mean"])

    def test_missing_pred_attr_counts_wrong(self):
        m = score([pred_row([f("injection", 1, 2)], [f("injection", 1, 2, severity="high")])], schema(attrs=["severity"]))
        self.assertEqual(m["attr_accuracy"]["severity"]["correct"], 0)


class HierarchyTests(unittest.TestCase):
    def rows(self):
        return [pred_row([f("injection", 1, 3)], [f("injection.query", 1, 3)])]

    def test_parent_credit_only_in_hierarchical_block(self):
        off = score(self.rows(), schema(hierarchy=False))
        on = score(self.rows(), schema(hierarchy=True, parent_credit=0.5))
        for key in ("precision", "recall", "f1", "tp_total", "fp_total", "fn_total", "per_class", "exact_match"):
            self.assertEqual(off[key], on[key], key)
        self.assertNotIn("hierarchical", off)
        h = on["hierarchical"]
        self.assertEqual(h["partial_matches"], 1)
        self.assertEqual(h["parent_credit"], 0.5)
        # tp=.5 fp=.5 fn=.5 -> P=R=F1=.5
        self.assertEqual((h["precision"], h["recall"], h["f1"]), (0.5, 0.5, 0.5))
        self.assertEqual(on["f1"], 0.0)

    def test_child_predicted_for_parent_ref_gets_no_credit(self):
        rows = [pred_row([f("injection.query", 1, 3)], [f("injection", 1, 3)])]
        h = score(rows, schema(hierarchy=True))["hierarchical"]
        self.assertEqual(h["partial_matches"], 0)

    def test_partial_credit_needs_iou(self):
        rows = [pred_row([f("injection", 10, 12)], [f("injection.query", 1, 3)])]
        self.assertEqual(score(rows, schema(hierarchy=True))["hierarchical"]["partial_matches"], 0)

    def test_strict_match_is_not_double_counted(self):
        rows = [pred_row([f("injection.query", 1, 3), f("injection", 1, 3)], [f("injection.query", 1, 3)])]
        h = score(rows, schema(hierarchy=True))["hierarchical"]
        self.assertEqual(h["partial_matches"], 0)  # ref already consumed by the strict match


class MacroMicroTests(unittest.TestCase):
    def test_macro_differs_from_micro(self):
        rows = [pred_row([f("injection", i, i) for i in range(1, 10)], [f("injection", i, i) for i in range(1, 10)])]
        rows.append(pred_row([], [f("secret.hardcoded", 1, 1)]))
        m = score(rows)
        self.assertAlmostEqual(m["recall"], 0.9)
        self.assertEqual(m["per_class"]["injection"]["f1"], 1.0)
        self.assertEqual(m["per_class"]["secret.hardcoded"]["f1"], 0.0)
        self.assertEqual(m["f1_macro"], 0.5)
        self.assertGreater(m["f1"], m["f1_macro"])
        self.assertEqual(m["per_class"]["injection"]["support"], 9)
        self.assertEqual(m["per_class"]["secret.hardcoded"]["fn"], 1)

    def test_per_class_has_prediction_only_labels(self):
        m = score([pred_row([f("logging.sensitive", 1, 1)], [])])
        self.assertEqual(m["per_class"]["logging.sensitive"], {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0, "tp": 0, "fp": 1, "fn": 0})


class PerRowKeyTests(unittest.TestCase):
    def test_per_row_keys_and_pairing_export(self):
        row = pred_row([f("injection", 1, 3), f("secret.hardcoded", 9, 9)],
                       [f("injection", 1, 4), f("logging.sensitive", 20, 21)])
        m = score([row])
        self.assertEqual(row["scoring_mode"], "finding_set")
        self.assertTrue(row["is_valid_json"])
        self.assertFalse(row["row_malformed"])
        self.assertEqual(row["row_matched_findings"], [{
            "label": "injection", "start_line": 1, "end_line": 4,
            "pred_start_line": 1, "pred_end_line": 3, "iou": 0.75}])
        self.assertEqual(row["row_missed_findings"], [{"label": "logging.sensitive", "start_line": 20, "end_line": 21}])
        self.assertEqual(row["row_spurious_findings"], [{"label": "secret.hardcoded", "start_line": 9, "end_line": 9}])
        self.assertEqual(row["parsed_prediction"]["findings"][0]["label"], "injection")
        self.assertEqual((row["row_precision"], row["row_recall"]), (0.5, 0.5))
        self.assertEqual(row["row_exact_match"], 0.0)
        scores = _row_scores_for_pairing([row])
        self.assertEqual((scores["span_tp"], scores["span_fp"], scores["span_fn"]), ([1], [1], [1]))
        self.assertEqual(m["total"], 1)

    def test_empty_prediction_list(self):
        m = score([])
        self.assertEqual(m["total"], 0)
        self.assertEqual(m["scoring_mode"], "finding_set")


class SchemaResolutionTests(unittest.TestCase):
    def test_wraps_own_prompt_is_mode_aware(self):
        h = StructuredExtractionHandler()
        span = EvalContext(0, 0, "f1", "structured_extraction", "s", Path("."), "t",
                           manifest={"output_schema": {"scoring_mode": "span_set"}})
        find = EvalContext(0, 0, "f1", "structured_extraction", "s", Path("."), "t",
                           manifest={"output_schema": schema()})
        self.assertTrue(h.wraps_own_prompt())
        self.assertTrue(h.wraps_own_prompt(span))
        self.assertFalse(h.wraps_own_prompt(find))

    def test_per_class_flatten_gives_per_label_gate_keys(self):
        m = score([pred_row([f("injection.query", 1, 3)], [f("injection.query", 1, 3)])])
        values: dict = {}

        class _Row:  # minimal EvalResult stand-in
            id = 1
            experiment_id = 1
            eval_type = "f1"
            dataset_name = "test"

        _flatten_per_class_metrics(values, {}, payload_key="per_class", payload_value=m["per_class"],
                                   row=_Row(), eval_type="f1")
        self.assertEqual(values["f1_injection.query"], 1.0)
        self.assertEqual(values["per_class.injection.query.recall"], 1.0)


if __name__ == "__main__":
    unittest.main()

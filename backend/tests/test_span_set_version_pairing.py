"""R1/R2: the eval summary and Compare flag (never block) span_set results
scored under different ``span_set_scoring_version``s, and the version stamp
never shows up as a metric."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.services.compare_row_evidence_service import _scoring, compare_provenance
from app.services.eval_summary_service import _checked_decoding_block
from app.services.sft_lift_summary_service import (
    _compute_metric_lifts,
    _normalize_metrics,
    span_set_version_mismatch,
)


def _span(version=None, f1=0.5):
    m = {"scoring_mode": "span_set", "span_f1": f1}
    if version is not None:
        m["span_set_scoring_version"] = version
    return m


def _prov(version):
    return {
        "answer_key": {"dataset_id": 1, "dataset_type": "test", "version": 3, "sha256": "a" * 64,
                       "version_source": "recorded"},
        "judge": {"answer": None, "rubric": None, "rubric_sha": None, "code_example": None},
        "scoring": {"eval_type": "exact_match", "scoring_mode": "span_set", "span_set_scoring_version": version,
                    "fingerprints": {}},
    }


class VersionStampHiddenTests(unittest.TestCase):
    def test_version_is_not_a_metric(self):
        self.assertNotIn("span_set_scoring_version", _normalize_metrics(_span(2)))

    def test_no_version_row_in_lifts(self):
        rows = _compute_metric_lifts(_normalize_metrics(_span(2, 0.4)), _normalize_metrics(_span(2, 0.6)))
        self.assertNotIn("span_set_scoring_version", [r["metric_id"] for r in rows])


class LiftSummaryMismatchTests(unittest.TestCase):
    def test_same_version_no_note(self):
        self.assertIsNone(span_set_version_mismatch(_span(2), _span(2)))

    def test_both_unversioned_no_note(self):
        self.assertIsNone(span_set_version_mismatch(_span(), _span()))

    def test_non_span_results_ignored(self):
        self.assertIsNone(span_set_version_mismatch({"scoring_mode": "json_fields"}, {"scoring_mode": "json_fields"}))

    def test_missing_base_version_flags_rescore_base(self):
        note = span_set_version_mismatch(_span(), _span(2))
        self.assertEqual((note["base"], note["tuned"]), (None, 2))
        self.assertIn("Re-score the base-model result", note["message"])

    def test_older_tuned_flags_the_run(self):
        note = span_set_version_mismatch(_span(2), _span(1))
        self.assertIn("Re-score the run's result", note["message"])

    def test_eval_summary_passes_note_through(self):
        note = span_set_version_mismatch(_span(), _span(2))
        out = _checked_decoding_block({}, {"scoring_version_differs": note})
        self.assertEqual(out["scoring_version_differs"], note)
        self.assertNotIn("scoring_version_differs", _checked_decoding_block({}, {}))


class CompareNoteTests(unittest.TestCase):
    def test_scoring_reads_version_for_span_set_only(self):
        er = SimpleNamespace(metrics=_span(2), eval_type="exact_match", details={})
        self.assertEqual(_scoring(er)["span_set_scoring_version"], 2)
        er = SimpleNamespace(metrics={"scoring_mode": "json_fields", "span_set_scoring_version": 2},
                             eval_type="exact_match", details={})
        self.assertIsNone(_scoring(er)["span_set_scoring_version"])

    def test_version_mismatch_is_a_note_not_a_block(self):
        result = compare_provenance(_prov(None), _prov(2))
        self.assertTrue(result["comparable"])
        self.assertTrue(any("different span scorers" in n and "Evaluate A again" in n for n in result["notes"]))

    def test_same_version_no_note(self):
        result = compare_provenance(_prov(2), _prov(2))
        self.assertFalse(any("span scorers" in n for n in result["notes"]))


if __name__ == "__main__":
    unittest.main()

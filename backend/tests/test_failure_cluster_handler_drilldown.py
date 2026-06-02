"""Arc 2 — failure-cluster drill-down backend pass-through.

The eval pipeline writes ``predictions_preview`` rows carrying
handler-specific diagnostic fields (RAG context + faithfulness,
alignment chosen/rejected/preference, structured-extraction
is_valid_json + missing fields). Pre-Arc-2 these were extracted
into the failure cluster's exemplars only as
``prompt``/``reference``/``prediction`` — the handler-specific
"why did this row fail" diagnostics were dropped at the extract
step. This commit pulls them through end-to-end so the frontend
drill-down can render them.

Pins:
  * ``_extract_failures_from_eval_result`` forwards handler-
    specific fields from each predictions_preview row into the
    failure dict.
  * ``_build_exemplar`` carries those fields into the exemplar
    that ships to the frontend.
  * Long-text fields (rag_context, alignment_chosen/rejected)
    are truncated to the exemplar limit so a 50KB context block
    doesn't bloat the cluster payload.
  * Rows without handler fields produce exemplars unchanged
    (regression guard).
"""

from __future__ import annotations

import unittest

from app.services.evaluation_remediation_service import (
    _extract_failures_from_eval_result,
)
from app.services.failure_cluster_service import _build_exemplar


class _FakeEvalResult:
    """Minimal shape that ``_extract_failures_from_eval_result`` needs
    — eval_type + metrics + details + pass_rate. Sidesteps the full
    SQLAlchemy model so tests stay pure."""

    def __init__(
        self,
        *,
        eval_type: str = "f1",
        metrics: dict | None = None,
        details: dict | None = None,
        pass_rate: float = 0.5,
    ):
        self.eval_type = eval_type
        self.metrics = metrics or {}
        self.details = details or {}
        self.pass_rate = pass_rate


class ExtractFailuresHandlerFieldsTests(unittest.TestCase):
    def test_rag_context_and_faithfulness_pass_through(self):
        # Row failing on f1 with RAG diagnostics attached. The
        # extract step should preserve every handler field the
        # frontend drill-down knows how to render.
        rag_row = {
            "prompt": "When was Acme founded?",
            "reference": "1999",
            "prediction": "2003",
            "rag_context": "Acme was founded in 1999 by Jane Doe.",
            "rag_has_context": True,
            "rag_faithfulness": 0.2,
            "rag_context_recall": 0.5,
            "rag_is_faithful": False,
            "rag_unsupported_rate": 0.8,
        }
        eval_result = _FakeEvalResult(
            eval_type="f1",
            details={"predictions_preview": [rag_row]},
        )
        failures = _extract_failures_from_eval_result(
            eval_result, max_failures=10,
        )
        self.assertEqual(len(failures), 1)
        for field in (
            "rag_context", "rag_has_context", "rag_faithfulness",
            "rag_context_recall", "rag_is_faithful", "rag_unsupported_rate",
        ):
            self.assertIn(field, failures[0], f"missing field {field!r}")
        self.assertEqual(failures[0]["rag_is_faithful"], False)

    def test_alignment_fields_pass_through(self):
        # Same shape for DPO/ORPO alignment diagnostics.
        align_row = {
            "prompt": "Which is better?",
            "reference": "Answer A",
            "prediction": "Answer B",
            "alignment_chosen": "Answer A — correct one",
            "alignment_rejected": "Answer B — wrong one",
            "alignment_chosen_sim": 0.30,
            "alignment_rejected_sim": 0.95,
            "alignment_preference_correct": False,
        }
        eval_result = _FakeEvalResult(
            eval_type="f1",
            details={"predictions_preview": [align_row]},
        )
        failures = _extract_failures_from_eval_result(
            eval_result, max_failures=10,
        )
        self.assertEqual(len(failures), 1)
        for field in (
            "alignment_chosen", "alignment_rejected",
            "alignment_chosen_sim", "alignment_rejected_sim",
            "alignment_preference_correct",
        ):
            self.assertIn(field, failures[0])
        self.assertEqual(failures[0]["alignment_preference_correct"], False)

    def test_structured_extraction_fields_pass_through(self):
        structured_row = {
            "prompt": "Extract fields.",
            "reference": '{"a":1}',
            "prediction": "not-json",
            "is_valid_json": False,
            "missing_required_fields": ["a", "b"],
            "row_field_results": [{"field": "a", "passed": False}],
        }
        eval_result = _FakeEvalResult(
            eval_type="f1",
            details={"predictions_preview": [structured_row]},
        )
        failures = _extract_failures_from_eval_result(
            eval_result, max_failures=10,
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["is_valid_json"], False)
        self.assertEqual(
            failures[0]["missing_required_fields"], ["a", "b"],
        )
        self.assertIn("row_field_results", failures[0])

    def test_row_level_em_f1_pass_through(self):
        # When the row carries its own EM/F1 (handler set them
        # during the score pass), forward them so the
        # drill-down's metric scoreboard can render.
        em_f1_row = {
            "prompt": "x", "reference": "y", "prediction": "z",
            "row_exact_match": 0.0, "row_f1": 0.25,
        }
        eval_result = _FakeEvalResult(
            eval_type="f1",
            details={"predictions_preview": [em_f1_row]},
        )
        failures = _extract_failures_from_eval_result(
            eval_result, max_failures=10,
        )
        self.assertEqual(failures[0]["row_exact_match"], 0.0)
        self.assertEqual(failures[0]["row_f1"], 0.25)

    def test_rows_without_handler_fields_unchanged(self):
        # Legacy / generic row — just prompt/ref/pred. The
        # extract should NOT inject any of the new fields.
        plain_row = {
            "prompt": "x", "reference": "y", "prediction": "z",
        }
        eval_result = _FakeEvalResult(
            eval_type="f1",
            details={"predictions_preview": [plain_row]},
        )
        failures = _extract_failures_from_eval_result(
            eval_result, max_failures=10,
        )
        for absent in (
            "rag_context", "alignment_chosen", "is_valid_json",
            "row_exact_match",
        ):
            self.assertNotIn(absent, failures[0], f"unexpected field {absent}")


class BuildExemplarHandlerFieldsTests(unittest.TestCase):
    def test_rag_fields_forwarded_to_exemplar(self):
        # The exemplar shape ships to the frontend via the cluster
        # response. Verify the new handler fields land on it.
        row = {
            "prompt": "Q", "reference": "R", "prediction": "P",
            "rag_context": "Some context.",
            "rag_faithfulness": 0.3,
            "rag_is_faithful": False,
        }
        exemplar = _build_exemplar(row)
        self.assertIn("rag_context", exemplar)
        self.assertEqual(exemplar["rag_faithfulness"], 0.3)
        self.assertEqual(exemplar["rag_is_faithful"], False)

    def test_long_rag_context_is_truncated(self):
        # 50KB context blocks would bloat the cluster payload. The
        # exemplar builder truncates rag_context to 600 chars.
        long_ctx = "x" * 5000
        exemplar = _build_exemplar({
            "prompt": "Q", "reference": "R", "prediction": "P",
            "rag_context": long_ctx,
        })
        self.assertLessEqual(len(exemplar["rag_context"]), 600)
        # Truncation mark is the ``…`` suffix the builder uses for
        # text that didn't fit.
        self.assertTrue(exemplar["rag_context"].endswith("…"))

    def test_alignment_fields_forwarded(self):
        row = {
            "prompt": "Q", "reference": "R", "prediction": "P",
            "alignment_chosen": "good answer",
            "alignment_rejected": "bad answer",
            "alignment_chosen_sim": 0.9,
            "alignment_preference_correct": True,
        }
        exemplar = _build_exemplar(row)
        self.assertEqual(exemplar["alignment_chosen"], "good answer")
        self.assertEqual(exemplar["alignment_rejected"], "bad answer")
        self.assertEqual(exemplar["alignment_chosen_sim"], 0.9)
        self.assertEqual(exemplar["alignment_preference_correct"], True)

    def test_structured_fields_forwarded(self):
        row = {
            "prompt": "Q", "reference": "R", "prediction": "P",
            "is_valid_json": False,
            "missing_required_fields": ["company", "country"],
        }
        exemplar = _build_exemplar(row)
        self.assertEqual(exemplar["is_valid_json"], False)
        self.assertEqual(
            exemplar["missing_required_fields"],
            ["company", "country"],
        )

    def test_empty_missing_fields_list_not_forwarded(self):
        # An empty list of missing fields is "no failures on
        # this dimension" — don't bother the frontend with it.
        row = {
            "prompt": "Q", "reference": "R", "prediction": "P",
            "is_valid_json": True,
            "missing_required_fields": [],
        }
        exemplar = _build_exemplar(row)
        self.assertIn("is_valid_json", exemplar)
        self.assertNotIn("missing_required_fields", exemplar)

    def test_no_handler_fields_yields_legacy_exemplar_shape(self):
        # Regression guard — rows without any handler fields
        # produce the same exemplar shape as pre-Arc-2.
        row = {"prompt": "Q", "reference": "R", "prediction": "P"}
        exemplar = _build_exemplar(row)
        for absent in (
            "rag_context", "alignment_chosen", "is_valid_json",
            "row_exact_match", "missing_required_fields",
        ):
            self.assertNotIn(absent, exemplar)
        # Base fields preserved.
        self.assertEqual(exemplar["prompt"], "Q")
        self.assertEqual(exemplar["reference"], "R")
        self.assertEqual(exemplar["prediction"], "P")


if __name__ == "__main__":
    unittest.main()

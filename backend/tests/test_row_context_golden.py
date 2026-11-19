"""No pipeline breakage: rows WITHOUT ``row_context`` render byte-identically for training (prepared
rows + ``scripts/train.py``'s re-wrap, with and without a chat template) and for eval (held-out pairs +
the text each is generated from), across task types. The hashes were recorded with
``tests/_row_context_golden.py`` on the code before per-row context existed (df7ffea).

pytest-only. Pure functions: no DB, no model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _row_context_golden import CASES, fingerprints, scoring_fingerprints, training_render

# Recorded on df7ffea (before row_context): `python tests/_row_context_golden.py` on that tree.
BEFORE_ROW_CONTEXT = {
    "chat": {"eval": "6600901b9da1706b", "prepared": "c6f4b422cdd657e3", "train_text": "b8e8ab1f15faf2ac"},
    "classification": {"eval": "4c744375264ab065", "prepared": "c303fb2a851a1fa7", "train_text": "300d15c5c7a4e0db"},
    # Legacy ``system_prompt`` on a handler-built prompt: training text AND eval are unchanged.
    "classification_legacy_system": {"eval": "4c744375264ab065", "prepared": "458ad2965fa923bd",
                                     "train_text": "2d276ac8678e27fd"},
    "extraction": {"eval": "6f9930e705ef17c0", "prepared": "ef73c7bf13087aee", "train_text": "6326a6c6a879bfa1"},
    "qa": {"eval": "59d463d71aebab92", "prepared": "42f4d6c1402d8a19", "train_text": "7c3966199d9ba68b"},
    "tool_calling": {"eval": "d991e36480f8a630", "prepared": "0259f5151191e2b8", "train_text": "45afd00db75da3f6"},
    "tool_calling_no_catalog": {"eval": "8201870c93508999", "prepared": "41e6447dabc4d564",
                                "train_text": "4635da629c46b80a"},
}

# Recorded on f717c2d (Phases 1-3, before per-row output schemas) with `git archive f717c2d` + the
# current `tests/_row_context_golden.py`: every case (incl. the Phase 4 additions: JSON answers without a
# schema, Phase 2 context kinds only) and the scoring cases render and score byte-identically.
BEFORE_OUTPUT_SCHEMA = {
    "chat": {"eval": "6600901b9da1706b", "prepared": "c6f4b422cdd657e3", "train_text": "b8e8ab1f15faf2ac"},
    "classification": {"eval": "4c744375264ab065", "prepared": "c303fb2a851a1fa7", "train_text": "300d15c5c7a4e0db"},
    "classification_legacy_system": {"eval": "4c744375264ab065", "prepared": "458ad2965fa923bd", "train_text": "2d276ac8678e27fd"},
    "extraction": {"eval": "6f9930e705ef17c0", "prepared": "ef73c7bf13087aee", "train_text": "6326a6c6a879bfa1"},
    "extraction_nested": {"eval": "52bc3d74677b25c3", "prepared": "1eedc251b3dcd498", "train_text": "db2dbd45b7060ef4"},
    "qa": {"eval": "59d463d71aebab92", "prepared": "42f4d6c1402d8a19", "train_text": "7c3966199d9ba68b"},
    "qa_instruction_sft_row_system": {"eval": "38d376e40ef32366", "prepared": "0e331f83473624cc", "train_text": "07b5d99e4adf24ca"},
    "qa_json_answer": {"eval": "bca3ef5c049350af", "prepared": "cf30d8fb6adee79d", "train_text": "5ee5d4a795ba2e40"},
    "tool_calling": {"eval": "d991e36480f8a630", "prepared": "0259f5151191e2b8", "train_text": "45afd00db75da3f6"},
    "tool_calling_no_catalog": {"eval": "8201870c93508999", "prepared": "41e6447dabc4d564", "train_text": "4635da629c46b80a"},
    # P5 gap 6 (2026-11-19): a prepared row stores its row_context in the row's own key order (was sorted), so
    # its "prepared" hash moved; its training text and eval prompt are unchanged.
    "tool_calling_own_tools": {"eval": "d97a69e32f9d0d04", "prepared": "002b8ee104870011", "train_text": "e439e1e8cdbda333"},
}
SCORING_BEFORE_OUTPUT_SCHEMA = {
    "extraction_derived_fields": "754fe0470c73e408",
    "extraction_field_match": "d68c64a892a59eaf",
    "qa_json_text": "c4854b5701c6c783",
    "tool_call_strict": "34313d5b01fecc63",
}


class GoldenRenderingTests(unittest.TestCase):
    def test_every_case_maps_every_row(self):
        for name, case in CASES.items():
            with self.subTest(case=name):
                self.assertEqual(len(training_render(case)), len(case["rows"]))

    def test_rendering_is_unchanged_for_rows_without_row_context(self):
        with tempfile.TemporaryDirectory() as d:
            now = fingerprints(Path(d))
        for name, expected in BEFORE_ROW_CONTEXT.items():
            for part, digest in expected.items():
                with self.subTest(case=name, part=part):
                    self.assertEqual(now[name][part], digest)

    def test_rendering_and_scoring_are_unchanged_since_f717c2d(self):
        with tempfile.TemporaryDirectory() as d:
            now = fingerprints(Path(d))
        self.assertEqual(set(now), set(BEFORE_OUTPUT_SCHEMA))
        for name, expected in BEFORE_OUTPUT_SCHEMA.items():
            for part, digest in expected.items():
                with self.subTest(case=name, part=part):
                    self.assertEqual(now[name][part], digest)
        self.assertEqual(scoring_fingerprints(), SCORING_BEFORE_OUTPUT_SCHEMA)


if __name__ == "__main__":
    unittest.main()

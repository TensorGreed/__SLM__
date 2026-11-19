"""No pipeline breakage: rows WITHOUT ``row_context`` render byte-identically for training (prepared
rows + ``scripts/train.py``'s re-wrap, with and without a chat template) and for eval (held-out pairs +
the text each is generated from), across task types. The hashes were recorded with
``tests/_row_context_golden.py`` on the code before per-row context existed (df7ffea).

pytest-only. Pure functions: no DB, no model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _row_context_golden import CASES, fingerprints, training_render

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


if __name__ == "__main__":
    unittest.main()

"""Unit tests for ``_dedup_entries_for_split`` — the engine behind the
"Re-split with dedup" leakage remediation.

The contract that matters: deduping the combined corpus before a split
must drop exactly the rows the leakage scan would later flag as
cross-split duplicates, so that a ``leakage.split_overlap`` re-scan
clears. We verify that by matching the dedup helper's decisions against
the scan's own matcher (``data_health_service._match_row_against_index``)
over the SAME rows — they must agree on what's a duplicate.

Covers:
  * Exact duplicates (modulo case + whitespace) collapse to one row.
  * Near-duplicates (token-set Jaccard ≥ threshold) collapse.
  * Genuinely distinct rows all survive.
  * First occurrence wins; stable order preserved.
  * The kept set is internally duplicate-free under the scan's matcher
    (the property that guarantees disjoint splits downstream).
"""

from __future__ import annotations

import unittest

from app.services.dataset_service import _dedup_entries_for_split
from app.services.data_health_service import (
    LEAKAGE_FUZZY_THRESHOLD,
    _build_leakage_index,
    _match_row_against_index,
)
from app.services.trainability_forecast_service import _row_to_text


class DedupForSplitTests(unittest.TestCase):

    def test_exact_duplicates_collapse_modulo_case_and_whitespace(self):
        entries = [
            {"text": "The package never arrived and I want a refund."},
            {"text": "the   PACKAGE  never arrived and I want a refund.  "},
            {"text": "A completely different sentence about login errors."},
        ]
        kept, dropped = _dedup_entries_for_split(entries)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 2)
        # First occurrence wins.
        self.assertEqual(kept[0]["text"], entries[0]["text"])

    def test_near_duplicates_collapse(self):
        # Two rows differing by a single token over a long shared body
        # land above the 0.9 Jaccard threshold → one is dropped.
        base = "the quick brown fox jumps over the lazy sleeping dog by the river bank today"
        entries = [
            {"text": base},
            {"text": base + " quietly"},  # 1 extra token over ~14 → Jaccard ≈ 0.93
            {"text": "totally unrelated content about billing invoices and taxes here"},
        ]
        kept, dropped = _dedup_entries_for_split(entries)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 2)

    def test_distinct_rows_all_survive(self):
        entries = [
            {"text": "billing question about a double charge on my card"},
            {"text": "the mobile app crashes on the reports screen"},
            {"text": "where is my order it has been a week"},
            {"text": "how do I reset my account password"},
        ]
        kept, dropped = _dedup_entries_for_split(entries)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 4)

    def test_empty_input(self):
        kept, dropped = _dedup_entries_for_split([])
        self.assertEqual(kept, [])
        self.assertEqual(dropped, 0)

    def test_kept_set_is_internally_duplicate_free_under_scan_matcher(self):
        # The downstream guarantee: after dedup, no kept row matches any
        # OTHER kept row under the leakage scan's own matcher. If that
        # holds, no split can share a row, so split_overlap clears.
        base = "customer support ticket about a delayed international shipment to canada"
        entries = [
            {"text": base},
            {"text": base.upper()},                       # exact dup (case)
            {"text": base + " please help"},              # near dup
            {"text": "unrelated: enable two factor authentication on my profile"},
            {"text": "another distinct one: merge my two separate accounts into one"},
        ]
        kept, _ = _dedup_entries_for_split(entries)

        # For each kept row, build the index from the OTHER kept rows and
        # assert it does not match (exact or fuzzy ≥ threshold).
        for i in range(len(kept)):
            others = [kept[j] for j in range(len(kept)) if j != i]
            index = _build_leakage_index(others)
            kind, jaccard, _idx = _match_row_against_index(_row_to_text(kept[i]), index)
            self.assertIsNone(
                kind,
                f"kept row {i} still matches another kept row "
                f"(kind={kind}, jaccard={jaccard})",
            )
            self.assertLess(jaccard, LEAKAGE_FUZZY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()

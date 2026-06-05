"""Unit tests for ``_disjoint_split_entries`` (Gap #4 slice 3).

Disjoint-by-key split = group rows by a key field (author, template_id,
document_id, customer_id …), then assign each group WHOLE to exactly
one split. Guards against same-key leakage that inflates eval numbers
when uniform random splitting puts chunks of the same document — or
prose by the same author — into both train and test.

Tests cover:
  * The core guarantee: no key appears in more than one split.
  * Split sizes track the requested ratios (within reason — exact
    integer counts can't always hit a target when group sizes vary).
  * Missing / null / empty key rows bucket to ``__missing__`` and go
    entirely to train (so the disjoint guarantee on non-missing keys
    is unconditional).
  * Deterministic under seed; order-independent.
  * Report payload shape matches the documented contract.
  * Degenerate cases: every row shares one key → that key lands in
    one split; ``val_ratio=0`` → no rows in val.
"""

from __future__ import annotations

import unittest

from app.services.dataset_service import _disjoint_split_entries


class DisjointSplitTests(unittest.TestCase):

    # ─────────────────────────────────────────────────────────────
    # Core guarantee: each key appears in exactly one split
    # ─────────────────────────────────────────────────────────────

    def test_no_key_appears_in_more_than_one_split(self):
        # 20 authors, each with 5 rows. The disjoint guarantee says
        # an author's 5 rows land entirely in one split — never
        # scattered. This is the failure mode uniform random has
        # and the whole reason this feature exists.
        entries = [
            {"author": f"author_{a}", "row_id": a * 100 + i, "text": f"row {i}"}
            for a in range(20)
            for i in range(5)
        ]
        splits, report = _disjoint_split_entries(
            entries,
            disjoint_field="author",
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42,
        )

        def _authors(rows):
            return {r["author"] for r in rows}

        train_authors = _authors(splits["train"])
        val_authors = _authors(splits["val"])
        test_authors = _authors(splits["test"])

        # Hard contract: no author is in two splits.
        self.assertEqual(train_authors & val_authors, set())
        self.assertEqual(train_authors & test_authors, set())
        self.assertEqual(val_authors & test_authors, set())
        # All authors accounted for.
        self.assertEqual(
            train_authors | val_authors | test_authors,
            {f"author_{a}" for a in range(20)},
        )
        # Total row count preserved.
        total_rows = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        self.assertEqual(total_rows, 100)

    # ─────────────────────────────────────────────────────────────
    # Split sizes track the requested ratios
    # ─────────────────────────────────────────────────────────────

    def test_split_sizes_approximate_target_ratios(self):
        # 50 templates × 4 rows each → 200 rows. With 0.8/0.1/0.1
        # the greedy bin-packer should land within ±2 templates of
        # the ideal counts. We don't demand exact ratios — group-
        # whole assignment makes that impossible in general — but
        # we do demand it's close enough that drift stays small.
        entries = [
            {"template_id": f"tmpl_{t}", "id": t * 10 + i}
            for t in range(50)
            for i in range(4)
        ]
        splits, report = _disjoint_split_entries(
            entries,
            disjoint_field="template_id",
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42,
        )
        # The report includes ratio_drift = |actual − target|; the
        # greedy bin-packer with uniform group sizes should converge
        # to within a few percent of target.
        for split_name in ("train", "val", "test"):
            self.assertLess(report["ratio_drift"][split_name], 0.05)
        # And every split got at least one group.
        self.assertGreater(report["per_split"]["train"]["group_count"], 0)
        self.assertGreater(report["per_split"]["val"]["group_count"], 0)
        self.assertGreater(report["per_split"]["test"]["group_count"], 0)

    # ─────────────────────────────────────────────────────────────
    # Missing-key rows bucket to train
    # ─────────────────────────────────────────────────────────────

    def test_missing_key_rows_go_entirely_to_train(self):
        # 12 rows with valid authors + 5 rows where the key is
        # missing/null/empty. The 5 missing rows should all land in
        # train so the disjoint guarantee on non-missing keys is a
        # hard contract.
        entries = (
            [{"author": "alice", "id": i} for i in range(4)]
            + [{"author": "bob", "id": 100 + i} for i in range(4)]
            + [{"author": "carol", "id": 200 + i} for i in range(4)]
            + [
                {"id": 901},  # no key at all
                {"author": None, "id": 902},  # explicit null
                {"author": "", "id": 903},  # empty string
                {"author": "   ", "id": 904},  # whitespace
                {"id": 905},
            ]
        )
        splits, report = _disjoint_split_entries(
            entries,
            disjoint_field="author",
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )
        # All 5 missing-shaped rows landed in train.
        missing_ids = {901, 902, 903, 904, 905}
        train_ids = {r["id"] for r in splits["train"]}
        val_ids = {r["id"] for r in splits["val"]}
        test_ids = {r["id"] for r in splits["test"]}
        self.assertEqual(missing_ids & train_ids, missing_ids)
        self.assertEqual(missing_ids & val_ids, set())
        self.assertEqual(missing_ids & test_ids, set())
        # Report surfaces the missing-count.
        self.assertEqual(report["missing_count"], 5)

    # ─────────────────────────────────────────────────────────────
    # Determinism + order independence
    # ─────────────────────────────────────────────────────────────

    def test_deterministic_under_seed(self):
        entries = [
            {"author": f"author_{i % 10}", "id": i}
            for i in range(50)
        ]
        splits_a, _ = _disjoint_split_entries(
            entries, disjoint_field="author", train_ratio=0.7, val_ratio=0.15, seed=42,
        )
        splits_b, _ = _disjoint_split_entries(
            entries, disjoint_field="author", train_ratio=0.7, val_ratio=0.15, seed=42,
        )
        for split_name in ("train", "val", "test"):
            self.assertEqual(
                sorted(r["id"] for r in splits_a[split_name]),
                sorted(r["id"] for r in splits_b[split_name]),
            )

    def test_different_seed_produces_different_assignment(self):
        # Same authors, different seed → at least one author lands in
        # a different split. With 10 authors and a seeded shuffle this
        # is overwhelmingly likely; if you happen to hit a seed that
        # collides, swap one of the seeds and re-run.
        entries = [
            {"author": f"author_{i % 10}", "id": i}
            for i in range(50)
        ]
        splits_a, _ = _disjoint_split_entries(
            entries, disjoint_field="author", train_ratio=0.7, val_ratio=0.15, seed=42,
        )
        splits_b, _ = _disjoint_split_entries(
            entries, disjoint_field="author", train_ratio=0.7, val_ratio=0.15, seed=7,
        )
        # Different seeds → at least one author moves splits.
        train_a = {r["author"] for r in splits_a["train"]}
        train_b = {r["author"] for r in splits_b["train"]}
        self.assertNotEqual(train_a, train_b)

    # ─────────────────────────────────────────────────────────────
    # Report payload shape
    # ─────────────────────────────────────────────────────────────

    def test_report_payload_matches_documented_contract(self):
        entries = [
            {"author": "alice", "id": 1},
            {"author": "alice", "id": 2},
            {"author": "bob", "id": 3},
            {"author": "carol", "id": 4},
            {"author": "carol", "id": 5},
            {"id": 99},  # missing
        ]
        _splits, report = _disjoint_split_entries(
            entries,
            disjoint_field="author",
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )
        self.assertEqual(report["disjoint_field"], "author")
        # 3 real keys + 1 missing bucket = 4 total.
        self.assertEqual(report["group_count"], 4)
        self.assertEqual(report["missing_count"], 1)
        # per_split has all three keys with group_count + row_count + groups.
        for split_name in ("train", "val", "test"):
            entry = report["per_split"][split_name]
            self.assertIn("group_count", entry)
            self.assertIn("row_count", entry)
            self.assertIn("groups", entry)
            self.assertEqual(entry["group_count"], len(entry["groups"]))
        # Sum of per-split row_counts equals total entries.
        self.assertEqual(
            sum(report["per_split"][s]["row_count"] for s in ("train", "val", "test")),
            6,
        )

    # ─────────────────────────────────────────────────────────────
    # Degenerate cases
    # ─────────────────────────────────────────────────────────────

    def test_single_key_lands_entirely_in_one_split(self):
        # All 20 rows share one author. The disjoint guarantee
        # demands they all land in the same split — and that's the
        # split with the largest deficit, which is train at 0.8.
        entries = [{"author": "only_one", "id": i} for i in range(20)]
        splits, _report = _disjoint_split_entries(
            entries,
            disjoint_field="author",
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42,
        )
        # Exactly one split holds all 20 rows.
        sizes = [len(splits[s]) for s in ("train", "val", "test")]
        self.assertEqual(sorted(sizes), [0, 0, 20])
        # And it's the split with the largest target — train.
        self.assertEqual(len(splits["train"]), 20)

    def test_val_ratio_zero_leaves_val_empty(self):
        entries = [
            {"author": f"author_{i % 8}", "id": i} for i in range(40)
        ]
        splits, _report = _disjoint_split_entries(
            entries,
            disjoint_field="author",
            train_ratio=0.8,
            val_ratio=0.0,
            seed=42,
        )
        self.assertEqual(len(splits["val"]), 0)
        # And the rest is split between train and test.
        self.assertEqual(
            len(splits["train"]) + len(splits["test"]),
            40,
        )


if __name__ == "__main__":
    unittest.main()

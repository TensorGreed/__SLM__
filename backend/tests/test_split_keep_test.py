"""split_dataset(keep_test=True): the active test rows stay verbatim, the
rest is re-divided train / val, and no row with a test row's text reaches
train or val — adding data never moves the answer key into training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.config import settings
from app.services import dataset_service as ds


def _rows(prefix, n):
    return [{"text": f"{prefix} example {i}", "entities": [], "_source_dataset": "synthetic"} for i in range(n)]


class KeepTestTests(unittest.TestCase):
    def test_active_test_rows_are_kept_and_never_trained_on(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(settings, "DATA_DIR", Path(tmp)):
            prep = ds._prep_dir(7)
            old_test = _rows("old test", 5)
            (prep / "test.jsonl").write_text("\n".join(json.dumps(r) for r in old_test) + "\n")
            # A fresh split of old + new rows put two old test rows in train
            # (with different metadata and spacing) and new rows in test.
            moved = [dict(old_test[0], _source_dataset="cleaned"), dict(old_test[1], text="Old  TEST example 1")]
            splits = {
                "train": _rows("old train", 40) + moved,
                "val": _rows("old val", 5),
                "test": _rows("new", 6),
            }
            out, report = ds._keep_test_rows(7, splits, train_ratio=0.8, val_ratio=0.1, seed=42)
            self.assertEqual(out["test"], old_test, "verbatim, in order")
            test_keys = {ds._entry_text_key(r) for r in old_test}
            for name in ("train", "val"):
                self.assertFalse(any(ds._entry_text_key(r) in test_keys for r in out[name]), name)
            self.assertEqual(len(out["train"]) + len(out["val"]), 51, "40 + 5 + 6 new, minus nothing else")
            self.assertEqual(len(out["val"]), round(51 / 9))
            self.assertEqual(report, {"applied": True, "test_rows_kept": 5, "test_rows_added": 0, "test_rows": 5,
                                  "rows_matching_test_excluded": 2, "train_val_pool": 51,
                                  "val_rows_requested": None, "question_leakage": None, "fixed_val_rows": None})

    def test_no_active_test_split_is_a_reported_no_op(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(settings, "DATA_DIR", Path(tmp)):
            splits = {"train": _rows("a", 3), "val": [], "test": _rows("b", 1)}
            out, report = ds._keep_test_rows(8, splits, train_ratio=0.8, val_ratio=0.1, seed=1)
            self.assertIs(out, splits)
            self.assertEqual(report, {"applied": False, "reason": "no active test split"})


if __name__ == "__main__":
    unittest.main()


class ReviewQueueRowIdTests(unittest.TestCase):
    def test_rows_without_an_id_get_one_and_become_reviewable(self):
        from app.services import synth_review_queue_service as rq

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.jsonl"
            rows = [
                {"id": 4, "text": "a", "review_status": "pending"},
                {"text": "b", "review_status": "pending"},
                {"id": True, "text": "c"},
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            fixed = rq._ensure_row_ids(path, rq._read_all_rows(path))
            self.assertEqual([r["id"] for r in fixed], [4, 5, 6], "a bool is not an id")
            self.assertEqual([r["id"] for r in rq._read_all_rows(path)], [4, 5, 6], "saved")
            again = rq._ensure_row_ids(path, rq._read_all_rows(path))
            self.assertEqual([r["id"] for r in again], [4, 5, 6], "idempotent")


class ReviewGateTests(unittest.TestCase):
    def test_pending_and_rejected_rows_never_reach_training_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.jsonl"
            rows = [
                {"id": 1, "text": "accepted row", "review_status": "accepted"},
                {"id": 2, "text": "legacy row without a status"},
                {"id": 3, "text": "pending row", "review_status": "pending"},
                {"id": 4, "text": "rejected row", "review_status": "rejected", "reject_reason": "evaluation only"},
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            kept = [r["text"] for r in ds._load_records_from_file(path)]
            self.assertEqual(kept, ["accepted row", "legacy row without a status"])
            everything = ds._load_records_from_file(path, include_pending_synth=True)
            self.assertEqual(len(everything), 4, "the review queue still sees every row")


class SqliteUrlTests(unittest.TestCase):
    def test_in_memory_sqlite_urls_are_not_turned_into_files(self):
        from app.config import Settings

        s = Settings.__new__(Settings)
        for url in ("sqlite+aiosqlite:///:memory:", "sqlite:///:memory:"):
            self.assertEqual(s._normalize_sqlite_database_url(url), url)
        relative = s._normalize_sqlite_database_url("sqlite+aiosqlite:///./data/x.db")
        self.assertTrue(relative.startswith("sqlite+aiosqlite:////"), "relative paths still become absolute")


class LegacyPreparedRowKeepTestTests(unittest.TestCase):
    """The random path is exactly as before: prepared rows carry no input key,
    and keep_test matches test rows by ``_entry_text_key`` - for every row shape,
    including rows prepared by older versions."""

    def _check(self, rows, test_idx):
        from app.models.dataset import DatasetType

        fresh = ds._normalize_rows_for_training(rows, DatasetType.RAW, "llama3")
        self.assertTrue(all("_input_key" not in e for e in fresh))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(settings, "DATA_DIR", Path(tmp)):
            (ds._prep_dir(3) / "test.jsonl").write_text("".join(json.dumps(fresh[i]) + "\n" for i in test_idx))
            splits, report = ds._keep_test_rows(3, {"train": fresh, "val": [], "test": []}, train_ratio=0.8, val_ratio=0.2, seed=1)
        test_keys = {ds._entry_text_key(fresh[i]) for i in test_idx}
        leaked = [e for e in splits["train"] + splits["val"] if ds._entry_text_key(e) in test_keys]
        self.assertEqual(report["rows_matching_test_excluded"], len(test_idx))
        self.assertEqual(leaked, [])

    def test_instruction_input_output(self):
        self._check([{"instruction": "Summarize", "input": f"Doc {i}", "output": f"o{i}"} for i in range(10)], [0, 1, 2])

    def test_prompt_completion(self):
        self._check([{"prompt": f"{i}+{i}=", "completion": str(2 * i)} for i in range(10)], [0, 1, 2])

    def test_source_text_target_text(self):
        self._check([{"source_text": f"hello {i}", "target_text": f"bonjour {i}"} for i in range(10)], [0, 1, 2])

    def test_same_question_different_answers_survive_the_pool(self):
        from app.models.dataset import DatasetType

        rows = [{"question": "Name a colour", "answer": c} for c in ("red", "blue", "green", "pink")] + [{"question": "q-test", "answer": "x"}]
        fresh = ds._normalize_rows_for_training(rows, DatasetType.RAW, "llama3")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(settings, "DATA_DIR", Path(tmp)):
            (ds._prep_dir(3) / "test.jsonl").write_text(json.dumps(fresh[-1]) + "\n")
            splits, report = ds._keep_test_rows(3, {"train": fresh[:4], "val": [], "test": []}, train_ratio=0.8, val_ratio=0.2, seed=1)
        self.assertEqual(len(splits["train"]) + len(splits["val"]), 4)
        self.assertEqual(report["rows_matching_test_excluded"], 0)

"""``scripts/train.py`` loads the train and validation JSONL files in separate
``load_dataset`` calls.

One call over ``{"train": …, "validation": …}`` casts both files to one Arrow
schema and fails with ``CastError: column names don't match`` when the files
carry different columns. They did on 2026-10-19: the validation split had
become eval-only answer-key rows (``answer_key_source``, no ``id`` /
``synth_*`` / ``status``) next to synthetic training rows, and run #154 of the
teacher-distilled reader died four seconds in.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.train import _load_jsonl_splits


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class TrainSplitLoadingTests(unittest.TestCase):
    def test_train_and_validation_files_with_different_columns_load(self):
        from datasets import load_dataset

        with tempfile.TemporaryDirectory() as tmp:
            train = _write(
                Path(tmp) / "train.jsonl",
                [
                    {"id": 1, "question": "q1", "answer": "a1", "synth_source": "flow", "status": "accepted"},
                    {"id": 2, "question": "q2", "answer": "a2", "synth_source": "flow", "status": "accepted"},
                ],
            )
            val = _write(
                Path(tmp) / "val.jsonl",
                [{"question": "q3", "answer": "a3", "answer_key_source": "flow", "generated_by": "teacher"}],
            )
            splits = _load_jsonl_splits(load_dataset, train, val)
        self.assertEqual(len(splits["train"]), 2)
        self.assertEqual(len(splits["validation"]), 1)
        self.assertIn("synth_source", splits["train"].column_names)
        self.assertIn("answer_key_source", splits["validation"].column_names)
        self.assertNotIn("answer_key_source", splits["train"].column_names)

    def test_no_validation_file(self):
        from datasets import load_dataset

        with tempfile.TemporaryDirectory() as tmp:
            train = _write(Path(tmp) / "train.jsonl", [{"question": "q", "answer": "a"}])
            splits = _load_jsonl_splits(load_dataset, train, None)
        self.assertEqual(list(splits), ["train"])
        self.assertEqual(len(splits["train"]), 1)


if __name__ == "__main__":
    unittest.main()


class CudaAllocatorConfigTests(unittest.TestCase):
    """train.py defaults PYTORCH_CUDA_ALLOC_CONF to expandable segments before
    torch touches CUDA (run #154 reserved 33 GB for a < 20 GB working set);
    an explicit value is never overridden."""

    def test_default_set_when_unset(self):
        from scripts.train import CUDA_ALLOC_CONF_DEFAULT, configure_cuda_allocator

        env: dict[str, str] = {}
        self.assertEqual(configure_cuda_allocator(env), CUDA_ALLOC_CONF_DEFAULT)
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], CUDA_ALLOC_CONF_DEFAULT)

    def test_explicit_value_kept(self):
        from scripts.train import configure_cuda_allocator

        env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
        self.assertEqual(configure_cuda_allocator(env), "max_split_size_mb:128")

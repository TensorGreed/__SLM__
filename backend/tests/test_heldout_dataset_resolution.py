"""Held-out eval dataset resolution: aliases resolve by type priority.

Regression: ``"test"`` meant {TEST, GOLD_TEST} resolved by "most recently
updated", so editing the gold test set between the base-model and fine-tuned
evals silently switched the dataset under a lift comparison.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.models.dataset import DatasetType
from app.services.evaluation_service import _resolve_dataset_alias, _resolve_heldout_dataset


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self._rows))


class _FakeDb:
    """Returns datasets in the order given — the real query is newest-first."""

    def __init__(self, newest_first):
        self._rows = newest_first

    async def execute(self, _stmt):
        return _Result(self._rows)


def _ds(dataset_type: DatasetType, name: str, path: str | None = "/tmp/x.jsonl"):
    return SimpleNamespace(dataset_type=dataset_type, name=name, file_path=path)


def _resolve(datasets, dataset_name):
    return asyncio.run(_resolve_heldout_dataset(_FakeDb(datasets), 1, dataset_name))


class DatasetAliasPriorityTests(unittest.TestCase):
    def test_aliases_are_ordered_prepared_split_first(self):
        self.assertEqual(_resolve_dataset_alias("test"), [DatasetType.TEST, DatasetType.GOLD_TEST])
        self.assertEqual(_resolve_dataset_alias("val"), [DatasetType.VALIDATION, DatasetType.GOLD_DEV])
        self.assertEqual(_resolve_dataset_alias("gold_test"), [DatasetType.GOLD_TEST])
        self.assertEqual(_resolve_dataset_alias("nonsense"), [])

    def test_test_prefers_prepared_split_even_when_gold_is_newer(self):
        prepared = _ds(DatasetType.TEST, "test")
        gold = _ds(DatasetType.GOLD_TEST, "gold test")
        # Gold test updated more recently (listed first) — "test" must still
        # mean the prepared split, before AND after the gold edit.
        self.assertIs(_resolve([gold, prepared], "test"), prepared)
        self.assertIs(_resolve([prepared, gold], "test"), prepared)

    def test_test_falls_back_to_gold_when_no_prepared_split(self):
        gold = _ds(DatasetType.GOLD_TEST, "gold test")
        self.assertIs(_resolve([_ds(DatasetType.TRAIN, "train"), gold], "test"), gold)

    def test_explicit_gold_test_still_means_gold(self):
        prepared = _ds(DatasetType.TEST, "test")
        gold = _ds(DatasetType.GOLD_TEST, "gold test")
        self.assertIs(_resolve([prepared, gold], "gold_test"), gold)

    def test_within_one_type_the_newest_wins(self):
        newer = _ds(DatasetType.TEST, "test v2")
        older = _ds(DatasetType.TEST, "test v1")
        self.assertIs(_resolve([newer, older], "test"), newer)

    def test_dataset_without_file_is_skipped(self):
        empty = _ds(DatasetType.TEST, "test", path=None)
        gold = _ds(DatasetType.GOLD_TEST, "gold test")
        self.assertIs(_resolve([empty, gold], "test"), gold)


if __name__ == "__main__":
    unittest.main()


class RowScoresForPairingTests(unittest.TestCase):
    """Held-out evals keep a compact per-row score for EVERY row so a base
    and a fine-tuned eval of the same split can be compared row by row."""

    def test_keys_follow_the_row_and_scores_cover_every_row(self):
        from app.services.evaluation_service import _row_scores_for_pairing

        predictions = [
            {"prompt": "q1", "reference": "billing", "prediction": "billing", "row_f1": 1.0, "row_exact_match": 1.0},
            {"prompt": "q2", "reference": "billing", "prediction": "shipping", "row_f1": 0.0, "row_exact_match": 0.0},
            {"prompt": "q3", "reference": "Refund", "prediction": "refund", "row_f1": 0.5},
        ]
        scores = _row_scores_for_pairing(predictions)
        self.assertEqual(scores["correct"], [1, 0, 1])
        self.assertEqual(scores["f1"], [1.0, 0.0, 0.5])
        self.assertEqual(len(set(scores["keys"])), 3)
        # Same row → same key, whatever the model answered or the order.
        again = _row_scores_for_pairing([{"prompt": "q2", "reference": "billing", "prediction": "billing"}])
        self.assertEqual(again["keys"][0], scores["keys"][1])
        self.assertNotIn("f1", again)  # handler recorded no per-row F1
        self.assertIsNone(_row_scores_for_pairing([]))


class CpuDtypeTests(unittest.TestCase):
    """transformers >= 5 loads a checkpoint in its saved dtype (bf16 for most
    small instruct models). On a CPU without native bf16 that is emulated —
    CI's AVX2 runner trained ~45x slower — so CPU loads must be fp32."""

    def test_cpu_inference_loads_fp32_and_gpu_keeps_half(self):
        from app.services.evaluation_service import _inference_dtype

        class _Cuda:
            def __init__(self, available, bf16=True):
                self._a, self._b = available, bf16

            def is_available(self):
                return self._a

            def is_bf16_supported(self):
                return self._b

        self.assertEqual(_inference_dtype(SimpleNamespace(cuda=_Cuda(False), float32="f32", bfloat16="bf16", float16="f16")), "f32")
        self.assertEqual(_inference_dtype(SimpleNamespace(cuda=_Cuda(True), float32="f32", bfloat16="bf16", float16="f16")), "bf16")
        self.assertEqual(_inference_dtype(SimpleNamespace(cuda=_Cuda(True, bf16=False), float32="f32", bfloat16="bf16", float16="f16")), "f16")

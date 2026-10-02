"""Coach coverage for the synthetic, data-prep and export stages (Wave 3b).

Handlers are exercised directly with their data reads patched (same approach
as test_coach_service's stage tests), so no database is touched.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

from app.models.dataset import DatasetType
from app.services import coach_service
from app.services.coach_service import (
    SYNTHETIC_SHARE_MIN_ROWS,
    TEST_SPLIT_MIN_ROWS,
    CoachStage,
    _dataprep_stage_suggestions,
    _export_stage_suggestions,
    _synthetic_stage_suggestions,
)


def _project(with_task_type: bool = True):
    return SimpleNamespace(
        id=7,
        selected_recipe={"recipe_id": "qa-sft"} if with_task_type else None,
    )


def _ids(suggestions):
    return [s["id"] for s in suggestions]


async def _true(*_a, **_k):
    return True


async def _none(*_a, **_k):
    return None


class StageRegistrationTests(unittest.TestCase):
    def test_new_stages_are_routable(self):
        stages = set(get_args(CoachStage))
        for stage in ("synthetic", "dataprep", "export"):
            self.assertIn(stage, stages)
            self.assertIn(stage, coach_service._STAGE_HANDLERS)


class SyntheticStageTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, *, queue, cleaned_rows=100, with_task_type=True, has_data=True):
        async def _queue(*_a, **_k):
            return queue

        async def _dataset(_db, _pid, dataset_type):
            if dataset_type == DatasetType.CLEANED:
                return SimpleNamespace(record_count=cleaned_rows, file_path=None)
            return None

        async def _has_data(*_a, **_k):
            return has_data

        with (
            patch("app.services.synth_review_queue_service.list_review_queue", side_effect=_queue),
            patch.object(coach_service, "_dataset_of_type", side_effect=_dataset),
            patch.object(coach_service, "_project_has_any_data", side_effect=_has_data),
        ):
            return await _synthetic_stage_suggestions(None, _project(with_task_type))  # type: ignore[arg-type]

    async def test_empty_project_is_silent(self):
        out = await self._run(queue={"total_pending": 0, "total_accepted": 0, "groups": []}, has_data=False)
        self.assertEqual(out, [])

    async def test_pending_rows_nudge_uses_synthetic_prefix(self):
        out = await self._run(queue={
            "total_pending": 12,
            "total_accepted": 0,
            "groups": [{"synth_source": "positives_paraphrase"}],
        })
        nudge = next(s for s in out if s["id"] == "synthetic:synth-review-pending")
        self.assertEqual(nudge["severity"], "warning")
        self.assertEqual(nudge["action"]["params"]["target"], "synthetic-review-queue")
        self.assertEqual(nudge["action"]["params"]["synth_source"], "positives_paraphrase")

    async def test_synthetic_heavy_mix_warns(self):
        out = await self._run(
            queue={"total_pending": 0, "total_accepted": 300, "groups": []},
            cleaned_rows=100,
        )
        nudge = next(s for s in out if s["id"] == "synthetic:share-high")
        self.assertEqual(nudge["title"], "75% of your training rows are synthetic")
        self.assertEqual(nudge["context"]["share"], 0.75)

    async def test_balanced_mix_and_tiny_synthetic_are_quiet(self):
        balanced = await self._run(queue={"total_pending": 0, "total_accepted": 100, "groups": []}, cleaned_rows=100)
        tiny = await self._run(
            queue={"total_pending": 0, "total_accepted": SYNTHETIC_SHARE_MIN_ROWS - 1, "groups": []},
            cleaned_rows=0,
        )
        self.assertNotIn("synthetic:share-high", _ids(balanced))
        self.assertNotIn("synthetic:share-high", _ids(tiny))

    async def test_no_task_type_routes_to_picker(self):
        out = await self._run(queue={"total_pending": 0, "total_accepted": 0, "groups": []}, with_task_type=False)
        nudge = next(s for s in out if s["id"] == "synthetic:no-task-type")
        self.assertEqual(nudge["action"]["params"]["target"], "recipe-picker")


class DataprepStageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _file(self, name: str, age_s: float) -> str:
        path = Path(self.tmp.name) / name
        path.write_text("{}\n")
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
        return str(path)

    async def _run(self, datasets: dict, leak=None):
        async def _dataset(_db, _pid, dataset_type):
            return datasets.get(dataset_type)

        async def _leak(*_a, **_k):
            return leak

        with (
            patch.object(coach_service, "_dataset_of_type", side_effect=_dataset),
            patch("app.services.data_health_service.scan_prepared_split_leakage", side_effect=_leak),
        ):
            return await _dataprep_stage_suggestions(None, _project())  # type: ignore[arg-type]

    async def test_no_source_data_is_silent(self):
        self.assertEqual(await self._run({}), [])

    async def test_data_without_split_asks_to_split(self):
        out = await self._run({
            DatasetType.CLEANED: SimpleNamespace(record_count=50, file_path=None),
        })
        self.assertEqual(_ids(out), ["dataprep:no-split"])
        self.assertEqual(out[0]["action"]["params"]["target"], "dataprep-split")

    async def test_split_older_than_cleaned_data_is_stale(self):
        out = await self._run({
            DatasetType.CLEANED: SimpleNamespace(record_count=50, file_path=self._file("cleaned.jsonl", 10)),
            DatasetType.TRAIN: SimpleNamespace(record_count=40, file_path=self._file("train.jsonl", 600)),
            DatasetType.TEST: SimpleNamespace(record_count=TEST_SPLIT_MIN_ROWS, file_path=None),
        })
        self.assertEqual(_ids(out), ["dataprep:stale-split"])

    async def test_fresh_split_with_enough_test_rows_is_quiet(self):
        out = await self._run({
            DatasetType.CLEANED: SimpleNamespace(record_count=500, file_path=self._file("cleaned.jsonl", 600)),
            DatasetType.TRAIN: SimpleNamespace(record_count=400, file_path=self._file("train.jsonl", 10)),
            DatasetType.TEST: SimpleNamespace(record_count=50, file_path=None),
        })
        self.assertEqual(out, [])

    async def test_small_and_empty_test_split(self):
        base = {
            DatasetType.CLEANED: SimpleNamespace(record_count=60, file_path=self._file("cleaned.jsonl", 600)),
            DatasetType.TRAIN: SimpleNamespace(record_count=50, file_path=self._file("train.jsonl", 10)),
        }
        small = await self._run({**base, DatasetType.TEST: SimpleNamespace(record_count=6, file_path=None)})
        empty = await self._run(dict(base))
        small_nudge = next(s for s in small if s["id"] == "dataprep:test-split-small")
        empty_nudge = next(s for s in empty if s["id"] == "dataprep:test-split-small")
        self.assertEqual((small_nudge["severity"], small_nudge["title"]), ("warning", "Only 6 test examples"))
        self.assertEqual((empty_nudge["severity"], empty_nudge["title"]), ("critical", "No test examples"))

    async def test_split_leakage_reuses_shared_nudge(self):
        leak = {
            "severity": "block",
            "total_leaked": 4,
            "total_scanned": 40,
            "worst_frac": 0.1,
            "per_pair": {"train_in_test": {"leaked": 4}},
            "examples": [],
        }
        out = await self._run({
            DatasetType.CLEANED: SimpleNamespace(record_count=500, file_path=self._file("cleaned.jsonl", 600)),
            DatasetType.TRAIN: SimpleNamespace(record_count=400, file_path=self._file("train.jsonl", 10)),
            DatasetType.TEST: SimpleNamespace(record_count=50, file_path=None),
        }, leak=leak)
        nudge = next(s for s in out if s["id"] == "dataprep:split-leakage")
        self.assertEqual(nudge["severity"], "critical")


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeDb:
    def __init__(self, last_export):
        self.last_export = last_export

    async def execute(self, _stmt):
        return _FakeResult(self.last_export)


class ExportStageTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, summary: dict, last_export=None):
        async def _summary(*_a, **_k):
            return summary

        with patch("app.services.eval_summary_service.build_eval_summary", side_effect=_summary):
            return await _export_stage_suggestions(_FakeDb(last_export), _project())  # type: ignore[arg-type]

    async def test_nothing_trained(self):
        out = await self._run({"verdict": "no_trained_run", "experiment_id": None})
        self.assertEqual(_ids(out), ["export:no-trained-run"])
        self.assertEqual(out[0]["action"]["params"], {"target": "pipeline-tab", "tab": "training"})

    async def test_worse_than_base_is_critical(self):
        out = await self._run({
            "verdict": "worse",
            "experiment_id": 12,
            "headline": {"metric_id": "exact_match", "baseline_value": 0.4, "trained_value": 0.3},
        })
        self.assertEqual(_ids(out), ["export:worse-than-base"])
        self.assertEqual(out[0]["severity"], "critical")
        self.assertIn("0.4 → 0.3", out[0]["body"])
        self.assertIn("run #12", out[0]["title"])

    async def test_within_noise_is_neither_a_downgrade_nor_a_win(self):
        """A "better" or "worse" headline whose row-level evidence is within
        noise must not be called a downgrade, and must not be recommended as a
        model that beats its base."""
        headline = {"metric_id": "f1", "baseline_value": 0.106, "trained_value": 0.136}
        noise = {"verdict": "within_noise", "n": 21, "better": 14, "worse": 6, "same": 1}
        for verdict in ("better", "worse"):
            out = await self._run(
                {"verdict": verdict, "experiment_id": 12, "headline": headline, "evidence": noise},
                last_export=SimpleNamespace(experiment_id=9),
            )
            self.assertEqual(_ids(out), ["export:within-noise"], verdict)
            self.assertEqual(out[0]["severity"], "warning")
            self.assertIn("isn't clearly different", out[0]["title"])
            self.assertIn("helped 14 test examples and hurt 6", out[0]["body"])
            self.assertIn("0.106 → 0.136", out[0]["body"])

        few = await self._run({
            "verdict": "better", "experiment_id": 12, "headline": headline,
            "evidence": {"verdict": "too_few_rows", "n": 2, "better": 2, "worse": 0, "same": 0},
        }, last_export=SimpleNamespace(experiment_id=9))
        self.assertEqual(_ids(few), ["export:within-noise"])
        self.assertIn("only 2 test examples could be compared", few[0]["body"])

    async def test_evidence_beyond_noise_keeps_the_plain_verdicts(self):
        headline = {"metric_id": "f1", "baseline_value": 0.4, "trained_value": 0.3}
        worse = await self._run({
            "verdict": "worse", "experiment_id": 12, "headline": headline,
            "evidence": {"verdict": "worse", "n": 21, "better": 3, "worse": 15, "same": 3},
        })
        self.assertEqual((_ids(worse), worse[0]["severity"]), (["export:worse-than-base"], "critical"))
        better = await self._run({
            "verdict": "better", "experiment_id": 12, "headline": headline,
            "evidence": {"verdict": "better", "n": 21, "better": 17, "worse": 4, "same": 0},
        }, last_export=SimpleNamespace(experiment_id=9))
        self.assertEqual(_ids(better), ["export:newer-better-run"])

    async def test_not_evaluated_and_same_warn(self):
        not_eval = await self._run({"verdict": "not_evaluated", "experiment_id": 12})
        same = await self._run({"verdict": "same", "experiment_id": 12, "headline": {}})
        self.assertEqual((_ids(not_eval), not_eval[0]["severity"]), (["export:not-evaluated"], "warning"))
        self.assertEqual(_ids(same), ["export:no-better-than-base"])

    async def test_better_run_newer_than_last_export(self):
        summary = {"verdict": "better", "experiment_id": 12, "headline": {}}
        stale = await self._run(summary, last_export=SimpleNamespace(experiment_id=9))
        current = await self._run(summary, last_export=SimpleNamespace(experiment_id=12))
        never = await self._run(summary, last_export=None)
        self.assertEqual(_ids(stale), ["export:newer-better-run"])
        self.assertEqual(current, [])
        self.assertEqual(never, [])


if __name__ == "__main__":
    unittest.main()

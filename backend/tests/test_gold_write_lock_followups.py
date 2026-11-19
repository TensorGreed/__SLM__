"""Gold writers share one lock and unique temp names: ``set_review`` and the eval-gap rebalance
autofix take ``gold_write_lock``; no fixed ``<file>.tmp`` is ever used, and none is left behind.
Pure unit tests (no TestClient, no Jobs, no DB engine)."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

from app.models.dataset import DatasetType
from app.services import eval_gap_service, gold_review_service, gold_service


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _tmps(d: Path) -> list[Path]:
    return sorted(d.glob("*.tmp"))


class _Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gold-lock-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.path = self.root / "gold_dev.jsonl"


class UniqueTempTests(_Base):
    def test_names_are_unique_and_beside_the_file(self):
        names = {gold_service.unique_tmp_path(self.path).name for _ in range(50)}
        self.assertEqual(len(names), 50)
        for n in names:
            self.assertTrue(n.startswith("gold_dev.jsonl.") and n.endswith(".tmp"))
            self.assertNotEqual(n, "gold_dev.jsonl.tmp")

    def test_write_rows_atomic_leaves_no_tmp(self):
        gold_service.write_rows_atomic(self.path, [{"id": 1}, {"id": 2}])
        self.assertEqual(len(self.path.read_text().splitlines()), 2)
        self.assertEqual(_tmps(self.root), [])

    def test_failed_write_leaves_no_tmp_and_keeps_old_file(self):
        _write(self.path, [{"id": 1}])
        with self.assertRaises(TypeError):
            gold_service.write_rows_atomic(self.path, [{"id": object()}])
        self.assertEqual(json.loads(self.path.read_text()), {"id": 1})
        self.assertEqual(_tmps(self.root), [])

    def test_atomic_append_cleans_up_on_failure_and_success(self):
        class _DB:
            def __init__(self, fail):
                self.fail = fail

            async def flush(self):
                if self.fail:
                    raise RuntimeError("boom")

            async def commit(self):
                pass

            async def rollback(self):
                pass

        ds = mock.Mock()
        _write(self.path, [{"id": 1}])
        asyncio.run(gold_service._atomic_append(_DB(False), ds, self.path, ['{"id": 2}\n']))
        self.assertEqual(len(self.path.read_text().splitlines()), 2)
        self.assertEqual(_tmps(self.root), [])
        with self.assertRaises(RuntimeError):
            asyncio.run(gold_service._atomic_append(_DB(True), ds, self.path, ['{"id": 3}\n']))
        self.assertEqual(len(self.path.read_text().splitlines()), 2)
        self.assertEqual(_tmps(self.root), [])


class SetReviewLockTests(_Base):
    def test_set_review_waits_for_the_lock_and_keeps_a_concurrent_append(self):
        gp = self.root / "gold_test.jsonl"
        _write(gp, [{"text": "a", "entities": []}])

        async def go():
            async with gold_service.gold_write_lock(7, DatasetType.GOLD_TEST):
                task = asyncio.ensure_future(gold_review_service.set_review(
                    None, 7, dataset_type="gold_test", index=0, key=None, verdict="accepted"))
                await asyncio.sleep(0.05)
                self.assertFalse(task.done(), "set_review ran without taking the lock")
                with open(gp, "a", encoding="utf-8") as f:  # a concurrent append
                    f.write(json.dumps({"text": "b", "entities": []}) + "\n")
            return await task

        with mock.patch.object(gold_review_service, "gold_path", lambda pid, dt: gp), \
             mock.patch.object(gold_review_service, "_sync_record_count", mock.AsyncMock()):
            asyncio.run(go())
        rows = gold_review_service.load_rows(gp)
        self.assertEqual(len(rows), 2)
        self.assertEqual(gold_review_service.review_of(rows[0])["verdict"], "accepted")
        self.assertEqual(_tmps(self.root), [])


class AutofixLockTests(_Base):
    def test_rebalance_autofix_takes_the_lock_and_rereads_rows(self):
        _write(self.path, [{"label": "a"}, {"label": "a"}, {"label": "b"}])
        entered: list[tuple] = []

        @asynccontextmanager
        async def spy(pid, dt):
            entered.append((pid, dt))
            # a concurrent append lands while the autofix waits for the lock
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"label": "b"}) + "\n")
            yield

        preview = {"patch_kind": "label_kl_rebalance_eval", "gold_dev_path": str(self.path),
                   "after": {"counts": {"a": 1, "b": 5}}}

        async def load(db, pid):
            return self.path, [json.loads(line) for line in self.path.read_text().splitlines()]

        class _Res:
            def scalars(self):
                return []

        db = mock.Mock()
        db.execute = mock.AsyncMock(return_value=_Res())
        with mock.patch.object(eval_gap_service, "preview_patch", mock.AsyncMock(return_value=preview)), \
             mock.patch.object(eval_gap_service, "_load_gold_dev_jsonl", load), \
             mock.patch.object(gold_service, "gold_write_lock", spy):
            out = asyncio.run(eval_gap_service.apply_patch(db, 3, "sig"))
        self.assertEqual(entered, [(3, DatasetType.GOLD_DEV)])
        self.assertEqual(out["rows_after"], 3)  # 1 of a, both b (incl. the concurrent one)
        labels = sorted(json.loads(line)["label"] for line in self.path.read_text().splitlines())
        self.assertEqual(labels, ["a", "b", "b"])
        self.assertEqual(_tmps(self.root), [])


if __name__ == "__main__":
    unittest.main()

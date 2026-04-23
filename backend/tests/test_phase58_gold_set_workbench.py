"""Phase 58 tests — gold-set annotation workbench backend (priority.md P10).

Exercises:
- `POST /gold-sets/{id}/rows/sample` (random + stratified + seed + dedup + reviewer auto-assign + empty source + non-gold 400 + stratify_by-missing 400)
- `PATCH /gold-sets/{id}/rows/{row_id}` (field updates, status transitions, reviewer reassignment / clearing, cross-gold-set 404)
- `GET  /gold-sets/{id}/queue` (reviewer/status filters, priority/assigned_at ordering)
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase58_gold_set_workbench.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase58_gold_set_workbench_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.auth import GlobalRole, User
from app.models.dataset import Dataset, DatasetType
from app.models.gold_set_annotation import (
    GoldSetReviewerQueue,
    GoldSetRow,
    GoldSetVersion,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class Phase58GoldSetWorkbenchTests(unittest.TestCase):
    @classmethod
    def _cleanup_artifacts(cls):
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        for suffix in ("", "-shm", "-wal"):
            path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

    @classmethod
    def setUpClass(cls):
        # Skip (rather than error) when the runtime DB binding drifted away
        # from our test file — e.g. an earlier-loaded test module froze
        # ``settings.DATABASE_URL`` onto a different path and the engine is
        # already connected. Refuse to run so we don't pollute the shared DB;
        # skip, not fail, so the suite still completes cleanly.
        resolved = str(settings.DATABASE_URL)
        if TEST_DB_PATH.resolve().as_posix() not in resolved:
            raise unittest.SkipTest(
                f"phase58 requires DATABASE_URL at {TEST_DB_PATH}, "
                f"but engine is bound to {resolved!r} (run this file standalone)."
            )
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        cls._cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_artifacts()

    # -- helpers ----------------------------------------------------------

    def _create_project(self, name: str = "phase58") -> int:
        unique = f"{name}-{uuid.uuid4().hex[:8]}"
        resp = self.client.post(
            "/api/projects",
            json={"name": unique, "description": "phase58 gold workbench"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _insert_dataset(
        self,
        project_id: int,
        *,
        name: str,
        dataset_type: DatasetType,
        file_path: str | None,
        record_count: int = 0,
    ) -> int:
        async with async_session_factory() as db:
            dataset = Dataset(
                project_id=project_id,
                name=name,
                dataset_type=dataset_type,
                file_path=file_path,
                record_count=record_count,
                metadata_={},
            )
            db.add(dataset)
            await db.commit()
            await db.refresh(dataset)
            return int(dataset.id)

    async def _insert_user(self, username: str) -> int:
        async with async_session_factory() as db:
            user = User(
                username=f"{username}-{uuid.uuid4().hex[:8]}",
                role=GlobalRole.ENGINEER,
                is_active=True,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return int(user.id)

    def _setup_source_jsonl(self, rows: list[dict], name: str) -> Path:
        path = TEST_DATA_DIR / f"{name}.jsonl"
        _write_jsonl(path, rows)
        return path

    def _create_gold_and_source(
        self,
        source_rows: list[dict],
        source_name: str = "phase58-source",
    ) -> tuple[int, int, int]:
        project_id = self._create_project()
        source_path = self._setup_source_jsonl(source_rows, source_name)
        gold_id = asyncio.run(
            self._insert_dataset(
                project_id,
                name="phase58-gold",
                dataset_type=DatasetType.GOLD_DEV,
                file_path=None,
            )
        )
        source_id = asyncio.run(
            self._insert_dataset(
                project_id,
                name=source_name,
                dataset_type=DatasetType.RAW,
                file_path=str(source_path),
                record_count=len(source_rows),
            )
        )
        return project_id, gold_id, source_id

    def _sample(self, gold_id: int, **body) -> dict:
        resp = self.client.post(
            f"/api/gold-sets/{gold_id}/rows/sample",
            json=body,
        )
        return resp

    # -- tests ------------------------------------------------------------

    def test_sample_random_creates_rows_in_draft_v1(self):
        _, gold_id, source_id = self._create_gold_and_source(
            [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]
        )
        resp = self._sample(
            gold_id,
            source_dataset_id=source_id,
            target_count=4,
            strategy="random",
            seed=42,
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["created"], 4)
        self.assertEqual(payload["skipped_duplicates"], 0)
        self.assertEqual(len(payload["rows"]), 4)
        for row in payload["rows"]:
            self.assertEqual(row["status"], "pending")
            self.assertIn("question", row["input"])

    def test_sample_stratified_distributes_across_buckets(self):
        rows = []
        for i in range(6):
            rows.append({"text": f"a{i}", "label": "A"})
        for i in range(6):
            rows.append({"text": f"b{i}", "label": "B"})
        for i in range(6):
            rows.append({"text": f"c{i}", "label": "C"})
        _, gold_id, source_id = self._create_gold_and_source(rows, source_name="stratified")

        resp = self._sample(
            gold_id,
            source_dataset_id=source_id,
            target_count=9,
            strategy="stratified",
            stratify_by="label",
            seed=7,
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()
        self.assertEqual(payload["created"], 9)
        label_counts: dict[str, int] = {}
        for row in payload["rows"]:
            label = row["input"]["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        # Each bucket should contribute roughly equally; with 3 equal buckets we
        # expect exactly 3 per bucket.
        self.assertEqual(sorted(label_counts.values()), [3, 3, 3])
        self.assertEqual(sorted(label_counts.keys()), ["A", "B", "C"])

    def test_sample_seed_is_reproducible(self):
        rows = [{"q": f"q{i}"} for i in range(30)]
        _, gold_id_a, source_id_a = self._create_gold_and_source(rows, source_name="seed-a")
        _, gold_id_b, source_id_b = self._create_gold_and_source(rows, source_name="seed-b")

        resp_a = self._sample(gold_id_a, source_dataset_id=source_id_a, target_count=5, seed=123)
        resp_b = self._sample(gold_id_b, source_dataset_id=source_id_b, target_count=5, seed=123)
        self.assertEqual(resp_a.status_code, 201)
        self.assertEqual(resp_b.status_code, 201)
        keys_a = [row["source_row_key"] for row in resp_a.json()["rows"]]
        keys_b = [row["source_row_key"] for row in resp_b.json()["rows"]]
        self.assertEqual(keys_a, keys_b)

    def test_sample_dedup_on_repeated_call(self):
        _, gold_id, source_id = self._create_gold_and_source(
            [{"q": f"q{i}"} for i in range(5)]
        )
        first = self._sample(gold_id, source_dataset_id=source_id, target_count=5, seed=1)
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(first.json()["created"], 5)

        # Re-sample the same pool entirely — all rows should be deduped.
        second = self._sample(gold_id, source_dataset_id=source_id, target_count=5, seed=1)
        self.assertEqual(second.status_code, 201, second.text)
        self.assertEqual(second.json()["created"], 0)
        self.assertEqual(second.json()["skipped_duplicates"], 5)

    def test_sample_auto_assigns_reviewer_queue_entries(self):
        _, gold_id, source_id = self._create_gold_and_source(
            [{"q": f"q{i}"} for i in range(5)]
        )
        reviewer_id = asyncio.run(self._insert_user("reviewer"))
        resp = self._sample(
            gold_id,
            source_dataset_id=source_id,
            target_count=3,
            reviewer_id=reviewer_id,
            seed=11,
        )
        self.assertEqual(resp.status_code, 201, resp.text)

        queue = self.client.get(f"/api/gold-sets/{gold_id}/queue")
        self.assertEqual(queue.status_code, 200, queue.text)
        entries = queue.json()["items"]
        self.assertEqual(len(entries), 3)
        for item in entries:
            self.assertEqual(item["reviewer_id"], reviewer_id)
            self.assertEqual(item["status"], "pending")
            self.assertIn("input_snippet", item["row_preview"])

    def test_sample_empty_source_returns_zero_not_error(self):
        _, gold_id, source_id = self._create_gold_and_source([])
        resp = self._sample(gold_id, source_dataset_id=source_id, target_count=5, seed=1)
        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()
        self.assertEqual(payload["created"], 0)
        self.assertEqual(payload["skipped_duplicates"], 0)
        self.assertEqual(payload["rows"], [])

    def test_sample_non_gold_dataset_returns_400(self):
        project_id = self._create_project()
        source_path = self._setup_source_jsonl([{"q": "q"}], "non-gold-src")
        raw_id = asyncio.run(
            self._insert_dataset(
                project_id,
                name="raw-only",
                dataset_type=DatasetType.RAW,
                file_path=str(source_path),
                record_count=1,
            )
        )
        # Using the RAW dataset as the "gold set" must be rejected.
        resp = self.client.post(
            f"/api/gold-sets/{raw_id}/rows/sample",
            json={"source_dataset_id": raw_id, "target_count": 1},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["detail"], "not_a_gold_set")

    def test_sample_stratified_without_field_returns_400(self):
        _, gold_id, source_id = self._create_gold_and_source([{"q": "q"}])
        resp = self._sample(
            gold_id,
            source_dataset_id=source_id,
            target_count=1,
            strategy="stratified",
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["detail"], "stratify_by_required")

    def test_patch_row_updates_fields_and_sets_reviewed_at_on_terminal_status(self):
        _, gold_id, source_id = self._create_gold_and_source(
            [{"q": f"q{i}"} for i in range(3)]
        )
        sample = self._sample(gold_id, source_dataset_id=source_id, target_count=3, seed=5)
        row_id = sample.json()["rows"][0]["id"]

        patch = self.client.patch(
            f"/api/gold-sets/{gold_id}/rows/{row_id}",
            json={
                "expected": {"answer": "42"},
                "rationale": "42 is canonical.",
                "labels": {"difficulty": "medium", "flags": ["canonical"]},
                "status": "approved",
            },
        )
        self.assertEqual(patch.status_code, 200, patch.text)
        body = patch.json()
        self.assertEqual(body["expected"], {"answer": "42"})
        self.assertEqual(body["labels"]["difficulty"], "medium")
        self.assertEqual(body["rationale"], "42 is canonical.")
        self.assertEqual(body["status"], "approved")
        self.assertIsNotNone(body["reviewed_at"])

    def test_patch_reviewer_change_moves_queue_entry(self):
        _, gold_id, source_id = self._create_gold_and_source(
            [{"q": f"q{i}"} for i in range(2)]
        )
        reviewer_a = asyncio.run(self._insert_user("rev-a"))
        reviewer_b = asyncio.run(self._insert_user("rev-b"))

        sample = self._sample(
            gold_id,
            source_dataset_id=source_id,
            target_count=2,
            reviewer_id=reviewer_a,
            seed=2,
        )
        row_id = sample.json()["rows"][0]["id"]

        # Reassign to reviewer B.
        reassign = self.client.patch(
            f"/api/gold-sets/{gold_id}/rows/{row_id}",
            json={"reviewer_id": reviewer_b, "status": "in_review"},
        )
        self.assertEqual(reassign.status_code, 200, reassign.text)

        q_for_b = self.client.get(
            f"/api/gold-sets/{gold_id}/queue",
            params={"reviewer_id": reviewer_b},
        )
        self.assertEqual(q_for_b.status_code, 200)
        for_b_rows = [item["row_id"] for item in q_for_b.json()["items"]]
        self.assertIn(row_id, for_b_rows)

        q_for_a = self.client.get(
            f"/api/gold-sets/{gold_id}/queue",
            params={"reviewer_id": reviewer_a},
        )
        self.assertEqual(q_for_a.status_code, 200)
        for_a_rows = [item["row_id"] for item in q_for_a.json()["items"]]
        self.assertNotIn(row_id, for_a_rows)

        # Now clear the reviewer entirely — queue entry should vanish.
        clear = self.client.patch(
            f"/api/gold-sets/{gold_id}/rows/{row_id}",
            json={"reviewer_id": None},
        )
        self.assertEqual(clear.status_code, 200, clear.text)
        q_after_clear = self.client.get(
            f"/api/gold-sets/{gold_id}/queue",
            params={"reviewer_id": reviewer_b},
        )
        rows_after = [item["row_id"] for item in q_after_clear.json()["items"]]
        self.assertNotIn(row_id, rows_after)

    def test_patch_cross_gold_set_returns_404(self):
        # Build two gold sets under the same project, sample into gold A, then
        # try to PATCH gold A's row via gold B's id.
        project_id = self._create_project()
        source_path = self._setup_source_jsonl([{"q": f"q{i}"} for i in range(2)], "cross")
        gold_a = asyncio.run(
            self._insert_dataset(
                project_id,
                name="gold-a",
                dataset_type=DatasetType.GOLD_DEV,
                file_path=None,
            )
        )
        gold_b = asyncio.run(
            self._insert_dataset(
                project_id,
                name="gold-b",
                dataset_type=DatasetType.GOLD_TEST,
                file_path=None,
            )
        )
        source_id = asyncio.run(
            self._insert_dataset(
                project_id,
                name="src",
                dataset_type=DatasetType.RAW,
                file_path=str(source_path),
                record_count=2,
            )
        )
        sample = self._sample(gold_a, source_dataset_id=source_id, target_count=2, seed=3)
        row_id = sample.json()["rows"][0]["id"]

        mis = self.client.patch(
            f"/api/gold-sets/{gold_b}/rows/{row_id}",
            json={"status": "approved"},
        )
        self.assertEqual(mis.status_code, 404, mis.text)
        self.assertEqual(mis.json()["detail"], "row_not_found")

    def test_queue_filters_by_status_and_reviewer(self):
        _, gold_id, source_id = self._create_gold_and_source(
            [{"q": f"q{i}"} for i in range(5)]
        )
        reviewer = asyncio.run(self._insert_user("rev-filter"))
        self._sample(
            gold_id,
            source_dataset_id=source_id,
            target_count=4,
            reviewer_id=reviewer,
            seed=9,
        )

        # Move one row to in_review, one to approved.
        rows_resp = self.client.get(f"/api/gold-sets/{gold_id}/queue")
        row_ids = [item["row_id"] for item in rows_resp.json()["items"]]
        self.client.patch(
            f"/api/gold-sets/{gold_id}/rows/{row_ids[0]}",
            json={"status": "in_review"},
        )
        self.client.patch(
            f"/api/gold-sets/{gold_id}/rows/{row_ids[1]}",
            json={"status": "approved"},
        )

        pending = self.client.get(
            f"/api/gold-sets/{gold_id}/queue",
            params={"status": "pending"},
        )
        self.assertEqual(pending.status_code, 200)
        self.assertEqual(pending.json()["count"], 2)

        in_progress = self.client.get(
            f"/api/gold-sets/{gold_id}/queue",
            params={"status": "in_progress"},
        )
        self.assertEqual(in_progress.json()["count"], 1)

        completed = self.client.get(
            f"/api/gold-sets/{gold_id}/queue",
            params={"status": "completed"},
        )
        self.assertEqual(completed.json()["count"], 1)

        other_reviewer = self.client.get(
            f"/api/gold-sets/{gold_id}/queue",
            params={"reviewer_id": reviewer + 999},
        )
        self.assertEqual(other_reviewer.json()["count"], 0)

    def test_version_is_created_on_first_sample_and_reused_on_second(self):
        _, gold_id, source_id = self._create_gold_and_source(
            [{"q": f"q{i}"} for i in range(10)]
        )
        self._sample(gold_id, source_dataset_id=source_id, target_count=2, seed=1)
        self._sample(gold_id, source_dataset_id=source_id, target_count=2, seed=2)

        async def count_versions() -> int:
            from sqlalchemy import func, select

            async with async_session_factory() as db:
                result = await db.execute(
                    select(func.count(GoldSetVersion.id)).where(
                        GoldSetVersion.gold_set_id == gold_id
                    )
                )
                return int(result.scalar_one() or 0)

        self.assertEqual(asyncio.run(count_versions()), 1)

    def test_service_persists_rows_and_queue_rows_to_db(self):
        """Covers serialization-independent DB-level invariants."""
        _, gold_id, source_id = self._create_gold_and_source(
            [{"q": f"q{i}"} for i in range(3)]
        )
        reviewer = asyncio.run(self._insert_user("persistence"))
        self._sample(
            gold_id,
            source_dataset_id=source_id,
            target_count=3,
            reviewer_id=reviewer,
            seed=4,
        )

        async def counts() -> tuple[int, int]:
            from sqlalchemy import func, select

            async with async_session_factory() as db:
                row_count = int(
                    (
                        await db.execute(
                            select(func.count(GoldSetRow.id)).where(
                                GoldSetRow.gold_set_id == gold_id
                            )
                        )
                    ).scalar_one()
                    or 0
                )
                queue_count = int(
                    (
                        await db.execute(
                            select(func.count(GoldSetReviewerQueue.id)).where(
                                GoldSetReviewerQueue.gold_set_id == gold_id
                            )
                        )
                    ).scalar_one()
                    or 0
                )
                return row_count, queue_count

        row_count, queue_count = asyncio.run(counts())
        self.assertEqual(row_count, 3)
        self.assertEqual(queue_count, 3)


if __name__ == "__main__":
    unittest.main()

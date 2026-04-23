"""Phase 60 regression test — gold service tolerates duplicate gold datasets.

A real user hit ``MultipleResultsFound`` on ``POST /projects/{id}/gold/add``
because their database had two ``(project_id=1, dataset_type=GOLD_DEV)`` rows
from a double-click race against the unguarded
``get_or_create_gold_dataset`` lookup. The fix switches from
``scalar_one_or_none`` to ``scalars().first()`` with a stable order-by. This
test pins that behavior so the crash cannot regress.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase60_gold_dataset_duplicates.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase60_gold_dataset_duplicates_data"

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
from app.models.dataset import Dataset, DatasetType
from app.services.gold_service import add_qa_pair, get_or_create_gold_dataset


class Phase60GoldDatasetDuplicateTests(unittest.TestCase):
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
        resolved = str(settings.DATABASE_URL)
        if TEST_DB_PATH.resolve().as_posix() not in resolved:
            raise unittest.SkipTest(
                f"phase60 requires DATABASE_URL at {TEST_DB_PATH}, "
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

    def _create_project(self) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"phase60-{uuid.uuid4().hex[:8]}", "description": "phase60"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _seed_duplicate_gold_datasets(self, project_id: int) -> tuple[int, int]:
        async with async_session_factory() as db:
            first = Dataset(
                project_id=project_id,
                name="Gold Dev Set",
                dataset_type=DatasetType.GOLD_DEV,
                description="first",
            )
            second = Dataset(
                project_id=project_id,
                name="Gold Dev Set",
                dataset_type=DatasetType.GOLD_DEV,
                description="second (race)",
            )
            db.add(first)
            db.add(second)
            await db.commit()
            await db.refresh(first)
            await db.refresh(second)
            return int(first.id), int(second.id)

    def test_get_or_create_returns_lowest_id_when_duplicates_exist(self):
        project_id = self._create_project()
        first_id, second_id = asyncio.run(self._seed_duplicate_gold_datasets(project_id))
        self.assertLess(first_id, second_id, "fixture precondition: first insert has lower id")

        async def _call():
            async with async_session_factory() as db:
                return await get_or_create_gold_dataset(
                    db, project_id, DatasetType.GOLD_DEV
                )

        dataset = asyncio.run(_call())
        # The fix must pick the deterministic lowest-id match without raising.
        self.assertEqual(dataset.id, first_id)

    def test_add_qa_pair_succeeds_when_duplicate_gold_datasets_exist(self):
        project_id = self._create_project()
        asyncio.run(self._seed_duplicate_gold_datasets(project_id))

        async def _call():
            async with async_session_factory() as db:
                result = await add_qa_pair(
                    db,
                    project_id,
                    question="What is the capital of Canada?",
                    answer="Ottawa",
                )
                await db.commit()
                return result

        entry = asyncio.run(_call())
        # Before the fix this raised MultipleResultsFound (500 in the API).
        # After the fix it returns a well-formed Q&A entry.
        self.assertEqual(entry["question"], "What is the capital of Canada?")
        self.assertEqual(entry["answer"], "Ottawa")

    def test_http_add_endpoint_does_not_500_with_duplicates(self):
        project_id = self._create_project()
        asyncio.run(self._seed_duplicate_gold_datasets(project_id))

        resp = self.client.post(
            f"/api/projects/{project_id}/gold/add",
            json={
                "question": "Which court hears constitutional cases in Canada?",
                "answer": "Supreme Court of Canada",
            },
        )
        self.assertIn(resp.status_code, (200, 201), resp.text)
        payload = resp.json()
        self.assertEqual(payload["answer"], "Supreme Court of Canada")


if __name__ == "__main__":
    unittest.main()

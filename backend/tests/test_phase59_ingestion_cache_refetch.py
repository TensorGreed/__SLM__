"""Phase 59 tests — remote-import idempotency check respects requested row count.

The previous check short-circuited on `(source, split, config)` only, so a user
who first imported 200 rows and then asked for 10000 got the cached 200 back
without any re-fetch. This phase verifies:

- Repeat imports with `max_samples <= cached_rows` still hit the cache.
- Imports with `max_samples > cached_rows` fall through and re-fetch.
- Different splits remain independent of each other's caches.
"""

from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


TEST_DB_PATH = Path(__file__).resolve().parent / "phase59_ingestion_cache_refetch.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase59_ingestion_cache_refetch_data"

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
from app.services.ingestion_service import ingest_remote_dataset


def _fake_hf_dataset(row_count: int) -> list[dict]:
    """Return a deterministic list of dict rows resembling an HF dataset."""
    return [
        {
            "instruction": f"row-{i}",
            "input": f"input-{i}",
            "output": f"output-{i}",
        }
        for i in range(row_count)
    ]


class Phase59IngestionCacheRefetchTests(unittest.TestCase):
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

    def _create_project(self, name: str = "phase59") -> int:
        unique = f"{name}-{uuid.uuid4().hex[:8]}"
        resp = self.client.post(
            "/api/projects",
            json={"name": unique, "description": "phase59 import cache"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _ingest(
        self,
        *,
        project_id: int,
        max_samples: int | None,
        split: str = "train",
        identifier: str = "fake/dataset-a",
    ):
        async with async_session_factory() as db:
            result = await ingest_remote_dataset(
                db=db,
                project_id=project_id,
                source_type="huggingface",
                identifier=identifier,
                split=split,
                max_samples=max_samples,
                use_saved_secrets=False,
            )
            await db.commit()
            return result

    # -- tests ------------------------------------------------------------

    def test_larger_max_samples_triggers_refetch(self):
        import asyncio

        project_id = self._create_project("cache-miss")

        # First call: fetch 50 rows.
        with patch("datasets.load_dataset", return_value=_fake_hf_dataset(200)):
            first = asyncio.run(self._ingest(project_id=project_id, max_samples=50))
        self.assertFalse(first.get("already_exists"), first)
        self.assertEqual(first["raw_samples"], 50)

        # Second call asks for more than the cached 50 — should re-fetch.
        with patch("datasets.load_dataset", return_value=_fake_hf_dataset(500)):
            second = asyncio.run(self._ingest(project_id=project_id, max_samples=300))
        self.assertFalse(second.get("already_exists"), second)
        self.assertEqual(second["raw_samples"], 300)
        self.assertNotEqual(second["document_id"], first["document_id"])

    def test_equal_or_smaller_max_samples_reuses_cache(self):
        import asyncio

        project_id = self._create_project("cache-hit")

        with patch("datasets.load_dataset", return_value=_fake_hf_dataset(200)):
            first = asyncio.run(self._ingest(project_id=project_id, max_samples=100))
        self.assertFalse(first.get("already_exists"))
        self.assertEqual(first["raw_samples"], 100)

        # Second call with equal max_samples → cache hit, no re-fetch.
        load_calls: list[tuple] = []

        def _fail_on_call(*args, **kwargs):
            load_calls.append((args, kwargs))
            return _fake_hf_dataset(100)  # would be taken only if re-fetched

        with patch("datasets.load_dataset", side_effect=_fail_on_call):
            second = asyncio.run(self._ingest(project_id=project_id, max_samples=100))
        self.assertTrue(second.get("already_exists"))
        self.assertEqual(second["raw_samples"], 100)
        self.assertEqual(second["document_id"], first["document_id"])
        self.assertEqual(load_calls, [])  # no upstream load

        # Third call with a smaller max_samples → still cache hit.
        with patch("datasets.load_dataset", side_effect=_fail_on_call):
            third = asyncio.run(self._ingest(project_id=project_id, max_samples=25))
        self.assertTrue(third.get("already_exists"))
        self.assertEqual(third["document_id"], first["document_id"])
        self.assertEqual(load_calls, [])

    def test_cache_picks_largest_existing_when_multiple_match(self):
        """If a project has both a 50-row and a 300-row cache of the same
        `(source, split, config)`, a request for 200 rows should reuse the
        300-row document, not the 50-row one."""
        import asyncio

        project_id = self._create_project("cache-largest")

        # Force two separate imports of increasing sizes.
        with patch("datasets.load_dataset", return_value=_fake_hf_dataset(400)):
            first = asyncio.run(self._ingest(project_id=project_id, max_samples=50))
            second = asyncio.run(self._ingest(project_id=project_id, max_samples=300))
        self.assertFalse(first.get("already_exists"))
        self.assertFalse(second.get("already_exists"))

        # A request for 200 rows should reuse the 300-row cached doc.
        with patch("datasets.load_dataset", return_value=_fake_hf_dataset(400)):
            third = asyncio.run(self._ingest(project_id=project_id, max_samples=200))
        self.assertTrue(third.get("already_exists"))
        self.assertEqual(third["document_id"], second["document_id"])
        self.assertEqual(third["raw_samples"], 300)

    def test_different_split_does_not_share_cache(self):
        """A cached import of split=train must not satisfy a request for split=test."""
        import asyncio

        project_id = self._create_project("split-isolation")

        with patch("datasets.load_dataset", return_value=_fake_hf_dataset(200)):
            train = asyncio.run(
                self._ingest(project_id=project_id, max_samples=100, split="train")
            )
            test = asyncio.run(
                self._ingest(project_id=project_id, max_samples=50, split="test")
            )

        self.assertFalse(train.get("already_exists"))
        self.assertFalse(test.get("already_exists"))
        self.assertNotEqual(train["document_id"], test["document_id"])


if __name__ == "__main__":
    unittest.main()

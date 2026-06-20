"""Epic G phase G2 — demo sample reset lifecycle.

Isolated DB (not the shared phase87 suite) so the reset's delete +
re-seed doesn't collide with order-sensitive shared-state tests via
SQLite rowid reuse.

Reset proof: ``seed_demo_project`` returns ``created=False`` when it finds
an existing project, so ``created=True`` *after* a reset means the old
project was dropped and a fresh one was seeded.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "demo_reset.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "demo_reset_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

import asyncio  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import Dataset  # noqa: E402
from app.models.gold_set_annotation import (  # noqa: E402
    GoldSetReviewerQueue,
    GoldSetRow,
    GoldSetVersion,
)


class DemoResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()
        # Defensive clean slate: under the shared-engine isolation gotcha, an
        # earlier demo test file (run first in the same process) may have left
        # projects in the DB. Our `created=True` assertions assume an empty DB,
        # so purge any leftovers before seeding. Order-independent now.
        cls._purge_all_projects()

    def setUp(self):
        # Each test assumes a clean DB (the `created=True` assertions only hold
        # on a first seed). Purge all projects before every test so the methods
        # are order-independent within the class.
        self._purge_all_projects()

    @classmethod
    def _purge_all_projects(cls):
        try:
            listing = cls.client.get("/api/projects?limit=500")
            for proj in listing.json().get("projects", []):
                cls.client.delete(f"/api/projects/{proj['id']}")
        except Exception:
            pass

    @classmethod
    def tearDownClass(cls):
        # This file seeds + resets real demo projects, leaving them (and their
        # gold-set workbench rows) in the DB. Because every test module shares a
        # single SQLAlchemy engine (created from the first-imported test file),
        # those leftovers pollute *other* demo test files run in the same pytest
        # process (e.g. test_phase87_demo_projects, which expects a fresh seed).
        # Delete every project on the way out so the shared DB is left clean. The
        # DELETE endpoint now purges gold-set artifacts too, so nothing leaks.
        cls._purge_all_projects()
        cls._cm.__exit__(None, None, None)

    def test_reset_drops_existing_and_reseeds_fresh_in_beginner_mode(self):
        # Seed once (creates), then seed again (idempotent → not created).
        first = self.client.post("/api/demo-projects/sentiment-classifier")
        self.assertEqual(first.status_code, 200, first.text)
        self.assertTrue(first.json()["summary"]["created"])
        again = self.client.post("/api/demo-projects/sentiment-classifier")
        self.assertFalse(again.json()["summary"]["created"])  # idempotent

        # Reset → the old project is dropped and a fresh one is seeded, so
        # created flips back to True. Sample stays in beginner mode.
        reset = self.client.post("/api/demo-projects/sentiment-classifier/reset")
        self.assertEqual(reset.status_code, 200, reset.text)
        body = reset.json()
        self.assertTrue(body["summary"]["reset"])
        self.assertTrue(body["summary"]["created"])
        self.assertTrue(body["project"]["beginner_mode"])

    def test_reset_on_unseeded_slug_just_seeds(self):
        # Reset is valid even with nothing to drop (first-run path).
        resp = self.client.post("/api/demo-projects/pii-detector/reset")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["summary"]["created"])

    def test_reset_unknown_slug_returns_404(self):
        resp = self.client.post("/api/demo-projects/no-such-demo/reset")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertIn("demo_slug_unknown", resp.json()["detail"])

    def test_reset_does_not_leak_gold_set_rows(self):
        # The bug: db.delete(project) cascades datasets but NOT the gold-set
        # workbench tables (no ORM relationship; ambiguous multi-FK), so a
        # reset left orphaned gold versions/rows pointing at freed dataset ids.
        # After the fix (purge_gold_sets_for_project), every gold artifact in
        # the DB must reference a dataset that still exists.
        seed = self.client.post("/api/demo-projects/sentiment-classifier")
        self.assertEqual(seed.status_code, 200, seed.text)
        reset = self.client.post("/api/demo-projects/sentiment-classifier/reset")
        self.assertEqual(reset.status_code, 200, reset.text)

        async def _count_orphans():
            async with async_session_factory() as db:
                live = set(
                    (await db.execute(select(Dataset.id))).scalars().all()
                )
                orphans = 0
                for model in (GoldSetVersion, GoldSetRow, GoldSetReviewerQueue):
                    gold_set_ids = (
                        await db.execute(select(model.gold_set_id))
                    ).scalars().all()
                    orphans += sum(1 for gsid in gold_set_ids if gsid not in live)
                return orphans

        self.assertEqual(asyncio.run(_count_orphans()), 0)


if __name__ == "__main__":
    unittest.main()

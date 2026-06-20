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

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


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

    @classmethod
    def tearDownClass(cls):
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


if __name__ == "__main__":
    unittest.main()

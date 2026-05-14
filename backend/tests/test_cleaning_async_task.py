"""Cleaning background-task pattern (Phase F UX fix for 100K-row cleans).

Pins:
- POST /clean-batch-async returns 202 + a task_id + initial status.
- GET /cleaning/tasks/{task_id} reports running → completed/failed.
- The task survives missing-document errors (per-doc errors recorded
  on the task; status still 'completed').
- 404 path for an unknown task id + a task id from another project.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


TEST_DATA_DIR = Path(tempfile.gettempdir()) / f"brewslm-cleaning-async-{uuid.uuid4().hex[:8]}"


class CleaningAsyncTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"{name}-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _wait_terminal(self, pid: int, task_id: str, timeout: float = 5.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self.client.get(
                f"/api/projects/{pid}/cleaning/tasks/{task_id}"
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            payload = resp.json()
            if payload["status"] in ("completed", "failed"):
                return payload
            time.sleep(0.05)
        self.fail(f"task {task_id} did not finish within {timeout}s")

    def test_start_returns_202_with_task_id(self):
        pid = self._create_project("cleaning-async-start")
        resp = self.client.post(
            f"/api/projects/{pid}/cleaning/clean-batch-async",
            json={"document_ids": [999_001]},
        )
        self.assertEqual(resp.status_code, 202, resp.text)
        body = resp.json()
        self.assertTrue(body["task_id"].startswith("clean-"))
        self.assertIn(body["status"], ("pending", "running", "completed", "failed"))

    def test_task_completes_with_per_doc_errors_for_missing_documents(self):
        # No documents exist for this project; the task should run to
        # completion and record one entry per missing-doc error.
        pid = self._create_project("cleaning-async-missing")
        resp = self.client.post(
            f"/api/projects/{pid}/cleaning/clean-batch-async",
            json={"document_ids": [999_002, 999_003]},
        )
        self.assertEqual(resp.status_code, 202)
        body = resp.json()
        final = self._wait_terminal(pid, body["task_id"])
        self.assertEqual(final["status"], "completed")
        self.assertEqual(len(final["errors"]), 2)
        self.assertEqual(final["results"], [])
        # Reason captured per-document for the UI.
        reasons = [err["error"] for err in final["errors"]]
        self.assertTrue(all("not found" in r for r in reasons))

    def test_unknown_task_id_returns_404(self):
        pid = self._create_project("cleaning-async-404")
        resp = self.client.get(
            f"/api/projects/{pid}/cleaning/tasks/nope-xyz"
        )
        self.assertEqual(resp.status_code, 404)

    def test_task_from_other_project_returns_404(self):
        pid_a = self._create_project("cleaning-async-cross-a")
        pid_b = self._create_project("cleaning-async-cross-b")
        start = self.client.post(
            f"/api/projects/{pid_a}/cleaning/clean-batch-async",
            json={"document_ids": [999_004]},
        )
        task_id = start.json()["task_id"]
        # Right project — visible.
        self.assertEqual(
            self.client.get(
                f"/api/projects/{pid_a}/cleaning/tasks/{task_id}"
            ).status_code,
            200,
        )
        # Wrong project — 404.
        self.assertEqual(
            self.client.get(
                f"/api/projects/{pid_b}/cleaning/tasks/{task_id}"
            ).status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()

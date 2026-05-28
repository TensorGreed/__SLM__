"""Teacher logit capture — Track 1, Epic A, slice 1.

Pins:
- POST /distillation/capture returns 202 + a 'distill-' task_id.
- With the teacher call mocked, the task completes and writes captured
  rows carrying teacher_logits + provenance to the distillation artifact.
- Per-row failures land in chunk_errors but the batch still completes.
- `limit` caps the captured rows.
- A distillation_teacher_capture RunEvent is emitted (asserted via the
  run-events API).
- 404 paths: unknown dataset on start; unknown task id; cross-project task.
"""

from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import os

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

_MODULE_CLIENT_CM = TestClient(app)

_CAPTURE_FN = "app.services.distillation.teacher_capture.call_teacher_with_logprobs"


def setUpModule() -> None:  # noqa: N802 — unittest convention
    # A non-empty teacher URL makes capture_teacher_outputs pass its
    # config pre-check; the actual call is mocked in every test.
    settings.TEACHER_MODEL_API_URL = "http://teacher.local/v1/chat/completions"
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802 — unittest convention
    _MODULE_CLIENT_CM.__exit__(None, None, None)


async def _fake_teacher(prompt: str, **kwargs):
    """Deterministic stand-in for the OpenAI-compatible logprobs call."""
    return {
        "content": "positive",
        "teacher_logits": [
            {"token": "pos", "top_k": [["pos", -0.05], ["neg", -3.0]]},
        ],
        "model": kwargs.get("model_name", "fake-teacher"),
    }


class TeacherCaptureTests(unittest.TestCase):
    client: TestClient

    @classmethod
    def setUpClass(cls):
        cls.client = _MODULE_CLIENT_CM

    # ── helpers ────────────────────────────────────────────────────────

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects", json={"name": f"{name}-{uuid.uuid4().hex[:6]}"}
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_dataset(self, project_id: int, rows: list[dict]) -> int:
        from app.database import async_session_factory
        from app.models.dataset import Dataset, DatasetType

        src_dir = Path(settings.DATA_DIR) / "projects" / str(project_id) / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        file_path = src_dir / "source.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        async def _create() -> int:
            async with async_session_factory() as db:
                ds = Dataset(
                    project_id=project_id,
                    name="Source",
                    dataset_type=DatasetType.SYNTHETIC,
                    file_path=str(file_path),
                    record_count=len(rows),
                )
                db.add(ds)
                await db.flush()
                await db.refresh(ds)
                ds_id = ds.id
                await db.commit()
                return ds_id

        return asyncio.run(_create())

    def _wait_terminal(self, pid: int, task_id: str, timeout: float = 6.0) -> dict:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self.client.get(
                f"/api/projects/{pid}/distillation/tasks/{task_id}"
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            payload = resp.json()
            if payload["status"] in ("completed", "failed"):
                return payload
            time.sleep(0.05)
        self.fail(f"capture task {task_id} did not finish within {timeout}s")

    def _sample_rows(self, n: int) -> list[dict]:
        return [
            {"question": f"Is item {i} good?", "answer": "positive"}
            for i in range(n)
        ]

    # ── tests ──────────────────────────────────────────────────────────

    def test_start_returns_202_with_task_id(self):
        pid = self._create_project("distill-start")
        ds_id = self._seed_dataset(pid, self._sample_rows(3))
        with patch(_CAPTURE_FN, new=_fake_teacher):
            resp = self.client.post(
                f"/api/projects/{pid}/distillation/capture",
                json={"dataset_id": ds_id, "top_k": 5},
            )
            self.assertEqual(resp.status_code, 202, resp.text)
            body = resp.json()
            self.assertTrue(body["task_id"].startswith("distill-"))
            self.assertEqual(body["dataset_id"], ds_id)
            self.assertEqual(body["top_k"], 5)
            self._wait_terminal(pid, body["task_id"])

    def test_capture_writes_rows_with_logits_and_provenance(self):
        pid = self._create_project("distill-happy")
        ds_id = self._seed_dataset(pid, self._sample_rows(4))
        with patch(_CAPTURE_FN, new=_fake_teacher):
            start = self.client.post(
                f"/api/projects/{pid}/distillation/capture",
                json={"dataset_id": ds_id},
            )
            task_id = start.json()["task_id"]
            final = self._wait_terminal(pid, task_id)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["produced_count"], 4)
        self.assertEqual(final["chunk_errors"], [])
        self.assertTrue(final["written_path"])

        lines = [
            json.loads(line)
            for line in Path(final["written_path"]).read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), 4)
        first = lines[0]
        self.assertEqual(first["source"], "teacher_capture")
        self.assertEqual(first["status"], "accepted")
        self.assertIn("teacher_logits", first)
        self.assertEqual(first["teacher_logits"][0]["token"], "pos")
        self.assertEqual(first["teacher_logits"][0]["top_k"][0], ["pos", -0.05])
        # Original payload is preserved alongside the captured logits.
        self.assertIn("question", first)
        self.assertEqual(first["answer"], "positive")

    def test_per_row_failure_isolated_in_chunk_errors(self):
        pid = self._create_project("distill-isolate")
        ds_id = self._seed_dataset(pid, self._sample_rows(3))

        state = {"calls": 0}

        async def _flaky(prompt: str, **kwargs):
            state["calls"] += 1
            if state["calls"] == 2:
                raise RuntimeError("teacher timeout")
            return await _fake_teacher(prompt, **kwargs)

        with patch(_CAPTURE_FN, new=_flaky):
            start = self.client.post(
                f"/api/projects/{pid}/distillation/capture",
                json={"dataset_id": ds_id},
            )
            final = self._wait_terminal(pid, start.json()["task_id"])

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["produced_count"], 2)
        self.assertEqual(len(final["chunk_errors"]), 1)
        self.assertIn("teacher timeout", final["chunk_errors"][0]["error"])

    def test_limit_caps_captured_rows(self):
        pid = self._create_project("distill-limit")
        ds_id = self._seed_dataset(pid, self._sample_rows(5))
        with patch(_CAPTURE_FN, new=_fake_teacher):
            start = self.client.post(
                f"/api/projects/{pid}/distillation/capture",
                json={"dataset_id": ds_id, "limit": 2},
            )
            final = self._wait_terminal(pid, start.json()["task_id"])
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["produced_count"], 2)

    def test_run_event_emitted(self):
        pid = self._create_project("distill-event")
        ds_id = self._seed_dataset(pid, self._sample_rows(2))
        with patch(_CAPTURE_FN, new=_fake_teacher):
            start = self.client.post(
                f"/api/projects/{pid}/distillation/capture",
                json={"dataset_id": ds_id},
            )
            self._wait_terminal(pid, start.json()["task_id"])

        events = self.client.get(
            f"/api/projects/{pid}/run-events", params={"stage": "ingestion"}
        )
        self.assertEqual(events.status_code, 200, events.text)
        codes = [e["reason_code"] for e in events.json()["events"]]
        self.assertIn("distillation_teacher_capture", codes)

    def test_capture_unknown_dataset_returns_404(self):
        pid = self._create_project("distill-no-ds")
        resp = self.client.post(
            f"/api/projects/{pid}/distillation/capture",
            json={"dataset_id": 999_999},
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_unknown_task_id_returns_404(self):
        pid = self._create_project("distill-404")
        resp = self.client.get(
            f"/api/projects/{pid}/distillation/tasks/nope-xyz"
        )
        self.assertEqual(resp.status_code, 404)

    def test_task_from_other_project_returns_404(self):
        pid_a = self._create_project("distill-cross-a")
        pid_b = self._create_project("distill-cross-b")
        ds_id = self._seed_dataset(pid_a, self._sample_rows(2))
        with patch(_CAPTURE_FN, new=_fake_teacher):
            start = self.client.post(
                f"/api/projects/{pid_a}/distillation/capture",
                json={"dataset_id": ds_id},
            )
            task_id = start.json()["task_id"]
            self.assertEqual(
                self.client.get(
                    f"/api/projects/{pid_a}/distillation/tasks/{task_id}"
                ).status_code,
                200,
            )
            self.assertEqual(
                self.client.get(
                    f"/api/projects/{pid_b}/distillation/tasks/{task_id}"
                ).status_code,
                404,
            )
            self._wait_terminal(pid_a, task_id)


if __name__ == "__main__":
    unittest.main()

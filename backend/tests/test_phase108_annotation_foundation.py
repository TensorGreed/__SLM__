"""Phase 108 — Annotation foundation (Story 1.1).

Pins:
- label-job CRUD + label_type validation.
- seed-from-dataset materializes LabelRows from a project's dataset file.
- next-row hands one unlabeled row to a reviewer + flags it assigned.
- Two next-row calls never return the same row (the compare-and-set
  contract assign_next promises).
- submit-label persists the payload + emits an
  ``annotation_label_submitted`` RunEvent.
- Job stats reports total / labeled / assigned / unlabeled.
- Job creation emits ``annotation_job_created``.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import Dataset, DatasetType  # noqa: E402
from app.models.reason_codes import (  # noqa: E402
    ANNOTATION_JOB_CREATED,
    ANNOTATION_LABEL_SUBMITTED,
)
from app.models.run_event import (  # noqa: E402
    SEVERITY_INFO,
    STAGE_INGESTION,
    RunEvent,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir()) / f"brewslm-phase108-{uuid.uuid4().hex[:8]}"
)


def _write_jsonl(rows: list[dict]) -> str:
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    try:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    finally:
        fh.close()
    return fh.name


async def _create_dataset(
    project_id: int, file_path: str, record_count: int
) -> int:
    async with async_session_factory() as session:
        ds = Dataset(
            project_id=project_id,
            name="phase108-fixture",
            dataset_type=DatasetType.SYNTHETIC,
            record_count=record_count,
            file_path=file_path,
        )
        session.add(ds)
        await session.commit()
        return int(ds.id)


def _create_fixture_dataset(project_id: int, n_rows: int = 6) -> int:
    """Build a JSONL file with ``n_rows`` rows + a Dataset row that
    points at it. Returns the dataset id."""
    rows = [
        {
            "id": i + 1,
            "text": f"sample text number {i + 1}",
            "label": "positive" if i % 2 == 0 else "negative",
        }
        for i in range(n_rows)
    ]
    path = _write_jsonl(rows)
    dataset_id = asyncio.run(
        _create_dataset(project_id, path, record_count=n_rows)
    )
    return dataset_id


def _read_events_for_project(project_id: int) -> list[RunEvent]:
    async def _go() -> list[RunEvent]:
        async with async_session_factory() as session:
            result = await session.execute(
                select(RunEvent)
                .where(RunEvent.project_id == project_id)
                .order_by(RunEvent.id.asc())
            )
            return list(result.scalars().all())

    return asyncio.run(_go())


class Phase108AnnotationFoundationTests(unittest.TestCase):
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

    def _create_project(self, label: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"{label}-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_job(
        self,
        project_id: int,
        *,
        name: str = "phase108-job",
        label_type: str = "classification",
        label_schema: dict | None = None,
        target_rows: int | None = None,
    ) -> dict:
        resp = self.client.post(
            f"/api/projects/{project_id}/label-jobs/",
            json={
                "name": name,
                "label_type": label_type,
                "label_schema": label_schema
                or {"allowed_labels": ["positive", "negative", "neutral"]},
                "instructions": "Pick the sentiment.",
                "target_rows": target_rows,
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    # ── CRUD ────────────────────────────────────────────────────────

    def test_create_then_list_then_delete(self):
        pid = self._create_project("phase108-crud")
        job = self._create_job(pid, name="crud-job")
        self.assertEqual(job["name"], "crud-job")
        self.assertEqual(job["label_type"], "classification")
        self.assertEqual(job["status"], "active")

        listing = self.client.get(f"/api/projects/{pid}/label-jobs/")
        self.assertEqual(listing.status_code, 200)
        ids = [j["id"] for j in listing.json()["jobs"]]
        self.assertIn(job["id"], ids)

        delete = self.client.delete(
            f"/api/projects/{pid}/label-jobs/{job['id']}"
        )
        self.assertEqual(delete.status_code, 204, delete.text)

        delete_again = self.client.delete(
            f"/api/projects/{pid}/label-jobs/{job['id']}"
        )
        self.assertEqual(delete_again.status_code, 404)

    def test_unknown_label_type_rejected(self):
        pid = self._create_project("phase108-bad-type")
        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/",
            json={
                "name": "bad",
                "label_type": "summarization",
                "label_schema": {},
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("summarization", resp.json()["detail"])

    def test_creation_emits_audit_event(self):
        pid = self._create_project("phase108-audit-create")
        self._create_job(pid, name="audit-job")
        events = _read_events_for_project(pid)
        creates = [
            ev
            for ev in events
            if ev.reason_code == ANNOTATION_JOB_CREATED
            and ev.severity == SEVERITY_INFO
            and ev.stage == STAGE_INGESTION
        ]
        self.assertEqual(len(creates), 1)
        self.assertEqual(creates[0].payload["label_type"], "classification")

    # ── Seeding ─────────────────────────────────────────────────────

    def test_seed_from_dataset_materializes_rows(self):
        pid = self._create_project("phase108-seed")
        job = self._create_job(pid)
        dataset_id = _create_fixture_dataset(pid, n_rows=6)

        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": 4},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["seeded"], 4)

        detail = self.client.get(
            f"/api/projects/{pid}/label-jobs/{job['id']}"
        )
        self.assertEqual(detail.status_code, 200)
        stats = detail.json()["stats"]
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["labeled"], 0)
        self.assertEqual(stats["assigned"], 0)
        self.assertEqual(stats["unlabeled"], 4)

    def test_seed_from_cross_project_dataset_rejected(self):
        pid_a = self._create_project("phase108-cross-a")
        pid_b = self._create_project("phase108-cross-b")
        dataset_id = _create_fixture_dataset(pid_a, n_rows=3)
        job_b = self._create_job(pid_b)

        resp = self.client.post(
            f"/api/projects/{pid_b}/label-jobs/{job_b['id']}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": 3},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("different project", resp.json()["detail"])

    # ── Assignment ──────────────────────────────────────────────────

    def test_next_row_marks_assigned(self):
        pid = self._create_project("phase108-assign")
        job = self._create_job(pid)
        dataset_id = _create_fixture_dataset(pid, n_rows=3)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": 3},
        )

        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 7},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["queue_empty"])
        self.assertEqual(body["row"]["assigned_to"], 7)
        self.assertIsNotNone(body["row"]["assigned_at"])
        self.assertIsNone(body["row"]["labeled_at"])

    def test_two_next_row_calls_return_different_rows(self):
        pid = self._create_project("phase108-no-double-assign")
        job = self._create_job(pid)
        dataset_id = _create_fixture_dataset(pid, n_rows=3)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": 3},
        )

        first = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 1},
        )
        second = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 2},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        first_id = first.json()["row"]["id"]
        second_id = second.json()["row"]["id"]
        self.assertNotEqual(
            first_id,
            second_id,
            "next-row handed the same row to two reviewers",
        )

    def test_queue_empty_returns_null_row(self):
        pid = self._create_project("phase108-empty-queue")
        job = self._create_job(pid)
        # No seeding — queue is empty from the start.
        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 1},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["queue_empty"])
        self.assertIsNone(body["row"])

    # ── Submit + audit ──────────────────────────────────────────────

    def test_submit_label_persists_and_emits_audit(self):
        pid = self._create_project("phase108-submit")
        job = self._create_job(pid)
        dataset_id = _create_fixture_dataset(pid, n_rows=2)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": 2},
        )
        assigned = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 42},
        ).json()
        row_id = assigned["row"]["id"]

        submit = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/rows/{row_id}/submit",
            json={
                "label_payload": {"label": "positive"},
                "reviewer_notes": "clear sentiment",
            },
        )
        self.assertEqual(submit.status_code, 200, submit.text)
        body = submit.json()
        self.assertEqual(body["label_payload"], {"label": "positive"})
        self.assertIsNotNone(body["labeled_at"])
        self.assertEqual(body["reviewer_notes"], "clear sentiment")

        events = _read_events_for_project(pid)
        submits = [
            ev
            for ev in events
            if ev.reason_code == ANNOTATION_LABEL_SUBMITTED
            and ev.severity == SEVERITY_INFO
            and ev.stage == STAGE_INGESTION
        ]
        self.assertEqual(len(submits), 1)
        payload = submits[0].payload
        self.assertEqual(payload["job_id"], job["id"])
        self.assertEqual(payload["row_id"], row_id)
        self.assertEqual(payload["user_id"], 42)
        self.assertEqual(payload["label_type"], "classification")

    def test_submit_label_rejects_empty_payload(self):
        pid = self._create_project("phase108-empty-payload")
        job = self._create_job(pid)
        dataset_id = _create_fixture_dataset(pid, n_rows=1)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": 1},
        )
        assigned = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 1},
        ).json()
        row_id = assigned["row"]["id"]

        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/rows/{row_id}/submit",
            json={"label_payload": {}},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("label_payload", resp.json()["detail"])

    # ── Skip + re-assign ────────────────────────────────────────────

    def test_skip_returns_row_to_queue(self):
        pid = self._create_project("phase108-skip")
        job = self._create_job(pid)
        dataset_id = _create_fixture_dataset(pid, n_rows=1)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": 1},
        )
        assigned = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 1},
        ).json()
        row_id = assigned["row"]["id"]

        skip = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/rows/{row_id}/skip"
        )
        self.assertEqual(skip.status_code, 200, skip.text)
        self.assertIsNone(skip.json()["assigned_to"])
        self.assertIsNone(skip.json()["assigned_at"])

        # Same row is now available again, possibly to a different
        # reviewer.
        reclaim = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 2},
        ).json()
        self.assertEqual(reclaim["row"]["id"], row_id)
        self.assertEqual(reclaim["row"]["assigned_to"], 2)

    def test_skip_after_submit_is_409(self):
        pid = self._create_project("phase108-skip-after-submit")
        job = self._create_job(pid)
        dataset_id = _create_fixture_dataset(pid, n_rows=1)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": 1},
        )
        assigned = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 1},
        ).json()
        row_id = assigned["row"]["id"]
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/rows/{row_id}/submit",
            json={"label_payload": {"label": "positive"}},
        )
        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/rows/{row_id}/skip"
        )
        self.assertEqual(resp.status_code, 409, resp.text)

    # ── Stats reflects state ────────────────────────────────────────

    def test_stats_reports_total_labeled_assigned_unlabeled(self):
        pid = self._create_project("phase108-stats")
        job = self._create_job(pid, target_rows=10)
        dataset_id = _create_fixture_dataset(pid, n_rows=5)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": 5},
        )
        # Assign one (no submit) + assign + submit another → expect
        # total=5, labeled=1, assigned=1, unlabeled=3.
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 1},
        )
        second = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/next-row",
            json={"user_id": 2},
        ).json()
        second_row_id = second["row"]["id"]
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job['id']}/rows/{second_row_id}/submit",
            json={"label_payload": {"label": "negative"}},
        )

        detail = self.client.get(
            f"/api/projects/{pid}/label-jobs/{job['id']}"
        ).json()
        stats = detail["stats"]
        self.assertEqual(stats["total"], 5)
        self.assertEqual(stats["labeled"], 1)
        self.assertEqual(stats["assigned"], 1)
        self.assertEqual(stats["unlabeled"], 3)
        self.assertEqual(stats["target_rows"], 10)


if __name__ == "__main__":
    unittest.main()

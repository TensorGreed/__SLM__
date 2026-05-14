"""Phase G — saved mapping configs + audit log.

Pins:
- Configs CRUD (create, list, delete) + name uniqueness per project.
- ``run_from_config`` persists rows and bumps ``last_run_*``.
- ``run_import`` emits a RunEvent on success (severity=info,
  reason_code=dataset_import_run) and on failure (severity=error,
  reason_code=dataset_import_failed).
- API endpoints behave as the wizard expects.
"""

from __future__ import annotations

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
from app.models.dataset_import_config import DatasetImportConfig  # noqa: E402
from app.models.run_event import (  # noqa: E402
    SEVERITY_ERROR,
    SEVERITY_INFO,
    STAGE_INGESTION,
    RunEvent,
)
from app.models.reason_codes import (  # noqa: E402
    DATASET_IMPORT_FAILED,
    DATASET_IMPORT_RUN,
)


TEST_DATA_DIR = Path(tempfile.gettempdir()) / f"brewslm-phase106-{uuid.uuid4().hex[:8]}"


def _write_jsonl(rows: list[dict]) -> str:
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    try:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    finally:
        fh.close()
    return fh.name


class Phase106DatasetImportConfigsTests(unittest.TestCase):
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

    def _make_high_conf_fixture(self) -> str:
        return _write_jsonl(
            [
                {
                    "text": "outstanding customer service exceeded my expectations",
                    "label": "positive",
                },
                {
                    "text": "absolute disaster, refund was denied flat out",
                    "label": "negative",
                },
                {
                    "text": "perfectly average, nothing remarkable to say",
                    "label": "neutral",
                },
            ]
        )

    # ── CRUD ────────────────────────────────────────────────────────

    def test_create_then_list_then_delete(self):
        pid = self._create_project("phase106-crud")
        body = {
            "name": "weekly-pii-refresh",
            "locator": "jsonl:/tmp/never-used.jsonl",
            "mapper_id": "label_to_classification",
            "field_map": {"text_field": "text", "label_field": "label"},
            "drop_reasons": ["missing_text"],
            "description": "weekly refresh of the PII dataset",
        }
        create = self.client.post(
            f"/api/projects/{pid}/dataset-import/configs", json=body
        )
        self.assertEqual(create.status_code, 201, create.text)
        cfg = create.json()
        self.assertEqual(cfg["name"], "weekly-pii-refresh")
        self.assertEqual(cfg["drop_reasons"], ["missing_text"])
        self.assertIsNone(cfg["last_run_at"])

        # Listed back.
        listing = self.client.get(f"/api/projects/{pid}/dataset-import/configs")
        self.assertEqual(listing.status_code, 200)
        ids = [c["id"] for c in listing.json()["configs"]]
        self.assertIn(cfg["id"], ids)

        # Delete.
        delete = self.client.delete(
            f"/api/projects/{pid}/dataset-import/configs/{cfg['id']}"
        )
        self.assertEqual(delete.status_code, 204, delete.text)
        # 404 on next delete.
        delete_again = self.client.delete(
            f"/api/projects/{pid}/dataset-import/configs/{cfg['id']}"
        )
        self.assertEqual(delete_again.status_code, 404)

    def test_duplicate_name_returns_409(self):
        pid = self._create_project("phase106-dup")
        body = {
            "name": "first",
            "locator": "jsonl:/tmp/x.jsonl",
            "mapper_id": "label_to_classification",
        }
        first = self.client.post(
            f"/api/projects/{pid}/dataset-import/configs", json=body
        )
        self.assertEqual(first.status_code, 201)
        second = self.client.post(
            f"/api/projects/{pid}/dataset-import/configs", json=body
        )
        self.assertEqual(second.status_code, 409, second.text)
        self.assertIn("already exists", second.json()["detail"])

    def test_blank_name_rejected(self):
        pid = self._create_project("phase106-blank")
        # FastAPI's pydantic min_length=1 catches this at validation time.
        resp = self.client.post(
            f"/api/projects/{pid}/dataset-import/configs",
            json={
                "name": "",
                "locator": "jsonl:/tmp/x.jsonl",
                "mapper_id": "label_to_classification",
            },
        )
        self.assertEqual(resp.status_code, 422)

    # ── run_from_config + audit ─────────────────────────────────────

    def test_run_from_saved_config_writes_rows_and_bumps_audit(self):
        pid = self._create_project("phase106-run")
        path = self._make_high_conf_fixture()
        try:
            create = self.client.post(
                f"/api/projects/{pid}/dataset-import/configs",
                json={
                    "name": "happy-path",
                    "locator": f"jsonl:{path}",
                    "mapper_id": "label_to_classification",
                    "field_map": {
                        "text_field": "text",
                        "label_field": "label",
                    },
                },
            )
            self.assertEqual(create.status_code, 201, create.text)
            cfg_id = create.json()["id"]

            run = self.client.post(
                f"/api/projects/{pid}/dataset-import/configs/{cfg_id}/run"
            )
            self.assertEqual(run.status_code, 200, run.text)
            result = run.json()
            self.assertEqual(result["accepted_count"], 3)
            self.assertFalse(result["dry_run"])

            # last_run_at + last_run_accepted populated.
            listing = self.client.get(
                f"/api/projects/{pid}/dataset-import/configs"
            )
            cfg = next(c for c in listing.json()["configs"] if c["id"] == cfg_id)
            self.assertIsNotNone(cfg["last_run_at"])
            self.assertEqual(cfg["last_run_accepted"], 3)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_run_emits_success_audit_run_event(self):
        pid = self._create_project("phase106-audit-ok")
        path = self._make_high_conf_fixture()
        try:
            # Bypass the saved-config path and go through /run directly
            # to confirm the audit hook fires for every run, not just
            # re-runs from saved mappings.
            resp = self.client.post(
                f"/api/projects/{pid}/dataset-import/run",
                json={
                    "locator": f"jsonl:{path}",
                    "mapper_id": "label_to_classification",
                    "field_map": {
                        "text_field": "text",
                        "label_field": "label",
                    },
                },
            )
            self.assertEqual(resp.status_code, 200, resp.text)

            # Verify an info RunEvent with the right reason_code is in
            # the project's audit log.
            events = _read_events_for_project(pid)
            success_events = [
                ev
                for ev in events
                if ev.reason_code == DATASET_IMPORT_RUN
                and ev.severity == SEVERITY_INFO
                and ev.stage == STAGE_INGESTION
            ]
            self.assertEqual(len(success_events), 1)
            payload = success_events[0].payload
            self.assertEqual(payload["mapper_id"], "label_to_classification")
            self.assertEqual(payload["accepted_count"], 3)
            self.assertEqual(payload["source_id"], "jsonl")
            self.assertIn("synthetic.jsonl", payload["written_path"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_run_emits_failure_audit_event_on_unknown_mapper(self):
        pid = self._create_project("phase106-audit-fail")
        path = self._make_high_conf_fixture()
        try:
            resp = self.client.post(
                f"/api/projects/{pid}/dataset-import/run",
                json={
                    "locator": f"jsonl:{path}",
                    "mapper_id": "does_not_exist",
                    "field_map": {},
                },
            )
            self.assertEqual(resp.status_code, 400, resp.text)

            events = _read_events_for_project(pid)
            failures = [
                ev
                for ev in events
                if ev.reason_code == DATASET_IMPORT_FAILED
                and ev.severity == SEVERITY_ERROR
                and ev.stage == STAGE_INGESTION
            ]
            self.assertGreaterEqual(len(failures), 1)
            self.assertIn("does_not_exist", failures[-1].payload["error"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_run_from_config_payload_links_config_id(self):
        pid = self._create_project("phase106-link")
        path = self._make_high_conf_fixture()
        try:
            create = self.client.post(
                f"/api/projects/{pid}/dataset-import/configs",
                json={
                    "name": "trace-me",
                    "locator": f"jsonl:{path}",
                    "mapper_id": "label_to_classification",
                    "field_map": {
                        "text_field": "text",
                        "label_field": "label",
                    },
                },
            )
            cfg_id = create.json()["id"]
            self.client.post(
                f"/api/projects/{pid}/dataset-import/configs/{cfg_id}/run"
            )
            events = _read_events_for_project(pid)
            success_events = [
                ev
                for ev in events
                if ev.reason_code == DATASET_IMPORT_RUN
            ]
            self.assertTrue(success_events)
            self.assertEqual(success_events[-1].payload["config_id"], cfg_id)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_run_saved_config_404_when_missing(self):
        pid = self._create_project("phase106-missing")
        resp = self.client.post(
            f"/api/projects/{pid}/dataset-import/configs/999999/run"
        )
        self.assertEqual(resp.status_code, 404)


def _read_events_for_project(project_id: int) -> list[RunEvent]:
    """Run the async query synchronously — the test client owns the
    async loop for the FastAPI app, but for read-only inspection we
    spin up a fresh session via the existing factory."""

    import asyncio

    async def _go() -> list[RunEvent]:
        async with async_session_factory() as session:
            result = await session.execute(
                select(RunEvent)
                .where(RunEvent.project_id == project_id)
                .order_by(RunEvent.id.asc())
            )
            return list(result.scalars().all())

    return asyncio.run(_go())


if __name__ == "__main__":
    unittest.main()

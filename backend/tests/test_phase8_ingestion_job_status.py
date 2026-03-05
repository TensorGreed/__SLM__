"""Phase 8 tests: ingestion import job status and queue metadata."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["DEBUG"] = "false"

from app.config import settings
from app.services.ingestion_service import get_import_job_status, queue_remote_import


class IngestionImportJobStatusTests(unittest.TestCase):
    def test_import_job_status_running_when_report_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_data_dir = settings.DATA_DIR
            try:
                settings.DATA_DIR = Path(tmp)
                status = get_import_job_status(31, "missing_report.json")
                self.assertEqual(status["status"], "running")
            finally:
                settings.DATA_DIR = previous_data_dir

    def test_import_job_status_completed_when_report_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_data_dir = settings.DATA_DIR
            try:
                settings.DATA_DIR = Path(tmp)
                report_path = (
                    Path(tmp) / "projects" / "42" / "raw" / "import_jobs" / "remote_import_1.json"
                )
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "project_id": 42,
                            "source_type": "huggingface",
                            "identifier": "squad",
                            "started_at": "2026-03-04T12:00:00Z",
                            "finished_at": "2026-03-04T12:00:03Z",
                            "result": {"samples_ingested": 123},
                        }
                    ),
                    encoding="utf-8",
                )
                status = get_import_job_status(42, str(report_path))
                self.assertEqual(status["status"], "completed")
                self.assertEqual(status["result"]["samples_ingested"], 123)
            finally:
                settings.DATA_DIR = previous_data_dir

    def test_import_job_status_rejects_paths_outside_project_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_data_dir = settings.DATA_DIR
            try:
                settings.DATA_DIR = Path(tmp)
                outside = Path(tmp) / "outside_import_report.json"
                outside.write_text("{}", encoding="utf-8")
                with self.assertRaises(ValueError):
                    get_import_job_status(99, str(outside))
            finally:
                settings.DATA_DIR = previous_data_dir


class IngestionImportQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_remote_import_returns_queued_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_data_dir = settings.DATA_DIR
            try:
                settings.DATA_DIR = Path(tmp)

                class DummyTask:
                    id = "task-123"

                with patch("app.worker.celery_app.send_task", return_value=DummyTask()) as send_task:
                    queued = await queue_remote_import(
                        project_id=7,
                        source_type="huggingface",
                        identifier="squad",
                        split="train",
                        max_samples=100,
                    )

                self.assertEqual(queued["status"], "queued")
                self.assertEqual(queued["task_id"], "task-123")
                self.assertTrue(str(queued["report_path"]).endswith(".json"))
                send_task.assert_called_once()
            finally:
                settings.DATA_DIR = previous_data_dir


if __name__ == "__main__":
    unittest.main()

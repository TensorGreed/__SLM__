"""Phase 13 tests: workflow DAG runner persistence, retries, and dependencies."""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database as database_module
import app.main as main_module
from app.config import settings


class WorkflowRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_settings = {
            "AUTH_ENABLED": settings.AUTH_ENABLED,
            "DB_AUTO_CREATE": settings.DB_AUTO_CREATE,
            "DB_REQUIRE_ALEMBIC_HEAD": settings.DB_REQUIRE_ALEMBIC_HEAD,
            "DATABASE_URL": settings.DATABASE_URL,
        }
        cls._original_engine = database_module.engine
        cls._original_session_factory = database_module.async_session_factory
        cls._original_main_session_factory = main_module.async_session_factory

        cls._tmp_db_path = Path(tempfile.gettempdir()) / f"phase13_runner_{uuid4().hex}.db"
        settings.DATABASE_URL = f"sqlite+aiosqlite:///{cls._tmp_db_path.as_posix()}"
        settings.AUTH_ENABLED = False
        settings.DB_AUTO_CREATE = True
        settings.DB_REQUIRE_ALEMBIC_HEAD = False

        database_module.engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
            future=True,
        )
        database_module.async_session_factory = async_sessionmaker(
            database_module.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        main_module.async_session_factory = database_module.async_session_factory

        cls._client_cm = TestClient(main_module.app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        asyncio.run(database_module.engine.dispose())
        database_module.engine = cls._original_engine
        database_module.async_session_factory = cls._original_session_factory
        main_module.async_session_factory = cls._original_main_session_factory
        for key, value in cls._original_settings.items():
            setattr(settings, key, value)
        if cls._tmp_db_path.exists():
            cls._tmp_db_path.unlink()

    def _create_project(self, prefix: str) -> int:
        create = self.client.post(
            "/api/projects",
            json={
                "name": f"{prefix}-{uuid4().hex[:8]}",
                "description": "phase13 workflow runner test",
                "base_model_name": "microsoft/phi-2",
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        return int(create.json()["id"])

    def _publish_source_artifacts(self, project_id: int) -> None:
        publish = self.client.post(
            f"/api/projects/{project_id}/artifacts/publish-batch",
            json={
                "artifacts": [
                    {"artifact_key": "source.file", "producer_stage": "ingestion"},
                    {"artifact_key": "source.remote_dataset", "producer_stage": "ingestion"},
                ]
            },
        )
        self.assertEqual(publish.status_code, 201, publish.text)
        self.assertEqual(publish.json()["count"], 2)

    def test_workflow_run_blocks_when_inputs_missing(self):
        project_id = self._create_project("phase13-block")
        templates = self.client.get(f"/api/projects/{project_id}/pipeline/graph/templates")
        self.assertEqual(templates.status_code, 200, templates.text)
        template_ids = [item["template_id"] for item in templates.json()["templates"]]
        self.assertIn("template.sft", template_ids)
        self.assertIn("template.lora", template_ids)
        self.assertIn("template.distill", template_ids)
        self.assertIn("template.eval_only", template_ids)

        run = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run",
            json={},
        )
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "blocked")
        first = payload["nodes"][0]
        self.assertEqual(first["stage"], "ingestion")
        self.assertEqual(first["status"], "blocked")
        self.assertIn("source.file", first["missing_inputs"])

    def test_workflow_run_completes_and_publishes_outputs(self):
        project_id = self._create_project("phase13-complete")
        self._publish_source_artifacts(project_id)

        run = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run",
            json={"execution_backend": "local"},
        )
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(len(payload["nodes"]), 10)
        self.assertTrue(all(node["status"] == "completed" for node in payload["nodes"]))

        latest_model = self.client.get(f"/api/projects/{project_id}/artifacts/latest/model.checkpoint")
        self.assertEqual(latest_model.status_code, 200, latest_model.text)
        self.assertEqual(latest_model.json()["version"], 1)

    def test_workflow_run_retries_and_succeeds(self):
        project_id = self._create_project("phase13-retry")
        self._publish_source_artifacts(project_id)

        run = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run",
            json={
                "execution_backend": "local",
                "max_retries": 1,
                "config": {"simulate_fail_once_stages": ["ingestion"]},
            },
        )
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        first = payload["nodes"][0]
        self.assertEqual(first["stage"], "ingestion")
        self.assertEqual(first["attempt_count"], 2)
        self.assertEqual(first["status"], "completed")

        run_id = payload["id"]
        detail = self.client.get(f"/api/projects/{project_id}/pipeline/graph/workflow-runs/{run_id}")
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["id"], run_id)
        self.assertEqual(detail.json()["status"], "completed")

    def test_workflow_run_async_queue_and_poll(self):
        project_id = self._create_project("phase13-async")
        self._publish_source_artifacts(project_id)

        queued = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run-async",
            json={"execution_backend": "local"},
        )
        self.assertEqual(queued.status_code, 200, queued.text)
        payload = queued.json()
        self.assertTrue(payload["queued"])
        run_id = payload["run_id"]

        terminal = None
        for _ in range(300):
            detail = self.client.get(f"/api/projects/{project_id}/pipeline/graph/workflow-runs/{run_id}")
            self.assertEqual(detail.status_code, 200, detail.text)
            terminal = detail.json()
            if terminal["status"] in {"completed", "failed", "blocked", "cancelled"}:
                break
            time.sleep(0.2)

        self.assertIsNotNone(terminal)
        self.assertEqual(terminal["status"], "completed")


if __name__ == "__main__":
    unittest.main()

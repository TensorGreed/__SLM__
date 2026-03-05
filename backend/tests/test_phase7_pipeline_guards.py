"""Phase 7 tests: pipeline advancement guardrails."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database as database_module
import app.main as main_module
from app.config import settings


class PipelineGuardTests(unittest.TestCase):
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

        cls._tmp_db_path = Path(tempfile.gettempdir()) / f"phase7_pipeline_{uuid4().hex}.db"
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

    def test_pipeline_advance_blocked_without_ingestion_artifacts(self):
        create = self.client.post(
            "/api/projects",
            json={
                "name": f"phase7-{uuid4().hex[:8]}",
                "description": "pipeline guard test",
                "base_model_name": "microsoft/phi-2",
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = create.json()["id"]

        advance = self.client.post(f"/api/projects/{project_id}/pipeline/advance")
        self.assertEqual(advance.status_code, 400, advance.text)
        self.assertIn("ingest and process at least one raw document", advance.json()["detail"])

    def test_pipeline_graph_returns_readonly_linear_contract(self):
        create = self.client.post(
            "/api/projects",
            json={
                "name": f"phase7-graph-{uuid4().hex[:8]}",
                "description": "pipeline graph test",
                "base_model_name": "microsoft/phi-2",
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = create.json()["id"]

        response = self.client.get(f"/api/projects/{project_id}/pipeline/graph")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()

        self.assertEqual(payload["project_id"], project_id)
        self.assertEqual(payload["mode"], "readonly_preview")
        self.assertEqual(payload["current_stage"], "ingestion")
        self.assertEqual(payload["graph_id"], "default-linear-v1")
        self.assertEqual(len(payload["nodes"]), 10)
        self.assertEqual(len(payload["edges"]), 9)

        first = payload["nodes"][0]
        self.assertEqual(first["stage"], "ingestion")
        self.assertEqual(first["status"], "active")
        self.assertTrue(first["input_artifacts"])
        self.assertTrue(first["output_artifacts"])

    def test_pipeline_graph_validate_fallback_on_invalid_override(self):
        create = self.client.post(
            "/api/projects",
            json={
                "name": f"phase7-graph-validate-{uuid4().hex[:8]}",
                "description": "pipeline graph validate test",
                "base_model_name": "microsoft/phi-2",
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = create.json()["id"]

        response = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/validate",
            json={
                "graph": {
                    "nodes": [{"id": "broken-node"}],
                    "edges": [],
                }
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["fallback_used"])
        self.assertTrue(payload["errors"])
        self.assertEqual(payload["graph"]["graph_id"], "default-linear-v1")

    def test_pipeline_graph_dry_run_reports_missing_ingestion_inputs(self):
        create = self.client.post(
            "/api/projects",
            json={
                "name": f"phase7-graph-dryrun-{uuid4().hex[:8]}",
                "description": "pipeline graph dry-run test",
                "base_model_name": "microsoft/phi-2",
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = create.json()["id"]

        response = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/dry-run",
            json={},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["current_stage"], "ingestion")
        self.assertEqual(payload["active_step"]["stage"], "ingestion")
        self.assertIn("source.file", payload["active_step"]["missing_inputs"])
        self.assertIn("source.remote_dataset", payload["active_step"]["missing_inputs"])

    def test_pipeline_graph_run_step_advances_when_stage_requirements_pass(self):
        create = self.client.post(
            "/api/projects",
            json={
                "name": f"phase7-graph-run-{uuid4().hex[:8]}",
                "description": "pipeline graph run-step test",
                "base_model_name": "microsoft/phi-2",
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = create.json()["id"]

        upload = self.client.post(
            f"/api/projects/{project_id}/ingestion/upload",
            files={"file": ("sample.txt", b"hello pipeline runtime", "text/plain")},
            data={"source": "upload", "sensitivity": "internal", "license_info": ""},
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        document_id = upload.json()["id"]

        processed = self.client.post(f"/api/projects/{project_id}/ingestion/documents/{document_id}/process")
        self.assertEqual(processed.status_code, 200, processed.text)

        run = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run-step",
            json={},
        )
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        self.assertTrue(payload["advanced"])
        self.assertEqual(payload["previous_stage"], "ingestion")
        self.assertEqual(payload["current_stage"], "cleaning")

    def test_pipeline_graph_run_step_blocks_when_requirements_missing(self):
        create = self.client.post(
            "/api/projects",
            json={
                "name": f"phase7-graph-block-{uuid4().hex[:8]}",
                "description": "pipeline graph blocked run-step test",
                "base_model_name": "microsoft/phi-2",
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = create.json()["id"]

        run = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run-step",
            json={},
        )
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "blocked")
        self.assertFalse(payload["advanced"])
        self.assertIn("source.file", payload["missing_inputs"])


if __name__ == "__main__":
    unittest.main()

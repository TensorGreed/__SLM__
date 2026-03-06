"""Phase 12 tests: typed artifact registry and workflow runtime wiring."""

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


class ArtifactRegistryApiTests(unittest.TestCase):
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

        cls._tmp_db_path = Path(tempfile.gettempdir()) / f"phase12_artifacts_{uuid4().hex}.db"
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
                "description": "phase12 artifact registry test",
                "base_model_name": "microsoft/phi-2",
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        return int(create.json()["id"])

    def test_artifact_publish_versions_and_latest(self):
        project_id = self._create_project("phase12-publish")

        v1 = self.client.post(
            f"/api/projects/{project_id}/artifacts/publish",
            json={"artifact_key": "dataset.cleaned", "producer_stage": "cleaning"},
        )
        self.assertEqual(v1.status_code, 201, v1.text)
        self.assertEqual(v1.json()["version"], 1)

        v2 = self.client.post(
            f"/api/projects/{project_id}/artifacts/publish",
            json={"artifact_key": "dataset.cleaned", "producer_stage": "cleaning"},
        )
        self.assertEqual(v2.status_code, 201, v2.text)
        self.assertEqual(v2.json()["version"], 2)

        latest = self.client.get(f"/api/projects/{project_id}/artifacts/latest/dataset.cleaned")
        self.assertEqual(latest.status_code, 200, latest.text)
        self.assertEqual(latest.json()["version"], 2)

        listed = self.client.get(f"/api/projects/{project_id}/artifacts", params={"artifact_key": "dataset.cleaned"})
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(listed.json()["count"], 2)

        keys = self.client.get(f"/api/projects/{project_id}/artifacts/keys")
        self.assertEqual(keys.status_code, 200, keys.text)
        self.assertIn("dataset.cleaned", keys.json()["keys"])

    def test_pipeline_run_step_publishes_declared_outputs(self):
        project_id = self._create_project("phase12-runstep")

        upload = self.client.post(
            f"/api/projects/{project_id}/ingestion/upload",
            files={"file": ("sample.txt", b"phase12 ingestion text", "text/plain")},
            data={"source": "upload", "sensitivity": "internal", "license_info": ""},
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        document_id = upload.json()["id"]

        processed = self.client.post(f"/api/projects/{project_id}/ingestion/documents/{document_id}/process")
        self.assertEqual(processed.status_code, 200, processed.text)

        run = self.client.post(f"/api/projects/{project_id}/pipeline/graph/run-step", json={})
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        self.assertIn("dataset.raw", payload["published_artifact_keys"])

        latest_raw = self.client.get(f"/api/projects/{project_id}/artifacts/latest/dataset.raw")
        self.assertEqual(latest_raw.status_code, 200, latest_raw.text)
        self.assertEqual(latest_raw.json()["version"], 1)
        self.assertEqual(latest_raw.json()["producer_stage"], "ingestion")
        self.assertEqual(latest_raw.json()["producer_run_id"], payload["run_id"])

    def test_dry_run_reads_artifacts_from_registry(self):
        project_id = self._create_project("phase12-dryrun")

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

        dry_run = self.client.post(f"/api/projects/{project_id}/pipeline/graph/dry-run", json={})
        self.assertEqual(dry_run.status_code, 200, dry_run.text)
        payload = dry_run.json()
        self.assertEqual(payload["active_step"]["stage"], "ingestion")
        self.assertTrue(payload["active_step"]["can_run_now"])
        self.assertEqual(payload["active_step"]["missing_inputs"], [])
        self.assertIn("source.file", payload["available_artifacts"])
        self.assertIn("source.remote_dataset", payload["available_artifacts"])

    def test_runtime_requirements_block_run_step_when_missing(self):
        project_id = self._create_project("phase12-runtime")

        upload = self.client.post(
            f"/api/projects/{project_id}/ingestion/upload",
            files={"file": ("sample.txt", b"runtime check", "text/plain")},
            data={"source": "upload", "sensitivity": "internal", "license_info": ""},
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        document_id = upload.json()["id"]

        processed = self.client.post(f"/api/projects/{project_id}/ingestion/documents/{document_id}/process")
        self.assertEqual(processed.status_code, 200, processed.text)

        graph_resp = self.client.get(f"/api/projects/{project_id}/pipeline/graph")
        self.assertEqual(graph_resp.status_code, 200, graph_resp.text)
        graph = graph_resp.json()
        graph["nodes"][0]["runtime_requirements"]["required_env"] = ["PHASE12_REQUIRED_ENV_NOT_SET"]

        run = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run-step",
            json={
                "graph": graph,
                "allow_fallback": False,
                "use_saved_override": False,
            },
        )
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "blocked")
        self.assertFalse(payload["advanced"])
        self.assertIn("PHASE12_REQUIRED_ENV_NOT_SET", payload["missing_runtime_requirements"])


if __name__ == "__main__":
    unittest.main()

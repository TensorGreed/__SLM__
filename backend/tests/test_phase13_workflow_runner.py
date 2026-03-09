"""Phase 13 tests: workflow DAG runner persistence, retries, and dependencies."""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database as database_module
import app.main as main_module
from app.config import settings
import app.services.workflow_runner_service as workflow_runner_module


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

    def test_stage_catalog_includes_data_adapter_preview(self):
        project_id = self._create_project("phase13-stage-catalog")
        catalog = self.client.get(f"/api/projects/{project_id}/pipeline/graph/stage-catalog")
        self.assertEqual(catalog.status_code, 200, catalog.text)
        stages = [item["stage"] for item in catalog.json()["stages"]]
        self.assertIn("data_adapter_preview", stages)

    def test_workflow_run_executes_local_data_adapter_preview_node(self):
        project_id = self._create_project("phase13-adapter-node")
        graph_override = {
            "graph_id": "custom-adapter-preview",
            "graph_label": "Adapter Preview Graph",
            "graph_version": "1.0.0",
            "nodes": [
                {
                    "id": "step:data_adapter_preview",
                    "stage": "data_adapter_preview",
                    "display_name": "Data Adapter Preview",
                    "index": 0,
                    "kind": "custom_step",
                    "status": "pending",
                    "step_type": "core.data_adapter_preview",
                    "description": "Validate adapter coverage before splitting.",
                    "input_artifacts": [],
                    "output_artifacts": ["analysis.data_adapter"],
                    "config_schema_ref": "slm.step.data_adapter_preview/v1",
                    "runtime_requirements": {
                        "execution_modes": ["local"],
                        "required_services": [],
                        "required_env": [],
                        "required_settings": [],
                        "requires_gpu": False,
                        "min_vram_gb": 0.0,
                    },
                    "position": {"x": 40, "y": 40},
                    "config": {
                        "dataset_type": "raw",
                        "adapter_id": "auto",
                        "sample_size": 100,
                        "min_mapping_ratio": 0.6,
                    },
                }
            ],
            "edges": [],
        }

        preview_payload = {
            "project_id": project_id,
            "requested_adapter_id": "auto",
            "resolved_adapter_id": "qa-pair",
            "detection_scores": {"qa-pair": 1.0},
            "sampled_records": 100,
            "mapped_records": 95,
            "dropped_records": 5,
            "error_count": 0,
            "errors": [],
            "validation_report": {"status": "ok"},
            "preview_rows": [],
            "source": {"dataset_type": "raw"},
        }

        with patch(
            "app.services.dataset_service.preview_project_data_adapter",
            AsyncMock(return_value=preview_payload),
        ) as mocked_preview:
            run = self.client.post(
                f"/api/projects/{project_id}/pipeline/graph/run",
                json={
                    "execution_backend": "local",
                    "allow_fallback": False,
                    "graph": graph_override,
                },
            )

        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(len(payload["nodes"]), 1)
        node = payload["nodes"][0]
        self.assertEqual(node["stage"], "data_adapter_preview")
        self.assertEqual(node["status"], "completed")
        self.assertEqual(node["published_artifact_keys"], ["analysis.data_adapter"])
        self.assertGreaterEqual(len(node["node_log"]), 1)
        self.assertIn("preview_summary", node["node_log"][-1])
        mocked_preview.assert_awaited()

    def test_workflow_run_training_node_defaults_to_noop(self):
        project_id = self._create_project("phase13-training-noop")
        graph_override = {
            "graph_id": "custom-training-noop",
            "graph_label": "Training Node Noop",
            "graph_version": "1.0.0",
            "nodes": [
                {
                    "id": "step:training",
                    "stage": "training",
                    "display_name": "Training",
                    "index": 0,
                    "kind": "custom_step",
                    "status": "pending",
                    "step_type": "core.training",
                    "description": "Training node with default noop execution mode.",
                    "input_artifacts": [],
                    "output_artifacts": ["model.checkpoint"],
                    "config_schema_ref": "slm.step.training/v1",
                    "runtime_requirements": {
                        "execution_modes": ["local"],
                        "required_services": [],
                        "required_env": [],
                        "required_settings": [],
                        "requires_gpu": False,
                        "min_vram_gb": 0.0,
                    },
                    "position": {"x": 40, "y": 40},
                }
            ],
            "edges": [],
        }

        run = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run",
            json={
                "execution_backend": "local",
                "allow_fallback": False,
                "graph": graph_override,
            },
        )
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(len(payload["nodes"]), 1)
        node = payload["nodes"][0]
        self.assertEqual(node["stage"], "training")
        self.assertEqual(node["status"], "completed")
        self.assertGreaterEqual(len(node["node_log"]), 1)
        training_log = node["node_log"][-1].get("training", {})
        self.assertEqual(training_log.get("mode"), "noop")

    def test_workflow_run_training_node_create_and_start(self):
        project_id = self._create_project("phase13-training-start")
        graph_override = {
            "graph_id": "custom-training-start",
            "graph_label": "Training Node Execute",
            "graph_version": "1.0.0",
            "nodes": [
                {
                    "id": "step:training",
                    "stage": "training",
                    "display_name": "Training",
                    "index": 0,
                    "kind": "custom_step",
                    "status": "pending",
                    "step_type": "core.training",
                    "description": "Training node with create_and_start mode.",
                    "input_artifacts": [],
                    "output_artifacts": ["model.checkpoint"],
                    "config_schema_ref": "slm.step.training/v1",
                    "runtime_requirements": {
                        "execution_modes": ["local"],
                        "required_services": [],
                        "required_env": [],
                        "required_settings": [],
                        "requires_gpu": False,
                        "min_vram_gb": 0.0,
                    },
                    "position": {"x": 40, "y": 40},
                    "config": {
                        "mode": "create_and_start",
                        "name": "wf-train",
                        "base_model": "microsoft/phi-2",
                        "config": {
                            "num_epochs": 1,
                            "batch_size": 1,
                        },
                        "wait_for_terminal": False,
                    },
                }
            ],
            "edges": [],
        }

        with patch(
            "app.services.training_service.create_experiment",
            AsyncMock(return_value=SimpleNamespace(id=777)),
        ) as mocked_create, patch(
            "app.services.training_service.start_training",
            AsyncMock(
                return_value={
                    "experiment_id": 777,
                    "status": "running",
                    "backend": "external",
                    "task_id": "task-777",
                }
            ),
        ) as mocked_start:
            run = self.client.post(
                f"/api/projects/{project_id}/pipeline/graph/run",
                json={
                    "execution_backend": "local",
                    "allow_fallback": False,
                    "graph": graph_override,
                },
            )

        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        node = payload["nodes"][0]
        self.assertEqual(node["status"], "completed")
        training_log = node["node_log"][-1].get("training", {})
        self.assertEqual(training_log.get("mode"), "create_and_start")
        self.assertEqual(training_log.get("experiment_id"), 777)
        mocked_create.assert_awaited()
        mocked_start.assert_awaited()

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

    def test_workflow_run_celery_backend_dispatches_node_attempts(self):
        project_id = self._create_project("phase13-celery")
        self._publish_source_artifacts(project_id)

        original_execute = workflow_runner_module._execute_celery_node_attempt
        calls: list[str] = []

        def _fake_execute_celery(**kwargs):
            stage = str(kwargs.get("stage", ""))
            calls.append(stage)
            return True, {"message": f"fake celery completed {stage}"}, ""

        workflow_runner_module._execute_celery_node_attempt = _fake_execute_celery
        try:
            run = self.client.post(
                f"/api/projects/{project_id}/pipeline/graph/run",
                json={"execution_backend": "celery"},
            )
        finally:
            workflow_runner_module._execute_celery_node_attempt = original_execute

        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        self.assertGreaterEqual(len(calls), 1)
        self.assertTrue(all(node["execution_backend"] == "celery" for node in payload["nodes"]))

    def test_workflow_run_branching_conditions_skip_inactive_path(self):
        project_id = self._create_project("phase13-branching")
        graph_override = {
            "graph_id": "custom-branching",
            "graph_label": "Conditional Branching Graph",
            "graph_version": "2.0.0",
            "nodes": [
                {
                    "id": "step:start",
                    "stage": "ingestion",
                    "display_name": "Start",
                    "index": 0,
                    "kind": "custom_step",
                    "status": "pending",
                    "step_type": "core.ingestion",
                    "description": "Branch source node.",
                    "input_artifacts": [],
                    "output_artifacts": ["dataset.raw"],
                    "config_schema_ref": "slm.step.ingestion/v2",
                    "runtime_requirements": {
                        "execution_modes": ["local"],
                        "required_services": [],
                        "required_env": [],
                        "required_settings": [],
                        "requires_gpu": False,
                        "min_vram_gb": 0.0,
                    },
                    "position": {"x": 0, "y": 0},
                },
                {
                    "id": "step:branch_ok",
                    "stage": "cleaning",
                    "display_name": "Branch Success",
                    "index": 1,
                    "kind": "custom_step",
                    "status": "pending",
                    "step_type": "core.cleaning",
                    "description": "Runs when upstream is completed.",
                    "input_artifacts": [],
                    "output_artifacts": ["dataset.cleaned"],
                    "config_schema_ref": "slm.step.cleaning/v2",
                    "runtime_requirements": {
                        "execution_modes": ["local"],
                        "required_services": [],
                        "required_env": [],
                        "required_settings": [],
                        "requires_gpu": False,
                        "min_vram_gb": 0.0,
                    },
                    "position": {"x": 280, "y": 0},
                },
                {
                    "id": "step:branch_fail",
                    "stage": "synthetic",
                    "display_name": "Branch Failure",
                    "index": 2,
                    "kind": "custom_step",
                    "status": "pending",
                    "step_type": "core.synthetic",
                    "description": "Runs only when upstream fails.",
                    "input_artifacts": [],
                    "output_artifacts": ["dataset.synthetic"],
                    "config_schema_ref": "slm.step.synthetic/v2",
                    "runtime_requirements": {
                        "execution_modes": ["local"],
                        "required_services": [],
                        "required_env": [],
                        "required_settings": [],
                        "requires_gpu": False,
                        "min_vram_gb": 0.0,
                    },
                    "position": {"x": 560, "y": 0},
                },
            ],
            "edges": [
                {
                    "id": "edge:start->ok",
                    "source": "step:start",
                    "target": "step:branch_ok",
                    "kind": "branch",
                    "condition": {"source_status_in": ["completed"]},
                },
                {
                    "id": "edge:start->fail",
                    "source": "step:start",
                    "target": "step:branch_fail",
                    "kind": "branch",
                    "condition": {"source_status_in": ["failed"]},
                },
            ],
        }

        run = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run",
            json={
                "execution_backend": "local",
                "allow_fallback": False,
                "graph": graph_override,
            },
        )
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        by_node = {row["node_id"]: row for row in payload["nodes"]}
        self.assertEqual(by_node["step:start"]["status"], "completed")
        self.assertEqual(by_node["step:branch_ok"]["status"], "completed")
        self.assertEqual(by_node["step:branch_fail"]["status"], "skipped")
        self.assertIn("branch condition", str(by_node["step:branch_fail"]["error_message"]))

    def test_workflow_retry_policy_honors_failure_types(self):
        project_id = self._create_project("phase13-retry-policy")
        graph_override = {
            "graph_id": "custom-retry-policy",
            "graph_label": "Retry Policy Graph",
            "graph_version": "2.0.0",
            "nodes": [
                {
                    "id": "step:ingestion",
                    "stage": "ingestion",
                    "display_name": "Ingestion",
                    "index": 0,
                    "kind": "custom_step",
                    "status": "pending",
                    "step_type": "core.ingestion",
                    "description": "Ingestion with failure-type retry policy.",
                    "input_artifacts": [],
                    "output_artifacts": ["dataset.raw"],
                    "config_schema_ref": "slm.step.ingestion/v2",
                    "runtime_requirements": {
                        "execution_modes": ["local"],
                        "required_services": [],
                        "required_env": [],
                        "required_settings": [],
                        "requires_gpu": False,
                        "min_vram_gb": 0.0,
                    },
                    "position": {"x": 20, "y": 20},
                    "retry_policy": {
                        "max_retries": 3,
                        "retry_on": ["transient"],
                    },
                }
            ],
            "edges": [],
        }

        run = self.client.post(
            f"/api/projects/{project_id}/pipeline/graph/run",
            json={
                "execution_backend": "local",
                "allow_fallback": False,
                "graph": graph_override,
                "config": {"simulate_fail_once_stages": ["ingestion"]},
            },
        )
        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "failed")
        node = payload["nodes"][0]
        self.assertEqual(node["status"], "failed")
        self.assertEqual(node["attempt_count"], 1)
        self.assertIn("simulated one-time failure", str(node.get("error_message", "")))

    def test_workflow_sweep_loop_runs_multiple_iterations(self):
        project_id = self._create_project("phase13-sweep-loop")
        graph_override = {
            "graph_id": "custom-sweep-loop",
            "graph_label": "Sweep Loop Graph",
            "graph_version": "2.0.0",
            "nodes": [
                {
                    "id": "step:data_adapter_preview",
                    "stage": "data_adapter_preview",
                    "display_name": "Data Adapter Preview",
                    "index": 0,
                    "kind": "custom_step",
                    "status": "pending",
                    "step_type": "core.data_adapter_preview",
                    "description": "Data adapter preview with sweep loop.",
                    "input_artifacts": [],
                    "output_artifacts": ["analysis.data_adapter"],
                    "config_schema_ref": "slm.step.data_adapter_preview/v2",
                    "runtime_requirements": {
                        "execution_modes": ["local"],
                        "required_services": [],
                        "required_env": [],
                        "required_settings": [],
                        "requires_gpu": False,
                        "min_vram_gb": 0.0,
                    },
                    "position": {"x": 30, "y": 30},
                    "config": {
                        "dataset_type": "raw",
                        "adapter_id": "baseline",
                        "sample_size": 100,
                        "min_mapping_ratio": 0.1,
                    },
                    "loop": {
                        "type": "sweep",
                        "objective_path": "preview_summary.mapping_ratio",
                        "objective_mode": "max",
                        "items": [
                            {
                                "label": "trial-low",
                                "node_config": {"adapter_id": "trial-low"},
                            },
                            {
                                "label": "trial-high",
                                "node_config": {"adapter_id": "trial-high"},
                            },
                        ],
                    },
                }
            ],
            "edges": [],
        }

        def _mock_preview(*, adapter_id: str, **kwargs):
            mapped = 60 if adapter_id == "trial-low" else 95
            return {
                "project_id": project_id,
                "requested_adapter_id": adapter_id,
                "resolved_adapter_id": adapter_id,
                "detection_scores": {adapter_id: 1.0},
                "sampled_records": 100,
                "mapped_records": mapped,
                "dropped_records": 100 - mapped,
                "error_count": 0,
                "errors": [],
                "validation_report": {"status": "ok"},
                "preview_rows": [],
                "source": {"dataset_type": "raw"},
            }

        with patch(
            "app.services.dataset_service.preview_project_data_adapter",
            AsyncMock(side_effect=_mock_preview),
        ):
            run = self.client.post(
                f"/api/projects/{project_id}/pipeline/graph/run",
                json={
                    "execution_backend": "local",
                    "allow_fallback": False,
                    "graph": graph_override,
                },
            )

        self.assertEqual(run.status_code, 200, run.text)
        payload = run.json()
        self.assertEqual(payload["status"], "completed")
        node = payload["nodes"][0]
        self.assertEqual(node["status"], "completed")
        self.assertEqual(node["attempt_count"], 2)
        loop_entry = next((item for item in node.get("node_log", []) if "loop_summary" in item), None)
        self.assertIsNotNone(loop_entry)
        summary = loop_entry.get("loop_summary") or {}
        self.assertEqual(int(summary.get("total_iterations", 0)), 2)
        self.assertEqual(int(summary.get("successful_iterations", 0)), 2)
        selected_entry = next((item for item in node.get("node_log", []) if "selected_iteration_payload" in item), None)
        self.assertIsNotNone(selected_entry)
        selected_payload = selected_entry.get("selected_iteration_payload") or {}
        preview_summary = selected_payload.get("preview_summary") or {}
        self.assertEqual(preview_summary.get("resolved_adapter_id"), "trial-high")

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

"""Phase 16 tests: end-to-end pipeline recipe blueprints."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase16_pipeline_recipes_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase16_pipeline_recipes_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase16PipelineRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "phase16"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_pipeline_recipe_catalog_and_resolve(self):
        project_id = self._create_project("phase16-recipes-1")

        catalog = self.client.get(f"/api/projects/{project_id}/pipeline/recipes")
        self.assertEqual(catalog.status_code, 200, catalog.text)
        catalog_payload = catalog.json()
        self.assertEqual(catalog_payload.get("default_recipe_id"), "recipe.pipeline.sft_default")
        self.assertTrue(bool(catalog_payload.get("recommended_recipe_id")))
        recipe_ids = {str(item.get("recipe_id")) for item in catalog_payload.get("recipes", []) if isinstance(item, dict)}
        self.assertIn("recipe.pipeline.sft_default", recipe_ids)
        self.assertIn("recipe.pipeline.lora_fast", recipe_ids)

        resolved = self.client.post(
            f"/api/projects/{project_id}/pipeline/recipes/resolve",
            json={
                "recipe_id": "recipe.pipeline.sft_default",
                "include_preflight": True,
            },
        )
        self.assertEqual(resolved.status_code, 200, resolved.text)
        payload = resolved.json()
        workflow = payload.get("resolved", {}).get("workflow", {})
        self.assertEqual(workflow.get("template_id"), "template.sft")
        self.assertIn("preflight", payload)

    def test_pipeline_recipe_program_v2_resolves_compiled_graph(self):
        project_id = self._create_project("phase16-recipes-program-v2")

        resolved = self.client.post(
            f"/api/projects/{project_id}/pipeline/recipes/resolve",
            json={
                "recipe_id": "recipe.pipeline.program.sft_sweep",
                "include_preflight": True,
            },
        )
        self.assertEqual(resolved.status_code, 200, resolved.text)
        payload = resolved.json()
        workflow = payload.get("resolved", {}).get("workflow", {})
        self.assertEqual(workflow.get("resolution_mode"), "program_v2")
        graph = workflow.get("graph") or {}
        self.assertEqual(str(graph.get("graph_id") or ""), "program.sft.sweep.v2")
        nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
        node_by_id = {str(item.get("id")): item for item in nodes}
        self.assertIn("step:training", node_by_id)
        self.assertIn("step:evaluation_fast_lane", node_by_id)
        training = node_by_id["step:training"]
        self.assertIsInstance(training.get("retry_policy"), dict)
        self.assertIsInstance(training.get("loop"), dict)
        edges = [item for item in graph.get("edges", []) if isinstance(item, dict)]
        self.assertTrue(
            any(
                str(item.get("source")) == "step:training"
                and str(item.get("target")) == "step:evaluation_fast_lane"
                for item in edges
            )
        )

    def test_pipeline_recipe_recommend_endpoint_respects_context(self):
        project_id = self._create_project("phase16-recipes-1b")

        fast_resp = self.client.get(
            f"/api/projects/{project_id}/pipeline/recipes/recommend",
            params={
                "task_profile": "instruction_sft",
                "preferred_plan_profile": "safe",
                "prefer_fast": "true",
            },
        )
        self.assertEqual(fast_resp.status_code, 200, fast_resp.text)
        fast_payload = fast_resp.json()
        self.assertEqual(fast_payload.get("recommended_recipe_id"), "recipe.pipeline.lora_fast")
        self.assertEqual(fast_payload.get("context", {}).get("task_profile"), "instruction_sft")

        balanced_resp = self.client.get(
            f"/api/projects/{project_id}/pipeline/recipes/recommend",
            params={
                "task_profile": "instruction_sft",
                "preferred_plan_profile": "balanced",
                "prefer_fast": "false",
            },
        )
        self.assertEqual(balanced_resp.status_code, 200, balanced_resp.text)
        balanced_payload = balanced_resp.json()
        self.assertEqual(balanced_payload.get("recommended_recipe_id"), "recipe.pipeline.sft_default")

    def test_pipeline_recipe_apply_updates_project_and_state(self):
        project_id = self._create_project("phase16-recipes-2")

        applied = self.client.post(
            f"/api/projects/{project_id}/pipeline/recipes/apply",
            json={
                "recipe_id": "recipe.pipeline.lora_fast",
                "include_preflight": True,
                "enforce_preflight_ok": False,
                "mark_active": True,
            },
        )
        self.assertEqual(applied.status_code, 200, applied.text)
        body = applied.json()
        self.assertEqual(body.get("project", {}).get("evaluation_preferred_pack_id"), "evalpack.fast.iteration")
        self.assertEqual(body.get("project", {}).get("training_preferred_plan_profile"), "safe")
        self.assertTrue(bool(body.get("manifest_path")))
        self.assertTrue(bool(body.get("artifact", {}).get("id")))

        state = self.client.get(f"/api/projects/{project_id}/pipeline/recipes/state")
        self.assertEqual(state.status_code, 200, state.text)
        state_payload = state.json()
        self.assertTrue(state_payload.get("has_state"))
        self.assertEqual(
            state_payload.get("state", {}).get("active_recipe_id"),
            "recipe.pipeline.lora_fast",
        )

        graph_contract = self.client.get(f"/api/projects/{project_id}/pipeline/graph/contract")
        self.assertEqual(graph_contract.status_code, 200, graph_contract.text)
        graph_payload = graph_contract.json()
        self.assertTrue(bool(graph_payload.get("has_saved_override")))

    def test_pipeline_recipe_run_persists_execution_lineage(self):
        project_id = self._create_project("phase16-recipes-3")

        run_resp = self.client.post(
            f"/api/projects/{project_id}/pipeline/recipes/run",
            json={
                "recipe_id": "recipe.pipeline.sft_default",
                "include_preflight": True,
                "enforce_preflight_ok": False,
                "mark_active": True,
                "execution_backend": "local",
                "max_retries": 0,
                "stop_on_blocked": True,
                "stop_on_failure": True,
                "async_run": False,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        run_body = run_resp.json()
        recipe_run_id = str(run_body.get("recipe_run_id") or "")
        self.assertTrue(recipe_run_id)

        execution = run_body.get("execution", {})
        workflow_run_id = execution.get("workflow_run_id")
        self.assertTrue(bool(workflow_run_id))
        self.assertIn(
            str(execution.get("workflow_status") or ""),
            {"completed", "blocked", "failed", "running", "pending", "cancelled", "skipped", "unknown"},
        )

        list_resp = self.client.get(f"/api/projects/{project_id}/pipeline/recipes/runs")
        self.assertEqual(list_resp.status_code, 200, list_resp.text)
        list_body = list_resp.json()
        self.assertGreaterEqual(int(list_body.get("count", 0)), 1)
        run_ids = {
            str(item.get("recipe_run_id"))
            for item in list_body.get("runs", [])
            if isinstance(item, dict)
        }
        self.assertIn(recipe_run_id, run_ids)

        detail_resp = self.client.get(
            f"/api/projects/{project_id}/pipeline/recipes/runs/{recipe_run_id}"
        )
        self.assertEqual(detail_resp.status_code, 200, detail_resp.text)
        detail_body = detail_resp.json()
        self.assertEqual(str(detail_body.get("recipe_run_id")), recipe_run_id)
        self.assertEqual(int(detail_body.get("workflow_run_id")), int(workflow_run_id))

        state_resp = self.client.get(f"/api/projects/{project_id}/pipeline/recipes/state")
        self.assertEqual(state_resp.status_code, 200, state_resp.text)
        state_body = state_resp.json()
        self.assertEqual(
            str((state_body.get("state") or {}).get("last_execution_recipe_run_id") or ""),
            recipe_run_id,
        )

    def test_pipeline_recipe_run_cancel_retry_resume_controls(self):
        project_id = self._create_project("phase16-recipes-4")

        initial = self.client.post(
            f"/api/projects/{project_id}/pipeline/recipes/run",
            json={
                "recipe_id": "recipe.pipeline.sft_default",
                "include_preflight": True,
                "execution_backend": "local",
                "async_run": False,
                "config": {"simulate_fail_stages": ["training"]},
            },
        )
        self.assertEqual(initial.status_code, 200, initial.text)
        initial_body = initial.json()
        source_recipe_run_id = str(initial_body.get("recipe_run_id") or "")
        self.assertTrue(source_recipe_run_id)
        source_execution = initial_body.get("execution", {})
        source_workflow = source_execution.get("workflow_run") or {}
        source_nodes = source_workflow.get("nodes") if isinstance(source_workflow, dict) else []
        training_node_id = ""
        if isinstance(source_nodes, list):
            for node in source_nodes:
                if isinstance(node, dict) and str(node.get("stage")) == "training":
                    training_node_id = str(node.get("node_id") or "")
                    break
        self.assertTrue(training_node_id)

        retry = self.client.post(
            f"/api/projects/{project_id}/pipeline/recipes/runs/{source_recipe_run_id}/retry",
            json={
                "execution_backend": "local",
                "async_run": False,
                "config": {"simulate_fail_stages": []},
            },
        )
        self.assertEqual(retry.status_code, 200, retry.text)
        retry_body = retry.json()
        retry_recipe_run_id = str(retry_body.get("recipe_run_id") or "")
        self.assertTrue(retry_recipe_run_id)
        self.assertNotEqual(retry_recipe_run_id, source_recipe_run_id)

        resume = self.client.post(
            f"/api/projects/{project_id}/pipeline/recipes/runs/{source_recipe_run_id}/resume",
            json={
                "execution_backend": "local",
                "async_run": False,
                "resume_from_node_id": training_node_id,
                "config": {"simulate_fail_stages": []},
            },
        )
        self.assertEqual(resume.status_code, 200, resume.text)
        resume_body = resume.json()
        resume_recipe_run_id = str(resume_body.get("recipe_run_id") or "")
        self.assertTrue(resume_recipe_run_id)
        self.assertNotEqual(resume_recipe_run_id, source_recipe_run_id)

        cancel = self.client.post(
            f"/api/projects/{project_id}/pipeline/recipes/runs/{source_recipe_run_id}/cancel"
        )
        self.assertEqual(cancel.status_code, 400, cancel.text)

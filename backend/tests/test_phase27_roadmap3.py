"""Phase 27 tests: roadmap3 implementation slices (items 1-4)."""

from __future__ import annotations

import asyncio
import json
import os
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase27_roadmap3_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase27_roadmap3_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"
os.environ["TRAINING_RUNTIME_PLUGIN_MODULES"] = '["app.plugins.training_runtimes.example_runtime"]'

from fastapi.testclient import TestClient

from app.main import app


class Phase27Roadmap3Tests(unittest.TestCase):
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
        unique_name = f"{name}-{uuid.uuid4().hex[:8]}"
        resp = self.client.post(
            "/api/projects",
            json={"name": unique_name, "description": "phase27"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_item1_active_learning_downvote_capture_and_alignment_compose(self):
        project_id = self._create_project("phase27-item1")

        import_resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/preference-dataset/import",
            json={
                "rows": [
                    {
                        "prompt": "How should API keys be rotated?",
                        "chosen": "Rotate API keys every 90 days and after incidents.",
                        "rejected": "Never rotate API keys.",
                    }
                ],
                "mode": "replace",
                "target": "prepared_train",
            },
        )
        self.assertEqual(import_resp.status_code, 200, import_resp.text)

        feedback_resp = self.client.post(
            f"/api/projects/{project_id}/training/playground/logs",
            json={
                "prompt": "How should API keys be rotated?",
                "reply": "Never rotate API keys.",
                "rating": -1,
                "preferred_reply": "Rotate API keys every 90 days and after incidents.",
                "tags": ["security", "policy"],
            },
        )
        self.assertEqual(feedback_resp.status_code, 200, feedback_resp.text)
        feedback_payload = feedback_resp.json()
        active_learning = feedback_payload.get("active_learning", {})
        self.assertTrue(bool(active_learning.get("captured")))

        summary_resp = self.client.get(
            f"/api/projects/{project_id}/training/alignment/active-learning",
            params={"refresh_pairs": True},
        )
        self.assertEqual(summary_resp.status_code, 200, summary_resp.text)
        summary = summary_resp.json()
        self.assertEqual(int(summary.get("rejected_count", 0)), 1)
        self.assertEqual(int(summary.get("auto_pair_count", 0)), 1)

        compose_resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/active-learning/compose",
            json={
                "include_playground_pairs": True,
                "max_playground_pairs": 100,
            },
        )
        self.assertEqual(compose_resp.status_code, 200, compose_resp.text)
        compose_payload = compose_resp.json()
        self.assertTrue(bool(compose_payload.get("written")))
        self.assertGreaterEqual(int(compose_payload.get("rows_written", 0)), 1)

        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase27-active-learning-dpo",
                "config": {
                    "base_model": "microsoft/phi-2",
                    "training_mode": "dpo",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "alignment_include_playground_feedback": True,
                },
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        experiment_id = int(exp_resp.json()["id"])

        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            return_value={
                "torch": True,
                "transformers": True,
                "datasets": True,
                "accelerate": True,
                "trl": True,
                "peft": True,
                "bitsandbytes": False,
            },
        ):
            start_resp = self.client.post(
                f"/api/projects/{project_id}/training/experiments/{experiment_id}/start"
            )
        self.assertEqual(start_resp.status_code, 200, start_resp.text)
        runtime = start_resp.json().get("config", {}).get("_runtime", {})
        active_learning_runtime = runtime.get("active_learning_feedback", {})
        self.assertTrue(bool(active_learning_runtime.get("enabled")))
        self.assertGreaterEqual(int(active_learning_runtime.get("playground_rows", 0)), 1)

    def test_item2_observability_telemetry_record_and_simulate_emission(self):
        project_id = self._create_project("phase27-item2")
        manual_event_resp = self.client.post(
            f"/api/projects/{project_id}/training/observability/telemetry",
            json={
                "experiment_id": 123,
                "step": 10,
                "epoch": 0.1,
                "split": "train",
                "layer_gradients": [
                    {"layer": "transformer.layers.0", "grad_norm": 0.42, "update_ratio": 0.0012},
                    {"layer": "transformer.layers.1", "grad_norm": 0.58, "update_ratio": 0.0016},
                ],
                "attention_focus": [
                    {"token": "domain_fact", "weight": 0.34, "source": "context"},
                ],
                "notes": "manual observability signal",
            },
        )
        self.assertEqual(manual_event_resp.status_code, 200, manual_event_resp.text)

        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase27-observability-sim",
                "config": {
                    "base_model": "microsoft/phi-2",
                    "num_epochs": 1,
                },
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        experiment_id = int(exp_resp.json()["id"])
        start_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/{experiment_id}/start"
        )
        self.assertEqual(start_resp.status_code, 200, start_resp.text)

        time.sleep(3.2)
        summary_resp = self.client.get(
            f"/api/projects/{project_id}/training/observability/telemetry",
            params={"experiment_id": experiment_id, "limit": 50},
        )
        self.assertEqual(summary_resp.status_code, 200, summary_resp.text)
        payload = summary_resp.json()
        summary = payload.get("summary", {})
        recent = payload.get("recent", {})
        self.assertGreaterEqual(int(summary.get("event_count", 0)), 1)
        self.assertGreaterEqual(int(recent.get("count", 0)), 1)
        self.assertTrue(bool(summary.get("top_layers")))

    def test_item2_observability_stream_event_ingest_from_external_monitor(self):
        from app.services.training_service import TRAINING_EVENT_PREFIX, _monitor_external_training

        project_id = self._create_project("phase27-item2-stream")
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase27-observability-stream-ingest",
                "config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        exp_payload = exp_resp.json()
        experiment_id = int(exp_payload["id"])
        output_dir = Path(str(exp_payload.get("output_dir") or ""))
        self.assertTrue(output_dir.exists(), output_dir)

        event_payload = {
            "event": "observability",
            "experiment_id": experiment_id,
            "step": 7,
            "epoch": 0.1,
            "split": "train",
            "layer_gradients": [
                {
                    "layer": "transformer.layers.0",
                    "grad_norm": 0.77,
                    "weight_norm": 23.0,
                    "update_ratio": 0.0033,
                }
            ],
            "attention_focus": [{"token": "domain_fact", "weight": 0.41, "source": "context"}],
            "notes": "monitor-stream-ingest",
        }
        captured_stdout = "\n".join(
            [
                "plain training log",
                f"{TRAINING_EVENT_PREFIX}{json.dumps(event_payload)}",
            ]
        )

        class _StubProcess:
            returncode = 0

            async def communicate(self):
                return b"", b""

        status = asyncio.run(
            _monitor_external_training(
                experiment_id=experiment_id,
                process=_StubProcess(),
                command="python backend/scripts/train.py --dry-run",
                log_path=output_dir / "external_monitor_stream_ingest.json",
                output_dir=output_dir,
                captured_stdout=captured_stdout,
                captured_stderr="",
            )
        )
        self.assertEqual(status, "completed")

        summary_resp = self.client.get(
            f"/api/projects/{project_id}/training/observability/telemetry",
            params={"experiment_id": experiment_id, "limit": 20},
        )
        self.assertEqual(summary_resp.status_code, 200, summary_resp.text)
        payload = summary_resp.json()
        recent_events = list(payload.get("recent", {}).get("events") or [])
        matched = next(
            (
                event
                for event in recent_events
                if int(event.get("step") or 0) == 7
                and str(event.get("notes") or "") == "monitor-stream-ingest"
            ),
            None,
        )
        self.assertIsNotNone(matched)

    def test_item3_local_judge_mode_uses_serve_target(self):
        project_id = self._create_project("phase27-item3")
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase27-local-judge-exp",
                "config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        experiment_id = int(exp_resp.json()["id"])

        with patch(
            "app.services.evaluation_service._resolve_local_judge_target",
            return_value={
                "run_id": "serve123",
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "method": "POST",
                "transport": "openai_chat",
                "model": "local-judge",
            },
        ), patch(
            "app.services.evaluation_service._judge_with_local_serve",
            new=AsyncMock(return_value=(5, "Local serve judge score")),
        ):
            judge_resp = self.client.post(
                f"/api/projects/{project_id}/evaluation/llm-judge",
                json={
                    "experiment_id": experiment_id,
                    "dataset_name": "heldout",
                    "judge_model": "local_serve:auto",
                    "predictions": [
                        {
                            "prompt": "How should logs be retained?",
                            "reference": "Retain logs for 30 days.",
                            "prediction": "Retain logs for 30 days.",
                        },
                        {
                            "prompt": "How should secrets be stored?",
                            "reference": "Use encrypted secret managers.",
                            "prediction": "Use encrypted secret managers.",
                        },
                    ],
                },
            )
        self.assertEqual(judge_resp.status_code, 201, judge_resp.text)
        metrics = judge_resp.json().get("metrics", {})
        self.assertEqual(str(metrics.get("judge_provider")), "local_serve")
        local_judge = metrics.get("local_judge", {})
        self.assertEqual(local_judge.get("run_id"), "serve123")
        self.assertEqual(int(metrics.get("fallback_count", 0)), 0)

    def test_item3_local_judge_list_serve_runs_filters_candidates(self):
        project_id = self._create_project("phase27-item3-runs")
        mocked_runs = {
            "runs": [
                {
                    "run_id": "run-usable-smoke",
                    "status": "running",
                    "source": "export",
                    "template_id": "builtin.fastapi",
                    "template_name": "FastAPI",
                    "export_id": 11,
                    "model_id": None,
                    "telemetry": {
                        "smoke_url": "http://127.0.0.1:8100/v1/chat/completions",
                        "first_healthy_at": "2026-03-11T12:00:00Z",
                        "startup_latency_ms": 1345,
                    },
                },
                {
                    "run_id": "run-not-usable",
                    "status": "running",
                    "source": "export",
                    "template_id": "builtin.fastapi",
                    "template_name": "FastAPI",
                    "telemetry": {},
                },
                {
                    "run_id": "run-usable-first-token",
                    "status": "completed",
                    "source": "registry",
                    "template_id": "runner.vllm",
                    "template_name": "vLLM",
                    "export_id": 12,
                    "model_id": 99,
                    "telemetry": {
                        "first_token_url": "http://127.0.0.1:8200/v1/chat/completions",
                        "first_healthy_at": "2026-03-11T12:00:03Z",
                        "startup_latency_ms": 1560,
                    },
                },
            ]
        }
        with patch("app.api.evaluation.list_serve_runs", new=AsyncMock(return_value=mocked_runs)):
            resp = self.client.get(f"/api/projects/{project_id}/evaluation/local-judge/serve-runs")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(int(payload.get("count", 0)), 2)
        runs = [item for item in payload.get("runs", []) if isinstance(item, dict)]
        run_ids = {str(item.get("run_id")) for item in runs}
        self.assertEqual(run_ids, {"run-usable-smoke", "run-usable-first-token"})

    def test_item3_local_judge_auto_start_and_auto_stop(self):
        project_id = self._create_project("phase27-item3-auto-start")
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase27-local-judge-auto-start-exp",
                "config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        experiment_id = int(exp_resp.json()["id"])

        resolve_mock = AsyncMock(
            side_effect=[
                ValueError(
                    "No serve runtime found for local judge. Start a serve run first or configure remote judge API."
                ),
                {
                    "run_id": "serve456",
                    "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                    "method": "POST",
                    "transport": "openai_chat",
                    "model": "local-judge",
                },
            ]
        )
        start_mock = AsyncMock(return_value={"run_id": "serve456", "status": "running"})
        wait_mock = AsyncMock(return_value=None)
        judge_mock = AsyncMock(return_value=(4, "Auto-started local serve judge score"))
        stop_mock = AsyncMock(return_value={"run_id": "serve456", "status": "cancelled"})

        with patch("app.services.evaluation_service._resolve_local_judge_target", new=resolve_mock), patch(
            "app.services.evaluation_service._start_local_judge_serve_run",
            new=start_mock,
        ), patch(
            "app.services.evaluation_service._wait_for_local_judge_ready",
            new=wait_mock,
        ), patch(
            "app.services.evaluation_service._judge_with_local_serve",
            new=judge_mock,
        ), patch(
            "app.services.serve_runtime_service.stop_serve_run",
            new=stop_mock,
        ):
            judge_resp = self.client.post(
                f"/api/projects/{project_id}/evaluation/llm-judge",
                json={
                    "experiment_id": experiment_id,
                    "dataset_name": "heldout",
                    "judge_model": "local_serve:auto?auto_start=1&auto_stop=1&source=export&export_id=15",
                    "predictions": [
                        {
                            "prompt": "How should logs be retained?",
                            "reference": "Retain logs for 30 days.",
                            "prediction": "Retain logs for 30 days.",
                        }
                    ],
                },
            )
        self.assertEqual(judge_resp.status_code, 201, judge_resp.text)
        metrics = judge_resp.json().get("metrics", {})
        self.assertEqual(str(metrics.get("judge_provider")), "local_serve")
        local_judge = metrics.get("local_judge", {})
        self.assertEqual(local_judge.get("auto_started_run_id"), "serve456")
        self.assertEqual(bool(local_judge.get("auto_stop_enabled")), True)
        notes = [str(item) for item in list(local_judge.get("notes") or [])]
        self.assertIn("auto_started_serve_run", notes)
        self.assertIn("auto_stopped_serve_run", notes)
        self.assertEqual(resolve_mock.await_count, 2)
        start_mock.assert_awaited_once()
        wait_mock.assert_awaited_once()
        stop_mock.assert_awaited_once()

    def test_item3_local_judge_auto_start_does_not_mask_endpoint_errors(self):
        project_id = self._create_project("phase27-item3-auto-start-errors")
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase27-local-judge-auto-start-errors-exp",
                "config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        experiment_id = int(exp_resp.json()["id"])

        resolve_mock = AsyncMock(
            side_effect=ValueError(
                "Unsupported local judge endpoint format 'http://127.0.0.1:9999/foo'. "
                "Supported formats: /v1/chat/completions (OpenAI-compatible)."
            )
        )
        start_mock = AsyncMock(return_value={"run_id": "serve999"})
        with patch("app.services.evaluation_service._resolve_local_judge_target", new=resolve_mock), patch(
            "app.services.evaluation_service._start_local_judge_serve_run",
            new=start_mock,
        ):
            judge_resp = self.client.post(
                f"/api/projects/{project_id}/evaluation/llm-judge",
                json={
                    "experiment_id": experiment_id,
                    "dataset_name": "heldout",
                    "judge_model": "local_serve:auto?auto_start=1&source=export&export_id=21",
                    "predictions": [
                        {
                            "prompt": "How should secrets be stored?",
                            "reference": "Use encrypted secret managers.",
                            "prediction": "Use encrypted secret managers.",
                        }
                    ],
                },
            )
        self.assertEqual(judge_resp.status_code, 404, judge_resp.text)
        self.assertIn("Unsupported local judge endpoint format", judge_resp.text)
        self.assertEqual(start_mock.await_count, 0)

    def test_item4_runtime_plugin_reload_and_catalog(self):
        project_id = self._create_project("phase27-item4")
        reload_resp = self.client.post(
            f"/api/projects/{project_id}/training/runtimes/plugins/reload"
        )
        self.assertEqual(reload_resp.status_code, 200, reload_resp.text)
        reload_payload = reload_resp.json()
        reload_status = reload_payload.get("reload", {})
        self.assertIn(
            "app.plugins.training_runtimes.example_runtime",
            [str(item) for item in reload_status.get("loaded_modules", [])],
        )

        status_resp = self.client.get(
            f"/api/projects/{project_id}/training/runtimes/plugins/status"
        )
        self.assertEqual(status_resp.status_code, 200, status_resp.text)
        status_payload = status_resp.json()
        self.assertIn(
            "app.plugins.training_runtimes.example_runtime",
            [str(item) for item in status_payload.get("loaded_modules", [])],
        )

        catalog_resp = self.client.get(f"/api/projects/{project_id}/training/runtimes")
        self.assertEqual(catalog_resp.status_code, 200, catalog_resp.text)
        runtimes = [item for item in catalog_resp.json().get("runtimes", []) if isinstance(item, dict)]
        plugin_runtime = next(
            (item for item in runtimes if item.get("runtime_id") == "plugin.example_simulate"),
            None,
        )
        self.assertIsNotNone(plugin_runtime)
        self.assertEqual(
            plugin_runtime.get("source_module"),
            "app.plugins.training_runtimes.example_runtime",
        )
        self.assertEqual(plugin_runtime.get("supported_modalities"), ["text"])
        self.assertTrue(bool(plugin_runtime.get("declares_supported_modalities")))


if __name__ == "__main__":
    unittest.main()

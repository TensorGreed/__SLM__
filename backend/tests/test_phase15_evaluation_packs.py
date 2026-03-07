"""Phase 15 tests: evaluation packs and auto-gates."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase15_eval_packs_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase15_eval_packs_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase15EvaluationPackTests(unittest.TestCase):
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
            json={"name": name, "description": "phase15"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_experiment(self, project_id: int, name: str = "phase15-exp") -> int:
        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": name,
                "config": {
                    "base_model": "microsoft/phi-2",
                },
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_eval_pack_preference_roundtrip(self):
        project_id = self._create_project("phase15-packs-1")

        packs = self.client.get(f"/api/projects/{project_id}/evaluation/packs")
        self.assertEqual(packs.status_code, 200, packs.text)
        packs_payload = packs.json()
        pack_ids = {str(item.get("pack_id")) for item in packs_payload.get("packs", []) if isinstance(item, dict)}
        self.assertIn("evalpack.general.default", pack_ids)
        self.assertIn("evalpack.fast.iteration", pack_ids)
        self.assertEqual(packs_payload.get("default_pack_id"), "evalpack.general.default")

        pref = self.client.get(f"/api/projects/{project_id}/evaluation/pack-preference")
        self.assertEqual(pref.status_code, 200, pref.text)
        self.assertTrue(bool(pref.json().get("active_pack_id")))

        update = self.client.put(
            f"/api/projects/{project_id}/evaluation/pack-preference",
            json={"pack_id": "evalpack.fast.iteration"},
        )
        self.assertEqual(update.status_code, 200, update.text)
        update_payload = update.json()
        self.assertEqual(update_payload.get("preferred_pack_id"), "evalpack.fast.iteration")
        self.assertEqual(update_payload.get("active_pack_id"), "evalpack.fast.iteration")
        self.assertEqual(update_payload.get("active_pack_source"), "project")

        clear = self.client.put(
            f"/api/projects/{project_id}/evaluation/pack-preference",
            json={"pack_id": None},
        )
        self.assertEqual(clear.status_code, 200, clear.text)
        self.assertIsNone(clear.json().get("preferred_pack_id"))
        self.assertTrue(bool(clear.json().get("active_pack_id")))

    def test_eval_gate_report_and_pipeline_auto_gate(self):
        project_id = self._create_project("phase15-packs-2")
        experiment_id = self._create_experiment(project_id)

        pref = self.client.put(
            f"/api/projects/{project_id}/evaluation/pack-preference",
            json={"pack_id": "evalpack.fast.iteration"},
        )
        self.assertEqual(pref.status_code, 200, pref.text)

        exact_eval = self.client.post(
            f"/api/projects/{project_id}/evaluation/run",
            json={
                "experiment_id": experiment_id,
                "dataset_name": "gold_test",
                "eval_type": "exact_match",
                "predictions": [
                    {"prediction": "Answer A", "reference": "Answer A"},
                    {"prediction": "Answer B", "reference": "Answer B"},
                ],
            },
        )
        self.assertEqual(exact_eval.status_code, 201, exact_eval.text)

        f1_eval = self.client.post(
            f"/api/projects/{project_id}/evaluation/run",
            json={
                "experiment_id": experiment_id,
                "dataset_name": "gold_test",
                "eval_type": "f1",
                "predictions": [
                    {"prediction": "Correct response", "reference": "Correct response"},
                ],
            },
        )
        self.assertEqual(f1_eval.status_code, 201, f1_eval.text)

        gates = self.client.get(
            f"/api/projects/{project_id}/evaluation/gates/{experiment_id}",
            params={"pack_id": "evalpack.fast.iteration"},
        )
        self.assertEqual(gates.status_code, 200, gates.text)
        gates_payload = gates.json()
        self.assertTrue(gates_payload.get("passed"), gates_payload)
        checks = gates_payload.get("checks", [])
        self.assertTrue(any(str(item.get("gate_id")) == "min_exact_match" for item in checks))

        status = self.client.get(f"/api/projects/{project_id}/pipeline/status")
        self.assertEqual(status.status_code, 200, status.text)
        auto_gate = status.json().get("auto_gate")
        self.assertIsInstance(auto_gate, dict)
        self.assertEqual(auto_gate.get("experiment_id"), experiment_id)
        self.assertEqual(auto_gate.get("pack_id"), "evalpack.fast.iteration")
        self.assertTrue(auto_gate.get("passed"))

    def test_eval_pack_rejects_unknown_ids(self):
        project_id = self._create_project("phase15-packs-3")
        experiment_id = self._create_experiment(project_id)

        update = self.client.put(
            f"/api/projects/{project_id}/evaluation/pack-preference",
            json={"pack_id": "evalpack.unknown"},
        )
        self.assertEqual(update.status_code, 400, update.text)
        self.assertIn("Unsupported evaluation pack", str(update.json().get("detail", "")))

        gates = self.client.get(
            f"/api/projects/{project_id}/evaluation/gates/{experiment_id}",
            params={"pack_id": "evalpack.unknown"},
        )
        self.assertEqual(gates.status_code, 400, gates.text)
        self.assertIn("Unsupported pack_id", str(gates.json().get("detail", "")))

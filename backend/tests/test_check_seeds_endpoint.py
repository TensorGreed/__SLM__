"""``POST /evaluation/summary/check-seeds`` — the multi-seed option for the
lift check: re-train a run's exact config with N seeds as one seed group.
Training itself is stubbed; this pins the clone + fan-out request."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


class CheckSeedsEndpointTests(unittest.TestCase):
    def setUp(self):
        resp = client.post("/api/projects", json={"name": f"seeds-{uuid4().hex[:6]}", "description": ""})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.pid = int(resp.json()["id"])
        exp = client.post(
            f"/api/projects/{self.pid}/training/experiments",
            json={"name": "Support FAQ · default config", "config": {
                "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct", "learning_rate": 0.0005, "lora_r": 32,
            }},
        )
        self.assertEqual(exp.status_code, 201, exp.text)
        self.exp_id = int(exp.json()["id"])

    def test_clones_the_config_as_a_seed_group_and_starts_it(self):
        started = AsyncMock(return_value={"status": "running", "seed_group_id": "abc"})
        with patch("app.api.training.start", started):
            resp = client.post(
                f"/api/projects/{self.pid}/evaluation/summary/check-seeds",
                json={"experiment_id": self.exp_id, "num_seeds": 3},
            )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual(body["source_experiment_id"], self.exp_id)
        self.assertEqual(body["num_seeds"], 3)
        self.assertEqual(body["experiment_name"], "Support FAQ · default config · 3 seeds")
        self.assertNotEqual(body["experiment_id"], self.exp_id)
        started.assert_awaited_once()
        self.assertEqual(started.await_args.args[:2], (self.pid, body["experiment_id"]))

        experiments = client.get(f"/api/projects/{self.pid}/training/experiments").json()
        leader = next(e for e in experiments if e["id"] == body["experiment_id"])
        cfg = leader["config"]
        # Same recipe (the user's knobs survive), N seeds, no stale stamps.
        self.assertEqual((cfg["learning_rate"], cfg["lora_r"], cfg["num_seeds"]), (0.0005, 32, 3))
        self.assertIsNone(cfg.get("seeds"))
        self.assertNotIn("_runtime", cfg)

    def test_rejects_unknown_runs_and_bad_seed_counts(self):
        resp = client.post(
            f"/api/projects/{self.pid}/evaluation/summary/check-seeds",
            json={"experiment_id": 999999, "num_seeds": 3},
        )
        self.assertEqual(resp.status_code, 404)
        resp = client.post(
            f"/api/projects/{self.pid}/evaluation/summary/check-seeds",
            json={"experiment_id": self.exp_id, "num_seeds": 1},
        )
        self.assertEqual(resp.status_code, 422)

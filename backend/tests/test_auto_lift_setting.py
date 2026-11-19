"""Per-project switch for the automatic lift eval: setter, GET payload, precedence with the global
switch, and ``auto_lift_skip_reason`` honouring it. No Jobs are started. pytest-only."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project
from app.services.post_training_eval_service import auto_lift_setting, auto_lift_skip_reason

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def _project() -> int:
    return client.post("/api/projects", json={"name": f"auto-lift-{time.time_ns()}", "description": "x"}).json()["id"]


class AutoLiftSettingApiTests(unittest.TestCase):
    def test_default_follows_global_and_is_on(self):
        pid = _project()
        body = client.get(f"/api/projects/{pid}/settings/auto-lift-eval").json()
        self.assertEqual(body, {"enabled": None, "global_enabled": True, "effective": True, "reason": None})

    def test_off_then_on_then_follow_default(self):
        pid = _project()
        url = f"/api/projects/{pid}/settings/auto-lift-eval"
        off = client.put(url, json={"enabled": False}).json()
        self.assertEqual((off["enabled"], off["effective"], off["reason"]), (False, False, "disabled_for_project"))
        self.assertFalse(client.get(url).json()["effective"])
        on = client.put(url, json={"enabled": True}).json()
        self.assertEqual((on["enabled"], on["effective"]), (True, True))
        cleared = client.put(url, json={"enabled": None}).json()
        self.assertEqual((cleared["enabled"], cleared["effective"]), (None, True))

    def test_other_runtime_config_keys_survive(self):
        pid = _project()

        async def seed():
            async with async_session_factory() as db:
                p = await db.get(Project, pid)
                p.runtime_config = {"keep_me": {"a": 1}}
                await db.commit()

        asyncio.run(seed())
        client.put(f"/api/projects/{pid}/settings/auto-lift-eval", json={"enabled": False})

        async def read():
            async with async_session_factory() as db:
                return (await db.get(Project, pid)).runtime_config

        cfg = asyncio.run(read())
        self.assertEqual(cfg["keep_me"], {"a": 1})
        self.assertIs(cfg["auto_lift_eval"], False)

    def test_global_off_wins_over_project_on(self):
        pid = _project()
        url = f"/api/projects/{pid}/settings/auto-lift-eval"
        client.put(url, json={"enabled": True})
        with patch.object(settings, "AUTO_LIFT_EVAL_ENABLED", False):
            body = client.get(url).json()
        self.assertEqual((body["enabled"], body["global_enabled"], body["effective"], body["reason"]),
                         (True, False, False, "disabled_globally"))

    def test_unknown_project_404(self):
        self.assertEqual(client.get("/api/projects/99999999/settings/auto-lift-eval").status_code, 404)
        self.assertEqual(client.put("/api/projects/99999999/settings/auto-lift-eval", json={"enabled": False}).status_code, 404)


class SkipReasonTests(unittest.TestCase):
    def _exp(self) -> Experiment:
        return Experiment(project_id=1, name="e", status=ExperimentStatus.COMPLETED, config={})

    def test_project_setting_drives_skip_reason(self):
        exp = self._exp()
        off = Project(name="p", runtime_config={"auto_lift_eval": False})
        on = Project(name="p", runtime_config={"auto_lift_eval": True})
        unset = Project(name="p", runtime_config={})
        self.assertEqual(auto_lift_skip_reason(exp, off), "disabled_for_project")
        self.assertNotEqual(auto_lift_skip_reason(exp, on), "disabled_for_project")
        self.assertNotEqual(auto_lift_skip_reason(exp, unset), "disabled_for_project")
        # The user asking for the check by hand ignores the opt-out.
        self.assertNotEqual(auto_lift_skip_reason(exp, off, manual=True), "disabled_for_project")

    def test_global_off_beats_project_on(self):
        exp = self._exp()
        with patch.object(settings, "AUTO_LIFT_EVAL_ENABLED", False):
            self.assertEqual(auto_lift_skip_reason(exp, Project(name="p", runtime_config={"auto_lift_eval": True})),
                             "disabled_globally")

    def test_setting_helper_ignores_non_bool_values(self):
        self.assertIsNone(auto_lift_setting(Project(name="p", runtime_config={"auto_lift_eval": "no"}))["enabled"])
        self.assertIsNone(auto_lift_setting(None)["enabled"])


if __name__ == "__main__":
    unittest.main()

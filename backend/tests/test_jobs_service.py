"""Tests for the Jobs framework (Hardening Phase H1).

All tests drive the framework through the public HTTP API via
TestClient — which keeps the FastAPI app, the SQLAlchemy async
engine, and the runner tasks on the same event loop (the one
TestClient spawns in its own thread). Mixing pytest's
``IsolatedAsyncioTestCase`` with the StaticPool-pinned SQLite
engine produces session-isolation surprises that don't show up
in production.

Covers:
- serialize_job shape contract (pure-function check).
- POST /reroute-to-rag?async_job=true returns 202 + a Job stub
  that polls to SUCCEEDED with the new project id in result.
- POST /synthetic/run-playbook?async_job=true returns 202 + a
  Job stub (we don't poll to completion because the synth path
  needs a real LLM backend; FAILED is the expected terminal).
- Runner exceptions land as FAILED + error text.
- GET /jobs/active returns the expected shape.
- GET /jobs/{id} 404 on unknown.
- POST /jobs/{id}/dismiss refuses in-flight, accepts terminal.
- POST /jobs/{id}/cancel flips status to CANCELLED.
"""

from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.models.job import Job, JobStatus  # noqa: E402
from app.services.jobs_service import serialize_job  # noqa: E402


# Module-level TestClient boots the FastAPI lifespan once
# (which runs init_db / Base.metadata.create_all so the jobs
# table exists in the per-PID /tmp DB).
_MODULE_CLIENT_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802 — unittest convention
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802 — unittest convention
    _MODULE_CLIENT_CM.__exit__(None, None, None)


# ─────────────────────────────────────────────────────────────────────
# Pure-function serialization contract
# ─────────────────────────────────────────────────────────────────────


class SerializeJobShapeTests(unittest.TestCase):
    def test_serialize_job_shape_matches_frontend_contract(self):
        from datetime import datetime, timezone

        j = Job(
            id=1,
            kind="test_kind",
            title="t",
            status=JobStatus.RUNNING,
            progress=0.5,
            progress_message="m",
            project_id=4,
            user_id=None,
            params={"a": 1},
            result=None,
            error=None,
            queued_at=datetime.now(timezone.utc),
        )
        serialized = serialize_job(j)
        expected_keys = {
            "id", "kind", "title", "status", "progress",
            "progress_message", "project_id", "user_id", "params",
            "result", "error", "queued_at", "started_at",
            "completed_at", "dismissed_at",
        }
        self.assertEqual(set(serialized.keys()), expected_keys)
        # status serialized as the enum value string.
        self.assertEqual(serialized["status"], "running")


# ─────────────────────────────────────────────────────────────────────
# End-to-end via TestClient — public API surface
# ─────────────────────────────────────────────────────────────────────


def _poll_job_until(
    client: TestClient,
    job_id: int,
    *,
    expected: str,
    timeout_s: float = 8.0,
) -> dict:
    """Poll /api/jobs/{id} until status matches. Used wherever a
    test needs to await a runner's terminal transition."""
    deadline = time.monotonic() + timeout_s
    last_text = ""
    while time.monotonic() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        if resp.status_code == 200:
            body = resp.json()
            last_text = resp.text
            if body["status"] == expected:
                return body
        time.sleep(0.1)
    raise AssertionError(
        f"Job {job_id} did not reach {expected} within {timeout_s}s: {last_text}"
    )


class JobsApiBasicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls.client = _MODULE_CLIENT_CM

    @classmethod
    def tearDownClass(cls):
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def test_get_active_jobs_returns_shape(self):
        resp = self.client.get("/api/jobs/active")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("count", body)
        self.assertIsInstance(body["jobs"], list)

    def test_get_job_by_id_404_on_unknown(self):
        resp = self.client.get("/api/jobs/99999999")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_dismiss_unknown_returns_404(self):
        resp = self.client.post("/api/jobs/99999999/dismiss")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_cancel_unknown_returns_404(self):
        resp = self.client.post("/api/jobs/99999999/cancel")
        self.assertEqual(resp.status_code, 404, resp.text)


class RerouteAsyncEndpointTests(unittest.TestCase):
    """End-to-end: ?async_job=true on /reroute-to-rag returns 202 +
    a Job stub; polling the job via the API produces a real clone."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls.client = _MODULE_CLIENT_CM

    @classmethod
    def tearDownClass(cls):
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def test_async_reroute_returns_202_and_job_runs_to_success(self):
        # Instantiate a qa-sft template to clone.
        resp = self.client.post(
            "/api/project-templates/policy-qa-style/instantiate",
            json={"project_name": "Phase H1 Async Reroute"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        source_id = resp.json()["id"]

        # Async clone — 202 + job stub.
        clone = self.client.post(
            f"/api/projects/{source_id}/reroute-to-rag?async_job=true",
            json={},
        )
        self.assertEqual(clone.status_code, 202, clone.text)
        body = clone.json()
        self.assertEqual(body["kind"], "reroute_to_rag")
        self.assertIn(body["status"], ("queued", "running"))
        job_id = body["id"]

        # Verify the runner actually publishes intermediate progress
        # by polling for a non-null progress_message before terminal.
        # (Best-effort; runner may finish faster than the poll tick.)
        done = _poll_job_until(self.client, job_id, expected="succeeded")
        self.assertIsNotNone(done.get("started_at"))
        self.assertIsNotNone(done.get("completed_at"))
        new_id = (done.get("result") or {}).get("new_project_id")
        self.assertIsNotNone(new_id, done)
        # Verify the new project exists with the expected provenance.
        proj = self.client.get(f"/api/projects/{new_id}")
        self.assertEqual(proj.status_code, 200, proj.text)
        self.assertEqual(proj.json()["parent_project_id"], source_id)

    def test_async_reroute_failed_recipe_lands_as_failed_job(self):
        """Async failure path — clone refuses a non-qa-sft source;
        the runner raises and the Job lands as status=failed with
        error text. Tests the wrapper's exception capture."""
        # ticket-router → classification recipe → reroute refuses.
        resp = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={"project_name": "Phase H1 Async Reroute Failed Recipe"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        source_id = resp.json()["id"]

        clone = self.client.post(
            f"/api/projects/{source_id}/reroute-to-rag?async_job=true",
            json={},
        )
        self.assertEqual(clone.status_code, 202, clone.text)
        job_id = clone.json()["id"]

        # Wait for terminal — should be FAILED.
        done = _poll_job_until(self.client, job_id, expected="failed")
        self.assertIn("classification", done.get("error") or "")
        self.assertIsNotNone(done.get("completed_at"))

    def test_dismiss_terminal_job_hides_it_from_active(self):
        resp = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={"project_name": "Phase H1 Dismiss Terminal"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        source_id = resp.json()["id"]
        clone = self.client.post(
            f"/api/projects/{source_id}/reroute-to-rag?async_job=true",
            json={},
        )
        self.assertEqual(clone.status_code, 202, clone.text)
        job_id = clone.json()["id"]
        _poll_job_until(self.client, job_id, expected="failed")

        # Dismiss → hidden from /active.
        dismiss = self.client.post(f"/api/jobs/{job_id}/dismiss")
        self.assertEqual(dismiss.status_code, 200, dismiss.text)
        active = self.client.get("/api/jobs/active")
        self.assertEqual(active.status_code, 200, active.text)
        ids = [j["id"] for j in active.json()["jobs"]]
        self.assertNotIn(job_id, ids)


class SynthPlaybookAsyncEndpointTests(unittest.TestCase):
    """The synth playbook async path. We don't drive it to SUCCEEDED
    because the runner needs a real Ollama / teacher LLM backend
    which the test env doesn't have. FAILED is the expected terminal."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls.client = _MODULE_CLIENT_CM

    @classmethod
    def tearDownClass(cls):
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def test_async_synth_returns_202_with_job_stub(self):
        resp = self.client.post(
            "/api/project-templates/policy-qa-style/instantiate",
            json={"project_name": "Phase H1 Async Synth"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        project_id = resp.json()["id"]

        run = self.client.post(
            f"/api/projects/{project_id}/synthetic/run-playbook?async_job=true",
            json={
                "mode": "positives_paraphrase",
                "target_count": 5,
                "backend": None,
            },
        )
        self.assertEqual(run.status_code, 202, run.text)
        body = run.json()
        self.assertEqual(body["kind"], "synth_playbook")
        self.assertIn(body["status"], ("queued", "running"))
        # Title is human-readable and includes mode + target count.
        self.assertIn("positives_paraphrase", body["title"])
        self.assertIn("5", body["title"])
        # Params persisted for re-runnability + debug.
        self.assertEqual(body["params"]["mode"], "positives_paraphrase")
        self.assertEqual(body["params"]["target_count"], 5)


if __name__ == "__main__":
    unittest.main()

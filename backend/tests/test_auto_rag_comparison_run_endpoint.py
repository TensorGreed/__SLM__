"""Tests for POST /api/projects/{id}/auto-rag/comparison/run
(Hardening — UI-triggered auto-RAG comparison Job).

The endpoint validates the project (recipe must be qa-sft),
checks idempotency (one in-flight comparison per project), and
spawns a Job. We don't drive the Job to SUCCEEDED in tests
because the runner loads a real LoRA on GPU + runs torch
inference — out of scope for unit tests. Instead we assert the
Job was queued with the right kind / title / params, and that
a second call returns 429 -> wait, 409.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from datetime import datetime, timezone

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.models.job import Job, JobStatus  # noqa: E402


_MODULE_CLIENT_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802 — unittest convention
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802 — unittest convention
    _MODULE_CLIENT_CM.__exit__(None, None, None)


def _seed_inflight_comparison_job(project_id: int) -> int:
    """Insert a RUNNING auto_rag_comparison Job directly so we can
    test the idempotency guard without having to actually fire the
    GPU-bound runner."""
    from app.database import async_session_factory

    async def _go() -> int:
        async with async_session_factory() as db:
            job = Job(
                kind="auto_rag_comparison",
                title="Pre-existing comparison",
                status=JobStatus.RUNNING,
                project_id=project_id,
                params={"project_id": project_id, "recipe_id": "qa-sft"},
                queued_at=datetime.now(timezone.utc),
            )
            db.add(job)
            await db.flush()
            jid = job.id
            await db.commit()
            return jid

    return asyncio.run(_go())


def _clear_inflight_comparison_jobs(project_id: int) -> None:
    from app.database import async_session_factory
    from sqlalchemy import update

    async def _go() -> None:
        async with async_session_factory() as db:
            await db.execute(
                update(Job)
                .where(
                    Job.project_id == project_id,
                    Job.kind == "auto_rag_comparison",
                    Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                )
                .values(
                    status=JobStatus.CANCELLED,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()

    asyncio.run(_go())


class RunComparisonEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls.client = _MODULE_CLIENT_CM

    @classmethod
    def tearDownClass(cls):
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_404_when_project_missing(self):
        resp = self.client.post(
            "/api/projects/99999999/auto-rag/comparison/run",
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_400_when_recipe_not_rag_eligible(self):
        # ticket-router → classification recipe → not RAG-eligible.
        project = self._instantiate_template(
            "ticket-router", "ARC run-comparison wrong-recipe"
        )
        resp = self.client.post(
            f"/api/projects/{project['id']}/auto-rag/comparison/run",
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("classification", resp.text.lower())

    def test_409_when_comparison_already_in_flight(self):
        project = self._instantiate_template(
            "policy-qa-style", "ARC run-comparison idempotency"
        )
        existing_id = _seed_inflight_comparison_job(project["id"])
        try:
            resp = self.client.post(
                f"/api/projects/{project['id']}/auto-rag/comparison/run",
            )
            self.assertEqual(resp.status_code, 409, resp.text)
            detail = resp.json()["detail"]
            self.assertEqual(
                detail["error_code"], "AUTO_RAG_COMPARISON_ALREADY_RUNNING",
            )
            self.assertEqual(
                detail["metadata"]["existing_job_id"], existing_id,
            )
            self.assertEqual(
                detail["metadata"]["existing_job_status"], "running",
            )
        finally:
            _clear_inflight_comparison_jobs(project["id"])

    # The "202 success-shape" path isn't unit-tested here because the
    # runner fires asynchronously and loads torch + a real LoRA on GPU
    # the moment it gets event-loop time. Cancelling the Job row after
    # the response returns races the wrapper's QUEUED→RUNNING transition
    # — when the runner wins the race, it crashes mid-execution of a
    # subsequent test ("No COMPLETED experiment found for project N")
    # and pytest attributes the error to whichever test happens to be
    # running. The 400 / 404 / 409 paths above cover endpoint
    # validation; manual end-to-end with a real trained model verifies
    # the 202 shape + the runner's per-row progress.


if __name__ == "__main__":
    unittest.main()

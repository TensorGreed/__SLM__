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
            # The structured-error envelope hoists error_code + metadata to the
            # top level (`detail` is now the message string), so read the body.
            detail = resp.json()
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

    def test_base_model_run_shares_the_in_flight_guard(self):
        """``?model=base`` is the same GPU-heavy Job kind: one comparison per
        project at a time, whichever model it scores."""
        project = self._instantiate_template(
            "policy-qa-style", "ARC run-comparison base model"
        )
        _seed_inflight_comparison_job(project["id"])
        try:
            resp = self.client.post(
                f"/api/projects/{project['id']}/auto-rag/comparison/run",
                params={"model": "base"},
            )
            self.assertEqual(resp.status_code, 409, resp.text)
            bad = self.client.post(
                f"/api/projects/{project['id']}/auto-rag/comparison/run",
                params={"model": "teacher"},
            )
            self.assertEqual(bad.status_code, 422, bad.text)
        finally:
            _clear_inflight_comparison_jobs(project["id"])

    def _add_run(self, project_id: int, status: str = "completed") -> int:
        from app.database import async_session_factory
        from app.models.experiment import Experiment, ExperimentStatus, TrainingMode

        async def _go() -> int:
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=project_id, name="scored run", status=ExperimentStatus(status),
                    training_mode=TrainingMode.SFT, base_model="HuggingFaceTB/SmolLM2-135M-Instruct", config={},
                )
                db.add(exp)
                await db.commit()
                return exp.id

        return asyncio.run(_go())

    def test_experiment_id_must_be_a_completed_run_of_this_project(self):
        project = self._instantiate_template("policy-qa-style", "ARC run-comparison experiment_id checks")
        other = self._instantiate_template("policy-qa-style", "ARC run-comparison other project")
        url = f"/api/projects/{project['id']}/auto-rag/comparison/run"
        done = self._add_run(project["id"])

        with_base = self.client.post(url, params={"model": "base", "experiment_id": done})
        self.assertEqual(with_base.status_code, 400, with_base.text)
        self.assertEqual(with_base.json()["error_code"], "EXPERIMENT_ID_WITH_BASE_MODEL")

        self.assertEqual(self.client.post(url, params={"experiment_id": 99999999}).status_code, 404)
        foreign = self.client.post(url, params={"experiment_id": self._add_run(other["id"])})
        self.assertEqual(foreign.status_code, 404, foreign.text)

        pending = self._add_run(project["id"], status="pending")
        not_done = self.client.post(url, params={"experiment_id": pending})
        self.assertEqual(not_done.status_code, 400, not_done.text)
        body = not_done.json()
        self.assertEqual(body["error_code"], "EXPERIMENT_NOT_COMPLETED")
        self.assertEqual(body["metadata"], {"experiment_id": pending, "status": "pending"})

    def test_experiment_id_reaches_the_harness_and_labels_the_job(self):
        """A run other than the latest is scored as a Job (it used to need the
        CLI's --experiment-id, which the bell and watchdogs can't see). The
        harness is patched: no model is loaded."""
        from unittest.mock import patch

        from _job_wait import wait_for_jobs_idle

        project = self._instantiate_template("policy-qa-style", "ARC run-comparison older run")
        older = self._add_run(project["id"])
        self._add_run(project["id"])  # the latest, which is NOT scored
        calls: list[dict] = []

        def _fake_comparison(project_id, **kwargs):
            calls.append({"project_id": project_id, **kwargs})
            return {"experiment_id": kwargs.get("experiment_id"), "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
                    "summary": {"off_mean_f1": 0.1, "on_mean_f1": 0.3, "n_val_rows": 0}, "rows": []}

        with patch("scripts.auto_rag_ab.run_project_comparison", _fake_comparison):
            resp = self.client.post(
                f"/api/projects/{project['id']}/auto-rag/comparison/run",
                params={"corpus": "documents", "split": "val", "experiment_id": older},
            )
            self.assertEqual(resp.status_code, 202, resp.text)
            job = resp.json()
            wait_for_jobs_idle(self.client)
        self.assertEqual(job["params"]["experiment_id"], older)
        self.assertIn(f"run #{older}", job["title"])
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["experiment_id"], calls[0]["split"], calls[0]["base_only"]), (older, "val", False))
        done = self.client.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(done["status"], "succeeded", done)
        self.assertEqual(done["result"]["experiment_id"], older)

    # The unpatched "202 success-shape" path isn't unit-tested here because the
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

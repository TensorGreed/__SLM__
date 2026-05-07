"""Phase 79 — canonical RunEvent schema (priority.md P31, Wave G).

Covers the foundation + the per-stage emit hooks:

- Direct emit + retrieve via ``run_event_service``: stage / severity /
  reason_code / parent_run_id round-trip.
- ``GET /api/projects/{id}/run-events`` with the documented filter set
  (run_id / parent_run_id / stage / severity / since / limit).
- ``GET /api/run-events/run/{run_id}`` returns events for a run in
  oldest-first order, suitable for a per-experiment drill-in.
- 404 ``project_not_found`` for unknown projects, 400
  ``invalid_stage`` / ``invalid_severity`` / ``invalid_window`` for bad
  input.
- Hook coverage: a non-dry-run deploy execute emits a
  ``stage=deployment, action=record`` event; promote then emits
  ``action=promote``; rollback emits both ``rollback`` and
  ``re_promote`` referencing each other; ingest_file emits a
  ``stage=ingestion`` event.
- Emit-failure isolation: inject a transient session error inside
  ``emit_event`` and confirm the calling stage operation still
  succeeds (the calling code wraps emit in try/except).
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase79_run_events.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase79_run_events_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"
os.environ["ALLOW_SYNTHETIC_DEMO_FALLBACK"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.run_event import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    STAGE_DEPLOYMENT,
    STAGE_TRAINING,
)
from app.services.run_event_service import emit_event, list_run_events


def _cleanup_artifacts() -> None:
    if TEST_DATA_DIR.exists():
        for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
            if path.is_file():
                try:
                    path.unlink()
                except PermissionError:
                    pass
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    for suffix in ("", "-shm", "-wal"):
        path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass


class Phase79RunEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        _cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        _cleanup_artifacts()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_project(self, name: str = "phase79") -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"{name}-{uuid.uuid4().hex[:8]}",
                "description": "phase79",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_completed_export(self, project_id: int) -> int:
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase79-exp",
                "description": "phase79",
                "config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        exp_payload = exp_resp.json()
        experiment_id = int(exp_payload["id"])
        output_dir = Path(str(exp_payload.get("output_dir") or ""))
        self.assertTrue(output_dir.exists(), output_dir)
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            '{"model_type":"phi"}', encoding="utf-8"
        )
        (model_dir / "weights.safetensors").write_bytes(b"phase79-model")
        (model_dir / "tokenizer.json").write_text(
            '{"version":1}', encoding="utf-8"
        )
        create_resp = self.client.post(
            f"/api/projects/{project_id}/export/create",
            json={
                "experiment_id": experiment_id,
                "export_format": "huggingface",
                "quantization": "none",
            },
        )
        self.assertEqual(create_resp.status_code, 201, create_resp.text)
        export_id = int(create_resp.json()["id"])
        run_resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/run",
            json={
                "deployment_targets": ["exporter.huggingface"],
                "run_smoke_tests": False,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        return export_id

    def _execute_deploy(
        self, project_id: int, export_id: int
    ) -> int:
        resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/deploy-as-api/execute",
            json={"target_id": "sdk.apple_coreml_stub", "dry_run": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return int(resp.json()["deployment_version_id"])

    # ------------------------------------------------------------------
    # Direct service tests
    # ------------------------------------------------------------------

    def test_emit_event_persists_and_round_trips(self):
        project_id = self._create_project("emit")

        async def _go():
            async with async_session_factory() as db:
                row = await emit_event(
                    db,
                    project_id=project_id,
                    run_id="exp-1",
                    parent_run_id="autopilot-abc",
                    stage=STAGE_TRAINING,
                    severity=SEVERITY_INFO,
                    summary="hello",
                    actor="alice",
                    payload={"foo": "bar"},
                )
                await db.commit()
                return row.id

        row_id = asyncio.run(_go())
        self.assertGreater(row_id, 0)

        resp = self.client.get(
            f"/api/projects/{project_id}/run-events"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(len(body["events"]), 1)
        event = body["events"][0]
        self.assertEqual(event["run_id"], "exp-1")
        self.assertEqual(event["parent_run_id"], "autopilot-abc")
        self.assertEqual(event["stage"], "training")
        self.assertEqual(event["severity"], "info")
        self.assertEqual(event["summary"], "hello")
        self.assertEqual(event["actor"], "alice")
        self.assertEqual(event["payload"], {"foo": "bar"})

    def test_emit_event_rejects_unknown_stage_and_severity(self):
        project_id = self._create_project("reject")

        async def _go(stage: str, severity: str):
            async with async_session_factory() as db:
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="r",
                    stage=stage,
                    severity=severity,
                )

        with self.assertRaises(ValueError) as cm:
            asyncio.run(_go("not-a-stage", SEVERITY_INFO))
        self.assertIn("invalid_stage", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            asyncio.run(_go(STAGE_TRAINING, "not-a-severity"))
        self.assertIn("invalid_severity", str(cm.exception))

    # ------------------------------------------------------------------
    # API filter tests
    # ------------------------------------------------------------------

    def test_list_filters_stage_and_severity(self):
        project_id = self._create_project("filters")

        async def _seed():
            async with async_session_factory() as db:
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="exp-10",
                    stage=STAGE_TRAINING,
                    severity=SEVERITY_INFO,
                    summary="train info",
                )
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="exp-10",
                    stage=STAGE_TRAINING,
                    severity=SEVERITY_ERROR,
                    reason_code="training_runtime_error",
                    summary="train error",
                )
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="deploy-5",
                    stage=STAGE_DEPLOYMENT,
                    severity=SEVERITY_INFO,
                    summary="deploy info",
                )
                await db.commit()

        asyncio.run(_seed())

        # By stage
        resp = self.client.get(
            f"/api/projects/{project_id}/run-events",
            params={"stage": "training"},
        )
        names = sorted(e["summary"] for e in resp.json()["events"])
        self.assertEqual(names, ["train error", "train info"])

        # By severity
        resp = self.client.get(
            f"/api/projects/{project_id}/run-events",
            params={"severity": "error"},
        )
        events = resp.json()["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reason_code"], "training_runtime_error")

        # By run_id
        resp = self.client.get(
            f"/api/projects/{project_id}/run-events",
            params={"run_id": "deploy-5"},
        )
        self.assertEqual(len(resp.json()["events"]), 1)

    def test_per_run_endpoint_returns_oldest_first(self):
        project_id = self._create_project("perrun")

        async def _seed():
            async with async_session_factory() as db:
                base = datetime.now(timezone.utc)
                for offset, summary in (
                    (-30, "first"),
                    (-20, "second"),
                    (-10, "third"),
                ):
                    await emit_event(
                        db,
                        project_id=project_id,
                        run_id="exp-22",
                        stage=STAGE_TRAINING,
                        summary=summary,
                        ts=base + timedelta(seconds=offset),
                    )
                await db.commit()

        asyncio.run(_seed())

        resp = self.client.get("/api/run-events/run/exp-22")
        self.assertEqual(resp.status_code, 200)
        summaries = [e["summary"] for e in resp.json()["events"]]
        self.assertEqual(summaries, ["first", "second", "third"])

    def test_invalid_stage_filter_400(self):
        project_id = self._create_project("badstage")
        resp = self.client.get(
            f"/api/projects/{project_id}/run-events",
            params={"stage": "not-real"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("invalid_stage", resp.json()["detail"])

    def test_invalid_severity_filter_400(self):
        project_id = self._create_project("badsev")
        resp = self.client.get(
            f"/api/projects/{project_id}/run-events",
            params={"severity": "not-real"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("invalid_severity", resp.json()["detail"])

    def test_invalid_window_400(self):
        project_id = self._create_project("badwin")
        now = datetime.now(timezone.utc)
        resp = self.client.get(
            f"/api/projects/{project_id}/run-events",
            params={
                "since": now.isoformat(),
                "until": (now - timedelta(hours=1)).isoformat(),
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("invalid_window", resp.json()["detail"])

    def test_unknown_project_404(self):
        resp = self.client.get("/api/projects/999999/run-events")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "project_not_found")

    # ------------------------------------------------------------------
    # Stage hook coverage
    # ------------------------------------------------------------------

    def test_deployment_record_promote_emits_run_events(self):
        project_id = self._create_project("dvevents")
        export_id = self._create_completed_export(project_id)
        dv_id = self._execute_deploy(project_id, export_id)

        # Promote should emit a second event.
        promote_resp = self.client.post(
            f"/api/deployments/{dv_id}/promote", json={"reason": "ready"}
        )
        self.assertEqual(promote_resp.status_code, 200, promote_resp.text)

        events_resp = self.client.get(
            f"/api/run-events/run/deploy-{dv_id}"
        )
        self.assertEqual(events_resp.status_code, 200)
        events = events_resp.json()["events"]
        actions = [e["payload"].get("action") for e in events]
        # At minimum we should see the record + promote actions, in order.
        self.assertIn("record", actions)
        self.assertIn("promote", actions)
        self.assertLess(actions.index("record"), actions.index("promote"))

        for event in events:
            self.assertEqual(event["stage"], "deployment")

    def test_rollback_emits_paired_events(self):
        project_id = self._create_project("rbevents")
        export_id = self._create_completed_export(project_id)
        v1_id = self._execute_deploy(project_id, export_id)
        self.assertEqual(
            self.client.post(
                f"/api/deployments/{v1_id}/promote", json={}
            ).status_code,
            200,
        )
        v2_id = self._execute_deploy(project_id, export_id)
        self.assertEqual(
            self.client.post(
                f"/api/deployments/{v2_id}/promote", json={}
            ).status_code,
            200,
        )

        rb = self.client.post(
            f"/api/deployments/{v2_id}/rollback",
            json={"reason": "regression"},
        )
        self.assertEqual(rb.status_code, 200, rb.text)

        v2_events = self.client.get(
            f"/api/run-events/run/deploy-{v2_id}"
        ).json()["events"]
        v1_events = self.client.get(
            f"/api/run-events/run/deploy-{v1_id}"
        ).json()["events"]

        self.assertTrue(
            any(
                e["payload"].get("action") == "rollback"
                and e["payload"].get("rolled_back_to_id") == v1_id
                for e in v2_events
            ),
            f"v2 events: {v2_events}",
        )
        self.assertTrue(
            any(
                e["payload"].get("action") == "re_promote"
                and e["payload"].get("rollback_source_id") == v2_id
                for e in v1_events
            ),
            f"v1 events: {v1_events}",
        )

    def test_ingest_emits_run_event(self):
        project_id = self._create_project("ingest")
        upload = self.client.post(
            f"/api/projects/{project_id}/ingestion/upload",
            files={
                "file": (
                    "phase79.txt",
                    b"hello phase 79 ingestion test",
                    "text/plain",
                ),
            },
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        document_id = int(upload.json()["id"])

        events = self.client.get(
            f"/api/run-events/run/doc-{document_id}"
        ).json()["events"]
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["stage"], "ingestion")
        self.assertEqual(events[0]["payload"]["document_id"], document_id)

    # ------------------------------------------------------------------
    # Emit-failure isolation
    # ------------------------------------------------------------------

    def test_emit_failure_does_not_break_calling_stage(self):
        """A failing emit in a stage hook must not poison the operation.

        The deployment helper ``_emit_deployment_event`` wraps the
        canonical ``emit_event`` call in try/except. We patch the inner
        ``emit_event`` to raise so the helper's catch is exercised — if
        the helper ever loses its wrapper, this test will fail because
        the promote call will surface the simulated error.
        """
        project_id = self._create_project("isolation")
        export_id = self._create_completed_export(project_id)

        from app.services import run_event_service

        original = run_event_service.emit_event

        async def _boom(*args, **kwargs):  # noqa: ARG001
            raise RuntimeError("simulated transient")

        run_event_service.emit_event = _boom  # type: ignore[assignment]
        try:
            dv_id = self._execute_deploy(project_id, export_id)
            promote_resp = self.client.post(
                f"/api/deployments/{dv_id}/promote", json={"reason": "ok"}
            )
            self.assertEqual(promote_resp.status_code, 200, promote_resp.text)
            self.assertEqual(
                promote_resp.json()["deployment_version"]["status"],
                "promoted",
            )
        finally:
            run_event_service.emit_event = original


if __name__ == "__main__":
    unittest.main()

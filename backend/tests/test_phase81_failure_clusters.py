"""Phase 81 — reason-code taxonomy + failure clustering (priority.md P33).

Two surfaces under test:

1. **Reason-code lint enforcement** in
   :func:`app.services.run_event_service.emit_event`.
   - severity=info accepts free-form (or absent) reason_code.
   - severity=error / critical require a code from
     :data:`app.models.reason_codes.KNOWN_REASON_CODES`.
   - Missing → ``reason_code_required`` (400).
   - Unknown → ``invalid_reason_code:<value>`` (400).

2. **Failure clustering** via
   :mod:`app.services.run_event_clustering_service` and its API:
   - ``compute_failure_clusters`` is idempotent — a second compute on
     the same event log produces the same set of cluster rows with
     the same counts.
   - Two error events with the same (stage, reason_code) but
     different summaries collapse into a **single** cluster when
     their summaries differ only in step/line/hash (signature
     normalisation).
   - Two error events with truly different summaries become **two**
     clusters under the same (stage, reason_code).
   - ``GET /api/projects/{id}/failure-clusters`` returns clusters
     ordered by ``failure_count`` DESC, supports stage / reason_code
     filters.
   - ``POST /api/projects/{id}/failure-clusters/recompute`` returns
     the compute summary; new events between recomputes update
     existing cluster rows in place.
   - 404 ``project_not_found`` for unknown projects on both routes.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase81_failure_clusters.db"
TEST_DATA_DIR = (
    Path(__file__).resolve().parent / "phase81_failure_clusters_data"
)

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.reason_codes import (
    EXPORT_RUN_FAILED,
    TRAINING_DISPATCH_ERROR,
    TRAINING_RUNTIME_ERROR,
)
from app.models.run_event import (
    SEVERITY_CRITICAL,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    STAGE_EXPORT,
    STAGE_TRAINING,
)
from app.services.run_event_clustering_service import compute_signature
from app.services.run_event_service import emit_event


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


class Phase81FailureClusterTests(unittest.TestCase):
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

    def _create_project(self, name: str = "phase81") -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"{name}-{uuid.uuid4().hex[:8]}",
                "description": "phase81",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_error_events(
        self,
        project_id: int,
        events: list[dict],
    ) -> None:
        async def _runner():
            async with async_session_factory() as db:
                base = datetime.now(timezone.utc) - timedelta(hours=1)
                for index, entry in enumerate(events):
                    await emit_event(
                        db,
                        project_id=project_id,
                        run_id=entry.get("run_id", f"r-{index}"),
                        stage=entry["stage"],
                        severity=entry.get("severity", SEVERITY_ERROR),
                        reason_code=entry["reason_code"],
                        summary=entry["summary"],
                        ts=base + timedelta(seconds=entry.get("ts_offset", index)),
                    )
                await db.commit()

        asyncio.run(_runner())

    # ------------------------------------------------------------------
    # 1. Reason-code lint enforcement
    # ------------------------------------------------------------------

    def test_emit_info_severity_does_not_require_reason_code(self):
        project_id = self._create_project("lintinfo")

        async def _go():
            async with async_session_factory() as db:
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="r-1",
                    stage=STAGE_TRAINING,
                    severity=SEVERITY_INFO,
                    summary="hello",
                )
                await db.commit()

        # Must not raise.
        asyncio.run(_go())

    def test_emit_error_without_reason_code_raises(self):
        project_id = self._create_project("lintmissing")

        async def _go():
            async with async_session_factory() as db:
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="r-1",
                    stage=STAGE_TRAINING,
                    severity=SEVERITY_ERROR,
                    summary="boom",
                )

        with self.assertRaises(ValueError) as cm:
            asyncio.run(_go())
        self.assertEqual(str(cm.exception), "reason_code_required")

    def test_emit_critical_without_reason_code_raises(self):
        project_id = self._create_project("lintcrit")

        async def _go():
            async with async_session_factory() as db:
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="r-1",
                    stage=STAGE_TRAINING,
                    severity=SEVERITY_CRITICAL,
                    summary="boom",
                )

        with self.assertRaises(ValueError) as cm:
            asyncio.run(_go())
        self.assertEqual(str(cm.exception), "reason_code_required")

    def test_emit_error_with_unknown_reason_code_raises(self):
        project_id = self._create_project("lintunknown")

        async def _go():
            async with async_session_factory() as db:
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="r-1",
                    stage=STAGE_TRAINING,
                    severity=SEVERITY_ERROR,
                    reason_code="not_in_taxonomy_garbage",
                    summary="boom",
                )

        with self.assertRaises(ValueError) as cm:
            asyncio.run(_go())
        self.assertIn("invalid_reason_code", str(cm.exception))
        self.assertIn("not_in_taxonomy_garbage", str(cm.exception))

    def test_emit_error_with_known_reason_code_succeeds(self):
        project_id = self._create_project("lintok")

        async def _go():
            async with async_session_factory() as db:
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="r-1",
                    stage=STAGE_TRAINING,
                    severity=SEVERITY_ERROR,
                    reason_code=TRAINING_RUNTIME_ERROR,
                    summary="oom at step 1200",
                )
                await db.commit()

        asyncio.run(_go())

    # ------------------------------------------------------------------
    # 2. Failure clustering
    # ------------------------------------------------------------------

    def test_signature_collapses_step_numbered_summaries(self):
        sig_a = compute_signature(
            stage=STAGE_TRAINING,
            reason_code=TRAINING_RUNTIME_ERROR,
            summary="CUDA OOM at step 1200",
        )
        sig_b = compute_signature(
            stage=STAGE_TRAINING,
            reason_code=TRAINING_RUNTIME_ERROR,
            summary="CUDA OOM at step 4500",
        )
        self.assertEqual(sig_a, sig_b)
        # But a meaningfully different summary gets a different signature.
        sig_c = compute_signature(
            stage=STAGE_TRAINING,
            reason_code=TRAINING_RUNTIME_ERROR,
            summary="dataloader corrupted",
        )
        self.assertNotEqual(sig_a, sig_c)

    def test_same_signature_collapses_into_one_cluster(self):
        project_id = self._create_project("collapse")
        self._seed_error_events(
            project_id,
            [
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "CUDA OOM at step 1200",
                },
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "CUDA OOM at step 4500",
                },
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "CUDA OOM at step 9876",
                },
            ],
        )
        recompute = self.client.post(
            f"/api/projects/{project_id}/failure-clusters/recompute",
            json={},
        )
        self.assertEqual(recompute.status_code, 200, recompute.text)
        body = recompute.json()
        self.assertEqual(body["clusters_created"], 1)
        self.assertEqual(body["clusters_total"], 1)

        clusters = self.client.get(
            f"/api/projects/{project_id}/failure-clusters"
        ).json()["clusters"]
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["failure_count"], 3)
        self.assertEqual(clusters[0]["reason_code"], TRAINING_RUNTIME_ERROR)
        self.assertLessEqual(
            len(clusters[0]["exemplar_event_ids"]), 5
        )
        # P36 deep-link support: exemplar_run_ids array runs parallel
        # to exemplar_event_ids so the failure-analysis UI can deep-link
        # without a chained per-event lookup.
        self.assertEqual(
            len(clusters[0]["exemplar_run_ids"]),
            len(clusters[0]["exemplar_event_ids"]),
        )
        # All seeded events shared run_id "r-0" / "r-1" / "r-2" via
        # the helper's default; just check the field is populated.
        self.assertTrue(
            all(rid for rid in clusters[0]["exemplar_run_ids"])
        )

    def test_distinct_summaries_yield_separate_clusters(self):
        project_id = self._create_project("distinct")
        self._seed_error_events(
            project_id,
            [
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "CUDA OOM at step 1200",
                },
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "dataloader corrupted",
                },
            ],
        )
        self.client.post(
            f"/api/projects/{project_id}/failure-clusters/recompute",
            json={},
        )
        clusters = self.client.get(
            f"/api/projects/{project_id}/failure-clusters"
        ).json()["clusters"]
        self.assertEqual(len(clusters), 2)
        signatures = {c["signature"] for c in clusters}
        self.assertEqual(len(signatures), 2)

    def test_distinct_reason_codes_yield_separate_clusters(self):
        project_id = self._create_project("multireason")
        self._seed_error_events(
            project_id,
            [
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "boom",
                },
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_DISPATCH_ERROR,
                    "summary": "boom",
                },
            ],
        )
        self.client.post(
            f"/api/projects/{project_id}/failure-clusters/recompute",
            json={},
        )
        clusters = self.client.get(
            f"/api/projects/{project_id}/failure-clusters"
        ).json()["clusters"]
        self.assertEqual(len(clusters), 2)
        codes = sorted(c["reason_code"] for c in clusters)
        self.assertEqual(
            codes, sorted([TRAINING_DISPATCH_ERROR, TRAINING_RUNTIME_ERROR])
        )

    def test_recompute_is_idempotent(self):
        project_id = self._create_project("idempotent")
        self._seed_error_events(
            project_id,
            [
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "OOM at step 1",
                },
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "OOM at step 2",
                },
            ],
        )
        first = self.client.post(
            f"/api/projects/{project_id}/failure-clusters/recompute",
            json={},
        ).json()
        self.assertEqual(first["clusters_created"], 1)
        self.assertEqual(first["clusters_updated"], 0)

        second = self.client.post(
            f"/api/projects/{project_id}/failure-clusters/recompute",
            json={},
        ).json()
        self.assertEqual(second["clusters_created"], 0)
        self.assertEqual(second["clusters_updated"], 1)

        clusters = self.client.get(
            f"/api/projects/{project_id}/failure-clusters"
        ).json()["clusters"]
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["failure_count"], 2)

    def test_new_events_between_recomputes_update_count(self):
        project_id = self._create_project("growing")
        self._seed_error_events(
            project_id,
            [
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "OOM at step 1",
                    "ts_offset": 0,
                },
            ],
        )
        first = self.client.post(
            f"/api/projects/{project_id}/failure-clusters/recompute",
            json={},
        ).json()
        self.assertEqual(first["clusters_total"], 1)

        # Inject more failures and recompute.
        self._seed_error_events(
            project_id,
            [
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "OOM at step 99",
                    "ts_offset": 60,
                },
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "OOM at step 200",
                    "ts_offset": 120,
                },
            ],
        )
        second = self.client.post(
            f"/api/projects/{project_id}/failure-clusters/recompute",
            json={},
        ).json()
        self.assertEqual(second["clusters_total"], 1)
        self.assertEqual(second["clusters_updated"], 1)

        clusters = self.client.get(
            f"/api/projects/{project_id}/failure-clusters"
        ).json()["clusters"]
        self.assertEqual(clusters[0]["failure_count"], 3)

    def test_list_orders_by_failure_count_desc(self):
        project_id = self._create_project("ranking")
        # Bigger cluster on training_runtime_error, smaller on dispatch.
        self._seed_error_events(
            project_id,
            [
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "OOM at step 1",
                },
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "OOM at step 2",
                },
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "OOM at step 3",
                },
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_DISPATCH_ERROR,
                    "summary": "kicked off but dispatch died",
                },
            ],
        )
        self.client.post(
            f"/api/projects/{project_id}/failure-clusters/recompute",
            json={},
        )
        clusters = self.client.get(
            f"/api/projects/{project_id}/failure-clusters"
        ).json()["clusters"]
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0]["reason_code"], TRAINING_RUNTIME_ERROR)
        self.assertEqual(clusters[0]["failure_count"], 3)
        self.assertEqual(clusters[1]["reason_code"], TRAINING_DISPATCH_ERROR)

    def test_list_filters_by_stage_and_reason_code(self):
        project_id = self._create_project("filter")
        self._seed_error_events(
            project_id,
            [
                {
                    "stage": STAGE_TRAINING,
                    "reason_code": TRAINING_RUNTIME_ERROR,
                    "summary": "boom",
                },
                {
                    "stage": STAGE_EXPORT,
                    "reason_code": EXPORT_RUN_FAILED,
                    "summary": "missing artifact",
                },
            ],
        )
        self.client.post(
            f"/api/projects/{project_id}/failure-clusters/recompute",
            json={},
        )
        export_only = self.client.get(
            f"/api/projects/{project_id}/failure-clusters",
            params={"stage": "export"},
        ).json()["clusters"]
        self.assertEqual(len(export_only), 1)
        self.assertEqual(export_only[0]["stage"], "export")

        by_code = self.client.get(
            f"/api/projects/{project_id}/failure-clusters",
            params={"reason_code": TRAINING_RUNTIME_ERROR},
        ).json()["clusters"]
        self.assertEqual(len(by_code), 1)
        self.assertEqual(by_code[0]["reason_code"], TRAINING_RUNTIME_ERROR)

    def test_list_unknown_project_404(self):
        resp = self.client.get("/api/projects/999999/failure-clusters")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "project_not_found")

    def test_recompute_unknown_project_404(self):
        resp = self.client.post(
            "/api/projects/999999/failure-clusters/recompute", json={}
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "project_not_found")


if __name__ == "__main__":
    unittest.main()

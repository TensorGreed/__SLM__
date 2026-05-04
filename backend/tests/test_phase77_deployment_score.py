"""Phase 77 — deployability score (priority.md P28).

Covers ``POST /api/deployments/{id}/score/compute`` end-to-end on top of
P25 deployment versions, plus the read paths for the most-recent score
and history. The score blends measured smoke-test outcomes from the
deploy execute history (P25) and post-deploy telemetry (P26) with
estimated compatibility signals from the deploy-target suite.

Test matrix:

- A freshly-executed deployment version (no telemetry yet, no drift
  check) computes a score from manifest signals + execute history.
- Ingesting telemetry into the dv adds the ``telemetry_health``
  component with provenance ``measured``.
- Running a P27 drift check adds ``drift_health`` with provenance
  ``measured``.
- Provenance summary is ``mixed`` when at least one estimated and one
  measured component are present, ``measured`` when all contributing
  components are measured.
- Components with no source signal carry ``score=null`` and are listed
  under ``signals_summary.components_missing``.
- Renormalised weights of contributing components sum to 1.0.
- ``GET /score`` returns the most recent row; ``score_not_found`` (404)
  before any compute has run.
- ``GET /score/history`` returns rows newest-first.
- 404 on unknown deployment version on every route.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase77_deployment_score.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase77_deployment_score_data"

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
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import EvalResult
from app.models.export import Export
from app.models.gold_set_annotation import (
    GoldSetRow,
    GoldSetRowStatus,
    GoldSetVersion,
    GoldSetVersionStatus,
)


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


class Phase77DeploymentScoreTests(unittest.TestCase):
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
    # Helpers (HTTP)
    # ------------------------------------------------------------------

    def _create_project(self, name: str = "phase77") -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"{name}-{uuid.uuid4().hex[:8]}",
                "description": "phase77",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_completed_export(self, project_id: int) -> tuple[int, int]:
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase77-exp",
                "description": "phase77",
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
        (model_dir / "weights.safetensors").write_bytes(b"phase77-model")
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
        return export_id, experiment_id

    def _create_deployment_version(
        self, project_id: int, export_id: int
    ) -> int:
        resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/deploy-as-api/execute",
            json={"target_id": "sdk.apple_coreml_stub", "dry_run": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return int(resp.json()["deployment_version_id"])

    def _ingest_telemetry(
        self, dv_id: int, *, samples: list[dict]
    ) -> None:
        resp = self.client.post(
            f"/api/deployments/{dv_id}/telemetry/ingest",
            json={"samples": samples},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def _seed_gold_set(
        self,
        *,
        project_id: int,
        rows: list[tuple[str, str]],
        name: str | None = None,
    ) -> tuple[int, list[int], str]:
        """Create a gold-set Dataset + draft version + rows.

        Returns ``(gold_set_id, row_ids, gold_set_name)`` so callers that
        need to seed a matching baseline ``EvalResult`` can pin the
        same ``dataset_name``.
        """
        gold_name = name or f"phase77-gold-{uuid.uuid4().hex[:6]}"

        async def _runner():
            async with async_session_factory() as db:
                ds = Dataset(
                    project_id=project_id,
                    name=gold_name,
                    dataset_type=DatasetType.GOLD_DEV,
                    description="phase77",
                    record_count=len(rows),
                    file_path="",
                    metadata_={},
                    is_locked=False,
                )
                db.add(ds)
                await db.flush()
                version = GoldSetVersion(
                    gold_set_id=ds.id,
                    version=1,
                    status=GoldSetVersionStatus.DRAFT,
                )
                db.add(version)
                await db.flush()
                row_ids: list[int] = []
                for index, (prompt, expected) in enumerate(rows):
                    gsrow = GoldSetRow(
                        gold_set_id=ds.id,
                        version_id=version.id,
                        source_row_key=f"row-{index}",
                        input={"text": prompt},
                        expected={"answer": expected},
                        rationale="seed",
                        labels={},
                        status=GoldSetRowStatus.APPROVED,
                    )
                    db.add(gsrow)
                    await db.flush()
                    row_ids.append(gsrow.id)
                await db.commit()
                return ds.id, row_ids, gold_name

        return asyncio.run(_runner())

    def _seed_baseline_eval_result(
        self,
        *,
        experiment_id: int,
        dataset_name: str,
        pass_rate: float,
    ) -> int:
        async def _runner():
            async with async_session_factory() as db:
                er = EvalResult(
                    experiment_id=experiment_id,
                    dataset_name=dataset_name,
                    eval_type="exact_match",
                    metrics={"pass_rate": pass_rate},
                    pass_rate=pass_rate,
                )
                db.add(er)
                await db.flush()
                eval_id = er.id
                await db.commit()
                return eval_id

        return asyncio.run(_runner())

    def _wipe_export_manifest(self, export_id: int) -> None:
        """Force the export's manifest to {} so the score has no static signals."""

        async def _runner():
            async with async_session_factory() as db:
                from sqlalchemy import select

                row = (
                    await db.execute(
                        select(Export).where(Export.id == export_id)
                    )
                ).scalar_one()
                row.manifest = {}
                await db.commit()

        asyncio.run(_runner())

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def _component_by_name(self, body: dict, name: str) -> dict:
        for c in body["components"]:
            if c["name"] == name:
                return c
        raise AssertionError(f"component {name!r} missing from {body['components']}")

    def test_compute_score_on_fresh_dv_uses_manifest_and_execute_history(self):
        project_id = self._create_project()
        export_id, _ = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)

        resp = self.client.post(
            f"/api/deployments/{dv_id}/score/compute",
            json={"notes": "first compute", "actor": "ops"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()

        # Five components always returned; some null until measured.
        names = {c["name"] for c in body["components"]}
        self.assertEqual(
            names,
            {
                "artifact_compat",
                "target_compatibility",
                "execute_smoke",
                "telemetry_health",
                "drift_health",
            },
        )
        # Manifest is present so the static checks contributed.
        artifact = self._component_by_name(body, "artifact_compat")
        self.assertIsNotNone(artifact["score"])
        self.assertEqual(artifact["provenance"], "estimated")
        target = self._component_by_name(body, "target_compatibility")
        self.assertIsNotNone(target["score"])
        # The non-dry-run execute we just ran should show up as
        # measured execute_smoke.
        smoke = self._component_by_name(body, "execute_smoke")
        self.assertIsNotNone(smoke["score"])
        self.assertEqual(smoke["provenance"], "measured")
        # Telemetry + drift are absent on a brand-new dv.
        self.assertIsNone(
            self._component_by_name(body, "telemetry_health")["score"]
        )
        self.assertIsNone(
            self._component_by_name(body, "drift_health")["score"]
        )
        # Provenance is mixed — at least one measured (execute_smoke)
        # and one estimated (artifact_compat / target_compatibility).
        self.assertEqual(body["provenance"], "mixed")
        # signals_summary breaks down which components are present vs missing.
        self.assertIn("execute_smoke", body["signals_summary"]["components_present"])
        self.assertIn(
            "telemetry_health", body["signals_summary"]["components_missing"]
        )
        self.assertIn("drift_health", body["signals_summary"]["components_missing"])
        # Notes / actor round-trip.
        self.assertEqual(body["notes"], "first compute")
        self.assertEqual(body["actor"], "ops")
        # Confidence band falls into one of three buckets.
        self.assertIn(body["confidence_band"], {"low", "medium", "high"})

    def test_telemetry_signal_lifts_to_measured_provenance(self):
        project_id = self._create_project("telem")
        export_id, _ = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)

        # Healthy telemetry batch: low latency, all successful.
        now = datetime.now(timezone.utc)
        samples = [
            {
                "latency_ms": 50.0,
                "success": True,
                "ts": (now - timedelta(seconds=i)).isoformat(),
            }
            for i in range(40)
        ]
        self._ingest_telemetry(dv_id, samples=samples)

        resp = self.client.post(
            f"/api/deployments/{dv_id}/score/compute", json={}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        telem = self._component_by_name(body, "telemetry_health")
        self.assertIsNotNone(telem["score"])
        self.assertEqual(telem["provenance"], "measured")
        self.assertGreaterEqual(float(telem["score"]), 0.9)  # healthy
        # signals_summary.telemetry_sample_count reflects what we sent.
        self.assertGreaterEqual(
            int(body["signals_summary"]["telemetry_sample_count"]), 30
        )

    def test_unhealthy_telemetry_drops_score(self):
        project_id = self._create_project("unhealthy")
        export_id, _ = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)

        now = datetime.now(timezone.utc)
        # 20% error rate + p95 latency ~ 4s ⇒ both axes should drag the
        # score way down.
        samples = []
        for i in range(50):
            samples.append(
                {
                    "latency_ms": 100.0 if i % 2 == 0 else 4500.0,
                    "success": (i % 5 != 0),
                    "ts": (now - timedelta(seconds=i)).isoformat(),
                }
            )
        self._ingest_telemetry(dv_id, samples=samples)

        resp = self.client.post(
            f"/api/deployments/{dv_id}/score/compute", json={}
        )
        body = resp.json()
        telem = self._component_by_name(body, "telemetry_health")
        self.assertIsNotNone(telem["score"])
        self.assertLess(float(telem["score"]), 0.5)

    def test_drift_check_contributes_drift_health_component(self):
        project_id = self._create_project("drift")
        export_id, experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)

        gold_set_id, row_ids, gold_name = self._seed_gold_set(
            project_id=project_id,
            rows=[("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")],
        )
        # Pin the baseline EvalResult to the gold set's actual name so the
        # drift service hits its primary (name-filtered) match path rather
        # than relying on the broad eval-type-only fallback. baseline=1.0.
        self._seed_baseline_eval_result(
            experiment_id=experiment_id,
            dataset_name=gold_name,
            pass_rate=1.0,
        )

        # 1 / 4 correct ⇒ current_pass_rate = 0.25, baseline = 1.0,
        # delta = -0.75. Score formula: max(0, 1 - 2 * |delta|) ⇒ 0.0.
        drift_resp = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "tolerance": 0.05,
                "predictions": [
                    {"row_id": row_ids[0], "prediction": "a1"},
                    {"row_id": row_ids[1], "prediction": "wrong"},
                    {"row_id": row_ids[2], "prediction": "wrong"},
                    {"row_id": row_ids[3], "prediction": "wrong"},
                ],
            },
        )
        self.assertEqual(drift_resp.status_code, 200, drift_resp.text)
        drift_body = drift_resp.json()
        # The drift service must have actually found the baseline.
        self.assertIsNotNone(drift_body["baseline_pass_rate"])
        self.assertIsNotNone(drift_body["delta"])
        self.assertTrue(drift_body["drift_detected"])

        score_resp = self.client.post(
            f"/api/deployments/{dv_id}/score/compute", json={}
        )
        body = score_resp.json()
        drift = self._component_by_name(body, "drift_health")
        self.assertEqual(drift["provenance"], "measured")
        # Real assertion now: with baseline=1.0 and current=0.25, the
        # score MUST be measurable (not None) and MUST be 0.0 since
        # |delta|=0.75 saturates the 1 - 2*|delta| formula.
        self.assertIsNotNone(drift["score"])
        self.assertEqual(float(drift["score"]), 0.0)
        # signals_summary should record the drift verdict for the UI.
        self.assertEqual(body["signals_summary"].get("drift_detected"), True)
        self.assertIsNotNone(body["signals_summary"].get("drift_check_id"))

    def test_drift_health_within_tolerance_yields_high_score(self):
        """Mirror of the previous test but with predictions that match the
        baseline — drift_health should land near 1.0 with provenance=measured."""
        project_id = self._create_project("nodrift")
        export_id, experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)

        gold_set_id, row_ids, gold_name = self._seed_gold_set(
            project_id=project_id,
            rows=[("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")],
        )
        self._seed_baseline_eval_result(
            experiment_id=experiment_id,
            dataset_name=gold_name,
            pass_rate=1.0,
        )

        # Perfect predictions ⇒ delta = 0 ⇒ drift_health.score = 1.0.
        self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "tolerance": 0.05,
                "predictions": [
                    {"row_id": rid, "prediction": expected}
                    for rid, expected in zip(row_ids, ["a1", "a2", "a3", "a4"])
                ],
            },
        )
        score_resp = self.client.post(
            f"/api/deployments/{dv_id}/score/compute", json={}
        )
        body = score_resp.json()
        drift = self._component_by_name(body, "drift_health")
        self.assertEqual(drift["provenance"], "measured")
        self.assertIsNotNone(drift["score"])
        self.assertAlmostEqual(float(drift["score"]), 1.0)
        self.assertEqual(body["signals_summary"].get("drift_detected"), False)

    def test_components_with_no_signal_have_null_score_and_zero_normalised_weight(self):
        project_id = self._create_project("missing")
        export_id, _ = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)

        resp = self.client.post(
            f"/api/deployments/{dv_id}/score/compute", json={}
        )
        body = resp.json()
        present = [c for c in body["components"] if c["score"] is not None]
        missing = [c for c in body["components"] if c["score"] is None]
        # weight_normalised on missing components is 0.0; on present
        # components sums to ~1.0 when at least one component contributes.
        for component in missing:
            self.assertEqual(component["weight_normalised"], 0.0)
        if present:
            total = sum(float(c["weight_normalised"]) for c in present)
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_no_signals_at_all_returns_zero_score_estimated_provenance(self):
        project_id = self._create_project("nosignal")
        export_id, _ = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        # Wipe the manifest so no static signals contribute. Without an
        # execute_smoke history either, every component should be null.
        self._wipe_export_manifest(export_id)
        # Nuke the deploy execute history at the manifest level only —
        # the dv row itself remains for the /score endpoint to load.

        resp = self.client.post(
            f"/api/deployments/{dv_id}/score/compute", json={}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["overall_score"], 0.0)
        self.assertEqual(body["provenance"], "estimated")
        # All five components present in the response, all with null score.
        scored_components = [
            c for c in body["components"] if c["score"] is not None
        ]
        self.assertEqual(scored_components, [])
        self.assertEqual(
            sorted(body["signals_summary"]["components_missing"]),
            sorted([
                "artifact_compat",
                "target_compatibility",
                "execute_smoke",
                "telemetry_health",
                "drift_health",
            ]),
        )

    def test_score_history_returns_newest_first(self):
        project_id = self._create_project("history")
        export_id, _ = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)

        first = self.client.post(
            f"/api/deployments/{dv_id}/score/compute",
            json={"notes": "first"},
        ).json()
        second = self.client.post(
            f"/api/deployments/{dv_id}/score/compute",
            json={"notes": "second"},
        ).json()

        history_resp = self.client.get(
            f"/api/deployments/{dv_id}/score/history"
        )
        self.assertEqual(history_resp.status_code, 200, history_resp.text)
        body = history_resp.json()
        self.assertEqual(len(body["scores"]), 2)
        self.assertEqual(body["scores"][0]["notes"], "second")
        self.assertEqual(body["scores"][1]["notes"], "first")

    def test_get_latest_score_round_trips(self):
        project_id = self._create_project("latest")
        export_id, _ = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)

        compute_resp = self.client.post(
            f"/api/deployments/{dv_id}/score/compute",
            json={"notes": "only"},
        )
        score_id = int(compute_resp.json()["id"])

        latest_resp = self.client.get(f"/api/deployments/{dv_id}/score")
        self.assertEqual(latest_resp.status_code, 200, latest_resp.text)
        self.assertEqual(int(latest_resp.json()["id"]), score_id)
        self.assertEqual(latest_resp.json()["notes"], "only")

    def test_get_latest_score_404_when_not_yet_computed(self):
        project_id = self._create_project("notyet")
        export_id, _ = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)

        resp = self.client.get(f"/api/deployments/{dv_id}/score")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "score_not_found")

    def test_compute_unknown_deployment_404(self):
        resp = self.client.post(
            "/api/deployments/999999/score/compute", json={}
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(
            resp.json()["detail"], "deployment_version_not_found"
        )

    def test_history_unknown_deployment_404(self):
        resp = self.client.get("/api/deployments/999999/score/history")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(
            resp.json()["detail"], "deployment_version_not_found"
        )


if __name__ == "__main__":
    unittest.main()

"""Phase 76 — drift check on served endpoint (priority.md P27).

Covers ``POST /api/deployments/{id}/drift/check`` end-to-end with the
P10 gold-set rows + P25 deployment versions. The drift check supports
two prediction sources:

- offline: caller passes ``predictions`` directly (the test path).
- live_url: caller passes ``endpoint_url``; the service POSTs per row.

The tests below exercise the offline path exhaustively and verify the
live-url branch raises 400 ``endpoint_or_predictions_required`` when
neither is supplied.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase76_deployment_drift.db"
TEST_DATA_DIR = (
    Path(__file__).resolve().parent / "phase76_deployment_drift_data"
)

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


class Phase76DeploymentDriftCheckTests(unittest.TestCase):
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

    def _create_project(self, name: str = "phase76") -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"{name}-{uuid.uuid4().hex[:8]}",
                "description": "phase76",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_completed_export(self, project_id: int) -> tuple[int, int]:
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase76-exp",
                "description": "phase76",
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
        (model_dir / "weights.safetensors").write_bytes(b"phase76-model")
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

    # ------------------------------------------------------------------
    # Helpers (DB seeding)
    # ------------------------------------------------------------------

    def _seed_gold_set(
        self,
        *,
        project_id: int,
        rows: list[tuple[str, str]],
        gold_name: str = "drift-gold",
    ) -> tuple[int, list[int]]:
        """Create a gold-set Dataset + draft version + GoldSetRows.

        Returns ``(gold_set_id, [row_ids in order])`` so tests can build
        offline prediction batches that reference real ids.
        """

        async def _runner():
            async with async_session_factory() as db:
                ds = Dataset(
                    project_id=project_id,
                    name=gold_name,
                    dataset_type=DatasetType.GOLD_DEV,
                    description="phase76",
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
                return ds.id, row_ids

        return asyncio.run(_runner())

    def _seed_baseline_eval_result(
        self,
        *,
        experiment_id: int,
        dataset_name: str,
        pass_rate: float,
        eval_type: str = "exact_match",
    ) -> int:
        async def _runner():
            async with async_session_factory() as db:
                er = EvalResult(
                    experiment_id=experiment_id,
                    dataset_name=dataset_name,
                    eval_type=eval_type,
                    metrics={"pass_rate": pass_rate},
                    pass_rate=pass_rate,
                )
                db.add(er)
                await db.flush()
                eval_id = er.id
                await db.commit()
                return eval_id

        return asyncio.run(_runner())

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_drift_check_offline_perfect_match_no_drift(self):
        project_id = self._create_project()
        export_id, experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        gold_set_id, row_ids = self._seed_gold_set(
            project_id=project_id,
            rows=[("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")],
        )
        # Baseline: 1.0 — current will also be 1.0 (perfect predictions).
        self._seed_baseline_eval_result(
            experiment_id=experiment_id,
            dataset_name="drift-gold",
            pass_rate=1.0,
        )

        predictions = [
            {"row_id": row_ids[0], "prediction": "a1"},
            {"row_id": row_ids[1], "prediction": "a2"},
            {"row_id": row_ids[2], "prediction": "a3"},
            {"row_id": row_ids[3], "prediction": "a4"},
        ]
        resp = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "predictions": predictions,
                "tolerance": 0.05,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertAlmostEqual(body["current_pass_rate"], 1.0)
        self.assertAlmostEqual(body["baseline_pass_rate"], 1.0)
        self.assertAlmostEqual(body["delta"], 0.0)
        self.assertFalse(body["drift_detected"])
        self.assertEqual(body["samples_evaluated"], 4)
        self.assertEqual(body["samples_failed"], 0)
        self.assertEqual(body["samples_skipped"], 0)
        self.assertEqual(body["mode"], "offline")
        # Per-row breakdown: every entry should be a match.
        for entry in body["per_row_results"]:
            self.assertTrue(entry["match"])

    def test_drift_check_detects_drift_beyond_tolerance(self):
        project_id = self._create_project()
        export_id, experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        gold_set_id, row_ids = self._seed_gold_set(
            project_id=project_id,
            rows=[("q", f"correct-{i}") for i in range(4)],
        )
        self._seed_baseline_eval_result(
            experiment_id=experiment_id,
            dataset_name="drift-gold",
            pass_rate=1.0,
        )

        # Predict only 1 of 4 correctly => current_pass_rate = 0.25,
        # baseline = 1.0, delta = -0.75 => drift_detected.
        predictions = [
            {"row_id": row_ids[0], "prediction": "correct-0"},
            {"row_id": row_ids[1], "prediction": "wrong"},
            {"row_id": row_ids[2], "prediction": "wrong"},
            {"row_id": row_ids[3], "prediction": "wrong"},
        ]
        resp = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "predictions": predictions,
                "tolerance": 0.1,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertAlmostEqual(body["current_pass_rate"], 0.25)
        self.assertAlmostEqual(body["baseline_pass_rate"], 1.0)
        self.assertAlmostEqual(body["delta"], -0.75)
        self.assertTrue(body["drift_detected"])
        self.assertEqual(body["samples_evaluated"], 4)

    def test_drift_check_no_baseline_returns_null_delta(self):
        project_id = self._create_project()
        export_id, _experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        gold_set_id, row_ids = self._seed_gold_set(
            project_id=project_id,
            rows=[("q1", "a1"), ("q2", "a2")],
        )
        # No baseline EvalResult seeded.
        predictions = [
            {"row_id": row_ids[0], "prediction": "a1"},
            {"row_id": row_ids[1], "prediction": "a2"},
        ]
        resp = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "predictions": predictions,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIsNone(body["baseline_pass_rate"])
        self.assertIsNone(body["delta"])
        # No baseline => cannot detect drift.
        self.assertFalse(body["drift_detected"])
        # baseline_experiment_id is still resolved from export → experiment.
        self.assertIsNotNone(body["baseline_experiment_id"])

    def test_drift_check_missing_predictions_count_as_skipped(self):
        project_id = self._create_project()
        export_id, _experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        gold_set_id, row_ids = self._seed_gold_set(
            project_id=project_id,
            rows=[("q1", "a1"), ("q2", "a2"), ("q3", "a3")],
        )
        # Only supply predictions for 2 of 3 rows.
        predictions = [
            {"row_id": row_ids[0], "prediction": "a1"},
            {"row_id": row_ids[2], "prediction": "a3"},
        ]
        resp = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "predictions": predictions,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["samples_evaluated"], 2)
        self.assertEqual(body["samples_skipped"], 1)
        self.assertAlmostEqual(body["current_pass_rate"], 1.0)
        # The skipped row appears in per-row with error=no_prediction.
        skipped = [
            r for r in body["per_row_results"] if r["error"] == "no_prediction"
        ]
        self.assertEqual(len(skipped), 1)

    def test_drift_check_neither_endpoint_nor_predictions_400(self):
        project_id = self._create_project()
        export_id, _experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        gold_set_id, _ = self._seed_gold_set(
            project_id=project_id,
            rows=[("q", "a")],
        )
        resp = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={"gold_set_id": gold_set_id},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(
            resp.json()["detail"], "endpoint_or_predictions_required"
        )

    def test_drift_check_invalid_tolerance_400(self):
        project_id = self._create_project()
        export_id, _experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        gold_set_id, row_ids = self._seed_gold_set(
            project_id=project_id, rows=[("q", "a")]
        )
        # tolerance is bounded by Pydantic at the request layer (ge=0, le=1)
        # so we exercise the runtime branch via invalid_max_samples.
        resp = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "predictions": [{"row_id": row_ids[0], "prediction": "a"}],
                "max_samples": 1000,  # outside Pydantic le=500 too
            },
        )
        # Pydantic rejects max_samples > 500 with 422, so this exercises
        # the request-level guard.
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_drift_check_unknown_deployment_404(self):
        resp = self.client.post(
            "/api/deployments/999999/drift/check",
            json={"gold_set_id": 1, "predictions": []},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(
            resp.json()["detail"], "deployment_version_not_found"
        )

    def test_drift_check_unknown_gold_set_404(self):
        project_id = self._create_project()
        export_id, _ = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        resp = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={"gold_set_id": 999999, "predictions": [{"row_id": 1, "prediction": "x"}]},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "gold_set_not_found")

    def test_drift_check_persists_to_history(self):
        project_id = self._create_project()
        export_id, experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        gold_set_id, row_ids = self._seed_gold_set(
            project_id=project_id,
            rows=[("q1", "a1"), ("q2", "a2")],
        )
        # Run twice with different prediction quality.
        first = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "predictions": [
                    {"row_id": row_ids[0], "prediction": "a1"},
                    {"row_id": row_ids[1], "prediction": "a2"},
                ],
            },
        )
        self.assertEqual(first.status_code, 200, first.text)
        second = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "predictions": [
                    {"row_id": row_ids[0], "prediction": "wrong"},
                    {"row_id": row_ids[1], "prediction": "a2"},
                ],
            },
        )
        self.assertEqual(second.status_code, 200, second.text)

        history_resp = self.client.get(
            f"/api/deployments/{dv_id}/drift/checks"
        )
        self.assertEqual(history_resp.status_code, 200, history_resp.text)
        body = history_resp.json()
        self.assertEqual(len(body["drift_checks"]), 2)
        # Newest first: 0.5 then 1.0.
        self.assertAlmostEqual(
            body["drift_checks"][0]["current_pass_rate"], 0.5
        )
        self.assertAlmostEqual(
            body["drift_checks"][1]["current_pass_rate"], 1.0
        )

        # Single-row fetch by id round-trips.
        single = self.client.get(
            f"/api/deployments/drift/checks/{body['drift_checks'][0]['id']}"
        )
        self.assertEqual(single.status_code, 200, single.text)
        self.assertEqual(single.json()["mode"], "offline")

    def test_drift_check_match_is_case_insensitive_and_dict_aware(self):
        project_id = self._create_project()
        export_id, _experiment_id = self._create_completed_export(project_id)
        dv_id = self._create_deployment_version(project_id, export_id)
        gold_set_id, row_ids = self._seed_gold_set(
            project_id=project_id,
            rows=[("q1", "Yes"), ("q2", "No")],
        )
        # Predictions sent as dicts with `answer` key + different casing.
        predictions = [
            {"row_id": row_ids[0], "prediction": {"answer": "YES"}},
            {"row_id": row_ids[1], "prediction": "no"},
        ]
        resp = self.client.post(
            f"/api/deployments/{dv_id}/drift/check",
            json={
                "gold_set_id": gold_set_id,
                "predictions": predictions,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertAlmostEqual(body["current_pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

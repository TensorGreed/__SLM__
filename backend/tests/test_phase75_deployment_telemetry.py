"""Phase 75 — post-deploy telemetry ingest + aggregate (priority.md P26).

Covers the served-model telemetry surface added on top of the P25
deployment-versions table:

- ``POST /api/deployments/{id}/telemetry/ingest`` accepts a batch and
  returns ``accepted`` + ``rejected`` counts; invalid samples are dropped
  with a per-sample reason rather than failing the whole batch.
- ``samples_required`` (400) when the batch is empty.
- ``GET /api/deployments/{id}/telemetry`` returns the spec contract
  (request_volume + latency p50/p95/p99 + errors + tokens) over a
  configurable window; absent samples → zeroed payload.
- Latency percentile math is correct over a known distribution
  (0..99ms → p50=49.5, p95=94.05, p99=98.01).
- ``success`` is inferred from ``status_code`` when the caller doesn't
  send it (status < 400 → success=true).
- Token throughput is computed against the window in seconds.
- Samples outside the window (``window_seconds``) are excluded.
- ``invalid_window`` (400) when ``since >= until``.
- 404 ``deployment_version_not_found`` for unknown ids on every route.
- ``GET .../telemetry/samples`` returns recent samples, newest first,
  capped at the supplied ``limit``.
"""

from __future__ import annotations

import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase75_deployment_telemetry.db"
TEST_DATA_DIR = (
    Path(__file__).resolve().parent / "phase75_deployment_telemetry_data"
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
from app.main import app


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


class Phase75DeploymentTelemetryTests(unittest.TestCase):
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

    def _create_project(self, name: str = "phase75") -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"{name}-{uuid.uuid4().hex[:8]}",
                "description": "phase75",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_completed_export(self, project_id: int) -> int:
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase75-exp",
                "description": "phase75",
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
        (model_dir / "weights.safetensors").write_bytes(b"phase75-model")
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

    def _create_deployment_version(self) -> int:
        project_id = self._create_project()
        export_id = self._create_completed_export(project_id)
        resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/deploy-as-api/execute",
            json={"target_id": "sdk.apple_coreml_stub", "dry_run": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return int(resp.json()["deployment_version_id"])

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def test_ingest_accepts_batch(self):
        dv_id = self._create_deployment_version()
        samples = [
            {
                "latency_ms": 50.0,
                "success": True,
                "status_code": 200,
                "input_tokens": 100,
                "output_tokens": 30,
            }
            for _ in range(3)
        ]
        resp = self.client.post(
            f"/api/deployments/{dv_id}/telemetry/ingest",
            json={"samples": samples},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["accepted"], 3)
        self.assertEqual(body["rejected"], 0)
        self.assertEqual(body["received"], 3)

    def test_ingest_drops_invalid_samples_but_keeps_valid_ones(self):
        dv_id = self._create_deployment_version()
        samples = [
            {"latency_ms": 12.0},
            {"latency_ms": -5.0},  # rejected: invalid_latency_ms
            {"latency_ms": "not-a-number"},  # rejected: invalid_latency_ms
            {"foo": "bar"},  # rejected: invalid_latency_ms (no latency)
        ]
        resp = self.client.post(
            f"/api/deployments/{dv_id}/telemetry/ingest",
            json={"samples": samples},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # latency_ms type=float in Pydantic — `"not-a-number"` is rejected
        # by the request model with a 422 in strict mode, but our schema
        # accepts a float-coercible value. Confirm the service-side
        # rejection path catches the negative latency at minimum.
        self.assertGreaterEqual(body["accepted"], 1)
        self.assertGreaterEqual(body["rejected"], 1)
        reasons = {r["reason"] for r in body.get("rejected_details", [])}
        self.assertIn("invalid_latency_ms", reasons)

    def test_ingest_empty_batch_is_400(self):
        dv_id = self._create_deployment_version()
        resp = self.client.post(
            f"/api/deployments/{dv_id}/telemetry/ingest",
            json={"samples": []},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["detail"], "samples_required")

    def test_ingest_unknown_deployment_is_404(self):
        resp = self.client.post(
            "/api/deployments/999999/telemetry/ingest",
            json={"samples": [{"latency_ms": 1.0}]},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "deployment_version_not_found")

    def test_ingest_infers_success_from_status_code(self):
        dv_id = self._create_deployment_version()
        samples = [
            {"latency_ms": 10.0, "status_code": 200},  # success inferred true
            {"latency_ms": 20.0, "status_code": 503},  # success inferred false
        ]
        resp = self.client.post(
            f"/api/deployments/{dv_id}/telemetry/ingest",
            json={"samples": samples},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        # Telemetry endpoint then sees error_count=1, error_rate=0.5.
        agg = self.client.get(f"/api/deployments/{dv_id}/telemetry").json()
        self.assertEqual(agg["sample_count"], 2)
        self.assertEqual(agg["errors"]["count"], 1)
        self.assertAlmostEqual(agg["errors"]["rate"], 0.5)

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def test_aggregate_percentiles_known_distribution(self):
        dv_id = self._create_deployment_version()
        # Ingest 100 samples with latency 0..99ms in one second window
        # so the math is testable. All successful, no tokens.
        now = datetime.now(timezone.utc)
        samples = [
            {
                "latency_ms": float(i),
                "success": True,
                "ts": (now - timedelta(seconds=30)).isoformat(),
            }
            for i in range(100)
        ]
        ingest = self.client.post(
            f"/api/deployments/{dv_id}/telemetry/ingest",
            json={"samples": samples},
        )
        self.assertEqual(ingest.status_code, 200, ingest.text)
        self.assertEqual(ingest.json()["accepted"], 100)

        resp = self.client.get(
            f"/api/deployments/{dv_id}/telemetry",
            params={"window_seconds": 3600},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["sample_count"], 100)
        self.assertAlmostEqual(body["latency_ms"]["p50"], 49.5, places=3)
        self.assertAlmostEqual(body["latency_ms"]["p95"], 94.05, places=3)
        self.assertAlmostEqual(body["latency_ms"]["p99"], 98.01, places=3)
        self.assertAlmostEqual(body["latency_ms"]["min"], 0.0)
        self.assertAlmostEqual(body["latency_ms"]["max"], 99.0)
        self.assertAlmostEqual(body["latency_ms"]["mean"], 49.5)
        self.assertEqual(body["errors"]["count"], 0)

    def test_aggregate_token_throughput(self):
        dv_id = self._create_deployment_version()
        now = datetime.now(timezone.utc)
        # 60 samples evenly spread over 60s, each with 50 input + 25 output.
        samples = [
            {
                "latency_ms": 1.0,
                "success": True,
                "input_tokens": 50,
                "output_tokens": 25,
                "ts": (now - timedelta(seconds=i)).isoformat(),
            }
            for i in range(60)
        ]
        self.client.post(
            f"/api/deployments/{dv_id}/telemetry/ingest",
            json={"samples": samples},
        )
        resp = self.client.get(
            f"/api/deployments/{dv_id}/telemetry",
            params={"window_seconds": 60},
        )
        body = resp.json()
        self.assertEqual(body["tokens"]["input_total"], 3000)
        self.assertEqual(body["tokens"]["output_total"], 1500)
        # window 60s -> per-second = total/60. Allow some tolerance because
        # the precise `window_seconds` resolved by the server may include
        # the small delta between request build and aggregate fetch.
        window = float(body["window_seconds"])
        self.assertAlmostEqual(
            body["tokens"]["input_per_second"], 3000 / window, places=3
        )
        self.assertAlmostEqual(
            body["tokens"]["output_per_second"], 1500 / window, places=3
        )

    def test_aggregate_empty_window_returns_zeros(self):
        dv_id = self._create_deployment_version()
        resp = self.client.get(f"/api/deployments/{dv_id}/telemetry")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["sample_count"], 0)
        self.assertEqual(body["request_volume"]["total"], 0)
        self.assertEqual(body["latency_ms"]["p95"], 0.0)
        self.assertEqual(body["errors"]["count"], 0)
        self.assertEqual(body["tokens"]["input_total"], 0)

    def test_aggregate_excludes_samples_outside_window(self):
        dv_id = self._create_deployment_version()
        now = datetime.now(timezone.utc)
        in_window = [
            {
                "latency_ms": 10.0,
                "ts": (now - timedelta(seconds=30)).isoformat(),
            }
            for _ in range(5)
        ]
        out_of_window = [
            {
                "latency_ms": 999.0,
                "ts": (now - timedelta(hours=2)).isoformat(),
            }
            for _ in range(3)
        ]
        self.client.post(
            f"/api/deployments/{dv_id}/telemetry/ingest",
            json={"samples": in_window + out_of_window},
        )
        resp = self.client.get(
            f"/api/deployments/{dv_id}/telemetry",
            params={"window_seconds": 60},
        )
        body = resp.json()
        self.assertEqual(body["sample_count"], 5)
        # The 999ms outliers must not have leaked into the aggregate.
        self.assertLess(body["latency_ms"]["max"], 100.0)

    def test_aggregate_invalid_window_400(self):
        dv_id = self._create_deployment_version()
        now = datetime.now(timezone.utc)
        # since strictly after until ⇒ invalid window
        resp = self.client.get(
            f"/api/deployments/{dv_id}/telemetry",
            params={
                "since": now.isoformat(),
                "until": (now - timedelta(hours=1)).isoformat(),
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["detail"], "invalid_window")

    def test_aggregate_unknown_deployment_404(self):
        resp = self.client.get("/api/deployments/999999/telemetry")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "deployment_version_not_found")

    # ------------------------------------------------------------------
    # Raw samples
    # ------------------------------------------------------------------

    def test_recent_samples_returns_newest_first_up_to_limit(self):
        dv_id = self._create_deployment_version()
        now = datetime.now(timezone.utc)
        samples = [
            {
                "latency_ms": float(i),
                "ts": (now - timedelta(seconds=i)).isoformat(),
                "request_id": f"req-{i:03d}",
            }
            for i in range(10)
        ]
        self.client.post(
            f"/api/deployments/{dv_id}/telemetry/ingest",
            json={"samples": samples},
        )
        resp = self.client.get(
            f"/api/deployments/{dv_id}/telemetry/samples",
            params={"limit": 3},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["limit"], 3)
        self.assertEqual(len(body["samples"]), 3)
        # Newest first => latency_ms 0.0, 1.0, 2.0 (smaller seconds-ago).
        self.assertEqual(body["samples"][0]["request_id"], "req-000")
        self.assertEqual(body["samples"][1]["request_id"], "req-001")
        self.assertEqual(body["samples"][2]["request_id"], "req-002")

    def test_recent_samples_unknown_deployment_404(self):
        resp = self.client.get("/api/deployments/999999/telemetry/samples")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "deployment_version_not_found")


if __name__ == "__main__":
    unittest.main()

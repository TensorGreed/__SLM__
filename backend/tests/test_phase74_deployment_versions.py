"""Phase 74 — deployment versions + rollback (priority.md P25).

Covers the end-to-end deploy-version lifecycle layered on top of the
existing ``deploy-as-api/execute`` path:

- Non-dry-run execute persists a ``PENDING`` ``DeploymentVersion`` and
  returns its id; secret-bearing keys never reach the persisted
  ``plan_payload``.
- Dry-run execute does **not** persist a deployment version (a dry run
  has not actually deployed anything to the target slot).
- ``POST /deployments/{id}/promote`` flips PENDING → PROMOTED and writes
  a sequence-1 audit row; promoting again is rejected with
  ``not_promotable`` (409).
- A second execute of the same export+target produces v2; promoting v2
  moves v1 to ``SUPERSEDED`` in the same transaction.
- ``POST /deployments/{id}/rollback`` (target = currently-PROMOTED v2)
  re-promotes v1 and appends both a ``rollback`` and a ``promote``
  audit row; rolling back when nothing has ever been promoted before
  returns ``no_promoted_predecessor`` (409).
- ``POST /deployments/{id}/reject`` flips PENDING → REJECTED; rejecting
  a PROMOTED row is forbidden with ``not_rejectable`` (409).
- ``GET  /deployments/{id}`` returns the deployment-version row plus
  its full audit chain.
- ``GET  /projects/{id}/deployments`` returns rows for a project and
  supports the ``status`` / ``export_id`` / ``target_id`` filters.
- 404s for unknown deployment-version ids and unknown projects.
"""

from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase74_deployment_versions.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase74_deployment_versions_data"

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


class Phase74DeploymentVersionsTests(unittest.TestCase):
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

    def _create_project(self, name: str = "phase74") -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"{name}-{uuid.uuid4().hex[:8]}",
                "description": "phase74",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_completed_export(self, project_id: int) -> int:
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase74-exp",
                "description": "phase74",
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
        (model_dir / "config.json").write_text('{"model_type":"phi"}', encoding="utf-8")
        (model_dir / "weights.safetensors").write_bytes(b"phase74-model")
        (model_dir / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")

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
            json={"deployment_targets": ["exporter.huggingface"], "run_smoke_tests": False},
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        self.assertEqual(str(run_resp.json().get("status")), "completed")
        return export_id

    def _execute_deploy(
        self,
        *,
        project_id: int,
        export_id: int,
        target_id: str = "sdk.apple_coreml_stub",
        dry_run: bool = False,
        endpoint_name: str | None = None,
        hf_token: str | None = None,
    ) -> dict:
        # The SDK stub targets execute locally without making any network
        # calls, so they're the right choice for tests that exercise the
        # non-dry-run recording path. The HF inference endpoint hits the
        # real HuggingFace API on execute and is exercised separately
        # in the dry-run test.
        body: dict = {"target_id": target_id, "dry_run": dry_run}
        if endpoint_name is not None:
            body["endpoint_name"] = endpoint_name
        if hf_token is not None:
            body["hf_token"] = hf_token
        resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/deploy-as-api/execute",
            json=body,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def test_execute_non_dry_run_records_pending_version(self):
        project_id = self._create_project("record")
        export_id = self._create_completed_export(project_id)

        payload = self._execute_deploy(
            project_id=project_id,
            export_id=export_id,
            dry_run=False,
            endpoint_name="phase74-endpoint",
            hf_token="hf-secret-must-not-persist",
        )
        self.assertIn("deployment_version_id", payload)
        deployment_version_id = int(payload["deployment_version_id"])

        get_resp = self.client.get(f"/api/deployments/{deployment_version_id}")
        self.assertEqual(get_resp.status_code, 200, get_resp.text)
        body = get_resp.json()
        dv = body["deployment_version"]
        self.assertEqual(dv["status"], "pending")
        self.assertEqual(dv["export_id"], export_id)
        self.assertEqual(dv["project_id"], project_id)
        self.assertEqual(dv["version"], 1)
        self.assertEqual(dv["target_id"], "sdk.apple_coreml_stub")
        self.assertEqual(dv["endpoint_name"], "phase74-endpoint")
        # Secret tokens must never be persisted.
        plan_payload = dv["plan_payload"]
        for key in plan_payload:
            for forbidden in ("token", "secret", "key", "password", "credential"):
                self.assertNotIn(forbidden, key.lower(), plan_payload)
        self.assertEqual(body["audit"], [])

    def test_dry_run_execute_does_not_record_version(self):
        project_id = self._create_project("dryrun")
        export_id = self._create_completed_export(project_id)

        payload = self._execute_deploy(
            project_id=project_id,
            export_id=export_id,
            dry_run=True,
        )
        self.assertNotIn("deployment_version_id", payload)

        list_resp = self.client.get(
            f"/api/projects/{project_id}/deployments",
            params={"export_id": export_id},
        )
        self.assertEqual(list_resp.status_code, 200, list_resp.text)
        self.assertEqual(list_resp.json()["deployment_versions"], [])

    # ------------------------------------------------------------------
    # Promote
    # ------------------------------------------------------------------

    def test_promote_pending_marks_promoted_and_audits(self):
        project_id = self._create_project("promote")
        export_id = self._create_completed_export(project_id)
        execute_payload = self._execute_deploy(
            project_id=project_id, export_id=export_id
        )
        dv_id = int(execute_payload["deployment_version_id"])

        promote_resp = self.client.post(
            f"/api/deployments/{dv_id}/promote",
            json={"reason": "ready for prod", "actor": "alice"},
        )
        self.assertEqual(promote_resp.status_code, 200, promote_resp.text)
        body = promote_resp.json()
        self.assertEqual(body["deployment_version"]["status"], "promoted")
        self.assertEqual(body["deployment_version"]["promoted_reason"], "ready for prod")
        self.assertEqual(body["deployment_version"]["actor"], "alice")
        self.assertIsNotNone(body["deployment_version"]["promoted_at"])

        self.assertEqual(body["audit"]["sequence"], 1)
        self.assertEqual(body["audit"]["action"], "promote")
        self.assertEqual(body["audit"]["actor"], "alice")
        self.assertEqual(body["audit"]["status_after"], "promoted")

    def test_promote_already_promoted_is_409(self):
        project_id = self._create_project("repromote")
        export_id = self._create_completed_export(project_id)
        dv_id = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        first = self.client.post(f"/api/deployments/{dv_id}/promote", json={})
        self.assertEqual(first.status_code, 200, first.text)

        second = self.client.post(f"/api/deployments/{dv_id}/promote", json={})
        self.assertEqual(second.status_code, 409, second.text)
        self.assertEqual(second.json()["detail"], "not_promotable")

    def test_second_promote_supersedes_prior(self):
        project_id = self._create_project("supersede")
        export_id = self._create_completed_export(project_id)

        v1 = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        promote_v1 = self.client.post(f"/api/deployments/{v1}/promote", json={})
        self.assertEqual(promote_v1.status_code, 200, promote_v1.text)

        v2 = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        # Second execute increments the per-export version counter.
        v2_get = self.client.get(f"/api/deployments/{v2}").json()
        self.assertEqual(v2_get["deployment_version"]["version"], 2)

        promote_v2 = self.client.post(f"/api/deployments/{v2}/promote", json={})
        self.assertEqual(promote_v2.status_code, 200, promote_v2.text)

        v1_after = self.client.get(f"/api/deployments/{v1}").json()
        self.assertEqual(v1_after["deployment_version"]["status"], "superseded")
        self.assertIsNotNone(v1_after["deployment_version"]["superseded_at"])

        v2_after = self.client.get(f"/api/deployments/{v2}").json()
        self.assertEqual(v2_after["deployment_version"]["status"], "promoted")

    # ------------------------------------------------------------------
    # Reject
    # ------------------------------------------------------------------

    def test_reject_pending_marks_rejected(self):
        project_id = self._create_project("reject")
        export_id = self._create_completed_export(project_id)
        dv_id = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )

        resp = self.client.post(
            f"/api/deployments/{dv_id}/reject",
            json={"reason": "smoke test failed", "actor": "bob"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["deployment_version"]["status"], "rejected")
        self.assertEqual(body["deployment_version"]["rejected_reason"], "smoke test failed")
        self.assertIsNotNone(body["deployment_version"]["rejected_at"])
        self.assertEqual(body["audit"]["action"], "reject")
        self.assertEqual(body["audit"]["sequence"], 1)

    def test_reject_promoted_is_409(self):
        project_id = self._create_project("rejectprom")
        export_id = self._create_completed_export(project_id)
        dv_id = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        promote = self.client.post(f"/api/deployments/{dv_id}/promote", json={})
        self.assertEqual(promote.status_code, 200, promote.text)

        resp = self.client.post(f"/api/deployments/{dv_id}/reject", json={})
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["detail"], "not_rejectable")

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def test_rollback_re_promotes_predecessor(self):
        project_id = self._create_project("rollback")
        export_id = self._create_completed_export(project_id)

        v1 = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        self.assertEqual(
            self.client.post(f"/api/deployments/{v1}/promote", json={}).status_code,
            200,
        )

        v2 = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        self.assertEqual(
            self.client.post(f"/api/deployments/{v2}/promote", json={}).status_code,
            200,
        )

        rollback_resp = self.client.post(
            f"/api/deployments/{v2}/rollback",
            json={"reason": "regression", "actor": "carol"},
        )
        self.assertEqual(rollback_resp.status_code, 200, rollback_resp.text)
        body = rollback_resp.json()

        self.assertEqual(body["rolled_back"]["status"], "rolled_back")
        self.assertEqual(body["rolled_back"]["rolled_back_to_id"], v1)
        self.assertEqual(body["promoted"]["id"], v1)
        self.assertEqual(body["promoted"]["status"], "promoted")
        # Predecessor's superseded_at is cleared on re-promotion so the
        # row reads as "currently live" without lingering tombstone state.
        self.assertIsNone(body["promoted"]["superseded_at"])

        self.assertEqual(len(body["audit"]), 2)
        actions = [a["action"] for a in body["audit"]]
        self.assertEqual(sorted(actions), ["promote", "rollback"])

        # Audit chain on v2 should now contain promote (seq 1, when first
        # promoted), the supersede-by-v? promote-action audit (seq 2,
        # appended when v2 was promoted), and the rollback (seq 3 or 2).
        v2_audit = self.client.get(f"/api/deployments/{v2}/audit").json()
        seqs = [row["sequence"] for row in v2_audit["audit"]]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(v2_audit["audit"][-1]["action"], "rollback")

    def test_rollback_without_predecessor_is_409(self):
        project_id = self._create_project("nopred")
        export_id = self._create_completed_export(project_id)
        dv_id = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        self.assertEqual(
            self.client.post(f"/api/deployments/{dv_id}/promote", json={}).status_code,
            200,
        )
        resp = self.client.post(f"/api/deployments/{dv_id}/rollback", json={})
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["detail"], "no_promoted_predecessor")

    def test_rollback_pending_is_409(self):
        project_id = self._create_project("rbpending")
        export_id = self._create_completed_export(project_id)
        dv_id = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        resp = self.client.post(f"/api/deployments/{dv_id}/rollback", json={})
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["detail"], "not_rollbackable")

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def test_get_deployment_includes_audit_chain(self):
        project_id = self._create_project("read")
        export_id = self._create_completed_export(project_id)
        dv_id = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        self.assertEqual(
            self.client.post(
                f"/api/deployments/{dv_id}/promote",
                json={"reason": "ok", "actor": "ops"},
            ).status_code,
            200,
        )
        resp = self.client.get(f"/api/deployments/{dv_id}")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["deployment_version"]["status"], "promoted")
        self.assertEqual(len(body["audit"]), 1)
        self.assertEqual(body["audit"][0]["action"], "promote")
        self.assertEqual(body["audit"][0]["actor"], "ops")

    def test_list_for_project_filters(self):
        project_id = self._create_project("list")
        export_id = self._create_completed_export(project_id)

        first = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        second = int(
            self._execute_deploy(project_id=project_id, export_id=export_id)[
                "deployment_version_id"
            ]
        )
        self.assertEqual(
            self.client.post(f"/api/deployments/{first}/promote", json={}).status_code,
            200,
        )

        all_resp = self.client.get(f"/api/projects/{project_id}/deployments")
        self.assertEqual(all_resp.status_code, 200)
        all_ids = [d["id"] for d in all_resp.json()["deployment_versions"]]
        self.assertEqual(set(all_ids), {first, second})

        promoted_resp = self.client.get(
            f"/api/projects/{project_id}/deployments",
            params={"status": "promoted"},
        )
        self.assertEqual(promoted_resp.status_code, 200)
        promoted_ids = [d["id"] for d in promoted_resp.json()["deployment_versions"]]
        self.assertEqual(promoted_ids, [first])

        # Filter by export_id round-trips through the same query path.
        export_filter = self.client.get(
            f"/api/projects/{project_id}/deployments",
            params={"export_id": export_id},
        )
        self.assertEqual(export_filter.status_code, 200)
        export_ids = [d["id"] for d in export_filter.json()["deployment_versions"]]
        self.assertEqual(set(export_ids), {first, second})

    # ------------------------------------------------------------------
    # 404s
    # ------------------------------------------------------------------

    def test_get_unknown_deployment_is_404(self):
        resp = self.client.get("/api/deployments/9999999")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "deployment_version_not_found")

    def test_promote_unknown_deployment_is_404(self):
        resp = self.client.post("/api/deployments/9999999/promote", json={})
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "deployment_version_not_found")

    def test_list_unknown_project_is_404(self):
        resp = self.client.get("/api/projects/9999999/deployments")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "project_not_found")


if __name__ == "__main__":
    unittest.main()

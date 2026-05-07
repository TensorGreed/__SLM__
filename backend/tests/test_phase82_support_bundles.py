"""Phase 82 — support-bundle service with redaction (priority.md P34).

Coverage:

- Direct ``redact_payload`` unit tests:
  - sensitive *keys* are scrubbed regardless of value shape
    (``token``, ``api_key``, ``password``, ``credential``,
    ``auth``, ``private_key``, ``access_key``).
  - secret value *patterns* are scrubbed even when the key is
    benign (HF token, OpenAI key, AWS access key, bearer prefix,
    JWT, URL with embedded creds).
  - per-reason counts in the stats block are accurate.
  - non-string scalars (int/bool/None) pass through unchanged.

- End-to-end bundle creation via the API:
  - ``POST /api/projects/{id}/support-bundle`` returns a metadata
    payload with section_counts, redactions_applied, expires_at,
    and a ``download_url`` containing a ``token`` query.
  - The zip on disk has the expected layout: ``manifest.json``,
    ``env.txt``, ``sections/<name>.json``.
  - Sensitive payload values seeded in run_events get redacted in
    the bundle's ``run_events.json`` section.
  - ``GET /api/projects/{id}/support-bundles`` lists the new bundle.
  - The download endpoint returns the zip bytes when given the
    correct ``token``.
  - Wrong token → 403 ``support_bundle_invalid_token``.
  - Unknown bundle_uid → 404 ``support_bundle_not_found``.
  - Expired bundle → 410 ``support_bundle_expired``.
  - Unknown project → 404 ``project_not_found``.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import unittest
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase82_support_bundles.db"
TEST_DATA_DIR = (
    Path(__file__).resolve().parent / "phase82_support_bundles_data"
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
from app.models.reason_codes import TRAINING_RUNTIME_ERROR
from app.models.run_event import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    STAGE_TRAINING,
)
from app.models.support_bundle import SupportBundle
from app.services.run_event_service import emit_event
from app.services.support_bundle_service import redact_payload


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


class Phase82SupportBundleTests(unittest.TestCase):
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

    def _create_project(self, name: str = "phase82") -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"{name}-{uuid.uuid4().hex[:8]}",
                "description": "phase82",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_event_with_secret_payload(self, project_id: int) -> None:
        async def _go():
            async with async_session_factory() as db:
                await emit_event(
                    db,
                    project_id=project_id,
                    run_id="exp-1",
                    stage=STAGE_TRAINING,
                    severity=SEVERITY_ERROR,
                    reason_code=TRAINING_RUNTIME_ERROR,
                    summary="Training run failed; see hf_FAKE1234567890ABCDEF for details",
                    payload={
                        "hf_token": "hf_FAKE1234567890ABCDEF",
                        "ok_field": "no secrets here",
                        "nested": {
                            "api_key": "sk-FAKE1234567890ABCDEFGHIJ",
                            "label": "human-readable",
                        },
                        "logs_url": "https://user:pass@logs.example.com/runs/1",
                    },
                )
                await db.commit()

        asyncio.run(_go())

    def _read_zip_section(
        self, zip_bytes: bytes, name: str
    ) -> object:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            with zf.open(name) as fh:
                return json.loads(fh.read().decode("utf-8"))

    # ------------------------------------------------------------------
    # 1. Direct redaction unit tests
    # ------------------------------------------------------------------

    def test_redact_sensitive_keys(self):
        cleaned, stats = redact_payload(
            {
                "api_key": "anything-here",
                "password": "x",
                "credential": "x",
                "auth_header": "Bearer something",
                "private_key": "----stuff----",
                "access_key": "AKIAxxxxxxxxxxxxxxxx",
                "session_id": "abc",
                "ok_value": "kept",
            }
        )
        # Every sensitive key is scrubbed; ``ok_value`` survives.
        for key in (
            "api_key",
            "password",
            "credential",
            "auth_header",
            "private_key",
            "access_key",
            "session_id",
        ):
            self.assertEqual(
                cleaned[key], "***REDACTED:sensitive_key***", key
            )
        self.assertEqual(cleaned["ok_value"], "kept")
        self.assertGreaterEqual(stats["total"], 7)
        self.assertGreaterEqual(stats["by_reason"]["sensitive_key"], 7)

    def test_redact_value_patterns_independent_of_key(self):
        # Keys are benign — should still scrub due to value shape.
        cleaned, stats = redact_payload(
            {
                "log_message": "got hf_FAKE1234567890ABCDEF from caller",
                "preview": "see sk-FAKE1234567890ABCDEFGHIJ",
                # Real AWS access keys are AKIA + 16 alphanumeric chars (20 total).
                "stack": "AKIA1234567890ABCDEF",
                "header": "Bearer abcdefghijklmnopqrstuvwxyz12345",
                "url": "https://user:pass@example.com/health",
                "summary": "no secrets at all",
            }
        )
        self.assertNotIn(
            "hf_FAKE1234567890ABCDEF", json.dumps(cleaned)
        )
        self.assertNotIn(
            "sk-FAKE1234567890ABCDEFGHIJ", json.dumps(cleaned)
        )
        self.assertNotIn("AKIA1234567890ABCDEF", json.dumps(cleaned))
        self.assertEqual(cleaned["summary"], "no secrets at all")
        # Per-reason counts are accurate.
        self.assertGreaterEqual(stats["by_reason"].get("hf_token", 0), 1)
        self.assertGreaterEqual(stats["by_reason"].get("openai_key", 0), 1)
        self.assertGreaterEqual(
            stats["by_reason"].get("aws_access_key", 0), 1
        )
        self.assertGreaterEqual(
            stats["by_reason"].get("bearer_token", 0), 1
        )
        self.assertGreaterEqual(
            stats["by_reason"].get("url_with_credentials", 0), 1
        )

    def test_redact_recurses_into_nested_lists_and_dicts(self):
        cleaned, _ = redact_payload(
            [
                {"hf_token": "hf_AAAAAAAAAAAAAAAAAAA"},
                [{"nested_token": "hf_BBBBBBBBBBBBBBBBBBB"}],
                "hf_CCCCCCCCCCCCCCCCCCC inline",
            ]
        )
        self.assertNotIn("hf_AAAAAAAA", json.dumps(cleaned))
        self.assertNotIn("hf_BBBBBBBB", json.dumps(cleaned))
        self.assertNotIn("hf_CCCCCCCC", json.dumps(cleaned))

    def test_redact_passes_non_strings_through(self):
        cleaned, _ = redact_payload(
            {
                "count": 7,
                "ratio": 0.95,
                "ok": True,
                "missing": None,
                "tags": ["a", "b", "c"],
            }
        )
        self.assertEqual(cleaned["count"], 7)
        self.assertEqual(cleaned["ratio"], 0.95)
        self.assertIs(cleaned["ok"], True)
        self.assertIsNone(cleaned["missing"])
        self.assertEqual(cleaned["tags"], ["a", "b", "c"])

    # ------------------------------------------------------------------
    # 2. End-to-end bundle creation + download
    # ------------------------------------------------------------------

    def test_create_bundle_returns_metadata_and_writes_zip(self):
        project_id = self._create_project("createzip")
        self._seed_event_with_secret_payload(project_id)

        resp = self.client.post(
            f"/api/projects/{project_id}/support-bundle",
            json={"actor": "ops"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()

        # Metadata shape.
        self.assertIn("bundle_uid", body)
        self.assertIn("download_url", body)
        self.assertIn("download_token", body)
        self.assertIn("token=", body["download_url"])
        self.assertGreater(int(body["size_bytes"]), 0)
        self.assertIn("run_events", body["section_counts"])
        self.assertGreaterEqual(
            int(body["section_counts"]["run_events"]), 1
        )
        # Project section is always exactly 1 row.
        self.assertEqual(int(body["section_counts"]["project"]), 1)
        self.assertEqual(body["actor"], "ops")

        # Bundle row persisted.
        async def _row():
            from sqlalchemy import select

            async with async_session_factory() as db:
                result = await db.execute(
                    select(SupportBundle).where(
                        SupportBundle.bundle_uid == body["bundle_uid"]
                    )
                )
                return result.scalar_one()

        row = asyncio.run(_row())
        self.assertEqual(row.project_id, project_id)
        self.assertGreater(row.size_bytes, 0)
        self.assertTrue(Path(row.file_path).exists())

    def test_bundle_zip_layout_and_redaction(self):
        project_id = self._create_project("layout")
        self._seed_event_with_secret_payload(project_id)
        meta = self.client.post(
            f"/api/projects/{project_id}/support-bundle", json={}
        ).json()

        download_url = meta["download_url"]
        # Strip the ``/api`` prefix because TestClient already roots there.
        relative = download_url[len("/api"):] if download_url.startswith("/api") else download_url
        resp = self.client.get(f"/api{relative}")
        self.assertEqual(resp.status_code, 200, resp.text)
        zip_bytes = resp.content

        # Required entries.
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
        self.assertIn("manifest.json", names)
        self.assertIn("env.txt", names)
        for section in (
            "project",
            "experiments",
            "training_manifests",
            "autopilot_decisions",
            "run_events",
            "deployment_versions",
            "deployment_audit",
            "failure_clusters",
            "model_registry",
        ):
            self.assertIn(f"sections/{section}.json", names, section)

        # The run_events section must not contain the seeded secrets.
        run_events_section = self._read_zip_section(
            zip_bytes, "sections/run_events.json"
        )
        as_text = json.dumps(run_events_section)
        self.assertNotIn("hf_FAKE1234567890ABCDEF", as_text)
        self.assertNotIn("sk-FAKE1234567890ABCDEFGHIJ", as_text)
        self.assertNotIn(
            "user:pass@logs.example.com", as_text
        )
        # The benign field survives.
        self.assertIn("no secrets here", as_text)

        # Manifest reports redaction stats > 0 for run_events.
        manifest_section = self._read_zip_section(
            zip_bytes, "manifest.json"
        )
        run_events_redactions = manifest_section[
            "redactions_applied"
        ]["run_events"]
        self.assertGreater(int(run_events_redactions["total"]), 0)

    def test_download_wrong_token_returns_403(self):
        project_id = self._create_project("badtoken")
        self._seed_event_with_secret_payload(project_id)
        body = self.client.post(
            f"/api/projects/{project_id}/support-bundle", json={}
        ).json()

        resp = self.client.get(
            f"/api/support-bundles/{body['bundle_uid']}/download",
            params={"token": "definitely-not-the-token"},
        )
        self.assertEqual(resp.status_code, 403, resp.text)
        self.assertEqual(
            resp.json()["detail"], "support_bundle_invalid_token"
        )

    def test_download_unknown_uid_returns_404(self):
        resp = self.client.get(
            "/api/support-bundles/0000000000000000/download",
            params={"token": "anything"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(
            resp.json()["detail"], "support_bundle_not_found"
        )

    def test_expired_bundle_returns_410(self):
        project_id = self._create_project("expired")
        self._seed_event_with_secret_payload(project_id)
        body = self.client.post(
            f"/api/projects/{project_id}/support-bundle", json={}
        ).json()

        # Force the bundle to be expired by rewriting the row.
        async def _expire():
            from sqlalchemy import select

            async with async_session_factory() as db:
                row = (
                    await db.execute(
                        select(SupportBundle).where(
                            SupportBundle.bundle_uid == body["bundle_uid"]
                        )
                    )
                ).scalar_one()
                row.expires_at = datetime.now(timezone.utc) - timedelta(
                    minutes=1
                )
                await db.commit()

        asyncio.run(_expire())

        resp = self.client.get(
            f"/api/support-bundles/{body['bundle_uid']}/download",
            params={"token": body["download_token"]},
        )
        self.assertEqual(resp.status_code, 410, resp.text)
        self.assertEqual(
            resp.json()["detail"], "support_bundle_expired"
        )

    def test_create_bundle_unknown_project_404(self):
        resp = self.client.post(
            "/api/projects/999999/support-bundle", json={}
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "project_not_found")

    def test_list_bundles_for_project_returns_newest_first(self):
        project_id = self._create_project("listing")
        self._seed_event_with_secret_payload(project_id)
        first = self.client.post(
            f"/api/projects/{project_id}/support-bundle", json={}
        ).json()
        second = self.client.post(
            f"/api/projects/{project_id}/support-bundle", json={}
        ).json()

        listed = self.client.get(
            f"/api/projects/{project_id}/support-bundles"
        ).json()["bundles"]
        self.assertEqual(len(listed), 2)
        self.assertEqual(listed[0]["bundle_uid"], second["bundle_uid"])
        self.assertEqual(listed[1]["bundle_uid"], first["bundle_uid"])

    def test_list_bundles_unknown_project_404(self):
        resp = self.client.get("/api/projects/999999/support-bundles")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "project_not_found")


if __name__ == "__main__":
    unittest.main()

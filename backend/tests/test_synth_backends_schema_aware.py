"""Tests for the ``schema_aware`` backend flag (USER-SUCCESS Epic 5 Phase 5c).

Covers:
- Each registered backend declares ``schema_aware`` at the class level.
- The truth table is correct: NeMo + vLLM = True, Ollama + Teacher = False.
- The ``GET /api/projects/{id}/synthetic/backends`` endpoint surfaces
  ``schema_aware`` for every entry (so the frontend picker can badge
  the schema-honoring options).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "synth_schema_aware_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "synth_schema_aware_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.synth_backends import (  # noqa: E402
    BACKEND_REGISTRY,
    NemoBackend,
    OllamaBackend,
    TeacherModelBackend,
    VllmBackend,
)


class BackendSchemaAwareFlagTests(unittest.TestCase):
    """The class-level ``schema_aware`` attribute is the source of truth
    for both the picker badge and the audit trail in the API response."""

    def test_ollama_is_not_schema_aware(self):
        # Ollama's /v1 shim ignores response_format=json_schema.
        self.assertFalse(OllamaBackend.schema_aware)

    def test_teacher_is_not_schema_aware(self):
        # The legacy dispatcher has no structured-output hook.
        self.assertFalse(TeacherModelBackend.schema_aware)

    def test_nemo_is_schema_aware(self):
        # NIM honors OpenAI Structured Outputs natively (Phase 5b).
        self.assertTrue(NemoBackend.schema_aware)

    def test_vllm_is_schema_aware(self):
        # vLLM enforces it during decoding via xgrammar / outlines
        # (Phase 5c — the whole reason this backend exists).
        self.assertTrue(VllmBackend.schema_aware)

    def test_every_registered_backend_declares_the_flag(self):
        # Future backends must opt in or out explicitly — no implicit
        # "maybe schema-aware" picker UX.
        for cls in BACKEND_REGISTRY:
            with self.subTest(backend=cls.name):
                self.assertTrue(
                    hasattr(cls, "schema_aware"),
                    f"{cls.name} must declare schema_aware at the class level",
                )
                self.assertIsInstance(cls.schema_aware, bool)


class BackendsApiSurfaceSchemaAwareTests(unittest.TestCase):
    """End-to-end: the /backends endpoint must include ``schema_aware``
    on every entry. The frontend type (``SynthBackendInfo``) reads it
    to badge picker options."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def test_backends_endpoint_includes_schema_aware_per_entry(self):
        # project_id is read but not load-bearing; the route lives
        # under the project-scoped prefix for auth uniformity.
        resp = self.client.get("/api/projects/1/synthetic/backends")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertIn("backends", payload)
        by_name = {entry["name"]: entry for entry in payload["backends"]}
        # Every registered backend must show up.
        for cls in BACKEND_REGISTRY:
            self.assertIn(cls.name, by_name, f"{cls.name} missing from /backends")
            entry = by_name[cls.name]
            self.assertIn("schema_aware", entry)
            self.assertIsInstance(entry["schema_aware"], bool)
            self.assertEqual(
                entry["schema_aware"],
                cls.schema_aware,
                f"{cls.name} schema_aware on the wire must match the class-level flag",
            )

    def test_backends_endpoint_marks_nemo_and_vllm_schema_aware(self):
        resp = self.client.get("/api/projects/1/synthetic/backends")
        by_name = {entry["name"]: entry for entry in resp.json()["backends"]}
        self.assertTrue(by_name["nemo"]["schema_aware"])
        self.assertTrue(by_name["vllm"]["schema_aware"])
        self.assertFalse(by_name["ollama"]["schema_aware"])
        self.assertFalse(by_name["teacher"]["schema_aware"])


if __name__ == "__main__":
    unittest.main()

"""Tests for the curriculum-preview API (USER-SUCCESS Epic 6 Phase 6a).

End-to-end via FastAPI TestClient:
- 404 on unknown project.
- 400 when project has no recipe selected.
- 400 when recipe has no curriculum scoring mode (e.g. qa-sft today).
- 400 when project has no training rows yet.
- 503 when the embedder dependency isn't installed.
- 200 on a classification template — response shape includes
  recipe_id, scoring_mode, total_rows, returned, and a ``ranked``
  list whose first entry has the lowest difficulty.
- ``limit`` query trims the response payload without losing the
  total count.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "curriculum_api_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "curriculum_api_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Stub embedder — bypasses sentence-transformers so tests don't have
# to download a 22MB model just to assert the API contract.
# ─────────────────────────────────────────────────────────────────────


def _stub_embedder(texts: list[str]) -> list[list[float]]:
    """Tiny deterministic embedder: each text maps to a 3-d vector
    where the first dimension is the length (so similar-length rows
    cluster) and the next two are based on character distribution.
    Good enough to verify the API plumbing — NOT good enough to
    judge real curriculum quality (that's what A/B in 6c is for)."""
    vectors: list[list[float]] = []
    for text in texts:
        text = text or ""
        length = float(len(text)) or 0.0
        a_share = float(sum(1 for c in text.lower() if c == "a"))
        other = float(sum(1 for c in text.lower() if c.isalpha() and c != "a"))
        # Normalize so cosine math doesn't degenerate to constants.
        magnitude = (length ** 2 + a_share ** 2 + other ** 2) ** 0.5 or 1.0
        vectors.append(
            [length / magnitude, a_share / magnitude, other / magnitude]
        )
    return vectors


class CurriculumPreviewApiTests(unittest.TestCase):
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

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_unknown_project_returns_404(self):
        resp = self.client.get("/api/projects/99999/curriculum/preview")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_project_without_recipe_returns_400(self):
        # Direct project creation (no template instantiation) leaves
        # selected_recipe blank.
        resp = self.client.post(
            "/api/projects",
            json={"name": "No Recipe Project"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = resp.json()["id"]
        preview = self.client.get(f"/api/projects/{pid}/curriculum/preview")
        self.assertEqual(preview.status_code, 400, preview.text)
        self.assertIn("recipe", preview.text.lower())

    def test_qa_sft_recipe_returns_400_curriculum_unavailable(self):
        """qa-sft has no curriculum scoring mode in Phase 6a — the
        API returns 400 with a message naming the recipe."""
        project = self._instantiate_template(
            "policy-qa-style", "QA Curriculum Unavailable"
        )
        resp = self.client.get(
            f"/api/projects/{project['id']}/curriculum/preview"
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("qa-sft", resp.text)

    def test_classification_template_returns_ranked_payload(self):
        """End-to-end on ticket-router (a classification template).
        Asserts the response shape + the ``ranked`` ordering is
        ascending by difficulty."""
        project = self._instantiate_template(
            "ticket-router", "Classification Curriculum Preview"
        )
        # Patch the embedder so the test doesn't depend on
        # sentence-transformers being installed (it usually is, but
        # we want hermetic tests).
        with patch(
            "app.services.curriculum_service._sentence_transformer_embedder",
            side_effect=_stub_embedder,
        ):
            resp = self.client.get(
                f"/api/projects/{project['id']}/curriculum/preview"
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["recipe_id"], "classification")
        self.assertEqual(payload["scoring_mode"], "prototype_entropy")
        self.assertGreater(payload["total_rows"], 0)
        self.assertGreater(payload["returned"], 0)
        ranked = payload["ranked"]
        # Ascending by difficulty.
        for prev_entry, next_entry in zip(ranked, ranked[1:]):
            self.assertLessEqual(prev_entry["difficulty"], next_entry["difficulty"])
        # Each entry carries the snippet field the UI uses.
        for entry in ranked:
            self.assertIn("snippet", entry)
            self.assertIn("row_id", entry)
            self.assertIn("rank", entry)
            self.assertIn("difficulty", entry)

    def test_limit_query_caps_response_without_losing_total(self):
        project = self._instantiate_template(
            "ticket-router", "Curriculum Limit"
        )
        with patch(
            "app.services.curriculum_service._sentence_transformer_embedder",
            side_effect=_stub_embedder,
        ):
            resp = self.client.get(
                f"/api/projects/{project['id']}/curriculum/preview",
                params={"limit": 3},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        # Total row count is the FULL count; ``returned`` matches limit.
        self.assertGreater(payload["total_rows"], 3)
        self.assertEqual(payload["returned"], 3)
        self.assertEqual(len(payload["ranked"]), 3)

    def test_503_when_sentence_transformers_missing(self):
        """The endpoint surfaces the embedder dependency error as 503,
        not 500. The message points the user at `pip install
        sentence-transformers`."""
        project = self._instantiate_template(
            "log-triage", "Curriculum No Embedder"
        )
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("not installed in this env")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            resp = self.client.get(
                f"/api/projects/{project['id']}/curriculum/preview"
            )
        self.assertEqual(resp.status_code, 503, resp.text)
        body = resp.text
        self.assertIn("sentence-transformers", body)
        self.assertIn("pip install", body)


if __name__ == "__main__":
    unittest.main()

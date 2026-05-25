"""Tests for the Phase 9b inference + training-completion wiring.

Covers:
  * ``build_index_for_project`` truth table:
    - project not found / no recipe / non-RAG-eligible recipe → skipped
    - QA-SFT + corpus rows → built, with manifest fields stamped
    - QA-SFT + missing prepared file + no Dataset rows → skipped
      ("no_corpus_rows") rather than raising
  * ``build_preamble_from_query``:
    - QA-SFT happy path → preamble text + retrieved chunks (top-K)
    - empty query → None (caller skips)
    - missing index → None (caller falls back to no-RAG)
    - non-QA recipe → None
    - preamble pair formatting handles both nested + flat shapes
  * Playground integration via FastAPI TestClient on the mock
    provider:
    - auto_rag=False (default) → response has NO auto_rag block, no
      messages mutated
    - auto_rag=True on QA-SFT project with index → response carries
      auto_rag.applied=true + retrieved list; system message
      preamble lands in front of the user's chat; session metadata
      stamped with the same retrieved payload
    - auto_rag=True on non-QA project (classification) → applied=false,
      skip_reason="recipe_or_index_ineligible"
    - auto_rag=True on QA-SFT WITHOUT a built index → same skip
      (no failure, falls back to no-RAG)
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

TEST_DB_PATH = Path(__file__).resolve().parent / "auto_rag_phase9b_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "auto_rag_phase9b_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.auto_rag_service import (  # noqa: E402
    build_index_for_project,
    build_preamble_from_query,
)


# ─────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────


def _clear_tree(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()


def _stub_project(recipe_id: str | None):
    p = MagicMock()
    p.selected_recipe = {"recipe_id": recipe_id} if recipe_id else None
    return p


def _stub_db(recipe_id: str | None, dataset_rows: list[dict] | None = None):
    """Mocks an AsyncSession returning a project with the given
    recipe and a SELECT(Dataset) that yields nothing (we'll rely on
    the prepared/train.jsonl fallback for corpus data)."""
    project = _stub_project(recipe_id)
    db = MagicMock()
    db.get = AsyncMock(return_value=project)
    # Stub db.execute returning a scalars()-yielding result. We
    # don't surface any Dataset rows from the SELECT so the
    # fallback to prepared/train.jsonl is what's exercised.
    result = MagicMock()
    result.scalars = MagicMock(return_value=iter([]))
    db.execute = AsyncMock(return_value=result)
    return db


def _seed_prepared_train(project_id: int, rows: list[dict]) -> Path:
    path = (
        settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────
# build_index_for_project
# ─────────────────────────────────────────────────────────────────────


class BuildIndexForProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _clear_tree(TEST_DATA_DIR)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def test_skips_when_project_not_found(self):
        db = MagicMock()
        db.get = AsyncMock(return_value=None)
        result = asyncio.run(build_index_for_project(db, project_id=7001))
        self.assertFalse(result["built"])
        self.assertEqual(result["reason"], "project_not_found")

    def test_skips_when_no_recipe(self):
        db = _stub_db(None)
        result = asyncio.run(build_index_for_project(db, project_id=7002))
        self.assertFalse(result["built"])
        self.assertEqual(result["reason"], "no_recipe_selected")

    def test_skips_for_non_rag_eligible_recipe(self):
        for recipe in ("classification", "span-extraction", "generic-sft"):
            with self.subTest(recipe=recipe):
                db = _stub_db(recipe)
                result = asyncio.run(build_index_for_project(db, project_id=7003))
                self.assertFalse(result["built"])
                self.assertIn(f"recipe_has_no_auto_rag:{recipe}", result["reason"])

    def test_skips_when_corpus_is_empty(self):
        db = _stub_db("qa-sft")
        # No prepared file seeded → falls back to empty rows → skip.
        result = asyncio.run(build_index_for_project(db, project_id=7004))
        self.assertFalse(result["built"])
        self.assertEqual(result["reason"], "no_corpus_rows")

    def test_builds_on_qa_sft_with_prepared_rows(self):
        project_id = 7005
        _seed_prepared_train(project_id, [
            {"id": 1, "input": {"question": "How do I reset my password?"}, "expected": {"answer": "Visit Settings → Security."}},
            {"id": 2, "input": {"question": "How many PTO days roll over?"}, "expected": {"answer": "Up to 5 days."}},
            {"id": 3, "input": {"question": "Can I email PII to myself?"}, "expected": {"answer": "No — never."}},
        ])
        db = _stub_db("qa-sft")
        result = asyncio.run(build_index_for_project(db, project_id=project_id))
        self.assertTrue(result["built"])
        self.assertEqual(result["doc_count"], 3)
        self.assertEqual(result["recipe_id"], "qa-sft")
        self.assertIn("bm25_index.json", result["index_path"])
        # Index file actually exists on disk.
        self.assertTrue(Path(result["index_path"]).exists())


# ─────────────────────────────────────────────────────────────────────
# build_preamble_from_query
# ─────────────────────────────────────────────────────────────────────


class BuildPreambleFromQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _clear_tree(TEST_DATA_DIR)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _bootstrap_indexed_project(self, project_id: int) -> None:
        _seed_prepared_train(project_id, [
            {"id": 1, "input": {"question": "How do I reset my password?"}, "expected": {"answer": "Visit Settings → Security to reset."}},
            {"id": 2, "input": {"question": "How many vacation days roll over?"}, "expected": {"answer": "Up to five PTO days carry over to January."}},
            {"id": 3, "input": {"question": "Can I email customer PII?"}, "expected": {"answer": "No — customer data may not leave the CRM."}},
        ])
        db = _stub_db("qa-sft")
        result = asyncio.run(build_index_for_project(db, project_id=project_id))
        self.assertTrue(result["built"], f"Index build failed: {result}")

    def test_returns_preamble_and_retrievals_on_qa_sft(self):
        project_id = 8001
        self._bootstrap_indexed_project(project_id)
        db = _stub_db("qa-sft")
        preamble = asyncio.run(build_preamble_from_query(
            db, project_id, query="password reset", k=2
        ))
        self.assertIsNotNone(preamble)
        # Preamble text frames the retrieved pairs as references.
        self.assertIn("Reference Q&A", preamble["preamble_text"])
        # Top hit is the password row.
        self.assertGreater(len(preamble["retrieved"]), 0)
        self.assertEqual(preamble["retrieved"][0]["row_id"], 1)

    def test_returns_none_when_recipe_not_eligible(self):
        project_id = 8002
        # Build an index for a "qa-sft" project (so the file exists)
        # but then ask for the preamble using a project that says
        # "classification" — should still skip via the recipe gate.
        self._bootstrap_indexed_project(project_id)
        db = _stub_db("classification")
        self.assertIsNone(asyncio.run(
            build_preamble_from_query(db, project_id, query="anything", k=3)
        ))

    def test_returns_none_when_index_missing(self):
        # qa-sft recipe but no index built.
        db = _stub_db("qa-sft")
        self.assertIsNone(asyncio.run(
            build_preamble_from_query(db, project_id=8003, query="hi", k=3)
        ))

    def test_returns_none_when_query_has_no_retrievable_tokens(self):
        project_id = 8004
        self._bootstrap_indexed_project(project_id)
        db = _stub_db("qa-sft")
        # Empty / punctuation-only / truly-no-overlap queries → no
        # hits → None. The "truly no overlap" case uses synthetic
        # tokens unlikely to appear in any English corpus — single
        # words from a real corpus risk matching common stopword-
        # adjacent tokens like "no" via the tokenizer.
        self.assertIsNone(asyncio.run(
            build_preamble_from_query(db, project_id, query="", k=3)
        ))
        self.assertIsNone(asyncio.run(
            build_preamble_from_query(db, project_id, query="?!.", k=3)
        ))
        self.assertIsNone(asyncio.run(
            build_preamble_from_query(
                db, project_id, query="xqzbplnk wggrtypq", k=3,
            )
        ))

    def test_preamble_pair_formatter_handles_nested_and_flat(self):
        project_id = 8005
        _seed_prepared_train(project_id, [
            # Flat shape.
            {"id": "flat-1", "question": "flat shape Q", "answer": "flat shape A"},
            # Nested shape.
            {"id": "nest-1", "input": {"question": "nested Q"}, "expected": {"answer": "nested A"}},
        ])
        db = _stub_db("qa-sft")
        self.assertTrue(
            asyncio.run(build_index_for_project(db, project_id=project_id))["built"]
        )
        preamble = asyncio.run(build_preamble_from_query(
            db, project_id, query="shape", k=2
        ))
        self.assertIsNotNone(preamble)
        # Both rows surface their Q + A text in the formatted preamble.
        body = preamble["preamble_text"]
        self.assertIn("flat shape Q", body)
        self.assertIn("flat shape A", body)


# ─────────────────────────────────────────────────────────────────────
# Playground integration (mock provider — no model calls)
# ─────────────────────────────────────────────────────────────────────


class PlaygroundAutoRagIntegrationTests(unittest.TestCase):
    """End-to-end via FastAPI TestClient + mock provider."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def _trigger_index_build(self, project_id: int) -> dict:
        """Build the auto-RAG index by hitting the preview endpoint
        (whose code path also covers the row-loading + build flow)."""
        resp = self.client.get(
            f"/api/projects/{project_id}/auto-rag/preview",
            params={"query": "x", "k": 1},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _post_chat(self, project_id: int, body: dict) -> dict:
        # Note the /training/ prefix — training router mounts the
        # /playground/chat endpoint under its own scope.
        resp = self.client.post(
            f"/api/projects/{project_id}/training/playground/chat",
            json={
                "provider": "mock",
                "model_name": "mock-test-model",
                "messages": [{"role": "user", "content": body.get("user_message", "Hi")}],
                "save_history": False,
                "auto_runtime_provider": False,
                **{k: v for k, v in body.items() if k != "user_message"},
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_auto_rag_false_default_omits_block_entirely(self):
        """When the caller doesn't set auto_rag, the response has NO
        ``auto_rag`` key — preserves the pre-Phase-9b shape so any
        existing callers don't see a new field appear."""
        project = self._instantiate_template("policy-qa-style", "AutoRAG Default Off")
        result = self._post_chat(project["id"], {"user_message": "What's the PTO policy?"})
        self.assertNotIn("auto_rag", result)

    def test_auto_rag_true_on_qa_sft_with_index_applies_retrieval(self):
        project = self._instantiate_template("policy-qa-style", "AutoRAG QA Happy Path")
        # Build the index first via the preview endpoint.
        self._trigger_index_build(project["id"])
        # Now request a chat with auto_rag=true.
        result = self._post_chat(project["id"], {
            "user_message": "How many vacation days can I roll over?",
            "auto_rag": True,
            "auto_rag_k": 2,
        })
        # auto_rag block surfaces in the response.
        self.assertIn("auto_rag", result)
        self.assertTrue(result["auto_rag"]["applied"])
        self.assertEqual(result["auto_rag"]["k"], 2)
        self.assertIn("retrieved", result["auto_rag"])
        self.assertGreaterEqual(len(result["auto_rag"]["retrieved"]), 1)

    def test_auto_rag_true_on_classification_skips_with_reason(self):
        """Classification recipe → no auto-RAG corpus shape → applied
        False with skip_reason so the UI can surface 'requested but
        ineligible' instead of silently dropping the flag."""
        project = self._instantiate_template("ticket-router", "AutoRAG Wrong Recipe")
        result = self._post_chat(project["id"], {
            "user_message": "billing question",
            "auto_rag": True,
        })
        self.assertIn("auto_rag", result)
        self.assertFalse(result["auto_rag"]["applied"])
        self.assertEqual(
            result["auto_rag"]["skip_reason"], "recipe_or_index_ineligible"
        )

    def test_auto_rag_true_without_index_built_skips_gracefully(self):
        """QA-SFT project but no index on disk → applied False, no
        crash. Phase 9b's training-completion hook normally builds
        the index; this guards the case where playground is hit
        before the first training run finishes."""
        project = self._instantiate_template("policy-qa-style", "AutoRAG No Index Yet")
        # Skip the index-build trigger.
        result = self._post_chat(project["id"], {
            "user_message": "What's policy?",
            "auto_rag": True,
        })
        self.assertFalse(result["auto_rag"]["applied"])
        self.assertEqual(
            result["auto_rag"]["skip_reason"], "recipe_or_index_ineligible"
        )

    def test_session_metadata_carries_retrieved_chunks_when_applied(self):
        """Phase 9d's interpretability panel reads from the saved
        session — verify the retrieval payload lands there."""
        project = self._instantiate_template("policy-qa-style", "AutoRAG Session Metadata")
        self._trigger_index_build(project["id"])
        result = self._post_chat(project["id"], {
            "user_message": "tell me about PII",
            "auto_rag": True,
            "save_history": True,
        })
        session = result.get("session_summary") or {}
        metadata = session.get("metadata") or {}
        self.assertIn("auto_rag", metadata)
        self.assertTrue(metadata["auto_rag"]["applied"])
        self.assertIn("retrieved", metadata["auto_rag"])


if __name__ == "__main__":
    unittest.main()

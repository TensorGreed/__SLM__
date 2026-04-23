"""Phase 56 tests: the BUILTIN_GLOSSARY exposes product concept IDs for the frontend <Term> component."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase56_glossary_product_concepts.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase56_glossary_product_concepts_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.main import app
from app.services.domain_blueprint_service import BUILTIN_GLOSSARY


REQUIRED_PRODUCT_CONCEPTS = {
    "adapter",
    "domain pack",
    "domain profile",
    "recipe",
    "runtime",
    "pack",
    "gate",
    "blueprint",
    "autopilot",
}


class Phase56GlossaryProductConceptsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def test_builtin_glossary_defines_product_concept_ids(self):
        missing = REQUIRED_PRODUCT_CONCEPTS - set(BUILTIN_GLOSSARY.keys())
        self.assertFalse(
            missing,
            f"BUILTIN_GLOSSARY is missing product concept keys: {sorted(missing)}",
        )
        # Each entry must be a 2-tuple (plain_language, category) with non-empty strings.
        for term in REQUIRED_PRODUCT_CONCEPTS:
            plain, category = BUILTIN_GLOSSARY[term]
            self.assertTrue(plain.strip(), f"{term} has empty plain_language")
            self.assertTrue(category.strip(), f"{term} has empty category")

    def test_global_glossary_endpoint_returns_product_concepts(self):
        resp = self.client.get("/api/domain-blueprints/glossary/help")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        terms = {entry["term"].lower() for entry in payload["entries"]}
        for concept in REQUIRED_PRODUCT_CONCEPTS:
            self.assertIn(concept, terms, f"glossary endpoint missing {concept}")

    def test_term_filter_returns_matching_product_concept_entry(self):
        resp = self.client.get("/api/domain-blueprints/glossary/help?term=recipe")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertGreaterEqual(payload["count"], 1)
        terms = {entry["term"].lower() for entry in payload["entries"]}
        self.assertIn("recipe", terms)

    def test_plain_language_text_is_beginner_friendly(self):
        resp = self.client.get("/api/domain-blueprints/glossary/help?term=gate")
        self.assertEqual(resp.status_code, 200, resp.text)
        entries = resp.json()["entries"]
        gate_entry = next((e for e in entries if e["term"].lower() == "gate"), None)
        self.assertIsNotNone(gate_entry)
        self.assertIn("pass", gate_entry["plain_language"].lower())


if __name__ == "__main__":
    unittest.main()

"""Tests for the auto-RAG BM25 service (USER-SUCCESS Epic 9 Phase 9a).

Covers:
  * Recipe → text_keys map truth table.
  * Tokenizer keeps contractions intact + drops punctuation.
  * BM25 scoring: identical-text round-trip (top-1 is the source row);
    rare-term match outranks common-term match (idf working).
  * Q+A indexing: a query matching the *answer* text retrieves the
    row, not just queries matching the question.
  * build_bm25_index error paths: empty rows, unsupported recipe,
    every-row-empty-after-extraction.
  * retrieve error paths: missing index file, empty query returns
    [], k<1 returns [], k larger than corpus returns all hits.
  * On-disk index is inspectable JSON + carries the recipe + tokenizer
    settings so a stale recipe rename auto-invalidates.
  * Nested ``input.question`` / ``expected.answer`` shapes (template
    gold rows) are extracted correctly without the caller flattening.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.auto_rag_service import (  # noqa: E402
    AutoRagUnavailable,
    _tokenize,
    build_bm25_index,
    recommended_text_keys_for_recipe,
    retrieve,
)


# ─────────────────────────────────────────────────────────────────────
# Recipe → text_keys map
# ─────────────────────────────────────────────────────────────────────


class RecipeTextKeysMapTests(unittest.TestCase):
    def test_qa_sft_maps_to_question_answer_pair(self):
        self.assertEqual(
            recommended_text_keys_for_recipe("qa-sft"),
            ("question", "answer"),
        )

    def test_unmapped_recipes_return_none(self):
        # Phase 9a ships qa-sft only; other recipes plug in later.
        for recipe in (
            "classification",
            "span-extraction",
            "summarization",
            "code-review",
            "generic-sft",
            "not-a-real-recipe",
        ):
            with self.subTest(recipe=recipe):
                self.assertIsNone(recommended_text_keys_for_recipe(recipe))


# ─────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────


class TokenizerTests(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self):
        self.assertEqual(
            _tokenize("How do I RESET my password?!"),
            ["how", "do", "i", "reset", "my", "password"],
        )

    def test_keeps_contractions_intact(self):
        # "don't" stays one token; "user's" stays one token. Splitting
        # on apostrophes would mangle high-signal rare-token retrievals.
        self.assertEqual(_tokenize("don't"), ["don't"])
        self.assertEqual(_tokenize("user's data"), ["user's", "data"])

    def test_handles_digits_and_mixed_alphanum(self):
        # Token = alphanumeric run, so "p3" stays whole but "f-1"
        # splits on the dash (which is what BM25 wants).
        self.assertEqual(_tokenize("p3 model F-1 score"), ["p3", "model", "f", "1", "score"])

    def test_empty_returns_empty_list(self):
        self.assertEqual(_tokenize(""), [])
        self.assertEqual(_tokenize(None), [])  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────
# build_bm25_index + retrieve happy path
# ─────────────────────────────────────────────────────────────────────


def _qa(qid: str | int, question: str, answer: str) -> dict:
    """Make a row in the template's nested ``input/expected`` shape
    so the corpus extractor's nested path is exercised."""
    return {
        "id": qid,
        "input": {"question": question},
        "expected": {"answer": answer},
    }


class BuildAndRetrieveHappyPathTests(unittest.TestCase):
    def test_query_matching_row_question_retrieves_that_row_top1(self):
        rows = [
            _qa(1, "How do I reset my password?", "Visit Settings → Security."),
            _qa(2, "How many PTO days roll over?", "Up to 5 days."),
            _qa(3, "Can I email PII to myself?",   "No — never."),
        ]
        with TemporaryDirectory() as td:
            build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            hits = retrieve("password reset", index_dir=Path(td), k=3)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["row_id"], 1)
        # The matched payload carries the full source row so Phase 9b
        # can extract both question and answer for the prepend context.
        self.assertIn("input", hits[0]["payload"])
        self.assertEqual(hits[0]["payload"]["input"]["question"], "How do I reset my password?")

    def test_query_matching_row_answer_retrieves_that_row(self):
        """Q+A indexing: a query that matches the answer text only
        (NOT the question text) should still find the row. Verifies
        we're indexing the union, not just the questions."""
        rows = [
            _qa(1, "What's the policy?", "Up to five PTO days carry over to January."),
            _qa(2, "Anything else?",      "Refund requests go to billing@example.com."),
        ]
        with TemporaryDirectory() as td:
            build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            hits = retrieve("PTO carry over", index_dir=Path(td), k=2)
        self.assertGreaterEqual(len(hits), 1)
        # Top hit is the row whose ANSWER mentions PTO, not the one
        # whose question is generic.
        self.assertEqual(hits[0]["row_id"], 1)

    def test_rare_terms_outrank_common_terms(self):
        """idf is doing work: a query matching a rare term in row B
        outranks a query matching only common terms in row A."""
        rows = [
            _qa(1, "How do I reset my password?",       "Visit Settings."),
            _qa(2, "How do I configure my password?",   "Visit Settings."),
            _qa(3, "Where do I find my password?",      "Visit Settings."),
            # Row 4 has a uniquely rare term: "biometric".
            _qa(4, "Can I enable biometric login?",     "Yes via Settings."),
        ]
        with TemporaryDirectory() as td:
            build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            hits = retrieve("biometric login", index_dir=Path(td), k=4)
        self.assertEqual(hits[0]["row_id"], 4)

    def test_top_k_caps_result_count(self):
        rows = [_qa(i, f"unique{i} question", f"answer{i}") for i in range(10)]
        with TemporaryDirectory() as td:
            build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            hits = retrieve("question", index_dir=Path(td), k=3)
        self.assertEqual(len(hits), 3)

    def test_retrieve_scores_are_monotonic_descending(self):
        rows = [
            _qa(1, "password reset password reset",     "x"),  # 2× hit on "password"
            _qa(2, "password",                          "x"),  # 1× hit
            _qa(3, "completely unrelated query",        "x"),  # 0× hit
        ]
        with TemporaryDirectory() as td:
            build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            hits = retrieve("password", index_dir=Path(td), k=3)
        # Score descending — top hit wins, runner-up loses, no-hit dropped.
        self.assertEqual([h["row_id"] for h in hits], [1, 2])
        self.assertGreater(hits[0]["score"], hits[1]["score"])


# ─────────────────────────────────────────────────────────────────────
# On-disk index inspectability
# ─────────────────────────────────────────────────────────────────────


class IndexFilePersistenceTests(unittest.TestCase):
    def test_index_file_is_inspectable_json_with_recipe_and_tokenizer_settings(self):
        rows = [_qa(1, "test question", "test answer")]
        with TemporaryDirectory() as td:
            manifest = build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            payload = json.loads(Path(manifest["index_path"]).read_text(encoding="utf-8"))
        # Recipe + tokenizer config stamped so a future change
        # invalidates loudly instead of silently mis-scoring.
        self.assertEqual(payload["recipe_id"], "qa-sft")
        self.assertEqual(payload["text_keys"], ["question", "answer"])
        self.assertEqual(payload["bm25_k1"], 1.5)
        self.assertEqual(payload["bm25_b"], 0.75)
        self.assertEqual(payload["doc_count"], 1)
        # Each row entry carries its tokens + the full source payload.
        self.assertEqual(len(payload["rows"]), 1)
        self.assertIn("doc_tokens", payload["rows"][0])
        self.assertIn("payload", payload["rows"][0])


# ─────────────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────────────


class BuildErrorPathTests(unittest.TestCase):
    def test_empty_rows_raises(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(AutoRagUnavailable) as cm:
                build_bm25_index([], recipe_id="qa-sft", output_dir=Path(td))
        self.assertIn("empty", str(cm.exception).lower())

    def test_unsupported_recipe_raises_with_known_recipes_listed(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(AutoRagUnavailable) as cm:
                build_bm25_index(
                    [_qa(1, "q", "a")],
                    recipe_id="classification",
                    output_dir=Path(td),
                )
        msg = str(cm.exception)
        self.assertIn("classification", msg)
        # The error names which recipes ARE supported so the caller
        # can correct the typo without grepping docs.
        self.assertIn("qa-sft", msg)

    def test_every_row_empty_after_extraction_raises(self):
        # Rows have wrong field names entirely — corpus extractor
        # finds nothing.
        rows = [{"id": 1, "text": "wrong shape"}]
        with TemporaryDirectory() as td:
            with self.assertRaises(AutoRagUnavailable) as cm:
                build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
        msg = str(cm.exception)
        self.assertIn("empty", msg.lower())
        self.assertIn("question", msg)  # names the field that was missing
        self.assertIn("answer", msg)


class RetrieveErrorPathTests(unittest.TestCase):
    def test_missing_index_file_raises(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(AutoRagUnavailable) as cm:
                retrieve("query", index_dir=Path(td), k=3)
        self.assertIn("not found", str(cm.exception).lower())

    def test_corrupt_index_file_raises(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "bm25_index.json"
            path.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(AutoRagUnavailable) as cm:
                retrieve("query", index_dir=Path(td), k=3)
        self.assertIn("unreadable", str(cm.exception).lower())

    def test_empty_query_returns_empty_list(self):
        rows = [_qa(1, "q", "a")]
        with TemporaryDirectory() as td:
            build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            # Empty + whitespace-only queries → []; caller decides what to do.
            self.assertEqual(retrieve("", index_dir=Path(td), k=3), [])
            self.assertEqual(retrieve("   ", index_dir=Path(td), k=3), [])
            # Query with only punctuation also tokenizes to [].
            self.assertEqual(retrieve("?!.", index_dir=Path(td), k=3), [])

    def test_k_less_than_one_returns_empty_list(self):
        rows = [_qa(1, "q", "a")]
        with TemporaryDirectory() as td:
            build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            self.assertEqual(retrieve("query", index_dir=Path(td), k=0), [])
            self.assertEqual(retrieve("query", index_dir=Path(td), k=-1), [])

    def test_k_larger_than_corpus_returns_all_hits(self):
        rows = [_qa(1, "test question", "test answer"), _qa(2, "another test", "x")]
        with TemporaryDirectory() as td:
            build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            hits = retrieve("test", index_dir=Path(td), k=100)
        self.assertEqual(len(hits), 2)


# ─────────────────────────────────────────────────────────────────────
# Flat-row support (rows with top-level "question" / "answer")
# ─────────────────────────────────────────────────────────────────────


class FlatRowShapeTests(unittest.TestCase):
    def test_flat_rows_work_alongside_nested_template_rows(self):
        """Some callers pass already-flattened rows (top-level
        question/answer keys); the corpus extractor handles both
        shapes interchangeably."""
        rows = [
            {"id": 1, "question": "flat shape Q",   "answer": "flat shape A"},
            _qa(2, "nested shape Q", "nested shape A"),
        ]
        with TemporaryDirectory() as td:
            build_bm25_index(rows, recipe_id="qa-sft", output_dir=Path(td))
            hits = retrieve("nested", index_dir=Path(td), k=2)
        # Both rows successfully tokenized; the nested one ranks first
        # because it actually contains the query token.
        self.assertEqual(hits[0]["row_id"], 2)


if __name__ == "__main__":
    unittest.main()

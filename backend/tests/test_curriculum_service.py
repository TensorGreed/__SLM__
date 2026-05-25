"""Tests for the curriculum-ranking service (USER-SUCCESS Epic 6 Phase 6a).

Covers:
- ``recommended_scoring_mode_for_recipe`` truth table
  (classification → prototype_entropy; others → None).
- ``rank_rows`` happy path:
    * Prototypical-classmate rows rank easier than outliers.
    * Singletons (single-row class) get max difficulty.
    * Multi-class input keeps per-class scoring isolated.
    * Result list preserves row_id mapping + 0-indexed rank order.
- Edge cases:
    * Empty rows → CurriculumUnavailable.
    * Unknown scoring_mode → CurriculumUnavailable.
    * Missing sentence-transformers (no injected embedder) →
      CurriculumUnavailable surfaces a useful message.
- Embedding cache: a second call with the same texts hits the cache
  (no re-embedding); adding a new row only embeds the new text.
- Text + group extraction handles nested ``expected.label`` /
  ``input.question`` shapes used by some templates.

The tests use a deterministic stub embedder so we can assert exact
ordering — sentence-transformers would produce stable but harder-to-
assert values across machines.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.curriculum_service import (  # noqa: E402
    CurriculumEntry,
    CurriculumUnavailable,
    rank_rows,
    recommended_scoring_mode_for_recipe,
)


# ─────────────────────────────────────────────────────────────────────
# Stub embedder — deterministic + small
# ─────────────────────────────────────────────────────────────────────


def _embedder_with_map(vectors_by_text: dict[str, list[float]]):
    """Returns an embedder that looks each text up in a map.
    Missing texts get a zero vector (which the service treats as
    max-difficulty)."""

    def _embed(texts: list[str]) -> list[list[float]]:
        return [
            list(vectors_by_text.get(t, [0.0, 0.0, 0.0]))
            for t in texts
        ]

    return _embed


# ─────────────────────────────────────────────────────────────────────
# recommended_scoring_mode_for_recipe
# ─────────────────────────────────────────────────────────────────────


class RecipeScoringModeMapTests(unittest.TestCase):
    def test_classification_maps_to_prototype_entropy(self):
        self.assertEqual(
            recommended_scoring_mode_for_recipe("classification"),
            "prototype_entropy",
        )

    def test_unmapped_recipes_return_none(self):
        # Phase 6a ships classification only; the others land in
        # later phases. None signals "curriculum unavailable" to
        # callers without raising.
        for recipe in (
            "qa-sft",
            "span-extraction",
            "summarization",
            "code-review",
            "generic-sft",
            "not-a-real-recipe",
        ):
            with self.subTest(recipe=recipe):
                self.assertIsNone(recommended_scoring_mode_for_recipe(recipe))


# ─────────────────────────────────────────────────────────────────────
# rank_rows — prototype_entropy
# ─────────────────────────────────────────────────────────────────────


class PrototypeEntropyRankingTests(unittest.TestCase):
    def test_prototypical_row_ranks_easier_than_outlier(self):
        """In a class of 3, the two near-duplicate rows should rank
        easier than the rotation-away outlier."""
        # Two rows nearly aligned along [1, 0, 0]; one outlier
        # rotated 90 degrees onto [0, 1, 0].
        vectors = {
            "row a": [1.0, 0.0, 0.0],
            "row b": [0.99, 0.14, 0.0],  # ~ cos 0.99 with row a
            "row c": [0.0, 1.0, 0.0],    # outlier
        }
        rows = [
            {"id": 1, "text": "row a", "label": "billing"},
            {"id": 2, "text": "row b", "label": "billing"},
            {"id": 3, "text": "row c", "label": "billing"},
        ]
        ranked = rank_rows(
            rows,
            scoring_mode="prototype_entropy",
            embedder=_embedder_with_map(vectors),
        )
        by_id = {entry["row_id"]: entry for entry in ranked}
        # The two near-duplicates are EASIER (lower difficulty) than the outlier.
        self.assertLess(by_id[1]["difficulty"], by_id[3]["difficulty"])
        self.assertLess(by_id[2]["difficulty"], by_id[3]["difficulty"])
        # Outlier ends up at the highest rank position.
        self.assertEqual(by_id[3]["rank"], 2)

    def test_multi_class_scoring_is_per_class(self):
        """A prototypical row in class A shouldn't be made hard by
        rows in class B looking different — scoring compares each row
        only to its classmates."""
        vectors = {
            "a1": [1.0, 0.0, 0.0],
            "a2": [0.99, 0.14, 0.0],   # near-duplicate of a1
            "b1": [0.0, 1.0, 0.0],
            "b2": [0.14, 0.99, 0.0],   # near-duplicate of b1
        }
        rows = [
            {"id": 1, "text": "a1", "label": "billing"},
            {"id": 2, "text": "a2", "label": "billing"},
            {"id": 3, "text": "b1", "label": "technical"},
            {"id": 4, "text": "b2", "label": "technical"},
        ]
        ranked = rank_rows(
            rows,
            scoring_mode="prototype_entropy",
            embedder=_embedder_with_map(vectors),
        )
        # Each class has two near-duplicate rows → both classes' rows
        # should have low (similar) difficulty. The four difficulties
        # should be close to each other, all well below 1.0.
        for entry in ranked:
            self.assertLess(entry["difficulty"], 0.5,
                            f"row {entry['row_id']} should be easy "
                            f"(found difficulty={entry['difficulty']})")

    def test_singleton_class_gets_max_difficulty(self):
        """A row that's alone in its class has no classmates to
        compare with — by definition an outlier → difficulty 1.0."""
        vectors = {
            "a1": [1.0, 0.0, 0.0],
            "a2": [0.99, 0.14, 0.0],
            "lone": [1.0, 0.0, 0.0],  # cos similarity = 1 to a1 but in a different class
        }
        rows = [
            {"id": 1, "text": "a1", "label": "billing"},
            {"id": 2, "text": "a2", "label": "billing"},
            {"id": 3, "text": "lone", "label": "only-class"},
        ]
        ranked = rank_rows(
            rows,
            scoring_mode="prototype_entropy",
            embedder=_embedder_with_map(vectors),
        )
        by_id = {entry["row_id"]: entry for entry in ranked}
        self.assertEqual(by_id[3]["difficulty"], 1.0)
        # And the singleton ranks last (hardest).
        self.assertEqual(by_id[3]["rank"], 2)

    def test_rank_field_is_0_indexed_and_dense(self):
        vectors = {
            "x": [1.0, 0.0, 0.0],
            "y": [0.5, 0.5, 0.0],
            "z": [0.0, 0.0, 1.0],
        }
        rows = [
            {"id": "x", "text": "x", "label": "L"},
            {"id": "y", "text": "y", "label": "L"},
            {"id": "z", "text": "z", "label": "L"},
        ]
        ranked = rank_rows(
            rows,
            scoring_mode="prototype_entropy",
            embedder=_embedder_with_map(vectors),
        )
        # Ranks form a permutation of [0, 1, 2].
        self.assertEqual(sorted(entry["rank"] for entry in ranked), [0, 1, 2])
        # row_id type round-trips (string ids).
        self.assertEqual({entry["row_id"] for entry in ranked}, {"x", "y", "z"})

    def test_falls_back_to_position_when_row_has_no_id(self):
        """Rows without ``id`` get their 0-indexed position as the
        row_id — so downstream callers can still match entries back
        to the source list."""
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
        }
        rows = [
            {"text": "a", "label": "x"},
            {"text": "b", "label": "x"},
        ]
        ranked = rank_rows(
            rows,
            scoring_mode="prototype_entropy",
            embedder=_embedder_with_map(vectors),
        )
        ids = sorted(entry["row_id"] for entry in ranked)
        self.assertEqual(ids, [0, 1])

    def test_zero_vector_row_gets_max_difficulty(self):
        """Embeddings that happen to be all-zero (very rare; usually
        means the embedder choked on empty text) can't compute cosine
        similarity — degrade gracefully to max difficulty."""
        vectors = {
            "real": [1.0, 0.0, 0.0],
            "":     [0.0, 0.0, 0.0],
        }
        rows = [
            {"id": 1, "text": "real", "label": "L"},
            {"id": 2, "text": "", "label": "L"},
        ]
        ranked = rank_rows(
            rows,
            scoring_mode="prototype_entropy",
            embedder=_embedder_with_map(vectors),
        )
        by_id = {entry["row_id"]: entry for entry in ranked}
        self.assertEqual(by_id[2]["difficulty"], 1.0)


# ─────────────────────────────────────────────────────────────────────
# Text + group extraction (template-shape compatibility)
# ─────────────────────────────────────────────────────────────────────


class TextGroupExtractionTests(unittest.TestCase):
    def test_extracts_text_from_nested_input_dict(self):
        """Templates with input: {question: "..."} shape (QA-style)
        should still find the text without the caller having to
        flatten the row first."""
        vectors = {
            "Q1": [1.0, 0.0],
            "Q2": [0.99, 0.14],
        }
        rows = [
            {"id": 1, "input": {"question": "Q1"}, "label": "L"},
            {"id": 2, "input": {"question": "Q2"}, "label": "L"},
        ]
        # No exception, and the embeddings were resolved correctly.
        ranked = rank_rows(
            rows,
            scoring_mode="prototype_entropy",
            embedder=_embedder_with_map(vectors),
        )
        self.assertEqual(len(ranked), 2)
        for entry in ranked:
            self.assertLess(entry["difficulty"], 0.5)

    def test_extracts_group_from_nested_expected_label(self):
        """Templates with expected: {label: "..."} shape should
        resolve the grouping key via the nested path."""
        vectors = {
            "a1": [1.0, 0.0],
            "a2": [0.99, 0.14],
            "b1": [0.0, 1.0],
        }
        rows = [
            {"id": 1, "text": "a1", "expected": {"label": "L"}},
            {"id": 2, "text": "a2", "expected": {"label": "L"}},
            {"id": 3, "text": "b1", "expected": {"label": "OTHER"}},
        ]
        ranked = rank_rows(
            rows,
            scoring_mode="prototype_entropy",
            embedder=_embedder_with_map(vectors),
        )
        by_id = {entry["row_id"]: entry for entry in ranked}
        # Row 3 is alone in its class → max difficulty.
        self.assertEqual(by_id[3]["difficulty"], 1.0)


# ─────────────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────────────


class CurriculumErrorPathTests(unittest.TestCase):
    def test_empty_rows_raises(self):
        with self.assertRaises(CurriculumUnavailable) as cm:
            rank_rows(
                [],
                scoring_mode="prototype_entropy",
                embedder=_embedder_with_map({}),
            )
        self.assertIn("empty", str(cm.exception).lower())

    def test_unknown_scoring_mode_raises(self):
        with self.assertRaises(CurriculumUnavailable) as cm:
            rank_rows(
                [{"id": 1, "text": "x", "label": "L"}],
                scoring_mode="not_a_real_mode",  # type: ignore[arg-type]
                embedder=_embedder_with_map({"x": [1.0, 0.0]}),
            )
        self.assertIn("scoring_mode", str(cm.exception))

    def test_missing_sentence_transformers_raises_actionable_error(self):
        """The default embedder hard-fails (rather than silently
        falling back to hashing) when sentence-transformers isn't
        installed. The error message names the package + install
        command so the user can fix it without grepping docs."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("not installed in this env")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            with self.assertRaises(CurriculumUnavailable) as cm:
                rank_rows(
                    [{"id": 1, "text": "x", "label": "L"}],
                    scoring_mode="prototype_entropy",
                )
        msg = str(cm.exception)
        self.assertIn("sentence-transformers", msg)
        self.assertIn("pip install", msg)


# ─────────────────────────────────────────────────────────────────────
# Embedding cache
# ─────────────────────────────────────────────────────────────────────


class EmbeddingCacheTests(unittest.TestCase):
    def test_second_call_with_same_texts_hits_cache(self):
        """The embedder is called once on the first rank_rows call;
        a second call with the same texts must not re-invoke it."""
        embedder_calls: list[list[str]] = []
        vectors_map = {"a": [1.0, 0.0], "b": [0.99, 0.14]}

        def counting_embedder(texts: list[str]) -> list[list[float]]:
            embedder_calls.append(list(texts))
            return [list(vectors_map[t]) for t in texts]

        rows = [
            {"id": 1, "text": "a", "label": "L"},
            {"id": 2, "text": "b", "label": "L"},
        ]
        with TemporaryDirectory() as td:
            cache_dir = Path(td)
            rank_rows(
                rows,
                scoring_mode="prototype_entropy",
                embedder=counting_embedder,
                cache_dir=cache_dir,
            )
            # Second call → cache hit, embedder NOT invoked.
            rank_rows(
                rows,
                scoring_mode="prototype_entropy",
                embedder=counting_embedder,
                cache_dir=cache_dir,
            )
        # Embedder called exactly once.
        self.assertEqual(len(embedder_calls), 1)
        self.assertEqual(sorted(embedder_calls[0]), ["a", "b"])

    def test_adding_a_new_row_only_embeds_the_new_text(self):
        """Cache is keyed per-row (text hash). Adding a new row
        should only re-embed the new text, not all rows."""
        embedder_calls: list[list[str]] = []
        vectors_map = {
            "a": [1.0, 0.0],
            "b": [0.99, 0.14],
            "c": [0.0, 1.0],
        }

        def counting_embedder(texts: list[str]) -> list[list[float]]:
            embedder_calls.append(list(texts))
            return [list(vectors_map[t]) for t in texts]

        rows_before = [
            {"id": 1, "text": "a", "label": "L"},
            {"id": 2, "text": "b", "label": "L"},
        ]
        rows_after = rows_before + [{"id": 3, "text": "c", "label": "L"}]

        with TemporaryDirectory() as td:
            cache_dir = Path(td)
            rank_rows(
                rows_before,
                scoring_mode="prototype_entropy",
                embedder=counting_embedder,
                cache_dir=cache_dir,
            )
            rank_rows(
                rows_after,
                scoring_mode="prototype_entropy",
                embedder=counting_embedder,
                cache_dir=cache_dir,
            )
        # Two calls into the embedder — first embeds {a, b}; second
        # embeds only {c} (the new row).
        self.assertEqual(len(embedder_calls), 2)
        self.assertEqual(sorted(embedder_calls[0]), ["a", "b"])
        self.assertEqual(embedder_calls[1], ["c"])

    def test_cache_file_is_inspectable_json(self):
        """JSON, not pickle/npz, so a debugging human can `cat` it."""
        vectors_map = {"a": [1.0, 0.0]}
        with TemporaryDirectory() as td:
            cache_dir = Path(td)
            rank_rows(
                [{"id": 1, "text": "a", "label": "L"}],
                scoring_mode="prototype_entropy",
                embedder=_embedder_with_map(vectors_map),
                cache_dir=cache_dir,
            )
            cache_path = cache_dir / "embeddings.json"
            self.assertTrue(cache_path.exists())
            parsed = json.loads(cache_path.read_text(encoding="utf-8"))
            # Cache holds at least one key + the value is a list of floats.
            self.assertGreaterEqual(len(parsed), 1)
            vec = next(iter(parsed.values()))
            self.assertIsInstance(vec, list)
            for v in vec:
                self.assertIsInstance(v, float)


# ─────────────────────────────────────────────────────────────────────
# CurriculumEntry shape sanity
# ─────────────────────────────────────────────────────────────────────


class CurriculumEntryShapeTests(unittest.TestCase):
    def test_entry_has_all_typeddict_keys(self):
        rows = [
            {"id": 1, "text": "a", "label": "L"},
            {"id": 2, "text": "b", "label": "L"},
        ]
        ranked = rank_rows(
            rows,
            scoring_mode="prototype_entropy",
            embedder=_embedder_with_map(
                {"a": [1.0, 0.0], "b": [0.99, 0.14]}
            ),
        )
        for entry in ranked:
            # TypedDict at runtime is just dict — assert the keys are
            # all present + types are right.
            self.assertIn("row_id", entry)
            self.assertIn("difficulty", entry)
            self.assertIn("rank", entry)
            self.assertIsInstance(entry["difficulty"], float)
            self.assertIsInstance(entry["rank"], int)
            # Difficulty stays in [0, 1].
            self.assertGreaterEqual(entry["difficulty"], 0.0)
            self.assertLessEqual(entry["difficulty"], 1.0)
        # TypedDict import is used (silences linter).
        _: CurriculumEntry = ranked[0]
        self.assertTrue(_)


if __name__ == "__main__":
    unittest.main()

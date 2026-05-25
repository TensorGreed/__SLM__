"""Tests for the curriculum training-pipeline integration
(USER-SUCCESS Epic 6 Phase 6b).

Exercises ``_maybe_apply_curriculum`` end-to-end with a stubbed
embedder so the tests don't depend on sentence-transformers being
installed. Four scenarios:

  1. ``curriculum=true`` on a classification project → the helper
     writes the shard, overrides train_file, and the returned block
     names the manifest fields the UI / debug surfaces read.
  2. ``curriculum=true`` on a non-classification project → block
     records ``skip_reason:unsupported_recipe:...``; resolved_config
     does NOT get the disable-shuffle flag; train_file unchanged.
  3. ``curriculum=true`` but sentence-transformers missing → block
     records ``skip_reason:embedder_unavailable:...``; same no-op.
  4. ``curriculum`` unset / false (default) → block reports
     ``requested=false, applied=false``; pure no-op.
  5. DPO/ORPO training modes → skipped even with curriculum=true.

The fifth case guards against a future contributor enabling
curriculum on an alignment run, which would silently misorder the
preference pairs.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.training_service import _maybe_apply_curriculum  # noqa: E402


def _stub_embedder(texts: list[str]) -> list[list[float]]:
    """3-d embeddings keyed by row text. Lets tests assert exact
    ordering without pulling sentence-transformers."""
    vectors: dict[str, list[float]] = {
        "easy A": [1.0, 0.0, 0.0],
        "easy B": [0.99, 0.14, 0.0],
        "hard outlier": [0.0, 1.0, 0.0],
        "hard far": [-0.5, -0.5, 0.7],
    }
    return [list(vectors.get(t, [0.0, 0.0, 1.0])) for t in texts]


def _write_train_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _project_stub(recipe_id: str | None):
    """Mimics a Project ORM row with ``selected_recipe`` populated."""
    project = MagicMock()
    project.selected_recipe = (
        {"recipe_id": recipe_id} if recipe_id else None
    )
    return project


def _db_stub(recipe_id: str | None):
    """Mimics an AsyncSession whose ``.get(Project, project_id)``
    returns our project stub. The real start_training also writes
    via the session; the curriculum helper only reads."""
    project = _project_stub(recipe_id)
    db = MagicMock()
    db.get = AsyncMock(return_value=project)
    return db


class MaybeApplyCurriculumScenarioTests(unittest.TestCase):
    """Direct exercise of the helper, asserting the runtime_config
    block + resolved_config mutations the training pipeline relies on."""

    def _rows(self) -> list[dict]:
        return [
            {"id": 1, "text": "easy A", "label": "L", "answer": "y"},
            {"id": 2, "text": "easy B", "label": "L", "answer": "y"},
            {"id": 3, "text": "hard outlier", "label": "L", "answer": "y"},
            {"id": 4, "text": "hard far", "label": "L", "answer": "y"},
        ]

    def test_curriculum_disabled_is_pure_noop(self):
        """When ``curriculum`` is unset (the default for every existing
        run), the helper does nothing — no file reads, no embedder
        load, no resolved_config mutation, no skip_reason."""
        resolved = {}  # no "curriculum" key at all
        with TemporaryDirectory() as td:
            out = Path(td)
            train_file = out / "train.jsonl"
            _write_train_file(train_file, self._rows())
            block = asyncio.run(
                _maybe_apply_curriculum(
                    db=_db_stub("classification"),
                    project_id=11,
                    train_file=train_file,
                    output_dir=out,
                    resolved_config=resolved,
                    training_mode="sft",
                )
            )
        self.assertEqual(block, {"requested": False, "applied": False})
        self.assertNotIn("curriculum_disable_shuffle", resolved)

    def test_classification_curriculum_writes_shard_and_overrides(self):
        """Happy path — classification recipe + working embedder."""
        resolved = {"curriculum": True}
        with TemporaryDirectory() as td:
            out = Path(td)
            train_file = out / "train.jsonl"
            _write_train_file(train_file, self._rows())
            with patch(
                "app.services.curriculum_service._sentence_transformer_embedder",
                side_effect=_stub_embedder,
            ):
                block = asyncio.run(
                    _maybe_apply_curriculum(
                        db=_db_stub("classification"),
                        project_id=11,
                        train_file=train_file,
                        output_dir=out,
                        resolved_config=resolved,
                        training_mode="sft",
                    )
                )
            shard_path = Path(str(block["shard_path"]))
            self.assertTrue(shard_path.exists())
            # easy_half (2 rows) + full set (4 rows) = 6 lines on disk.
            with shard_path.open() as f:
                lines = [line for line in f if line.strip()]
            self.assertEqual(len(lines), 6)
        # Block matches the manifest fields the UI / debug consume.
        self.assertTrue(block["requested"])
        self.assertTrue(block["applied"])
        self.assertEqual(block["scoring_mode"], "prototype_entropy")
        self.assertEqual(block["easy_count"], 2)
        self.assertEqual(block["total_rows"], 4)
        self.assertEqual(block["recipe_id"], "classification")
        # The disable-shuffle flag MUST land on resolved_config so
        # train.py knows to swap RandomSampler → SequentialSampler.
        self.assertTrue(resolved["curriculum_disable_shuffle"])

    def test_non_classification_recipe_skips_with_reason(self):
        """Today only classification has a scoring mode. Other recipes
        get skipped with a reason naming the recipe — Phase 6b
        deliberately doesn't try to invent a scoring mode for them."""
        resolved = {"curriculum": True}
        with TemporaryDirectory() as td:
            out = Path(td)
            train_file = out / "train.jsonl"
            _write_train_file(train_file, self._rows())
            block = asyncio.run(
                _maybe_apply_curriculum(
                    db=_db_stub("qa-sft"),
                    project_id=11,
                    train_file=train_file,
                    output_dir=out,
                    resolved_config=resolved,
                    training_mode="sft",
                )
            )
        self.assertTrue(block["requested"])
        self.assertFalse(block["applied"])
        self.assertIn("unsupported_recipe:qa-sft", str(block["skip_reason"]))
        self.assertNotIn("curriculum_disable_shuffle", resolved)

    def test_missing_embedder_skips_with_install_hint(self):
        """If sentence-transformers isn't installed, the helper
        records the skip but doesn't fail the run.

        We patch ``_sentence_transformer_embedder`` to raise the same
        ``CurriculumUnavailable`` it would raise on ``ImportError``;
        a `sys.modules`-level patch wouldn't work here because
        another test in the same process may have already cached the
        sentence_transformers import. We also use a distinct
        project_id so the embedding cache populated by the happy-
        path test above doesn't short-circuit the embedder call."""
        from app.services.curriculum_service import CurriculumUnavailable

        resolved = {"curriculum": True}
        with TemporaryDirectory() as td:
            out = Path(td)
            train_file = out / "train.jsonl"
            _write_train_file(train_file, self._rows())
            with patch(
                "app.services.curriculum_service._sentence_transformer_embedder",
                side_effect=CurriculumUnavailable(
                    "Curriculum ranking needs `sentence-transformers` "
                    "(≈ 22MB for all-MiniLM-L6-v2). Install it with "
                    "`pip install sentence-transformers` and retry."
                ),
            ):
                block = asyncio.run(
                    _maybe_apply_curriculum(
                        db=_db_stub("classification"),
                        project_id=991,  # distinct project_id → empty cache
                        train_file=train_file,
                        output_dir=out,
                        resolved_config=resolved,
                        training_mode="sft",
                    )
                )
        self.assertTrue(block["requested"])
        self.assertFalse(block["applied"])
        reason = str(block["skip_reason"])
        self.assertIn("embedder_unavailable", reason)
        self.assertIn("sentence-transformers", reason)
        self.assertNotIn("curriculum_disable_shuffle", resolved)

    def test_dpo_orpo_modes_skip_curriculum(self):
        """Alignment runs (DPO/ORPO) have their own dataset path with
        different shape semantics — curriculum row ordering would
        silently misorder preference pairs. Guard explicitly."""
        for mode in ("dpo", "orpo"):
            with self.subTest(training_mode=mode):
                resolved = {"curriculum": True}
                with TemporaryDirectory() as td:
                    out = Path(td)
                    train_file = out / "train.jsonl"
                    _write_train_file(train_file, self._rows())
                    block = asyncio.run(
                        _maybe_apply_curriculum(
                            db=_db_stub("classification"),
                            project_id=11,
                            train_file=train_file,
                            output_dir=out,
                            resolved_config=resolved,
                            training_mode=mode,
                        )
                    )
                self.assertTrue(block["requested"])
                self.assertFalse(block["applied"])
                self.assertIn(
                    f"unsupported_training_mode:{mode}",
                    str(block["skip_reason"]),
                )
                self.assertNotIn("curriculum_disable_shuffle", resolved)

    def test_empty_train_file_skips_with_reason(self):
        """Defensive — a malformed prepared file shouldn't crash
        start_training; record the skip and let standard training
        proceed (the data-shape gate will catch a truly empty file)."""
        resolved = {"curriculum": True}
        with TemporaryDirectory() as td:
            out = Path(td)
            train_file = out / "train.jsonl"
            train_file.touch()  # empty
            block = asyncio.run(
                _maybe_apply_curriculum(
                    db=_db_stub("classification"),
                    project_id=11,
                    train_file=train_file,
                    output_dir=out,
                    resolved_config=resolved,
                    training_mode="sft",
                )
            )
        self.assertTrue(block["requested"])
        self.assertFalse(block["applied"])
        self.assertEqual(block["skip_reason"], "train_file_empty")


class CurriculumDisableShuffleFlagTests(unittest.TestCase):
    """Sanity that train.py reads the same flag the helper sets.
    Verifies the integration contract without spinning up a real
    Trainer (which would require torch + transformers + the actual
    model — out of scope for a unit test)."""

    def test_train_py_module_reads_curriculum_disable_shuffle_from_config(self):
        """The flag is loaded into a local variable named ``config``
        and consulted before the sampler-swap block runs. We assert
        the literal key name appears in train.py — guards against a
        future rename that would silently disable the wiring."""
        train_py = Path(
            "/home/anuragj/Desktop/GitHub/__SLM__/backend/scripts/train.py"
        )
        source = train_py.read_text(encoding="utf-8")
        # The helper sets resolved_config["curriculum_disable_shuffle"];
        # train.py reads config.get("curriculum_disable_shuffle").
        self.assertIn('curriculum_disable_shuffle', source)
        self.assertIn("SequentialSampler", source)
        # The swap is wrapped in a guard so unknown trainer versions
        # don't crash training.
        self.assertIn("Curriculum sampler swap failed", source)


if __name__ == "__main__":
    unittest.main()

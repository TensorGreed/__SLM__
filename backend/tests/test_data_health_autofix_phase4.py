"""Phase-4 additions to the data-side autofix engine.

Covers the five new fix kinds end-to-end (preview → apply → idempotent
re-run) plus the data_health signals that emit each autofix_kind.

Each fix operates on the cleaned-text file referenced by
``metadata_.cleaned_path``; tests seed both a RawDocument row and a
real on-disk file so the rewrite path exercises actual I/O.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "autofix_phase4.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "autofix_phase4_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import (  # noqa: E402
    Dataset, DatasetType, DocumentStatus, RawDocument,
)


def _signal_by_id(body: dict, signal_id: str) -> dict | None:
    for group in body["groups"]:
        for sig in group["signals"]:
            if sig["id"] == signal_id:
                return sig
    return None


class AutofixPhase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    # ── helpers ─────────────────────────────────────────────────────

    def _create_project(self) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"af-p4-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _add_cleaned_docs(
        self,
        project_id: int,
        *,
        cleaned_contents: list[str],
    ) -> list[int]:
        """Seed N ACCEPTED RawDocuments with on-disk cleaned files
        containing ``cleaned_contents[i]``. Returns the doc ids."""
        project_dir = TEST_DATA_DIR / "projects" / str(project_id) / "raw"
        project_dir.mkdir(parents=True, exist_ok=True)
        cleaned_paths: list[Path] = []
        for i, _ in enumerate(cleaned_contents):
            cleaned_path = project_dir / f"doc-{i}.cleaned.txt"
            cleaned_paths.append(cleaned_path)
            cleaned_path.write_text(cleaned_contents[i], encoding="utf-8")

        async def _add() -> list[int]:
            async with async_session_factory() as db:
                ds = Dataset(
                    project_id=project_id,
                    name="raw",
                    dataset_type=DatasetType.RAW,
                    file_path="",
                    record_count=len(cleaned_contents),
                )
                db.add(ds)
                await db.flush()
                ids: list[int] = []
                from app.services.cleaning_service import compute_text_hash
                for i, content in enumerate(cleaned_contents):
                    file_path = project_dir / f"doc-{i}.txt"
                    file_path.write_text(f"raw {i}", encoding="utf-8")
                    doc = RawDocument(
                        dataset_id=ds.id,
                        filename=f"doc-{i}.txt",
                        file_type="txt",
                        file_path=str(file_path),
                        file_size_bytes=10,
                        source="upload",
                        sensitivity="internal",
                        status=DocumentStatus.ACCEPTED,
                        quality_score=0.8,
                        chunk_count=2,
                        metadata_={
                            "cleaned_path": str(cleaned_paths[i]),
                            "text_hash": compute_text_hash(content),
                        },
                    )
                    db.add(doc)
                    await db.flush()
                    ids.append(int(doc.id))
                await db.commit()
                return ids
        return asyncio.run(_add())

    def _read_cleaned(self, project_id: int, doc_index: int) -> str:
        path = (
            TEST_DATA_DIR / "projects" / str(project_id) / "raw"
            / f"doc-{doc_index}.cleaned.txt"
        )
        return path.read_text(encoding="utf-8")

    # ── strip_html ──────────────────────────────────────────────────

    def test_strip_html_preview_and_apply(self):
        pid = self._create_project()
        self._add_cleaned_docs(
            pid,
            cleaned_contents=[
                "<p>Hello <b>world</b></p>",   # has tags
                "Plain text without tags.",     # untouched
                "<script>nope()</script>after",  # script block dropped
            ],
        )
        preview = self.client.post(
            f"/api/projects/{pid}/data-health/autofix/preview",
            json={"fix_kind": "strip_html"},
        ).json()
        self.assertEqual(preview["fix_kind"], "strip_html")
        self.assertEqual(preview["would_apply_count"], 2)
        self.assertTrue(preview["safe_to_apply"])

        result = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "strip_html"},
        ).json()
        self.assertEqual(result["applied_count"], 2)
        # Tags removed; entities decoded; plain doc untouched.
        self.assertEqual(self._read_cleaned(pid, 0), "Hello world")
        self.assertEqual(self._read_cleaned(pid, 1), "Plain text without tags.")
        self.assertEqual(self._read_cleaned(pid, 2), "after")

        # Idempotent: second apply finds nothing to change.
        result2 = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "strip_html"},
        ).json()
        self.assertEqual(result2["applied_count"], 0)

    # ── normalize_whitespace ────────────────────────────────────────

    def test_normalize_whitespace_collapses_runs_and_blank_lines(self):
        pid = self._create_project()
        self._add_cleaned_docs(
            pid,
            cleaned_contents=[
                "Hello    world   \n\n\n\nGoodbye  ",
                "Already normalised text.",
                "Excess    blanks   here\n  ",
            ],
        )
        result = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "normalize_whitespace"},
        ).json()
        self.assertEqual(result["applied_count"], 2)
        # Doc 0: runs collapsed, blank-line burst trimmed, trailing
        # spaces stripped.
        self.assertEqual(self._read_cleaned(pid, 0), "Hello world\n\nGoodbye")
        # Doc 1 (already normalised) stays as-is.
        self.assertEqual(self._read_cleaned(pid, 1), "Already normalised text.")
        # Doc 2: runs collapsed + trailing whitespace stripped.
        self.assertEqual(self._read_cleaned(pid, 2), "Excess blanks here")

    # ── length_cap ──────────────────────────────────────────────────

    def test_length_cap_truncates_to_max_seq_length(self):
        pid = self._create_project()
        # Default TrainingConfig().max_seq_length = 2048 → cap = 8192 chars.
        # Doc 0: under cap (kept). Doc 1: way over cap (truncated).
        long_text = "a" * 12000
        self._add_cleaned_docs(
            pid,
            cleaned_contents=[
                "short doc",
                long_text,
            ],
        )
        preview = self.client.post(
            f"/api/projects/{pid}/data-health/autofix/preview",
            json={"fix_kind": "length_cap"},
        ).json()
        self.assertEqual(preview["would_apply_count"], 1)
        self.assertEqual(preview["details"]["cap_chars"], 8192)

        result = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "length_cap"},
        ).json()
        self.assertEqual(result["applied_count"], 1)
        self.assertEqual(self._read_cleaned(pid, 0), "short doc")
        self.assertEqual(len(self._read_cleaned(pid, 1)), 8192)

    # ── near_duplicate_dedup ────────────────────────────────────────

    def test_near_duplicate_dedup_drops_paraphrases(self):
        pid = self._create_project()
        # Three docs share the same opening that fully fills the
        # 500-char prefix window; the unique trailing content sits
        # AFTER the prefix and doesn't differentiate the soft hash.
        # exact text_hash differs across the three because of trailing
        # content + punctuation, so the existing exact-hash dedup
        # would miss them. The 4th is unrelated.
        shared_opening = (
            "The quick brown fox jumps over the lazy dog. "
            * 20  # ~ 900 chars — fills the 500-char prefix
        )
        self._add_cleaned_docs(
            pid,
            cleaned_contents=[
                shared_opening + " Trailer A with very different unique content.",
                shared_opening + " Trailer B with completely different ending!",
                shared_opening + " Trailer C also wholly distinct?",
                "Completely unrelated content about apples and trains.",
            ],
        )
        preview = self.client.post(
            f"/api/projects/{pid}/data-health/autofix/preview",
            json={"fix_kind": "near_duplicate_dedup"},
        ).json()
        self.assertEqual(preview["would_apply_count"], 2)
        result = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "near_duplicate_dedup"},
        ).json()
        self.assertEqual(result["applied_count"], 2)
        # Idempotent: second apply is a no-op.
        result2 = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "near_duplicate_dedup"},
        ).json()
        self.assertEqual(result2["applied_count"], 0)

    # ── normalize_schema ────────────────────────────────────────────

    def test_normalize_schema_renames_gold_field_variants(self):
        pid = self._create_project()
        # Seed a GOLD_DEV file with mixed canonical + non-canonical
        # field names. _normalize_schema renames `class` → `label`
        # and `text` → `input`.
        gold_dir = TEST_DATA_DIR / "projects" / str(pid) / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)
        gold_path = gold_dir / "dev.jsonl"
        rows = [
            {"text": "Doc A", "class": "pos"},        # both rename
            {"input": "Doc B", "label": "neg"},       # already canonical
            {"text": "Doc C", "label": "pos"},        # one rename
            {"input": "Doc D", "class": "neg"},       # one rename
        ]
        with gold_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        async def _add():
            async with async_session_factory() as db:
                db.add(Dataset(
                    project_id=pid,
                    name="gold_dev",
                    dataset_type=DatasetType.GOLD_DEV,
                    file_path=str(gold_path),
                    record_count=len(rows),
                ))
                await db.commit()
        asyncio.run(_add())

        preview = self.client.post(
            f"/api/projects/{pid}/data-health/autofix/preview",
            json={"fix_kind": "normalize_schema"},
        ).json()
        # 2 `text→input` + 2 `class→label` = 4 renames.
        self.assertEqual(preview["would_apply_count"], 4)

        result = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "normalize_schema"},
        ).json()
        self.assertEqual(result["applied_count"], 4)
        # File now has canonical names everywhere.
        new_rows = []
        with gold_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    new_rows.append(json.loads(line))
        for row in new_rows:
            self.assertIn("label", row)
            self.assertIn("input", row)
            self.assertNotIn("class", row)
            self.assertNotIn("text", row)

        # Idempotent re-run.
        result2 = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "normalize_schema"},
        ).json()
        self.assertEqual(result2["applied_count"], 0)

    # ── Signals emit the autofix_kinds ─────────────────────────────

    def test_data_health_emits_strip_html_signal(self):
        pid = self._create_project()
        # Seed enough docs above the 5% warn threshold (just 1 doc
        # with HTML tags out of 5 is 20%, well above).
        self._add_cleaned_docs(
            pid,
            cleaned_contents=[
                "<p>tags here</p>",
                "no tags",
                "no tags",
                "no tags",
                "no tags",
            ],
        )
        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        sig = _signal_by_id(body, "cleaning.html_tags_present")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["autofix_kind"], "strip_html")

    def test_data_health_emits_normalize_schema_signal(self):
        pid = self._create_project()
        gold_dir = TEST_DATA_DIR / "projects" / str(pid) / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)
        gold_path = gold_dir / "dev.jsonl"
        with gold_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "x", "class": "y"}) + "\n")

        async def _add():
            async with async_session_factory() as db:
                db.add(Dataset(
                    project_id=pid,
                    name="gold_dev",
                    dataset_type=DatasetType.GOLD_DEV,
                    file_path=str(gold_path),
                    record_count=1,
                ))
                # Also need a recipe so shape_group runs the gold
                # schema scan path (the no-recipe early return would
                # otherwise short-circuit).
                from app.models.project import Project
                proj = await db.get(Project, pid)
                proj.selected_recipe = {"recipe_id": "classification"}
                await db.commit()
        asyncio.run(_add())

        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        sig = _signal_by_id(body, "shape.gold_field_variants")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["autofix_kind"], "normalize_schema")

    # ── Dispatcher exposes phase-4 kinds ────────────────────────────

    def test_supported_endpoint_lists_phase4_kinds(self):
        pid = self._create_project()
        resp = self.client.get(
            f"/api/projects/{pid}/data-health/autofix/supported"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        kinds = resp.json()["fix_kinds"]
        for k in (
            "near_duplicate_dedup", "normalize_whitespace",
            "strip_html", "length_cap", "normalize_schema",
        ):
            self.assertIn(k, kinds)

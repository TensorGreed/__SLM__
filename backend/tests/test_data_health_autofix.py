"""D3 of the data-quality arc — safe auto-fix engine.

Covers each of the three D3 fixes:
- drop_failed_docs: removes RawDocuments with status=ERROR + their files.
- dedupe_duplicate_docs: keeps lowest-id of each text_hash, drops the rest.
- redact_pii: re-cleans every doc with PII findings + redact_pii=False.

Plus:
- Idempotency (running the same fix twice = 0 changes the second time).
- Unknown fix_kind → 400.
- Unknown project → 404.
- The data-health report's parse_failure_rate / pii_unredacted /
  duplicate_chunks signals carry the right autofix_kind hint so the
  panel can render the button.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "data_health_autofix.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "data_health_autofix_data"

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


class D3AutofixTests(unittest.TestCase):
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

    def _create_project(self) -> int:
        resp = self.client.post("/api/projects", json={"name": f"d3-{uuid.uuid4().hex[:8]}"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _add_docs(
        self,
        project_id: int,
        *,
        records: list[dict],
    ) -> list[int]:
        """records: list of {status: DocumentStatus, filename: str, hash?: str, pii?: bool, redact?: bool, cleaned?: bool}"""
        async def _add():
            async with async_session_factory() as db:
                ds = Dataset(
                    project_id=project_id,
                    name="raw",
                    dataset_type=DatasetType.RAW,
                    file_path="",
                    record_count=len(records),
                )
                db.add(ds)
                await db.flush()
                ids: list[int] = []
                for rec in records:
                    meta: dict = {}
                    if rec.get("cleaned"):
                        meta["cleaned_path"] = f"/tmp/{rec['filename']}.cleaned.txt"
                    if rec.get("hash"):
                        meta["text_hash"] = rec["hash"]
                    if rec.get("pii"):
                        meta["pii_findings"] = [
                            {"type": "email", "match": "a@b.com", "position": 0}
                        ] * 3
                    if rec.get("redact"):
                        meta["redact_pii"] = True
                    # Write a real on-disk file so the autofix delete
                    # has something to unlink (exercises the file
                    # cleanup path).
                    file_path = TEST_DATA_DIR / "projects" / str(project_id) / "raw" / rec["filename"]
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(f"content for {rec['filename']}")
                    doc = RawDocument(
                        dataset_id=ds.id,
                        filename=rec["filename"],
                        file_type="txt",
                        file_path=str(file_path),
                        file_size_bytes=10,
                        source="upload",
                        sensitivity="internal",
                        status=rec["status"],
                        chunk_count=2 if rec.get("cleaned") else 0,
                        metadata_=meta,
                    )
                    db.add(doc)
                    await db.flush()
                    ids.append(int(doc.id))
                await db.commit()
                return ids
        return asyncio.run(_add())

    # ── drop_failed_docs ───────────────────────────────────────

    def test_drop_failed_docs_removes_error_status_rows_and_files(self):
        pid = self._create_project()
        self._add_docs(pid, records=[
            {"status": DocumentStatus.ACCEPTED, "filename": "good-1.txt"},
            {"status": DocumentStatus.ERROR, "filename": "bad-1.pdf"},
            {"status": DocumentStatus.ERROR, "filename": "bad-2.pdf"},
            {"status": DocumentStatus.ACCEPTED, "filename": "good-2.txt"},
        ])
        # File on disk exists for one of the doomed docs.
        bad_file = TEST_DATA_DIR / "projects" / str(pid) / "raw" / "bad-1.pdf"
        self.assertTrue(bad_file.exists())

        resp = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "drop_failed_docs"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["fix_kind"], "drop_failed_docs")
        self.assertEqual(body["applied_count"], 2)
        self.assertEqual(
            sorted(body["details"]["dropped_filenames"]),
            ["bad-1.pdf", "bad-2.pdf"],
        )
        # Files removed from disk too.
        self.assertFalse(bad_file.exists())

    def test_drop_failed_docs_is_idempotent(self):
        """Running the same fix twice = 0 changes the second time."""
        pid = self._create_project()
        self._add_docs(pid, records=[
            {"status": DocumentStatus.ERROR, "filename": "bad.pdf"},
        ])
        resp1 = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "drop_failed_docs"},
        )
        self.assertEqual(resp1.json()["applied_count"], 1)
        resp2 = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "drop_failed_docs"},
        )
        self.assertEqual(resp2.json()["applied_count"], 0)
        self.assertIn("No failed documents", resp2.json()["summary"])

    # ── dedupe_duplicate_docs ──────────────────────────────────

    def test_dedupe_keeps_first_and_drops_the_rest(self):
        pid = self._create_project()
        self._add_docs(pid, records=[
            # Three docs share one hash; lowest-id (uniqueA) is kept.
            {"status": DocumentStatus.ACCEPTED, "filename": "uniqueA.txt", "cleaned": True, "hash": "hash-A"},
            {"status": DocumentStatus.ACCEPTED, "filename": "dupA-1.txt", "cleaned": True, "hash": "hash-A"},
            {"status": DocumentStatus.ACCEPTED, "filename": "dupA-2.txt", "cleaned": True, "hash": "hash-A"},
            # One-off — not a duplicate.
            {"status": DocumentStatus.ACCEPTED, "filename": "uniqueB.txt", "cleaned": True, "hash": "hash-B"},
            # Two more share a different hash.
            {"status": DocumentStatus.ACCEPTED, "filename": "uniqueC.txt", "cleaned": True, "hash": "hash-C"},
            {"status": DocumentStatus.ACCEPTED, "filename": "dupC-1.txt", "cleaned": True, "hash": "hash-C"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "dedupe_duplicate_docs"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # 3 dups dropped (2 from hash-A + 1 from hash-C), 2 groups affected.
        self.assertEqual(body["applied_count"], 3)
        self.assertEqual(body["details"]["group_count"], 2)
        dropped = set(body["details"]["dropped_filenames"])
        self.assertEqual(dropped, {"dupA-1.txt", "dupA-2.txt", "dupC-1.txt"})

    def test_dedupe_skips_docs_without_text_hash(self):
        """Docs with no text_hash (e.g. cleaning hasn't run) aren't
        considered duplicates — the autofix can only act on docs whose
        content was hashed."""
        pid = self._create_project()
        self._add_docs(pid, records=[
            {"status": DocumentStatus.ACCEPTED, "filename": "a.txt"},  # no hash
            {"status": DocumentStatus.ACCEPTED, "filename": "b.txt"},  # no hash
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "dedupe_duplicate_docs"},
        )
        self.assertEqual(resp.json()["applied_count"], 0)

    # ── redact_pii ─────────────────────────────────────────────

    def test_redact_pii_skips_already_redacted_docs(self):
        """A doc with redact_pii=True is treated as done — no re-clean."""
        pid = self._create_project()
        self._add_docs(pid, records=[
            {"status": DocumentStatus.ACCEPTED, "filename": "already.txt",
             "cleaned": True, "pii": True, "redact": True},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "redact_pii"},
        )
        self.assertEqual(resp.json()["applied_count"], 0)

    def test_redact_pii_skips_uncleaned_docs(self):
        """A doc with PII but no cleaned_path hasn't gone through
        cleaning yet — autofix doesn't re-run from scratch (would
        surprise the user). They need to click Clean first."""
        pid = self._create_project()
        self._add_docs(pid, records=[
            {"status": DocumentStatus.ACCEPTED, "filename": "uncleaned.txt",
             "cleaned": False, "pii": True},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "redact_pii"},
        )
        self.assertEqual(resp.json()["applied_count"], 0)

    # ── PII safeguard for span-extraction projects ────────────

    def _set_recipe(self, project_id: int, recipe_id: str) -> None:
        async def _set():
            async with async_session_factory() as db:
                from app.models.project import Project
                proj = await db.get(Project, project_id)
                proj.selected_recipe = {"recipe_id": recipe_id}
                await db.commit()
        asyncio.run(_set())

    def test_redact_pii_signal_silenced_for_span_extraction_project(self):
        """A structured_extraction recipe (PII detection, NER, etc.)
        needs PII in source documents to teach the model. The
        data-health report should report the signal as ok and refuse
        the auto-fix — the model would lose its training signal."""
        pid = self._create_project()
        # Find a real structured_extraction recipe — the platform's
        # built-in pii.spans recipe maps to task_profile=structured_extraction.
        from app.services.recipe_service import list_recipes
        se_recipes = [
            r for r in list_recipes() if getattr(r, "task_profile", None) == "structured_extraction"
        ]
        self.assertGreater(len(se_recipes), 0, "no structured_extraction recipe found in catalog")
        self._set_recipe(pid, se_recipes[0].id)
        # Seed docs with PII findings + cleaned but no redact flag —
        # exactly the state that would normally fire the warn signal.
        self._add_docs(pid, records=[
            {"status": DocumentStatus.ACCEPTED, "filename": "u1.txt", "cleaned": True, "pii": True},
            {"status": DocumentStatus.ACCEPTED, "filename": "u2.txt", "cleaned": True, "pii": True},
        ])

        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        pii = _signal_by_id(body, "cleaning.pii_unredacted")
        self.assertIsNotNone(pii)
        assert pii is not None
        # Severity flipped to ok — the platform isn't telling the user
        # "this is wrong" when for their use case it's right.
        self.assertEqual(pii["severity"], "ok")
        # No auto-fix offered.
        self.assertIsNone(pii["autofix_kind"])
        # Plain-English explains why PII is kept (so the user
        # understands the data-health UX, not a bug).
        self.assertIn("span-extraction", pii["plain_english"])
        # Context carries the blocked-reason so logs / debugging can
        # trace why the autofix didn't fire.
        self.assertEqual(pii["context"]["autofix_blocked_reason"], "span_extraction_needs_pii")

    def test_redact_pii_autofix_refuses_for_span_extraction_project(self):
        """Even if the API is called directly (bypassing the panel
        which hides the button), the autofix endpoint refuses with
        400 + a clear explanation. Defence in depth — the signal
        logic is the primary guard; this is the safety net."""
        pid = self._create_project()
        from app.services.recipe_service import list_recipes
        se_recipes = [
            r for r in list_recipes() if getattr(r, "task_profile", None) == "structured_extraction"
        ]
        self._set_recipe(pid, se_recipes[0].id)
        self._add_docs(pid, records=[
            {"status": DocumentStatus.ACCEPTED, "filename": "u1.txt", "cleaned": True, "pii": True},
        ])

        resp = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "redact_pii"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        body = resp.json()
        # Error message names the protected use case in plain language.
        self.assertIn("span-extraction", body["detail"])
        self.assertIn("training signal", body["detail"])

    # ── error paths ────────────────────────────────────────────

    def test_unknown_fix_kind_returns_400(self):
        pid = self._create_project()
        resp = self.client.post(
            f"/api/projects/{pid}/data-health/autofix",
            json={"fix_kind": "make-coffee"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_unknown_project_returns_404(self):
        resp = self.client.post(
            "/api/projects/999999/data-health/autofix",
            json={"fix_kind": "drop_failed_docs"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_supported_kinds_endpoint(self):
        pid = self._create_project()
        body = self.client.get(
            f"/api/projects/{pid}/data-health/autofix/supported"
        ).json()
        self.assertIn("drop_failed_docs", body["fix_kinds"])
        self.assertIn("dedupe_duplicate_docs", body["fix_kinds"])
        self.assertIn("redact_pii", body["fix_kinds"])

    # ── report-level integration ───────────────────────────────

    def test_report_signals_carry_autofix_kind_hint(self):
        """The data-health report's three D3-eligible signals should
        each surface an autofix_kind hint so the panel can render
        the button. Other signals leave it null."""
        pid = self._create_project()
        # Seed enough state to fire all three: one failed doc, two
        # duplicates, two PII-with-no-redact.
        self._add_docs(pid, records=[
            {"status": DocumentStatus.ERROR, "filename": "bad.pdf"},
            {"status": DocumentStatus.ACCEPTED, "filename": "u1.txt", "cleaned": True, "hash": "h", "pii": True},
            {"status": DocumentStatus.ACCEPTED, "filename": "u2.txt", "cleaned": True, "hash": "h", "pii": True},
        ])
        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        # The three D3-eligible signals.
        parse = _signal_by_id(body, "ingestion.parse_failure_rate")
        self.assertEqual(parse["autofix_kind"], "drop_failed_docs")
        pii = _signal_by_id(body, "cleaning.pii_unredacted")
        self.assertEqual(pii["autofix_kind"], "redact_pii")
        dup = _signal_by_id(body, "cleaning.duplicate_chunks")
        self.assertEqual(dup["autofix_kind"], "dedupe_duplicate_docs")
        # Other signals (shape group, no_documents block) have None.
        no_recipe = _signal_by_id(body, "shape.no_recipe_selected")
        self.assertIsNone(no_recipe["autofix_kind"])


if __name__ == "__main__":
    unittest.main()

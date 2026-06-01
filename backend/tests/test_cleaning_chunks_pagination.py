"""GET /cleaning/chunks pagination + sampling contract.

Pins (the endpoint used to stream every chunk in one shot — for a 74k
chunk project that's ~37MB of JSON per click; this test exists so we
never regress to that):

- Empty project returns {chunks: [], total: 0, returned: 0}.
- Random-sample mode returns at most ``limit`` chunks, regardless of
  pool size, with ``total`` reflecting the full count.
- Random-sample with the same ``seed`` is deterministic.
- Random-sample without a seed yields different rows on successive
  calls when the pool is much larger than the limit.
- Paginated mode (random_sample=false) honors ``offset`` + ``limit``
  and respects ``total`` as the full count.
- ``limit=0`` returns total + no rows (count-only mode).
- ``limit < 0`` / ``limit > 5000`` / ``offset < 0`` are 400s.
- ``document_id`` is injected into each chunk dict so the UI can
  group / link back to its source document.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import (  # noqa: E402
    Dataset,
    DatasetType,
    RawDocument,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-chunks-pagination-{uuid.uuid4().hex[:8]}"
)


def _write_chunks_file(file_path: Path, chunks: list[dict]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")


async def _seed_doc_with_chunks(
    project_id: int, doc_label: str, chunks: list[dict]
) -> None:
    """Materialize one RawDocument row + its .chunks.jsonl on disk."""
    doc_file = TEST_DATA_DIR / f"{doc_label}.txt"
    chunks_file = doc_file.with_suffix(".chunks.jsonl")
    doc_file.touch()
    _write_chunks_file(chunks_file, chunks)

    async with async_session_factory() as session:
        ds = Dataset(
            project_id=project_id,
            name=f"{doc_label}-dataset",
            dataset_type=DatasetType.RAW,
        )
        session.add(ds)
        await session.flush()
        doc = RawDocument(
            dataset_id=ds.id,
            filename=doc_file.name,
            file_type="txt",
            file_path=str(doc_file),
        )
        session.add(doc)
        await session.commit()


def _seed(project_id: int, doc_label: str, chunks: list[dict]) -> None:
    asyncio.run(_seed_doc_with_chunks(project_id, doc_label, chunks))


async def _seed_dataset_import_rows(
    project_id: int, label: str, rows: list[dict], dataset_type: DatasetType = DatasetType.SYNTHETIC,
) -> None:
    """Mimic ``/dataset-import/run`` — write labelled JSONL + register a Dataset row.
    No RawDocument, no .chunks.jsonl. This is the path that produced
    the original 'load from cleaned data' bug: 30K labelled Kaggle
    rows existed but the chunks endpoint couldn't see them because it
    only scanned RawDocument.chunks.jsonl files."""
    ds_path = TEST_DATA_DIR / f"projects/{project_id}/{label}.jsonl"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    with ds_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    async with async_session_factory() as session:
        ds = Dataset(
            project_id=project_id,
            name=label,
            dataset_type=dataset_type,
            file_path=str(ds_path),
            record_count=len(rows),
        )
        session.add(ds)
        await session.commit()


def _seed_imported(project_id: int, label: str, rows: list[dict], dataset_type: DatasetType = DatasetType.SYNTHETIC) -> None:
    asyncio.run(_seed_dataset_import_rows(project_id, label, rows, dataset_type))


class ChunksPaginationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _create_project(self, label: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"{label}-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    # ── Empty project ─────────────────────────────────────────────

    def test_empty_project_returns_empty(self):
        pid = self._create_project("chunks-empty")
        resp = self.client.get(f"/api/projects/{pid}/cleaning/chunks")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["total"], 0)
        self.assertEqual(body["returned"], 0)
        self.assertEqual(body["chunks"], [])

    # ── Random sampling ───────────────────────────────────────────

    def test_random_sample_caps_to_limit(self):
        pid = self._create_project("chunks-sample-cap")
        rows = [
            {"chunk_id": i, "text": f"chunk {i}", "source_doc": "doc-a.txt"}
            for i in range(500)
        ]
        _seed(pid, f"doc-a-{pid}", rows)

        resp = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 50, "random_sample": True},
        )
        body = resp.json()
        self.assertEqual(body["total"], 500)
        self.assertEqual(body["returned"], 50)
        self.assertEqual(len(body["chunks"]), 50)
        # document_id injected on each chunk.
        for chunk in body["chunks"]:
            self.assertIn("document_id", chunk)

    def test_random_sample_with_seed_is_deterministic(self):
        pid = self._create_project("chunks-sample-seed")
        rows = [
            {"chunk_id": i, "text": f"c-{i}", "source_doc": "s.txt"}
            for i in range(200)
        ]
        _seed(pid, f"doc-seed-{pid}", rows)

        a = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 20, "random_sample": True, "seed": 42},
        ).json()
        b = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 20, "random_sample": True, "seed": 42},
        ).json()
        self.assertEqual(
            [c["chunk_id"] for c in a["chunks"]],
            [c["chunk_id"] for c in b["chunks"]],
        )

    def test_random_sample_without_seed_varies(self):
        pid = self._create_project("chunks-sample-varies")
        rows = [
            {"chunk_id": i, "text": f"c-{i}", "source_doc": "s.txt"}
            for i in range(500)
        ]
        _seed(pid, f"doc-varies-{pid}", rows)

        # Two unseeded samples of size 20 from a pool of 500 — the
        # probability of matching exactly is astronomically small.
        a = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 20, "random_sample": True},
        ).json()
        b = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 20, "random_sample": True},
        ).json()
        self.assertNotEqual(
            [c["chunk_id"] for c in a["chunks"]],
            [c["chunk_id"] for c in b["chunks"]],
        )

    # ── Paginated ─────────────────────────────────────────────────

    def test_paginated_respects_offset_and_limit(self):
        pid = self._create_project("chunks-paginated")
        rows = [
            {"chunk_id": i, "text": f"c-{i}", "source_doc": "s.txt"}
            for i in range(120)
        ]
        _seed(pid, f"doc-paged-{pid}", rows)

        page1 = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 25, "offset": 0, "random_sample": False},
        ).json()
        page2 = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 25, "offset": 25, "random_sample": False},
        ).json()
        self.assertEqual(page1["total"], 120)
        self.assertEqual(page1["returned"], 25)
        self.assertEqual(page2["returned"], 25)
        # No overlap between consecutive pages.
        ids1 = {c["chunk_id"] for c in page1["chunks"]}
        ids2 = {c["chunk_id"] for c in page2["chunks"]}
        self.assertFalse(ids1 & ids2)

    def test_limit_zero_returns_count_only(self):
        pid = self._create_project("chunks-count-only")
        rows = [
            {"chunk_id": i, "text": f"c-{i}", "source_doc": "s.txt"}
            for i in range(73)
        ]
        _seed(pid, f"doc-count-{pid}", rows)

        body = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 0, "random_sample": True},
        ).json()
        self.assertEqual(body["total"], 73)
        self.assertEqual(body["returned"], 0)
        self.assertEqual(body["chunks"], [])

    # ── Validation ────────────────────────────────────────────────

    def test_negative_limit_rejected(self):
        pid = self._create_project("chunks-neg-limit")
        resp = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": -1},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_limit_too_high_rejected(self):
        pid = self._create_project("chunks-too-high")
        resp = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 99_999},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_negative_offset_rejected(self):
        pid = self._create_project("chunks-neg-offset")
        resp = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"offset": -1, "random_sample": False},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    # ── Dataset-import fallback ───────────────────────────────────

    def test_chunks_surface_dataset_import_rows_alongside_chunks(self):
        """Regression for the 'fake CSV in cleaned data' bug.

        The user's SQLi-detector project had 30K labelled Kaggle rows
        ingested via /dataset-import/run, plus a single one-document
        'unlock.txt' that went through the cleaning pipeline. The
        Synth panel's 'Load from Cleaned Data' button only surfaced
        the unlock.txt chunk because /cleaning/chunks scanned just
        the RawDocument .chunks.jsonl files. The fix sweeps SYNTHETIC
        + CLEANED Datasets too and treats each labelled row as a
        pseudo-chunk."""
        pid = self._create_project("chunks-import-fallback")
        # One real document chunk.
        _seed(pid, f"doc-{pid}", [
            {"chunk_id": 0, "text": "real chunk from a cleaned document"},
        ])
        # 50 labelled rows from a dataset-import path.
        _seed_imported(pid, "kaggle_corpus", [
            {"text": f"' OR 1=1-- ({i})", "label": "injection"}
            for i in range(50)
        ])

        resp = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 5000, "random_sample": False},
        )
        body = resp.json()
        self.assertEqual(body["total"], 51, body)
        paths = [c.get("ingest_path") for c in body["chunks"]]
        self.assertIn("document", paths)
        self.assertIn("dataset_import", paths)
        # Imported rows carry the label through so the picker can
        # filter on it.
        imported = [c for c in body["chunks"] if c.get("ingest_path") == "dataset_import"]
        self.assertEqual(len(imported), 50)
        self.assertEqual(imported[0]["label"], "injection")

    def test_chunks_silent_when_dataset_jsonl_missing_on_disk(self):
        """If the Dataset row says file_path=foo.jsonl but the file is
        gone, skip it silently instead of 500ing. This used to bite
        when a project's data dir got wiped but the DB rows survived."""
        pid = self._create_project("chunks-missing-import-file")
        # Register an import Dataset pointing at a non-existent file.
        async def _bad():
            async with async_session_factory() as session:
                ds = Dataset(
                    project_id=pid,
                    name="ghost",
                    dataset_type=DatasetType.SYNTHETIC,
                    file_path="/tmp/this-file-does-not-exist-xyz.jsonl",
                    record_count=999,
                )
                session.add(ds)
                await session.commit()
        asyncio.run(_bad())
        resp = self.client.get(
            f"/api/projects/{pid}/cleaning/chunks",
            params={"limit": 10, "random_sample": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["total"], 0)


if __name__ == "__main__":
    unittest.main()

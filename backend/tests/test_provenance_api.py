"""Licence & provenance ledger through the API: lazily attached legacy
documents, bulk PATCH rules, upload / hub / kaggle / web / git / dataset-import
auto-fill, predefined-split lock, project duplicate. Network and the hub
libraries are stubbed."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.services import provenance_service as prov
from app.services.ingestion_service import get_or_create_raw_dataset, ingest_remote_dataset
from test_git_repo_source import make_repo
from test_web_crawl_source import SEED, Site

_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802
    _CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _CM.__exit__(None, None, None)


def _poll_job(client: TestClient, job_id: int, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body.get("status") in {"succeeded", "failed", "cancelled"}:
            return body
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} still {body.get('status')}")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _CM
        self.pid = self.client.post("/api/projects", json={"name": f"prov-{uuid4().hex[:6]}", "description": ""}).json()["id"]
        self.base = f"/api/projects/{self.pid}"

    def tearDown(self) -> None:
        wait_for_jobs_idle(self.client)

    def sources(self) -> dict:
        resp = self.client.get(f"{self.base}/provenance/sources")
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def by_kind(self, kind: str) -> list[dict]:
        return [s for s in self.sources()["sources"] if s["source_kind"] == kind]

    def documents(self) -> list[dict]:
        return self.client.get(f"{self.base}/ingestion/documents").json()

    def seed_legacy_documents(self) -> list[int]:
        """Documents as they were created before the ledger (no source)."""

        async def go() -> list[int]:
            async with async_session_factory() as db:
                raw = await get_or_create_raw_dataset(db, self.pid)
                docs = [
                    RawDocument(dataset_id=raw.id, filename="notes.txt", file_type="txt", file_path="/x/notes.txt",
                                source="upload", license_info="MIT", status=DocumentStatus.ACCEPTED, metadata_={}),
                    RawDocument(dataset_id=raw.id, filename="more.txt", file_type="txt", file_path="/x/more.txt",
                                source="upload", license_info="", status=DocumentStatus.ACCEPTED, metadata_={}),
                    RawDocument(dataset_id=raw.id, filename="hf_a.jsonl", file_type="jsonl", file_path="/x/hf_a.jsonl",
                                source="huggingface:toy/dataset", license_info="Imported from huggingface",
                                status=DocumentStatus.ACCEPTED,
                                metadata_={"source_type": "huggingface", "identifier": "toy/dataset", "split": "train",
                                           "num_samples": 12}),
                    RawDocument(dataset_id=raw.id, filename="hf_b.jsonl", file_type="jsonl", file_path="/x/hf_b.jsonl",
                                source="huggingface:toy/dataset", license_info="Imported from huggingface",
                                status=DocumentStatus.ACCEPTED,
                                metadata_={"source_type": "huggingface", "identifier": "toy/dataset", "split": "val",
                                           "num_samples": 3}),
                ]
                db.add_all(docs)
                await db.commit()
                return [d.id for d in docs]

        return asyncio.run(go())


class LegacyAndPatchTests(_Base):
    def test_legacy_documents_are_attached_once_and_listed(self):
        ids = self.seed_legacy_documents()
        first = self.sources()
        self.assertEqual(first["totals"]["sources"], 3, "two uploads + one grouped hub import")
        self.assertEqual(first["totals"]["documents"], 4)
        self.assertEqual(self.sources()["totals"]["sources"], 3, "idempotent")
        hub = self.by_kind("hf")[0]
        self.assertEqual(hub["document_count"], 2)
        self.assertEqual(hub["row_count"], 15)
        self.assertIsNone(hub["licence"], "the old 'Imported from ...' placeholder is not a licence")
        self.assertEqual(hub["licence_flags"], ["unknown"])
        self.assertEqual(hub["locator"], "toy/dataset")
        self.assertFalse(hub["usage_locked"])
        uploads = {s["title"]: s for s in self.by_kind("upload")}
        self.assertEqual(uploads["notes.txt"]["licence"], "MIT")
        self.assertEqual(uploads["notes.txt"]["licence_flags"], [])
        self.assertEqual(first["totals"]["unknown_licence"], 2)
        docs = {d["filename"]: d for d in self.documents()}
        self.assertEqual(docs["notes.txt"]["licence"], "MIT")
        self.assertEqual(docs["notes.txt"]["usage"], "any")
        self.assertEqual(docs["notes.txt"]["licence_flags"], [])
        self.assertIsNotNone(docs["more.txt"]["data_source_id"])
        self.assertEqual(docs["more.txt"]["licence_flags"], ["unknown"])
        self.assertEqual(docs["hf_a.jsonl"]["license_info"], "Imported from huggingface", "old column untouched")
        self.assertEqual(len(ids), 4)

    def test_legacy_uploads_group_by_day_and_licence_and_a_race_makes_no_twins(self):
        async def seed() -> None:
            async with async_session_factory() as db:
                raw = await get_or_create_raw_dataset(db, self.pid)
                db.add_all([
                    RawDocument(dataset_id=raw.id, filename=f"n{i}.txt", file_type="txt", file_path=f"/x/n{i}.txt",
                                source="upload", license_info="MIT", status=DocumentStatus.ACCEPTED, metadata_={})
                    for i in range(5)
                ])
                await db.commit()

        asyncio.run(seed())

        async def race() -> list[int]:
            async def one() -> int:
                async with async_session_factory() as db:
                    n = await prov.ensure_sources(db, self.pid)
                    await db.commit()
                    return n
            return list(await asyncio.gather(one(), one()))

        asyncio.run(race())
        uploads = self.by_kind("upload")
        self.assertEqual(len(uploads), 1, "five same-day uploads share one source, and no race twin is left")
        self.assertEqual(uploads[0]["document_count"], 5)
        self.assertEqual(uploads[0]["licence"], "MIT")

    def test_deleting_the_last_document_removes_its_source_but_not_import_sources(self):
        ids = self.seed_legacy_documents()
        before = self.sources()["totals"]["sources"]
        notes = next(d for d in self.documents() if d["filename"] == "notes.txt")
        self.assertEqual(self.client.delete(f"{self.base}/ingestion/documents/{notes['id']}").status_code, 204)
        self.assertEqual(self.sources()["totals"]["sources"], before - 1)
        # hub source still has a document left after deleting one of two
        hub_doc = next(d for d in self.documents() if d["filename"] == "hf_a.jsonl")
        self.client.delete(f"{self.base}/ingestion/documents/{hub_doc['id']}")
        self.assertEqual(len(self.by_kind("hf")), 1)

        async def make_import_source() -> int:
            async with async_session_factory() as db:
                src = await prov.create_source(db, self.pid, kind="dataset_import", title="imp", locator="jsonl:x")
                n = await prov.prune_empty_sources(db, self.pid, [src.id])
                await db.commit()
                return n

        self.assertEqual(asyncio.run(make_import_source()), 0, "dataset_import sources have no documents by design")
        self.assertEqual(len(ids), 4)

    def test_patch_stamps_verification_resets_on_licence_change_and_validates(self):
        self.seed_legacy_documents()
        hub = self.by_kind("hf")[0]
        url = f"{self.base}/provenance/sources"
        resp = self.client.patch(url, json={"source_ids": [hub["id"]], "licence": "CC-BY-NC-4.0", "licence_verified": True})
        self.assertEqual(resp.status_code, 200, resp.text)
        updated = resp.json()["sources"][0]
        self.assertEqual(updated["licence"], "CC-BY-NC-4.0")
        self.assertTrue(updated["licence_verified"])
        self.assertTrue(updated["licence_verified_by"])
        self.assertTrue(updated["licence_verified_at"])
        self.assertEqual(updated["licence_flags"], ["non_commercial"])
        # Member documents keep the old column in sync.
        docs = [d for d in self.documents() if d["data_source_id"] == hub["id"]]
        self.assertEqual({d["license_info"] for d in docs}, {"CC-BY-NC-4.0"})
        # Changing the licence alone resets verification.
        resp = self.client.patch(url, json={"source_ids": [hub["id"]], "licence": "MIT"})
        again = resp.json()["sources"][0]
        self.assertFalse(again["licence_verified"])
        self.assertIsNone(again["licence_verified_by"])
        # Verifying alone keeps the licence; "unknown" / null clears it.
        resp = self.client.patch(url, json={"source_ids": [hub["id"]], "licence_verified": True})
        self.assertEqual(resp.json()["sources"][0]["licence"], "MIT")
        self.assertTrue(resp.json()["sources"][0]["licence_verified"])
        for clear in ("unknown", None, ""):
            resp = self.client.patch(url, json={"source_ids": [hub["id"]], "licence": clear})
            self.assertIsNone(resp.json()["sources"][0]["licence"], clear)
        # Usage.
        resp = self.client.patch(url, json={"source_ids": [hub["id"]], "usage": "eval"})
        self.assertEqual(resp.json()["sources"][0]["usage"], "eval")
        docs = {d["filename"]: d for d in self.documents()}
        self.assertEqual(docs["hf_a.jsonl"]["usage"], "eval")
        # Errors: bad usage 422 (nothing changed), unknown source 404, other project's source 404, empty list 422.
        self.assertEqual(self.client.patch(url, json={"source_ids": [hub["id"]], "usage": "nope"}).status_code, 422)
        self.assertEqual(self.client.patch(url, json={"source_ids": [99999999], "usage": "rag"}).status_code, 404)
        self.assertEqual(self.client.patch(url, json={"source_ids": [], "usage": "rag"}).status_code, 422)
        self.assertEqual(self.client.patch("/api/projects/99999999/provenance/sources",
                                           json={"source_ids": [1], "usage": "rag"}).status_code, 404)
        other = self.client.post("/api/projects", json={"name": f"prov-o-{uuid4().hex[:6]}", "description": ""}).json()["id"]
        self.assertEqual(self.client.patch(f"/api/projects/{other}/provenance/sources",
                                           json={"source_ids": [hub["id"]], "usage": "rag"}).status_code, 404)
        self.assertEqual(self.by_kind("hf")[0]["usage"], "eval")
        self.assertEqual(self.client.get("/api/projects/99999999/provenance/sources").status_code, 404)

    def test_bulk_patch_is_all_or_nothing(self):
        self.seed_legacy_documents()
        ids = [s["id"] for s in self.sources()["sources"]]
        resp = self.client.patch(f"{self.base}/provenance/sources", json={"source_ids": ids + [99999999], "usage": "rag"})
        self.assertEqual(resp.status_code, 404)
        self.assertEqual({s["usage"] for s in self.sources()["sources"]}, {"any"})
        resp = self.client.patch(f"{self.base}/provenance/sources", json={"source_ids": ids, "usage": "rag"})
        self.assertEqual({s["usage"] for s in resp.json()["sources"]}, {"rag"})

    def test_predefined_split_usage_is_locked_but_its_licence_is_editable(self):
        async def go() -> int:
            async with async_session_factory() as db:
                raw = await get_or_create_raw_dataset(db, self.pid)
                source = await prov.create_source(db, self.pid, kind="predefined_split", title="Predefined test split",
                                                  usage="eval")  # a forced usage is ignored at creation
                db.add(RawDocument(dataset_id=raw.id, filename="test.jsonl", file_type="jsonl", file_path="/x/t.jsonl",
                                   source="predefined_split:test", status=DocumentStatus.ACCEPTED,
                                   metadata_={"split_role": "test"}, data_source_id=source.id))
                await db.commit()
                return source.id

        sid = asyncio.run(go())
        listed = [s for s in self.sources()["sources"] if s["id"] == sid][0]
        self.assertEqual(listed["usage"], "any")
        self.assertTrue(listed["usage_locked"])
        url = f"{self.base}/provenance/sources"
        resp = self.client.patch(url, json={"source_ids": [sid], "usage": "rag"})
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["error_code"], "PROVENANCE_USAGE_LOCKED")
        resp = self.client.patch(url, json={"source_ids": [sid], "licence": "CC0-1.0"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["sources"][0]["licence"], "CC0-1.0")
        self.assertEqual(resp.json()["sources"][0]["usage"], "any")


class UploadTests(_Base):
    def test_upload_creates_a_source_with_the_form_licence_and_usage(self):
        resp = self.client.post(
            f"{self.base}/ingestion/upload", files={"file": ("a.txt", b"toy document text", "text/plain")},
            data={"license_info": "CC-BY-4.0", "usage": "rag"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        (src,) = self.by_kind("upload")
        self.assertEqual((src["licence"], src["usage"], src["title"], src["document_count"]), ("CC-BY-4.0", "rag", "a.txt", 1))
        self.assertTrue(src["retrieved_at"])
        doc = self.documents()[0]
        self.assertEqual((doc["data_source_id"], doc["usage"], doc["licence"]), (src["id"], "rag", "CC-BY-4.0"))
        self.assertTrue(doc["retrieved_at"])
        # No licence typed: unknown, usage any by default.
        self.client.post(f"{self.base}/ingestion/upload", files={"file": ("b.txt", b"other text", "text/plain")})
        b = [s for s in self.by_kind("upload") if s["title"] == "b.txt"][0]
        self.assertEqual((b["licence"], b["usage"], b["licence_flags"]), (None, "any", ["unknown"]))
        self.assertEqual(self.client.post(f"{self.base}/ingestion/upload", files={"file": ("c.txt", b"x y z", "text/plain")},
                                          data={"usage": "bogus"}).status_code, 422)
        self.assertEqual(len(self.documents()), 2, "a refused upload leaves nothing behind")

    def test_batch_upload_is_one_source(self):
        resp = self.client.post(
            f"{self.base}/ingestion/upload-batch",
            files=[("files", ("a.txt", b"text one", "text/plain")), ("files", ("b.txt", b"text two", "text/plain")),
                   ("files", ("bad.exe", b"MZ", "application/octet-stream"))],
            data={"license_info": "MIT", "usage": "eval"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["uploaded"], 2)
        (src,) = self.by_kind("upload")
        self.assertEqual(src["title"], "Upload batch of 2 files")
        self.assertEqual((src["document_count"], src["licence"], src["usage"]), (2, "MIT", "eval"))


def _fake_hf_rows(n: int) -> list[dict]:
    return [{"instruction": f"row-{i}", "input": f"input-{i}", "output": f"output-{i}"} for i in range(n)]


class RemoteImportTests(_Base):
    def _ingest(self, source_type: str, identifier: str, **kw) -> dict:
        async def go():
            async with async_session_factory() as db:
                out = await ingest_remote_dataset(
                    db=db, project_id=self.pid, source_type=source_type, identifier=identifier, max_samples=10,
                    use_saved_secrets=False, **kw,
                )
                await db.commit()
                return out

        return asyncio.run(go())

    def test_hub_import_records_the_declared_licence_and_revision(self):
        infos = {"default": SimpleNamespace(license="cc-by-4.0")}
        with mock.patch("datasets.load_dataset", return_value=_fake_hf_rows(30)), \
                mock.patch("datasets.get_dataset_infos", return_value=infos):
            self._ingest("huggingface", "toy/dataset", split="train", config_name=None)
        (src,) = self.by_kind("hf")
        self.assertEqual((src["locator"], src["licence"], src["source_ref"]), ("toy/dataset", "cc-by-4.0", "train"))
        self.assertEqual(src["licence_flags"], [])
        doc = self.documents()[0]
        self.assertEqual(doc["license_info"], "cc-by-4.0", "no 'Imported from' placeholder any more")
        self.assertEqual(doc["data_source_id"], src["id"])

    def test_hub_import_falls_back_to_the_card_then_to_unknown(self):
        card = SimpleNamespace(card_data=SimpleNamespace(to_dict=lambda: {"license": "cc-by-nc-4.0"}))
        api = mock.MagicMock()
        api.dataset_info.return_value = card
        with mock.patch("datasets.load_dataset", return_value=_fake_hf_rows(30)), \
                mock.patch("datasets.get_dataset_infos", side_effect=RuntimeError("offline")), \
                mock.patch("huggingface_hub.HfApi", return_value=api):
            self._ingest("huggingface", "toy/card-licence")
        with mock.patch("datasets.load_dataset", return_value=_fake_hf_rows(30)), \
                mock.patch("datasets.get_dataset_infos", side_effect=RuntimeError("offline")), \
                mock.patch("huggingface_hub.HfApi", side_effect=RuntimeError("offline")):
            self._ingest("huggingface", "toy/no-licence")
        by_locator = {s["locator"]: s for s in self.by_kind("hf")}
        self.assertEqual(by_locator["toy/card-licence"]["licence"], "cc-by-nc-4.0")
        self.assertEqual(by_locator["toy/card-licence"]["licence_flags"], ["non_commercial"])
        self.assertIsNone(by_locator["toy/no-licence"]["licence"])

    def test_kaggle_import_reads_the_licence_through_the_api_object(self):
        class FakeApi:
            def authenticate(self):
                pass

            def dataset_download_files(self, slug, path, unzip, quiet):
                (Path(path) / "data.jsonl").write_text("\n".join(json.dumps({"text": f"row {i}"}) for i in range(20)))

            def dataset_list(self, search=None, **kwargs):
                return [SimpleNamespace(ref="other/toy-set-2", license_name="GPL-3.0"),
                        SimpleNamespace(ref="owner/toy-set", license_name="CC0-1.0")]

        fake_module = SimpleNamespace(KaggleApi=FakeApi)
        with mock.patch("app.services.ingestion_service._kaggle_credentials_available", return_value=True), \
                mock.patch.dict("sys.modules", {"kaggle": SimpleNamespace(),
                                                "kaggle.api": SimpleNamespace(),
                                                "kaggle.api.kaggle_api_extended": fake_module}):
            self._ingest("kaggle", "owner/toy-set")
        (src,) = self.by_kind("kaggle")
        self.assertEqual((src["locator"], src["licence"]), ("owner/toy-set", "CC0-1.0"))


class DocumentSourceImporterTests(_Base):
    def test_web_crawl_is_one_source_with_the_declared_licence(self):
        site = Site()
        with mock.patch("app.services.document_sources.web_crawl.httpx.Client", lambda *a, **k: site.client()):
            resp = self.client.post(
                f"{self.base}/document-sources/web-crawl",
                json={"seed_urls": [SEED], "delay_seconds": 0, "max_page_bytes": 10_000,
                      "exclude_patterns": ["/skip/"], "licence": "CC-BY-4.0"},
            )
            self.assertEqual(resp.status_code, 202, resp.text)
            done = _poll_job(self.client, resp.json()["id"])
        self.assertEqual(done["status"], "succeeded", done)
        (src,) = self.by_kind("web_crawl")
        self.assertEqual((src["locator"], src["licence"], src["document_count"]), (SEED, "CC-BY-4.0", 3))
        self.assertEqual(done["result"]["data_source_id"], src["id"])
        docs = self.documents()
        self.assertEqual({d["data_source_id"] for d in docs}, {src["id"]})
        self.assertEqual({d["source_url"] for d in docs}, {SEED, f"{SEED}a.html", f"{SEED}b.html"})
        self.assertTrue(all(d["retrieved_at"] and d["licence"] == "CC-BY-4.0" for d in docs))

    def test_web_crawl_without_a_licence_is_unknown_and_an_empty_crawl_leaves_no_source(self):
        site = Site()
        with mock.patch("app.services.document_sources.web_crawl.httpx.Client", lambda *a, **k: site.client()):
            resp = self.client.post(f"{self.base}/document-sources/web-crawl",
                                    json={"seed_urls": [SEED], "delay_seconds": 0, "max_pages": 1})
            _poll_job(self.client, resp.json()["id"])
            self.assertEqual(self.by_kind("web_crawl")[0]["licence_flags"], ["unknown"])
            other = self.client.post("/api/projects", json={"name": f"prov-e-{uuid4().hex[:6]}", "description": ""}).json()["id"]
            resp = self.client.post(f"/api/projects/{other}/document-sources/web-crawl",
                                    json={"seed_urls": ["https://docs.example.test/docs/private/secret.html"],
                                          "allowed_prefixes": ["https://docs.example.test/docs/private/"],
                                          "delay_seconds": 0}, params={"async_job": "false"})
            self.assertEqual(resp.status_code, 422, resp.text)
            listing = self.client.get(f"/api/projects/{other}/provenance/sources").json()
            self.assertEqual(listing["sources"], [])
        self.assertEqual(self.client.post(f"{self.base}/document-sources/web-crawl",
                                          json={"seed_urls": [SEED], "licence": "x" * 300}).status_code, 422)

    def test_git_repo_is_one_source_pinned_to_the_commit(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(settings, "DOCUMENT_SOURCES_ALLOW_LOCAL_GIT", True):
            ids = make_repo(Path(tmp))
            resp = self.client.post(f"{self.base}/document-sources/git-repo", json={"url": ids["path"], "ref": "v1"})
            self.assertEqual(resp.status_code, 202, resp.text)
            done = _poll_job(self.client, resp.json()["id"])
        self.assertEqual(done["status"], "succeeded", done)
        (src,) = self.by_kind("git_repo")
        self.assertEqual((src["locator"], src["source_ref"], src["licence"]), (ids["path"], ids["v1"], "MIT"))
        self.assertEqual(src["document_count"], 5)
        self.assertEqual(done["result"]["data_source_id"], src["id"])
        docs = self.documents()
        self.assertEqual({d["source_url"] for d in docs}, {ids["path"]})
        self.assertEqual({d["commit"] for d in docs}, {ids["v1"]})
        self.assertTrue(all(d["retrieved_at"] for d in docs))

    def test_git_repo_without_a_licence_file_is_unknown(self):
        self.assertIsNone(prov.normalize_licence("NOASSERTION"))


class DatasetImportRunTests(_Base):
    def _write(self, tmp: str, n: int) -> str:
        path = Path(tmp) / "toy.jsonl"
        path.write_text("\n".join(json.dumps({"question": f"question {i}?", "answer": f"answer {i}"}) for i in range(n)))
        return str(path)

    def test_run_creates_a_source_and_stamps_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            locator = f"jsonl:{self._write(tmp, 6)}"
            resp = self.client.post(f"{self.base}/dataset-import/run",
                                    json={"locator": locator, "mapper_id": "qa_pair_passthrough", "usage": "rag"})
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertEqual(resp.json()["accepted_count"], 6)
            (src,) = self.by_kind("dataset_import")
            self.assertEqual((src["locator"], src["usage"], src["row_count"]), (locator, "rag", 6))
            self.assertEqual(src["licence_flags"], ["unknown"])
            rows = [json.loads(line) for line in
                    (settings.DATA_DIR / "projects" / str(self.pid) / "synthetic" / "synthetic.jsonl").read_text().splitlines()]
            self.assertEqual({r["data_source_id"] for r in rows}, {src["id"]})
            # A second run is its own source.
            self.client.post(f"{self.base}/dataset-import/run", json={"locator": locator, "mapper_id": "qa_pair_passthrough"})
            self.assertEqual(len(self.by_kind("dataset_import")), 2)
            self.assertEqual(self.client.post(f"{self.base}/dataset-import/run",
                                              json={"locator": locator, "mapper_id": "qa_pair_passthrough",
                                                    "usage": "nope"}).status_code, 400)

    def test_hub_source_describe_and_run_carry_the_licence(self):
        from app.services.dataset_import.sources.hf import HuggingFaceSource

        source = HuggingFaceSource()
        with mock.patch("datasets.get_dataset_infos", return_value={"default": SimpleNamespace(license="mit")}):
            self.assertEqual(source.licence_of("toy/dataset:train:abc123"), ("mit", "abc123"))
            self.assertEqual(source.licence_of("toy/dataset:train"), ("mit", "train"))
            with mock.patch.object(HuggingFaceSource, "_open_streaming",
                                   return_value=([{"text": "a"}], "toy/dataset", "train", None)):
                self.assertEqual(source.describe("toy/dataset:train")["licence"], "mit")

    def test_kaggle_source_licence_of(self):
        from app.services.dataset_import.sources.kaggle import KaggleSource

        api = SimpleNamespace(dataset_list=lambda search=None, **kw: [SimpleNamespace(ref="owner/toy", license_name="ODbL")])
        with mock.patch("app.services.dataset_import.sources.kaggle._import_kaggle_api", return_value=api):
            self.assertEqual(KaggleSource().licence_of("dataset:owner/toy"), ("ODbL", None))
            self.assertEqual(KaggleSource().licence_of("competition:toy-comp"), (None, None))
        with mock.patch("app.services.dataset_import.sources.kaggle._import_kaggle_api", side_effect=PermissionError("no creds")):
            self.assertEqual(KaggleSource().licence_of("dataset:owner/toy"), (None, None))


class DuplicateTests(_Base):
    def test_duplicate_copies_sources_and_remaps_ids(self):
        self.seed_legacy_documents()
        sources = self.sources()["sources"]
        hub = [s for s in sources if s["source_kind"] == "hf"][0]
        self.client.patch(f"{self.base}/provenance/sources",
                          json={"source_ids": [hub["id"]], "licence": "MIT", "licence_verified": True, "usage": "rag"})
        root = settings.DATA_DIR / "projects" / str(self.pid)
        (root / "synthetic").mkdir(parents=True, exist_ok=True)
        (root / "synthetic" / "synthetic.jsonl").write_text(
            json.dumps({"id": 1, "question": "q", "answer": "a", "data_source_id": hub["id"]}) + "\n"
            + json.dumps({"id": 2, "question": "q2", "answer": "a2", "data_source_id": 987654}) + "\n")
        copy = self.client.post(f"{self.base}/duplicate", json={}).json()
        copy_sources = self.client.get(f"/api/projects/{copy['id']}/provenance/sources").json()["sources"]
        self.assertEqual(len(copy_sources), len(sources))
        chub = [s for s in copy_sources if s["source_kind"] == "hf"][0]
        self.assertNotEqual(chub["id"], hub["id"])
        self.assertEqual((chub["licence"], chub["licence_verified"], chub["usage"]), ("MIT", True, "rag"))
        self.assertEqual(chub["document_count"], 2)
        rows = [json.loads(line) for line in
                (settings.DATA_DIR / "projects" / str(copy["id"]) / "synthetic" / "synthetic.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["data_source_id"], chub["id"])
        self.assertNotIn("data_source_id", rows[1], "a row whose source was not copied loses the stale id")
        # The original is untouched.
        self.assertEqual({s["id"] for s in self.sources()["sources"]}, {s["id"] for s in sources})

    def test_deleting_a_project_with_sources_works(self):
        self.seed_legacy_documents()
        self.sources()
        self.assertEqual(self.client.delete(f"{self.base}").status_code, 204)


if __name__ == "__main__":
    unittest.main()

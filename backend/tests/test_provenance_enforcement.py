"""The ledger's usage rules at prepare time and in seeding: ``eval`` sources
are held out of train / val and land in the test split, ``rag`` sources are in
no split, a pinned test is never added to, and a project with no restricted
source prepares byte-for-byte as before. Plus the training manifest, the
synthetic seeding filter (the passage index keeps every document) and the
export manifest."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import Base, async_session_factory
from app.main import app
from app.models.data_source import DataSource
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.experiment import Experiment, ExperimentStatus
from app.services import provenance_service as prov
from app.services.ingestion_service import get_or_create_raw_dataset

_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802
    _CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _CM.__exit__(None, None, None)


def _qa(prefix: str, n: int) -> list[dict]:
    return [{"question": f"{prefix} question number {i}?", "answer": f"{prefix} answer {i}"} for i in range(n)]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


class _Project(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _CM
        self.pid = self.client.post("/api/projects", json={"name": f"enf-{uuid4().hex[:6]}", "description": ""}).json()["id"]
        self.root = settings.DATA_DIR / "projects" / str(self.pid)
        self.docs_dir = "docs"
        (self.root / "docs").mkdir(parents=True, exist_ok=True)
        (self.root / "raw").mkdir(parents=True, exist_ok=True)

    # seeding ------------------------------------------------------------

    def add_document(self, name: str, rows: list[dict], *, kind: str = "upload", licence: str | None = "MIT") -> tuple[int, int]:
        """An ACCEPTED row-file document with its own source -> (document id, source id)."""
        path = self.root / self.docs_dir / name
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        async def go() -> tuple[int, int]:
            async with async_session_factory() as db:
                raw = await get_or_create_raw_dataset(db, self.pid)
                source = await prov.create_source(db, self.pid, kind=kind, title=name, locator=name, licence=licence)
                doc = RawDocument(dataset_id=raw.id, filename=name, file_type="jsonl", file_path=str(path),
                                  source="upload", status=DocumentStatus.ACCEPTED, metadata_={"num_samples": len(rows)})
                db.add(doc)
                await db.flush()
                doc.data_source_id = source.id
                await db.commit()
                return doc.id, source.id

        return asyncio.run(go())

    def add_import_batch(self, rows: list[dict], *, licence: str | None = None) -> int:
        """Rows a dataset-import run appended to synthetic.jsonl -> its source id."""

        async def go() -> int:
            async with async_session_factory() as db:
                source = await prov.create_source(db, self.pid, kind="dataset_import", title="import batch",
                                                  locator="jsonl:toy.jsonl", licence=licence)
                ds = (await db.execute(
                    Dataset.__table__.select().where(Dataset.project_id == self.pid, Dataset.dataset_type == DatasetType.SYNTHETIC)
                )).first()
                syn = self.root / "synthetic"
                syn.mkdir(parents=True, exist_ok=True)
                path = syn / "synthetic.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps({**row, "data_source_id": source.id, "status": "accepted"}) + "\n")
                if ds is None:
                    db.add(Dataset(project_id=self.pid, name="Synthetic", dataset_type=DatasetType.SYNTHETIC,
                                   record_count=len(rows), file_path=str(path)))
                await db.commit()
                return source.id

        return asyncio.run(go())

    def set_usage(self, source_ids: list[int], usage: str) -> None:
        resp = self.client.patch(f"/api/projects/{self.pid}/provenance/sources", json={"source_ids": source_ids, "usage": usage})
        self.assertEqual(resp.status_code, 200, resp.text)

    def prepare(self, **extra) -> dict:
        body = {"include_types": ["raw", "synthetic"], "seed": 7, "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
                "auto_stratify": False, **extra}
        resp = self.client.post(f"/api/projects/{self.pid}/dataset/split", json=body)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def split_rows(self, name: str) -> list[dict]:
        path = self.root / "prepared" / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    @staticmethod
    def questions(rows: list[dict]) -> set[str]:
        return {r["question"] for r in rows}


class EnforcementTests(_Project):
    def test_eval_rows_go_only_to_test_and_rag_rows_nowhere(self):
        a_doc, a_src = self.add_document("a.jsonl", _qa("alpha", 40))
        b_doc, b_src = self.add_document("b.jsonl", _qa("beta", 6), licence="CC-BY-NC-4.0")
        c_src = self.add_import_batch(_qa("gamma", 8))
        self.set_usage([b_src], "eval")
        self.set_usage([c_src], "rag")
        manifest = self.prepare()

        train, val, test = (self.split_rows(n) for n in ("train", "val", "test"))
        beta, gamma, alpha = (self.questions(_qa(p, 40)) for p in ("beta", "gamma", "alpha"))
        for name, rows in (("train", train), ("val", val)):
            self.assertFalse(self.questions(rows) & beta, f"eval rows in {name}")
        self.assertTrue(self.questions(_qa("beta", 6)) <= self.questions(test), "every eval row is in test")
        for name, rows in (("train", train), ("val", val), ("test", test)):
            self.assertFalse(self.questions(rows) & gamma, f"rag rows in {name}")
        self.assertTrue(self.questions(train) <= alpha | beta and self.questions(train) <= alpha)
        self.assertEqual(len(train) + len(val) + len(test) - 6, 40, "the other source splits normally")

        prov_section = manifest["provenance"]
        self.assertEqual(prov_section["excluded_rag"], 8)
        self.assertEqual(prov_section["eval_routed_to_test"], 6)
        self.assertEqual(prov_section["eval_not_added_pinned_test"], 0)
        by = prov_section["by_source"]
        self.assertEqual(by[str(b_src)]["test"], 6)
        self.assertEqual((by[str(b_src)]["train"], by[str(b_src)]["val"]), (0, 0))
        self.assertEqual(by[str(b_src)]["usage"], "eval")
        self.assertEqual(by[str(c_src)]["excluded"], 8)
        self.assertEqual(by[str(c_src)]["usage"], "rag")
        self.assertEqual(sum(by[str(a_src)][k] for k in ("train", "val", "test")), 40)
        self.assertEqual(manifest["splits"], {"train": len(train), "val": len(val), "test": len(test)})
        self.assertEqual(manifest["total_entries"], 46)
        # b is eval-only, so its non-commercial licence does not feed the model; a is MIT.
        self.assertEqual(prov_section["licence_warnings"], [])
        on_disk = json.loads((self.root / "prepared" / "manifest.json").read_text())
        self.assertEqual(on_disk["provenance"], prov_section)
        listed = {s["id"]: s for s in self.client.get(f"/api/projects/{self.pid}/provenance/sources").json()["sources"]}
        self.assertEqual(listed[b_src]["in_prepared"], {"train": 0, "val": 0, "test": 6, "excluded": 0})
        self.assertEqual(listed[c_src]["in_prepared"]["excluded"], 8)
        self.assertIsNotNone(a_doc and b_doc)

    def test_eval_rows_equal_to_pool_rows_are_removed_from_train_and_val(self):
        shared = _qa("shared", 5)
        self.add_document("a.jsonl", _qa("alpha", 30) + shared)
        _, b_src = self.add_document("b.jsonl", shared + _qa("beta", 2))
        self.set_usage([b_src], "eval")
        manifest = self.prepare(dedup_rows=False)
        pool = self.questions(self.split_rows("train")) | self.questions(self.split_rows("val"))
        self.assertFalse(pool & self.questions(shared), "duplicates of eval rows left the training pool")
        test_questions = self.questions(self.split_rows("test"))
        self.assertTrue(self.questions(shared) <= test_questions)
        self.assertEqual(sum(1 for r in self.split_rows("test") if r["question"].startswith("shared")), 5, "once each")
        self.assertEqual(manifest["provenance"]["eval_routed_to_test"], 7)

    def test_pinned_test_split_is_not_added_to(self):
        self.add_document("a.jsonl", _qa("alpha", 40))
        self.prepare()  # an active test split now exists
        pinned = self.split_rows("test")
        _, b_src = self.add_document("b.jsonl", _qa("beta", 6))
        self.set_usage([b_src], "eval")
        manifest = self.prepare(keep_test=True)
        self.assertEqual(self.split_rows("test"), pinned, "the answer key is kept verbatim")
        self.assertFalse(self.questions(self.split_rows("train") + self.split_rows("val")) & self.questions(_qa("beta", 6)))
        self.assertEqual(manifest["provenance"]["eval_not_added_pinned_test"], 6)
        self.assertEqual(manifest["provenance"]["eval_routed_to_test"], 0)
        self.assertTrue(any("pinned" in w for w in manifest["provenance"]["warnings"]))

    def test_near_duplicates_of_eval_rows_leave_the_pool_when_dedup_is_on(self):
        long_q = "what is the standard procedure for resetting the toy widget controller after a power failure"
        a_rows = _qa("alpha", 30) + [{"question": long_q + " today?", "answer": "reset the toy controller by holding the button"}]
        b_rows = [{"question": long_q + " ?", "answer": "reset the toy controller by holding the button"}] + _qa("beta", 2)
        self.add_document("a.jsonl", a_rows)
        _, b_src = self.add_document("b.jsonl", b_rows)
        self.set_usage([b_src], "eval")
        manifest = self.prepare(dedup_rows=True)
        pool = self.split_rows("train") + self.split_rows("val")
        self.assertFalse([r for r in pool if r["question"].startswith(long_q)], "the near-identical pool row is gone")
        self.assertEqual(manifest["provenance"]["eval_duplicates_removed"], 1)
        total = sum(len(self.split_rows(n)) for n in ("train", "val", "test"))
        self.assertEqual(manifest["total_entries"], total, "total_entries is what the splits hold")
        # Without dedup the near-identical row is only an exact-key match away: it stays.
        manifest = self.prepare(dedup_rows=False)
        pool = self.split_rows("train") + self.split_rows("val") + self.split_rows("test")
        self.assertEqual(manifest["provenance"]["eval_duplicates_removed"], 0)
        self.assertTrue([r for r in pool if r["question"] == long_q + " today?"])

    def test_disjoint_split_notes_the_appended_eval_rows(self):
        rows = [{"question": f"alpha q {i}?", "answer": "a", "group": f"g{i % 10}"} for i in range(40)]
        self.add_document("a.jsonl", rows)
        _, b_src = self.add_document("b.jsonl", [{"question": "beta q?", "answer": "b", "group": "g1"}])
        self.set_usage([b_src], "eval")
        manifest = self.prepare(disjoint_by="group")
        self.assertEqual(manifest["disjoint_report"]["eval_rows_appended"], 1)
        self.assertTrue(any("disjoint" in w for w in manifest["disjoint_report"]["warnings"]))
        self.assertTrue(any("disjoint" in w for w in manifest["provenance"]["warnings"]))

    def test_a_source_absent_from_an_existing_prepared_manifest_lists_zeros(self):
        _, a_src = self.add_document("a.jsonl", _qa("alpha", 30))
        listed = {s["id"]: s for s in self.client.get(f"/api/projects/{self.pid}/provenance/sources").json()["sources"]}
        self.assertIsNone(listed[a_src]["in_prepared"], "nothing prepared yet")
        self.prepare()
        _, late_src = self.add_document("late.jsonl", _qa("late", 3))
        listed = {s["id"]: s for s in self.client.get(f"/api/projects/{self.pid}/provenance/sources").json()["sources"]}
        self.assertEqual(listed[late_src]["in_prepared"], {"train": 0, "val": 0, "test": 0, "excluded": 0})
        self.assertGreater(sum(listed[a_src]["in_prepared"][k] for k in ("train", "val", "test")), 0)

    def test_all_rows_restricted_is_a_clear_error(self):
        _, a_src = self.add_document("a.jsonl", _qa("alpha", 10))
        self.set_usage([a_src], "rag")
        resp = self.client.post(f"/api/projects/{self.pid}/dataset/split", json={"include_types": ["raw"]})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("retrieval-only", json.dumps(resp.json()))

    def test_licence_warnings_for_sources_that_train(self):
        _, a_src = self.add_document("a.jsonl", _qa("alpha", 20), licence=None)
        _, b_src = self.add_document("b.jsonl", _qa("beta", 20), licence="CC-BY-NC-4.0")
        self.add_document("c.jsonl", _qa("gamma", 20), licence="MIT")
        manifest = self.prepare()
        warnings = {(w["source_id"], w["flag"]) for w in manifest["provenance"]["licence_warnings"]}
        self.assertEqual(warnings, {(a_src, "unknown"), (b_src, "non_commercial")})


class UnchangedWhenNothingIsRestrictedTests(_Project):
    def test_an_all_any_project_prepares_identically_with_and_without_the_ledger_step(self):
        self.add_document("a.jsonl", _qa("alpha", 40))
        self.add_document("b.jsonl", _qa("beta", 12), licence=None)
        self.add_import_batch(_qa("gamma", 9))

        def run() -> tuple[dict, dict]:
            manifest = self.prepare()
            files = {n: _digest(self.root / "prepared" / f"{n}.jsonl") for n in ("train", "val", "test")}
            return manifest, files

        with_ledger, files_with = run()
        with mock.patch.object(prov, "build_usage_context", new=mock.AsyncMock(return_value=prov.UsageContext())):
            without, files_without = run()
        self.assertEqual(files_with, files_without, "train / val / test are byte-for-byte the same")
        self.assertTrue(all(files_with.values()))
        drop = {"created_at", "provenance", "dataset_versions", "prepared_version"}
        self.assertEqual({k: v for k, v in with_ledger.items() if k not in drop},
                         {k: v for k, v in without.items() if k not in drop})
        self.assertEqual(with_ledger["provenance"]["excluded_rag"], 0)
        self.assertEqual(with_ledger["provenance"]["eval_routed_to_test"], 0)
        # The ledger still attributes the rows.
        self.assertEqual(len(with_ledger["provenance"]["by_source"]), 3)
        self.assertEqual(sum(sum(s[k] for k in ("train", "val", "test")) for s in with_ledger["provenance"]["by_source"].values()),
                         with_ledger["total_entries"])

    def test_the_rows_of_an_all_any_project_do_not_pick_up_extra_keys(self):
        self.add_document("a.jsonl", _qa("alpha", 20))
        self.prepare()
        for row in self.split_rows("train"):
            self.assertEqual(set(row) - {"question", "answer", "text", "_source_dataset", "_adapter_id", "_task_profile",
                                         "_source_document_id", "_source_document"}, set())


class AcceptanceTests(_Project):
    def test_marking_a_source_eval_then_reprepare_moves_it_and_the_manifests_show_it(self):
        self.add_document("a.jsonl", _qa("alpha", 40))
        _, b_src = self.add_document("b.jsonl", _qa("beta", 20))
        first = self.prepare()
        beta = self.questions(_qa("beta", 20))
        before = self.questions(self.split_rows("train")) | self.questions(self.split_rows("val"))
        self.assertTrue(before & beta, "to begin with it trains")
        self.assertEqual(first["provenance"]["eval_routed_to_test"], 0)

        self.set_usage([b_src], "eval")
        second = self.prepare()
        after = self.questions(self.split_rows("train")) | self.questions(self.split_rows("val"))
        self.assertFalse(after & beta, "its rows left train / val")
        self.assertTrue(beta <= self.questions(self.split_rows("test")))
        self.assertEqual(second["provenance"]["by_source"][str(b_src)]["usage"], "eval")
        self.assertEqual(second["provenance"]["eval_routed_to_test"], 20)

        # A training run records it.
        from app.services.training_manifest_service import capture_training_manifest, get_training_manifest

        async def go() -> dict:
            async with async_session_factory() as db:
                exp = Experiment(project_id=self.pid, name="exp-prov", description="", status=ExperimentStatus.COMPLETED,
                                 base_model="org/toy-model", config={})
                db.add(exp)
                await db.commit()
                await capture_training_manifest(db, project_id=self.pid, experiment_id=exp.id, collect_env=False)
                return await get_training_manifest(db, project_id=self.pid, experiment_id=exp.id)

        recorded = asyncio.run(go())["data_provenance"]
        self.assertEqual(recorded["eval_routed_to_test"], 20)
        self.assertEqual(recorded["by_source"][str(b_src)]["test"], 20)
        self.assertEqual(recorded["by_source"][str(b_src)]["usage"], "eval")
        self.assertEqual(recorded["sources"], 2)


class SeedingTests(_Project):
    def test_eval_and_rag_documents_never_seed_synthetic_data_but_stay_in_the_index(self):
        docs = {}
        for name, usage in (("any", "any"), ("eval", "eval"), ("rag", "rag")):
            doc_id, src = self.add_document(f"{name}.jsonl", _qa(name, 2))
            docs[name] = (doc_id, src)
            if usage != "any":
                self.set_usage([src], usage)
            chunks = Path(self.root / "docs" / f"{name}.chunks.jsonl")
            chunks.write_text("\n".join(json.dumps({"text": f"{name} passage {i} " + "word " * 30, "chunk_id": i}) for i in range(3)))
        cleaned = self.root / "cleaned"
        cleaned.mkdir(exist_ok=True)
        (cleaned / "cleaned.jsonl").write_text("\n".join(
            json.dumps({"text": f"{name} passage {i} " + "word " * 30, "source_document_id": doc_id, "chunk_id": i,
                        "source_doc": f"{name}.jsonl"})
            for name, (doc_id, _src) in docs.items() for i in range(3)) + "\n")

        from app.services.auto_rag_service import load_document_passages
        from app.services.documents_qa_flow_service import preview_documents_qa_flow
        from app.services.synthetic_service import _load_project_cleaned_chunks

        index_passages = load_document_passages(self.pid)
        self.assertEqual(len(index_passages), 9, "the passage index keeps every document")
        self.assertEqual({p["source_document_id"] for p in index_passages}, {d for d, _ in docs.values()})

        texts = asyncio.run(_load_project_cleaned_chunks(self.pid))
        self.assertEqual(len(texts), 3)
        self.assertTrue(all(t.startswith("any passage") for t in texts), texts)

        resp = self.client.get(f"/api/projects/{self.pid}/cleaning/chunks", params={"limit": 50, "random_sample": "false"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual({c["document_id"] for c in resp.json()["chunks"]}, {docs["any"][0]})

        async def preview() -> dict:
            async with async_session_factory() as db:
                return await preview_documents_qa_flow(db, self.pid)

        self.assertEqual(asyncio.run(preview())["passages"], 3, "the documents-to-Q&A flow sizes on trainable passages only")
        # Back to any: everything seeds again.
        self.set_usage([docs["eval"][1], docs["rag"][1]], "any")
        self.assertEqual(len(asyncio.run(_load_project_cleaned_chunks(self.pid))), 9)

    def test_gold_grounding_keeps_eval_documents_but_not_retrieval_only_ones(self):
        docs = {}
        for name, usage in (("any", "any"), ("eval", "eval"), ("rag", "rag")):
            doc_id, src = self.add_document(f"{name}.jsonl", _qa(name, 2))
            docs[name] = doc_id
            if usage != "any":
                self.set_usage([src], usage)
            (self.root / "docs" / f"{name}.chunks.jsonl").write_text(
                json.dumps({"text": f"{name} passage " + "word " * 30, "chunk_id": 0}) + "\n")
        from app.services.synthetic_service import _load_project_cleaned_chunks

        seeding = asyncio.run(_load_project_cleaned_chunks(self.pid))
        gold = asyncio.run(_load_project_cleaned_chunks(self.pid, include_eval=True))
        self.assertEqual([t.split()[0] for t in seeding], ["any"])
        self.assertEqual(sorted(t.split()[0] for t in gold), ["any", "eval"])
        self.assertEqual({t.doc_id for t in gold}, {docs["any"], docs["eval"]})

    def test_generated_rows_carry_the_documents_they_were_cut_from(self):
        a_doc, a_src = self.add_document("a.jsonl", _qa("alpha", 2))
        b_doc, b_src = self.add_document("b.jsonl", _qa("beta", 2))
        for name in ("a", "b"):
            (self.root / "docs" / f"{name}.chunks.jsonl").write_text(
                json.dumps({"text": f"{name} passage " + "word " * 30, "chunk_id": 0}) + "\n")
        from app.services import synthetic_service as ss

        async def fake_generate(db, project_id, source_text, n, *args, **kwargs):
            return [{"question": f"q{i}?", "answer": "a"} for i in range(n)]

        async def run() -> ss.SyntheticQaTask:
            task = ss.SyntheticQaTask(task_id="t", project_id=self.pid, target_rows=2, api_url="", api_key="",
                                      model_name="m", use_all_chunks=True, source_text="")
            with mock.patch.object(ss, "generate_qa_pairs", fake_generate):
                await ss._run_qa_generation_task(task)
            return task

        task = asyncio.run(run())
        self.assertEqual(task.status, "completed", task.error)
        ids = {tuple(r.get("source_document_ids") or [r.get("source_document_id")]) for r in task.rows}
        self.assertTrue(ids <= {(a_doc,), (b_doc,), (a_doc, b_doc)} and ids != {(None,)}, ids)
        # A later eval mark on one of those documents reaches the rows.
        self.set_usage([b_src], "eval")
        async def ctx():
            async with async_session_factory() as db:
                return await prov.build_usage_context(db, self.pid)
        context = asyncio.run(ctx())
        self.assertEqual(context.row_usage({"source_document_ids": [a_doc, b_doc]}), "eval")
        self.assertEqual(context.row_usage({"source_document_id": a_doc}), "any")
        self.assertEqual(context.row_usage({"source_document_ids": [a_doc]}), "any")

    def test_a_duplicated_project_keeps_the_eval_mark_on_its_cleaned_rows(self):
        self.docs_dir = "raw"
        a_doc, _a_src = self.add_document("a.jsonl", _qa("alpha", 2))
        b_doc, b_src = self.add_document("b.jsonl", _qa("beta", 2))
        self.set_usage([b_src], "eval")
        (self.root / "raw" / "a.chunks.jsonl").write_text(
            json.dumps({"text": "alpha passage " + "word " * 30, "source_document_id": a_doc}) + "\n")
        cleaned = self.root / "cleaned"
        cleaned.mkdir(exist_ok=True)
        rows = []
        for name, doc_id in (("alpha", a_doc), ("beta", b_doc)):
            for i in range(30):
                rows.append({"question": f"{name} cleaned question {i}?", "answer": f"{name} cleaned answer {i}",
                             "text": f"{name} passage {i} " + "word " * 30, "source_document_id": doc_id, "chunk_id": i})
        (cleaned / "cleaned.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (self.root / "raw" / "b.chunks.jsonl").write_text(json.dumps({"text": "beta passage", "source_document_id": b_doc}) + "\n")

        async def add_cleaned() -> None:
            async with async_session_factory() as db:
                db.add(Dataset(project_id=self.pid, name="Cleaned", dataset_type=DatasetType.CLEANED,
                               record_count=len(rows), file_path=str(cleaned / "cleaned.jsonl")))
                await db.commit()

        asyncio.run(add_cleaned())

        out = self.client.post(f"/api/projects/{self.pid}/duplicate", json={})
        self.assertEqual(out.status_code, 201, out.text)
        new_pid = out.json()["id"]
        new_root = settings.DATA_DIR / "projects" / str(new_pid)

        async def docs() -> dict[str, int]:
            from sqlalchemy import select
            async with async_session_factory() as db:
                res = await db.execute(select(RawDocument).join(Dataset, Dataset.id == RawDocument.dataset_id)
                                       .where(Dataset.project_id == new_pid))
                return {d.filename: int(d.id) for d in res.scalars()}

        new_docs = asyncio.run(docs())
        self.assertNotEqual(new_docs["b.jsonl"], b_doc)
        copied = [json.loads(line) for line in (new_root / "cleaned" / "cleaned.jsonl").read_text().splitlines()]
        self.assertEqual({r["source_document_id"] for r in copied}, {new_docs["a.jsonl"], new_docs["b.jsonl"]})

        sidecar = json.loads((new_root / "raw" / "b.chunks.jsonl").read_text().splitlines()[0])
        self.assertEqual(sidecar["source_document_id"], new_docs["b.jsonl"], "chunk sidecars are remapped too")

        resp = self.client.post(f"/api/projects/{new_pid}/dataset/split", json={
            "include_types": ["cleaned"], "seed": 7, "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
            "auto_stratify": False})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertGreater(resp.json()["provenance"]["eval_routed_to_test"], 0)
        def split(name):
            path = new_root / "prepared" / f"{name}.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        pool = split("train") + split("val")
        self.assertFalse([r for r in pool if r["question"].startswith("beta")], "eval rows stay out of train / val")
        self.assertEqual(sum(1 for r in split("test") if r["question"].startswith("beta")), 30)

        from app.services.synthetic_service import _load_project_cleaned_chunks
        texts = asyncio.run(_load_project_cleaned_chunks(new_pid))
        self.assertTrue(texts and all(t.startswith("alpha") for t in texts), "seeding in the copy skips the eval document")

    def test_dataset_import_rows_of_a_restricted_source_are_not_offered_as_seeds(self):
        src = self.add_import_batch(_qa("imported", 4))
        resp = self.client.get(f"/api/projects/{self.pid}/cleaning/chunks", params={"limit": 50, "random_sample": "false"})
        self.assertEqual(resp.json()["total"], 4)
        self.set_usage([src], "rag")
        resp = self.client.get(f"/api/projects/{self.pid}/cleaning/chunks", params={"limit": 50, "random_sample": "false"})
        self.assertEqual(resp.json()["total"], 0)


class ExportManifestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._patch = mock.patch.object(settings, "DATA_DIR", self.root)
        self._patch.start()
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.root / 'x.db'}", future=True)
        self.sf = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._patch.stop()
        self._tmp.cleanup()

    async def _export(self, *, with_training_manifest: bool):
        from app.models.export import ExportFormat, ExportStatus
        from app.models.project import Project
        from app.models.training_manifest import TrainingManifest
        from app.services.export_service import create_export, run_export

        out = self.root / "run"
        (out / "model").mkdir(parents=True)
        (out / "model" / "config.json").write_text("{}")
        (out / "model" / "model.safetensors").write_bytes(b"w")
        async with self.sf() as db:
            project = Project(name=f"exp-{uuid4().hex[:6]}", description="", base_model_name="org/toy-model")
            db.add(project)
            await db.flush()
            unknown = await prov.create_source(db, project.id, kind="url", title="Toy web pages", locator="https://docs.example.com/")
            nc = await prov.create_source(db, project.id, kind="hf", title="Toy dataset", locator="toy/dataset", ref="train",
                                          licence="CC-BY-NC-4.0")
            ok = await prov.create_source(db, project.id, kind="git_repo", title="Toy repo", locator="https://github.com/org/repo",
                                          ref="a" * 40, licence="MIT")
            ok.licence_verified = True
            await prov.create_source(db, project.id, kind="dataset_import", title="Held out", locator="jsonl:toy.jsonl", usage="rag")
            exp = Experiment(project_id=project.id, name="run", status=ExperimentStatus.COMPLETED, base_model="org/toy-model",
                             output_dir=str(out), config={})
            db.add(exp)
            await db.flush()
            prepared = {"by_source": {
                str(unknown.id): {"train": 10, "val": 2, "test": 1, "excluded": 0},
                str(nc.id): {"train": 5, "val": 1, "test": 0, "excluded": 0},
                str(ok.id): {"train": 20, "val": 3, "test": 2, "excluded": 0},
            }}
            (self.root / "projects" / str(project.id) / "prepared").mkdir(parents=True)
            (self.root / "projects" / str(project.id) / "prepared" / "manifest.json").write_text(json.dumps({"provenance": prepared}))
            if with_training_manifest:
                db.add(TrainingManifest(
                    experiment_id=exp.id, project_id=project.id, schema_version=1,
                    data_provenance={"by_source": {str(ok.id): {"train": 7, "val": 1, "test": 0, "excluded": 0}}},
                ))
            export = await create_export(db, project.id, exp.id, ExportFormat.HUGGINGFACE)
            export = await run_export(db, project.id, export.id, run_smoke_tests=False)
            await db.commit()
        manifest = export.manifest or {}
        self.assertEqual(export.status, ExportStatus.COMPLETED, manifest.get("error"))
        return export, manifest, (unknown.id, nc.id, ok.id)

    async def test_manifest_and_release_notes_list_sources_licences_and_warnings(self):
        export, manifest, (unknown_id, nc_id, ok_id) = await self._export(with_training_manifest=False)
        block = manifest["provenance"]
        self.assertEqual(block["counts_from"], "current_prepared_manifest")
        by_title = {s["title"]: s for s in block["sources"]}
        self.assertEqual(set(by_title), {"Toy web pages", "Toy dataset", "Toy repo", "Held out"})
        self.assertEqual((by_title["Toy dataset"]["train"], by_title["Toy dataset"]["val"]), (5, 1))
        self.assertEqual(by_title["Toy repo"]["ref"], "a" * 40)
        self.assertTrue(by_title["Toy repo"]["licence_verified"])
        self.assertEqual(by_title["Held out"]["usage"], "rag")
        self.assertEqual(block["licences"], {"unknown": 12, "CC-BY-NC-4.0": 6, "MIT": 23})
        self.assertEqual({(w["source_id"], w["flag"]) for w in block["warnings"]},
                         {(unknown_id, "unknown"), (nc_id, "non_commercial")})
        notes = (Path(export.output_path) / manifest["run_dir"] / "RELEASE_NOTES.md").read_text()
        self.assertIn("## Data sources & licences", notes)
        self.assertIn("Toy web pages (url, https://docs.example.com/): licence unknown (not verified)", notes)
        self.assertIn("licence CC-BY-NC-4.0", notes)
        self.assertIn("Toy web pages: licence unknown.", notes)
        self.assertIn("Toy dataset: non-commercial licence (CC-BY-NC-4.0).", notes)

    async def test_counts_come_from_the_training_manifest_when_the_run_has_one(self):
        _, manifest, (_, _, ok_id) = await self._export(with_training_manifest=True)
        block = manifest["provenance"]
        self.assertEqual(block["counts_from"], "training_manifest")
        by_title = {s["title"]: s for s in block["sources"]}
        self.assertEqual((by_title["Toy repo"]["train"], by_title["Toy repo"]["val"]), (7, 1))
        self.assertEqual(by_title["Toy dataset"]["train"], 0)
        self.assertEqual(block["licences"], {"MIT": 8})
        self.assertEqual(block["warnings"], [], "only sources that fed the run are warned about")


if __name__ == "__main__":
    unittest.main()

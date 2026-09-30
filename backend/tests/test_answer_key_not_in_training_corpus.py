"""The answer key is eval-only: auto-RAG and the curriculum preview read the
project's TRAINING rows, never the gold set or the val/test splits.

Both used to load GOLD_DEV + GOLD_TEST + SYNTHETIC. The auto-RAG index then
held the exact questions a model is scored on together with their reference
answers, so retrieval could hand the model the answer (and the auto-RAG
comparison's val rows could retrieve themselves). Pins:
  * before a split: cleaned + accepted synthetic rows, no gold rows;
  * after a split: exactly ``prepared/train.jsonl``;
  * a Q&A index built over the old corpus is rebuilt on the next retrieval;
  * the curriculum preview loads the same rows.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.curriculum import _load_curriculum_input_rows
from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.services import auto_rag_service as rag


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


def _qa(prefix: str, n: int) -> list[dict]:
    return [{"id": f"{prefix}-{i}", "question": f"{prefix} question {i}?", "answer": f"{prefix} answer {i}."} for i in range(n)]


class AnswerKeyStaysOutOfTrainingCorpusTests(unittest.TestCase):
    def setUp(self):
        resp = client.post("/api/projects", json={"name": f"corpus-{uuid4().hex[:6]}", "description": ""})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.pid = int(resp.json()["id"])
        recipe = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "qa-sft"})
        self.assertEqual(recipe.status_code, 200, recipe.text)
        self.root = Path(settings.DATA_DIR) / "projects" / str(self.pid)
        self._seed(DatasetType.GOLD_DEV, "gold/gold_dev.jsonl", _qa("golddev", 4))
        self._seed(DatasetType.GOLD_TEST, "gold/gold_test.jsonl", _qa("goldtest", 4))
        self._seed(DatasetType.SYNTHETIC, "synthetic/synthetic.jsonl", _qa("synth", 5))

    def _seed(self, dataset_type: DatasetType, rel: str, rows: list[dict]) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

        async def _add():
            async with async_session_factory() as db:
                db.add(Dataset(project_id=self.pid, name=rel, dataset_type=dataset_type,
                               file_path=str(path), record_count=len(rows)))
                await db.commit()

        asyncio.run(_add())

    def _write_train_split(self, rows: list[dict]) -> None:
        path = self.root / "prepared" / "train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def _load(self, loader) -> list[str]:
        async def _go():
            async with async_session_factory() as db:
                return await loader(db, self.pid)

        return [str(r.get("id")) for r in asyncio.run(_go())]

    def test_corpus_is_training_rows_never_the_answer_key(self):
        for loader in (rag._load_rag_corpus_rows, _load_curriculum_input_rows):
            ids = self._load(loader)
            self.assertEqual(len(ids), 5, ids)
            self.assertTrue(all(i.startswith("synth-") for i in ids), ids)

        # Once the project is split, the train split is the corpus — synthetic
        # rows that landed in val/test must not be retrievable either.
        self._write_train_split(_qa("synth", 3))
        for loader in (rag._load_rag_corpus_rows, _load_curriculum_input_rows):
            self.assertEqual(self._load(loader), ["synth-0", "synth-1", "synth-2"])

    def test_index_built_over_the_answer_key_is_rebuilt(self):
        index_dir = self.root / "auto_rag"
        # The old corpus: gold + synthetic, no corpus stamp.
        rag.build_bm25_index(_qa("golddev", 4) + _qa("goldtest", 4) + _qa("synth", 5),
                             recipe_id="qa-sft", output_dir=index_dir)
        self.assertFalse(rag.qa_index_is_current(index_dir / "bm25_index.json"))

        async def _go(query: str):
            async with async_session_factory() as db:
                return await rag.build_preamble_from_query(db, self.pid, query, k=5, corpus="qa")

        result = asyncio.run(_go("goldtest question 2"))
        self.assertTrue(rag.qa_index_is_current(index_dir / "bm25_index.json"))
        retrieved_ids = [str(hit["row_id"]) for hit in (result or {}).get("retrieved", [])]
        self.assertTrue(retrieved_ids)
        self.assertTrue(all(i.startswith("synth-") for i in retrieved_ids), retrieved_ids)
        self.assertNotIn("goldtest answer", (result or {}).get("preamble_text", ""))

    def test_preview_endpoint_indexes_training_rows_only(self):
        resp = client.get(f"/api/projects/{self.pid}/auto-rag/preview", params={"query": "golddev question 1", "k": 5})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["index"]["doc_count"], 5)
        self.assertTrue(all(str(hit["row_id"]).startswith("synth-") for hit in body["retrieved"]), body["retrieved"])

    def test_stale_index_is_dropped_when_there_are_no_training_rows(self):
        # A project with only an answer key: nothing to index, and the old
        # gold-backed index must not keep serving.
        resp = client.post("/api/projects", json={"name": f"corpus-{uuid4().hex[:6]}", "description": ""})
        pid = int(resp.json()["id"])
        client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "qa-sft"})
        index_dir = Path(settings.DATA_DIR) / "projects" / str(pid) / "auto_rag"
        rag.build_bm25_index(_qa("golddev", 4), recipe_id="qa-sft", output_dir=index_dir)

        async def _go():
            async with async_session_factory() as db:
                return await rag.build_preamble_from_query(db, pid, "golddev question 1", corpus="qa")

        self.assertIsNone(asyncio.run(_go()))
        self.assertFalse((index_dir / "bm25_index.json").exists())

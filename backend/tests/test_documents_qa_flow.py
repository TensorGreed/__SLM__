"""Documents → generated Q&A → answer key → split → train, as one Job.

Pins (with a fake generation backend — no LLM, no GPU):
  * each passage yields training pairs (synthetic, accepted, tagged with the
    flow's source) and one DIFFERENT evaluation question (answer key);
  * the project becomes a Q&A project, its chosen base model survives, and
    the split holds only Q&A rows (document passages drop out);
  * training starts through the given launcher with the project defaults;
  * too few pairs stop the flow before training, with the rows kept;
  * preview reports blockers (no passages / no backend); the endpoint runs
    it as a Job and refuses a second one while the first runs.
"""

from __future__ import annotations

import io
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.services import documents_qa_flow_service as flow
from app.services.synth_backends.base import SynthBackendError


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


POLICY = [
    f"Section {i}: Employees accrue {10 + i} vacation days per year after {i % 3 + 1} years of service, "
    f"and unused days up to {5 + i} carry over to January. Requests go to the team lead {2 + i % 4} weeks ahead."
    for i in range(1, 25)
]


def _docx_bytes(paragraphs: list[str]) -> bytes:
    from docx import Document

    doc = Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class FakeBackend:
    """Deterministic 'LLM': reads the section number out of the passage."""

    name = "fake"
    schema_aware = True
    calls = 0
    fail_on: set[int] = set()

    @classmethod
    def is_available(cls) -> bool:
        return True

    def describe(self) -> str:
        return "fake:deterministic"

    async def complete(self, prompt, *, system_prompt=None, max_tokens=1024, temperature=0.7, response_schema=None):
        FakeBackend.calls += 1
        import re

        section = int(re.search(r"Section (\d+):", prompt).group(1))
        if section in FakeBackend.fail_on:
            raise SynthBackendError("boom")
        days = 10 + section
        # Distinct wording per section so the split's near-duplicate dedup
        # (token-set Jaccard) treats the rows as different, like real
        # generated questions would be.
        topics = ["parental leave", "sabbaticals", "overtime", "remote work", "travel", "training budget",
                  "sick leave", "bereavement", "jury duty", "relocation", "equipment", "expenses"]
        topic = topics[section % len(topics)]
        return json.dumps({
            "pairs": [
                {"question": f"Under section {section}, how many {topic} days does an employee earn each year?", "answer": f"{days} days of {topic}."},
                {"question": f"Section {section}: what is the notice period for {topic} requests to the lead?", "answer": f"{2 + section % 4} weeks' notice for {topic}."},
                {"question": f"Under section {section}, how many {topic} days does an employee earn each year?", "answer": "duplicate"},
                {"question": f"What carries into January for {topic} per section {section}?", "answer": f"Up to {5 + section} unused {topic} days."},
            ],
            "eval": {"question": f"Who qualifies for {topic} under section {section}, by years of service?", "answer": f"Staff with {section % 3 + 1} years."},
        })


class DocumentsQaFlowTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.calls = 0
        FakeBackend.fail_on = set()
        resp = client.post("/api/projects", json={"name": f"flow-{uuid4().hex[:6]}", "description": "HR policy handbook"})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.pid = int(resp.json()["id"])
        self.root = Path(settings.DATA_DIR) / "projects" / str(self.pid)

    def _ingest(self, name: str, paragraphs: list[str]) -> None:
        doc = client.post(
            f"/api/projects/{self.pid}/ingestion/upload",
            files={"file": (name, _docx_bytes(paragraphs), "application/octet-stream")},
        ).json()
        self.assertEqual(client.post(f"/api/projects/{self.pid}/ingestion/documents/{doc['id']}/process").json()["status"], "accepted")
        resp = client.post(f"/api/projects/{self.pid}/cleaning/clean", json={"document_id": doc["id"], "chunk_size": 220, "chunk_overlap": 0})
        self.assertEqual(resp.status_code, 200, resp.text)

    def _run(self, **kwargs):
        import asyncio

        async def _go():
            async with async_session_factory() as db:
                # The pre-training passages check runs the real harness (loads
                # a model) unless a test injects it or turns it off.
                kwargs.setdefault("passages_check", False)
                return await flow.run_documents_qa_flow(db, self.pid, backend_override=FakeBackend(), **kwargs)

        return asyncio.run(_go())

    def _rows(self, rel: str) -> list[dict]:
        path = self.root / rel
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_generates_pairs_answer_key_and_split_without_training(self):
        self._ingest("policy.docx", POLICY)
        passages = client.get(f"/api/projects/{self.pid}/auto-rag/documents").json()["passages"]
        self.assertGreaterEqual(passages, 20)

        summary = self._run(train=False, max_passages=20, pairs_per_passage=3)
        self.assertEqual(summary.passages_used, 20)
        self.assertEqual(FakeBackend.calls, 20)
        # 4 pairs offered, one a duplicate question → 3 kept per passage.
        self.assertEqual(summary.training_pairs, 60)
        self.assertEqual(summary.answer_key_rows, 20)
        self.assertEqual(summary.stopped_reason, "training skipped (train=False)")

        synth = [r for r in self._rows("synthetic/synthetic.jsonl") if r.get("synth_source") == flow.FLOW_SOURCE]
        self.assertEqual(len(synth), 60)
        self.assertTrue(all(r["review_status"] == "accepted" for r in synth))
        self.assertTrue(all(r.get("source_doc") == "policy.docx" and r.get("chunk_id") is not None for r in synth))
        gold = self._rows("gold/gold_dev.jsonl")
        self.assertEqual(len(gold), 20)
        self.assertTrue(all(r["source"] == flow.FLOW_SOURCE for r in gold))
        # Eval questions are different questions from the training ones.
        self.assertFalse({r["question"] for r in gold} & {r["question"] for r in synth})

        project = client.get(f"/api/projects/{self.pid}").json()
        self.assertEqual(project["selected_recipe"]["recipe_id"], "qa-sft")
        dropped = int(summary.split.get("dedup_dropped") or 0)
        self.assertEqual(summary.split["train"] + summary.split["val"] + summary.split["test"], 60 - dropped)
        self.assertLessEqual(dropped, 10)  # the fake's templated wording trips near-dup dedup on a few rows
        self.assertGreaterEqual(summary.split["test"], 5)
        train_rows = self._rows("prepared/train.jsonl")
        self.assertTrue(all("question" in r and "answer" in r for r in train_rows))
        self.assertFalse(any("Section" in str(r.get("text", "")) and "question" not in r for r in train_rows))

    def test_starts_training_with_the_projects_model(self):
        self._ingest("policy.docx", POLICY)
        client.put(f"/api/projects/{self.pid}", json={"base_model_name": "HuggingFaceTB/SmolLM2-360M-Instruct"})
        launched: dict = {}

        async def _launcher(project_id, experiment_id, db):
            launched.update({"project_id": project_id, "experiment_id": experiment_id})
            return {"status": "running", "fake": True}

        summary = self._run(train=True, max_passages=20, start_training=_launcher)
        self.assertIsNone(summary.stopped_reason)
        self.assertEqual(launched["project_id"], self.pid)
        self.assertEqual(launched["experiment_id"], summary.experiment_id)
        self.assertEqual(summary.training, {"status": "running", "fake": True})
        experiments = client.get(f"/api/projects/{self.pid}/training/experiments").json()
        exp = next(e for e in experiments if e["id"] == summary.experiment_id)
        self.assertEqual(exp["base_model"], "HuggingFaceTB/SmolLM2-360M-Instruct")
        self.assertIn("Q&A assistant from documents", exp["name"])
        self.assertEqual(client.get(f"/api/projects/{self.pid}").json()["base_model_name"], "HuggingFaceTB/SmolLM2-360M-Instruct")

    def test_reuse_existing_retrains_the_same_pairs_on_a_new_model(self):
        self._ingest("policy.docx", POLICY)
        with self.assertRaises(ValueError):
            self._run(train=False, reuse_existing=True)
        first = self._run(train=False, max_passages=20)
        self.assertEqual(FakeBackend.calls, 20)
        client.put(f"/api/projects/{self.pid}", json={"base_model_name": "Qwen/Qwen2.5-1.5B-Instruct"})
        launched: dict = {}

        async def _launcher(project_id, experiment_id, db):
            launched["experiment_id"] = experiment_id
            return {"status": "running"}

        again = self._run(train=True, reuse_existing=True, start_training=_launcher)
        self.assertEqual(FakeBackend.calls, 20)  # no new generation
        self.assertEqual(again.training_pairs, first.training_pairs)
        self.assertEqual(again.answer_key_rows, first.answer_key_rows)
        self.assertEqual(again.passages_used, 0)
        self.assertIn("reused", again.backend)
        self.assertEqual(again.split["train"], first.split["train"])
        self.assertEqual(len([r for r in self._rows("synthetic/synthetic.jsonl") if r.get("synth_source") == flow.FLOW_SOURCE]), 60)
        self.assertEqual(len(self._rows("gold/gold_dev.jsonl")), 20)
        experiments = client.get(f"/api/projects/{self.pid}/training/experiments").json()
        exp = next(e for e in experiments if e["id"] == launched["experiment_id"])
        self.assertEqual(exp["base_model"], "Qwen/Qwen2.5-1.5B-Instruct")

    def _fake_passages_check(self, *, with_score: float, correct: int, partial: int, wrong: int):
        """Writes the cached comparison the gate reads, like the harness would."""
        def _check(progress_callback=None):
            n = correct + partial + wrong
            verdicts = ["correct"] * correct + ["partial"] * partial + ["wrong"] * wrong
            score = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
            rows = [
                {"question": f"Q{i}?", "reference": "A long enough reference answer for the judge to read here.",
                 "without_rag": {"generated": "x", "f1": 0.1, "judge": {"score": 0.0, "verdict": "wrong", "reason": "r"}},
                 "with_rag": {"generated": "y", "f1": 0.4, "judge": {"score": score[v], "verdict": v, "reason": "r"},
                              "retrieved_row_count": 3}}
                for i, v in enumerate(verdicts)
            ]
            if progress_callback:
                progress_callback(n, n, "with-RAG")
            path = self.root / "auto_rag" / "comparison_base_documents.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "model": "base", "corpus": "documents", "split": "test", "base_model": "m", "cached_at": "t",
                "summary": {"n_val_rows": n, "judge": {"judge": "fake:judge",
                                                       "without_rag": {"score": 0.0, "counts": {"correct": 0, "partial": 0, "wrong": n}},
                                                       "with_rag": {"score": with_score, "judged": n,
                                                                    "counts": {"correct": correct, "partial": partial, "wrong": wrong}}}},
                "rows": rows,
            }), encoding="utf-8")
            # What the harness returns after a sweep: the chosen retrieval
            # and every arm tried.
            return {"summary": {
                "retrieval": {"k": 3, "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
                "retrieval_sweep": [
                    {"label": "top-3", "judge_score": 0.55, "chosen": False},
                    {"label": "top-3 + reranker ms-marco-MiniLM-L-6-v2", "judge_score": with_score, "chosen": True},
                ],
            }}

        return _check

    def test_pretraining_gate_stops_when_retrieval_already_answers(self):
        self._ingest("policy.docx", POLICY)
        launched: list = []

        async def _launcher(project_id, experiment_id, db):
            launched.append(experiment_id)
            return {"status": "running"}

        summary = self._run(
            train=True, max_passages=20, start_training=_launcher, passages_check=True,
            passages_check_fn=self._fake_passages_check(with_score=0.74, correct=12, partial=4, wrong=3),
        )
        self.assertEqual(summary.passages_gate["status"], "retrieval_ready")
        self.assertEqual(summary.passages_gate["correct"], 12)
        # The gate judged the sweep's winner, and the project now serves it.
        self.assertEqual(summary.passages_gate["retrieval"]["k"], 3)
        self.assertEqual(summary.passages_gate["retrieval"]["reranker"], "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertTrue(any(arm["chosen"] for arm in summary.passages_gate["retrieval_sweep"]))
        project = client.get(f"/api/projects/{self.pid}").json()
        self.assertEqual((project.get("runtime_config") or {}).get("auto_rag_retrieval", {}).get("reranker"),
                         "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertIn("Retrieval already answers", summary.stopped_reason)
        self.assertIn("12 of 19 right", summary.stopped_reason)
        self.assertIsNone(summary.experiment_id)
        self.assertEqual(launched, [])
        self.assertEqual(summary.as_dict()["passages_gate"]["judge"], "fake:judge")

        # Train anyway: the gate is recorded but training proceeds.
        again = self._run(
            train=True, reuse_existing=True, start_training=_launcher, passages_check=True,
            train_if_retrieval_ready=True,
            passages_check_fn=self._fake_passages_check(with_score=0.74, correct=12, partial=4, wrong=3),
        )
        self.assertEqual(again.passages_gate["status"], "retrieval_ready")
        self.assertIsNotNone(again.experiment_id)
        self.assertEqual(launched, [again.experiment_id])

    def test_pretraining_gate_lets_training_run_when_retrieval_is_weak_or_unjudged(self):
        self._ingest("policy.docx", POLICY)

        async def _launcher(project_id, experiment_id, db):
            return {"status": "running"}

        weak = self._run(
            train=True, max_passages=20, start_training=_launcher, passages_check=True,
            passages_check_fn=self._fake_passages_check(with_score=0.3, correct=3, partial=4, wrong=12),
        )
        self.assertEqual(weak.passages_gate["status"], "retrieval_weak")
        self.assertIsNotNone(weak.experiment_id)

        # Promising (a fair share right, too many wrong): trains, and says so.
        promising = self._run(
            train=True, reuse_existing=True, start_training=_launcher, passages_check=True,
            passages_check_fn=self._fake_passages_check(with_score=0.55, correct=8, partial=5, wrong=6),
        )
        self.assertEqual(promising.passages_gate["status"], "retrieval_promising")
        self.assertIsNotNone(promising.experiment_id)
        self.assertTrue(any("retrieval alone was promising" in w for w in promising.warnings))

        def _broken(progress_callback=None):
            raise RuntimeError("no GPU")

        errored = self._run(train=True, reuse_existing=True, start_training=_launcher, passages_check=True, passages_check_fn=_broken)
        self.assertEqual(errored.passages_gate["status"], "error")
        self.assertIsNotNone(errored.experiment_id, "a failed check never blocks training")

    def test_too_few_pairs_stops_before_training_and_keeps_the_rows(self):
        self._ingest("policy.docx", POLICY[:8])
        FakeBackend.fail_on = {3, 4}
        summary = self._run(train=True, max_passages=8, pairs_per_passage=2)
        self.assertEqual(summary.passages_failed, 2)
        self.assertLess(summary.training_pairs, flow.MIN_TRAINING_PAIRS)
        self.assertIn("training pairs were generated", summary.stopped_reason)
        self.assertIsNone(summary.experiment_id)
        self.assertEqual(len([r for r in self._rows("synthetic/synthetic.jsonl") if r.get("synth_source") == flow.FLOW_SOURCE]), summary.training_pairs)
        self.assertIn("2 of 8 passages produced no usable questions.", summary.warnings)

    def test_preview_reports_blockers(self):
        with patch("app.services.synth_backends.BACKEND_REGISTRY", []):
            preview = client.get(f"/api/projects/{self.pid}/flows/documents-to-qa/preview").json()
        self.assertFalse(preview["eligible"])
        self.assertEqual(len(preview["blockers"]), 2)
        self.assertIn("cleaned document passages", preview["blockers"][0])
        self.assertIn("No generation model", preview["blockers"][1])
        resp = client.post(f"/api/projects/{self.pid}/flows/documents-to-qa", json={})
        self.assertEqual(resp.status_code, 400)

        self._ingest("policy.docx", POLICY)
        with patch("app.services.synth_backends.BACKEND_REGISTRY", [FakeBackend]):
            preview = client.get(f"/api/projects/{self.pid}/flows/documents-to-qa/preview").json()
        self.assertTrue(preview["eligible"])
        self.assertEqual(preview["backend"], "fake")
        self.assertEqual(preview["plan"]["llm_calls"], min(preview["passages"], flow.DEFAULT_MAX_PASSAGES))

    def test_endpoint_runs_the_flow_as_a_job(self):
        self._ingest("policy.docx", POLICY)
        with patch("app.services.synth_backends.BACKEND_REGISTRY", [FakeBackend]):
            resp = client.post(f"/api/projects/{self.pid}/flows/documents-to-qa", json={"train": False, "max_passages": 20, "passages_check": False})
            self.assertEqual(resp.status_code, 202, resp.text)
            job_id = resp.json()["id"]
            dup = client.post(f"/api/projects/{self.pid}/flows/documents-to-qa", json={"train": False})
            self.assertIn(dup.status_code, (202, 409))  # 409 if the first is still in flight
            deadline = time.time() + 60
            while time.time() < deadline:
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.5)
        self.assertEqual(job["status"], "succeeded", job.get("error"))
        self.assertEqual(job["result"]["training_pairs"], 60)
        self.assertEqual(job["result"]["answer_key_rows"], 20)
        self.assertEqual(job["kind"], "documents_qa_flow")

    def test_parse_output_tolerates_fences_and_drops_duplicates(self):
        raw = '```json\n{"pairs": [{"question": "What is the cap?", "answer": "Ten."}, {"question": "what is the cap?", "answer": "x"}], "eval": {"question": "What is the cap?", "answer": "Ten."}}\n```'
        pairs, eval_pair = flow.parse_passage_output(raw, 3)
        self.assertEqual(len(pairs), 1)
        self.assertIsNone(eval_pair)  # the eval question repeated a training one
        self.assertEqual(flow.parse_passage_output("not json at all", 3), ([], None))

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
import re
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
            # Quotable: the passage's own statement (the flow drops answer-key
            # rows the passage doesn't support).
            "eval": {"question": f"Who qualifies for {topic} under section {section}, by years of service?",
                     "answer": f"Employees accrue {days} vacation days per year after {section % 3 + 1} years of service."},
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
                kwargs.setdefault("backend_override", FakeBackend())
                return await flow.run_documents_qa_flow(db, self.pid, **kwargs)

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

    def test_train_with_passages_marks_the_run_for_passage_prompts(self):
        self._ingest("policy.docx", POLICY)

        async def _launcher(project_id, experiment_id, db):
            return {"status": "running"}

        first = self._run(train=False, max_passages=20)
        test_before = (self.root / "prepared" / "test.jsonl").read_bytes()
        summary = self._run(
            train=True, reuse_existing=True, start_training=_launcher,
            train_with_passages=True, passages_withheld_fraction=0.1,
        )
        self.assertTrue(summary.as_dict()["train_with_passages"])
        self.assertEqual(summary.training_pairs, first.training_pairs)
        # Same pairs, same split: the test examples are byte-identical.
        self.assertEqual((self.root / "prepared" / "test.jsonl").read_bytes(), test_before)
        exp = next(
            e for e in client.get(f"/api/projects/{self.pid}/training/experiments").json()
            if e["id"] == summary.experiment_id
        )
        self.assertIn("reads your passages", exp["name"])
        self.assertTrue(exp["config"]["passage_prompts"])
        self.assertEqual(exp["config"]["passage_prompts_withheld_fraction"], 0.1)

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
        def _check(progress_callback=None, **kwargs):
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

        def _broken(progress_callback=None, **kwargs):
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

    def test_cloud_pin_uses_the_projects_saved_key(self):
        """backend="cloud:<provider>:<model>" — the flow uses a cloud model with
        the key saved on the project (like the Synthetic tab's span tools):
        400 for a malformed pin, 402 (and a preview blocker) with no key."""
        self._ingest("policy.docx", POLICY)
        pin = "cloud:anthropic:claude-test"
        with patch("app.services.synth_backends.BACKEND_REGISTRY", []):
            preview = client.get(f"/api/projects/{self.pid}/flows/documents-to-qa/preview", params={"backend": pin}).json()
            self.assertFalse(preview["eligible"])
            self.assertIn("No anthropic API key saved", " ".join(preview["blockers"]))
            missing = client.post(f"/api/projects/{self.pid}/flows/documents-to-qa", json={"backend": pin, "train": False})
            self.assertEqual(missing.status_code, 402, missing.text)
            self.assertEqual(missing.json()["error_code"], "CLOUD_KEY_MISSING")
            bad = client.post(f"/api/projects/{self.pid}/flows/documents-to-qa", json={"backend": "cloud:anthropic", "train": False})
            self.assertEqual(bad.status_code, 400, bad.text)
            self.assertEqual(bad.json()["error_code"], "BAD_BACKEND_PIN")

        saved = client.put(f"/api/projects/{self.pid}/secrets", json={"provider": "cloud_llm_anthropic", "key_name": "api_key", "value": "sk-test-key"})
        self.assertEqual(saved.status_code, 201, saved.text)
        built: dict = {}

        class FakeCloud(FakeBackend):
            def __init__(self, *, provider, model, api_key):
                built.update(provider=provider, model=model, api_key=api_key)

            def describe(self) -> str:
                return f"cloud:{built['provider']}:{built['model']}"

        with patch("app.services.synth_backends.BACKEND_REGISTRY", []), patch("app.services.synth_backends.CloudLlmBackend", FakeCloud):
            preview = client.get(f"/api/projects/{self.pid}/flows/documents-to-qa/preview", params={"backend": pin}).json()
            self.assertTrue(preview["eligible"], preview["blockers"])
            self.assertEqual(preview["backend"], pin)
            resp = client.post(
                f"/api/projects/{self.pid}/flows/documents-to-qa",
                json={"backend": pin, "train": False, "max_passages": 20, "passages_check": False},
            )
            self.assertEqual(resp.status_code, 202, resp.text)
            job_id = resp.json()["id"]
            deadline = time.time() + 60
            while time.time() < deadline:
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.5)
        self.assertEqual(job["status"], "succeeded", job.get("error"))
        self.assertEqual(built, {"provider": "anthropic", "model": "claude-test", "api_key": "sk-test-key"})
        self.assertEqual(job["result"]["backend"], pin)
        self.assertEqual(job["result"]["training_pairs"], 60)
        self.assertNotIn("sk-test-key", json.dumps(job))

    def test_spread_passages_covers_the_whole_corpus_in_order(self):
        items = list(range(1103))
        chosen = flow.spread_passages(items, 150)
        self.assertEqual(len(chosen), 150)
        self.assertEqual(chosen, sorted(set(chosen)))
        self.assertEqual(chosen[0], 0)
        self.assertGreater(chosen[-1], 1090)
        self.assertEqual(flow.spread_passages(items[:5], 10), items[:5])
        self.assertEqual(flow.spread_passages(items, 0), [])

    def test_sizes_generation_to_the_decision_gap(self):
        """45 paired rows for a 0.15 judge-score gap to clear noise (SD 0.5);
        more passages first, a larger test share only when they run out."""
        self.assertEqual(flow.target_test_rows(), 45)
        full = flow.size_for_decision(467, 3)
        self.assertEqual((full["passages"], full["test_ratio"], full["expected_test_rows"], full["capped_by_available"]), (150, 0.1, 45, False))
        capped = flow.size_for_decision(60, 3)
        self.assertEqual((capped["passages"], capped["test_ratio"], capped["expected_test_rows"]), (60, 0.25, 45))
        self.assertTrue(capped["reaches_target"])
        short = flow.size_for_decision(20, 3)
        self.assertEqual((short["passages"], short["test_ratio"]), (20, 0.3))
        self.assertFalse(short["reaches_target"])
        # A smaller gap needs more rows; the floor is MIN_TEST_ROWS.
        self.assertEqual(flow.target_test_rows(gap=0.3), 12)
        self.assertEqual(flow.target_test_rows(gap=1.0, diff_sd=0.1), flow.MIN_TEST_ROWS)

    def test_preview_plan_is_sized_and_an_unsized_run_uses_it(self):
        self._ingest("policy.docx", POLICY)
        preview = client.get(f"/api/projects/{self.pid}/flows/documents-to-qa/preview").json()
        sized = preview["plan"]["sized_for"]
        self.assertEqual(sized["target_test_rows"], 45)
        self.assertEqual(preview["plan"]["max_passages"], sized["passages"])
        self.assertTrue(sized["capped_by_available"], "the fixture has far fewer than 150 passages")
        # An unsized run (max_passages omitted) takes the sizing: the test
        # share rises so the split can still make the decision.
        summary = self._run(train=False)
        self.assertEqual(summary.split["sized_for"]["passages"], sized["passages"])
        self.assertGreater(summary.split["test_ratio"], 0.1)
        self.assertGreaterEqual(summary.split["test"], 5)

    def test_eval_pair_gate_drops_deictic_questions_and_unsupported_answers(self):
        """An answer-key row must be quotable from its passage: the question
        can't point at "the passage" / "this subsection", and the answer's
        content words must mostly be in the passage."""
        passage = (
            "Every individual who is given access to personal information is entitled to (a) request "
            "correction of the personal information where the individual believes there is an error or "
            "omission therein; (b) require that a notation be attached to the information."
        )
        ok = {"question": "Who may request a correction of personal information?",
              "answer": "Every individual who is given access to personal information is entitled to request correction of the personal information where the individual believes there is an error or omission therein."}
        self.assertIsNone(flow.eval_pair_rejection(ok, passage))
        self.assertEqual(flow.eval_pair_rejection({"question": "What does this subsection require?", "answer": ok["answer"]}, passage), "deictic")
        self.assertEqual(flow.eval_pair_rejection({"question": "What is required regarding access, as mentioned above?", "answer": ok["answer"]}, passage), "deictic")
        self.assertEqual(flow.eval_pair_rejection({"question": "What does the passage say about notations?", "answer": ok["answer"]}, passage), "deictic")
        self.assertEqual(flow.eval_pair_rejection({"question": "Who must be notified of a correction?",
                                                   "answer": "The Privacy Commissioner must report the correction to the Minister of Justice within two years."}, passage), "unsupported")
        self.assertLess(flow.answer_support("The Privacy Commissioner must report to the Minister of Justice", passage), flow.EVAL_ANSWER_SUPPORT_MIN)
        self.assertIsNone(flow.eval_pair_rejection(None, passage))
        # The prompt asks for it too.
        prompt = flow.build_passage_prompt("text", 3)
        self.assertIn("quotable", prompt)
        self.assertIn("this subsection", prompt)

    def test_flow_counts_dropped_eval_questions_and_keeps_the_training_pairs(self):
        """A passage whose eval question is unusable still contributes its
        training pairs; the summary says how many keys were dropped and why."""
        self._ingest("policy.docx", POLICY)

        class DeicticBackend(FakeBackend):
            async def complete(self, prompt, **kwargs):
                raw = await super().complete(prompt, **kwargs)
                payload = json.loads(raw)
                payload["eval"] = {"question": "What does this passage say about refunds?", "answer": payload["eval"]["answer"]}
                return json.dumps(payload)

        summary = self._run(train=False, backend_override=DeicticBackend())
        self.assertGreater(summary.training_pairs, 0)
        self.assertEqual(summary.answer_key_rows, 0)
        self.assertEqual(summary.eval_rejected, {"deictic": summary.passages_used})
        self.assertTrue(any("dropped from the answer key" in w and "pointed at the passage" in w for w in summary.warnings), summary.warnings)
        self.assertEqual(summary.as_dict()["eval_rejected"], {"deictic": summary.passages_used})

    def test_questions_that_point_at_the_passage_are_repaired_before_any_row_is_kept(self):
        """A pure attribution ("…, according to the passage?") is cut by code;
        any other reference to the passage goes back to the model in one
        rewrite call per passage; nothing saved points at the passage."""
        self._ingest("policy.docx", POLICY)
        rewrite_prompts: list[str] = []

        class PointingBackend(FakeBackend):
            async def complete(self, prompt, **kwargs):
                if prompt.startswith("Each question below"):
                    rewrite_prompts.append(prompt)
                    count = len(re.findall(r"^\d+\. ", prompt, flags=re.MULTILINE))
                    section = int(re.search(r"Section (\d+):", prompt).group(1))
                    return json.dumps({"questions": [f"What does section {section} define as a qualifying leave (rewrite {i})?" for i in range(count)]})
                payload = json.loads(await super().complete(prompt, **kwargs))
                pairs = payload["pairs"]
                pairs[0]["question"] = pairs[0]["question"].rstrip("?") + ", according to the passage?"
                pairs[1]["question"] = "What does the passage define as a notice period?"
                payload["eval"]["question"] = "Who qualifies, as listed in the text?"
                return json.dumps(payload)

        summary = self._run(train=False, max_passages=10, backend_override=PointingBackend())
        used = summary.passages_used
        self.assertEqual(summary.questions_repaired, {"stripped": used, "rewritten": 2 * used})
        self.assertEqual(len(rewrite_prompts), used)  # one rewrite call per passage, both questions in it
        self.assertEqual(summary.pairs_rejected, {})
        self.assertEqual(summary.answer_key_rows, used)
        self.assertEqual(summary.as_dict()["questions_repaired"], summary.questions_repaired)
        synth = [r for r in self._rows("synthetic/synthetic.jsonl") if r.get("synth_source") == flow.FLOW_SOURCE]
        gold = self._rows("gold/gold_dev.jsonl")
        self.assertTrue(synth and gold)
        self.assertFalse([r["question"] for r in synth + gold if flow.is_deictic(r["question"])])
        self.assertTrue(any(r["question"].endswith("each year?") for r in synth), "the stripped question keeps its own words")

    def test_a_question_the_rewrite_cannot_fix_is_dropped_and_counted(self):
        self._ingest("policy.docx", POLICY)

        class StubbornBackend(FakeBackend):
            async def complete(self, prompt, **kwargs):
                if prompt.startswith("Each question below"):
                    return json.dumps({"questions": ["What does the passage define?"]})  # still deictic
                payload = json.loads(await super().complete(prompt, **kwargs))
                payload["pairs"][1]["question"] = "What does the passage define as a notice period?"
                return json.dumps(payload)

        summary = self._run(train=False, max_passages=10, backend_override=StubbornBackend())
        used = summary.passages_used
        self.assertEqual(summary.pairs_rejected, {"deictic": used})
        self.assertEqual(summary.training_pairs, 2 * used)
        self.assertEqual(summary.answer_key_rows, used)  # the eval questions were fine
        self.assertTrue(any("training questions were dropped" in w for w in summary.warnings), summary.warnings)

    def test_strip_and_rewrite_helpers(self):
        strip = flow.strip_passage_reference
        self.assertEqual(strip("Who can claim the ACTC according to this passage?"), "Who can claim the ACTC?")
        self.assertEqual(strip("What are the ways to pay, according to the passage?"), "What are the ways to pay?")
        self.assertEqual(strip("According to the text, when is the return due?"), "When is the return due?")
        # A reference that carries meaning is left for the model.
        for question in ("What does the passage define as identity theft?", "Which states are listed in the text?",
                         "How much is the standard deduction for 2025?"):
            self.assertEqual(strip(question), question)
        self.assertTrue(flow.is_deictic("Who can claim it according to the excerpt?"))
        self.assertFalse(flow.is_deictic("Who can claim the additional child tax credit?"))

        prompt = flow.build_rewrite_prompt("Section 3: text.", ["Q one about the passage?", "Q two in the text?"])
        self.assertIn("1. Q one about the passage?\n2. Q two in the text?", prompt)
        self.assertIn("<passage>", prompt)
        ok = json.dumps({"questions": ["Who files Form 1040?", "What does the passage say?"]})
        self.assertEqual(flow.parse_rewrite_output(ok, 2), ["Who files Form 1040?", None])
        self.assertEqual(flow.parse_rewrite_output(ok, 3), [None, None, None])  # wrong count: order untrusted
        self.assertEqual(flow.parse_rewrite_output("garbage", 1), [None])
        # Every question, not only the evaluation one, must stand on its own.
        instructions = flow.build_passage_prompt("text", 3)
        self.assertIn("Every question — the pairs and the evaluation question", instructions)
        self.assertIn("according to the passage", instructions)

    def test_parse_output_tolerates_fences_and_drops_duplicates(self):
        raw = '```json\n{"pairs": [{"question": "What is the cap?", "answer": "Ten."}, {"question": "what is the cap?", "answer": "x"}], "eval": {"question": "What is the cap?", "answer": "Ten."}}\n```'
        pairs, eval_pair = flow.parse_passage_output(raw, 3)
        self.assertEqual(len(pairs), 1)
        self.assertIsNone(eval_pair)  # the eval question repeated a training one
        self.assertEqual(flow.parse_passage_output("not json at all", 3), ([], None))


class FakeTeacher:
    """Answers from the system message it is given; abstains on notice-period
    questions (as if the passages didn't cover them)."""

    name = "fake-teacher"
    asked: list[dict] = []

    def describe(self) -> str:
        return "fake:teacher"

    async def complete(self, prompt, *, system_prompt=None, max_tokens=1024, temperature=0.7, **kwargs):
        FakeTeacher.asked.append({"question": prompt, "system_prompt": system_prompt})
        if "notice period" in prompt:
            return "The passages don't say, so I don't know."
        return f"From the passages: {prompt[:40]} [1]"


class RevisingTeacher(FakeTeacher):
    """A FakeTeacher that also answers revision prompts: a corrected answer,
    unless the question is in ``stubborn`` (its revision stays wrong) or in
    ``abstain`` (its revision says the passages don't hold the answer)."""

    revised: list[dict] = []
    stubborn: set[str] = set()
    abstain: set[str] = set()

    async def complete(self, prompt, *, system_prompt=None, max_tokens=1024, temperature=0.7, **kwargs):
        if "YOUR PREVIOUS ANSWER:" not in prompt:
            return await FakeTeacher.complete(self, prompt, system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        question = prompt.split("QUESTION:\n", 1)[1].split("\n\n", 1)[0]
        RevisingTeacher.revised.append({"question": question, "prompt": prompt, "system_prompt": system_prompt})
        if question in RevisingTeacher.stubborn:
            return "Still the same answer [1]"
        if question in RevisingTeacher.abstain:
            return "The passages don't say."
        return f"Corrected: the right figure for {question[:30]} [2]"


class FakeCritic:
    """Writes the critique as JSON; records what it was shown."""

    name = "fake-critic"

    def __init__(self) -> None:
        self.prompts: list[dict] = []

    def describe(self) -> str:
        return "fake:critic"

    async def complete(self, prompt, *, system_prompt=None, max_tokens=1024, temperature=0.7, **kwargs):
        self.prompts.append({"prompt": prompt, "system_prompt": system_prompt})
        return '{"verdict": "partial", "feedback": "Name the carry-over cap."}'


def _fake_retrieve(question, retrieval):
    return [
        {"payload": {"text": f"Passage {i} about {question}", "source_doc": "policy.docx", "chunk_id": i}, "score": 1.0}
        for i in range(int(retrieval.get("k") or 3))
    ]


class FlowGrowthSweepTeacherTests(DocumentsQaFlowTests.__bases__[0]):
    """keep_test growth, the val sweep and the teacher step (fakes only)."""

    setUp = DocumentsQaFlowTests.setUp
    _ingest = DocumentsQaFlowTests._ingest
    _run = DocumentsQaFlowTests._run
    _rows = DocumentsQaFlowTests._rows

    def test_grow_test_set_spreads_new_passages_and_skips_known_questions(self):
        kept = [{"question": "Q kept", "source_document_id": 1, "chunk_id": 0}]
        candidates = [{"question": "Q kept", "source_document_id": 1, "chunk_id": 9}] + [
            {"question": f"Q{i}", "source_document_id": 1, "chunk_id": i} for i in range(10)
        ]
        additions, report = flow.grow_test_set(kept, candidates, 5)
        self.assertEqual(len(additions), 4)
        self.assertNotIn("Q kept", [a["question"] for a in additions])
        self.assertNotIn(0, [a["chunk_id"] for a in additions], "a passage that backs a test row is used last")
        self.assertEqual(report["added"], 4)
        self.assertTrue(report["reaches_target"])
        self.assertEqual(flow.target_test_rows(0.08), 157)

    def test_question_leakage_filter_drops_exact_and_near_matches(self):
        from app.services.dataset_service import question_leakage_filter

        test = [{"question": "How many vacation days does an employee accrue after three years of service?"}]
        rows = [
            {"question": "How many vacation days does an employee accrue after three years of service?"},
            {"question": "how many vacation days does an employee accrue after three years of service"},
            {"question": "What is the notice period for remote work requests to the team lead?"},
        ]
        kept, report = question_leakage_filter(rows, test)
        self.assertEqual([r["question"] for r in kept], [rows[2]["question"]])
        self.assertEqual(report["dropped"], 2)

    def test_keep_test_keeps_the_rows_verbatim_and_grows_them(self):
        self._ingest("policy.docx", POLICY)
        first = self._run(train=False, max_passages=20)
        test_path = self.root / "prepared" / "test.jsonl"
        before = test_path.read_text(encoding="utf-8").splitlines()
        target = flow.target_test_rows(0.3)  # 12
        self.assertLess(len(before), target)
        grown = self._run(train=False, reuse_existing=True, keep_test=True, decision_gap=0.3)
        after = test_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(after[: len(before)], before, "the kept test rows are byte-identical, first")
        self.assertEqual(len(after), target)
        self.assertEqual(grown.split["test"], target)
        self.assertEqual(grown.split["val"], target)
        self.assertEqual(grown.split["test_growth"]["added"], target - len(before))
        added = [json.loads(line) for line in after[len(before):]]
        self.assertTrue(all(r.get("_source_dataset") == "answer_key" for r in added))
        self.assertTrue(all(r.get("question") and r.get("answer") for r in added))
        test_questions = {json.loads(line)["question"].lower() for line in after}
        for split in ("train", "val"):
            self.assertFalse(test_questions & {r["question"].lower() for r in self._rows(f"prepared/{split}.jsonl")})
        self.assertIn("question_leakage", grown.split["keep_test"])
        self.assertEqual(grown.split["test_sha256"], __import__("hashlib").sha256(test_path.read_bytes()).hexdigest())
        # Validation = eval-only answer-key questions, none of them a test question.
        val = self._rows("prepared/val.jsonl")
        self.assertTrue(all(r.get("_source_dataset") == "answer_key" for r in val))
        self.assertFalse(test_questions & {r["question"].lower() for r in val})
        self.assertTrue(all(r.get("_source_dataset") != "answer_key" for r in self._rows("prepared/train.jsonl")))
        # Re-running with the target already met keeps both files byte-identical.
        sha = grown.split["test_sha256"]
        again = self._run(train=False, reuse_existing=True, keep_test=True, decision_gap=0.3)
        self.assertEqual(again.split["test_sha256"], sha)
        self.assertEqual(again.split["val_sha256"], grown.split["val_sha256"])
        self.assertTrue(again.split["test_growth"]["val"]["kept_verbatim"])
        self.assertEqual(again.split["test_growth"]["added"], 0)
        self.assertGreater(first.split["test"], 0)

    def test_sweep_picks_on_val_and_scores_once_on_test(self):
        self._ingest("policy.docx", POLICY)
        self._run(train=False, max_passages=20)
        calls: list[dict] = []

        def _check(progress_callback=None, **kwargs):
            calls.append(kwargs)
            split = kwargs.get("split")
            n = 10
            path = self.root / "auto_rag" / "comparison_base_documents.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            score = {"correct": 1.0, "wrong": 0.0}
            verdicts = ["correct"] * 8 + ["wrong"] * 2 if split == "test" else ["correct"] * 6 + ["wrong"] * 4
            rows = [{"question": f"Q{i}?", "reference": "A long enough reference answer for the judge to read here.",
                     "without_rag": {"generated": "x", "judge": {"score": 0.0, "verdict": "wrong"}},
                     "with_rag": {"generated": "y", "judge": {"score": score[v], "verdict": v}}} for i, v in enumerate(verdicts)]
            counts = {"correct": verdicts.count("correct"), "partial": 0, "wrong": verdicts.count("wrong")}
            judge = {"judge": "fake:judge", "without_rag": {"score": 0.0, "counts": {"correct": 0, "partial": 0, "wrong": n}},
                     "with_rag": {"score": counts["correct"] / n, "judged": n, "counts": counts}}
            path.write_text(json.dumps({"model": "base", "corpus": "documents", "split": split, "base_model": "m",
                                        "summary": {"n_val_rows": n, "judge": judge}, "rows": rows}), encoding="utf-8")
            return {"summary": {"n_val_rows": n, "judge": judge, "retrieval": {"k": 5, "reranker": None},
                                "retrieval_sweep": [{"label": "top-5", "judge_score": 0.6, "chosen": True}] if kwargs.get("sweep_retrieval") else None}}

        async def _launcher(project_id, experiment_id, db):
            return {"status": "running"}

        summary = self._run(train=True, reuse_existing=True, start_training=_launcher, passages_check=True, passages_check_fn=_check)
        self.assertEqual([(c["split"], c["sweep_retrieval"]) for c in calls], [("val", True), ("test", False)])
        gate = summary.passages_gate
        self.assertEqual(gate["sweep_split"], "val")
        self.assertEqual(gate["sweep"]["split"], "val")
        self.assertEqual(gate["sweep"]["chosen_score"], 0.6)
        self.assertEqual(gate["split"], "test")
        self.assertEqual(gate["score"], 0.8, "the gate reads the single test score, not the sweep's")
        self.assertEqual(gate["retrieval"]["k"], 5)
        # The older behaviour stays available.
        calls.clear()
        self._run(train=True, reuse_existing=True, start_training=_launcher, passages_check=True, passages_check_fn=_check, sweep_split="test")
        self.assertEqual([(c["split"], c["sweep_retrieval"]) for c in calls], [("test", True)])

    def test_teacher_answers_the_served_passages_and_become_the_targets(self):
        from app.services import passage_prompts_service as pps
        from app.services import teacher_passage_answers_service as tpa
        from app.services.answer_judge_service import ResolvedJudge

        self._ingest("policy.docx", POLICY)
        self._run(train=False, max_passages=20)
        FakeTeacher.asked = []

        async def _judge(question, reference, prediction):
            if "carries" in question.lower():
                return 0.0, "wrong", "contradicts the reference", 0
            return 1.0, "correct", "matches", 0

        launched: dict = {}

        async def _launcher(project_id, experiment_id, db):
            launched["experiment_id"] = experiment_id
            return {"status": "running"}

        summary = self._run(
            train=True, reuse_existing=True, start_training=_launcher,
            answers_from_teacher=True, teacher_backend_override=FakeTeacher(),
            teacher_retrieve_fn=_fake_retrieve,
            teacher_filter_judge=ResolvedJudge(label="fake:judge", judge=_judge),
        )
        teacher = summary.teacher
        train_rows = self._rows("prepared/train.jsonl")
        val_rows = self._rows("prepared/val.jsonl")
        test_rows = self._rows("prepared/test.jsonl")
        self.assertEqual(teacher["counts"]["train"]["rows"], len(train_rows))
        self.assertEqual(teacher["counts"]["val"]["rows"], len(val_rows))
        asked = {a["question"] for a in FakeTeacher.asked}
        self.assertFalse(asked & {r["question"] for r in test_rows}, "the teacher never sees a test question")
        # The teacher read the serving preamble (+ the style note only).
        first = FakeTeacher.asked[0]
        self.assertTrue(first["system_prompt"].startswith("Passages from this project's documents:"))
        self.assertTrue(first["system_prompt"].endswith(tpa.TEACHER_STYLE_NOTE))
        self.assertGreater(teacher["counts"]["train"]["abstained"], 0, "notice-period answers are natural abstentions")
        self.assertGreater(teacher["counts"]["train"]["filter_dropped"], 0)
        self.assertEqual(teacher["counts"]["val"]["filter_judged"], 0, "the filter only judges training rows")
        exp = next(e for e in client.get(f"/api/projects/{self.pid}/training/experiments").json() if e["id"] == launched["experiment_id"])
        self.assertTrue(exp["config"]["passage_prompts"])
        self.assertEqual(exp["config"]["passage_prompts_targets"], "teacher")
        self.assertEqual(exp["config"]["passage_prompts_withheld_fraction"], 0.0)
        self.assertIn("teacher answers", exp["name"])

        # Training rows: prompt = the preamble the teacher read, target = its answer.
        out_dir = self.root / "scratch-run"
        block = pps.apply_passage_prompts(
            self.pid, retrieval={"k": 3}, train_file=self.root / "prepared" / "train.jsonl",
            val_file=self.root / "prepared" / "val.jsonl", output_dir=out_dir, target_mode="teacher",
        )
        self.assertEqual(block["targets"], "teacher")
        rows = [json.loads(l) for l in Path(block["train_file"]).read_text(encoding="utf-8").splitlines()]
        by_q = {a["question"]: a for a in FakeTeacher.asked}
        for row in rows:
            self.assertEqual(row["system_prompt"] + "\n\n" + tpa.TEACHER_STYLE_NOTE, by_q[row["question"]]["system_prompt"])
            self.assertNotIn("carries", row["question"].lower(), "answers the judge graded wrong are dropped")
        self.assertEqual(block["counts"]["train"]["filtered"], teacher["counts"]["train"]["filter_dropped"])

        # A re-run reuses every cached answer: no new teacher calls.
        FakeTeacher.asked = []
        again = self._run(
            train=True, reuse_existing=True, start_training=_launcher,
            answers_from_teacher=True, teacher_backend_override=FakeTeacher(),
            teacher_retrieve_fn=_fake_retrieve, teacher_filter_judge=ResolvedJudge(label="fake:judge", judge=_judge),
        )
        self.assertEqual(FakeTeacher.asked, [])
        self.assertEqual(again.teacher["counts"]["train"]["reused"], len(train_rows))

    def test_teacher_filter_keep_correct_drops_partial_answers(self):
        from app.services import teacher_passage_answers_service as tpa
        from app.services.answer_judge_service import ResolvedJudge

        self.assertTrue(tpa.filter_drops("wrong"))
        self.assertFalse(tpa.filter_drops("partial"))
        self.assertTrue(tpa.filter_drops("partial", "correct"))
        self.assertFalse(tpa.filter_drops("correct", "correct"))
        self.assertFalse(tpa.filter_drops(None, "correct"), "an unjudged answer is never dropped")
        with self.assertRaises(ValueError):
            tpa.filter_drops("wrong", "strict")

        self._ingest("policy.docx", POLICY)
        self._run(train=False, max_passages=20)
        FakeTeacher.asked = []

        async def _judge(question, reference, prediction):
            if "carries" in question.lower():
                return 0.0, "wrong", "contradicts the reference", 0
            if "earn each year" in question.lower():
                return 0.5, "partial", "half of it", 0
            return 1.0, "correct", "matches", 0

        async def _launcher(project_id, experiment_id, db):
            return {"status": "running"}

        kwargs = dict(
            train=True, reuse_existing=True, start_training=_launcher,
            answers_from_teacher=True, teacher_backend_override=FakeTeacher(),
            teacher_retrieve_fn=_fake_retrieve, teacher_filter_judge=ResolvedJudge(label="fake:judge", judge=_judge),
        )
        default = self._run(**kwargs).teacher["counts"]["train"]
        by_verdict = default["filter_by_verdict"]
        self.assertGreater(by_verdict.get("partial", 0), 0, "the fake judge grades the earn-each-year answers partial")
        self.assertEqual(default["filter_dropped"], by_verdict.get("wrong", 0), "not_wrong keeps partial answers")
        strict = self._run(**kwargs, teacher_filter_keep="correct").teacher
        self.assertEqual(strict["filter"]["keep"], "correct")
        s_counts = strict["counts"]["train"]
        self.assertEqual(s_counts["filter_by_verdict"], by_verdict, "cached verdicts are re-read, not re-judged")
        self.assertEqual(s_counts["filter_dropped"], by_verdict.get("wrong", 0) + by_verdict.get("partial", 0))
        answers = tpa.load_answers(self.pid)
        for rec in answers.values():
            f = rec.get("filter") or {}
            if f.get("verdict") == "partial":
                self.assertTrue(rec["dropped"])
                self.assertEqual(f["keep"], "correct")
        # Back to the default rule: the same cache, partial answers train again.
        again = self._run(**kwargs).teacher["counts"]["train"]
        self.assertEqual(again["filter_dropped"], by_verdict.get("wrong", 0))
        with self.assertRaises(ValueError):
            self._run(**kwargs, teacher_filter_keep="strict")

    def _revise_fixture(self):
        """Pairs + split, a teacher whose revisions fix every dropped answer
        but the ``stubborn`` one, and a judge that grades "carries" answers
        wrong until they are corrected."""
        from app.services.answer_judge_service import ResolvedJudge

        self._ingest("policy.docx", POLICY)
        self._run(train=False, max_passages=20)
        FakeTeacher.asked = []
        RevisingTeacher.revised = []
        RevisingTeacher.stubborn = set()
        RevisingTeacher.abstain = set()

        async def _judge(question, reference, prediction):
            if "carries" in question.lower() and "Corrected" not in prediction:
                return 0.0, "wrong", "contradicts the reference", 0
            return 1.0, "correct", "matches", 0

        async def _launcher(project_id, experiment_id, db):
            return {"status": "running"}

        return dict(
            train=True, reuse_existing=True, start_training=_launcher,
            answers_from_teacher=True, teacher_backend_override=RevisingTeacher(),
            teacher_retrieve_fn=_fake_retrieve, teacher_filter_judge=ResolvedJudge(label="fake:judge", judge=_judge),
        )

    def test_revise_loop_fixes_dropped_answers_records_them_and_resumes(self):
        from app.services import teacher_passage_answers_service as tpa

        kwargs = self._revise_fixture()
        # rounds=0 (the default): today's behaviour, nothing revised.
        base = self._run(**kwargs).teacher
        self.assertEqual(base["revise"], {"rounds": 0})
        self.assertNotIn("revise", base["counts"]["train"])
        dropped_before = base["counts"]["train"]["filter_dropped"]
        self.assertGreater(dropped_before, 1)
        self.assertFalse(any(r.get("revised") for r in tpa.load_answers(self.pid).values()))
        self.assertEqual(RevisingTeacher.revised, [])
        dropped_questions = sorted(r["question"] for r in tpa.load_answers(self.pid).values() if r.get("dropped"))
        RevisingTeacher.stubborn = {dropped_questions[0]}

        # One round, no critic: the judge's reason is the feedback.
        FakeTeacher.asked = []
        one = self._run(**kwargs, teacher_revise_rounds=1).teacher
        self.assertEqual(one["revise"], {"rounds": 1, "critic": "fake:judge", "judge": "fake:judge", "critic_is_judge": True})
        revise = one["counts"]["train"]["revise"]
        self.assertEqual(revise, {"attempted": dropped_before, "fixed": dropped_before - 1, "still_dropped": 1, "abstained": 0, "calls": dropped_before})
        self.assertEqual(FakeTeacher.asked, [], "no answer is asked again")
        self.assertEqual(len(RevisingTeacher.revised), dropped_before)
        first = RevisingTeacher.revised[0]
        self.assertIn("contradicts the reference", first["prompt"], "the judge's reason is the critique")
        self.assertIn("YOUR PREVIOUS ANSWER:\nFrom the passages:", first["prompt"])
        self.assertTrue(first["system_prompt"].startswith("Passages from this project's documents:"))
        self.assertTrue(first["system_prompt"].endswith(tpa.TEACHER_STYLE_NOTE), "the same system message as the original answer")
        self.assertEqual(one["counts"]["train"]["filter_dropped"], 1)
        self.assertEqual(one["counts"]["train"]["filter_by_verdict"].get("wrong"), 1, "the counts describe what trains")
        answers = tpa.load_answers(self.pid)
        revised = [r for r in answers.values() if r.get("revised")]
        self.assertEqual(len(revised), dropped_before)
        for rec in revised:
            self.assertTrue(rec["original_answer"].startswith("From the passages:"))
            self.assertEqual(len(rec["revisions"]), 1)
            rev = rec["revisions"][0]
            self.assertEqual((rev["critic"], rev["judge"], rev["round"], rev["feedback_source"]), ("fake:judge", "fake:judge", 1, "judge_reason"))
            self.assertEqual(rev["feedback"], "contradicts the reference")
            self.assertEqual(rec["answer"], rev["answer"])
            self.assertEqual(rec["filter"]["verdict"], rev["verdict"])
            self.assertTrue(rec["filter"]["revised"])
            if rec["question"] in RevisingTeacher.stubborn:
                self.assertTrue(rec["dropped"])
                self.assertEqual(rec["filter"]["verdict"], "wrong")
            else:
                self.assertFalse(rec["dropped"])
                self.assertTrue(rec["answer"].startswith("Corrected:"))
                self.assertEqual(rec["filter"]["verdict"], "correct")
        # The training rows carry the kept revision; the stubborn one stays out.
        rows, counts = tpa.teacher_rows(self._rows("prepared/train.jsonl"), answers, {"k": 3})
        self.assertEqual(counts["filtered"], 1)
        carried = [r for r in rows if "carries" in r["question"].lower()]
        self.assertEqual(len(carried), dropped_before - 1)
        self.assertTrue(all(r["answer"].startswith("Corrected:") for r in carried))

        # Resume: the same options make no new call.
        RevisingTeacher.revised = []
        again = self._run(**kwargs, teacher_revise_rounds=1).teacher
        self.assertEqual(RevisingTeacher.revised, [])
        self.assertEqual(FakeTeacher.asked, [])
        self.assertEqual(again["counts"]["train"]["revise"], {"attempted": 0, "fixed": 0, "still_dropped": 0, "abstained": 0, "calls": 0})
        self.assertEqual(again["counts"]["train"]["filter_dropped"], 1)
        # A second round re-runs only what is missing: the stubborn answer.
        two = self._run(**kwargs, teacher_revise_rounds=2).teacher
        self.assertEqual(two["counts"]["train"]["revise"], {"attempted": 1, "fixed": 0, "still_dropped": 1, "abstained": 0, "calls": 1})
        self.assertEqual([r["question"] for r in RevisingTeacher.revised], list(RevisingTeacher.stubborn))
        stubborn = next(r for r in tpa.load_answers(self.pid).values() if r["question"] in RevisingTeacher.stubborn)
        self.assertEqual([r["round"] for r in stubborn["revisions"]], [1, 2])
        self.assertTrue(stubborn["dropped"])

    def test_a_revision_that_abstains_stays_judged_and_dropped_in_the_counts(self):
        # Project 38 (2026-10-22): 69 revisions came back as abstentions and
        # vanished from filter_judged / filter_dropped, so revise.attempted
        # (628) could not be reconciled with filter_dropped (317).
        from app.services import teacher_passage_answers_service as tpa

        kwargs = self._revise_fixture()
        base = self._run(**kwargs).teacher["counts"]["train"]
        dropped_before = base["filter_dropped"]
        self.assertGreater(dropped_before, 1)
        dropped_questions = sorted(r["question"] for r in tpa.load_answers(self.pid).values() if r.get("dropped"))
        RevisingTeacher.abstain = {dropped_questions[0]}

        counts = self._run(**kwargs, teacher_revise_rounds=1).teacher["counts"]["train"]
        self.assertEqual(counts["revise"], {"attempted": dropped_before, "fixed": dropped_before - 1, "still_dropped": 1,
                                            "abstained": 1, "calls": dropped_before})
        self.assertEqual(counts["filter_judged"], base["filter_judged"], "the abstaining revision is still a judged answer")
        self.assertEqual(counts["filter_dropped"], counts["revise"]["still_dropped"])
        self.assertEqual(counts["abstained"], base["abstained"], "abstained stays the teacher's own first-pass abstentions")
        rec = next(r for r in tpa.load_answers(self.pid).values() if r["question"] in RevisingTeacher.abstain)
        self.assertTrue(rec["abstained"])
        self.assertTrue(rec["dropped"])
        rows, row_counts = tpa.teacher_rows(self._rows("prepared/train.jsonl"), tpa.load_answers(self.pid), {"k": 3})
        self.assertFalse(any(r["question"] in RevisingTeacher.abstain for r in rows), "it never trains")
        self.assertEqual(row_counts["filtered"], 1)

    def test_critic_backend_reads_the_passages_and_its_feedback_reaches_the_teacher(self):
        from app.services import teacher_passage_answers_service as tpa

        kwargs = self._revise_fixture()
        critic = FakeCritic()
        dropped_before = self._run(**kwargs).teacher["counts"]["train"]["filter_dropped"]
        FakeTeacher.asked = []
        result = self._run(**kwargs, teacher_revise_rounds=1, teacher_critic_backend_override=critic).teacher
        self.assertEqual(result["revise"]["critic"], "fake:critic")
        self.assertFalse(result["revise"]["critic_is_judge"])
        revise = result["counts"]["train"]["revise"]
        self.assertEqual(revise["attempted"], dropped_before)
        self.assertEqual(revise["fixed"], dropped_before)
        self.assertEqual(revise["calls"], 2 * dropped_before, "one critic call + one teacher call per answer")
        self.assertEqual(len(critic.prompts), dropped_before)
        first = critic.prompts[0]
        self.assertIn("Passage 0 about", first["prompt"], "the critic sees the retrieved passages")
        self.assertIn("REFERENCE ANSWER", first["prompt"])
        self.assertIn("ANSWER TO CHECK:\nFrom the passages:", first["prompt"])
        self.assertEqual(first["system_prompt"], tpa.CRITIQUE_SYSTEM_PROMPT)
        self.assertTrue(all("Name the carry-over cap." in r["prompt"] for r in RevisingTeacher.revised), "the critic's feedback reaches the revise prompt")
        rec = next(r for r in tpa.load_answers(self.pid).values() if r.get("revised"))
        rev = rec["revisions"][0]
        self.assertEqual((rev["critic"], rev["critic_verdict"], rev["feedback_source"], rev["feedback"]), ("fake:critic", "partial", "critic", "Name the carry-over cap."))
        # Abstentions were never revised.
        self.assertFalse(any("notice period" in r["question"] for r in RevisingTeacher.revised))
        self.assertFalse(any("notice period" in p["prompt"] for p in critic.prompts))
        # Same critic again: nothing to do.
        critic.prompts = []
        RevisingTeacher.revised = []
        self._run(**kwargs, teacher_revise_rounds=1, teacher_critic_backend_override=critic)
        self.assertEqual((critic.prompts, RevisingTeacher.revised), ([], []))

    def test_revise_needs_the_filter_and_a_valid_round_count(self):
        import asyncio

        from app.services import teacher_passage_answers_service as tpa
        from app.services.answer_judge_service import ResolvedJudge

        kwargs = self._revise_fixture()
        with self.assertRaises(ValueError) as ctx:
            self._run(**kwargs, teacher_revise_rounds=1, teacher_filter=False)
        self.assertIn("teacher_filter", str(ctx.exception))
        with self.assertRaises(ValueError):
            self._run(**kwargs, teacher_revise_rounds=3)

        async def _judge(question, reference, prediction):
            return 0.0, "wrong", "no", 0

        async def _bad_rounds():
            return await tpa.critique_and_revise([], llm=RevisingTeacher(), judge=ResolvedJudge(label="j", judge=_judge), rounds=3)

        with self.assertRaises(ValueError):
            asyncio.run(_bad_rounds())

        async def _abstention_skipped():
            rec = {"question": "q", "reference": "r", "answer": "The passages don't say.", "abstained": True,
                   "system_prompt": "p", "filter": {"judge": "j", "verdict": "wrong", "reason": "no"}}
            return await tpa.critique_and_revise([rec], llm=RevisingTeacher(), judge=ResolvedJudge(label="j", judge=_judge), rounds=1), rec

        counts, rec = asyncio.run(_abstention_skipped())
        self.assertEqual(counts, {"attempted": 0, "fixed": 0, "still_dropped": 0, "abstained": 0, "calls": 0})
        self.assertNotIn("revised", rec)

    def test_generation_goes_through_one_message_batch(self):
        self._ingest("policy.docx", POLICY)

        class BatchBackend(FakeBackend):
            supports_batch = True
            use_batch = False
            batches: list[int] = []

            async def complete_batch(self, requests, *, progress=None):
                BatchBackend.batches.append(len(requests))
                out = [await FakeBackend.complete(self, r["prompt"]) for r in requests]
                out[0] = None  # an errored request falls back to a direct call
                return out

        backend = BatchBackend()
        summary = self._run(train=False, max_passages=20, backend_override=backend, use_batch=True)
        self.assertTrue(backend.use_batch)
        self.assertEqual(BatchBackend.batches, [20])
        self.assertEqual(summary.passages_used, 20)
        self.assertEqual(summary.training_pairs, 60)
        self.assertFalse(flow.reply_checkpoint_path(self.pid).exists(), "the checkpoint goes once the pairs are saved")
        with self.assertRaises(ValueError):
            self._run(train=False, max_passages=20, use_batch=True)  # the plain fake can't batch

    def test_abstention_phrases(self):
        from app.services.teacher_passage_answers_service import is_abstention

        self.assertTrue(is_abstention("The passages don't say, so I don't know."))
        self.assertTrue(is_abstention("The provided passages do not contain information about that."))
        self.assertTrue(is_abstention("That is not covered in the passages."))
        self.assertFalse(is_abstention("You can deduct up to $2,500 of student loan interest [2]."))


class FlowRequestValidationTests(unittest.TestCase):
    def test_batch_needs_an_anthropic_pin(self):
        pid = int(client.post("/api/projects", json={"name": f"flow-{uuid4().hex[:6]}"}).json()["id"])
        resp = client.post(f"/api/projects/{pid}/flows/documents-to-qa", json={"use_batch": True, "backend": "ollama"})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["error_code"], "BATCH_NEEDS_ANTHROPIC")
        bad = client.post(f"/api/projects/{pid}/flows/documents-to-qa", json={"sweep_split": "train"})
        self.assertEqual(bad.status_code, 422)

    def test_revise_needs_the_filter(self):
        pid = int(client.post("/api/projects", json={"name": f"flow-{uuid4().hex[:6]}"}).json()["id"])
        resp = client.post(f"/api/projects/{pid}/flows/documents-to-qa",
                           json={"answers_from_teacher": True, "teacher_revise_rounds": 1, "teacher_filter": False})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["error_code"], "REVISE_NEEDS_FILTER")
        bad = client.post(f"/api/projects/{pid}/flows/documents-to-qa", json={"answers_from_teacher": True, "teacher_revise_rounds": 3})
        self.assertEqual(bad.status_code, 422)


class TeacherFormatTests(unittest.TestCase):
    def test_critique_and_revise_prompts(self):
        from app.services import teacher_passage_answers_service as tpa

        prompt = tpa.build_critique_prompt("What is the cap?", "Ten days.", "Twelve days [1].", "Passages from this project's documents:\n[1] Ten days carry.")
        self.assertTrue(prompt.startswith("Passages from this project's documents:\n[1] Ten days carry."))
        self.assertIn("QUESTION:\nWhat is the cap?", prompt)
        self.assertIn("REFERENCE ANSWER (treat as ground truth):\nTen days.", prompt)
        self.assertIn("ANSWER TO CHECK:\nTwelve days [1].", prompt)
        self.assertIn('{"verdict": "correct|partial|wrong", "feedback": "..."}', prompt)
        req = tpa.revise_request("What is the cap?", "PREAMBLE", "Twelve days [1].", "It is ten, not twelve.")
        self.assertEqual(req["system_prompt"], tpa.teacher_request("What is the cap?", "PREAMBLE")["system_prompt"], "the revision reads the same system message")
        self.assertEqual((req["max_tokens"], req["temperature"]), (tpa.TEACHER_MAX_TOKENS, 0.0))
        self.assertIn("QUESTION:\nWhat is the cap?\n\nYOUR PREVIOUS ANSWER:\nTwelve days [1].\n\nFEEDBACK ON IT:\nIt is ten, not twelve.", req["prompt"])
        self.assertIn("one to three sentences", req["prompt"])
        self.assertIn("[n]", req["prompt"])

    def test_parse_critique_falls_back_to_the_judges_reason(self):
        from app.services.teacher_passage_answers_service import parse_critique

        good = parse_critique('{"verdict": "wrong", "feedback": "Misses  the  cap."}', "judge said no")
        self.assertEqual(good, {"verdict": "wrong", "feedback": "Misses the cap.", "parsed": True})
        fenced = parse_critique('```json\n{"verdict": "Partial", "feedback": "x"}\n```', "r")
        self.assertEqual((fenced["verdict"], fenced["feedback"]), ("partial", "x"))
        embedded = parse_critique('Sure: {"verdict": "maybe", "feedback": "y"} done', "r")
        self.assertEqual((embedded["verdict"], embedded["feedback"], embedded["parsed"]), (None, "y", True))
        for raw in ("not json", "", None, "[1, 2]"):
            self.assertEqual(parse_critique(raw, "judge said no"), {"verdict": None, "feedback": "judge said no", "parsed": False}, raw)
        self.assertEqual(parse_critique('{"verdict": "wrong"}', "judge said no")["feedback"], "judge said no", "no feedback → the judge's reason")

    def test_teacher_answer_is_normalised_to_the_serving_cite_format(self):
        from app.services.teacher_passage_answers_service import normalize_teacher_answer

        self.assertEqual(normalize_teacher_answer("It is **$31,500** 【1】."), "It is $31,500 [1].")
        self.assertEqual(normalize_teacher_answer("See 【2, 3】"), "See [2][3]")

    def test_gpt_oss_gets_low_reasoning_others_none(self):
        from app.services.synth_backends.ollama import think_setting

        self.assertEqual(think_setting("gpt-oss:120b"), "low")
        self.assertIs(think_setting("gemma4:12b"), False)

    def test_flow_unloads_ollama_only_when_it_used_it(self):
        from app.services.synth_backends.ollama import OllamaBackend

        summary = flow.FlowSummary(project_id=1, passages_total=0, passages_used=0, passages_failed=0,
                                   training_pairs=0, answer_key_rows=0, backend="x")
        self.assertFalse(flow._used_local_ollama(summary))
        summary._llms.append(OllamaBackend(model="gpt-oss:120b"))
        self.assertTrue(flow._used_local_ollama(summary))
        judged = flow.FlowSummary(project_id=1, passages_total=0, passages_used=0, passages_failed=0,
                                  training_pairs=0, answer_key_rows=0, backend="x",
                                  teacher={"filter": {"applied": True, "judge": "ollama:gemma4:12b"}})
        self.assertTrue(flow._used_local_ollama(judged))


class GenerationConcurrencyTests(unittest.TestCase):
    def test_generation_runs_up_to_concurrency_calls_at_once(self):
        import asyncio

        state = {"now": 0, "peak": 0}

        class SlowBackend:
            name = "fake"

            def describe(self):
                return "fake"

            async def complete(self, prompt, **kwargs):
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
                await asyncio.sleep(0.01)
                state["now"] -= 1
                return "{}"

        async def _report(_f, _m):
            return None

        passages = [{"id": i, "text": f"Passage {i}. It says something."} for i in range(10)]
        raws = asyncio.run(flow._generate_replies(SlowBackend(), passages, 3, "", _report, "fake", concurrency=3))
        self.assertEqual(len(raws), 10)
        self.assertTrue(all(r == "{}" for r in raws))
        self.assertEqual(state["peak"], 3)
        state["peak"] = 0
        asyncio.run(flow._generate_replies(SlowBackend(), passages, 3, "", _report, "fake"))
        self.assertEqual(state["peak"], 1)


class GenerationCheckpointTests(unittest.TestCase):
    """A run killed mid-way (crash, power cut, restart) resumes from disk."""

    def test_generation_resumes_from_the_checkpoint(self):
        import asyncio
        import tempfile

        state = {"calls": 0, "crash_at": None}

        class Backend:
            name = "fake"

            def describe(self):
                return "fake"

            async def complete(self, prompt, **kwargs):
                state["calls"] += 1
                if state["crash_at"] is not None and state["calls"] == state["crash_at"]:
                    raise RuntimeError("power cut")
                return json.dumps({"reply_for": prompt[-40:]})

        async def _report(_f, _m):
            return None

        passages = [{"id": i, "text": f"Passage {i}. It says something."} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "flows" / "generation_checkpoint.jsonl"
            state["crash_at"] = 6
            with self.assertRaises(RuntimeError):
                asyncio.run(flow._generate_replies(Backend(), passages, 3, "", _report, "fake", checkpoint=checkpoint))
            # (gather lets the calls already queued finish; a real power cut stops them too)
            saved = len(checkpoint.read_text(encoding="utf-8").splitlines())
            self.assertTrue(0 < saved < 10, saved)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write('{"key": "cut off by the cra')  # a torn last line is skipped

            state.update(calls=0, crash_at=None)
            raws = asyncio.run(flow._generate_replies(Backend(), passages, 3, "", _report, "fake", checkpoint=checkpoint))
            self.assertEqual(state["calls"], 10 - saved, "only the passages without a saved reply are asked again")
            self.assertTrue(all(raws))
            fresh = asyncio.run(flow._generate_replies(Backend(), passages, 3, "", _report, "fake"))
            self.assertEqual(raws, fresh, "a resumed reply is the reply the backend gave")

            state["calls"] = 0
            asyncio.run(flow._generate_replies(Backend(), passages, 3, "", _report, "other-model", checkpoint=checkpoint))
            self.assertEqual(state["calls"], 10, "another backend's replies are not reused")

    def test_teacher_answers_are_on_disk_as_they_arrive(self):
        import asyncio

        from app.services import teacher_passage_answers_service as tpa

        pid = 900000 + uuid4().int % 90000
        state = {"calls": 0}

        class Teacher:
            name = "fake-teacher"

            def describe(self):
                return "fake:teacher"

            async def complete(self, prompt, **kwargs):
                state["calls"] += 1
                if state["calls"] == 3:
                    raise RuntimeError("power cut")
                return f"From the passages: {prompt[:30]} [1]"

        rows = {"train": [{"question": f"How many days under section {i}?", "answer": f"{i} days."} for i in range(5)]}
        kwargs = dict(retrieval={"k": 3}, retrieve_fn=_fake_retrieve, concurrency=1)
        with self.assertRaises(RuntimeError):
            asyncio.run(tpa.write_teacher_answers(pid, rows, llm=Teacher(), **kwargs))
        saved = len(tpa.load_answers(pid))
        self.assertTrue(0 < saved < 5, saved)

        state["calls"] = -100  # no more crashes
        report = asyncio.run(tpa.write_teacher_answers(pid, rows, llm=Teacher(), **kwargs))
        self.assertEqual(report["counts"]["train"]["reused"], saved)
        self.assertEqual(state["calls"], -100 + 5 - saved, "only the unanswered questions are asked again")
        self.assertEqual(len(tpa.load_answers(pid)), 5)

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

    def test_parse_output_tolerates_fences_and_drops_duplicates(self):
        raw = '```json\n{"pairs": [{"question": "What is the cap?", "answer": "Ten."}, {"question": "what is the cap?", "answer": "x"}], "eval": {"question": "What is the cap?", "answer": "Ten."}}\n```'
        pairs, eval_pair = flow.parse_passage_output(raw, 3)
        self.assertEqual(len(pairs), 1)
        self.assertIsNone(eval_pair)  # the eval question repeated a training one
        self.assertEqual(flow.parse_passage_output("not json at all", 3), ([], None))

"""Unified task-shape detector + confirm (Wave 2b).

Pins:
  * one detector, one vocabulary: canonical task profiles (``dpo`` →
    ``preference``) with the adapter / recipe / mapper that implement each;
  * common shapes are recognised — incl. ``{article, summary}`` as
    summarization (it used to fall through to instruction_sft) — and a
    shape nobody can recognise asks for confirmation instead of guessing;
  * the user's goal is a small prior, never an override;
  * confirming writes ``dataset_adapter_preset`` — what dataset prep,
    training and eval read — and picking a recipe does too (before, the
    wizard's choice only reached ``selected_recipe`` and never training).
"""

from __future__ import annotations

import asyncio
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.services.dataset_service import resolve_project_dataset_adapter_preference
from app.services.task_shape_service import canonical_task_profile, detect_task_shape


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


def _qa_rows(n=12):
    return [{"question": f"How do I reset device {i}?", "answer": f"Hold the power button for {i} seconds."} for i in range(n)]


class DetectTests(unittest.TestCase):
    def assertTop(self, rows, profile, **kwargs):
        result = detect_task_shape(rows, **kwargs)
        self.assertEqual(result["top"]["task_profile"], profile, result["candidates"])
        return result

    def test_question_answer_is_confident_qa(self):
        result = self.assertTop(_qa_rows(), "qa")
        top = result["top"]
        self.assertGreaterEqual(top["confidence"], 0.8)
        self.assertFalse(result["needs_confirmation"])
        self.assertEqual((top["adapter_id"], top["recipe_id"]), ("qa-pair", "qa-sft"))
        self.assertTrue(any("of rows fit" in reason for reason in top["rationale"]))

    def test_article_summary_is_summarization(self):
        rows = [
            {"article": "The quarterly meeting covered budget, hiring and the roadmap in detail. " * 3,
             "summary": f"Budget approved; {i} hires."}
            for i in range(10)
        ]
        result = self.assertTop(rows, "summarization")
        self.assertEqual(result["top"]["field_map"], {"question_field": "article", "answer_field": "summary"})

    def test_text_label_is_classification(self):
        labels = ["billing", "shipping", "account"]
        rows = [{"text": f"My issue number {i} is about {labels[i % 3]} stuff please help", "label": labels[i % 3]} for i in range(15)]
        self.assertTop(rows, "classification")

    def test_rag_triple_and_preference_and_chat(self):
        rag = [{"question": f"q{i}?", "context": "Refunds are issued within 30 days of purchase. " * 2, "answer": "30 days"} for i in range(8)]
        self.assertTop(rag, "rag_qa")
        pref = [{"prompt": f"Write a greeting for customer {i}", "chosen": f"Hello customer {i}, how can I help you today?",
                 "rejected": f"what do you want {i}, i am busy right now"} for i in range(8)]
        result = self.assertTop(pref, "preference")  # introspector says "dpo"
        self.assertEqual(result["top"]["adapter_id"], "preference-pair")
        chat = [{"messages": [{"role": "user", "content": f"hi {i}"}, {"role": "assistant", "content": "hello"}]} for i in range(8)]
        self.assertTop(chat, "chat_sft")

    def test_plain_documents_are_language_modeling(self):
        rows = [{"text": f"Section {i}. The warranty covers manufacturing defects for fourteen months from purchase."} for i in range(10)]
        self.assertTop(rows, "language_modeling")

    def test_unrecognisable_rows_ask_for_confirmation(self):
        rows = [{"foo": str(i), "bar": "x"} for i in range(10)]
        result = detect_task_shape(rows)
        self.assertTrue(result["needs_confirmation"])
        self.assertLess(result["top"]["confidence"], 0.8)

    def test_intent_is_a_prior_not_an_override(self):
        rows = _qa_rows()
        plain = detect_task_shape(rows)["top"]
        with_goal = detect_task_shape(rows, intent="a support FAQ bot for customer questions")["top"]
        self.assertGreater(with_goal["confidence"], plain["confidence"])
        self.assertIn("matches the goal you described", with_goal["rationale"])
        # A goal that disagrees with the data doesn't flip a clear QA dataset.
        self.assertEqual(detect_task_shape(rows, intent="summarize meeting notes")["top"]["task_profile"], "qa")

    def test_vocabulary_aliases(self):
        self.assertEqual(canonical_task_profile("dpo"), "preference")
        self.assertEqual(canonical_task_profile("Summary"), "summarization")
        self.assertIsNone(canonical_task_profile("interpretive-dance"))


class ConfirmTests(unittest.TestCase):
    def setUp(self):
        resp = client.post("/api/projects", json={"name": f"shape-{uuid4().hex[:6]}", "description": ""})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.pid = int(resp.json()["id"])

    def _preference(self):
        async def _go():
            async with async_session_factory() as db:
                return await resolve_project_dataset_adapter_preference(db, self.pid)
        return asyncio.run(_go())

    def test_confirm_writes_what_training_reads(self):
        resp = client.post(f"/api/projects/{self.pid}/task-shape/confirm", json={"task_profile": "classification"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["recipe_id"], "classification")
        pref = self._preference()
        self.assertEqual((pref["source"], pref["adapter_id"], pref["task_profile"]),
                         ("project", "classification-label", "classification"))
        project = client.get(f"/api/projects/{self.pid}").json()
        self.assertEqual(project["selected_recipe"]["recipe_id"], "classification")
        self.assertEqual(project["selected_recipe"]["confirmed_via"], "task_shape")

    def test_confirm_keeps_an_existing_base_model_and_rejects_unknown(self):
        client.put(f"/api/projects/{self.pid}", json={"base_model_name": "Qwen/Qwen2.5-0.5B-Instruct"})
        client.post(f"/api/projects/{self.pid}/task-shape/confirm", json={"task_profile": "dpo"})
        project = client.get(f"/api/projects/{self.pid}").json()
        self.assertEqual(project["base_model_name"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(self._preference()["task_profile"], "preference")
        bad = client.post(f"/api/projects/{self.pid}/task-shape/confirm", json={"task_profile": "nope"})
        self.assertEqual(bad.status_code, 400)

    def test_picking_a_recipe_reaches_training_too(self):
        resp = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "classification"})
        self.assertEqual(resp.status_code, 200, resp.text)
        pref = self._preference()
        self.assertEqual((pref["adapter_id"], pref["task_profile"]), ("classification-label", "classification"))

    def test_project_detection_reads_cleaned_rows(self):
        csv_bytes = ("question,answer\n" + "".join(
            f"How do I reset device {i}?,Hold the power button for {i} seconds.\n" for i in range(12)
        )).encode()
        doc = client.post(
            f"/api/projects/{self.pid}/ingestion/upload",
            files={"file": ("faq.csv", csv_bytes, "text/csv")},
        ).json()
        client.post(f"/api/projects/{self.pid}/ingestion/documents/{doc['id']}/process")
        client.post(f"/api/projects/{self.pid}/cleaning/clean", json={"document_id": doc["id"]})
        body = client.get(f"/api/projects/{self.pid}/task-shape").json()
        self.assertEqual(body["detection"]["top"]["task_profile"], "qa")
        self.assertIsNone(body["confirmed"])
        self.assertIn("summarization", {c["task_profile"] for c in body["catalog"]})

    def test_introspect_includes_detection(self):
        resp = client.post(
            f"/api/projects/{self.pid}/dataset-import/upload",
            files={"file": ("sum.csv", b"document,summary\n" + b"".join(
                b"A long meeting transcript about budget and hiring and roadmap items and more,Budget ok\n"
                for _ in range(8)), "text/csv")},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        shape = resp.json()["introspection"]["task_shape"]
        self.assertEqual(shape["top"]["task_profile"], "summarization")


if __name__ == "__main__":
    unittest.main()

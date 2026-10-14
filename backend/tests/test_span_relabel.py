"""Answer key from a second model: relabel a split's texts (unchanged),
verify offsets from the entity text, write gold/gold_test.jsonl + the
GOLD_TEST dataset the eval's alias resolves, and report old-vs-new
agreement; the endpoint runs it as a Job."""

from __future__ import annotations

import asyncio
import json
import time
import unittest

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.services import span_relabel_service as svc

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


T1 = "Call Ada Lovelace at ada@example.test about card 4242 4242 4242 4242 before Friday."
T2 = "Patient Bob Stone, MRN 112233, was seen by Dr. Chen on 03/04/2024 for asthma."


class FakeLabeller:
    name = "fake"
    prompts: list[str] = []

    def describe(self):
        return "cloud:anthropic:claude-haiku-4-5-20251001"

    async def complete(self, prompt, *, system_prompt=None, max_tokens=1024, temperature=0.0, response_schema=None):
        FakeLabeller.prompts.append(prompt)
        # Wrong offsets on purpose; the exact text is right. Index 1 gets
        # one entity the old labels didn't have (diagnosis).
        return json.dumps({"rows": [
            {"index": 0, "entities": [
                {"type": "person_name", "text": "Ada Lovelace", "start": 0, "end": 3},
                {"type": "email", "text": "ada@example.test", "start": 9, "end": 9},
                {"type": "credit_card", "text": "4242 4242 4242 4242"},
            ]},
            {"index": 1, "entities": [
                {"type": "person_name", "text": "Bob Stone", "start": 8, "end": 17},
                {"type": "medical_record_number", "text": "112233"},
                {"type": "provider_name", "text": "Dr. Chen"},
                {"type": "diagnosis", "text": "asthma"},
                {"type": "phone", "text": "555-0100"},  # not in the text → dropped
            ]},
        ]})


class RelabelTests(unittest.TestCase):
    def setUp(self):
        FakeLabeller.prompts = []
        self.pid = client.post("/api/projects", json={"name": f"relabel-{time.time_ns()}", "description": "x"}).json()["id"]
        prepared = settings.DATA_DIR / "projects" / str(self.pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        old = [
            {"text": T1, "entities": [
                {"type": "person_name", "start": 5, "end": 17, "text": "Ada Lovelace"},
                {"type": "email", "start": 21, "end": 37, "text": "ada@example.test"},
                {"type": "credit_card", "start": 49, "end": 68, "text": "4242 4242 4242 4242"},
            ]},
            {"text": T2, "entities": [
                {"type": "person_name", "start": 8, "end": 17, "text": "Bob Stone"},
                {"type": "medical_record_number", "start": 23, "end": 29, "text": "112233"},
                {"type": "provider_name", "start": 43, "end": 51, "text": "Dr. Chen"},
                {"type": "visit_date", "start": 55, "end": 65, "text": "03/04/2024"},
            ]},
            {"no_text": True},
        ]
        (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in old) + "\n", encoding="utf-8")

    def test_relabel_writes_the_answer_key_and_reports_agreement(self):
        async def _go():
            async with async_session_factory() as db:
                return await svc.relabel_split_as_answer_key(
                    db, self.pid, split="test", backend_override=FakeLabeller(),
                    entity_types=["person_name", "email", "credit_card", "medical_record_number", "provider_name", "visit_date", "diagnosis"],
                )

        summary = asyncio.run(_go())
        self.assertEqual((summary.rows, summary.labelled_rows, summary.failed_calls), (2, 2, 0))
        self.assertEqual(summary.backend, "cloud:anthropic:claude-haiku-4-5-20251001")
        self.assertEqual(summary.entities, 7)
        self.assertEqual(summary.entity_counts["person_name"], 2)
        self.assertNotIn("phone", summary.entity_counts)
        path = settings.DATA_DIR / "projects" / str(self.pid) / "gold" / "gold_test.jsonl"
        self.assertEqual(summary.file_path, str(path))
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual([r["text"] for r in rows], [T1, T2], "texts unchanged, in order")
        for r in rows:
            for e in r["entities"]:
                self.assertEqual(r["text"][e["start"]:e["end"]], e["text"], "offsets repaired from the entity text")
            self.assertEqual(r["source"], "answer_key:cloud:anthropic:claude-haiku-4-5-20251001")
        # The prompt carried the split's own entity types and numbered texts.
        self.assertIn("[0] " + T1, FakeLabeller.prompts[0])
        self.assertIn("credit_card", FakeLabeller.prompts[0])
        # Agreement: old labels vs the new key — 6 of 7 old spans match
        # (visit_date missing from the key), the key has one extra (diagnosis).
        agr = summary.agreement
        self.assertEqual((agr["tp"], agr["fp"], agr["fn"]), (6, 1, 1))
        self.assertEqual(agr["per_type"]["visit_date"]["fp"], 1)
        self.assertEqual(agr["per_type"]["diagnosis"]["fn"], 1)
        # The GOLD_TEST dataset record points at the file (the eval's gold_test alias).
        from app.models.dataset import Dataset, DatasetType

        async def _ds():
            from sqlalchemy import select

            async with async_session_factory() as db:
                return (await db.execute(select(Dataset).where(Dataset.project_id == self.pid, Dataset.dataset_type == DatasetType.GOLD_TEST))).scalars().all()

        datasets = asyncio.run(_ds())
        self.assertEqual(len(datasets), 1)
        self.assertEqual((datasets[0].file_path, datasets[0].record_count), (str(path), 2))
        self.assertEqual(datasets[0].metadata_["source"], "span_relabel")

    def test_a_failed_call_is_retried_once(self):
        calls = {"n": 0}
        inner = FakeLabeller()

        class Flaky(FakeLabeller):
            async def complete(self, prompt, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise __import__("app.services.synth_backends.base", fromlist=["SynthBackendError"]).SynthBackendError("rate limited")
                return await inner.complete(prompt, **kw)

        async def _go():
            async with async_session_factory() as db:
                return await svc.relabel_split_as_answer_key(db, self.pid, split="test", backend_override=Flaky())

        summary = asyncio.run(_go())
        self.assertEqual((calls["n"], summary.failed_calls, summary.labelled_rows), (2, 0, 2))

    def test_endpoint_runs_as_a_job(self):
        from unittest import mock

        async def fake_relabel(db, project_id, **kwargs):
            await asyncio.sleep(0.4)
            return svc.RelabelSummary(project_id=project_id, split=kwargs["split"], backend="fake", rows=2, labelled_rows=2, failed_calls=0, entities=7, entity_counts={"email": 1}, agreement={"f1": 0.9})

        with mock.patch("app.services.span_relabel_service.relabel_split_as_answer_key", fake_relabel):
            res = client.post(f"/api/projects/{self.pid}/synthetic/relabel-answer-key", json={"split": "test", "backend": "ollama"})
            self.assertEqual(res.status_code, 202, res.text)
            job_id = res.json()["id"]
            for _ in range(40):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.25)
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["result"]["agreement"]["f1"], 0.9)
        bad = client.post(f"/api/projects/{self.pid}/synthetic/relabel-answer-key", json={"split": "nope"})
        self.assertEqual(bad.status_code, 422)


if __name__ == "__main__":
    unittest.main()

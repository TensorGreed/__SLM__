"""The eval summary of a token-classification run (kind="span_eval"): its
own span scores — answer key first when there is one — never a
"no base-model comparison yet" verdict."""

from __future__ import annotations

import asyncio
import json
import time
import unittest

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import EvalResult, Experiment, ExperimentStatus

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def _metrics(f1, total=44):
    return {
        "precision": f1 + 0.01, "recall": f1 - 0.01, "f1": f1, "total": total, "scoring_mode": "span_set",
        "per_class": {"email": {"f1": 1.0, "tp": 5, "fn": 0}, "bank_account": {"f1": 0.2, "tp": 1, "fn": 4}},
    }


class SpanEvalSummaryTests(unittest.TestCase):
    def _seed(self, results):
        pid = client.post("/api/projects", json={"name": f"span-sum-{time.time_ns()}", "description": "x"}).json()["id"]

        async def _go():
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=pid, name="deberta tc", status=ExperimentStatus.COMPLETED,
                    base_model="microsoft/deberta-v3-base", output_dir="/tmp/none",
                    config={"task_type": "token_classification"},
                )
                db.add(exp)
                await db.flush()
                for dataset, f1, failures in results:
                    db.add(EvalResult(
                        experiment_id=exp.id, dataset_name=dataset, eval_type="f1", metrics=_metrics(f1),
                        details={"failures_preview": failures, "failed_count": len(failures)},
                    ))
                await db.commit()
                return exp.id

        return pid, asyncio.run(_go())

    def test_answer_key_leads_and_test_follows(self):
        failure = {"prompt": "Pay to account 12345678.", "reference": '{"entities": []}', "prediction": '{"entities": []}'}
        pid, exp = self._seed([("test", 0.89, []), ("gold_test", 0.84, [failure])])
        gold = settings.DATA_DIR / "projects" / str(pid) / "gold"
        gold.mkdir(parents=True, exist_ok=True)
        (gold / "gold_test.jsonl").write_text(json.dumps({"text": "x", "entities": [], "source": "answer_key:cloud:anthropic:claude-haiku-4-5"}) + "\n")
        body = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertEqual((body["kind"], body["verdict"]), ("span_eval", "evaluated"))
        self.assertEqual(body["answer_key"]["f1"], 0.84)
        self.assertEqual(body["answer_key"]["written_by"], "cloud:anthropic:claude-haiku-4-5")
        self.assertEqual(body["test"]["f1"], 0.89)
        self.assertEqual(body["answer_key"]["weakest_types"][0]["type"], "bank_account")
        self.assertEqual(body["dataset_name"], "gold_test", "failures come from the answer key")
        self.assertEqual(len(body["failures"]), 1)
        self.assertIsNone(body["message"])

    def test_without_an_answer_key_it_says_so(self):
        pid, exp = self._seed([("test", 0.9, [])])
        body = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertEqual(body["kind"], "span_eval")
        self.assertIsNone(body["answer_key"])
        self.assertIn("answer key with a second model", body["message"])

    def test_not_yet_evaluated_offers_the_button(self):
        pid, exp = self._seed([])
        body = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertEqual(body["verdict"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()

"""Human review of the answer key (P6 Trust): verdicts recorded on the
gold rows, corrections replacing the labels the eval reads, rejected rows
dropped by the eval loader, the queue ordered by disagreement with the
trained model, and the eval re-totalled over checked vs unchecked rows."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.services import gold_review_service as svc

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


T1 = "Call Ada Lovelace at ada@example.test before Friday."
T2 = "Patient Bob Stone, MRN 112233, was seen on 03/04/2024."
T3 = "Nothing sensitive here."


def _row(text, entities, **extra):
    return {"text": text, "entities": entities, "source": "answer_key:cloud:anthropic:claude-sonnet-5-5", **extra}


ROWS = [
    _row(T1, [{"type": "person_name", "start": 5, "end": 17, "text": "Ada Lovelace"}, {"type": "email", "start": 21, "end": 37, "text": "ada@example.test"}]),
    _row(T2, [{"type": "person_name", "start": 8, "end": 17, "text": "Bob Stone"}, {"type": "medical_record_number", "start": 23, "end": 29, "text": "112233"}]),
    _row(T3, []),
]


class PureTests(unittest.TestCase):
    def test_accept_then_clear(self):
        row = svc.apply_review(ROWS[0], verdict="accepted", note="fine")
        self.assertEqual(row["review"]["verdict"], "accepted")
        self.assertEqual(row["entities"], ROWS[0]["entities"])
        self.assertNotIn("review", svc.apply_review(row, verdict=None))

    def test_correction_replaces_entities_and_keeps_the_original(self):
        fixed = [{"type": "person_name", "start": 5, "end": 17}]
        row = svc.apply_review(ROWS[0], verdict="corrected", corrected={"entities": fixed})
        self.assertEqual(row["entities"], [{"type": "person_name", "start": 5, "end": 17, "text": "Ada Lovelace"}])
        self.assertEqual(row["review"]["original"]["entities"], ROWS[0]["entities"])
        # A second correction keeps the FIRST original; clearing restores it.
        again = svc.apply_review(row, verdict="corrected", corrected={"entities": []})
        self.assertEqual(again["review"]["original"]["entities"], ROWS[0]["entities"])
        self.assertEqual(svc.apply_review(again, verdict=None)["entities"], ROWS[0]["entities"])

    def test_correction_validation(self):
        with self.assertRaises(svc.GoldReviewError):
            svc.apply_review(ROWS[0], verdict="corrected", corrected={"entities": [{"type": "email", "start": 30, "end": 999}]})
        with self.assertRaises(svc.GoldReviewError):
            svc.apply_review(ROWS[0], verdict="corrected", corrected={"entities": [
                {"type": "a", "start": 5, "end": 17}, {"type": "b", "start": 10, "end": 20},
            ]})
        with self.assertRaises(svc.GoldReviewError):
            svc.apply_review(ROWS[0], verdict="corrected", corrected={})
        with self.assertRaises(svc.GoldReviewError):
            svc.apply_review(ROWS[0], verdict="maybe")

    def test_non_span_rows_correct_the_answer(self):
        row = {"question": "Capital of France?", "answer": "Lyon"}
        out = svc.apply_review(row, verdict="corrected", corrected={"answer": "Paris"})
        self.assertEqual(out["answer"], "Paris")
        self.assertEqual(out["review"]["original"], {"answer": "Lyon"})

    def test_queue_order_is_unreviewed_then_most_disagreement(self):
        items = [
            {"index": 0, "review": None, "agreement": 0.9},
            {"index": 1, "review": {"verdict": "accepted"}, "agreement": 0.1},
            {"index": 2, "review": None, "agreement": 0.2},
            {"index": 3, "review": None, "agreement": None},
        ]
        self.assertEqual([i["index"] for i in svc.order_queue(items)], [2, 0, 3, 1])

    def test_split_scores_and_staleness(self):
        rows = [dict(r) for r in ROWS]
        rows[0] = svc.apply_review(rows[0], verdict="accepted")
        row_scores = {
            "inputs": [svc.row_key(r) for r in rows],
            "span_tp": [2, 1, 0], "span_fp": [0, 1, 1], "span_fn": [0, 1, 0],
        }
        later = datetime.now(timezone.utc) + timedelta(minutes=1)
        split = svc.split_scores(row_scores, rows, eval_at=later)
        self.assertEqual(split["reviewed_scores"], {"precision": 1.0, "recall": 1.0, "f1": 1.0, "rows": 1})
        self.assertEqual(split["unreviewed_scores"]["rows"], 2)
        self.assertAlmostEqual(split["unreviewed_scores"]["precision"], 1 / 3, places=3)
        self.assertFalse(split["stale"])
        # A correction AFTER the eval makes it stale; a rejected row is
        # left out of both buckets.
        rows[1] = svc.apply_review(rows[1], verdict="corrected", corrected={"entities": []})
        rows[2] = svc.apply_review(rows[2], verdict="rejected")
        earlier = datetime.now(timezone.utc) - timedelta(minutes=1)
        split = svc.split_scores(row_scores, rows, eval_at=earlier)
        self.assertTrue(split["stale"])
        self.assertEqual(split["changed_since_eval"], 2)
        self.assertEqual(split["reviewed_scores"]["rows"], 2)
        self.assertEqual(split["unreviewed_scores"]["rows"], 0)
        self.assertIsNone(svc.split_scores({"keys": ["x"]}, rows))

    def test_review_line(self):
        self.assertIsNone(svc.review_line({"total": 3, "reviewed": 0}))
        self.assertEqual(
            svc.review_line({"total": 43, "reviewed": 12, "corrected": 3, "rejected": 1}),
            "12 of 43 rows checked by a person (3 corrected, 1 rejected)",
        )


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"review-{time.time_ns()}", "description": "x"}).json()["id"]
        self.gold = settings.DATA_DIR / "projects" / str(self.pid) / "gold"
        self.gold.mkdir(parents=True, exist_ok=True)
        svc.write_rows(self.gold / "gold_test.jsonl", [dict(r) for r in ROWS])

        async def _seed():
            async with async_session_factory() as db:
                db.add(Dataset(
                    project_id=self.pid, name="Answer key (test)", dataset_type=DatasetType.GOLD_TEST,
                    file_path=str(self.gold / "gold_test.jsonl"), record_count=3,
                ))
                await db.commit()

        asyncio.run(_seed())

    def _queue(self, **params):
        res = client.get(f"/api/projects/{self.pid}/gold/review", params={"compare": "false", **params})
        self.assertEqual(res.status_code, 200, res.text)
        return res.json()

    def test_queue_lists_rows_and_counts(self):
        body = self._queue()
        self.assertEqual((body["total"], body["reviewed"], body["unreviewed"]), (3, 0, 3))
        self.assertIsNone(body["model_experiment_id"])
        self.assertEqual([r["index"] for r in body["rows"]], [0, 1, 2])
        self.assertEqual(body["rows"][0]["text"], T1)
        self.assertEqual(len(body["rows"][0]["entities"]), 2)
        self.assertIsNone(body["rows"][0]["agreement"])

    def test_queue_orders_by_disagreement_with_the_model(self):
        async def fake(db, project_id, texts, experiment_id=None):
            # Row 1: model misses the MRN (agreement 0.67); row 0 exact; row 2 exact (both empty).
            by_text = {
                T1: ROWS[0]["entities"],
                T2: [ROWS[1]["entities"][0]],
                T3: [],
            }
            return 77, [by_text[t] for t in texts]

        with mock.patch("app.services.span_relabel_service.latest_span_run_predictions", fake):
            body = client.get(f"/api/projects/{self.pid}/gold/review").json()
        self.assertEqual(body["model_experiment_id"], 77)
        self.assertEqual([r["index"] for r in body["rows"]], [1, 0, 2])
        self.assertAlmostEqual(body["rows"][0]["agreement"], 0.6667, places=3)
        self.assertEqual(len(body["rows"][0]["predicted"]), 1)
        # Cached: a second queue read doesn't call the model again.
        cache = json.loads((self.gold / svc.PREDICTIONS_CACHE).read_text())
        self.assertEqual(cache["experiment_id"], 77)

    def test_verdicts_land_on_the_row_and_in_the_dataset_count(self):
        key = self._queue()["rows"][1]["key"]
        res = client.put(f"/api/projects/{self.pid}/gold/review", json={
            "index": 1, "key": key, "verdict": "corrected",
            "corrected": {"entities": [{"type": "person_name", "start": 8, "end": 17}]}, "note": "MRN is a ticket id",
        })
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["row"]["review"]["verdict"], "corrected")
        self.assertEqual(res.json()["corrected"], 1)
        rows = svc.load_rows(self.gold / "gold_test.jsonl")
        self.assertEqual(rows[1]["entities"], [{"type": "person_name", "start": 8, "end": 17, "text": "Bob Stone"}])
        self.assertEqual(rows[1]["review"]["note"], "MRN is a ticket id")

        res = client.put(f"/api/projects/{self.pid}/gold/review", json={"index": 2, "verdict": "rejected"})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual((res.json()["reviewed"], res.json()["rejected"]), (2, 1))
        async def _count():
            async with async_session_factory() as db:
                from sqlalchemy import select

                return (await db.execute(
                    select(Dataset.record_count).where(Dataset.project_id == self.pid, Dataset.dataset_type == DatasetType.GOLD_TEST)
                )).scalar_one()

        self.assertEqual(asyncio.run(_count()), 2, "a rejected row doesn't count")

        # The eval loader never sees the rejected row.
        from app.services.evaluation_service import _load_eval_rows

        self.assertEqual([r["text"] for r in _load_eval_rows(self.gold / "gold_test.jsonl")], [T1, T2])

        # Stale key → 400, bad index → 400, clear → restores.
        self.assertEqual(client.put(f"/api/projects/{self.pid}/gold/review", json={"index": 1, "key": "nope", "verdict": "accepted"}).status_code, 400)
        self.assertEqual(client.put(f"/api/projects/{self.pid}/gold/review", json={"index": 9, "verdict": "accepted"}).status_code, 400)
        res = client.put(f"/api/projects/{self.pid}/gold/review", json={"index": 1, "verdict": None})
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(res.json()["row"]["review"])
        self.assertEqual(len(svc.load_rows(self.gold / "gold_test.jsonl")[1]["entities"]), 2)

    def _seed_span_run(self, row_scores, created_at=None):
        async def _go():
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=self.pid, name="deberta tc", status=ExperimentStatus.COMPLETED,
                    base_model="microsoft/deberta-v3-base", output_dir="/tmp/none",
                    config={"task_type": "token_classification"},
                )
                db.add(exp)
                await db.flush()
                metrics = {
                    "precision": 0.8, "recall": 0.75, "f1": 0.77, "total": 3, "scoring_mode": "span_set",
                    "per_class": {"email": {"f1": 1.0, "tp": 1, "fn": 0}, "person_name": {"f1": 0.8, "tp": 2, "fn": 0}},
                }
                result = EvalResult(experiment_id=exp.id, dataset_name="gold_test", eval_type="f1", metrics=metrics, details={"row_scores": row_scores})
                if created_at is not None:
                    result.created_at = created_at
                db.add(result)
                await db.commit()
                return exp.id

        return asyncio.run(_go())

    def test_summary_and_model_card_split_checked_rows(self):
        rows = svc.load_rows(self.gold / "gold_test.jsonl")
        row_scores = {"inputs": [svc.row_key(r) for r in rows], "span_tp": [2, 1, 0], "span_fp": [0, 0, 1], "span_fn": [0, 1, 0]}
        exp = self._seed_span_run(row_scores, created_at=datetime.now(timezone.utc) - timedelta(minutes=5))
        # Nothing reviewed → no review block.
        body = client.get(f"/api/projects/{self.pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertEqual(body["kind"], "span_eval")
        self.assertIsNone(body["answer_key"]["review"])

        client.put(f"/api/projects/{self.pid}/gold/review", json={"index": 0, "verdict": "accepted"})
        client.put(f"/api/projects/{self.pid}/gold/review", json={"index": 2, "verdict": "rejected"})
        body = client.get(f"/api/projects/{self.pid}/evaluation/summary", params={"experiment_id": exp}).json()
        review = body["answer_key"]["review"]
        self.assertEqual((review["reviewed"], review["accepted"], review["rejected"]), (2, 1, 1))
        self.assertEqual(review["reviewed_scores"], {"precision": 1.0, "recall": 1.0, "f1": 1.0, "rows": 1})
        self.assertEqual(review["unreviewed_scores"]["rows"], 1)
        self.assertAlmostEqual(review["unreviewed_scores"]["f1"], 2 / 3, places=3)
        self.assertTrue(review["stale"], "a rejection after the eval means the eval scored a row that is now out")

        from app.services.span_model_package_service import build_span_model_card, span_evaluations

        async def _blocks():
            async with async_session_factory() as db:
                return await span_evaluations(db, self.pid, exp)

        blocks = asyncio.run(_blocks())
        self.assertEqual(blocks[0]["review"]["line"], "2 of 3 rows checked by a person (1 rejected)")
        card = build_span_model_card(entity_types=["email"], evaluations=blocks)
        self.assertIn("Human review: 2 of 3 rows checked by a person (1 rejected). On the checked rows: precision 1.00 · recall 1.00 · F1 1.00 (1 texts).", card)


if __name__ == "__main__":
    unittest.main()

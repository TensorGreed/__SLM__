"""Refine weak types: the latest span_set evaluation names the types under
the threshold, weakest first, with provenance and a suggested row count;
the endpoint serves it."""

from __future__ import annotations

import asyncio
import time
import unittest

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.services.span_refine_service import weak_entity_types, weak_types_from_metrics

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


METRICS = {
    "scoring_mode": "span_set", "f1": 0.9,
    "per_class": {
        "email": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 10, "fp": 0, "fn": 0},
        "passport": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 3},
        "medication": {"precision": 0.6, "recall": 0.6, "f1": 0.6, "tp": 3, "fp": 2, "fn": 2},
        "provider_name": {"precision": 0.54, "recall": 0.88, "f1": 0.6667, "tp": 7, "fp": 6, "fn": 1},
        "national_id": {"precision": 0.67, "recall": 0.89, "f1": 0.7619, "tp": 8, "fp": 4, "fn": 1},
    },
}


class WeakTypesTests(unittest.TestCase):
    def test_pure_ranking(self):
        weak = weak_types_from_metrics(METRICS, 0.8)
        self.assertEqual([w["type"] for w in weak], ["passport", "medication", "provider_name", "national_id"])
        self.assertEqual(weak[0]["fn"], 3)
        self.assertEqual(weak_types_from_metrics(METRICS, 0.5), [{"type": "passport", "f1": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 3}])
        self.assertEqual(weak_types_from_metrics({}, 0.8), [])

    def test_endpoint_reads_the_latest_span_eval(self):
        pid = client.post("/api/projects", json={"name": f"refine-{time.time_ns()}", "description": "x"}).json()["id"]
        empty = client.get(f"/api/projects/{pid}/synthetic/weak-types").json()
        self.assertEqual((empty["eval_result_id"], empty["types"], empty["suggested_rows"]), (None, [], 0))

        async def _seed():
            async with async_session_factory() as db:
                exp = Experiment(project_id=pid, name="tc", description="", status=ExperimentStatus.COMPLETED, base_model="microsoft/deberta-v3-base", config={"task_type": "token_classification"}, output_dir="/tmp/x")
                db.add(exp)
                await db.flush()
                db.add(EvalResult(experiment_id=exp.id, dataset_name="test", eval_type="f1", metrics={"scoring_mode": "field_match", "f1": 0.2, "per_class": {}}, pass_rate=0.2))
                db.add(EvalResult(experiment_id=exp.id, dataset_name="test", eval_type="f1", metrics=METRICS, pass_rate=0.9))
                db.add(EvalResult(experiment_id=exp.id, dataset_name="test", eval_type="exact_match", metrics={"exact_match": 0.5}, pass_rate=0.5))
                await db.commit()
                return exp.id, await weak_entity_types(db, pid)

        exp_id, payload = asyncio.run(_seed())
        self.assertEqual(payload["experiment_id"], exp_id)
        self.assertEqual([t["type"] for t in payload["types"]], ["passport", "medication", "provider_name", "national_id"])
        self.assertEqual(payload["suggested_rows"], 100)
        self.assertEqual(payload["overall_f1"], 0.9)
        self.assertIn("email", payload["all_types"])
        strict = client.get(f"/api/projects/{pid}/synthetic/weak-types", params={"threshold": 0.5}).json()
        self.assertEqual([t["type"] for t in strict["types"]], ["passport"])
        self.assertEqual(strict["suggested_rows"], 60)


if __name__ == "__main__":
    unittest.main()

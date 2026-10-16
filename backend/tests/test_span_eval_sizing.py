"""Per-type answer-key sizing (P6): how many reference spans a type needs
before a DECISION_GAP F1 difference between runs clears noise, the gap
the current support can show, and where it surfaces — span_block
(Eval card), the refine endpoint (under-sampled types) and the model card."""

from __future__ import annotations

import asyncio
import time
import unittest

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.services import span_eval_sizing_service as svc

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


PER_CLASS = {
    "email": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 600, "fp": 0, "fn": 0},
    "medication": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "tp": 16, "fp": 4, "fn": 4},
    "api_key": {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "tp": 1, "fp": 0, "fn": 1},
}


class PureTests(unittest.TestCase):
    def test_spans_needed_follows_the_paired_rule(self):
        # 8·p·(1−p)/Δ²: p=0.8 → 512 at Δ=0.05; p clamped to 0.95 at F1 1.0 → 152; p=0.5 → 800.
        self.assertEqual(svc.spans_needed(0.8), 512)
        self.assertEqual(svc.spans_needed(1.0), 152)
        self.assertEqual(svc.spans_needed(0.0), 800)
        self.assertEqual(svc.spans_needed(0.8, gap=0.1), 128)
        self.assertEqual(svc.spans_needed(None), 800)

    def test_detectable_gap(self):
        self.assertAlmostEqual(svc.detectable_gap(0.8, 20), 0.2530, places=3)
        self.assertAlmostEqual(svc.detectable_gap(0.8, 512), 0.05, places=3)
        self.assertIsNone(svc.detectable_gap(0.8, 0))

    def test_size_types_orders_by_shortfall(self):
        sizing = svc.size_types(PER_CLASS)
        self.assertEqual([r["type"] for r in sizing["types"]], ["api_key", "medication", "email"])
        self.assertEqual(sizing["short"], ["api_key", "medication"])
        self.assertEqual(sizing["enough"], ["email"])
        api = sizing["types"][0]
        self.assertEqual((api["support"], api["needed"], api["shortfall"], api["enough"]), (2, 712, 710, False))
        self.assertEqual(sizing["shortfall"], 710 + (512 - 20))
        self.assertEqual(svc.size_types({})["types"], [])

    def test_sizing_line(self):
        self.assertEqual(
            svc.sizing_line(svc.size_types(PER_CLASS)),
            "1 of 3 types have enough reference spans to show a 0.05 F1 difference; the others can only show 0.94 or more (api_key: 2 spans, needs 712).",
        )
        self.assertEqual(
            svc.sizing_line(svc.size_types({"email": PER_CLASS["email"]})),
            "Every type has enough reference spans to show a 0.05 F1 difference between runs.",
        )
        self.assertIsNone(svc.sizing_line(None))


class SurfacesTests(unittest.TestCase):
    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"sizing-{time.time_ns()}", "description": "x"}).json()["id"]

        async def _seed():
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=self.pid, name="tc", status=ExperimentStatus.COMPLETED, base_model="microsoft/deberta-v3-base",
                    output_dir="/tmp/none", config={"task_type": "token_classification"},
                )
                db.add(exp)
                await db.flush()
                db.add(EvalResult(
                    experiment_id=exp.id, dataset_name="gold_test", eval_type="f1",
                    metrics={"scoring_mode": "span_set", "precision": 0.9, "recall": 0.9, "f1": 0.9, "total": 40, "per_class": PER_CLASS},
                    details={},
                ))
                await db.commit()
                return exp.id

        self.exp = asyncio.run(_seed())

    def test_eval_summary_carries_sizing(self):
        body = client.get(f"/api/projects/{self.pid}/evaluation/summary", params={"experiment_id": self.exp}).json()
        sizing = body["answer_key"]["sizing"]
        self.assertEqual((sizing["gap"], sizing["short"], sizing["enough"]), (0.05, ["api_key", "medication"], ["email"]))
        self.assertEqual(sizing["types"][1]["detectable_gap"], 0.253)

    def test_refine_targets_under_sampled_types_too(self):
        body = client.get(f"/api/projects/{self.pid}/synthetic/weak-types", params={"threshold": 0.8}).json()
        self.assertEqual([t["type"] for t in body["types"]], ["api_key"], "weak = under the F1 threshold")
        self.assertEqual([t["type"] for t in body["under_sampled"]], ["api_key", "medication"])
        self.assertEqual(body["types_to_refine"], ["api_key", "medication"])
        self.assertEqual(body["suggested_rows"], 60)
        self.assertEqual(body["sizing_gap"], 0.05)

    def test_model_card_states_the_sizing(self):
        from app.services.span_model_package_service import build_span_model_card

        card = build_span_model_card(entity_types=list(PER_CLASS), evaluations=[
            {"label": "answer key", "dataset_name": "gold_test", "precision": 0.9, "recall": 0.9, "f1": 0.9, "rows": 40, "per_type": PER_CLASS},
        ])
        self.assertIn("1 of 3 types have enough reference spans to show a 0.05 F1 difference", card)


if __name__ == "__main__":
    unittest.main()

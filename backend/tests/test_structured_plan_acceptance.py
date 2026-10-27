"""Acceptance: a structured-plan project (recipe applied, schema bundle + rubric + pointers uploaded
through the config API) with a baseline and a trained run, both evaluated on the same 20 prepared test
rows through the held-out evaluation endpoint (model inference and the rubric judge are stubbed), then
read back through the eval summary: the pass rate, schema-valid rate and the rubric per-criterion block.

pytest-only. No real model, no network; the Jobs-free path (the endpoint evaluates inline)."""

from __future__ import annotations

import asyncio
import json
import os
import time
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.services import answer_judge_service as ajs
from app.services.data_adapter_service import map_record_with_adapter

from test_structured_plan_recipe import FILES, RUBRIC

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


N_ROWS = 20
POINTERS = ["/environment", "/steps/*/action"]


def has_acceptable(i: int) -> bool:
    return i % 5 == 0


def gold_plan(i: int) -> dict:
    return {
        "name": f"service-{i} rollout",
        "environment": "staging",
        "steps": [
            {"action": "update", "target": f"service-{i}", "environment": "staging"},
            {"action": "update", "target": f"service-{i}", "environment": "prod"},
        ],
    }


def gold_row(i: int) -> dict:
    row = {
        "row_kind": "structured_plan",
        "scenario": f"Roll out service-{i} to staging, then production.",
        "plan_json": gold_plan(i),
        "grid_cell": {"region": ["north", "south"][i % 2], "tier": "gold"},
    }
    if has_acceptable(i):
        row["acceptable"] = {"/environment": ["staging", "dev"]}
    return row


def trained_answer(i: int) -> str:
    plan = gold_plan(i)
    if has_acceptable(i):
        plan["environment"] = "dev"  # one of the acceptable values
    return json.dumps(plan)


def base_answer(i: int) -> str:
    plan = gold_plan(i)
    if i % 4 == 0:
        return "I would update billing first."  # not JSON
    if i % 4 == 1:
        plan["environment"] = "qa"  # schema violation
        return json.dumps(plan)
    if i % 4 == 2:  # valid, but production is updated first (the critical rubric criterion)
        plan["steps"] = [{"action": "update", "target": "reversed-target", "environment": "prod"}, plan["steps"][0]]
        plan["steps"][1]["action"] = "update"
        return json.dumps(plan)
    return json.dumps(plan)


def run_async(coro_fn):
    async def go():
        async with async_session_factory() as db:
            return await coro_fn(db)

    return asyncio.run(go())


class StructuredPlanAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"plan-accept-{time.time_ns()}", "description": "x"}).json()["id"]
        put = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "structured-plan"})
        self.assertEqual(put.status_code, 200, put.text)
        cfg = client.put(f"/api/projects/{self.pid}/structured-plan-config", json={
            "schema_files": FILES, "root_schema": "plan.schema.json", "rubric_yaml": RUBRIC,
            "pointers": POINTERS, "judge_enabled": True, "retrieve_passages": False})
        self.assertEqual(cfg.status_code, 200, cfg.text)
        self.config = cfg.json()

    def seed(self) -> tuple[int, int]:
        from app.services.dataset_service import _recipe_output_schema

        schema = run_async(lambda db: _recipe_output_schema(db, self.pid))
        self.assertEqual(schema["scoring_mode"], "structured_plan")
        prepared = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({
            "task_profile": "structured_extraction", "adapter_id": "structured-extraction", "output_schema": schema}))
        rows = []
        for i in range(N_ROWS):
            row = gold_row(i)
            mapped = map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config={"structured_plan": {
                "schema_files": FILES, "root_schema": "plan.schema.json", "project_id": 0, "retrieve_passages": False}})
            rows.append({**row, **mapped})
        (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        async def add(db):
            db.add(Dataset(project_id=self.pid, name="test", dataset_type=DatasetType.TEST, file_path=str(prepared / "test.jsonl"),
                           record_count=len(rows)))
            now = datetime.now(timezone.utc)
            base = Experiment(project_id=self.pid, name="base model", status=ExperimentStatus.COMPLETED, base_model="toy",
                              config={"is_baseline": True}, output_dir=str(prepared), completed_at=now)
            trained = Experiment(project_id=self.pid, name="trained", status=ExperimentStatus.COMPLETED, base_model="toy",
                                 config={}, output_dir=str(prepared), completed_at=now)
            db.add_all([base, trained])
            await db.commit()
            return base.id, trained.id

        return run_async(add)

    @staticmethod
    def fake_inference(answers):
        def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True, decoding="neutral"):
            assert apply_chat_template is True  # plan rows train and evaluate through the chat template
            assert max_new_tokens == 1024
            preds = [{"prompt": p["prompt"], "formatted_prompt": "", "reference": p["reference"], "prediction": answers(i),
                      "generated_tokens": 5, "latency_ms": 1.0} for i, p in enumerate(pairs)]
            return preds, {"engine": "stub", "decoding": {"mode": decoding}}

        return run

    def evaluate(self, exp_id: int, answers, judge: bool = True) -> dict:
        def complete(prompt, system_prompt, response_schema, max_tokens):
            self.assertIn("[brewslm:rubric-judge]", prompt)
            ordering = 1 if "reversed-target" in prompt else 4
            return json.dumps({"scores": [
                {"id": "ordering", "score": ordering, "reason": "order"},
                {"id": "naming", "score": 4, "reason": "name"},
            ]})

        async def noop_judge(question, reference, prediction, instruction=None):
            return None

        resolved = ajs.ResolvedJudge(label="stub:rubric-judge", judge=noop_judge, complete=complete)

        async def resolve_judge(db, project_id, project=None):
            return resolved if judge else None

        with ExitStack() as stack:
            stack.enter_context(patch("app.services.evaluation_service._run_local_inference", self.fake_inference(answers)))
            stack.enter_context(patch.object(ajs, "resolve_answer_judge", resolve_judge))
            stack.enter_context(patch.object(ajs, "project_cache", lambda pid: ajs.AnswerJudgeCache(None)))
            resp = client.post(f"/api/projects/{self.pid}/evaluation/run-heldout", json={
                "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 40, "model_path": None})
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_base_and_trained_side_by_side_in_the_eval_summary(self):
        base_id, trained_id = self.seed()
        base = self.evaluate(base_id, base_answer)
        trained = self.evaluate(trained_id, trained_answer)
        bm, tm = base["metrics"], trained["metrics"]

        self.assertEqual(bm["scoring_mode"], "structured_plan")
        self.assertEqual((bm["total"], tm["total"]), (N_ROWS, N_ROWS))
        self.assertEqual(tm["structured_plan_pass_rate"], 1.0)
        self.assertEqual(tm["schema_valid_rate"], 1.0)
        self.assertEqual(tm["field_match_rate"], 1.0)
        # base: 5 rows are not JSON, 5 break the enum, 5 have production first (rubric critical fail), 5 are right
        self.assertEqual(bm["json_validity_rate"], 0.75)
        self.assertEqual(bm["schema_valid_rate"], 0.5)
        self.assertEqual(bm["structured_plan_pass_rate"], 0.25)
        self.assertEqual(bm["schema_errors_top"][0]["validator"], "enum")
        self.assertEqual(bm["schema_errors_top"][0]["pointer"], "/environment")
        self.assertEqual(bm["schema"]["root"], "plan.schema.json")
        self.assertEqual(set(tm["per_pointer"]), set(POINTERS))
        self.assertEqual(tm["per_pointer"]["/environment"]["match_rate"], 1.0)  # dev is acceptable on the multi-value rows

        rubric = bm["rubric"]
        self.assertEqual(rubric["status"], "judged")
        self.assertEqual(rubric["judge"], "stub:rubric-judge")
        self.assertEqual(rubric["criteria_order"], ["ordering", "naming"])
        self.assertEqual(rubric["per_criterion"]["ordering"]["critical"], True)
        self.assertEqual(rubric["per_criterion"]["ordering"]["low_count"], 5)
        self.assertEqual(rubric["critical_fail_count"], 5)
        self.assertEqual(rubric["judged"], N_ROWS)
        self.assertEqual(tm["rubric"]["pass_rate"], 1.0)
        self.assertEqual(tm["rubric_pass_rate"], 1.0)
        self.assertEqual(tm["rubric_mean_score"], 4.0)
        self.assertEqual(tm["structured_plan_fingerprint"], bm["structured_plan_fingerprint"])

        summary = client.get(f"/api/projects/{self.pid}/evaluation/summary", params={"experiment_id": trained_id})
        self.assertEqual(summary.status_code, 200, summary.text)
        body = summary.json()
        self.assertEqual(body["verdict"], "better", body)
        self.assertEqual(body["headline"]["metric_id"], "structured_plan_pass_rate")
        lifts = {row["metric_id"]: row for row in body["metric_lifts"]}
        for metric in ("structured_plan_pass_rate", "schema_valid_rate", "json_validity_rate"):
            self.assertIn(metric, lifts, sorted(lifts))
        self.assertEqual(lifts["structured_plan_pass_rate"]["baseline_value"], 0.25)
        self.assertEqual(lifts["structured_plan_pass_rate"]["trained_value"], 1.0)
        self.assertEqual(lifts["schema_valid_rate"]["baseline_value"], 0.5)
        self.assertEqual(lifts["schema_valid_rate"]["direction"], "improved")

        preview = trained["details"]["predictions_preview"][0]
        self.assertTrue(preview["row_structured_plan_pass"])
        self.assertEqual(preview["row_rubric_status"], "judged")
        self.assertEqual(set(preview["row_rubric_scores"]), {"ordering", "naming"})
        self.assertEqual(preview["row_pointer_results"]["/environment"]["source"], "acceptable")
        self.assertEqual(preview["grid_cell"], {"region": "north", "tier": "gold"})
        bad = next(p for p in base["details"]["predictions_preview"] if p["row_schema_valid"] is False and p["is_valid_json"])
        self.assertEqual(bad["schema_errors"][0]["validator"], "enum")

        dump = os.environ.get("STRUCTURED_PLAN_DUMP")
        if dump:
            Path(dump).write_text(json.dumps({"base_metrics": bm, "trained_metrics": tm, "base_preview": base["details"]["predictions_preview"][:3],
                                              "trained_preview": trained["details"]["predictions_preview"][0],
                                              "config": self.config, "summary": body}, indent=2, default=str))

    def test_without_a_judge_the_rubric_is_reported_not_applied(self):
        base_id, trained_id = self.seed()
        m = self.evaluate(base_id, base_answer, judge=False)["metrics"]
        self.assertEqual(m["rubric"]["status"], "no_judge")
        self.assertEqual(m["rubric"]["id"], "plan-quality")
        # the rubric is ignored: the production-first rows pass on schema + pointers alone
        self.assertEqual(m["structured_plan_pass_rate"], 0.5)
        self.assertNotIn("rubric_pass_rate", m)


if __name__ == "__main__":
    unittest.main()

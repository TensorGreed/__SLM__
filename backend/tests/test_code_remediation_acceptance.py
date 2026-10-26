"""Acceptance: a code-remediation project with a baseline and a trained run, evaluated on the same
gold test rows through the held-out evaluation endpoint (model inference, detector and judge are
stubbed), then read back through the eval summary. The trained run's remediation metrics show
side by side with the base run's, with the judge block present.

pytest-only. No real model, no network; the Jobs-free path (the endpoint evaluates inline)."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project
from app.services import answer_judge_service as ajs
from app.services import remediation_detector as rd
from app.services.data_adapter_service import map_record_with_adapter
from app.services.unified_diff import make_unified_diff

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


TAXONOMY = [
    {"id": "injection", "title": "Injection"},
    {"id": "injection.query", "title": "Query injection", "parent": "injection"},
    {"id": "secret.hardcoded", "title": "Hard-coded secret"},
    {"id": "logging.sensitive", "title": "Sensitive value logged"},
]
LEAK = '    log.info("password=%s", password)\n'
SAFE = '    log.info("login attempt for %s", user)\n'
PASSAGES = [
    {"text": "Never write credentials to a log; log the account name instead.", "source": "logging-guidelines"},
    {"text": "def audit(user):\n    log.info('audit %s', user)", "path": "lib/audit.py", "start_line": 3, "end_line": 4, "symbol": "audit"},
]
N_ROWS = 4


def code_for(i: int) -> str:
    return f"def login_{i}(user, password):\n{LEAK}    return check(user, password)\n"


def fixed_for(i: int) -> str:
    return code_for(i).replace(LEAK, SAFE)


def patch_for(old: str, new: str) -> str:
    return make_unified_diff(old, new, path="app/auth.py", start_line=10)


def gold_row(i: int) -> dict:
    return {
        "row_kind": "remediation", "code": code_for(i), "language": "python", "path": "app/auth.py", "start_line": 10,
        "end_line": 12,
        "finding": {"label": "logging.sensitive", "start_line": 11, "end_line": 11, "evidence": "password written to a log", "attrs": {}},
        "passages": [dict(p) for p in PASSAGES],
        "remediation_json": {"patch": patch_for(code_for(i), fixed_for(i)), "rationale": "Log the account, not the password.",
                             "citations": ["P1"]},
    }


def good_answer(i: int) -> str:
    return json.dumps({"patch": patch_for(code_for(i), fixed_for(i)), "rationale": "Do not log the password.", "citations": ["P1"]})


def base_answer(i: int) -> str:
    if i == 0:
        return "I would remove the logging."  # malformed
    if i == 1:  # context that is not in the code
        other = code_for(i).replace("check(user, password)", "verify(user)")
        return json.dumps({"patch": patch_for(other, other.replace(LEAK, SAFE)), "rationale": "x", "citations": []})
    if i == 2:  # applies and parses, but the password is still logged
        return json.dumps({"patch": patch_for(code_for(i), code_for(i).replace("return", "# checked\n    return")),
                           "rationale": "Add a comment.", "citations": []})
    return "{not json"


def run_async(coro_fn):
    async def go():
        async with async_session_factory() as db:
            return await coro_fn(db)

    return asyncio.run(go())


class RemediationAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"accept-{time.time_ns()}", "description": "x"}).json()["id"]
        put = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "code-remediation"})
        self.assertEqual(put.status_code, 200, put.text)
        client.put(f"/api/projects/{self.pid}/label-taxonomy", json={"entries": TAXONOMY})
        self.pid_for_detector = self.pid

    def seed(self) -> tuple[int, int]:
        from app.services.dataset_service import _recipe_output_schema

        schema = run_async(lambda db: _recipe_output_schema(db, self.pid))
        prepared = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({
            "task_profile": "structured_extraction", "adapter_id": "structured-extraction", "output_schema": schema}))
        rows = []
        for i in range(N_ROWS):
            row = gold_row(i)
            mapped = map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config={"remediation": {
                "extra_fields": {}, "project_id": 0, "retrieve_passages": False, "top_k": 3, "retrieval": None, "label_titles": {}}})
            rows.append({**row, **mapped})
        (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        async def add(db):
            project = await db.get(Project, self.pid)
            project.runtime_config = {"remediation": {
                "detector": {"kind": "project", "project_id": self.pid, "experiment_id": None}, "retrieve_passages": False}}
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
            preds = [{"prompt": p["prompt"], "formatted_prompt": "", "reference": p["reference"], "prediction": answers(i),
                      "generated_tokens": 5, "latency_ms": 1.0} for i, p in enumerate(pairs)]
            return preds, {"engine": "stub", "decoding": {"mode": decoding}}

        return run

    def evaluate(self, exp_id: int, answers) -> dict:
        async def judge(question, reference, prediction, instruction=None):
            # the stub judge: a patch that removes the logged password is correct
            if any(ln.startswith("+") and "login attempt" in ln for ln in prediction.splitlines()):
                return 1.0, "correct", "removes the leak", 0
            return 0.0, "wrong", "the finding remains", 0

        async def retrieval(question, reference, passages, instruction=None):
            return "found", 1, "p1", 0

        resolved = ajs.ResolvedJudge(label="stub:judge", judge=judge, retrieval_judge=retrieval)

        async def resolve_judge(db, project_id, project=None):
            return resolved

        detector = rd.ResolvedDetector(
            "project", "Toy detector · run 4", "acceptance:v1",
            lambda chunks: [["logging.sensitive"] if "password=%s" in c["code"] else [] for c in chunks],
        )

        async def resolve_detector(db, cfg):
            return detector

        with ExitStack() as stack:
            stack.enter_context(patch("app.services.evaluation_service._run_local_inference", self.fake_inference(answers)))
            stack.enter_context(patch.object(ajs, "resolve_answer_judge", resolve_judge))
            stack.enter_context(patch.object(ajs, "project_cache", lambda pid: ajs.AnswerJudgeCache(None)))
            stack.enter_context(patch.object(rd, "resolve_detector", resolve_detector))
            resp = client.post(f"/api/projects/{self.pid}/evaluation/run-heldout", json={
                "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 20, "model_path": None})
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_base_and_trained_side_by_side_in_the_eval_summary(self):
        base_id, trained_id = self.seed()
        base = self.evaluate(base_id, base_answer)
        trained = self.evaluate(trained_id, good_answer)

        bm, tm = base["metrics"], trained["metrics"]
        self.assertEqual(bm["scoring_mode"], "remediation")
        self.assertEqual((bm["total"], tm["total"]), (N_ROWS, N_ROWS))
        # the same denominators for both models
        self.assertEqual(bm["code_checkable_count"], tm["code_checkable_count"])
        self.assertEqual(tm["remediation_pass_rate"], 1.0)
        self.assertEqual(tm["patch_applies_rate"], 1.0)
        self.assertEqual(tm["code_parses_rate"], 1.0)
        self.assertEqual(tm["issue_resolved_overall_rate"], 1.0)
        self.assertEqual(tm["judge_correct"], 1.0)
        self.assertEqual(bm["patch_applies_rate"], 0.25)  # only the comment-adding patch applies
        self.assertEqual(bm["issue_resolved_overall_rate"], 0.0)
        self.assertEqual(bm["judge_correct"], 0.25)  # the judge reads the text: the non-applying patch looks right to it
        self.assertLess(bm["remediation_pass_rate"], tm["remediation_pass_rate"])
        self.assertTrue(tm["remediation_fingerprint"])
        self.assertEqual(tm["remediation_fingerprint"], bm["remediation_fingerprint"])

        summary = client.get(f"/api/projects/{self.pid}/evaluation/summary", params={"experiment_id": trained_id})
        self.assertEqual(summary.status_code, 200, summary.text)
        body = summary.json()
        self.assertEqual(body["verdict"], "better", body)
        self.assertEqual(body["headline"]["metric_id"], "remediation_pass_rate")
        lifts = {row["metric_id"]: row for row in body["metric_lifts"]}
        for metric in ("remediation_pass_rate", "patch_applies_rate", "code_parses_rate", "issue_resolved_overall_rate", "judge_correct"):
            self.assertIn(metric, lifts, sorted(lifts))
            self.assertIsNotNone(lifts[metric]["baseline_value"], metric)
            self.assertIsNotNone(lifts[metric]["trained_value"], metric)
        self.assertEqual(lifts["patch_applies_rate"]["baseline_value"], 0.25)
        self.assertEqual(lifts["patch_applies_rate"]["trained_value"], 1.0)
        self.assertEqual(lifts["judge_correct"]["trained_value"], 1.0)
        self.assertEqual(lifts["issue_resolved_overall_rate"]["direction"], "improved")
        # prediction-dependent denominators are not lift metrics
        for hidden in ("issue_resolved_rate", "new_findings_rate", "code_checkable_count"):
            self.assertNotIn(hidden, lifts)
        self.assertIsNotNone(body["judge"])
        self.assertEqual(body["judge"]["score"], 1.0)
        self.assertEqual(body["judge"]["judge"], "stub:judge")


if __name__ == "__main__":
    unittest.main()

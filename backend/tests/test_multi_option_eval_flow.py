"""End to end: a toy config-fix project in ``multi_option`` scoring, a base and a trained run evaluated
on the same held-out rows through the held-out evaluation endpoint (model inference stubbed; the option
verifier is a real tiny script registered in an operator file and run through the sandbox), then read
back through the eval summary; plus the settings API and the lift check's fingerprint.

pytest-only. No real model, no network; the Jobs-free path (the endpoint evaluates inline)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project
from app.services import external_checks as ec

import _multi_option_toy as toy

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


N_ROWS = 3
MO_CONFIG = {"formats": toy.FORMATS, "min_options": 2, "max_options": 3, "strategy_taxonomy": toy.TAXONOMY,
             "verifier": {"check_id": "toy_option_verifier"}, "require_agreement": True, "row_pass": "top_option"}


def gold_row(i: int) -> dict:
    start = {**toy.START, "name": f"svc{i}"}
    return {
        "context": "Service config (debug output must be off; the port must stay as it is):\n" + toy.config_text(start),
        "verifier_payload": {**toy.PAYLOAD, "start": start},
        "options_json": {"explanation": "Turn debug off.", "options": [toy.good_option(), toy.restrict_option()]},
    }


def _opt_for(i: int, option: dict) -> dict:
    """Rebase a toy option onto row i's starting config (the name line)."""
    out = json.loads(json.dumps(option))
    out["fixes"]["config_file"] = out["fixes"]["config_file"] + f"name=svc{i}\n"
    return out


def trained_answer(i: int) -> str:
    return toy.answer(_opt_for(i, toy.good_option()), _opt_for(i, toy.restrict_option()))


def base_answer(i: int) -> str:
    if i == 0:
        return "Just turn it off."  # malformed
    if i == 1:  # top option breaks the port; second disagrees across formats
        return toy.answer(_opt_for(i, toy.breaking_option()), _opt_for(i, toy.disagreeing_option()))
    return toy.answer(_opt_for(i, toy.good_option()), {"strategy": "exit-nonzero", "fixes": {"config_file": "x", "commands": "x"}})


def run_async(fn):
    async def go():
        async with async_session_factory() as db:
            return await fn(db)

    return asyncio.run(go())


class MultiOptionFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="brewslm-mo-flow-"))
        self.file = toy.write_operator_file(self.tmp)
        self._patches = [
            patch.object(settings, "EXTERNAL_CHECKS_FILE", str(self.file)),
            patch.object(settings, "CODE_CHECK_ALLOWED", "toy_option_verifier"),
        ]
        for p in self._patches:
            p.start()
        ec.ensure_loaded(force=True)
        self.pid = client.post("/api/projects", json={"name": f"mo-{time.time_ns()}", "description": "x"}).json()["id"]

    def tearDown(self):
        for p in self._patches:
            p.stop()
        ec.ensure_loaded(force=True)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def seed(self, schema_cfg: dict | None = None) -> tuple[int, int]:
        prepared = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({
            "task_profile": "structured_extraction", "adapter_id": "structured-extraction",
            "output_schema": {"scoring_mode": "multi_option", "multi_option": schema_cfg or MO_CONFIG}}))
        rows = [gold_row(i) for i in range(N_ROWS)]
        (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        async def add(db):
            db.add(Dataset(project_id=self.pid, name="test", dataset_type=DatasetType.TEST,
                           file_path=str(prepared / "test.jsonl"), record_count=len(rows)))
            now = datetime.now(timezone.utc)
            base = Experiment(project_id=self.pid, name="base model", status=ExperimentStatus.COMPLETED, base_model="toy",
                              config={"is_baseline": True}, output_dir=str(prepared), completed_at=now)
            trained = Experiment(project_id=self.pid, name="trained", status=ExperimentStatus.COMPLETED, base_model="toy",
                                 config={}, output_dir=str(prepared), completed_at=now)
            db.add_all([base, trained])
            await db.commit()
            return base.id, trained.id

        return run_async(add)

    def evaluate(self, exp_id: int, answers) -> dict:
        seen_prompts: list[str] = []

        def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True, decoding="neutral"):
            seen_prompts.extend(p["prompt"] for p in pairs)
            preds = [{"prompt": p["prompt"], "formatted_prompt": "", "reference": p["reference"], "prediction": answers(i),
                      "generated_tokens": 5, "latency_ms": 1.0} for i, p in enumerate(pairs)]
            return preds, {"engine": "stub", "decoding": {"mode": decoding}}

        with patch("app.services.evaluation_service._run_local_inference", run):
            resp = client.post(f"/api/projects/{self.pid}/evaluation/run-heldout", json={
                "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 20, "model_path": None})
        self.assertEqual(resp.status_code, 201, resp.text)
        for prompt in seen_prompts:
            # the hidden assertions never reach the model
            self.assertNotIn("must_change", prompt)
            self.assertNotIn("must_keep", prompt)
        return resp.json()

    def test_base_and_trained_side_by_side(self):
        base_id, trained_id = self.seed()
        base = self.evaluate(base_id, base_answer)
        trained = self.evaluate(trained_id, trained_answer)
        bm, tm = base["metrics"], trained["metrics"]

        self.assertEqual(tm["scoring_mode"], "multi_option")
        self.assertEqual(tm["multi_option_pass_rate"], 1.0)
        self.assertEqual(tm["option_pass_rate"], 1.0)
        self.assertEqual(tm["rows_with_two_distinct_passing_rate"], 1.0)
        self.assertEqual(tm["cross_format_agreement_rate"], 1.0)
        self.assertEqual(tm["verifier_error_rows"], 0)
        self.assertEqual(tm["multi_option"]["verifier"], {"id": "toy_option_verifier", "revision": 1})
        self.assertTrue(tm["multi_option_fingerprint"])
        self.assertEqual(tm["multi_option_fingerprint"], bm["multi_option_fingerprint"])

        # base: row 0 malformed (fail), row 1 top option breaks the port (fail), row 2 top option passes
        # and the verifier crashes on option 1 -- an option-level failure, so the row is decided by its
        # top option (pass) and nothing is excluded.
        self.assertEqual(bm["verifier_error_rows"], 0)
        self.assertEqual(bm["option_verifier_failures"], 1)
        self.assertEqual(bm["total"], 3)
        self.assertEqual(bm["multi_option_pass_rate"], round(1 / 3, 4))
        self.assertEqual(bm["malformed_rate"], round(1 / 3, 4))
        self.assertEqual(bm["option_pass_rate"], 0.25)  # good passes; breaking, disagreeing, crashed fail
        self.assertEqual(bm["cross_format_agreement_rate"], round(2 / 3, 4))  # breaking + good agree
        self.assertEqual(bm["check_pass_rate"]["behaviour_preserved"], round(2 / 3, 4))

        preview = trained["details"]["predictions_preview"]
        row = preview[0]
        self.assertTrue(row["row_multi_option_pass"])
        self.assertEqual([o["strategy"] for o in row["row_options"]], ["turn-off", "restrict"])
        self.assertEqual(set(row["row_options"][0]["formats"]), set(toy.FORMATS))
        self.assertNotIn("multi_option_row", row)
        self.assertNotIn("must_change", json.dumps(preview))  # the verifier payload never reaches the UI
        base_rows = base["details"]["predictions_preview"]
        self.assertFalse(base_rows[2]["row_verifier_error"])
        self.assertTrue(base_rows[2]["row_multi_option_pass"])
        crashed = base_rows[2]["row_options"][1]
        self.assertIs(crashed["passes"], False)
        self.assertEqual(crashed["failure_reason"], "verifier_crash")
        self.assertIn("exit 3", crashed["note"])

        summary = client.get(f"/api/projects/{self.pid}/evaluation/summary", params={"experiment_id": trained_id})
        self.assertEqual(summary.status_code, 200, summary.text)
        body = summary.json()
        self.assertEqual(body["headline"]["metric_id"], "multi_option_pass_rate")
        lifts = {r["metric_id"]: r for r in body["metric_lifts"]}
        self.assertIn("option_pass_rate", lifts)
        for hidden in ("verifier_error_rows", "options_scored"):
            self.assertNotIn(hidden, lifts)

        # verdicts are cached per project: a second trained eval does not run the verifier again
        with patch("app.services.sandbox_exec.run_sandboxed", side_effect=AssertionError("re-verified")):
            again = self.evaluate(trained_id, trained_answer)
        self.assertEqual(again["metrics"]["multi_option_pass_rate"], 1.0)

    def test_verifier_not_allow_listed_marks_rows_never_fails(self):
        _base_id, trained_id = self.seed()
        with patch.object(settings, "CODE_CHECK_ALLOWED", ""):
            result = self.evaluate(trained_id, trained_answer)
        m = result["metrics"]
        self.assertEqual(m["verifier_error_rows"], N_ROWS)
        self.assertIn("CODE_CHECK_ALLOWED", m["verifier_unavailable"])
        self.assertIsNone(m["multi_option_pass_rate"])
        self.assertIsNone(m["exact_match"])  # not scored, never 0.0
        row_scores = result["details"].get("row_scores") or {}
        self.assertEqual(row_scores.get("correct"), [None] * N_ROWS)  # excluded rows are never "correct"
        self.assertNotIn("multi_option_pass", row_scores)

    def test_revision_bump_changes_the_fingerprint_and_blocks_baseline_reuse(self):
        from app.services import post_training_eval_service as pte

        self.seed()
        before = run_async(lambda db: pte._multi_option_fingerprint_for(db, self.pid))
        toy.write_operator_file(self.tmp, revision=2)
        os.utime(self.file, (time.time() + 5, time.time() + 5))
        after = run_async(lambda db: pte._multi_option_fingerprint_for(db, self.pid))
        self.assertTrue(before and after)
        self.assertNotEqual(before, after)

        class Result:
            metrics = {"multi_option_fingerprint": before, "prompt_format_version": 2}

        self.assertTrue(pte._baseline_scoring_matches(Result(), before))
        self.assertFalse(pte._baseline_scoring_matches(Result(), after))

    def test_settings_api(self):
        self.seed()
        got = client.get(f"/api/projects/{self.pid}/multi-option-config")
        self.assertEqual(got.status_code, 200, got.text)
        body = got.json()
        self.assertEqual([v["id"] for v in body["verifiers"]], ["toy_option_verifier"])
        self.assertTrue(body["verifiers"][0]["allowed"])
        self.assertNotIn("argv", json.dumps(body))  # operators' command lines are never exposed
        self.assertEqual(body["effective"]["formats"], toy.FORMATS)

        bad = client.put(f"/api/projects/{self.pid}/multi-option-config", json={"verifier": {"check_id": "python_ast"}})
        self.assertEqual(bad.status_code, 422)
        envelope = bad.json()
        self.assertEqual(envelope["error_code"], "INVALID_MULTI_OPTION_CONFIG")
        self.assertIn("not an option verifier", envelope["detail"])

        bad = client.put(f"/api/projects/{self.pid}/multi-option-config", json={"min_options": 5, "max_options": 2})
        self.assertEqual(bad.status_code, 422)

        # enabled with formats from the manifest: fine
        ok = client.put(f"/api/projects/{self.pid}/multi-option-config", json={"enabled": True})
        self.assertEqual(ok.status_code, 200, ok.text)

        ok = client.put(f"/api/projects/{self.pid}/multi-option-config",
                        json={"row_pass": "any_option", "verifier": {"check_id": "toy_option_verifier", "argv": ["/bin/sh"]}})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(ok.json()["effective"]["row_pass"], "any_option")
        self.assertEqual(ok.json()["effective"]["verifier"], {"check_id": "toy_option_verifier"})


class EnabledWithoutFormatsTests(unittest.TestCase):
    def test_enabled_config_with_no_format_is_rejected(self):
        pid = client.post("/api/projects", json={"name": f"mo-nofmt-{time.time_ns()}", "description": "x"}).json()["id"]
        bad = client.put(f"/api/projects/{pid}/multi-option-config", json={"enabled": True, "row_pass": "any_option"})
        self.assertEqual(bad.status_code, 422, bad.text)
        envelope = bad.json()
        self.assertEqual(envelope["error_code"], "INVALID_MULTI_OPTION_CONFIG")
        self.assertIn("at least one format", envelope["detail"])
        self.assertNotIn("multi_option", client.get(f"/api/projects/{pid}/multi-option-config").json()["config"])
        ok = client.put(f"/api/projects/{pid}/multi-option-config", json={"enabled": True, "formats": ["format_a"]})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertTrue(ok.json()["enabled"])


if __name__ == "__main__":
    unittest.main()

"""Unjudged rows in a held-out eval (judge resilience A1), end to end on a throwaway engine.

An unreachable judge: the breaker stops the calls after 3 failed rows, the eval still finishes and
writes every answer; unjudged rows are left out of pass / fail (``correct`` None, not in
``failed_count`` / ``failures_preview``); the full predictions file stores what a later re-judge
needs (``judge_pending``); the Job-result fields say the judge is incomplete; and the lift check will
not reuse such a result as the base bar. Model and judge are stubbed (no model eval, no network).
"""

from __future__ import annotations

import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_eval_judge_calibration import LONG  # shared harness: run_eval (model + judge stubbed)

from app.config import settings
from app.services import eval_predictions_service as preds
from app.services import judge_call_guard as jcg
from app.services.answer_judge_service import ResolvedJudge
from app.services.synth_backends.base import SynthBackendTransientError


client = None


def setUpModule():
    global client
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


class _DownJudge:
    def __init__(self):
        self.calls = 0

    async def __call__(self, q, r, p, instruction=None):
        self.calls += 1
        raise SynthBackendTransientError("Command backend failed: connection_failed")


class _UpJudge:
    async def __call__(self, q, r, p, instruction=None):
        return 1.0, "correct", "ok", 0


class UnjudgedEndToEndTests(unittest.TestCase):
    ROWS = [{"question": f"What about item {i}?", "answer": f"Item {i} is {LONG}"} for i in range(6)]

    def setUp(self):
        jcg.reset_registry()
        patcher = patch.object(jcg.settings, "JUDGE_TRANSIENT_RETRIES", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        jcg.reset_registry()
        for pid in getattr(self, "_pids", []):
            shutil.rmtree(settings.DATA_DIR / "projects" / str(pid), ignore_errors=True)

    def _run(self, **kw):
        from test_eval_judge_calibration import run_eval as harness

        pid, result = harness(self.ROWS, **kw)
        self._pids = [*getattr(self, "_pids", []), pid]
        return pid, result

    def test_outage_leaves_rows_unscored_and_saves_every_answer(self):
        judge = _DownJudge()
        pid, result = self._run(answer_judge=ResolvedJudge(label="down:judge", judge=judge))
        snap = result.metrics["judge"]
        self.assertEqual(judge.calls, 3, "the breaker stops after the 3rd failed row")
        self.assertEqual((snap["judged"], snap["unjudged"], snap["status"]), (0, 6, "paused"))
        self.assertEqual(snap["unjudged_reasons"], {"transient_error": 3, "judge_unreachable": 3})
        self.assertNotIn("judge_correct", result.metrics)
        details = result.details
        # Half the stub predictions are not an exact match: before A1 they counted as failures.
        self.assertEqual(details["failed_count"], 0)
        self.assertEqual(details["failures_preview"], [])
        self.assertEqual(details["row_scores"]["correct"], [None] * 6)
        records = preds.read_records(pid, result.id)
        self.assertEqual(len(records), 6, "every answer is saved")
        for record in records:
            self.assertNotIn("correct", record)
            self.assertNotIn("judge", record)
            self.assertEqual(record["judge_pending"]["status"], "unjudged")
            self.assertEqual(record["judge_pending"]["task"], "answer")
            self.assertEqual(record["judge_pending"]["label"], "down:judge")
            self.assertTrue(record["judge_pending"]["content_key"])
        self.assertEqual(records[5]["judge_pending"]["reason"], "judge_unreachable")
        fields = jcg.job_result_fields(result)
        self.assertEqual(fields["judge_status"], "paused")
        self.assertEqual((fields["judge_unjudged"], fields["judge_judged"]), (6, 0))
        self.assertEqual(fields["judge_label"], "down:judge")
        self.assertEqual(fields["judge_eval_result_id"], result.id)

        from app.services.post_training_eval_service import _baseline_judge_matches

        self.assertFalse(_baseline_judge_matches(result, "down:judge"),
                         "a base result with no judge verdicts is no lift bar (re-run)")

    def test_healthy_judge_output_is_unchanged(self):
        _, result = self._run(answer_judge=ResolvedJudge(label="up:judge", judge=_UpJudge()))
        snap = result.metrics["judge"]
        for key in ("status", "unjudged_reasons", "coverage", "breaker"):
            self.assertNotIn(key, snap)
        self.assertEqual(jcg.job_result_fields(result), {})
        from app.services.post_training_eval_service import _baseline_judge_matches

        self.assertTrue(_baseline_judge_matches(result, "up:judge"))

    def test_partial_outage_marks_base_incomplete_and_unparseable_does_not(self):
        from app.services.post_training_eval_service import _baseline_judge_matches

        partial = SimpleNamespace(metrics={"judge": {"judge": "j", "judged": 9, "unjudged": 1,
                                                     "unjudged_reasons": {"transient_error": 1}}})
        self.assertFalse(_baseline_judge_matches(partial, "j"))
        unparseable = SimpleNamespace(metrics={"judge": {"judge": "j", "judged": 9, "unjudged": 1,
                                                         "unjudged_reasons": {"unparseable": 1}}})
        self.assertTrue(_baseline_judge_matches(unparseable, "j"))
        paused_code = SimpleNamespace(metrics={"judge": {"judge": "j", "judged": 3, "unjudged": 0},
                                               "code_example_judge": {"judge": "j", "status": "paused",
                                                                      "judged": 0, "unjudged": 4}})
        self.assertFalse(_baseline_judge_matches(paused_code, "j"))


class PipelineEvalUnscoredTests(unittest.TestCase):
    def test_none_correct_is_row_unscored_not_wrong(self):
        """A pipeline stage row the scorer excluded (``correct`` None, e.g. a code example without a judge
        verdict) is unscored with ``row_unscored`` -- before A1 it was counted correct (1)."""
        import asyncio

        from app.services import pipeline_eval_service as pes

        units = [SimpleNamespace(error=None, prompt="task", output="{}", correct=None, unscored_reason=None,
                                 reference="{}")]
        scorer = SimpleNamespace(handler=SimpleNamespace(profile_id="x", build_prompts=lambda rows, ctx: [
            SimpleNamespace(prompt="task", reference="{}", extras={})]), eval_ctx=None, reference_fields=[], object_fields=[])
        sc = SimpleNamespace(stage={"name": "s", "output": "text"}, project=SimpleNamespace(id=1))

        async def fake_score(db, pid, preds, **kw):
            return {"handler": "x", "scoring_mode": "remediation", "judge": None, "correct": [None]}

        with patch.object(pes, "pseudo_row", lambda u, s: {}), \
                patch("app.services.evaluation_service.score_predictions_like_heldout", fake_score):
            asyncio.run(pes._score_slot(None, sc, units, scorer))
        self.assertIsNone(units[0].correct)
        self.assertEqual(units[0].unscored_reason, "row_unscored")


if __name__ == "__main__":
    unittest.main()

"""LLM-judge correctness for long-answer held-out evals (answer_judge_service).

Pins:
  * when the judge applies (generative profile + long references) and what
    it parses;
  * judge_predictions annotates rows, counts verdicts, uses the cache and
    survives a judge that fails on a row;
  * the eval service folds it in: judged rows decide pass/fail, row_scores
    carry ``judge_correct`` with None for unjudged rows, and the lift
    pairing skips those rows instead of dropping the comparison;
  * resolution honours EVAL_JUDGE_BACKEND=none and the project opt-out.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("DEBUG", "false")

from app.services import answer_judge_service as judge_svc
from app.services.answer_judge_service import (
    AnswerJudgeCache,
    ResolvedJudge,
    judge_predictions,
    parse_verdict,
    should_judge,
)

LONG = "The head of a government institution may refuse to disclose records containing personal information unless consent is given."
SHORT = "Section 19"


class ApplicabilityAndParsingTests(unittest.TestCase):
    def test_applies_to_long_answer_generative_tasks_only(self):
        self.assertTrue(should_judge("qa", [LONG] * 5))
        self.assertTrue(should_judge("instruction_sft", [LONG, LONG, SHORT]))
        self.assertFalse(should_judge("qa", [SHORT] * 5), "short answers keep exact match / F1")
        self.assertFalse(should_judge("classification", [LONG] * 5))
        self.assertFalse(should_judge("structured_extraction", [LONG] * 5))
        self.assertFalse(should_judge(None, [LONG]))
        self.assertFalse(should_judge("qa", []))

    def test_parse_verdict_tolerates_fences_and_rejects_unknown(self):
        self.assertEqual(parse_verdict('{"verdict": "correct", "reason": "same facts"}'), (1.0, "correct", "same facts"))
        self.assertEqual(parse_verdict('```json\n{"verdict":"Partial","reason":"misses consent"}\n```')[1], "partial")
        self.assertEqual(parse_verdict('Sure: {"verdict": "wrong", "reason": "x"} ok')[0], 0.0)
        self.assertIsNone(parse_verdict('{"verdict": "maybe"}'))
        self.assertIsNone(parse_verdict(""))
        self.assertIsNone(parse_verdict("not json"))


class JudgePredictionsTests(unittest.TestCase):
    def _rows(self):
        return [
            {"prompt": "Q1", "reference": LONG, "prediction": "Refuse unless consent."},
            {"prompt": "Q2", "reference": LONG, "prediction": "Anything goes."},
            {"prompt": "Q3", "reference": LONG, "prediction": "Some of it."},
            {"prompt": "Q4", "reference": LONG, "prediction": "boom"},
        ]

    def test_annotates_counts_and_tolerates_a_failing_row(self):
        calls = []

        async def judge(q, r, p):
            calls.append(q)
            if p == "boom":
                raise RuntimeError("judge down")
            return {"Refuse unless consent.": (1.0, "correct", "ok", 12), "Anything goes.": (0.0, "wrong", "contradicts", 10), "Some of it.": (0.5, "partial", "misses", 11)}[p]

        rows = self._rows()
        snap = asyncio.run(judge_predictions(rows, judge, label="fake:judge"))
        self.assertEqual(snap["counts"], {"correct": 1, "partial": 1, "wrong": 1})
        self.assertEqual(snap["judged"], 3)
        self.assertEqual(snap["unjudged"], 1)
        self.assertEqual(snap["score"], 0.5)
        self.assertEqual(snap["strict_correct_rate"], round(1 / 3, 4))
        self.assertEqual(snap["judge_calls"], 4)
        self.assertEqual(snap["judge_tokens"], 33)
        self.assertEqual(rows[0]["row_judge_score"], 1.0)
        self.assertEqual(rows[1]["row_judge_verdict"], "wrong")
        self.assertEqual(rows[2]["row_judge_reason"], "misses")
        self.assertNotIn("row_judge_score", rows[3])

    def test_cache_makes_a_repeat_eval_free(self):
        calls = []

        async def judge(q, r, p):
            calls.append(p)
            return (1.0, "correct", "ok", 5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            rows = self._rows()[:2]
            first = asyncio.run(judge_predictions(rows, judge, label="fake", cache=AnswerJudgeCache(path)))
            self.assertEqual(first["judge_calls"], 2)
            self.assertTrue(path.exists())
            again = asyncio.run(judge_predictions(self._rows()[:2], judge, label="fake", cache=AnswerJudgeCache(path)))
            self.assertEqual(again["judge_calls"], 0)
            self.assertEqual(again["judge_cached"], 2)
            self.assertEqual(again["score"], 1.0)
            # A different judge label is a different cache entry.
            other = asyncio.run(judge_predictions(self._rows()[:2], judge, label="other", cache=AnswerJudgeCache(path)))
            self.assertEqual(other["judge_calls"], 2)


class EvalServiceHooksTests(unittest.TestCase):
    def test_judged_rows_decide_failure_and_row_scores(self):
        from app.services.evaluation_service import _prediction_failed, _row_scores_for_pairing

        rows = [
            {"prompt": "Q1", "reference": LONG, "prediction": "paraphrase", "row_exact_match": 0.0, "row_f1": 0.2, "row_judge_score": 1.0},
            {"prompt": "Q2", "reference": LONG, "prediction": "half", "row_exact_match": 0.0, "row_f1": 0.5, "row_judge_score": 0.5},
            {"prompt": "Q3", "reference": LONG, "prediction": "?", "row_exact_match": 0.0, "row_f1": 0.1},
        ]
        self.assertFalse(_prediction_failed(rows[0]), "a correct paraphrase is not a failure once judged")
        self.assertTrue(_prediction_failed(rows[1]))
        self.assertTrue(_prediction_failed(rows[2]), "unjudged rows fall back to exact match")
        scores = _row_scores_for_pairing(rows)
        self.assertEqual(scores["correct"], [1, 0, 0])
        self.assertEqual(scores["f1"], [0.2, 0.5, 0.1])
        self.assertEqual(scores["judge_correct"], [1.0, 0.5, None])

    def test_safe_judge_answers_skips_short_answers_and_missing_judge(self):
        from app.services.evaluation_service import _safe_judge_answers

        async def no_judge(db, pid, project):
            return None

        with mock.patch.object(judge_svc, "resolve_answer_judge", no_judge):
            short = asyncio.run(_safe_judge_answers(
                _FakeDb(), project_id=1, task_profile="qa",
                predictions=[{"prompt": "q", "reference": SHORT, "prediction": "x"}], judge=None,
            ))
            self.assertEqual(short, {"skipped": "not_long_answer_task"})
            none = asyncio.run(_safe_judge_answers(
                _FakeDb(), project_id=1, task_profile="qa",
                predictions=[{"prompt": "q", "reference": LONG, "prediction": "x"}], judge=None,
            ))
            self.assertEqual(none, {"skipped": "no_judge_available"})

    def test_safe_judge_answers_uses_an_injected_judge(self):
        from app.services.evaluation_service import _safe_judge_answers

        async def judge(q, r, p):
            return (1.0, "correct", "ok", 0)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(judge_svc.settings, "DATA_DIR", Path(tmp)):
            rows = [{"prompt": "q", "reference": LONG, "prediction": "x"}]
            snap = asyncio.run(_safe_judge_answers(
                _FakeDb(), project_id=1, task_profile="rag_qa", predictions=rows,
                judge=ResolvedJudge(label="fake", judge=judge),
            ))
        self.assertEqual(snap["score"], 1.0)
        self.assertEqual(snap["judge"], "fake")
        self.assertEqual(rows[0]["row_judge_verdict"], "correct")

    def test_lift_pairing_skips_unjudged_rows(self):
        from app.models.experiment import EvalResult
        from app.services.sft_lift_summary_service import _paired_row_scores

        base = EvalResult(details={"row_scores": {"keys": ["a", "b", "c"], "judge_correct": [0.0, None, 0.5]}})
        fine = EvalResult(details={"row_scores": {"keys": ["a", "b", "c"], "judge_correct": [1.0, 1.0, None]}})
        self.assertEqual(_paired_row_scores(base, fine, "judge_correct"), ([0.0], [1.0]))
        self.assertIsNone(_paired_row_scores(base, fine, "f1"))


class ResolutionTests(unittest.TestCase):
    def test_env_none_and_project_opt_out_disable_the_judge(self):
        class P:
            runtime_config = {"eval_judge": {"enabled": False}}

        with mock.patch.dict(os.environ, {"EVAL_JUDGE_BACKEND": "none"}):
            self.assertIsNone(asyncio.run(judge_svc.resolve_answer_judge(_FakeDb(), 1, None)))
        with mock.patch.dict(os.environ, {"EVAL_JUDGE_BACKEND": ""}):
            self.assertIsNone(asyncio.run(judge_svc.resolve_answer_judge(_FakeDb(), 1, P())))

    def test_requested_local_backend_is_used(self):
        class FakeBackend:
            name = "fakeb"

            @classmethod
            def is_available(cls):
                return True

            def describe(self):
                return "fakeb:model-x"

            async def complete(self, prompt, *, system_prompt=None, max_tokens=0, temperature=0.0, response_schema=None):
                assert "REFERENCE ANSWER" in prompt
                return '{"verdict": "partial", "reason": "half"}'

        with mock.patch("app.services.synth_backends.BACKEND_REGISTRY", [FakeBackend]), \
                mock.patch.dict(os.environ, {"EVAL_JUDGE_BACKEND": ""}):
            resolved = asyncio.run(judge_svc.resolve_answer_judge(_FakeDb(), 1, None))
        self.assertEqual(resolved.label, "fakeb:model-x")
        self.assertEqual(asyncio.run(resolved.judge("q", LONG, "x")), (0.5, "partial", "half", 0))


class _FakeDb:
    async def get(self, *_args, **_kwargs):
        return None


if __name__ == "__main__":
    unittest.main()

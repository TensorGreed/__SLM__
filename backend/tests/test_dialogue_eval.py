"""Multi-turn rows at eval time: last-turn history in the prompt + the per-turn replay judge.

Neutral toy data; the model and the judge are stubbed (no model eval, no network).
pytest-only. Each eval scenario builds its own throwaway engine inside ONE ``asyncio.run`` (never the
app's engine, never IsolatedAsyncioTestCase).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"

import app.models  # noqa: F401
from app.database import Base
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project
from app.services import dialogue_eval_service as des
from app.services import evaluation_service as ev
from app.services.answer_judge_service import ResolvedJudge, VERDICT_SCORES

SYSTEM = {"role": "system", "content": "You are terse."}
CHAT_ROW = {
    "messages": [
        SYSTEM,
        {"role": "user", "content": "What colour is the sky?"},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "And the grass?"},
        {"role": "assistant", "content": "Green."},
        {"role": "user", "content": "And snow?"},
        {"role": "assistant", "content": "White."},
    ]
}
SINGLE_ROW = {"question": "What is 2+2?", "answer": "4"}


class FakeTokenizer:
    chat_template = "x"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}</s>" for m in messages) + "<assistant>"


class HistoryTemplateTests(unittest.TestCase):
    def test_history_replaces_the_single_user_message(self):
        hist = [SYSTEM, {"role": "user", "content": "hi"}]
        text, applied = ev._apply_chat_template_if_present(FakeTokenizer(), "hi", hist)
        self.assertTrue(applied)
        self.assertEqual(text, "<system>You are terse.</s><user>hi</s><assistant>")

    def test_no_history_is_the_old_single_user_prompt(self):
        text, applied = ev._apply_chat_template_if_present(FakeTokenizer(), "hi")
        self.assertEqual(text, "<user>hi</s><assistant>")
        self.assertTrue(applied)

    def test_no_template_falls_back_to_role_lines(self):
        class Bare:
            chat_template = None

        text, applied = ev._apply_chat_template_if_present(Bare(), "hi", [SYSTEM, {"role": "user", "content": "hi"}])
        self.assertFalse(applied)
        self.assertEqual(text, "system: You are terse.\nuser: hi\nassistant: ")
        self.assertEqual(ev._apply_chat_template_if_present(Bare(), "hi"), ("hi", False))

    def test_a_template_that_raises_falls_back_to_role_lines_for_a_history(self):
        class Broken:
            chat_template = "x"

            def apply_chat_template(self, *a, **k):
                raise ValueError("bad template")

        hist = [SYSTEM, {"role": "user", "content": "hi"}]
        self.assertEqual(
            ev._apply_chat_template_if_present(Broken(), "hi", hist),
            ("system: You are terse.\nuser: hi\nassistant: ", False),
        )
        # a bare prompt keeps its old behaviour
        self.assertEqual(ev._apply_chat_template_if_present(Broken(), "hi"), ("hi", False))


class PairsTests(unittest.TestCase):
    def _pairs(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            return ev._load_heldout_pairs(path, 10)

    def test_chat_row_is_scored_on_its_last_turn_with_history(self):
        pair = self._pairs([CHAT_ROW])[0]
        self.assertEqual(pair["reference"], "White.")
        self.assertEqual(len(pair["history"]), 6)  # system + 2 exchanges + the last user turn
        self.assertEqual(pair["history"][0], SYSTEM)
        self.assertEqual(pair["history"][-1]["content"], "And snow?")
        self.assertEqual(len(pair["dialogue_messages"]), 7)

    def test_system_plus_one_exchange_keeps_the_system_prompt(self):
        pair = self._pairs([{"messages": [SYSTEM, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}])[0]
        self.assertEqual([m["role"] for m in pair["history"]], ["system", "user"])

    def test_single_turn_rows_are_unchanged(self):
        for row in (SINGLE_ROW, {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}):
            pair = self._pairs([row])[0]
            self.assertNotIn("history", pair)
        pair = self._pairs([SINGLE_ROW])[0]
        self.assertEqual((pair["prompt"], pair["reference"]), ("What is 2+2?", "4"))
        self.assertNotIn("dialogue_messages", pair)

    def test_history_is_rewritten_only_when_the_handler_changed_the_prompt(self):
        hist = [SYSTEM, {"role": "user", "content": "  Why?  "}]
        # the extracted prompt is the stripped last user turn: not a rewrite
        self.assertEqual(des.history_for_pair(hist, "Why?")[-1]["content"], "  Why?  ")
        # a handler that added context rewrote it
        self.assertEqual(des.history_for_pair(hist, "Context: x\n\nWhy?")[-1]["content"], "Context: x\n\nWhy?")
        # a trailing non-user message is left alone; the last USER turn is the one replaced
        tool_tail = [{"role": "user", "content": "q"}, {"role": "tool", "content": "t"}]
        self.assertEqual(des.history_for_pair(tool_tail, "rewritten")[-1]["content"], "t")
        self.assertEqual(des.history_for_pair(tool_tail, "rewritten")[0]["content"], "rewritten")

    def test_replay_pairs_cover_every_non_final_assistant_turn(self):
        pairs = self._pairs([CHAT_ROW, SINGLE_ROW])
        replay = des.build_replay_pairs(pairs)
        self.assertEqual([(r["_replay"]["pair_index"], r["_replay"]["turn"], r["reference"]) for r in replay],
                         [(0, 1, "Blue."), (0, 2, "Green.")])
        self.assertEqual([m["role"] for m in replay[0]["history"]], ["system", "user"])
        self.assertEqual(replay[1]["history"][-1]["content"], "And the grass?")


class _Judge:
    """Scores a prediction 'correct' when it equals the reference, else 'wrong'."""

    def __init__(self):
        self.calls = []

    async def __call__(self, question, reference, prediction):
        self.calls.append((question, reference, prediction))
        verdict = "correct" if prediction.strip() == reference.strip() else "wrong"
        return VERDICT_SCORES[verdict], verdict, "", 0


class Outcome:
    def __init__(self, result, seen_pairs, project_id, rows):
        self.result, self.seen_pairs, self.project_id, self.rows = result, seen_pairs, project_id, rows


def _fake_inference_factory(seen):
    def fake(model_ref, pairs, *a, **kw):
        seen.extend(dict(p) for p in pairs)
        # The model gets every turn right except the final turn of the chat row.
        preds = []
        for p in pairs:
            wrong = p["reference"] == "White."
            preds.append({"prompt": p["prompt"], "reference": p["reference"],
                          "prediction": "Pink." if wrong else p["reference"], "latency_ms": 1.0})
        return preds, {"engine": "stub"}

    return fake


def run_eval(rows, *, dataset_type=DatasetType.TEST, **kw) -> Outcome:
    """One whole eval scenario (throwaway engine + seed + run + side-file read) in a single event loop."""

    async def scenario() -> Outcome:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'g8.db'}", future=True)
            sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                heldout = root / "test.jsonl"
                heldout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
                async with sf() as db:
                    # a unique id keeps the on-disk judge cache / side files of other scenarios out of this one
                    project = Project(id=10_000 + uuid4().int % 10**9, name=f"g8-{uuid4().hex[:8]}",
                                      description="x", base_model_name="toy/model")
                    db.add(project)
                    await db.flush()
                    exp = Experiment(project_id=project.id, name="e", description="", status=ExperimentStatus.COMPLETED,
                                     base_model="toy/model", output_dir=str(root))
                    db.add(exp)
                    db.add(Dataset(project_id=project.id, name="t", dataset_type=dataset_type,
                                   file_path=str(heldout), record_count=len(rows)))
                    await db.commit()
                    pid, eid = project.id, exp.id
                seen: list = []
                with patch("app.services.evaluation_service._run_local_inference", _fake_inference_factory(seen)), \
                        patch("app.services.evaluation_service._resolve_model_reference", return_value="toy"):
                    async with sf() as db:
                        result = await ev.run_heldout_evaluation(
                            db=db, project_id=pid, experiment_id=eid, dataset_name=dataset_type.value,
                            eval_type="exact_match", max_samples=100, **kw)
                        await db.commit()
                stored = ev.read_row_predictions_file(pid, (result.details or {}).get("row_predictions_file"))
                return Outcome(result, seen, pid, stored)
            finally:
                await engine.dispose()

    return asyncio.run(scenario())


class DialogueJudgeEndToEnd(unittest.TestCase):
    def test_off_by_default_and_history_reaches_inference(self):
        out = run_eval([CHAT_ROW, SINGLE_ROW])
        self.assertEqual(len(out.seen_pairs), 2)
        self.assertEqual(len(out.seen_pairs[0]["history"]), 6)
        self.assertNotIn("history", out.seen_pairs[1])
        self.assertNotIn("dialogue_judge", out.result.metrics)
        # No judge scored anything: nothing is stored, and no row array rides in the result.
        self.assertNotIn("row_predictions", out.result.details)
        self.assertNotIn("row_predictions_file", out.result.details)
        self.assertIsNone(out.rows)

    def test_per_turn_replay_judge(self):
        judge = _Judge()
        out = run_eval([CHAT_ROW, SINGLE_ROW], dialogue_judge=True, answer_judge=ResolvedJudge(label="stub", judge=judge))
        result = out.result
        # 2 held-out rows + 2 replayed turns (turns 1 and 2 of the chat row)
        self.assertEqual(len(out.seen_pairs), 4)
        self.assertEqual(sum(1 for p in out.seen_pairs if p.get("_replay")), 2)
        summary = result.metrics["dialogue_judge"]
        self.assertEqual(summary["rows"], 1)
        self.assertAlmostEqual(summary["mean"], round(2 / 3, 4))
        self.assertEqual(summary["worst_turn_mean"], 0.0)
        self.assertEqual(summary["by_turn_index"], [{"turn": 1, "mean": 1.0, "n": 1}, {"turn": 2, "mean": 1.0, "n": 1},
                                                    {"turn": 3, "mean": 0.0, "n": 1}])
        self.assertEqual(summary["replay_generations"], 2)
        self.assertEqual((summary["rows_not_replayed"], summary["replay_cap_hit"]), (0, False))
        # the replayed turns never leak into the scored predictions
        self.assertEqual(result.metrics["evaluated_samples"], 2)
        # the judge saw the gold history as its question
        self.assertTrue(judge.calls[0][0].startswith("system: You are terse.\nuser: What colour"))

    def test_judged_rows_go_to_a_side_file_not_the_result(self):
        out = run_eval([CHAT_ROW, SINGLE_ROW], dialogue_judge=True,
                       answer_judge=ResolvedJudge(label="stub", judge=_Judge()))
        details = out.result.details
        self.assertNotIn("row_predictions", details)
        pointer = details["row_predictions_file"]
        self.assertEqual(pointer["path"], f"experiments/{out.result.experiment_id}/evals/result_{out.result.id}_rows.jsonl")
        self.assertEqual((pointer["count"], pointer["truncated"]), (1, 0))
        # judged rows only: the single-turn row (no judge scored it) is not stored
        self.assertEqual([r["row_index"] for r in out.rows], [0])
        row = out.rows[0]
        self.assertEqual(row["prediction"], "Pink.")
        self.assertEqual(row["history"][0]["role"], "system")
        self.assertNotIn("prompt", row)  # the history carries it
        self.assertEqual(row["dialogue_judge"]["worst_turn"], 3)
        self.assertEqual([t["verdict"] for t in row["dialogue_judge"]["turns"]], ["correct", "correct", "wrong"])
        self.assertTrue((ev._project_dir_for_evals(out.project_id) / pointer["path"]).is_file())

    def test_history_is_capped_at_the_last_forty_messages(self):
        messages = [SYSTEM]
        for n in range(24):
            messages += [{"role": "user", "content": f"q{n}"}, {"role": "assistant", "content": f"a{n}"}]
        out = run_eval([{"messages": messages}], dialogue_judge=True,
                       answer_judge=ResolvedJudge(label="stub", judge=_Judge()))
        row = out.rows[0]
        self.assertEqual(len(row["history"]), ev.ROW_HISTORY_MESSAGE_CAP)
        self.assertTrue(row["history_truncated"])
        self.assertEqual(row["history"][-1]["content"], "q23")

    def test_the_replay_cap_leaves_later_rows_out_and_says_so(self):
        # Each chat row needs 2 replayed generations; a cap of 3 fits one row.
        with patch.object(des, "MAX_REPLAY_GENERATIONS", 3):
            out = run_eval([CHAT_ROW, CHAT_ROW], dialogue_judge=True,
                           answer_judge=ResolvedJudge(label="stub", judge=_Judge()))
        summary = out.result.metrics["dialogue_judge"]
        self.assertEqual(summary["replay_generations"], 2)
        self.assertEqual(summary["rows"], 1)
        self.assertEqual(summary["rows_not_replayed"], 1)
        self.assertTrue(summary["replay_cap_hit"])
        # the un-replayed row was not judged on its final turn alone
        self.assertEqual([r["row_index"] for r in out.rows], [0])
        self.assertEqual(summary["turns_judged"], 3)

    def test_no_judge_available_costs_no_generation(self):
        async def none(*a, **k):
            return None

        with patch("app.services.answer_judge_service.resolve_answer_judge", none):
            out = run_eval([CHAT_ROW, SINGLE_ROW], dialogue_judge=True)
        self.assertEqual(len(out.seen_pairs), 2)
        self.assertEqual(out.result.metrics["dialogue_judge"], {"skipped": "no_judge_available"})

    def test_no_multi_turn_rows_is_reported(self):
        out = run_eval([SINGLE_ROW], dialogue_judge=True, answer_judge=ResolvedJudge(label="stub", judge=_Judge()))
        self.assertEqual(out.result.metrics["dialogue_judge"], {"skipped": "no_multi_turn_rows"})


class ChatSftAnswerJudgeTests(unittest.TestCase):
    LONG_REF = "Grass is green because chlorophyll absorbs red and blue light and reflects the green part of it."

    def test_chat_sft_is_a_judge_profile_with_the_long_answer_rule(self):
        from app.services.answer_judge_service import should_judge

        self.assertTrue(should_judge("chat_sft", [self.LONG_REF] * 3))
        self.assertFalse(should_judge("chat_sft", ["Green."] * 3))

    def test_a_chat_row_is_judged_on_the_whole_dialogue_not_its_last_user_turn(self):
        judge = _Judge()
        history = [SYSTEM, {"role": "user", "content": "Why is grass green?"}]
        predictions = [
            {"prompt": "Why is grass green?", "history": history, "reference": self.LONG_REF,
             "prediction": self.LONG_REF},
            {"prompt": "plain question", "reference": self.LONG_REF, "prediction": "nope"},
        ]
        snapshot = asyncio.run(ev._safe_judge_answers(
            None, project_id=987654, task_profile="chat_sft", predictions=predictions,
            judge=ResolvedJudge(label="stub", judge=judge),
        ))
        self.assertEqual(snapshot["judged"], 2)
        self.assertEqual(judge.calls[0][0], des.history_text(history))
        self.assertEqual(judge.calls[1][0], "plain question")
        self.assertEqual([p["row_judge_verdict"] for p in predictions], ["correct", "wrong"])


class ReplayStatsTests(unittest.TestCase):
    """Replayed dialogue-judge generations stay out of latency / throughput / template counts."""

    def _run_llama(self, pairs, **kw):
        import sys
        import types

        calls = []

        class FakeLlama:
            chat_handler = object()

            def __init__(self, **k):
                pass

            def create_chat_completion(self, **k):
                calls.append(k["messages"])
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 3}}

        fake = types.ModuleType("llama_cpp")
        fake.Llama = FakeLlama
        with patch.dict(sys.modules, {"llama_cpp": fake}):
            preds, runtime = ev._run_llama_cpp_inference("m.gguf", pairs, 8, 0.0, None, **kw)
        return preds, runtime, calls

    def _pairs(self):
        hist = [SYSTEM, {"role": "user", "content": "hi"}]
        return [
            {"prompt": "hi", "reference": "r", "history": hist},
            {"prompt": "hi", "reference": "r", "history": hist, "_replay": {"pair_index": 0, "turn": 1, "turns": 2}},
        ]

    def test_replay_generations_are_counted_apart(self):
        _preds, runtime, _calls = self._run_llama(self._pairs())
        self.assertEqual(runtime["total_generated_tokens"], 3)
        self.assertEqual(runtime["chat_template_applied_count"], 1)
        self.assertEqual(runtime["replay_generations"], 1)
        self.assertEqual(runtime["replay_generated_tokens"], 3)
        self.assertIn("replay_generation_seconds", runtime)

    def test_no_replay_means_no_replay_keys(self):
        _p, runtime, _c = self._run_llama(self._pairs()[:1])
        self.assertNotIn("replay_generations", runtime)

    def test_llama_cpp_ignores_history_for_a_handler_that_wraps_its_own_prompt(self):
        _p, _r, calls = self._run_llama(self._pairs()[:1], apply_chat_template=False)
        self.assertEqual(calls[0], [{"role": "user", "content": "hi"}])
        _p, _r, calls = self._run_llama(self._pairs()[:1])
        self.assertEqual(calls[0][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()

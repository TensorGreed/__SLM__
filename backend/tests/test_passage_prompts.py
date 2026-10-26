"""Train with retrieved passages (passage_prompts_service) and its lift check
(passage_reader_eval_service).

Pins (fakes only — no retrieval model, no LLM, no GPU):
  * a row becomes the SERVED prompt: ``system_prompt`` is exactly
    ``auto_rag_service.document_preamble`` of the shown passages; the answer
    passage found by retrieval is cited ``[n]``; a missed one is put back in
    place of the lowest-ranked hit (``inserted``); a withheld row loses every
    passage holding the answer and its target becomes the fixed "don't say";
  * the mix is deterministic per seed and the withheld share follows the
    configured fraction; the files land under the run's output dir and the
    prepared split is untouched;
  * train.py renders ``system_prompt`` as the chat template's system message;
  * the reader check pairs base + passages with fine-tuned + passages row by
    row, reuses a matching cached base comparison, re-runs a stale one, and
    the summary / verdict say which arms they compare.
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.config import settings
from app.services import passage_prompts_service as pps
from app.services import passage_reader_eval_service as reader
from app.services.auto_rag_service import document_preamble

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _hit(text: str, chunk: int, doc: int = 7) -> dict:
    return {"row_id": f"{doc}:{chunk}", "score": 1.0, "payload": {
        "id": f"{doc}:{chunk}", "text": text, "source_doc": "guide.pdf", "source_document_id": doc, "chunk_id": chunk,
    }}


GOLD_TEXT = "The standard deduction for a single filer is 15,750 dollars for the tax year."
ANSWER = "The standard deduction for a single filer is 15,750 dollars."
DISTRACTORS = [_hit(f"Unrelated paragraph {i} about mileage rates and records kept for audits.", 100 + i) for i in range(8)]
ROW = {"id": 1, "question": "What is the standard deduction for a single filer?", "answer": ANSWER,
       "source_document_id": 7, "chunk_id": 3, "text": "<rendered old prompt>"}


class PassagePromptRowTests(unittest.TestCase):
    def _row(self, hits, *, withhold=False, candidates=None, seed=1):
        import random

        return pps.passage_prompt_row(
            ROW, hits, k=3, withhold=withhold, gold_candidates=candidates or [], rng=random.Random(seed)
        )

    def test_retrieved_answer_passage_is_cited_and_the_prompt_is_the_served_one(self):
        hits = [DISTRACTORS[0], _hit(GOLD_TEXT, 3), DISTRACTORS[1], DISTRACTORS[2]]
        row, outcome = self._row(hits)
        self.assertEqual(outcome, "retrieved")
        self.assertEqual(row["system_prompt"], document_preamble(hits[:3]))
        self.assertEqual(row["answer"], ANSWER + " [2]")
        self.assertEqual(row["passage_prompt"], {"gold": "retrieved", "cited": 2, "passages": 3})
        # The old rendered composite (and its bare prompt) never trains.
        self.assertNotIn("text", row)

    def test_a_missed_answer_passage_is_put_back_in_place_of_the_last_hit(self):
        row, outcome = self._row(DISTRACTORS[:4], candidates=[DISTRACTORS[5], _hit(GOLD_TEXT, 3)])
        self.assertEqual(outcome, "inserted")
        self.assertEqual(row["passage_prompt"]["passages"], 3)
        self.assertIn(GOLD_TEXT, row["system_prompt"])
        self.assertNotIn(DISTRACTORS[2]["payload"]["text"], row["system_prompt"])
        self.assertTrue(row["answer"].startswith(ANSWER + " ["))

    def test_withheld_rows_lose_every_answer_passage_and_say_they_dont_know(self):
        hits = [_hit(GOLD_TEXT, 3), DISTRACTORS[0], _hit(GOLD_TEXT + " Again.", 4), DISTRACTORS[1], DISTRACTORS[2]]
        row, outcome = self._row(hits, withhold=True)
        self.assertEqual(outcome, "withheld")
        self.assertEqual(row["answer"], pps.NOT_IN_PASSAGES_ANSWER)
        self.assertNotIn("15,750", row["system_prompt"])
        self.assertEqual(row["passage_prompt"]["passages"], 3)
        self.assertIsNone(row["passage_prompt"]["cited"])

    def test_no_passage_holds_the_answer_keeps_the_answer_uncited(self):
        row, outcome = self._row(DISTRACTORS[:3], candidates=[DISTRACTORS[4]])
        self.assertEqual(outcome, "not_found")
        self.assertEqual(row["answer"], ANSWER)

    def test_config_switch_and_fraction_are_clamped(self):
        self.assertFalse(pps.is_enabled({}))
        self.assertTrue(pps.is_enabled({"passage_prompts": True}))
        self.assertEqual(pps.withheld_fraction({}), pps.DEFAULT_WITHHELD_FRACTION)
        self.assertEqual(pps.withheld_fraction({"passage_prompts_withheld_fraction": 0.9}), pps.MAX_WITHHELD_FRACTION)


class BuildRowsTests(unittest.TestCase):
    def _rows(self, n=200):
        return [{**ROW, "id": i, "question": f"Question {i}: what is the standard deduction for a single filer?"} for i in range(n)]

    def _retrieve(self, question, config):
        # Retrieval finds the answer passage for even rows only; wide k.
        self.seen_k.append(config["k"])
        i = int(question.split()[1].rstrip(":"))
        hits = list(DISTRACTORS[: config["k"]])
        if i % 2 == 0:
            hits[1] = _hit(GOLD_TEXT, 3)
        return hits

    def test_deterministic_mix_follows_the_fraction(self):
        self.seen_k = []
        kwargs = dict(
            retrieval={"k": 3}, retrieve_fn=self._retrieve,
            corpus_passages=[_hit(GOLD_TEXT, 3)["payload"]], source_passages=[], withheld=0.2, seed=5,
        )
        rows, counts = pps.build_passage_prompt_rows(self._rows(), **kwargs)
        again, _ = pps.build_passage_prompt_rows(self._rows(), **kwargs)
        self.assertEqual(rows, again)
        self.assertEqual(set(self.seen_k), {6})  # k + extra hits, so withheld rows still show k
        self.assertEqual(counts["rows_out"], 200)
        self.assertTrue(25 <= counts["withheld"] <= 55, counts)
        self.assertGreater(counts["inserted"], 50)  # odd rows: retrieval missed, put back
        self.assertGreater(counts["retrieved"], 50)
        self.assertEqual(counts["not_found"], 0)


class ApplyTests(unittest.TestCase):
    def test_writes_run_files_and_never_touches_the_prepared_split(self):
        pid = 900000 + int(uuid4().int % 90000)
        root = Path(settings.DATA_DIR) / "projects" / str(pid)
        (root / "cleaned").mkdir(parents=True)
        (root / "cleaned" / "cleaned.jsonl").write_text(
            json.dumps({"text": GOLD_TEXT, "source_doc": "guide.pdf", "source_document_id": 7, "chunk_id": 3}) + "\n",
            encoding="utf-8",
        )
        prepared = root / "prepared"
        prepared.mkdir()
        for split in ("train", "val", "test"):
            (prepared / f"{split}.jsonl").write_text(json.dumps({**ROW, "id": split}) + "\n", encoding="utf-8")
        test_before = (prepared / "test.jsonl").read_bytes()
        out = root / "experiments" / "5"
        block = pps.apply_passage_prompts(
            pid, retrieval={"k": 2}, train_file=prepared / "train.jsonl", val_file=prepared / "val.jsonl",
            output_dir=out, withheld=0.0, retrieve_fn=lambda q, c: DISTRACTORS[: c["k"]],
        )
        self.assertTrue(block["applied"])
        self.assertEqual(block["counts"]["train"]["inserted"], 1)
        self.assertEqual(block["gold_inserted_share"], 1.0)
        train = [json.loads(l) for l in Path(block["train_file"]).read_text(encoding="utf-8").splitlines()]
        self.assertIn(GOLD_TEXT, train[0]["system_prompt"])
        self.assertTrue(Path(block["val_file"]).exists())
        self.assertEqual((prepared / "test.jsonl").read_bytes(), test_before)
        self.assertTrue((out / "passage_prompts" / "report.json").exists())

    def test_refuses_a_project_without_document_passages(self):
        with self.assertRaises(ValueError):
            pps.apply_passage_prompts(
                987654321, retrieval={"k": 2}, train_file=Path("/nonexistent"), val_file=None,
                output_dir=Path("/tmp"), retrieve_fn=lambda q, c: [],
            )


class TrainScriptRewrapTests(unittest.TestCase):
    def test_system_prompt_becomes_the_chat_templates_system_message(self):
        from scripts.train import _rewrap_prompt_completion_row

        class Tok:
            chat_template = "x"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"

        row = {"source_text": "Q?", "target_text": "A [1]", "system_prompt": "Passages: [1] x"}
        out = _rewrap_prompt_completion_row(row, Tok())
        self.assertEqual(out["source_text"], "<system>Passages: [1] x</system><user>Q?</user><assistant>")
        self.assertEqual(out["text"], out["source_text"] + "A [1]")
        plain = _rewrap_prompt_completion_row({"source_text": "Q?", "target_text": "A", "system_prompt": ""}, Tok())
        self.assertEqual(plain["source_text"], "<user>Q?</user><assistant>")


def _comparison(model, *, rows, on_scores, off_scores, judge="ollama:j", retrieval=None, base_model="B", split="test",
                decoding="neutral"):
    from app.services.decoding_settings import decoding_record, generation_kwargs, model_decoding_record

    out_rows = []
    for (q, ref), on, off in zip(rows, on_scores, off_scores):
        out_rows.append({
            "question": q, "reference": ref,
            "without_rag": {"generated": "x", "judge": {"score": off, "verdict": "correct" if off == 1 else "wrong"}},
            "with_rag": {"generated": "y", "judge": {"score": on, "verdict": "correct" if on == 1 else "wrong"},
                         "retrieval_judge": {"verdict": "found", "passage": 1}},
        })
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return {
        "model": model, "corpus": "documents", "split": split, "base_model": base_model,
        "cached_at": "2026-10-19T00:00:00+00:00",
        "summary": {
            "retrieval": retrieval or {"k": 5, "reranker": None, "pool": None, "chunk_size": None},
            "decoding": (
                model_decoding_record({"repetition_penalty": 1.1}) if decoding == "model"
                else decoding_record(generation_kwargs(max_new_tokens=1, temperature=0.0))
            ),
            "judge": {
                "judge": judge,
                "with_rag": {"score": mean(on_scores), "counts": {"correct": on_scores.count(1)}, "judged": len(rows),
                             "retrieval": {"found": len(rows), "wrong_missed": 0, "wrong_misread": on_scores.count(0)}},
                "without_rag": {"score": mean(off_scores), "counts": {"correct": off_scores.count(1)}},
            },
        },
        "rows": out_rows,
    }


ROWS = [(f"q{i}", f"r{i}") for i in range(12)]


class ReaderRecordTests(unittest.TestCase):
    def test_pairs_the_two_with_passages_arms_row_by_row(self):
        base = _comparison("base", rows=ROWS, on_scores=[1] * 6 + [0] * 6, off_scores=[0] * 12)
        ft = _comparison("fine_tuned", rows=list(reversed(ROWS)), on_scores=[1] * 12, off_scores=[0] * 12)
        record = reader.build_reader_record(base, ft, experiment_id=9)
        self.assertTrue(record["comparable"])
        self.assertEqual(record["n_rows"], 12)
        self.assertEqual(record["evidence"]["better"], 6)
        self.assertEqual(record["evidence"]["worse"], 0)
        self.assertEqual(record["evidence"]["verdict"], "better")
        self.assertEqual(record["arms"]["base_with_passages"]["score"], 0.5)
        self.assertEqual(record["arms"]["finetune_with_passages"]["score"], 1.0)

    def test_different_judge_or_retrieval_is_not_comparable(self):
        base = _comparison("base", rows=ROWS, on_scores=[1] * 12, off_scores=[0] * 12, judge="ollama:other")
        ft = _comparison("fine_tuned", rows=ROWS, on_scores=[1] * 12, off_scores=[0] * 12)
        self.assertFalse(reader.build_reader_record(base, ft, experiment_id=9)["comparable"])
        base = _comparison("base", rows=ROWS, on_scores=[1] * 12, off_scores=[0] * 12, retrieval={"k": 3})
        self.assertFalse(reader.build_reader_record(base, ft, experiment_id=9)["comparable"])

    def test_cached_base_comparison_is_the_bar_only_when_it_matches(self):
        r = {"k": 5}
        base = _comparison("base", rows=ROWS, on_scores=[1] * 12, off_scores=[0] * 12, retrieval=r)
        self.assertTrue(reader.base_comparison_usable(base, base_model="B", retrieval=r, test_rows=12))
        self.assertFalse(reader.base_comparison_usable(base, base_model="other", retrieval=r))
        self.assertFalse(reader.base_comparison_usable(base, base_model="B", retrieval={"k": 3}))
        self.assertFalse(reader.base_comparison_usable(base, base_model="B", retrieval=r, test_rows=13))
        val = _comparison("base", rows=ROWS, on_scores=[1] * 12, off_scores=[0] * 12, retrieval=r, split="val")
        self.assertFalse(reader.base_comparison_usable(val, base_model="B", retrieval=r))
        # Decoded another way, or before decoding was recorded: not this bar.
        self.assertFalse(reader.base_comparison_usable(base, base_model="B", retrieval=r, decoding="model"))
        own = _comparison("base", rows=ROWS, on_scores=[1] * 12, off_scores=[0] * 12, retrieval=r, decoding="model")
        self.assertTrue(reader.base_comparison_usable(own, base_model="B", retrieval=r, decoding="model"))
        self.assertFalse(reader.base_comparison_usable(own, base_model="B", retrieval=r))
        del base["summary"]["decoding"]
        self.assertFalse(reader.base_comparison_usable(base, base_model="B", retrieval=r))


class ReaderEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.pid = 800000 + int(uuid4().int % 90000)
        root = Path(settings.DATA_DIR) / "projects" / str(self.pid)
        (root / "prepared").mkdir(parents=True)
        (root / "prepared" / "test.jsonl").write_text("".join(json.dumps({"q": i}) + "\n" for i in range(12)), encoding="utf-8")
        self.retrieval = {"k": 5, "reranker": None, "pool": None, "chunk_size": None}
        self.project = SimpleNamespace(id=self.pid, base_model_name="B", runtime_config={"auto_rag_retrieval": self.retrieval})
        self.exp = SimpleNamespace(id=41, project_id=self.pid, base_model="B", name="reads", config={"passage_prompts": True})
        self.calls: list[dict] = []
        # The base model's own decoding settings when they differ (None:
        # neutral only); its base + passages score under them.
        self.own_settings: dict | None = None
        self.own_on_scores = [1] * 6 + [0] * 6
        patcher = mock.patch("app.services.decoding_settings.model_decoding_differs", lambda ref: self.own_settings)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fn(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("base_only"):
            own = kwargs.get("decoding") == "model"
            payload = _comparison(
                "base", rows=ROWS, on_scores=self.own_on_scores if own else [1] * 6 + [0] * 6,
                off_scores=[0] * 12, retrieval=self.retrieval, decoding="model" if own else "neutral",
            )
            name = "comparison_base_documents_model_decoding.json" if own else "comparison_base_documents.json"
            path = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "auto_rag" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        return _comparison("fine_tuned", rows=ROWS, on_scores=[1] * 10 + [0] * 2, off_scores=[0] * 12, retrieval=self.retrieval)

    def _run(self):
        async def _report(fraction, message):
            pass

        return asyncio.run(reader.run_passage_reader_evaluation(
            None, project=self.project, experiment=self.exp, report=_report, comparison_fn=self._fn,
        ))

    def test_runs_the_bar_once_then_reuses_it_and_writes_the_record(self):
        result = self._run()
        self.assertEqual([c.get("base_only") for c in self.calls], [True, False])
        self.assertEqual(self.calls[1]["experiment_id"], 41)
        self.assertEqual(self.calls[1]["split"], "test")
        self.assertEqual(result["kind"], "passage_reader_eval")
        self.assertFalse(result["base_reused"])
        self.assertEqual(result["evidence"]["better"], 4)
        self.calls.clear()
        again = self._run()
        self.assertEqual([c.get("base_only") for c in self.calls], [False])
        self.assertTrue(again["base_reused"])

        summary = reader.reader_summary(self.pid, self.exp)
        self.assertEqual(summary["kind"], "passage_reader")
        self.assertEqual(summary["headline"]["baseline_value"], 0.5)
        self.assertAlmostEqual(summary["headline"]["trained_value"], 10 / 12, places=3)
        self.assertEqual(summary["verdict"], "better")
        self.assertEqual(summary["failed_count"], 2)
        self.assertEqual(summary["failures"][0]["row_retrieval_verdict"], "found")

    def test_base_decoded_its_own_way_is_the_bar_when_it_scores_higher(self):
        """Strongest-baseline rule: the base is also scored with its own
        settings (repetition penalty 1.1); 9/12 beats neutral's 6/12, so the
        fine-tune (10/12) is measured from 0.75, not 0.5. Both are reused."""
        self.own_settings = {"num_beams": 1, "repetition_penalty": 1.1, "no_repeat_ngram_size": 0}
        self.own_on_scores = [1] * 9 + [0] * 3
        result = self._run()
        self.assertEqual(
            [(c.get("base_only"), c.get("decoding", "neutral")) for c in self.calls],
            [(True, "neutral"), (True, "model"), (False, "neutral")],
        )
        bar = result["base_decoding"]
        self.assertEqual((bar["chosen"], bar["neutral"], bar["model"]), ("model", 0.5, 0.75))
        self.assertEqual(result["base_with_passages"]["score"], 0.75)
        summary = reader.reader_summary(self.pid, self.exp)
        self.assertEqual(summary["headline"]["baseline_value"], 0.75)
        self.assertEqual(summary["baseline"]["decoding_bar"]["chosen"], "model")
        self.calls.clear()
        again = self._run()
        self.assertEqual([c.get("base_only") for c in self.calls], [False])
        self.assertTrue(again["base_reused"])

    def test_own_decoding_comparison_judged_by_another_judge_is_re_run(self):
        self.own_settings = {"num_beams": 1, "repetition_penalty": 1.1, "no_repeat_ngram_size": 0}
        self._run()
        path = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "auto_rag" / "comparison_base_documents_model_decoding.json"
        stale = json.loads(path.read_text(encoding="utf-8"))
        stale["summary"]["judge"]["judge"] = "ollama:old"
        path.write_text(json.dumps(stale), encoding="utf-8")
        self.calls.clear()
        self._run()
        self.assertEqual([(c.get("base_only"), c.get("decoding", "neutral")) for c in self.calls],
                         [(True, "model"), (False, "neutral")])

    def test_a_neutral_bar_from_an_earlier_judge_is_re_run_beside_a_fresh_own_decoding_run(self):
        self.own_settings = {"num_beams": 1, "repetition_penalty": 1.1, "no_repeat_ngram_size": 0}
        auto = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "auto_rag"
        auto.mkdir(parents=True, exist_ok=True)
        old = _comparison("base", rows=ROWS, on_scores=[1] * 6 + [0] * 6, off_scores=[0] * 12,
                          retrieval=self.retrieval, judge="ollama:old")
        (auto / "comparison_base_documents.json").write_text(json.dumps(old), encoding="utf-8")
        result = self._run()
        self.assertEqual([(c.get("base_only"), c.get("decoding", "neutral")) for c in self.calls],
                         [(True, "model"), (True, "neutral"), (False, "neutral")])
        self.assertFalse(result["base_reused"])
        self.assertEqual(result["judge"], "ollama:j")

    def test_neutral_stays_the_bar_when_its_own_decoding_scores_lower(self):
        self.own_settings = {"num_beams": 1, "repetition_penalty": 1.1, "no_repeat_ngram_size": 0}
        self.own_on_scores = [1] * 2 + [0] * 10
        result = self._run()
        self.assertEqual(result["base_decoding"]["chosen"], "neutral")
        self.assertEqual(result["base_with_passages"]["score"], 0.5)

    def test_a_changed_project_base_model_refuses_rather_than_scoring_the_wrong_bar(self):
        self.project.base_model_name = "Other"
        with self.assertRaises(ValueError):
            self._run()

    def test_summary_before_the_check_says_so(self):
        self.assertEqual(reader.reader_summary(self.pid, self.exp)["verdict"], "not_evaluated")

    def test_verdict_compares_the_two_with_passages_arms(self):
        from app.services.passage_rag_verdict_service import _judged_reader_run, compare_passages_to_finetune

        self._run()
        finetune = _judged_reader_run(self.exp)
        self.assertTrue(finetune["with_passages"])
        passages = {"judge": "ollama:j", "with_passages": 0.9, "retrieval_evidence": {"verdict": "better"}, "split": "test"}
        verdict = compare_passages_to_finetune(passages, finetune)
        # The base arm of the reader check (0.5) replaces the gate's cached score.
        self.assertEqual(verdict["passages"]["with_passages"], 0.5)
        self.assertEqual(verdict["arms"], {"passages": "base_with_passages", "finetune": "finetune_with_passages"})
        self.assertFalse(verdict["passages_win"])
        self.assertEqual(verdict["reason"], "finetune_ahead")


if __name__ == "__main__":
    unittest.main()

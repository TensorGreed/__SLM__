"""Opt-in early cutoff for a prompt format the model never answers in (gap G7: ``stop_if_unanswered_after``).

Pins:
  * equivalence: a run where some of the first N answers parse is stored byte-identical (metrics, pass rate,
    details; ``json.dumps(sort_keys=True)``) with the option on and off -- the real transformers row loop runs
    over a fake tokenizer / model (no real model, CPU tensors only, a deterministic clock);
  * a never-answering run stops after N rows: N predictions, the partial-rows fields the Verdict card reads
    (``details.inference.samples`` < ``details.dataset.rows_available``), ``metrics.stopped_unanswered``;
  * the cutoff reuses each scorer's own parser and is ignored (recorded nowhere) for a free-text scorer;
  * the API forwards the field only when set and validates it; the eval Job's final progress says it stopped.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import unittest
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from fastapi.testclient import TestClient

_client_cm = None


def setUpModule():  # noqa: N802 - unittest hook
    global _client_cm
    from app.main import app

    _client_cm = TestClient(app)
    _client_cm.__enter__()


def tearDownModule():  # noqa: N802 - unittest hook
    if _client_cm is not None:
        _client_cm.__exit__(None, None, None)


# ── a fake causal LM: answer i of the list for the i-th generate call ─────────────


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __init__(self, answers):
        self.answers = answers

    def __len__(self):
        return 8

    def __call__(self, text, return_tensors=None, **kw):
        import torch

        return {"input_ids": torch.tensor([[2, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return self.answers[int(ids[0]) - 100]


class _FakeModel:
    def __init__(self):
        import torch

        self.dtype = torch.float32
        self.generation_config = SimpleNamespace()
        self.calls = 0

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=8)

    def eval(self):
        return self

    def to(self, *_a, **_k):
        return self

    def generate(self, **kw):
        import torch

        self.calls += 1
        return torch.tensor([[2, 3, 4, 100 + self.calls - 1]])


async def _none(*_a, **_k):
    """The probe pack / behavioral tests generate with the same model: stubbed so only held-out rows run."""
    return None


def _rows(n):
    return [{"question": f"Extract the name from: person {i}", "answer": json.dumps({"name": f"p{i}"})} for i in range(n)]


def _run(rows, answers_per_run, *, task_type="structured_extraction", option_values=(None,)):
    """One eval per entry of ``option_values`` (same project, set and fake answers) on a throwaway engine in one
    event loop; the real ``_run_transformers_inference`` loop runs over the fake model. Returns
    ``[(result, generate_calls)]``."""
    import tempfile
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    import app.models  # noqa: F401
    from app.database import Base
    from app.models.dataset import Dataset, DatasetType
    from app.models.experiment import Experiment, ExperimentStatus
    from app.models.project import Project
    from app.services import evaluation_service as ev

    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'su.db'}", future=True)
            sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            out = []
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                heldout = root / "test.jsonl"
                heldout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
                async with sf() as db:
                    project = Project(id=30_000 + uuid4().int % 10**9, name=f"su-{uuid4().hex[:8]}", description="x",
                                      base_model_name="toy/model", runtime_config={})
                    db.add(project)
                    await db.flush()
                    exp = Experiment(project_id=project.id, name="e", description="", status=ExperimentStatus.COMPLETED,
                                     base_model="toy/model", output_dir=str(root), config={"task_type": task_type})
                    db.add(exp)
                    db.add(Dataset(project_id=project.id, name="t", dataset_type=DatasetType.TEST,
                                   file_path=str(heldout), record_count=len(rows)))
                    await db.commit()
                    pid, eid = project.id, exp.id
                for value in option_values:
                    tok, model = _FakeTok(answers_per_run), _FakeModel()
                    clock = itertools.count(0, 0.25)  # deterministic: same calls -> same timings
                    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), \
                            mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
                            mock.patch("torch.cuda.is_available", return_value=False), \
                            mock.patch("app.services.evaluation_service.perf_counter", lambda: next(clock)), \
                            mock.patch("app.services.evaluation_service._resolve_model_reference", return_value="toy"), \
                            mock.patch("app.services.evaluation_service._safe_run_probe_pack", _none), \
                            mock.patch("app.services.evaluation_service._safe_run_behavioral_tests", _none):
                        async with sf() as db:
                            kw = {} if value is None else {"stop_if_unanswered_after": value}
                            result = await ev.run_heldout_evaluation(
                                db=db, project_id=pid, experiment_id=eid, dataset_name="test", eval_type="exact_match",
                                max_samples=100, answer_judge_enabled=False, **kw)
                            await db.commit()
                    out.append((result, model.calls))
                return out
            finally:
                await engine.dispose()

    return asyncio.run(scenario())


# Pointers to this result's own side files (they name the result id); everything else must match.
_POINTER_KEYS = ("row_predictions_file", "eval_predictions_file")


def _stored(result):
    details = {k: v for k, v in (result.details or {}).items() if k not in _POINTER_KEYS}
    return json.dumps({"metrics": result.metrics, "pass_rate": result.pass_rate, "details": details}, sort_keys=True)


class EquivalenceTests(unittest.TestCase):
    def test_a_run_that_answers_is_byte_identical_with_the_option_on(self):
        # 8 rows, N = 3: rows 1-2 are prose, row 3 parses -> the cutoff never fires; every row runs.
        answers = ["I cannot help.", "Sure! The name is p1.", '{"name": "p2"}', '{"name": "x"}',
                   "no", '{"name": "p5"}', "", '{"name": "p7"}']
        (off, off_calls), (on, on_calls) = _run(_rows(8), answers, option_values=(None, 3))
        self.assertEqual((off_calls, on_calls), (8, 8))
        self.assertEqual(_stored(on), _stored(off))
        self.assertNotIn("stopped_unanswered", on.metrics)
        self.assertEqual(on.details["inference"]["samples"], 8)

    def test_a_first_row_that_parses_also_keeps_every_row(self):
        answers = ['{"name": "p0"}'] + ["prose"] * 5
        (off, _), (on, calls) = _run(_rows(6), answers, option_values=(None, 2))
        self.assertEqual(calls, 6)
        self.assertEqual(_stored(on), _stored(off))


class CutoffTests(unittest.TestCase):
    def test_a_never_answering_run_stops_after_n_rows_as_a_partial_result(self):
        answers = ["Here is some prose about the person, no JSON at all."] * 10
        ((result, calls),) = _run(_rows(10), answers, option_values=(4,))
        self.assertEqual(calls, 4)  # 6 rows were never generated
        marker = result.metrics["stopped_unanswered"]
        self.assertEqual(marker, {
            "after_rows": 4, "rows_requested": 10, "format": None,
            "note": "The model never answered in this prompt format: stopped after 4 rows with 0 parsable answers.",
        })
        # The Verdict card's existing Partial warning: rows answered < rows in the set.
        self.assertEqual(result.details["inference"]["samples"], 4)
        self.assertEqual(result.details["dataset"]["rows_available"], 10)
        self.assertEqual(result.metrics["evaluated_samples"], 4)
        self.assertEqual(result.pass_rate, 0.0)

    def test_n_at_or_over_the_set_size_runs_every_row_unmarked(self):
        answers = ["prose"] * 3
        (off, _), (on, calls) = _run(_rows(3), answers, option_values=(None, 3))
        self.assertEqual(calls, 3)
        self.assertEqual(_stored(on), _stored(off))

    def test_a_free_text_scorer_ignores_the_option(self):
        rows = [{"question": f"Q{i}?", "answer": f"a{i}"} for i in range(5)]
        (off, _), (on, calls) = _run(rows, ["prose"] * 5, task_type="qa", option_values=(None, 2))
        self.assertEqual(calls, 5)
        self.assertEqual(_stored(on), _stored(off))


class ParserReuseTests(unittest.TestCase):
    def test_each_scorer_uses_its_own_parser(self):
        from app.services.eval_task_handler_service import (
            QAHandler, StructuredExtractionHandler, ToolCallHandler,
        )
        from app.services.evaluation_service import answer_parse_check

        def ctx(mode=None):
            return SimpleNamespace(manifest={"output_schema": {"scoring_mode": mode}} if mode else {})

        tool = answer_parse_check(ToolCallHandler(), ctx())
        self.assertTrue(tool('Tool Call: get_weather({"city": "x"})'))
        self.assertTrue(tool('{"tool": "a", "args": {}}'))
        self.assertFalse(tool("I would call the weather tool."))
        fields = answer_parse_check(StructuredExtractionHandler(), ctx("field_match"))
        self.assertTrue(fields('prose then {"a": 1}'))
        self.assertFalse(fields("[1, 2]"))  # field_match reads an object only
        spans = answer_parse_check(StructuredExtractionHandler(), ctx("span_set"))
        self.assertTrue(spans('[{"type": "NAME", "start": 0, "end": 1}]'))
        self.assertIsNone(answer_parse_check(QAHandler(), ctx()))


class ApiAndJobTests(unittest.TestCase):
    def _project(self):
        from app.database import async_session_factory
        from app.models.project import Project

        async def _go():
            async with async_session_factory() as db:
                p = Project(name=f"su-{uuid4().hex[:10]}", description="")
                db.add(p)
                await db.commit()
                return p.id

        return asyncio.run(_go())

    def test_run_heldout_forwards_the_field_only_when_set_and_validates_it(self):
        pid = self._project()
        bad = _client_cm.post(f"/api/projects/{pid}/evaluation/run-heldout",
                              json={"experiment_id": 1, "stop_if_unanswered_after": 0})
        self.assertEqual(bad.status_code, 422, bad.text)
        seen: list[dict] = []

        async def fake_run(**kwargs):
            seen.append(kwargs)
            raise ValueError("stop here")

        with mock.patch("app.api.evaluation.run_heldout_evaluation", fake_run):
            _client_cm.post(f"/api/projects/{pid}/evaluation/run-heldout",
                            json={"experiment_id": 1, "stop_if_unanswered_after": 50})
            _client_cm.post(f"/api/projects/{pid}/evaluation/run-heldout", json={"experiment_id": 1})
        self.assertEqual(seen[0]["stop_if_unanswered_after"], 50)
        self.assertNotIn("stop_if_unanswered_after", seen[1])

    def test_the_job_progress_says_it_stopped_early(self):
        from app.services import eval_jobs_service as ejs

        messages: list[str] = []

        class _Handle:
            async def set_progress(self, fraction=None, message=None):
                messages.append(message)

        async def work_stopped():
            return {"stopped_unanswered": {"after_rows": 50}}

        async def work_full():
            return {"pass_rate": 0.5}

        asyncio.run(ejs._run_with_heartbeat(_Handle(), describe="Eval on test examples", work=work_stopped))
        self.assertIn("stopped early after 50 rows (no parsable answer)", messages[-1])
        asyncio.run(ejs._run_with_heartbeat(_Handle(), describe="Eval on test examples", work=work_full))
        self.assertIn("finished in", messages[-1])
        result = SimpleNamespace(id=1, pass_rate=0.0, dataset_name="test", metrics={"stopped_unanswered": {"after_rows": 50}})
        with mock.patch("app.services.judge_call_guard.job_result_fields", return_value={}):
            self.assertEqual(ejs._summarize_eval_result(result, {})["stopped_unanswered"], {"after_rows": 50})
            plain = SimpleNamespace(id=2, pass_rate=1.0, dataset_name="test", metrics={})
            self.assertNotIn("stopped_unanswered", ejs._summarize_eval_result(plain, {}))

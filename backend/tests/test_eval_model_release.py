"""Held-out eval releases its model when the run ends.

Loading a checkpoint can import modules that keep a frame in a reference cycle
(``inspect.currentframe()`` held in a local at import time). That frame's ``f_back``
chain reaches the eval's own frame, so the loaded model survived the eval and was freed
only by the cyclic GC — whose object-count thresholds never notice a 50+ GB model — and a
second eval in the same process stacked a second copy on top.

Pins:
  * the sync inference path drops the model on return and on an error (weakref dead);
  * a ``heldout_evaluation`` Job (``?async_job=true``) releases it, and a second Job's
    load never sees the first Job's model alive;
  * CUDA's caching allocator is emptied when CUDA is available.

No real model runs: a fake tokenizer / model stands in for transformers, and the fake
loader reproduces the import-time frame cycle (the test runs with automatic GC off, so
only the eval's own release can free the model).
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle

_client_cm = None


def setUpModule():  # noqa: N802 - unittest hook
    global _client_cm
    from app.main import app

    _client_cm = TestClient(app)
    _client_cm.__enter__()


def tearDownModule():  # noqa: N802 - unittest hook
    if _client_cm is not None:
        _client_cm.__exit__(None, None, None)


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"
    chat_template = None

    def __len__(self):
        return 8

    def __call__(self, text, return_tensors=None, **kw):
        import torch

        return {"input_ids": torch.tensor([[2, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "an answer"


class _FakeModel:
    def __init__(self, fail: bool = False):
        import torch

        self.dtype = torch.float32
        self.generation_config = SimpleNamespace()
        self.fail = fail

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=8)

    def eval(self):
        return self

    def to(self, *_a, **_k):
        return self

    def generate(self, **kw):
        import torch

        if self.fail:
            raise RuntimeError("generation failed")
        return torch.tensor([[2, 3, 4, 5]])


class _Loader:
    """Stands in for ``AutoModelForCausalLM.from_pretrained``: hands out a fresh fake model,
    keeps only a weakref, and leaves a self-referencing frame behind (as an import-time
    ``inspect.currentframe()`` does), which pins the caller's frames until a GC runs."""

    def __init__(self, fail: bool = False):
        self.refs: list[weakref.ref] = []
        self.alive_at_load: list[list[bool]] = []
        self.fail = fail

    def __call__(self, *_a, **_k):
        self.alive_at_load.append([r() is not None for r in self.refs])
        frame = inspect.currentframe()  # noqa: F841 - the cycle under test
        model = _FakeModel(fail=self.fail)
        self.refs.append(weakref.ref(model))
        return model

    def alive(self) -> list[bool]:
        return [r() is not None for r in self.refs]


class _GcOff:
    """Automatic GC off for the block: only an explicit collect can free a cycle."""

    def __enter__(self):
        gc.collect()
        self.was = gc.isenabled()
        gc.disable()
        return self

    def __exit__(self, *exc):
        if self.was:
            gc.enable()
        return False


def _patches(loader):
    return (
        mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: _FakeTok()),
        mock.patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=loader),
        mock.patch("torch.cuda.is_available", return_value=False),
    )


def _infer(loader, pairs):
    from app.services import evaluation_service as ev

    p1, p2, p3 = _patches(loader)
    with p1, p2, p3:
        return ev._run_local_inference("fake/model-not-on-disk", pairs, 8, 0.0)


class InferenceReleaseTests(unittest.TestCase):
    PAIRS = [{"prompt": "q1", "reference": "a"}, {"prompt": "q2", "reference": "b"}]

    def test_the_model_is_dropped_when_inference_returns(self):
        loader = _Loader()
        with _GcOff():
            preds, runtime = _infer(loader, self.PAIRS)
            self.assertEqual([p["prediction"] for p in preds], ["an answer", "an answer"])
            self.assertEqual(runtime["engine"], "transformers")
            self.assertEqual(loader.alive(), [False])

    def test_the_cycle_would_pin_the_model_without_the_release(self):
        # Guards the test itself: with the release stubbed out the fake loader's frame cycle
        # keeps the model alive, so the test above really proves the release.
        from app.services import evaluation_service as ev

        loader = _Loader()
        with _GcOff(), mock.patch.object(ev, "release_inference_memory", lambda: None):
            _infer(loader, self.PAIRS)
            self.assertEqual(loader.alive(), [True])
        gc.collect()
        self.assertEqual(loader.alive(), [False])

    def test_the_model_is_dropped_when_inference_raises(self):
        loader = _Loader(fail=True)
        with _GcOff():
            try:
                _infer(loader, self.PAIRS)
            except RuntimeError as exc:
                self.assertIn("generation failed", str(exc))
                # Still inside the handler: the exception (and its traceback) is alive.
                self.assertEqual(loader.alive(), [False])
            else:
                self.fail("expected the generation error to propagate")

    def test_two_runs_never_hold_two_models(self):
        loader = _Loader()
        with _GcOff():
            _infer(loader, self.PAIRS)
            _infer(loader, self.PAIRS)
            self.assertEqual(loader.alive_at_load, [[], [False]])
            self.assertEqual(loader.alive(), [False, False])

    def test_cuda_cache_is_emptied_when_a_cuda_context_exists(self):
        from app.services import evaluation_service as ev

        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.is_initialized", return_value=True), \
                mock.patch("torch.cuda.empty_cache") as empty, \
                mock.patch("torch.cuda.ipc_collect") as ipc:
            ev.release_inference_memory()
        self.assertEqual((empty.call_count, ipc.call_count), (1, 1))

    def test_no_cuda_context_is_created_when_none_exists(self):
        # ipc_collect() lazily initialises CUDA; an eval that never used it (e.g. llama.cpp) must not open one.
        from app.services import evaluation_service as ev

        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.is_initialized", return_value=False), \
                mock.patch("torch.cuda.empty_cache") as empty, \
                mock.patch("torch.cuda.ipc_collect") as ipc:
            ev.release_inference_memory()
        self.assertEqual((empty.call_count, ipc.call_count), (0, 0))

    def test_nested_calls_release_once_at_the_outermost_return(self):
        from app.services import evaluation_service as ev

        loader = _Loader()
        p1, p2, p3 = _patches(loader)
        with p1, p2, p3, mock.patch.object(ev, "release_inference_memory") as release:
            ev._run_local_inference("fake/model-not-on-disk", self.PAIRS, 8, 0.0)
        self.assertEqual(release.call_count, 1)


# ── the heldout_evaluation Job ────────────────────────────────────────────


def _seed_project(root: Path) -> tuple[int, int]:
    from app.database import async_session_factory
    from app.models.dataset import Dataset, DatasetType
    from app.models.experiment import Experiment, ExperimentStatus
    from app.models.project import Project

    rows = [{"question": "What is 2+2?", "answer": "4"}, {"question": "Capital?", "answer": "Paris"}]
    heldout = root / "test.jsonl"
    heldout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    async def _seed():
        async with async_session_factory() as db:
            project = Project(name=f"release-{uuid4().hex[:8]}", description="x", base_model_name="toy/model")
            db.add(project)
            await db.flush()
            exp = Experiment(project_id=project.id, name="e", description="", status=ExperimentStatus.COMPLETED,
                             base_model="toy/model", output_dir=str(root))
            db.add(exp)
            db.add(Dataset(project_id=project.id, name="t", dataset_type=DatasetType.TEST,
                           file_path=str(heldout), record_count=len(rows)))
            await db.commit()
            return project.id, exp.id

    return asyncio.run(_seed())


class HeldoutJobReleaseTests(unittest.TestCase):
    def tearDown(self):
        wait_for_jobs_idle(_client_cm, timeout_s=60.0)

    def _run_job(self, pid: int, eid: int) -> dict:
        resp = _client_cm.post(
            f"/api/projects/{pid}/evaluation/run-heldout?async_job=true",
            json={"experiment_id": eid, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 5,
                  "max_new_tokens": 8, "temperature": 0.0, "model_path": "fake/model-not-on-disk"},
        )
        self.assertEqual(resp.status_code, 202, resp.text)
        job_id = resp.json()["id"]
        wait_for_jobs_idle(_client_cm, timeout_s=60.0)
        job = _client_cm.get(f"/api/jobs/{job_id}")
        self.assertEqual(job.status_code, 200, job.text)
        return job.json()

    def test_a_finished_job_holds_no_model_and_a_second_job_never_stacks(self):
        loader = _Loader()
        with tempfile.TemporaryDirectory() as tmp:
            pid, eid = _seed_project(Path(tmp))
            p1, p2, p3 = _patches(loader)
            with p1, p2, p3, _GcOff(), \
                    mock.patch("app.services.answer_judge_service.resolve_answer_judge",
                               mock.AsyncMock(return_value=None)):
                first = self._run_job(pid, eid)
                self.assertEqual(first["status"], "succeeded", first)
                self.assertEqual(loader.alive(), [False], "the finished Job still holds its model")
                second = self._run_job(pid, eid)
                self.assertEqual(second["status"], "succeeded", second)
                self.assertEqual(loader.alive_at_load, [[], [False]], "the second load saw the first model alive")
                self.assertEqual(loader.alive(), [False, False])

    def test_a_failed_job_holds_no_model(self):
        loader = _Loader(fail=True)
        with tempfile.TemporaryDirectory() as tmp:
            pid, eid = _seed_project(Path(tmp))
            p1, p2, p3 = _patches(loader)
            with p1, p2, p3, _GcOff(), \
                    mock.patch("app.services.answer_judge_service.resolve_answer_judge",
                               mock.AsyncMock(return_value=None)):
                job = self._run_job(pid, eid)
                self.assertEqual(job["status"], "failed", job)
                self.assertEqual(loader.alive(), [False], "the failed Job still holds its model")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

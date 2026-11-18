"""Cancelling a held-out eval Job stops the eval: the inference thread checks between rows
(``evaluation_service.CancellablePairs``), the Job ends CANCELLED, no EvalResult is stored (partial
predictions are discarded) and the rows after the cancel are never generated.

pytest-only. Model inference is stubbed (a slow per-row loop); no GPU, no network."""

from __future__ import annotations

import asyncio
import json
import threading
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
from app.services import evaluation_service as evs

client: TestClient | None = None
N_ROWS = 40


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    from _job_wait import wait_for_jobs_idle

    wait_for_jobs_idle(client, timeout_s=60)
    client.__exit__(None, None, None)


def run_async(fn):
    async def go():
        async with async_session_factory() as db:
            return await fn(db)

    return asyncio.run(go())


def seed() -> tuple[int, int]:
    pid = client.post("/api/projects", json={"name": f"eval-cancel-{time.time_ns()}", "description": "x"}).json()["id"]
    prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    rows = [{"question": f"What is {i}?", "answer": str(i)} for i in range(N_ROWS)]
    (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    async def add(db):
        db.add(Dataset(project_id=pid, name="test", dataset_type=DatasetType.TEST,
                       file_path=str(prepared / "test.jsonl"), record_count=len(rows)))
        exp = Experiment(project_id=pid, name="trained", status=ExperimentStatus.COMPLETED, base_model="toy",
                         config={}, output_dir=str(prepared), completed_at=datetime.now(timezone.utc))
        db.add(exp)
        await db.commit()
        return exp.id

    return pid, run_async(add)


class CancellablePairsTests(unittest.TestCase):
    def test_iterates_like_a_list_until_cancelled(self):
        event = threading.Event()
        pairs = evs.CancellablePairs([{"a": 1}, {"a": 2}, {"a": 3}], event)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[1], {"a": 2})
        self.assertEqual([p["a"] for p in pairs], [1, 2, 3])
        seen = []
        with self.assertRaises(evs.EvalCancelled):
            for i, p in enumerate(pairs):
                seen.append(p["a"])
                if i == 0:
                    event.set()
        self.assertEqual(seen, [1])

    def test_token_classification_tags_in_chunks_and_stops_between_them(self):
        rows = [{"structured_input": f"text {i}"} for i in range(7)]
        calls: list[list[str]] = []

        def predict(chunk):
            calls.append(list(chunk))
            return [[{"label": "X", "text": t}] for t in chunk]

        texts, spans = evs._predict_spans_in_chunks(evs.CancellablePairs(rows, threading.Event()), predict, 3)
        self.assertEqual([len(c) for c in calls], [3, 3, 1])
        self.assertEqual(texts, [f"text {i}" for i in range(7)])
        self.assertEqual([s[0]["text"] for s in spans], texts)

        event = threading.Event()
        calls.clear()

        def predict_then_cancel(chunk):
            calls.append(list(chunk))
            event.set()  # the Job is cancelled while the first chunk is tagged
            return [[] for _ in chunk]

        with self.assertRaises(evs.EvalCancelled):
            evs._predict_spans_in_chunks(evs.CancellablePairs(rows, event), predict_then_cancel, 3)
        self.assertEqual(len(calls), 1)  # the remaining chunks never ran


class EvalCancelJobTests(unittest.TestCase):
    def test_cancel_stops_the_eval_and_stores_nothing(self):
        pid, exp_id = seed()
        generated: list[int] = []
        first_row = threading.Event()
        exited = threading.Event()

        def slow_inference(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *,
                           apply_chat_template=True, decoding="neutral"):
            try:
                preds = []
                for i, p in enumerate(pairs):  # the engines' per-row loop
                    generated.append(i)
                    first_row.set()
                    time.sleep(0.1)
                    preds.append({**p, "formatted_prompt": "", "prediction": p["reference"],
                                  "generated_tokens": 1, "latency_ms": 1.0})
                return preds, {"engine": "stub", "decoding": {"mode": decoding}}
            finally:
                exited.set()

        with patch("app.services.evaluation_service._run_local_inference", slow_inference), \
                patch.object(evs, "EVAL_CANCEL_POLL_SECONDS", 0.05):
            resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", params={"async_job": "true"}, json={
                "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": N_ROWS})
            self.assertEqual(resp.status_code, 202, resp.text)
            job_id = resp.json()["id"]
            self.assertTrue(first_row.wait(15), "the eval never started generating")
            cancel = client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(cancel.status_code, 200, cancel.text)
            self.assertTrue(exited.wait(15), "the inference thread kept running after the cancel")

        self.assertLess(len(generated), N_ROWS)  # the rows after the cancel were never generated
        deadline = time.monotonic() + 15
        job = client.get(f"/api/jobs/{job_id}").json()
        while time.monotonic() < deadline and not job.get("completed_at"):
            time.sleep(0.1)
            job = client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(job["status"], "cancelled")
        time.sleep(0.3)  # let the runner unwind
        results = client.get(f"/api/projects/{pid}/evaluation/results/{exp_id}")
        self.assertEqual(results.status_code, 200, results.text)
        self.assertEqual(results.json(), [], "a cancelled eval must not store a (partial) result")


if __name__ == "__main__":
    unittest.main()

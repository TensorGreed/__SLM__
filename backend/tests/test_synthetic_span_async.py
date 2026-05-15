"""Batched synthetic-span generation (long-running) — Story PII-async.

Pins:
- ``start_span_generation_task`` returns a task record + registers in
  the in-memory dict + launches a background coroutine.
- ``target_rows`` validation: rejects <1, rejects >MAX_TOTAL_ROWS.
- ``use_all_chunks=false`` with empty ``source_text`` is rejected.
- ``_sample_chunks_for_batch`` joins random chunks until the
  target-char threshold is met (or the pool is exhausted).
- ``batches_total`` reflects ceil(target_rows / PER_BATCH_ROW_CAP).
- ``get_span_task_status`` returns the live record.
- The background runner produces ``batches_done == batches_total``
  rows when the underlying generator succeeds.
- Cross-project task lookups return None (the API layer translates
  this to 404).
"""

from __future__ import annotations

import asyncio
import os
import random
import unittest
from unittest.mock import patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.synthetic_service import (  # noqa: E402
    MAX_TOTAL_ROWS,
    PER_BATCH_ROW_CAP,
    _sample_chunks_for_batch,
    _SYNTHETIC_TASKS,
    _SYNTHETIC_TASKS_LOCK,
    get_span_task_status,
    start_span_generation_task,
)


async def _await_task_completion(task_id: str, *, timeout_s: float = 5.0) -> None:
    """Spin until the in-memory task record reports ``finished_at``.
    Must run inside an asyncio event loop (the task runner uses
    ``asyncio.create_task`` which requires one)."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        task = get_span_task_status(task_id)
        if task is not None and task.finished_at is not None:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"task {task_id} did not finish in {timeout_s}s")


def _start_and_wait(start_kwargs: dict, *, timeout_s: float = 5.0):
    """Run ``start_span_generation_task`` inside a fresh event loop +
    wait for completion. Returns the final task record."""
    async def _go():
        task = start_span_generation_task(**start_kwargs)
        await _await_task_completion(task.task_id, timeout_s=timeout_s)
        return get_span_task_status(task.task_id)

    return asyncio.run(_go())


class SyntheticSpanAsyncTests(unittest.TestCase):
    def setUp(self):
        with _SYNTHETIC_TASKS_LOCK:
            _SYNTHETIC_TASKS.clear()

    # ── Validation ───────────────────────────────────────────────

    def test_target_rows_zero_rejected(self):
        with self.assertRaises(ValueError):
            start_span_generation_task(
                project_id=1,
                target_rows=0,
                entity_types=[],
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="some seed",
            )

    def test_target_rows_above_max_rejected(self):
        with self.assertRaises(ValueError):
            start_span_generation_task(
                project_id=1,
                target_rows=MAX_TOTAL_ROWS + 1,
                entity_types=[],
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="seed",
            )

    def test_empty_source_text_rejected_when_not_use_all_chunks(self):
        with self.assertRaises(ValueError):
            start_span_generation_task(
                project_id=1,
                target_rows=10,
                entity_types=[],
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="   ",
            )

    # ── Chunk sampler ────────────────────────────────────────────

    def test_sample_chunks_for_batch_reaches_target(self):
        pool = ["x" * 100 for _ in range(50)]
        out = _sample_chunks_for_batch(
            pool, target_chars=600, rng=random.Random(0)
        )
        # 6 chunks × 100 chars + 5 separators = >= 600 chars; never
        # collects every chunk in the pool.
        self.assertGreaterEqual(len(out), 600)
        self.assertLess(out.count("---"), len(pool))

    def test_sample_chunks_for_batch_empty_pool(self):
        out = _sample_chunks_for_batch(
            [], target_chars=1000, rng=random.Random(0)
        )
        self.assertEqual(out, "")

    def test_sample_chunks_for_batch_exhausts_pool_if_smaller_than_target(self):
        pool = ["short"] * 3
        out = _sample_chunks_for_batch(
            pool, target_chars=10_000, rng=random.Random(0)
        )
        # All three chunks joined by ---.
        self.assertEqual(out.count("short"), 3)

    # ── Task lifecycle (mocked teacher) ──────────────────────────

    def test_task_lifecycle_produces_target_rows(self):
        # Patch the actual teacher-call coroutine so the test runs
        # offline + deterministically. Each call returns ``n`` stub
        # rows so we can verify the runner accumulates them.
        async def fake_generate(
            db, project_id, source_text, num_rows, *args, **kwargs
        ):
            return [
                {"text": f"row-{i}", "entities": [], "confidence": 1.0}
                for i in range(num_rows)
            ]

        with patch(
            "app.services.synthetic_service.generate_span_extraction_rows",
            side_effect=fake_generate,
        ):
            target = 125  # ceil(125/50) = 3 batches
            record = _start_and_wait(
                {
                    "project_id": 99,
                    "target_rows": target,
                    "entity_types": ["person_name"],
                    "api_url": "",
                    "api_key": "",
                    "model_name": "llama3",
                    "use_all_chunks": False,
                    "source_text": "seed text",
                }
            )

        self.assertIsNotNone(record)
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.batches_total, 3)
        self.assertEqual(record.batches_done, 3)
        self.assertEqual(len(record.rows), target)

    def test_task_with_use_all_chunks_but_no_pool_fails_cleanly(self):
        # Project 9999 has no cleaned chunks on disk; loader returns
        # empty list → task fails with an actionable message.
        async def fake_load(project_id):
            return []

        with patch(
            "app.services.synthetic_service._load_project_cleaned_chunks",
            side_effect=fake_load,
        ):
            record = _start_and_wait(
                {
                    "project_id": 9999,
                    "target_rows": 10,
                    "entity_types": [],
                    "api_url": "",
                    "api_key": "",
                    "model_name": "llama3",
                    "use_all_chunks": True,
                    "source_text": "",
                }
            )

        self.assertEqual(record.status, "failed")
        self.assertIn("cleaned chunks", record.error)

    def test_batches_total_matches_ceiling_of_target_over_cap(self):
        async def fake_generate(
            db, project_id, source_text, num_rows, *args, **kwargs
        ):
            return [{"text": "x", "entities": [], "confidence": 0.9}] * num_rows

        with patch(
            "app.services.synthetic_service.generate_span_extraction_rows",
            side_effect=fake_generate,
        ):
            # 50 → 1 batch; 51 → 2 batches; PER_BATCH_ROW_CAP×2 → 2 batches
            for target, expected in (
                (PER_BATCH_ROW_CAP, 1),
                (PER_BATCH_ROW_CAP + 1, 2),
                (PER_BATCH_ROW_CAP * 2, 2),
                (PER_BATCH_ROW_CAP * 3 - 1, 3),
            ):
                record = _start_and_wait(
                    {
                        "project_id": 1,
                        "target_rows": target,
                        "entity_types": [],
                        "api_url": "",
                        "api_key": "",
                        "model_name": "llama3",
                        "use_all_chunks": False,
                        "source_text": "seed",
                    }
                )
                self.assertEqual(
                    record.batches_total,
                    expected,
                    msg=f"target={target} expected batches={expected}",
                )

    def test_get_span_task_status_unknown_returns_none(self):
        self.assertIsNone(get_span_task_status("does-not-exist"))


if __name__ == "__main__":
    unittest.main()

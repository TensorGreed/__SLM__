"""Batched synthetic QA + conversation generation (USER-SUCCESS Epic 2c).

Mirrors ``test_synthetic_span_async.py`` for the QA + conversation
async runners that were added to lift the 50-row cap on
``/generate`` and the 20-dialogue cap on ``/generate-conversations``.

Pins:
- ``start_qa_generation_task`` + ``start_conversation_generation_task``
  validate target_rows + source_text the same way the span variant does.
- ``batches_total`` reflects ceil(target / per-batch cap), where the
  cap differs by kind (QA = 50, Conversation = 5).
- The shared ``get_synth_task_status`` returns each kind keyed by id.
- ``task_kind`` discriminator round-trips through ``to_dict()``.
- ``use_all_chunks=True`` with an empty cleaned-chunks pool fails the
  task cleanly (matches span behavior).
"""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.synthetic_service import (  # noqa: E402
    MAX_TOTAL_ROWS,
    PER_BATCH_CONVERSATION_CAP,
    PER_BATCH_ROW_CAP,
    SyntheticConversationTask,
    SyntheticQaTask,
    _SYNTHETIC_TASKS,
    _SYNTHETIC_TASKS_LOCK,
    get_synth_task_status,
    start_conversation_generation_task,
    start_qa_generation_task,
)


async def _await_task_completion(task_id: str, *, timeout_s: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        task = get_synth_task_status(task_id)
        if task is not None and task.finished_at is not None:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"task {task_id} did not finish in {timeout_s}s")


def _start_qa_and_wait(start_kwargs: dict, *, timeout_s: float = 5.0):
    async def _go():
        task = start_qa_generation_task(**start_kwargs)
        await _await_task_completion(task.task_id, timeout_s=timeout_s)
        return get_synth_task_status(task.task_id)

    return asyncio.run(_go())


def _start_conv_and_wait(start_kwargs: dict, *, timeout_s: float = 5.0):
    async def _go():
        task = start_conversation_generation_task(**start_kwargs)
        await _await_task_completion(task.task_id, timeout_s=timeout_s)
        return get_synth_task_status(task.task_id)

    return asyncio.run(_go())


class SyntheticQaAsyncTests(unittest.TestCase):
    def setUp(self):
        with _SYNTHETIC_TASKS_LOCK:
            _SYNTHETIC_TASKS.clear()

    def test_target_rows_zero_rejected(self):
        with self.assertRaises(ValueError):
            start_qa_generation_task(
                project_id=1,
                target_rows=0,
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="seed",
            )

    def test_target_rows_above_max_rejected(self):
        with self.assertRaises(ValueError):
            start_qa_generation_task(
                project_id=1,
                target_rows=MAX_TOTAL_ROWS + 1,
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="seed",
            )

    def test_empty_source_text_rejected_when_not_use_all_chunks(self):
        with self.assertRaises(ValueError):
            start_qa_generation_task(
                project_id=1,
                target_rows=10,
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="   ",
            )

    def test_task_lifecycle_produces_target_rows(self):
        async def fake_generate(
            db, project_id, source_text, num_rows, *args, **kwargs
        ):
            return [
                {"question": f"q-{i}", "answer": f"a-{i}", "confidence": 1.0}
                for i in range(num_rows)
            ]

        with patch(
            "app.services.synthetic_service.generate_qa_pairs",
            side_effect=fake_generate,
        ):
            # 125 with PER_BATCH_ROW_CAP=50 → 3 batches.
            target = 125
            record = _start_qa_and_wait(
                {
                    "project_id": 77,
                    "target_rows": target,
                    "api_url": "",
                    "api_key": "",
                    "model_name": "llama3",
                    "use_all_chunks": False,
                    "source_text": "seed text",
                }
            )

        self.assertIsInstance(record, SyntheticQaTask)
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.batches_total, 3)
        self.assertEqual(record.batches_done, 3)
        self.assertEqual(len(record.rows), target)
        # to_dict() carries the discriminator so the frontend can
        # branch on row shape (QA pairs vs span rows vs conversations).
        self.assertEqual(record.to_dict()["task_kind"], "qa")

    def test_task_with_use_all_chunks_but_no_pool_fails_cleanly(self):
        async def fake_load(project_id):
            return []

        with patch(
            "app.services.synthetic_service._load_project_cleaned_chunks",
            side_effect=fake_load,
        ):
            record = _start_qa_and_wait(
                {
                    "project_id": 9999,
                    "target_rows": 10,
                    "api_url": "",
                    "api_key": "",
                    "model_name": "llama3",
                    "use_all_chunks": True,
                    "source_text": "",
                }
            )

        self.assertEqual(record.status, "failed")
        self.assertIn("cleaned chunks", record.error)

    def test_batches_total_matches_ceiling_over_cap(self):
        async def fake_generate(
            db, project_id, source_text, num_rows, *args, **kwargs
        ):
            return [
                {"question": "q", "answer": "a", "confidence": 1.0}
                for _ in range(num_rows)
            ]

        with patch(
            "app.services.synthetic_service.generate_qa_pairs",
            side_effect=fake_generate,
        ):
            for target, expected in (
                (PER_BATCH_ROW_CAP, 1),
                (PER_BATCH_ROW_CAP + 1, 2),
                (PER_BATCH_ROW_CAP * 2, 2),
                (PER_BATCH_ROW_CAP * 3 - 1, 3),
            ):
                record = _start_qa_and_wait(
                    {
                        "project_id": 1,
                        "target_rows": target,
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


class SyntheticConversationAsyncTests(unittest.TestCase):
    def setUp(self):
        with _SYNTHETIC_TASKS_LOCK:
            _SYNTHETIC_TASKS.clear()

    def test_target_rows_zero_rejected(self):
        with self.assertRaises(ValueError):
            start_conversation_generation_task(
                project_id=1,
                target_rows=0,
                min_turns=3,
                max_turns=5,
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="seed",
            )

    def test_target_rows_above_max_rejected(self):
        with self.assertRaises(ValueError):
            start_conversation_generation_task(
                project_id=1,
                target_rows=MAX_TOTAL_ROWS + 1,
                min_turns=3,
                max_turns=5,
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="seed",
            )

    def test_min_turns_greater_than_max_rejected(self):
        with self.assertRaises(ValueError):
            start_conversation_generation_task(
                project_id=1,
                target_rows=10,
                min_turns=6,
                max_turns=3,
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="seed",
            )

    def test_empty_source_text_rejected_when_not_use_all_chunks(self):
        with self.assertRaises(ValueError):
            start_conversation_generation_task(
                project_id=1,
                target_rows=10,
                min_turns=3,
                max_turns=5,
                api_url="",
                api_key="",
                model_name="llama3",
                use_all_chunks=False,
                source_text="",
            )

    def test_task_lifecycle_produces_target_rows(self):
        async def fake_generate(
            db, project_id, source_text, num_dialogues, *args, **kwargs
        ):
            return [
                {"turns": [{"role": "user", "content": f"hi-{i}"}], "confidence": 1.0}
                for i in range(num_dialogues)
            ]

        with patch(
            "app.services.synthetic_service.generate_conversation_dialogues",
            side_effect=fake_generate,
        ):
            # 12 with PER_BATCH_CONVERSATION_CAP=5 → 3 batches.
            target = 12
            record = _start_conv_and_wait(
                {
                    "project_id": 77,
                    "target_rows": target,
                    "min_turns": 3,
                    "max_turns": 5,
                    "api_url": "",
                    "api_key": "",
                    "model_name": "llama3",
                    "use_all_chunks": False,
                    "source_text": "seed text",
                }
            )

        self.assertIsInstance(record, SyntheticConversationTask)
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.batches_total, 3)
        self.assertEqual(record.batches_done, 3)
        self.assertEqual(len(record.rows), target)
        self.assertEqual(record.to_dict()["task_kind"], "conversation")

    def test_batches_total_matches_ceiling_over_cap(self):
        async def fake_generate(
            db, project_id, source_text, num_dialogues, *args, **kwargs
        ):
            return [
                {"turns": [], "confidence": 1.0} for _ in range(num_dialogues)
            ]

        with patch(
            "app.services.synthetic_service.generate_conversation_dialogues",
            side_effect=fake_generate,
        ):
            for target, expected in (
                (PER_BATCH_CONVERSATION_CAP, 1),
                (PER_BATCH_CONVERSATION_CAP + 1, 2),
                (PER_BATCH_CONVERSATION_CAP * 4, 4),
            ):
                record = _start_conv_and_wait(
                    {
                        "project_id": 1,
                        "target_rows": target,
                        "min_turns": 3,
                        "max_turns": 5,
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

    def test_task_with_use_all_chunks_but_no_pool_fails_cleanly(self):
        async def fake_load(project_id):
            return []

        with patch(
            "app.services.synthetic_service._load_project_cleaned_chunks",
            side_effect=fake_load,
        ):
            record = _start_conv_and_wait(
                {
                    "project_id": 9999,
                    "target_rows": 5,
                    "min_turns": 3,
                    "max_turns": 5,
                    "api_url": "",
                    "api_key": "",
                    "model_name": "llama3",
                    "use_all_chunks": True,
                    "source_text": "",
                }
            )

        self.assertEqual(record.status, "failed")
        self.assertIn("cleaned chunks", record.error)


class GenericSynthTaskStatusTests(unittest.TestCase):
    def setUp(self):
        with _SYNTHETIC_TASKS_LOCK:
            _SYNTHETIC_TASKS.clear()

    def test_get_synth_task_status_unknown_returns_none(self):
        self.assertIsNone(get_synth_task_status("does-not-exist"))


if __name__ == "__main__":
    unittest.main()

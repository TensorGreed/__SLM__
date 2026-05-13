"""Phase 5.3.0 — task-aware eval dispatcher (foundation).

Asserts the dispatcher contract that everything in Phase 5.3.1+ relies
on:

- Unknown / missing / malformed ``task_profile`` falls through to
  ``GenericHandler`` (never raises, never returns ``None``).
- A registered handler is returned for matching profiles.
- A buggy handler factory falls back to GenericHandler instead of
  crashing the eval.
- ``GenericHandler.build_prompts`` mirrors the pre-dispatcher
  ``_extract_prompt_and_reference`` field-precedence rules.
- ``GenericHandler.score`` produces the same metric dict the old
  if/elif chain produced for each ``eval_type``.

Phase 5.3.0 is a no-op refactor by design; this file is the
contract-pinning suite that catches drift.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.eval_task_handler_service import (  # noqa: E402
    EvalContext,
    GenericHandler,
    TaskHandler,
    _HANDLER_FACTORIES,
    build_eval_context,
    list_registered_profiles,
    register_handler,
    resolve_task_handler,
)


def _ctx(eval_type: str = "exact_match") -> EvalContext:
    from pathlib import Path

    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type=eval_type,
        task_profile=None,
        handler_id="generic",
        prepared_dir=Path("."),
        dataset_name="",
    )


class DispatcherFallbackTests(unittest.TestCase):
    def setUp(self):
        # Snapshot registry so individual tests can register without
        # polluting later cases.
        self._registry_snapshot = dict(_HANDLER_FACTORIES)

    def tearDown(self):
        _HANDLER_FACTORIES.clear()
        _HANDLER_FACTORIES.update(self._registry_snapshot)

    def test_none_profile_returns_generic(self):
        handler = resolve_task_handler(None)
        self.assertIsInstance(handler, GenericHandler)
        self.assertEqual(handler.profile_id, "generic")

    def test_empty_string_profile_returns_generic(self):
        self.assertIsInstance(resolve_task_handler(""), GenericHandler)
        self.assertIsInstance(resolve_task_handler("   "), GenericHandler)

    def test_unknown_profile_returns_generic(self):
        self.assertIsInstance(
            resolve_task_handler("not_a_real_profile"), GenericHandler
        )

    def test_non_string_profile_returns_generic(self):
        # Defensive: callers reading from JSON can pass anything.
        self.assertIsInstance(resolve_task_handler(42), GenericHandler)  # type: ignore[arg-type]
        self.assertIsInstance(resolve_task_handler({"x": 1}), GenericHandler)  # type: ignore[arg-type]

    def test_registered_handler_takes_precedence(self):
        class FakeClassificationHandler:
            profile_id = "classification"

            def build_prompts(self, rows, ctx):  # noqa: ARG002
                return []

            def score(self, predictions, ctx):  # noqa: ARG002
                return {"accuracy": 0.42}

        register_handler("classification", FakeClassificationHandler)

        handler = resolve_task_handler("classification")
        self.assertIsInstance(handler, TaskHandler)
        self.assertEqual(handler.profile_id, "classification")
        # Profile normalization: case + whitespace tolerant.
        self.assertEqual(
            resolve_task_handler("  CLASSIFICATION  ").profile_id, "classification"
        )

    def test_buggy_factory_falls_back_to_generic(self):
        def broken_factory():
            raise RuntimeError("simulated registry corruption")

        register_handler("breakme", broken_factory)
        handler = resolve_task_handler("breakme")
        self.assertIsInstance(handler, GenericHandler)

    def test_list_registered_profiles_excludes_generic(self):
        register_handler("classification", lambda: GenericHandler())
        profiles = list_registered_profiles()
        self.assertIn("classification", profiles)
        self.assertNotIn("generic", profiles)

    def test_register_rejects_empty_id(self):
        with self.assertRaises(ValueError):
            register_handler("", lambda: GenericHandler())
        with self.assertRaises(ValueError):
            register_handler("   ", lambda: GenericHandler())


class GenericHandlerPromptBuildingTests(unittest.TestCase):
    """``GenericHandler.build_prompts`` must mirror the pre-dispatcher
    field-precedence in ``_extract_prompt_and_reference``."""

    def _build(self, rows):
        return GenericHandler().build_prompts(rows, _ctx())

    def test_qa_pair_fields(self):
        built = self._build(
            [{"question": "what's the capital of France?", "answer": "Paris"}]
        )
        self.assertEqual(built[0].prompt, "what's the capital of France?")
        self.assertEqual(built[0].reference, "Paris")

    def test_prompt_field_takes_precedence_over_question(self):
        # Same field-precedence as old _extract_prompt_and_reference:
        # prompt > question > instruction > input > source_text.
        built = self._build(
            [{"prompt": "from prompt", "question": "from question", "answer": "x"}]
        )
        self.assertEqual(built[0].prompt, "from prompt")

    def test_reference_falls_through_answer_to_target_text(self):
        built = self._build([{"question": "q", "target_text": "fallback"}])
        self.assertEqual(built[0].reference, "fallback")

    def test_image_path_flows_into_extras(self):
        built = self._build(
            [{"question": "describe", "answer": "a cat", "image_path": "a.png"}]
        )
        self.assertEqual(built[0].extras.get("image_path"), "a.png")

    def test_audio_path_flows_into_extras(self):
        built = self._build(
            [{"prompt": "transcribe", "transcript": "hello", "audio_path": "a.wav"}]
        )
        self.assertEqual(built[0].extras.get("audio_path"), "a.wav")


class GenericHandlerScoringTests(unittest.TestCase):
    """``GenericHandler.score`` must produce the same metric keys + values
    the pre-dispatcher if/elif chain produced."""

    def test_exact_match_metric_shape(self):
        handler = GenericHandler()
        out = handler.score(
            [
                {"prediction": "Paris", "reference": "paris"},
                {"prediction": "London", "reference": "Paris"},
            ],
            _ctx(eval_type="exact_match"),
        )
        self.assertEqual(out["total"], 2)
        self.assertEqual(out["correct"], 1)
        self.assertEqual(out["exact_match"], 0.5)

    def test_f1_metric_shape(self):
        out = GenericHandler().score(
            [{"prediction": "the cat sat", "reference": "the cat sat on the mat"}],
            _ctx(eval_type="f1"),
        )
        self.assertEqual(out["total"], 1)
        self.assertIn("f1", out)
        self.assertGreater(out["f1"], 0.0)

    def test_safety_metric_shape(self):
        from app.services.evaluation_service import evaluate_safety_response

        # Confirm we can score a safety prediction in the same shape.
        safety_pred = {
            "response": "I cannot help with that request.",
            "test_type": "prompt_injection",
        }
        # Sanity: the underlying evaluator at least returns a dict.
        self.assertIsInstance(evaluate_safety_response(safety_pred["response"], "prompt_injection"), dict)
        out = GenericHandler().score(
            [safety_pred],
            _ctx(eval_type="safety"),
        )
        self.assertIn("pass_rate", out)
        self.assertEqual(out["total_tests"], 1)

    def test_unknown_eval_type_returns_empty(self):
        out = GenericHandler().score([{"prediction": "x", "reference": "x"}], _ctx(eval_type="????"))
        self.assertEqual(out, {})


class BuildEvalContextTests(unittest.TestCase):
    def test_returns_generic_when_no_manifest(self):
        ctx, handler = build_eval_context(
            project_id=999_999,  # unlikely to exist on disk
            experiment_id=1,
            eval_type="exact_match",
            dataset_name="test",
        )
        self.assertIsNone(ctx.task_profile)
        self.assertEqual(ctx.handler_id, "generic")
        self.assertIsInstance(handler, GenericHandler)

    def test_handler_id_matches_ctx(self):
        # Even when the handler is generic, ctx.handler_id must equal
        # handler.profile_id — that field is the load-bearing log of
        # "which handler scored this run".
        ctx, handler = build_eval_context(
            project_id=999_999,
            experiment_id=1,
            eval_type="f1",
            dataset_name="test",
        )
        self.assertEqual(ctx.handler_id, handler.profile_id)


if __name__ == "__main__":
    unittest.main()

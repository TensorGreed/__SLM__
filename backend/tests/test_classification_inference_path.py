"""Classification-aware inference path (fix #3).

Held-out eval used to dispatch every project to GenericHandler/QAHandler
when the prepared manifest didn't declare a task_profile — and then
wrap every prompt in the model's chat template, converting
classification-tuned LoRAs into chat-completion models that emit
"I'm sorry, I'm here to help" instead of "injection" / "benign".

The fix has two layers, both pinned here:

  1. ``build_eval_context`` falls back to the experiment's
     ``config.task_type`` when the manifest's task_profile is empty,
     so classification experiments route to ClassificationHandler
     even on dataset-import projects (which don't set task_profile
     in the prepared manifest).

  2. Handlers can opt out of the model's chat-template wrap by
     declaring ``wraps_own_prompt() → True`` (ClassificationHandler
     does). The held-out eval orchestrator queries this and threads
     ``apply_chat_template=False`` through to the transformers
     inference call, so the classification handler's built prompt
     ("Classify the following text. … Label:") reaches the model
     unwrapped.

These tests don't actually run the model — they pin the dispatch +
plumbing so we don't silently regress to the broken behavior.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-cls-infer-{uuid.uuid4().hex[:8]}"
)
TEST_DB_PATH = Path(__file__).resolve().parent / f"cls_infer_{uuid.uuid4().hex[:8]}.db"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from app.config import settings  # noqa: E402
from app.services.eval_task_handler_service import (  # noqa: E402
    ClassificationHandler,
    GenericHandler,
    QAHandler,
    build_eval_context,
)


class ClassificationDispatchTests(unittest.TestCase):
    """Layer 1 — handler dispatch."""

    @classmethod
    def setUpClass(cls):
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def test_handler_falls_back_to_experiment_task_type_when_manifest_silent(self):
        """The bug we're pinning: dataset-import projects write a
        prepared manifest with ``task_profile: None``. Without the
        experiment-task-type fallback, build_eval_context picks
        GenericHandler + chat-template-wraps every prompt + eval
        produces 0% F1 for classification experiments.

        With the fallback (this test): the experiment's ``task_type=
        classification`` flag wins and ClassificationHandler is
        picked, which builds classification-style prompts."""
        # No manifest file means read_prepared_manifest returns {} +
        # read_task_profile_from_manifest returns None. Simulate by
        # using a project id with no manifest dir.
        ctx, handler = build_eval_context(
            project_id=99999,
            experiment_id=1,
            eval_type="f1",
            dataset_name="gold_test",
            experiment_task_type="classification",
        )
        self.assertIsInstance(handler, ClassificationHandler)
        self.assertEqual(ctx.handler_id, "classification")
        self.assertEqual(ctx.task_profile, "classification")

    def test_handler_picks_generic_when_neither_manifest_nor_exp_have_task_type(self):
        """Regression: when both manifest and experiment task_type
        are missing we should still get a stable default (Generic)
        and not crash."""
        ctx, handler = build_eval_context(
            project_id=99999,
            experiment_id=1,
            eval_type="f1",
            dataset_name="gold_test",
            experiment_task_type=None,
        )
        self.assertIsInstance(handler, GenericHandler)

    def test_handler_dispatch_respects_explicit_task_types(self):
        """Each supported task_type maps to its registered handler.
        Mapping is identity for most; ``causal_lm`` maps into the
        QA family because that's where ``language_modeling`` lives
        in the registry."""
        for task_type, expected_cls in [
            ("classification", ClassificationHandler),
            ("causal_lm", QAHandler),
        ]:
            ctx, handler = build_eval_context(
                project_id=99999,
                experiment_id=1,
                eval_type="f1",
                dataset_name="gold_test",
                experiment_task_type=task_type,
            )
            self.assertIsInstance(handler, expected_cls, task_type)

    def test_manifest_task_profile_still_wins_when_present(self):
        """The fallback is *only* a fallback. If the manifest declares
        a task_profile (the desired long-term state once dataset-prep
        writes it correctly), that value wins regardless of the
        experiment's task_type. This protects projects whose
        manifest is more authoritative than a misconfigured
        experiment."""
        # Materialize a manifest under the test DATA_DIR.
        pid = 12345
        manifest_dir = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        import json
        (manifest_dir / "manifest.json").write_text(
            json.dumps({"task_profile": "classification"}),
        )
        # Mismatch: manifest says classification, experiment says
        # causal_lm. Manifest should win.
        _ctx, handler = build_eval_context(
            project_id=pid,
            experiment_id=1,
            eval_type="f1",
            dataset_name="gold_test",
            experiment_task_type="causal_lm",
        )
        self.assertIsInstance(handler, ClassificationHandler)


class WrapsOwnPromptTests(unittest.TestCase):
    """Layer 2 — handler signals it builds its own prompt.

    Six handlers build complete instruction prompts in their
    ``build_prompts`` (classification / structured / RAG / seq2seq /
    vision / audio). Each opts out of the chat-template wrap. Four
    handlers (generic / qa / safety / alignment) pass through raw
    inputs and rely on the model's chat template, so they keep the
    default (chat-template-on)."""

    def test_classification_handler_opts_out_of_chat_template(self):
        """ClassificationHandler builds a complete prompt
        ("Classify the following text. Reply with exactly one of: …
        Label:") inside build_prompts. Wrapping that in the model's
        chat template would convert the classification-tuned model
        into a chat-completion model, which is exactly the bug this
        fix addresses."""
        self.assertTrue(ClassificationHandler().wraps_own_prompt())

    def test_all_instruction_handlers_opt_out_of_chat_template(self):
        """The 6 handlers whose ``build_prompts`` produces a full
        instruction template all return ``wraps_own_prompt: True``.
        These follow the same trap as ClassificationHandler — without
        the opt-out, the chat-template wrap converts the task-tuned
        model into chat-completion mode at eval time and yields
        gibberish for task-specific outputs."""
        from app.services.eval_task_handler_service import (
            AudioTranscriptHandler,
            RAGHandler,
            Seq2SeqHandler,
            StructuredExtractionHandler,
            VisionLanguageHandler,
        )
        for cls in [
            ClassificationHandler,
            StructuredExtractionHandler,
            RAGHandler,
            Seq2SeqHandler,
            VisionLanguageHandler,
            AudioTranscriptHandler,
        ]:
            instance = cls()
            self.assertTrue(
                bool(getattr(instance, "wraps_own_prompt", lambda: False)()),
                f"{cls.__name__} should opt out of chat-template wrap",
            )

    def test_default_handlers_keep_chat_template_wrap(self):
        """Handlers that pass through raw inputs (Generic / QA /
        Safety / Alignment) don't override wraps_own_prompt — the
        inference path treats the absence of the method as ``False``
        (apply chat template). These rely on the model's chat
        template because they pass raw user inputs without
        task-specific framing."""
        from app.services.eval_task_handler_service import (
            AlignmentHandler,
            SafetyHandler,
        )
        for handler in [
            GenericHandler(),
            QAHandler(),
            SafetyHandler(),
            AlignmentHandler(),
        ]:
            wraps = bool(getattr(handler, "wraps_own_prompt", lambda: False)())
            self.assertFalse(
                wraps,
                f"{type(handler).__name__} should keep chat-template wrap",
            )


class InferenceChatTemplatePlumbingTests(unittest.TestCase):
    """Layer 2 — apply_chat_template flag threads through inference.

    Tests the plumbing without instantiating a real model. We patch
    ``_run_transformers_inference`` and assert its kwargs to verify
    the orchestrator wires up the chat-template flag correctly given
    the chosen handler.
    """

    def test_run_local_inference_forwards_apply_chat_template_flag(self):
        """_run_local_inference is the dispatch layer that hands off
        to either transformers or llama.cpp. The flag must reach the
        transformers backend; the llama.cpp backend doesn't apply
        chat templates anyway."""
        from app.services.evaluation_service import _run_local_inference

        with patch(
            "app.services.evaluation_service._run_transformers_inference",
            return_value=([], {"engine": "transformers"}),
        ) as mock_tf:
            _run_local_inference(
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                pairs=[],
                max_new_tokens=16,
                temperature=0.0,
                stop_sequences=[],
                apply_chat_template=False,
            )
            mock_tf.assert_called_once()
            self.assertFalse(mock_tf.call_args.kwargs["apply_chat_template"])

    def test_run_local_inference_defaults_to_apply_chat_template_true(self):
        """Backward compat: callers that don't set the flag (qa-sft
        etc.) get the original chat-template-on behavior."""
        from app.services.evaluation_service import _run_local_inference

        with patch(
            "app.services.evaluation_service._run_transformers_inference",
            return_value=([], {}),
        ) as mock_tf:
            _run_local_inference("model_ref", [], 16, 0.0, [])
            self.assertTrue(mock_tf.call_args.kwargs["apply_chat_template"])


if __name__ == "__main__":
    unittest.main()

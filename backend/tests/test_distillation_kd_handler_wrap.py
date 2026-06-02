"""Distillation β-fix tests — teacher capture wraps with the eval
handler's prompt format so the student trains on the same
scaffold the held-out eval will build.

Pre-fix gap (audit-confirmed): the teacher capture wrote raw row
prompts into ``teacher_capture.jsonl``; the student trained on
``raw_prompt + teacher_completion`` tokens; held-out eval ran
through a handler that wrapped inputs with an instruction
template (``"Classify the following text. …\\nLabel:"`` for
classification; ``"Extract the following fields as JSON: …\\n
Output:"`` for structured-extraction; etc.). The student never
saw the eval scaffold → held-out metrics collapsed even though
train-time KD loss looked healthy. Structurally identical to the
SQLi pre-β collapse.

Post-fix:
  1. ``_resolve_handler_wrapped_prompts`` resolves the project's
     task_profile, picks the matching handler, and calls
     ``handler.build_prompts(rows, ctx)`` to get the
     handler-emitted prompt for each row. Returns ``None`` for
     handlers that don't wrap (QA / language_modeling have a
     separate chat-template gap, not addressed here).
  2. ``capture_teacher_outputs`` uses the wrapped prompt as the
     teacher's input (so the captured logits are over the answer
     tokens the eval handler will elicit) and persists the
     wrapped string on the captured row as ``wrapped_prompt``,
     plus ``task_profile`` and ``handler_id`` for provenance.
  3. ``kd_capture._extract_prompt_text`` reads ``wrapped_prompt``
     preferentially. Legacy captures (without the field) still
     build records via the existing raw-field fallback.
  4. Byte-for-byte: the wrapped prompt the capture stores
     matches what ``ClassificationHandler._build_prompt_text``
     produces for the same input + candidates.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.config import settings  # noqa: E402
from app.services.distillation.kd_capture import _extract_prompt_text  # noqa: E402
from app.services.distillation.teacher_capture import (  # noqa: E402
    _resolve_handler_wrapped_prompts,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-kd-wrap-{uuid.uuid4().hex[:8]}"
)


def _write_manifest(project_id: int, task_profile: str | None) -> Path:
    """Build a fake prepared/manifest.json so
    ``read_task_profile_from_manifest`` returns the value we want."""
    prepared_dir = (
        TEST_DATA_DIR
        / "projects"
        / str(project_id)
        / "prepared"
    )
    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"adapter_id": "classification-label"}
    if task_profile is not None:
        manifest["task_profile"] = task_profile
    (prepared_dir / "manifest.json").write_text(json.dumps(manifest))
    return prepared_dir / "manifest.json"


class ResolveHandlerWrappedPromptsTests(unittest.TestCase):
    """The resolver — returns per-row wrapped prompts when the
    project's handler wraps; ``None`` otherwise (signals the
    capture caller to fall back to raw extraction)."""

    @classmethod
    def setUpClass(cls):
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()

    def test_classification_project_returns_wrapped_prompts(self):
        # Classification handler wraps with the SQLi-style prompt.
        # Capture should pre-wrap so the teacher sees the same
        # scaffold the student will be eval'd on.
        pid = 9001
        _write_manifest(pid, "classification")
        rows = [
            {"text": "free entry to win", "label": "spam"},
            {"text": "team meeting at 3", "label": "ham"},
        ]
        wrapped, provenance = _resolve_handler_wrapped_prompts(pid, rows)
        self.assertIsNotNone(wrapped)
        self.assertEqual(len(wrapped), 2)
        # Each wrap carries the load-bearing pieces of the
        # classification scaffold.
        for w in wrapped:
            self.assertIsNotNone(w)
            self.assertIn("Classify the following text", w)
            self.assertIn("Label:", w)
        # The row's text appears verbatim in the wrap.
        self.assertIn("free entry to win", wrapped[0])
        self.assertEqual(provenance["task_profile"], "classification")
        self.assertEqual(provenance["handler_id"], "classification")

    def test_structured_extraction_project_returns_wrapped_prompts(self):
        # Same shape for structured-extraction (ζ-aligned). The
        # handler wraps with "Extract the following fields as JSON…"
        # at eval; the capture should pre-wrap to match.
        pid = 9002
        _write_manifest(pid, "structured_extraction")
        rows = [
            {
                "text": "John works at Acme.",
                "structured_output": {"name": "John", "company": "Acme"},
            },
        ]
        wrapped, provenance = _resolve_handler_wrapped_prompts(pid, rows)
        self.assertIsNotNone(wrapped)
        self.assertIn("Extract", wrapped[0])
        self.assertIn("Output:", wrapped[0])
        self.assertEqual(provenance["task_profile"], "structured_extraction")

    def test_qa_project_returns_none(self):
        # QA handler doesn't wrap — its eval applies the chat
        # template instead. That's a different gap; the wrapped-
        # prompt resolver explicitly opts out and signals the
        # caller to use raw extraction.
        pid = 9003
        _write_manifest(pid, "qa")
        rows = [{"question": "what is 2+2?", "answer": "4"}]
        wrapped, _ = _resolve_handler_wrapped_prompts(pid, rows)
        self.assertIsNone(wrapped)

    def test_project_without_manifest_returns_none(self):
        # Pre-data-prep state. Resolver can't know the handler
        # without a manifest; caller falls back to raw.
        pid = 9004
        # Don't write a manifest.
        rows = [{"text": "x"}]
        wrapped, provenance = _resolve_handler_wrapped_prompts(pid, rows)
        self.assertIsNone(wrapped)
        self.assertEqual(provenance, {})

    def test_handler_build_prompts_failure_is_caught(self):
        # Defensive: a handler raising during build_prompts must
        # not crash the entire capture. Resolver returns None
        # with a wrap_error stamped in provenance.
        pid = 9005
        _write_manifest(pid, "classification")
        rows = [{"text": "x", "label": "y"}]
        # Stub the handler factory to return one that raises.
        from app.services import eval_task_handler_service as _hndlr
        broken = MagicMock()
        broken.wraps_own_prompt = lambda: True
        broken.build_prompts = MagicMock(side_effect=RuntimeError("kaboom"))
        broken.profile_id = "classification"
        with patch.object(_hndlr, "resolve_task_handler", return_value=broken):
            wrapped, provenance = _resolve_handler_wrapped_prompts(pid, rows)
        self.assertIsNone(wrapped)
        self.assertEqual(provenance.get("wrap_error"), "build_prompts_failed")


class ByteForByteCapturePromptVsHandlerTests(unittest.TestCase):
    """The load-bearing pin: the wrapped string the capture
    persists matches what ``ClassificationHandler._build_prompt_text``
    produces for the same input. A drift here would silently
    re-introduce the SQLi-shaped gap for KD."""

    @classmethod
    def setUpClass(cls):
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()

    def test_capture_wrap_matches_handler_build_prompt_text(self):
        from app.services.eval_task_handler_service import (
            ClassificationHandler,
        )
        pid = 9101
        _write_manifest(pid, "classification")
        rows = [{"text": "hello world", "label": "spam"}]
        wrapped, _ = _resolve_handler_wrapped_prompts(pid, rows)
        self.assertIsNotNone(wrapped)
        # Build what the eval handler would emit for the same
        # row. ClassificationHandler picks candidates from the
        # row's labels — for a single-row capture, the candidate
        # set is just {spam}.
        handler = ClassificationHandler()
        expected = handler._build_prompt_text("hello world", ["spam"])
        self.assertEqual(wrapped[0], expected)


class KdCaptureExtractPromptTextPreferenceTests(unittest.TestCase):
    """The KD record builder side — reads ``wrapped_prompt``
    preferentially so the student trains on what the teacher saw
    (which is what the eval handler will rebuild). Legacy
    fallback for pre-β-KD captures preserved."""

    def test_prefers_wrapped_prompt_when_present(self):
        row = {
            "text": "raw text",
            "question": "ignored raw question",
            "wrapped_prompt": "Classify the following text. …\nText: raw text\nLabel:",
        }
        self.assertEqual(
            _extract_prompt_text(row),
            "Classify the following text. …\nText: raw text\nLabel:",
        )

    def test_falls_back_to_question_when_no_wrapped_prompt(self):
        # Legacy capture — no wrapped_prompt field. KD record
        # builder reads question via the existing alias walk.
        row = {"question": "what is 2+2?"}
        self.assertEqual(_extract_prompt_text(row), "what is 2+2?")

    def test_skips_empty_wrapped_prompt_and_falls_back(self):
        # Defensive: a wrapped_prompt that's empty/whitespace
        # shouldn't shadow the raw alias walk.
        row = {"wrapped_prompt": "   ", "text": "real"}
        self.assertEqual(_extract_prompt_text(row), "real")

    def test_non_string_wrapped_prompt_falls_back(self):
        row = {"wrapped_prompt": 42, "text": "real"}
        self.assertEqual(_extract_prompt_text(row), "real")


class BuildOfflineKdRecordsPromptTransformTests(unittest.TestCase):
    """Chat-template sub-gap fix (Arc 1, item 1) —
    ``build_offline_kd_records`` accepts an optional
    ``prompt_transform`` that wraps the row's prompt before
    tokenization. Used by the trainer to apply
    ``tokenizer.apply_chat_template`` for QA-family KD projects
    so the student trains on the same chat-template-wrapped
    scaffold the eval will build.

    Pins:
      * Transform is applied when the row has no
        ``wrapped_prompt`` (capture-time wrap absent → student
        needs the chat-template wrap to match eval).
      * Transform is SKIPPED when the row already has a
        ``wrapped_prompt`` (capture-time wrap present →
        already byte-aligned to a wraps-own-prompt handler;
        applying chat template again would double-wrap).
      * Transform = None preserves legacy behaviour (raw prompt
        tokenization).
      * A throwing transform doesn't crash the build — falls
        back to the untransformed prompt for that row.
    """

    def _capture_row(
        self,
        *,
        prompt: str = "what is 2+2?",
        completion: str = "4",
        wrapped: str | None = None,
        teacher_positions: int = 1,
    ) -> dict:
        # Minimal capture-row shape the builder needs.
        row: dict = {
            "question": prompt,
            "teacher_completion": completion,
            # One teacher position per token for simplicity — the
            # builder requires at least one position to keep the
            # row.
            "teacher_logits": [
                {"position": p, "top_k": [["x", -0.1]]}
                for p in range(teacher_positions)
            ],
        }
        if wrapped is not None:
            row["wrapped_prompt"] = wrapped
        return row

    @staticmethod
    def _fake_encoder():
        # Map each token in the input to a fake numeric id. Use
        # split() so we can write tests that look at exact token
        # counts.
        vocab: dict[str, int] = {"<unk>": 0}

        def encode(text: str) -> list[int]:
            ids: list[int] = []
            for tok in text.split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
                ids.append(vocab[tok])
            return ids

        def token_to_id(token: str):
            return vocab.get(token, 0)

        return encode, token_to_id

    def test_transform_applied_when_no_wrapped_prompt(self):
        from app.services.distillation.kd_capture import (
            build_offline_kd_records,
        )
        encode, token_to_id = self._fake_encoder()
        # The chat-template-shaped wrap: prepend two prefix tokens
        # the model expects (representing the user/assistant
        # turn structure).
        def _transform(p: str) -> str:
            return f"<|user|> {p} <|assistant|>"
        row = self._capture_row(
            prompt="hello world",
            completion="hi",
        )
        records, _ = build_offline_kd_records(
            [row], encode, token_to_id,
            top_k=1, max_seq_length=64,
            prompt_transform=_transform,
        )
        self.assertEqual(len(records), 1)
        # Prompt tokens are masked in labels. Count them via the
        # label prefix length. After the transform, "hello world"
        # has 4 tokens (<|user|>, hello, world, <|assistant|>) +
        # untransformed it'd have 2 (hello, world).
        ignore = -100
        masked_prefix = next(
            (i for i, lab in enumerate(records[0]["labels"]) if lab != ignore),
            len(records[0]["labels"]),
        )
        self.assertEqual(masked_prefix, 4)

    def test_transform_skipped_when_wrapped_prompt_present(self):
        # The β-fix path already populated ``wrapped_prompt`` from
        # the capture (handler.build_prompts ran at capture time).
        # The chat-template transform must NOT fire — else the
        # wrapped prompt would get double-wrapped and diverge
        # from what the eval handler rebuilds at inference.
        from app.services.distillation.kd_capture import (
            build_offline_kd_records,
        )
        encode, token_to_id = self._fake_encoder()
        called: list[str] = []
        def _transform(p: str) -> str:
            called.append(p)
            return f"<|user|> {p} <|assistant|>"
        row = self._capture_row(
            prompt="what is 2+2?",
            completion="4",
            wrapped="Classify the following text. …\nText: 2+2\nLabel:",
        )
        records, _ = build_offline_kd_records(
            [row], encode, token_to_id,
            top_k=1, max_seq_length=64,
            prompt_transform=_transform,
        )
        # Transform never called → no double-wrap.
        self.assertEqual(called, [])
        # Prompt tokens come from the wrapped_prompt (which
        # ``_extract_prompt_text`` already prefers over the raw
        # question field).
        ignore = -100
        masked_prefix = next(
            (i for i, lab in enumerate(records[0]["labels"]) if lab != ignore),
            len(records[0]["labels"]),
        )
        # ``Classify the following text. …\nText: 2+2\nLabel:``
        # split() gives 7 tokens.
        self.assertGreater(masked_prefix, 4)

    def test_no_transform_preserves_legacy_behaviour(self):
        # Regression guard for the existing trainer call sites
        # (and any test fixture) that didn't previously pass
        # prompt_transform. The default ``None`` keeps raw
        # tokenization.
        from app.services.distillation.kd_capture import (
            build_offline_kd_records,
        )
        encode, token_to_id = self._fake_encoder()
        row = self._capture_row(prompt="alpha beta", completion="gamma")
        records, _ = build_offline_kd_records(
            [row], encode, token_to_id,
            top_k=1, max_seq_length=64,
        )
        ignore = -100
        masked_prefix = next(
            (i for i, lab in enumerate(records[0]["labels"]) if lab != ignore),
            len(records[0]["labels"]),
        )
        # Raw "alpha beta" → 2 tokens, no chat-template prefix.
        self.assertEqual(masked_prefix, 2)

    def test_throwing_transform_falls_back_to_raw_prompt(self):
        # Defensive: a buggy transform must not crash the entire
        # build. Same shape as the capture-time handler-failure
        # tolerance.
        from app.services.distillation.kd_capture import (
            build_offline_kd_records,
        )
        encode, token_to_id = self._fake_encoder()
        def _transform(p: str) -> str:
            raise RuntimeError("kaboom")
        row = self._capture_row(prompt="alpha beta", completion="gamma")
        records, _ = build_offline_kd_records(
            [row], encode, token_to_id,
            top_k=1, max_seq_length=64,
            prompt_transform=_transform,
        )
        self.assertEqual(len(records), 1)
        # The build succeeded with the raw-prompt fallback.
        ignore = -100
        masked_prefix = next(
            (i for i, lab in enumerate(records[0]["labels"]) if lab != ignore),
            len(records[0]["labels"]),
        )
        self.assertEqual(masked_prefix, 2)

    def test_empty_transform_output_falls_back_to_raw(self):
        # Transform returns empty/whitespace → use the raw prompt
        # rather than train on an empty prompt block (which would
        # produce a record with zero prompt tokens, breaking
        # downstream loss masking).
        from app.services.distillation.kd_capture import (
            build_offline_kd_records,
        )
        encode, token_to_id = self._fake_encoder()
        row = self._capture_row(prompt="alpha beta", completion="gamma")
        records, _ = build_offline_kd_records(
            [row], encode, token_to_id,
            top_k=1, max_seq_length=64,
            prompt_transform=lambda _p: "   ",
        )
        ignore = -100
        masked_prefix = next(
            (i for i, lab in enumerate(records[0]["labels"]) if lab != ignore),
            len(records[0]["labels"]),
        )
        self.assertEqual(masked_prefix, 2)


if __name__ == "__main__":
    unittest.main()

"""Tests for the shared subtask-propagation infrastructure.

The 3 remaining β-shape gaps from the cross-task audit
(vision-language-pair, audio-transcript, seq2seq-pair) all branch
per-subtask at eval time — the handler picks one of
``{captioning, vqa}`` / ``{transcription, audio_qa}`` /
``{translation, summarization, paraphrase}`` from the manifest.
Adapters need the same signal at row-mapping time so they wrap
``source_text`` with the matching prompt shape.

``_resolve_adapter_subtask`` is the shared resolver. Pins:
  1. Returns ``None`` for adapters NOT in
     ``_ADAPTER_SUBTASK_SPECS`` — caller uses ``None`` to skip
     injection entirely (classification / structured / rag don't
     branch per-subtask).
  2. Resolution precedence: adapter_config override → manifest
     field → manifest output_schema.subtask → per-adapter
     default. Invalid values fall through rather than failing
     loudly.
  3. Allowed sets + defaults match the handler's ``SUBTASK_*``
     constants byte-for-byte. A drift here would silently
     re-introduce the train/eval mismatch β documented.
  4. ``_normalize_rows_for_training`` injects the resolved
     subtask into ``adapter_config`` for the right adapters
     before the per-row map loop. Non-subtask-aware adapters
     don't get an injection.
"""

from __future__ import annotations

import unittest

from app.services.dataset_service import (
    _ADAPTER_SUBTASK_SPECS,
    _resolve_adapter_subtask,
)


class ResolverPrecedenceTests(unittest.TestCase):
    """The four-tier precedence ladder, exercised one rung at a
    time. Each test isolates a single rung by null-ing out the
    higher-priority sources."""

    def test_adapter_config_override_wins_over_manifest(self):
        # Rung 1: caller-injected subtask trumps manifest.
        out = _resolve_adapter_subtask(
            "vision-language-pair",
            manifest={"subtask": "captioning"},
            adapter_config={"subtask": "vqa"},
        )
        self.assertEqual(out, "vqa")

    def test_manifest_subtask_wins_over_default(self):
        # Rung 2: manifest.subtask is the canonical source the
        # handler reads — the adapter must agree byte-for-byte.
        out = _resolve_adapter_subtask(
            "audio-transcript",
            manifest={"subtask": "audio_qa"},
            adapter_config=None,
        )
        self.assertEqual(out, "audio_qa")

    def test_manifest_output_schema_subtask_accepted(self):
        # Rung 3: some manifests nest subtask under output_schema.
        # The handler reads both shapes; the adapter must too.
        out = _resolve_adapter_subtask(
            "seq2seq-pair",
            manifest={"output_schema": {"subtask": "translation"}},
            adapter_config=None,
        )
        self.assertEqual(out, "translation")

    def test_per_adapter_default_when_no_source_carries_subtask(self):
        # Rung 4: handler's DEFAULT_SUBTASK is the safety net.
        # Vision-language: captioning. Audio: transcription.
        # Seq2seq: summarization.
        self.assertEqual(
            _resolve_adapter_subtask(
                "vision-language-pair", manifest=None, adapter_config=None,
            ),
            "captioning",
        )
        self.assertEqual(
            _resolve_adapter_subtask(
                "audio-transcript", manifest={}, adapter_config={},
            ),
            "transcription",
        )
        self.assertEqual(
            _resolve_adapter_subtask(
                "seq2seq-pair", manifest=None, adapter_config=None,
            ),
            "summarization",
        )

    def test_unknown_adapter_returns_none(self):
        # Rung 5: adapters NOT in the spec table (classification,
        # structured, rag) get ``None`` so the injection step at
        # the call site skips them cleanly.
        for adapter_id in (
            "classification-label",
            "structured-extraction",
            "rag-grounded",
            "qa-pair",
            "default-canonical",
            "non-existent",
        ):
            self.assertIsNone(
                _resolve_adapter_subtask(adapter_id, manifest=None, adapter_config=None),
                f"adapter_id={adapter_id} should return None",
            )


class ResolverInvalidValueFallthroughTests(unittest.TestCase):
    """Invalid subtask values at higher-priority rungs MUST fall
    through to lower-priority rungs (don't raise). A power user
    typo or a legacy adapter_config shouldn't break prep."""

    def test_invalid_adapter_config_subtask_falls_through_to_manifest(self):
        out = _resolve_adapter_subtask(
            "vision-language-pair",
            manifest={"subtask": "vqa"},
            adapter_config={"subtask": "not-a-real-subtask"},
        )
        self.assertEqual(out, "vqa")

    def test_invalid_manifest_subtask_falls_through_to_default(self):
        out = _resolve_adapter_subtask(
            "audio-transcript",
            manifest={"subtask": "🐛bogus"},
            adapter_config=None,
        )
        self.assertEqual(out, "transcription")

    def test_non_string_adapter_config_subtask_falls_through(self):
        # Wrong type shouldn't crash the resolver.
        out = _resolve_adapter_subtask(
            "seq2seq-pair",
            manifest={"subtask": "paraphrase"},
            adapter_config={"subtask": 42},
        )
        self.assertEqual(out, "paraphrase")

    def test_resolver_normalizes_whitespace_and_case(self):
        # The handler ``str(...).strip().lower()`` — adapter resolver
        # must do the same so adapter_config + manifest can carry
        # ``"Translation"`` / ``" translation "`` and still match.
        out = _resolve_adapter_subtask(
            "seq2seq-pair",
            manifest={"subtask": "  TRANSLATION  "},
            adapter_config=None,
        )
        self.assertEqual(out, "translation")


class ResolverHandlerParityTests(unittest.TestCase):
    """The allowed sets and defaults must mirror the handler's
    constants byte-for-byte. A drift here would silently
    re-introduce the train/eval mismatch the audit closed —
    pin equality so a future refactor can't drift one side
    without the other."""

    def test_vision_language_spec_matches_handler_constants(self):
        from app.services.eval_task_handler_service import VisionLanguageHandler
        spec = _ADAPTER_SUBTASK_SPECS["vision-language-pair"]
        self.assertEqual(
            spec["allowed"], frozenset(VisionLanguageHandler._SUPPORTED_SUBTASKS),
        )
        self.assertEqual(spec["default"], VisionLanguageHandler.DEFAULT_SUBTASK)

    def test_audio_transcript_spec_matches_handler_constants(self):
        from app.services.eval_task_handler_service import AudioTranscriptHandler
        spec = _ADAPTER_SUBTASK_SPECS["audio-transcript"]
        self.assertEqual(
            spec["allowed"], frozenset(AudioTranscriptHandler._SUPPORTED_SUBTASKS),
        )
        self.assertEqual(spec["default"], AudioTranscriptHandler.DEFAULT_SUBTASK)

    def test_seq2seq_pair_spec_matches_handler_constants(self):
        from app.services.eval_task_handler_service import Seq2SeqHandler
        spec = _ADAPTER_SUBTASK_SPECS["seq2seq-pair"]
        self.assertEqual(
            spec["allowed"], frozenset(Seq2SeqHandler._SUPPORTED_SUBTASKS),
        )
        self.assertEqual(spec["default"], Seq2SeqHandler.DEFAULT_SUBTASK)


class NormalizeRowsInjectionTests(unittest.TestCase):
    """End-to-end: ``_normalize_rows_for_training`` injects subtask
    into adapter_config for the right adapters before the per-row
    map loop. We pass a stub adapter via ``adapter_config`` and a
    trace dict that the stub writes to, so we can assert the
    config seen by the adapter carries the resolved subtask."""

    def test_vision_pair_injection_uses_default_when_no_manifest(self):
        # Inject via a patched map function so we don't have to
        # build a real vision-language record. Read the seen
        # adapter_config back via a side-channel.
        from unittest.mock import patch
        from app.services.dataset_service import _normalize_rows_for_training
        from app.models.dataset import DatasetType

        seen_configs: list[dict] = []

        def _trace_map(record, adapter_id, adapter_config, field_mapping, task_profile):
            seen_configs.append(dict(adapter_config or {}))
            return {"text": "x", "source_text": "x", "target_text": "y"}

        with patch(
            "app.services.dataset_service.map_record_with_adapter",
            side_effect=_trace_map,
        ), patch(
            "app.services.dataset_service.resolve_data_adapter_for_records",
            return_value=("vision-language-pair", None),
        ), patch(
            "app.services.dataset_service.resolve_task_profile_for_adapter",
            return_value="vision_language",
        ):
            _normalize_rows_for_training(
                [{"text": "x"}],
                DatasetType.SYNTHETIC,
                chat_template="chatml",
                adapter_id="vision-language-pair",
            )
        self.assertEqual(len(seen_configs), 1)
        # No manifest, no override → DEFAULT_SUBTASK = "captioning".
        self.assertEqual(seen_configs[0].get("subtask"), "captioning")

    def test_manifest_subtask_propagates_through_normalize(self):
        from unittest.mock import patch
        from app.services.dataset_service import _normalize_rows_for_training
        from app.models.dataset import DatasetType

        seen_configs: list[dict] = []

        def _trace_map(record, adapter_id, adapter_config, field_mapping, task_profile):
            seen_configs.append(dict(adapter_config or {}))
            return {"text": "x", "source_text": "x", "target_text": "y"}

        with patch(
            "app.services.dataset_service.map_record_with_adapter",
            side_effect=_trace_map,
        ), patch(
            "app.services.dataset_service.resolve_data_adapter_for_records",
            return_value=("audio-transcript", None),
        ), patch(
            "app.services.dataset_service.resolve_task_profile_for_adapter",
            return_value="audio_transcript",
        ):
            _normalize_rows_for_training(
                [{"text": "x"}],
                DatasetType.SYNTHETIC,
                chat_template="chatml",
                adapter_id="audio-transcript",
                manifest={"subtask": "audio_qa"},
            )
        # Manifest's ``audio_qa`` overrides the handler default
        # (``transcription``).
        self.assertEqual(seen_configs[0].get("subtask"), "audio_qa")

    def test_classification_adapter_gets_no_subtask_injection(self):
        # Regression guard: subtask injection must not leak into
        # adapters that don't branch per-subtask. classification-
        # label's adapter_config carries ``candidates`` from the
        # β pre-scan and nothing else from this code path.
        from unittest.mock import patch
        from app.services.dataset_service import _normalize_rows_for_training
        from app.models.dataset import DatasetType

        seen_configs: list[dict] = []

        def _trace_map(record, adapter_id, adapter_config, field_mapping, task_profile):
            seen_configs.append(dict(adapter_config or {}))
            return {"text": "x", "source_text": "x", "target_text": " y"}

        with patch(
            "app.services.dataset_service.map_record_with_adapter",
            side_effect=_trace_map,
        ), patch(
            "app.services.dataset_service.resolve_data_adapter_for_records",
            return_value=("classification-label", None),
        ), patch(
            "app.services.dataset_service.resolve_task_profile_for_adapter",
            return_value="classification",
        ):
            _normalize_rows_for_training(
                [{"text": "x", "label": "y"}],
                DatasetType.SYNTHETIC,
                chat_template="chatml",
                adapter_id="classification-label",
                manifest={"subtask": "captioning"},  # bogus, must not leak
            )
        self.assertNotIn("subtask", seen_configs[0])

    def test_caller_subtask_in_adapter_config_preserved(self):
        # Caller-provided subtask shouldn't be clobbered by the
        # resolver. We pass a non-default subtask in
        # adapter_config and assert it survives.
        from unittest.mock import patch
        from app.services.dataset_service import _normalize_rows_for_training
        from app.models.dataset import DatasetType

        seen_configs: list[dict] = []

        def _trace_map(record, adapter_id, adapter_config, field_mapping, task_profile):
            seen_configs.append(dict(adapter_config or {}))
            return {"text": "x", "source_text": "x", "target_text": "y"}

        with patch(
            "app.services.dataset_service.map_record_with_adapter",
            side_effect=_trace_map,
        ), patch(
            "app.services.dataset_service.resolve_data_adapter_for_records",
            return_value=("seq2seq-pair", None),
        ), patch(
            "app.services.dataset_service.resolve_task_profile_for_adapter",
            return_value="seq2seq",
        ):
            _normalize_rows_for_training(
                [{"text": "x"}],
                DatasetType.SYNTHETIC,
                chat_template="chatml",
                adapter_id="seq2seq-pair",
                adapter_config={"subtask": "paraphrase"},
                manifest={"subtask": "translation"},
            )
        # Caller's "paraphrase" wins over manifest's "translation".
        self.assertEqual(seen_configs[0].get("subtask"), "paraphrase")


if __name__ == "__main__":
    unittest.main()

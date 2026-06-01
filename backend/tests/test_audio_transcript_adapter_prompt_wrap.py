"""κ-fix tests — audio-transcript adapter writes the production
prompt format into training rows.

Mirror of ι (vision-language). Pre-κ ``_map_audio_transcript``
wrote ``source_text = "audio:{path}"`` (bare text, no
``<audio:…>`` placeholder, no instruction, no ``Transcript:``
cue) and ``text = "<audio:{path}> {transcript}"``. The adapter
also only extracted the transcript — audio_qa rows had a
``question`` field that the adapter never surfaced.
``AudioTranscriptHandler.build_prompts`` at eval time wraps inputs
with one of two subtask scaffolds (``"Transcribe the audio:
<audio:…>\\nTranscript:"`` for transcription,
``"Question: …\\nAudio: <audio:…>\\nAnswer:"`` for audio_qa), so
held-out WER/CER on audio projects came in artificially high.

Post-κ (this commit's tests pin):

  1. ``_build_audio_transcript_training_prompt`` and
     ``AudioTranscriptHandler.build_prompts`` produce IDENTICAL
     strings byte-for-byte under both subtask branches (with +
     without audio_path).
  2. Adapter reads ``subtask`` from ``adapter_config``. Missing /
     invalid → ``transcription`` (handler's ``DEFAULT_SUBTASK``).
  3. audio_qa rows extract ``question`` separately (pre-κ the
     adapter only picked the transcript/answer).
  4. ``target_text = f" {transcript}"`` (leading space — same
     trick as β/ζ/η/θ/ι).
  5. Raw ``audio_path`` / ``answer`` / ``question`` preserved.
  6. ``scripts/train.py:_adapt_record_to_text`` passes
     audio-wrapped ``source_text`` through untouched, with the
     ι-style tight VQA-equivalent signal
     (``\\nAudio: <audio:`` marker).
"""

from __future__ import annotations

import unittest

from app.services.data_adapter_service import (
    _build_audio_transcript_training_prompt,
    _map_audio_transcript,
)


class AudioTranscriptAdapterWrapTests(unittest.TestCase):
    def test_transcription_default_with_audio_path(self):
        # Missing subtask → handler's DEFAULT_SUBTASK ("transcription").
        out = _map_audio_transcript(
            {"audio_path": "audio/a.wav", "transcript": "hello world"},
            {},
        )
        assert out is not None
        self.assertEqual(
            out["source_text"],
            "Transcribe the audio: <audio:audio/a.wav>\nTranscript:",
        )
        self.assertEqual(out["target_text"], " hello world")

    def test_transcription_text_field_byte_for_byte(self):
        # Pre-κ ``text`` was ``"<audio:{path}> {transcript}"`` — no
        # instruction. κ aligns ``text`` with wrapped prompt +
        # target so trainer paths consuming ``text`` directly see
        # the same scaffold the handler will rebuild at eval.
        out = _map_audio_transcript(
            {"audio_path": "audio/b.wav", "transcript": "the quick brown fox"},
            {},
        )
        assert out is not None
        self.assertEqual(out["text"], f"{out['source_text']}{out['target_text']}")
        self.assertFalse(out["text"].startswith("<audio:"))

    def test_audio_qa_extracts_question_from_row(self):
        # The audio mirror of ι's load-bearing VQA fix: audio_qa
        # rows have a question that pre-κ adapter never picked
        # up. κ reads ``question`` (or its aliases) and stitches
        # into the handler's scaffold.
        out = _map_audio_transcript(
            {
                "audio_path": "audio/c.wav",
                "question": "Who is speaking?",
                "answer": "A child.",
            },
            {"subtask": "audio_qa"},
        )
        assert out is not None
        self.assertEqual(
            out["source_text"],
            "Question: Who is speaking?\n"
            "Audio: <audio:audio/c.wav>\n"
            "Answer:",
        )
        self.assertEqual(out["target_text"], " A child.")
        self.assertEqual(out["question"], "Who is speaking?")

    def test_audio_qa_question_alias_accepted(self):
        # ``prompt`` is an alias for ``question`` (matches the
        # handler's _extract_question aliases).
        out = _map_audio_transcript(
            {
                "audio_path": "audio/d.wav",
                "prompt": "What language?",
                "answer": "French.",
            },
            {"subtask": "audio_qa"},
        )
        assert out is not None
        self.assertIn("Question: What language?", out["source_text"])

    def test_target_text_has_leading_space(self):
        out = _map_audio_transcript(
            {"audio_path": "audio/e.wav", "transcript": "T"},
            {"subtask": "transcription"},
        )
        assert out is not None
        self.assertTrue(out["target_text"].startswith(" "))

    def test_raw_fields_preserved_for_downstream(self):
        out = _map_audio_transcript(
            {
                "audio_path": "audio/f.wav",
                "question": "raw q",
                "answer": "raw a",
            },
            {"subtask": "audio_qa"},
        )
        assert out is not None
        self.assertEqual(out["audio_path"], "audio/f.wav")
        self.assertEqual(out["question"], "raw q")
        self.assertEqual(out["answer"], "raw a")

    def test_invalid_subtask_falls_back_to_transcription(self):
        out = _map_audio_transcript(
            {"audio_path": "audio/g.wav", "transcript": "Stuff."},
            {"subtask": "🐛bogus"},
        )
        assert out is not None
        self.assertTrue(
            out["source_text"].startswith("Transcribe the audio:")
        )

    def test_record_missing_audio_path_returns_none(self):
        self.assertIsNone(
            _map_audio_transcript({"transcript": "no audio"}, {})
        )

    def test_record_missing_transcript_returns_none(self):
        self.assertIsNone(
            _map_audio_transcript({"audio_path": "audio/x.wav"}, {})
        )


class AudioTranscriptHandlerByteForByteTests(unittest.TestCase):
    """Pin equality with the handler's actual wrap. Same shape as
    ι's parity tests, swap image→audio."""

    def test_transcription_with_audio_matches_handler(self):
        adapter = _build_audio_transcript_training_prompt(
            "audio/x.wav", "transcription", "ignored-for-transcription",
        )
        expected = "Transcribe the audio: <audio:audio/x.wav>\nTranscript:"
        self.assertEqual(adapter, expected)

    def test_transcription_without_audio_matches_handler_fallback(self):
        # Handler's transcription-no-audio branch: just the
        # instruction (no audio token, no Transcript: cue).
        adapter = _build_audio_transcript_training_prompt(
            "", "transcription", "ignored",
        )
        self.assertEqual(adapter, "Transcribe the audio:")

    def test_audio_qa_with_audio_matches_handler(self):
        adapter = _build_audio_transcript_training_prompt(
            "audio/y.wav", "audio_qa", "What is said?",
        )
        expected = (
            "Question: What is said?\n"
            "Audio: <audio:audio/y.wav>\n"
            "Answer:"
        )
        self.assertEqual(adapter, expected)

    def test_audio_qa_without_audio_matches_handler_fallback(self):
        # Handler's audio_qa-no-audio branch: just the question.
        adapter = _build_audio_transcript_training_prompt(
            "", "audio_qa", "What is said?",
        )
        self.assertEqual(adapter, "What is said?")

    def test_adapter_source_text_carries_handler_expected_prefixes(self):
        # γ′ smoke check / row peek looks for any of the handler's
        # declared prefixes in the prepared row.
        from app.services.eval_task_handler_service import AudioTranscriptHandler
        prefixes = AudioTranscriptHandler().expected_prompt_prefixes()
        for subtask, fixture in (
            ("transcription", {"audio_path": "audio/p.wav", "transcript": "T"}),
            ("audio_qa", {
                "audio_path": "audio/q.wav",
                "question": "Who?",
                "answer": "Alice.",
            }),
        ):
            out = _map_audio_transcript(fixture, {"subtask": subtask})
            assert out is not None
            self.assertTrue(
                any(p in out["source_text"] for p in prefixes),
                f"subtask={subtask} prefixes={prefixes!r} "
                f"source_text={out['source_text']!r}",
            )


class AudioAdaptRecordPassthroughTests(unittest.TestCase):
    def test_transcription_wrap_passes_through_untouched(self):
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        wrapped = "Transcribe the audio: <audio:audio/a.wav>\nTranscript:"
        adapted = train_script._adapt_record_to_text(
            {
                "audio_path": "audio/a.wav",
                "answer": "hello",
                "text": f"{wrapped} hello",
                "source_text": wrapped,
                "target_text": " hello",
            },
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], wrapped)
        self.assertEqual(adapted["target_text"], " hello")

    def test_audio_qa_wrap_passes_through_untouched(self):
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        wrapped = (
            "Question: Who is speaking?\n"
            "Audio: <audio:audio/b.wav>\n"
            "Answer:"
        )
        adapted = train_script._adapt_record_to_text(
            {
                "audio_path": "audio/b.wav",
                "question": "Who is speaking?",
                "answer": "A child.",
                "text": f"{wrapped} A child.",
                "source_text": wrapped,
                "target_text": " A child.",
            },
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], wrapped)

    def test_plain_question_qa_row_not_misdetected_as_audio_qa(self):
        # Mirror of ι's regression guard: a plain Q/A row (no
        # ``\\nAudio: <audio:`` marker) must not be misrouted
        # through κ-tail. Same shape as vision regression guard.
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        adapted = train_script._adapt_record_to_text(
            {"question": "What is 2+2?", "answer": "4"},
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], "What is 2+2?")
        self.assertEqual(adapted["target_text"], "4")

    def test_vision_vqa_row_not_misdetected_as_audio_qa(self):
        # Adversarial regression guard: an ι-vision VQA row
        # carries the ``\\nImage: <image:…>`` marker, NOT
        # ``\\nAudio: <audio:…>``. The κ-tail signal must not
        # fire for it (otherwise the vision tail's text-shape
        # rendering would be silently overwritten by κ's
        # pass-through reconstruction).
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        vl_wrapped = (
            "Question: What is shown?\n"
            "Image: <image:imgs/x.jpg>\n"
            "Answer:"
        )
        adapted = train_script._adapt_record_to_text(
            {
                "image_path": "imgs/x.jpg",
                "question": "What is shown?",
                "answer": "A bird.",
                "text": f"{vl_wrapped} A bird.",
                "source_text": vl_wrapped,
                "target_text": " A bird.",
            },
            contract,
            "chatml",
        )
        # ι's vision-VQA tail handles this row — source_text passes
        # through. κ-tail must not have touched it.
        self.assertEqual(adapted["source_text"], vl_wrapped)
        self.assertIn("<image:", adapted["source_text"])
        self.assertNotIn("<audio:", adapted["source_text"])


if __name__ == "__main__":
    unittest.main()

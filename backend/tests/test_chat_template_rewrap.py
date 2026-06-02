"""Tests for the opt-in chat-template re-wrap pass
(``use_tokenizer_chat_template`` config flag).

Closes the residual train/eval format gap for QA-family rows
that the live-teacher KD audit surfaced: the hardcoded
``_qa_to_chat_text`` heuristic format differs at the byte level
from what ``tokenizer.apply_chat_template`` emits at eval. When
the flag is on, training rows for non-wraps_own_prompt handlers
get re-shaped via ``apply_chat_template`` so the student trains
on the byte-identical scaffold the eval will build.

Pins:
  * ``_is_wraps_own_prompt_row`` correctly detects β/ζ/η/θ/ι/κ
    adapter-wrap output via source_text prefix — the load-bearing
    distinguisher between "already byte-aligned" and "needs
    chat-template re-wrap."
  * Multimodal VQA rows (Question:…\nImage: <image:…> / Audio
    equivalent) are detected via the placeholder marker rather
    than just "Question:" prefix — mirrors ι/κ's tail detector.
  * Plain QA rows aren't classified as wraps_own_prompt.

The actual re-wrap pass is exercised end-to-end at runtime by the
trainer; pinning the helper here protects the load-bearing
classification logic.
"""

from __future__ import annotations

import unittest

from scripts.train import (
    _WRAPS_OWN_PROMPT_PREFIXES,
    _is_wraps_own_prompt_row,
)


class WrapsOwnPromptDetectionTests(unittest.TestCase):
    def test_classification_wrap_detected(self):
        # β adapter (classification-label) writes source_text
        # starting with this prefix. Re-wrap must skip — already
        # byte-aligned to ClassificationHandler.
        row = {
            "source_text": (
                "Classify the following text. Reply with exactly one of: "
                "spam, ham.\nText: hello\nLabel:"
            ),
            "target_text": " spam",
        }
        self.assertTrue(_is_wraps_own_prompt_row(row))

    def test_structured_extraction_wrap_detected(self):
        # ζ adapter — two possible prefixes (list vs no-list
        # branches). Both must be recognized.
        for prefix in (
            "Extract the following fields as JSON: name, age.\n"
            "Reply with a single JSON object, nothing else.\n"
            "Input: x\nOutput:",
            "Extract the relevant fields from the input as a single "
            "JSON object, nothing else.\nInput: x\nOutput:",
        ):
            self.assertTrue(
                _is_wraps_own_prompt_row(
                    {"source_text": prefix, "target_text": "x"},
                ),
                f"prefix={prefix[:40]!r}",
            )

    def test_rag_wrap_detected(self):
        # η adapter — grounded rag prompt.
        row = {
            "source_text": (
                "Answer the question using only the context. If the "
                "context does not contain the answer, say you don't "
                "know.\nContext: …\nQuestion: x\nAnswer:"
            ),
            "target_text": " y",
        }
        self.assertTrue(_is_wraps_own_prompt_row(row))

    def test_seq2seq_three_subtasks_all_detected(self):
        # θ adapter — three subtasks, each with its own prefix.
        for prefix in (
            "Translate the following to French.\nText: hi\nTranslation:",
            "Summarize the following text concisely.\nText: long\nSummary:",
            "Paraphrase the following text in different words.\n"
            "Text: x\nParaphrase:",
        ):
            self.assertTrue(
                _is_wraps_own_prompt_row(
                    {"source_text": prefix, "target_text": "x"},
                )
            )

    def test_vision_captioning_detected(self):
        # ι adapter — captioning branch starts with
        # "Describe the image:".
        row = {
            "source_text": "Describe the image: <image:x.jpg>\nCaption:",
            "target_text": " a cat",
        }
        self.assertTrue(_is_wraps_own_prompt_row(row))

    def test_audio_transcription_detected(self):
        # κ adapter — transcription branch starts with
        # "Transcribe the audio:".
        row = {
            "source_text": "Transcribe the audio: <audio:x.wav>\nTranscript:",
            "target_text": " hello",
        }
        self.assertTrue(_is_wraps_own_prompt_row(row))

    def test_vqa_marker_detected(self):
        # ι adapter — VQA branch starts with "Question:" AND
        # carries an "\nImage: <image:" marker. The marker is the
        # disambiguator vs plain QA "Question:" rows.
        row = {
            "source_text": (
                "Question: what's shown?\nImage: <image:cat.jpg>\nAnswer:"
            ),
            "target_text": " a cat",
        }
        self.assertTrue(_is_wraps_own_prompt_row(row))

    def test_audio_qa_marker_detected(self):
        # κ adapter — audio_qa branch.
        row = {
            "source_text": (
                "Question: who?\nAudio: <audio:speech.wav>\nAnswer:"
            ),
            "target_text": " Alice",
        }
        self.assertTrue(_is_wraps_own_prompt_row(row))

    def test_plain_qa_row_not_detected(self):
        # QA-family rows (qa-pair adapter, _qa_to_chat_text path)
        # start with whatever the row's question text is — they
        # must NOT match the wraps_own_prompt prefixes so the
        # re-wrap pass fires on them.
        for raw in (
            "what is 2+2?",
            "Hello world",
            "Tell me about Python.",
        ):
            self.assertFalse(
                _is_wraps_own_prompt_row(
                    {"source_text": raw, "target_text": "x"},
                ),
                f"raw={raw!r}",
            )

    def test_plain_question_prefixed_but_no_multimodal_marker_not_detected(self):
        # A QA row whose question text happens to start with
        # "Question:" (e.g., a row imported from a Q/A dataset
        # with that literal prefix in the field) must NOT be
        # misclassified as VQA/audio_qa — the marker check
        # prevents that.
        row = {
            "source_text": "Question: what is 2+2?",
            "target_text": "4",
        }
        self.assertFalse(_is_wraps_own_prompt_row(row))

    def test_non_dict_source_text_not_detected(self):
        # Defensive: a row whose source_text isn't a string
        # (None, int, etc.) doesn't crash the detector.
        for value in (None, 42, ["list"]):
            self.assertFalse(
                _is_wraps_own_prompt_row({"source_text": value}),
                f"value={value!r}",
            )

    def test_prefix_list_covers_every_canonical_adapter_output(self):
        # Audit-coverage guard via the adapters themselves: for
        # each β/ζ/η/θ/ι/κ adapter, build a sample row and confirm
        # _is_wraps_own_prompt_row classifies its source_text as
        # already-wrapped. The sweep test pins the (adapter, handler)
        # canonical pairs; this asserts the train.py rewrap pass
        # respects them.
        from app.services.data_adapter_service import (
            _build_audio_transcript_training_prompt,
            _build_classification_training_prompt,
            _build_rag_training_prompt,
            _build_seq2seq_training_prompt,
            _build_structured_extraction_training_prompt,
            _build_vision_language_training_prompt,
        )
        samples = [
            _build_classification_training_prompt("x", ["a", "b"]),
            _build_classification_training_prompt("x", None),
            _build_structured_extraction_training_prompt("x", ["a"]),
            _build_structured_extraction_training_prompt("x", None),
            _build_rag_training_prompt("ctx", "q"),
            _build_seq2seq_training_prompt("x", "translation", "French"),
            _build_seq2seq_training_prompt("x", "summarization", ""),
            _build_seq2seq_training_prompt("x", "paraphrase", ""),
            _build_vision_language_training_prompt("img.jpg", "captioning", ""),
            _build_vision_language_training_prompt("img.jpg", "vqa", "what?"),
            _build_audio_transcript_training_prompt("a.wav", "transcription", ""),
            _build_audio_transcript_training_prompt("a.wav", "audio_qa", "who?"),
        ]
        for src in samples:
            self.assertTrue(
                _is_wraps_own_prompt_row({"source_text": src}),
                f"adapter output not classified as wraps_own_prompt: "
                f"{src[:80]!r}",
            )


if __name__ == "__main__":
    unittest.main()

"""Phase 5.3.7 — Multimodal handlers (VisionLanguage + AudioTranscript).

Pins the contract:

- Dispatcher routes `vision_language` / `image_captioning` / `vqa` to
  VisionLanguageHandler. Routes `audio_transcript` /
  `audio_transcription` / `speech_to_text` to AudioTranscriptHandler.
- VisionLanguageHandler dispatches by manifest.subtask on
  {captioning, vqa}. Captioning emits BLEU-4 + ROUGE-L (via sacrebleu
  / rouge_score, both already in deps from Phase 5.3.3). VQA emits
  just EM/F1.
- AudioTranscriptHandler dispatches on {transcription, audio_qa}.
  Transcription emits WER + CER via jiwer. Audio QA emits EM/F1.
- Both handlers include image_path / audio_path tokens in the prompt
  so plain-text inference still has a paper trail of what was
  attached. Real multimodal inference passes the paths separately.
- Both handlers preserve EM/F1 (legacy gate compat) regardless of
  sub-task.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.eval_task_handler_service import (  # noqa: E402
    AudioTranscriptHandler,
    EvalContext,
    GenericHandler,
    VisionLanguageHandler,
    resolve_task_handler,
)


def _ctx_vl(subtask: str | None = None) -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile="vision_language",
        handler_id="vision_language",
        prepared_dir=Path("."),
        dataset_name="test",
        manifest={"subtask": subtask} if subtask else {},
    )


def _ctx_audio(subtask: str | None = None) -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile="audio_transcript",
        handler_id="audio_transcript",
        prepared_dir=Path("."),
        dataset_name="test",
        manifest={"subtask": subtask} if subtask else {},
    )


class DispatcherRoutingTests(unittest.TestCase):
    def test_vision_aliases_route_to_vl_handler(self):
        self.assertIsInstance(
            resolve_task_handler("vision_language"), VisionLanguageHandler
        )
        self.assertIsInstance(
            resolve_task_handler("image_captioning"), VisionLanguageHandler
        )
        self.assertIsInstance(resolve_task_handler("vqa"), VisionLanguageHandler)

    def test_audio_aliases_route_to_audio_handler(self):
        self.assertIsInstance(
            resolve_task_handler("audio_transcript"), AudioTranscriptHandler
        )
        self.assertIsInstance(
            resolve_task_handler("audio_transcription"), AudioTranscriptHandler
        )
        self.assertIsInstance(
            resolve_task_handler("speech_to_text"), AudioTranscriptHandler
        )

    def test_other_profiles_unaffected(self):
        self.assertIsInstance(resolve_task_handler(None), GenericHandler)
        self.assertIsInstance(
            resolve_task_handler("unknown_profile"), GenericHandler
        )


class VisionLanguageHandlerTests(unittest.TestCase):
    def test_captioning_default_subtask(self):
        h = VisionLanguageHandler()
        self.assertEqual(h._resolve_subtask(_ctx_vl()), "captioning")

    def test_explicit_vqa_subtask(self):
        h = VisionLanguageHandler()
        self.assertEqual(h._resolve_subtask(_ctx_vl("vqa")), "vqa")

    def test_captioning_prompt_references_image(self):
        h = VisionLanguageHandler()
        built = h.build_prompts(
            [
                {
                    "image_path": "assets/cat.png",
                    "caption": "A cat on a chair.",
                }
            ],
            _ctx_vl("captioning"),
        )
        self.assertIn("Describe the image", built[0].prompt)
        self.assertIn("assets/cat.png", built[0].prompt)
        self.assertEqual(built[0].reference, "A cat on a chair.")
        self.assertEqual(built[0].extras["vl_subtask"], "captioning")

    def test_vqa_prompt_includes_question_and_image(self):
        h = VisionLanguageHandler()
        built = h.build_prompts(
            [
                {
                    "image_path": "assets/cat.png",
                    "question": "What animal is in the image?",
                    "answer": "cat",
                }
            ],
            _ctx_vl("vqa"),
        )
        self.assertIn("Question: What animal is in the image?", built[0].prompt)
        self.assertIn("Image: <image:assets/cat.png>", built[0].prompt)
        self.assertEqual(built[0].reference, "cat")

    def test_captioning_scoring_produces_bleu_and_rouge(self):
        h = VisionLanguageHandler()
        predictions = [
            {
                "prediction": "A cat sitting on a chair.",
                "reference": "A cat sitting on a chair.",
            }
        ]
        out = h.score(predictions, _ctx_vl("captioning"))
        self.assertEqual(out["subtask"], "captioning")
        self.assertGreater(out["bleu_4"], 0.9)
        self.assertGreater(out["rouge_l"], 0.9)
        self.assertIn("length_ratio", out)

    def test_vqa_scoring_skips_caption_metrics(self):
        h = VisionLanguageHandler()
        out = h.score(
            [{"prediction": "cat", "reference": "cat"}], _ctx_vl("vqa")
        )
        self.assertEqual(out["subtask"], "vqa")
        # VQA → no BLEU/ROUGE/length_ratio.
        self.assertNotIn("bleu_4", out)
        self.assertNotIn("rouge_l", out)
        self.assertEqual(out["exact_match"], 1.0)
        self.assertEqual(out["f1"], 1.0)

    def test_per_row_enrichment_writes_subtask(self):
        h = VisionLanguageHandler()
        row = {"prediction": "cat", "reference": "cat"}
        h.score([row], _ctx_vl("vqa"))
        self.assertEqual(row["vl_subtask"], "vqa")
        self.assertEqual(row["row_exact_match"], 1.0)

    def test_max_new_tokens_caps_at_256(self):
        self.assertEqual(VisionLanguageHandler().max_new_tokens_override(1024), 256)

    def test_empty_predictions_returns_zeroed(self):
        out = VisionLanguageHandler().score([], _ctx_vl("captioning"))
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["exact_match"], 0.0)


class AudioTranscriptHandlerTests(unittest.TestCase):
    def test_transcription_default_subtask(self):
        h = AudioTranscriptHandler()
        self.assertEqual(h._resolve_subtask(_ctx_audio()), "transcription")

    def test_transcription_prompt_references_audio(self):
        h = AudioTranscriptHandler()
        built = h.build_prompts(
            [{"audio_path": "assets/hello.wav", "transcript": "hello world"}],
            _ctx_audio("transcription"),
        )
        self.assertIn("Transcribe the audio", built[0].prompt)
        self.assertIn("assets/hello.wav", built[0].prompt)
        self.assertEqual(built[0].reference, "hello world")

    def test_audio_qa_prompt_includes_question(self):
        h = AudioTranscriptHandler()
        built = h.build_prompts(
            [
                {
                    "audio_path": "assets/clip.wav",
                    "question": "What did the speaker order?",
                    "answer": "coffee",
                }
            ],
            _ctx_audio("audio_qa"),
        )
        self.assertIn("Question: What did the speaker order?", built[0].prompt)
        self.assertIn("Audio: <audio:assets/clip.wav>", built[0].prompt)
        self.assertEqual(built[0].reference, "coffee")

    def test_transcription_scoring_produces_wer_and_cer(self):
        h = AudioTranscriptHandler()
        predictions = [
            # Perfect transcript → WER = CER = 0
            {"prediction": "hello world", "reference": "hello world"},
            # 1/2 words wrong → WER = 0.5
            {"prediction": "hello there", "reference": "hello world"},
        ]
        out = h.score(predictions, _ctx_audio("transcription"))
        self.assertEqual(out["subtask"], "transcription")
        self.assertIn("wer", out)
        self.assertIn("cer", out)
        # Mean WER is around 0.25 ((0 + 0.5) / 2).
        self.assertLess(out["wer"], 0.5)
        self.assertGreater(out["wer"], 0)

    def test_audio_qa_scoring_skips_wer(self):
        h = AudioTranscriptHandler()
        out = h.score(
            [{"prediction": "coffee", "reference": "coffee"}],
            _ctx_audio("audio_qa"),
        )
        self.assertEqual(out["subtask"], "audio_qa")
        self.assertNotIn("wer", out)
        self.assertNotIn("cer", out)
        self.assertEqual(out["exact_match"], 1.0)

    def test_max_new_tokens_caps_at_512(self):
        # Audio transcripts can be long — bigger ceiling than VL.
        self.assertEqual(AudioTranscriptHandler().max_new_tokens_override(1024), 512)

    def test_empty_predictions_returns_zeroed(self):
        out = AudioTranscriptHandler().score([], _ctx_audio("transcription"))
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["exact_match"], 0.0)


if __name__ == "__main__":
    unittest.main()

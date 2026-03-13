"""Phase 32 tests: multimodal adapter behavior in training script."""

from __future__ import annotations

import unittest

from scripts import train as train_script


class Phase32MultimodalTrainingAdapterTests(unittest.TestCase):
    def test_extract_multimodal_paths_disambiguates_shared_audio_path(self):
        image_path, audio_path = train_script._extract_multimodal_paths(
            {"path": "clips/sample.wav"}
        )
        self.assertEqual(image_path, "")
        self.assertEqual(audio_path, "clips/sample.wav")

    def test_extract_multimodal_paths_disambiguates_shared_image_path(self):
        image_path, audio_path = train_script._extract_multimodal_paths(
            {"path": "images/sample.png"}
        )
        self.assertEqual(image_path, "images/sample.png")
        self.assertEqual(audio_path, "")

    def test_infer_modality_uses_media_markers_when_paths_missing(self):
        modality = train_script._infer_input_modality(
            text="Describe this <image:invoice.png> for me",
            image_path="",
            audio_path="",
        )
        self.assertEqual(modality, "vision_language")

    def test_adapt_record_preserves_multimodal_fields_for_seq2seq(self):
        contract = train_script._build_data_adapter_contract("seq2seq", "chatml")
        adapted = train_script._adapt_record_to_text(
            {
                "question": "What is shown?",
                "answer": "A city skyline at night.",
                "image_path": "images/city.jpg",
            },
            contract,
            "chatml",
        )
        self.assertEqual(str(adapted.get("input_modality")), "vision_language")
        self.assertEqual(str(adapted.get("image_path")), "images/city.jpg")
        self.assertIn("city skyline", str(adapted.get("text")))
        self.assertEqual(str(adapted.get("source_text")), "What is shown?")
        self.assertEqual(str(adapted.get("target_text")), "A city skyline at night.")

    def test_summarize_adapted_modalities_counts_detected_modalities(self):
        summary = train_script._summarize_adapted_modalities(
            [
                {"text": "hello world"},
                {"text": "caption", "image_path": "images/a.png", "input_modality": "vision_language"},
                {"text": "transcribe", "audio_path": "audio/a.wav", "input_modality": "audio_text"},
                {
                    "text": "combined",
                    "image_path": "images/a.png",
                    "audio_path": "audio/a.wav",
                    "input_modality": "multimodal",
                },
            ]
        )
        counts = dict(summary.get("counts") or {})
        self.assertEqual(int(summary.get("total") or 0), 4)
        self.assertEqual(int(counts.get("text") or 0), 1)
        self.assertEqual(int(counts.get("vision_language") or 0), 1)
        self.assertEqual(int(counts.get("audio_text") or 0), 1)
        self.assertEqual(int(counts.get("multimodal") or 0), 1)


if __name__ == "__main__":
    unittest.main()

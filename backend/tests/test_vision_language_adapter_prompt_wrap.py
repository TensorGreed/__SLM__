"""ι-fix tests — vision-language-pair adapter writes the
production prompt format into training rows.

Pre-ι: the adapter wrote ``source_text = "image:{path}"`` (bare
text, no ``<image:…>`` placeholder brackets, no instruction, no
``Caption:`` cue) and only extracted a caption — it never
surfaced the question field for VQA rows.
``VisionLanguageHandler.build_prompts`` at eval time wraps inputs
with one of two subtask scaffolds (``"Describe the image:
<image:…>\\nCaption:"`` for captioning,
``"Question: …\\nImage: <image:…>\\nAnswer:"`` for VQA). The model
never saw the eval-time scaffold so held-out BLEU_4/ROUGE-L on
vision-language projects came in artificially low — same shape as
the bug β closed for classification-label.

Post-ι (this commit's tests pin):

  1. ``_build_vision_language_training_prompt`` and
     ``VisionLanguageHandler.build_prompts`` produce IDENTICAL
     strings byte-for-byte under both subtask branches (with +
     without image_path).
  2. Adapter reads ``subtask`` from ``adapter_config`` (injected
     by the subtask-propagation infrastructure). Missing /
     invalid subtask falls back to ``captioning`` (handler's
     ``DEFAULT_SUBTASK``).
  3. VQA rows extract ``question`` separately from caption /
     answer (pre-ι the adapter only picked the answer and
     never surfaced the question).
  4. ``target_text = f" {caption}"`` (leading space — same trick
     as β / ζ / η / θ for clean BPE continuation).
  5. Raw ``image_path`` / ``answer`` / ``question`` preserved.
  6. ``scripts/train.py:_adapt_record_to_text`` passes
     vision-wrapped ``source_text`` through untouched, with a
     tighter VQA signal (``\\nImage: <image:`` marker) to
     distinguish from a plain "Question:" QA row.
"""

from __future__ import annotations

import unittest

from app.services.data_adapter_service import (
    _build_vision_language_training_prompt,
    _map_vision_language_pair,
)


# ── Adapter wrap, per subtask ────────────────────────────────────────


class VisionLanguageAdapterWrapTests(unittest.TestCase):
    def test_captioning_default_with_image_path(self):
        # Missing subtask → handler's DEFAULT_SUBTASK ("captioning").
        out = _map_vision_language_pair(
            {"image_path": "imgs/a.jpg", "caption": "A black cat sitting."},
            {},
        )
        assert out is not None
        self.assertEqual(
            out["source_text"],
            "Describe the image: <image:imgs/a.jpg>\nCaption:",
        )
        self.assertEqual(out["target_text"], " A black cat sitting.")

    def test_captioning_preserves_text_field_byte_for_byte(self):
        # The audit's open question — pre-ι ``text`` was
        # ``"<image:{path}> {caption}"`` (no instruction). ι aligns
        # ``text`` with the wrapped prompt + target so trainer
        # paths consuming ``text`` directly see the same scaffold.
        out = _map_vision_language_pair(
            {"image_path": "imgs/b.jpg", "caption": "Photo of a bird."},
            {},
        )
        assert out is not None
        self.assertEqual(out["text"], f"{out['source_text']}{out['target_text']}")
        # Specifically: no pre-ι ``"<image:imgs/b.jpg> Photo …"``
        # leftover starting the line.
        self.assertFalse(out["text"].startswith("<image:"))

    def test_vqa_extracts_question_from_row(self):
        # The load-bearing gap pre-ι: VQA rows have a question in
        # the row but the adapter never picked it up. ι reads
        # ``question`` (or its aliases) and stitches into the
        # handler's scaffold.
        out = _map_vision_language_pair(
            {
                "image_path": "imgs/c.jpg",
                "question": "How many people are in the image?",
                "answer": "3",
            },
            {"subtask": "vqa"},
        )
        assert out is not None
        self.assertEqual(
            out["source_text"],
            "Question: How many people are in the image?\n"
            "Image: <image:imgs/c.jpg>\n"
            "Answer:",
        )
        self.assertEqual(out["target_text"], " 3")
        # ``question`` preserved as a raw field for downstream.
        self.assertEqual(out["question"], "How many people are in the image?")

    def test_vqa_question_alias_accepted(self):
        # ``prompt`` is a documented alias for ``question``. Make
        # sure the adapter doesn't only look at the literal
        # ``question`` key.
        out = _map_vision_language_pair(
            {
                "image_path": "imgs/d.jpg",
                "prompt": "What color is the sky?",
                "answer": "Blue",
            },
            {"subtask": "vqa"},
        )
        assert out is not None
        self.assertIn(
            "Question: What color is the sky?", out["source_text"]
        )

    def test_target_text_has_leading_space(self):
        out = _map_vision_language_pair(
            {"image_path": "imgs/e.jpg", "caption": "Caption."},
            {"subtask": "captioning"},
        )
        assert out is not None
        self.assertTrue(out["target_text"].startswith(" "))

    def test_raw_fields_preserved_for_downstream(self):
        out = _map_vision_language_pair(
            {
                "image_path": "imgs/f.jpg",
                "question": "raw q",
                "answer": "raw a",
            },
            {"subtask": "vqa"},
        )
        assert out is not None
        self.assertEqual(out["image_path"], "imgs/f.jpg")
        self.assertEqual(out["question"], "raw q")
        # Note: in VQA the caption_aliases pick up ``answer`` (so the
        # extracted "caption" IS the answer). Raw ``answer`` is
        # preserved on the output.
        self.assertEqual(out["answer"], "raw a")

    def test_invalid_subtask_falls_back_to_captioning(self):
        out = _map_vision_language_pair(
            {"image_path": "imgs/g.jpg", "caption": "Stuff."},
            {"subtask": "🐛bogus"},
        )
        assert out is not None
        self.assertTrue(
            out["source_text"].startswith("Describe the image:")
        )

    def test_record_missing_image_path_returns_none(self):
        self.assertIsNone(
            _map_vision_language_pair({"caption": "no image"}, {}),
        )

    def test_record_missing_caption_returns_none(self):
        self.assertIsNone(
            _map_vision_language_pair({"image_path": "imgs/x.jpg"}, {}),
        )


# ── Byte-for-byte equality with handler ──────────────────────────────


class VisionLanguageHandlerByteForByteTests(unittest.TestCase):
    """Pin equality with the handler's actual wrap. Drift here
    would silently re-introduce the train/eval mismatch the audit
    closed."""

    def test_captioning_with_image_matches_handler(self):
        adapter = _build_vision_language_training_prompt(
            "imgs/x.jpg", "captioning", "ignored-for-captioning",
        )
        expected = "Describe the image: <image:imgs/x.jpg>\nCaption:"
        self.assertEqual(adapter, expected)

    def test_captioning_without_image_matches_handler_fallback(self):
        # Handler's captioning-no-image branch: just the instruction.
        adapter = _build_vision_language_training_prompt(
            "", "captioning", "ignored",
        )
        self.assertEqual(adapter, "Describe the image:")

    def test_vqa_with_image_matches_handler(self):
        adapter = _build_vision_language_training_prompt(
            "imgs/y.jpg", "vqa", "What is shown?",
        )
        expected = (
            "Question: What is shown?\n"
            "Image: <image:imgs/y.jpg>\n"
            "Answer:"
        )
        self.assertEqual(adapter, expected)

    def test_vqa_without_image_matches_handler_fallback(self):
        # Handler's VQA-no-image branch: just the question.
        adapter = _build_vision_language_training_prompt(
            "", "vqa", "What is shown?",
        )
        self.assertEqual(adapter, "What is shown?")

    def test_adapter_source_text_carries_handler_expected_prefixes(self):
        # γ′ smoke check / row peek looks for any of the handler's
        # declared prefixes in the prepared row. Both subtasks
        # must produce a prompt that carries at least one.
        from app.services.eval_task_handler_service import VisionLanguageHandler
        prefixes = VisionLanguageHandler().expected_prompt_prefixes()
        for subtask, fixture in (
            ("captioning", {"image_path": "imgs/p.jpg", "caption": "X"}),
            ("vqa", {
                "image_path": "imgs/q.jpg",
                "question": "What?",
                "answer": "A",
            }),
        ):
            out = _map_vision_language_pair(fixture, {"subtask": subtask})
            assert out is not None
            self.assertTrue(
                any(p in out["source_text"] for p in prefixes),
                f"subtask={subtask} prefixes={prefixes!r} "
                f"source_text={out['source_text']!r}",
            )


# ── train.py tail pass-through ───────────────────────────────────────


class VisionLanguageAdaptRecordPassthroughTests(unittest.TestCase):
    def test_captioning_wrap_passes_through_untouched(self):
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        wrapped = "Describe the image: <image:imgs/a.jpg>\nCaption:"
        adapted = train_script._adapt_record_to_text(
            {
                "image_path": "imgs/a.jpg",
                "answer": "Caption.",
                "text": f"{wrapped} Caption.",
                "source_text": wrapped,
                "target_text": " Caption.",
            },
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], wrapped)
        self.assertEqual(adapted["target_text"], " Caption.")

    def test_vqa_wrap_passes_through_untouched(self):
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        wrapped = (
            "Question: What is shown?\n"
            "Image: <image:imgs/b.jpg>\n"
            "Answer:"
        )
        adapted = train_script._adapt_record_to_text(
            {
                "image_path": "imgs/b.jpg",
                "question": "What is shown?",
                "answer": "A cat.",
                "text": f"{wrapped} A cat.",
                "source_text": wrapped,
                "target_text": " A cat.",
            },
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], wrapped)

    def test_plain_question_qa_row_not_misdetected_as_vqa(self):
        # The pass-through's VQA signal requires BOTH the
        # ``Question:`` start AND the ``\\nImage: <image:`` marker.
        # A plain QA row with just ``Question: …\\nAnswer:`` must
        # not be misrouted through this branch (it would still get
        # the right answer, but the regression guard catches a
        # future loosening of the marker that would conflate
        # adapter shapes).
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        # No vision passthrough should fire — this row goes
        # through the regular question+answer reconstruction.
        adapted = train_script._adapt_record_to_text(
            {"question": "What is 2+2?", "answer": "4"},
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], "What is 2+2?")
        self.assertEqual(adapted["target_text"], "4")


if __name__ == "__main__":
    unittest.main()

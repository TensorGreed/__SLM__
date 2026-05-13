"""Phase 5.2 — eval-time prompt template + SQuAD-style scorer normalization.

Three things land together because they're a single user-facing fix
("my QA eval scored 0.05 F1, why?"):

1. ``_apply_chat_template_if_present`` wraps a bare user prompt with the
   tokenizer's saved chat template when one exists, so SFT-trained models
   see the prompt shape they were trained on (Llama [INST]…[/INST],
   ChatML, Alpaca, etc.). Falls through cleanly for plain base models.

2. ``exact_match`` is now SQuAD-style: lowercase + article stripping
   ("a/an/the") + punctuation stripping + whitespace collapse before
   comparison. So ``"Paris."`` vs ``"Paris"`` now scores 1.0.

3. ``f1_score`` is SQuAD-style F1 over Counter-intersected tokens
   (not the old set-intersection which silently collapsed duplicate
   tokens and counted ``"the the the"`` as a match for ``"the"``).
"""

from __future__ import annotations

import os
import unittest
from typing import Any

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.evaluation_service import (  # noqa: E402  (env must be set first)
    _apply_chat_template_if_present,
    _normalize_answer,
    exact_match,
    f1_score,
)


class FakeTokenizer:
    """Minimal stand-in for an HF tokenizer so we don't need transformers."""

    def __init__(self, chat_template: str | None = None, raises: bool = False) -> None:
        self.chat_template = chat_template
        self._raises = raises

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        if self._raises:
            raise RuntimeError("simulated template render failure")
        if not self.chat_template:
            return ""
        # Tiny template implementation good enough for assertions.
        user = next((m for m in messages if m.get("role") == "user"), {})
        out = f"[INST] {user.get('content', '')} [/INST]"
        if add_generation_prompt:
            out += " "
        assert tokenize is False
        return out


class NormalizeAnswerTests(unittest.TestCase):
    def test_strips_articles_and_punctuation(self):
        self.assertEqual(_normalize_answer("The Paris."), "paris")
        self.assertEqual(_normalize_answer("a cat, sat!"), "cat sat")

    def test_collapses_whitespace(self):
        self.assertEqual(_normalize_answer("  hello   world  "), "hello world")

    def test_none_and_empty(self):
        self.assertEqual(_normalize_answer(None), "")
        self.assertEqual(_normalize_answer(""), "")

    def test_does_not_strip_inside_word(self):
        # "the" inside "theatre" should NOT be stripped.
        self.assertEqual(_normalize_answer("theatre"), "theatre")


class ExactMatchTests(unittest.TestCase):
    def test_punctuation_difference_now_matches(self):
        # The smoking-gun naive-scorer failure: "Paris." vs "Paris" → 0
        # under the old impl. SQuAD normalization makes it 1.0.
        self.assertEqual(exact_match("Paris.", "Paris"), 1.0)

    def test_article_difference_now_matches(self):
        self.assertEqual(exact_match("The Eiffel Tower", "Eiffel Tower"), 1.0)

    def test_case_difference_matches(self):
        self.assertEqual(exact_match("paris", "Paris"), 1.0)

    def test_genuinely_different_answers_still_miss(self):
        self.assertEqual(exact_match("London", "Paris"), 0.0)


class F1ScoreTests(unittest.TestCase):
    def test_counter_semantics_not_set_semantics(self):
        # Old set-based impl would score "the the the" vs "the cat" as 0.67
        # (perfect overlap of the only common token). SQuAD F1 over multisets
        # punishes the duplicates because they inflate prediction length.
        score = f1_score("the the the", "the cat")
        self.assertLess(score, 0.5)

    def test_punctuation_does_not_kill_overlap(self):
        # "cat," vs "cat" used to miss entirely; now it matches cleanly.
        self.assertEqual(f1_score("cat,", "cat"), 1.0)

    def test_perfect_overlap(self):
        self.assertEqual(f1_score("the quick brown fox", "the quick brown fox"), 1.0)

    def test_zero_overlap(self):
        self.assertEqual(f1_score("foo bar", "baz qux"), 0.0)

    def test_empty_prediction(self):
        self.assertEqual(f1_score("", "the answer"), 0.0)

    def test_empty_reference_with_empty_prediction(self):
        # Both empty → trivially correct; matches SQuAD convention.
        self.assertEqual(f1_score("", ""), 1.0)


class ChatTemplateTests(unittest.TestCase):
    def test_applies_template_when_present(self):
        tok = FakeTokenizer(chat_template="dummy")
        formatted, applied = _apply_chat_template_if_present(tok, "What is 2+2?")
        self.assertTrue(applied)
        self.assertIn("[INST]", formatted)
        self.assertIn("What is 2+2?", formatted)

    def test_falls_back_when_no_template(self):
        tok = FakeTokenizer(chat_template=None)
        formatted, applied = _apply_chat_template_if_present(tok, "raw prompt")
        self.assertFalse(applied)
        self.assertEqual(formatted, "raw prompt")

    def test_falls_back_when_template_raises(self):
        tok = FakeTokenizer(chat_template="dummy", raises=True)
        formatted, applied = _apply_chat_template_if_present(tok, "raw prompt")
        self.assertFalse(applied)
        self.assertEqual(formatted, "raw prompt")

    def test_falls_back_when_tokenizer_has_no_apply_method(self):
        class Bare:
            chat_template = "dummy"

        formatted, applied = _apply_chat_template_if_present(Bare(), "raw prompt")
        self.assertFalse(applied)
        self.assertEqual(formatted, "raw prompt")

    def test_falls_back_when_template_returns_empty(self):
        class EmptyOut:
            chat_template = "dummy"

            def apply_chat_template(self, *a, **kw):  # noqa: ARG002
                return ""

        formatted, applied = _apply_chat_template_if_present(EmptyOut(), "raw prompt")
        self.assertFalse(applied)
        self.assertEqual(formatted, "raw prompt")


if __name__ == "__main__":
    unittest.main()

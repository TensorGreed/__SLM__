"""η-fix tests — rag-grounded adapter writes the production
prompt format into training rows.

Pre-η: the adapter wrote ``source_text = f"{context}\\n\\n{question}"``
(no instruction, no labeled blocks, no refusal hint) and
``text = "Context:\\n{ctx}\\n\\nQuestion: {q}\\nAnswer: {a}"``
(newline after Context: rather than the handler's space-after).
``RAGHandler.build_prompts`` at eval time wrapped inputs with the
grounded instruction prompt — the model never saw the eval-time
scaffold so held-out faithfulness + context_recall came in
artificially low (same shape as the bug β surfaced for
classification-label).

Post-η (this commit's tests pin):

  1. ``_build_rag_training_prompt`` and the handler's grounded
     wrap produce IDENTICAL strings byte-for-byte (one branch —
     RAG is single-mode, unlike ζ's field-list/no-list pair).
  2. ``_map_rag_grounded`` writes ``source_text`` as the wrapped
     prompt and ``target_text`` as ``f" {answer}"`` so the
     decoder treats the answer as a clean continuation of
     ``Answer:`` (mirrors β + ζ).
  3. ``text`` is ALSO rebuilt byte-for-byte against the handler
     (the audit's open question #2 — pre-η ``text`` used
     ``Context:\\n{ctx}`` newline-after-colon, handler emits
     ``Context: {ctx}`` space-after-colon). The trainer
     consuming ``text`` directly now sees the same scaffold.
  4. Raw ``question`` / ``context`` / ``answer`` stay for
     downstream surfaces (data health, gold diagnostics, smoke
     peek, RAG context-recall introspection).
  5. ``scripts/train.py:_adapt_record_to_text`` passes
     RAG-wrapped ``source_text`` through untouched — same shape
     as β-tail + ζ-tail.
"""

from __future__ import annotations

import unittest

from app.services.data_adapter_service import (
    _build_rag_training_prompt,
    _map_rag_grounded,
)


class RAGAdapterWrapTests(unittest.TestCase):
    def test_wrap_renders_grounded_instruction_prompt(self):
        out = _map_rag_grounded(
            {
                "question": "When did Acme launch?",
                "context": "Acme was founded in 1999 and launched in 2001.",
                "answer": "2001",
            },
            {},
        )
        assert out is not None
        # All four load-bearing pieces of the grounded prompt.
        self.assertIn(
            "Answer the question using only the context", out["source_text"]
        )
        self.assertIn("say you don't know", out["source_text"])
        self.assertIn(
            "Context: Acme was founded in 1999 and launched in 2001.",
            out["source_text"],
        )
        self.assertIn("Question: When did Acme launch?", out["source_text"])
        self.assertTrue(out["source_text"].endswith("Answer:"))

    def test_target_text_has_leading_space(self):
        out = _map_rag_grounded(
            {"question": "Q?", "context": "C.", "answer": "A"}, {}
        )
        assert out is not None
        self.assertEqual(out["target_text"], " A")

    def test_text_field_matches_wrapped_prompt_byte_for_byte(self):
        """The audit's open question #2: pre-η ``text`` used
        ``Context:\\n{ctx}`` (newline) while the handler emits
        ``Context: {ctx}`` (space). η aligns both so the trainer
        consuming ``text`` directly sees the same scaffold as
        eval. Pin the equality so a future refactor can't drift."""
        out = _map_rag_grounded(
            {"question": "Q?", "context": "C.", "answer": "A"}, {}
        )
        assert out is not None
        self.assertEqual(out["text"], f"{out['source_text']}{out['target_text']}")
        # Concretely: ``Context: C.\nQuestion: Q?\nAnswer: A``.
        self.assertIn("Context: C.\n", out["text"])
        self.assertNotIn("Context:\nC.", out["text"])

    def test_raw_fields_preserved_for_downstream_surfaces(self):
        out = _map_rag_grounded(
            {
                "question": "raw q",
                "context": "raw ctx",
                "answer": "raw a",
            },
            {},
        )
        assert out is not None
        self.assertEqual(out["question"], "raw q")
        self.assertEqual(out["context"], "raw ctx")
        self.assertEqual(out["answer"], "raw a")

    def test_record_missing_question_returns_none(self):
        # Pre-η behaviour: rows without all three of
        # question/context/answer are skipped. Preserved.
        self.assertIsNone(_map_rag_grounded({"context": "c", "answer": "a"}, {}))

    def test_record_missing_context_returns_none(self):
        # The handler has an empty-context fallback (just passes
        # question through as the prompt) — but the adapter still
        # rejects context-less rows because they can't carry the
        # grounded scaffold that the handler-with-context path
        # builds. Pre-η behaviour preserved.
        self.assertIsNone(_map_rag_grounded({"question": "q", "answer": "a"}, {}))


class RAGHandlerByteForByteCompatibilityTests(unittest.TestCase):
    """The load-bearing test: η guarantees adapter + handler emit
    identical strings so the trained model sees the same prompt at
    train and eval time."""

    def test_adapter_prompt_matches_handler_grounded_wrap(self):
        # Mirror the handler's grounded branch (single-mode — no
        # alternatives the way ζ has list/no-list). Construct the
        # handler's expected output via duck-typed string
        # comparison, since RAGHandler doesn't expose
        # ``_build_prompt_text`` as a standalone method (the
        # wrap lives inline in ``build_prompts``).
        context = "Some grounding paragraph."
        question = "What is X?"
        adapter_prompt = _build_rag_training_prompt(context, question)
        # Reconstruct the handler's exact format from the source.
        # eval_task_handler_service.py:1495-1503.
        expected = (
            "Answer the question using only the context. If the "
            "context does not contain the answer, say you don't "
            "know.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )
        self.assertEqual(adapter_prompt, expected)

    def test_adapter_source_text_carries_handler_expected_prefixes(self):
        from app.services.eval_task_handler_service import RAGHandler

        out = _map_rag_grounded(
            {"question": "Q?", "context": "C.", "answer": "A"}, {}
        )
        assert out is not None
        for prefix in RAGHandler().expected_prompt_prefixes():
            self.assertIn(prefix, out["source_text"], f"prefix {prefix!r}")


class RAGAdaptRecordPassthroughTests(unittest.TestCase):
    """η-tail — scripts/train.py:_adapt_record_to_text must pass
    adapter-wrapped source_text through untouched for RAG rows.
    Without this the direct-text branch would clobber the wrapped
    prompt with the raw ``text`` field, same shape as the β-tail
    bug for classification + ζ-tail for structured."""

    def test_wrapped_rag_row_passes_through_untouched(self):
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        wrapped = (
            "Answer the question using only the context. If the "
            "context does not contain the answer, say you don't "
            "know.\n"
            "Context: C.\n"
            "Question: Q?\n"
            "Answer:"
        )
        adapted = train_script._adapt_record_to_text(
            {
                "question": "Q?",
                "context": "C.",
                "answer": "A",
                "text": f"{wrapped} A",
                "source_text": wrapped,
                "target_text": " A",
            },
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], wrapped)
        self.assertEqual(adapted["target_text"], " A")
        self.assertEqual(adapted["text"], f"{wrapped} A")

    def test_legacy_qa_row_unaffected_by_rag_tail(self):
        # A row without an "Answer the question using only the
        # context" prefix should still go through the existing
        # question/answer reconstruction path — η-tail must not
        # accidentally swallow non-RAG rows.
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        adapted = train_script._adapt_record_to_text(
            {"question": "What is 2+2?", "answer": "4"},
            contract,
            "chatml",
        )
        # The question+answer branch renders via
        # _qa_to_chat_text → chatml-wrapped text.
        self.assertIn("4", adapted["text"])
        self.assertEqual(adapted["source_text"], "What is 2+2?")
        self.assertEqual(adapted["target_text"], "4")


if __name__ == "__main__":
    unittest.main()

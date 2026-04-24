"""Phase 61 tests — teacher-model response parser tolerates Qwen3-style output.

A local Ollama teacher model (qwen3 family) surfaced a parse failure in the
synthetic pipeline because its responses wrap reasoning in ``<think>…</think>``
tags and often dress Q/A markers in markdown (``**Q:**``, ``### Question 1``).
This phase covers:

* ``<think>`` / ``<thinking>`` / ``<reasoning>`` blocks are stripped before
  JSON or plaintext extraction (including unterminated tags).
* Mixed ``<think>`` + JSON output round-trips to clean pairs.
* Markdown-dressed Q/A blocks parse when JSON isn't present.
* The ``call_teacher_model(force_json=True)`` flag surfaces the correct
  ``response_format`` + Ollama ``format`` fields on the request payload.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.config import settings
from app.services.synthetic_service import (
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    _extract_json_blocks,
    _parse_plaintext_qa_pairs,
    _parse_teacher_pairs,
    _strip_thinking_blocks,
    call_teacher_model,
    generate_qa_pairs,
)


class Phase61ThinkingStripTests(unittest.TestCase):
    def test_removes_balanced_think_block(self):
        raw = (
            "<think>First I need to identify the subjects in the text "
            "and form questions.</think>\n"
            '{"pairs":[{"question":"Q1","answer":"A1"}]}'
        )
        cleaned = _strip_thinking_blocks(raw)
        self.assertNotIn("<think", cleaned.lower())
        self.assertIn('"pairs"', cleaned)

    def test_removes_unterminated_think_block(self):
        raw = (
            "<think>The model got cut off mid-reasoning and never "
            "closed the tag because max_tokens hit…"
        )
        cleaned = _strip_thinking_blocks(raw)
        self.assertEqual(cleaned, "")

    def test_handles_alternate_reasoning_tags(self):
        raw = (
            "<thinking>outline</thinking>\n"
            "<reasoning>why</reasoning>\n"
            '{"pairs":[]}'
        )
        cleaned = _strip_thinking_blocks(raw)
        self.assertEqual(cleaned, '{"pairs":[]}')

    def test_extract_json_blocks_skips_thinking_prefix(self):
        raw = (
            "<think>planning</think>\n"
            '{"pairs":[{"question":"Is this parsed?","answer":"Yes"}]}'
        )
        blocks = _extract_json_blocks(raw)
        self.assertTrue(blocks)
        # Every candidate must parse as JSON once thinking is stripped.
        import json

        for block in blocks:
            json.loads(block)


class Phase61PlaintextParserTests(unittest.TestCase):
    def test_markdown_bold_q_a_markers(self):
        raw = (
            "Here are some example pairs:\n\n"
            "**Q:** What is the capital of Canada?\n"
            "**A:** Ottawa is the capital.\n\n"
            "**Q:** What river runs through Ottawa?\n"
            "**A:** The Ottawa River.\n"
        )
        pairs = _parse_plaintext_qa_pairs(raw)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["question"], "What is the capital of Canada?")
        self.assertEqual(pairs[1]["answer"], "The Ottawa River.")

    def test_numbered_question_answer_words(self):
        raw = (
            "1. Question: What is tokenization?\n"
            "   Answer: Breaking text into model-friendly units.\n"
            "2. Question: Why does it matter?\n"
            "   Answer: Because the model only sees tokens.\n"
        )
        pairs = _parse_plaintext_qa_pairs(raw)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[1]["question"], "Why does it matter?")

    def test_markdown_heading_and_trailing_numbering(self):
        raw = (
            "### Q1\n"
            "What is a gate in evaluation?\n"
            "### A1\n"
            "A pass/fail threshold on a metric.\n"
        )
        pairs = _parse_plaintext_qa_pairs(raw)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "What is a gate in evaluation?")
        self.assertIn("threshold", pairs[0]["answer"])

    def test_qwen3_thinking_then_plaintext_qa(self):
        raw = (
            "<think>I will answer briefly.</think>\n\n"
            "**Q1:** What does VRAM stand for?\n"
            "**A1:** Video RAM.\n"
            "**Q2:** What is QLoRA?\n"
            "**A2:** A parameter-efficient tuning technique.\n"
        )
        pairs = _parse_teacher_pairs(raw)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["question"], "What does VRAM stand for?")

    def test_json_path_still_works_unchanged(self):
        raw = '{"pairs":[{"question":"Q","answer":"A"}]}'
        pairs = _parse_teacher_pairs(raw)
        self.assertEqual(pairs, [{"question": "Q", "answer": "A"}])


class Phase61ForceJsonPayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_force_json_adds_response_format_and_ollama_format(self):
        captured: dict = {}

        class _FakeResponse:
            def raise_for_status(self) -> None: ...

            def json(self) -> dict:
                return {
                    "choices": [{"message": {"content": '{"pairs":[]}'}}],
                    "usage": {"total_tokens": 1},
                    "model": "ollama/qwen",
                }

        class _FakeClient:
            def __init__(self, *args, **kwargs): ...

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["payload"] = json
                captured["headers"] = headers
                return _FakeResponse()

        with patch.object(httpx, "AsyncClient", _FakeClient):
            await call_teacher_model(
                prompt="generate Q&A",
                api_url="http://localhost:11434/v1/chat/completions",
                model_name="qwen3.6-fast-64k:latest",
                force_json=True,
            )
        payload = captured.get("payload") or {}
        self.assertEqual(payload.get("response_format"), {"type": "json_object"})
        self.assertEqual(payload.get("format"), "json")
        self.assertEqual(payload.get("model"), "qwen3.6-fast-64k:latest")

    async def test_force_json_disabled_leaves_payload_minimal(self):
        captured: dict = {}

        class _FakeResponse:
            def raise_for_status(self) -> None: ...

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "ok"}}], "usage": {}, "model": "m"}

        class _FakeClient:
            def __init__(self, *args, **kwargs): ...

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None):
                captured["payload"] = json
                return _FakeResponse()

        with patch.object(httpx, "AsyncClient", _FakeClient):
            await call_teacher_model(
                prompt="generate",
                api_url="http://localhost:11434/v1/chat/completions",
                model_name="llama3",
            )
        payload = captured.get("payload") or {}
        self.assertNotIn("response_format", payload)
        self.assertNotIn("format", payload)


class Phase61UniversalTeacherConfigTests(unittest.IsolatedAsyncioTestCase):
    """Model-agnostic guardrails: defaults that serve every backend (OpenAI,
    llama, Mistral, Qwen, DeepSeek), plus opt-in knobs for the few that need
    model-specific tweaks."""

    async def _capture_payload(self) -> dict:
        captured: dict = {}

        class _FakeResponse:
            def raise_for_status(self) -> None: ...

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "ok"}}], "usage": {}, "model": "m"}

        class _FakeClient:
            def __init__(self, *args, **kwargs): ...

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None):
                captured["payload"] = json
                return _FakeResponse()

        with patch.object(httpx, "AsyncClient", _FakeClient):
            await call_teacher_model(
                prompt="generate",
                api_url="http://localhost:11434/v1/chat/completions",
                model_name="llama3",
            )
        return captured["payload"]

    async def test_default_max_tokens_is_4096(self):
        payload = await self._capture_payload()
        self.assertEqual(payload["max_tokens"], 4096)

    async def test_default_system_prompt_discourages_reasoning_preamble(self):
        payload = await self._capture_payload()
        system = next(
            m["content"] for m in payload["messages"] if m["role"] == "system"
        )
        # Model-agnostic hint, not a Qwen3-specific directive.
        self.assertIn("directly", system.lower())
        self.assertIn("reasoning", system.lower())
        self.assertEqual(system, DEFAULT_TEACHER_SYSTEM_PROMPT)

    async def test_no_think_suffix_is_blank_by_default(self):
        # Without any config, the user prompt goes through verbatim.
        # llama / OpenAI users should not see anything appended.
        self.assertEqual(
            getattr(settings, "TEACHER_MODEL_NO_THINK_SUFFIX", ""), ""
        )
        payload = await self._capture_payload()
        user_content = next(
            m["content"] for m in payload["messages"] if m["role"] == "user"
        )
        self.assertEqual(user_content, "generate")

    async def test_no_think_suffix_is_appended_when_configured(self):
        original = getattr(settings, "TEACHER_MODEL_NO_THINK_SUFFIX", "")
        settings.TEACHER_MODEL_NO_THINK_SUFFIX = "/no_think"
        try:
            payload = await self._capture_payload()
            user_content = next(
                m["content"] for m in payload["messages"] if m["role"] == "user"
            )
            self.assertTrue(user_content.endswith("/no_think"))
            self.assertIn("generate", user_content)
        finally:
            settings.TEACHER_MODEL_NO_THINK_SUFFIX = original


class Phase61EmptyResponseDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    """When the parser fails, the error message must tell the user what to do."""

    async def _run_generate_with_content(self, content: str) -> str:
        """Call the full Q&A generator against a fake teacher that returns
        ``content``, capture the ValueError, return its message."""

        async def fake_call(*args, **kwargs):
            return {"content": content, "tokens_used": 1, "model": "fake"}

        with patch(
            "app.services.synthetic_service.call_teacher_model",
            side_effect=fake_call,
        ):
            try:
                await generate_qa_pairs(
                    db=None,
                    project_id=1,
                    source_text="whatever",
                    num_pairs=2,
                    api_url="http://fake",
                    model_name="fake",
                )
            except ValueError as exc:
                return str(exc)
        self.fail("generate_qa_pairs should have raised ValueError")

    async def test_unterminated_think_triggers_specific_diagnostic(self):
        msg = await self._run_generate_with_content(
            "<think>I need to think about this carefully and the budget is"
        )
        self.assertIn("empty after stripping reasoning tags", msg)
        self.assertIn("max_tokens", msg)
        self.assertIn("TEACHER_MODEL_NO_THINK_SUFFIX", msg)

    async def test_completely_empty_content_triggers_endpoint_hint(self):
        msg = await self._run_generate_with_content("")
        self.assertIn("empty response", msg.lower())
        self.assertIn("teacher endpoint", msg.lower())

    async def test_garbage_content_triggers_generic_format_hint(self):
        msg = await self._run_generate_with_content(
            "Sure! Here are some pairs for you to enjoy without any structure."
        )
        self.assertIn("response_format", msg)
        self.assertIn("JSON mode", msg)


if __name__ == "__main__":
    unittest.main()

"""Tests for the LLM-assisted gold-set generation path.

Covers:
  * ``cloud_llm_service.extract_json_payload`` — code-fence tolerance,
    preamble stripping, fallback slicing.
  * ``gold_llm_service.generate_gold_qa_via_llm`` — recipe-shape
    gating, prompt incorporates project context, response parsing,
    structured error codes for caller-fixable failures.
  * ``POST /api/projects/{id}/gold/generate-via-llm`` — endpoint
    error codes (API_KEY_REQUIRED, RECIPE_REQUIRED, etc.), happy
    path returns parsed rows for preview.

LLM clients themselves are monkeypatched to avoid network calls.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "gold_llm_service.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "gold_llm_service_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["DOMAIN_BLUEPRINT_ENABLE_LLM_ENRICHMENT"] = "false"

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.cloud_llm_service import (  # noqa: E402
    CloudLlmError,
    CloudLlmResponse,
    extract_json_payload,
)


class ExtractJsonPayloadTests(unittest.TestCase):
    """Pure-function tolerance tests — every shape the LLM might
    return that wraps the actual JSON in something."""

    def test_plain_object(self):
        self.assertEqual(
            extract_json_payload('{"pairs":[{"question":"Q","answer":"A"}]}'),
            {"pairs": [{"question": "Q", "answer": "A"}]},
        )

    def test_code_fenced_json(self):
        raw = '```json\n{"pairs":[{"question":"Q","answer":"A"}]}\n```'
        self.assertEqual(
            extract_json_payload(raw),
            {"pairs": [{"question": "Q", "answer": "A"}]},
        )

    def test_code_fenced_no_lang_hint(self):
        raw = '```\n{"pairs":[]}\n```'
        self.assertEqual(extract_json_payload(raw), {"pairs": []})

    def test_preamble_then_json_slice(self):
        # No fences, but a chatty preamble before the object.
        raw = (
            'Here are the Q&A pairs you requested:\n\n'
            '{"pairs":[{"question":"Q","answer":"A"}]}'
        )
        parsed = extract_json_payload(raw)
        self.assertEqual(parsed, {"pairs": [{"question": "Q", "answer": "A"}]})

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            extract_json_payload("")

    def test_unparseable_raises(self):
        with self.assertRaises(ValueError):
            extract_json_payload("not json at all just words")

    def test_think_block_stripped_before_json_extraction(self):
        # Reasoning models (deepseek-reasoner, R1 family, o-series)
        # emit <think>...</think> preambles even with response_format
        # set to json_object. Those blocks frequently contain {} and
        # [] chars (rehearsing the output mid-reasoning), which would
        # derail the naïve "slice from first { to last }" fallback.
        raw = (
            '<think>\nLet me think about the user\'s request. They '
            'want pairs like {"question": "...", "answer": "..."}, '
            'maybe 3 of them. Let me draft them.\n</think>\n\n'
            '{"pairs":[{"question":"Q","answer":"A"}]}'
        )
        parsed = extract_json_payload(raw)
        self.assertEqual(parsed, {"pairs": [{"question": "Q", "answer": "A"}]})

    def test_unterminated_think_block_stripped(self):
        # Streaming truncation / max_tokens hit mid-thought leaves an
        # unterminated <think> — if it's all we have, fail with a
        # helpful message; if real JSON came after, parse it.
        from app.services.cloud_llm_service import extract_json_payload as _e

        # Case A: everything was the think block — clear error.
        with self.assertRaises(ValueError) as cm:
            _e('<think>\nI was thinking but ran out of tokens before')
        self.assertIn("reasoning preamble", str(cm.exception))

        # Case B: think block followed by valid JSON, but the closing
        # tag is missing. The unterminated stripper takes everything
        # from <think> to EOF, so JSON after the think block is also
        # lost. That's correct behavior — without the closing tag we
        # can't tell where reasoning ended and answer began.
        with self.assertRaises(ValueError):
            _e('<think>thinking...\n{"pairs":[{"question":"Q","answer":"A"}]}')

    def test_reasoning_with_code_fence_combo(self):
        # Worst-case real-world shape: think block + code fence
        # around the JSON. Both wrappers must strip cleanly.
        raw = (
            '<reasoning>analyzing the project context</reasoning>\n'
            'Here are the pairs:\n```json\n'
            '{"pairs":[{"question":"Q1","answer":"A1"}]}\n```'
        )
        parsed = extract_json_payload(raw)
        self.assertEqual(
            parsed, {"pairs": [{"question": "Q1", "answer": "A1"}]},
        )


class ParserToleranceTests(unittest.TestCase):
    """Direct tests of ``_parse_qa_payload`` — the alias map + container
    keys + improved error messages. Goal: real LLM outputs across
    vendors should parse without the user having to know what field
    names BrewSLM wants."""

    def _parse(self, content: str, expected_count: int = 2):
        from app.services.gold_llm_service import _parse_qa_payload
        return _parse_qa_payload(content, expected_count)

    # ── Alias paths — each accepted field-name variant ────────────

    def test_canonical_question_answer_keys(self):
        rows = self._parse(
            '{"pairs":[{"question":"Q1","answer":"A1"},'
            '{"question":"Q2","answer":"A2"}]}',
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].question, "Q1")

    def test_q_a_short_form(self):
        rows = self._parse(
            '{"pairs":[{"q":"Q1","a":"A1"},{"q":"Q2","a":"A2"}]}',
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].question, "Q1")
        self.assertEqual(rows[0].answer, "A1")

    def test_prompt_response_form(self):
        rows = self._parse(
            '{"pairs":[{"prompt":"P1","response":"R1"},'
            '{"prompt":"P2","response":"R2"}]}',
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].question, "P1")

    def test_input_output_form(self):
        rows = self._parse(
            '[{"input":"i1","output":"o1"},{"input":"i2","output":"o2"}]',
        )
        self.assertEqual(len(rows), 2)

    def test_case_insensitive_field_match(self):
        # Some models emit TitleCase or UPPERCASE keys; aliases match
        # case-insensitively.
        rows = self._parse(
            '{"pairs":[{"Question":"Q1","ANSWER":"A1"}]}',
            expected_count=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].question, "Q1")

    # ── Container key variants ────────────────────────────────────

    def test_questions_container_key(self):
        rows = self._parse(
            '{"questions":[{"question":"Q","answer":"A"}]}',
            expected_count=1,
        )
        self.assertEqual(len(rows), 1)

    def test_examples_container_key(self):
        rows = self._parse(
            '{"examples":[{"q":"Q","a":"A"}]}',
            expected_count=1,
        )
        self.assertEqual(len(rows), 1)

    def test_nested_container_one_level(self):
        # {"data": {"pairs": [...]}} shape — common from
        # over-specified prompt templates.
        rows = self._parse(
            '{"data":{"pairs":[{"question":"Q","answer":"A"}]}}',
            expected_count=1,
        )
        self.assertEqual(len(rows), 1)

    def test_single_arbitrary_list_value_fallback(self):
        # {"result": [...]} — single-key dict whose value is a
        # list. Last-ditch path before erroring.
        rows = self._parse(
            '{"result":[{"q":"Q","a":"A"}]}',
            expected_count=1,
        )
        self.assertEqual(len(rows), 1)

    # ── Improved errors ───────────────────────────────────────────

    def test_wrong_field_names_error_includes_first_item_keys(self):
        # LLM returned items but used field names we don't know.
        # User needs to see WHICH keys came back so they can either
        # request different keys via focus_hint or pick another
        # model.
        from app.services.gold_llm_service import GoldGenerationError
        with self.assertRaises(GoldGenerationError) as cm:
            self._parse(
                '{"pairs":[{"foo":"x","bar":"y"},{"foo":"x","bar":"y"}]}',
            )
        message = str(cm.exception)
        self.assertIn("foo", message)
        self.assertIn("bar", message)
        # Also the raw response preview so users can copy-paste for
        # debugging.
        self.assertIn("pairs", message)
        self.assertEqual(cm.exception.error_code, "LLM_RESPONSE_UNPARSEABLE")

    def test_empty_pairs_array_distinct_error_message(self):
        # Grounded mode + LLM couldn't anchor any questions — emits
        # {"pairs":[]}. Error must point at grounding, not field-name.
        from app.services.gold_llm_service import GoldGenerationError
        with self.assertRaises(GoldGenerationError) as cm:
            self._parse('{"pairs":[]}')
        self.assertEqual(cm.exception.error_code, "LLM_RESPONSE_UNPARSEABLE")
        message = str(cm.exception)
        self.assertIn("grounding", message.lower())

    def test_error_includes_raw_response_preview(self):
        # Any unparseable case must include the raw response so the
        # user can see what the LLM actually returned without having
        # to dig through server logs.
        from app.services.gold_llm_service import GoldGenerationError
        with self.assertRaises(GoldGenerationError) as cm:
            self._parse('{"pairs":[{"foo":"x"}]}')
        message = str(cm.exception)
        self.assertIn("foo", message)

    def test_rationale_and_source_excerpt_aliases(self):
        # Models often use "explanation" or "reasoning" for rationale
        # and "source" / "evidence" / "quote" for the citation.
        rows = self._parse(
            '{"pairs":[{"question":"Q","answer":"A",'
            '"reasoning":"because","evidence":"from doc"}]}',
            expected_count=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].rationale, "because")
        self.assertEqual(rows[0].source_excerpt, "from doc")


class GenerateGoldQaServiceTests(unittest.TestCase):
    """Service-level tests — patch the cloud clients so we don't
    hit OpenAI / Anthropic for real."""

    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    def _make_qa_sft_project(self, name: str) -> int:
        # Template instantiate is the cleanest way to get a project
        # with selected_recipe='qa-sft' + a domain blueprint already
        # attached.
        resp = self.client.post(
            "/api/project-templates/policy-qa-style/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _make_classification_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _patched_openai(self, content: str):
        async def _fake(**kwargs):
            return CloudLlmResponse(
                content=content,
                model=kwargs.get("model", "gpt-4o-mini"),
                prompt_tokens=120,
                completion_tokens=350,
            )
        return _fake

    def _patched_anthropic(self, content: str):
        async def _fake(**kwargs):
            return CloudLlmResponse(
                content=content,
                model=kwargs.get("model", "claude-haiku-4-5-20251001"),
                prompt_tokens=110,
                completion_tokens=320,
            )
        return _fake

    # ── Happy paths ──────────────────────────────────────────────

    def test_openai_happy_path_returns_parsed_rows_for_preview(self):
        pid = self._make_qa_sft_project("Gold LLM OpenAI Happy")
        fake_content = json.dumps({
            "pairs": [
                {"question": "Q1?", "answer": "A1.", "rationale": "r1"},
                {"question": "Q2?", "answer": "A2."},
                {"question": "Q3?", "answer": "A3."},
            ],
        })
        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=self._patched_openai(fake_content),
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 3,
                    "api_key": "sk-test",
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(len(body["rows"]), 3)
        self.assertEqual(body["rows"][0]["question"], "Q1?")
        self.assertEqual(body["rows"][0]["rationale"], "r1")
        # Row 2 had no rationale — default to empty string, not missing.
        self.assertEqual(body["rows"][1]["rationale"], "")
        self.assertEqual(body["provider"], "openai")
        self.assertEqual(body["usage"]["prompt_tokens"], 120)
        # prompt_preview should include the project name (template
        # instantiation gives the project a known name we can look for).
        self.assertIn("Gold LLM OpenAI Happy", body["prompt_preview"])

    def test_anthropic_happy_path_returns_parsed_rows(self):
        pid = self._make_qa_sft_project("Gold LLM Anthropic Happy")
        fake_content = (
            '```json\n'
            '{"pairs":[{"question":"Q?","answer":"A."}]}\n'
            '```'
        )
        with patch(
            "app.services.gold_llm_service.call_anthropic_chat",
            side_effect=self._patched_anthropic(fake_content),
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "anthropic",
                    "model": "claude-haiku-4-5-20251001",
                    "count": 1,
                    "api_key": "sk-ant-test",
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(len(body["rows"]), 1)
        self.assertEqual(body["provider"], "anthropic")

    # ── Caller-fixable errors (400 + structured error_code) ──────

    def test_missing_api_key_returns_structured_error(self):
        pid = self._make_qa_sft_project("Gold LLM No Key")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "count": 3,
                # api_key omitted; no stored secret either.
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json().get("detail") or {}
        self.assertEqual(detail.get("error_code"), "API_KEY_REQUIRED")

    def test_classification_project_returns_recipe_not_supported(self):
        pid = self._make_classification_project("Gold LLM Wrong Recipe")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "count": 3,
                "api_key": "sk-test",
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json().get("detail") or {}
        self.assertEqual(detail.get("error_code"), "RECIPE_NOT_SUPPORTED")
        self.assertIn("qa-sft", detail.get("message", ""))

    def test_null_recipe_project_returns_recipe_required(self):
        import asyncio
        from app.database import async_session_factory
        from app.services.recipe_apply_service import clear_recipe_from_project

        pid = self._make_qa_sft_project("Gold LLM Null Recipe")

        async def _clear() -> None:
            async with async_session_factory() as db:
                await clear_recipe_from_project(db, pid)
                await db.commit()
        asyncio.run(_clear())

        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "count": 3,
                "api_key": "sk-test",
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json().get("detail") or {}
        self.assertEqual(detail.get("error_code"), "RECIPE_REQUIRED")

    def test_unparseable_llm_response_returns_structured_error(self):
        pid = self._make_qa_sft_project("Gold LLM Bad Response")
        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=self._patched_openai("just some words, no json here"),
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 3,
                    "api_key": "sk-test",
                },
            )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json().get("detail") or {}
        # Either LLM_RESPONSE_UNPARSEABLE (preferred — no JSON found)
        # or LLM_RESPONSE_COUNT_MISMATCH (if the JSON path slid through
        # to count check). Both are structured + frontend-actionable.
        self.assertIn(
            detail.get("error_code"),
            {"LLM_RESPONSE_UNPARSEABLE", "LLM_RESPONSE_COUNT_MISMATCH"},
        )

    def test_count_out_of_range_returns_structured_error(self):
        pid = self._make_qa_sft_project("Gold LLM Bad Count")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "count": 999,  # > 50
                "api_key": "sk-test",
            },
        )
        # Pydantic field-level validation fires at 422 for the count
        # bound — that's a fine signal too.
        self.assertIn(resp.status_code, (400, 422), resp.text)

    # ── Upstream provider failure → 502 ──────────────────────────

    def test_upstream_provider_error_returns_502(self):
        pid = self._make_qa_sft_project("Gold LLM Upstream Boom")

        async def _boom(**kwargs):
            raise CloudLlmError("OpenAI returned 401: invalid API key")

        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=_boom,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 3,
                    "api_key": "sk-test",
                },
            )
        self.assertEqual(resp.status_code, 502, resp.text)
        # Detail is a plain string with the upstream error — no JSON
        # parsing required by the frontend.
        self.assertIn("invalid API key", resp.json()["detail"])

    # ── Project context flows into the prompt ────────────────────

    def test_prompt_includes_focus_hint_and_project_name(self):
        pid = self._make_qa_sft_project("Focus Hint Project")
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return CloudLlmResponse(
                content='{"pairs":[{"question":"Q","answer":"A"}]}',
                model="gpt-4o-mini",
                prompt_tokens=1,
                completion_tokens=1,
            )

        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=_capture,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 1,
                    "focus_hint": "cover edge cases around refunds",
                    "api_key": "sk-test",
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        user_prompt = captured.get("user_prompt") or ""
        self.assertIn("Focus Hint Project", user_prompt)
        self.assertIn("cover edge cases around refunds", user_prompt)
        # Output rules section is always present.
        self.assertIn("Return ONLY valid JSON", user_prompt)


class ChunkSamplerTests(unittest.TestCase):
    """Pure-function tests for ``_sample_reference_chunks`` — the
    cost-control guard. Caps MUST hold so the cost estimator's
    math stays trustworthy."""

    def test_empty_pool_returns_empty_list(self):
        from app.services.gold_llm_service import _sample_reference_chunks
        self.assertEqual(_sample_reference_chunks([]), [])

    def test_stratified_pick_spans_first_and_last(self):
        from app.services.gold_llm_service import _sample_reference_chunks
        # 100-chunk pool, ask for 5 — should include first + last.
        pool = [f"chunk-{i}" for i in range(100)]
        sampled = _sample_reference_chunks(pool, max_chunks=5)
        self.assertEqual(len(sampled), 5)
        # First and last chunks of the pool both made it in.
        labels = [c.source_label for c in sampled]
        self.assertIn("chunk-1-of-100", labels)
        self.assertIn("chunk-100-of-100", labels)

    def test_per_chunk_char_cap_is_enforced(self):
        from app.services.gold_llm_service import _sample_reference_chunks
        # One huge chunk — must be truncated to max_chars_per_chunk.
        huge = "x" * 10_000
        sampled = _sample_reference_chunks(
            [huge], max_chars_per_chunk=500,
        )
        self.assertEqual(len(sampled), 1)
        self.assertEqual(len(sampled[0].text), 500)

    def test_total_chars_cap_is_enforced_across_chunks(self):
        from app.services.gold_llm_service import _sample_reference_chunks
        pool = ["a" * 3000 for _ in range(10)]
        sampled = _sample_reference_chunks(
            pool,
            max_chunks=10,
            max_chars_per_chunk=3000,
            max_total_chars=5000,
        )
        total_chars = sum(len(c.text) for c in sampled)
        self.assertLessEqual(total_chars, 5000)
        # And there's at least one chunk (i.e. the cap kicked in
        # mid-iteration, didn't return empty).
        self.assertGreaterEqual(len(sampled), 1)

    def test_pool_smaller_than_max_chunks_returns_all(self):
        from app.services.gold_llm_service import _sample_reference_chunks
        pool = ["a", "b", "c"]
        sampled = _sample_reference_chunks(pool, max_chunks=10)
        self.assertEqual(len(sampled), 3)


class CostHelpersTests(unittest.TestCase):

    def test_lookup_pricing_known_model(self):
        from app.services.gold_llm_service import _lookup_pricing
        self.assertEqual(_lookup_pricing("gpt-4o-mini"), (0.15, 0.60))
        self.assertEqual(_lookup_pricing("gpt-4o"), (2.50, 10.00))

    def test_lookup_pricing_date_stamped_anthropic_model(self):
        # Production Anthropic model strings include a date suffix —
        # the prefix-match path must still resolve them.
        from app.services.gold_llm_service import _lookup_pricing
        self.assertEqual(
            _lookup_pricing("claude-haiku-4-5-20251001"),
            (1.00, 5.00),
        )

    def test_lookup_pricing_deepseek_models(self):
        # Deepseek's API is OpenAI-compatible so the frontend sends
        # provider=openai + api_url=<deepseek host>; cost-estimate
        # must price the model string correctly even though provider
        # is "openai" on the wire.
        from app.services.gold_llm_service import _lookup_pricing
        self.assertEqual(_lookup_pricing("deepseek-chat"), (0.27, 1.10))
        self.assertEqual(_lookup_pricing("deepseek-reasoner"), (0.55, 2.19))

    def test_lookup_pricing_unknown_deepseek_variant_falls_back(self):
        # An unconfirmed model like the "DeepSeek-V4-Pro" some users
        # ask about — prefix doesn't match a known SKU, so falls back
        # to the cheapest-tier estimate (defensive under-estimate
        # rather than wrong-direction over-estimate).
        from app.services.gold_llm_service import _lookup_pricing
        self.assertEqual(_lookup_pricing("DeepSeek-V4-Pro"), (0.15, 0.60))

    def test_lookup_pricing_unknown_falls_back_to_cheapest(self):
        # Falling back to the cheapest tier means an unknown model
        # under-estimates rather than over-estimates — defensible
        # since the user only cares about the ceiling.
        from app.services.gold_llm_service import _lookup_pricing
        self.assertEqual(_lookup_pricing("unknown-model"), (0.15, 0.60))

    def test_compute_estimated_cost_usd_known_numbers(self):
        # 100K input + 50K output on gpt-4o-mini:
        # (100000/1e6)*0.15 + (50000/1e6)*0.60 = 0.015 + 0.03 = 0.045
        from app.services.gold_llm_service import compute_estimated_cost_usd
        cost = compute_estimated_cost_usd(
            model="gpt-4o-mini",
            prompt_tokens=100_000,
            completion_tokens=50_000,
        )
        self.assertAlmostEqual(cost, 0.045, places=4)

    def test_estimate_call_cost_grounded_vs_ungrounded(self):
        from app.services.gold_llm_service import estimate_call_cost_usd
        # Grounding adds reference-material tokens to the prompt
        # estimate, so cost should be strictly higher than ungrounded
        # for the same model + count.
        ungrounded = estimate_call_cost_usd(
            model="gpt-4o-mini",
            count=10,
            grounded=False,
            reference_chunk_count=0,
            reference_total_chars=0,
        )
        grounded = estimate_call_cost_usd(
            model="gpt-4o-mini",
            count=10,
            grounded=True,
            reference_chunk_count=5,
            reference_total_chars=6000,
        )
        self.assertGreater(
            grounded["estimated_cost_usd"],
            ungrounded["estimated_cost_usd"],
        )
        # Sanity: grounded 10-row call on gpt-4o-mini should still
        # cost well under 1¢ — the worst-case spend the user is
        # exposed to per Generate click on the default model.
        self.assertLess(grounded["estimated_cost_usd"], 0.01)

    def test_estimate_call_cost_sonnet_50_rows_stays_bounded(self):
        # The "do not shoot too much" ceiling: max 50 rows on the
        # priciest supported model (sonnet) with full grounding
        # context. The realistic ceiling is ~12¢ — opt-in, deliberate,
        # and surfaced to the user via the cost badge BEFORE clicking
        # Generate. We assert <20¢ as the meaningful guardrail (catches
        # accidental price-table 10× errors without being arbitrary).
        from app.services.gold_llm_service import estimate_call_cost_usd
        out = estimate_call_cost_usd(
            model="claude-sonnet-4-6",
            count=50,
            grounded=True,
            reference_chunk_count=6,
            reference_total_chars=8000,
        )
        self.assertLess(out["estimated_cost_usd"], 0.20)

    def test_estimate_default_call_is_sub_cent(self):
        # Default UX (10 rows, gpt-4o-mini, grounded, typical chunk
        # context) MUST be sub-cent so first-click cost isn't scary.
        from app.services.gold_llm_service import estimate_call_cost_usd
        out = estimate_call_cost_usd(
            model="gpt-4o-mini",
            count=10,
            grounded=True,
            reference_chunk_count=6,
            reference_total_chars=8000,
        )
        self.assertLess(out["estimated_cost_usd"], 0.01)


class GroundingIntegrationTests(unittest.TestCase):
    """End-to-end through the endpoint: confirms grounding context
    flows into the prompt + falls back gracefully when no source
    material exists."""

    @classmethod
    def setUpClass(cls):
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _make_qa_sft_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/project-templates/policy-qa-style/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_grounded_call_when_no_chunks_falls_back_gracefully(self):
        """Fresh project with no cleaned chunks — grounding requested
        but pool is empty. Must not crash; payload reports
        reference_chunk_count=0 so the UI knows grounding was
        ineffective."""
        pid = self._make_qa_sft_project("Grounding Empty Pool")

        async def _fake(**kwargs):
            return CloudLlmResponse(
                content='{"pairs":[{"question":"Q","answer":"A"}]}',
                model="gpt-4o-mini",
                prompt_tokens=200,
                completion_tokens=80,
            )
        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=_fake,
        ), patch(
            "app.services.gold_llm_service._load_project_cleaned_chunks",
            return_value=[],
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 1,
                    "api_key": "sk-test",
                    "ground_in_source": True,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["reference_chunk_count"], 0)
        # Cost still reported, computed from the returned usage.
        self.assertGreater(body["estimated_cost_usd"], 0)

    def test_grounded_call_with_chunks_passes_them_into_the_prompt(self):
        pid = self._make_qa_sft_project("Grounding With Chunks")
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return CloudLlmResponse(
                content=(
                    '{"pairs":[{"question":"Q","answer":"A",'
                    '"source_excerpt":"chunk text snippet"}]}'
                ),
                model="gpt-4o-mini",
                prompt_tokens=300,
                completion_tokens=80,
            )

        fake_chunks = [
            "This is the first cleaned chunk about Topic A.",
            "Second chunk discussing the same Topic A in more detail.",
            "Third chunk introduces Topic B.",
        ]
        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=_capture,
        ), patch(
            "app.services.gold_llm_service._load_project_cleaned_chunks",
            return_value=fake_chunks,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 1,
                    "api_key": "sk-test",
                    "ground_in_source": True,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # All 3 chunks made it (pool < MAX_REFERENCE_CHUNKS).
        self.assertEqual(body["reference_chunk_count"], 3)
        # Source excerpt round-tripped from the LLM response.
        self.assertEqual(
            body["rows"][0]["source_excerpt"], "chunk text snippet",
        )
        # Prompt actually included the REFERENCE MATERIAL section.
        user_prompt = captured.get("user_prompt") or ""
        self.assertIn("REFERENCE MATERIAL", user_prompt)
        self.assertIn("Topic A", user_prompt)
        # Grounding-mode instructions prefer source-grounded answers
        # but allow fallback to project-domain knowledge when refs
        # are thin (softened from the v1 "SKIP that question" wording
        # which made some models return empty pairs / refusal text).
        self.assertIn("Prefer answers grounded in the REFERENCE MATERIAL", user_prompt)
        self.assertIn("source_excerpt", user_prompt)

    def test_ground_in_source_false_does_not_load_chunks(self):
        pid = self._make_qa_sft_project("Grounding Off")
        loader_calls = {"n": 0}

        async def _fake_loader(project_id):
            loader_calls["n"] += 1
            return ["should not be loaded"]

        async def _fake_llm(**kwargs):
            return CloudLlmResponse(
                content='{"pairs":[{"question":"Q","answer":"A"}]}',
                model="gpt-4o-mini",
                prompt_tokens=120,
                completion_tokens=50,
            )

        with patch(
            "app.services.gold_llm_service._load_project_cleaned_chunks",
            side_effect=_fake_loader,
        ), patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=_fake_llm,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 1,
                    "api_key": "sk-test",
                    "ground_in_source": False,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(loader_calls["n"], 0)
        self.assertEqual(resp.json()["reference_chunk_count"], 0)


class CostEstimateEndpointTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _make_qa_sft_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/project-templates/policy-qa-style/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_estimate_endpoint_reports_cost_and_chunk_count(self):
        pid = self._make_qa_sft_project("Cost Estimate Endpoint")
        fake_chunks = [f"chunk text {i}" * 100 for i in range(20)]

        with patch(
            "app.api.gold._load_project_cleaned_chunks",
            return_value=fake_chunks,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm/cost-estimate",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 10,
                    "ground_in_source": True,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # 20-chunk pool, cap at MAX_REFERENCE_CHUNKS (6).
        self.assertEqual(body["ground_in_source_effective"], True)
        self.assertEqual(body["reference_chunk_count"], 6)
        self.assertGreater(body["estimated_cost_usd"], 0)
        # Sanity ceiling — gpt-4o-mini × 10 rows × grounded should be
        # solidly under 1¢.
        self.assertLess(body["estimated_cost_usd"], 0.01)

    def test_estimate_when_no_chunks_marks_effective_false(self):
        pid = self._make_qa_sft_project("Cost Estimate Empty Pool")
        with patch(
            "app.api.gold._load_project_cleaned_chunks",
            return_value=[],
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm/cost-estimate",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 5,
                    "ground_in_source": True,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["ground_in_source_requested"], True)
        self.assertEqual(body["ground_in_source_effective"], False)
        self.assertEqual(body["reference_chunk_count"], 0)


if __name__ == "__main__":
    unittest.main()

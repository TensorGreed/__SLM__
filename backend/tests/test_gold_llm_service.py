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

    def test_unsupported_recipe_returns_recipe_not_supported(self):
        # qa-sft / classification / span-extraction / summarization are
        # supported; other recipes (e.g. code-review, generic-sft) are
        # not yet. Use a code-review project to hit the gate.
        resp = self.client.post(
            "/api/project-templates/code-review-style/instantiate",
            json={"project_name": "Gold LLM Code Review Recipe"},
        )
        # Some test fixtures don't ship a code-review template; fall
        # back to direct project creation + recipe set if so.
        if resp.status_code != 201:
            # Create a project then force-set an unsupported recipe.
            import asyncio
            from app.database import async_session_factory
            from app.services.recipe_apply_service import apply_recipe_to_project
            from app.models.project import Project

            create = self.client.post(
                "/api/projects",
                json={"name": "Gold LLM Unsupported Recipe"},
            )
            self.assertEqual(create.status_code, 201, create.text)
            pid = int(create.json()["id"])

            async def _force_unsupported() -> None:
                async with async_session_factory() as db:
                    try:
                        await apply_recipe_to_project(db, pid, "code-review")
                    except Exception:
                        # No code-review recipe → set raw selected_recipe.
                        project = await db.get(Project, pid)
                        project.selected_recipe = {"recipe_id": "code-review"}
                    await db.commit()
            asyncio.run(_force_unsupported())
        else:
            pid = int(resp.json()["id"])

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
        # Error message lists the supported recipes so the user knows
        # what to switch to.
        msg = detail.get("message", "")
        self.assertIn("qa-sft", msg)
        self.assertIn("classification", msg)

    def test_classification_project_now_supported_happy_path(self):
        # Classification was previously gated as RECIPE_NOT_SUPPORTED.
        # Now it generates classification-shape rows. The ticket-router
        # template seeds gold rows with labels {billing, account,
        # sales, technical, legal} — when the LLM is told to stay in
        # vocabulary, the parser will only accept rows using those
        # labels. The mock has to match.
        pid = self._make_classification_project("Gold LLM Classification Happy")
        fake_content = json.dumps({
            "rows": [
                {"text": "Charge me again for last month?", "label": "billing"},
                {"text": "App crashes on Android.", "label": "technical"},
                {"text": "How do I close my account?", "label": "account"},
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
                    "ground_in_source": False,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["recipe_id"], "classification")
        self.assertEqual(len(body["rows"]), 3)
        self.assertEqual(body["rows"][0]["text"], "Charge me again for last month?")
        self.assertEqual(body["rows"][0]["label"], "billing")
        # qa-sft fields should NOT appear on classification rows.
        self.assertNotIn("question", body["rows"][0])
        self.assertNotIn("answer", body["rows"][0])

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


class ClassificationParserTests(unittest.TestCase):
    """Per-recipe parser tests for classification rows."""

    def _parse(self, content: str, expected_count: int = 2, known_labels=None):
        from app.services.gold_llm_service import _parse_classification_rows
        return _parse_classification_rows(
            content, expected_count, known_labels=known_labels,
        )

    def test_canonical_text_label_shape(self):
        rows = self._parse(
            '{"rows":[{"text":"good","label":"pos"},'
            '{"text":"bad","label":"neg"}]}',
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "good")
        self.assertEqual(rows[0]["label"], "pos")

    def test_input_label_alias(self):
        rows = self._parse(
            '{"rows":[{"input":"x","label":"a"},{"input":"y","label":"b"}]}',
        )
        self.assertEqual(rows[0]["text"], "x")

    def test_known_labels_filters_out_of_vocab(self):
        # LLM emitted 3 rows but only 2 use vocab labels.
        rows = self._parse(
            '{"rows":[{"text":"a","label":"pos"},'
            '{"text":"b","label":"weird-new-label"},'
            '{"text":"c","label":"neg"}]}',
            expected_count=2,
            known_labels=["pos", "neg"],
        )
        self.assertEqual(len(rows), 2)
        labels = sorted(r["label"] for r in rows)
        self.assertEqual(labels, ["neg", "pos"])

    def test_missing_text_or_label_skipped(self):
        # First row has no label, second has no text — both dropped.
        # Use the smallest expected_count compatible with the
        # off-by-25% drift rule (1 row out of 2 expected is within
        # the soft tolerance of max(2, expected//4) = 2).
        rows = self._parse(
            '{"rows":[{"text":"a"},{"label":"x"},'
            '{"text":"c","label":"good"}]}',
            expected_count=2,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "c")

    def test_empty_payload_raises_structured_error(self):
        from app.services.gold_llm_service import GoldGenerationError
        with self.assertRaises(GoldGenerationError) as cm:
            self._parse('{"rows":[]}')
        self.assertEqual(cm.exception.error_code, "LLM_RESPONSE_UNPARSEABLE")


class SpanExtractionParserTests(unittest.TestCase):
    """Per-recipe parser tests for span-extraction rows. Most edge
    cases here are about offset validation — getting these wrong
    poisons eval."""

    def _parse(self, content: str, expected_count: int = 1):
        from app.services.gold_llm_service import _parse_span_rows
        return _parse_span_rows(content, expected_count)

    def test_canonical_text_entities_shape(self):
        rows = self._parse(
            json.dumps({
                "rows": [{
                    "text": "Contact jane@example.com today",
                    "entities": [
                        {"type": "email", "start": 8, "end": 24, "text": "jane@example.com"},
                    ],
                }],
            }),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["entities"]), 1)
        self.assertEqual(rows[0]["entities"][0]["type"], "email")
        self.assertEqual(rows[0]["entities"][0]["text"], "jane@example.com")

    def test_offset_mismatch_drops_span(self):
        # ``text[start:end]`` doesn't match the claimed span text → drop.
        # Row's only span is broken so the row itself is dropped, which
        # makes the parser raise (0 rows, items=1 → unparseable).
        from app.services.gold_llm_service import GoldGenerationError
        bad = json.dumps({
            "rows": [{
                "text": "abcdefghij",
                "entities": [
                    # text[0:3] is "abc" but span claims "xyz" — off.
                    {"type": "tag", "start": 0, "end": 3, "text": "xyz"},
                ],
            }],
        })
        with self.assertRaises(GoldGenerationError):
            self._parse(bad)

    def test_negative_or_end_lte_start_dropped(self):
        # Bad offsets: end<=start, out-of-range → spans dropped.
        # The row had a single bad span, so the row drops, parser errors.
        from app.services.gold_llm_service import GoldGenerationError
        bad = json.dumps({
            "rows": [{
                "text": "hello",
                "entities": [
                    {"type": "t", "start": 3, "end": 3, "text": ""},
                ],
            }],
        })
        with self.assertRaises(GoldGenerationError):
            self._parse(bad)

    def test_row_with_empty_entities_kept(self):
        # No spans is a legitimate negative example.
        rows = self._parse(
            json.dumps({"rows": [{"text": "clean text, no PII", "entities": []}]}),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entities"], [])

    def test_canonicalizes_span_text_from_offsets(self):
        # LLM emitted the span text with extra whitespace; parser
        # canonicalizes to text[start:end].
        rows = self._parse(
            json.dumps({
                "rows": [{
                    "text": "  hello world  ",
                    "entities": [
                        {"type": "greet", "start": 2, "end": 7, "text": "hello"},
                    ],
                }],
            }),
        )
        self.assertEqual(rows[0]["entities"][0]["text"], "hello")


class SummarizationParserTests(unittest.TestCase):

    def _parse(self, content: str, expected_count: int = 1):
        from app.services.gold_llm_service import _parse_summarization_rows
        return _parse_summarization_rows(content, expected_count)

    def test_canonical_document_summary_shape(self):
        rows = self._parse(
            json.dumps({
                "rows": [{
                    "document": "Long document " * 20,
                    "summary": "Short summary.",
                }],
            }),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["summary"], "Short summary.")

    def test_summary_longer_than_document_dropped(self):
        # A "summary" longer than the document is a guaranteed bad
        # eval row — parser drops it. With only one bad row, the
        # parser then errors (no valid rows).
        from app.services.gold_llm_service import GoldGenerationError
        bad = json.dumps({
            "rows": [{
                "document": "short",
                "summary": "this summary is way longer than the document",
            }],
        })
        with self.assertRaises(GoldGenerationError):
            self._parse(bad)

    def test_aliases_article_tldr_supported(self):
        rows = self._parse(
            json.dumps({
                "rows": [{
                    "article": "A long news story about something. " * 10,
                    "tldr": "News story summary.",
                }],
            }),
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("news story", rows[0]["summary"].lower())


class NonQaRecipeEndpointTests(unittest.TestCase):
    """End-to-end through the API for the 3 new recipes — checks
    response shape + recipe_id surfacing."""

    @classmethod
    def setUpClass(cls):
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _instantiate(self, template_slug: str, name: str) -> int:
        resp = self.client.post(
            f"/api/project-templates/{template_slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _patched(self, content: str):
        async def _fake(**kwargs):
            return CloudLlmResponse(
                content=content,
                model=kwargs.get("model", "gpt-4o-mini"),
                prompt_tokens=200,
                completion_tokens=400,
            )
        return _fake

    def test_classification_response_shape(self):
        pid = self._instantiate("ticket-router", "Gold LLM Class Route")
        # Use labels from the ticket-router template vocabulary so the
        # parser's vocab filter keeps the rows.
        content = json.dumps({
            "rows": [
                {"text": "Where's my refund?", "label": "billing"},
                {"text": "App crashes on startup.", "label": "technical"},
                {"text": "How do I cancel?", "label": "billing"},
            ],
        })
        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=self._patched(content),
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 3,
                    "api_key": "sk-test",
                    "ground_in_source": False,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["recipe_id"], "classification")
        for row in body["rows"]:
            self.assertIn("text", row)
            self.assertIn("label", row)

    def test_classification_prompt_includes_known_labels_when_seeded(self):
        # ticket-router template seeds gold rows with labels
        # {billing, account, sales, technical, legal}. The prompt
        # builder reads those + locks the LLM to that vocabulary.
        pid = self._instantiate("ticket-router", "Gold LLM Class Vocab")

        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return CloudLlmResponse(
                content=json.dumps({
                    "rows": [{"text": "Login fails", "label": "technical"}],
                }),
                model="gpt-4o-mini",
                prompt_tokens=200,
                completion_tokens=80,
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
                    "api_key": "sk-test",
                    "ground_in_source": False,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        user_prompt = captured.get("user_prompt") or ""
        # All template-seeded labels appear in the LABEL VOCABULARY section.
        self.assertIn("LABEL VOCABULARY", user_prompt)
        self.assertIn("billing", user_prompt)
        self.assertIn("technical", user_prompt)
        self.assertIn("legal", user_prompt)
        # Labels appear in deterministic (sorted) order so prompt-cache
        # behavior is stable across calls with the same vocab.
        vocab_section = user_prompt.split("LABEL VOCABULARY")[1].split("\n\n")[0]
        labels_in_order = [
            label for label in ["account", "billing", "legal", "sales", "technical"]
            if label in vocab_section
        ]
        self.assertEqual(labels_in_order, sorted(labels_in_order))

    def test_span_extraction_response_shape(self):
        # Look for a PII-style template; fall back to direct recipe set.
        pid = self._make_recipe_project("span-extraction", "Gold LLM Spans")
        content = json.dumps({
            "rows": [{
                "text": "Email me at user@example.com please.",
                "entities": [
                    {"type": "email", "start": 12, "end": 28, "text": "user@example.com"},
                ],
            }],
        })
        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=self._patched(content),
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
        body = resp.json()
        self.assertEqual(body["recipe_id"], "span-extraction")
        self.assertEqual(body["rows"][0]["entities"][0]["type"], "email")

    def test_summarization_response_shape(self):
        pid = self._make_recipe_project("summarization", "Gold LLM Summary")
        content = json.dumps({
            "rows": [{
                "document": "Board meeting on Tuesday covered hiring, "
                            "budget, and the planned office relocation. "
                            "Each item was reviewed in detail.",
                "summary": "Tuesday board reviewed hiring, budget, and relocation.",
            }],
        })
        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=self._patched(content),
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
        body = resp.json()
        self.assertEqual(body["recipe_id"], "summarization")
        self.assertIn("document", body["rows"][0])
        self.assertIn("summary", body["rows"][0])

    def _make_recipe_project(self, recipe_id: str, name: str) -> int:
        """Direct project creation + recipe override for recipes
        that don't have a template (or where the template's recipe
        differs from what the test needs)."""
        import asyncio
        from app.database import async_session_factory
        from app.models.project import Project

        create = self.client.post("/api/projects", json={"name": name})
        self.assertEqual(create.status_code, 201, create.text)
        pid = int(create.json()["id"])

        async def _set_recipe() -> None:
            async with async_session_factory() as db:
                project = await db.get(Project, pid)
                project.selected_recipe = {"recipe_id": recipe_id}
                await db.commit()
        asyncio.run(_set_recipe())
        return pid

    def test_import_preserves_non_qa_shape(self):
        # When the panel saves non-QA rows back via /gold/import,
        # the JSONL on disk must preserve text/label (not drop them
        # like the old QA-only extractor did).
        pid = self._make_recipe_project("classification", "Gold LLM Import Preserve")
        save = self.client.post(
            f"/api/projects/{pid}/gold/import",
            json={
                "pairs": [
                    {"text": "good experience", "label": "positive"},
                    {"text": "terrible app", "label": "negative"},
                ],
                "dataset_type": "gold_dev",
            },
        )
        self.assertEqual(save.status_code, 200, save.text)
        # Read back via the entries endpoint.
        entries = self.client.get(
            f"/api/projects/{pid}/gold/entries",
            params={"dataset_type": "gold_dev"},
        )
        self.assertEqual(entries.status_code, 200, entries.text)
        rows = entries.json()["entries"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "good experience")
        self.assertEqual(rows[0]["label"], "positive")
        # Backwards-compat QA-era fields should still be present as
        # defaults (so old eval code that reads them doesn't crash).
        self.assertIn("difficulty", rows[0])
        # System fields are owned by the service, not the caller.
        self.assertIn("id", rows[0])
        self.assertIn("created_at", rows[0])


class PreviewPromptEndpointTests(unittest.TestCase):
    """Tests for ``POST /generate-via-llm/preview-prompt`` — the
    endpoint that returns the would-be prompts without calling the
    LLM. Powers the panel's 'review & edit prompt before sending' UX."""

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

    def _make_classification_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_preview_returns_user_and_system_prompts_for_qa_sft(self):
        pid = self._make_qa_sft_project("Preview QA happy")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm/preview-prompt",
            json={"count": 5, "ground_in_source": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["recipe_id"], "qa-sft")
        self.assertIn("question/answer", body["user_prompt"])
        self.assertIn("Generate exactly 5", body["user_prompt"])
        # System prompt is non-empty + recipe-appropriate.
        self.assertIn("question/answer pairs", body["system_prompt"])
        self.assertEqual(body["reference_chunk_count"], 0)
        self.assertEqual(body["known_labels"], [])

    def test_preview_for_classification_returns_known_labels(self):
        pid = self._make_classification_project("Preview Class labels")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm/preview-prompt",
            json={"count": 3, "ground_in_source": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["recipe_id"], "classification")
        # ticket-router template seeds 5 labels.
        self.assertIn("billing", body["known_labels"])
        self.assertIn("technical", body["known_labels"])
        # Prompt mentions the vocabulary section.
        self.assertIn("LABEL VOCABULARY", body["user_prompt"])
        # Labels appear in deterministic (sorted) order in the prompt.
        self.assertIn("billing", body["user_prompt"])

    def test_preview_grounded_includes_reference_material(self):
        pid = self._make_qa_sft_project("Preview QA grounded")
        # Mock the chunk loader so we don't depend on data being
        # imported into the test project. Without this, fresh
        # projects have no cleaned chunks and grounding falls back.
        with patch(
            "app.services.gold_llm_service._load_project_cleaned_chunks",
            return_value=[
                "First cleaned chunk about Topic A.",
                "Second chunk about Topic A.",
                "Third chunk about Topic B.",
            ],
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm/preview-prompt",
                json={"count": 2, "ground_in_source": True},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["reference_chunk_count"], 3)
        # Reference section appears in the user prompt.
        self.assertIn("REFERENCE MATERIAL", body["user_prompt"])
        self.assertIn("Topic A", body["user_prompt"])

    def test_preview_recipe_required_returns_400(self):
        # Project without a selected_recipe.
        import asyncio
        from app.database import async_session_factory
        from app.services.recipe_apply_service import clear_recipe_from_project

        pid = self._make_qa_sft_project("Preview no recipe")

        async def _clear() -> None:
            async with async_session_factory() as db:
                await clear_recipe_from_project(db, pid)
                await db.commit()
        asyncio.run(_clear())

        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm/preview-prompt",
            json={"count": 3, "ground_in_source": False},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json().get("detail") or {}
        self.assertEqual(detail.get("error_code"), "RECIPE_REQUIRED")

    def test_preview_count_out_of_range_returns_422(self):
        pid = self._make_qa_sft_project("Preview bad count")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm/preview-prompt",
            json={"count": 999, "ground_in_source": False},
        )
        # Pydantic field-level constraint fires at 422.
        self.assertIn(resp.status_code, (400, 422), resp.text)


class PromptOverrideTests(unittest.TestCase):
    """Tests for the ``user_prompt_override`` / ``system_prompt_override``
    fields on ``/generate-via-llm``. Verifies:
      * the override is what reaches the cloud client (not the built prompt)
      * the classification vocab filter is suspended on parse when the
        user takes control (advanced-user trust contract)
      * either field can be overridden independently"""

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

    def _make_classification_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_user_prompt_override_reaches_the_cloud_client(self):
        pid = self._make_qa_sft_project("Override user-prompt")
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return CloudLlmResponse(
                content='{"pairs":[{"question":"Q","answer":"A"}]}',
                model="gpt-4o-mini",
                prompt_tokens=50,
                completion_tokens=20,
            )

        custom_prompt = (
            "CUSTOM PROMPT — this is what the advanced user wrote. "
            'Return JSON: {"pairs":[{"question":"X","answer":"Y"}]}.'
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
                    "api_key": "sk-test",
                    "user_prompt_override": custom_prompt,
                    "ground_in_source": False,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        # The literal override string is what reached the cloud client —
        # NOT the service's normally-built prompt.
        self.assertEqual(captured.get("user_prompt"), custom_prompt)
        # System prompt fell back to the service default.
        self.assertIn("question/answer", captured.get("system_prompt") or "")

    def test_system_prompt_override_alone_does_not_affect_user_prompt(self):
        pid = self._make_qa_sft_project("Override system-prompt")
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return CloudLlmResponse(
                content='{"pairs":[{"question":"Q","answer":"A"}]}',
                model="gpt-4o-mini",
                prompt_tokens=50,
                completion_tokens=20,
            )

        custom_system = "You are a helpful tester. Output exactly one Q/A pair."
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
                    "api_key": "sk-test",
                    "system_prompt_override": custom_system,
                    "ground_in_source": False,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(captured.get("system_prompt"), custom_system)
        # User prompt still built by the service (project name appears).
        self.assertIn("Override system-prompt", captured.get("user_prompt") or "")

    def test_user_override_suspends_classification_vocab_filter(self):
        # Without override: the parser drops out-of-vocab labels. With
        # override: the user has taken control, vocab filter is off,
        # arbitrary labels pass through.
        pid = self._make_classification_project("Override class vocab")
        # ticket-router seeds {billing, account, sales, technical, legal}.
        # The mock emits "spam" (not in vocab) — without override this
        # row would be filtered out and the parser would error.
        async def _fake(**kwargs):
            return CloudLlmResponse(
                content=json.dumps({
                    "rows": [{"text": "buy now!!!", "label": "spam"}],
                }),
                model="gpt-4o-mini",
                prompt_tokens=80,
                completion_tokens=30,
            )

        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=_fake,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 1,
                    "api_key": "sk-test",
                    "user_prompt_override": (
                        "Generate one classification row using labels of "
                        'your choice. Return {"rows":[{"text":"...","label":"..."}]}'
                    ),
                    "ground_in_source": False,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # The out-of-template-vocab label "spam" survived because we
        # suspended the vocab filter under override.
        self.assertEqual(body["rows"][0]["label"], "spam")


class DifficultyAndTrapLabelTests(unittest.TestCase):
    """Tests for the per-row difficulty + is_hallucination_trap labels
    on qa-sft generations — both the explicit distribution path and
    the always-on labeling that fixes the silent-medium bug."""

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

    # ── Parser-level: pure-function tests ─────────────────────────

    def test_parser_extracts_difficulty_and_trap_from_response(self):
        from app.services.gold_llm_service import _parse_qa_payload
        content = json.dumps({
            "pairs": [
                {"question": "Q1", "answer": "A1", "difficulty": "easy",
                 "is_hallucination_trap": False},
                {"question": "Q2", "answer": "A2", "difficulty": "hard",
                 "is_hallucination_trap": True},
            ],
        })
        rows = _parse_qa_payload(content, expected_count=2)
        self.assertEqual(rows[0].difficulty, "easy")
        self.assertEqual(rows[0].is_hallucination_trap, False)
        self.assertEqual(rows[1].difficulty, "hard")
        self.assertEqual(rows[1].is_hallucination_trap, True)

    def test_parser_defaults_difficulty_to_medium_and_trap_to_false(self):
        # LLM didn't include the fields — defaults apply.
        from app.services.gold_llm_service import _parse_qa_payload
        content = json.dumps({
            "pairs": [{"question": "Q", "answer": "A"}],
        })
        rows = _parse_qa_payload(content, expected_count=1)
        self.assertEqual(rows[0].difficulty, "medium")
        self.assertEqual(rows[0].is_hallucination_trap, False)

    def test_parser_normalizes_difficulty_synonyms(self):
        # LLM emits 'simple' / 'tough' / 'expert' / etc. — parser maps
        # them to the canonical {easy, medium, hard} set.
        from app.services.gold_llm_service import _parse_qa_payload
        content = json.dumps({
            "pairs": [
                {"question": "Q1", "answer": "A1", "difficulty": "Simple"},
                {"question": "Q2", "answer": "A2", "difficulty": "TOUGH"},
                {"question": "Q3", "answer": "A3", "difficulty": "expert"},
                {"question": "Q4", "answer": "A4", "difficulty": "bizarre"},
            ],
        })
        rows = _parse_qa_payload(content, expected_count=4)
        self.assertEqual(rows[0].difficulty, "easy")
        self.assertEqual(rows[1].difficulty, "hard")
        self.assertEqual(rows[2].difficulty, "hard")
        # Unrecognized → falls back to medium (not None / not the raw).
        self.assertEqual(rows[3].difficulty, "medium")

    def test_parser_trap_field_string_true_is_coerced(self):
        # LLM emits "true" (string) instead of true (bool) — accepted.
        from app.services.gold_llm_service import _parse_qa_payload
        content = json.dumps({
            "pairs": [
                {"question": "Q", "answer": "I don't know",
                 "is_hallucination_trap": "true"},
            ],
        })
        rows = _parse_qa_payload(content, expected_count=1)
        self.assertEqual(rows[0].is_hallucination_trap, True)

    # ── End-to-end: distribution drives prompt + saves labeled rows ──

    def test_distribution_enumerates_mix_in_user_prompt(self):
        pid = self._make_qa_sft_project("Distribution prompt mix")
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return CloudLlmResponse(
                content=json.dumps({
                    "pairs": [
                        # Match the requested mix so the count-drift
                        # check passes (5 + 3 + 2 + 2 = 12 total).
                        *[{"question": f"E{i}", "answer": f"A{i}",
                           "difficulty": "easy",
                           "is_hallucination_trap": False} for i in range(5)],
                        *[{"question": f"M{i}", "answer": f"A{i}",
                           "difficulty": "medium",
                           "is_hallucination_trap": False} for i in range(3)],
                        *[{"question": f"H{i}", "answer": f"A{i}",
                           "difficulty": "hard",
                           "is_hallucination_trap": False} for i in range(2)],
                        *[{"question": f"T{i}", "answer": "I don't know",
                           "difficulty": "hard",
                           "is_hallucination_trap": True} for i in range(2)],
                    ],
                }),
                model="gpt-4o-mini",
                prompt_tokens=300,
                completion_tokens=600,
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
                    # count is in Pydantic-valid range but FAR from
                    # the distribution total (12). Proves the service
                    # used distribution.total (12) — if it had used
                    # count (5), the parser's drift check would reject
                    # the 12-row response with COUNT_MISMATCH.
                    "count": 5,
                    "api_key": "sk-test",
                    "ground_in_source": False,
                    "distribution": {
                        "easy": 5,
                        "medium": 3,
                        "hard": 2,
                        "hallucination_traps": 2,
                    },
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        user_prompt = captured.get("user_prompt") or ""
        # Prompt enumerates the breakdown.
        self.assertIn("5 EASY", user_prompt)
        self.assertIn("3 MEDIUM", user_prompt)
        self.assertIn("2 HARD", user_prompt)
        self.assertIn("2 HALLUCINATION TRAP", user_prompt)
        # Total mentioned ("12 question/answer pairs total").
        self.assertIn("12 question/answer pairs", user_prompt)

    def test_distribution_response_rows_carry_per_row_labels(self):
        pid = self._make_qa_sft_project("Distribution row labels")
        fake_content = json.dumps({
            "pairs": [
                {"question": "Easy Q", "answer": "Easy A",
                 "difficulty": "easy", "is_hallucination_trap": False},
                {"question": "Trap Q", "answer": "I don't know",
                 "difficulty": "hard", "is_hallucination_trap": True},
            ],
        })

        async def _fake(**kwargs):
            return CloudLlmResponse(
                content=fake_content,
                model="gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=100,
            )

        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=_fake,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 2,
                    "api_key": "sk-test",
                    "ground_in_source": False,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["rows"][0]["difficulty"], "easy")
        self.assertEqual(body["rows"][0]["is_hallucination_trap"], False)
        self.assertEqual(body["rows"][1]["difficulty"], "hard")
        self.assertEqual(body["rows"][1]["is_hallucination_trap"], True)

    def test_distribution_out_of_range_returns_400(self):
        pid = self._make_qa_sft_project("Distribution oor")
        # Sum = 60 > 50 → reject.
        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "count": 10,
                "api_key": "sk-test",
                "ground_in_source": False,
                "distribution": {
                    "easy": 30, "medium": 20, "hard": 10, "hallucination_traps": 0,
                },
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json().get("detail") or {}
        self.assertEqual(detail.get("error_code"), "DISTRIBUTION_OUT_OF_RANGE")

    def test_distribution_total_zero_returns_400(self):
        pid = self._make_qa_sft_project("Distribution zero")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/generate-via-llm",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "count": 10,
                "api_key": "sk-test",
                "distribution": {
                    "easy": 0, "medium": 0, "hard": 0, "hallucination_traps": 0,
                },
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json().get("detail") or {}
        self.assertEqual(detail.get("error_code"), "DISTRIBUTION_OUT_OF_RANGE")

    def test_distribution_ignored_for_non_qa_recipes(self):
        # Classification project + a distribution → distribution is
        # silently ignored, prompt should NOT enumerate the mix.
        resp = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={"project_name": "Distribution non-qa"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return CloudLlmResponse(
                content=json.dumps({
                    "rows": [{"text": "good", "label": "billing"}],
                }),
                model="gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=100,
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
                    "api_key": "sk-test",
                    "ground_in_source": False,
                    "distribution": {
                        "easy": 5, "medium": 3, "hard": 2, "hallucination_traps": 2,
                    },
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        # Prompt should NOT enumerate the mix (distribution silently
        # ignored for non-qa-sft).
        user_prompt = captured.get("user_prompt") or ""
        self.assertNotIn("HALLUCINATION TRAP", user_prompt)
        self.assertNotIn("5 EASY", user_prompt)

    def _make_qa_sft_project_clean(self, name: str) -> int:
        """Create a qa-sft project directly (no template seeding +
        lock) so we can write fresh rows into gold_dev."""
        import asyncio
        from app.database import async_session_factory
        from app.models.project import Project

        resp = self.client.post("/api/projects", json={"name": name})
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])

        async def _set_recipe() -> None:
            async with async_session_factory() as db:
                project = await db.get(Project, pid)
                project.selected_recipe = {"recipe_id": "qa-sft"}
                await db.commit()
        asyncio.run(_set_recipe())
        return pid

    def test_import_round_trips_difficulty_and_trap_fields(self):
        # Generate + save in one round-trip; read back via /gold/entries
        # to confirm the labels persisted into the JSONL. Use a clean
        # project (no template seeding/locking) so gold_dev is writable.
        pid = self._make_qa_sft_project_clean("Round-trip labels")
        fake_content = json.dumps({
            "pairs": [
                {"question": "Easy Q", "answer": "Easy A",
                 "difficulty": "easy", "is_hallucination_trap": False},
                {"question": "Trap Q", "answer": "I don't know",
                 "difficulty": "hard", "is_hallucination_trap": True},
            ],
        })

        async def _fake(**kwargs):
            return CloudLlmResponse(
                content=fake_content,
                model="gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=100,
            )

        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=_fake,
        ):
            gen = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "count": 2,
                    "api_key": "sk-test",
                    "ground_in_source": False,
                },
            )
        self.assertEqual(gen.status_code, 200, gen.text)
        # Save the rows as-is (mirror what the panel does — pass the
        # full per-row dict so labels survive).
        save = self.client.post(
            f"/api/projects/{pid}/gold/import",
            json={
                "pairs": gen.json()["rows"],
                "dataset_type": "gold_dev",
            },
        )
        self.assertEqual(save.status_code, 200, save.text)
        # Read back. Templates seed a lot of rows; just check that our
        # two new entries (with their labels intact) are present.
        entries = self.client.get(
            f"/api/projects/{pid}/gold/entries",
            params={"dataset_type": "gold_dev"},
        )
        self.assertEqual(entries.status_code, 200, entries.text)
        rows = entries.json()["entries"]
        # Find by question.
        easy_row = next(r for r in rows if r.get("question") == "Easy Q")
        trap_row = next(r for r in rows if r.get("question") == "Trap Q")
        self.assertEqual(easy_row["difficulty"], "easy")
        self.assertEqual(easy_row["is_hallucination_trap"], False)
        self.assertEqual(trap_row["difficulty"], "hard")
        self.assertEqual(trap_row["is_hallucination_trap"], True)


class SavedKeyEndpointTests(unittest.TestCase):
    """Tests for the new GET/PUT/DELETE /generate-via-llm/saved-key
    endpoints — the panel-local stored-key UX. The plumbing for
    stored-key fallback in /generate-via-llm itself is already
    exercised by the existing API_KEY_REQUIRED + happy-path tests;
    here we test the explicit saved-key surface."""

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

    # ── GET ───────────────────────────────────────────────────────

    def test_get_saved_key_returns_false_when_no_secret_stored(self):
        pid = self._make_qa_sft_project("Saved Key GET empty")
        resp = self.client.get(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            params={"provider": "openai"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["has_stored_key"], False)
        self.assertIsNone(body["value_hint"])

    def test_get_saved_key_returns_hint_when_secret_exists(self):
        pid = self._make_qa_sft_project("Saved Key GET present")
        # Seed a secret via PUT first.
        put_resp = self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "anthropic", "api_key": "sk-ant-test-123456"},
        )
        self.assertEqual(put_resp.status_code, 200, put_resp.text)

        get_resp = self.client.get(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            params={"provider": "anthropic"},
        )
        self.assertEqual(get_resp.status_code, 200, get_resp.text)
        body = get_resp.json()
        self.assertTrue(body["has_stored_key"])
        # Hint format from _mask_secret: first 2 + stars + last 2 chars.
        self.assertIsNotNone(body["value_hint"])
        self.assertNotIn("sk-ant-test-123456", body["value_hint"])
        # Raw API key MUST NOT appear anywhere in the response.
        self.assertNotIn("sk-ant-test-123456", get_resp.text)

    def test_get_saved_key_isolated_per_provider(self):
        # Seeding the OpenAI secret must not appear under deepseek.
        pid = self._make_qa_sft_project("Saved Key GET per-provider")
        self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "openai", "api_key": "sk-openai-only"},
        )
        # Deepseek lookup should still return empty.
        resp = self.client.get(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            params={"provider": "deepseek"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertFalse(resp.json()["has_stored_key"])

    # ── PUT ───────────────────────────────────────────────────────

    def test_put_saved_key_creates_and_then_replaces(self):
        pid = self._make_qa_sft_project("Saved Key PUT create+replace")
        # Initial PUT.
        r1 = self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "deepseek", "api_key": "sk-deepseek-aaaaaa"},
        )
        self.assertEqual(r1.status_code, 200, r1.text)
        hint_1 = r1.json()["value_hint"]
        # Replace with a new key.
        r2 = self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "deepseek", "api_key": "sk-deepseek-zzzzzz"},
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        hint_2 = r2.json()["value_hint"]
        self.assertNotEqual(hint_1, hint_2)
        # Listing all secrets should show only one deepseek row, not two.
        list_resp = self.client.get(f"/api/projects/{pid}/secrets")
        self.assertEqual(list_resp.status_code, 200)
        rows = [
            s for s in list_resp.json()["secrets"]
            if s["provider"] == "cloud_llm_deepseek"
        ]
        self.assertEqual(len(rows), 1)

    def test_put_saved_key_rejects_short_api_key(self):
        # min_length=8 — guards against silently overwriting a real key
        # with a typo'd stub like "sk-" or "abc".
        pid = self._make_qa_sft_project("Saved Key PUT too short")
        resp = self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "openai", "api_key": "sk-"},
        )
        # Pydantic field validation → 422.
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_put_saved_key_rejects_unknown_provider(self):
        pid = self._make_qa_sft_project("Saved Key PUT bad provider")
        resp = self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "cohere", "api_key": "abcdefghij"},
        )
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_put_response_does_not_leak_raw_key(self):
        pid = self._make_qa_sft_project("Saved Key PUT no leak")
        raw = "sk-leak-canary-xyz12345"
        resp = self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "openai", "api_key": raw},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertNotIn(raw, resp.text)

    # ── DELETE ────────────────────────────────────────────────────

    def test_delete_saved_key_removes_existing_secret(self):
        pid = self._make_qa_sft_project("Saved Key DELETE exists")
        self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "openai", "api_key": "sk-delete-me-abc"},
        )
        resp = self.client.delete(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            params={"provider": "openai"},
        )
        self.assertEqual(resp.status_code, 204, resp.text)
        # GET now reports no stored key.
        get_resp = self.client.get(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            params={"provider": "openai"},
        )
        self.assertFalse(get_resp.json()["has_stored_key"])

    def test_delete_saved_key_is_idempotent_when_missing(self):
        # No secret stored — DELETE still returns 204 so the panel's
        # Remove button never lies after a stale cache.
        pid = self._make_qa_sft_project("Saved Key DELETE idempotent")
        resp = self.client.delete(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            params={"provider": "anthropic"},
        )
        self.assertEqual(resp.status_code, 204, resp.text)

    # ── End-to-end: stored key drives a generate-via-llm call ─────

    def test_stored_key_falls_back_into_generate_call(self):
        # PUT a key, then call /generate-via-llm WITHOUT inline api_key
        # — the call must succeed (would 400 with API_KEY_REQUIRED if
        # the fallback wired up wrong).
        pid = self._make_qa_sft_project("Saved Key drives generate")
        self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "openai", "api_key": "sk-stored-fallback"},
        )

        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return CloudLlmResponse(
                content='{"pairs":[{"question":"Q","answer":"A"}]}',
                model="gpt-4o-mini",
                prompt_tokens=10,
                completion_tokens=10,
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
                    # api_key omitted — stored key must be used.
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        # The stored key reached the cloud client.
        self.assertEqual(captured.get("api_key"), "sk-stored-fallback")

    def test_deepseek_secret_used_when_api_url_is_deepseek(self):
        # Deepseek arrives on the wire as provider=openai +
        # api_url=<deepseek host>. The fallback must consult the
        # cloud_llm_deepseek secret, NOT cloud_llm_openai, even though
        # provider says openai.
        pid = self._make_qa_sft_project("Deepseek secret routing")
        # Seed both keys so we can prove the right one was picked.
        self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "openai", "api_key": "sk-openai-wrong"},
        )
        self.client.put(
            f"/api/projects/{pid}/gold/generate-via-llm/saved-key",
            json={"provider": "deepseek", "api_key": "sk-deepseek-right"},
        )

        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return CloudLlmResponse(
                content='{"pairs":[{"question":"Q","answer":"A"}]}',
                model="deepseek-chat",
                prompt_tokens=10,
                completion_tokens=10,
            )

        with patch(
            "app.services.gold_llm_service.call_openai_chat",
            side_effect=_capture,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/gold/generate-via-llm",
                json={
                    "provider": "openai",
                    "model": "deepseek-chat",
                    "count": 1,
                    "api_url": "https://api.deepseek.com/v1/chat/completions",
                    # api_key omitted — stored deepseek key must be used.
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(captured.get("api_key"), "sk-deepseek-right")


if __name__ == "__main__":
    unittest.main()

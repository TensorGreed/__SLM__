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


if __name__ == "__main__":
    unittest.main()

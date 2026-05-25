"""Tests for the vLLM synth backend (USER-SUCCESS Epic 5 Phase 5c).

Covers:
- Constructor reads from settings + accepts explicit overrides.
- ``is_available`` returns False without VLLM_API_URL, False on unreachable,
  True on HTTP 200, True on 401/403 (auth-required is still "reachable").
- ``describe()`` includes the pinned model when set.
- ``complete()`` happy path returns the OpenAI-shaped content.
- ``complete()`` forwards ``response_schema`` as ``response_format=json_schema``
  end-to-end (this is the whole point of the vLLM backend vs. Ollama:
  vLLM honors structured outputs, Ollama silently ignores them).
- ``complete()`` omits ``response_format`` when no schema is passed.
- ``complete()`` raises ``SynthBackendError`` for: missing URL, missing
  model, timeout, HTTP 401 (with key hint), HTTP 404 (with model
  hint), HTTP 400 with schema-rejection hint, generic HTTPError,
  JSON-decode error, unexpected response shape.
- Registry ordering: vLLM is registered LAST so existing auto-pick
  behavior is unchanged.
- ``pick_backend("vllm:<model>")`` routes correctly.

Mocking pattern mirrors test_synth_nemo_backend.py.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

import httpx  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.synth_backends import (  # noqa: E402
    BACKEND_REGISTRY,
    NemoBackend,
    OllamaBackend,
    SynthBackendError,
    VllmBackend,
    pick_backend,
)


# ─────────────────────────────────────────────────────────────────────
# Test doubles (mirrors test_synth_nemo_backend.py)
# ─────────────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = "",
    ):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text or (json.dumps(self._json) if self._json else "")

    def json(self):
        if self._json == "__decode_error__":
            raise ValueError("not json")
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            req = httpx.Request("POST", "http://test/v1/chat/completions")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=req,
                response=httpx.Response(
                    self.status_code, text=self.text, request=req
                ),
            )


class _FakeAsyncClient:
    def __init__(self, handler):
        self._handler = handler

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        return self._handler(url=url, **kwargs)


def _patch_async_client(handler):
    return patch.object(httpx, "AsyncClient", _FakeAsyncClient(handler))


class _SettingsCtx:
    def __init__(self, overrides):
        self._overrides = overrides
        self._previous: dict = {}

    def __enter__(self):
        for key, value in self._overrides.items():
            self._previous[key] = getattr(settings, key)
            setattr(settings, key, value)
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self._previous.items():
            setattr(settings, key, value)
        return False


def _settings_overrides(**kwargs):
    return _SettingsCtx(kwargs)


# ─────────────────────────────────────────────────────────────────────
# Reachability + describe + constructor
# ─────────────────────────────────────────────────────────────────────


class VllmBackendReachabilityTests(unittest.TestCase):
    def test_is_available_false_when_url_unset(self):
        with _settings_overrides(VLLM_API_URL=""):
            self.assertFalse(VllmBackend.is_available())

    def test_is_available_true_when_url_returns_200(self):
        def fake_get(url, headers=None, timeout=None):
            return _FakeResponse(status_code=200, json_data={"data": []})

        with _settings_overrides(VLLM_API_URL="http://localhost:8000"):
            with patch.object(httpx, "get", side_effect=fake_get):
                self.assertTrue(VllmBackend.is_available())

    def test_is_available_treats_auth_failures_as_reachable(self):
        for status in (401, 403):
            with self.subTest(status=status):
                with _settings_overrides(VLLM_API_URL="http://test"):
                    with patch.object(
                        httpx,
                        "get",
                        return_value=_FakeResponse(status_code=status),
                    ):
                        self.assertTrue(VllmBackend.is_available())

    def test_is_available_false_when_endpoint_unreachable(self):
        def boom(url, headers=None, timeout=None):
            raise httpx.ConnectError("no route to host")

        with _settings_overrides(VLLM_API_URL="http://localhost:8000"):
            with patch.object(httpx, "get", side_effect=boom):
                self.assertFalse(VllmBackend.is_available())

    def test_describe_includes_pinned_model(self):
        backend = VllmBackend(
            host="http://vllm",
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        )
        self.assertEqual(
            backend.describe(),
            "vllm:meta-llama/Meta-Llama-3.1-8B-Instruct",
        )

    def test_describe_falls_back_to_name_without_model(self):
        with _settings_overrides(VLLM_DEFAULT_MODEL=""):
            backend = VllmBackend(host="http://vllm")
            self.assertEqual(backend.describe(), "vllm")


# ─────────────────────────────────────────────────────────────────────
# complete()
# ─────────────────────────────────────────────────────────────────────


class VllmBackendCompleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_happy_path_returns_content(self):
        captured: dict = {}

        def handler(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            return _FakeResponse(
                status_code=200,
                json_data={
                    "choices": [
                        {"message": {"content": "vllm-generated"}}
                    ]
                },
            )

        backend = VllmBackend(
            host="http://vllm",
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            api_key="vllm-secret",
        )
        with _patch_async_client(handler):
            out = await backend.complete(
                "Generate 3 paraphrases.",
                system_prompt="You are a data labeler.",
                max_tokens=512,
                temperature=0.3,
            )

        self.assertEqual(out, "vllm-generated")
        self.assertEqual(captured["url"], "http://vllm/v1/chat/completions")
        self.assertEqual(
            captured["headers"]["Authorization"], "Bearer vllm-secret"
        )
        msgs = captured["json"]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(captured["json"]["max_tokens"], 512)
        self.assertEqual(captured["json"]["temperature"], 0.3)
        # No schema passed → no response_format on the wire.
        self.assertNotIn("response_format", captured["json"])

    async def test_complete_forwards_response_schema_end_to_end(self):
        """The whole reason vLLM exists as a backend: it honors
        ``response_format=json_schema`` (vLLM uses xgrammar/outlines
        for constrained decoding). Verify the schema lands on the
        wire in OpenAI Structured-Outputs shape with ``strict: true``."""
        captured: dict = {}

        def handler(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _FakeResponse(
                status_code=200,
                json_data={
                    "choices": [{"message": {"content": '{"x": 1}'}}]
                },
            )

        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "label": {"type": "string", "enum": ["rare"]},
            },
            "required": ["text", "label"],
            "additionalProperties": False,
        }
        backend = VllmBackend(host="http://vllm", model="m")
        with _patch_async_client(handler):
            await backend.complete("Generate.", response_schema=schema)

        rf = captured["json"].get("response_format")
        self.assertIsNotNone(rf, "vLLM payload must carry response_format")
        self.assertEqual(rf["type"], "json_schema")
        self.assertEqual(rf["json_schema"]["name"], "synth_row")
        self.assertEqual(rf["json_schema"]["schema"], schema)
        self.assertTrue(rf["json_schema"]["strict"])

    async def test_complete_raises_when_url_unset(self):
        with _settings_overrides(VLLM_API_URL=""):
            backend = VllmBackend(model="some/model")
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
            self.assertIn("VLLM_API_URL", str(cm.exception))

    async def test_complete_raises_when_model_unset(self):
        with _settings_overrides(VLLM_DEFAULT_MODEL=""):
            backend = VllmBackend(host="http://vllm")
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
            self.assertIn("VLLM_DEFAULT_MODEL", str(cm.exception))

    async def test_complete_wraps_timeout(self):
        def handler(url, **kwargs):
            raise httpx.TimeoutException("timed out")

        backend = VllmBackend(host="http://vllm", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        msg = str(cm.exception)
        self.assertIn("timed out", msg.lower())
        self.assertIn("VLLM_TIMEOUT_SECONDS", msg)

    async def test_complete_wraps_401_with_api_key_hint(self):
        def handler(url, **kwargs):
            return _FakeResponse(status_code=401, text="unauthorized")

        backend = VllmBackend(host="http://vllm", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        self.assertIn("VLLM_API_KEY", str(cm.exception))

    async def test_complete_wraps_404_with_model_listing_hint(self):
        def handler(url, **kwargs):
            return _FakeResponse(status_code=404, text="model not found")

        backend = VllmBackend(host="http://vllm", model="missing/model")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        msg = str(cm.exception)
        self.assertIn("missing/model", msg)
        self.assertIn("/v1/models", msg)

    async def test_complete_wraps_400_schema_rejection_with_hint(self):
        """vLLM rejects unsupported JSON Schema features (e.g. ``$ref``,
        certain enum shapes) with HTTP 400. The wrapper should surface
        an actionable hint pointing the playbook author at
        response_schema()."""
        def handler(url, **kwargs):
            return _FakeResponse(
                status_code=400,
                text="Invalid json_schema: unsupported feature $ref",
            )

        backend = VllmBackend(host="http://vllm", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete(
                    "hi",
                    response_schema={"$ref": "#/definitions/Bad"},
                )
        msg = str(cm.exception)
        self.assertIn("rejected the JSON Schema", msg)
        self.assertIn("response_schema", msg)

    async def test_complete_wraps_generic_http_error(self):
        def handler(url, **kwargs):
            raise httpx.ConnectError("connection reset")

        backend = VllmBackend(host="http://vllm", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        self.assertIn("connection reset", str(cm.exception))

    async def test_complete_wraps_json_decode_error(self):
        def handler(url, **kwargs):
            return _FakeResponse(status_code=200, json_data="__decode_error__")  # type: ignore[arg-type]

        backend = VllmBackend(host="http://vllm", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        self.assertIn("non-JSON", str(cm.exception))

    async def test_complete_wraps_unexpected_response_shape(self):
        def handler(url, **kwargs):
            return _FakeResponse(
                status_code=200,
                json_data={"unexpected": "shape"},
            )

        backend = VllmBackend(host="http://vllm", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        self.assertIn("unexpected response shape", str(cm.exception))


# ─────────────────────────────────────────────────────────────────────
# Registry + pick_backend
# ─────────────────────────────────────────────────────────────────────


class VllmRegistryTests(unittest.TestCase):
    def test_registry_includes_vllm_last(self):
        """vLLM must land AFTER Ollama, Teacher, AND NeMo. Auto-pick
        for existing installs (Ollama users) is unchanged; same goes
        for power users who already configured NeMo."""
        names = [c.name for c in BACKEND_REGISTRY]
        self.assertIn("vllm", names)
        self.assertEqual(names[-1], "vllm")
        # NeMo is still second-to-last (Phase 5a ordering preserved).
        self.assertEqual(names[-2], "nemo")

    def test_pick_backend_routes_explicit_vllm_pin(self):
        with _settings_overrides(VLLM_API_URL="http://vllm"):
            with patch.object(
                httpx,
                "get",
                return_value=_FakeResponse(status_code=200, json_data={}),
            ):
                backend = pick_backend(
                    "vllm:meta-llama/Meta-Llama-3.1-8B-Instruct"
                )
        self.assertIsInstance(backend, VllmBackend)
        self.assertEqual(
            backend.describe(),
            "vllm:meta-llama/Meta-Llama-3.1-8B-Instruct",
        )

    def test_pick_backend_auto_pick_skips_vllm_when_ollama_available(self):
        """Auto-pick walks BACKEND_REGISTRY in order. Even with vLLM +
        NeMo + Ollama all reachable, Ollama wins — existing users see
        no change."""
        with _settings_overrides(
            VLLM_API_URL="http://vllm",
            NEMO_API_URL="http://nim",
        ):
            with (
                patch.object(OllamaBackend, "is_available", return_value=True),
                patch.object(NemoBackend, "is_available", return_value=True),
                patch.object(VllmBackend, "is_available", return_value=True),
            ):
                backend = pick_backend(None)
        self.assertIsInstance(backend, OllamaBackend)


if __name__ == "__main__":
    unittest.main()

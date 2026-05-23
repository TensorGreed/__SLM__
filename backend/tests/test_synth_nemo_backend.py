"""Tests for the NeMo Data Designer / NIM synth backend (USER-SUCCESS Epic 5 Phase 5a).

Covers:
- Constructor reads from settings + accepts explicit overrides.
- ``is_available`` returns False without NEMO_API_URL, False on unreachable,
  True on HTTP 200, True on 401/403 (auth-required is still "reachable").
- ``describe()`` includes the pinned model when set.
- ``complete()`` happy path returns the OpenAI-shaped content.
- ``complete()`` raises ``SynthBackendError`` for: missing URL,
  missing model, timeout, HTTP 401 (with key hint), HTTP 404 (with
  model-name hint), generic HTTPError, JSON-decode error, unexpected
  response shape.

The httpx-mocking pattern mirrors ``test_phase61_synthetic_teacher_parsing.py``:
patch ``httpx.AsyncClient`` (async path) or ``httpx.get`` (sync
reachability probe) with a fake context manager.
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
    pick_backend,
)


# ─────────────────────────────────────────────────────────────────────
# Test doubles
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
            # Sentinel: simulate a JSON decode failure (e.g. HTML 502
            # from a reverse proxy).
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
    """Async context manager mimicking ``httpx.AsyncClient`` for the
    purposes of mocking ``NemoBackend.complete``. The ``handler`` is a
    callable that takes the POST kwargs and returns a ``_FakeResponse``
    (or raises an httpx error)."""

    def __init__(self, handler):
        self._handler = handler

    def __init_subclass__(cls, **kwargs):  # pragma: no cover
        super().__init_subclass__(**kwargs)

    def __call__(self, *args, **kwargs):
        # Some call sites instantiate AsyncClient with a timeout kwarg;
        # we ignore it.
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        return self._handler(url=url, **kwargs)


def _patch_async_client(handler):
    fake_factory = _FakeAsyncClient(handler)
    return patch.object(httpx, "AsyncClient", fake_factory)


def _settings_overrides(**kwargs):
    """Patch every NEMO_* attribute on the settings singleton + clean
    up after the test. ``unittest.mock.patch.object`` doesn't compose
    nicely for 4 attributes so we apply them directly with a finally."""
    return _SettingsCtx(kwargs)


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


# ─────────────────────────────────────────────────────────────────────
# Reachability + describe + constructor
# ─────────────────────────────────────────────────────────────────────


class NemoBackendReachabilityTests(unittest.TestCase):
    def test_is_available_false_when_url_unset(self):
        with _settings_overrides(NEMO_API_URL=""):
            self.assertFalse(NemoBackend.is_available())

    def test_is_available_true_when_url_returns_200(self):
        def fake_get(url, headers=None, timeout=None):
            return _FakeResponse(status_code=200, json_data={"data": []})

        with _settings_overrides(NEMO_API_URL="http://localhost:8000"):
            with patch.object(httpx, "get", side_effect=fake_get):
                self.assertTrue(NemoBackend.is_available())

    def test_is_available_treats_auth_failures_as_reachable(self):
        # The picker shouldn't hide NeMo when the user has the
        # endpoint but a bad/missing key — they'd think NeMo isn't
        # supported when it really is + just needs configuration.
        for status in (401, 403):
            with self.subTest(status=status):
                with _settings_overrides(NEMO_API_URL="http://test"):
                    with patch.object(
                        httpx,
                        "get",
                        return_value=_FakeResponse(status_code=status),
                    ):
                        self.assertTrue(NemoBackend.is_available())

    def test_is_available_false_when_endpoint_unreachable(self):
        def boom(url, headers=None, timeout=None):
            raise httpx.ConnectError("no route to host")

        with _settings_overrides(NEMO_API_URL="http://localhost:8000"):
            with patch.object(httpx, "get", side_effect=boom):
                self.assertFalse(NemoBackend.is_available())

    def test_describe_includes_pinned_model(self):
        backend = NemoBackend(
            host="http://nim", model="meta/llama-3.1-70b-instruct"
        )
        self.assertEqual(backend.describe(), "nemo:meta/llama-3.1-70b-instruct")

    def test_describe_falls_back_to_name_without_model(self):
        with _settings_overrides(NEMO_DEFAULT_MODEL=""):
            backend = NemoBackend(host="http://nim")
            self.assertEqual(backend.describe(), "nemo")


# ─────────────────────────────────────────────────────────────────────
# complete()
# ─────────────────────────────────────────────────────────────────────


class NemoBackendCompleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_happy_path_returns_content(self):
        # Capture the outbound request so we can also assert prompt
        # shaping (system + user messages) + auth header.
        captured: dict = {}

        def handler(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            return _FakeResponse(
                status_code=200,
                json_data={
                    "choices": [
                        {"message": {"content": "  generated json blob  "}}
                    ]
                },
            )

        backend = NemoBackend(
            host="http://nim",
            model="meta/llama-3.1-70b-instruct",
            api_key="secret-key",
        )
        with _patch_async_client(handler):
            out = await backend.complete(
                "Generate 3 paraphrases.",
                system_prompt="You are a data labeler.",
                max_tokens=512,
                temperature=0.3,
            )

        self.assertEqual(out, "  generated json blob  ")
        self.assertEqual(
            captured["url"], "http://nim/v1/chat/completions"
        )
        # Auth header propagates from settings/constructor.
        self.assertEqual(
            captured["headers"]["Authorization"], "Bearer secret-key"
        )
        # OpenAI-chat shaping: system message comes first.
        msgs = captured["json"]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        # max_tokens + temperature flow through.
        self.assertEqual(captured["json"]["max_tokens"], 512)
        self.assertEqual(captured["json"]["temperature"], 0.3)

    async def test_complete_raises_when_url_unset(self):
        with _settings_overrides(NEMO_API_URL=""):
            backend = NemoBackend(model="some/model")
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
            self.assertIn("NEMO_API_URL", str(cm.exception))

    async def test_complete_raises_when_model_unset(self):
        with _settings_overrides(NEMO_DEFAULT_MODEL=""):
            backend = NemoBackend(host="http://nim")
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
            self.assertIn("NEMO_DEFAULT_MODEL", str(cm.exception))

    async def test_complete_wraps_timeout(self):
        def handler(url, **kwargs):
            raise httpx.TimeoutException("timed out")

        backend = NemoBackend(host="http://nim", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        msg = str(cm.exception)
        self.assertIn("timed out", msg.lower())
        # Hint surfaces the env var for the user to bump.
        self.assertIn("NEMO_TIMEOUT_SECONDS", msg)

    async def test_complete_wraps_401_with_api_key_hint(self):
        def handler(url, **kwargs):
            return _FakeResponse(status_code=401, text="unauthorized")

        backend = NemoBackend(host="http://nim", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        self.assertIn("NEMO_API_KEY", str(cm.exception))

    async def test_complete_wraps_404_with_model_listing_hint(self):
        def handler(url, **kwargs):
            return _FakeResponse(status_code=404, text="model not found")

        backend = NemoBackend(host="http://nim", model="missing/model")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        msg = str(cm.exception)
        self.assertIn("missing/model", msg)
        self.assertIn("/v1/models", msg)

    async def test_complete_wraps_generic_http_error(self):
        def handler(url, **kwargs):
            raise httpx.ConnectError("connection reset")

        backend = NemoBackend(host="http://nim", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        self.assertIn("connection reset", str(cm.exception))

    async def test_complete_wraps_json_decode_error(self):
        def handler(url, **kwargs):
            # __decode_error__ sentinel forces _FakeResponse.json() to
            # raise ValueError, mimicking an HTML 502 page from a
            # reverse proxy.
            return _FakeResponse(status_code=200, json_data="__decode_error__")  # type: ignore[arg-type]

        backend = NemoBackend(host="http://nim", model="m")
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

        backend = NemoBackend(host="http://nim", model="m")
        with _patch_async_client(handler):
            with self.assertRaises(SynthBackendError) as cm:
                await backend.complete("hi")
        self.assertIn("unexpected response shape", str(cm.exception))


# ─────────────────────────────────────────────────────────────────────
# Registry + pick_backend
# ─────────────────────────────────────────────────────────────────────


class SynthBackendRegistryTests(unittest.TestCase):
    def test_registry_includes_nemo_last(self):
        # Order matters: existing auto-pick must keep returning Ollama
        # / Teacher first so existing local-only installs see no
        # change in behavior.
        names = [c.name for c in BACKEND_REGISTRY]
        self.assertIn("nemo", names)
        self.assertEqual(names[-1], "nemo")

    def test_pick_backend_routes_explicit_nemo_pin(self):
        with _settings_overrides(NEMO_API_URL="http://nim"):
            with patch.object(
                httpx,
                "get",
                return_value=_FakeResponse(status_code=200, json_data={}),
            ):
                backend = pick_backend("nemo:meta/llama-3.1-70b-instruct")
        self.assertIsInstance(backend, NemoBackend)
        self.assertEqual(
            backend.describe(), "nemo:meta/llama-3.1-70b-instruct"
        )

    def test_pick_backend_unknown_name_raises(self):
        with self.assertRaises(SynthBackendError):
            pick_backend("not-a-real-backend")

    def test_pick_backend_auto_pick_skips_nemo_when_ollama_available(self):
        # Auto-pick walks BACKEND_REGISTRY in order. Even with NeMo
        # reachable, Ollama wins when present — existing users see no
        # change. Patch Ollama's classmethod to claim available; NeMo's
        # too, just to prove the order is what protects us.
        with _settings_overrides(NEMO_API_URL="http://nim"):
            with (
                patch.object(OllamaBackend, "is_available", return_value=True),
                patch.object(NemoBackend, "is_available", return_value=True),
            ):
                backend = pick_backend(None)
        self.assertIsInstance(backend, OllamaBackend)


if __name__ == "__main__":
    unittest.main()

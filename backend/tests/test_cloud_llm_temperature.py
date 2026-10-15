"""Newer Anthropic models reject `temperature`: the call retries without it
once and remembers the model."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

import httpx

from app.services import cloud_llm_service as svc


class TemperatureRetryTests(unittest.TestCase):
    def setUp(self):
        svc._NO_TEMPERATURE_MODELS.clear()

    def test_retries_without_temperature_and_remembers(self):
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body)
            if "temperature" in body:
                return httpx.Response(400, text='{"type":"error","error":{"type":"invalid_request_error","message":"`temperature` is deprecated for this model."}}')
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "model": body["model"],
                                             "usage": {"input_tokens": 3, "output_tokens": 1}, "stop_reason": "end_turn"})

        real = httpx.AsyncClient

        def client(*a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            return real(*a, **kw)

        async def go():
            return await svc.call_anthropic_chat(api_key="k", model="new-model", system_prompt="s", user_prompt="u")

        with mock.patch.object(svc.httpx, "AsyncClient", client):
            first = asyncio.run(go())
            second = asyncio.run(go())
        self.assertEqual((first.content, second.content), ("ok", "ok"))
        self.assertEqual(["temperature" in b for b in seen], [True, False, False], "retried once, then never sent")

    def test_other_400s_still_raise(self):
        def handler(request):
            return httpx.Response(400, text='{"error":{"message":"bad model"}}')

        real = httpx.AsyncClient

        def client(*a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            return real(*a, **kw)

        with mock.patch.object(svc.httpx, "AsyncClient", client), self.assertRaises(svc.CloudLlmError):
            asyncio.run(svc.call_anthropic_chat(api_key="k", model="m", system_prompt="s", user_prompt="u"))

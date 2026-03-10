"""Inference playground service for project-scoped chat experiments."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, AsyncIterator

import httpx

DEFAULT_OPENAI_COMPATIBLE_URL = "http://localhost:11434/v1/chat/completions"


def _normalize_provider(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"", "openai", "openai_compatible", "ollama"}:
        return "openai_compatible"
    if token in {"mock", "simulate"}:
        return "mock"
    return token


def normalize_playground_messages(
    *,
    messages: list[dict[str, str]],
    system_prompt: str | None,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        normalized.append({"role": "system", "content": system_prompt.strip()})

    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _mock_chat_reply(
    *,
    messages: list[dict[str, str]],
    model_name: str,
) -> str:
    for message in reversed(messages):
        if str(message.get("role")).strip().lower() != "user":
            continue
        text = str(message.get("content") or "").strip()
        if not text:
            continue
        return (
            f"[mock:{model_name}] I received your message:\n\n{text[:1200]}"
        )
    return f"[mock:{model_name}] Send a user message to start the conversation."


def _iter_text_chunks(text: str, chunk_size: int = 28) -> list[str]:
    cleaned = str(text or "")
    if not cleaned:
        return []
    size = max(1, int(chunk_size))
    return [cleaned[i : i + size] for i in range(0, len(cleaned), size)]


async def _openai_compatible_chat(
    *,
    model_name: str,
    messages: list[dict[str, str]],
    api_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any] | None, str | None, str]:
    endpoint = str(api_url or DEFAULT_OPENAI_COMPATIBLE_URL).strip() or DEFAULT_OPENAI_COMPATIBLE_URL
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPStatusError as e:
        response_text = e.response.text.strip()
        detail = response_text[:400] if response_text else "no response body"
        raise ValueError(f"Provider request failed ({e.response.status_code}): {detail}") from e
    except httpx.HTTPError as e:
        raise ValueError(f"Provider request failed: {e}") from e

    if not isinstance(body, dict):
        raise ValueError("Provider returned non-object JSON payload.")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Provider response missing choices.")

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    reply = str(message.get("content") or "").strip()
    if not reply:
        raise ValueError("Provider response missing assistant message content.")

    usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    finish_reason = str(first.get("finish_reason") or "").strip() or None
    response_id = str(body.get("id") or "").strip()
    return reply, usage, finish_reason, response_id


async def _openai_compatible_chat_stream(
    *,
    model_name: str,
    messages: list[dict[str, str]],
    api_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[dict[str, Any]]:
    endpoint = str(api_url or DEFAULT_OPENAI_COMPATIBLE_URL).strip() or DEFAULT_OPENAI_COMPATIBLE_URL
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    chunks: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                endpoint,
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()

                async for raw_line in response.aiter_lines():
                    line = str(raw_line or "").strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue

                    try:
                        body = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(body, dict):
                        continue

                    current_id = str(body.get("id") or "").strip()
                    if current_id:
                        response_id = current_id

                    if isinstance(body.get("usage"), dict):
                        usage = dict(body.get("usage") or {})

                    choices = body.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue

                    first = choices[0] if isinstance(choices[0], dict) else {}
                    candidate_finish_reason = str(first.get("finish_reason") or "").strip()
                    if candidate_finish_reason:
                        finish_reason = candidate_finish_reason

                    delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
                    message = first.get("message") if isinstance(first.get("message"), dict) else {}
                    content = str(
                        delta.get("content")
                        or message.get("content")
                        or first.get("text")
                        or ""
                    )
                    if content:
                        chunks.append(content)
                        yield {"type": "delta", "content": content}
    except httpx.HTTPStatusError as e:
        response_text = e.response.text.strip()
        detail = response_text[:400] if response_text else "no response body"
        raise ValueError(f"Provider request failed ({e.response.status_code}): {detail}") from e
    except httpx.HTTPError as e:
        raise ValueError(f"Provider request failed: {e}") from e

    reply = "".join(chunks).strip()
    if not reply:
        reply, fallback_usage, fallback_finish_reason, fallback_response_id = await _openai_compatible_chat(
            model_name=model_name,
            messages=messages,
            api_url=api_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for chunk in _iter_text_chunks(reply):
            yield {"type": "delta", "content": chunk}
        usage = usage or fallback_usage
        finish_reason = finish_reason or fallback_finish_reason
        response_id = response_id or fallback_response_id

    yield {
        "type": "final",
        "reply": reply,
        "usage": usage,
        "finish_reason": finish_reason or "stop",
        "response_id": response_id,
        "endpoint": endpoint,
    }


async def run_playground_chat(
    *,
    provider: str,
    model_name: str,
    messages: list[dict[str, str]],
    api_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run chat request against mock or OpenAI-compatible provider."""
    normalized_provider = _normalize_provider(provider)
    normalized_messages = normalize_playground_messages(
        messages=messages,
        system_prompt=system_prompt,
    )
    if not normalized_messages:
        raise ValueError("At least one non-empty chat message is required.")

    started = perf_counter()
    if normalized_provider == "mock":
        reply = _mock_chat_reply(messages=normalized_messages, model_name=model_name)
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return {
            "provider": normalized_provider,
            "model_name": model_name,
            "reply": reply,
            "usage": None,
            "finish_reason": "stop",
            "response_id": None,
            "endpoint": None,
            "latency_ms": latency_ms,
        }

    if normalized_provider != "openai_compatible":
        raise ValueError(
            f"Unsupported provider '{normalized_provider}'. Use 'openai_compatible' or 'mock'."
        )

    reply, usage, finish_reason, response_id = await _openai_compatible_chat(
        model_name=model_name,
        messages=normalized_messages,
        api_url=api_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = round((perf_counter() - started) * 1000, 2)
    endpoint = str(api_url or DEFAULT_OPENAI_COMPATIBLE_URL).strip() or DEFAULT_OPENAI_COMPATIBLE_URL
    return {
        "provider": normalized_provider,
        "model_name": model_name,
        "reply": reply,
        "usage": usage,
        "finish_reason": finish_reason,
        "response_id": response_id,
        "endpoint": endpoint,
        "latency_ms": latency_ms,
    }


async def stream_playground_chat(
    *,
    provider: str,
    model_name: str,
    messages: list[dict[str, str]],
    api_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    system_prompt: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield incremental chat events and a final payload envelope."""
    normalized_provider = _normalize_provider(provider)
    normalized_messages = normalize_playground_messages(
        messages=messages,
        system_prompt=system_prompt,
    )
    if not normalized_messages:
        raise ValueError("At least one non-empty chat message is required.")

    started = perf_counter()
    if normalized_provider == "mock":
        reply = _mock_chat_reply(messages=normalized_messages, model_name=model_name)
        for chunk in _iter_text_chunks(reply):
            yield {"type": "delta", "content": chunk}
        latency_ms = round((perf_counter() - started) * 1000, 2)
        yield {
            "type": "final",
            "provider": normalized_provider,
            "model_name": model_name,
            "reply": reply,
            "usage": None,
            "finish_reason": "stop",
            "response_id": None,
            "endpoint": None,
            "latency_ms": latency_ms,
        }
        return

    if normalized_provider != "openai_compatible":
        raise ValueError(
            f"Unsupported provider '{normalized_provider}'. Use 'openai_compatible' or 'mock'."
        )

    async for event in _openai_compatible_chat_stream(
        model_name=model_name,
        messages=normalized_messages,
        api_url=api_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        if event.get("type") != "final":
            yield event
            continue
        payload = dict(event)
        payload["provider"] = normalized_provider
        payload["model_name"] = model_name
        payload["latency_ms"] = round((perf_counter() - started) * 1000, 2)
        yield payload

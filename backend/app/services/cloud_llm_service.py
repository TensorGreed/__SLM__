"""Thin client wrappers for flagship cloud LLM providers (OpenAI +
Anthropic), used by the gold-set LLM-assisted generation path.

Why two clients instead of one OpenAI-compatible client:
  * OpenAI (and Deepseek, which is OpenAI-compatible) use the
    ``/v1/chat/completions`` endpoint with the messages/role/content
    shape and an optional ``response_format`` field.
  * Anthropic uses ``/v1/messages`` with a top-level ``system`` field
    (not a system role inside ``messages``), a different
    ``x-api-key`` header, and no ``response_format`` field (it
    enforces JSON via prompt engineering — we ask the model to emit
    pure JSON in the user prompt).

Both clients return the same shape::

    {"content": str, "model": str, "usage": {prompt_tokens, completion_tokens}}

so the caller (``gold_llm_service``) doesn't have to branch on
provider when parsing or accounting for tokens.

The existing ``synthetic_service.call_teacher_model`` covers
OpenAI-compatible endpoints for the synth path; we deliberately keep
a separate entry point for gold generation so the prompt-shape +
schema-strictness rules can evolve independently. Gold rows are the
evaluation ground truth — looser validation than synth would let
hallucinations slip into evals.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx


# ─────────────────────────────────────────────────────────────────────
# Public response shape — identical across providers
# ─────────────────────────────────────────────────────────────────────


@dataclass
class CloudLlmResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class CloudLlmError(RuntimeError):
    """Raised when the upstream provider returns an unrecoverable
    error (bad API key, model not found, rate-limited, etc.). The
    message is safe to surface to end users — it never includes the
    API key."""


# ─────────────────────────────────────────────────────────────────────
# OpenAI client — also speaks to Deepseek + any OpenAI-compatible API
# ─────────────────────────────────────────────────────────────────────


_OPENAI_DEFAULT_URL = "https://api.openai.com/v1/chat/completions"
_DEFAULT_TIMEOUT_SECONDS = 180.0


async def call_openai_chat(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.4,
    api_url: str | None = None,
    force_json: bool = True,
) -> CloudLlmResponse:
    """Call OpenAI's chat-completions endpoint (or a compatible one).

    ``api_url`` defaults to the OpenAI production URL; override to
    point at Deepseek (``https://api.deepseek.com/v1/chat/completions``)
    or a custom OpenAI-compatible host. ``force_json`` adds
    ``response_format=json_object`` so the model is constrained to
    emit valid JSON when supported (gold parsing depends on it)."""
    if not api_key:
        raise CloudLlmError("OpenAI API key is required.")
    url = api_url or _OPENAI_DEFAULT_URL
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
        try:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise CloudLlmError(
                f"OpenAI request failed: {type(exc).__name__}",
            ) from exc

    if resp.status_code >= 400:
        # OpenAI returns ``{"error": {"message": "..."}}``. Surface
        # the message but never the API key (httpx logs headers via
        # ``request`` so we don't ``str(resp.request)`` the whole
        # thing; just the body).
        body = resp.text[:400]
        raise CloudLlmError(
            f"OpenAI returned {resp.status_code}: {body}",
        )

    data = resp.json()
    choices = data.get("choices") or []
    content = ""
    if choices:
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
    usage = data.get("usage") or {}
    return CloudLlmResponse(
        content=content,
        model=str(data.get("model") or model),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


# ─────────────────────────────────────────────────────────────────────
# Anthropic client
# ─────────────────────────────────────────────────────────────────────


_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"


async def call_anthropic_chat(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.4,
) -> CloudLlmResponse:
    """Call Anthropic's messages endpoint.

    Notable shape differences from OpenAI:
      * ``system`` is a top-level field, NOT a message with role=system
      * Auth header is ``x-api-key``, not ``Authorization: Bearer``
      * Requires the ``anthropic-version`` header
      * No ``response_format`` field — we ask for JSON in the prompt
        and rely on the parser's tolerance for code fences / preamble.
      * Response content is a list of ``{type, text}`` blocks; we
        concatenate the text blocks.
    """
    if not api_key:
        raise CloudLlmError("Anthropic API key is required.")
    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
        try:
            resp = await client.post(
                _ANTHROPIC_URL,
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": _ANTHROPIC_API_VERSION,
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise CloudLlmError(
                f"Anthropic request failed: {type(exc).__name__}",
            ) from exc

    if resp.status_code >= 400:
        body = resp.text[:400]
        raise CloudLlmError(
            f"Anthropic returned {resp.status_code}: {body}",
        )

    data = resp.json()
    # ``content`` is a list of blocks: [{"type": "text", "text": "..."}]
    blocks = data.get("content") or []
    text_parts = [
        str(b.get("text") or "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    content = "".join(text_parts)
    usage = data.get("usage") or {}
    return CloudLlmResponse(
        content=content,
        model=str(data.get("model") or model),
        prompt_tokens=int(usage.get("input_tokens") or 0),
        completion_tokens=int(usage.get("output_tokens") or 0),
    )


# ─────────────────────────────────────────────────────────────────────
# Lightweight JSON extractor — tolerates code fences + preamble
# ─────────────────────────────────────────────────────────────────────


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_payload(raw: str) -> Any:
    """Pull a JSON object/array out of an LLM response.

    Real responses from both providers occasionally wrap the JSON in
    triple-backtick fences, prefix it with a sentence ("Here are the
    Q&A pairs:"), or both. This helper strips the most common
    wrappers and tries ``json.loads``; raises ``ValueError`` if no
    parseable JSON is found.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("LLM returned an empty response.")

    # Prefer the first code-fenced block when present.
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try the whole thing first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to slice from the first { or [ to the matching closer.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            slice_ = text[start : end + 1]
            try:
                return json.loads(slice_)
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"LLM response was not parseable JSON. First 200 chars: {raw[:200]!r}",
    )

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
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx


_LOG = logging.getLogger("cloud_llm")


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
# Bumped from 180s → 300s. Reasoning-style models (DeepSeek-R1, o-series,
# any "Pro/Reasoner" Deepseek variant) routinely take 60-120s on grounded
# 10-pair requests because they emit a long ``<think>`` preamble before
# the actual JSON. The frontend's axios timeout is now 420s, so this
# stays below it — the backend will surface a structured 502
# (CloudLlmError) before the frontend gives up with "Network Error",
# which used to happen when both timeouts were 180s and raced.
_DEFAULT_TIMEOUT_SECONDS = 300.0


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

    started = time.monotonic()
    _LOG.info(
        "openai-compat call → url=%s model=%s max_tokens=%d force_json=%s",
        url, model, max_tokens, force_json,
    )
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
            elapsed = time.monotonic() - started
            _LOG.warning(
                "openai-compat FAILED after %.1fs: %s url=%s model=%s",
                elapsed, type(exc).__name__, url, model,
            )
            raise CloudLlmError(
                f"OpenAI-compat request failed after {elapsed:.0f}s: "
                f"{type(exc).__name__}. "
                "The provider may be slow / unreachable, or the model id is "
                "rejected. Try a known-good model (gpt-4o-mini / deepseek-chat) "
                "to isolate.",
            ) from exc

    elapsed = time.monotonic() - started

    if resp.status_code >= 400:
        # OpenAI returns ``{"error": {"message": "..."}}``. Surface
        # the message but never the API key (httpx logs headers via
        # ``request`` so we don't ``str(resp.request)`` the whole
        # thing; just the body).
        body = resp.text[:400]
        _LOG.warning(
            "openai-compat returned HTTP %d after %.1fs: %s",
            resp.status_code, elapsed, body[:200],
        )
        raise CloudLlmError(
            f"OpenAI-compat returned {resp.status_code} after {elapsed:.0f}s: "
            f"{body}",
        )

    data = resp.json()
    choices = data.get("choices") or []
    content = ""
    finish_reason = ""
    if choices:
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        finish_reason = str(choices[0].get("finish_reason") or "")
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    _LOG.info(
        "openai-compat OK after %.1fs: model=%s prompt_tokens=%d "
        "completion_tokens=%d finish_reason=%s content_chars=%d",
        elapsed, data.get("model") or model, prompt_tokens,
        completion_tokens, finish_reason, len(content),
    )
    if finish_reason == "length":
        _LOG.warning(
            "openai-compat hit max_tokens (%d) on model=%s — response was "
            "truncated; downstream parse will likely fail. Consider raising "
            "max_tokens or reducing the count.",
            max_tokens, data.get("model") or model,
        )
    return CloudLlmResponse(
        content=content,
        model=str(data.get("model") or model),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
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
    started = time.monotonic()
    _LOG.info(
        "anthropic call → model=%s max_tokens=%d",
        model, max_tokens,
    )
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
            elapsed = time.monotonic() - started
            _LOG.warning(
                "anthropic FAILED after %.1fs: %s model=%s",
                elapsed, type(exc).__name__, model,
            )
            raise CloudLlmError(
                f"Anthropic request failed after {elapsed:.0f}s: "
                f"{type(exc).__name__}. The provider may be slow / "
                "unreachable, or the model id is rejected.",
            ) from exc

    elapsed = time.monotonic() - started

    if resp.status_code >= 400:
        body = resp.text[:400]
        _LOG.warning(
            "anthropic returned HTTP %d after %.1fs: %s",
            resp.status_code, elapsed, body[:200],
        )
        raise CloudLlmError(
            f"Anthropic returned {resp.status_code} after {elapsed:.0f}s: "
            f"{body}",
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
    stop_reason = str(data.get("stop_reason") or "")
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    _LOG.info(
        "anthropic OK after %.1fs: model=%s prompt_tokens=%d "
        "completion_tokens=%d stop_reason=%s content_chars=%d",
        elapsed, data.get("model") or model, prompt_tokens,
        completion_tokens, stop_reason, len(content),
    )
    if stop_reason == "max_tokens":
        _LOG.warning(
            "anthropic hit max_tokens (%d) on model=%s — response was "
            "truncated; downstream parse will likely fail.",
            max_tokens, data.get("model") or model,
        )
    return CloudLlmResponse(
        content=content,
        model=str(data.get("model") or model),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ─────────────────────────────────────────────────────────────────────
# Lightweight JSON extractor — tolerates code fences + preamble
# ─────────────────────────────────────────────────────────────────────


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Reasoning models (DeepSeek-R1 family, Qwen3 "/think", Claude's
# extended thinking, the openai o-series) wrap their internal chain
# of thought in ``<think>...</think>`` / ``<reasoning>...</reasoning>``
# / ``<scratchpad>...</scratchpad>`` blocks BEFORE emitting the
# user-facing answer. Many of them do this even when response_format
# is set to json_object, since the format constraint only applies to
# the final answer. The thinking text often contains JSON-shaped
# fragments (the model rehearses the output mid-reasoning), so a
# naïve "slice from first { to last }" picks up nonsense. Strip
# these blocks BEFORE looking for JSON.
_THINK_TAG_PATTERN = re.compile(
    r"<\s*(think|thinking|reasoning|reflection|scratchpad|analysis)\s*>"
    r".*?"
    r"<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# Unterminated opening tags happen on streaming truncation — strip
# everything from the open tag to end-of-text so the partial think
# block doesn't poison the JSON slice path either.
_UNTERMINATED_THINK_PATTERN = re.compile(
    r"<\s*(think|thinking|reasoning|reflection|scratchpad|analysis)\s*>.*$",
    re.IGNORECASE | re.DOTALL,
)


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` and friends. Idempotent + safe
    to call on non-reasoning model output (no tags → no change)."""
    cleaned = _THINK_TAG_PATTERN.sub("", text)
    cleaned = _UNTERMINATED_THINK_PATTERN.sub("", cleaned)
    return cleaned


def extract_json_payload(raw: str) -> Any:
    """Pull a JSON object/array out of an LLM response.

    Real responses from both providers occasionally wrap the JSON in
    triple-backtick fences, prefix it with a sentence ("Here are the
    Q&A pairs:"), include a ``<think>...</think>`` reasoning preamble
    (R1-style models), or any combination. This helper strips the
    most common wrappers and tries ``json.loads``; raises
    ``ValueError`` if no parseable JSON is found.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("LLM returned an empty response.")

    # Strip reasoning blocks FIRST — their content frequently
    # contains JSON-shaped fragments that would derail the slice
    # path below.
    text = _strip_thinking(text).strip()
    if not text:
        # Whole response was inside an unterminated <think> — common
        # when the model ran out of tokens mid-thought.
        raise ValueError(
            "LLM response was entirely a reasoning preamble — no "
            "JSON output was reached. Raise max_tokens or pick a "
            f"non-reasoning model. First 200 chars: {raw[:200]!r}",
        )

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

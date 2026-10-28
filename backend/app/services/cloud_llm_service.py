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
from urllib.parse import urlparse

import httpx

from app.services import llm_budget_guard, llm_usage


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
    # Anthropic prompt caching: input tokens written to / read from the
    # cache (billed at 1.25x / 0.1x the input price), on top of
    # ``prompt_tokens`` (the uncached input).
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    # Anthropic: why the reply ended ("end_turn", "max_tokens", "refusal", ...).
    stop_reason: str = ""


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
    # Hard monthly API budget: a metered (non-local) call is refused once a cap is reached.
    if not llm_usage.is_local_url(url):
        await llm_budget_guard.acheck_budget(llm_usage.provider_for_url(url), model)
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
    response = CloudLlmResponse(
        content=content,
        model=str(data.get("model") or model),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    if not llm_usage.is_local_url(url):
        host = (urlparse(url).hostname or "").lower()
        llm_usage.record_response("deepseek" if "deepseek" in host else "openai" if "openai" in host else host, response)
    return response


# ─────────────────────────────────────────────────────────────────────
# Anthropic client
# ─────────────────────────────────────────────────────────────────────


_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"


# Models that reject a ``temperature`` field (newer Anthropic models answer
# 400 "`temperature` is deprecated for this model"). Learned on the first
# such 400; later calls to that model leave it out.
_NO_TEMPERATURE_MODELS: set[str] = set()


# Anthropic models that take ``output_config.effort`` (how much the model
# thinks before answering), and the ones that think by default when no
# ``thinking`` field is sent. Thinking tokens count against ``max_tokens``
# and come back as non-text blocks: a big prompt at the default effort can
# spend the whole budget thinking and return no text at all. Prefix match.
_EFFORT_MODEL_PREFIXES = (
    "claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
    "claude-opus-4-6", "claude-sonnet-5", "claude-sonnet-4-6", "claude-haiku-5",
)
_THINKS_BY_DEFAULT_PREFIXES = ("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5", "claude-haiku-5")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def anthropic_supports_effort(model: str) -> bool:
    return str(model or "").lower().startswith(_EFFORT_MODEL_PREFIXES)


def anthropic_thinks_by_default(model: str) -> bool:
    return str(model or "").lower().startswith(_THINKS_BY_DEFAULT_PREFIXES)


def _rejects_temperature(status: int, body: str) -> bool:
    return status == 400 and "temperature" in body and ("deprecated" in body or "not supported" in body)


def anthropic_message_params(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.4,
    cached_system: str | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    """The ``/v1/messages`` body — shared by the synchronous call and a
    Message Batches request's ``params`` so both send the same prompt.
    ``effort`` is sent only to models that take it (``anthropic_supports_effort``)."""
    system: Any = system_prompt
    if cached_system:
        system = [
            *([{"type": "text", "text": system_prompt}] if system_prompt else []),
            {"type": "text", "text": cached_system, "cache_control": {"type": "ephemeral"}},
        ]
    payload: dict[str, Any] = {
        "model": model,
        "system": system,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    if model not in _NO_TEMPERATURE_MODELS:
        payload["temperature"] = temperature
    if effort in EFFORT_LEVELS and anthropic_supports_effort(model):
        payload["output_config"] = {"effort": effort}
    return payload


async def call_anthropic_chat(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.4,
    cached_system: str | None = None,
    effort: str | None = None,
) -> CloudLlmResponse:
    """Call Anthropic's messages endpoint. ``cached_system``: a stable
    block (the same across many calls) sent as a cached system block —
    repeats read it from Anthropic's prompt cache at a tenth of the input
    price. Below the model's minimum cacheable length it is simply sent
    uncached.

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
    await llm_budget_guard.acheck_budget("anthropic", model)  # hard monthly API budget
    payload = anthropic_message_params(
        model=model, system_prompt=system_prompt, user_prompt=user_prompt,
        max_tokens=max_tokens, temperature=temperature, cached_system=cached_system, effort=effort,
    )
    started = time.monotonic()
    _LOG.info(
        "anthropic call → model=%s max_tokens=%d",
        model, max_tokens,
    )
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_API_VERSION,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
        try:
            resp = await client.post(_ANTHROPIC_URL, json=payload, headers=headers)
            if "temperature" in payload and _rejects_temperature(resp.status_code, resp.text):
                _NO_TEMPERATURE_MODELS.add(model)
                payload.pop("temperature")
                _LOG.info("anthropic model=%s rejects temperature; retrying without it", model)
                resp = await client.post(_ANTHROPIC_URL, json=payload, headers=headers)
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
    out = anthropic_response_from_message(data, model)
    # Every open usage meter (the Job / operation this call belongs to)
    # counts it — see llm_usage. Only the synchronous path: batch results
    # are accounted by the batch path itself, at the batch discount.
    llm_usage.record_response("anthropic", out)
    stop_reason = str(data.get("stop_reason") or "")
    _LOG.info(
        "anthropic OK after %.1fs: model=%s prompt_tokens=%d "
        "completion_tokens=%d stop_reason=%s content_chars=%d",
        elapsed, out.model, out.prompt_tokens,
        out.completion_tokens, stop_reason, len(out.content),
    )
    if stop_reason == "max_tokens" and not out.content:
        _LOG.warning(
            "anthropic spent the whole max_tokens budget (%d) on model=%s thinking — "
            "no answer text; lower the effort or raise the budget.",
            max_tokens, out.model,
        )
    elif stop_reason == "max_tokens":
        _LOG.warning(
            "anthropic hit max_tokens (%d) on model=%s — response was "
            "truncated; downstream parse will likely fail.",
            max_tokens, out.model,
        )
    return out


def anthropic_response_from_message(data: dict[str, Any], model: str) -> CloudLlmResponse:
    """A ``/v1/messages`` response body (or a batch result's ``message``)
    → ``CloudLlmResponse``: text blocks joined, usage incl. cache tokens."""
    # ``content`` is a list of blocks: [{"type": "text", "text": "..."}]
    blocks = data.get("content") or []
    text_parts = [
        str(b.get("text") or "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    usage = data.get("usage") or {}
    return CloudLlmResponse(
        content="".join(text_parts),
        model=str(data.get("model") or model),
        prompt_tokens=int(usage.get("input_tokens") or 0),
        completion_tokens=int(usage.get("output_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        stop_reason=str(data.get("stop_reason") or ""),
    )


# ─────────────────────────────────────────────────────────────────────
# Anthropic Message Batches — half the list price, results in minutes
# to hours (at most 24 h). Used for non-urgent labelling jobs.
# ─────────────────────────────────────────────────────────────────────


_ANTHROPIC_BATCHES_URL = "https://api.anthropic.com/v1/messages/batches"
# Seconds between status polls (ANTHROPIC_BATCH_POLL_SECONDS overrides).
BATCH_POLL_SECONDS = 30.0
# A batch that has not ended after this long is given up on (Anthropic
# expires unfinished requests at 24 h; the margin covers the last poll).
BATCH_MAX_WAIT_SECONDS = 25 * 3600.0
_CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_API_VERSION,
        "Content-Type": "application/json",
    }


async def _anthropic_batch_request(method: str, url: str, api_key: str, what: str, **kwargs: Any) -> httpx.Response:
    if not api_key:
        raise CloudLlmError("Anthropic API key is required.")
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
        try:
            resp = await client.request(method, url, headers=_anthropic_headers(api_key), **kwargs)
        except httpx.HTTPError as exc:
            raise CloudLlmError(f"Anthropic batch {what} failed: {type(exc).__name__}.") from exc
    if resp.status_code >= 400:
        body = resp.text[:400]
        _LOG.warning("anthropic batch %s returned HTTP %d: %s", what, resp.status_code, body[:200])
        raise CloudLlmError(f"Anthropic batch {what} returned {resp.status_code}: {body}")
    return resp


async def create_anthropic_batch(*, api_key: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Submit ``requests`` (``[{custom_id, params}]``, ``params`` = a
    ``/v1/messages`` body — see ``anthropic_message_params``) as one Message
    Batch. Returns the batch object (``id``, ``processing_status``,
    ``request_counts``, …)."""
    if not requests:
        raise CloudLlmError("A message batch needs at least one request.")
    ids = [str(r.get("custom_id") or "") for r in requests]
    bad = [i for i in ids if not _CUSTOM_ID_RE.match(i)]
    if bad or len(set(ids)) != len(ids):
        raise CloudLlmError("Batch custom_ids must be unique and match [a-zA-Z0-9_-]{1,64}.")
    resp = await _anthropic_batch_request("POST", _ANTHROPIC_BATCHES_URL, api_key, "create", json={"requests": requests})
    data = resp.json()
    _LOG.info("anthropic batch created: id=%s requests=%d", data.get("id"), len(requests))
    return data


async def get_anthropic_batch(*, api_key: str, batch_id: str) -> dict[str, Any]:
    """The batch object: ``processing_status`` (in_progress / canceling /
    ended), ``request_counts`` and, once ended, ``results_url``."""
    resp = await _anthropic_batch_request("GET", f"{_ANTHROPIC_BATCHES_URL}/{batch_id}", api_key, "status")
    return resp.json()


async def cancel_anthropic_batch(*, api_key: str, batch_id: str) -> dict[str, Any]:
    resp = await _anthropic_batch_request("POST", f"{_ANTHROPIC_BATCHES_URL}/{batch_id}/cancel", api_key, "cancel")
    return resp.json()


async def fetch_anthropic_batch_results(*, api_key: str, results_url: str) -> dict[str, dict[str, Any]]:
    """Download an ended batch's results (JSONL, one line per request, in
    any order) → ``{custom_id: result}`` where ``result`` is ``{"type":
    "succeeded", "message": {...}}`` or errored / canceled / expired."""
    if not results_url:
        raise CloudLlmError("The batch has no results_url yet (has it ended?).")
    resp = await _anthropic_batch_request("GET", results_url, api_key, "results")
    out: dict[str, dict[str, Any]] = {}
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("custom_id"):
            out[str(item["custom_id"])] = item.get("result") if isinstance(item.get("result"), dict) else {}
    return out


def batch_result_error(result: dict[str, Any] | None) -> str:
    """Why a batch request has no message ("" when it succeeded)."""
    if not result:
        return "missing from the batch results"
    kind = str(result.get("type") or "")
    if kind == "succeeded":
        return ""
    err = result.get("error") or {}
    inner = err.get("error") if isinstance(err.get("error"), dict) else err
    message = str((inner or {}).get("message") or "") if isinstance(inner, dict) else ""
    return f"{kind}: {message}" if message else kind or "unknown result"


async def wait_for_anthropic_batch(
    *,
    api_key: str,
    batch_id: str,
    on_status: Any = None,
    poll_seconds: float | None = None,
    max_wait_seconds: float | None = None,
) -> dict[str, Any]:
    """Poll until the batch has ended; ``on_status(batch)`` (async) after
    each poll. Raises ``CloudLlmError`` when it hasn't ended within
    ``max_wait_seconds`` (the batch is cancelled first, best-effort)."""
    import asyncio
    import os

    if poll_seconds is None:
        try:
            poll_seconds = float(os.environ.get("ANTHROPIC_BATCH_POLL_SECONDS") or BATCH_POLL_SECONDS)
        except ValueError:
            poll_seconds = BATCH_POLL_SECONDS
    limit = BATCH_MAX_WAIT_SECONDS if max_wait_seconds is None else max_wait_seconds
    started = time.monotonic()
    while True:
        batch = await get_anthropic_batch(api_key=api_key, batch_id=batch_id)
        if on_status is not None:
            await on_status(batch)
        if batch.get("processing_status") == "ended":
            return batch
        if time.monotonic() - started > limit:
            try:
                await cancel_anthropic_batch(api_key=api_key, batch_id=batch_id)
            except CloudLlmError:
                pass
            raise CloudLlmError(f"Anthropic batch {batch_id} had not ended after {limit / 3600:.1f} h; cancelled it.")
        try:
            await asyncio.sleep(max(0.0, poll_seconds))
        except asyncio.CancelledError:
            # The Job was cancelled: stop the batch too, so unfinished
            # requests aren't processed (and billed) for nothing.
            try:
                await cancel_anthropic_batch(api_key=api_key, batch_id=batch_id)
            except Exception:  # noqa: BLE001 — best-effort
                pass
            raise


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

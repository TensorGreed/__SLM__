"""Ollama-based synthetic-data backend (USER-SUCCESS Epic 2).

Talks to a local Ollama server via its OpenAI-compatible
`/v1/chat/completions` endpoint. Picks the largest installed model
from a preference order (Llama 3.1 8B → Qwen 2.5 7B → Mistral 7B →
first listed) unless the caller pins a specific one.

Reachability is cheap to check: a `GET /api/tags` is non-streaming
and returns immediately when the daemon is up.
"""

from __future__ import annotations

import os
import re
from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - tests guard via mocks
    httpx = None  # type: ignore[assignment]

from .base import SynthBackendError


DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# 10 minutes — generations of 30-50 rows on a CPU-loaded Llama 3.1 8B
# routinely take 3-5 minutes. The legacy `call_teacher_model` flow
# defaults to 600s for the same reason. Matches the Vite proxy ceiling
# (10 min) — anything longer than that is broken regardless.
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "600"))

# Preference order — we want the strongest realistic model that
# Ollama users typically have pulled. Tags are matched as substrings
# (so `qwen2.5:14b-instruct-q4_K_M` matches `qwen2.5`).
#
# Qwen 2.5 is preferred over Llama 3 because:
#   - It scales up to 14B / 32B / 72B in the Ollama catalog (Llama 3
#     family caps at 8B without going to 70B).
#   - It's measurably less guard-rail-trigger-happy on legitimate
#     classifier-training data (security / spam / abuse detection),
#     where Llama 3 will refuse outright on category names like
#     "injection" or "toxicity" even with the defensive-use system
#     prompt. The model that won't refuse is more useful to users.
PREFERRED_MODEL_PATTERNS: list[str] = [
    "qwen2.5",
    "qwen2",
    "llama3.1",
    "llama3",
    "mistral",
    "phi3",
    "gemma",
]


def _strip_thinking_blocks(text: str) -> str:
    """Reasoning models (Qwen QwQ, DeepSeek-R1) emit `<think>...</think>`
    blocks. We discard those before handing the response to the
    playbook parser."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class OllamaBackend:
    """Ollama OpenAI-compatible chat completion backend."""

    name: str = "ollama"
    # Ollama's /v1 shim ignores OpenAI's response_format=json_schema —
    # the playbook parser does all structure enforcement.
    schema_aware: bool = False

    def __init__(
        self,
        *,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self._host = host.rstrip("/")
        self._timeout = timeout_seconds
        # Resolve the model lazily on first complete() if not pinned.
        self._model: str | None = model or None

    @classmethod
    def is_available(cls) -> bool:
        """True when an Ollama daemon answers `GET /api/tags` on the
        configured host. Uses a 1-second timeout so the check is cheap
        even when Ollama is down."""
        if httpx is None:
            return False
        try:
            resp = httpx.get(f"{DEFAULT_OLLAMA_HOST}/api/tags", timeout=1.0)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001 — any error means "not available"
            return False

    def describe(self) -> str:
        if self._model:
            return f"{self.name}:{self._model}"
        return self.name

    async def _resolve_model(self) -> str:
        if self._model:
            return self._model
        if httpx is None:
            raise SynthBackendError("httpx is not installed; install httpx to use OllamaBackend.")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._host}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise SynthBackendError(
                f"Couldn't reach Ollama at {self._host}: {e}. Is `ollama serve` running?"
            ) from e
        models = [m.get("name", "") for m in (data.get("models") or [])]
        if not models:
            raise SynthBackendError(
                f"Ollama is reachable at {self._host} but has no models installed. "
                f"Run `ollama pull llama3.1:8b` (or any model) first."
            )
        # Pick the first model in the preference list that matches any
        # of the installed tags. Substring match so quantization
        # suffixes (`:8b-q4`) don't break the match.
        for pattern in PREFERRED_MODEL_PATTERNS:
            for tag in models:
                if pattern in tag:
                    self._model = tag
                    return tag
        # Nothing matched — fall back to the first available tag.
        self._model = models[0]
        return self._model

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_schema: dict | None = None,  # noqa: ARG002 — Ollama's /v1/chat/completions ignores OpenAI's response_format=json_schema; the playbook parser handles structure.
    ) -> str:
        if httpx is None:
            raise SynthBackendError("httpx is not installed; install httpx to use OllamaBackend.")
        model = await self._resolve_model()
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._host}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as e:
            raise SynthBackendError(
                f"Ollama timed out after {self._timeout:.0f}s generating with {model!r}. "
                f"Try a smaller target_count, a faster model, or increase OLLAMA_TIMEOUT_SECONDS."
            ) from e
        except httpx.HTTPStatusError as e:
            raise SynthBackendError(
                f"Ollama returned HTTP {e.response.status_code} for model {model!r}: "
                f"{(e.response.text or '')[:200]}"
            ) from e
        except httpx.HTTPError as e:
            raise SynthBackendError(
                f"Ollama request failed for model {model!r}: {e}. Is `ollama serve` still running?"
            ) from e
        except ValueError as e:
            # JSON decode error.
            raise SynthBackendError(
                f"Ollama returned a non-JSON response for model {model!r}: {e}"
            ) from e
        # OpenAI-compatible response shape.
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise SynthBackendError(
                f"Ollama returned an unexpected response shape: {str(data)[:200]!r}"
            ) from e
        return _strip_thinking_blocks(str(content or ""))

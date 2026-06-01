"""Cloud-LLM synth backend — OpenAI, Anthropic, Deepseek.

A thin wrapper around the existing ``cloud_llm_service`` helpers so
the playbook framework can call cloud providers the same way it
calls Ollama. The constructor expects credentials + model already
resolved by the API layer (the synth router pulls the API key from
project secrets before instantiating).

Unlike Ollama, ``is_available()`` here is a static True — cloud
availability is per-project (depends on a saved API key) and the
frontend handles 'no key' UX before submitting. The backend will
raise ``SynthBackendError`` at ``complete()`` time if it was
mis-constructed (e.g. no api_key passed).

Pin tokens accepted by the picker:

    cloud:openai:gpt-4o-mini
    cloud:anthropic:claude-haiku-4-5-20251001
    cloud:deepseek:deepseek-chat

The trailing model id is forwarded to the chat-completions call
verbatim; provider-specific routing is on the ``provider`` arg.
"""

from __future__ import annotations

from typing import Literal

from app.services.cloud_llm_service import (
    CloudLlmError,
    call_anthropic_chat,
    call_openai_chat,
)

from .base import SynthBackendError


Provider = Literal["openai", "anthropic", "deepseek"]


# Provider → base URL. Deepseek is OpenAI-compatible so it routes
# through call_openai_chat with the deepseek host on api_url.
_PROVIDER_URLS: dict[str, str | None] = {
    "openai": None,  # cloud_llm_service default
    "anthropic": None,  # Anthropic uses its own helper
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
}


class CloudLlmBackend:
    """OpenAI / Anthropic / Deepseek backend for synth playbooks."""

    name: str = "cloud"
    # Cloud providers honor OpenAI's ``response_format={"type":
    # "json_object"}`` (call_openai_chat sets force_json=True by
    # default). Not the same as the playbook's full JSON Schema
    # (``response_format=json_schema``) — that's vLLM/NeMo territory —
    # but enough that the picker can badge cloud as schema-adjacent.
    schema_aware: bool = False

    def __init__(
        self,
        *,
        provider: Provider,
        model: str,
        api_key: str,
    ):
        if provider not in _PROVIDER_URLS:
            raise SynthBackendError(
                f"Unknown cloud provider {provider!r}. Supported: "
                f"{', '.join(_PROVIDER_URLS.keys())}."
            )
        if not model or not model.strip():
            raise SynthBackendError(
                f"Cloud backend requires a model id (provider={provider})."
            )
        if not api_key or not api_key.strip():
            raise SynthBackendError(
                f"Cloud provider {provider!r} requires an API key. Save "
                f"one under Project Settings → Secrets, or pass it via "
                f"the request."
            )
        self._provider: Provider = provider
        self._model = model.strip()
        self._api_key = api_key.strip()

    @classmethod
    def is_available(cls) -> bool:
        """Always True at the registry level — cloud availability is
        per-project (depends on a saved API key) and the API layer
        gates instantiation on key presence. Returning True here keeps
        the cloud picker visible in the synth panel UX even when no
        keys are saved yet, so the user knows the option exists."""
        return True

    def describe(self) -> str:
        return f"cloud:{self._provider}:{self._model}"

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_schema: dict | None = None,  # noqa: ARG002 — cloud_llm_service uses force_json instead
    ) -> str:
        # call_*_chat both return CloudLlmResponse with a ``content``
        # str. Map any CloudLlmError to SynthBackendError so the
        # playbook framework's existing 5xx handling kicks in.
        try:
            if self._provider == "anthropic":
                resp = await call_anthropic_chat(
                    api_key=self._api_key,
                    model=self._model,
                    system_prompt=system_prompt or "",
                    user_prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                # openai + deepseek both ride on call_openai_chat with
                # different api_url. force_json=True puts the model
                # into JSON-object mode which the playbook parser
                # tolerates (JSON wrapping a single object instead of
                # JSONL is rare but our parser handles it).
                resp = await call_openai_chat(
                    api_key=self._api_key,
                    model=self._model,
                    system_prompt=system_prompt or "",
                    user_prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_url=_PROVIDER_URLS[self._provider],
                    # Disable force_json — the playbook prompt asks for
                    # JSONL (one object per line), but OpenAI's
                    # response_format=json_object forces a SINGLE
                    # top-level JSON object, which breaks multi-row
                    # playbooks. The playbook parser already handles
                    # markdown-wrapped + free-form JSONL.
                    force_json=False,
                )
        except CloudLlmError as e:
            raise SynthBackendError(
                f"Cloud LLM ({self._provider}:{self._model}) failed: {e}"
            ) from e
        return resp.content or ""

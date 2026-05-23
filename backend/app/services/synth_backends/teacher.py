"""Teacher-model backend — fallback that talks to whatever
OpenAI-compatible endpoint `TEACHER_MODEL_API_URL` points at
(USER-SUCCESS Epic 2).

This is the configured-teacher-endpoint path the legacy
``synthetic_service.call_teacher_model`` already exercises. We
reuse the same dispatcher so any OpenAI / Anthropic-proxy /
HuggingFace-Inference / self-hosted vLLM setup the user already
has wired for the existing synthetic surface works for playbooks
without additional config.

Availability check: backend is available iff ``TEACHER_MODEL_API_URL``
is set. We do NOT actively probe the endpoint at availability-check
time because that would synchronously block the picker; latency
errors surface when ``complete()`` is actually called.
"""

from __future__ import annotations

from app.config import settings


class TeacherModelBackend:
    """Wrapper around `synthetic_service.call_teacher_model`."""

    name: str = "teacher"

    def __init__(self, *, model: str | None = None, api_url: str | None = None):
        self._api_url = api_url or settings.TEACHER_MODEL_API_URL
        self._api_key = settings.TEACHER_MODEL_API_KEY
        # Default to a small chat-capable model; callers/playbooks
        # can override per-request via the `model` constructor arg.
        self._model = model or "llama3"

    @classmethod
    def is_available(cls) -> bool:
        url = (settings.TEACHER_MODEL_API_URL or "").strip()
        return bool(url)

    def describe(self) -> str:
        return f"{self.name}:{self._model}@{self._api_url}"

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        # Import lazily to avoid a circular import at module load time.
        from app.services.synthetic_service import call_teacher_model  # noqa: WPS433

        # The legacy dispatcher returns a dict — content lives at
        # ['choices'][0]['message']['content'] (OpenAI-compatible) or
        # ['raw_text'] (fallback). Normalize to a single string.
        result = await call_teacher_model(
            prompt=prompt,
            system_prompt=system_prompt or "",
            api_url=self._api_url,
            api_key=self._api_key,
            model_name=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
            force_json=False,
        )
        if not isinstance(result, dict):
            return str(result or "")
        # OpenAI-compatible shape first.
        try:
            return str(result["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            pass
        if "raw_text" in result and isinstance(result["raw_text"], str):
            return result["raw_text"]
        if "content" in result and isinstance(result["content"], str):
            return result["content"]
        return ""

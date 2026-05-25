"""SynthBackend protocol + registry helpers (USER-SUCCESS Epic 2).

A SynthBackend is a thin async wrapper around one LLM transport. The
Playbook framework calls `complete(prompt, …)` and consumes the
returned string; backends handle all transport details (provider URL,
auth, model selection, JSON-mode hints, etc.).

Keeping the protocol minimal — just `complete()` and `is_available()`
— means new backends (NeMo for Epic 5, vLLM, etc.) can plug in
without touching playbook code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class SynthBackendError(RuntimeError):
    """Raised when no available backend can satisfy a request."""


@runtime_checkable
class SynthBackend(Protocol):
    """One LLM transport (Ollama, configured teacher endpoint, …).

    Implementations are classes with a `name` class attribute, an
    `is_available()` classmethod for cheap reachability checks, and an
    async `complete()` instance method that returns the model's raw
    text output for a given prompt.

    ``schema_aware`` is True for backends that actually honor the
    ``response_schema`` kwarg on ``complete()`` (NeMo / vLLM forward
    it as OpenAI ``response_format=json_schema``). False for backends
    that accept-and-ignore the kwarg (Ollama, the legacy teacher
    dispatcher). The picker surfaces this so users can tell which
    picks let the playbook's schema actually constrain decoding.
    """

    name: str
    schema_aware: bool = False

    @classmethod
    def is_available(cls) -> bool:  # pragma: no cover - default no-op
        return False

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_schema: dict | None = None,
    ) -> str:  # pragma: no cover - default no-op
        """Generate text for `prompt`.

        `response_schema` is an optional JSON Schema. Backends that
        natively support schema-constrained decoding (NeMo / NIM)
        forward it as ``response_format`` on the chat-completion call;
        backends that don't (Ollama, the legacy teacher dispatcher)
        silently ignore it — playbooks always also enforce structure
        via parse_output/validate, so an ignored schema is a soft
        downgrade, never a failure.
        """
        ...

    def describe(self) -> str:  # pragma: no cover - default no-op
        """Human-readable backend identifier (for PlaybookResult.backend_used).
        e.g. "ollama:llama3.1:8b" or "teacher:http://localhost:11434/v1/…"."""
        return self.name


def pick_backend(
    requested: str | None = None,
    *,
    registry: list[type[SynthBackend]] | None = None,
) -> SynthBackend:
    """Resolve a backend instance.

    - When `requested` is None, walks the registry in order and returns
      the first available backend.
    - When `requested` is a string, looks for an exact match on
      `<backend.name>`. The backend may include extra config (e.g.
      `ollama:llama3.1:8b`) — split on the first colon for routing.

    Raises `SynthBackendError` when nothing satisfies the request.
    """
    # Import lazily to avoid a circular import at module load time.
    if registry is None:
        from . import BACKEND_REGISTRY  # noqa: WPS433

        registry = BACKEND_REGISTRY

    if requested is None:
        for cls in registry:
            try:
                if cls.is_available():
                    return cls()
            except Exception:  # noqa: BLE001 — never let a broken backend block selection
                continue
        raise SynthBackendError(
            "No synthetic-data backend is available. Install Ollama "
            "(https://ollama.com) or set TEACHER_MODEL_API_URL to an "
            "OpenAI-compatible endpoint."
        )

    head, _, tail = requested.partition(":")
    for cls in registry:
        if cls.name != head:
            continue
        if not cls.is_available():
            raise SynthBackendError(
                f"Backend '{head}' is registered but not reachable. "
                f"Check the relevant configuration."
            )
        # If the caller passed extra config (e.g. "ollama:llama3.1:8b"),
        # forward it to the constructor as `model`. Backends that
        # don't accept a model arg should ignore it.
        try:
            return cls(model=tail) if tail else cls()
        except TypeError:
            return cls()

    raise SynthBackendError(
        f"Unknown backend '{requested}'. Registered: "
        f"{', '.join(cls.name for cls in registry)}"
    )

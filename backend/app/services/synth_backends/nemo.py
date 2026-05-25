"""NVIDIA NeMo Data Designer / NIM synth backend (USER-SUCCESS Epic 5 Phase 5a + 5b).

Talks to a locally-running NIM (NVIDIA Inference Microservice) or
NeMo Data Designer endpoint over its OpenAI-compatible
``/v1/chat/completions`` API.

Phase 5b adds schema-constrained decoding: when the playbook passes
a JSON Schema via ``response_schema=``, the backend forwards it as
``response_format={"type": "json_schema", "json_schema": {...}}`` on
the chat completion. NeMo / NIM honor this exactly like the OpenAI
Structured Outputs API — the model is constrained to emit JSON that
validates against the schema, so playbooks get clean parses even on
small / instruction-shaky models. The playbook parser still runs
afterwards as a defense-in-depth check (and a no-op fall-through on
backends that ignore the schema).

Configuration (set in env / .env):

    NEMO_API_URL          — base URL of the NIM, e.g. ``http://localhost:8000``
    NEMO_API_KEY          — optional bearer token (NGC keys, hosted NIMs, etc.)
    NEMO_DEFAULT_MODEL    — required; the model identifier the NIM serves
    NEMO_TIMEOUT_SECONDS  — request timeout (default 600s, matches Ollama)

The NeMo backend is **opt-in**. It only appears in ``pick_backend``'s
auto-resolution path when ``NEMO_API_URL`` is set and the endpoint
answers a reachability probe. Existing Ollama / Teacher setups see
no change.
"""

from __future__ import annotations

from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - tests guard via mocks
    httpx = None  # type: ignore[assignment]

from app.config import settings

from .base import SynthBackendError


class NemoBackend:
    """NeMo Data Designer / NIM OpenAI-compatible chat backend."""

    name: str = "nemo"
    # NIM honors response_format=json_schema natively (OpenAI
    # Structured-Outputs shape). Phase 5b forwards it on every
    # complete() that carries a response_schema.
    schema_aware: bool = True

    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ):
        host = (host if host is not None else settings.NEMO_API_URL) or ""
        self._host = host.rstrip("/")
        self._api_key = (
            api_key if api_key is not None else settings.NEMO_API_KEY
        ) or ""
        self._model = (
            model if model else (settings.NEMO_DEFAULT_MODEL or None)
        )
        self._timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.NEMO_TIMEOUT_SECONDS
        )

    @classmethod
    def is_available(cls) -> bool:
        """True when ``NEMO_API_URL`` is configured AND the endpoint
        answers ``GET /v1/models`` within a 1-second budget.

        A 401 / 403 is still treated as "reachable" — the user has the
        endpoint but a missing/invalid key. We surface the NIM in the
        picker so the user can see it + fix the key from the docs;
        otherwise they'd think the backend isn't supported."""
        if httpx is None:
            return False
        host = (settings.NEMO_API_URL or "").rstrip("/")
        if not host:
            return False
        headers: dict[str, str] = {}
        if settings.NEMO_API_KEY:
            headers["Authorization"] = f"Bearer {settings.NEMO_API_KEY}"
        try:
            resp = httpx.get(
                f"{host}/v1/models",
                headers=headers,
                timeout=1.0,
            )
        except Exception:  # noqa: BLE001 — any transport error means "not available"
            return False
        if resp.status_code == 200:
            return True
        # Auth failures still mean the endpoint is reachable; surface it
        # so the picker can show "NeMo (auth required)" rather than
        # silently hiding the option.
        return resp.status_code in (401, 403)

    def describe(self) -> str:
        if self._model:
            return f"{self.name}:{self._model}"
        return self.name

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_schema: dict | None = None,
    ) -> str:
        if httpx is None:
            raise SynthBackendError(
                "httpx is not installed; install httpx to use NemoBackend."
            )
        if not self._host:
            raise SynthBackendError(
                "NEMO_API_URL is not set. Point it at your NIM / NeMo "
                "Data Designer endpoint (e.g. http://localhost:8000)."
            )
        if not self._model:
            raise SynthBackendError(
                "NEMO_DEFAULT_MODEL is not set and no model was passed. "
                "Pin a model name (e.g. 'meta/llama-3.1-70b-instruct') "
                "either via the env var or via the backend picker."
            )

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_schema is not None:
            # NIM / NeMo Data Designer mirrors the OpenAI Structured
            # Outputs shape. `strict: true` forces the decoder to stay
            # on-schema; `name` is a label NIM logs back in errors.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "synth_row",
                    "schema": response_schema,
                    "strict": True,
                },
            }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._host}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as e:
            raise SynthBackendError(
                f"NeMo timed out after {self._timeout:.0f}s generating with "
                f"{self._model!r}. Lower target_count, pick a smaller "
                f"served model, or raise NEMO_TIMEOUT_SECONDS."
            ) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_excerpt = (e.response.text or "")[:200]
            if status in (401, 403):
                hint = " — check NEMO_API_KEY"
            elif status == 404:
                hint = (
                    f" — is {self._model!r} loaded on this NIM? "
                    f"Hit GET {self._host}/v1/models to list available models."
                )
            else:
                hint = ""
            raise SynthBackendError(
                f"NeMo returned HTTP {status} for model {self._model!r}: "
                f"{body_excerpt}{hint}"
            ) from e
        except httpx.HTTPError as e:
            raise SynthBackendError(
                f"NeMo request failed for model {self._model!r}: {e}. "
                f"Is the NIM at {self._host} still running?"
            ) from e
        except ValueError as e:
            # JSON decode error — NIM returned non-JSON (e.g. an
            # HTML 502 from a reverse proxy).
            raise SynthBackendError(
                f"NeMo returned a non-JSON response for model "
                f"{self._model!r}: {e}"
            ) from e

        # OpenAI-compatible response shape — same as Ollama.
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise SynthBackendError(
                f"NeMo returned an unexpected response shape: "
                f"{str(data)[:200]!r}"
            ) from e
        return str(content or "")

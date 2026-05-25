"""vLLM synth backend (USER-SUCCESS Epic 5 Phase 5c).

vLLM exposes a full OpenAI-compatible server (``vllm serve <model>
--host 0.0.0.0 --port 8000``). Unlike Ollama's partial ``/v1`` shim,
vLLM honors ``response_format={"type": "json_schema", ...}``
**end-to-end** — the decoder is constrained to JSON validating
against the supplied schema. This is the backend that actually
exercises the Phase 5b schema on small / local models.

Transport-wise it's identical to ``NemoBackend``: OpenAI-compatible
HTTP POSTs to ``/v1/chat/completions``, optional bearer auth, a
reachability probe via ``GET /v1/models``. The only real difference
is the env-var namespace (``VLLM_*`` instead of ``NEMO_*``) so
operators running both can pin them independently.

Configuration (set in env / .env):

    VLLM_API_URL          — base URL of the vLLM server,
                            e.g. ``http://localhost:8000``
    VLLM_API_KEY          — optional bearer token (when vLLM was
                            launched with ``--api-key …``)
    VLLM_DEFAULT_MODEL    — required; the model id vLLM is serving
                            (vLLM serves one model per process)
    VLLM_TIMEOUT_SECONDS  — request timeout (default 600s)

Auto-pick order: vLLM lands **after** Ollama + Teacher + NeMo in
``BACKEND_REGISTRY``. Existing installs see no behavior change —
vLLM is only used when the user pins it via the picker, or when
nothing else is reachable.
"""

from __future__ import annotations

from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - tests guard via mocks
    httpx = None  # type: ignore[assignment]

from app.config import settings

from .base import SynthBackendError


class VllmBackend:
    """vLLM OpenAI-compatible chat backend with schema-constrained decoding."""

    name: str = "vllm"

    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ):
        host = (host if host is not None else settings.VLLM_API_URL) or ""
        self._host = host.rstrip("/")
        self._api_key = (
            api_key if api_key is not None else settings.VLLM_API_KEY
        ) or ""
        self._model = (
            model if model else (settings.VLLM_DEFAULT_MODEL or None)
        )
        self._timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.VLLM_TIMEOUT_SECONDS
        )

    @classmethod
    def is_available(cls) -> bool:
        """True when ``VLLM_API_URL`` is configured AND the endpoint
        answers ``GET /v1/models`` within a 1-second budget.

        401/403 is still treated as "reachable" (auth required but
        endpoint exists) — same rationale as NemoBackend: surface it
        in the picker so the user can see the option + fix the key,
        rather than silently hiding the backend."""
        if httpx is None:
            return False
        host = (settings.VLLM_API_URL or "").rstrip("/")
        if not host:
            return False
        headers: dict[str, str] = {}
        if settings.VLLM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.VLLM_API_KEY}"
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
                "httpx is not installed; install httpx to use VllmBackend."
            )
        if not self._host:
            raise SynthBackendError(
                "VLLM_API_URL is not set. Point it at your vLLM server "
                "(e.g. http://localhost:8000)."
            )
        if not self._model:
            raise SynthBackendError(
                "VLLM_DEFAULT_MODEL is not set and no model was passed. "
                "Pin the model id vLLM is serving (e.g. "
                "'meta-llama/Meta-Llama-3.1-8B-Instruct') either via the "
                "env var or via the backend picker."
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
            # vLLM honors response_format=json_schema natively
            # (https://docs.vllm.ai/en/latest/features/structured_outputs.html).
            # ``strict: true`` is OpenAI-spec for "stay on-schema or fail";
            # vLLM's xgrammar/outlines backend enforces it during decoding.
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
                f"vLLM timed out after {self._timeout:.0f}s generating with "
                f"{self._model!r}. Lower target_count, pick a smaller "
                f"served model, or raise VLLM_TIMEOUT_SECONDS."
            ) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_excerpt = (e.response.text or "")[:200]
            if status in (401, 403):
                hint = " — check VLLM_API_KEY"
            elif status == 404:
                hint = (
                    f" — is {self._model!r} the model vLLM is serving? "
                    f"Hit GET {self._host}/v1/models to list available models."
                )
            elif status == 400 and "json_schema" in body_excerpt.lower():
                # vLLM rejects unsupported schema features (e.g. $ref,
                # certain enum shapes) with a 400. Surface that hint so
                # the playbook author can simplify the schema.
                hint = (
                    " — vLLM rejected the JSON Schema. Check the playbook's "
                    "response_schema() output; vLLM's structured-outputs "
                    "backend doesn't support every JSON Schema feature."
                )
            else:
                hint = ""
            raise SynthBackendError(
                f"vLLM returned HTTP {status} for model {self._model!r}: "
                f"{body_excerpt}{hint}"
            ) from e
        except httpx.HTTPError as e:
            raise SynthBackendError(
                f"vLLM request failed for model {self._model!r}: {e}. "
                f"Is the vLLM server at {self._host} still running?"
            ) from e
        except ValueError as e:
            raise SynthBackendError(
                f"vLLM returned a non-JSON response for model "
                f"{self._model!r}: {e}"
            ) from e

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise SynthBackendError(
                f"vLLM returned an unexpected response shape: "
                f"{str(data)[:200]!r}"
            ) from e
        return str(content or "")

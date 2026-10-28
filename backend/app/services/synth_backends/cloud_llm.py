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

import asyncio
from typing import Any, Awaitable, Callable, Literal

from app.config import settings
from app.services import cloud_llm_service, llm_budget_guard
from app.services.cloud_llm_service import (
    CloudLlmError,
    call_anthropic_chat,
    call_openai_chat,
)

from .base import SynthBackendError


Provider = Literal["openai", "anthropic", "deepseek"]

# Output tokens per reply that every catalogued model of the provider accepts and
# that finish inside the 300 s HTTP timeout. Playbooks size a call's max_tokens to
# the rows it asks for (and a scenario grid its plans per call) up to this.
_MAX_OUTPUT_TOKENS: dict[str, int] = {"anthropic": 16000, "openai": 16384, "deepseek": 8192}
# A model that thinks by default spends thinking tokens from max_tokens: each call
# gets this much on top of the answer budget, up to the direct ceiling above, or up
# to _BATCH_MAX_OUTPUT_TOKENS in a Message Batch (no HTTP timeout there).
THINKING_HEADROOM_TOKENS = 8000
_BATCH_MAX_OUTPUT_TOKENS = 32000


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
    # Prompts leave this machine; billed per token against API credits.
    sends_data_off_machine: bool = True
    billing: str = "metered"

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
        # Running totals for this instance (one per job) — what a relabel
        # reports as its token use and estimated cost.
        self.usage: dict[str, int] = {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0,
        }
        # Message Batches (Anthropic only): set by the API when the caller
        # asked for a batch; labelling then submits its calls through
        # ``complete_batch``. Batch tokens are counted apart (half price).
        self.use_batch: bool = False
        self.batch_usage: dict[str, int] = {k: 0 for k in self.usage}
        self.batch_report: list[dict[str, Any]] = []

    # Labelling prompts can pass their stable part separately (``cache_prefix``).
    supports_prompt_cache: bool = True

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_output_tokens(self) -> int:
        return _MAX_OUTPUT_TOKENS.get(self._provider, 4096)

    @property
    def effort(self) -> str | None:
        """``SYNTH_CLOUD_EFFORT`` when it's a valid level (sent only to models that take it)."""
        value = str(getattr(settings, "SYNTH_CLOUD_EFFORT", "") or "").strip().lower()
        return value if value in cloud_llm_service.EFFORT_LEVELS else None

    def _reply_budget(self, max_tokens: int, *, batch: bool = False) -> int:
        """``max_tokens`` plus thinking headroom for a model that thinks by default."""
        if self._provider != "anthropic" or not cloud_llm_service.anthropic_thinks_by_default(self._model):
            return max_tokens
        ceiling = _BATCH_MAX_OUTPUT_TOKENS if batch else self.max_output_tokens
        return max(max_tokens, min(ceiling, max_tokens + THINKING_HEADROOM_TOKENS))

    @property
    def playbook_rows_per_call(self) -> int:
        """A playbook run over this many rows is chunked + checkpointed
        (``synth_playbook_service._run_chunked``), like the command backend."""
        try:
            n = int(settings.SYNTH_CLOUD_PLAYBOOK_ROWS_PER_CALL)
        except (TypeError, ValueError):
            n = 10
        return max(1, min(n, 100))

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

    @property
    def supports_batch(self) -> bool:
        return self._provider == "anthropic"

    async def complete_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        progress: Callable[[str, int, int], Awaitable[None]] | None = None,
    ) -> list[str | None]:
        """Send ``requests`` (each ``{prompt, system_prompt, max_tokens,
        temperature, cache_prefix}`` — what ``complete`` takes) as ONE
        Anthropic Message Batch, wait for it, and return each request's text
        in order (None where it errored / expired — the caller retries those
        synchronously). ``progress(message, done, total)`` after each poll.
        A model that rejects ``temperature`` fails every request with the
        same 400; those are resubmitted once without it."""
        if not self.supports_batch:
            raise SynthBackendError(f"Message batches are Anthropic-only (this backend is {self.describe()}).")
        await llm_budget_guard.acheck_budget(self._provider, self._model)  # hard monthly API budget, before submitting
        out: list[str | None] = [None] * len(requests)
        todo = list(range(len(requests)))
        for attempt in range(2):
            if attempt:  # the temperature resubmit is a new spend: re-check the cap
                await llm_budget_guard.acheck_budget(self._provider, self._model)
            batch_req = [
                {
                    "custom_id": f"r{i}",
                    "params": cloud_llm_service.anthropic_message_params(
                        model=self._model,
                        system_prompt=str(requests[i].get("system_prompt") or ""),
                        user_prompt=str(requests[i]["prompt"]),
                        max_tokens=self._reply_budget(int(requests[i].get("max_tokens") or 1024), batch=True),
                        temperature=float(requests[i].get("temperature", 0.7)),
                        cached_system=requests[i].get("cache_prefix"),
                        effort=self.effort,
                    ),
                }
                for i in todo
            ]
            batch_id = ""
            try:
                batch = await cloud_llm_service.create_anthropic_batch(api_key=self._api_key, requests=batch_req)
                batch_id = str(batch.get("id") or "")
                total = len(batch_req)

                async def _on_status(b: dict[str, Any]) -> None:
                    if progress is None:
                        return
                    counts = b.get("request_counts") or {}
                    done = sum(int(counts.get(k) or 0) for k in ("succeeded", "errored", "canceled", "expired"))
                    await progress(f"batch submitted · {total} requests · waiting ({done}/{total} done)", done, total)

                if progress is not None:
                    await progress(f"batch submitted · {total} requests · waiting", 0, total)
                ended = await cloud_llm_service.wait_for_anthropic_batch(
                    api_key=self._api_key, batch_id=batch_id, on_status=_on_status,
                )
                results = await cloud_llm_service.fetch_anthropic_batch_results(
                    api_key=self._api_key, results_url=str(ended.get("results_url") or ""),
                )
            except CloudLlmError as e:
                raise SynthBackendError(f"Cloud LLM batch ({self._provider}:{self._model}) failed: {e}") from e
            except asyncio.CancelledError:
                # The Job was cancelled (``progress`` may raise it after a
                # poll): stop the batch so unfinished requests aren't billed.
                if batch_id:
                    try:
                        await cloud_llm_service.cancel_anthropic_batch(api_key=self._api_key, batch_id=batch_id)
                    except Exception:  # noqa: BLE001 — best-effort
                        pass
                raise
            failed: list[int] = []
            no_text = 0
            attempt_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0}
            temperature_rejected = False
            for i in todo:
                result = results.get(f"r{i}")
                why = cloud_llm_service.batch_result_error(result)
                if why:
                    failed.append(i)
                    low = why.lower()
                    if "temperature" in low and ("deprecated" in low or "not supported" in low):
                        temperature_rejected = True
                    continue
                resp = cloud_llm_service.anthropic_response_from_message(result.get("message") or {}, self._model)
                out[i] = resp.content or ""
                if not resp.content and resp.stop_reason == "max_tokens":
                    no_text += 1
                self.batch_usage["calls"] += 1
                self.batch_usage["input_tokens"] += resp.prompt_tokens
                self.batch_usage["output_tokens"] += resp.completion_tokens
                self.batch_usage["cache_write_tokens"] += resp.cache_write_tokens
                self.batch_usage["cache_read_tokens"] += resp.cache_read_tokens
                attempt_usage["calls"] += 1
                attempt_usage["input_tokens"] += resp.prompt_tokens
                attempt_usage["output_tokens"] += resp.completion_tokens
                attempt_usage["cache_write_tokens"] += resp.cache_write_tokens
                attempt_usage["cache_read_tokens"] += resp.cache_read_tokens
            # Batch spend feeds the budget ledger at half price. Not through
            # ``llm_usage.record``: the Job meter already counts batch usage via
            # ``batch_usage`` (recording would double count it).
            if attempt_usage["calls"]:
                llm_budget_guard.note_usage(self._provider, self._model, attempt_usage, batch=True)
            report = {
                "batch_id": batch_id, "requests": len(todo),
                "succeeded": len(todo) - len(failed), "failed": len(failed),
            }
            if no_text:
                # Answered, but the whole budget went to thinking: no text to parse.
                report["budget_spent_thinking"] = no_text
            self.batch_report.append(report)
            if temperature_rejected and attempt == 0 and self._model not in cloud_llm_service._NO_TEMPERATURE_MODELS:
                cloud_llm_service._NO_TEMPERATURE_MODELS.add(self._model)
                todo = failed
                continue
            break
        return out

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_schema: dict | None = None,  # noqa: ARG002 — cloud_llm_service uses force_json instead
        cache_prefix: str | None = None,
        cache_system: bool = False,
    ) -> str:
        """``cache_prefix``: the stable part of the prompt (instructions that
        repeat across calls). Anthropic gets it as a cached system block;
        other providers get it prepended to ``prompt``. ``cache_system``:
        mark the system prompt itself as the cached block (same content —
        for callers whose stable instructions already live there; OpenAI
        caches long prefixes on its own, so nothing changes there)."""
        await llm_budget_guard.acheck_budget(self._provider, self._model)  # hard monthly API budget
        # call_*_chat both return CloudLlmResponse with a ``content``
        # str. Map any CloudLlmError to SynthBackendError so the
        # playbook framework's existing 5xx handling kicks in.
        try:
            if self._provider == "anthropic":
                cached = cache_prefix
                system = system_prompt or ""
                if cache_system and not cache_prefix and system:
                    cached, system = system, ""
                resp = await call_anthropic_chat(
                    api_key=self._api_key,
                    model=self._model,
                    system_prompt=system,
                    user_prompt=prompt,
                    max_tokens=self._reply_budget(max_tokens),
                    temperature=temperature,
                    cached_system=cached,
                    effort=self.effort,
                )
                if not resp.content and resp.stop_reason == "max_tokens":
                    raise SynthBackendError(
                        f"Cloud LLM ({self._provider}:{self._model}) spent its whole reply budget "
                        f"({resp.completion_tokens} tokens) thinking and returned no answer. Lower "
                        f"SYNTH_CLOUD_EFFORT (now {self.effort or 'the model default'}) or ask for fewer rows per call."
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
                    user_prompt=(cache_prefix + "\n\n" + prompt) if cache_prefix else prompt,
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
        self.usage["calls"] += 1
        self.usage["input_tokens"] += int(getattr(resp, "prompt_tokens", 0) or 0)
        self.usage["output_tokens"] += int(getattr(resp, "completion_tokens", 0) or 0)
        self.usage["cache_write_tokens"] += int(getattr(resp, "cache_write_tokens", 0) or 0)
        self.usage["cache_read_tokens"] += int(getattr(resp, "cache_read_tokens", 0) or 0)
        return resp.content or ""

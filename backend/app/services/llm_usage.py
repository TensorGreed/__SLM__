"""Cloud-LLM token metering: who spent what, per operation.

Every cloud call (``cloud_llm_service.call_anthropic_chat`` /
``call_openai_chat``, the teacher endpoints when they point at a non-local
host) reports its token counts with ``record(...)``. A ``UsageMeter`` opened
with ``metered()`` collects every call made inside it — through ``await``s,
``asyncio.to_thread`` and nested ``asyncio.run`` (contextvars are copied into
both) — so an operation reads its own spend without threading counters
through every helper.

* Every Job gets a root meter (``jobs_service._runner_wrapper``); its result
  gains ``usage`` (list of ``llm_pricing.estimate_cost`` dicts, one per
  backend) + ``cost_usd`` + ``tokens`` when the runner spent any — the same
  shape as a relabel summary — also on a FAILED Job (tokens spent are spent).
* Operations that are not Jobs (the LLM answer key, the plan-refinement
  strategy pass, the probe judge inside a synchronous eval, distillation
  capture) use ``metered_operation(project_id, kind)``: when it is the
  outermost meter, the spend is appended to
  ``data/projects/<id>/llm_spend.jsonl`` so the project's spend tally
  (``llm_spend_service``) sees it. Inside a Job it only nests — the Job
  result already counts those calls.

Best-effort throughout: accounting never fails the operation.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlparse

from app.config import settings
from app.services import llm_budget_guard

_LOG = logging.getLogger(__name__)

LEDGER_FILE = "llm_spend.jsonl"
_COUNTERS = ("calls", "input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens")

_ACTIVE: contextvars.ContextVar[tuple["UsageMeter", ...]] = contextvars.ContextVar("llm_usage_meters", default=())


class UsageMeter:
    """Per-backend counters (``provider:model``) for the calls made inside it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_backend: dict[str, dict[str, Any]] = {}
        # True when some counts were estimated (a stream that sent no usage
        # event): the ledger entry says so.
        self.estimated = False

    def add(self, provider: str, model: str, counts: dict[str, int]) -> None:
        key = f"{provider}:{model}" if provider else str(model)
        with self._lock:
            entry = self._by_backend.setdefault(key, {"model": model, **{c: 0 for c in _COUNTERS}})
            for c in _COUNTERS:
                entry[c] += int(counts.get(c) or 0)

    @property
    def calls(self) -> int:
        with self._lock:
            return sum(e["calls"] for e in self._by_backend.values())

    def entries(self) -> list[dict[str, Any]]:
        """``[{backend, **estimate_cost(...)}]`` — the relabel summary's ``usage``."""
        from app.services.llm_pricing import estimate_cost

        with self._lock:
            items = [(k, dict(v)) for k, v in self._by_backend.items()]
        return [{"backend": k, **estimate_cost(v["model"], v)} for k, v in items if v["calls"]]

    def block(self) -> dict[str, Any]:
        """``{usage, cost_usd, tokens}`` — empty dict when nothing was spent."""
        return usage_fields(self.entries())


def usage_fields(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {}
    priced = [u["usd"] for u in entries if u.get("usd") is not None]
    return {
        "usage": entries,
        "cost_usd": round(sum(priced), 4) if priced else None,
        "tokens": sum(int(u.get("total_tokens") or 0) for u in entries),
    }


def is_local_url(url: str | None) -> bool:
    """A teacher endpoint on this machine / LAN (Ollama, vLLM) costs nothing
    and is not cloud spend."""
    try:
        parsed = urlparse(str(url or ""))
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return True
    if host in {"localhost", "0.0.0.0", "::1", "host.docker.internal", "ollama"} or host.startswith("127."):
        return True
    if host.startswith(("10.", "192.168.")) or host.endswith(".local") or parsed.port == 11434:
        return True
    if host.startswith("172."):
        try:
            return 16 <= int(host.split(".")[1]) <= 31
        except (IndexError, ValueError):
            return False
    return False


def record(
    provider: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> None:
    """Report one call to every open meter (and, when priced, to the monthly
    budget ledger — ``llm_budget_guard``). Never raises."""
    try:
        counts = {
            "calls": 1, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_write_tokens": cache_write_tokens, "cache_read_tokens": cache_read_tokens,
        }
        llm_budget_guard.note_usage(provider, str(model or "unknown"), counts)
        meters = _ACTIVE.get()
        if not meters:
            return
        for meter in meters:
            meter.add(provider, str(model or "unknown"), counts)
    except Exception:  # noqa: BLE001 — accounting never breaks a call
        _LOG.debug("llm usage record failed", exc_info=True)


def record_response(provider: str, resp: Any, model: str | None = None) -> None:
    """``record`` from a ``CloudLlmResponse``."""
    record(
        provider,
        str(getattr(resp, "model", None) or model or "unknown"),
        input_tokens=int(getattr(resp, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(resp, "completion_tokens", 0) or 0),
        cache_write_tokens=int(getattr(resp, "cache_write_tokens", 0) or 0),
        cache_read_tokens=int(getattr(resp, "cache_read_tokens", 0) or 0),
    )


def provider_for_url(url: str | None) -> str:
    """``anthropic`` / ``deepseek`` / ``openai`` from an endpoint host, else the host."""
    host = (urlparse(str(url or "")).hostname or "openai-compatible").lower()
    return "anthropic" if "anthropic" in host else "deepseek" if "deepseek" in host else "openai" if "openai" in host else host


def record_openai_usage(url: str | None, model: str, usage: Any) -> None:
    """``record`` from an OpenAI-shape ``usage`` dict, skipped for local hosts."""
    if is_local_url(url) or not isinstance(usage, dict):
        return
    provider = provider_for_url(url)
    record(
        provider, model,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
    )


@contextlib.contextmanager
def metered(*, root: bool = False) -> Iterator[UsageMeter]:
    """Collect the calls made inside. ``root=True`` detaches from any outer
    meter (a Job spawned from another Job counts its own calls only)."""
    meter = UsageMeter()
    token = _ACTIVE.set((meter,) if root else (*_ACTIVE.get(), meter))
    try:
        yield meter
    finally:
        _ACTIVE.reset(token)


def has_active_meter() -> bool:
    return bool(_ACTIVE.get())


@contextlib.contextmanager
def metered_operation(project_id: int | None, kind: str) -> Iterator[UsageMeter]:
    """A meter for a non-Job operation; the outermost one records the spend in
    the project's ledger (also when the operation raises)."""
    outermost = not has_active_meter()
    # The project context: which project's monthly cap a metered call inside
    # counts against (and its command-backend overrides). Inside a Job the
    # Job's context stays unless this operation names a project.
    outer = llm_budget_guard.current_project.get()
    token = llm_budget_guard.current_project.set(
        llm_budget_guard.ProjectContext(project_id=project_id, synth_command=dict(outer.synth_command) if outer else {})
    ) if project_id is not None else None
    try:
        with metered() as meter:
            try:
                yield meter
            finally:
                if outermost and project_id is not None:
                    append_ledger(project_id, kind, meter)
    finally:
        if token is not None:
            llm_budget_guard.current_project.reset(token)


def ledger_path(project_id: int):
    return settings.DATA_DIR / "projects" / str(project_id) / LEDGER_FILE


def append_ledger(project_id: int, kind: str, meter: UsageMeter) -> None:
    try:
        fields = meter.block()
        if not fields:
            return
        path = ledger_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"at": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields}
        if meter.estimated:
            entry["estimated"] = True
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: BLE001
        _LOG.debug("llm spend ledger write failed", exc_info=True)


def merge_into_result(result: dict[str, Any], meter: UsageMeter) -> dict[str, Any]:
    """Add the meter's ``usage`` / ``cost_usd`` / ``tokens`` to a Job result
    unless the runner reported its own (a relabel does)."""
    try:
        fields = meter.block()
        if not fields:
            return result
        if is_usage_list(result.get("usage")):
            return result
        out = dict(result)
        if "usage" in out:
            out["llm_usage"] = fields["usage"]
        else:
            out["usage"] = fields["usage"]
        out.setdefault("cost_usd", fields["cost_usd"])
        out.setdefault("tokens", fields["tokens"])
        return out
    except Exception:  # noqa: BLE001
        return result


# A run that paid for calls the Job meter never sees (Message Batches) and then
# fails hangs its own usage on the exception; the Job writes it on FAILED /
# CANCELLED too, so spent money never drops out of the spend tally.
RESULT_USAGE_ATTR = "llm_result_usage"
_RESULT_USAGE_KEYS = ("usage", "cost_usd", "tokens", "batch")


def attach_result_usage(exc: BaseException, result: Any) -> None:
    """Hang ``result``'s own ``usage`` / ``cost_usd`` / ``tokens`` / ``batch`` on ``exc``."""
    if isinstance(result, dict) and is_usage_list(result.get("usage")):
        try:
            setattr(exc, RESULT_USAGE_ATTR, {k: result[k] for k in _RESULT_USAGE_KEYS if k in result})
        except Exception:  # noqa: BLE001 — best-effort
            pass


def failure_block(meter: UsageMeter, exc: BaseException) -> dict[str, Any]:
    """The usage a failed / cancelled Job reports: the run's own (it supersedes the
    meter, as in ``merge_into_result``) when it attached one, else the meter's."""
    own = getattr(exc, RESULT_USAGE_ATTR, None)
    return dict(own) if isinstance(own, dict) else meter.block()


def is_usage_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(u, dict) and "total_tokens" in u for u in value)


def result_usage(result: Any) -> list[dict[str, Any]]:
    """The usage entries a Job result carries (``usage`` or ``llm_usage``)."""
    if not isinstance(result, dict):
        return []
    for key in ("usage", "llm_usage"):
        if is_usage_list(result.get(key)):
            return list(result[key])
    return []


def cache_kwargs(llm: Any) -> dict[str, Any]:
    """``{"cache_system": True}`` for a backend that can mark its system
    prompt as a cached block (CloudLlmBackend), else ``{}``."""
    return {"cache_system": True} if getattr(llm, "supports_prompt_cache", False) else {}

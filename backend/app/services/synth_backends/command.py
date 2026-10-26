"""Command (local CLI) synth backend.

Runs an allow-listed command-line tool as the teacher — typically a vendor
CLI signed in to a *subscription* (e.g. ``claude -p``), so generation costs no
per-call API credits. Pin token: ``command`` or ``command:<model>``.

Safety model (all install-wide settings; a project can never choose them):

* ``SYNTH_COMMAND_ALLOWED_BINARIES`` — comma list of bare names (match
  ``basename(argv[0])``) or absolute paths (match the resolved executable's
  realpath). Empty = backend disabled. Checked at construction AND per call.
* never a shell: ``asyncio.create_subprocess_exec`` with an argv list;
  placeholders (``{model}``, ``{prompt}``) are substituted inside single argv
  elements only. A prompt starting with ``-`` is prefixed with a space in
  ``arg`` mode so it can never parse as a flag.
* minimal environment (PATH, HOME, locale, XDG dirs, TMPDIR + the names in
  ``SYNTH_COMMAND_ENV_PASSTHROUGH``) minus API-key variables that would make
  such a CLI bill the metered API instead of the subscription.
* a fresh temporary working directory per call (the CLI never sees the repo).
* the process is killed on timeout and on task cancellation.

``max_tokens`` / ``temperature`` / ``response_schema`` are accepted for the
protocol but a CLI has no such knobs: ``response_schema`` becomes an
instruction in the prompt; the other two are ignored.

Usage is recorded as ``subscription:<binary>`` (priced $0, shown apart from
metered spend, never counted against the API budget).

A usage-limit failure (``SYNTH_COMMAND_USAGE_LIMIT_REGEX`` on stdout+stderr)
pauses ALL calls of the process (shared cool-down), publishes the running
Job's message "waiting for usage window — resumes ~HH:MM UTC", then retries
the same call (the pause does not consume a retry).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from app.config import settings
from app.services import sandbox_exec

from .base import SynthBackendError

_LOG = logging.getLogger(__name__)

# No leading "-": a model name is substituted into argv and must never parse as a flag.
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}$")
# A usage-limit pause is never shorter than this (a reset epoch in the past, or
# a tiny cool-down, must not turn into a hot loop of failing calls).
_MIN_COOLDOWN_SECONDS = 60.0
# Consecutive usage-limit hits one call tolerates before it gives up.
_MAX_LIMIT_HITS = 20
_CANCEL_POLL_SECONDS = 2.0
_WAITING_PREFIX = "waiting for usage window"
# Minimal-env / kill helpers are shared with the code-check sandbox.
_BASE_ENV = sandbox_exec.BASE_ENV
_DENIED_ENV = sandbox_exec.DENIED_ENV
_OVERRIDE_KEYS = ("model", "concurrency", "batch_size", "timeout_seconds", "cooldown_seconds", "max_retries")
_PAUSE_SLICE_SECONDS = 5.0
_TAIL_CHARS = 400


# ── settings readers (read per call so tests / runtime overrides apply) ──


def _allowed_entries() -> list[str]:
    return [e.strip() for e in str(settings.SYNTH_COMMAND_ALLOWED_BINARIES or "").split(",") if e.strip()]


def _argv_template() -> list[str]:
    try:
        argv = json.loads(settings.SYNTH_COMMAND_ARGV)
    except (TypeError, ValueError) as exc:
        raise SynthBackendError(f"SYNTH_COMMAND_ARGV is not a JSON list: {exc}") from exc
    # Later elements may be empty strings (e.g. `--tools ""` turns a CLI's tools off).
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv) or not argv[0]:
        raise SynthBackendError("SYNTH_COMMAND_ARGV must be a non-empty JSON list of strings.")
    return argv


def _prompt_mode() -> str:
    mode = str(settings.SYNTH_COMMAND_PROMPT_MODE or "stdin").strip().lower()
    return mode if mode in {"stdin", "arg"} else "stdin"


def _minimal_env() -> dict[str, str]:
    return sandbox_exec.minimal_env(settings.SYNTH_COMMAND_ENV_PASSTHROUGH)


def _resolve_executable(argv0: str) -> str | None:
    env_path = os.environ.get("PATH")
    found = shutil.which(argv0, path=env_path) if not os.path.isabs(argv0) else (argv0 if os.access(argv0, os.X_OK) else None)
    return found


def _is_allowed(argv0: str, resolved: str | None) -> bool:
    if resolved is None:
        return False
    for entry in _allowed_entries():
        if os.path.isabs(entry):
            if os.path.realpath(entry) == os.path.realpath(resolved):
                return True
        elif "/" not in argv0 and entry == argv0:  # a bare name only matches a bare argv[0]
            return True
    return False


def _check_allowed(argv0: str) -> str:
    if not _allowed_entries():
        raise SynthBackendError("command backend is disabled: set SYNTH_COMMAND_ALLOWED_BINARIES to the CLI's name or absolute path.")
    resolved = _resolve_executable(argv0)
    if not _is_allowed(argv0, resolved):
        raise SynthBackendError(f"binary not allow-listed: {os.path.basename(argv0)}")
    assert resolved is not None
    return resolved


def _clamp(value: Any, lo: float, hi: float, default: float, *, integer: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    number = max(lo, min(hi, number))
    return int(number) if integer else number


# ── shared state: usage-limit pause, per-loop concurrency ──

_PAUSE: dict[str, float] = {"until_mono": 0.0, "until_wall": 0.0}
_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Any, asyncio.Semaphore]]" = weakref.WeakKeyDictionary()


def is_paused() -> bool:
    return time.monotonic() < _PAUSE["until_mono"]


def reset_pause() -> None:
    _PAUSE["until_mono"] = 0.0
    _PAUSE["until_wall"] = 0.0


def _global_limit() -> int:
    return int(_clamp(getattr(settings, "SYNTH_COMMAND_CONCURRENCY", 1), 1, 8, 1, integer=True))


def _semaphore(key: Any, limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    per_loop = _SEMAPHORES.setdefault(loop, {})
    return per_loop.setdefault((key, limit), asyncio.Semaphore(limit))


@dataclass
class _Outcome:
    ok: bool
    text: str = ""
    usage: dict[str, int] | None = None
    error: str = ""
    # stdout + stderr of a failed call (what the usage-limit regex reads)
    raw: str = ""
    # the usage window closed while this call waited for a slot: not run, ask again
    paused: bool = False


def _dig(data: Any, path: str) -> Any:
    """``.result`` / ``.data.text`` / ``.choices.0.text`` → the value (KeyError when absent)."""
    cur = data
    for part in [p for p in path.strip().split(".") if p]:
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(part)
    return cur


def _usage_from(data: Any) -> dict[str, int]:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {}

    def _n(key: str) -> int:
        try:
            return max(0, int(usage.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    return {
        "input_tokens": _n("input_tokens"), "output_tokens": _n("output_tokens"),
        "cache_read_tokens": _n("cache_read_input_tokens"), "cache_write_tokens": _n("cache_creation_input_tokens"),
    }


class CommandBackend:
    """An allow-listed local CLI as the teacher (subscription billing)."""

    name: str = "command"
    schema_aware: bool = False
    supports_batch: bool = True
    # Only ever chosen by an explicit pin: never by ``pick_backend(None)`` or
    # the judge auto-walk.
    auto_pick: bool = False
    # Prompts leave this machine (the CLI sends them to its vendor).
    sends_data_off_machine: bool = True
    billing: str = "subscription"

    def __init__(self, *, model: str | None = None) -> None:
        argv = _argv_template()
        _check_allowed(argv[0])
        ctx = _project_overrides()
        chosen = (model or "").strip() or str(ctx.get("model") or "").strip() or str(settings.SYNTH_COMMAND_DEFAULT_MODEL)
        if not _MODEL_RE.match(chosen):
            raise ValueError(f"Invalid command-backend model {chosen!r}: use letters, digits and . _ : / @ - (max 128).")
        self._model = chosen
        self._overrides = ctx
        if _prompt_mode() == "arg" and "{prompt}" not in argv:
            raise SynthBackendError("SYNTH_COMMAND_PROMPT_MODE=arg needs an argv element that is exactly {prompt}.")
        # Packed requests per CLI call; on by default when it packs.
        self.use_batch: bool = self.batch_size > 1
        # One summary per ``complete_batch`` call (CloudLlmBackend-shaped:
        # batch_id / requests / succeeded / failed) + a flat list of the
        # rejected items ({id, reason}).
        self.batch_report: list[dict[str, Any]] = []
        self.rejected_items: list[dict[str, str]] = []
        self._fallback_tally = None

    # ── settings (+ per-project overrides, clamped) ──

    def _setting(self, key: str, lo: float, hi: float, default: float, *, integer: bool) -> float:
        raw = self._overrides.get(key) if key in self._overrides else None
        if raw is None:
            raw = getattr(settings, f"SYNTH_COMMAND_{key.upper()}", default)
        return _clamp(raw, lo, hi, default, integer=integer)

    @property
    def model(self) -> str:
        return self._model

    @property
    def concurrency(self) -> int:
        return int(self._setting("concurrency", 1, 8, 1, integer=True))

    @property
    def batch_size(self) -> int:
        return int(self._setting("batch_size", 1, 32, 8, integer=True))

    @property
    def timeout_seconds(self) -> float:
        return float(self._setting("timeout_seconds", 1, 86400, 600.0, integer=False))

    @property
    def cooldown_seconds(self) -> float:
        return float(self._setting("cooldown_seconds", 0, 86400, 900.0, integer=False))

    @property
    def max_retries(self) -> int:
        return int(self._setting("max_retries", 0, 10, 2, integer=True))

    @property
    def playbook_rows_per_call(self) -> int:
        return int(_clamp(getattr(settings, "SYNTH_COMMAND_PLAYBOOK_ROWS_PER_CALL", 25), 1, 200, 25, integer=True))

    @property
    def binary(self) -> str:
        return os.path.basename(_argv_template()[0])

    @classmethod
    def is_available(cls) -> bool:
        try:
            argv = _argv_template()
            return _is_allowed(argv[0], _resolve_executable(argv[0])) and bool(_allowed_entries())
        except Exception:  # noqa: BLE001 — a broken config is "unavailable", not an error
            return False

    def describe(self) -> str:
        return "command" if self._model == str(settings.SYNTH_COMMAND_DEFAULT_MODEL) else f"command:{self._model}"

    def detail(self) -> str:
        return f"{self.binary} CLI (subscription) · batch {self.batch_size} · concurrency {self.concurrency}"

    # ── one CLI invocation ──

    def _build_argv(self, prompt: str) -> tuple[list[str], str | None]:
        template = _argv_template()
        mode = _prompt_mode()
        argv: list[str] = []
        for element in template:
            if mode == "arg" and element == "{prompt}":
                argv.append(" " + prompt if prompt.startswith("-") else prompt)
            else:
                argv.append(element.replace("{model}", self._model))
        return argv, (prompt if mode == "stdin" else None)

    async def _invoke(self, prompt: str) -> _Outcome:
        from app.services import jobs_service

        argv, stdin_text = self._build_argv(prompt)
        resolved = _check_allowed(argv[0])  # per call too: settings can change
        argv[0] = resolved
        output_path = str(settings.SYNTH_COMMAND_OUTPUT or ".result").strip()
        handle = jobs_service.current_job_handle.get()
        # ONE process-wide limit (the setting); a per-project override can only
        # lower it, via an extra semaphore taken first.
        async with contextlib.AsyncExitStack() as slots:
            if self.concurrency < _global_limit():
                await slots.enter_async_context(
                    _semaphore(("project", llm_budget_guard_project_id()), self.concurrency)
                )
            await slots.enter_async_context(_semaphore("global", _global_limit()))
            if is_paused():  # a limit was hit while this call queued for a slot
                return _Outcome(False, paused=True)
            try:
                with tempfile.TemporaryDirectory(prefix="brewslm-cmd-") as cwd:
                    proc = await asyncio.create_subprocess_exec(
                        *argv,
                        stdin=asyncio.subprocess.PIPE if stdin_text is not None else asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                        env=_minimal_env(), cwd=cwd, start_new_session=True,
                    )
                    task = asyncio.ensure_future(
                        proc.communicate(stdin_text.encode("utf-8") if stdin_text is not None else None)
                    )
                    deadline = time.monotonic() + self.timeout_seconds
                    try:
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                _kill(proc)
                                task.cancel()
                                with contextlib.suppress(BaseException):
                                    await task
                                with contextlib.suppress(Exception):
                                    await proc.wait()
                                return _Outcome(False, error=f"timed out after {self.timeout_seconds:g}s")
                            done, _pending = await asyncio.wait({task}, timeout=min(_CANCEL_POLL_SECONDS, remaining))
                            if done:
                                out_b, err_b = task.result()
                                break
                            if handle is not None and await handle.check_cancelled():
                                raise asyncio.CancelledError()  # finally: kills the in-flight CLI
                    except BaseException:  # cancelled / interrupted: never leave the CLI running
                        if not task.done():
                            _kill(proc)
                            task.cancel()
                            with contextlib.suppress(BaseException):
                                await task
                            with contextlib.suppress(Exception):
                                await proc.wait()
                        raise
            except OSError as exc:
                return _Outcome(False, error=f"could not run {os.path.basename(argv[0])}: {exc}")
        stdout = out_b.decode("utf-8", "replace")
        stderr = err_b.decode("utf-8", "replace")
        raw = f"{stdout}\n{stderr}"
        data: Any = None
        with contextlib.suppress(ValueError):
            data = json.loads(stdout)
        if proc.returncode != 0:
            return _Outcome(False, error=f"exit {proc.returncode}: {_tail(stderr or stdout)}", raw=raw)
        if isinstance(data, dict) and data.get("is_error"):
            return _Outcome(False, error=f"CLI reported an error: {_tail(str(data.get('result') or stdout))}", raw=raw)
        if output_path.lower() == "raw":
            text = stdout.strip()
        else:
            if data is None:
                return _Outcome(False, error=f"output is not JSON: {_tail(stdout)}", raw=raw)
            try:
                value = _dig(data, output_path)
            except (KeyError, IndexError, ValueError):
                return _Outcome(False, error=f"output path {output_path!r} not found in the CLI's JSON reply", raw=raw)
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return _Outcome(True, text=text, usage=_usage_from(data))

    # ── pause / retry ──

    def _tally(self):
        from app.services import jobs_service

        tally = jobs_service.current_pause_tally.get()
        if tally is None:
            if self._fallback_tally is None:
                self._fallback_tally = jobs_service.PauseTally()
            tally = self._fallback_tally
        return tally

    def _start_pause(self, raw: str) -> None:
        match = re.search(str(settings.SYNTH_COMMAND_USAGE_LIMIT_REGEX), raw, re.IGNORECASE)
        duration = self.cooldown_seconds
        if match is not None:
            reset = (match.groupdict() or {}).get("reset")
            if reset:
                with contextlib.suppress(TypeError, ValueError):
                    until_reset = float(reset) - time.time()
                    if until_reset > 0:  # an epoch in the past falls back to the cool-down
                        duration = until_reset
        duration = max(duration, _MIN_COOLDOWN_SECONDS)
        duration = min(duration, float(settings.SYNTH_COMMAND_MAX_PAUSE_SECONDS))
        now_m, now_w = time.monotonic(), time.time()
        if now_m + duration > _PAUSE["until_mono"]:
            _PAUSE["until_mono"] = now_m + duration
            _PAUSE["until_wall"] = now_w + duration

    def _limit_hit(self, raw: str) -> bool:
        try:
            return re.search(str(settings.SYNTH_COMMAND_USAGE_LIMIT_REGEX), raw, re.IGNORECASE) is not None
        except re.error:
            return False

    async def _wait_if_paused(self) -> None:
        from app.services import jobs_service

        if not is_paused():
            return
        handle = jobs_service.current_job_handle.get()
        tally = self._tally()
        if handle is not None and not str(handle.last_message or "").startswith(_WAITING_PREFIX):
            handle.pre_pause_message = handle.last_message  # remembered once per Job
        announced_until = 0.0
        while is_paused():
            if _PAUSE["until_wall"] != announced_until and handle is not None:
                announced_until = _PAUSE["until_wall"]
                at = datetime.fromtimestamp(announced_until, tz=timezone.utc).strftime("%H:%M")
                with contextlib.suppress(Exception):
                    await handle.set_message(f"{_WAITING_PREFIX} — resumes ~{at} UTC")
            remaining = _PAUSE["until_mono"] - time.monotonic()
            started = time.monotonic()
            await asyncio.sleep(max(0.0, min(_PAUSE_SLICE_SECONDS, remaining)))
            tally.add_interval(started, time.monotonic())
            if tally.seconds > float(settings.SYNTH_COMMAND_MAX_PAUSE_SECONDS):
                raise SynthBackendError(
                    f"usage window did not reopen within {int(float(settings.SYNTH_COMMAND_MAX_PAUSE_SECONDS))}s "
                    "— try again later or switch backend"
                )
            if handle is not None and await handle.check_cancelled():
                raise asyncio.CancelledError()
        previous = getattr(handle, "pre_pause_message", None) if handle is not None else None
        if handle is not None and previous and str(handle.last_message or "").startswith(_WAITING_PREFIX) \
                and not str(previous).startswith(_WAITING_PREFIX):
            with contextlib.suppress(Exception):
                await handle.set_message(previous)

    def _record(self, usage: dict[str, int] | None) -> None:
        from app.services import llm_usage

        llm_usage.record("", f"subscription:{self.binary}", **(usage or {}))

    async def _ask(self, text: str) -> str:
        from app.services import jobs_service

        failures = 0
        limit_hits = 0
        while True:
            await self._wait_if_paused()
            handle = jobs_service.current_job_handle.get()
            if handle is not None and await handle.check_cancelled():
                raise asyncio.CancelledError()
            outcome = await self._invoke(text)
            if outcome.paused:
                continue
            if outcome.ok:
                self._record(outcome.usage)
                return outcome.text
            if self._limit_hit(outcome.raw):
                limit_hits += 1
                if limit_hits > _MAX_LIMIT_HITS:
                    raise SynthBackendError(
                        f"Command backend ({self.binary}) hit the usage limit {_MAX_LIMIT_HITS} times in a row — giving up"
                    )
                self._start_pause(outcome.raw)
                continue  # the same call again once the window reopens
            failures += 1
            if failures > self.max_retries:
                raise SynthBackendError(f"Command backend ({self.binary}) failed: {outcome.error}")
            await asyncio.sleep(float(settings.SYNTH_COMMAND_BACKOFF_SECONDS) * (2 ** (failures - 1)))

    # ── protocol ──

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,  # noqa: ARG002 — a CLI has no such knob
        temperature: float = 0.7,  # noqa: ARG002
        response_schema: dict | None = None,
        cache_prefix: str | None = None,
        **_ignored: Any,  # cache_system etc.: no prompt cache on a CLI
    ) -> str:
        if cache_prefix:  # the stable instructions a cloud backend would cache
            prompt = f"{cache_prefix}\n\n{prompt}"
        text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        if response_schema:
            text += "\n\nReply with ONLY JSON matching this JSON Schema (no prose, no code fences):\n" + json.dumps(response_schema)
        return await self._ask(text)

    async def complete_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        progress: Callable[[str, int, int], Awaitable[None]] | None = None,
    ) -> list[str | None]:
        """Same contract as ``CloudLlmBackend.complete_batch``: each request
        ``{prompt, system_prompt, max_tokens, temperature, cache_prefix?}``;
        replies in order, None where one could not be obtained.
        Up to ``batch_size`` requests are packed into ONE CLI call under a
        strict JSON-lines contract; items missing / malformed / duplicated in
        the reply are retried once each through ``complete()``; still bad →
        None + a ``{id, reason}`` entry (reasons: missing_from_batch,
        malformed_json, empty_output, call_failed)."""
        total = len(requests)
        out: list[str | None] = [None] * total
        size = self.batch_size
        chunks = [list(range(i, min(i + size, total))) for i in range(0, total, size)]
        rejected: list[dict[str, str]] = []
        state = {"done": 0}

        async def _one(i: int) -> None:
            req = requests[i]
            try:
                reply = await self.complete(
                    str(req["prompt"]), system_prompt=req.get("system_prompt"), cache_prefix=req.get("cache_prefix"),
                    response_schema=req.get("response_schema"),
                )
            except SynthBackendError:
                rejected.append({"id": f"r{i}", "reason": "call_failed"})
                return
            if reply.strip():
                out[i] = reply
            else:
                rejected.append({"id": f"r{i}", "reason": "empty_output"})

        async def _chunk(idxs: list[int]) -> None:
            if len(idxs) == 1:
                await _one(idxs[0])
            else:
                try:
                    reply = await self._ask(_pack_prompt(requests, idxs))
                except SynthBackendError:
                    rejected.extend({"id": f"r{i}", "reason": "call_failed"} for i in idxs)
                else:
                    answers, bad = _parse_packed(reply, idxs)
                    for i, text in answers.items():
                        out[i] = text
                    for i in bad:  # missing / malformed / duplicate: once each, individually
                        await _one(i)
            state["done"] += len(idxs)
            if progress is not None:
                await progress(f"packed calls · {state['done']}/{total} answered", state["done"], total)

        if progress is not None and total:
            await progress(f"packing {total} requests, {size} per call", 0, total)
        tasks = [asyncio.ensure_future(_chunk(c)) for c in chunks]
        try:
            await asyncio.gather(*tasks)
        except BaseException:  # one chunk cancelled / failed hard: stop its siblings too
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        self.rejected_items.extend(rejected)
        self.batch_report.append({
            "batch_id": f"command-{len(self.batch_report) + 1}", "requests": total,
            "succeeded": sum(1 for r in out if r is not None), "failed": sum(1 for r in out if r is None),
            "rejected": rejected,
        })
        return out


def validate_pin(pin: str | None) -> None:
    """Raise ValueError for a ``command:<model>`` pin whose model is not a safe
    argv value (no leading "-", limited charset). Other pins pass."""
    text = str(pin or "")
    if text.split(":", 1)[0].strip().lower() != "command" or ":" not in text:
        return
    model = text.split(":", 1)[1].strip()
    if model and not _MODEL_RE.match(model):
        raise ValueError(f"Invalid command-backend model {model!r}: use letters, digits and . _ : / @ - (max 128, no leading dash).")


def llm_budget_guard_project_id() -> int | None:
    from app.services import llm_budget_guard

    return llm_budget_guard.current_project_id()


def _tail(text: str) -> str:
    text = (text or "").strip()
    return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text


def _kill(proc: "asyncio.subprocess.Process") -> None:
    sandbox_exec.kill_process(proc)


def _request_text(req: dict[str, Any]) -> str:
    prefix = str(req.get("cache_prefix") or "")
    return f"{prefix}\n\n{req['prompt']}" if prefix else str(req["prompt"])


def _pack_prompt(requests: list[dict[str, Any]], idxs: list[int]) -> str:
    def _item(i: int) -> dict[str, Any]:
        item: dict[str, Any] = {"id": f"r{i}", "system": str(requests[i].get("system_prompt") or ""), "prompt": _request_text(requests[i])}
        if requests[i].get("response_schema"):  # the answer string must itself be JSON matching this schema
            item["response_schema"] = requests[i]["response_schema"]
        return item

    items = "\n".join(json.dumps(_item(i), ensure_ascii=False) for i in idxs)
    return (
        f"You will answer {len(idxs)} independent requests. Each request is one JSON line below with an "
        "\"id\", an optional \"system\" instruction and a \"prompt\". Answer every request on its own, as if it were "
        "the only one.\n\n"
        "Reply with EXACTLY one JSON line per request, in the same order, in this form and nothing else "
        "(no prose, no code fences, no extra lines):\n"
        '{"id": "<the request id>", "output": "<your complete answer to that request, as a JSON string>"}\n\n'
        "Requests:\n" + items
    )


def _parse_packed(reply: str, idxs: list[int]) -> tuple[dict[int, str], dict[int, str]]:
    """(answers by request index, bad items → reason). Tolerates a code fence."""
    known = {f"r{i}": i for i in idxs}
    answers: dict[int, str] = {}
    bad: dict[int, str] = {}
    seen: set[str] = set()
    unparsable: list[str] = []
    for line in reply.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            unparsable.append(line)
            continue
        rid = obj.get("id") if isinstance(obj, dict) else None
        if not isinstance(rid, str) or rid not in known:  # list / dict ids are unhashable: skip
            continue
        if rid in seen:  # duplicate: neither copy is trusted
            answers.pop(known[rid], None)
            bad[known[rid]] = "malformed_json"
            continue
        seen.add(rid)
        output = obj.get("output")
        if not isinstance(output, str):
            bad[known[rid]] = "malformed_json"
        elif not output.strip():
            bad[known[rid]] = "empty_output"
        else:
            answers[known[rid]] = output
    for rid, i in known.items():
        if i in answers or i in bad:
            continue
        bad[i] = "malformed_json" if any(f'"{rid}"' in line for line in unparsable) else "missing_from_batch"
    return answers, bad


def _project_overrides() -> dict[str, Any]:
    """The current project's ``synth_command`` overrides (known keys only)."""
    from app.services import llm_budget_guard

    ctx = llm_budget_guard.current_project.get()
    raw = ctx.synth_command if ctx is not None else {}
    return {k: v for k, v in raw.items() if k in _OVERRIDE_KEYS and v is not None} if isinstance(raw, dict) else {}

"""Run an external command in a throw-away sandbox (synchronous).

Used by code checks (compilers, linters) whose input is untrusted model output.
Defence in depth, none of it a security boundary on its own:

* a fresh temp dir per call; ``files`` are written inside it (relative names
  only, no ``..``), the command's cwd and ``HOME`` / ``TMPDIR`` are that dir;
* never a shell; a minimal environment (no API keys, no proxy variables);
* its own session, killed as a process group on timeout;
* ``setrlimit`` before exec: CPU seconds, file size, data segment (not
  ``RLIMIT_AS`` -- the JVM reserves huge virtual address space), process count,
  no core dumps;
* output capped (the rest is drained and dropped);
* no network when ``unshare -rn`` works on the host (``CODE_CHECK_NETWORK_ISOLATION``:
  ``auto`` (default) uses it when available, ``off`` never, ``require`` refuses
  to run without it). ``SandboxResult.network_isolated`` reports what happened.

The minimal-env / kill helpers live here and are shared with the command synth
backend.
"""

from __future__ import annotations

import contextlib
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings

BASE_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
)
# A key in the environment makes such CLIs bill the metered API instead of
# the subscription: never forwarded, even when named in a passthrough list.
DENIED_ENV = frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"})
_PROXY_RE = re.compile(r"(^|_)(proxy)$", re.IGNORECASE)
_SECRETISH_RE = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)

_FSIZE_BYTES = 16 * 1024 * 1024


class SandboxError(RuntimeError):
    """The sandbox refused to run (e.g. network isolation required but unavailable)."""


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool
    network_isolated: bool
    duration_s: float


# ── helpers shared with synth_backends/command.py ──


def minimal_env(passthrough: str = "") -> dict[str, str]:
    names = list(BASE_ENV) + [n.strip() for n in str(passthrough or "").split(",") if n.strip()]
    return {n: os.environ[n] for n in names if n in os.environ and n not in DENIED_ENV}


def kill_process(proc: Any) -> None:
    """SIGKILL the process group (the child runs in its own session), then the process."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


# ── network isolation ──

_ISOLATION_LOCK = threading.Lock()
_ISOLATION_PROBE: dict[str, bool] = {}


def network_isolation_mode() -> str:
    mode = str(getattr(settings, "CODE_CHECK_NETWORK_ISOLATION", "auto") or "auto").strip().lower()
    return mode if mode in {"auto", "off", "require"} else "auto"


def network_isolation_available() -> bool:
    """Whether ``unshare -rn true`` works on this host (probed once, cached)."""
    with _ISOLATION_LOCK:
        if "ok" not in _ISOLATION_PROBE:
            ok = False
            exe = shutil.which("unshare")
            if exe:
                try:
                    ok = subprocess.run(
                        [exe, "-rn", "true"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL
                    ).returncode == 0
                except (OSError, subprocess.SubprocessError):
                    ok = False
            _ISOLATION_PROBE["ok"] = ok
        return _ISOLATION_PROBE["ok"]


# ── execution ──


def _safe_name(name: str) -> Path:
    p = Path(name)
    if not name or p.is_absolute() or ".." in p.parts or "\x00" in name:
        raise SandboxError(f"unsafe file name: {name!r}")
    return p


def _thread_count_estimate() -> int:
    try:
        return sum(1 for _ in Path("/proc").glob("[0-9]*/task/*"))
    except OSError:
        return 0


def _sandbox_env(tmp: str, extra: dict[str, str] | None) -> dict[str, str]:
    env = {k: v for k, v in minimal_env().items() if not _PROXY_RE.search(k) and not _SECRETISH_RE.search(k)}
    env.update({"HOME": tmp, "TMPDIR": tmp, "LC_ALL": env.get("LC_ALL", "C.UTF-8")})
    for k, v in (extra or {}).items():
        if not _PROXY_RE.search(k) and not _SECRETISH_RE.search(k):
            env[k] = str(v).replace("{tmp}", tmp)  # "{tmp}" = this call's temp dir
    return env


def _drain(stream: Any, cap: int, sink: list[bytes], flag: list[bool]) -> None:
    size = 0
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            room = cap - size
            if room > 0:
                sink.append(chunk[:room])
                size += min(len(chunk), room)
            if len(chunk) > room:
                flag[0] = True
    except (OSError, ValueError):
        return


def run_sandboxed(
    argv: list[str],
    *,
    files: dict[str, str],
    timeout_s: float,
    cpu_s: int,
    mem_mb: int,
    max_output_bytes: int = 65536,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
    nproc: int | None = None,
) -> SandboxResult:
    if not argv or not all(isinstance(a, str) for a in argv):
        raise SandboxError("argv must be a non-empty list of strings")
    mode = network_isolation_mode()
    isolate = False
    if mode != "off":
        isolate = network_isolation_available()
        if mode == "require" and not isolate:
            raise SandboxError("network isolation is required (CODE_CHECK_NETWORK_ISOLATION=require) but 'unshare -rn' is unavailable on this host")
    exe = shutil.which(argv[0], path=os.environ.get("PATH")) if not os.path.isabs(argv[0]) else argv[0]
    if not exe or not os.access(exe, os.X_OK):
        raise SandboxError(f"binary not found: {os.path.basename(argv[0])}")
    full = ([shutil.which("unshare") or "unshare", "-rn"] if isolate else []) + [exe] + list(argv[1:])
    limit_nproc = nproc if nproc is not None else _thread_count_estimate() + 512
    mem_bytes = max(64, int(mem_mb)) * 1024 * 1024

    def _preexec() -> None:  # runs in the child between fork and exec
        for res, val in (
            (resource.RLIMIT_CPU, max(1, int(cpu_s))),
            (resource.RLIMIT_FSIZE, _FSIZE_BYTES),
            (resource.RLIMIT_DATA, mem_bytes),
            (resource.RLIMIT_NPROC, limit_nproc),
            (resource.RLIMIT_CORE, 0),
        ):
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(res, (val, val))

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="brewslm-sbx-") as tmp:
        for name, content in (files or {}).items():
            rel = _safe_name(name)
            target = Path(tmp) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        try:
            proc = subprocess.Popen(
                full, cwd=tmp, env=_sandbox_env(tmp, env), stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                start_new_session=True, preexec_fn=_preexec, close_fds=True,
            )
        except OSError as exc:
            raise SandboxError(f"could not run {os.path.basename(argv[0])}: {exc}") from exc
        out_b: list[bytes] = []
        err_b: list[bytes] = []
        flag = [False]
        threads = [
            threading.Thread(target=_drain, args=(proc.stdout, max_output_bytes, out_b, flag), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, max_output_bytes, err_b, flag), daemon=True),
        ]
        if stdin is not None:
            def _feed() -> None:
                with contextlib.suppress(OSError, ValueError):
                    proc.stdin.write(stdin.encode("utf-8"))  # type: ignore[union-attr]
                with contextlib.suppress(OSError, ValueError):
                    proc.stdin.close()  # type: ignore[union-attr]
            threads.append(threading.Thread(target=_feed, daemon=True))
        for t in threads:
            t.start()
        timed_out = False
        try:
            proc.wait(timeout=max(0.1, float(timeout_s)))
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process(proc)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        except BaseException:
            kill_process(proc)
            raise
        # A grandchild may keep the pipes open: kill the group before joining readers.
        kill_process(proc)
        for t in threads:
            t.join(timeout=5)
    return SandboxResult(
        returncode=proc.returncode if proc.returncode is not None else -9,
        stdout=b"".join(out_b).decode("utf-8", "replace"),
        stderr=b"".join(err_b).decode("utf-8", "replace"),
        timed_out=timed_out,
        truncated=flag[0],
        network_isolated=isolate,
        duration_s=round(time.monotonic() - started, 3),
    )

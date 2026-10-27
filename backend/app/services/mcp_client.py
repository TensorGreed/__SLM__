"""Minimal in-house MCP client (JSON-RPC 2.0) for the tool catalog.

Two transports, no third-party MCP dependency:

* **stdio** -- launches a *user-configured* process, so it is the dangerous one. Safety model (all
  install-wide; a project can never choose them): disabled unless ``MCP_STDIO_ALLOWED_BINARIES``
  allows ``argv[0]`` (same match rules as the command synth backend: a bare name matches a bare
  argv[0], an absolute path matches by realpath); never a shell; ``start_new_session`` + the whole
  process group killed on exit; minimal environment (``sandbox_exec.minimal_env`` with
  ``MCP_STDIO_ENV_PASSTHROUGH``) plus the env values the user declared (those come from project
  secrets); a fresh temporary working directory; an overall timeout; 4 MB cap per stdout line.
* **streamable HTTP** -- one POST per message (``Accept: application/json, text/event-stream``);
  the reply is JSON or an SSE stream. ``Mcp-Session-Id`` is carried, ``MCP-Protocol-Version`` is sent
  after ``initialize``, the session is DELETEd at the end (best effort). http(s) only, redirects are
  never followed, responses are capped at 8 MB, and with ``MCP_HTTP_ALLOW_PRIVATE_HOSTS=false``
  loopback / private / link-local addresses are refused. The legacy two-endpoint HTTP+SSE transport
  is NOT supported.

Every error message and stderr tail passes through a redactor that replaces each known secret value
(env values / header values) and ``NAME=value`` pairs whose name looks secret-ish with ``***``.

``call_tool_stdio`` / ``call_tool_http`` are PRIVATE to ``tool_catalog_service.dry_run_call`` (the one
function that enforces "read-only tools only"); nothing else may import them.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import os
import re
import shutil
import socket
import tempfile
from typing import Any, Awaitable, Callable, Iterable
from urllib.parse import urlsplit

import httpx

from app.config import settings
from app.services import sandbox_exec

PROTOCOL_VERSION = "2025-06-18"
CLIENT_NAME = "brewslm"
CLIENT_VERSION = "1.0"
MAX_PAGES = 50
MAX_TOOLS = 1000
MAX_LINE_BYTES = 4 * 1024 * 1024
MAX_HTTP_BYTES = 8 * 1024 * 1024
_STDERR_TAIL = 600
_MSG_MAX = 500

_NAME_VALUE_RE = re.compile(
    r"([A-Za-z0-9_.\-]*(?:" + sandbox_exec._SECRETISH_RE.pattern.strip("()") + r")[A-Za-z0-9_.\-]*)(['\"]?\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]+")


_DENIED_ENV_RE = re.compile(
    r"^(?:LD_.*|DYLD_.*|PYTHON.*|PERL5.*|PERLLIB|RUBY.*|MALLOC_.*|GIT_.*|PATH|NODE_OPTIONS|NODE_PATH|BASH_ENV|ENV|HOME|IFS|SHELL|TMPDIR"
    r"|GCONV_PATH|LOCPATH|GLIBC_TUNABLES|JAVA_TOOL_OPTIONS|_JAVA_OPTIONS|CLASSPATH)$",
    re.IGNORECASE,
)


def env_name_denied(name: Any) -> bool:
    """True for env names that change how a binary loads or runs code (loader / interpreter / shell
    injection vectors) and for every base-environment key: a user env value can never set these."""
    n = str(name)
    return bool(_DENIED_ENV_RE.match(n)) or n.upper() in {b.upper() for b in sandbox_exec.BASE_ENV}


class McpError(RuntimeError):
    """An MCP transport / protocol failure. The message is already redacted."""


class McpStdioDisabled(McpError):
    """The stdio binary is not allow-listed (or the allow-list is empty)."""


def redact(text: Any, secrets: Iterable[str] = ()) -> str:
    """Replace every known secret value and secret-looking ``NAME=value`` pair with ``***``."""
    out = str(text)
    for value in sorted({str(s) for s in secrets if s}, key=len, reverse=True):
        out = out.replace(value, "***")
    out = _NAME_VALUE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}***", out)
    return _BEARER_RE.sub(lambda m: f"{m.group(1)} ***", out)


# ── stdio allow-list (same rules as the command synth backend) ──


def _allowed_entries() -> list[str]:
    return [e.strip() for e in str(settings.MCP_STDIO_ALLOWED_BINARIES or "").split(",") if e.strip()]


def check_stdio_allowed(argv0: str) -> str:
    """Return the resolved executable path, or raise ``McpStdioDisabled``."""
    if not _allowed_entries():
        raise McpStdioDisabled("MCP stdio is disabled: set MCP_STDIO_ALLOWED_BINARIES to the server binary's name or absolute path.")
    if not isinstance(argv0, str) or not argv0:
        raise McpStdioDisabled("MCP stdio command is empty.")
    env_path = os.environ.get("PATH")
    resolved = shutil.which(argv0, path=env_path) if not os.path.isabs(argv0) else (argv0 if os.access(argv0, os.X_OK) else None)
    if resolved is not None:
        for entry in _allowed_entries():
            if os.path.isabs(entry):
                if os.path.realpath(entry) == os.path.realpath(resolved):
                    return resolved
            elif "/" not in argv0 and entry == argv0:
                return resolved
    raise McpStdioDisabled(
        f"binary not allow-listed in MCP_STDIO_ALLOWED_BINARIES: {os.path.basename(argv0) or argv0}"
    )


# ── JSON-RPC session core (transport = request / notify / close) ──


class _Transport:
    secrets: list[str]

    async def request(self, method: str, params: dict[str, Any] | None) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    async def notify(self, method: str, params: dict[str, Any] | None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def set_protocol_version(self, version: str) -> None:
        pass

    async def close(self) -> None:
        pass


def _server_info(init: Any) -> dict[str, Any]:
    init = init if isinstance(init, dict) else {}
    info = init.get("serverInfo") if isinstance(init.get("serverInfo"), dict) else {}
    return {
        "name": str(info.get("name") or "")[:200],
        "version": str(info.get("version") or "")[:100],
        "protocol_version": str(init.get("protocolVersion") or ""),
    }


async def _handshake(tr: _Transport) -> dict[str, Any]:
    init = await tr.request("initialize", {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
    })
    info = _server_info(init)
    tr.set_protocol_version(info["protocol_version"] or PROTOCOL_VERSION)
    await tr.notify("notifications/initialized", None)
    return info


async def _list(tr: _Transport) -> dict[str, Any]:
    server = await _handshake(tr)
    tools: list[dict[str, Any]] = []
    truncated = False
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        res = await tr.request("tools/list", {"cursor": cursor} if cursor else {})
        page = res.get("tools") if isinstance(res, dict) else None
        if not isinstance(page, list):
            raise McpError("tools/list returned no tools array.")
        for t in page:
            if isinstance(t, dict):
                if len(tools) >= MAX_TOOLS:
                    truncated = True
                    break
                tools.append(t)
        cursor = res.get("nextCursor") if isinstance(res, dict) else None
        if truncated or not isinstance(cursor, str) or not cursor:
            break
    else:
        truncated = True
    out: dict[str, Any] = {"server": server, "tools": tools}
    if truncated:
        out["truncated"] = True
    return out


async def _call(tr: _Transport, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    await _handshake(tr)
    res = await tr.request("tools/call", {"name": name, "arguments": arguments or {}})
    res = res if isinstance(res, dict) else {}
    content = res.get("content")
    return {
        "is_error": bool(res.get("isError")),
        "content": content if isinstance(content, list) else [],
        "structured": res.get("structuredContent"),
    }


async def _run(tr: _Transport, op: Callable[[], Awaitable[Any]], timeout_s: float) -> Any:
    try:
        return await asyncio.wait_for(op(), timeout=max(0.1, float(timeout_s)))
    except McpError:
        raise
    except asyncio.TimeoutError:
        raise McpError(f"MCP server did not answer within {timeout_s:g}s.") from None
    except httpx.HTTPError as exc:
        raise McpError(redact(f"HTTP error: {type(exc).__name__}: {exc}", tr.secrets)[:_MSG_MAX]) from None
    finally:
        with contextlib.suppress(Exception):
            await asyncio.shield(tr.close())


def _rpc_result(msg: dict[str, Any], secrets: list[str]) -> Any:
    err = msg.get("error")
    if isinstance(err, dict):
        raise McpError(redact(f"MCP error {err.get('code')}: {err.get('message')}", secrets)[:_MSG_MAX])
    if err is not None:
        raise McpError("MCP server returned a malformed error.")
    return msg.get("result")


# ── stdio ──


class _StdioTransport(_Transport):
    def __init__(self, command: list[str], env: dict[str, str]) -> None:
        self.command = [str(c) for c in command]
        self.user_env = {str(k): str(v) for k, v in (env or {}).items()}
        self.secrets = [v for v in self.user_env.values() if v]
        self.proc: asyncio.subprocess.Process | None = None
        self.tmp: str | None = None
        self._id = 0
        self._stderr = bytearray()
        self._stderr_task: asyncio.Task | None = None

    async def start(self) -> None:
        if not self.command:
            raise McpError("MCP stdio command is empty.")
        resolved = check_stdio_allowed(self.command[0])
        bad = sorted(k for k in self.user_env if env_name_denied(k))
        if bad:
            raise McpError(f"environment variable {bad[0]!r} is not allowed for an MCP server.")
        env = sandbox_exec.minimal_env(settings.MCP_STDIO_ENV_PASSTHROUGH)
        env.update(self.user_env)
        self.tmp = tempfile.mkdtemp(prefix="brewslm-mcp-")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                resolved, *self.command[1:],
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=self.tmp, env=env, start_new_session=True, limit=MAX_LINE_BYTES,
            )
        except OSError as exc:
            raise McpError(redact(f"could not start MCP server: {exc}", self.secrets)[:_MSG_MAX]) from None
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        with contextlib.suppress(Exception):
            while True:
                chunk = await self.proc.stderr.read(4096)
                if not chunk:
                    return
                self._stderr.extend(chunk)
                del self._stderr[:-_STDERR_TAIL * 4]

    def _tail(self) -> str:
        text = bytes(self._stderr).decode("utf-8", "replace").strip()[-_STDERR_TAIL:]
        return redact(text, self.secrets)

    async def _send(self, msg: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write((json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8"))
            await self.proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            raise McpError(self._exit_message("MCP server closed its input")) from None

    def _exit_message(self, prefix: str) -> str:
        tail = self._tail()
        return redact(prefix + (f": {tail}" if tail else "."), self.secrets)[:_MSG_MAX + _STDERR_TAIL]

    async def _read_msg(self) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            try:
                line = await self.proc.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError):
                raise McpError(f"MCP server sent a line longer than {MAX_LINE_BYTES // (1024 * 1024)} MB.") from None
            if not line:
                await asyncio.sleep(0.05)  # let the stderr drainer catch up
                raise McpError(self._exit_message("MCP server exited before answering"))
            try:
                msg = json.loads(line.decode("utf-8", "replace"))
            except RecursionError:
                raise McpError("MCP server sent a message nested too deeply.") from None
            except ValueError:
                continue  # banner / log noise on stdout
            if isinstance(msg, dict):
                return msg

    async def request(self, method: str, params: dict[str, Any] | None) -> Any:
        self._id += 1
        rid = self._id
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)
        while True:
            got = await self._read_msg()
            if "method" in got:
                if "id" in got:  # a server -> client request
                    if got.get("method") == "ping":
                        await self._send({"jsonrpc": "2.0", "id": got["id"], "result": {}})
                    else:
                        await self._send({"jsonrpc": "2.0", "id": got["id"], "error": {"code": -32601, "message": "method not supported"}})
                continue  # notification
            if got.get("id") == rid:
                return _rpc_result(got, self.secrets)

    async def notify(self, method: str, params: dict[str, Any] | None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    async def close(self) -> None:
        proc = self.proc
        if proc is not None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.close()
            sandbox_exec.kill_process(proc)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5)
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(BaseException):
                await self._stderr_task
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)


async def _stdio_op(command: list[str], env: dict[str, str], timeout_s: float, op: Callable[[_Transport], Awaitable[Any]]) -> Any:
    tr = _StdioTransport(command, env)

    async def go() -> Any:
        await tr.start()
        return await op(tr)

    try:
        return await _run(tr, go, timeout_s)
    except McpError as exc:
        raise type(exc)(redact(str(exc), tr.secrets)) from None


async def list_tools_stdio(command: list[str], env: dict[str, str], *, timeout_s: float = 30) -> dict[str, Any]:
    return await _stdio_op(command, env, timeout_s, _list)


async def call_tool_stdio(
    command: list[str], env: dict[str, str], name: str, arguments: dict[str, Any], *, timeout_s: float = 30,
) -> dict[str, Any]:
    """PRIVATE: only ``tool_catalog_service.dry_run_call`` may import this."""
    return await _stdio_op(command, env, timeout_s, lambda tr: _call(tr, name, arguments))


# ── streamable HTTP ──


def _host_is_private(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_unspecified or ip.is_multicast)


async def check_http_url(url: str) -> None:
    """http(s) only; when ``MCP_HTTP_ALLOW_PRIVATE_HOSTS`` is false, refuse private resolved addresses."""
    parts = urlsplit(str(url or ""))
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise McpError("MCP URL must be an http(s) URL.")
    if parts.username or parts.password:
        raise McpError("MCP URL must not embed credentials; use a header secret.")
    if settings.MCP_HTTP_ALLOW_PRIVATE_HOSTS:
        return
    host = parts.hostname
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError:
        raise McpError(f"could not resolve host {host!r}.") from None
    if not infos or any(_host_is_private(i[4][0]) for i in infos):
        raise McpError(f"host {host!r} resolves to a private address and MCP_HTTP_ALLOW_PRIVATE_HOSTS is false.")


class _HttpTransport(_Transport):
    def __init__(self, url: str, headers: dict[str, str], client: httpx.AsyncClient | None, timeout_s: float) -> None:
        self.url = url
        self.headers = {str(k): str(v) for k, v in (headers or {}).items()}
        self.secrets = [v for v in self.headers.values() if v]
        self._own = client is None
        self.client = client or httpx.AsyncClient(follow_redirects=False, timeout=httpx.Timeout(max(1.0, float(timeout_s))))
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self._id = 0

    def set_protocol_version(self, version: str) -> None:
        self.protocol_version = version

    def _headers(self) -> dict[str, str]:
        h = {k: v for k, v in self.headers.items()}
        h["Accept"] = "application/json, text/event-stream"
        h["Content-Type"] = "application/json"
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            h["MCP-Protocol-Version"] = self.protocol_version
        return h

    async def _post(self, msg: dict[str, Any], want_id: int | None) -> Any:
        body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        async with self.client.stream("POST", self.url, content=body, headers=self._headers(), follow_redirects=False) as resp:
            if 300 <= resp.status_code < 400:
                raise McpError(f"MCP server answered HTTP {resp.status_code}; redirects are not followed.")
            if resp.status_code >= 400:
                text = (await self._read_capped(resp))[:300].decode("utf-8", "replace")
                raise McpError(redact(f"MCP server answered HTTP {resp.status_code}: {text}".strip(), self.secrets)[:_MSG_MAX])
            sid = resp.headers.get("mcp-session-id")
            if sid and not self.session_id:
                self.session_id = sid
            if want_id is None:
                return None
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype == "text/event-stream":
                return await self._read_sse(resp, want_id)
            raw = await self._read_capped(resp)
            try:
                payload = json.loads(raw.decode("utf-8", "replace"))
            except RecursionError:
                raise McpError("MCP server response is nested too deeply.") from None
            except ValueError:
                raise McpError("MCP server returned a non-JSON response.") from None
            for item in payload if isinstance(payload, list) else [payload]:
                if isinstance(item, dict) and item.get("id") == want_id and "method" not in item:
                    return _rpc_result(item, self.secrets)
            raise McpError("MCP server response did not contain the answer.")

    async def _read_capped(self, resp: httpx.Response) -> bytes:
        buf = bytearray()
        async for chunk in resp.aiter_bytes():
            buf.extend(chunk)
            if len(buf) > MAX_HTTP_BYTES:
                raise McpError(f"MCP server response exceeded {MAX_HTTP_BYTES // (1024 * 1024)} MB.")
        return bytes(buf)

    async def _read_sse(self, resp: httpx.Response, want_id: int) -> Any:
        total = 0
        pending = b""
        data_lines: list[str] = []

        def handle() -> Any:
            if not data_lines:
                return _MISSING
            text = "\n".join(data_lines)
            data_lines.clear()
            try:
                item = json.loads(text)
            except RecursionError:
                raise McpError("MCP server response is nested too deeply.") from None
            except ValueError:
                return _MISSING
            for it in item if isinstance(item, list) else [item]:
                if isinstance(it, dict) and it.get("id") == want_id and "method" not in it:
                    return _rpc_result(it, self.secrets)
            return _MISSING

        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > MAX_HTTP_BYTES:
                raise McpError(f"MCP server response exceeded {MAX_HTTP_BYTES // (1024 * 1024)} MB.")
            pending += chunk
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").rstrip("\r")
                if line == "":
                    got = handle()
                    if got is not _MISSING:
                        return got
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
        if pending:
            line = pending.decode("utf-8", "replace").rstrip("\r")
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        got = handle()
        if got is not _MISSING:
            return got
        raise McpError("MCP event stream ended without an answer.")

    async def request(self, method: str, params: dict[str, Any] | None) -> Any:
        self._id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        return await self._post(msg, self._id)

    async def notify(self, method: str, params: dict[str, Any] | None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._post(msg, None)

    async def _delete_session(self) -> None:
        # Best effort: send the DELETE and close without reading the (untrusted, possibly endless) body.
        headers = {k: v for k, v in self._headers().items() if k != "Content-Type"}
        async with self.client.stream("DELETE", self.url, headers=headers, follow_redirects=False):
            pass

    async def close(self) -> None:
        if self.session_id:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._delete_session(), timeout=5)
        if self._own:
            with contextlib.suppress(Exception):
                await self.client.aclose()


_MISSING = object()


async def _http_op(
    url: str, headers: dict[str, str], timeout_s: float, client: httpx.AsyncClient | None,
    op: Callable[[_Transport], Awaitable[Any]],
) -> Any:
    secrets = [str(v) for v in (headers or {}).values() if v]
    try:
        await check_http_url(url)
    except McpError as exc:
        raise McpError(redact(str(exc), secrets)) from None
    tr = _HttpTransport(url, headers, client, timeout_s)

    async def go() -> Any:
        return await op(tr)

    try:
        return await _run(tr, go, timeout_s)
    except McpError as exc:
        raise McpError(redact(str(exc), tr.secrets)) from None


async def list_tools_http(
    url: str, headers: dict[str, str], *, timeout_s: float = 30, client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    return await _http_op(url, headers, timeout_s, client, _list)


async def call_tool_http(
    url: str, headers: dict[str, str], name: str, arguments: dict[str, Any], *,
    timeout_s: float = 30, client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """PRIVATE: only ``tool_catalog_service.dry_run_call`` may import this."""
    return await _http_op(url, headers, timeout_s, client, lambda tr: _call(tr, name, arguments))

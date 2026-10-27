"""BrewSLM MCP server: a thin client of the serving gateway (G11).

Run ``python -m app.mcp_server`` (stdio, newline-delimited JSON-RPC 2.0) or
``python -m app.mcp_server --http --port 8765`` (minimal streamable HTTP:
``POST /mcp``). Standard library + httpx only: it must not import
``app.models`` / ``app.database`` / ``app.services`` -- it never touches the
platform database, only the gateway's ``/v1`` HTTP API with a gateway key.

Environment: ``BREWSLM_GATEWAY_URL`` (default ``http://localhost:8000``) and
``BREWSLM_API_KEY`` (required; tools answer with ``isError`` when missing).
Logs go to stderr only. Tools: ``list_models``, ``run_model``, ``retrieve``
plus every named tool of the key's projects (``GET /v1/brewslm/tools``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from typing import Any

import httpx

SERVER_NAME = "brewslm"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL = "2025-06-18"
TOOLS_CACHE_SECONDS = 30.0
HTTP_TIMEOUT_S = 330.0
ANNOTATIONS = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}

# Tests inject an ``httpx.MockTransport`` here.
_TRANSPORT: httpx.BaseTransport | None = None


def _log(message: str) -> None:
    sys.stderr.write(f"[brewslm-mcp] {message}\n")
    sys.stderr.flush()


class ToolFailure(Exception):
    """A tool-level failure reported to the client as ``isError`` (never a stack trace)."""


BUILTIN_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_models",
        "description": "List the BrewSLM models this key can call (project, run and base model for each).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "run_model",
        "description": (
            "Ask a BrewSLM model. 'model' is a model id from list_models (e.g. my-project or my-project:base). "
            "Sources are listed when retrieval is applied."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model id from list_models."},
                "input": {"type": "string", "description": "The user message."},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": 4096},
                "system": {"type": "string", "description": "Optional system prompt."},
                "retrieval": {"type": "boolean", "description": "Force retrieval on or off (default: per model)."},
            },
            "required": ["model", "input"],
        },
    },
    {
        "name": "retrieve",
        "description": "Retrieve source passages (with citations) from a project's documents without running a model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project slug or p<id>."},
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["project", "query"],
        },
    },
]


class Gateway:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("BREWSLM_GATEWAY_URL") or "http://localhost:8000").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("BREWSLM_API_KEY", "")

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            raise ToolFailure("BREWSLM_API_KEY is not set. Create a gateway key in BrewSLM (Ship > Use from Claude Code) and export it.")
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT_S, transport=_TRANSPORT) as client:
                resp = client.request(
                    method, self.base_url + path, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}
                )
        except httpx.HTTPError as exc:
            raise ToolFailure(f"Could not reach the BrewSLM gateway at {self.base_url} ({type(exc).__name__}).") from None
        try:
            data = resp.json()
        except ValueError:
            data = None
        if resp.status_code >= 400:
            raise ToolFailure(f"Gateway error {resp.status_code}: {_error_message(data) or resp.reason_phrase}")
        return data


def _error_message(data: Any) -> str:
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if data.get("detail"):
            return str(data["detail"])[:300]
    return ""


def _sources_block(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""
    lines = []
    for c in citations:
        where = c.get("source_url") or c.get("document") or "document"
        lines.append(f"[{c.get('n')}] {where}")
    return "\n\nSources:\n" + "\n".join(lines)


def _json_or_none(text: str) -> Any:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, (dict, list)) else None


class McpServer:
    def __init__(self, gateway: Gateway | None = None) -> None:
        self.gateway = gateway or Gateway()
        self._tools_cache: tuple[float, list[dict[str, Any]]] | None = None

    # ---- tools

    def _named_tools(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        if self._tools_cache and now - self._tools_cache[0] < TOOLS_CACHE_SECONDS:
            return self._tools_cache[1]
        try:
            data = self.gateway.request("GET", "/v1/brewslm/tools")
        except ToolFailure as exc:
            _log(f"named tools unavailable: {exc}")
            return []
        tools = [
            {
                "name": t["name"],
                "description": t.get("description") or f"Named tool on {t.get('model')}",
                "inputSchema": t.get("input_schema") or {"type": "object"},
            }
            for t in (data or {}).get("tools", [])
            if isinstance(t, dict) and t.get("name")
        ]
        self._tools_cache = (now, tools)
        return tools

    def list_tools(self) -> list[dict[str, Any]]:
        builtin = {t["name"] for t in BUILTIN_TOOLS}
        named = [t for t in self._named_tools() if t["name"] not in builtin]
        return [{**t, "annotations": dict(ANNOTATIONS)} for t in [*BUILTIN_TOOLS, *named]]

    def _result(self, text: str, structured: dict[str, Any], *, error: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": error}
        if structured:
            out["structuredContent"] = structured
        return out

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "list_models":
                data = self.gateway.request("GET", "/v1/models") or {}
                models = data.get("data", [])
                lines = []
                for m in models:
                    b = m.get("brewslm") or {}
                    lines.append(f"{m.get('id')}  (project {b.get('project_name')}, run {b.get('run_id')}, base {b.get('base_model')})")
                return self._result("\n".join(lines) or "No models.", {"models": models})
            if name == "run_model":
                return self._run_model(arguments)
            if name == "retrieve":
                data = self.gateway.request(
                    "POST", "/v1/brewslm/retrieve",
                    {k: arguments[k] for k in ("project", "query", "k") if arguments.get(k) is not None},
                ) or {}
                passages = data.get("passages", [])
                text = "\n\n".join(
                    f"[{p.get('n')}] {p.get('source_url') or p.get('document')}\n{p.get('text')}" for p in passages
                ) or f"No passages ({(data.get('retrieval') or {}).get('reason', 'no match')})."
                return self._result(text, data)
            if name in {t["name"] for t in self._named_tools()}:
                data = self.gateway.request("POST", f"/v1/brewslm/tools/{name}/call", {"arguments": arguments}) or {}
                return self._result(
                    str(data.get("content") or "") + _sources_block(data.get("citations") or []),
                    {k: data.get(k) for k in ("model", "citations", "parsed_json", "brewslm") if data.get(k) is not None},
                )
        except ToolFailure as exc:
            return self._result(str(exc), {}, error=True)
        return self._result(f"Unknown tool '{name}'.", {}, error=True)

    def _run_model(self, args: dict[str, Any]) -> dict[str, Any]:
        model, text = args.get("model"), args.get("input")
        if not isinstance(model, str) or not model or not isinstance(text, str) or not text:
            raise ToolFailure("'model' and 'input' are required strings.")
        messages = []
        if isinstance(args.get("system"), str) and args["system"].strip():
            messages.append({"role": "system", "content": args["system"]})
        messages.append({"role": "user", "content": text})
        body: dict[str, Any] = {"model": model, "messages": messages}
        if args.get("max_tokens") is not None:
            body["max_tokens"] = args["max_tokens"]
        if args.get("retrieval") is not None:
            body["brewslm"] = {"retrieval": bool(args["retrieval"])}
        data = self.gateway.request("POST", "/v1/chat/completions", body) or {}
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ToolFailure("The gateway returned an unexpected response.") from None
        citations = data.get("citations") or []
        structured: dict[str, Any] = {"model": data.get("model"), "content": content, "citations": citations}
        parsed = _json_or_none(content)
        if parsed is not None:
            structured["parsed_json"] = parsed
        if data.get("brewslm"):
            structured["brewslm"] = data["brewslm"]
        return self._result(content + _sources_block(citations), structured)

    # ---- JSON-RPC

    def handle(self, message: Any) -> dict[str, Any] | None:
        """One JSON-RPC message -> its response, or ``None`` for a notification."""
        if not isinstance(message, dict):
            return _error(None, -32600, "Invalid request")
        rid, method = message.get("id"), message.get("method")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if rid is None:  # notification (notifications/initialized, ...)
            return None
        if method == "initialize":
            wanted = params.get("protocolVersion")
            return _ok(rid, {
                "protocolVersion": wanted if wanted in SUPPORTED_PROTOCOLS else DEFAULT_PROTOCOL,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })
        if method == "ping":
            return _ok(rid, {})
        if method == "tools/list":
            return _ok(rid, {"tools": self.list_tools()})
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                return _error(rid, -32602, "tools/call needs a tool name")
            args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            if name not in {t["name"] for t in self.list_tools()}:
                return _error(rid, -32602, f"Unknown tool: {name}")
            return _ok(rid, self.call_tool(name, args))
        return _error(rid, -32601, f"Method not found: {method}")


def _ok(rid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _error(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def serve_stdio(server: McpServer, stdin=None, stdout=None) -> None:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            stdout.write(json.dumps(_error(None, -32700, "Parse error")) + "\n")
            stdout.flush()
            continue
        try:
            response = server.handle(message)
        except Exception as exc:  # noqa: BLE001 - never a stack trace to the client
            _log(f"internal error: {type(exc).__name__}")
            response = _error(message.get("id") if isinstance(message, dict) else None, -32603, "Internal error")
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")


def check_http_headers(headers: Any, port: int, *, allow_remote: bool = False) -> tuple[int, str] | None:
    """Guard against DNS rebinding / cross-site calls: a present ``Origin`` must be
    a loopback http(s) origin, ``Host`` must be this server's loopback address
    (skipped with ``--allow-remote``), and the body must be JSON. ``(status,
    message)`` to reject, ``None`` to accept."""
    origin = headers.get("Origin")
    if origin is not None and not _ORIGIN_RE.match(origin.strip()):
        return 403, "Origin not allowed"
    if not allow_remote:
        host = (headers.get("Host") or "").strip().lower()
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}:
            return 403, "Host not allowed"
    ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        return 415, "Content-Type must be application/json"
    return None


def make_http_server(server: McpServer, host: str, port: int, *, allow_remote: bool = False):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    sessions: set[str] = set()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # stderr only
            _log(fmt % args)

        def _send(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            if body:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/mcp":
                return self._send(404)
            rejected = check_http_headers(self.headers, self.server.server_address[1], allow_remote=allow_remote)
            if rejected:
                return self._send(rejected[0], json.dumps({"error": rejected[1]}).encode())
            try:
                length = int(self.headers.get("Content-Length") or 0)
                message = json.loads(self.rfile.read(length) or b"null")
            except ValueError:
                return self._send(400, json.dumps(_error(None, -32700, "Parse error")).encode())
            try:
                response = server.handle(message)
            except Exception as exc:  # noqa: BLE001 - never a stack trace to the client
                _log(f"internal error: {type(exc).__name__}")
                response = _error(message.get("id") if isinstance(message, dict) else None, -32603, "Internal error")
            if response is None:
                return self._send(202)
            headers: dict[str, str] = {}
            if isinstance(message, dict) and message.get("method") == "initialize":
                sid = uuid.uuid4().hex
                with lock:
                    sessions.add(sid)
                headers["Mcp-Session-Id"] = sid
            self._send(200, json.dumps(response).encode(), headers)

        def do_GET(self) -> None:  # noqa: N802
            self._send(405, headers={"Allow": "POST"})

    return ThreadingHTTPServer((host, port), Handler)


def serve_http(server: McpServer, host: str, port: int, *, allow_remote: bool = False) -> None:
    httpd = make_http_server(server, host, port, allow_remote=allow_remote)
    _log(f"streamable HTTP on http://{host}:{httpd.server_address[1]}/mcp")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.mcp_server", description="BrewSLM MCP server")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-remote", action="store_true",
        help="permit --host to be a non-loopback address (the server has no auth of its own)",
    )
    args = parser.parse_args(argv)
    if args.http and args.host not in LOOPBACK_HOSTS:
        if not args.allow_remote:
            _log(f"refusing to bind non-loopback host '{args.host}' without --allow-remote.")
            return 2
        _log(
            f"WARNING: binding {args.host}; this MCP server has no authentication of its own and uses "
            "the gateway key from its environment, so anyone who can reach it can use that key."
        )
    server = McpServer()
    if not server.gateway.api_key:
        _log("BREWSLM_API_KEY is not set; tools will report an error until it is.")
    if args.http:
        serve_http(server, args.host, args.port, allow_remote=args.allow_remote)
    else:
        serve_stdio(server)
    return 0


if __name__ == "__main__":
    sys.exit(main())

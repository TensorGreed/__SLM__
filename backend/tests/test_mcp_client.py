"""MCP client: stdio via the fake server (allow-listed by absolute path), streamable HTTP via
``httpx.MockTransport`` (JSON + SSE, session header), errors, redaction, the stdio allow-list gate."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from app.config import settings
from app.services import mcp_client

HERE = Path(__file__).resolve().parent
FAKE = str(HERE / "fake_mcp_server.py")
PY = sys.executable


def run(coro):
    return asyncio.run(coro)


def allow_python():
    return mock.patch.object(settings, "MCP_STDIO_ALLOWED_BINARIES", PY)


class StdioTests(unittest.TestCase):
    def test_lists_two_pages(self):
        with allow_python():
            out = run(mcp_client.list_tools_stdio([PY, FAKE], {}))
        self.assertEqual([t["name"] for t in out["tools"]], ["list_services", "get_service", "create_service", "delete_service"])
        self.assertEqual(out["server"], {"name": "deploy-service-fake", "version": "0.1", "protocol_version": "2025-06-18"})

    def test_noise_and_ping_are_tolerated(self):
        with allow_python():
            out = run(mcp_client.list_tools_stdio([PY, FAKE], {"FAKE_MCP_NOISE": "1", "FAKE_MCP_PING": "1"}))
        self.assertEqual(len(out["tools"]), 4)

    def test_disallowed_binary(self):
        with mock.patch.object(settings, "MCP_STDIO_ALLOWED_BINARIES", ""):
            with self.assertRaises(mcp_client.McpStdioDisabled) as cm:
                run(mcp_client.list_tools_stdio([PY, FAKE], {}))
        self.assertIn("MCP_STDIO_ALLOWED_BINARIES", str(cm.exception))
        with mock.patch.object(settings, "MCP_STDIO_ALLOWED_BINARIES", "/usr/bin/true"):
            with self.assertRaises(mcp_client.McpStdioDisabled):
                run(mcp_client.list_tools_stdio([PY, FAKE], {}))
        # a bare allow-list name never matches an absolute argv[0]
        with mock.patch.object(settings, "MCP_STDIO_ALLOWED_BINARIES", os.path.basename(PY)):
            with self.assertRaises(mcp_client.McpStdioDisabled):
                run(mcp_client.list_tools_stdio([PY, FAKE], {}))

    def test_stderr_secret_is_redacted(self):
        with allow_python():
            with self.assertRaises(mcp_client.McpError) as cm:
                run(mcp_client.list_tools_stdio([PY, FAKE], {"FAKE_MCP_LEAK": "hunter2-value", "SOME_TOKEN": "abc123xyz"}))
        msg = str(cm.exception)
        self.assertNotIn("hunter2-value", msg)
        self.assertIn("***", msg)

    def test_timeout_kills_process(self):
        with allow_python():
            with self.assertRaises(mcp_client.McpError) as cm:
                run(mcp_client.list_tools_stdio([PY, FAKE], {"FAKE_MCP_HANG": "1"}, timeout_s=1))
        self.assertIn("did not answer", str(cm.exception))

    def test_env_is_minimal_plus_declared(self):
        script = (
            "import json,os,sys\n"
            "env={k:os.environ.get(k) for k in ('BREWSLM_LEAK_PROBE','DECLARED_ONE')}\n"
            "sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':1,'result':{'protocolVersion':'x','serverInfo':{'name':json.dumps(env)}}})+'\\n');sys.stdout.flush()\n"
            "sys.stdin.readline();sys.stdin.readline()\n"
            "sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':2,'result':{'tools':[]}})+'\\n');sys.stdout.flush()\n"
        )
        with tempfile.TemporaryDirectory() as d, allow_python(), mock.patch.dict(os.environ, {"BREWSLM_LEAK_PROBE": "leaked"}):
            p = Path(d) / "s.py"
            p.write_text(script)
            out = run(mcp_client.list_tools_stdio([PY, str(p)], {"DECLARED_ONE": "yes"}))
        env = json.loads(out["server"]["name"])
        self.assertIsNone(env["BREWSLM_LEAK_PROBE"])
        self.assertEqual(env["DECLARED_ONE"], "yes")

    def test_call_tool(self):
        with allow_python():
            res = run(mcp_client.call_tool_stdio([PY, FAKE], {}, "get_service", {"name": "api"}))
        self.assertFalse(res["is_error"])
        self.assertEqual(res["structured"]["tool"], "get_service")


def _rpc(req: httpx.Request):
    return json.loads(req.content)


class HttpTests(unittest.TestCase):
    def client(self, handler):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def make_handler(self, *, sse: bool, seen: list):
        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req)
            if req.method == "DELETE":
                return httpx.Response(204)
            msg = _rpc(req)
            method = msg.get("method")
            if "id" not in msg:
                return httpx.Response(202)
            if method == "initialize":
                result = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "h", "version": "2"}}
                hdr = {"Mcp-Session-Id": "sess-1"}
            else:
                cursor = (msg.get("params") or {}).get("cursor")
                result = {"tools": [{"name": "b" if cursor else "a", "inputSchema": {"type": "object"}}]}
                if not cursor:
                    result["nextCursor"] = "c2"
                hdr = {}
            body = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
            if sse:
                text = "event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "method": "notifications/x"}) + "\n\n" \
                    + "event: message\ndata: " + json.dumps(body) + "\n\n"
                return httpx.Response(200, text=text, headers={"content-type": "text/event-stream", **hdr})
            return httpx.Response(200, json=body, headers=hdr)
        return handler

    def check(self, sse: bool):
        seen: list = []
        out = run(mcp_client.list_tools_http("http://mcp.test/mcp", {"Authorization": "Bearer s3cret"}, client=self.client(self.make_handler(sse=sse, seen=seen))))
        self.assertEqual([t["name"] for t in out["tools"]], ["a", "b"])
        self.assertEqual(out["server"]["name"], "h")
        posts = [r for r in seen if r.method == "POST"]
        self.assertIn("text/event-stream", posts[0].headers["accept"])
        self.assertNotIn("mcp-session-id", posts[0].headers)
        self.assertEqual(posts[1].headers["mcp-session-id"], "sess-1")
        self.assertEqual(posts[1].headers["mcp-protocol-version"], "2025-06-18")
        self.assertEqual(posts[1].headers["authorization"], "Bearer s3cret")
        self.assertEqual(seen[-1].method, "DELETE")
        self.assertEqual(seen[-1].headers["mcp-session-id"], "sess-1")

    def test_json(self):
        self.check(False)

    def test_sse(self):
        self.check(True)

    def test_http_error_redacts_header_value(self):
        def handler(req):
            return httpx.Response(401, text="bad credentials Bearer s3cret-value for user")
        with self.assertRaises(mcp_client.McpError) as cm:
            run(mcp_client.list_tools_http("http://mcp.test/mcp", {"X-Key": "s3cret-value"}, client=self.client(handler)))
        self.assertNotIn("s3cret-value", str(cm.exception))
        self.assertIn("401", str(cm.exception))

    def test_redirect_not_followed(self):
        def handler(req):
            return httpx.Response(302, headers={"location": "http://evil.test/"})
        with self.assertRaises(mcp_client.McpError) as cm:
            run(mcp_client.list_tools_http("http://mcp.test/mcp", {}, client=self.client(handler)))
        self.assertIn("redirect", str(cm.exception))

    def test_rpc_error(self):
        def handler(req):
            msg = _rpc(req)
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32000, "message": "boom api_key=abc"}})
        with self.assertRaises(mcp_client.McpError) as cm:
            run(mcp_client.list_tools_http("http://mcp.test/mcp", {}, client=self.client(handler)))
        self.assertIn("boom", str(cm.exception))
        self.assertNotIn("abc", str(cm.exception))

    def test_scheme_and_private_hosts(self):
        with self.assertRaises(mcp_client.McpError):
            run(mcp_client.list_tools_http("file:///etc/passwd", {}))
        with mock.patch.object(settings, "MCP_HTTP_ALLOW_PRIVATE_HOSTS", False):
            for url in ("http://127.0.0.1:9/mcp", "http://10.0.0.5/mcp", "http://169.254.169.254/", "http://[::1]/mcp"):
                with self.assertRaises(mcp_client.McpError, msg=url) as cm:
                    run(mcp_client.list_tools_http(url, {}))
                self.assertIn("private", str(cm.exception))

    def test_response_cap(self):
        big = "x" * (mcp_client.MAX_HTTP_BYTES + 10)
        def handler(req):
            return httpx.Response(200, text=big, headers={"content-type": "application/json"})
        with self.assertRaises(mcp_client.McpError) as cm:
            run(mcp_client.list_tools_http("http://mcp.test/mcp", {}, client=self.client(handler)))
        self.assertIn("exceeded", str(cm.exception))

    def test_session_delete_does_not_read_the_body(self):
        pulled = {"chunks": 0}

        class Endless(httpx.AsyncByteStream):
            async def __aiter__(self):
                while True:
                    pulled["chunks"] += 1
                    yield b"x" * 1024
                    await asyncio.sleep(0)

        seen: list = []
        base = self.make_handler(sse=False, seen=seen)

        def handler(req):
            if req.method == "DELETE":
                seen.append(req)
                return httpx.Response(200, stream=Endless())
            return base(req)

        import time

        t = time.monotonic()
        out = run(mcp_client.list_tools_http("http://mcp.test/mcp", {}, client=self.client(handler)))
        self.assertEqual(len(out["tools"]), 2)
        self.assertLess(time.monotonic() - t, 4)
        self.assertEqual(seen[-1].method, "DELETE")
        self.assertEqual(pulled["chunks"], 0)

    def test_deeply_nested_json_is_an_mcp_error(self):
        deep = "[" * 200000 + "]" * 200000

        def handler(req):
            return httpx.Response(200, text=deep, headers={"content-type": "application/json"})
        with self.assertRaises(mcp_client.McpError) as cm:
            run(mcp_client.list_tools_http("http://mcp.test/mcp", {}, client=self.client(handler)))
        self.assertIn("nested too deeply", str(cm.exception))

    def test_stdio_deeply_nested_line_is_an_mcp_error(self):
        async def go():
            tr = mcp_client._StdioTransport([PY], {})
            reader = asyncio.StreamReader()
            reader.feed_data(("[" * 200000 + "]" * 200000 + "\n").encode())
            tr.proc = mock.Mock(stdout=reader)
            await tr._read_msg()
        with self.assertRaises(mcp_client.McpError):
            run(go())

    def test_denied_env_names(self):
        for name in ("LD_PRELOAD", "DYLD_X", "PATH", "PYTHONPATH", "NODE_OPTIONS", "GIT_DIR", "HOME", "LANG", "USER"):
            self.assertTrue(mcp_client.env_name_denied(name), name)
        for name in ("API_TOKEN", "MY_PYTHON", "FAKE_MCP_CALL_LOG"):
            self.assertFalse(mcp_client.env_name_denied(name), name)


class RedactTests(unittest.TestCase):
    def test_redact(self):
        out = mcp_client.redact("failed with API_KEY=abc123 and password: 'p w' plus value zzz", ["zzz"])
        self.assertNotIn("abc123", out)
        self.assertNotIn("zzz", out)
        self.assertNotIn("p w", out)


if __name__ == "__main__":
    unittest.main()

"""BrewSLM MCP server (G11): protocol, tools, error mapping, import hygiene.
The gateway is stubbed with ``httpx.MockTransport``; no network."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import httpx
import pytest

from app import mcp_server
from app.mcp_server import Gateway, McpServer

BACKEND = Path(__file__).resolve().parents[1]
CITATIONS = [{"n": 1, "passage_id": "1:0", "document": "guide.md", "source_url": "https://docs.example.org/g",
              "source_document_id": 1, "snippet": "s", "score": 1.0}]


class FakeGateway:
    """Records requests; answers like the real gateway."""

    def __init__(self):
        self.requests = []
        self.overrides = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, path, body, request.headers.get("authorization")))
        if path in self.overrides:
            return self.overrides[path]
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "demo", "object": "model", "brewslm": {"project_name": "Demo", "run_id": 4, "base_model": "org/b"}}]})
        if path == "/v1/chat/completions":
            return httpx.Response(200, json={
                "model": body["model"], "choices": [{"message": {"role": "assistant", "content": '{"findings": []}'}}],
                "citations": CITATIONS, "brewslm": {"project_id": 1}})
        if path == "/v1/brewslm/retrieve":
            return httpx.Response(200, json={"project": "demo", "retrieval": {"applied": True}, "passages": [
                {"n": 1, "document": "guide.md", "source_url": "https://docs.example.org/g", "text": "hello passage"}]})
        if path == "/v1/brewslm/tools":
            return httpx.Response(200, json={"tools": [
                {"name": "detect_issues", "description": "Find issues", "input_schema": {
                    "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
                 "project": "demo", "model": "demo"}]})
        if path == "/v1/brewslm/tools/detect_issues/call":
            return httpx.Response(200, json={"content": '{"findings": []}', "model": "demo", "citations": CITATIONS,
                                             "brewslm": {}, "parsed_json": {"findings": []}})
        return httpx.Response(404, json={"error": {"message": "nope", "type": "x", "code": "y"}})


@pytest.fixture()
def fake(monkeypatch):
    fg = FakeGateway()
    monkeypatch.setattr(mcp_server, "_TRANSPORT", httpx.MockTransport(fg))
    return fg


@pytest.fixture()
def server(fake):
    return McpServer(Gateway("http://gw.test", "bsk_testkey"))


def rpc(server, method, params=None, rid=1):
    msg = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        msg["params"] = params
    return server.handle(msg)


def test_initialize_negotiates_protocol(server):
    for asked, got in (("2024-11-05", "2024-11-05"), ("2025-03-26", "2025-03-26"), ("1999-01-01", "2025-06-18"), (None, "2025-06-18")):
        res = rpc(server, "initialize", {"protocolVersion": asked} if asked else {})["result"]
        assert res["protocolVersion"] == got
        assert res["capabilities"] == {"tools": {"listChanged": False}}
        assert res["serverInfo"]["name"] == "brewslm"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert rpc(server, "ping")["result"] == {}
    assert rpc(server, "resources/list")["error"]["code"] == -32601


def test_tools_list_includes_named_tools_and_read_only_annotations(server, fake):
    tools = rpc(server, "tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["list_models", "run_model", "retrieve", "detect_issues"]
    for t in tools:
        assert t["annotations"] == {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
        assert t["inputSchema"]["type"] == "object"
    assert tools[3]["inputSchema"]["required"] == ["code"]
    rpc(server, "tools/list")
    assert sum(1 for r in fake.requests if r[1] == "/v1/brewslm/tools") == 1  # cached
    assert all(r[3] == "Bearer bsk_testkey" for r in fake.requests)


def test_run_model_text_sources_and_structured(server, fake):
    res = rpc(server, "tools/call", {"name": "run_model", "arguments": {
        "model": "demo", "input": "check", "max_tokens": 50, "system": "sys", "retrieval": True}})["result"]
    assert res["isError"] is False
    text = res["content"][0]["text"]
    assert text.startswith('{"findings": []}') and "Sources:\n[1] https://docs.example.org/g" in text
    assert res["structuredContent"]["citations"] == CITATIONS and res["structuredContent"]["parsed_json"] == {"findings": []}
    sent = fake.requests[-1][2]
    assert sent == {"model": "demo", "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "check"}],
                    "max_tokens": 50, "brewslm": {"retrieval": True}}


def test_retrieve_list_models_and_named_tool(server, fake):
    r = rpc(server, "tools/call", {"name": "retrieve", "arguments": {"project": "demo", "query": "q", "k": 2}})["result"]
    assert "[1] https://docs.example.org/g" in r["content"][0]["text"] and "hello passage" in r["content"][0]["text"]
    assert fake.requests[-1][2] == {"project": "demo", "query": "q", "k": 2}
    m = rpc(server, "tools/call", {"name": "list_models", "arguments": {}})["result"]
    assert "demo" in m["content"][0]["text"] and m["structuredContent"]["models"][0]["id"] == "demo"
    t = rpc(server, "tools/call", {"name": "detect_issues", "arguments": {"code": "x = 1"}})["result"]
    assert t["isError"] is False and t["structuredContent"]["parsed_json"] == {"findings": []}
    assert fake.requests[-1][2] == {"arguments": {"code": "x = 1"}}


def test_gateway_error_becomes_is_error_without_key_or_trace(server, fake):
    fake.overrides["/v1/chat/completions"] = httpx.Response(
        403, json={"error": {"message": "This API key has no access to the project.", "type": "x", "code": "project_not_in_scope"}})
    res = rpc(server, "tools/call", {"name": "run_model", "arguments": {"model": "other", "input": "q"}})["result"]
    assert res["isError"] is True and "no access to the project" in res["content"][0]["text"]
    assert "bsk_testkey" not in json.dumps(res) and "Traceback" not in json.dumps(res)
    assert rpc(server, "tools/call", {"name": "run_model", "arguments": {"model": "x"}})["result"]["isError"] is True
    unknown = rpc(server, "tools/call", {"name": "no_such_tool", "arguments": {}})
    assert unknown["error"]["code"] == -32602 and "no_such_tool" in unknown["error"]["message"]


def test_unreachable_gateway_is_clean_error(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(mcp_server, "_TRANSPORT", httpx.MockTransport(boom))
    res = rpc(McpServer(Gateway("http://gw.test", "bsk_testkey")), "tools/call", {"name": "list_models", "arguments": {}})["result"]
    assert res["isError"] is True and "Could not reach" in res["content"][0]["text"] and "bsk_testkey" not in json.dumps(res)


def test_missing_key_is_is_error(fake):
    srv = McpServer(Gateway("http://gw.test", ""))
    res = rpc(srv, "tools/call", {"name": "list_models", "arguments": {}})["result"]
    assert res["isError"] is True and "BREWSLM_API_KEY" in res["content"][0]["text"]
    assert fake.requests == []
    names = [t["name"] for t in rpc(srv, "tools/list")["result"]["tools"]]
    assert names == ["list_models", "run_model", "retrieve"]  # still lists the built-ins


def test_http_transport_roundtrip(server):
    from http.server import ThreadingHTTPServer

    # run the real handler on an ephemeral port
    started = {}
    orig = ThreadingHTTPServer.serve_forever

    def capture(self, *a, **k):
        started["httpd"] = self
        return orig(self, *a, **k)

    ThreadingHTTPServer.serve_forever = capture
    try:
        th = threading.Thread(target=mcp_server.serve_http, args=(server, "127.0.0.1", 0), daemon=True)
        th.start()
        for _ in range(100):
            if "httpd" in started:
                break
            threading.Event().wait(0.02)
        port = started["httpd"].server_address[1]

        def post(obj):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", data=json.dumps(obj).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            return urllib.request.urlopen(req, timeout=5)

        init = post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init.status == 200 and init.headers["Mcp-Session-Id"]
        assert json.loads(init.read())["result"]["serverInfo"]["name"] == "brewslm"
        assert post({"jsonrpc": "2.0", "method": "notifications/initialized"}).status == 202
        listed = json.loads(post({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).read())
        assert len(listed["result"]["tools"]) == 4
    finally:
        ThreadingHTTPServer.serve_forever = orig
        if "httpd" in started:
            started["httpd"].shutdown()


def test_imports_nothing_from_platform_internals():
    code = (
        "import sys, app.mcp_server\n"
        "bad = [m for m in sys.modules if m == 'app.models' or m == 'app.database' or m == 'app.services'"
        " or m.startswith(('app.models.', 'app.database.', 'app.services.'))]\n"
        "print(bad)\n"
        "assert not bad\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r)\n%s" % (str(BACKEND), code)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr + out.stdout


def test_stdio_subprocess_roundtrip():
    env = {**os.environ, "BREWSLM_API_KEY": "", "BREWSLM_GATEWAY_URL": "http://127.0.0.1:9"}
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_models", "arguments": {}}},
    ]
    proc = subprocess.run([sys.executable, "-m", "app.mcp_server"], input="\n".join(json.dumps(m) for m in msgs) + "\nnot json\n",
                          capture_output=True, text=True, cwd=BACKEND, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert [r.get("id") for r in replies] == [1, 2, 3, None]  # no reply to the notification
    assert replies[0]["result"]["protocolVersion"] == "2025-03-26"
    assert [t["name"] for t in replies[1]["result"]["tools"]] == ["list_models", "run_model", "retrieve"]
    assert replies[2]["result"]["isError"] is True and "BREWSLM_API_KEY" in replies[2]["result"]["content"][0]["text"]
    assert replies[3]["error"]["code"] == -32700
    assert "brewslm-mcp" in proc.stderr  # logs on stderr only


# ------------------------------------------------------------------ HTTP hardening


@pytest.fixture()
def http_server(server):
    httpd = mcp_server.make_http_server(server, "127.0.0.1", 0)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def raw_post(port, body=b'{"jsonrpc":"2.0","id":1,"method":"ping"}', **headers):
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    hdrs = {"Content-Type": "application/json", **headers}
    conn.putrequest("POST", "/mcp", skip_host="Host" in hdrs, skip_accept_encoding=True)
    for k, v in hdrs.items():
        if v is not None:
            conn.putheader(k, v)
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders(body)
    resp = conn.getresponse()
    out = resp.status, resp.read()
    conn.close()
    return out


def test_http_rejects_foreign_origin_host_and_non_json(http_server):
    port = http_server.server_address[1]
    assert raw_post(port)[0] == 200
    assert raw_post(port, Origin="http://localhost:3000")[0] == 200
    assert raw_post(port, Origin="https://127.0.0.1")[0] == 200
    for origin in ("https://evil.example", "http://localhost.evil.example", "null", "http://127.0.0.1.evil.test:80"):
        assert raw_post(port, Origin=origin)[0] == 403, origin
    assert raw_post(port, Host="evil.example")[0] == 403
    assert raw_post(port, Host=f"evil.example:{port}")[0] == 403
    assert raw_post(port, Host=f"localhost:{port}")[0] == 200
    assert raw_post(port, **{"Content-Type": "text/plain"})[0] == 415
    assert raw_post(port, **{"Content-Type": None})[0] == 415
    assert raw_post(port, **{"Content-Type": "application/json; charset=utf-8"})[0] == 200


def test_http_internal_error_is_a_jsonrpc_32603(fake):
    class Exploding(McpServer):
        def handle(self, message):
            raise RuntimeError("secret internals")

    httpd = mcp_server.make_http_server(Exploding(Gateway("http://gw.test", "k")), "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status, body = raw_post(httpd.server_address[1], body=b'{"jsonrpc":"2.0","id":7,"method":"ping"}')
    finally:
        httpd.shutdown()
        httpd.server_close()
    reply = json.loads(body)
    assert status == 200 and reply["id"] == 7 and reply["error"]["code"] == -32603
    assert "secret internals" not in body.decode()


def test_non_loopback_host_needs_allow_remote(monkeypatch, capsys):
    served = []
    monkeypatch.setattr(mcp_server, "serve_http", lambda srv, host, port, allow_remote=False: served.append((host, allow_remote)))
    assert mcp_server.main(["--http", "--host", "0.0.0.0"]) == 2
    assert served == [] and "--allow-remote" in capsys.readouterr().err
    assert mcp_server.main(["--http", "--host", "0.0.0.0", "--allow-remote"]) == 0
    assert served == [("0.0.0.0", True)] and "WARNING" in capsys.readouterr().err
    assert mcp_server.main(["--http"]) == 0 and served[-1] == ("127.0.0.1", False)

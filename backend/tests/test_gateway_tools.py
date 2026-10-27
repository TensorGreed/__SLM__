"""Named gateway tools: settings validation, safe templates, argument
validation, ``@recipe`` for code-findings, name clashes (G11)."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from uuid import uuid4
from pathlib import Path
from unittest import mock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _gateway_helpers as h  # noqa: E402
from app.services import gateway_tools_service as gts  # noqa: E402
from app.services.auto_rag_service import AutoRagUnavailable  # noqa: E402
from app.services.code_findings_prompt import build_code_findings_prompt  # noqa: E402

SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"], "additionalProperties": False}


@pytest.fixture(scope="module")
def client():
    cm, c, prev = h.open_client()
    yield c
    h.close_client(cm, prev)


@pytest.fixture(autouse=True)
def _reset():
    h.reset()


@pytest.fixture()
def root():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class Gen:
    def __init__(self, reply="done"):
        self.reply, self.calls = reply, []

    def __call__(self, model_ref, messages, **kw):
        self.calls.append((model_ref, messages, kw))
        return {"reply": self.reply, "usage": None, "finish_reason": "stop"}


def put_tools(client, pid, tools):
    resp = client.put(f"/api/projects/{pid}/gateway/settings", json={"tools": tools})
    assert resp.status_code == 200, resp.text


def test_render_template_is_safe():
    args = {"x": "VAL", "n": 3, "obj": {"a": 1}}
    assert gts.render_template("{x}-{n}-{obj}", args) == 'VAL-3-{"a": 1}'
    assert gts.render_template("{x.__class__}", args) == "{x.__class__}"
    assert gts.render_template("{x[0]} {missing} {0} {{x}}", args) == "{x[0]} {missing} {0} {VAL}"
    assert gts.render_template("{x!r:>10}", args) == "{x!r:>10}"


def test_argument_validation():
    assert gts.argument_errors(SCHEMA, {"q": "ok"}) == []
    assert gts.argument_errors(SCHEMA, {}) and gts.argument_errors(SCHEMA, {"q": 1})
    assert gts.argument_errors(SCHEMA, {"q": "a", "extra": 1})
    assert gts.argument_errors(SCHEMA, "text") == ["arguments must be an object"]


def test_tool_call_end_to_end_and_validation(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    put_tools(client, pid, [{
        "name": "ask_model", "description": "Ask", "input_schema": SCHEMA,
        "prompt_template": "Question: {q}\nAnswer in {q.__class__}", "system_prompt": "Be terse.",
        "max_tokens": 64, "model": ":base"}])
    key = h.make_key(client, [pid])
    listing = client.get("/v1/brewslm/tools", headers=h.auth(key)).json()["tools"]
    slug = h.slug_of(pid)
    assert listing == [{"name": "ask_model", "description": "Ask", "input_schema": SCHEMA,
                        "project": slug, "model": f"{slug}:base"}]
    gen = Gen('{"ok": true}')
    with mock.patch("app.services.local_chat_service.generate_reply", gen):
        resp = client.post("/v1/brewslm/tools/ask_model/call", headers=h.auth(key), json={"arguments": {"q": "why"}})
        bad = client.post("/v1/brewslm/tools/ask_model/call", headers=h.auth(key), json={"arguments": {"q": 5}})
        missing = client.post("/v1/brewslm/tools/nope/call", headers=h.auth(key), json={"arguments": {}})
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["content"] == '{"ok": true}' and out["parsed_json"] == {"ok": True}
    assert out["model"] == f"{slug}:base" and out["citations"] == []
    model_ref, messages, kw = gen.calls[0]
    assert model_ref == h.BASE and kw["max_tokens"] == 64
    assert messages == [{"role": "system", "content": "Be terse."},
                        {"role": "user", "content": "Question: why\nAnswer in {q.__class__}"}]
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_arguments"
    assert len(gen.calls) == 1  # invalid arguments never reach the model
    assert missing.status_code == 404


def test_recipe_template_uses_the_training_prompt(client, root):
    pid = h.make_project(client, recipe="code-findings")
    h.seed_run(pid, root)
    labels = ["sql-injection", "hardcoded-secret"]
    tax = {"entries": [{"id": label} for label in labels]}
    h.set_runtime(pid, label_taxonomy=tax)
    schema = {"type": "object", "properties": {"code": {"type": "string"}, "path": {"type": "string"},
                                               "language": {"type": "string"}, "start_line": {"type": "integer"}},
              "required": ["code"]}
    put_tools(client, pid, [{"name": "detect_issues", "description": "Find issues", "input_schema": schema,
                             "prompt_template": "@recipe"}])
    key = h.make_key(client, [pid])
    args = {"code": "def f(x):\n    return x\n", "path": "src/app.py", "language": "python", "start_line": 10}
    gen = Gen('{"findings": []}')
    manifest = {"output_schema": {"prompt_labels": labels}}
    with mock.patch("app.services.local_chat_service.generate_reply", gen), \
            mock.patch("app.services.eval_task_handler_service.read_prepared_manifest", return_value=manifest):
        resp = client.post("/v1/brewslm/tools/detect_issues/call", headers=h.auth(key), json={"arguments": args})
    assert resp.status_code == 200, resp.text
    expected = build_code_findings_prompt(args["code"], "python", "src/app.py", 10, labels)
    assert gen.calls[0][1] == [{"role": "user", "content": expected}]
    assert resp.json()["parsed_json"] == {"findings": []}

    # without a prepared manifest the project taxonomy supplies the labels
    gen2 = Gen('{"findings": []}')
    with mock.patch("app.services.local_chat_service.generate_reply", gen2):
        client.post("/v1/brewslm/tools/detect_issues/call", headers=h.auth(key), json={"arguments": args})
    assert gen2.calls[0][1][0]["content"] == expected
    miss = client.post("/v1/brewslm/tools/detect_issues/call", headers=h.auth(key), json={"arguments": {"path": "x"}})
    assert miss.status_code == 400


def test_recipe_requires_a_recipe_builder(client, root):
    pid = h.make_project(client)  # no recipe
    resp = client.put(f"/api/projects/{pid}/gateway/settings", json={"tools": [
        {"name": "t", "input_schema": {"type": "object"}, "prompt_template": "@recipe"}]})
    assert resp.status_code == 400
    assert "no recipe prompt builder; write a prompt_template" in resp.text


def test_name_clash_gets_slug_suffix(client, root):
    p1, p2, p3 = (h.make_project(client) for _ in range(3))
    for p in (p1, p2, p3):
        h.seed_run(p, root)
    shared = {"input_schema": {"type": "object"}, "prompt_template": "go"}
    put_tools(client, p1, [{"name": "dup", **shared}, {"name": "solo", **shared}])
    put_tools(client, p2, [{"name": "dup", **shared}])
    key = h.make_key(client, [p1, p2, p3])
    names = {t["name"]: t for t in client.get("/v1/brewslm/tools", headers=h.auth(key)).json()["tools"]}
    s1, s2 = h.slug_of(p1), h.slug_of(p2)
    assert set(names) == {f"dup__{s1}", f"dup__{s2}", "solo"}
    assert names[f"dup__{s2}"]["project"] == s2
    gen = Gen("x")
    with mock.patch("app.services.local_chat_service.generate_reply", gen):
        resp = client.post(f"/v1/brewslm/tools/dup__{s2}/call", headers=h.auth(key), json={"arguments": {}})
    assert resp.status_code == 200 and resp.json()["model"] == s2
    # a key limited to one project sees the plain name
    only = h.make_key(client, [p2])
    assert [t["name"] for t in client.get("/v1/brewslm/tools", headers=h.auth(only)).json()["tools"]] == ["dup"]


def test_retrieve_endpoint(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    doc = h.seed_document(pid, "https://docs.example.org/a")
    key = h.make_key(client, [pid])
    hit = [{"row_id": 0, "score": 1.5, "payload": {"id": f"{doc}:0", "text": "x" * 2000, "source_doc": "a.md",
                                                   "source_document_id": doc, "chunk_id": 0}}]
    other = h.make_project(client)
    with mock.patch("app.services.auto_rag_service.retrieve_document_passages",
                    lambda project_id, q, cfg: hit):
        ok = client.post("/v1/brewslm/retrieve", headers=h.auth(key), json={"project": f"p{pid}", "query": "q", "k": 1})
        out_of_scope = client.post("/v1/brewslm/retrieve", headers=h.auth(key), json={"project": h.slug_of(other), "query": "q"})
    body = ok.json()
    assert ok.status_code == 200 and body["project"] == h.slug_of(pid) and body["retrieval"]["k"] == 1
    p = body["passages"][0]
    assert p["n"] == 1 and p["source_url"] == "https://docs.example.org/a" and len(p["text"]) == 1500
    assert out_of_scope.status_code == 403
    def _no_index(project_id, q, cfg):
        raise AutoRagUnavailable("no index")

    with mock.patch("app.services.auto_rag_service.retrieve_document_passages", _no_index):
        empty = client.post("/v1/brewslm/retrieve", headers=h.auth(key), json={"project": f"p{pid}", "query": "q"}).json()
    assert empty["passages"] == [] and empty["retrieval"]["applied"] is False


def test_tool_names_with_double_underscore_resolve_by_exact_exposed_name(client, root):
    sfx = uuid4().hex[:6]
    p1 = h.make_project(client, name=f"pa{sfx}")
    p2 = h.make_project(client, name=f"pb{sfx}")
    h.seed_run(p1, root)
    h.seed_run(p2, root)
    s1, s2 = h.slug_of(p1), h.slug_of(p2)
    shared = {"input_schema": {"type": "object"}, "prompt_template": "go"}
    put_tools(client, p1, [{"name": "dup", **shared}, {"name": f"dup__{s2}", **shared}])
    put_tools(client, p2, [{"name": "dup", **shared}])
    key = h.make_key(client, [p1, p2])
    tools = client.get("/v1/brewslm/tools", headers=h.auth(key)).json()["tools"]
    # p2's "dup" would be exposed as "dup__<s2>" too: the collision is skipped, not served ambiguously
    assert sorted(t["name"] for t in tools) == sorted([f"dup__{s1}", f"dup__{s2}"])
    assert all("_raw_name" not in t and "_project_id" not in t for t in tools)
    gen = Gen("x")
    with mock.patch("app.services.local_chat_service.generate_reply", gen):
        a = client.post(f"/v1/brewslm/tools/dup__{s2}/call", headers=h.auth(key), json={"arguments": {}})
        b = client.post(f"/v1/brewslm/tools/dup__{s1}/call", headers=h.auth(key), json={"arguments": {}})
    assert a.json()["model"] == s1  # p1's tool literally named dup__<s2>
    assert b.json()["model"] == s1


def test_reserved_tool_names_rejected(client):
    pid = h.make_project(client)
    for name in ("list_models", "run_model", "retrieve"):
        resp = client.put(f"/api/projects/{pid}/gateway/settings", json={"tools": [
            {"name": name, "input_schema": {"type": "object"}, "prompt_template": "x"}]})
        assert resp.status_code == 400 and "reserved" in resp.text


def test_tool_model_suffix_of_another_projects_run_is_404(client, root):
    pid, other = h.make_project(client), h.make_project(client)
    h.seed_run(pid, root)
    foreign = h.seed_run(other, root)
    put_tools(client, pid, [{"name": "t", "input_schema": {"type": "object"}, "prompt_template": "go",
                             "model": f":run-{foreign}"}])
    key = h.make_key(client, [pid, other])
    with mock.patch("app.services.local_chat_service.generate_reply", Gen()):
        resp = client.post("/v1/brewslm/tools/t/call", headers=h.auth(key), json={"arguments": {}})
    assert resp.status_code == 404 and resp.json()["error"]["code"] == "model_not_found"


def test_recipe_labels_follow_training_precedence(client, root):
    from app.database import async_session_factory
    from app.models.project import Project

    pid = h.make_project(client, recipe="code-findings")
    h.seed_run(pid, root)
    h.set_runtime(pid, label_taxonomy={"entries": [{"id": "from-taxonomy"}]})
    put_tools(client, pid, [{"name": "detect_issues", "input_schema": {"type": "object"}, "prompt_template": "@recipe"}])
    key = h.make_key(client, [pid])
    args = {"code": "x = 1\n", "language": "python"}
    manifest = {"output_schema": {"prompt_labels": ["from-manifest"]}}

    def run():
        gen = Gen('{"findings": []}')
        with mock.patch("app.services.local_chat_service.generate_reply", gen), \
                mock.patch("app.services.eval_task_handler_service.read_prepared_manifest", return_value=manifest):
            assert client.post("/v1/brewslm/tools/detect_issues/call", headers=h.auth(key),
                               json={"arguments": args}).status_code == 200
        return gen.calls[0][1][0]["content"]

    assert run() == build_code_findings_prompt(args["code"], "python", None, None, ["from-manifest"])

    async def _preset():
        async with async_session_factory() as db:
            p = await db.get(Project, pid)
            p.dataset_adapter_preset = {"adapter_config": {"finding_labels": ["explicit-one", "explicit-two"]}}
            await db.commit()

    asyncio.run(_preset())
    assert run() == build_code_findings_prompt(args["code"], "python", None, None, ["explicit-one", "explicit-two"])


class _AppTransport(httpx.BaseTransport):
    """Routes the MCP server's HTTP calls into the in-process gateway (no sockets)."""

    def __init__(self, client):
        self.client = client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        resp = self.client.request(request.method, request.url.path, content=request.content,
                                   headers={k: v for k, v in request.headers.items() if k.lower() != "host"})
        return httpx.Response(resp.status_code, headers={"content-type": "application/json"}, content=resp.content)


def test_mcp_server_round_trip_against_the_real_gateway(client, root, monkeypatch):
    from app import mcp_server

    pid = h.make_project(client)
    h.seed_run(pid, root)
    doc = h.seed_document(pid, "https://docs.example.org/a")
    key = h.make_key(client, [pid])
    slug = h.slug_of(pid)
    monkeypatch.setattr(mcp_server, "_TRANSPORT", _AppTransport(client))
    srv = mcp_server.McpServer(mcp_server.Gateway("http://testserver", key))
    hit = [{"row_id": 0, "score": 1.5, "payload": {"id": f"{doc}:0", "text": "passage body", "source_doc": "a.md",
                                                   "source_document_id": doc, "chunk_id": 0}}]

    def call(name, arguments):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        return srv.handle(msg)["result"]

    with mock.patch("app.services.local_chat_service.generate_reply", Gen("answer [1]")), \
            mock.patch("app.services.auto_rag_service.retrieve_document_passages", lambda p, q, c: hit):
        listed = call("list_models", {})
        ran = call("run_model", {"model": slug, "input": "what?", "retrieval": True})
        got = call("retrieve", {"project": slug, "query": "what?", "k": 1})
    assert listed["isError"] is False and slug in [m["id"] for m in listed["structuredContent"]["models"]]
    assert ran["isError"] is False and ran["content"][0]["text"].startswith("answer [1]")
    assert "Sources:" in ran["content"][0]["text"] and "https://docs.example.org/a" in ran["content"][0]["text"]
    assert ran["structuredContent"]["citations"][0]["source_url"] == "https://docs.example.org/a"
    assert got["isError"] is False and "passage body" in got["content"][0]["text"]
    assert got["structuredContent"]["passages"][0]["n"] == 1


def test_recipe_registry_unregistered_is_400_and_registered_builder_is_used(client, root):
    pid = h.make_project(client, recipe="qa-sft")
    h.seed_run(pid, root)
    tool = {"name": "ask", "input_schema": {"type": "object"}, "prompt_template": "@recipe"}
    resp = client.put(f"/api/projects/{pid}/gateway/settings", json={"tools": [tool]})
    assert resp.status_code == 400 and "no recipe prompt builder" in resp.text
    seen = {}

    def dummy(project, arguments, adapter_config):
        seen["project"] = project.id
        return "DUMMY " + str(arguments.get("q"))

    gts.register_recipe_prompt_builder("qa-sft", dummy)
    try:
        put_tools(client, pid, [{**tool, "input_schema": SCHEMA}])
        key = h.make_key(client, [pid])
        gen = Gen("ok")
        with mock.patch("app.services.local_chat_service.generate_reply", gen):
            r = client.post("/v1/brewslm/tools/ask/call", headers=h.auth(key), json={"arguments": {"q": "hi"}})
        assert r.status_code == 200, r.text
        assert gen.calls[0][1] == [{"role": "user", "content": "DUMMY hi"}] and seen["project"] == pid
    finally:
        gts.RECIPE_PROMPT_BUILDERS.pop("qa-sft", None)

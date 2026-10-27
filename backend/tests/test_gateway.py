"""Serving gateway ``/v1`` (G11): models, resolution, retrieval, guided JSON,
streaming, auth, rate limit and telemetry. Stub runtime + stub upstream only."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import httpx
import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))
import _gateway_helpers as h  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.services import gateway_service  # noqa: E402
from app.services.auto_rag_service import AutoRagUnavailable, document_preamble  # noqa: E402

SCHEMA = {"type": "object", "properties": {"findings": {"type": "array"}}, "required": ["findings"]}


@pytest.fixture(scope="module")
def client():
    cm, c, prev = h.open_client()
    yield c
    h.close_client(cm, prev)


@pytest.fixture(autouse=True)
def _reset():
    h.reset()
    yield
    h.reset()


@pytest.fixture()
def root():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class Calls:
    def __init__(self, reply="ok"):
        self.reply, self.calls = reply, []

    def __call__(self, model_ref, messages, **kw):
        self.calls.append((model_ref, messages, kw))
        return {"reply": self.reply, "usage": {"prompt_tokens": 7, "completion_tokens": 3}, "finish_reason": "stop"}


def chat(client, key, model, content="hello", **extra):
    return client.post(
        "/v1/chat/completions", headers=h.auth(key),
        json={"model": model, "messages": [{"role": "user", "content": content}], **extra},
    )


def hits(n=2):
    return [
        {"row_id": i, "score": 2.5 - i,
         "payload": {"id": f"7:{i}", "text": f"passage text {i}", "source_doc": "guide.md",
                     "source_document_id": DOC_ID[0], "chunk_id": i}}
        for i in range(n)
    ]


DOC_ID = [0]


def fake_retrieve(result):
    """Stand-in for ``auto_rag_service.retrieve_document_passages`` (called once per request)."""
    calls: list = []

    def fn(project_id, question, cfg):
        calls.append((project_id, question))
        if isinstance(result, Exception):
            raise result
        return result

    fn.calls = calls
    return fn


# ------------------------------------------------------------------ models / resolution


def test_models_scoped_with_provenance(client, root):
    p1, p2 = h.make_project(client), h.make_project(client)
    r1 = h.seed_run(p1, root)
    h.seed_run(p2, root)
    h.set_runtime(p1, guided_json={"schema": SCHEMA})
    key = h.make_key(client, [p1])
    resp = client.get("/v1/models", headers=h.auth(key))
    assert resp.status_code == 200
    data = resp.json()["data"]
    s1 = h.slug_of(p1)
    ids = [m["id"] for m in data]
    assert ids == [s1, f"{s1}:base", f"{s1}:run-{r1}"]
    assert all(h.slug_of(p2) not in i for i in ids)
    top = data[0]
    assert top["object"] == "model" and top["owned_by"] == "brewslm"
    assert top["brewslm"]["project_id"] == p1 and top["brewslm"]["run_id"] == r1
    assert top["brewslm"]["base_model"] == h.BASE and top["brewslm"]["guided_json"] is True
    assert data[1]["brewslm"]["run_id"] is None


def test_model_resolution_forms(client, root):
    pid = h.make_project(client)
    old = h.seed_run(pid, root)
    new = h.seed_run(pid, root)
    other = h.make_project(client)
    foreign = h.seed_run(other, root)
    key = h.make_key(client, [pid])
    slug = h.slug_of(pid)
    gen = Calls()
    with mock.patch("app.services.local_chat_service.generate_reply", gen):
        assert chat(client, key, slug).status_code == 200
        assert chat(client, key, f"p{pid}").status_code == 200
        assert chat(client, key, f"{slug}:run-{old}").status_code == 200
        r = chat(client, key, f"{slug}:base")
        assert r.status_code == 200
    refs = [c[0] for c in gen.calls]
    assert refs[0] == refs[1]  # slug and p<id> both serve the latest run
    assert refs[0] != refs[2] and refs[2].endswith("model")  # :run-<old> is a different checkpoint
    assert Path(refs[0]).parent.exists() and new > old
    assert gen.calls[3][0] == h.BASE and gen.calls[3][2]["base_model_hint"] == h.BASE
    assert chat(client, key, "nope-nothing").status_code == 404
    assert chat(client, key, f"{slug}:run-999999").status_code == 404
    assert chat(client, key, f"{slug}:run-{foreign}").status_code == 404
    # a real project outside the key's scope
    r = chat(client, key, h.slug_of(other))
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "project_not_in_scope"


def test_default_run_id_setting_wins(client, root):
    pid = h.make_project(client)
    first = h.seed_run(pid, root)
    h.seed_run(pid, root)
    h.set_runtime(pid, gateway={"default_run_id": first})
    key = h.make_key(client, [pid])
    gen = Calls()
    with mock.patch("app.services.local_chat_service.generate_reply", gen):
        resp = chat(client, key, h.slug_of(pid))
    assert resp.status_code == 200
    assert resp.json()["brewslm"]["run_id"] == first


def test_non_servable_run_is_400(client, root):
    pid = h.make_project(client)
    bad = h.seed_run(pid, root, weights=False)
    key = h.make_key(client, [pid])
    r = chat(client, key, f"{h.slug_of(pid)}:run-{bad}")
    assert r.status_code == 400 and "no saved model weights" in r.json()["error"]["message"]


# ------------------------------------------------------------------ chat basics


def test_chat_shape_defaults_and_usage(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    key = h.make_key(client, [pid])
    gen = Calls("Routed to Billing.")
    with mock.patch("app.services.local_chat_service.generate_reply", gen):
        resp = chat(client, key, h.slug_of(pid))
    body = resp.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["model"] == h.slug_of(pid)
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Routed to Billing."}
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert body["citations"] == [] and body["brewslm"]["upstream"] == "in_process"
    assert gen.calls[0][2]["temperature"] == 0.0 and gen.calls[0][2]["max_tokens"] == 512
    with mock.patch("app.services.local_chat_service.generate_reply", gen):
        chat(client, key, h.slug_of(pid), max_tokens=99999, temperature=0.7)
    assert gen.calls[1][2]["max_tokens"] == 4096 and gen.calls[1][2]["temperature"] == 0.7


def test_bad_request_is_openai_envelope(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    key = h.make_key(client, [pid])
    resp = client.post("/v1/chat/completions", headers=h.auth(key), json={"model": h.slug_of(pid), "messages": []})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert set(err) == {"message", "type", "code"} and "detail" not in resp.json()


# ------------------------------------------------------------------ retrieval


def _retrieval_project(client, root, passage_trained=True):
    pid = h.make_project(client)
    run = h.seed_run(pid, root, config={"passage_prompts": True} if passage_trained else {})
    DOC_ID[0] = h.seed_document(pid, "https://docs.example.org/guide")
    return pid, run, h.make_key(client, [pid])


def test_retrieval_injection_and_citations(client, root):
    pid, run, key = _retrieval_project(client, root)
    h_hits = hits(2)
    gen = Calls("see [1]")
    with mock.patch("app.services.local_chat_service.generate_reply", gen), \
            mock.patch("app.services.auto_rag_service.retrieve_document_passages", fake_retrieve(h_hits)):
        resp = client.post("/v1/chat/completions", headers=h.auth(key), json={
            "model": h.slug_of(pid),
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "How?"}],
            "brewslm": {"retrieval": True, "k": 2},
        })
    assert resp.status_code == 200, resp.text
    msgs = gen.calls[0][1]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == document_preamble(h_hits) + "\n\nBe brief."
    body = resp.json()
    assert [c["n"] for c in body["citations"]] == [1, 2]
    assert body["citations"][0]["source_url"] == "https://docs.example.org/guide"
    assert body["citations"][0]["document"] == "guide.md" and body["citations"][1]["passage_id"] == "7:1"
    assert len(body["citations"][0]["snippet"]) <= 300
    r = body["brewslm"]["retrieval"]
    assert r["applied"] is True and r["count"] == 2 and r["k"] == 2
    assert {"reranker", "pool", "chunk_size", "source"} <= set(r)


def test_retrieval_defaults_follow_passage_training(client, root):
    from app.services import passage_prompts_service as pps

    pid = h.make_project(client)
    h.seed_run(pid, root, config={pps.CONFIG_FLAG: True})
    key = h.make_key(client, [pid])
    gen = Calls()
    with mock.patch("app.services.local_chat_service.generate_reply", gen), \
            mock.patch("app.services.auto_rag_service.retrieve_document_passages", fake_retrieve(hits(1))):
        on = chat(client, key, h.slug_of(pid)).json()
        off = chat(client, key, h.slug_of(pid), brewslm={"retrieval": False}).json()
        base = chat(client, key, f"{h.slug_of(pid)}:base").json()
    assert on["brewslm"]["retrieval"]["applied"] is True and on["citations"]
    assert off["brewslm"]["retrieval"]["applied"] is False and off["citations"] == []
    assert base["brewslm"]["retrieval"]["applied"] is False
    assert gen.calls[1][1][0]["role"] == "user"  # untouched messages when off


def test_retrieval_without_index_is_skipped_not_500(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    key = h.make_key(client, [pid])
    gen = Calls()
    err = AutoRagUnavailable("Project has no document passages to retrieve from.")
    with mock.patch("app.services.local_chat_service.generate_reply", gen), \
            mock.patch("app.services.auto_rag_service.retrieve_document_passages", fake_retrieve(err)):
        resp = chat(client, key, h.slug_of(pid), brewslm={"retrieval": True})
    assert resp.status_code == 200
    info = resp.json()["brewslm"]["retrieval"]
    assert info["applied"] is False and "no_index" in info["reason"]
    assert gen.calls[0][1] == [{"role": "user", "content": "hello"}]


# ------------------------------------------------------------------ guided JSON


def test_in_process_guided_json_is_validated_only(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    h.set_runtime(pid, guided_json={"schema": SCHEMA})
    key = h.make_key(client, [pid])
    with mock.patch("app.services.local_chat_service.generate_reply", Calls('{"findings": []}')):
        ok = chat(client, key, h.slug_of(pid)).json()["brewslm"]["guided_json"]
    with mock.patch("app.services.local_chat_service.generate_reply", Calls('{"oops": 1}')):
        bad = chat(client, key, h.slug_of(pid)).json()["brewslm"]["guided_json"]
    with mock.patch("app.services.local_chat_service.generate_reply", Calls("not json")):
        junk = chat(client, key, h.slug_of(pid)).json()["brewslm"]["guided_json"]
        off = chat(client, key, h.slug_of(pid), brewslm={"guided_json": False}).json()["brewslm"]["guided_json"]
    assert ok == {"mode": "validated_only", "valid": True, "errors": []}
    assert bad["valid"] is False and 0 < len(bad["errors"]) <= 5
    assert junk["valid"] is False and "not valid JSON" in junk["errors"][0]
    assert off is None


def _upstream_project(client, root, kind):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    h.set_runtime(pid, guided_json={"schema": SCHEMA}, gateway={
        "upstream": "openai_compatible", "base_url": "http://upstream.test:8001", "upstream_kind": kind})
    return pid, h.make_key(client, [pid])


def _mock_upstream(captured, content='{"findings": []}', status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((str(request.url), json.loads(request.content)))
        if status != 200:
            return httpx.Response(status, text="boom")
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
    gateway_service._HTTP_TRANSPORT = httpx.MockTransport(handler)


@pytest.mark.parametrize("kind", ["vllm", "ollama", "openai"])
def test_guided_json_response_format_for_every_upstream(client, root, kind):
    pid, key = _upstream_project(client, root, kind)
    cap: list = []
    _mock_upstream(cap)
    resp = chat(client, key, h.slug_of(pid))
    assert resp.status_code == 200, resp.text
    url, sent = cap[0]
    assert url == "http://upstream.test:8001/v1/chat/completions"
    assert sent["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "output", "schema": SCHEMA, "strict": True}}
    assert "guided_json" not in sent and "format" not in sent
    assert resp.json()["brewslm"]["guided_json"] == {
        "mode": "enforced", "via": "response_format", "valid": True, "errors": []}
    assert resp.json()["brewslm"]["upstream"] == "openai_compatible"
    assert sent["model"].startswith(h.slug_of(pid) + ":run-")  # canonical id, no map needed


def test_enforced_reply_is_validated_not_assumed(client, root):
    pid, key = _upstream_project(client, root, "vllm")
    _mock_upstream([], content='{"oops": 1}')
    info = chat(client, key, h.slug_of(pid)).json()["brewslm"]["guided_json"]
    assert info["mode"] == "enforced" and info["valid"] is False and info["errors"]
    _mock_upstream([], content="plain words")
    info = chat(client, key, h.slug_of(pid)).json()["brewslm"]["guided_json"]
    assert info["valid"] is False and "not valid JSON" in info["errors"][0]


def test_client_override_reports_client_mode(client, root):
    pid, key = _upstream_project(client, root, "vllm")
    _mock_upstream([])
    info = chat(client, key, h.slug_of(pid), response_format={"type": "json_object"}).json()["brewslm"]["guided_json"]
    assert info["mode"] == "client"


def test_client_response_format_wins_and_opt_out(client, root):
    pid, key = _upstream_project(client, root, "vllm")
    cap: list = []
    _mock_upstream(cap)
    rf = {"type": "json_object"}
    chat(client, key, h.slug_of(pid), response_format=rf)
    assert cap[0][1]["response_format"] == rf and "guided_json" not in cap[0][1]
    chat(client, key, h.slug_of(pid), brewslm={"guided_json": False})
    assert "guided_json" not in cap[1][1] and "response_format" not in cap[1][1]


def test_upstream_model_map_and_failure(client, root):
    pid = h.make_project(client)
    run = h.seed_run(pid, root)
    slug = h.slug_of(pid)
    h.set_runtime(pid, gateway={"upstream": "openai_compatible", "base_url": "http://u.test/v1",
                                "upstream_kind": "openai", "upstream_model_map": {f"{slug}:base": "my-base"}})
    key = h.make_key(client, [pid])
    cap: list = []
    _mock_upstream(cap)
    chat(client, key, f"{slug}:base")
    assert cap[0][0] == "http://u.test/v1/chat/completions" and cap[0][1]["model"] == "my-base"
    _mock_upstream([], status=500)
    resp = chat(client, key, f"{slug}:run-{run}")
    assert resp.status_code == 502 and resp.json()["error"]["code"] == "upstream_error"
    assert resp.json()["error"]["message"] == "upstream returned HTTP 500"  # upstream body not echoed
    assert "boom" not in resp.text


# ------------------------------------------------------------------ streaming


def test_stream_sse_shape(client, root):
    pid, run, key = _retrieval_project(client, root)
    with mock.patch("app.services.local_chat_service.generate_reply", Calls("Hi there")), \
            mock.patch("app.services.auto_rag_service.retrieve_document_passages", fake_retrieve(hits(1))):
        resp = client.post("/v1/chat/completions", headers=h.auth(key), json={
            "model": h.slug_of(pid), "stream": True, "brewslm": {"retrieval": True},
            "messages": [{"role": "user", "content": "q"}]})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    lines = [ln for ln in resp.text.split("\n\n") if ln]
    assert lines[-1] == "data: [DONE]"
    first, last = (json.loads(ln[6:]) for ln in lines[:2])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "Hi there"}
    assert first["choices"][0]["finish_reason"] is None
    assert last["choices"][0]["finish_reason"] == "stop" and last["choices"][0]["delta"] == {}
    assert last["citations"][0]["n"] == 1 and last["brewslm"]["retrieval"]["applied"] is True
    assert first["id"] == last["id"]


# ------------------------------------------------------------------ auth / rate limit


def test_auth_failures_and_envelope(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    key = h.make_key(client, [pid])
    for headers in ({}, {"Authorization": "Bearer bsk_wrong"}, {"X-API-Key": "nope"}):
        r = client.get("/v1/models", headers=headers)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert client.get(f"/v1/models?token={key}").status_code == 401  # never ?token=
    assert client.get("/v1/models", headers={"X-API-Key": key}).status_code == 200
    # revoke
    kid = next(k["id"] for k in client.get("/api/gateway/keys").json()["keys"] if key.startswith(k["key_prefix"]))
    assert client.delete(f"/api/gateway/keys/{kid}").json()["record"]["is_active"] is False
    assert client.get("/v1/models", headers=h.auth(key)).status_code == 401


def test_rate_limit_429_retry_after(client, root):
    pid = h.make_project(client)
    key = h.make_key(client, [pid], rate=2)
    assert client.get("/v1/models", headers=h.auth(key)).status_code == 200
    assert client.get("/v1/models", headers=h.auth(key)).status_code == 200
    r = client.get("/v1/models", headers=h.auth(key))
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limit_exceeded"
    assert 1 <= int(r.headers["Retry-After"]) <= 61
    # another key is unaffected
    assert client.get("/v1/models", headers=h.auth(h.make_key(client, [pid]))).status_code == 200


def test_rate_limit_window_slides():
    key = gateway_service.KeyInfo(id=987654, prefix="bsk_x", project_ids=(1,), rate_limit_per_minute=1)
    gateway_service.check_rate_limit(key, now=1000.0)
    with pytest.raises(gateway_service.GatewayError) as ei:
        gateway_service.check_rate_limit(key, now=1010.0)
    assert ei.value.status_code == 429
    gateway_service.check_rate_limit(key, now=1061.0)


# ------------------------------------------------------------------ telemetry


def test_telemetry_rows_without_text(client, root):
    pid = h.make_project(client)
    run = h.seed_run(pid, root)
    key = h.make_key(client, [pid])
    marker = "SECRET-PROMPT-MARKER-4821"
    with mock.patch("app.services.local_chat_service.generate_reply", Calls("SECRET-REPLY-MARKER-9917")):
        assert chat(client, key, h.slug_of(pid), content=marker).status_code == 200
    assert chat(client, key, f"{h.slug_of(pid)}:run-424242").status_code == 404

    async def _dump():
        async with async_session_factory() as db:
            rows = (await db.execute(text("select * from gateway_request_logs where project_id = :p"), {"p": pid})).all()
            cols = (await db.execute(text("select * from gateway_request_logs limit 0"))).keys()
            return list(cols), [tuple(r) for r in rows]

    cols, rows = asyncio.run(_dump())
    assert len(rows) == 1  # the unresolvable model is not logged (no project)
    blob = json.dumps(rows, default=str)
    assert "MARKER" not in blob and not {"prompt", "reply", "content", "messages"} & set(cols)
    row = dict(zip(cols, rows[0]))
    assert row["success"] and row["status_code"] == 200 and row["input_tokens"] == 7 and row["output_tokens"] == 3
    assert row["run_id"] == run and row["upstream"] == "in_process" and row["model_id"] == h.slug_of(pid)

    tel = client.get(f"/api/projects/{pid}/gateway/telemetry").json()
    assert tel["summary"]["total"] == 1
    entry = tel["summary"]["by_model"][0]
    assert entry["model_id"] == h.slug_of(pid) and entry["requests"] == 1 and entry["error_rate"] == 0.0
    rec = tel["recent"][0]
    assert rec["key_prefix"] == key[:12] and rec["status_code"] == 200 and rec["run_id"] == run
    assert set(rec) == {"ts", "model_id", "run_id", "status_code", "latency_ms", "retrieval_applied",
                        "guided_json_mode", "key_prefix"}


# ------------------------------------------------------------------ review fixes


def test_retrieval_runs_once_per_request(client, root):
    pid, run, key = _retrieval_project(client, root)
    fake = fake_retrieve(hits(1))
    with mock.patch("app.services.local_chat_service.generate_reply", Calls("ok")), \
            mock.patch("app.services.auto_rag_service.retrieve_document_passages", fake):
        assert chat(client, key, h.slug_of(pid), brewslm={"retrieval": True}).status_code == 200
    assert len(fake.calls) == 1


def test_telemetry_failure_never_fails_the_request(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    key = h.make_key(client, [pid])

    async def boom(*a, **k):
        raise RuntimeError("log table gone")

    with mock.patch("app.services.local_chat_service.generate_reply", Calls("fine")), \
            mock.patch("app.services.served_model_telemetry_service.record_gateway_request", boom):
        resp = chat(client, key, h.slug_of(pid))
    assert resp.status_code == 200 and resp.json()["choices"][0]["message"]["content"] == "fine"


def test_unexpected_exception_is_logged_as_500(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    key = h.make_key(client, [pid])

    async def kaboom(*a, **k):
        raise RuntimeError("unexpected")

    with mock.patch("app.services.gateway_service._execute", kaboom):
        with pytest.raises(RuntimeError):
            chat(client, key, h.slug_of(pid))
    rec = client.get(f"/api/projects/{pid}/gateway/telemetry").json()["recent"]
    assert rec and rec[0]["status_code"] == 500


def test_v1_errors_stay_in_the_openai_envelope(client):
    from starlette.requests import Request

    from app.main import last_resort_exception_handler

    key = h.make_key(client, [h.make_project(client)])
    for resp, status in (
        (client.get("/v1/nope", headers=h.auth(key)), 404),
        (client.delete("/v1/models", headers=h.auth(key)), 405),
        (client.post("/v1/brewslm/retrieve", headers=h.auth(key), json={"query": 3}), 422),
    ):
        assert resp.status_code == status, resp.text
        err = resp.json()["error"]
        assert set(err) == {"message", "type", "code"} and "detail" not in resp.json()
    assert resp.json()["error"]["type"] == "invalid_request_error"
    # the api surface keeps its own shapes
    assert "error" not in client.get("/api/projects/99999999").json()
    scope = {"type": "http", "method": "GET", "path": "/v1/models", "headers": [], "query_string": b""}
    out = asyncio.run(last_resort_exception_handler(Request(scope), RuntimeError("secret detail")))
    body = json.loads(out.body)
    assert out.status_code == 500 and body["error"]["type"] == "server_error" and "secret detail" not in out.body.decode()


def test_conflicting_key_headers_are_rejected(client):
    key = h.make_key(client, [h.make_project(client)])
    other = h.make_key(client, [h.make_project(client)])
    both_same = client.get("/v1/models", headers={"Authorization": f"Bearer {key}", "X-API-Key": key})
    assert both_same.status_code == 200
    clash = client.get("/v1/models", headers={"Authorization": f"Bearer {key}", "X-API-Key": other})
    assert clash.status_code == 401 and clash.json()["error"]["code"] == "invalid_api_key"


def test_slug_collisions_check_every_fallback(client):
    first = h.make_project(client, name="Collide Name")
    second = h.make_project(client, name="collide name")
    third = h.make_project(client, name=f"collide-name-{second}")
    blank = h.make_project(client, name="!!!")
    shadow = h.make_project(client, name=f"p{blank}")

    async def _slugs():
        async with async_session_factory() as db:
            return await gateway_service.all_slugs(db)

    slugs = asyncio.run(_slugs())
    assert slugs[first] == "collide-name" and slugs[second] == f"collide-name-{second}"
    assert slugs[third] == f"collide-name-{second}-{third}"
    assert slugs[blank] == f"p{blank}" and slugs[shadow] == f"p{blank}-{shadow}"
    assert len(set(slugs.values())) == len(slugs)


def test_project_delete_cuts_keys_and_logs_and_survives_id_reuse(client, root):
    from app.models.project import Project

    keep = h.make_project(client)
    doomed = h.make_project(client)
    h.seed_run(doomed, root)
    only = h.make_key(client, [doomed])
    both = h.make_key(client, [doomed, keep])
    with mock.patch("app.services.local_chat_service.generate_reply", Calls("ok")):
        assert chat(client, only, h.slug_of(doomed)).status_code == 200
    assert client.delete(f"/api/projects/{doomed}").status_code == 204

    async def _state():
        async with async_session_factory() as db:
            logs = (await db.execute(text("select count(*) from gateway_request_logs where project_id = :p"), {"p": doomed})).scalar()
            keys = (await db.execute(text("select key_prefix, project_ids, is_active, revoked_at from gateway_api_keys"))).all()
            return logs, {k[0]: k for k in keys}

    logs, keys = asyncio.run(_state())
    assert logs == 0
    only_row, both_row = keys[only[:12]], keys[both[:12]]
    assert only_row[2] == 0 and only_row[3] is not None and json.loads(only_row[1]) == []
    assert both_row[2] == 1 and json.loads(both_row[1]) == [keep]

    async def _reuse():
        async with async_session_factory() as db:  # SQLite hands the freed id to a new project
            db.add(Project(id=doomed, name="Reused Id", description=""))
            await db.commit()

    asyncio.run(_reuse())
    assert chat(client, only, "p%d" % doomed).status_code == 401  # revoked
    resp = chat(client, both, "p%d" % doomed)
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "project_not_in_scope"

"""Gateway keys, per-project settings and snippets (G11)."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _gateway_helpers as h  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.models.auth import GlobalRole, ProjectRole  # noqa: E402
from app.security import create_user_with_key, upsert_project_membership  # noqa: E402


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


def test_key_lifecycle_and_list_has_no_hash(client):
    pid = h.make_project(client)
    resp = client.post("/api/gateway/keys", json={"name": "ci", "project_ids": [pid, pid], "rate_limit_per_minute": 5})
    assert resp.status_code == 201
    body = resp.json()
    assert body["key"].startswith("bsk_") and len(body["key"]) > 40
    rec = body["record"]
    assert rec["key_prefix"] == body["key"][:12] and rec["project_ids"] == [pid]
    assert rec["rate_limit_per_minute"] == 5 and rec["is_active"] is True and rec["revoked_at"] is None
    listed = client.get("/api/gateway/keys").json()["keys"]
    mine = next(k for k in listed if k["id"] == rec["id"])
    assert "key_hash" not in mine and body["key"] not in str(listed)
    assert set(mine) == {"id", "name", "key_prefix", "project_ids", "rate_limit_per_minute", "is_active",
                         "created_at", "last_used_at", "revoked_at"}
    client.get("/v1/models", headers=h.auth(body["key"]))
    assert next(k for k in client.get("/api/gateway/keys").json()["keys"] if k["id"] == rec["id"])["last_used_at"]
    gone = client.delete(f"/api/gateway/keys/{rec['id']}").json()["record"]
    assert gone["is_active"] is False and gone["revoked_at"]
    assert client.delete(f"/api/gateway/keys/{rec['id']}").status_code == 200  # idempotent
    assert client.delete("/api/gateway/keys/99999999").status_code == 404


@pytest.mark.parametrize("body", [
    {"name": "x", "project_ids": []},
    {"name": "x", "project_ids": [99999999]},
    {"name": "  ", "project_ids": [1]},
    {"name": "x", "project_ids": [1], "rate_limit_per_minute": 0},
])
def test_key_validation(client, body):
    if body["project_ids"] == [1]:
        body["project_ids"] = [h.make_project(client)]
    assert client.post("/api/gateway/keys", json=body).status_code == 400


def test_key_creation_needs_editor_on_every_project_when_auth_on(client):
    p1, p2 = h.make_project(client), h.make_project(client)

    async def _seed():
        async with async_session_factory() as db:
            u = await create_user_with_key(db, f"eng-{uuid4().hex[:6]}", GlobalRole.ENGINEER)
            await upsert_project_membership(db, p1, u["user_id"], ProjectRole.EDITOR)
            await upsert_project_membership(db, p2, u["user_id"], ProjectRole.VIEWER)
            await db.commit()
            return u["api_key"]

    api_key = asyncio.run(_seed())
    settings.AUTH_ENABLED = True
    try:
        hdr = {"x-api-key": api_key}
        ok = client.post("/api/gateway/keys", json={"name": "a", "project_ids": [p1]}, headers=hdr)
        assert ok.status_code == 201, ok.text
        denied = client.post("/api/gateway/keys", json={"name": "b", "project_ids": [p1, p2]}, headers=hdr)
        assert denied.status_code == 403
        # /v1 is always key-gated and ignores the user's API key
        assert client.get("/v1/models", headers=hdr).status_code == 401
    finally:
        settings.AUTH_ENABLED = False
    # /v1 requires a gateway key even with AUTH_ENABLED=false
    assert client.get("/v1/models").status_code == 401


def test_settings_roundtrip_partial_merge_and_contract(client, root):
    pid = h.make_project(client, name=f"Gateway Demo {uuid4().hex[:4]}")
    old = h.seed_run(pid, root, config={"passage_prompts": True}, name="first")
    new = h.seed_run(pid, root, adapter=False, name="second")
    h.seed_run(pid, root, weights=False)  # not servable
    got = client.get(f"/api/projects/{pid}/gateway/settings").json()
    slug = got["slug"]
    assert slug.startswith("gateway-demo-")
    assert [r["id"] for r in got["servable_runs"]] == [old, new]  # ascending
    assert got["servable_runs"][0] == {"id": old, "name": "first", "base_model": h.BASE,
                                       "passage_trained": True, "adapter": True}
    assert got["servable_runs"][1]["adapter"] is False
    assert got["model_ids"] == [slug, f"{slug}:base", f"{slug}:run-{old}", f"{slug}:run-{new}"]
    assert got["base_model"] == h.BASE and got["default_run_resolved"] == new
    assert got["default_resolves_to"] == {"kind": "run", "run_id": new}
    assert got["guided_json_available"] is False
    assert got["settings"] == {"upstream": "in_process", "base_url": None, "upstream_kind": None,
                               "upstream_model_map": {}, "default_run_id": None, "tools": []}

    tool = {"name": "ask", "description": "d", "input_schema": {"type": "object"}, "prompt_template": "{q}"}
    put = client.put(f"/api/projects/{pid}/gateway/settings", json={"tools": [tool], "default_run_id": old})
    assert put.status_code == 200, put.text
    assert put.json()["default_run_resolved"] == old
    # partial: upstream fields without the map/tools keep what was stored
    put = client.put(f"/api/projects/{pid}/gateway/settings", json={
        "upstream": "openai_compatible", "base_url": "http://x.test:8001/", "upstream_kind": "vllm"})
    s = put.json()["settings"]
    assert s["upstream"] == "openai_compatible" and s["base_url"] == "http://x.test:8001"
    assert [t["name"] for t in s["tools"]] == ["ask"] and s["default_run_id"] == old
    assert client.get(f"/api/projects/{pid}/gateway/settings").json()["settings"] == s


def test_settings_validation_errors_shape(client, root):
    pid = h.make_project(client)
    bad = [
        {"name": "Bad Name", "input_schema": {"type": "object"}, "prompt_template": "x"},
        {"name": "ok_tool", "input_schema": {"type": "object", "properties": {"a": {"type": 5}}}, "prompt_template": "x"},
        {"name": "ok_tool", "input_schema": {"type": "object"}, "prompt_template": "@recipe"},
        {"name": "t3", "input_schema": {"type": "object"}, "prompt_template": "x", "model": "run-3"},
    ]
    resp = client.put(f"/api/projects/{pid}/gateway/settings", json={"tools": bad, "upstream": "nope", "default_run_id": 12345})
    assert resp.status_code == 400
    body = resp.json()
    fields = {e["field"] for e in body["errors"]}
    assert {"upstream", "default_run_id", "tools[0].name", "tools[1].input_schema",
            "tools[2].prompt_template", "tools[3].model"} <= fields
    assert body["metadata"]["errors"] == body["errors"] and isinstance(body["detail"], str)
    # nothing persisted
    assert client.get(f"/api/projects/{pid}/gateway/settings").json()["settings"]["tools"] == []
    r = client.put(f"/api/projects/{pid}/gateway/settings", json={"upstream": "openai_compatible"})
    assert r.status_code == 400 and r.json()["errors"][0]["field"] == "base_url"


def test_snippets_use_placeholder_never_a_key(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    key = h.make_key(client, [pid])
    snip = client.get(f"/api/projects/{pid}/gateway/snippets").json()
    slug = h.slug_of(pid)
    assert snip["gateway_url"] == settings.GATEWAY_PUBLIC_URL.rstrip("/")
    assert slug in snip["model_ids"] and f"/v1/chat/completions" in snip["curl"]
    assert "${BREWSLM_API_KEY}" in snip["curl"] and "BREWSLM_API_KEY" in snip["openai_python"]
    server = snip["mcp_json"]["mcpServers"]["brewslm"]
    assert server["type"] == "stdio" and server["args"] == ["-m", "app.mcp_server"]
    assert server["command"] == sys.executable and Path(server["cwd"]).name == "backend"
    assert server["env"] == {"BREWSLM_GATEWAY_URL": snip["gateway_url"], "BREWSLM_API_KEY": "${BREWSLM_API_KEY}"}
    assert key not in str(snip)


def test_default_resolves_to_for_rag_first_and_empty(client, root):
    empty = h.make_project(client)
    got = client.get(f"/api/projects/{empty}/gateway/settings").json()
    assert got["default_resolves_to"] == {"kind": "none", "run_id": None} and got["default_run_resolved"] is None
    rag = h.make_project(client)
    h.set_runtime(rag, rag_first=True)
    got = client.get(f"/api/projects/{rag}/gateway/settings").json()
    assert got["default_resolves_to"] == {"kind": "base_with_retrieval", "run_id": None}
    assert got["default_run_resolved"] is None and got["slug"] in got["model_ids"]


def test_key_listing_and_revoke_for_creator_owner_and_others_with_auth_on(client):
    p1, p2 = h.make_project(client), h.make_project(client)

    async def _user(role_by_project):
        async with async_session_factory() as db:
            u = await create_user_with_key(db, f"u-{uuid4().hex[:6]}", GlobalRole.ENGINEER)
            for pid, role in role_by_project.items():
                await upsert_project_membership(db, pid, u["user_id"], role)
            await db.commit()
            return {"x-api-key": u["api_key"]}

    creator = asyncio.run(_user({p1: ProjectRole.EDITOR}))
    owner_all = asyncio.run(_user({p1: ProjectRole.OWNER, p2: ProjectRole.OWNER}))
    owner_one = asyncio.run(_user({p1: ProjectRole.OWNER}))
    other = asyncio.run(_user({}))
    settings.AUTH_ENABLED = True
    try:
        rec = client.post("/api/gateway/keys", json={"name": "scoped", "project_ids": [p1]}, headers=creator).json()["record"]
        wide = client.post("/api/gateway/keys", json={"name": "wide", "project_ids": [p1, p2]}, headers=owner_all).json()["record"]

        def ids(hdr):
            return {k["id"] for k in client.get("/api/gateway/keys", headers=hdr).json()["keys"]}

        assert rec["id"] in ids(creator) and rec["id"] in ids(owner_all) and rec["id"] in ids(owner_one)
        assert rec["id"] not in ids(other)
        assert wide["id"] in ids(owner_all) and wide["id"] not in ids(owner_one)  # must own EVERY project in scope
        assert client.delete(f"/api/gateway/keys/{rec['id']}", headers=other).status_code == 403
        assert client.delete(f"/api/gateway/keys/{wide['id']}", headers=owner_one).status_code == 403
        assert client.delete(f"/api/gateway/keys/{rec['id']}", headers=owner_one).json()["record"]["is_active"] is False
        assert client.delete(f"/api/gateway/keys/{wide['id']}", headers=creator).status_code == 403
        assert client.delete(f"/api/gateway/keys/{wide['id']}", headers=owner_all).status_code == 200
    finally:
        settings.AUTH_ENABLED = False

"""Tool catalog: store / version / flags / history / secrets / API / domain-pack hookup / dry-run safety.

Stdio goes through ``fake_mcp_server.py`` (allow-listed by absolute path via a settings patch); HTTP goes
through ``httpx.MockTransport``. No network."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.project import Project
from app.services import json_schema_bundle as jsb
from app.services import mcp_client
from app.services import tool_catalog_service as tcs

HERE = Path(__file__).resolve().parent
FAKE = str(HERE / "fake_mcp_server.py")
PY = sys.executable
SECRET = "sekret-value-12345"
_REAL_ASYNC_CLIENT = httpx.AsyncClient
client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    wait_for_jobs_idle(client)
    client.__exit__(None, None, None)


def new_project() -> int:
    return client.post("/api/projects", json={"name": f"tools-{time.time_ns()}", "description": "x"}).json()["id"]


def stdio_body(**over):
    body = {"kind": "mcp_stdio", "label": "deploy fake", "command": [PY, FAKE], "env": {"API_TOKEN": SECRET}}
    body.update(over)
    return body


def allow():
    return mock.patch.object(settings, "MCP_STDIO_ALLOWED_BINARIES", PY)


def run_db(fn):
    async def go():
        async with async_session_factory() as db:
            project = await db.get(Project, fn.pid)
            out = await fn(db, project)
            await db.commit()
            return out
    return asyncio.run(go())


def simple_tools(*names):
    return [{"name": n, "description": f"{n} tool", "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}} for n in names]


class CatalogApiTests(unittest.TestCase):
    def setUp(self):
        self.pid = new_project()
        self.base = f"/api/projects/{self.pid}/tool-catalog"

    def tearDown(self):
        wait_for_jobs_idle(client)

    def import_stdio(self, **over):
        with allow():
            return client.post(f"{self.base}/import/mcp", json=stdio_body(**over))

    def test_empty(self):
        self.assertEqual(client.get(self.base).json(), {"catalog": None, "history": []})
        self.assertEqual(tcs.render_tool_block({}), "")
        self.assertEqual(tcs.catalog_fingerprint({}), "none")

    def test_stdio_import_provenance_and_secret_hygiene(self):
        with self.assertLogs(level=logging.DEBUG) as logs:
            logging.getLogger("probe").debug("start")
            r = self.import_stdio()
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        cat = body["catalog"]
        self.assertEqual([t["name"] for t in cat["tools"]], ["create_service", "delete_service", "get_service", "list_services"])
        self.assertEqual(cat["source"]["kind"], "mcp_stdio")
        self.assertEqual(cat["source"]["env_names"], ["API_TOKEN"])
        self.assertEqual(cat["source"]["secret_provider"], "mcp_stdio_deploy_fake")
        self.assertEqual(cat["server"]["name"], "deploy-service-fake")
        self.assertEqual(len(cat["version"]), 12)
        by = {t["name"]: t for t in cat["tools"]}
        self.assertTrue(by["get_service"]["read_only_effective"] and not by["get_service"]["mutating"])
        self.assertTrue(by["create_service"]["mutating"] and by["delete_service"]["mutating"])
        self.assertEqual(by["get_service"]["signature"], "get_service(name: string) — Fetch one service. [read-only]")
        # the value is nowhere: not in the payload, the project, the logs; the secret store has it
        self.assertNotIn(SECRET, r.text)
        self.assertNotIn(SECRET, client.get(f"/api/projects/{self.pid}").text)
        self.assertNotIn(SECRET, client.get(self.base).text)
        self.assertNotIn(SECRET, "\n".join(logs.output))
        secrets = client.get(f"/api/projects/{self.pid}/secrets").json()["secrets"]
        self.assertEqual([(s["provider"], s["key_name"]) for s in secrets], [("mcp_stdio_deploy_fake", "api_token")])
        self.assertNotIn(SECRET, json.dumps(secrets))

    def test_reimport_is_noop_keeps_flags_and_reuses_secret(self):
        v1 = self.import_stdio().json()["catalog"]["version"]
        client.patch(f"{self.base}/flags", json={"flags": {"get_service": {"allowed": False}}})
        r = self.import_stdio(env={"API_TOKEN": None})
        self.assertEqual(r.status_code, 200, r.text)
        status = r.json()
        self.assertEqual(status["catalog"]["version"], v1)
        self.assertEqual(len(status["history"]), 1)
        got = {t["name"]: t for t in status["catalog"]["tools"]}
        self.assertFalse(got["get_service"]["allowed"])
        # an undeclared-secret null is a 422, not a silent empty value
        r = self.import_stdio(env={"OTHER": None})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["error_code"], "INVALID_TOOL_CATALOG")

    def test_stdio_disabled_and_role_gate(self):
        r = client.post(f"{self.base}/import/mcp", json=stdio_body())
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["error_code"], "MCP_STDIO_DISABLED")
        self.assertIn("MCP_STDIO_ALLOWED_BINARIES", r.json()["message"])
        self.assertEqual(client.get(self.base).json()["catalog"], None)
        # not-admin principal while auth is on
        with allow(), mock.patch.object(settings, "AUTH_ENABLED", True), mock.patch("app.api.admin.get_request_principal", return_value=None):
            r = client.post(f"{self.base}/import/mcp", json=stdio_body())
        self.assertEqual(r.status_code, 401)

    def test_server_error_is_502_and_redacted(self):
        r = self.import_stdio(env={"FAKE_MCP_LEAK": SECRET})
        self.assertEqual(r.status_code, 502, r.text)
        self.assertEqual(r.json()["error_code"], "MCP_SERVER_ERROR")
        self.assertNotIn(SECRET, r.text)
        self.assertEqual(client.get(f"/api/projects/{self.pid}/secrets").json()["secrets"], [])  # nothing stored on failure

    def test_bad_requests(self):
        for body in ({"kind": "nope"}, {"kind": "mcp_http", "url": "ftp://x"}, {"kind": "mcp_stdio", "command": []}):
            with allow():
                r = client.post(f"{self.base}/import/mcp", json=body)
            self.assertEqual(r.status_code, 422, body)

    def test_http_import(self):
        seen = []

        def handler(req):
            seen.append(req)
            if req.method == "DELETE":
                return httpx.Response(204)
            msg = json.loads(req.content)
            if "id" not in msg:
                return httpx.Response(202)
            if msg["method"] == "initialize":
                res = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "h", "version": "1"}}
            else:
                res = {"tools": [{"name": "ping_it", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}}]}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "result": res})

        def factory(*a, **kw):
            kw.pop("transport", None)
            return _REAL_ASYNC_CLIENT(*a, transport=httpx.MockTransport(handler), **kw)

        body = {"kind": "mcp_http", "url": "http://mcp.test/mcp", "headers": {"X-Api-Key": SECRET}, "label": "hosted"}
        with mock.patch.object(mcp_client.httpx, "AsyncClient", factory):
            r = client.post(f"{self.base}/import/mcp", json=body)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertNotIn(SECRET, r.text)
            cat = r.json()["catalog"]
            self.assertEqual(cat["source"], {"kind": "mcp_http", "label": "hosted", "url": "http://mcp.test/mcp", "header_names": ["X-Api-Key"], "secret_provider": "mcp_http_hosted", "endpoint_binding": "http://mcp.test:80/mcp"})
            self.assertEqual(seen[0].headers["x-api-key"], SECRET)
            secrets = client.get(f"/api/projects/{self.pid}/secrets").json()["secrets"]
            self.assertEqual(secrets[0]["key_name"], "x_api_key")

    def test_async_job(self):
        with allow():
            r = client.post(f"{self.base}/import/mcp", params={"async_job": "true"}, json=stdio_body())
            self.assertEqual(r.status_code, 202, r.text)
            job = r.json()
            self.assertEqual(job["kind"], "tool_catalog_import")
            self.assertNotIn(SECRET, json.dumps(job))
            wait_for_jobs_idle(client)
        done = client.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(done["status"].lower(), "succeeded", done)
        self.assertEqual(len(done["result"]["catalog"]["tools"]), 4)
        self.assertNotIn(SECRET, json.dumps(done))

    def test_flags_and_destructive_override(self):
        self.import_stdio()
        r = client.patch(f"{self.base}/flags", json={"flags": {"create_service": {"read_only": True}}})
        self.assertEqual(r.status_code, 200, r.text)
        by = {t["name"]: t for t in r.json()["catalog"]["tools"]}
        self.assertTrue(by["create_service"]["read_only_effective"])
        fp = r.json()["catalog"]["flags_version"]
        r = client.patch(f"{self.base}/flags", json={"flags": {"delete_service": {"read_only": True}}})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["error_code"], "INVALID_TOOL_CATALOG")
        r = client.patch(f"{self.base}/flags", json={"flags": {"nope": {"allowed": False}}})
        self.assertEqual(r.status_code, 422)
        # null clears the override; omitted keys stay as they were
        r = client.patch(f"{self.base}/flags", json={"flags": {"create_service": {"read_only": None}}})
        by = {t["name"]: t for t in r.json()["catalog"]["tools"]}
        self.assertFalse(by["create_service"]["read_only_effective"])
        self.assertNotEqual(r.json()["catalog"]["flags_version"], fp)
        r = client.patch(f"{self.base}/flags", json={"flags": {"get_service": {"allowed": False}}})
        by = {t["name"]: t for t in r.json()["catalog"]["tools"]}
        self.assertFalse(by["get_service"]["allowed"])
        self.assertTrue(by["get_service"]["read_only_effective"])

    def test_history_restore_and_clear(self):
        a = client.put(f"{self.base}/tools", json={"tools": simple_tools("alpha", "beta")}).json()["catalog"]["version"]
        same = client.put(f"{self.base}/tools", json={"tools": simple_tools("beta", "alpha")}).json()
        self.assertEqual(same["catalog"]["version"], a)
        self.assertEqual(len(same["history"]), 1)
        b = client.put(f"{self.base}/tools", json={"tools": simple_tools("gamma")}).json()
        self.assertEqual([h["current"] for h in b["history"]], [True, False])
        r = client.post(f"{self.base}/restore", json={"version": a})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["catalog"]["version"], a)
        self.assertEqual(r.json()["catalog"]["source"]["restored_from"], a)
        r = client.post(f"{self.base}/restore", json={"version": "000000000000"})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["error_code"], "TOOL_CATALOG_VERSION_NOT_FOUND")
        cleared = client.delete(self.base).json()
        self.assertIsNone(cleared["catalog"])
        self.assertTrue(cleared["history"])
        # history trims to 10, only the newest 5 restorable
        for i in range(12):
            st = client.put(f"{self.base}/tools", json={"tools": simple_tools(f"t{i}")}).json()
        self.assertEqual(len(st["history"]), 10)
        self.assertEqual([h["restorable"] for h in st["history"]], [True] * 5 + [False] * 5)
        old = st["history"][-1]["version"]
        self.assertEqual(client.post(f"{self.base}/restore", json={"version": old}).status_code, 404)

    def test_manual_validation(self):
        for tools in ([], [{"name": "bad name"}], [{"name": "a"}, {"name": "a"}], [{"description": "x"}]):
            r = client.put(f"{self.base}/tools", json={"tools": tools})
            self.assertEqual(r.status_code, 422, tools)
        r = client.put(f"{self.base}/tools", json={"tools": [{"name": "broken", "inputSchema": {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}}}]})
        self.assertEqual(r.status_code, 200)
        self.assertIn("schema_error", r.json()["catalog"]["tools"][0])
        self.assertIsNone(tcs.tool_bundle(self._rc(), "broken"))

    def _rc(self):
        return client.get(f"/api/projects/{self.pid}").json()["runtime_config"]

    def test_openapi_import(self):
        doc = {"openapi": "3.0.0", "info": {"title": "Deploy API", "version": "2"}, "paths": {"/s": {"get": {"operationId": "listS"}, "delete": {"operationId": "dropS"}}}}
        r = client.post(f"{self.base}/import/openapi", json={"filename": "d.json", "content": json.dumps(doc)})
        self.assertEqual(r.status_code, 200, r.text)
        cat = r.json()["catalog"]
        self.assertEqual(cat["source"], {"kind": "openapi", "label": "Deploy API", "filename": "d.json", "title": "Deploy API", "version": "2"})
        by = {t["name"]: t for t in cat["tools"]}
        self.assertTrue(by["listS"]["read_only_effective"] and by["dropS"]["mutating"] and by["dropS"]["destructive"])
        for bad in ("", "openapi: 2.0.0", "a: &x [1]\nb: *x", "{"):
            r = client.post(f"{self.base}/import/openapi", json={"filename": "x", "content": bad})
            self.assertEqual(r.status_code, 422, bad)

    def test_fingerprint_changes_with_flags(self):
        self.import_stdio()
        before = tcs.catalog_fingerprint(self._rc())
        client.patch(f"{self.base}/flags", json={"flags": {"get_service": {"allowed": False}}})
        self.assertNotEqual(tcs.catalog_fingerprint(self._rc()), before)

    def test_prompt_block_and_effective_tools(self):
        self.import_stdio()
        client.patch(f"{self.base}/flags", json={"flags": {"list_services": {"allowed": False}}})
        rc = self._rc()
        block = tcs.render_tool_block(rc)
        lines = block.splitlines()
        self.assertEqual(lines[0], "Available tools (call only these):")
        self.assertEqual([ln.split("(")[0] for ln in lines[1:]], ["- create_service", "- delete_service", "- get_service"])
        self.assertIn("create_service(name: string, tier: free|pro|enterprise) — Create a service. [mutating]", block)
        self.assertIn("[destructive]", block)
        self.assertEqual([t["name"] for t in tcs.allowed_tools(rc)], ["create_service", "delete_service", "get_service"])
        small = tcs.render_tool_block(rc, max_chars=150)
        self.assertLessEqual(len(small), 150)
        self.assertIn("truncated", small)
        bundle = tcs.tool_bundle(rc, "create_service")
        self.assertEqual(bundle.validate({"name": "a", "tier": "pro"}), [])
        self.assertTrue(bundle.validate({"name": "a", "tier": "gold"}))
        self.assertIs(bundle, tcs.tool_bundle(rc, "create_service"))
        self.assertIsNone(tcs.tool_bundle(rc, "nope"))


class SchemaTests(unittest.TestCase):
    def test_normalize(self):
        s = tcs.normalize_input_schema({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"pair": {"type": "array", "items": [{"type": "string"}, {"type": "integer"}], "additionalItems": False},
                           "n": {"type": "number", "minimum": 1, "exclusiveMinimum": True}},
            "definitions": {"d": {"type": "string"}},
        })
        self.assertNotIn("$schema", s)
        self.assertEqual(s["properties"]["pair"]["prefixItems"], [{"type": "string"}, {"type": "integer"}])
        self.assertIs(s["properties"]["pair"]["items"], False)
        self.assertEqual(s["properties"]["n"], {"type": "number", "exclusiveMinimum": 1})
        self.assertIn("definitions", s)
        self.assertEqual(tcs.normalize_input_schema(None), {"type": "object"})
        t = tcs.normalize_tool({"name": "x", "inputSchema": s})
        self.assertNotIn("schema_error", t)

    def test_signature_types(self):
        sig = tcs.compact_signature({
            "name": "f", "description": "d" * 300, "read_only": True, "destructive": None,
            "input_schema": {"type": "object", "required": ["a"], "properties": {
                "a": {"type": "string", "enum": list("abcdefghij")}, "b": {"type": "object"},
                "c": {"type": "array", "items": {"type": "integer"}}, "d": {"$ref": "#/$defs/x"}},
                "$defs": {"x": {"type": "boolean"}}},
        })
        self.assertTrue(sig.startswith("f(a: a|b|c|d|e|f|g|h|…, b?: object, c?: array<integer>, d?: boolean) — "))
        self.assertTrue(sig.endswith("… [read-only]"))
        self.assertLessEqual(len(sig.split(" — ", 1)[1]) - len(" [read-only]"), 160)


class DryRunTests(unittest.TestCase):
    def setUp(self):
        self.pid = new_project()
        self.base = f"/api/projects/{self.pid}/tool-catalog"
        self.log = Path(tempfile.mkdtemp()) / "calls.log"
        with allow():
            r = client.post(f"{self.base}/import/mcp", json=stdio_body(env={"API_TOKEN": SECRET, "FAKE_MCP_CALL_LOG": str(self.log)}))
        self.assertEqual(r.status_code, 200, r.text)

    def calls(self):
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []

    def dry(self, name, args=None, **patches):
        async def fn(db, project):
            return await tcs.dry_run_call(db, project, name, args or {})
        fn.pid = self.pid
        with allow(), mock.patch.object(settings, "MCP_DRY_RUN_ENABLED", patches.get("enabled", True)):
            return run_db(fn)

    def test_read_only_call_executes(self):
        out = self.dry("get_service", {"name": "api"})
        self.assertEqual(out["ok"], True)
        self.assertIn("get_service", out["snippet"])
        self.assertLessEqual(len(out["snippet"]), 300)
        self.assertEqual(self.calls(), [{"name": "get_service", "arguments": {"name": "api"}}])

    def test_invalid_args_do_not_call(self):
        out = self.dry("get_service", {"nope": 1, "name": 5})
        self.assertFalse(out["ok"])
        self.assertIn("invalid arguments", out["error"])
        self.assertEqual(self.calls(), [])

    def test_disabled(self):
        with self.assertRaises(tcs.DryRunDisabled):
            self.dry("get_service", {"name": "a"}, enabled=False)
        self.assertEqual(self.calls(), [])

    def test_mutating_and_destructive_are_refused(self):
        for name, args in (("create_service", {"name": "a", "tier": "pro"}), ("delete_service", {"name": "a"}), ("missing", {})):
            with self.assertRaises(tcs.MutatingToolRefused, msg=name):
                self.dry(name, args)
        self.assertEqual(self.calls(), [])

    def test_override_attempts(self):
        # allowed override on a mutating tool via the API
        client.patch(f"{self.base}/flags", json={"flags": {"create_service": {"read_only": True}}})
        # the destructive tool cannot be overridden through the API...
        self.assertEqual(client.patch(f"{self.base}/flags", json={"flags": {"delete_service": {"read_only": True}}}).status_code, 422)
        # ...and a tampered store (flags written behind the API's back) is still refused
        async def tamper(db, project):
            cfg = dict(project.runtime_config)
            cat = dict(cfg["tool_catalog"])
            cat["flags"] = {"delete_service": {"allowed": True, "read_only": True}}
            cfg["tool_catalog"] = cat
            project.runtime_config = cfg
        tamper.pid = self.pid
        run_db(tamper)
        with self.assertRaises(tcs.MutatingToolRefused):
            self.dry("delete_service", {"name": "a"})
        self.assertEqual(self.calls(), [])
        # a user-overridden mutating tool (read_only true, not destructive) is callable; that is the user's decision
        client.patch(f"{self.base}/flags", json={"flags": {"create_service": {"read_only": True}}})

    def test_not_allowed_is_refused(self):
        client.patch(f"{self.base}/flags", json={"flags": {"get_service": {"allowed": False}}})
        with self.assertRaises(tcs.MutatingToolRefused):
            self.dry("get_service", {"name": "a"})
        self.assertEqual(self.calls(), [])

    def test_non_mcp_source_is_refused(self):
        client.put(f"{self.base}/tools", json={"tools": [{"name": "get_service", "annotations": {"readOnlyHint": True}}]})
        with self.assertRaises(tcs.MutatingToolRefused):
            self.dry("get_service", {})
        self.assertEqual(self.calls(), [])


def http_mock(seen: list, tools=None):
    """An ``httpx.AsyncClient`` factory (patched into mcp_client) over a MockTransport that records requests."""
    tools = tools or [{"name": "ping_it", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}}]

    def handler(req):
        seen.append(req)
        if req.method == "DELETE":
            return httpx.Response(204)
        msg = json.loads(req.content)
        if "id" not in msg:
            return httpx.Response(202)
        res = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "h", "version": "1"}} if msg["method"] == "initialize" else {"tools": tools}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "result": res})

    def factory(*a, **kw):
        kw.pop("transport", None)
        return _REAL_ASYNC_CLIENT(*a, transport=httpx.MockTransport(handler), **kw)

    return factory


class SecretBindingTests(unittest.TestCase):
    """A stored secret is only ever reused for the same kind AND the same endpoint."""

    def setUp(self):
        self.pid = new_project()
        self.base = f"/api/projects/{self.pid}/tool-catalog"

    def tearDown(self):
        wait_for_jobs_idle(client)

    def http(self, url, headers, label="deploy"):
        return client.post(f"{self.base}/import/mcp", json={"kind": "mcp_http", "label": label, "url": url, "headers": headers})

    def test_same_label_new_url_does_not_reuse_the_secret(self):
        seen: list = []
        with mock.patch.object(mcp_client.httpx, "AsyncClient", http_mock(seen)):
            self.assertEqual(self.http("http://good.test/mcp", {"Authorization": "Bearer " + SECRET}).status_code, 200)
            n = len(seen)
            r = self.http("http://evil.test/mcp", {"Authorization": None})
            self.assertEqual(r.status_code, 422, r.text)
            err = r.json()["metadata"]["errors"][0]
            self.assertEqual(err["code"], "missing_secret")
            self.assertIn("Authorization", err["message"])
            self.assertIn("endpoint changed", err["message"])
            self.assertEqual(len(seen), n, "no request may be made once the secret cannot be reused")
            # a different path on the same host is also a different endpoint
            self.assertEqual(self.http("http://good.test/other", {"Authorization": None}).status_code, 422)
            self.assertEqual(len(seen), n)
        self.assertTrue(all(r.url.host == "good.test" for r in seen))
        self.assertNotIn("evil.test", [r.url.host for r in seen])

    def test_other_label_has_no_fallback(self):
        seen: list = []
        with mock.patch.object(mcp_client.httpx, "AsyncClient", http_mock(seen)):
            self.assertEqual(self.http("http://good.test/mcp", {"Authorization": "Bearer " + SECRET}).status_code, 200)
            n = len(seen)
            r = self.http("http://good.test/mcp", {"Authorization": None}, label="other")
            self.assertEqual(r.status_code, 422, r.text)
            self.assertEqual(len(seen), n)

    def test_same_endpoint_still_reuses(self):
        seen: list = []
        with mock.patch.object(mcp_client.httpx, "AsyncClient", http_mock(seen)):
            self.assertEqual(self.http("http://good.test/mcp", {"Authorization": "Bearer " + SECRET}).status_code, 200)
            r = self.http("http://GOOD.test:80/mcp", {"Authorization": None})
            self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(seen[-2].headers["authorization"], "Bearer " + SECRET)

    def test_stdio_label_secret_is_not_usable_from_http(self):
        with allow():
            self.assertEqual(client.post(f"{self.base}/import/mcp", json=stdio_body(label="shared")).status_code, 200)
        seen: list = []
        with mock.patch.object(mcp_client.httpx, "AsyncClient", http_mock(seen)):
            r = self.http("http://evil.test/mcp", {"API_TOKEN": None}, label="shared")
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(seen, [])
        # the provider carries the kind
        self.assertEqual(tcs.secret_provider_for("shared", "mcp_stdio"), "mcp_stdio_shared")
        self.assertNotEqual(tcs.secret_provider_for("shared", "mcp_stdio"), tcs.secret_provider_for("shared", "mcp_http"))

    def test_stdio_reuse_needs_the_same_command(self):
        with allow():
            self.assertEqual(client.post(f"{self.base}/import/mcp", json=stdio_body()).status_code, 200)
            r = client.post(f"{self.base}/import/mcp", json=stdio_body(command=[PY, FAKE, "--other"], env={"API_TOKEN": None}))
            self.assertEqual(r.status_code, 422, r.text)
            self.assertEqual(r.json()["metadata"]["errors"][0]["code"], "missing_secret")
            self.assertEqual(client.post(f"{self.base}/import/mcp", json=stdio_body(env={"API_TOKEN": None})).status_code, 200)

    def test_secret_too_long_is_422_before_any_connection(self):
        seen: list = []
        with mock.patch.object(mcp_client.httpx, "AsyncClient", http_mock(seen)):
            r = self.http("http://good.test/mcp", {"Authorization": "x" * 5000})
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["metadata"]["errors"][0]["code"], "secret_too_long")
        self.assertEqual(seen, [])
        self.assertEqual(client.get(f"/api/projects/{self.pid}/secrets").json()["secrets"], [])


class EnvInjectionTests(unittest.TestCase):
    DENIED = ["LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "PATH", "PYTHONPATH", "PYTHONSTARTUP", "NODE_OPTIONS", "NODE_PATH",
              "PERL5OPT", "PERLLIB", "RUBYOPT", "BASH_ENV", "ENV", "HOME", "IFS", "SHELL", "TMPDIR", "GCONV_PATH", "LOCPATH",
              "MALLOC_CHECK_", "GLIBC_TUNABLES", "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "CLASSPATH", "GIT_SSH_COMMAND", "path", "USER", "LANG"]

    def test_service_rejects_loader_and_interpreter_vectors(self):
        for name in self.DENIED:
            with self.assertRaises(tcs.ToolCatalogError, msg=name) as cm:
                tcs.validate_mcp_request({"kind": "mcp_stdio", "command": ["server"], "env": {name: "x"}})
            self.assertEqual(cm.exception.errors[0]["code"], "env_name_denied")
        out = tcs.validate_mcp_request({"kind": "mcp_stdio", "command": ["server"], "env": {"API_TOKEN": "x", "MY_SERVER_MODE": "a"}})
        self.assertEqual(sorted(out["values"]), ["API_TOKEN", "MY_SERVER_MODE"])

    def test_api_is_422_and_nothing_starts(self):
        pid = new_project()
        with allow():
            r = client.post(f"/api/projects/{pid}/tool-catalog/import/mcp", json=stdio_body(env={"LD_PRELOAD": "/tmp/x.so"}))
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["metadata"]["errors"][0]["code"], "env_name_denied")

    def test_client_refuses_denied_names_as_defence_in_depth(self):
        for name in ("LD_PRELOAD", "PYTHONPATH", "PATH", "HOME"):
            with allow():
                with self.assertRaises(mcp_client.McpError, msg=name) as cm:
                    asyncio.run(mcp_client.list_tools_stdio([PY, FAKE], {name: "/tmp"}))
            self.assertIn("not allowed", str(cm.exception))


class HardeningTests(unittest.TestCase):
    def test_unknown_destructiveness_fails_closed(self):
        rc = {"tool_catalog": {}}
        tools, _ = tcs.normalize_tools([
            {"name": "a_ro_false", "annotations": {"readOnlyHint": False}},
            {"name": "b_ro_false_destr_false", "annotations": {"readOnlyHint": False, "destructiveHint": False}},
            {"name": "c_no_ann"},
            {"name": "d_ro_true", "annotations": {"readOnlyHint": True}},
        ], strict=True)
        project = types.SimpleNamespace(runtime_config={})
        tcs._apply(project, tools, {"kind": "manual"}, None)
        by = {t["name"]: t for t in tcs.effective_tools(project)}
        self.assertTrue(by["a_ro_false"]["destructive"])
        self.assertEqual(tcs.compact_signature(by["a_ro_false"]).rsplit(" ", 1)[1], "[destructive]")
        self.assertFalse(by["b_ro_false_destr_false"]["destructive"] is True)
        self.assertIsNot(by["c_no_ann"]["destructive"], True)
        # cannot be made read-only: neither through the flags API nor with a tampered store
        with self.assertRaises(tcs.ToolCatalogError) as cm:
            tcs.set_tool_flags(project, {"a_ro_false": {"read_only": True}})
        self.assertEqual(cm.exception.errors[0]["code"], "destructive_read_only")
        # a tool with no annotations at all can be marked read-only by the user
        tcs.set_tool_flags(project, {"c_no_ann": {"read_only": True}})
        by = {t["name"]: t for t in tcs.effective_tools(project)}
        self.assertTrue(by["c_no_ann"]["read_only_effective"])
        self.assertFalse(by["a_ro_false"]["read_only_effective"])

    def test_dry_run_refuses_a_fail_closed_destructive_tool(self):
        tools, _ = tcs.normalize_tools([{"name": "wipe", "annotations": {"readOnlyHint": False}}], strict=True)
        project = types.SimpleNamespace(runtime_config={})
        tcs._apply(project, tools, {"kind": "mcp_http", "url": "http://x.test/mcp", "header_names": [], "secret_provider": "mcp_http_x"}, None)
        cfg = dict(project.runtime_config)
        cat = dict(cfg["tool_catalog"])
        cat["flags"] = {"wipe": {"allowed": True, "read_only": True}}  # tampered store
        cfg["tool_catalog"] = cat
        project.runtime_config = cfg
        with mock.patch.object(settings, "MCP_DRY_RUN_ENABLED", True):
            with self.assertRaises(tcs.MutatingToolRefused):
                asyncio.run(tcs.dry_run_call(None, project, "wipe", {}))

    def test_draft07_dependencies_are_translated(self):
        out = tcs.normalize_input_schema({
            "type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "c": {"type": "string"}},
            "dependencies": {"a": ["b"], "c": {"required": ["b"]}},
        })
        self.assertNotIn("dependencies", out)
        self.assertEqual(out["dependentRequired"], {"a": ["b"]})
        self.assertEqual(out["dependentSchemas"], {"c": {"required": ["b"]}})
        tool = tcs.normalize_tool({"name": "t", "inputSchema": {"type": "object", "properties": {"a": {}, "b": {}}, "dependencies": {"a": ["b"]}}})
        self.assertNotIn("schema_error", tool)
        bundle = jsb.validate_bundle({"tool.json": tool["input_schema"]}, "tool.json")
        self.assertTrue(bundle.validate({"a": 1}))  # a without b
        self.assertFalse(bundle.validate({"a": 1, "b": 2}))
        # a property literally named "dependencies" is untouched
        keep = tcs.normalize_input_schema({"type": "object", "properties": {"dependencies": {"type": "array", "items": {"type": "string"}}}})
        self.assertEqual(keep["properties"]["dependencies"]["items"], {"type": "string"})

    def test_deeply_nested_schema_is_a_tool_error_not_a_crash(self):
        deep: dict = {"type": "object"}
        for _ in range(400):
            deep = {"type": "object", "properties": {"a": deep}}
        tool = tcs.normalize_tool({"name": "deep", "inputSchema": deep})
        self.assertIn("nested deeper", tool["schema_error"])
        pid = new_project()
        base = f"/api/projects/{pid}/tool-catalog"
        r = client.put(f"{base}/tools", json={"tools": [{"name": "deep", "inputSchema": deep}]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("nested deeper", r.json()["catalog"]["tools"][0]["schema_error"])
        doc = {"openapi": "3.0.0", "info": {"title": "t"}, "paths": {"/x": {"post": {"operationId": "x", "requestBody": {"content": {"application/json": {"schema": deep}}}}}}}
        r = client.post(f"{base}/import/openapi", json={"content": json.dumps(doc)})
        self.assertIn(r.status_code, (200, 422), r.text)

    def test_catalog_size_cap(self):
        big = [{"name": f"t{i}", "inputSchema": {"type": "object", "properties": {"x": {"type": "string", "enum": [f"{j}" + "v" * 1000 for j in range(1100)]}}}} for i in range(5)]
        with self.assertRaises(tcs.ToolCatalogError) as cm:
            tcs.normalize_tools(big, strict=False)
        self.assertEqual(cm.exception.errors[0]["code"], "catalog_too_large")
        pid = new_project()
        r = client.put(f"/api/projects/{pid}/tool-catalog/tools", json={"tools": big})
        self.assertEqual(r.status_code, 422, r.text)

    def test_catalog_work_runs_off_the_event_loop(self):
        pid = new_project()
        base = f"/api/projects/{pid}/tool-catalog"
        calls: list = []
        real = asyncio.to_thread

        async def spy(fn, *a, **kw):
            calls.append(getattr(fn, "__name__", str(fn)))
            return await real(fn, *a, **kw)

        with mock.patch.object(tcs.asyncio, "to_thread", spy):
            self.assertEqual(client.put(f"{base}/tools", json={"tools": simple_tools("a")}).status_code, 200)
            doc = {"openapi": "3.0.0", "info": {"title": "t"}, "paths": {"/x": {"get": {"operationId": "x"}}}}
            self.assertEqual(client.post(f"{base}/import/openapi", json={"content": json.dumps(doc)}).status_code, 200)
        self.assertIn("normalize_tools", calls)
        self.assertIn("prepare_openapi", calls)

    def test_pack_export_derives_annotations_from_top_level_flags(self):
        import test_domain_pack_content as dp

        dp._stub_ready()
        pid = dp.new_id()
        self.assertEqual(dp.post_zip(client, "/api/domain-packs/import", dp.make_zip(dp.toy_files(pid))).status_code, 200)
        project = client.post(f"/api/domain-packs/{pid}/instantiate", json={"project_name": f"proj-{pid}"}).json()["project"]
        client.put(f"/api/projects/{project['id']}/tool-catalog/tools", json={"tools": [
            {"name": "peek", "read_only": True}, {"name": "drop", "read_only": False, "destructive": True}, {"name": "plain"}]})
        exp = client.get(f"/api/projects/{project['id']}/domain-pack/export")
        import io
        import zipfile

        import yaml

        with zipfile.ZipFile(io.BytesIO(exp.content)) as zf:
            exported = yaml.safe_load(zf.read(next(n for n in zf.namelist() if n.endswith("tool_catalog.yaml"))))
        by = {t["name"]: t for t in exported["tools"]}
        self.assertEqual(by["peek"]["annotations"], {"readOnlyHint": True})
        self.assertEqual(by["drop"]["annotations"], {"readOnlyHint": False, "destructiveHint": True})
        self.assertNotIn("annotations", by["plain"])


class DomainPackTests(unittest.TestCase):
    def test_instantiate_and_export(self):
        import test_domain_pack_content as dp

        dp._stub_ready()
        pid = dp.new_id()
        self.assertEqual(dp.post_zip(client, "/api/domain-packs/import", dp.make_zip(dp.toy_files(pid))).status_code, 200)
        res = client.post(f"/api/domain-packs/{pid}/instantiate", json={"project_name": f"proj-{pid}"})
        self.assertEqual(res.status_code, 201, res.text)
        project = res.json()["project"]
        cat = client.get(f"/api/projects/{project['id']}/tool-catalog").json()["catalog"]
        self.assertEqual([t["name"] for t in cat["tools"]], ["grep", "read"])
        self.assertEqual(cat["source"], {"kind": "domain_pack", "pack_id": pid, "version": "1.0.0"})
        # edit the catalog, then export reads from the live catalog
        client.put(f"/api/projects/{project['id']}/tool-catalog/tools", json={"tools": [{"name": "fetch_doc", "description": "Get a doc", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}}]})
        exp = client.get(f"/api/projects/{project['id']}/domain-pack/export")
        self.assertEqual(exp.status_code, 200)
        import io
        import zipfile

        import yaml

        with zipfile.ZipFile(io.BytesIO(exp.content)) as zf:
            name = next(n for n in zf.namelist() if n.endswith("tool_catalog.yaml"))
            exported = yaml.safe_load(zf.read(name))
        self.assertEqual([t["name"] for t in exported["tools"]], ["fetch_doc"])
        self.assertIn("inputSchema", exported["tools"][0])

    def test_pack_tool_checks(self):
        errors, warnings = tcs.check_pack_tools({"tools": [{"id": "a"}, {"name": "a"}, {"id": "b", "inputSchema": {"type": "object", "properties": {"x": {"$ref": "#/$defs/zz"}}}}, {"description": "x"}]})
        self.assertEqual([e["code"] for e in errors], ["duplicate_id"])
        self.assertEqual(sorted(w["code"] for w in warnings), ["tool_schema_error", "tool_without_name"])


if __name__ == "__main__":
    unittest.main()

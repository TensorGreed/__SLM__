"""brewslm CLI: a configured backend URL and a saved login are honoured.

Fresh-clone test (2026-11-13): on a non-default port the CLI kept talking to
:8000, commands spelled with the full ``/api/...`` path went to
``/api/api/...``, a token saved by ``auth login --save`` was never read back,
and a 401 gave no hint to log in.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import httpx


def _load_cli() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "brewslm.py"
    spec = importlib.util.spec_from_file_location("brewslm_cli_api_base", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


class ApiBaseTests(unittest.TestCase):
    def test_host_port_gets_api_appended(self):
        self.assertEqual(cli.normalize_api_base("http://localhost:8100"), "http://localhost:8100/api")
        self.assertEqual(cli.normalize_api_base("http://localhost:8100/"), "http://localhost:8100/api")
        self.assertEqual(cli.normalize_api_base("http://localhost:8100/api/"), "http://localhost:8100/api")
        self.assertEqual(cli.normalize_api_base("http://proxy/brewslm/api"), "http://proxy/brewslm/api")
        self.assertEqual(cli.normalize_api_base(""), "")

    def _urls(self, base: str) -> tuple[str, str]:
        client = cli.ApiClient(api_base=base, token="", timeout_seconds=5)
        try:
            full = str(client._client.build_request("GET", client._path("/api/auth/me")).url)
            short = str(client._client.build_request("GET", client._path("/projects/1/runtime/readiness")).url)
        finally:
            client.close()
        return full, short

    def test_full_api_paths_are_not_doubled(self):
        full, short = self._urls("http://127.0.0.1:8100/api")
        self.assertEqual(full, "http://127.0.0.1:8100/api/auth/me")
        self.assertEqual(short, "http://127.0.0.1:8100/api/projects/1/runtime/readiness")

    def test_bare_host_port_reaches_the_same_urls(self):
        self.assertEqual(
            self._urls("http://127.0.0.1:8100"),
            ("http://127.0.0.1:8100/api/auth/me", "http://127.0.0.1:8100/api/projects/1/runtime/readiness"),
        )

    def test_flag_overrides_default(self):
        args = cli.build_parser().parse_args(["--api-base", "http://127.0.0.1:8100", "doctor", "--project", "1"])
        self.assertEqual(args.api_base, "http://127.0.0.1:8100")

    def test_request_uses_the_configured_base(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if request.url.path.endswith("/readiness"):
                return httpx.Response(401, json={"detail": "Authentication required"})
            return httpx.Response(200, json={"token": "t"})

        client = cli.ApiClient(api_base="http://127.0.0.1:8100", token="", timeout_seconds=5)
        client._client = httpx.Client(base_url=client._client.base_url, transport=httpx.MockTransport(handler))
        try:
            self.assertEqual(client.request("POST", "/api/auth/local/login", json_body={}), {"token": "t"})
            with self.assertRaises(RuntimeError) as ctx:
                client.request("GET", "/projects/1/runtime/readiness")
        finally:
            client.close()
        self.assertEqual(seen, [
            "http://127.0.0.1:8100/api/auth/local/login",
            "http://127.0.0.1:8100/api/projects/1/runtime/readiness",
        ])
        self.assertIn("(401)", str(ctx.exception))
        self.assertIn("auth login --username admin --save", str(ctx.exception))


class TokenTests(unittest.TestCase):
    def test_explicit_token_wins(self):
        self.assertEqual(cli.resolve_token("  abc  ", token_file="/nonexistent/token"), "abc")

    def test_saved_token_file_is_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            path.write_text("saved-jwt\n", encoding="utf-8")
            self.assertEqual(cli.resolve_token("", token_file=str(path)), "saved-jwt")
            self.assertEqual(cli.resolve_token("", token_file=str(Path(tmp) / "missing")), "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

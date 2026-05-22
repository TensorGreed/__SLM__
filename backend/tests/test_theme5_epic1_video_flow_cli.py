"""Theme 5 Epic 1 — brewslm CLI gap-fill commands for the 11-video flow.

Covers the four new top-level command groups:

- ``brewslm auth login``   → POST /api/auth/local/login
- ``brewslm auth whoami``  → GET  /api/auth/me
- ``brewslm template list``        → GET  /api/project-templates
- ``brewslm template get <slug>``  → GET  /api/project-templates/{slug}
- ``brewslm template instantiate`` → POST /api/project-templates/{slug}/instantiate
- ``brewslm serve plan``   → POST /api/projects/{pid}/export/{eid}/serve-plan
- ``brewslm serve start``  → POST /api/projects/{pid}/export/{eid}/serve-runs/start
- ``brewslm serve get``    → GET  /api/projects/{pid}/export/serve-runs/{run_id}
- ``brewslm serve stop``   → POST /api/projects/{pid}/export/serve-runs/{run_id}/stop
- ``brewslm version``      → printed locally; ``--remote`` hits GET /api/health
- ``brewslm --version``    → top-level version flag exits 0
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "brewslm.py"
    spec = importlib.util.spec_from_file_location("brewslm_cli_theme5e1", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeBaseUrl:
    def __init__(self, value: str):
        self._value = value

    def __str__(self) -> str:
        return self._value


class _FakeInnerClient:
    def __init__(self, base_url: str):
        self.base_url = _FakeBaseUrl(base_url)


class _FakeClient:
    def __init__(self, handler: Callable[[dict[str, Any]], Any] | None = None, *, base_url: str = "http://127.0.0.1:8000/api"):
        self._handler = handler or (lambda _call: {})
        self.calls: list[dict[str, Any]] = []
        # Mimic the real ApiClient surface — run_version reads
        # client._client.base_url.
        self._client = _FakeInnerClient(base_url)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ):
        call = {
            "method": method.upper(),
            "path": path,
            "json_body": dict(json_body or {}) if json_body is not None else None,
            "params": dict(params or {}) if params is not None else None,
        }
        self.calls.append(call)
        return self._handler(call)

    def close(self) -> None:  # pragma: no cover
        return None


class Theme5Epic1AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def test_login_posts_credentials_and_prints_bare_token_by_default(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "auth", "login",
                "--username", "admin",
                "--password", "letmein",
            ]
        )
        client = _FakeClient(lambda _call: {"token": "tok-xyz"})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls, [
            {
                "method": "POST",
                "path": "/api/auth/local/login",
                "json_body": {"username": "admin", "password": "letmein"},
                "params": None,
            }
        ])
        self.assertEqual(buf.getvalue().strip(), "tok-xyz")

    def test_login_json_mode_emits_full_response(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["auth", "login", "--username", "admin", "--password", "x", "--json"]
        )
        client = _FakeClient(lambda _call: {"token": "tok-1", "extra": "value"})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertIn('"token"', buf.getvalue())
        self.assertIn('"extra"', buf.getvalue())

    def test_login_save_writes_token_file_with_tight_perms(self):
        parser = self.cli.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "token"
            args = parser.parse_args(
                [
                    "auth", "login",
                    "--username", "admin",
                    "--password", "x",
                    "--save",
                    "--token-file", str(target),
                ]
            )
            client = _FakeClient(lambda _call: {"token": "secret-token"})
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = args.func(args, client)
            self.assertEqual(rc, 0)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text().strip(), "secret-token")
            # POSIX: perms should be 0600.
            if os.name == "posix":
                mode = target.stat().st_mode & 0o777
                self.assertEqual(mode, 0o600)
            # Confirmation message goes to stderr, not stdout — stdout
            # must stay clean for pipe-into-eval flows.
            self.assertIn(str(target), err.getvalue())

    def test_login_password_from_env_when_flag_absent(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["auth", "login", "--username", "admin"]
        )
        client = _FakeClient(lambda _call: {"token": "tok-env"})
        prev = os.environ.get("BREWSLM_PASSWORD")
        os.environ["BREWSLM_PASSWORD"] = "env-secret"
        try:
            with redirect_stdout(io.StringIO()):
                args.func(args, client)
        finally:
            if prev is None:
                del os.environ["BREWSLM_PASSWORD"]
            else:
                os.environ["BREWSLM_PASSWORD"] = prev
        self.assertEqual(
            client.calls[0]["json_body"],
            {"username": "admin", "password": "env-secret"},
        )

    def test_login_raises_when_token_missing_from_response(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["auth", "login", "--username", "admin", "--password", "x"]
        )
        client = _FakeClient(lambda _call: {"error": "wrong password"})
        with self.assertRaisesRegex(RuntimeError, "did not include a token"):
            args.func(args, client)

    def test_whoami_calls_me_and_renders_principal(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["auth", "whoami"])
        client = _FakeClient(lambda _call: {
            "auth_enabled": True,
            "principal": {
                "user_id": 7,
                "username": "admin",
                "role": "admin",
                "api_key_prefix": "abc12",
            },
            "memberships": [
                {"project_id": 1, "role": "owner"},
                {"project_id": 2, "role": "engineer"},
            ],
        })
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls[0], {
            "method": "GET",
            "path": "/api/auth/me",
            "json_body": None,
            "params": None,
        })
        text = buf.getvalue()
        self.assertIn("admin", text)
        self.assertIn("project 1: owner", text)
        self.assertIn("project 2: engineer", text)
        self.assertIn("abc12", text)

    def test_whoami_handles_auth_disabled_message(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["auth", "whoami"])
        client = _FakeClient(lambda _call: {
            "auth_enabled": False,
            "principal": None,
            "memberships": [],
        })
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertIn("Auth disabled", buf.getvalue())


class Theme5Epic1TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def test_template_list_gets_catalog_and_renders_table(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["template", "list"])
        client = _FakeClient(lambda _call: {
            "templates": [
                {
                    "slug": "ticket-router",
                    "name": "Ticket Router SLM",
                    "task_profile": "classification",
                    "minimum_dataset_size": 200,
                },
                {
                    "slug": "data-to-sql",
                    "name": "Data to SQL",
                    "task_profile": "generic_sft",
                    "minimum_dataset_size": 200,
                },
            ],
        })
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls[0]["method"], "GET")
        self.assertEqual(client.calls[0]["path"], "/api/project-templates")
        text = buf.getvalue()
        self.assertIn("ticket-router", text)
        self.assertIn("Ticket Router SLM", text)
        self.assertIn("data-to-sql", text)

    def test_template_list_empty_state_message(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["template", "list"])
        client = _FakeClient(lambda _call: {"templates": []})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertIn("no templates registered", buf.getvalue())

    def test_template_get_emits_full_detail_json(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["template", "get", "ticket-router"])
        client = _FakeClient(lambda _call: {"slug": "ticket-router", "labels": ["billing"]})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls[0]["path"], "/api/project-templates/ticket-router")
        self.assertIn("ticket-router", buf.getvalue())
        self.assertIn("billing", buf.getvalue())

    def test_template_instantiate_posts_with_optional_name(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["template", "instantiate", "ticket-router", "--name", "Acme Tickets"]
        )
        client = _FakeClient(lambda _call: {
            "id": 42,
            "name": "Acme Tickets",
            "base_model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "selected_recipe": {"recipe_id": "classification"},
        })
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls[0], {
            "method": "POST",
            "path": "/api/project-templates/ticket-router/instantiate",
            "json_body": {"project_name": "Acme Tickets"},
            "params": None,
        })
        text = buf.getvalue()
        self.assertIn("Created project #42", text)
        self.assertIn("classification", text)
        self.assertIn("SmolLM2", text)

    def test_template_instantiate_omits_name_when_not_set(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["template", "instantiate", "log-triage"])
        client = _FakeClient(lambda _call: {"id": 9, "name": "Log Triage SLM"})
        with redirect_stdout(io.StringIO()):
            args.func(args, client)
        # Empty body — server defaults to the template name.
        self.assertEqual(client.calls[0]["json_body"], {})


class Theme5Epic1ServeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def test_serve_plan_posts_with_only_provided_fields(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["serve", "plan", "--project", "3", "--export-id", "11", "--port", "8080"]
        )
        client = _FakeClient(lambda _call: {"templates": []})
        with redirect_stdout(io.StringIO()):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/api/projects/3/export/11/serve-plan")
        # Only port is sent (host + smoke prompt omitted).
        self.assertEqual(call["json_body"], {"port": 8080})

    def test_serve_start_renders_friendly_summary(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "serve", "start",
                "--project", "4",
                "--export-id", "12",
                "--template-id", "ollama_local",
            ]
        )
        client = _FakeClient(lambda _call: {
            "run_id": "srv-abc",
            "status": "starting",
            "host": "127.0.0.1",
            "port": 11434,
        })
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls[0]["path"], "/api/projects/4/export/12/serve-runs/start")
        self.assertEqual(client.calls[0]["json_body"], {"template_id": "ollama_local"})
        self.assertIn("srv-abc", buf.getvalue())
        self.assertIn("127.0.0.1:11434", buf.getvalue())

    def test_serve_get_uses_get_with_logs_tail_param(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["serve", "get", "--project", "4", "--run-id", "srv-abc", "--logs-tail", "50"]
        )
        client = _FakeClient(lambda _call: {"status": "running"})
        with redirect_stdout(io.StringIO()):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "/api/projects/4/export/serve-runs/srv-abc")
        self.assertEqual(call["params"], {"logs_tail": 50})

    def test_serve_stop_posts_to_stop_endpoint(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["serve", "stop", "--project", "4", "--run-id", "srv-abc"]
        )
        client = _FakeClient(lambda _call: {"status": "stopped"})
        with redirect_stdout(io.StringIO()):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(
            client.calls[0]["path"],
            "/api/projects/4/export/serve-runs/srv-abc/stop",
        )
        self.assertEqual(client.calls[0]["method"], "POST")


class Theme5Epic1VersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def test_version_subcommand_prints_cli_version(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["version"])
        client = _FakeClient()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertIn(self.cli.__version__, buf.getvalue())
        # No remote probe → no API call.
        self.assertEqual(client.calls, [])

    def test_version_remote_probes_health(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["version", "--remote"])
        client = _FakeClient(lambda _call: {"version": "9.9.9", "status": "ok"})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls[0]["path"], "/api/health")
        text = buf.getvalue()
        self.assertIn(self.cli.__version__, text)
        self.assertIn("9.9.9", text)

    def test_version_remote_swallows_backend_error(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["version", "--remote"])

        def handler(_call):
            raise RuntimeError("connection refused")

        client = _FakeClient(handler)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = args.func(args, client)
        # Backend down is not a CLI failure — exit 0, surface message.
        self.assertEqual(rc, 0)
        self.assertIn("backend probe failed", buf.getvalue())

    def test_top_level_version_flag_exits_zero(self):
        parser = self.cli.build_parser()
        with self.assertRaises(SystemExit) as cm:
            with redirect_stdout(io.StringIO()):
                parser.parse_args(["--version"])
        self.assertEqual(cm.exception.code, 0)


class Theme5Epic1ParserShapeTests(unittest.TestCase):
    """Parser-level guarantees — required subcommands, exit codes."""

    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def test_auth_requires_a_subcommand(self):
        parser = self.cli.build_parser()
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parser.parse_args(["auth"])

    def test_template_requires_a_subcommand(self):
        parser = self.cli.build_parser()
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parser.parse_args(["template"])

    def test_serve_requires_a_subcommand(self):
        parser = self.cli.build_parser()
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parser.parse_args(["serve"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""Phase 86 — scaffold / extensions CLI (priority.md P39, Wave H).

Covers the new ``brewslm scaffold {adapter,runtime,domain-pack,eval-pack}``
and ``brewslm extensions {list,validate,reload}`` subcommands that wire
through to the P37/P38 endpoints.

Each verb is exercised against a fake client that captures every
HTTP call (method + path + body + params), so we assert both the
shape of the request the CLI sends and that the CLI surfaces the
right exit code / output for each response.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "brewslm.py"
    spec = importlib.util.spec_from_file_location("brewslm_cli_p39", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRawClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.next_response: SimpleNamespace = SimpleNamespace(
            status_code=200, content=b"", text=""
        )

    def request(self, method: str, path: str, *, params=None, **_: Any):
        self.calls.append({"method": method.upper(), "path": path, "params": params})
        return self.next_response


class _FakeClient:
    def __init__(self, handler: Callable[[dict[str, Any]], Any] | None = None):
        self._handler = handler or (lambda _call: {})
        self.calls: list[dict[str, Any]] = []
        self._client = _FakeRawClient()

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
            "json_body": dict(json_body) if json_body is not None else None,
            "params": dict(params) if params else None,
        }
        self.calls.append(call)
        return self._handler(call)

    def close(self) -> None:  # pragma: no cover
        return None


class Phase86ExtensionCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def _parse(self, argv: list[str]):
        return self.cli.build_parser().parse_args(argv)

    def _capture_stdout(
        self, argv: list[str], handler: Callable[[dict], Any] | None = None
    ) -> tuple[int, _FakeClient, str]:
        args = self._parse(argv)
        client = _FakeClient(handler)
        buffer = io.StringIO()
        original = sys.stdout
        sys.stdout = buffer
        try:
            rc = args.func(args, client)
        finally:
            sys.stdout = original
        return rc, client, buffer.getvalue()

    # ------------------------------------------------------------------
    # scaffold
    # ------------------------------------------------------------------

    def test_scaffold_adapter_posts_normalised_body(self):
        captured: dict[str, Any] = {}

        def handler(call: dict[str, Any]) -> Any:
            captured.update(call)
            return {
                "kind": "data_adapter",
                "plugin_id": "phase86-cli",
                "module_basename": "phase86_cli",
                "contract_version": "slm.data_adapter/v3",
                "files": {"phase86_cli.py": "stub"},
                "written_files": ["/tmp/phase86_cli.py"],
                "output_dir": "/tmp",
            }

        rc, _client, out = self._capture_stdout(
            [
                "scaffold",
                "adapter",
                "--plugin-id",
                "phase86-cli",
                "--description",
                "Phase 86 stub",
                "--no-write",
            ],
            handler,
        )
        self.assertEqual(rc, 0, out)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/extensions/scaffold")
        body = captured["json_body"]
        # The CLI translates "adapter" → "data_adapter" before sending.
        self.assertEqual(body["kind"], "data_adapter")
        self.assertEqual(body["plugin_id"], "phase86-cli")
        self.assertEqual(body["description"], "Phase 86 stub")
        # --no-write forces write=False, every other optional flag stays
        # absent rather than being sent as None.
        self.assertFalse(body["write"])
        self.assertNotIn("display_name", body)
        self.assertNotIn("author", body)

    def test_scaffold_runtime_alias_resolves(self):
        captured: dict[str, Any] = {}

        def handler(call: dict[str, Any]) -> Any:
            captured.update(call)
            return {"kind": "training_runtime"}

        rc, _client, _ = self._capture_stdout(
            ["scaffold", "runtime", "--plugin-id", "rt-id"], handler
        )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["json_body"]["kind"], "training_runtime")

    def test_scaffold_domain_pack_and_eval_pack_aliases(self):
        kinds_seen: list[str] = []

        def handler(call: dict[str, Any]) -> Any:
            kinds_seen.append(call["json_body"]["kind"])
            return {}

        self._capture_stdout(
            ["scaffold", "domain-pack", "--plugin-id", "dp"], handler
        )
        self._capture_stdout(
            ["scaffold", "eval-pack", "--plugin-id", "ep"], handler
        )
        self.assertEqual(kinds_seen, ["domain_pack", "eval_pack"])

    def test_scaffold_optional_flags_are_forwarded(self):
        captured: dict[str, Any] = {}

        def handler(call: dict[str, Any]) -> Any:
            captured.update(call)
            return {}

        self._capture_stdout(
            [
                "scaffold",
                "adapter",
                "--plugin-id",
                "full-options",
                "--display-name",
                "Full Options",
                "--description",
                "All optional flags exercised.",
                "--author",
                "phase86",
                "--version",
                "1.2.3",
                "--export-dir",
                "/tmp/phase86-scaffold",
            ],
            handler,
        )
        body = captured["json_body"]
        self.assertEqual(body["display_name"], "Full Options")
        self.assertEqual(body["author"], "phase86")
        self.assertEqual(body["version"], "1.2.3")
        self.assertEqual(body["export_dir"], "/tmp/phase86-scaffold")
        # write defaults to True so the body should omit the override.
        self.assertNotIn("write", body)

    # ------------------------------------------------------------------
    # extensions list / validate / reload
    # ------------------------------------------------------------------

    def test_extensions_list_hits_get_extensions(self):
        captured: dict[str, Any] = {}

        def handler(call: dict[str, Any]) -> Any:
            captured.update(call)
            return {"kinds": [{"kind": "data_adapter"}]}

        rc, _client, out = self._capture_stdout(["extensions", "list"], handler)
        self.assertEqual(rc, 0, out)
        self.assertEqual(captured["method"], "GET")
        self.assertEqual(captured["path"], "/extensions")
        # Body should not be sent on GET.
        self.assertIsNone(captured["json_body"])

    def test_extensions_validate_ok_exits_zero(self):
        def handler(call: dict[str, Any]) -> Any:
            self.assertEqual(call["path"], "/extensions/validate")
            self.assertEqual(call["json_body"]["kind"], "data_adapter")
            self.assertEqual(call["json_body"]["module"], "phase86.module.path")
            return {"ok": True, "checks": [], "kind": "data_adapter"}

        rc, _client, _ = self._capture_stdout(
            [
                "extensions",
                "validate",
                "--kind",
                "adapter",
                "--module",
                "phase86.module.path",
            ],
            handler,
        )
        self.assertEqual(rc, 0)

    def test_extensions_validate_failure_exits_nonzero(self):
        def handler(call: dict[str, Any]) -> Any:
            return {"ok": False, "checks": [], "kind": "data_adapter"}

        rc, _client, _ = self._capture_stdout(
            [
                "extensions",
                "validate",
                "--kind",
                "adapter",
                "--module",
                "phase86.bad.module",
            ],
            handler,
        )
        self.assertEqual(rc, 1)

    def test_extensions_validate_force_reload_propagates(self):
        captured: dict[str, Any] = {}

        def handler(call: dict[str, Any]) -> Any:
            captured.update(call)
            return {"ok": True}

        self._capture_stdout(
            [
                "extensions",
                "validate",
                "--kind",
                "runtime",
                "--module",
                "phase86.module",
                "--force-reload",
            ],
            handler,
        )
        self.assertTrue(captured["json_body"]["force_reload"])

    def test_extensions_reload_all_kinds_sends_empty_body(self):
        captured: dict[str, Any] = {}

        def handler(call: dict[str, Any]) -> Any:
            captured.update(call)
            return {
                "results": [
                    {"kind": "data_adapter", "status": "ok"},
                    {"kind": "training_runtime", "status": "ok"},
                    {"kind": "domain_pack", "status": "not_supported"},
                    {"kind": "eval_pack", "status": "not_supported"},
                ]
            }

        rc, _client, _ = self._capture_stdout(
            ["extensions", "reload"], handler
        )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["path"], "/extensions/reload")
        self.assertEqual(captured["json_body"], {})

    def test_extensions_reload_single_kind_resolves_alias(self):
        captured: dict[str, Any] = {}

        def handler(call: dict[str, Any]) -> Any:
            captured.update(call)
            return {"results": [{"kind": "training_runtime", "status": "ok"}]}

        rc, _client, _ = self._capture_stdout(
            ["extensions", "reload", "--kind", "runtime"], handler
        )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["json_body"]["kind"], "training_runtime")

    def test_extensions_reload_partial_failure_exits_nonzero(self):
        def handler(call: dict[str, Any]) -> Any:
            return {
                "results": [
                    {"kind": "data_adapter", "status": "ok"},
                    {
                        "kind": "training_runtime",
                        "status": "partial",
                        "failed_modules": {"x": "boom"},
                    },
                ]
            }

        rc, _client, _ = self._capture_stdout(
            ["extensions", "reload"], handler
        )
        self.assertEqual(rc, 1)

    def test_unknown_scaffold_alias_raises_value_error(self):
        # The CLI module exposes the helper; resolving an unknown alias
        # raises before any HTTP call so we never even hit the API.
        with self.assertRaises(ValueError) as cm:
            self.cli._resolve_scaffold_kind("widget")
        self.assertIn("Unsupported scaffold alias", str(cm.exception))

    def test_scaffold_output_is_pretty_json(self):
        def handler(call: dict[str, Any]) -> Any:
            return {"kind": "data_adapter", "ok": True}

        _, _, out = self._capture_stdout(
            ["scaffold", "adapter", "--plugin-id", "p", "--no-write"],
            handler,
        )
        # _print_json formats with indent=2 by convention; sanity-check
        # via round-trip rather than asserting whitespace exactly.
        parsed = json.loads(out)
        self.assertEqual(parsed, {"kind": "data_adapter", "ok": True})


if __name__ == "__main__":
    unittest.main()

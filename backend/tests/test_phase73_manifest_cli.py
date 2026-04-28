"""Phase 73 — brewslm CLI manifest + pipeline subcommands (priority.md P23).

Covers the new ``brewslm manifest`` subparser wiring through to the P21+P22
endpoints, plus ``brewslm pipeline run`` orchestrating apply + autopilot
one-click:

- ``manifest export``    → GET  /projects/{id}/manifest/export?format=yaml|json (P21)
- ``manifest validate``  → POST /manifest/validate                              (P22)
- ``manifest diff``      → POST /projects/{id}/manifest/diff                    (P22)
- ``manifest apply``     → POST /projects/{id}/manifest/apply                   (P22)
                          or POST /manifest/apply when --project is omitted
- ``pipeline run``       → manifest apply + POST /projects/{pid}/training/autopilot/one-click-run

Also asserts that:
- ``manifest export`` writes the body to stdout when ``--out`` is omitted and
  to the supplied path when given.
- ``pipeline run --no-train`` stops after the apply (no training call captured).
- Reading a non-existent manifest file raises a clear ``ValueError``.
- ``manifest`` and ``pipeline`` parents both ``SystemExit`` when invoked
  without a subcommand (mirror the ``repro`` parent behaviour from phase70).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "brewslm.py"
    spec = importlib.util.spec_from_file_location("brewslm_cli_p23", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MINIMAL_MANIFEST_YAML = """\
api_version: brewslm/v1
kind: Project
metadata:
  name: phase73-cli-test
  description: phase73 cli fixture
spec:
  workflow:
    target_profile_id: vllm_server
"""


class _FakeClient:
    def __init__(self, handler: Callable[[dict[str, Any]], Any] | None = None):
        self._handler = handler or (lambda _call: {})
        self.calls: list[dict[str, Any]] = []

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


class Phase73ManifestCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def _write_manifest(self, body: str | None = None) -> str:
        text = body if body is not None else _MINIMAL_MANIFEST_YAML
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        handle.write(text)
        handle.close()
        self.addCleanup(lambda p=handle.name: Path(p).unlink(missing_ok=True))
        return handle.name

    # -- manifest export --------------------------------------------------

    def test_manifest_export_yaml_writes_body_to_stdout(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["manifest", "export", "--project", "7"]
        )
        # YAML responses come back from ApiClient as {"raw": "<yaml>"}.
        client = _FakeClient(lambda _call: {"raw": "api_version: brewslm/v1\n"})
        captured = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured
        try:
            rc = args.func(args, client)
        finally:
            sys.stdout = original_stdout
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "/projects/7/manifest/export")
        self.assertEqual(call["params"], {"format": "yaml"})
        self.assertIn("api_version: brewslm/v1", captured.getvalue())

    def test_manifest_export_yaml_writes_to_out_path(self):
        parser = self.cli.build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "manifest.yaml")
            args = parser.parse_args(
                [
                    "manifest",
                    "export",
                    "--project",
                    "7",
                    "--out",
                    out_path,
                ]
            )
            yaml_body = "api_version: brewslm/v1\nkind: Project\n"
            client = _FakeClient(lambda _call: {"raw": yaml_body})
            rc = args.func(args, client)
            self.assertEqual(rc, 0)
            written = Path(out_path).read_text(encoding="utf-8")
            self.assertEqual(written, yaml_body)
            # The path is still passed to the GET endpoint.
            self.assertEqual(
                client.calls[0]["path"], "/projects/7/manifest/export"
            )
            self.assertEqual(client.calls[0]["params"], {"format": "yaml"})

    def test_manifest_export_json_format_passes_format_param(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "manifest",
                "export",
                "--project",
                "9",
                "--format",
                "json",
            ]
        )
        client = _FakeClient(
            lambda _call: {"api_version": "brewslm/v1", "kind": "Project"}
        )
        captured = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured
        try:
            rc = args.func(args, client)
        finally:
            sys.stdout = original_stdout
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "/projects/9/manifest/export")
        self.assertEqual(call["params"], {"format": "json"})

    # -- manifest validate -----------------------------------------------

    def test_manifest_validate_posts_yaml_body(self):
        parser = self.cli.build_parser()
        manifest_path = self._write_manifest()
        args = parser.parse_args(
            ["manifest", "validate", manifest_path]
        )
        client = _FakeClient(lambda _call: {"ok": True, "errors": [], "warnings": []})
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/manifest/validate")
        # Body is the file contents under the manifest_yaml key.
        self.assertIn("manifest_yaml", call["json_body"])
        self.assertIn("api_version: brewslm/v1", call["json_body"]["manifest_yaml"])

    def test_manifest_validate_returns_nonzero_on_validation_failure(self):
        parser = self.cli.build_parser()
        manifest_path = self._write_manifest()
        args = parser.parse_args(
            ["manifest", "validate", manifest_path]
        )
        client = _FakeClient(
            lambda _call: {
                "ok": False,
                "errors": [
                    {
                        "code": "UNKNOWN_TARGET_PROFILE",
                        "severity": "error",
                        "field": "spec.workflow.target_profile_id",
                        "message": "...",
                        "actionable_fix": "...",
                    }
                ],
                "warnings": [],
            }
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 1)

    # -- manifest diff ---------------------------------------------------

    def test_manifest_diff_posts_to_project_diff(self):
        parser = self.cli.build_parser()
        manifest_path = self._write_manifest()
        args = parser.parse_args(
            ["manifest", "diff", manifest_path, "--project", "11"]
        )
        client = _FakeClient(
            lambda _call: {"actions": [], "warnings": []}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/projects/11/manifest/diff")
        self.assertIn("manifest_yaml", call["json_body"])
        self.assertIn("api_version: brewslm/v1", call["json_body"]["manifest_yaml"])

    # -- manifest apply --------------------------------------------------

    def test_manifest_apply_project_scoped_writes(self):
        parser = self.cli.build_parser()
        manifest_path = self._write_manifest()
        args = parser.parse_args(
            ["manifest", "apply", manifest_path, "--project", "5"]
        )
        client = _FakeClient(
            lambda _call: {"project_id": 5, "plan_only": False, "applied_actions": []}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/projects/5/manifest/apply")
        self.assertIn("manifest_yaml", call["json_body"])
        self.assertEqual(call["json_body"]["plan_only"], False)

    def test_manifest_apply_plan_only_sets_flag(self):
        parser = self.cli.build_parser()
        manifest_path = self._write_manifest()
        args = parser.parse_args(
            [
                "manifest",
                "apply",
                manifest_path,
                "--project",
                "5",
                "--plan-only",
            ]
        )
        client = _FakeClient(
            lambda _call: {"project_id": 5, "plan_only": True, "applied_actions": []}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["path"], "/projects/5/manifest/apply")
        self.assertEqual(call["json_body"]["plan_only"], True)

    def test_manifest_apply_without_project_hits_top_level_endpoint(self):
        parser = self.cli.build_parser()
        manifest_path = self._write_manifest()
        args = parser.parse_args(
            ["manifest", "apply", manifest_path]
        )
        client = _FakeClient(
            lambda _call: {"project_id": 42, "plan_only": False, "applied_actions": []}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        # No --project → top-level new-project endpoint.
        self.assertEqual(call["path"], "/manifest/apply")
        self.assertIn("manifest_yaml", call["json_body"])
        self.assertEqual(call["json_body"]["plan_only"], False)

    # -- pipeline run ----------------------------------------------------

    def test_pipeline_run_chains_apply_then_one_click(self):
        parser = self.cli.build_parser()
        manifest_path = self._write_manifest()
        args = parser.parse_args(
            ["pipeline", "run", manifest_path, "--project", "8"]
        )

        def _handler(call: dict[str, Any]) -> Any:
            if call["path"].endswith("/manifest/apply"):
                return {
                    "project_id": 8,
                    "plan_only": False,
                    "applied_actions": [
                        {"kind": "project", "action": "update", "name": "phase73-cli-test"}
                    ],
                }
            if call["path"].endswith("/training/autopilot/one-click-run"):
                return {"started": True, "experiment_id": 99}
            return {}

        client = _FakeClient(_handler)
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.calls), 2)
        apply_call, train_call = client.calls
        self.assertEqual(apply_call["method"], "POST")
        self.assertEqual(apply_call["path"], "/projects/8/manifest/apply")
        self.assertEqual(apply_call["json_body"]["plan_only"], False)
        self.assertEqual(train_call["method"], "POST")
        self.assertEqual(
            train_call["path"], "/projects/8/training/autopilot/one-click-run"
        )
        # Body shape per task spec: {"intent": ..., "target_profile_id": ...}.
        self.assertEqual(train_call["json_body"]["intent"], "Apply manifest")
        self.assertEqual(train_call["json_body"]["target_profile_id"], "vllm_server")

    def test_pipeline_run_no_train_skips_training_call(self):
        parser = self.cli.build_parser()
        manifest_path = self._write_manifest()
        args = parser.parse_args(
            ["pipeline", "run", manifest_path, "--project", "8", "--no-train"]
        )

        def _handler(call: dict[str, Any]) -> Any:
            if call["path"].endswith("/manifest/apply"):
                return {
                    "project_id": 8,
                    "plan_only": False,
                    "applied_actions": [],
                }
            return {}

        client = _FakeClient(_handler)
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        # Only the apply call — no autopilot kick-off.
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["path"], "/projects/8/manifest/apply")

    # -- file-not-found behaviour ---------------------------------------

    def test_validate_with_missing_file_raises_value_error(self):
        parser = self.cli.build_parser()
        bogus = "/tmp/__phase73_does_not_exist__.yaml"
        # Defensive: ensure we genuinely have no file at this path.
        Path(bogus).unlink(missing_ok=True)
        args = parser.parse_args(["manifest", "validate", bogus])
        with self.assertRaises(ValueError):
            args.func(args, _FakeClient())

    # -- subparsers required --------------------------------------------

    def test_manifest_without_subcommand_exits(self):
        parser = self.cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["manifest"])

    def test_pipeline_without_subcommand_exits(self):
        parser = self.cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["pipeline"])


if __name__ == "__main__":
    unittest.main()

"""Phase 83 — observability CLI (priority.md P35, Wave G).

Covers the new ``brewslm doctor --deep`` / ``brewslm logs tail`` /
``brewslm support-bundle {create,list,download}`` subcommands wired
through to the P31/P32/P33/P34 endpoints.

Each verb is exercised against a ``_FakeClient`` that captures both
the high-level ``request()`` calls (json) AND the low-level
``_client.request()`` calls used by the binary download paths.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "brewslm.py"
    spec = importlib.util.spec_from_file_location(
        "brewslm_cli_p35", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRawClient:
    """Stand-in for httpx — intercepts the download path used by the
    support-bundle CLI's binary fetch."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.next_response: SimpleNamespace = SimpleNamespace(
            status_code=200, content=b"", text=""
        )

    def request(self, method: str, path: str, *, params=None, **_: Any):
        self.calls.append(
            {
                "method": method.upper(),
                "path": path,
                "params": dict(params) if params else None,
            }
        )
        return self.next_response


class _FakeClient:
    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Any] | None = None,
    ):
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
            "json_body": (
                dict(json_body) if json_body is not None else None
            ),
            "params": dict(params) if params else None,
        }
        self.calls.append(call)
        return self._handler(call)

    def close(self) -> None:  # pragma: no cover
        return None


class Phase83ObservabilityCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse(self, argv: list[str]):
        parser = self.cli.build_parser()
        return parser.parse_args(argv)

    def _run(
        self,
        argv: list[str],
        handler: Callable[[dict], Any] | None = None,
    ) -> tuple[int, _FakeClient]:
        args = self._parse(argv)
        client = _FakeClient(handler)
        rc = args.func(args, client)
        return rc, client

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
    # doctor --deep
    # ------------------------------------------------------------------

    def test_doctor_default_only_hits_readiness(self):
        def handler(call: dict[str, Any]) -> Any:
            if call["path"].endswith("/runtime/readiness"):
                return {
                    "status": "pass",
                    "strict_mode": False,
                    "checks": [
                        {
                            "status": "pass",
                            "name": "GPU",
                            "message": "ok",
                            "fix": "",
                        },
                    ],
                }
            raise AssertionError(f"unexpected call {call}")

        rc, client, _ = self._capture_stdout(
            ["doctor", "--project", "7"], handler
        )
        self.assertEqual(rc, 0)
        # Only readiness was fetched — no timeline / cluster fetch.
        paths = [c["path"] for c in client.calls]
        self.assertEqual(paths, ["/projects/7/runtime/readiness"])

    def test_doctor_deep_fetches_timeline_and_clusters(self):
        responses = {
            "/projects/7/runtime/readiness": {
                "status": "pass",
                "strict_mode": False,
                "checks": [],
            },
            "/projects/7/timeline": {
                "total_events": 12,
                "total_runs": 3,
                "orphaned_count": 0,
                "truncated": False,
                "tree": [
                    {
                        "run_id": "exp-42",
                        "highest_severity": "error",
                        "summary": "Training failed: cuda oom",
                    },
                    {
                        "run_id": "exp-41",
                        "highest_severity": "info",
                        "summary": "Training completed",
                    },
                ],
            },
            "/projects/7/failure-clusters": {
                "clusters": [
                    {
                        "stage": "training",
                        "reason_code": "training_runtime_error",
                        "signature": "abcdef123456",
                        "failure_count": 4,
                        "last_seen_at": "2026-05-07T00:00:00Z",
                        "exemplar_summaries": ["cuda oom at step 1"],
                    }
                ]
            },
        }

        def handler(call: dict[str, Any]) -> Any:
            return responses[call["path"]]

        rc, client, output = self._capture_stdout(
            ["doctor", "--project", "7", "--deep"], handler
        )
        self.assertEqual(rc, 0)
        paths = [c["path"] for c in client.calls]
        self.assertEqual(
            paths,
            [
                "/projects/7/runtime/readiness",
                "/projects/7/timeline",
                "/projects/7/failure-clusters",
            ],
        )
        self.assertIn("Deep observability", output)
        self.assertIn("most-severe recent run: exp-42", output)
        self.assertIn("training::training_runtime_error", output)

    def test_doctor_deep_json_returns_structured_payload(self):
        responses = {
            "/projects/7/runtime/readiness": {
                "status": "pass",
                "strict_mode": False,
                "checks": [],
            },
            "/projects/7/timeline": {
                "total_events": 0,
                "total_runs": 0,
                "tree": [],
            },
            "/projects/7/failure-clusters": {"clusters": []},
        }

        def handler(call: dict[str, Any]) -> Any:
            return responses[call["path"]]

        rc, _, output = self._capture_stdout(
            ["doctor", "--project", "7", "--deep", "--json"], handler
        )
        self.assertEqual(rc, 0)
        body = json.loads(output)
        self.assertIn("readiness", body)
        self.assertIn("timeline", body)
        self.assertIn("failure_clusters", body)

    def test_doctor_deep_passes_since_and_limit_to_timeline(self):
        captured: list[dict[str, Any]] = []

        def handler(call: dict[str, Any]) -> Any:
            captured.append(call)
            if call["path"].endswith("/runtime/readiness"):
                return {"status": "pass", "strict_mode": False, "checks": []}
            return {"tree": [], "total_events": 0, "total_runs": 0, "clusters": []}

        rc, _, _ = self._capture_stdout(
            [
                "doctor",
                "--project",
                "9",
                "--deep",
                "--deep-since",
                "2026-05-01T00:00:00Z",
                "--deep-limit",
                "200",
            ],
            handler,
        )
        self.assertEqual(rc, 0)
        timeline_call = next(
            c for c in captured if c["path"].endswith("/timeline")
        )
        self.assertEqual(timeline_call["params"]["limit"], 200)
        self.assertEqual(
            timeline_call["params"]["since"], "2026-05-01T00:00:00Z"
        )

    def test_doctor_failing_status_returns_rc_1(self):
        def handler(call: dict[str, Any]) -> Any:
            return {
                "status": "fail",
                "strict_mode": True,
                "checks": [
                    {
                        "status": "fail",
                        "name": "GPU",
                        "message": "no cuda",
                        "fix": "install drivers",
                    }
                ],
            }

        rc, _, _ = self._capture_stdout(["doctor", "--project", "7"], handler)
        self.assertEqual(rc, 1)

    # ------------------------------------------------------------------
    # logs tail
    # ------------------------------------------------------------------

    def test_logs_tail_default_path_and_params(self):
        def handler(call: dict[str, Any]) -> Any:
            return {
                "events": [
                    {
                        "ts": "2026-05-07T10:00:00Z",
                        "severity": "info",
                        "stage": "training",
                        "run_id": "exp-1",
                        "summary": "Training started",
                    }
                ]
            }

        rc, client, output = self._capture_stdout(
            ["logs", "tail", "--project", "7"], handler
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "/projects/7/run-events")
        self.assertEqual(call["params"]["limit"], 50)
        self.assertIn("Training started", output)
        self.assertIn("training", output)

    def test_logs_tail_passes_filters(self):
        captured: list[dict[str, Any]] = []

        def handler(call: dict[str, Any]) -> Any:
            captured.append(call)
            return {"events": []}

        rc, _, _ = self._capture_stdout(
            [
                "logs",
                "tail",
                "--project",
                "9",
                "--stage",
                "training",
                "--severity",
                "error",
                "--run-id",
                "exp-42",
                "--since",
                "2026-05-01T00:00:00Z",
                "--limit",
                "10",
            ],
            handler,
        )
        self.assertEqual(rc, 0)
        params = captured[0]["params"]
        self.assertEqual(params["stage"], "training")
        self.assertEqual(params["severity"], "error")
        self.assertEqual(params["run_id"], "exp-42")
        self.assertEqual(params["since"], "2026-05-01T00:00:00Z")
        self.assertEqual(params["limit"], 10)

    def test_logs_tail_json_emits_raw_payload(self):
        def handler(call: dict[str, Any]) -> Any:
            return {"events": [{"ts": "x", "severity": "info"}]}

        rc, _, output = self._capture_stdout(
            ["logs", "tail", "--project", "7", "--json"], handler
        )
        self.assertEqual(rc, 0)
        body = json.loads(output)
        self.assertEqual(len(body["events"]), 1)

    def test_logs_tail_empty_prints_no_match(self):
        def handler(call: dict[str, Any]) -> Any:
            return {"events": []}

        rc, _, output = self._capture_stdout(
            ["logs", "tail", "--project", "7"], handler
        )
        self.assertEqual(rc, 0)
        self.assertIn("(no events match)", output)

    def test_logs_parent_without_subcommand_exits(self):
        with self.assertRaises(SystemExit):
            self.cli.build_parser().parse_args(["logs"])

    # ------------------------------------------------------------------
    # support-bundle
    # ------------------------------------------------------------------

    def test_support_bundle_create_posts_with_actor_and_ttl(self):
        def handler(call: dict[str, Any]) -> Any:
            return {
                "bundle_uid": "abc",
                "download_token": "tok",
                "size_bytes": 1234,
                "section_counts": {"run_events": 3},
            }

        rc, client, _ = self._capture_stdout(
            [
                "support-bundle",
                "create",
                "--project",
                "7",
                "--actor",
                "ops",
                "--ttl-seconds",
                "3600",
            ],
            handler,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/projects/7/support-bundle")
        self.assertEqual(
            call["json_body"], {"actor": "ops", "ttl_seconds": 3600}
        )

    def test_support_bundle_create_with_download_streams_zip(self):
        def handler(call: dict[str, Any]) -> Any:
            return {
                "bundle_uid": "abc1234567890",
                "download_token": "tok",
                "size_bytes": 9,
                "section_counts": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "bundle.zip"
            args = self._parse(
                [
                    "support-bundle",
                    "create",
                    "--project",
                    "7",
                    "--download",
                    "--out",
                    str(out_path),
                ]
            )
            client = _FakeClient(handler)
            client._client.next_response = SimpleNamespace(
                status_code=200, content=b"PKZIPDATA", text=""
            )
            buffer = io.StringIO()
            original = sys.stdout
            sys.stdout = buffer
            try:
                rc = args.func(args, client)
            finally:
                sys.stdout = original

            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.read_bytes(), b"PKZIPDATA")
            # Download path was hit with the right bundle_uid + token.
            download_calls = client._client.calls
            self.assertEqual(len(download_calls), 1)
            self.assertEqual(
                download_calls[0]["path"],
                "/support-bundles/abc1234567890/download",
            )
            self.assertEqual(
                download_calls[0]["params"], {"token": "tok"}
            )
            # Output JSON includes downloaded_to.
            body = json.loads(buffer.getvalue())
            self.assertEqual(body["downloaded_to"], str(out_path.resolve()))

    def test_support_bundle_create_download_failure_raises(self):
        def handler(call: dict[str, Any]) -> Any:
            return {
                "bundle_uid": "abc",
                "download_token": "tok",
                "size_bytes": 0,
                "section_counts": {},
            }

        args = self._parse(
            [
                "support-bundle",
                "create",
                "--project",
                "7",
                "--download",
            ]
        )
        client = _FakeClient(handler)
        client._client.next_response = SimpleNamespace(
            status_code=410,
            content=b"",
            text="support_bundle_expired",
        )
        with self.assertRaises(RuntimeError) as cm:
            args.func(args, client)
        self.assertIn("support_bundle_expired", str(cm.exception))

    def test_support_bundle_list_path_and_limit(self):
        def handler(call: dict[str, Any]) -> Any:
            return {"bundles": []}

        rc, client, _ = self._capture_stdout(
            [
                "support-bundle",
                "list",
                "--project",
                "9",
                "--limit",
                "10",
            ],
            handler,
        )
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "/projects/9/support-bundles")
        self.assertEqual(call["params"], {"limit": 10})

    def test_support_bundle_download_writes_to_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "x.zip"
            args = self._parse(
                [
                    "support-bundle",
                    "download",
                    "--bundle-uid",
                    "abc1234",
                    "--token",
                    "tok",
                    "--out",
                    str(out_path),
                ]
            )
            client = _FakeClient()
            client._client.next_response = SimpleNamespace(
                status_code=200, content=b"PK_BYTES", text=""
            )
            buffer = io.StringIO()
            original = sys.stdout
            sys.stdout = buffer
            try:
                rc = args.func(args, client)
            finally:
                sys.stdout = original
            self.assertEqual(rc, 0)
            self.assertEqual(out_path.read_bytes(), b"PK_BYTES")
            body = json.loads(buffer.getvalue())
            self.assertEqual(body["bundle_uid"], "abc1234")

    def test_support_bundle_parent_without_subcommand_exits(self):
        with self.assertRaises(SystemExit):
            self.cli.build_parser().parse_args(["support-bundle"])


if __name__ == "__main__":
    unittest.main()

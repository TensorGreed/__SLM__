"""Phase 78 — brewslm CLI ``deploy`` subparser (priority.md P29).

Covers the new ``brewslm deploy`` subparser wiring through to the
P25/P26/P27/P28 endpoints. Each verb is exercised against a
``_FakeClient`` that captures method/path/body/params, mirroring the
phase70 (train) and phase73 (manifest) test patterns.

Verbs covered:
- ``plan``            → POST /projects/{pid}/export/{eid}/deploy-as-api  (P25)
- ``smoke-test``      → POST /projects/{pid}/export/{eid}/deployment-validate  (P25)
- ``execute``         → POST /projects/{pid}/export/{eid}/deploy-as-api/execute  (P25)
- ``promote``         → POST /deployments/{id}/promote                  (P25)
- ``reject``          → POST /deployments/{id}/reject                   (P25)
- ``rollback``        → POST /deployments/{id}/rollback                 (P25)
- ``get``             → GET  /deployments/{id}                          (P25)
- ``list``            → GET  /projects/{pid}/deployments + filters      (P25)
- ``telemetry``       → GET  /deployments/{id}/telemetry + window       (P26)
- ``telemetry-ingest``→ POST /deployments/{id}/telemetry/ingest         (P26)
- ``drift-check``     → POST /deployments/{id}/drift/check              (P27)
- ``drift-history``   → GET  /deployments/{id}/drift/checks             (P27)
- ``score``           → GET  /deployments/{id}/score                    (P28)
- ``score-compute``   → POST /deployments/{id}/score/compute            (P28)
- ``score-history``   → GET  /deployments/{id}/score/history            (P28)

Plus edge cases:
- ``deploy`` without a subcommand → ``SystemExit`` (mirrors the
  ``manifest`` / ``repro`` parent behaviour).
- ``telemetry-ingest`` with malformed JSON → ``ValueError``.
- ``drift-check`` with neither predictions nor endpoint URL →
  ``ValueError``.
- ``drift-check`` accepts predictions either as a top-level list or
  ``{"predictions": [...]}``.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "brewslm.py"
    spec = importlib.util.spec_from_file_location("brewslm_cli_p29", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeClient:
    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Any] | None = None,
    ):
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
            "json_body": (
                dict(json_body) if json_body is not None else None
            ),
            "params": dict(params) if params is not None else None,
        }
        self.calls.append(call)
        return self._handler(call)

    def close(self) -> None:  # pragma: no cover
        return None


class Phase78DeployCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def _parse(self, argv: list[str]):
        parser = self.cli.build_parser()
        return parser.parse_args(argv)

    def _run(self, argv: list[str], handler: Callable[[dict], Any] | None = None):
        args = self._parse(argv)
        client = _FakeClient(handler)
        rc = args.func(args, client)
        return rc, client.calls

    def _write_json_file(self, payload: Any, *, suffix: str = ".json") -> str:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=suffix, delete=False, encoding="utf-8"
        )
        if isinstance(payload, str):
            handle.write(payload)
        else:
            json.dump(payload, handle)
        handle.close()
        self.addCleanup(lambda p=handle.name: Path(p).unlink(missing_ok=True))
        return handle.name

    # -- plan -----------------------------------------------------------

    def test_plan_posts_with_optional_fields(self):
        rc, calls = self._run(
            [
                "deploy",
                "plan",
                "--project",
                "7",
                "--export-id",
                "42",
                "--target-id",
                "deployment.hf_inference_endpoint",
                "--endpoint-name",
                "my-ep",
                "--region",
                "us-east-1",
            ]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(
            call["path"],
            "/projects/7/export/42/deploy-as-api",
        )
        self.assertEqual(
            call["json_body"],
            {
                "target_id": "deployment.hf_inference_endpoint",
                "endpoint_name": "my-ep",
                "region": "us-east-1",
            },
        )

    def test_plan_only_target_id_when_optionals_unset(self):
        _, calls = self._run(
            [
                "deploy",
                "plan",
                "--project",
                "1",
                "--export-id",
                "2",
                "--target-id",
                "sdk.apple_coreml_stub",
            ]
        )
        self.assertEqual(
            calls[0]["json_body"],
            {"target_id": "sdk.apple_coreml_stub"},
        )

    # -- smoke-test -----------------------------------------------------

    def test_smoke_test_posts_target_list(self):
        _, calls = self._run(
            [
                "deploy",
                "smoke-test",
                "--project",
                "9",
                "--export-id",
                "8",
                "--targets",
                "exporter.huggingface, deployment.hf_inference_endpoint",
            ]
        )
        self.assertEqual(
            calls[0]["path"], "/projects/9/export/8/deployment-validate"
        )
        self.assertEqual(calls[0]["json_body"]["run_smoke_tests"], True)
        self.assertEqual(
            calls[0]["json_body"]["deployment_targets"],
            ["exporter.huggingface", "deployment.hf_inference_endpoint"],
        )

    def test_smoke_test_no_smoke_flips_run_smoke_tests(self):
        _, calls = self._run(
            [
                "deploy",
                "smoke-test",
                "--project",
                "1",
                "--export-id",
                "2",
                "--no-smoke",
            ]
        )
        self.assertEqual(calls[0]["json_body"]["run_smoke_tests"], False)

    # -- execute --------------------------------------------------------

    def test_execute_dry_run_flag(self):
        _, calls = self._run(
            [
                "deploy",
                "execute",
                "--project",
                "4",
                "--export-id",
                "5",
                "--target-id",
                "sdk.apple_coreml_stub",
                "--dry-run",
            ]
        )
        self.assertEqual(
            calls[0]["path"], "/projects/4/export/5/deploy-as-api/execute"
        )
        body = calls[0]["json_body"]
        self.assertEqual(body["target_id"], "sdk.apple_coreml_stub")
        self.assertTrue(body["dry_run"])
        # Optional credentials are absent when not supplied.
        self.assertNotIn("hf_token", body)
        self.assertNotIn("sagemaker_role_arn", body)

    def test_execute_passes_credentials_when_supplied(self):
        _, calls = self._run(
            [
                "deploy",
                "execute",
                "--project",
                "1",
                "--export-id",
                "2",
                "--target-id",
                "deployment.hf_inference_endpoint",
                "--hf-token",
                "hf_xxx",
            ]
        )
        body = calls[0]["json_body"]
        self.assertEqual(body["hf_token"], "hf_xxx")
        self.assertFalse(body["dry_run"])

    # -- promote / reject / rollback ------------------------------------

    def test_promote_posts_reason_and_actor(self):
        _, calls = self._run(
            [
                "deploy",
                "promote",
                "--deployment-id",
                "11",
                "--reason",
                "ready",
                "--actor",
                "alice",
            ]
        )
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(calls[0]["path"], "/deployments/11/promote")
        self.assertEqual(
            calls[0]["json_body"], {"reason": "ready", "actor": "alice"}
        )

    def test_reject_posts_with_empty_body_when_no_reason(self):
        _, calls = self._run(
            ["deploy", "reject", "--deployment-id", "13"]
        )
        self.assertEqual(calls[0]["path"], "/deployments/13/reject")
        self.assertEqual(calls[0]["json_body"], {})

    def test_rollback_path_and_method(self):
        _, calls = self._run(
            [
                "deploy",
                "rollback",
                "--deployment-id",
                "21",
                "--reason",
                "regression",
            ]
        )
        self.assertEqual(calls[0]["path"], "/deployments/21/rollback")
        self.assertEqual(calls[0]["json_body"], {"reason": "regression"})

    # -- get / list -----------------------------------------------------

    def test_get_uses_get_method(self):
        _, calls = self._run(
            ["deploy", "get", "--deployment-id", "5"]
        )
        self.assertEqual(calls[0]["method"], "GET")
        self.assertEqual(calls[0]["path"], "/deployments/5")
        self.assertIsNone(calls[0]["json_body"])

    def test_list_filters_translate_to_query_params(self):
        _, calls = self._run(
            [
                "deploy",
                "list",
                "--project",
                "9",
                "--export-id",
                "11",
                "--target-id",
                "sdk.apple_coreml_stub",
                "--status",
                "promoted",
            ]
        )
        self.assertEqual(calls[0]["method"], "GET")
        self.assertEqual(calls[0]["path"], "/projects/9/deployments")
        self.assertEqual(
            calls[0]["params"],
            {
                "export_id": 11,
                "target_id": "sdk.apple_coreml_stub",
                "status": "promoted",
            },
        )

    def test_list_omits_params_when_no_filters(self):
        _, calls = self._run(
            ["deploy", "list", "--project", "9"]
        )
        self.assertIsNone(calls[0]["params"])

    # -- telemetry ------------------------------------------------------

    def test_telemetry_window_seconds_param(self):
        _, calls = self._run(
            [
                "deploy",
                "telemetry",
                "--deployment-id",
                "7",
                "--window-seconds",
                "120",
            ]
        )
        self.assertEqual(calls[0]["method"], "GET")
        self.assertEqual(calls[0]["path"], "/deployments/7/telemetry")
        self.assertEqual(calls[0]["params"], {"window_seconds": 120})

    def test_telemetry_ingest_reads_list_file(self):
        path = self._write_json_file(
            [{"latency_ms": 10.0}, {"latency_ms": 20.0}]
        )
        _, calls = self._run(
            ["deploy", "telemetry-ingest", "--deployment-id", "5", path]
        )
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(
            calls[0]["path"], "/deployments/5/telemetry/ingest"
        )
        self.assertEqual(
            calls[0]["json_body"],
            {"samples": [{"latency_ms": 10.0}, {"latency_ms": 20.0}]},
        )

    def test_telemetry_ingest_accepts_object_with_samples_key(self):
        path = self._write_json_file({"samples": [{"latency_ms": 33.0}]})
        _, calls = self._run(
            ["deploy", "telemetry-ingest", "--deployment-id", "5", path]
        )
        self.assertEqual(
            calls[0]["json_body"], {"samples": [{"latency_ms": 33.0}]}
        )

    def test_telemetry_ingest_invalid_json_raises(self):
        path = self._write_json_file("not json at all")
        with self.assertRaises(ValueError):
            self._run(
                ["deploy", "telemetry-ingest", "--deployment-id", "5", path]
            )

    # -- drift-check / drift-history ------------------------------------

    def test_drift_check_with_predictions_file(self):
        path = self._write_json_file(
            [{"row_id": 1, "prediction": "yes"}, {"row_id": 2, "prediction": "no"}]
        )
        _, calls = self._run(
            [
                "deploy",
                "drift-check",
                "--deployment-id",
                "9",
                "--gold-set-id",
                "44",
                "--predictions-file",
                path,
                "--tolerance",
                "0.10",
                "--max-samples",
                "20",
            ]
        )
        self.assertEqual(calls[0]["path"], "/deployments/9/drift/check")
        body = calls[0]["json_body"]
        self.assertEqual(body["gold_set_id"], 44)
        self.assertEqual(body["tolerance"], 0.10)
        self.assertEqual(body["max_samples"], 20)
        self.assertEqual(len(body["predictions"]), 2)
        self.assertNotIn("endpoint_url", body)

    def test_drift_check_with_endpoint_url_and_headers(self):
        _, calls = self._run(
            [
                "deploy",
                "drift-check",
                "--deployment-id",
                "9",
                "--gold-set-id",
                "44",
                "--endpoint-url",
                "https://api.example/inference",
                "--endpoint-headers",
                '{"Authorization": "Bearer xxx"}',
            ]
        )
        body = calls[0]["json_body"]
        self.assertEqual(body["endpoint_url"], "https://api.example/inference")
        self.assertEqual(
            body["endpoint_headers"], {"Authorization": "Bearer xxx"}
        )
        self.assertNotIn("predictions", body)

    def test_drift_check_without_predictions_or_endpoint_raises(self):
        with self.assertRaises(ValueError):
            self._run(
                [
                    "deploy",
                    "drift-check",
                    "--deployment-id",
                    "9",
                    "--gold-set-id",
                    "44",
                ]
            )

    def test_drift_check_predictions_object_form(self):
        path = self._write_json_file(
            {"predictions": [{"row_id": 7, "prediction": "ok"}]}
        )
        _, calls = self._run(
            [
                "deploy",
                "drift-check",
                "--deployment-id",
                "9",
                "--gold-set-id",
                "1",
                "--predictions-file",
                path,
            ]
        )
        self.assertEqual(
            calls[0]["json_body"]["predictions"],
            [{"row_id": 7, "prediction": "ok"}],
        )

    def test_drift_history_path(self):
        _, calls = self._run(
            [
                "deploy",
                "drift-history",
                "--deployment-id",
                "8",
                "--limit",
                "10",
            ]
        )
        self.assertEqual(calls[0]["method"], "GET")
        self.assertEqual(calls[0]["path"], "/deployments/8/drift/checks")
        self.assertEqual(calls[0]["params"], {"limit": 10})

    # -- score / score-compute / score-history --------------------------

    def test_score_get_path(self):
        _, calls = self._run(
            ["deploy", "score", "--deployment-id", "12"]
        )
        self.assertEqual(calls[0]["method"], "GET")
        self.assertEqual(calls[0]["path"], "/deployments/12/score")

    def test_score_compute_posts_with_actor(self):
        _, calls = self._run(
            [
                "deploy",
                "score-compute",
                "--deployment-id",
                "12",
                "--notes",
                "post-deploy verification",
                "--actor",
                "ops",
            ]
        )
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(
            calls[0]["path"], "/deployments/12/score/compute"
        )
        self.assertEqual(
            calls[0]["json_body"],
            {"notes": "post-deploy verification", "actor": "ops"},
        )

    def test_score_history_path_and_limit(self):
        _, calls = self._run(
            [
                "deploy",
                "score-history",
                "--deployment-id",
                "3",
                "--limit",
                "25",
            ]
        )
        self.assertEqual(calls[0]["path"], "/deployments/3/score/history")
        self.assertEqual(calls[0]["params"], {"limit": 25})

    # -- parent without subcommand --------------------------------------

    def test_deploy_parent_without_subcommand_exits(self):
        with self.assertRaises(SystemExit):
            self.cli.build_parser().parse_args(["deploy"])


if __name__ == "__main__":
    unittest.main()

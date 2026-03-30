"""Phase 49 tests: brewslm models CLI commands."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "brewslm.py"
    spec = importlib.util.spec_from_file_location("brewslm_cli", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeClient:
    def __init__(self, handler):
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None, params: dict[str, Any] | None = None):
        call = {
            "method": method.upper(),
            "path": path,
            "json_body": dict(json_body or {}),
            "params": dict(params or {}),
        }
        self.calls.append(call)
        return self._handler(call)

    def close(self) -> None:  # pragma: no cover
        return None


class Phase49UniversalModelRegistryCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def test_models_import_and_list_json(self):
        parser = self.cli.build_parser()

        import_args = parser.parse_args(
            [
                "models",
                "import",
                "--hf-id",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--json",
            ]
        )

        def import_handler(call: dict[str, Any]) -> dict[str, Any]:
            if call["path"] == "/models/import":
                return {
                    "created": True,
                    "model": {
                        "id": 7,
                        "model_key": "huggingface:qwen:abc123",
                        "display_name": "Qwen/Qwen2.5-1.5B-Instruct",
                    },
                }
            raise AssertionError(f"Unexpected call: {call}")

        import_client = _FakeClient(import_handler)
        import_rc = import_args.func(import_args, import_client)
        self.assertEqual(import_rc, 0)
        self.assertEqual(import_client.calls[0]["path"], "/models/import")
        self.assertEqual(import_client.calls[0]["json_body"].get("source_type"), "huggingface")

        list_args = parser.parse_args(
            ["models", "list", "--family", "qwen", "--hardware-fit", "laptop", "--json"]
        )

        def list_handler(call: dict[str, Any]) -> dict[str, Any]:
            if call["path"] == "/models":
                return {
                    "count": 1,
                    "models": [
                        {
                            "id": 7,
                            "model_key": "huggingface:qwen:abc123",
                            "display_name": "Qwen/Qwen2.5-1.5B-Instruct",
                            "model_family": "qwen",
                            "architecture": "causal_lm",
                        }
                    ],
                }
            raise AssertionError(f"Unexpected call: {call}")

        list_client = _FakeClient(list_handler)
        list_rc = list_args.func(list_args, list_client)
        self.assertEqual(list_rc, 0)
        self.assertEqual(list_client.calls[0]["path"], "/models")
        self.assertEqual(list_client.calls[0]["params"].get("family"), "qwen")
        self.assertEqual(list_client.calls[0]["params"].get("hardware_fit"), "laptop")

    def test_models_validate_success_and_failure(self):
        parser = self.cli.build_parser()

        success_args = parser.parse_args(
            ["models", "validate", "--project", "55", "--model", "7", "--json"]
        )

        def success_handler(call: dict[str, Any]) -> dict[str, Any]:
            if call["path"] == "/projects/55/models/validate":
                return {
                    "project_id": 55,
                    "model_id": 7,
                    "model_key": "catalog:qwen:abc",
                    "compatible": True,
                    "compatibility_score": 0.9,
                    "reason_codes": ["TASK_FAMILY_SUPPORTED"],
                    "why_risky": [],
                    "recommended_next_actions": [],
                }
            raise AssertionError(f"Unexpected call: {call}")

        success_client = _FakeClient(success_handler)
        success_rc = success_args.func(success_args, success_client)
        self.assertEqual(success_rc, 0)
        self.assertEqual(success_client.calls[0]["json_body"].get("model_id"), 7)

        fail_args = parser.parse_args(
            ["models", "validate", "--project", "55", "--model", "catalog:bad:model"]
        )

        def fail_handler(call: dict[str, Any]) -> dict[str, Any]:
            if call["path"] == "/projects/55/models/validate":
                return {
                    "project_id": 55,
                    "model_id": 9,
                    "model_key": "catalog:bad:model",
                    "compatible": False,
                    "compatibility_score": 0.31,
                    "reason_codes": ["TASK_FAMILY_UNSUPPORTED"],
                    "why_risky": [
                        {
                            "code": "TASK_FAMILY_UNSUPPORTED",
                            "severity": "blocker",
                            "message": "Task family mismatch.",
                        }
                    ],
                    "recommended_next_actions": ["Choose a model that supports the project task family."],
                }
            raise AssertionError(f"Unexpected call: {call}")

        fail_client = _FakeClient(fail_handler)
        fail_rc = fail_args.func(fail_args, fail_client)
        self.assertEqual(fail_rc, 1)
        self.assertEqual(fail_client.calls[0]["json_body"].get("model_key"), "catalog:bad:model")


if __name__ == "__main__":
    unittest.main()

"""Phase 50 tests: Adapter Studio CLI commands."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "brewslm.py"
    spec = importlib.util.spec_from_file_location("brewslm_cli_phase50", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeClient:
    def __init__(self, handler):
        self._handler = handler
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
            "json_body": dict(json_body or {}),
            "params": dict(params or {}),
        }
        self.calls.append(call)
        return self._handler(call)

    def upload_file(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("upload_file not expected in this test")

    def close(self) -> None:  # pragma: no cover
        return None


class Phase50AdapterStudioCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def test_dataset_profile_and_adapter_infer_json(self):
        parser = self.cli.build_parser()

        dataset_args = parser.parse_args(
            [
                "dataset",
                "profile",
                "--project",
                "11",
                "--source-type",
                "csv",
                "--source-ref",
                "/tmp/data.csv",
                "--sample-size",
                "250",
                "--json",
            ]
        )

        def profile_handler(call: dict[str, Any]) -> dict[str, Any]:
            if call["path"] == "/projects/11/adapter-studio/profile":
                return {"sampled_rows": 250, "schema": {"field_count": 6}}
            raise AssertionError(f"Unexpected call: {call}")

        profile_client = _FakeClient(profile_handler)
        rc = dataset_args.func(dataset_args, profile_client)
        self.assertEqual(rc, 0)
        self.assertEqual(profile_client.calls[0]["path"], "/projects/11/adapter-studio/profile")
        payload = profile_client.calls[0]["json_body"]
        self.assertEqual(payload.get("sample_size"), 250)
        self.assertEqual((payload.get("source") or {}).get("source_type"), "csv")

        infer_args = parser.parse_args(
            [
                "adapter",
                "infer",
                "--project",
                "11",
                "--source-type",
                "jsonl",
                "--source-ref",
                "/tmp/chat.jsonl",
                "--task-profile",
                "chat_sft",
                "--json",
            ]
        )

        def infer_handler(call: dict[str, Any]) -> dict[str, Any]:
            if call["path"] == "/projects/11/adapter-studio/infer":
                return {
                    "inference": {
                        "resolved_adapter_id": "chat-messages",
                        "resolved_task_profile": "chat_sft",
                        "confidence": 0.93,
                    }
                }
            raise AssertionError(f"Unexpected call: {call}")

        infer_client = _FakeClient(infer_handler)
        infer_rc = infer_args.func(infer_args, infer_client)
        self.assertEqual(infer_rc, 0)
        self.assertEqual(infer_client.calls[0]["path"], "/projects/11/adapter-studio/infer")
        self.assertEqual((infer_client.calls[0]["json_body"].get("source") or {}).get("source_ref"), "/tmp/chat.jsonl")

    def test_adapter_validate_and_export(self):
        parser = self.cli.build_parser()

        validate_args = parser.parse_args(
            [
                "adapter",
                "validate",
                "--project",
                "9",
                "--source-type",
                "csv",
                "--source-ref",
                "/tmp/raw.csv",
                "--adapter-id",
                "qa-pair",
                "--field-mapping",
                '{"question":"prompt","answer":"response"}',
                "--json",
            ]
        )

        def validate_handler(call: dict[str, Any]) -> dict[str, Any]:
            if call["path"] == "/projects/9/adapter-studio/validate":
                return {
                    "status": "warning",
                    "reason_codes": ["HIGH_DROP_RATE"],
                    "recommended_next_actions": ["Apply suggested field mapping."],
                }
            raise AssertionError(f"Unexpected call: {call}")

        validate_client = _FakeClient(validate_handler)
        validate_rc = validate_args.func(validate_args, validate_client)
        self.assertEqual(validate_rc, 0)
        self.assertEqual(validate_client.calls[0]["path"], "/projects/9/adapter-studio/validate")
        self.assertEqual((validate_client.calls[0]["json_body"].get("field_mapping") or {}).get("question"), "prompt")

        export_args = parser.parse_args(
            [
                "adapter",
                "export",
                "--project",
                "9",
                "--adapter-name",
                "qa_studio",
                "--version",
                "3",
                "--json",
            ]
        )

        def export_handler(call: dict[str, Any]) -> dict[str, Any]:
            if call["path"] == "/projects/9/adapter-studio/adapters/qa_studio/versions/3/export":
                return {
                    "written_files": {
                        "template_json": "/tmp/adapter_template.json",
                        "plugin_python": "/tmp/adapter_plugin.py",
                    }
                }
            raise AssertionError(f"Unexpected call: {call}")

        export_client = _FakeClient(export_handler)
        export_rc = export_args.func(export_args, export_client)
        self.assertEqual(export_rc, 0)
        self.assertEqual(
            export_client.calls[0]["path"],
            "/projects/9/adapter-studio/adapters/qa_studio/versions/3/export",
        )


if __name__ == "__main__":
    unittest.main()

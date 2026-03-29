"""Phase 48 tests: brewslm CLI domain blueprint commands."""

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

    def close(self) -> None:  # pragma: no cover - interface compatibility
        return None


class Phase48DomainBlueprintCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    def test_project_bootstrap_create_project_calls_analyze_then_create(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "project",
                "bootstrap",
                "--name",
                "cli-blueprint-project",
                "--brief",
                "Build a support FAQ assistant for ticket responses.",
                "--sample-input",
                "How do I change my password?",
                "--sample-output",
                '{"answer":"Use the account settings password reset flow."}',
                "--target",
                "edge_gpu",
                "--create-project",
            ]
        )

        def handler(call: dict[str, Any]) -> dict[str, Any]:
            if call["path"] == "/domain-blueprints/analyze":
                return {
                    "blueprint": {
                        "domain_name": "Support",
                        "problem_statement": "Answer support FAQs.",
                        "target_user_persona": "Support agents",
                        "task_family": "qa",
                        "input_modality": "text",
                        "expected_output_schema": {"type": "object", "properties": {"answer": "string"}, "required": ["answer"]},
                        "expected_output_examples": [{"answer": "Use reset flow."}],
                        "safety_compliance_notes": ["Do not expose customer secrets."],
                        "deployment_target_constraints": {"target_profile_id": "edge_gpu"},
                        "success_metrics": [{"metric_id": "answer_correctness", "label": "Answer Correctness"}],
                        "glossary": [{"term": "task family", "plain_language": "Type of model behavior", "category": "general"}],
                        "confidence_score": 0.8,
                        "unresolved_assumptions": [],
                    },
                    "guidance": {},
                    "validation": {"ok": True, "errors": [], "warnings": []},
                    "llm_enrichment": {"enabled": False, "applied": False},
                }
            if call["path"] == "/projects":
                return {"id": 11, "name": "cli-blueprint-project", "beginner_mode": True}
            raise AssertionError(f"Unexpected API call: {call}")

        client = _FakeClient(handler)
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["path"], "/domain-blueprints/analyze")
        self.assertEqual(client.calls[1]["path"], "/projects")
        create_payload = client.calls[1]["json_body"]
        self.assertTrue(bool(create_payload.get("beginner_mode")))
        self.assertIn("domain_blueprint", create_payload)

    def test_project_blueprint_show_and_diff(self):
        parser = self.cli.build_parser()

        show_args = parser.parse_args(
            ["project", "blueprint", "show", "--project", "42", "--latest"]
        )
        show_client = _FakeClient(lambda call: {"version": 3, "status": "active"} if call["path"].endswith("/latest") else {})
        show_rc = show_args.func(show_args, show_client)
        self.assertEqual(show_rc, 0)
        self.assertEqual(show_client.calls[0]["method"], "GET")
        self.assertEqual(show_client.calls[0]["path"], "/projects/42/domain-blueprints/latest")

        diff_args = parser.parse_args(
            ["project", "blueprint", "diff", "--project", "42", "--from-version", "1", "--to-version", "3"]
        )
        diff_client = _FakeClient(lambda _call: {"changed_fields": [{"field": "task_family"}]})
        diff_rc = diff_args.func(diff_args, diff_client)
        self.assertEqual(diff_rc, 0)
        self.assertEqual(diff_client.calls[0]["path"], "/projects/42/domain-blueprints/diff")
        self.assertEqual(diff_client.calls[0]["params"], {"from_version": 1, "to_version": 3})


if __name__ == "__main__":
    unittest.main()

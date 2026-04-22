"""Phase 54 tests — brewslm CLI autopilot subparser (priority.md P4).

Covers:

- `autopilot plan`  → POSTs to /autopilot/repair-preview with the orchestrate
  body, exits 0 when the plan is runnable and 1 when not.
- `autopilot run`   → POSTs to /projects/{id}/training/autopilot/v2/orchestrate/run
  and reports `started`.
- `autopilot repair --plan-token …` → POSTs to /autopilot/repair-apply
  with optional `--force`/`--actor`/`--reason`/`--expected-state-hash`.
- `autopilot rollback <id>` → POSTs to /autopilot/rollback/{id}, or to the
  preview sibling when `--preview` is passed.
- `autopilot decisions` → GETs /autopilot/decisions with filter/pagination
  params plumbed through.
- All five subcommands also accept `--json` and emit raw JSON on stdout.
"""

from __future__ import annotations

import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _load_cli_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "brewslm.py"
    spec = importlib.util.spec_from_file_location("brewslm_cli_p4", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeClient:
    def __init__(self, handler: Callable[[dict[str, Any]], Any]):
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

    def close(self) -> None:  # pragma: no cover
        return None


class Phase54AutopilotCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    # -- autopilot plan ---------------------------------------------------

    def test_plan_posts_to_repair_preview_and_returns_zero_when_runnable(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "autopilot",
                "plan",
                "--project",
                "17",
                "--intent",
                "Build a concise support assistant.",
                "--target-profile",
                "edge_gpu",
                "--plan-profile",
                "balanced",
            ]
        )

        response = {
            "preview": {
                "plan_token": "plan-abc",
                "project_id": 17,
                "expires_at": "2026-04-22T14:30:00+00:00",
                "state_hash": "aaaa",
            },
            "config_diff": {
                "summary": "Will create experiment and start training.",
                "would_create_experiment": True,
                "selected_profile": "balanced",
                "effective_target_profile_id": "edge_gpu",
                "repairs_planned": [{"kind": "intent_rewrite", "applied": True}],
                "guardrails": {"blockers": [], "warnings": [], "can_run": True},
            },
            "dry_run_response": {"dry_run": True, "started": False},
            "state_hash": "aaaa",
        }

        client = _FakeClient(lambda _call: response)
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/autopilot/repair-preview")
        body = call["json_body"]
        self.assertEqual(body["project_id"], 17)
        self.assertEqual(body["intent"], "Build a concise support assistant.")
        self.assertEqual(body["target_profile_id"], "edge_gpu")
        self.assertEqual(body["plan_profile"], "balanced")
        # No `--no-*` flags → should default to True.
        self.assertTrue(body["auto_prepare_data"])
        self.assertTrue(body["auto_apply_rewrite"])
        self.assertTrue(body["allow_target_fallback"])
        self.assertTrue(body["allow_profile_autotune"])

    def test_plan_returns_non_zero_when_not_runnable(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["autopilot", "plan", "--project", "3", "--intent", "X"]
        )
        response = {
            "preview": {"plan_token": "tok"},
            "config_diff": {
                "would_create_experiment": False,
                "guardrails": {"blockers": ["dataset not ready"], "warnings": []},
            },
        }
        client = _FakeClient(lambda _call: response)
        rc = args.func(args, client)
        self.assertEqual(rc, 1)

    def test_plan_honors_no_flags_and_json_flag(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "autopilot",
                "plan",
                "--project",
                "5",
                "--intent",
                "Do stuff",
                "--no-auto-prepare-data",
                "--no-auto-rewrite",
                "--no-target-fallback",
                "--no-profile-autotune",
                "--available-vram-gb",
                "16",
                "--base-model",
                "microsoft/phi-2",
                "--intent-rewrite",
                "Rewritten text",
                "--run-name",
                "cli-run",
                "--description",
                "cli desc",
                "--json",
            ]
        )
        captured = {"preview": {"plan_token": "tk"}, "config_diff": {"would_create_experiment": True}}
        client = _FakeClient(lambda _call: captured)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        body = client.calls[0]["json_body"]
        self.assertFalse(body["auto_prepare_data"])
        self.assertFalse(body["auto_apply_rewrite"])
        self.assertFalse(body["allow_target_fallback"])
        self.assertFalse(body["allow_profile_autotune"])
        self.assertEqual(body["available_vram_gb"], 16.0)
        self.assertEqual(body["base_model"], "microsoft/phi-2")
        self.assertEqual(body["intent_rewrite"], "Rewritten text")
        self.assertEqual(body["run_name"], "cli-run")
        self.assertEqual(body["description"], "cli desc")
        # --json mode produces parseable JSON.
        out = stdout.getvalue().strip()
        parsed = json.loads(out)
        self.assertEqual(parsed, captured)

    # -- autopilot run ----------------------------------------------------

    def test_run_posts_to_orchestrate_run_with_dry_run_false(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["autopilot", "run", "--project", "8", "--intent", "Ship it"]
        )
        response = {
            "run_id": "run-xyz",
            "dry_run": False,
            "started": True,
            "experiment": {"id": 99, "name": "Autopilot Balanced"},
            "decision_log": [],
        }
        client = _FakeClient(lambda _call: response)
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/projects/8/training/autopilot/v2/orchestrate/run")
        self.assertFalse(call["json_body"]["dry_run"])

    def test_run_exits_non_zero_when_not_started(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["autopilot", "run", "--project", "8", "--intent", "Ship it"]
        )
        client = _FakeClient(
            lambda _call: {"run_id": "r", "started": False, "dry_run": False}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 1)

    # -- autopilot repair -------------------------------------------------

    def test_repair_posts_to_repair_apply_with_token_and_optional_fields(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "autopilot",
                "repair",
                "--plan-token",
                "plan-abc",
                "--actor",
                "cli-user",
                "--reason",
                "confirmed",
                "--expected-state-hash",
                "abcd" * 16,
                "--force",
            ]
        )
        response = {
            "ok": True,
            "preview": {"plan_token": "plan-abc", "applied_at": "2026-04-22T14:00:00+00:00"},
            "response": {
                "run_id": "r1",
                "dry_run": False,
                "started": True,
                "experiment": {"id": 4, "name": "auto"},
                "decision_log": [],
            },
        }
        client = _FakeClient(lambda _call: response)
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["path"], "/autopilot/repair-apply")
        body = call["json_body"]
        self.assertEqual(body["plan_token"], "plan-abc")
        self.assertTrue(body["force"])
        self.assertEqual(body["actor"], "cli-user")
        self.assertEqual(body["reason"], "confirmed")
        self.assertEqual(body["expected_state_hash"], "abcd" * 16)

    def test_repair_returns_non_zero_on_failure_payload(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["autopilot", "repair", "--plan-token", "tok-x"]
        )
        client = _FakeClient(
            lambda _call: {"ok": False, "reason": "state_drift", "preview": {"plan_token": "tok-x"}}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 1)

    # -- autopilot rollback ----------------------------------------------

    def test_rollback_posts_to_rollback_endpoint(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "autopilot",
                "rollback",
                "17",
                "--actor",
                "phase54-test",
                "--reason",
                "mistake",
            ]
        )
        response = {
            "ok": True,
            "decision_id": 17,
            "rollback_decision": {"id": 33, "stage": "rollback", "action": "rolled_back"},
            "snapshot": {"id": 4},
            "outcomes": [{"kind": "cancel_experiment", "status": "applied", "experiment_id": 9}],
        }
        client = _FakeClient(lambda _call: response)
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/autopilot/rollback/17")
        self.assertEqual(call["json_body"]["actor"], "phase54-test")
        self.assertEqual(call["json_body"]["reason"], "mistake")

    def test_rollback_preview_hits_preview_subpath(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["autopilot", "rollback", "17", "--preview"]
        )
        response = {
            "reversible": True,
            "steps": [{"kind": "cancel_experiment", "experiment_id": 9}],
        }
        client = _FakeClient(lambda _call: response)
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls[0]["path"], "/autopilot/rollback/17/preview")

    def test_rollback_returns_non_zero_on_failure(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["autopilot", "rollback", "42"])
        client = _FakeClient(
            lambda _call: {"ok": False, "reason": "already_rolled_back"}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 1)

    # -- autopilot decisions ---------------------------------------------

    def test_decisions_plumbs_filter_params(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "autopilot",
                "decisions",
                "--project",
                "8",
                "--run-id",
                "run-xyz",
                "--stage",
                "start_training",
                "--status",
                "completed",
                "--action",
                "applied",
                "--reason-code",
                "AUTOPILOT_ROLLBACK",
                "--limit",
                "25",
                "--offset",
                "10",
                "--since",
                "2026-04-22T00:00:00+00:00",
            ]
        )
        client = _FakeClient(lambda _call: {"items": []})
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "/autopilot/decisions")
        params = call["params"]
        self.assertEqual(params["project_id"], 8)
        self.assertEqual(params["run_id"], "run-xyz")
        self.assertEqual(params["stage"], "start_training")
        self.assertEqual(params["status"], "completed")
        self.assertEqual(params["action"], "applied")
        self.assertEqual(params["reason_code"], "AUTOPILOT_ROLLBACK")
        self.assertEqual(params["since"], "2026-04-22T00:00:00+00:00")
        self.assertEqual(params["limit"], 25)
        self.assertEqual(params["offset"], 10)

    def test_decisions_json_flag_emits_parseable_json(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(["autopilot", "decisions", "--json"])
        response = {"items": [{"id": 1, "stage": "start_training"}]}
        client = _FakeClient(lambda _call: response)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = args.func(args, client)
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.getvalue().strip())
        self.assertEqual(parsed, response)

    # -- structural --------------------------------------------------------

    def test_autopilot_subparser_has_all_five_subcommands(self):
        parser = self.cli.build_parser()
        for sub in ("plan", "run", "repair", "rollback", "decisions"):
            args = parser.parse_args(
                [
                    "autopilot",
                    sub,
                    # Add required args per subcommand:
                    *self._required_args_for(sub),
                ]
            )
            self.assertTrue(callable(args.func), f"{sub} must be wired to a handler")

    def _required_args_for(self, sub: str) -> list[str]:
        if sub in {"plan", "run"}:
            return ["--project", "1", "--intent", "x"]
        if sub == "repair":
            return ["--plan-token", "tk"]
        if sub == "rollback":
            return ["1"]
        return []


if __name__ == "__main__":
    unittest.main()

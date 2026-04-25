"""Phase 70 — brewslm CLI training subparser additions (priority.md P19, RM3).

Covers the new ``brewslm train`` subcommands wiring through to the P14/P15/
P16/P17 endpoints, plus ``brewslm repro manifest``:

- ``train rerun``      → POST /api/projects/{pid}/training/runs/{eid}/rerun-from-manifest (P15)
- ``train clone``      → POST /api/projects/{pid}/training/runs/{eid}/clone (P15)
- ``train pause``      → POST /api/projects/{pid}/training/runs/{eid}/pause (P17)
- ``train resume``     → POST /api/projects/{pid}/training/runs/{eid}/resume (P17)
- ``train checkpoints``→ GET / POST under /api/projects/{pid}/training/runs/{eid}/... (P16)
                         (default = list, --promote-step = promote, --resume-from-step = fork)
- ``repro manifest``   → GET /api/projects/{pid}/training/runs/{eid}/manifest (P14)

Also asserts that:
- ``train clone`` accepts ``--config-overrides`` inline JSON and
  ``--config-overrides-file`` path, and ships the parsed overrides in the
  body.
- ``train checkpoints --promote-step`` and ``--resume-from-step`` are
  mutually exclusive (raises ValueError on combined usage).
- ``train`` and ``repro`` both require a subcommand (argparse `required=True`).
- The previously-default ``brewslm train`` autopilot one-click is reachable
  as ``brewslm train start`` and still POSTs to the autopilot endpoint.
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
    spec = importlib.util.spec_from_file_location("brewslm_cli_p19", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class Phase70TrainCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    # -- train rerun -------------------------------------------------------

    def test_train_rerun_posts_to_rerun_from_manifest(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "train",
                "rerun",
                "--project",
                "7",
                "--experiment-id",
                "42",
                "--run-name",
                "rerun of phi-2",
                "--description",
                "deterministic replay",
                "--json",
            ]
        )
        client = _FakeClient(lambda _call: {"experiment_id": 99, "status": "pending"})
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(
            call["path"], "/api/projects/7/training/runs/42/rerun-from-manifest"
        )
        self.assertEqual(call["json_body"], {"run_name": "rerun of phi-2", "description": "deterministic replay"})

    def test_train_rerun_omits_optional_fields_when_unset(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["train", "rerun", "--project", "1", "--experiment-id", "3"]
        )
        client = _FakeClient(lambda _call: {})
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        # No run_name / description authored → empty body.
        self.assertEqual(client.calls[0]["json_body"], {})

    # -- train clone -------------------------------------------------------

    def test_train_clone_inline_overrides_round_trip_in_body(self):
        parser = self.cli.build_parser()
        overrides = {"learning_rate": 3e-4, "num_epochs": 5}
        args = parser.parse_args(
            [
                "train",
                "clone",
                "--project",
                "5",
                "--experiment-id",
                "12",
                "--config-overrides",
                json.dumps(overrides),
                "--run-name",
                "lr-bump",
            ]
        )
        client = _FakeClient(lambda _call: {"experiment_id": 13, "status": "pending"})
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/api/projects/5/training/runs/12/clone")
        self.assertEqual(call["json_body"]["config_overrides"], overrides)
        self.assertEqual(call["json_body"]["run_name"], "lr-bump")

    def test_train_clone_reads_overrides_from_file(self):
        parser = self.cli.build_parser()
        overrides = {"seed": 99, "batch_size": 8}
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(overrides, handle)
            tmp_path = handle.name
        try:
            args = parser.parse_args(
                [
                    "train",
                    "clone",
                    "--project",
                    "5",
                    "--experiment-id",
                    "12",
                    "--config-overrides-file",
                    tmp_path,
                ]
            )
            client = _FakeClient()
            rc = args.func(args, client)
            self.assertEqual(rc, 0)
            self.assertEqual(client.calls[0]["json_body"]["config_overrides"], overrides)
        finally:
            Path(tmp_path).unlink()

    def test_train_clone_with_no_overrides_sends_null(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["train", "clone", "--project", "1", "--experiment-id", "1"]
        )
        client = _FakeClient()
        args.func(args, client)
        # Empty config-overrides → null in body so the API treats it as a no-op clone.
        self.assertIsNone(client.calls[0]["json_body"]["config_overrides"])

    # -- train pause / resume ---------------------------------------------

    def test_train_pause_posts_to_pause_endpoint(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["train", "pause", "--project", "9", "--experiment-id", "21"]
        )
        client = _FakeClient(
            lambda _call: {"experiment_id": 21, "status": "running", "pause_requested": True}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/api/projects/9/training/runs/21/pause")

    def test_train_resume_posts_to_resume_endpoint(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["train", "resume", "--project", "9", "--experiment-id", "21"]
        )
        client = _FakeClient(
            lambda _call: {"experiment_id": 21, "status": "running", "resumed_from_step": 80}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/api/projects/9/training/runs/21/resume")

    # -- train checkpoints (3 modes) --------------------------------------

    def test_train_checkpoints_default_lists_via_get(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["train", "checkpoints", "--project", "3", "--experiment-id", "7"]
        )
        client = _FakeClient(
            lambda _call: {"experiment_id": 7, "count": 4, "checkpoints": []}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "/api/projects/3/training/runs/7/checkpoints")

    def test_train_checkpoints_promote_step_posts_to_promote(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "train",
                "checkpoints",
                "--project",
                "3",
                "--experiment-id",
                "7",
                "--promote-step",
                "150",
            ]
        )
        client = _FakeClient(lambda _call: {"step": 150, "is_best": True})
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(
            call["path"], "/api/projects/3/training/runs/7/checkpoints/150/promote"
        )

    def test_train_checkpoints_resume_from_step_posts_to_fork(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "train",
                "checkpoints",
                "--project",
                "3",
                "--experiment-id",
                "7",
                "--resume-from-step",
                "200",
                "--resume-from-run-name",
                "fork at 200",
                "--resume-from-description",
                "ablation branch",
            ]
        )
        client = _FakeClient(lambda _call: {"experiment_id": 99, "status": "pending"})
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/api/projects/3/training/runs/7/resume-from/200")
        self.assertEqual(call["json_body"]["run_name"], "fork at 200")
        self.assertEqual(call["json_body"]["description"], "ablation branch")

    def test_train_checkpoints_mutually_exclusive_promote_and_resume(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "train",
                "checkpoints",
                "--project",
                "3",
                "--experiment-id",
                "7",
                "--promote-step",
                "100",
                "--resume-from-step",
                "200",
            ]
        )
        with self.assertRaises(ValueError):
            args.func(args, _FakeClient())

    # -- train start (existing autopilot one-click) -----------------------

    def test_train_start_posts_to_autopilot_one_click(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            [
                "train",
                "start",
                "--project",
                "1",
                "--autopilot",
                "--one-click",
                "--intent",
                "Build a support assistant.",
            ]
        )
        client = _FakeClient(lambda _call: {"started": True})
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        # Existing path kept its un-prefixed URL form (not /api/...).
        self.assertIn("training/autopilot/one-click-run", client.calls[0]["path"])

    # -- repro manifest ---------------------------------------------------

    def test_repro_manifest_gets_manifest_endpoint(self):
        parser = self.cli.build_parser()
        args = parser.parse_args(
            ["repro", "manifest", "--project", "4", "--experiment-id", "55"]
        )
        client = _FakeClient(
            lambda _call: {"experiment_id": 55, "git_sha": "abc", "env_digest": "xyz"}
        )
        rc = args.func(args, client)
        self.assertEqual(rc, 0)
        call = client.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "/api/projects/4/training/runs/55/manifest")

    # -- subparsers required ---------------------------------------------

    def test_train_without_subcommand_exits(self):
        parser = self.cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["train"])

    def test_repro_without_subcommand_exits(self):
        parser = self.cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["repro"])


if __name__ == "__main__":
    unittest.main()

"""Phase 64 tests — `brewslm eval` CLI subparser (priority.md P13).

Covers every documented subcommand:
- `eval generate`            → POST /evaluation/packs/generate (P9).
- `eval gold-set create`     → POST /gold/add (lazy-creates dataset).
- `eval gold-set add-row`    → P10 sampling OR legacy Q/A path (validated).
- `eval gold-set submit`     → POST /api/gold-sets/{id}/versions/lock (P10 + new).
- `eval label`               → PATCH /api/gold-sets/{id}/rows/{row_id} (P10).
- `eval run`                 → POST /evaluation/run-heldout.
- `eval compare`             → GETs both experiments' results + computes diffs.
- `eval clusters`            → GET /evaluation/{id}/failure-clusters (P12).
- `eval remediate`           → POST /evaluation/remediation-plans/generate.

All subcommands emit JSON and exit 0 on success; validation errors
surface as argparse/ValueError and exit non-zero.
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
    spec = importlib.util.spec_from_file_location("brewslm_cli_p13", script_path)
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


def _run_with_client(cli: ModuleType, argv: list[str], handler: Callable[[dict[str, Any]], Any]):
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    client = _FakeClient(handler)
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = args.func(args, client)
    return exit_code, buf.getvalue(), client.calls


class Phase64EvalCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli_module()

    # -- generate ---------------------------------------------------------

    def test_generate_posts_to_packs_generate_with_optional_ids(self):
        def handler(call):
            return {"pack_id": "slm.eval.pack.generated", "default_task_profile": "rag_qa"}

        exit_code, stdout, calls = _run_with_client(
            self.cli,
            [
                "eval", "generate",
                "--project", "7",
                "--blueprint-id", "3",
                "--dataset-id", "12",
                "--no-judge-rubric",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(calls[0]["path"], "/api/projects/7/evaluation/packs/generate")
        self.assertEqual(calls[0]["json_body"]["blueprint_id"], 3)
        self.assertEqual(calls[0]["json_body"]["dataset_id"], 12)
        self.assertNotIn("adapter_id", calls[0]["json_body"])
        self.assertEqual(calls[0]["json_body"]["include_judge_rubric"], False)
        self.assertIn("pack_id", json.loads(stdout))

    # -- gold-set create --------------------------------------------------

    def test_gold_set_create_seeds_via_legacy_gold_add(self):
        def handler(call):
            if call["path"].endswith("/gold/add"):
                return {"id": 101, "question": "Q", "answer": "A"}
            if call["path"].endswith("/gold/entries"):
                return [{"id": 101, "dataset_id": 55, "question": "Q"}]
            raise AssertionError(f"unexpected call: {call['path']}")

        exit_code, stdout, calls = _run_with_client(
            self.cli,
            [
                "eval", "gold-set", "create",
                "--project", "4",
                "--dataset-type", "gold_test",
                "--seed-question", "What is Paris?",
                "--seed-answer", "Capital of France.",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(calls[0]["path"], "/api/projects/4/gold/add")
        self.assertEqual(calls[0]["json_body"]["dataset_type"], "gold_test")
        self.assertEqual(calls[0]["json_body"]["question"], "What is Paris?")
        self.assertEqual(calls[1]["method"], "GET")
        self.assertEqual(calls[1]["path"], "/api/projects/4/gold/entries")
        payload = json.loads(stdout)
        self.assertEqual(payload["gold_set_id"], 55)
        self.assertEqual(payload["dataset_type"], "gold_test")

    # -- gold-set add-row -------------------------------------------------

    def test_gold_set_add_row_via_sample(self):
        def handler(call):
            return {
                "gold_set_id": 9,
                "version_id": 1,
                "version": 1,
                "requested": 5,
                "created": 5,
                "skipped_duplicates": 0,
                "strategy": "random",
                "rows": [],
            }

        exit_code, _, calls = _run_with_client(
            self.cli,
            [
                "eval", "gold-set", "add-row",
                "--project", "4",
                "--gold-set-id", "9",
                "--from-source-dataset-id", "22",
                "--target-count", "5",
                "--strategy", "random",
                "--seed", "42",
                "--reviewer-id", "17",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["path"], "/api/gold-sets/9/rows/sample")
        body = calls[0]["json_body"]
        self.assertEqual(body["source_dataset_id"], 22)
        self.assertEqual(body["target_count"], 5)
        self.assertEqual(body["strategy"], "random")
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["reviewer_id"], 17)

    def test_gold_set_add_row_stratified_without_field_errors(self):
        parser = self.cli.build_parser()
        args = parser.parse_args([
            "eval", "gold-set", "add-row",
            "--project", "4",
            "--gold-set-id", "9",
            "--from-source-dataset-id", "22",
            "--strategy", "stratified",
            "--target-count", "5",
        ])

        with self.assertRaises(ValueError) as ctx:
            args.func(args, _FakeClient(lambda call: {}))
        self.assertIn("stratify-by", str(ctx.exception))

    def test_gold_set_add_row_via_legacy_qa_path(self):
        def handler(call):
            return {"id": 42, "question": "Q", "answer": "A"}

        exit_code, _, calls = _run_with_client(
            self.cli,
            [
                "eval", "gold-set", "add-row",
                "--project", "4",
                "--dataset-type", "gold_dev",
                "--question", "Q",
                "--answer", "A",
                "--difficulty", "hard",
                "--hallucination-trap",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["path"], "/api/projects/4/gold/add")
        self.assertEqual(calls[0]["json_body"]["difficulty"], "hard")
        self.assertTrue(calls[0]["json_body"]["is_hallucination_trap"])

    def test_gold_set_add_row_with_no_paths_errors(self):
        parser = self.cli.build_parser()
        args = parser.parse_args([
            "eval", "gold-set", "add-row",
            "--project", "4",
        ])
        with self.assertRaises(ValueError) as ctx:
            args.func(args, _FakeClient(lambda call: {}))
        self.assertIn("--from-source-dataset-id", str(ctx.exception))

    # -- gold-set submit --------------------------------------------------

    def test_gold_set_submit_posts_to_versions_lock(self):
        def handler(call):
            return {
                "gold_set_id": 9,
                "version_id": 1,
                "version": 1,
                "status": "locked",
                "locked_at": "2026-04-24T12:00:00Z",
            }

        exit_code, stdout, calls = _run_with_client(
            self.cli,
            [
                "eval", "gold-set", "submit",
                "--gold-set-id", "9",
                "--notes", "approved-by-anurag",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(calls[0]["path"], "/api/gold-sets/9/versions/lock")
        self.assertEqual(calls[0]["json_body"]["notes"], "approved-by-anurag")
        self.assertEqual(json.loads(stdout)["status"], "locked")

    # -- label -----------------------------------------------------------

    def test_label_builds_patch_from_flags(self):
        def handler(call):
            return {"id": 11, "status": "approved"}

        exit_code, _, calls = _run_with_client(
            self.cli,
            [
                "eval", "label",
                "--gold-set-id", "9",
                "--row-id", "11",
                "--expected-json", '{"answer":"Paris"}',
                "--labels-json", '{"difficulty":"hard"}',
                "--rationale", "Because that is the capital.",
                "--status", "approved",
                "--reviewer-id", "33",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["method"], "PATCH")
        self.assertEqual(calls[0]["path"], "/api/gold-sets/9/rows/11")
        body = calls[0]["json_body"]
        self.assertEqual(body["expected"], {"answer": "Paris"})
        self.assertEqual(body["labels"], {"difficulty": "hard"})
        self.assertEqual(body["rationale"], "Because that is the capital.")
        self.assertEqual(body["status"], "approved")
        self.assertEqual(body["reviewer_id"], 33)

    def test_label_reviewer_minus_one_clears_assignment(self):
        def handler(call):
            return {}

        _, _, calls = _run_with_client(
            self.cli,
            [
                "eval", "label",
                "--gold-set-id", "9",
                "--row-id", "11",
                "--reviewer-id", "-1",
            ],
            handler,
        )
        self.assertIsNone(calls[0]["json_body"]["reviewer_id"])

    def test_label_without_any_fields_errors(self):
        parser = self.cli.build_parser()
        args = parser.parse_args([
            "eval", "label",
            "--gold-set-id", "9",
            "--row-id", "11",
        ])
        with self.assertRaises(ValueError) as ctx:
            args.func(args, _FakeClient(lambda call: {}))
        self.assertIn("No fields to update", str(ctx.exception))

    # -- run --------------------------------------------------------------

    def test_run_posts_to_run_heldout(self):
        def handler(call):
            return {"id": 501, "eval_type": "f1", "metrics": {"f1": 0.73}}

        exit_code, _, calls = _run_with_client(
            self.cli,
            [
                "eval", "run",
                "--project", "4",
                "--experiment-id", "7",
                "--dataset-name", "gold_dev",
                "--eval-type", "f1",
                "--max-samples", "25",
                "--max-new-tokens", "200",
                "--temperature", "0.1",
                "--model-path", "/tmp/model",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["path"], "/api/projects/4/evaluation/run-heldout")
        body = calls[0]["json_body"]
        self.assertEqual(body["experiment_id"], 7)
        self.assertEqual(body["dataset_name"], "gold_dev")
        self.assertEqual(body["eval_type"], "f1")
        self.assertEqual(body["max_samples"], 25)
        self.assertEqual(body["model_path"], "/tmp/model")
        self.assertNotIn("judge_model", body)  # f1 doesn't carry a judge

    # -- compare ----------------------------------------------------------

    def test_compare_computes_per_metric_delta(self):
        def handler(call):
            if call["path"].endswith("/results/11"):
                return [
                    {"id": 101, "eval_type": "f1", "metrics": {"f1": 0.60}, "pass_rate": 0.60},
                    {"id": 102, "eval_type": "llm_judge", "metrics": {"pass_rate": 0.70}, "pass_rate": 0.70},
                ]
            if call["path"].endswith("/results/22"):
                return [
                    {"id": 201, "eval_type": "f1", "metrics": {"f1": 0.78}, "pass_rate": 0.78},
                    {"id": 202, "eval_type": "llm_judge", "metrics": {"pass_rate": 0.82}, "pass_rate": 0.82},
                ]
            raise AssertionError(f"unexpected: {call['path']}")

        exit_code, stdout, calls = _run_with_client(
            self.cli,
            [
                "eval", "compare",
                "--project", "4",
                "--baseline", "11",
                "--candidate", "22",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 2)
        payload = json.loads(stdout)
        diffs = {d["eval_type"]: d for d in payload["diffs"]}
        f1_delta = next(m for m in diffs["f1"]["metrics"] if m["metric"] == "f1")
        self.assertAlmostEqual(f1_delta["baseline"], 0.60)
        self.assertAlmostEqual(f1_delta["candidate"], 0.78)
        self.assertAlmostEqual(f1_delta["delta"], 0.18, places=2)
        judge_delta = next(m for m in diffs["llm_judge"]["metrics"] if m["metric"] == "pass_rate")
        self.assertAlmostEqual(judge_delta["delta"], 0.12, places=2)

    # -- clusters ---------------------------------------------------------

    def test_clusters_passes_optional_params(self):
        def handler(call):
            return {"eval_result_id": 501, "clusters": [], "remediation_plans": []}

        exit_code, _, calls = _run_with_client(
            self.cli,
            [
                "eval", "clusters",
                "--project", "4",
                "--eval-result-id", "501",
                "--max-failures", "50",
                "--max-exemplars-per-cluster", "2",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["method"], "GET")
        self.assertEqual(calls[0]["path"], "/api/projects/4/evaluation/501/failure-clusters")
        self.assertEqual(calls[0]["params"]["max_failures"], 50)
        self.assertEqual(calls[0]["params"]["max_exemplars_per_cluster"], 2)

    # -- remediate --------------------------------------------------------

    def test_remediate_posts_to_remediation_plans_generate(self):
        def handler(call):
            return {"plan_id": "plan-xyz"}

        exit_code, stdout, calls = _run_with_client(
            self.cli,
            [
                "eval", "remediate",
                "--project", "4",
                "--experiment-id", "7",
                "--evaluation-result-id", "501",
                "--max-failures", "75",
            ],
            handler,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(calls[0]["path"], "/api/projects/4/evaluation/remediation-plans/generate")
        body = calls[0]["json_body"]
        self.assertEqual(body["experiment_id"], 7)
        self.assertEqual(body["evaluation_result_id"], 501)
        self.assertEqual(body["max_failures"], 75)
        self.assertIn("plan_id", json.loads(stdout))


if __name__ == "__main__":
    unittest.main()

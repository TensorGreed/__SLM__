"""Tests for the eval-aware experiment comparison service + API (E3).

Covers:
  - Per-metric delta direction logic (higher-is-better vs loss-style).
  - Failure-cluster diff (only_in_a / only_in_b / shared with delta).
  - Config diff with primary fields always present + other changes.
  - Winner + regressed verdict logic.
  - ``GET /api/projects/{id}/evaluation/compare`` endpoint contract.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "experiment_compare_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "experiment_compare_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import (
    EvalResult,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)
from app.models.project import Project, ProjectStatus
from app.services.experiment_comparison_service import (
    _decide_winner,
    _diff_configs,
    _diff_metrics,
    _direction,
    compare_experiments,
)


def _run(coro):
    return asyncio.run(coro)


class CompareUnitTests(unittest.TestCase):
    def test_tool_call_results_compare_on_one_pass_rate_definition(self):
        """P5 review: a cached tool-call base result stored the handler's pre-catalog rate in ``pass_rate``;
        Compare reads ``metrics.tool_call_pass_rate`` for both sides and labels the definition."""
        from types import SimpleNamespace

        from app.services import experiment_comparison_service as ecs

        old = SimpleNamespace(pass_rate=0.8, metrics={"scoring_mode": "tool_call", "tool_call_pass_rate": 0.5})
        new = SimpleNamespace(pass_rate=0.0, metrics={"scoring_mode": "tool_call", "tool_call_pass_rate": 0.0})
        other = SimpleNamespace(pass_rate=0.7, metrics={"scoring_mode": "field_match"})
        self.assertEqual(ecs.comparable_pass_rate(old), 0.5)
        self.assertEqual(ecs.comparable_pass_rate(new), 0.0)
        self.assertEqual(ecs.comparable_pass_rate(other), 0.7)
        self.assertIsNone(ecs.comparable_pass_rate(None))

    def test_the_stored_pass_rate_keeps_its_fallback_rule(self):
        """P5 re-review: a result with EM 0 / F1 0.4 stores 0.4, as it always did (only the display of a real
        0.0 changed, in the frontend)."""
        from app.services.evaluation_service import stored_pass_rate

        self.assertEqual(stored_pass_rate({"exact_match": 0.0, "f1": 0.4}), 0.4)
        self.assertEqual(stored_pass_rate({"pass_rate": 0.7, "exact_match": 0.2}), 0.7)
        self.assertEqual(stored_pass_rate({"exact_match": 0.0, "f1": 0.0}), 0.0)
        self.assertIsNone(stored_pass_rate({"exact_match": 0.0}))  # the UI shows 0% from the metrics

    """Pure-function helpers — no DB required."""

    def test_direction_for_higher_is_better_metric(self):
        # F1 is higher-is-better. b > a → improved.
        self.assertEqual(_direction("f1", 0.4, 0.6), "improved")
        self.assertEqual(_direction("f1", 0.6, 0.4), "regressed")
        self.assertEqual(_direction("f1", 0.5, 0.5), "unchanged")

    def test_direction_for_loss_style_metric(self):
        # eval_loss is lower-is-better. b < a → improved.
        self.assertEqual(_direction("eval_loss", 2.0, 1.2), "improved")
        self.assertEqual(_direction("eval_loss", 1.2, 2.0), "regressed")

    def test_direction_of_every_eval_metric_brewslm_emits(self):
        """P5b gap 2 (+ review): every numeric eval metric key the scorers write (grep of the producers in
        app/services: handlers, tool-call / SQL / structured-plan / multi-option / remediation / code-example /
        span-set / perplexity scoring, lift + judge folds), with its direction. Counts / versions are neutral."""
        from app.services.experiment_comparison_service import metric_direction

        higher = """
            accuracy macro_f1 micro_f1 f1 em exact_match precision recall pass_rate llm_judge_pass_rate
            safety_pass_rate tool_success_rate rouge_1 rouge_2 rouge_l bleu bleu_4 chrf groundedness
            all_options_pass_rate answer_span_extracted_rate attr_accuracy_mean extra_field_accuracy_mean
            citation_rate citations_valid_rate code_example_citations_valid_rate code_example_code_checks_pass_rate
            code_example_grounding_reference_pass_rate code_example_judge_pass_rate code_example_output_parses_rate
            code_example_pass_rate code_parses_rate correct cross_format_agreement_rate exec_match_rate
            fail_recall wrong_recall pass_recall pass_precision field_match_rate found_rate gold_pass_rate iou
            issue_resolved_rate issue_resolved_overall_rate json_validity_rate json_valid_rate kappa match_rate
            multi_option_pass_rate option_count_in_range_rate option_pass_rate probe_pass_rate user_probe_pass_rate
            refusal_correct_rate retrieval_found_rate rows_with_passing_option_rate
            rows_with_two_distinct_passing_rate rubric_mean_score rubric_pass_rate schema_compliance_rate
            schema_valid_rate strict_correct_rate structured_plan_pass_rate tool_args_match_rate
            tool_args_strict_match_rate tool_call_pass_rate tool_call_strict_pass_rate tool_call_valid_rate
            tool_selection_accuracy top_option_pass_rate span_set_f1 span_set_precision span_set_recall
            remediation_pass_rate patch_applies_rate judge_correct throughput_tps token_throughput_tps
            exact_agreement strict_agreement
        """.split()
        lower = """
            perplexity mean_nll bits_per_byte eval_loss train_loss wer cer ece brier_score mismatch_rate
            unparseable_rate out_of_taxonomy_rate new_findings_rate hallucination_rate hallucination_signal_rate
            malformed_rate malformed_count clean_chunk_fp_rate clean_chunk_fp_count fp_total fn_total
            invalid_finding_count error_rate failure_rate false_call_rate latency_ms average_latency_ms
            missing_rate toxicity_score truncation_fraction cost_usd llm_cost_usd detector_error_count
        """.split()
        neutral = "total evaluated_samples support prompt_format_version".split()
        for metric in higher:
            self.assertEqual(metric_direction(metric), "higher", metric)
            self.assertEqual(_direction(metric, 0.9, 0.8), "regressed", metric)
        for metric in lower:
            self.assertEqual(metric_direction(metric), "lower", metric)
            self.assertEqual(_direction(metric, 0.2, 0.1), "improved", metric)
        for metric in neutral:
            self.assertIsNone(metric_direction(metric), metric)
            self.assertEqual(_direction(metric, 1, 2), "changed", metric)
        # an unrecognised metric keeps the old loss-style default (FT-vs-FT Compare unchanged)
        self.assertEqual(metric_direction("some_custom_hook_metric"), "lower")
        # one source with the lift card's lower-is-better list
        from app.services.sft_lift_summary_service import _LOWER_IS_BETTER

        self.assertTrue(all(metric_direction(m) == "lower" for m in _LOWER_IS_BETTER))

    def test_direction_new_and_removed(self):
        # Metric appeared in B only / disappeared in B → new/removed.
        self.assertEqual(_direction("f1", None, 0.5), "new")
        self.assertEqual(_direction("f1", 0.5, None), "removed")
        self.assertEqual(_direction("f1", None, None), "unchanged")

    def test_diff_metrics_sorts_regressions_first(self):
        a = EvalResult(
            experiment_id=1, dataset_name="gold", eval_type="f1",
            metrics={"f1": 0.6, "exact_match": 0.5, "eval_loss": 1.0},
            pass_rate=0.6,
        )
        b = EvalResult(
            experiment_id=2, dataset_name="gold", eval_type="f1",
            metrics={"f1": 0.4, "exact_match": 0.7, "eval_loss": 0.8, "new_metric": 0.9},
            pass_rate=0.4,
        )
        rows = _diff_metrics(a, b)
        # First row is a regression — eval_loss IMPROVED (loss went down)
        # so the regression is f1. Verify the regression heads the list.
        self.assertEqual(rows[0]["metric_id"], "f1")
        self.assertEqual(rows[0]["direction"], "regressed")
        # The "new" metric ranks above "improved" / "unchanged".
        new_idx = next(i for i, r in enumerate(rows) if r["metric_id"] == "new_metric")
        improved_idx = next(i for i, r in enumerate(rows) if r["direction"] == "improved")
        self.assertLess(new_idx, improved_idx)

    def test_diff_metrics_handles_None_evals(self):
        # When one side has no eval, the other side's metrics show
        # as ``new``.
        rows = _diff_metrics(None, EvalResult(
            experiment_id=2, dataset_name="g", eval_type="f1",
            metrics={"f1": 0.5}, pass_rate=0.5,
        ))
        self.assertEqual(rows, [{
            "metric_id": "f1",
            "a_value": None,
            "b_value": 0.5,
            "delta": None,
            "direction": "new",
            "higher_is_better": True,
        }])

    def test_diff_configs_always_renders_primary_fields(self):
        a = {"base_model": "x", "learning_rate": 2e-4}
        b = {"base_model": "y", "learning_rate": 2e-4}
        rows = _diff_configs(a, b)
        # All primary fields present even when unchanged or missing.
        primary_fields = {r["field"] for r in rows if r["primary"]}
        self.assertIn("base_model", primary_fields)
        self.assertIn("learning_rate", primary_fields)
        self.assertIn("num_epochs", primary_fields)  # missing from both
        base_row = next(r for r in rows if r["field"] == "base_model")
        self.assertTrue(base_row["changed"])
        lr_row = next(r for r in rows if r["field"] == "learning_rate")
        self.assertFalse(lr_row["changed"])

    def test_diff_configs_appends_other_changed_fields(self):
        a = {"base_model": "x", "weird_knob": 1}
        b = {"base_model": "x", "weird_knob": 2}
        rows = _diff_configs(a, b)
        other = [r for r in rows if r["field"] == "weird_knob"]
        self.assertEqual(len(other), 1)
        self.assertFalse(other[0]["primary"])
        self.assertTrue(other[0]["changed"])

    def test_diff_configs_does_not_dump_unchanged_other_fields(self):
        a = {"base_model": "x", "other_unchanged": 42}
        b = {"base_model": "x", "other_unchanged": 42}
        rows = _diff_configs(a, b)
        # Primary fields land regardless; non-primary unchanged stay
        # hidden so the config diff doesn't drown in noise.
        self.assertFalse(any(r["field"] == "other_unchanged" for r in rows))

    def test_diff_configs_never_lists_platform_written_keys(self):
        # Phase 7i: ``_runtime`` carried absolute local paths onto Compare.
        a = {"base_model": "x", "_runtime": {"log_path": "/home/u/data/a.log"}, "_warm_start": {"a": 1}}
        b = {"base_model": "x", "_runtime": {"log_path": "/home/u/data/b.log"}, "_custom_flag": True,
             "lora_r": 8}
        rows = _diff_configs(a, b)
        fields = {r["field"] for r in rows}
        self.assertFalse(any(f.startswith("_") for f in fields), fields)
        self.assertIn("lora_r", fields)
        self.assertNotIn("/home/", repr(rows))

    def test_decide_winner_higher_pass_rate_wins(self):
        self.assertEqual(_decide_winner(0.4, 0.6), ("b", False))
        self.assertEqual(_decide_winner(0.6, 0.4), ("a", True))
        self.assertEqual(_decide_winner(0.5, 0.5), ("tie", False))

    def test_decide_winner_handles_missing_pass_rates(self):
        # Either side missing → still a sensible verdict.
        self.assertEqual(_decide_winner(None, 0.6), ("b", False))
        # A had a pass rate, B didn't → A wins, B regressed (failed
        # to produce a pass rate).
        self.assertEqual(_decide_winner(0.6, None), ("a", True))
        self.assertEqual(_decide_winner(None, None), ("unknown", False))


class CompareIntegrationTests(unittest.TestCase):
    """Wire end-to-end via the API + a real DB."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    _counter = 0

    def _seed_two_experiments(
        self,
        *,
        a_pass_rate: float,
        b_pass_rate: float,
        a_config: dict | None = None,
        b_config: dict | None = None,
        a_metrics: dict | None = None,
        b_metrics: dict | None = None,
    ) -> tuple[int, int, int]:
        """Insert a Project + two Experiments + one EvalResult each.
        Returns (project_id, exp_a_id, exp_b_id)."""
        CompareIntegrationTests._counter += 1
        tag = CompareIntegrationTests._counter

        async def _seed():
            async with async_session_factory() as session:
                proj = Project(
                    name=f"Compare Test Project #{tag}",
                    status=ProjectStatus.DRAFT,
                    selected_recipe={"recipe_id": "qa-sft"},
                )
                session.add(proj)
                await session.flush()

                exp_a = Experiment(
                    project_id=proj.id, name="A",
                    base_model="model-a", training_mode=TrainingMode.SFT,
                    config=a_config or {"base_model": "model-a", "learning_rate": 2e-4},
                    status=ExperimentStatus.COMPLETED,
                )
                exp_b = Experiment(
                    project_id=proj.id, name="B",
                    base_model="model-b", training_mode=TrainingMode.SFT,
                    config=b_config or {"base_model": "model-b", "learning_rate": 2e-4},
                    status=ExperimentStatus.COMPLETED,
                )
                session.add_all([exp_a, exp_b])
                await session.flush()

                session.add(EvalResult(
                    experiment_id=exp_a.id, dataset_name="gold_test",
                    eval_type="f1",
                    metrics=a_metrics or {"f1": a_pass_rate, "exact_match": a_pass_rate * 0.9},
                    pass_rate=a_pass_rate,
                ))
                session.add(EvalResult(
                    experiment_id=exp_b.id, dataset_name="gold_test",
                    eval_type="f1",
                    metrics=b_metrics or {"f1": b_pass_rate, "exact_match": b_pass_rate * 0.9},
                    pass_rate=b_pass_rate,
                ))
                await session.commit()
                return proj.id, exp_a.id, exp_b.id
        return _run(_seed())

    def test_compare_returns_regressed_True_when_b_pass_rate_lower(self):
        pid, a, b = self._seed_two_experiments(a_pass_rate=0.7, b_pass_rate=0.5)

        async def _compare():
            async with async_session_factory() as session:
                return await compare_experiments(
                    session, project_id=pid, exp_a_id=a, exp_b_id=b,
                )
        result = _run(_compare())
        self.assertTrue(result["regressed"])
        self.assertEqual(result["winner"], "a")
        # Both metrics regressed because b values are lower than a values.
        # Regressed rows sort to the top — every leading row is a regression.
        directions = [r["direction"] for r in result["metric_deltas"]]
        first_non_regressed = next(
            (i for i, d in enumerate(directions) if d != "regressed"),
            len(directions),
        )
        # f1 is in the regressed prefix.
        regressed_ids = {
            result["metric_deltas"][i]["metric_id"]
            for i in range(first_non_regressed)
        }
        self.assertIn("f1", regressed_ids)
        self.assertIn("exact_match", regressed_ids)

    def test_phase_7k_score_of_record_labels_token_f1_and_decides_the_winner_on_a_shared_key(self):
        """Phase 7k: the stored pass_rate column holds token F1 when exact match is 0 (A: 0.30) and exact match
        otherwise (B: 0.10), so the old rule crowned A. Each side now carries its score of record (token F1,
        a 0-1 mean) and the winner is decided on it (B: 0.60 > 0.30); ``eval_pass_rate`` is unchanged."""
        pid, a, b = self._seed_two_experiments(
            a_pass_rate=0.30, b_pass_rate=0.10,
            a_metrics={"exact_match": 0.0, "f1": 0.30}, b_metrics={"exact_match": 0.10, "f1": 0.60},
        )

        async def _compare():
            async with async_session_factory() as session:
                return await compare_experiments(session, project_id=pid, exp_a_id=a, exp_b_id=b)
        result = _run(_compare())
        self.assertEqual(result["a"]["eval_pass_rate"], 0.30)
        self.assertEqual(result["b"]["eval_pass_rate"], 0.10)
        self.assertEqual(result["a"]["score"]["key"], "f1")
        self.assertEqual(result["a"]["score"]["label"], "Token F1")
        self.assertEqual(result["a"]["score"]["scale"], "mean01")
        self.assertEqual(result["a"]["score"]["text"], "0.300")
        self.assertEqual(result["b"]["score"]["value"], 0.60)
        self.assertEqual(result["winner"], "b")
        self.assertFalse(result["regressed"])
        self.assertEqual(result["winner_metric"]["key"], "f1")

    def test_phase_7k_winner_falls_back_to_the_stored_pass_rate_without_a_shared_key(self):
        from types import SimpleNamespace

        from app.services.experiment_comparison_service import shared_score_key

        a = SimpleNamespace(metrics={"scoring_mode": "structured_plan", "structured_plan_pass_rate": 0.4}, eval_type="exact_match")
        b = SimpleNamespace(metrics={"perplexity": 12.0}, eval_type="perplexity")
        self.assertIsNone(shared_score_key(a, b))
        self.assertIsNone(shared_score_key(a, None))
        # Perplexity: lower is better, so a drop wins.
        pa = SimpleNamespace(metrics={"perplexity": 12.0}, eval_type="perplexity")
        pb = SimpleNamespace(metrics={"perplexity": 9.0}, eval_type="perplexity")
        self.assertFalse(shared_score_key(pa, pb)["higher_is_better"])

    def test_a_baseline_side_goes_on_a_and_reads_as_lift(self):
        """P5b gap 2: Compare(fine-tuned, baseline) puts the base on A; a fine-tune below its base is not
        "B regressed" and gets no rerun offer."""
        pid, ft, base = self._seed_two_experiments(
            a_pass_rate=0.9, b_pass_rate=0.0,
            b_config={"base_model": "model-b", "is_baseline": True},
        )

        async def _compare(a, b):
            async with async_session_factory() as session:
                return await compare_experiments(session, project_id=pid, exp_a_id=a, exp_b_id=b)
        result = _run(_compare(ft, base))
        self.assertEqual((result["a"]["experiment_id"], result["b"]["experiment_id"]), (base, ft))
        self.assertEqual(result["lift"]["swapped"], True)
        self.assertEqual(result["winner"], "b")
        self.assertFalse(result["regressed"])
        f1 = next(r for r in result["metric_deltas"] if r["metric_id"] == "f1")
        self.assertEqual(f1["direction"], "improved")
        # already base on A: not swapped; the fine-tune worse than base is still no rerun offer
        pid2, base2, ft2 = self._seed_two_experiments(
            a_pass_rate=0.9, b_pass_rate=0.5, a_config={"base_model": "model-a", "is_baseline": True},
        )
        pid = pid2
        result = _run(_compare(base2, ft2))
        self.assertEqual(result["lift"]["swapped"], False)
        self.assertEqual(result["winner"], "a")
        self.assertFalse(result["regressed"])

    def test_a_base_label_needs_a_baseline_side(self):
        """P5b gap 3: the base-model label is for base vs fine-tuned only (400 via the API otherwise)."""
        pid, a, b = self._seed_two_experiments(a_pass_rate=0.5, b_pass_rate=0.7)
        resp = self.client.get(f"/api/projects/{pid}/evaluation/compare",
                               params={"a": a, "b": b, "base_dataset_name": "eval_set:x@v1+y"})
        self.assertEqual(resp.status_code, 400)

    def test_compare_returns_regressed_False_when_b_wins(self):
        pid, a, b = self._seed_two_experiments(a_pass_rate=0.5, b_pass_rate=0.7)

        async def _compare():
            async with async_session_factory() as session:
                return await compare_experiments(
                    session, project_id=pid, exp_a_id=a, exp_b_id=b,
                )
        result = _run(_compare())
        self.assertFalse(result["regressed"])
        self.assertEqual(result["winner"], "b")

    def test_compare_surfaces_config_diff_for_changed_fields(self):
        pid, a, b = self._seed_two_experiments(
            a_pass_rate=0.6, b_pass_rate=0.5,
            a_config={"base_model": "smollm-135m", "learning_rate": 2e-4},
            b_config={"base_model": "qwen-0.5b", "learning_rate": 1e-4},
        )

        async def _compare():
            async with async_session_factory() as session:
                return await compare_experiments(
                    session, project_id=pid, exp_a_id=a, exp_b_id=b,
                )
        result = _run(_compare())
        bm_row = next(r for r in result["config_diff"] if r["field"] == "base_model")
        self.assertTrue(bm_row["changed"])
        self.assertEqual(bm_row["a_value"], "smollm-135m")
        self.assertEqual(bm_row["b_value"], "qwen-0.5b")
        lr_row = next(r for r in result["config_diff"] if r["field"] == "learning_rate")
        self.assertTrue(lr_row["changed"])

    def test_compare_404s_on_experiment_outside_project(self):
        # Build a project + an unrelated experiment in another project.
        pid_a, a, b = self._seed_two_experiments(a_pass_rate=0.5, b_pass_rate=0.5)
        pid_b, c, d = self._seed_two_experiments(a_pass_rate=0.5, b_pass_rate=0.5)
        # Compare a (project pid_a) with c (project pid_b) under pid_a.
        resp = self.client.get(
            f"/api/projects/{pid_a}/evaluation/compare?a={a}&b={c}"
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_compare_400s_when_a_equals_b(self):
        pid, a, b = self._seed_two_experiments(a_pass_rate=0.5, b_pass_rate=0.5)
        resp = self.client.get(
            f"/api/projects/{pid}/evaluation/compare?a={a}&b={a}"
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_compare_endpoint_returns_full_payload(self):
        pid, a, b = self._seed_two_experiments(a_pass_rate=0.7, b_pass_rate=0.4)
        resp = self.client.get(
            f"/api/projects/{pid}/evaluation/compare?a={a}&b={b}"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Shape contract.
        self.assertEqual(body["project_id"], pid)
        self.assertEqual(body["a"]["experiment_id"], a)
        self.assertEqual(body["b"]["experiment_id"], b)
        self.assertEqual(body["regressed"], True)
        self.assertEqual(body["winner"], "a")
        self.assertGreater(len(body["metric_deltas"]), 0)
        # cluster_diff payload present + structurally complete.
        # The cluster service may surface metric-level "failures" off
        # the pass_rate even without per-row data, so we don't assert
        # specific counts here — only that the keys are well-formed.
        self.assertIn("a_total", body["cluster_diff"])
        self.assertIn("b_total", body["cluster_diff"])
        self.assertIsInstance(body["cluster_diff"]["only_in_a"], list)
        self.assertIsInstance(body["cluster_diff"]["only_in_b"], list)
        self.assertIsInstance(body["cluster_diff"]["shared"], list)
        # config_diff always carries the primary fields, even when unchanged.
        primary_fields = {r["field"] for r in body["config_diff"] if r["primary"]}
        self.assertIn("base_model", primary_fields)
        self.assertIn("learning_rate", primary_fields)

    def test_compare_endpoint_handles_one_side_missing_eval(self):
        # Build a project where only A has an eval result; B never ran.
        CompareIntegrationTests._counter += 1
        tag = CompareIntegrationTests._counter

        async def _seed():
            async with async_session_factory() as session:
                proj = Project(
                    name=f"Compare One Eval Missing #{tag}",
                    status=ProjectStatus.DRAFT,
                    selected_recipe={"recipe_id": "qa-sft"},
                )
                session.add(proj)
                await session.flush()
                exp_a = Experiment(
                    project_id=proj.id, name="A", base_model="m",
                    training_mode=TrainingMode.SFT,
                    status=ExperimentStatus.COMPLETED,
                )
                exp_b = Experiment(
                    project_id=proj.id, name="B", base_model="m",
                    training_mode=TrainingMode.SFT,
                    status=ExperimentStatus.COMPLETED,
                )
                session.add_all([exp_a, exp_b])
                await session.flush()
                session.add(EvalResult(
                    experiment_id=exp_a.id, dataset_name="gold_test",
                    eval_type="f1", metrics={"f1": 0.5}, pass_rate=0.5,
                ))
                await session.commit()
                return proj.id, exp_a.id, exp_b.id
        pid, a, b = _run(_seed())

        resp = self.client.get(
            f"/api/projects/{pid}/evaluation/compare?a={a}&b={b}"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # A wins by default — B didn't produce a pass_rate.
        self.assertEqual(body["winner"], "a")
        self.assertTrue(body["regressed"])


if __name__ == "__main__":
    unittest.main()

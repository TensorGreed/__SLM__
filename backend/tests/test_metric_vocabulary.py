"""Phase 7k: one metric vocabulary (names, scales, score of record) shared by backend and frontend."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.services import metric_vocabulary as mv

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "frontend" / "src" / "generated" / "metricVocabulary.json"


class ScoreOfRecordTests(unittest.TestCase):
    def test_score_of_record_per_shape(self):
        cases = [
            # (metrics, eval_type, key, label, scale)
            ({"exact_match": 0.0, "f1": 0.184}, "exact_match", "f1", "Token F1", "mean01"),
            ({"exact_match": 0.1, "f1": 0.6, "judge_correct": 0.45}, "exact_match", "judge_correct",
             "Judge score (correct 1, partial 0.5)", "mean01"),
            ({"scoring_mode": "span_set", "exact_match": 0.12, "f1": 0.74}, "exact_match", "f1", "Span F1 (micro)", "mean01"),
            ({"scoring_mode": "finding_set", "f1": 0.5}, "exact_match", "f1", "Finding F1 (micro)", "mean01"),
            ({"scoring_mode": "field_match", "f1": 0.7, "exact_match": 0.1}, "exact_match", "f1", "Mean per-field F1", "mean01"),
            ({"scoring_mode": "tool_call", "tool_call_strict_pass_rate": 0.8, "tool_call_pass_rate": 0.9},
             "exact_match", "tool_call_strict_pass_rate", "Strict tool-call pass rate (extra arguments count as wrong)", "share"),
            ({"scoring_mode": "tool_call", "tool_call_pass_rate": 0.9, "exact_match": 0.9}, "exact_match",
             "tool_call_pass_rate", "Tool-call pass rate", "share"),
            ({"scoring_mode": "structured_plan", "structured_plan_pass_rate": 0.857}, "exact_match",
             "structured_plan_pass_rate", "Plan pass rate", "share"),
            ({"accuracy": 0.9, "macro_f1": 0.8, "f1": 0.8, "exact_match": 0.9}, "exact_match", "f1", "Macro F1", "mean01"),
            ({"per_test_type": {"x": 1}, "f1": 0.7, "pass_rate": 0.7}, "safety", "f1", "Pass rate", "share"),
            ({"pass_rate": 0.6}, "llm_judge", "pass_rate", "Judge pass rate", "share"),
            ({"perplexity": 12.3}, "perplexity", "perplexity", "Perplexity (lower is better)", "unbounded"),
        ]
        for metrics, eval_type, key, label, scale in cases:
            with self.subTest(key=key, label=label):
                record = mv.score_of_record(metrics, eval_type)
                self.assertEqual(record["key"], key)
                self.assertEqual(record["label"], label)
                self.assertEqual(record["scale"], scale)

    def test_the_metric_never_depends_on_the_value(self):
        base = mv.score_of_record({"scoring_mode": "span_set", "exact_match": 0.0, "f1": 0.31}, "exact_match")
        tuned = mv.score_of_record({"scoring_mode": "span_set", "exact_match": 0.12, "f1": 0.74}, "exact_match")
        self.assertEqual(base["key"], tuned["key"])

    def test_no_stored_key(self):
        record = mv.score_of_record({"total": 3}, "exact_match")
        self.assertIsNone(record["key"])
        self.assertEqual(record["text"], "—")

    def test_seed_group_aggregate_classification_keeps_its_shape(self):
        # Review fix: ``compute_variance_stats`` turns numbers into {mean, std, …}; the shape check must read them.
        from app.services.experiment_aggregation_service import compute_variance_stats

        seeds = [{"accuracy": a, "macro_f1": f, "f1": f, "exact_match": a} for a, f in ((0.7, 0.62), (0.8, 0.68))]
        agg = compute_variance_stats(seeds)
        self.assertIsInstance(agg["macro_f1"], dict)
        self.assertEqual(mv.result_shape(agg, "exact_match"), "classification")
        score = mv.score_of_record(agg, "exact_match", None)
        self.assertEqual(score["label"], "Macro F1")
        self.assertAlmostEqual(score["value"], 0.65)
        self.assertEqual(score["text"], "0.650")
        self.assertNotIn("free text", score["scored_as"])

    def test_judged_result_reads_graded_by_a_judge(self):
        record = mv.score_of_record({"exact_match": 0.1, "f1": 0.3, "judge_correct": 0.5}, "exact_match")
        self.assertEqual(record["scored_as"], "free text vs answer key, graded by a judge model")

    def test_format_value_and_delta_follow_the_scale(self):
        self.assertEqual(mv.format_value("exact_match", 0.857), "85.7%")
        self.assertEqual(mv.format_value("f1", 0.184), "0.184")
        self.assertEqual(mv.format_value("perplexity", 12.3456), "12.346")
        self.assertEqual(mv.format_delta("exact_match", 0.048), "+4.8 pts")
        self.assertEqual(mv.format_delta("f1", 0.048), "+0.048")
        self.assertEqual(mv.format_delta("f1", -0.048), "-0.048")
        self.assertEqual(mv.format_value("f1", None), "—")
        # Safety stores its pass rate under f1 too: a share there.
        self.assertEqual(mv.format_value("f1", 0.7, "safety"), "70.0%")

    def test_metric_name_inside_a_sentence(self):
        self.assertEqual(mv.metric_name("f1"), "F1")
        self.assertEqual(mv.metric_name("f1", "text"), "token F1")
        self.assertEqual(mv.metric_name("f1", "span_set"), "span F1 (micro)")
        self.assertEqual(mv.metric_name("exec_match_rate", "text"), "SQL execution match")
        self.assertEqual(mv.metric_name("unknown_metric"), "unknown metric")

    def test_unknown_keys_get_a_sensible_scale(self):
        self.assertEqual(mv.metric_scale("rubric_pass_rate"), "share")
        self.assertEqual(mv.metric_scale("dpo_loss"), "unbounded")
        self.assertEqual(mv.metric_scale("rouge_l"), "mean01")


class HeadlineOrderTests(unittest.TestCase):
    def test_headline_order_equals_the_lift_service(self):
        from app.services import sft_lift_summary_service as lift

        self.assertEqual(tuple(mv.PREFERRED_HEADLINE_METRICS), tuple(lift._PREFERRED_HEADLINE_METRICS))

    def test_every_shape_key_list_is_a_subsequence_of_the_headline_order(self):
        order = mv.PREFERRED_HEADLINE_METRICS
        for shape, keys in mv.SCORE_KEYS.items():
            positions = [order.index(k) for k in keys if k != "tool_call_strict_pass_rate"]
            self.assertEqual(positions, sorted(positions), shape)


class GeneratedJsonTests(unittest.TestCase):
    def test_frontend_json_matches_the_backend_tables(self):
        data = json.loads(GENERATED.read_text(encoding="utf-8"))
        payload = mv.export_payload()
        self.assertEqual(data, payload)
        self.assertEqual(data["score_keys"]["text"][:3], ["exec_match_rate", "judge_correct", "f1"])

    def test_generated_json_is_not_stale(self):
        spec = importlib.util.spec_from_file_location("_export_metric_vocabulary", ROOT / "scripts" / "export_metric_vocabulary.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(GENERATED.read_text(encoding="utf-8"), module.render(),
                         "run python3 scripts/export_metric_vocabulary.py")


class CoachWordingTests(unittest.TestCase):
    """Phase 7k: the Coach names the number it read (trigger unchanged: stored pass_rate < 0.90)."""

    def test_token_f1_only_result(self):
        from app.services.coach_service import _low_score_title, _score_today_text

        latest = SimpleNamespace(metrics={"exact_match": 0.0, "f1": 0.184}, eval_type="exact_match", pass_rate=0.184)
        self.assertEqual(_low_score_title(latest, 0.184), "Token F1 is 0.184 — below the 0.90 healthy threshold")
        self.assertEqual(_score_today_text(latest, 0.184), "Token F1 today is 0.184.")

    def test_share_result_keeps_percent_wording(self):
        from app.services.coach_service import _low_score_title, _score_today_text

        latest = SimpleNamespace(metrics={"exact_match": 0.5, "f1": 0.6}, eval_type="exact_match", pass_rate=0.5)
        self.assertEqual(_low_score_title(latest, 0.5), "Exact match is 50.0% — below the 90% healthy threshold")
        # The score of record of a free-text result is token F1, so "today" names it.
        self.assertEqual(_score_today_text(latest, 0.5), "Token F1 today is 0.600.")
        plan = SimpleNamespace(metrics={"scoring_mode": "structured_plan", "structured_plan_pass_rate": 0.5,
                                        "pass_rate": 0.5}, eval_type="exact_match", pass_rate=0.5)
        self.assertEqual(_score_today_text(plan, 0.5), "Plan pass rate today is 50.0%.")

    def test_trigger_threshold_unchanged(self):
        from app.services import coach_service

        self.assertEqual(coach_service.EVAL_PASS_RATE_HEALTHY, 0.90)


if __name__ == "__main__":
    unittest.main()

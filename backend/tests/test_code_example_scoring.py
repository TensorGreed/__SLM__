"""Opt-in held-out scoring of ``code_example`` (task -> code) rows in a ``remediation`` eval
(``runtime_config["remediation"]["score_code_examples"]``): the scorer, byte-identity with the setting
off, the judge, lift pairing / headline precedence, the base-reuse fingerprint, the settings API, the
manifest stamp and run-heldout end to end.

Neutral toy code only; code checks, model inference and the judge are stubbed where they decide."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.services import answer_judge_service as ajs
from app.services import code_example_scoring as ces
from app.services.data_adapter_service import map_record_with_adapter
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler
from app.services.unified_diff import make_unified_diff

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


CODE = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = " + uid\n    cur.execute(q)\n    return cur.fetchone()\n'
FIXED = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = ?"\n    cur.execute(q, (uid,))\n    return cur.fetchone()\n'
PASSAGES = [{"text": "Pass user input as bound parameters.", "source": "query-guidelines"}]
FINDING = {"label": "injection.query", "start_line": 11, "end_line": 12, "evidence": "q = ... + uid", "attrs": {}}
PATCH = make_unified_diff(CODE, FIXED, path="app/db.py", start_line=10)
REM_ROW = {"row_kind": "remediation", "code": CODE, "language": "python", "path": "app/db.py", "start_line": 10,
           "end_line": 13, "finding": FINDING, "passages": PASSAGES,
           "remediation_json": {"patch": PATCH, "rationale": "Bind it.", "citations": ["P1"]}}
EX_REF = "def get(cur, i):\n    cur.execute('SELECT 1 WHERE id = ?', (i,))\n"
EX1 = {"row_kind": "code_example", "task": "Read one row by id.", "language": "python", "passages": PASSAGES,
       "code_example_json": {"code": EX_REF, "rationale": "Bound.", "citations": ["P1"]}}
EX2 = {"row_kind": "code_example", "task": "Count rows.", "language": "python",
       "code_example_json": {"code": "def n(cur):\n    return cur.execute('SELECT COUNT(*) FROM t').fetchone()\n",
                             "rationale": "Count.", "citations": []}}
ADAPTER = {"remediation": {"extra_fields": {}, "project_id": 0, "retrieve_passages": False, "top_k": 3, "retrieval": None,
                           "label_titles": {}}}
GOOD_PATCH = json.dumps({"patch": PATCH, "rationale": "Bind it.", "citations": ["P1"]})


def prepared(row: dict) -> dict:
    return {**row, **map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config=ADAPTER)}


def schema(score_code_examples: bool | None = None, checks=("python_ast",)) -> dict:
    rem = {"checks": list(checks) if checks is not None else None, "retrieve_passages": False, "top_k": 3, "retrieval": None}
    if score_code_examples is not None:
        rem["score_code_examples"] = score_code_examples
    return {
        "scoring_mode": "remediation", "required": ["patch", "rationale", "citations"], "properties": {},
        "remediation": rem,
        "label_parents": {"injection.query": "injection"}, "label_titles": {"injection.query": "Query injection"},
    }


def handler_ctx(sch: dict):
    ctx = EvalContext(project_id=0, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
                      handler_id="structured_extraction", prepared_dir=Path("."), dataset_name="t",
                      manifest={"output_schema": sch})
    return StructuredExtractionHandler(), ctx


def predictions_for(rows: list[dict], answers: list[str], sch: dict) -> tuple[list[dict], object, object]:
    handler, ctx = handler_ctx(sch)
    built = handler.build_prompts([prepared(r) for r in rows], ctx)
    preds = []
    for b, a in zip(built, answers):
        p = {"prompt": b.prompt, "reference": b.reference, "prediction": a}
        for key in ("remediation_row", "code_example_row", "path", "language", "start_line"):
            if b.extras.get(key) is not None:
                p[key] = b.extras[key]
        preds.append(p)
    return preds, handler, ctx


def verdict(v: str | None, label: str = "stub:judge") -> dict:
    return {"verdict": v, "reason": f"{v} because", "label": label} if v else {"unscored": "no_judge_available", "label": None}


# ── off by default: byte-identical to before the setting existed ─────────


class OffByDefaultTests(unittest.TestCase):
    # sha256 of the fixture below scored by the code BEFORE this setting existed (commit 89eb0e5):
    # metrics (incl. the schema echo), every scored row dict, row_scores for pairing and _prediction_failed.
    GOLDEN_SHA256 = "d20e6625dd1a7c12e6d83bae99525fa0b991c952157772e5c4e93fdfc098de61"

    def run_fixture(self, sch: dict) -> dict:
        from app.services.evaluation_service import _prediction_failed, _row_scores_for_pairing

        answers = [GOOD_PATCH, json.dumps({"code": "def get(cur, i):\n    return 1\n", "rationale": "r", "citations": ["P1"]}),
                   "not json"]
        preds, handler, ctx = predictions_for([REM_ROW, EX1, EX2], answers, sch)
        metrics = handler.score(preds, ctx)
        return {
            "metrics": metrics,
            "rows": [{k: v for k, v in p.items() if not k.startswith("_")} for p in preds],
            "row_scores": _row_scores_for_pairing(preds),
            "failed": [_prediction_failed(p) for p in preds],
        }

    def test_flag_absent_matches_the_pre_change_output_byte_for_byte(self):
        out = self.run_fixture(schema())
        blob = json.dumps(out, sort_keys=True, default=str)
        self.assertEqual(hashlib.sha256(blob.encode()).hexdigest(), self.GOLDEN_SHA256)
        self.assertEqual(out["metrics"]["skipped_non_remediation_rows"], 2)
        self.assertEqual(out["row_scores"]["remediation_pass"], [1.0, None, None])
        self.assertNotIn("code_example_pass", out["row_scores"])
        self.assertFalse(any(k.startswith("code_example") for k in out["metrics"]))

    def test_flag_false_is_the_same_as_absent(self):
        absent = self.run_fixture(schema())
        off = self.run_fixture(schema(score_code_examples=False))
        off["metrics"]["schema"]["remediation"].pop("score_code_examples")
        self.assertEqual(json.dumps(absent, sort_keys=True, default=str), json.dumps(off, sort_keys=True, default=str))

    def test_prompt_is_the_same_with_the_flag_on(self):
        off, _, _ = predictions_for([EX1], ["{}"], schema())
        on, _, _ = predictions_for([EX1], ["{}"], schema(True))
        self.assertEqual((off[0]["prompt"], off[0]["reference"]), (on[0]["prompt"], on[0]["reference"]))
        self.assertNotIn("code_example_row", off[0])
        self.assertEqual(on[0]["code_example_row"]["task"], "Read one row by id.")
        self.assertEqual([p["id"] for p in on[0]["code_example_row"]["passages"]], ["P1"])


# ── the scorer ───────────────────────────────────────────────────────────


def fake_checks(code, *, language, check_ids=None, context=None):
    """A stub registry: python code containing BROKEN fails; language 'prose' has no check."""
    if language != "python":
        return {"ok": None, "results": [{"check_id": "stub", "ok": None, "messages": []}]}
    ok = "BROKEN" not in code
    return {"ok": ok, "results": [{"check_id": "stub", "ok": ok, "messages": [] if ok else ["syntax error"]}]}


class ScorerTests(unittest.TestCase):
    def score(self, rows, answers, verdicts, sch=None):
        preds, handler, ctx = predictions_for(rows, answers, sch or schema(True))
        for p, v in zip(preds, verdicts):
            if v is not ...:
                p["_code_example_judge"] = verdict(v)
        with patch("app.services.code_check_service.run_code_checks", side_effect=fake_checks):
            metrics = handler.score(preds, ctx)
        return metrics, preds

    def ans(self, code=EX_REF, citations=("P1",)):
        return json.dumps({"code": code, "rationale": "r", "citations": list(citations)})

    def test_every_row_flag(self):
        prose = {**EX2, "language": "prose"}
        rows = [EX1, EX1, EX1, EX1, EX1, EX2, prose, EX1]
        answers = [
            self.ans(),                          # 0 pass
            "not json",                          # 1 output does not parse
            self.ans(code="BROKEN"),             # 2 checks fail
            self.ans(citations=("P9",)),         # 3 citation not a given passage
            self.ans(citations=()),              # 4 no citation on a row with passages
            self.ans(citations=()),              # 5 no passages: citations not applicable -> pass
            self.ans(code="BROKEN"),             # 6 no check for the language: not checkable, judge decides
            self.ans(),                          # 7 judge partial -> fail
        ]
        verdicts = ["correct", "correct", "correct", "correct", "correct", "correct", "correct", "partial"]
        metrics, preds = self.score(rows, answers, verdicts)
        passes = [p["row_code_example_pass"] for p in preds]
        self.assertEqual(passes, [True, False, False, False, False, True, True, False])
        self.assertFalse(preds[1]["row_code_example_output_parses"])
        self.assertFalse(preds[2]["row_code_example_checks_pass"])
        self.assertEqual(preds[2]["row_code_example_check_messages"], ["stub: syntax error"])
        self.assertFalse(preds[3]["row_code_example_citations_valid"])
        self.assertFalse(preds[4]["row_code_example_citations_valid"])
        self.assertIsNone(preds[5]["row_code_example_citations_valid"])
        self.assertFalse(preds[6]["row_code_example_checkable"])
        self.assertIsNone(preds[6]["row_code_example_checks_pass"])
        self.assertFalse(preds[7]["row_code_example_judge_pass"])
        self.assertEqual(preds[7]["row_code_example_judge_verdict"], "partial")
        self.assertEqual(metrics["code_example_count"], 8)
        self.assertEqual(metrics["code_example_pass_rate"], round(3 / 8, 4))
        self.assertEqual(metrics["code_example_checkable_count"], 7)
        self.assertEqual(metrics["code_example_not_checkable_count"], 1)
        self.assertEqual(metrics["code_example_code_checks_pass_rate"], round(5 / 7, 4))
        self.assertEqual(metrics["code_example_output_parses_rate"], round(7 / 8, 4))
        self.assertEqual(metrics["code_example_citations_valid_rate"], 0.5)
        self.assertEqual(metrics["code_example_judge_pass_rate"], round(7 / 8, 4))
        self.assertEqual(metrics["code_example_judge_label"], "stub:judge")
        self.assertEqual(metrics["code_example_by_language"]["prose"],
                         {"scored": 1, "decided": 1, "passed": 1, "not_checkable": 1, "pass_rate": 1.0})
        self.assertEqual(metrics["code_example_by_language"]["python"]["scored"], 7)
        # only code examples: their pass rate is the headline, there is no remediation_pass_rate
        self.assertNotIn("remediation_pass_rate", metrics)
        self.assertEqual(metrics["exact_match"], round(3 / 8, 4))
        self.assertEqual((metrics["total"], metrics["correct"]), (8, 3))
        self.assertEqual(metrics["skipped_non_remediation_rows"], 0)

    def test_unscored_judge_never_passes_and_deterministic_failures_still_fail(self):
        rows = [EX1, EX1, EX1]
        metrics, preds = self.score(rows, [self.ans(), "nope", self.ans()], [None, None, ...])
        self.assertIsNone(preds[0]["row_code_example_pass"])
        self.assertEqual(preds[0]["row_code_example_judge_unscored"], "no_judge_available")
        self.assertFalse(preds[1]["row_code_example_pass"])  # does not parse: fails whatever the judge
        self.assertIsNone(preds[2]["row_code_example_pass"])
        self.assertEqual(preds[2]["row_code_example_judge_unscored"], "judge_not_run")
        self.assertEqual(metrics["code_example_decided_count"], 1)
        self.assertEqual(metrics["code_example_pass_rate"], 0.0)
        self.assertEqual(metrics["code_example_judge_unscored_count"], 3)
        self.assertEqual(metrics["code_example_judge_unscored_reasons"], {"judge_not_run": 1, "no_judge_available": 2})

    def test_reference_without_a_passing_check_is_not_checkable(self):
        bad_ref = {**EX1, "code_example_json": {"code": "BROKEN", "rationale": "r", "citations": ["P1"]}}
        _, preds = self.score([bad_ref], [self.ans(code="BROKEN")], ["correct"])
        self.assertFalse(preds[0]["row_code_example_checkable"])
        self.assertTrue(preds[0]["row_code_example_pass"])

    def test_mixed_split_reports_both_and_remediation_stays_the_headline(self):
        metrics, preds = self.score([REM_ROW, EX1], [GOOD_PATCH, self.ans()], [..., "wrong"])
        self.assertEqual(metrics["remediation_pass_rate"], 1.0)
        self.assertEqual(metrics["code_example_pass_rate"], 0.0)
        self.assertEqual((metrics["total"], metrics["exact_match"]), (1, 1.0))
        self.assertTrue(preds[0]["row_remediation_pass"])
        self.assertNotIn("row_remediation_pass", preds[1])

    def test_failure_and_pairing_rules(self):
        from app.services.evaluation_service import _prediction_failed, _row_scores_for_pairing, judge_decides_row

        rows = [{"prompt": "a", "reference": "r", "row_code_example_pass": True, "row_judge_score": 0.0},
                {"prompt": "b", "reference": "r", "row_code_example_pass": False, "row_judge_score": 1.0},
                {"prompt": "c", "reference": "r", "row_code_example_pass": None}]
        self.assertEqual([_prediction_failed(r) for r in rows], [False, True, False])
        self.assertEqual(_row_scores_for_pairing(rows)["code_example_pass"], [1.0, 0.0, None])
        self.assertFalse(judge_decides_row(rows[0]))


# ── the judge ────────────────────────────────────────────────────────────


class JudgeTests(unittest.TestCase):
    def run_judge(self, preds, *, resolved=None, rc=None, enabled=True):
        from app.services import evaluation_service as ev

        project = SimpleNamespace(runtime_config=rc or {})

        class Db:
            async def get(self, model, pk):
                return project

        async def go():
            with patch.object(ajs, "project_cache", lambda pid: ajs.AnswerJudgeCache(None)), \
                    patch.object(ajs, "resolve_answer_judge", side_effect=AssertionError("pinned judge expected")) \
                    if resolved is not None else patch.object(ajs, "resolve_answer_judge", return_value=None):
                return await ev._judge_code_example_predictions(
                    Db(), 1, StructuredExtractionHandler(), preds, judge=resolved, judge_enabled=enabled)

        return asyncio.run(go())

    def test_judge_reads_task_reference_and_prediction_with_the_code_example_rubric(self):
        seen = []

        async def judge(question, reference, prediction, instruction=None):
            seen.append((question, reference, prediction, instruction))
            if "unreadable" in prediction:
                return None
            return (1.0, "correct", "ok", 3) if "cur.execute" in prediction else (0.0, "wrong", "no", 3)

        answers = [json.dumps({"code": EX_REF, "rationale": "r", "citations": ["P1"]}),
                   json.dumps({"code": "x = 1", "rationale": "r", "citations": []}), "unreadable"]
        preds, _, _ = predictions_for([EX1, EX1, EX1, REM_ROW], answers + [GOOD_PATCH], schema(True))
        snap = self.run_judge(preds, resolved=ajs.ResolvedJudge(label="stub:judge", judge=judge))
        self.assertEqual(len(seen), 3)  # the remediation row is not sent
        question, reference, prediction, instruction = seen[0]
        self.assertIn("Task:\nRead one row by id.", question)
        self.assertIn("[P1] query-guidelines", question)
        self.assertTrue(reference.startswith("CODE:\ndef get"))
        self.assertEqual(instruction, ces.JUDGE_INSTRUCTION)
        self.assertEqual(preds[0]["_code_example_judge"]["verdict"], "correct")
        self.assertEqual(preds[1]["_code_example_judge"]["verdict"], "wrong")
        self.assertEqual(preds[2]["_code_example_judge"], {"unscored": "judge_no_verdict", "label": "stub:judge"})
        self.assertNotIn("_code_example_judge", preds[3])
        for p in preds:  # the answer judge's own per-row keys are not written by this judge
            self.assertNotIn("row_judge_verdict", p)
        self.assertEqual(snap["judge"], "stub:judge")
        self.assertEqual(snap["judged"], 2)

    def test_unavailable_or_disabled_judge_marks_rows_unscored(self):
        preds, _, _ = predictions_for([EX1], ["{}"], schema(True))
        self.assertEqual(self.run_judge(preds)["skipped"], "no_judge_available")
        self.assertEqual(preds[0]["_code_example_judge"]["unscored"], "no_judge_available")
        self.run_judge(preds, rc={"remediation": {"judge": False}})
        self.assertEqual(preds[0]["_code_example_judge"]["unscored"], "disabled_for_project")
        self.run_judge(preds, enabled=False)
        self.assertEqual(preds[0]["_code_example_judge"]["unscored"], "judge_disabled")

    def test_a_crashing_judge_leaves_rows_unscored(self):
        async def judge(*args):
            raise RuntimeError("boom")

        preds, _, _ = predictions_for([EX1], ["{}"], schema(True))
        self.run_judge(preds, resolved=ajs.ResolvedJudge(label="stub:judge", judge=judge))
        self.assertEqual(preds[0]["_code_example_judge"]["unscored"], "judge_no_verdict")

    def test_no_code_example_rows_means_no_judge(self):
        preds, _, _ = predictions_for([EX1, REM_ROW], ["{}", GOOD_PATCH], schema())
        self.assertIsNone(self.run_judge(preds))


# ── lift + fingerprint ───────────────────────────────────────────────────


class LiftAndFingerprintTests(unittest.TestCase):
    def test_headline_order_and_row_pairing(self):
        from app.services import sft_lift_summary_service as lift

        order = lift._PREFERRED_HEADLINE_METRICS
        self.assertEqual(order.index("code_example_pass_rate"), order.index("remediation_pass_rate") + 1)
        self.assertEqual(lift._ROW_SCORE_FOR_METRIC["code_example_pass_rate"], "code_example_pass")
        only = lift._compute_metric_lifts(
            lift._normalize_metrics({"code_example_pass_rate": 0.2, "code_example_count": 9, "exact_match": 0.2}),
            lift._normalize_metrics({"code_example_pass_rate": 0.6, "code_example_count": 9, "exact_match": 0.6}),
        )
        self.assertEqual(only[0]["metric_id"], "code_example_pass_rate")
        self.assertNotIn("code_example_count", [r["metric_id"] for r in only])
        mixed = lift._compute_metric_lifts(
            lift._normalize_metrics({"remediation_pass_rate": 0.1, "code_example_pass_rate": 0.2}),
            lift._normalize_metrics({"remediation_pass_rate": 0.3, "code_example_pass_rate": 0.6}),
        )
        self.assertEqual(mixed[0]["metric_id"], "remediation_pass_rate")

        def result(keys, passes):
            return SimpleNamespace(details={"row_scores": {"keys": keys, "code_example_pass": passes}})

        paired = lift._paired_row_scores(result(["a", "b", "c"], [0.0, None, 1.0]), result(["a", "b", "c"], [1.0, 1.0, 1.0]),
                                         "code_example_pass_rate")
        self.assertEqual(paired, ([0.0, 1.0], [1.0, 1.0]))

    def test_fingerprint_changes_only_when_on(self):
        from app.services.post_training_eval_service import _baseline_scoring_matches
        from app.services.remediation_detector import remediation_scoring_fingerprint

        def fp(rc):
            return asyncio.run(remediation_scoring_fingerprint(None, 0, rc, schema()))

        absent = fp({"remediation": {"retrieve_passages": False}})
        off = fp({"remediation": {"retrieve_passages": False, "score_code_examples": False}})
        on = fp({"remediation": {"retrieve_passages": False, "score_code_examples": True}})
        self.assertEqual(absent, off)
        self.assertNotEqual(absent, on)
        cached_off = SimpleNamespace(metrics={"scoring_mode": "remediation", "remediation_fingerprint": absent})
        self.assertTrue(_baseline_scoring_matches(cached_off, absent))
        self.assertFalse(_baseline_scoring_matches(cached_off, on))


# ── API, manifest stamp, run-heldout ─────────────────────────────────────


def run_async(coro_fn):
    async def go():
        async with async_session_factory() as db:
            return await coro_fn(db)

    return asyncio.run(go())


def new_project() -> int:
    pid = client.post("/api/projects", json={"name": f"ce-{time.time_ns()}", "description": "x"}).json()["id"]
    assert client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "code-remediation"}).status_code == 200
    return pid


class ApiAndStampTests(unittest.TestCase):
    def test_setting_round_trips_and_is_stamped_only_when_on(self):
        from app.services.dataset_service import _recipe_output_schema

        pid = new_project()
        url = f"/api/projects/{pid}/remediation-config"
        body = client.get(url).json()
        self.assertIsNone(body["config"]["score_code_examples"])
        self.assertFalse(body["effective"]["score_code_examples"])
        stamp_off = run_async(lambda db: _recipe_output_schema(db, pid))
        self.assertNotIn("score_code_examples", stamp_off["remediation"])

        put = client.put(url, json={"score_code_examples": True})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertTrue(put.json()["config"]["score_code_examples"])
        self.assertTrue(put.json()["effective"]["score_code_examples"])
        stamp_on = run_async(lambda db: _recipe_output_schema(db, pid))
        self.assertTrue(stamp_on["remediation"]["score_code_examples"])

        # PUT replaces the settings: leaving it out turns it off again
        self.assertFalse(client.put(url, json={"judge": True}).json()["effective"]["score_code_examples"])


class HeldoutIntegrationTests(unittest.TestCase):
    def seed(self, pid: int, rows: list[dict]) -> int:
        from app.services.dataset_service import _recipe_output_schema

        sch = run_async(lambda db: _recipe_output_schema(db, pid))
        sch["remediation"]["checks"] = ["python_ast"]
        sch["remediation"].pop("score_code_examples", None)  # prepared before the setting was turned on
        prep = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        prep.mkdir(parents=True, exist_ok=True)
        (prep / "manifest.json").write_text(json.dumps({"task_profile": "structured_extraction",
                                                        "adapter_id": "structured-extraction", "output_schema": sch}))
        mapped = [prepared(r) for r in rows]
        (prep / "test.jsonl").write_text("\n".join(json.dumps(r) for r in mapped) + "\n")

        async def add(db):
            db.add(Dataset(project_id=pid, name="test", dataset_type=DatasetType.TEST, file_path=str(prep / "test.jsonl"),
                           record_count=len(mapped)))
            exp = Experiment(project_id=pid, name="ce run", status=ExperimentStatus.COMPLETED, base_model="toy", config={},
                             output_dir=str(prep), completed_at=datetime.now(timezone.utc))
            db.add(exp)
            await db.commit()
            return exp.id

        return run_async(add)

    def run_heldout(self, pid, exp_id, answers):
        def infer(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True,
                  decoding="neutral"):
            return ([{"prompt": p["prompt"], "formatted_prompt": "", "reference": p["reference"], "prediction": answers[i],
                      "generated_tokens": 5, "latency_ms": 1.0} for i, p in enumerate(pairs)],
                    {"engine": "stub", "decoding": {"mode": decoding}})

        async def judge(question, reference, prediction, instruction=None):
            if instruction != ces.JUDGE_INSTRUCTION:  # the remediation patch judge
                return 1.0, "correct", "fine", 1
            return (1.0, "correct", "ok", 1) if "cur.execute" in prediction else (0.0, "wrong", "no", 1)

        resolved = ajs.ResolvedJudge(label="stub:judge", judge=judge)

        async def resolve(*args, **kwargs):
            return resolved

        with patch("app.services.evaluation_service._run_local_inference", infer), \
                patch.object(ajs, "resolve_answer_judge", resolve), \
                patch.object(ajs, "project_cache", lambda pid: ajs.AnswerJudgeCache(None)):
            resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", json={
                "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 20, "model_path": None})
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def ans(self, code):
        return json.dumps({"code": code, "rationale": "r", "citations": ["P1"]})

    def test_only_code_examples_with_the_live_setting_on(self):
        pid = new_project()
        exp_id = self.seed(pid, [EX1, EX1, EX1])
        # off (the default): every row skipped, exactly as before
        off = self.run_heldout(pid, exp_id, [self.ans(EX_REF), self.ans("x = 1"), "garbage"])
        self.assertEqual((off["metrics"]["total"], off["metrics"]["skipped_non_remediation_rows"]), (0, 3))
        self.assertNotIn("code_example_judge", off["metrics"])
        self.assertNotIn("row_code_example_pass", off["details"]["predictions_preview"][0])

        client.put(f"/api/projects/{pid}/remediation-config", json={"score_code_examples": True})
        on = self.run_heldout(pid, exp_id, [self.ans(EX_REF), self.ans("x = 1"), "garbage"])
        m = on["metrics"]
        self.assertNotIn("remediation_pass_rate", m)
        self.assertEqual(m["code_example_count"], 3)
        self.assertEqual(m["code_example_pass_rate"], round(1 / 3, 4))
        self.assertEqual(m["code_example_judge_label"], "stub:judge")
        self.assertEqual(m["code_example_judge"]["judged"], 3)
        self.assertEqual(on["pass_rate"], m["code_example_pass_rate"])
        self.assertTrue(m["remediation_fingerprint"])
        scores = on["details"]["row_scores"]
        self.assertEqual(scores["code_example_pass"], [1.0, 0.0, 0.0])
        preview = on["details"]["predictions_preview"]
        self.assertTrue(preview[0]["row_code_example_pass"])
        self.assertEqual(preview[1]["row_code_example_judge_verdict"], "wrong")
        self.assertFalse(preview[2]["row_code_example_output_parses"])
        self.assertEqual(preview[0]["row_kind"], "code_example")
        self.assertEqual(on["details"]["failed_count"], 2)

    def test_mixed_split_keeps_the_remediation_headline(self):
        pid = new_project()
        client.put(f"/api/projects/{pid}/remediation-config", json={"score_code_examples": True})
        exp_id = self.seed(pid, [REM_ROW, EX1])
        result = self.run_heldout(pid, exp_id, [GOOD_PATCH, self.ans(EX_REF)])
        m = result["metrics"]
        self.assertEqual(m["remediation_pass_rate"], 1.0)
        self.assertEqual(m["code_example_pass_rate"], 1.0)
        self.assertEqual(result["pass_rate"], m["remediation_pass_rate"])
        self.assertEqual(result["details"]["row_scores"]["remediation_pass"], [1.0, None])
        self.assertEqual(result["details"]["row_scores"]["code_example_pass"], [None, 1.0])


class SplitPreservationTests(unittest.TestCase):
    """Accepted synthetic code-example rows keep ``row_kind`` / ``passages`` / ``code_example_json`` through
    combine + split (incl. ``disjoint_by`` on a seed-document field), so held-out generation sends the prompt the
    training row used. (The predefined-split import does not take code-remediation rows at all yet: it never adds
    the recipe's adapter config, so it reports no recognised input column -- a pre-existing gap.)"""

    def rows(self, n: int) -> list[dict]:
        out = []
        for i in range(n):
            out.append({**EX1, "task": f"Read row {i} by id.", "seed_doc": f"doc-{i % 4}",
                        "code_example_json": {"code": EX_REF + f"# variant {i}\n", "rationale": "Bound.", "citations": ["P1"]}})
        return out

    def check_prepared(self, pid: int, prepared_rows: list[dict], manifest: dict) -> None:
        from app.services import code_remediation_prompt as crp

        self.assertTrue(prepared_rows)
        for r in prepared_rows:
            self.assertEqual(r["row_kind"], "code_example")
            self.assertEqual(r["language"], "python")
            self.assertTrue(r["task"].startswith("Read row"))
            self.assertEqual([p["id"] for p in r["passages"]], ["P1"])
            self.assertEqual(r["code_example_json"]["citations"], ["P1"])
            self.assertEqual(r["source_text"], crp.build_code_example_prompt_from_row(r))
        sch = dict(manifest["output_schema"])
        sch["remediation"] = {**sch["remediation"], "score_code_examples": True}
        handler, ctx = handler_ctx(sch)
        for r, built in zip(prepared_rows, handler.build_prompts(prepared_rows, ctx)):
            self.assertEqual(built.prompt, r["source_text"])
            self.assertEqual(built.reference, r["target_text"])
            self.assertEqual(built.extras["code_example_row"]["task"], r["task"])

    def test_random_split_with_disjoint_by_keeps_the_rows(self):
        pid = new_project()
        rows = self.rows(16)
        path = Path(settings.DATA_DIR) / "projects" / str(pid) / "synthetic_code_examples.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        async def add(db):
            db.add(Dataset(project_id=pid, name="synthetic", dataset_type=DatasetType.SYNTHETIC, file_path=str(path),
                           record_count=len(rows)))
            await db.commit()

        run_async(add)
        split = client.post(f"/api/projects/{pid}/dataset/split", json={
            "train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25, "dedup_rows": False, "disjoint_by": "seed_doc"})
        self.assertEqual(split.status_code, 200, split.text)
        manifest = split.json()
        prep = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        by_split = {name: [json.loads(x) for x in (prep / f"{name}.jsonl").read_text().splitlines() if x.strip()]
                    for name in ("train", "val", "test") if (prep / f"{name}.jsonl").exists()}
        self.assertEqual(sum(len(v) for v in by_split.values()), 16)
        self.assertTrue(by_split.get("test"))
        train_docs = {r["seed_doc"] for r in by_split["train"]}
        self.assertFalse(train_docs & {r["seed_doc"] for r in by_split["test"]})
        self.check_prepared(pid, [r for rows_ in by_split.values() for r in rows_], manifest)


if __name__ == "__main__":
    unittest.main()

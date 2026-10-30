"""Judge calibration V4: persisted eval predictions, blind eval-judge labels keyed by content, the
per-judge report (agreement, too strict / too lenient, weighted kappa), trust status and the
reference-model Job.

Neutral toy data; the model and the judges are stubbed (no model eval, no network). pytest-only
(the Jobs these tests start are awaited in tearDown).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services import eval_judge_calibration_service as ecal
from app.services import eval_predictions_service as preds
from app.services import judge_calibration_service as calib
from app.services import llm_budget_guard
from app.services.answer_judge_service import ResolvedJudge, VERDICT_SCORES, content_key

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


LONG = "the answer has enough words in it to be a long answer for the judge"


def rec(i, verdict, judge="judge:a", *, q=None, ref=None, pred=None, task="answer"):
    """One stored record judged on the verdict scale."""
    q = q or f"question {i}"
    ref = ref or f"reference {i} {LONG}"
    pred = pred or f"prediction {i}"
    return {"row_key": f"k{i}", "row_index": i, "prompt": q, "reference": ref, "prediction": pred, "correct": 1,
            "judge": {"scale": "verdict", "task": task, "label": judge, "verdict": verdict,
                      "score": VERDICT_SCORES[verdict], "reason": f"secret reason {i}", "content_key": content_key(q, ref, pred)}}


def rubric_rec(i, passed, judge="judge:a", sha="rubric-sha"):
    q, ref, pred = f"scenario {i}", f"gold plan {i}", f"plan {i}"
    return {"row_key": f"r{i}", "row_index": i, "prompt": f"prompt {i}", "reference": ref, "prediction": pred,
            "rubric": {"scale": "rubric", "label": judge, "passed": passed, "weighted": 3.0, "critical_failed": [],
                       "rubric_sha": sha, "content_key": content_key(q, ref, pred, "rubric"), "question": q}}


class _ProjectDir(unittest.TestCase):
    PID = 991_237

    def setUp(self):
        self.root = settings.DATA_DIR / "projects" / str(self.PID)
        shutil.rmtree(self.root, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def label(self, record, label, tier="human", labeller="human", scale="verdict"):
        block = record["judge"] if scale == "verdict" else record["rubric"]
        ecal.save_labels(self.PID, scale, tier, {block["content_key"]: ecal.label_entry(label, labeller=labeller)})


# ── maths ───────────────────────────────────────────────────────────────


class MathTests(unittest.TestCase):
    # Judge strata: correct pop 8 (4 labelled → weight 2), partial pop 2 (2 → 1), wrong pop 3 (3 → 1).
    ITEMS = (
        [{"verdict": "correct", "label": lab, "weight": 2.0} for lab in ("correct", "correct", "partial", "wrong")]
        + [{"verdict": "partial", "label": lab, "weight": 1.0} for lab in ("correct", "partial")]
        + [{"verdict": "wrong", "label": lab, "weight": 1.0} for lab in ("wrong", "wrong", "partial")]
    )

    def test_verdict_metrics_hand_computed(self):
        m = ecal.verdict_metrics(self.ITEMS, "verdict")
        # total weight 13
        self.assertAlmostEqual(m["exact_agreement"]["value"], round(7 / 13, 4))
        self.assertAlmostEqual(m["strict_agreement"]["value"], round(8 / 13, 4))
        self.assertAlmostEqual(m["strict_agreement"]["n_eff"], round(169 / 21, 2))
        self.assertAlmostEqual(m["pass_precision"]["value"], 0.5)          # 4 / 8
        self.assertAlmostEqual(m["pass_recall"]["value"], 0.8)             # 4 / 5
        self.assertAlmostEqual(m["wrong_recall"]["value"], 0.5)            # 2 / 4
        self.assertAlmostEqual(m["too_strict"]["value"], round(1 / 13, 4))   # partial where labeller said correct
        self.assertAlmostEqual(m["too_lenient"]["value"], round(4 / 13, 4))  # correct where labeller said partial / wrong
        # Linear-weighted kappa: Po = 9/13, Pe = 91/169 = 7/13 → (2/13) / (6/13) = 1/3
        self.assertAlmostEqual(m["kappa"], round(1 / 3, 4))
        self.assertEqual(m["kappa_kind"], "linear_weighted")
        self.assertEqual(m["confusion"]["correct"], {"correct": 2, "partial": 1, "wrong": 1})
        self.assertEqual(m["by_label"], {"correct": 3, "partial": 3, "wrong": 3})
        self.assertTrue(m["insufficient"])  # n 9 < 20, 3 labelled correct < 5
        low, high = calib.wilson(8 / 13, 169 / 21)
        self.assertAlmostEqual(m["strict_agreement"]["low"], round(low, 4))
        self.assertAlmostEqual(m["strict_agreement"]["high"], round(high, 4))

    def test_rubric_metrics_are_cohen(self):
        items = ([{"verdict": "pass", "label": "pass", "weight": 1.0}] * 3 + [{"verdict": "pass", "label": "fail", "weight": 1.0}]
                 + [{"verdict": "fail", "label": "fail", "weight": 1.0}, {"verdict": "fail", "label": "pass", "weight": 1.0}])
        m = ecal.verdict_metrics(items, "rubric")
        # a=1 both fail, b=1 judge fail / label pass, c=1 judge pass / label fail, d=3 → po 4/6, pe 20/36 → 0.25
        self.assertAlmostEqual(m["kappa"], 0.25)
        self.assertAlmostEqual(m["kappa"], round(calib.cohen_kappa(1, 1, 1, 3), 4))
        self.assertEqual(m["kappa_kind"], "cohen")
        self.assertAlmostEqual(m["too_strict"]["value"], round(1 / 6, 4))
        self.assertAlmostEqual(m["too_lenient"]["value"], round(1 / 6, 4))
        self.assertAlmostEqual(m["wrong_recall"]["value"], 0.5)  # fail-recall
        self.assertAlmostEqual(m["exact_agreement"]["value"], round(4 / 6, 4))

    def test_weighted_kappa_edges(self):
        self.assertIsNone(ecal.weighted_kappa({}, ecal.ORDER["verdict"]))
        self.assertIsNone(ecal.weighted_kappa({("correct", "correct"): 5}, ecal.ORDER["verdict"]))
        self.assertAlmostEqual(ecal.weighted_kappa({("correct", "correct"): 3, ("wrong", "wrong"): 3}, ecal.ORDER["verdict"]), 1.0)


class TrustTests(unittest.TestCase):
    @staticmethod
    def entry(strict=(0.9, 0.8, 0.95), ts=(0.02, 0.0, 0.1), tl=(0.02, 0.0, 0.1), n=40, insufficient=False):
        r = lambda t: {"value": t[0], "low": t[1], "high": t[2]}  # noqa: E731
        return {"judge": "j", "n": n, "exact_agreement": r((0.84, 0.7, 0.9)), "strict_agreement": r(strict),
                "too_strict": r(ts), "too_lenient": r(tl), "kappa": 0.6, "insufficient": insufficient,
                "insufficient_reasons": ["few"] if insufficient else []}

    def status(self, entry, source="human", reports=None):
        reports = reports or [{"tier": "human", "judges": [entry]}]
        return ecal.judge_trust("j", reports, source)

    def test_thresholds(self):
        self.assertEqual(ecal.judge_trust("j", [], "human")["status"], "uncalibrated")
        self.assertEqual(self.status(self.entry(n=0))["status"], "uncalibrated")
        self.assertEqual(self.status(self.entry(insufficient=True))["status"], "insufficient")
        ok = self.status(self.entry())
        self.assertEqual((ok["status"], ok["agreement"], ok["n"], ok["tier"]), ("calibrated_ok", 0.84, 40, "human"))
        # upper bound of strict agreement just under / at the floor
        low_agree = self.status(self.entry(strict=(0.5, 0.4, ecal.STRICT_AGREEMENT_FLOOR - 0.01)))
        self.assertEqual((low_agree["status"], low_agree["flags"]), ("below_floor", ["low_agreement"]))
        self.assertEqual(self.status(self.entry(strict=(0.6, 0.5, ecal.STRICT_AGREEMENT_FLOOR)))["status"], "calibrated_ok")
        # too strict / lenient: the LOWER bound over the ceiling flags; a high point estimate with a wide interval does not
        strict = self.status(self.entry(ts=(0.3, ecal.TOO_STRICT_CEILING + 0.01, 0.4)))
        self.assertEqual((strict["status"], strict["flags"]), ("below_floor", ["too_strict"]))
        self.assertEqual(self.status(self.entry(ts=(0.3, ecal.TOO_STRICT_CEILING, 0.5)))["status"], "calibrated_ok")
        lenient = self.status(self.entry(tl=(0.3, ecal.TOO_LENIENT_CEILING + 0.05, 0.4)))
        self.assertEqual(lenient["flags"], ["too_lenient"])

    def test_both_tiers_worse_wins(self):
        reports = [{"tier": "human", "judges": [self.entry()]},
                   {"tier": "reference_model", "judges": [self.entry(ts=(0.4, 0.3, 0.5))]}]
        t = ecal.judge_trust("j", reports, "both")
        self.assertEqual((t["status"], t["tier"]), ("below_floor", "reference_model"))
        none = ecal.judge_trust("j", [{"tier": "human", "judges": []}, {"tier": "reference_model", "judges": []}], "both")
        self.assertEqual((none["status"], none["tier"]), ("uncalibrated", None))


# ── persisted predictions ───────────────────────────────────────────────


class RecordTests(unittest.TestCase):
    def test_records_hold_every_row_and_only_differing_views(self):
        predictions = [
            {"prompt": "q1", "reference": "r1", "prediction": "p1", "row_judge_verdict": "wrong", "row_judge_score": 0.0,
             "row_judge_reason": "nope", "row_exact_match": 0.0},
            {"prompt": "q2", "reference": "r2", "prediction": "p2"},  # unjudged: still stored
            {"prompt": "last", "reference": "r3", "prediction": "p3", "row_judge_verdict": "correct", "row_judge_score": 1.0,
             "history": [{"role": "user", "content": "first"}, {"role": "user", "content": "last"}]},
        ]
        records, not_stored = preds.build_records(predictions, answer_label="judge:a", correct=[0, 0, 1])
        self.assertEqual((len(records), not_stored), (3, 0))
        first, second, chat = records
        self.assertEqual(first["row_key"], preds.row_key(predictions[0]))
        self.assertEqual(first["judge"]["label"], "judge:a")
        self.assertEqual(first["judge"]["content_key"], content_key("q1", "r1", "p1"))
        self.assertNotIn("question", first["judge"])  # the judge read the row's own texts: not duplicated
        self.assertEqual((first["correct"], first["row_exact_match"]), (0, 0.0))
        self.assertNotIn("judge", second)
        self.assertNotIn("prompt", chat)
        self.assertEqual(chat["judge"]["question"], "user: first\nuser: last")
        self.assertEqual(chat["judge"]["content_key"], content_key("user: first\nuser: last", "r3", "p3"))

    def test_rubric_and_code_example_records(self):
        plan = {"prompt": "p", "reference": "gold", "prediction": "cand", "plan_row": {"scenario": "S", "passages": [{"id": "a", "text": "x" * 5000}]},
                "row_rubric_passed": False, "row_rubric_weighted": 2.0, "row_rubric_critical_failed": ["safety"]}
        (r,), _ = preds.build_records([plan], rubric_label="judge:r", rubric_sha="abc")
        self.assertEqual((r["rubric"]["passed"], r["rubric"]["rubric_sha"], r["rubric"]["question"]), (False, "abc", "S"))
        self.assertEqual(r["rubric"]["content_key"], content_key("S", "gold", "cand", "rubric"))
        self.assertEqual(len(r["rubric"]["passages"][0]["text"]), preds.PASSAGE_CAP)
        ce = {"prompt": "p", "reference": '{"code": "a"}', "prediction": '{"code": "b"}',
              "code_example_row": {"task": "do x", "language": "python", "passages": []},
              "_code_example_judge": {"verdict": "partial", "reason": "r", "label": "judge:c"}}
        (c,), _ = preds.build_records([ce], parse_json=json.loads)
        self.assertEqual((c["judge"]["task"], c["judge"]["label"], c["judge"]["verdict"]), ("code_example", "judge:c", "partial"))
        self.assertIn("CODE:\nb", c["judge"]["prediction"])

    def test_safe_write_never_raises(self):
        with patch.object(preds, "write_file", side_effect=OSError("disk full")):
            self.assertIsNone(preds.safe_write(1, 2, [{"prompt": "q"}]))
        with patch.object(preds, "build_records", side_effect=RuntimeError("boom")):
            self.assertIsNone(preds.safe_write(1, 2, [{"prompt": "q"}]))


class _Judge:
    """correct when the prediction equals the reference, else wrong."""

    async def __call__(self, question, reference, prediction):
        verdict = "correct" if prediction.strip() == reference.strip() else "wrong"
        return VERDICT_SCORES[verdict], verdict, "stub reason", 0


def _fake_inference(model_ref, pairs, *a, **kw):
    preds_out = []
    for i, p in enumerate(pairs):
        preds_out.append({"prompt": p["prompt"], "reference": p["reference"],
                          "prediction": p["reference"] if i % 2 == 0 else "something else", "latency_ms": 1.0})
    return preds_out, {"engine": "stub"}


def run_eval(rows, **kw):
    """A held-out eval on a throwaway engine (model + judge stubbed), in one event loop."""

    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'v4.db'}", future=True)
            sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            import app.models  # noqa: F401
            from app.database import Base
            from app.models.dataset import Dataset, DatasetType
            from app.models.experiment import Experiment, ExperimentStatus
            from app.models.project import Project
            from app.services import evaluation_service as ev

            try:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                heldout = root / "test.jsonl"
                heldout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
                async with sf() as db:
                    project = Project(id=20_000 + uuid4().int % 10**9, name=f"v4-{uuid4().hex[:8]}", description="x",
                                      base_model_name="toy/model")
                    db.add(project)
                    await db.flush()
                    exp = Experiment(project_id=project.id, name="e", description="", status=ExperimentStatus.COMPLETED,
                                     base_model="toy/model", output_dir=str(root), config={"task_type": "qa"})
                    db.add(exp)
                    db.add(Dataset(project_id=project.id, name="t", dataset_type=DatasetType.TEST, file_path=str(heldout),
                                   record_count=len(rows)))
                    await db.commit()
                    pid, eid = project.id, exp.id
                with patch("app.services.evaluation_service._run_local_inference", _fake_inference), \
                        patch("app.services.evaluation_service._resolve_model_reference", return_value="toy"):
                    async with sf() as db:
                        result = await ev.run_heldout_evaluation(db=db, project_id=pid, experiment_id=eid, dataset_name="test",
                                                                 eval_type="exact_match", max_samples=100, **kw)
                        await db.commit()
                return pid, result
            finally:
                await engine.dispose()

    return asyncio.run(scenario())


class EndToEndWriteTests(unittest.TestCase):
    ROWS = [{"question": f"What about item {i}?", "answer": f"Item {i} is {LONG}"} for i in range(4)]

    def tearDown(self):
        for pid in getattr(self, "_pids", []):
            shutil.rmtree(settings.DATA_DIR / "projects" / str(pid), ignore_errors=True)

    def _run(self, **kw):
        pid, result = run_eval(self.ROWS, **kw)
        self._pids = [*getattr(self, "_pids", []), pid]
        return pid, result

    def test_every_row_is_written_and_readable(self):
        pid, result = self._run(answer_judge=ResolvedJudge(label="stub:judge", judge=_Judge()))
        pointer = result.details["eval_predictions_file"]
        self.assertEqual((pointer["path"], pointer["count"], pointer["judged"]), (f"eval_predictions/{result.id}.jsonl", 4, 4))
        records = preds.read_records(pid, result.id)
        self.assertEqual([r["judge"]["verdict"] for r in records], ["correct", "wrong", "correct", "wrong"])
        self.assertEqual([r["row_key"] for r in records], result.details["row_scores"]["keys"])
        self.assertEqual([r["correct"] for r in records], result.details["row_scores"]["correct"])
        self.assertEqual(records[1]["judge"]["label"], "stub:judge")
        self.assertEqual(records[1]["judge"]["reason"], "stub reason")
        page = preds.read_page(pid, result.id, offset=1, limit=2)
        self.assertEqual((page["available"], page["total"], [r["row_index"] for r in page["rows"]]), (True, 4, [1, 2]))
        # existing details are unchanged in shape
        self.assertIn("row_predictions_file", result.details)
        self.assertEqual(len(result.details["predictions_preview"]), 4)

    def test_eval_never_fails_on_a_write_error(self):
        with patch.object(preds, "write_file", side_effect=OSError("read-only")):
            pid, result = self._run(answer_judge=ResolvedJudge(label="stub:judge", judge=_Judge()))
        self.assertNotIn("eval_predictions_file", result.details)
        self.assertIn("row_predictions_file", result.details)
        self.assertEqual(result.metrics["judge"]["judged"], 4)
        with patch("app.services.eval_predictions_service.build_records", side_effect=RuntimeError("bug")):
            _, result = self._run(answer_judge=ResolvedJudge(label="stub:judge", judge=_Judge()))
        self.assertNotIn("eval_predictions_file", result.details)

    def test_absent_file_is_harmless(self):
        self.assertIsNone(preds.read_records(424242, 1))
        self.assertEqual(preds.read_page(424242, 1)["available"], False)
        self.assertEqual(list(preds.iter_project_records(424242)), [])
        self.assertEqual(ecal.population(424242, "verdict"), {})
        self.assertEqual(ecal.trust_for(424242, "j", {})["status"], "uncalibrated")


# ── population, sampling, labels ────────────────────────────────────────


class PopulationTests(_ProjectDir):
    def test_content_keys_are_stable_across_eval_results(self):
        a = [rec(1, "correct"), rec(2, "wrong")]
        b = [rec(1, "partial"), rec(3, "correct", judge="judge:b")]  # the same answer judged again later
        preds.write_file(self.PID, 10, a)
        preds.write_file(self.PID, 11, b)
        pop = ecal.population(self.PID, "verdict")
        self.assertEqual(len(pop), 3)
        ck = a[0]["judge"]["content_key"]
        self.assertEqual(ck, b[0]["judge"]["content_key"])
        self.assertEqual(pop[ck]["verdicts"], {"judge:a": "partial"})  # the newest verdict wins
        self.assertEqual(pop[ck]["eval_result_ids"], [10, 11])
        self.label(a[0], "correct")
        _, rows = ecal.sample(self.PID, "verdict", "human", 20, judge="judge:a")
        self.assertNotIn(ck, [r["content_key"] for r in rows])  # one label covers both evals

    def test_sample_is_blind_stratified_shuffled_and_excludes_done(self):
        records = ([rec(i, "correct") for i in range(10)] + [rec(i, "partial") for i in range(10, 13)]
                   + [rec(i, "wrong") for i in range(13, 16)] + [rec(99, "wrong", judge="judge:other")])
        preds.write_file(self.PID, 5, records)
        judge, rows = ecal.sample(self.PID, "verdict", "human", 6, judge="judge:a", seed=1)
        pop = ecal.population(self.PID, "verdict")
        verdicts = [pop[r["content_key"]]["verdicts"]["judge:a"] for r in rows]
        self.assertEqual(sorted(verdicts), ["correct", "correct", "partial", "partial", "wrong", "wrong"])
        self.assertNotIn(records[-1]["judge"]["content_key"], [r["content_key"] for r in rows])
        self.assertEqual(judge, "judge:a")
        # seeded: same seed same order; the order is not grouped by stratum for every seed
        self.assertEqual(rows, ecal.sample(self.PID, "verdict", "human", 6, judge="judge:a", seed=1)[1])
        orders = {tuple(pop[r["content_key"]]["verdicts"]["judge:a"]
                        for r in ecal.sample(self.PID, "verdict", "human", 6, judge="judge:a", seed=s)[1]) for s in range(6)}
        self.assertGreater(len(orders), 1)
        blind = [ecal.render_blind(pop[r["content_key"]], "verdict") for r in rows]
        text = json.dumps(blind)
        for banned in ("\"verdict\":", "verdicts", "secret reason", "judge:a", "score", "\"correct\"", "eval_result"):
            self.assertNotIn(banned, text)
        self.assertEqual(set(blind[0]), {"content_key", "scale", "question", "reference", "prediction"})
        # labelled and skipped answers are not offered again
        self.label(records[0], "correct")
        ecal.save_labels(self.PID, "verdict", "human", {records[10]["judge"]["content_key"]: ecal.skip_entry(reason="unsure", labeller="human")})
        _, rows = ecal.sample(self.PID, "verdict", "human", 100, judge="judge:a")
        keys = {r["content_key"] for r in rows}
        self.assertEqual(len(keys), 14)
        self.assertNotIn(records[0]["judge"]["content_key"], keys)
        self.assertNotIn(records[10]["judge"]["content_key"], keys)
        # the reference tier still offers them
        self.assertEqual(len(ecal.sample(self.PID, "verdict", "reference_model", 100, judge="judge:a")[1]), 16)

    def test_report_weights_by_the_judge_strata(self):
        # The MathTests example built from files: correct pop 8 (4 labelled), partial 2 (2), wrong 3 (3).
        correct = [rec(i, "correct") for i in range(8)]
        partial = [rec(i, "partial") for i in range(8, 10)]
        wrong = [rec(i, "wrong") for i in range(10, 13)]
        preds.write_file(self.PID, 1, correct + partial + wrong)
        for r, lab in zip(correct[:4], ("correct", "correct", "partial", "wrong")):
            self.label(r, lab)
        for r, lab in zip(partial, ("correct", "partial")):
            self.label(r, lab)
        for r, lab in zip(wrong, ("wrong", "wrong", "partial")):
            self.label(r, lab)
        ecal.save_labels(self.PID, "verdict", "human", {correct[5]["judge"]["content_key"]: ecal.skip_entry(reason="hard", labeller="human")})
        (rep,) = ecal.report(self.PID, "verdict", ["human"])
        (j,) = rep["judges"]
        self.assertEqual((j["n"], j["population"]), (9, 13))
        self.assertEqual(j["strata"]["correct"], {"population": 8, "labelled": 4, "skipped": 1, "skip_share": 0.2, "weight": 2.0})
        self.assertEqual(j["skip_warning_strata"], ["correct"])
        self.assertAlmostEqual(j["exact_agreement"]["value"], round(7 / 13, 4))
        self.assertAlmostEqual(j["too_lenient"]["value"], round(4 / 13, 4))
        self.assertAlmostEqual(j["kappa"], round(1 / 3, 4))
        self.assertEqual(j["tasks"], {"answer": 9})
        self.assertEqual(rep["skipped"], 1)

    def test_rubric_scale_population_and_report(self):
        rows = [rubric_rec(i, i < 4) for i in range(6)]
        preds.write_file(self.PID, 2, rows)
        pop = ecal.population(self.PID, "rubric")
        self.assertEqual(sorted(v["verdicts"]["judge:a"] for v in pop.values()), ["fail", "fail", "pass", "pass", "pass", "pass"])
        self.assertEqual(ecal.population(self.PID, "verdict"), {})
        for r, lab in zip(rows, ("pass", "pass", "pass", "fail", "fail", "pass")):
            self.label(r, lab, scale="rubric")
        (rep,) = ecal.report(self.PID, "rubric", ["human"])
        (j,) = rep["judges"]
        self.assertAlmostEqual(j["kappa"], 0.25)
        self.assertEqual(j["kappa_kind"], "cohen")
        self.assertEqual(j["rubric_shas"], {"rubric-sha": 6})
        with self.assertRaises(ValueError):
            ecal._check_scale("bogus")
        # a label outside the scale's vocabulary is not read back
        path = ecal.labels_path(self.PID)
        data = json.loads(path.read_text())
        data["rubric"]["junk"] = {"human": {"label": "correct"}}
        path.write_text(json.dumps(data))
        self.assertNotIn("junk", ecal.read_labels(self.PID)["rubric"])

    def test_over_cap_rows_are_left_out_and_counted(self):
        long_pred = "x" * preds.TEXT_CAP
        rows = [rec(1, "correct"), rec(2, "wrong", pred=long_pred), rec(3, "partial")]
        rows[2]["history_truncated"] = True
        preds.write_file(self.PID, 4, rows)
        pop, over = ecal.population_ex(self.PID, "verdict")
        self.assertEqual((len(pop), over), (1, 2))
        self.assertEqual([r["content_key"] for r in ecal.sample(self.PID, "verdict", "reference_model", 20)[1]],
                         [rows[0]["judge"]["content_key"]])
        self.assertEqual(ecal.progress(self.PID)["scales"]["verdict"]["over_cap"], 2)
        self.assertEqual(ecal.report(self.PID, "verdict", ["human"])[0]["over_cap"], 2)
        # built records: a prediction at the cap is stored cut, and so excluded
        (r,), _ = preds.build_records([{"prompt": "q", "reference": "r", "prediction": "y" * (preds.TEXT_CAP + 50),
                                        "row_judge_verdict": "wrong", "row_judge_score": 0.0}], answer_label="j")
        self.assertEqual(len(r["prediction"]), preds.TEXT_CAP)

    def test_content_key_includes_the_task_and_is_stable_per_task(self):
        plain = content_key("q", "r", "p")
        self.assertEqual(plain, content_key(" q", "r ", "p", "answer"))
        self.assertNotEqual(plain, content_key("q", "r", "p", "code_example"))
        self.assertNotEqual(content_key("q", "r", "p", "remediation"), content_key("q", "r", "p", "code_example"))
        self.assertNotEqual(plain, content_key("q", "r", "p", "rubric"))
        a = rec(1, "correct", q="q", ref="r", pred="p")
        b = rec(1, "wrong", q="q", ref="r", pred="p", task="code_example")
        b["judge"]["content_key"] = content_key("q", "r", "p", "code_example")
        preds.write_file(self.PID, 1, [a])
        preds.write_file(self.PID, 2, [a, b])
        pop = ecal.population(self.PID, "verdict")
        self.assertEqual(len(pop), 2)  # same texts, two instructions → two items
        self.assertEqual(pop[a["judge"]["content_key"]]["eval_result_ids"], [1, 2])  # stable across results
        # the builder keys a code example and a plain answer with the same texts apart
        ce = {"prompt": "p", "reference": "r", "prediction": "x", "code_example_row": {"task": "t", "language": "python", "passages": []},
              "_code_example_judge": {"verdict": "correct", "label": "j"}}
        (c,), _ = preds.build_records([ce], parse_json=lambda t: None)
        self.assertEqual(c["judge"]["content_key"], content_key(c["judge"]["question"], "r", "x", "code_example"))
        # live ids: a file whose result is gone is ignored
        self.assertEqual(len(ecal.population(self.PID, "verdict", live_ids={1})), 1)

    def test_write_replaces_a_stale_file_for_a_reused_id(self):
        preds.write_file(self.PID, 3, [rec(1, "correct"), rec(2, "wrong")])
        preds.write_file(self.PID, 3, [rec(5, "partial")])
        self.assertEqual([r["row_index"] for r in preds.read_records(self.PID, 3)], [5])
        preds.write_file(self.PID, 3, [])
        self.assertIsNone(preds.read_records(self.PID, 3))
        self.assertEqual(preds.delete_files(self.PID, [3, 4]), 0)

    def test_rubric_passages_are_shown_to_the_labeller_as_the_reference_model_gets_them(self):
        row = rubric_rec(1, True)
        row["rubric"]["passages"] = [{"id": f"P{i}", "text": "w" * 1600} for i in range(20)]
        preds.write_file(self.PID, 1, [row])
        item = next(iter(ecal.population(self.PID, "rubric").values()))
        blind = ecal.render_blind(item, "rubric")
        passages, cut = ecal.budgeted_passages(item)
        self.assertEqual((blind["passages"], blind["passages_truncated"], cut), (passages, True, True))
        self.assertLess(len(json.dumps(passages)), 21000)

    def test_local_backend_labels_are_self_excluded(self):
        self.assertTrue(ecal.same_model("ollama:qwen3:8b", "ollama:qwen3:8b"))
        self.assertTrue(ecal.same_model("command", "command"))
        self.assertTrue(ecal.same_model("cloud:anthropic:m", "anthropic:m"))
        self.assertFalse(ecal.same_model("ollama:qwen3:8b", "ollama:qwen3:14b"))
        rows = [rec(i, "correct", judge="ollama:qwen3:8b") for i in range(3)]
        preds.write_file(self.PID, 9, rows)
        for r in rows:
            self.label(r, "wrong", tier="reference_model", labeller="ollama:qwen3:8b")
        (j,) = ecal.report(self.PID, "verdict", ["reference_model"])[0]["judges"]
        self.assertEqual((j["n"], j["self_rows"], j["self"]), (0, 3, True))

    def test_reference_labels_never_grade_their_own_model(self):
        rows = [rec(i, "correct" if i % 2 else "wrong", judge="cloud-x:big") for i in range(4)]
        preds.write_file(self.PID, 3, rows)
        for r in rows:
            self.label(r, "correct", tier="reference_model", labeller="cloud:cloud-x:big")
        (rep,) = ecal.report(self.PID, "verdict", ["reference_model"])
        (j,) = rep["judges"]
        self.assertEqual((j["n"], j["self_rows"], j["self"]), (0, 4, True))
        self.assertEqual(ecal.judge_trust("cloud-x:big", [rep], "reference_model")["status"], "uncalibrated")


# ── API + reference Job ─────────────────────────────────────────────────


def seed_results(pid, count):
    """An experiment with ``count`` EvalResults in the app DB (seeding only): ``(experiment id, *result ids)``."""
    from app.database import async_session_factory
    from app.models.experiment import EvalResult, Experiment, ExperimentStatus

    async def seed():
        async with async_session_factory() as db:
            exp = Experiment(project_id=pid, name="e", description="", status=ExperimentStatus.COMPLETED, base_model="toy/m")
            db.add(exp)
            await db.flush()
            results = [EvalResult(experiment_id=exp.id, dataset_name="test", eval_type="f1", metrics={}, details={}) for _ in range(count)]
            db.add_all(results)
            await db.commit()
            return (exp.id, *[r.id for r in results])

    return asyncio.run(seed())


class Ref:
    """A stub reference backend: 'correct' when the model answer contains 'good', else 'wrong'."""
    name = "refmodel"
    max_output_tokens = 4096
    prompts: list = []
    gate: threading.Event | None = None

    @classmethod
    def is_available(cls):
        return True

    def __init__(self, model=None):
        self.model = model

    def describe(self):
        return f"{self.name}:{self.model}" if self.model else self.name

    async def complete(self, prompt, **kw):
        if type(self).gate is not None:
            await asyncio.to_thread(type(self).gate.wait, 10)
        type(self).prompts.append((prompt, kw.get("system_prompt")))
        answer = prompt.split("MODEL ANSWER:\n", 1)[-1].split("\n\n", 1)[0]
        verdict = "correct" if "good" in answer else "wrong"
        return json.dumps({"verdict": verdict, "reason": "ref"})


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls._patches = [patch("app.services.synth_backends.BACKEND_REGISTRY", [Ref]),
                        patch("app.services.synth_playbook_service.BACKEND_REGISTRY", [Ref])]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        settings.AUTH_ENABLED = cls._prev_auth

    def setUp(self):
        Ref.prompts, Ref.gate = [], None
        r = client.post("/api/projects", json={"name": f"v4-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        self.base = f"/api/projects/{self.pid}/eval-judge/calibration"
        self.records = [rec(i, ("correct", "partial", "wrong")[i % 3], pred=f"good answer {i}" if i % 2 else f"bad answer {i}")
                        for i in range(9)]
        # The reference model is also a judge, of the same answer as row 1.
        self.records.append(rec(1, "correct", judge="refmodel:big", pred="good answer 1"))
        self.exp_id, self.rid, self.rid2 = seed_results(self.pid, 2)
        preds.write_file(self.pid, self.rid, self.records)

    def tearDown(self):
        if Ref.gate is not None:
            Ref.gate.set()
        wait_for_jobs_idle(client)
        llm_budget_guard.reset()
        shutil.rmtree(settings.DATA_DIR / "projects" / str(self.pid), ignore_errors=True)

    def test_sample_label_skip_delete_progress(self):
        body = client.get(f"{self.base}/sample", params={"n": 20, "judge": "judge:a"}).json()
        self.assertEqual(len(body["rows"]), 9)
        self.assertEqual(body["labels"], ["correct", "partial", "wrong"])
        text = json.dumps(body["rows"])
        for banned in ("judge:a", "secret reason", "\"verdict\":", "verdicts", "score"):
            self.assertNotIn(banned, text)
        ck = body["rows"][0]["content_key"]
        resp = client.post(f"{self.base}/labels", json={"content_key": ck, "label": "partial", "judge": "judge:a", "reason": " meh "})
        self.assertEqual(resp.status_code, 200, resp.text)
        stored = json.loads(ecal.labels_path(self.pid).read_text())["verdict"][ck]["human"]
        self.assertEqual((stored["label"], stored["labeller"], stored["reason"], stored["judge"]), ("partial", "human", "meh", "judge:a"))
        self.assertIn(stored["stratum"], ("correct", "partial", "wrong"))
        prog = client.get(f"{self.base}/labels").json()["scales"]["verdict"]
        self.assertEqual((prog["population"], prog["tiers"]["human"]["labelled"]), (9, 1))
        self.assertEqual(prog["judges"][0]["judge"], "judge:a")
        ck2 = body["rows"][1]["content_key"]
        self.assertEqual(client.post(f"{self.base}/skips", json={"content_key": ck2}).status_code, 200)
        self.assertEqual(len(client.get(f"{self.base}/sample", params={"judge": "judge:a"}).json()["rows"]), 7)
        self.assertEqual(client.delete(f"{self.base}/labels/{ck2}").status_code, 200)
        self.assertEqual(client.delete(f"{self.base}/labels/{ck2}").status_code, 404)
        # guards
        self.assertEqual(client.post(f"{self.base}/labels", json={"content_key": ck, "label": "pass"}).status_code, 400)
        self.assertEqual(client.post(f"{self.base}/labels", json={"content_key": "nope", "label": "correct"}).status_code, 404)
        self.assertEqual(client.post(f"{self.base}/labels", json={"content_key": ck, "label": "correct", "tier": "reference_model"}).status_code, 400)
        self.assertEqual(client.get(f"{self.base}/sample", params={"n": 0}).status_code, 400)
        self.assertEqual(client.get(f"{self.base}/sample", params={"scale": "bogus"}).status_code, 400)

    def test_report_and_trust_use_the_label_source(self):
        for r in self.records[:9]:
            client.post(f"{self.base}/labels", json={"content_key": r["judge"]["content_key"], "label": r["judge"]["verdict"]})
        rep = client.get(f"{self.base}/report").json()
        self.assertEqual((rep["label_source"], rep["tier"], [t["tier"] for t in rep["reports"]]), ("human", "human", ["human"]))
        j = next(x for x in rep["reports"][0]["judges"] if x["judge"] == "judge:a")
        self.assertEqual((j["n"], j["exact_agreement"]["value"], j["insufficient"]), (9, 1.0, True))
        self.assertEqual(rep["trust"]["judge:a"]["status"], "insufficient")
        client.put(f"/api/projects/{self.pid}/synthetic/calibration/settings", json={"label_source": "both"})
        self.assertEqual([t["tier"] for t in client.get(f"{self.base}/report").json()["reports"]], ["human", "reference_model"])
        self.assertEqual(client.get(f"{self.base}/trust", params={"judge": "judge:a"}).json()["label_source"], "both")
        # the eval-judge settings payload carries trust for the judge in use (refmodel is the only backend)
        payload = client.get(f"/api/projects/{self.pid}/eval-judge").json()
        self.assertIn("trust", payload)

    def test_predictions_endpoint(self):
        rid, old_id = self.rid, self.rid2
        url = f"/api/projects/{self.pid}/evaluation/{rid}/predictions"
        page = client.get(url, params={"offset": 8, "limit": 5}).json()
        self.assertEqual((page["available"], page["total"], len(page["rows"])), (True, 10, 2))
        self.assertEqual(page["rows"][0]["judge"]["verdict"], self.records[8]["judge"]["verdict"])
        old = client.get(f"/api/projects/{self.pid}/evaluation/{old_id}/predictions").json()
        self.assertEqual((old["available"], old["rows"]), (False, []))
        self.assertEqual(client.get(f"/api/projects/{self.pid}/evaluation/999999/predictions").status_code, 404)
        self.assertEqual(client.get(url, params={"limit": 501}).status_code, 400)

    def test_deleted_results_leave_the_population_and_their_files_go(self):
        # a file whose EvalResult does not exist is ignored
        preds.write_file(self.pid, 987654, [rec(77, "wrong", q="orphan q")])
        prog = client.get(f"{self.base}/labels").json()["scales"]["verdict"]
        self.assertEqual(prog["population"], 9)
        # deleting the experiment removes its results' files (cascade) and empties the population
        resp = client.delete(f"/api/projects/{self.pid}/training/experiments/{self.exp_id}")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertFalse(preds.predictions_path(self.pid, self.rid).exists())
        self.assertEqual(client.get(f"{self.base}/labels").json()["scales"]["verdict"]["population"], 0)
        self.assertEqual(client.get(f"{self.base}/sample").json()["rows"], [])

    def test_reference_job_tier_isolation_self_exclusion_and_409(self):
        ck0 = self.records[0]["judge"]["content_key"]
        client.post(f"{self.base}/labels", json={"content_key": ck0, "label": "partial"})
        files_before = (preds.predictions_path(self.pid, self.rid)).read_text()
        Ref.gate = threading.Event()
        resp = client.post(f"{self.base}/reference-labelling?async_job=true", json={"backend": "refmodel:big", "n": 20})
        self.assertEqual(resp.status_code, 202, resp.text)
        second = client.post(f"{self.base}/reference-labelling?async_job=true", json={"backend": "refmodel:big", "n": 20})
        self.assertEqual(second.status_code, 409, second.text)
        self.assertEqual(second.json()["error_code"], "EVAL_CALIBRATION_REFERENCE_ALREADY_RUNNING")
        Ref.gate.set()
        wait_for_jobs_idle(client)
        Ref.gate = None
        job = client.get(f"/api/jobs/{resp.json()['id']}").json()
        self.assertEqual((job["kind"], job["status"]), ("eval_judge_calibration_reference", "succeeded"), job)
        res = job["result"]
        self.assertEqual((res["rows_sampled"], res["labelled"], res["labeller"]), (9, 9, "refmodel:big"))
        # the judge's own prompt (system prompt + question / reference / answer), blind
        prompt, system = Ref.prompts[0]
        self.assertIn("REFERENCE ANSWER (treat as ground truth)", prompt)
        self.assertIn("You grade a small language model", system)
        for banned in ("secret reason", "judge:a"):
            self.assertNotIn(banned, prompt)
        stored = json.loads(ecal.labels_path(self.pid).read_text())["verdict"]
        self.assertEqual(stored[ck0]["human"]["label"], "partial")  # the human tier is untouched
        self.assertEqual(stored[ck0]["reference_model"]["labeller"], "refmodel:big")
        self.assertEqual(preds.predictions_path(self.pid, self.rid).read_text(), files_before)
        good = self.records[1]["judge"]["content_key"]
        self.assertEqual(stored[good]["reference_model"]["label"], "correct")
        # the reference model is also a judge: never graded on its own labels
        (rep,) = client.get(f"{self.base}/report", params={"tier": "reference_model"}).json()["reports"]
        me = next(j for j in rep["judges"] if j["judge"] == "refmodel:big")
        self.assertEqual((me["n"], me["self_rows"], me["self"]), (0, 1, True))
        other = next(j for j in rep["judges"] if j["judge"] == "judge:a")
        self.assertEqual(other["n"], 9)
        # nothing left to label in the reference tier
        again = client.post(f"{self.base}/reference-labelling?async_job=true", json={"backend": "refmodel:big"})
        self.assertEqual((again.status_code, again.json()["error_code"]), (400, "no_calibration_rows"))
        self.assertEqual(client.post(f"{self.base}/reference-labelling", json={"backend": "refmodel:big"}).status_code, 400)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

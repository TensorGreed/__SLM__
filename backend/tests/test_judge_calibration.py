"""Judge calibration V2: blind stratified labels, stratum-weighted per-judge metrics, the label-source
setting and the reference-model labelling Job.

pytest-only (the Jobs these tests start are awaited in tearDown).
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services import judge_calibration_service as calib
from app.services import llm_budget_guard
from app.services import synth_review_assist_service as assist

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def row(rid, status="pending", **fields):
    return {"id": rid, "review_status": status, "synth_confidence": 0.8, "synth_source": "playbook:r:m",
            "synth_backend": "secret-generator:x", "label": "greeting", "text": f"t{rid}", **fields}


def verdict_entry(r, verdict, judge, sig=None, sha="aaaaaaaaaaaa", reason=None):
    return {"verdict": verdict, "reason_code": reason or ("ok" if verdict == "pass" else "incorrect"), "note": "judge note",
            "backend": judge, "job_id": 1, "at": "2026-10-29T10:00:00+00:00",
            "sig": sig or assist.row_signature(r), "rubric_sha": sha}


class Ref:
    """A stub reference backend: pass on even ids, fail on odd; ``unsure_ids`` answer unsure."""
    name = "refmodel"
    max_output_tokens = 4096
    prompts: list = []
    unsure_ids: set = set()

    @classmethod
    def is_available(cls):
        return True

    def __init__(self, model=None):
        self.model = model

    def describe(self):
        return f"{self.name}:{self.model}" if self.model else self.name

    async def complete(self, prompt, **kw):
        type(self).prompts.append(prompt)
        ids = [int(i) for i in re.findall(r'<row id="(\d+)"', prompt)]
        out = []
        for rid in ids:
            v = "unsure" if rid in type(self).unsure_ids else ("pass" if rid % 2 == 0 else "fail")
            out.append({"id": rid, "verdict": v, "reason_code": "ok" if v == "pass" else "incorrect", "note": f"ref {rid}"})
        return json.dumps(out)


class StoreBase(unittest.TestCase):
    PID = 987711

    def setUp(self):
        self.root = settings.DATA_DIR / "projects" / str(self.PID)
        shutil.rmtree(self.root, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class StatsTests(unittest.TestCase):
    def test_wilson_matches_the_textbook_value(self):
        low, high = calib.wilson(0.5, 10)
        self.assertAlmostEqual(low, 0.2366, places=3)
        self.assertAlmostEqual(high, 0.7634, places=3)
        low, high = calib.wilson(1.0, 20)
        self.assertAlmostEqual(low, 0.8389, places=3)
        self.assertEqual(high, 1.0)
        self.assertIsNone(calib.wilson(None, 5))
        self.assertIsNone(calib.wilson(0.5, 0))

    def test_cohen_kappa(self):
        # a=20 both fail, b=5, c=10, d=15 → po=0.7, pe=0.5 → 0.4
        self.assertAlmostEqual(calib.cohen_kappa(20, 5, 10, 15), 0.4)
        self.assertAlmostEqual(calib.cohen_kappa(10, 0, 0, 10), 1.0)
        self.assertIsNone(calib.cohen_kappa(10, 0, 0, 0))  # chance agreement 1: undefined
        self.assertIsNone(calib.cohen_kappa(0, 0, 0, 0))

    def test_kappa_hand_computed_cells(self):
        self.assertAlmostEqual(calib.cohen_kappa(4, 0, 4, 1), 2 / 11)
        # a=10, b=2, c=3, d=5: po=15/20, pe=(12*13 + 8*7)/400 = 212/400 -> (300-212)/(400-212) = 88/188
        self.assertAlmostEqual(calib.cohen_kappa(10, 2, 3, 5), 88 / 188)

    def test_degenerate_judge_with_one_label_class_does_not_crash(self):
        items = [{"verdict": "pass", "label": "pass", "weight": 1.0}] * 6 + [{"verdict": "fail", "label": "pass", "weight": 1.0}] * 2
        m = calib.judge_metrics(items)
        self.assertIn(m["kappa"], (None, 0.0))
        self.assertIsNone(m["fail_recall"]["value"])  # no labelled fail
        self.assertTrue(m["insufficient"])
        all_same = [{"verdict": "pass", "label": "pass", "weight": 1.0}] * 6
        self.assertIsNone(calib.judge_metrics(all_same)["kappa"])

    def test_effective_n(self):
        self.assertAlmostEqual(calib.effective_n([1, 1, 1, 1]), 4)
        self.assertAlmostEqual(calib.effective_n([3, 1]), 16 / 10)

    def test_weighted_metrics_on_a_hand_computed_example(self):
        # Stratum A weight 4 (2 labelled), stratum B weight 1 (2 labelled).
        items = [
            {"verdict": "fail", "label": "fail", "weight": 4.0},   # tp
            {"verdict": "pass", "label": "fail", "weight": 4.0},   # fn
            {"verdict": "pass", "label": "pass", "weight": 1.0},   # tn
            {"verdict": "unsure", "label": "fail", "weight": 1.0},
        ]
        m = calib.judge_metrics(items)
        self.assertEqual(m["counts"], {"tp": 1, "fp": 0, "fn": 1, "tn": 1, "unsure_label_fail": 1, "unsure_label_pass": 0})
        # fail recall = 4 / (4 + 4 + 1) (unsure counts as not saying fail)
        self.assertAlmostEqual(m["fail_recall"]["value"], round(4 / 9, 4))
        self.assertAlmostEqual(m["fail_recall"]["n_eff"], round(81 / 33, 2))
        # pass precision = 1 / (4 + 1)
        self.assertAlmostEqual(m["pass_precision"]["value"], 0.2)
        self.assertAlmostEqual(m["unsure_rate"], round(1 / 10, 4))
        # kappa on decisive weighted cells a=4 (both fail), b=0, c=4 (judge pass, label fail), d=1:
        # po = 5/9, pe = (4*8 + 5*1)/81 = 37/81 -> kappa = (45-37)/(81-37) = 8/44 = 0.1818
        self.assertAlmostEqual(m["kappa"], 0.1818)
        low, high = calib.wilson(4 / 9, 81 / 33)
        self.assertAlmostEqual(m["fail_recall"]["low"], round(low, 4))
        self.assertAlmostEqual(m["fail_recall"]["high"], round(high, 4))

    def test_insufficient_flag(self):
        few = [{"verdict": "pass", "label": "pass", "weight": 1.0}] * 3
        m = calib.judge_metrics(few)
        self.assertTrue(m["insufficient"])
        self.assertEqual(len(m["insufficient_reasons"]), 3)
        enough = ([{"verdict": "fail", "label": "fail", "weight": 1.0}] * 10
                  + [{"verdict": "pass", "label": "pass", "weight": 1.0}] * 10)
        m = calib.judge_metrics(enough)
        self.assertFalse(m["insufficient"])
        self.assertEqual(m["kappa"], 1.0)
        lopsided = ([{"verdict": "fail", "label": "fail", "weight": 1.0}] * 4
                    + [{"verdict": "pass", "label": "pass", "weight": 1.0}] * 20)
        m = calib.judge_metrics(lopsided)
        self.assertTrue(m["insufficient"])
        self.assertEqual(m["insufficient_reasons"], ["fewer than 5 labelled fails (4)"])


class SamplerTests(StoreBase):
    def build(self):
        rows = [row(i, status="accepted" if i == 3 else "pending", decided_via="manual") for i in range(1, 13)]
        rows[3].update(review_status="rejected", reject_reason="assist:incorrect")
        # 1-4 single pass; 5-6 single fail; 7 single unsure; 8-9 agree_pass; 10 disagree; 11-12 no verdict.
        plan = {1: ["pass"], 2: ["pass"], 3: ["pass"], 4: ["pass"], 5: ["fail"], 6: ["fail"], 7: ["unsure"],
                8: ["pass", "pass"], 9: ["pass", "pass"], 10: ["pass", "fail"]}
        for r in rows:
            for k, v in enumerate(plan.get(r["id"], [])):
                assist.save_verdicts(self.PID, {r["id"]: verdict_entry(r, v, f"judge{k}")})
        return rows

    def test_population_and_strata(self):
        rows = self.build()
        pop = {int(r["id"]): s for r, _, s in calib.population(self.PID, rows)}
        self.assertEqual(set(pop), set(range(1, 11)))  # any review status; no verdict = not in it
        self.assertEqual(pop[1], "single_pass")
        self.assertEqual(pop[5], "single_fail")
        self.assertEqual(pop[7], "single_unsure")
        self.assertEqual(pop[8], "agree_pass")
        self.assertEqual(pop[10], "disagree")

    def test_round_robin_spreads_across_strata_and_is_seeded(self):
        rows = self.build()
        picked = calib.sample_rows(self.PID, rows, "human", 5)
        strata = {int(r["id"]): s for r, _, s in calib.population(self.PID, rows)}
        self.assertEqual(sorted(strata[int(r["id"])] for r in picked),
                         sorted(["single_pass", "single_fail", "single_unsure", "agree_pass", "disagree"]))
        again = calib.sample_rows(self.PID, rows, "human", 5)
        self.assertEqual([r["id"] for r in picked], [r["id"] for r in again])
        everything = calib.sample_rows(self.PID, rows, "human", 100)
        self.assertEqual(sorted(r["id"] for r in everything), list(range(1, 11)))

    def test_sample_order_does_not_follow_the_strata_cycle(self):
        cands = [({"id": i}, calib.STRATA[i % 3]) for i in range(30)]
        cycle = [calib.STRATA[k % 3] for k in range(12)]
        orders = []
        for seed in range(5):
            picked = calib.stratified_sample(cands, 12, seed)
            self.assertEqual(sorted(calib.STRATA[r["id"] % 3] for r in picked), sorted(cycle))  # still 4 per stratum
            orders.append([calib.STRATA[r["id"] % 3] for r in picked])
        self.assertTrue(all(o != cycle for o in orders), orders)
        self.assertEqual(calib.stratified_sample(cands, 12, 3), calib.stratified_sample(cands, 12, 3))  # seeded

    def test_skipped_rows_are_not_resampled_and_are_not_labels(self):
        rows = self.build()
        by_id = {r["id"]: r for r in rows}
        calib.save_labels(self.PID, "human", {6: calib.skip_entry(by_id[6], reason="hard", labeller="human")})
        self.assertNotIn(6, {r["id"] for r in calib.sample_rows(self.PID, rows, "human", 100)})
        self.assertIn(6, {r["id"] for r in calib.sample_rows(self.PID, rows, "reference_model", 100)})
        prog = calib.progress(self.PID, rows)["tiers"]["human"]
        self.assertEqual((prog["labelled"], prog["skipped"], prog["strata"]["single_fail"]["skipped"]), (0, 1, 1))
        # a label replaces the skip
        calib.save_labels(self.PID, "human", {6: calib.label_entry(by_id[6], "fail", stratum="single_fail", labeller="human")})
        prog = calib.progress(self.PID, rows)["tiers"]["human"]
        self.assertEqual((prog["labelled"], prog["skipped"]), (1, 0))

    def test_already_labelled_rows_are_excluded_per_tier(self):
        rows = self.build()
        by_id = {r["id"]: r for r in rows}
        calib.save_labels(self.PID, "human", {
            rid: calib.label_entry(by_id[rid], "pass", stratum="x", labeller="human") for rid in (1, 2, 5, 10)
        })
        human = {r["id"] for r in calib.sample_rows(self.PID, rows, "human", 100)}
        self.assertEqual(human, {3, 4, 6, 7, 8, 9})
        ref = {r["id"] for r in calib.sample_rows(self.PID, rows, "reference_model", 100)}
        self.assertEqual(ref, set(range(1, 11)))  # tiers are separate

    def test_blind_render_carries_no_verdict_or_review_field(self):
        rows = self.build()
        rows[0]["review_assist"] = {"verdict": "pass"}  # a legacy inline verdict
        att = [dict(r) for r in rows]
        assist.attach_verdicts(self.PID, att)
        for r in att[:10]:
            out = calib.render_blind(r)
            text = json.dumps(out)
            for banned in ("verdict", "consensus", "review_status", "reject_reason", "decided_via", "synth_backend",
                           "secret-generator", "judge0", "review_assist", "accepted"):
                self.assertNotIn(banned, text, (banned, out))
            self.assertEqual(out["fields"]["text"], f"t{r['id']}")

    def test_stale_labels_are_ignored(self):
        rows = self.build()
        calib.save_labels(self.PID, "human", {1: calib.label_entry(rows[0], "fail", stratum="single_pass", labeller="human")})
        edited = [dict(r) for r in rows]
        edited[0]["text"] = "edited"
        # the edited row's verdict is stale too, so re-judge it to keep it in the population
        assist.save_verdicts(self.PID, {1: verdict_entry(edited[0], "pass", "judge0")})
        prog = calib.progress(self.PID, edited)
        self.assertEqual(prog["tiers"]["human"]["labelled"], 0)
        self.assertEqual(prog["tiers"]["human"]["stale"], 1)
        self.assertIn(1, {r["id"] for r in calib.sample_rows(self.PID, edited, "human", 100)})

    def test_legacy_project_without_sidecars(self):
        rows = [row(1), row(2)]
        self.assertEqual(calib.population(self.PID, rows), [])
        self.assertEqual(calib.sample_rows(self.PID, rows, "human", 20), [])
        self.assertEqual(calib.progress(self.PID, rows)["population"], 0)
        rep = calib.tier_report(self.PID, rows, "human")
        self.assertEqual((rep["labelled"], rep["judges"], rep["coverage"]), (0, [], None))


class ReportTests(StoreBase):
    def test_weights_by_stratum_population_over_labelled(self):
        # single_pass: 8 rows, 2 labelled (weight 4); single_fail: 2 rows, 2 labelled (weight 1).
        rows = [row(i) for i in range(1, 11)]
        for r in rows:
            assist.save_verdicts(self.PID, {r["id"]: verdict_entry(r, "pass" if r["id"] <= 8 else "fail", "j:a",
                                                                   sha="s1" if r["id"] != 9 else "s2")})
        by_id = {r["id"]: r for r in rows}
        labels = {1: "fail", 2: "pass", 9: "fail", 10: "pass"}
        calib.save_labels(self.PID, "human", {
            rid: calib.label_entry(by_id[rid], lab, stratum="x", labeller="human") for rid, lab in labels.items()
        })
        rep = calib.tier_report(self.PID, rows, "human")
        self.assertEqual(rep["strata"]["single_pass"], {"population": 8, "labelled": 2, "skipped": 0, "skip_share": 0.0, "weight": 4.0})
        self.assertEqual(rep["strata"]["single_fail"], {"population": 2, "labelled": 2, "skipped": 0, "skip_share": 0.0, "weight": 1.0})
        self.assertEqual(rep["coverage"], 1.0)
        (j,) = rep["judges"]
        # judge: 1 pass/label fail (w4) fn, 2 pass/pass (w4) tn, 9 fail/fail (w1) tp, 10 fail/pass (w1) fp
        self.assertEqual(j["counts"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "unsure_label_fail": 0, "unsure_label_pass": 0})
        self.assertAlmostEqual(j["fail_recall"]["value"], 0.2)       # 1 / (1 + 4)
        self.assertAlmostEqual(j["pass_precision"]["value"], 0.5)    # 4 / (4 + 4)
        self.assertEqual(j["n"], 4)
        self.assertAlmostEqual(j["n_eff"], round(100 / 34, 2))
        self.assertTrue(j["insufficient"])
        self.assertEqual(j["rubric_shas"], {"s1": 3, "s2": 1})
        self.assertTrue(j["rubric_mixed"])
        self.assertFalse(j["self"])

    def test_reference_labeller_grading_itself_is_flagged_self(self):
        rows = [row(1), row(2)]
        for r in rows:
            assist.save_verdicts(self.PID, {r["id"]: verdict_entry(r, "pass", "refmodel:big")})
            assist.save_verdicts(self.PID, {r["id"]: verdict_entry(r, "fail", "other:small")})
        calib.save_labels(self.PID, "reference_model", {
            r["id"]: calib.label_entry(r, "pass", stratum="disagree", labeller="refmodel:big") for r in rows
        })
        judges = {j["judge"]: j for j in calib.tier_report(self.PID, rows, "reference_model")["judges"]}
        ref = judges["refmodel:big"]
        self.assertTrue(ref["self"])
        self.assertEqual((ref["self_rows"], ref["n"]), (2, 0))  # nothing left to measure
        self.assertTrue(ref["insufficient"])
        self.assertFalse(judges["other:small"]["self"])
        self.assertEqual(judges["other:small"]["n"], 2)
        # the human tier is separate and empty
        self.assertEqual(calib.tier_report(self.PID, rows, "human")["judges"], [])

    def test_self_labelled_rows_are_excluded_and_weights_recomputed(self):
        # Stratum single_pass: rows 1-6 (judge j:a passes all). Rows 1-2 labelled by j:a itself (pass),
        # rows 3-4 by another reference model (fail). Stratum single_fail: rows 7-8 (j:a fails), labelled fail by other.
        rows = [row(i) for i in range(1, 9)]
        for r in rows:
            assist.save_verdicts(self.PID, {r["id"]: verdict_entry(r, "pass" if r["id"] <= 6 else "fail", "j:a")})
        by_id = {r["id"]: r for r in rows}
        labels = {1: ("pass", "j:a"), 2: ("pass", "j:a"), 3: ("fail", "ref:x"), 4: ("fail", "ref:x"),
                  7: ("fail", "ref:x"), 8: ("fail", "ref:x")}
        calib.save_labels(self.PID, "reference_model", {
            rid: calib.label_entry(by_id[rid], lab, stratum="x", labeller=who) for rid, (lab, who) in labels.items()
        })
        (j,) = calib.tier_report(self.PID, rows, "reference_model")["judges"]
        self.assertFalse(j["self"])
        self.assertEqual((j["self_rows"], j["n"]), (2, 4))
        # without rows 1-2: single_pass weight 6/2 = 3 (rows 3, 4), single_fail weight 2/2 = 1 (rows 7, 8)
        # fail recall = (1 + 1) / (3 + 3 + 1 + 1) = 0.25; the self passes would have made precision 6/12
        self.assertAlmostEqual(j["fail_recall"]["value"], 0.25)
        self.assertAlmostEqual(j["pass_precision"]["value"], 0.0)
        self.assertEqual(j["counts"], {"tp": 2, "fp": 0, "fn": 2, "tn": 0, "unsure_label_fail": 0, "unsure_label_pass": 0})

    def test_skip_share_note(self):
        rows = [row(i) for i in range(1, 6)]
        for r in rows:
            assist.save_verdicts(self.PID, {r["id"]: verdict_entry(r, "pass", "j:a")})
        by_id = {r["id"]: r for r in rows}
        calib.save_labels(self.PID, "human", {
            **{rid: calib.label_entry(by_id[rid], "pass", stratum="single_pass", labeller="human") for rid in (1, 2, 3)},
            4: calib.skip_entry(by_id[4], reason="hard", labeller="human"),
        })
        rep = calib.tier_report(self.PID, rows, "human")
        self.assertEqual(rep["skipped"], 1)
        self.assertEqual(rep["strata"]["single_pass"]["skipped"], 1)
        self.assertEqual(rep["strata"]["single_pass"]["skip_share"], 0.25)
        self.assertEqual(rep["skip_warning_strata"], ["single_pass"])
        self.assertIn("optimistic", rep["skip_note"])
        self.assertEqual(rep["labelled"], 3)  # a skip is not a label
        calib.delete_label(self.PID, 4, "human")
        rep = calib.tier_report(self.PID, rows, "human")
        self.assertEqual((rep["skip_warning_strata"], rep["skip_note"]), ([], None))

    def test_unparseable_verdicts_are_counted_apart(self):
        rows = [row(1)]
        assist.save_verdicts(self.PID, {1: verdict_entry(rows[0], "unsure", "j:a", reason=assist.UNPARSEABLE_REASON)})
        calib.save_labels(self.PID, "human", {1: calib.label_entry(rows[0], "fail", stratum="x", labeller="human")})
        (j,) = calib.tier_report(self.PID, rows, "human")["judges"]
        self.assertEqual((j["n"], j["unparseable"]), (0, 1))


class CalibrationApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls._patches = [
            patch("app.services.synth_backends.BACKEND_REGISTRY", [Ref]),
            patch("app.services.synth_playbook_service.BACKEND_REGISTRY", [Ref]),
        ]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        settings.AUTH_ENABLED = cls._prev_auth

    def setUp(self):
        Ref.prompts = []
        Ref.unsure_ids = set()
        r = client.post("/api/projects", json={"name": f"calib-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        self.base = f"/api/projects/{self.pid}/synthetic"
        self.syn = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "synthetic"
        self.syn.mkdir(parents=True, exist_ok=True)
        self.rows = [row(i, status="accepted" if i == 2 else "pending") for i in range(1, 9)]
        (self.syn / "synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in self.rows) + "\n", encoding="utf-8")
        for r in self.rows[:6]:  # 7-8 have no verdict
            assist.save_verdicts(self.pid, {r["id"]: verdict_entry(r, "pass" if r["id"] <= 3 else "fail", "other:small")})

    def tearDown(self):
        wait_for_jobs_idle(client)
        llm_budget_guard.reset()

    def test_settings_default_get_put(self):
        self.assertEqual(client.get(f"{self.base}/calibration/settings").json()["label_source"], "human")
        resp = client.put(f"{self.base}/calibration/settings", json={"label_source": "both"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(client.get(f"{self.base}/calibration/settings").json()["label_source"], "both")
        self.assertEqual(client.put(f"{self.base}/calibration/settings", json={"label_source": "nope"}).status_code, 422)
        rep = client.get(f"{self.base}/calibration/report").json()
        self.assertEqual([t["tier"] for t in rep["reports"]], ["human", "reference_model"])
        rep = client.get(f"{self.base}/calibration/report", params={"tier": "human"}).json()
        self.assertEqual([t["tier"] for t in rep["reports"]], ["human"])
        self.assertEqual(client.get(f"{self.base}/calibration/report", params={"tier": "bogus"}).status_code, 400)

    def test_sample_is_blind_and_labels_round_trip(self):
        body = client.get(f"{self.base}/calibration/sample", params={"n": 20}).json()
        ids = sorted(r["row_id"] for r in body["rows"])
        self.assertEqual(ids, [1, 2, 3, 4, 5, 6])
        text = json.dumps(body)
        for banned in ("verdict", "consensus", "stratum", "review_status", "decided_via", "synth_backend", "other:small"):
            self.assertNotIn(banned, text)
        resp = client.post(f"{self.base}/calibration/labels", json={"row_id": 4, "label": "fail", "reason": "wrong"})
        self.assertEqual(resp.status_code, 200, resp.text)
        stored = json.loads((self.syn / "calibration_labels.json").read_text())
        self.assertEqual(stored["4"]["human"]["stratum"], "single_fail")
        self.assertEqual(stored["4"]["human"]["labeller"], "human")
        prog = client.get(f"{self.base}/calibration/labels").json()
        self.assertEqual(prog["tiers"]["human"]["labelled"], 1)
        self.assertEqual(prog["tiers"]["human"]["strata"]["single_fail"], {"population": 3, "labelled": 1, "skipped": 0})
        self.assertNotIn(4, [r["row_id"] for r in client.get(f"{self.base}/calibration/sample").json()["rows"]])
        # relabel replaces; delete removes
        client.post(f"{self.base}/calibration/labels", json={"row_id": 4, "label": "pass"})
        self.assertEqual(json.loads((self.syn / "calibration_labels.json").read_text())["4"]["human"]["label"], "pass")
        self.assertEqual(client.delete(f"{self.base}/calibration/labels/4").status_code, 200)
        self.assertEqual(client.delete(f"{self.base}/calibration/labels/4").status_code, 404)
        # guards
        self.assertEqual(client.post(f"{self.base}/calibration/labels", json={"row_id": 7, "label": "pass"}).status_code, 400)
        self.assertEqual(client.post(f"{self.base}/calibration/labels", json={"row_id": 99, "label": "pass"}).status_code, 404)
        self.assertEqual(client.post(f"{self.base}/calibration/labels", json={"row_id": 1, "label": "maybe"}).status_code, 400)
        self.assertEqual(client.post(f"{self.base}/calibration/labels",
                                     json={"row_id": 1, "label": "pass", "tier": "reference_model"}).status_code, 400)
        self.assertEqual(client.get(f"{self.base}/calibration/sample", params={"n": 0}).status_code, 400)
        self.assertEqual(client.get(f"{self.base}/calibration/sample", params={"n": 101}).status_code, 400)

    def test_review_queue_is_unchanged_by_labels(self):
        # Labels change nothing in the queue but the V3 trust status of the judges (never rows or groups).
        def without_trust(body):
            for s in body["facets"]["judge_stats"]:
                s.pop("trust", None)
            return body

        before = client.get(f"{self.base}/review-queue").json()
        client.post(f"{self.base}/calibration/labels", json={"row_id": 1, "label": "fail"})
        after = client.get(f"{self.base}/review-queue").json()
        self.assertEqual(after["facets"]["judge_stats"][0]["trust"]["status"], "insufficient")
        self.assertEqual(without_trust(before), without_trust(after))

    def test_reference_job_writes_only_the_reference_tier(self):
        Ref.unsure_ids = {3}
        assist_before = (self.syn / "review_assist.json").read_text()
        jsonl_before = (self.syn / "synthetic.jsonl").read_text()
        client.post(f"{self.base}/calibration/labels", json={"row_id": 1, "label": "fail"})
        resp = client.post(f"{self.base}/calibration/reference-labelling?async_job=true", json={"backend": "refmodel:big", "n": 20})
        self.assertEqual(resp.status_code, 202, resp.text)
        wait_for_jobs_idle(client)
        job = client.get(f"/api/jobs/{resp.json()['id']}").json()
        self.assertEqual(job["kind"], "judge_calibration_reference")
        self.assertEqual(job["status"], "succeeded", job)
        res = job["result"]
        self.assertEqual((res["rows_sampled"], res["labelled"], res["skipped_unsure"]), (6, 5, 1))
        self.assertEqual(res["labeller"], "refmodel:big")
        # untouched: verdicts, rows (review status), and the human tier
        self.assertEqual((self.syn / "review_assist.json").read_text(), assist_before)
        self.assertEqual((self.syn / "synthetic.jsonl").read_text(), jsonl_before)
        stored = json.loads((self.syn / "calibration_labels.json").read_text())
        self.assertEqual(stored["1"]["human"]["label"], "fail")
        self.assertEqual(stored["1"]["reference_model"]["label"], "fail")  # odd → fail
        self.assertEqual(stored["2"]["reference_model"]["labeller"], "refmodel:big")
        self.assertTrue(stored["3"]["reference_model"]["skipped"])  # unsure: a skip marker, not a label
        self.assertNotIn("label", stored["3"]["reference_model"])
        self.assertEqual(res["skips_recorded"], 1)
        # the prompt is the review-assist prompt and carries no other judge's verdict
        prompt = "\n".join(Ref.prompts)
        self.assertIn("RUBRIC for", prompt)
        for banned in ("other:small", "judge note", "verdict\": \"", "review_status", "consensus"):
            self.assertNotIn(banned, prompt)
        rep = client.get(f"{self.base}/calibration/report", params={"tier": "reference_model"}).json()["reports"][0]
        self.assertEqual(rep["skipped"], 1)
        # every row is labelled or skipped: a re-run has nothing to pick (the unsure row is not re-picked)
        resp = client.post(f"{self.base}/calibration/reference-labelling?async_job=true", json={"backend": "refmodel:big"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error_code"], "no_calibration_rows")

    def test_all_unsure_reference_run_fails_but_its_rows_are_not_repicked(self):
        Ref.unsure_ids = set(range(1, 9))
        resp = client.post(f"{self.base}/calibration/reference-labelling?async_job=true", json={"backend": "refmodel:big"})
        self.assertEqual(resp.status_code, 202, resp.text)
        wait_for_jobs_idle(client)
        job = client.get(f"/api/jobs/{resp.json()['id']}").json()
        self.assertEqual(job["status"], "failed")
        self.assertIn("labelled 0 of 6", job["error"] or "")
        stored = json.loads((self.syn / "calibration_labels.json").read_text())
        self.assertEqual(sorted(int(k) for k, v in stored.items() if v["reference_model"].get("skipped")), [1, 2, 3, 4, 5, 6])
        resp = client.post(f"{self.base}/calibration/reference-labelling?async_job=true", json={"backend": "refmodel:big"})
        self.assertEqual(resp.status_code, 400)
        # the human tier still offers every row
        self.assertEqual(len(client.get(f"{self.base}/calibration/sample").json()["rows"]), 6)

    def test_human_skip_endpoint(self):
        resp = client.post(f"{self.base}/calibration/skips", json={"row_id": 5, "reason": "can't tell"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertNotIn(5, [r["row_id"] for r in client.get(f"{self.base}/calibration/sample").json()["rows"]])
        prog = client.get(f"{self.base}/calibration/labels").json()["tiers"]["human"]
        self.assertEqual((prog["labelled"], prog["skipped"]), (0, 1))
        self.assertEqual(client.post(f"{self.base}/calibration/skips", json={"row_id": 5, "tier": "reference_model"}).status_code, 400)
        self.assertEqual(client.post(f"{self.base}/calibration/skips", json={"row_id": 99}).status_code, 404)
        self.assertEqual(client.delete(f"{self.base}/calibration/labels/5").status_code, 200)
        self.assertIn(5, [r["row_id"] for r in client.get(f"{self.base}/calibration/sample").json()["rows"]])

    def test_reference_job_needs_async_and_rows(self):
        resp = client.post(f"{self.base}/calibration/reference-labelling", json={"backend": "refmodel:big"})
        self.assertEqual(resp.status_code, 400)
        (self.syn / "review_assist.json").unlink()
        resp = client.post(f"{self.base}/calibration/reference-labelling?async_job=true", json={"backend": "refmodel:big"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error_code"], "no_calibration_rows")

    def test_report_endpoint_shapes(self):
        for rid, lab in ((1, "pass"), (4, "fail")):
            client.post(f"{self.base}/calibration/labels", json={"row_id": rid, "label": lab})
        rep = client.get(f"{self.base}/calibration/report").json()
        self.assertEqual(rep["label_source"], "human")
        (tier,) = rep["reports"]
        (j,) = tier["judges"]
        self.assertEqual(j["judge"], "other:small")
        self.assertEqual(j["counts"]["tp"] + j["counts"]["tn"], 2)
        self.assertTrue(j["insufficient"])
        self.assertTrue(math.isclose(j["fail_recall"]["value"], 1.0))


# ── V3: trust status on the consensus gates ───────────────────────────────


def tier_entry(judge, value, low, high, n=24, insufficient=False):
    return {"judge": judge, "n": n, "fail_recall": {"value": value, "low": low, "high": high},
            "insufficient": insufficient, "insufficient_reasons": ["fewer than 20 labelled rows (3)"] if insufficient else []}


class TrustStatusTests(unittest.TestCase):
    def stat(self, judge="j", rubber_stamp=False):
        return {"judge": judge, "judged": 60, "pass": 30, "fail": 30, "unsure": 0, "pass_rate": 0.5, "rubber_stamp": rubber_stamp}

    def test_floor_compares_the_upper_wilson_bound(self):
        # 0.40 over 20: upper bound 0.613 < 0.7 → confidently below the floor.
        low, high = calib.wilson(0.4, 20)
        self.assertAlmostEqual(high, 0.6134, places=3)
        t = calib._tier_trust(tier_entry("j", 0.4, low, high, n=20), "human")
        self.assertEqual(t["status"], "below_floor")
        self.assertEqual((t["tier"], t["fail_recall"], t["n"]), ("human", 0.4, 20))
        self.assertEqual(t["ci"], [low, high])
        # 0.55 over 20: the point estimate is under the floor, but the upper bound (0.742) is not → not flagged.
        low, high = calib.wilson(0.55, 20)
        self.assertAlmostEqual(high, 0.7418, places=3)
        self.assertEqual(calib._tier_trust(tier_entry("j", 0.55, low, high, n=20), "human")["status"], "calibrated_ok")
        self.assertEqual(calib.FAIL_RECALL_FLOOR, 0.7)

    def test_precedence(self):
        bad = [{"tier": "human", "judges": [tier_entry("j", 0.2, 0.05, 0.5)]}]
        good = [{"tier": "human", "judges": [tier_entry("j", 0.9, 0.75, 0.97)]}]
        few = [{"tier": "human", "judges": [tier_entry("j", 1.0, 0.4, 1.0, n=3, insufficient=True)]}]
        # rubber stamp wins over everything, keeping the calibrated status alongside
        t = calib.judge_trust(self.stat(rubber_stamp=True), good, "human")
        self.assertEqual((t["status"], t["calibrated_status"]), ("rubber_stamp", "calibrated_ok"))
        self.assertEqual(calib.judge_trust(self.stat(rubber_stamp=True), [], "human")["status"], "rubber_stamp")
        self.assertEqual(calib.judge_trust(self.stat(), bad, "human")["status"], "below_floor")
        self.assertEqual(calib.judge_trust(self.stat(), good, "human")["status"], "calibrated_ok")
        t = calib.judge_trust(self.stat(), few, "human")
        self.assertEqual((t["status"], t["n"]), ("insufficient", 3))
        self.assertTrue(t["reasons"])
        t = calib.judge_trust(self.stat(), [], "human")
        self.assertEqual((t["status"], t["n"], t["fail_recall"]), ("uncalibrated", 0, None))
        # another judge's labels do not calibrate this one; a self-graded judge (n 0) is uncalibrated
        self.assertEqual(calib.judge_trust(self.stat("other"), good, "human")["status"], "uncalibrated")
        zero = [{"tier": "human", "judges": [{**tier_entry("j", None, None, None, n=0, insufficient=True), "self": True}]}]
        self.assertEqual(calib.judge_trust(self.stat(), zero, "human")["status"], "uncalibrated")
        # the label source picks the tier: human labels do not count under reference_model
        self.assertEqual(calib.judge_trust(self.stat(), good, "reference_model")["status"], "uncalibrated")
        self.assertEqual(calib.judge_trust(self.stat(), good, "human")["floor"], 0.7)

    def test_both_tiers(self):
        def reps(human, ref):
            return [{"tier": "human", "judges": [human] if human else []},
                    {"tier": "reference_model", "judges": [ref] if ref else []}]
        ok = tier_entry("j", 0.9, 0.75, 0.97)
        bad = tier_entry("j", 0.2, 0.05, 0.5, n=30)
        few = tier_entry("j", 1.0, 0.4, 1.0, n=3, insufficient=True)
        # either tier sufficient → calibrated, measured against that tier
        t = calib.judge_trust(self.stat(), reps(None, ok), "both")
        self.assertEqual((t["status"], t["tier"]), ("calibrated_ok", "reference_model"))
        t = calib.judge_trust(self.stat(), reps(bad, few), "both")
        self.assertEqual((t["status"], t["tier"]), ("below_floor", "human"))
        # both sufficient and disagreeing → the worse one
        t = calib.judge_trust(self.stat(), reps(ok, bad), "both")
        self.assertEqual((t["status"], t["tier"], t["n"]), ("below_floor", "reference_model", 30))
        t = calib.judge_trust(self.stat(), reps(bad, ok), "both")
        self.assertEqual((t["status"], t["tier"]), ("below_floor", "human"))
        # neither sufficient
        self.assertEqual(calib.judge_trust(self.stat(), reps(few, None), "both")["status"], "insufficient")
        t = calib.judge_trust(self.stat(), reps(None, None), "both")
        self.assertEqual((t["status"], t["tier"]), ("uncalibrated", None))

    def test_worst_trust(self):
        self.assertEqual(calib.worst_trust(["calibrated_ok", "insufficient"]), "insufficient")
        self.assertEqual(calib.worst_trust(["calibrated_ok", "uncalibrated", "below_floor"]), "below_floor")
        self.assertEqual(calib.worst_trust(["below_floor", "rubber_stamp"]), "rubber_stamp")
        self.assertEqual(calib.worst_trust(["calibrated_ok"]), "calibrated_ok")
        self.assertIsNone(calib.worst_trust([]))


class TrustCacheTests(StoreBase):
    def test_no_labels_file_computes_nothing(self):
        rows = [row(1)]
        assist.save_verdicts(self.PID, {1: verdict_entry(rows[0], "pass", "j:a")})
        with patch.object(calib, "report", side_effect=AssertionError("must not compute")):
            self.assertEqual(calib.cached_report(self.PID, rows, ["human"], self.root / "synthetic" / "synthetic.jsonl"), [])
            trust = calib.trust_for_queue(self.PID, rows, [{"judge": "j:a", "rubber_stamp": False}],
                                          None, self.root / "synthetic" / "synthetic.jsonl")
        self.assertEqual(trust["j:a"]["status"], "uncalibrated")

    def test_cached_until_a_file_changes(self):
        rows = [row(i) for i in range(1, 4)]
        syn = self.root / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows_path = syn / "synthetic.jsonl"
        rows_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        for r in rows:
            assist.save_verdicts(self.PID, {r["id"]: verdict_entry(r, "pass", "j:a")})
        calib.save_labels(self.PID, "human", {1: calib.label_entry(rows[0], "pass", stratum="x", labeller="human")})
        calls = []
        real = calib.report

        def counting(*a, **kw):
            calls.append(1)
            return real(*a, **kw)

        with patch.object(calib, "report", side_effect=counting):
            first = calib.cached_report(self.PID, lambda: rows, ["human"], rows_path)
            second = calib.cached_report(self.PID, lambda: rows, ["human"], rows_path)
            self.assertIs(first, second)
            self.assertEqual(len(calls), 1)
            time.sleep(0.01)
            calib.save_labels(self.PID, "human", {2: calib.label_entry(rows[1], "fail", stratum="x", labeller="human")})
            third = calib.cached_report(self.PID, lambda: rows, ["human"], rows_path)
            self.assertEqual(len(calls), 2)
            self.assertEqual(third[0]["labelled"], 2)

    def test_an_edited_row_invalidates_the_report(self):
        rows = [row(i) for i in range(1, 4)]
        syn = self.root / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows_path = syn / "synthetic.jsonl"

        def write(rs):
            rows_path.write_text("\n".join(json.dumps(r) for r in rs) + "\n", encoding="utf-8")

        write(rows)
        for r in rows:
            assist.save_verdicts(self.PID, {r["id"]: verdict_entry(r, "pass", "j:a")})
        calib.save_labels(self.PID, "human", {1: calib.label_entry(rows[0], "pass", stratum="x", labeller="human")})
        state = calib.report_state(self.PID, rows_path)
        first = calib.cached_report(self.PID, rows, ["human"], rows_path, state=state)
        self.assertEqual(first[0]["labelled"], 1)
        # Same-size edit of the labelled row ("t1" -> "x1"): its label goes stale and the report is recomputed.
        edited = [dict(rows[0], text="x1"), *rows[1:]]
        size = rows_path.stat().st_size
        time.sleep(0.01)
        write(edited)
        self.assertEqual(rows_path.stat().st_size, size)
        state2 = calib.report_state(self.PID, rows_path)
        self.assertNotEqual(state, state2)
        self.assertIsNone(calib.cached_report_hit(self.PID, ["human"], state2))
        second = calib.cached_report(self.PID, edited, ["human"], rows_path, state=state2)
        self.assertEqual((second[0]["labelled"], second[0]["stale"]), (0, 1))

    def test_a_state_taken_before_a_write_never_serves_the_newer_rows(self):
        rows = [row(1)]
        syn = self.root / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows_path = syn / "synthetic.jsonl"
        rows_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        assist.save_verdicts(self.PID, {1: verdict_entry(rows[0], "pass", "j:a")})
        calib.save_labels(self.PID, "human", {1: calib.label_entry(rows[0], "pass", stratum="x", labeller="human")})
        old_state = calib.report_state(self.PID, rows_path)
        time.sleep(0.01)
        rows_path.write_text(json.dumps(dict(rows[0], text="later")) + "\n", encoding="utf-8")  # lands after the stat
        calib.cached_report(self.PID, rows, ["human"], rows_path, state=old_state)
        self.assertIsNone(calib.cached_report_hit(self.PID, ["human"], calib.report_state(self.PID, rows_path)))

    def test_overflow_evicts_the_least_recently_used_entry(self):
        with patch.object(calib, "_TRUST_CACHE_MAX", 2), patch.object(calib, "report", return_value=[]):
            calib._TRUST_CACHE.clear()
            calib.compute_report(1, [], ["human"], "s1")
            calib.compute_report(2, [], ["human"], "s2")
            self.assertEqual(calib.cached_report_hit(1, ["human"], "s1"), [])  # 1 is now the most recent
            calib.compute_report(3, [], ["human"], "s3")
            self.assertIsNotNone(calib.cached_report_hit(1, ["human"], "s1"))
            self.assertIsNone(calib.cached_report_hit(2, ["human"], "s2"))
            self.assertIsNotNone(calib.cached_report_hit(3, ["human"], "s3"))
            calib._TRUST_CACHE.clear()


class TrustQueueApi(unittest.TestCase):
    """The queue adds ``trust`` to each ``facets.judge_stats`` entry and, grouped by consensus, the judges
    behind each group with the worst status. Rows: 1-24 judged by good:a (fail on odd ids) and lax:b
    (always pass), all labelled by a human (odd = fail); 25 judged by a, b and new:c (agree_pass,
    unlabelled); 26 judged by few:d only (single pass), labelled pass."""

    def setUp(self):
        r = client.post("/api/projects", json={"name": f"trust-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        self.base = f"/api/projects/{self.pid}/synthetic"
        syn = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows = [row(i) for i in range(1, 27)]
        (syn / "synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        for r in rows[:25]:
            rid = r["id"]
            assist.save_verdicts(self.pid, {rid: verdict_entry(r, "fail" if rid % 2 and rid != 25 else "pass", "good:a")})
            assist.save_verdicts(self.pid, {rid: verdict_entry(r, "pass", "lax:b")})
        assist.save_verdicts(self.pid, {25: verdict_entry(rows[24], "pass", "new:c")})
        assist.save_verdicts(self.pid, {26: verdict_entry(rows[25], "pass", "few:d")})
        self.rows = rows

    def label_all(self):
        for r in self.rows[:24]:
            resp = client.post(f"{self.base}/calibration/labels", json={"row_id": r["id"], "label": "fail" if r["id"] % 2 else "pass"})
            self.assertEqual(resp.status_code, 200, resp.text)
        client.post(f"{self.base}/calibration/labels", json={"row_id": 26, "label": "pass"})

    def test_legacy_project_without_labels_is_all_uncalibrated(self):
        body = client.get(f"{self.base}/review-queue", params={"group_by": "assist_consensus"}).json()
        trust = {s["judge"]: s["trust"] for s in body["facets"]["judge_stats"]}
        self.assertEqual({j: t["status"] for j, t in trust.items()},
                         {"good:a": "uncalibrated", "lax:b": "uncalibrated", "new:c": "uncalibrated", "few:d": "uncalibrated"})
        self.assertEqual(trust["good:a"]["tier"], "human")
        # the V1 fields are all still there
        for s in body["facets"]["judge_stats"]:
            self.assertTrue({"judge", "judged", "pass", "fail", "unsure", "unparseable", "pass_rate", "rubber_stamp"} <= set(s))

    def test_trust_on_judge_stats_and_consensus_groups(self):
        self.label_all()
        body = client.get(f"{self.base}/review-queue", params={"group_by": "assist_consensus"}).json()
        trust = {s["judge"]: s["trust"] for s in body["facets"]["judge_stats"]}
        self.assertEqual(trust["good:a"]["status"], "calibrated_ok")
        self.assertEqual((trust["good:a"]["fail_recall"], trust["good:a"]["n"], trust["good:a"]["tier"]), (1.0, 24, "human"))
        self.assertEqual(trust["lax:b"]["status"], "below_floor")
        self.assertEqual(trust["lax:b"]["fail_recall"], 0.0)
        self.assertLess(trust["lax:b"]["ci"][1], calib.FAIL_RECALL_FLOOR)
        self.assertEqual(trust["new:c"]["status"], "uncalibrated")
        self.assertEqual(trust["few:d"]["status"], "insufficient")
        groups = {g["synth_source"]: g for g in body["groups"]}
        self.assertEqual(groups["agree_pass"]["judge_trust"], [
            {"judge": "good:a", "status": "calibrated_ok"}, {"judge": "lax:b", "status": "below_floor"},
            {"judge": "new:c", "status": "uncalibrated"},
        ])
        self.assertEqual(groups["agree_pass"]["trust"], "below_floor")
        self.assertEqual(groups["single:pass"]["judge_trust"], [{"judge": "few:d", "status": "insufficient"}])
        self.assertEqual(groups["single:pass"]["trust"], "insufficient")
        # other groupings carry no trust fields
        plain = client.get(f"{self.base}/review-queue").json()
        self.assertNotIn("trust", plain["groups"][0])
        # the label source picks the tier: no reference labels → uncalibrated
        client.put(f"{self.base}/calibration/settings", json={"label_source": "reference_model"})
        body = client.get(f"{self.base}/review-queue").json()
        self.assertEqual({s["trust"]["status"] for s in body["facets"]["judge_stats"]}, {"uncalibrated"})
        client.put(f"{self.base}/calibration/settings", json={"label_source": "both"})
        body = client.get(f"{self.base}/review-queue").json()
        self.assertEqual({s["judge"]: s["trust"]["status"] for s in body["facets"]["judge_stats"]}["lax:b"], "below_floor")

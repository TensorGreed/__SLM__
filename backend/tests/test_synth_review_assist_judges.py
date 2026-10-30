"""Review assist with several judges: the sidecar keeps one verdict per (row, judge), the queue shows
where judges agree or disagree, and the Job can resume / target rows by judge or consensus.

pytest-only (the Jobs these tests start are awaited in tearDown).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services import llm_budget_guard
from app.services import synth_review_assist_service as assist

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def pending(rid, **fields):
    return {"id": rid, "review_status": "pending", "synth_confidence": 0.8,
            "synth_source": "playbook:r:m", "label": "greeting", "text": f"t{rid}", **fields}


def ids_in(prompt):
    import re
    return [int(i) for i in re.findall(r'<row id="(\d+)"', prompt)]


def verdicts_reply(prompt, verdict_for):
    return json.dumps([
        {"id": rid, "verdict": verdict_for(rid), "reason_code": "ok" if verdict_for(rid) == "pass" else "incorrect", "note": f"note {rid}"}
        for rid in ids_in(prompt)
    ])


class JudgeA:
    name = "judgea"
    max_output_tokens = 4096
    prompts: list = []
    verdict_for = staticmethod(lambda rid: "pass")

    @classmethod
    def is_available(cls):
        return True

    def __init__(self, model=None):
        self.model = model

    def describe(self):
        return f"{self.name}:{self.model}" if self.model else self.name

    async def complete(self, prompt, **kw):
        type(self).prompts.append(prompt)
        return verdicts_reply(prompt, type(self).verdict_for)


class JudgeB(JudgeA):
    name = "judgeb"
    prompts: list = []
    verdict_for = staticmethod(lambda rid: "pass")


def run(coro):
    return asyncio.run(coro)


def real_shape_sidecar(n=10000):
    """Mirrors the real file: {row id: {verdict, reason_code, note, backend, job_id, at, sig}}, one judge."""
    return {
        str(i): {
            "verdict": "pass" if i % 7 else "fail", "reason_code": "ok" if i % 7 else "incorrect",
            "note": f"Note for {i}.", "backend": "ollama:gpt-oss:120b", "job_id": 768,
            "at": f"2026-10-29T10:49:{i % 60:02d}.742082+00:00", "sig": f"{i:016x}",
            **({"batch": True} if i % 11 == 0 else {}),
        }
        for i in range(1, n + 1)
    }


class StoreBase(unittest.TestCase):
    PID = 987631

    def setUp(self):
        shutil.rmtree(settings.DATA_DIR / "projects" / str(self.PID), ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(settings.DATA_DIR / "projects" / str(self.PID), ignore_errors=True)

    def sidecar_file(self) -> Path:
        return assist.sidecar_path(self.PID)

    def raw(self):
        return json.loads(self.sidecar_file().read_text(encoding="utf-8"))

    def entry(self, row, verdict, judge, at="2026-10-29T10:00:00+00:00", job_id=1, sig=None):
        return {"verdict": verdict, "reason_code": "ok" if verdict == "pass" else "incorrect", "note": "n",
                "backend": judge, "job_id": job_id, "at": at, "sig": sig or assist.row_signature(row)}

    def attached(self, rows):
        fresh = [dict(r) for r in rows]
        assist.attach_verdicts(self.PID, fresh)
        return fresh


class MigrationTests(StoreBase):
    def test_the_real_single_verdict_shape_migrates_losslessly_on_read(self):
        old = real_shape_sidecar()
        self.sidecar_file().parent.mkdir(parents=True, exist_ok=True)
        self.sidecar_file().write_text(json.dumps(old), encoding="utf-8")
        data = assist._read_sidecar(self.sidecar_file())
        self.assertEqual(len(data), 10000)
        for rid, entry in old.items():
            self.assertEqual(list(data[rid]), ["ollama:gpt-oss:120b"])
            migrated = data[rid]["ollama:gpt-oss:120b"]
            self.assertEqual({**migrated, "backend": "ollama:gpt-oss:120b"}, entry)  # nothing lost, nothing added
        # reading does not rewrite the file...
        self.assertEqual(self.raw(), old)

    def test_a_write_converts_the_whole_file_and_keeps_every_old_verdict(self):
        old = real_shape_sidecar(500)
        self.sidecar_file().parent.mkdir(parents=True, exist_ok=True)
        self.sidecar_file().write_text(json.dumps(old), encoding="utf-8")
        assist.save_verdicts(self.PID, {1: self.entry(pending(1), "fail", "judgeb:m")})
        new = self.raw()
        self.assertEqual(len(new), 500)
        self.assertTrue(all(isinstance(v, dict) and "verdict" not in v for v in new.values()))
        self.assertEqual(set(new["1"]), {"ollama:gpt-oss:120b", "judgeb:m"})
        self.assertEqual(new["1"]["ollama:gpt-oss:120b"]["sig"], old["1"]["sig"])
        self.assertEqual(new["2"], {"ollama:gpt-oss:120b": {k: v for k, v in old["2"].items() if k != "backend"}})

    def test_an_old_entry_without_a_backend_gets_a_placeholder_judge(self):
        self.sidecar_file().parent.mkdir(parents=True, exist_ok=True)
        self.sidecar_file().write_text(json.dumps({"1": {"verdict": "pass", "note": "n", "sig": "x"}}), encoding="utf-8")
        self.assertEqual(list(assist._read_sidecar(self.sidecar_file())["1"]), [assist.UNKNOWN_JUDGE])

    def test_junk_rows_are_dropped_not_fatal(self):
        self.sidecar_file().parent.mkdir(parents=True, exist_ok=True)
        self.sidecar_file().write_text(json.dumps({"1": "oops", "2": {"j": "bad"}, "3": {}}), encoding="utf-8")
        self.assertEqual(assist._read_sidecar(self.sidecar_file()), {})


class StorageAndConsensusTests(StoreBase):
    def test_same_judge_replaces_only_its_own_entry(self):
        row = pending(1)
        assist.save_verdicts(self.PID, {1: self.entry(row, "fail", "judgea:m", job_id=1)})
        assist.save_verdicts(self.PID, {1: self.entry(row, "pass", "judgeb:m", job_id=2)})
        assist.save_verdicts(self.PID, {1: self.entry(row, "pass", "judgea:m", job_id=3)})
        stored = self.raw()["1"]
        self.assertEqual({j: e["verdict"] for j, e in stored.items()}, {"judgea:m": "pass", "judgeb:m": "pass"})
        self.assertEqual((stored["judgea:m"]["job_id"], stored["judgeb:m"]["job_id"]), (3, 2))
        self.assertNotIn("backend", stored["judgea:m"])

    def test_saving_the_same_entry_twice_adds_no_history(self):
        row = pending(1)
        first = self.entry(row, "fail", "judgea:m", job_id=1)
        assist.save_verdicts(self.PID, {1: dict(first)})
        assist.save_verdicts(self.PID, {1: dict(first)})  # a flush retried after a cancel
        self.assertNotIn("history", self.raw()["1"]["judgea:m"])
        assist.save_verdicts(self.PID, {1: self.entry(row, "pass", "judgea:m", job_id=2)})
        self.assertEqual([h["job_id"] for h in self.raw()["1"]["judgea:m"]["history"]], [1])

    def test_consensus_rules(self):
        c = assist.consensus_of
        self.assertEqual(c([]), "none")
        self.assertEqual((c(["pass"]), c(["fail"]), c(["unsure"])), ("single",) * 3)
        self.assertEqual(c(["pass", "pass"]), "agree_pass")
        self.assertEqual(c(["fail", "fail", "fail"]), "agree_fail")
        self.assertEqual(c(["pass", "fail"]), "disagree")
        self.assertEqual(c(["pass", "unsure"]), "disagree")
        self.assertEqual(c(["unsure", "unsure"]), "disagree")
        self.assertEqual(c(["pass", "pass", "fail"]), "disagree")

    def test_attach_sets_judges_consensus_and_the_most_recent_as_review_assist(self):
        row = pending(1)
        assist.save_verdicts(self.PID, {1: self.entry(row, "pass", "judgea:m", at="2026-10-29T10:00:00+00:00")})
        assist.save_verdicts(self.PID, {1: self.entry(row, "fail", "judgeb:m", at="2026-10-29T11:00:00+00:00")})
        (out,) = self.attached([row])
        self.assertEqual(out["assist_consensus"], "disagree")
        self.assertEqual([e["judge"] for e in out["review_assist_judges"]], ["judgea:m", "judgeb:m"])
        self.assertEqual((out["review_assist"]["verdict"], out["review_assist"]["backend"]), ("fail", "judgeb:m"))
        self.assertNotIn("sig", out["review_assist"])
        self.assertTrue(all("sig" not in e for e in out["review_assist_judges"]))

    def test_a_stale_signature_is_excluded_from_the_consensus(self):
        row = pending(1)
        assist.save_verdicts(self.PID, {1: self.entry(row, "pass", "judgea:m")})
        assist.save_verdicts(self.PID, {1: self.entry(row, "fail", "judgeb:m", sig="stale0000000000")})
        (out,) = self.attached([row])
        self.assertEqual(out["assist_consensus"], "single")
        self.assertEqual([e["judge"] for e in out["review_assist_judges"]], ["judgea:m"])
        edited = {**row, "text": "edited"}
        (gone,) = self.attached([edited])
        self.assertNotIn("assist_consensus", gone)
        self.assertNotIn("review_assist", gone)

    def test_accept_keeps_the_verdicts_but_an_edit_drops_them(self):
        row = pending(1)
        assist.save_verdicts(self.PID, {1: self.entry(row, "pass", "judgea:m")})
        (out,) = self.attached([{**row, "review_status": "accepted"}])
        self.assertEqual(out["assist_consensus"], "single")


class JudgeProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls._patches = [
            patch("app.services.synth_backends.BACKEND_REGISTRY", [JudgeA, JudgeB]),
            patch("app.services.synth_playbook_service.BACKEND_REGISTRY", [JudgeA, JudgeB]),
        ]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        settings.AUTH_ENABLED = cls._prev_auth

    def tearDown(self):
        wait_for_jobs_idle(client)
        llm_budget_guard.reset()

    def setUp(self):
        for j in (JudgeA, JudgeB):
            j.prompts = []
            j.verdict_for = staticmethod(lambda rid: "pass")
        r = client.post("/api/projects", json={"name": f"judges-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        self.base = f"/api/projects/{self.pid}/synthetic"
        self.path = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "synthetic" / "synthetic.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(json.dumps(pending(i)) for i in range(1, 7)) + "\n", encoding="utf-8")

    def queue(self, **params):
        resp = client.get(f"{self.base}/review-queue", params=params)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def start(self, backend, **body):
        resp = client.post(f"{self.base}/review-assist?async_job=true", json={"backend": backend, **body})
        self.assertEqual(resp.status_code, 202, resp.text)
        wait_for_jobs_idle(client)
        return client.get(f"/api/jobs/{resp.json()['id']}").json()

    def two_judges(self):
        """A passes 1-4 and fails 5-6; B passes 1-2, fails 3, is unsure on 4 and fails 5; 6 has only A."""
        JudgeA.verdict_for = staticmethod(lambda rid: "fail" if rid >= 5 else "pass")
        self.start("judgea:m")
        JudgeB.verdict_for = staticmethod(lambda rid: {1: "pass", 2: "pass", 3: "fail", 4: "unsure", 5: "fail"}[rid])
        self.start("judgeb:m", row_ids=[1, 2, 3, 4, 5])

    def test_a_second_judge_adds_to_the_first_and_the_job_reports_consensus(self):
        JudgeA.verdict_for = staticmethod(lambda rid: "fail" if rid >= 5 else "pass")
        first = self.start("judgea:m")
        self.assertEqual(first["result"]["consensus"]["single"], 6)
        JudgeB.verdict_for = staticmethod(lambda rid: "pass" if rid <= 2 else "fail")
        second = self.start("judgeb:m")
        self.assertEqual(second["status"], "succeeded", second)
        self.assertEqual(second["result"]["verdicts"], {"pass": 2, "fail": 4, "unsure": 0})
        self.assertEqual(second["result"]["consensus"], {"none": 0, "single": 0, "agree_pass": 2, "agree_fail": 2, "disagree": 2})
        self.assertIn("agree 4 · disagree 2", second["progress_message"])
        self.assertNotIn("agree", first["progress_message"])

    def test_queue_filters_and_groups_by_consensus_with_judges_in_the_label(self):
        self.two_judges()
        q = self.queue(group_by="assist_consensus")
        groups = {g["synth_source"]: g for g in q["groups"]}
        self.assertEqual({k: g["count"] for k, g in groups.items()},
                         {"agree_pass": 2, "disagree": 2, "agree_fail": 1, "single:fail": 1})
        self.assertEqual(groups["agree_pass"]["label"], "agree pass · judgea:m + judgeb:m")
        self.assertEqual(groups["single:fail"]["label"], "single · fail · judgea:m")
        row = next(r for r in groups["disagree"]["rows"] if r["id"] == 3)
        self.assertEqual([(e["judge"], e["verdict"], e["note"]) for e in row["review_assist_judges"]],
                         [("judgea:m", "pass", "note 3"), ("judgeb:m", "fail", "note 3")])
        self.assertEqual(row["assist_consensus"], "disagree")
        self.assertEqual([g["count"] for g in self.queue(assist_consensus="disagree")["groups"]], [2])
        self.assertEqual(self.queue(assist_consensus="single")["total_pending"], 1)
        self.assertEqual(self.queue(assist_consensus="single:fail")["total_pending"], 1)
        facets = self.queue()["facets"]
        self.assertEqual({f["consensus"]: f["count"] for f in facets["consensus"]},
                         {"agree_pass": 2, "disagree": 2, "agree_fail": 1, "single": 1})
        self.assertEqual({f["judge"]: f["count"] for f in facets["judges"]}, {"judgea:m": 6, "judgeb:m": 5})

    def test_the_judge_filter_shows_one_judges_verdicts_and_group_by_uses_it(self):
        self.two_judges()
        q = self.queue(judge="judgeb:m", group_by="verdict")
        self.assertEqual({g["synth_source"]: g["count"] for g in q["groups"]}, {"pass": 2, "fail": 2, "unsure": 1})
        self.assertEqual(q["total_pending"], 5)  # row 6 was never judged by B
        self.assertEqual(self.queue(judge="judgea:m", verdict="fail")["total_pending"], 2)
        self.assertEqual(self.queue(judge="judgeb:m", verdict="fail")["total_pending"], 2)
        reasons = {g["synth_source"] for g in self.queue(judge="judgeb:m", group_by="assist_reason")["groups"]}
        self.assertEqual(reasons, {"pass:ok", "fail:incorrect", "unsure:incorrect"})

    def test_a_lone_judge_groups_into_single_pass_and_single_fail(self):
        JudgeA.verdict_for = staticmethod(lambda rid: "fail" if rid >= 5 else "pass")
        self.start("judgea:m")
        keys = {g["synth_source"]: g["count"] for g in self.queue(group_by="assist_consensus")["groups"]}
        self.assertEqual(keys, {"single:pass": 4, "single:fail": 2})

    def test_rows_without_a_verdict_are_the_none_group(self):
        self.start("judgea:m", row_ids=[1, 2])
        keys = {g["synth_source"]: g["count"] for g in self.queue(group_by="assist_consensus")["groups"]}
        self.assertEqual(keys, {"single:pass": 2, "none": 4})

    def test_bulk_accept_of_an_agree_pass_group_leaves_the_disagreements_pending(self):
        self.two_judges()
        groups = {g["synth_source"]: g for g in self.queue(group_by="assist_consensus")["groups"]}
        ids = [r["id"] for r in groups["agree_pass"]["rows"]]
        resp = client.post(f"{self.base}/review-queue/bulk-update", json={"row_ids": ids, "action": "accept"})
        self.assertEqual(resp.status_code, 200, resp.text)
        after = {g["synth_source"]: g["count"] for g in self.queue(group_by="assist_consensus")["groups"]}
        self.assertNotIn("agree_pass", after)
        self.assertEqual(after["disagree"], 2)

    def test_only_missing_judge_skips_rows_this_judge_already_has(self):
        self.start("judgea:m", row_ids=[1, 2, 3])
        self.assertEqual(len(JudgeA.prompts), 1)
        JudgeA.prompts = []
        done = self.start("judgea:m", only_missing_judge=True)
        self.assertEqual(done["result"]["rows_judged"], 3)
        self.assertEqual(sorted(i for p in JudgeA.prompts for i in ids_in(p)), [4, 5, 6])
        # another judge is not "missing" just because A has judged the row
        done = self.start("judgeb:m", only_missing_judge=True)
        self.assertEqual(done["result"]["rows_judged"], 6)

    def test_only_missing_judge_retries_rows_whose_reply_was_unusable(self):
        # A garbled reply / failed call leaves the rows unsure + judge_unparseable; a resume must retry them.
        original = JudgeA.complete

        async def garbled(self, prompt, **kw):
            type(self).prompts.append(prompt)
            return "not json"

        JudgeA.complete = garbled
        try:
            self.start("judgea:m", row_ids=[1, 2])
        finally:
            JudgeA.complete = original
        self.start("judgea:m", row_ids=[3])
        JudgeA.prompts = []
        done = self.start("judgea:m", only_missing_judge=True)
        self.assertEqual(sorted(i for p in JudgeA.prompts for i in ids_in(p)), [1, 2, 4, 5, 6])
        self.assertEqual(done["result"]["rows_judged"], 5)

    def test_only_missing_judge_with_nothing_left_is_a_400(self):
        self.start("judgea:m")
        resp = client.post(f"{self.base}/review-assist?async_job=true", json={"backend": "judgea:m", "only_missing_judge": True})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["error_code"], "no_pending_rows")

    def test_the_consensus_filter_judges_only_matching_rows(self):
        self.two_judges()
        JudgeA.prompts = []
        JudgeB.prompts = []
        JudgeB.verdict_for = staticmethod(lambda rid: "pass")
        self.start("judgeb:m", consensus=["disagree"])
        self.assertEqual(sorted(i for p in JudgeB.prompts for i in ids_in(p)), [3, 4])

    def test_prior_verdicts_picks_rows_another_judge_failed_or_was_unsure_on(self):
        self.two_judges()
        JudgeB.prompts = []
        self.start("judgeb:m", prior_verdicts=["fail", "unsure"], only_missing_judge=False)
        # A said fail on 5, 6; B's own earlier verdicts do not count as "prior"
        self.assertEqual(sorted(i for p in JudgeB.prompts for i in ids_in(p)), [5, 6])

    def test_dry_run_applies_the_judge_filters_and_bad_values_are_400(self):
        self.start("judgea:m", row_ids=[1, 2])
        body = client.post(f"{self.base}/review-assist", json={"backend": "judgea:m", "only_missing_judge": True, "dry_run": True}).json()
        self.assertEqual(body["rows"], 4)
        resp = client.post(f"{self.base}/review-assist?async_job=true", json={"backend": "judgea:m", "consensus": ["maybe"]})
        self.assertEqual(resp.status_code, 400, resp.text)
        resp = client.post(f"{self.base}/review-assist?async_job=true", json={"backend": "judgea:m", "prior_verdicts": ["maybe"]})
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_a_duplicated_project_nulls_job_ids_in_both_formats(self):
        from app.services.project_duplicate_service import _drop_copied_job_ids

        root = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "synthetic"
        (root / "review_assist.json").write_text(json.dumps({
            "1": {"verdict": "pass", "backend": "a", "job_id": 5, "sig": "x"},
            "2": {"a": {"verdict": "pass", "job_id": 5, "sig": "x", "history": [{"verdict": "fail", "job_id": 4, "sig": "x"}]},
                  "b": {"verdict": "fail", "job_id": 6, "sig": "y"}},
        }), encoding="utf-8")
        _drop_copied_job_ids(self.pid)
        data = json.loads((root / "review_assist.json").read_text(encoding="utf-8"))
        self.assertIsNone(data["1"]["job_id"])
        self.assertEqual([e["job_id"] for e in data["2"].values()], [None, None])
        self.assertIsNone(data["2"]["a"]["history"][0]["job_id"])



# ── Epic V1: decision provenance, judge identity, rubber-stamp flag ─────────


class JudgeIdentityTests(StoreBase):
    def test_rubric_sha_is_the_hash_of_the_rubric_text_the_prompt_carries(self):
        row = pending(1)
        kind = assist.row_kind_of(row)
        prompt = assist.build_prompt([row], rubric_text="  only allowed ops: a, b  ")
        self.assertIn(assist._rubric_block(kind), prompt)
        self.assertIn(assist._extra_rubric_block("  only allowed ops: a, b  "), prompt)
        import hashlib
        expected = hashlib.sha256(
            (assist._rubric_block(kind) + assist._extra_rubric_block("only allowed ops: a, b")).encode("utf-8")
        ).hexdigest()[:12]
        self.assertEqual(assist.rubric_sha(kind, "  only allowed ops: a, b  "), expected)
        self.assertNotEqual(assist.rubric_sha(kind, "x"), assist.rubric_sha(kind, "y"))
        self.assertNotEqual(assist.rubric_sha(kind, None), assist.rubric_sha(kind, "x"))
        self.assertEqual(assist.rubric_sha(kind, None), assist.rubric_sha(kind, "   "))  # blank = no extra block
        self.assertNotEqual(assist.rubric_sha("qa"), assist.rubric_sha("code_example"))
        entry = assist._entry({"verdict": "pass", "reason_code": "ok", "note": ""}, row, backend="j", job_id=1,
                              batch=False, rubric_text="x")
        self.assertEqual(entry["rubric_sha"], assist.rubric_sha(kind, "x"))

    def test_a_rejudge_keeps_up_to_five_previous_entries_newest_first(self):
        row = pending(1)
        for job in range(1, 9):
            assist.save_verdicts(self.PID, {1: self.entry(row, "pass" if job % 2 else "fail", "judgea:m", job_id=job)})
        assist.save_verdicts(self.PID, {1: self.entry(row, "pass", "judgeb:m", job_id=99)})
        stored = self.raw()["1"]
        self.assertEqual(stored["judgea:m"]["job_id"], 8)
        history = stored["judgea:m"]["history"]
        self.assertEqual([h["job_id"] for h in history], [7, 6, 5, 4, 3])
        self.assertTrue(all("history" not in h for h in history))
        self.assertNotIn("history", stored["judgeb:m"])  # first verdict of that judge
        # the queue view shows current verdicts only
        (out,) = self.attached([row])
        self.assertTrue(all("history" not in e for e in out["review_assist_judges"]))
        self.assertNotIn("history", out["review_assist"])
        self.assertEqual(out["assist_consensus"], "disagree")  # only the current entries vote: A fail (job 8), B pass

    def test_legacy_entries_without_identity_fields_count_as_before(self):
        row = pending(1)
        legacy = {"verdict": "fail", "reason_code": "incorrect", "note": "n", "backend": "old:m", "job_id": 3,
                  "at": "2026-10-01T00:00:00+00:00", "sig": assist.row_signature(row)}
        self.sidecar_file().parent.mkdir(parents=True, exist_ok=True)
        self.sidecar_file().write_text(json.dumps({"1": legacy}), encoding="utf-8")
        assist.save_verdicts(self.PID, {1: {**self.entry(row, "fail", "new:m"), "rubric_sha": "abc"}})
        (out,) = self.attached([row])
        self.assertEqual(out["assist_consensus"], "agree_fail")
        self.assertEqual({e["judge"] for e in out["review_assist_judges"]}, {"old:m", "new:m"})
        stats = {s["judge"]: s for s in assist.judge_stats([out])}
        self.assertEqual((stats["old:m"]["judged"], stats["old:m"]["fail"]), (1, 1))
        # a same-judge re-run over a legacy entry keeps it as history
        assist.save_verdicts(self.PID, {1: self.entry(row, "pass", "old:m", job_id=4)})
        self.assertEqual(self.raw()["1"]["old:m"]["history"][0]["job_id"], 3)

    def test_decision_stamps_do_not_change_the_signature_or_reach_the_judge(self):
        row = pending(1)
        decided = {**row, "review_status": "accepted", "decided_via": "manual", "decided_at": "2026-10-30T00:00:00+00:00"}
        self.assertEqual(assist.row_signature(row), assist.row_signature(decided))
        self.assertNotIn("decided_", assist.render_row(decided))


def stats_rows(judge, passes, fails=0, unsure=0, unparseable=0):
    verdicts = ["pass"] * passes + ["fail"] * fails + ["unsure"] * unsure
    rows = [{"review_assist_judges": [{"judge": judge, "verdict": v, "reason_code": "ok" if v == "pass" else "x"}]}
            for v in verdicts]
    rows += [{"review_assist_judges": [{"judge": judge, "verdict": "unsure", "reason_code": assist.UNPARSEABLE_REASON}]}
             for _ in range(unparseable)]
    return rows


class RubberStampTests(unittest.TestCase):
    def stat(self, rows):
        (s,) = assist.judge_stats(rows)
        return s

    def test_counts_and_pass_rate(self):
        s = self.stat(stats_rows("j", 6, 3, 1, unparseable=2))
        self.assertEqual({k: s[k] for k in ("judge", "judged", "pass", "fail", "unsure", "unparseable", "pass_rate", "rubber_stamp")},
                         {"judge": "j", "judged": 10, "pass": 6, "fail": 3, "unsure": 1, "unparseable": 2,
                          "pass_rate": 0.6, "rubber_stamp": False})

    def test_threshold_edges(self):
        self.assertEqual((assist.RUBBER_STAMP_MIN_ROWS, assist.RUBBER_STAMP_PASS_RATE), (50, 0.98))
        self.assertTrue(self.stat(stats_rows("j", 49, 1))["rubber_stamp"])        # 50 rows, exactly 98%
        self.assertTrue(self.stat(stats_rows("j", 50))["rubber_stamp"])           # 50 rows, 100%
        self.assertFalse(self.stat(stats_rows("j", 49))["rubber_stamp"])          # 49 rows, 100%: too few
        self.assertFalse(self.stat(stats_rows("j", 97, 3))["rubber_stamp"])       # 97%
        self.assertFalse(self.stat(stats_rows("j", 48, 0, 2))["rubber_stamp"])    # 96%: unsure is not a pass
        # unparseable verdicts are no opinion: they neither count as judged nor dilute the pass rate
        self.assertTrue(self.stat(stats_rows("j", 50, unparseable=40))["rubber_stamp"])
        self.assertFalse(self.stat(stats_rows("j", 10, unparseable=60))["rubber_stamp"])

    def test_no_verdicts_no_stats(self):
        self.assertEqual(assist.judge_stats([{"id": 1}]), [])
        (s,) = assist.judge_stats(stats_rows("j", 0, unparseable=3))
        self.assertEqual((s["judged"], s["pass_rate"], s["rubber_stamp"]), (0, None, False))


class DecisionProvenanceTests(JudgeProject):
    def rows_on_disk(self):
        return {r["id"]: r for r in (json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip())}

    def bulk(self, **body):
        resp = client.post(f"{self.base}/review-queue/bulk-update", json=body)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def assert_recent(self, stamp):
        from datetime import datetime, timezone

        at = datetime.fromisoformat(stamp)
        self.assertIsNotNone(at.tzinfo)
        self.assertLess(abs((datetime.now(timezone.utc) - at).total_seconds()), 120)

    def test_accept_reject_and_by_source_stamp_how_and_when(self):
        self.bulk(row_ids=[1], action="accept", decided_via="manual")
        self.bulk(row_ids=[2], action="reject", reject_reason="assist:agree_fail",
                  decided_via="group:assist_consensus:agree_fail")
        self.bulk(row_ids=[3], action="accept")  # omitted -> api
        rows = self.rows_on_disk()
        self.assertEqual(rows[1]["decided_via"], "manual")
        self.assertEqual(rows[2]["decided_via"], "group:assist_consensus:agree_fail")
        self.assertEqual(rows[2]["reject_reason"], "assist:agree_fail")
        self.assertEqual(rows[3]["decided_via"], "api")
        for rid in (1, 2, 3):
            self.assert_recent(rows[rid]["decided_at"])
        for rid in (4, 5, 6):
            self.assertNotIn("decided_via", rows[rid])
            self.assertNotIn("decided_at", rows[rid])
        resp = client.post(f"{self.base}/review-queue/bulk-update-by-source",
                           json={"source": "playbook:r:m", "action": "accept"})
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = self.rows_on_disk()
        self.assertEqual({rows[r]["decided_via"] for r in (4, 5, 6)}, {"source:playbook:r:m"})
        self.assertEqual(rows[1]["decided_via"], "manual")  # already decided: untouched
        # the queue returns the stamps per row, out of the preview
        accepted = {r["id"]: r for g in self.queue()["accepted_groups"] for r in g["rows"]}
        self.assertEqual(accepted[1]["decided_via"], "manual")
        self.assertNotIn("decided", accepted[1]["preview"])

    def test_decided_via_is_validated_loosely(self):
        long_key = "group:verdict:" + "x" * 300
        self.bulk(row_ids=[1], action="accept", decided_via=long_key)
        self.assertEqual(self.rows_on_disk()[1]["decided_via"], long_key[:128])
        self.bulk(row_ids=[2], action="accept", decided_via="  ")
        self.assertEqual(self.rows_on_disk()[2]["decided_via"], "api")
        for bad in ("robot", "group:", "group:   "):
            resp = client.post(f"{self.base}/review-queue/bulk-update",
                               json={"row_ids": [3], "action": "accept", "decided_via": bad})
            self.assertEqual(resp.status_code, 422, (bad, resp.text))
        self.assertEqual(self.rows_on_disk()[3]["review_status"], "pending")

    def test_verdicts_survive_an_accept_and_a_restore_clears_the_stamps(self):
        self.start("judgea:m")
        before = assist.row_signature(self.rows_on_disk()[1])
        self.bulk(row_ids=[1], action="accept", decided_via="manual")
        self.bulk(row_ids=[2], action="reject", decided_via="manual")
        rows = self.rows_on_disk()
        self.assertEqual(assist.row_signature(rows[1]), before)
        accepted = {r["id"]: r for g in self.queue()["accepted_groups"] for r in g["rows"]}
        self.assertEqual([e["judge"] for e in accepted[1]["review_assist_judges"]], ["judgea:m"])
        stats = {s["judge"]: s for s in self.queue()["facets"]["judge_stats"]}
        self.assertEqual(stats["judgea:m"]["judged"], 6)  # accepted / rejected rows still count
        resp = client.post(f"{self.base}/review-queue/restore", json={"row_ids": [2], "dry_run": False})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertNotIn("decided_via", self.rows_on_disk()[2])
        self.assertNotIn("decided_at", self.rows_on_disk()[2])

    def test_a_job_stamps_rubric_sha_and_a_rerun_keeps_history(self):
        self.start("judgea:m", rubric_text="first rubric")
        self.start("judgea:m", rubric_text="second rubric")
        stored = json.loads(assist.sidecar_path(self.pid).read_text(encoding="utf-8"))["1"]["judgea:m"]
        kind = assist.row_kind_of(pending(1))
        self.assertEqual(stored["rubric_sha"], assist.rubric_sha(kind, "second rubric"))
        self.assertEqual([h["rubric_sha"] for h in stored["history"]], [assist.rubric_sha(kind, "first rubric")])

    def test_the_queue_reports_judge_stats_and_flags_a_rubber_stamp(self):
        self.path.write_text("\n".join(json.dumps(pending(i)) for i in range(1, 61)) + "\n", encoding="utf-8")
        JudgeA.verdict_for = staticmethod(lambda rid: "pass")
        JudgeB.verdict_for = staticmethod(lambda rid: "fail" if rid % 3 == 0 else "pass")
        self.start("judgea:m", rows_per_call=25)
        self.start("judgeb:m", rows_per_call=25)
        stats = {s["judge"]: s for s in self.queue()["facets"]["judge_stats"]}
        self.assertEqual((stats["judgea:m"]["judged"], stats["judgea:m"]["pass_rate"], stats["judgea:m"]["rubber_stamp"]),
                         (60, 1.0, True))
        self.assertEqual((stats["judgeb:m"]["pass"], stats["judgeb:m"]["fail"], stats["judgeb:m"]["rubber_stamp"]),
                         (40, 20, False))


if __name__ == "__main__":
    unittest.main()

"""Compare page row-paired evidence (Epic W1).

Covers ``paired_comparison_stats.pair_row_scores`` (pairing by row key),
``compare_row_evidence_service.compare_provenance`` (answer key / judge /
scoring comparability) and ``compare_experiments(...)["row_evidence"]``
end to end on seeded EvalResults: the 78-row within-noise example, a
mismatched split (recorded and inferred), a mismatched judge, missing
row_scores, and the re-score offer.

The numbers are worked by hand in the comments, not recomputed from the
code under test.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType, DatasetVersion
from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode
from app.models.job import Job, JobStatus
from app.models.project import Project, ProjectStatus
from app.services.compare_row_evidence_service import NO_ROW_SCORES_MESSAGE, answer_key_stamp, compare_provenance
from app.services.experiment_comparison_service import compare_experiments
from app.services.paired_comparison_stats import pair_row_scores

_client_cm = None


def setUpModule():  # noqa: N802 - unittest hook
    global _client_cm
    _client_cm = TestClient(app)
    _client_cm.__enter__()


def tearDownModule():  # noqa: N802 - unittest hook
    if _client_cm is not None:
        _client_cm.__exit__(None, None, None)


def _rows_78(before: bool) -> dict:
    """78 rows scored 0/1 on ``correct``: rows 0-1 B fixed (A 0 -> B 1),
    rows 2-8 B broke (A 1 -> B 0), rows 9-78 unchanged (half right)."""
    keys = [f"row{i:02d}" for i in range(78)]
    correct = []
    for i in range(78):
        if i < 2:
            correct.append(0 if before else 1)
        elif i < 9:
            correct.append(1 if before else 0)
        else:
            correct.append(i % 2)
    return {"keys": keys, "correct": correct}


class PairRowScoresTests(unittest.TestCase):
    def test_pairs_by_key_when_order_differs(self):
        a = {"keys": ["k1", "k2", "k3", "k5"], "f1": [1.0, 0.0, 0.5, 0.2]}
        # Same rows k1-k3 in another order, plus k4 only in B; k5 only in A.
        b = {"keys": ["k3", "k4", "k1", "k2"], "f1": [0.0, 1.0, 1.0, 1.0]}
        paired = pair_row_scores(a, b, "f1")
        # Pairs follow B's order: k3 (0.5 -> 0.0), k1 (1.0 -> 1.0), k2 (0.0 -> 1.0).
        self.assertEqual(paired["before"], [0.5, 1.0, 0.0])
        self.assertEqual(paired["after"], [0.0, 1.0, 1.0])
        self.assertEqual(paired["only_before"], 1)
        self.assertEqual(paired["only_after"], 1)
        self.assertEqual(paired["unscored"], 0)
        self.assertFalse(paired["same_order"])

    def test_unscored_row_is_dropped_and_counted(self):
        a = {"keys": ["k1", "k2"], "judge_correct": [1.0, None]}
        b = {"keys": ["k2", "k1"], "judge_correct": [0.5, 0.0]}
        paired = pair_row_scores(a, b, "judge_correct")
        self.assertEqual((paired["before"], paired["after"]), ([1.0], [0.0]))
        self.assertEqual(paired["unscored"], 1)

    def test_missing_score_list_is_none(self):
        self.assertIsNone(pair_row_scores({"keys": ["k1"], "f1": [1.0]}, {"keys": ["k1"]}, "f1"))
        self.assertIsNone(pair_row_scores(None, {"keys": ["k1"], "f1": [1.0]}, "f1"))


def _prov(*, version=3, sha="a" * 64, dtype="test", judge="judge-x", mode=None, eval_type="exact_match"):
    return {
        "answer_key": {"dataset_id": 1, "dataset_type": dtype, "version": version, "sha256": sha,
                       "version_source": "recorded"},
        "judge": {"answer": judge, "rubric": None, "rubric_sha": None, "code_example": None},
        "scoring": {"eval_type": eval_type, "scoring_mode": mode, "fingerprints": {}},
    }


class CompareProvenanceTests(unittest.TestCase):
    def test_same_everything_is_comparable(self):
        result = compare_provenance(_prov(), _prov())
        self.assertTrue(result["comparable"])
        self.assertEqual(result["differences"], [])

    def test_mismatched_split(self):
        result = compare_provenance(_prov(version=3, sha="a" * 64), _prov(version=5, sha="b" * 64))
        self.assertFalse(result["comparable"])
        self.assertEqual([d["message"] for d in result["differences"]], ["different answer key: split v3 vs v5"])

    def test_same_hash_different_version_is_still_the_same_answer_key(self):
        # A re-prepare that wrote an identical test file: content wins over the counter.
        self.assertTrue(compare_provenance(_prov(version=3), _prov(version=4))["comparable"])

    def test_mismatched_judge(self):
        result = compare_provenance(_prov(judge="judge-x"), _prov(judge="judge-y"))
        self.assertFalse(result["comparable"])
        self.assertEqual([d["message"] for d in result["differences"]], ["different judge: judge-x vs judge-y"])

    def test_judged_vs_unjudged(self):
        result = compare_provenance(_prov(judge=None), _prov(judge="judge-y"))
        self.assertEqual(result["differences"][0]["message"], "different judge: no judge vs judge-y")

    def test_mismatched_scoring_mode(self):
        result = compare_provenance(_prov(mode="span_set"), _prov(mode="field_match"))
        self.assertEqual(
            [d["message"] for d in result["differences"]], ["different scoring mode: span_set vs field_match"]
        )

    def test_different_dataset_type(self):
        result = compare_provenance(_prov(dtype="test"), _prov(dtype="gold_test"))
        self.assertEqual(
            result["differences"][0]["message"],
            "different answer key: test examples, split v3 vs answer key (final), split v3",
        )

    def test_llm_judge_scorer_is_still_a_scoring_difference(self):
        result = compare_provenance(_prov(eval_type="exact_match"), _prov(eval_type="llm_judge"))
        self.assertEqual([d["message"] for d in result["differences"]], ["different scoring: exact_match vs llm_judge"])

    def test_exact_match_and_f1_labels_are_the_same_scorer(self):
        self.assertTrue(compare_provenance(_prov(eval_type="exact_match"), _prov(eval_type="f1"))["comparable"])

    def test_different_decoding_is_a_caveat_not_a_block(self):
        a, b = _prov(), _prov()
        a["scoring"]["decoding"] = {"do_sample": False, "repetition_penalty": 1.1}
        b["scoring"]["decoding"] = {"do_sample": False, "repetition_penalty": 1.0}
        result = compare_provenance(a, b)
        self.assertTrue(result["comparable"])
        self.assertEqual(result["caveats"], [
            "Decoded differently: repetition_penalty 1.1 vs 1.0. The gap may partly come from decoding, not the model."
        ])
        b["scoring"]["decoding"] = None
        self.assertIn("B's eval predates recorded decoding", compare_provenance(a, b)["caveats"][0])
        b["scoring"]["decoding"] = dict(a["scoring"]["decoding"])
        self.assertEqual(compare_provenance(a, b)["caveats"], [])

    def test_unrecorded_answer_key_is_a_note_not_a_difference(self):
        a = _prov()
        a["answer_key"] = {"dataset_id": None, "dataset_type": "test", "version": None, "sha256": None}
        result = compare_provenance(a, _prov())
        self.assertTrue(result["comparable"])
        self.assertEqual(len(result["notes"]), 1)


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
_seed_counter = 0


def _seed(*, a: dict, b: dict, versions=((3, "a" * 64, 0), (5, "b" * 64, 10))) -> tuple[int, int, int]:
    """A project with a TEST dataset (versions: (version, sha, day offset)) and
    two completed runs, each with one EvalResult built from ``a`` / ``b``:
    ``{metrics, details, eval_type?, day}``. ``details.dataset.id`` = the TEST
    dataset unless the caller set one."""

    global _seed_counter
    _seed_counter += 1
    tag = _seed_counter

    async def _go():
        async with async_session_factory() as session:
            proj = Project(name=f"Row evidence #{tag}", status=ProjectStatus.DRAFT, selected_recipe={"recipe_id": "qa-sft"})
            session.add(proj)
            await session.flush()
            ds = Dataset(project_id=proj.id, name="Test Set", dataset_type=DatasetType.TEST)
            session.add(ds)
            await session.flush()
            for version, sha, day in versions:
                session.add(DatasetVersion(
                    dataset_id=ds.id, version=version, file_path="/tmp/none.jsonl",
                    manifest={"split": "test", "sha256": sha}, created_at=T0 + timedelta(days=day),
                ))
            exps = []
            for name in ("A", "B"):
                exp = Experiment(project_id=proj.id, name=name, base_model="base-m", training_mode=TrainingMode.SFT,
                                 config={"base_model": "base-m"}, status=ExperimentStatus.COMPLETED)
                session.add(exp)
                exps.append(exp)
            await session.flush()
            for exp, spec in zip(exps, (a, b)):
                details = dict(spec.get("details") or {})
                dataset = dict(details.get("dataset") or {})
                dataset.setdefault("id", ds.id)
                dataset.setdefault("dataset_type", "test")
                details["dataset"] = dataset
                session.add(EvalResult(
                    experiment_id=exp.id, dataset_name="test", eval_type=spec.get("eval_type", "exact_match"),
                    metrics=spec["metrics"], details=details, pass_rate=spec["metrics"].get("exact_match"),
                    created_at=T0 + timedelta(days=spec.get("day", 20)),
                ))
            await session.commit()
            return proj.id, exps[0].id, exps[1].id

    return asyncio.run(_go())


def _compare(pid: int, a: int, b: int) -> dict:
    async def _go():
        async with async_session_factory() as session:
            return await compare_experiments(session, project_id=pid, exp_a_id=a, exp_b_id=b)

    return asyncio.run(_go())


def _add_job(pid: int, experiment_id: int, status) -> int:
    """A lift-check Job row with no runner behind it (``_finish_job`` ends it)."""

    async def _go():
        async with async_session_factory() as session:
            job = Job(kind="post_training_lift_eval", title="lift", status=status, project_id=pid,
                      params={"experiment_id": experiment_id})
            session.add(job)
            await session.commit()
            return job.id

    return asyncio.run(_go())


def _finish_job(job_id: int) -> None:
    async def _go():
        async with async_session_factory() as session:
            job = await session.get(Job, job_id)
            job.status = JobStatus.CANCELLED
            await session.commit()

    asyncio.run(_go())


def _active_lift(pid: int, experiment_id: int):
    async def _go():
        from app.services.post_training_eval_service import active_lift_job_id

        async with async_session_factory() as session:
            return await active_lift_job_id(session, pid, experiment_id)

    return asyncio.run(_go())


def _judge(label):
    return {"judge": label, "score": 0.5, "judged": 78}


# A: rows 0-1 wrong, 2-8 right, 9-77 alternate -> 7 + 35 = 42 right of 78.
# B: rows 0-1 right, 2-8 wrong -> 2 + 35 = 37 right of 78.
A_EM = 42 / 78
B_EM = 37 / 78


class CompareRowEvidenceTests(unittest.TestCase):
    def test_78_rows_two_better_seven_worse_is_within_noise(self):
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM}, "day": 20,
               "details": {"dataset": {"sha256": "b" * 64, "version": 5}, "row_scores": _rows_78(True)}},
            b={"metrics": {"exact_match": B_EM}, "day": 21,
               "details": {"dataset": {"sha256": "b" * 64, "version": 5}, "row_scores": _rows_78(False)}},
        )
        ev = _compare(pid, a, b)["row_evidence"]
        self.assertEqual(ev["metric_id"], "exact_match")
        self.assertEqual((ev["n"], ev["better"], ev["worse"], ev["same"]), (78, 2, 7, 69))
        # Per-row diffs: two +1, seven -1, 69 zeros. mean = -5/78 = -0.0641;
        # sample variance = (9 - 78 * (5/78)^2) / 77 = 0.11272; t(77) = 1.980;
        # half-width = 1.980 * sqrt(0.11272 / 78) = 0.07527 -> CI -0.1394 to +0.0112.
        self.assertAlmostEqual(ev["mean_diff"], -0.064103, places=5)
        self.assertAlmostEqual(ev["ci_low"], -0.13937, places=4)
        self.assertAlmostEqual(ev["ci_high"], 0.01117, places=4)
        self.assertEqual(ev["verdict"], "within_noise")
        self.assertEqual((ev["only_in_a"], ev["only_in_b"]), (0, 0))
        self.assertIsNone(ev["rescore"])
        self.assertTrue(ev["comparability"]["comparable"])

    def test_pairs_by_key_with_rows_in_another_order(self):
        rows_b = _rows_78(False)
        order = list(reversed(range(78)))
        rows_b = {"keys": [rows_b["keys"][i] for i in order], "correct": [rows_b["correct"][i] for i in order]}
        same_key = {"dataset": {"sha256": "b" * 64, "version": 5}}
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM}, "details": {**same_key, "row_scores": _rows_78(True)}},
            b={"metrics": {"exact_match": B_EM}, "details": {**same_key, "row_scores": rows_b}},
        )
        ev = _compare(pid, a, b)["row_evidence"]
        self.assertEqual((ev["n"], ev["better"], ev["worse"], ev["same"]), (78, 2, 7, 69))
        self.assertEqual(ev["verdict"], "within_noise")

    def test_mismatched_split_recorded_gives_no_verdict(self):
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM}, "day": 5,
               "details": {"dataset": {"sha256": "a" * 64, "version": 3}, "row_scores": _rows_78(True)}},
            b={"metrics": {"exact_match": B_EM}, "day": 21,
               "details": {"dataset": {"sha256": "b" * 64, "version": 5}, "row_scores": _rows_78(False)}},
        )
        body = _compare(pid, a, b)
        ev = body["row_evidence"]
        self.assertEqual(ev["verdict"], "not_comparable")
        self.assertEqual(ev["message"], "different answer key: split v3 vs v5")
        self.assertEqual(ev["n"], 0)
        self.assertIsNone(ev["ci_low"])
        # Metric deltas still shown.
        self.assertEqual(body["metric_deltas"][0]["metric_id"], "exact_match")
        # Older eval is A's -> re-score A. No weights on disk in a test.
        self.assertEqual(ev["rescore"]["side"], "a")
        self.assertEqual(ev["rescore"]["experiment_id"], a)
        self.assertFalse(ev["rescore"]["available"])
        self.assertEqual(ev["rescore"]["unavailable_reason"], "no_model_weights")

    def test_mismatched_split_inferred_for_old_results(self):
        # Neither result recorded its split: A evaluated on day 5 (v3 was
        # the newest version then), B on day 20 (v5). Inferred hashes differ.
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM}, "day": 5, "details": {"row_scores": _rows_78(True)}},
            b={"metrics": {"exact_match": B_EM}, "day": 20, "details": {"row_scores": _rows_78(False)}},
        )
        body = _compare(pid, a, b)
        ev = body["row_evidence"]
        # Both versions rest on inference: still no verdict, worded "may differ".
        self.assertEqual(ev["verdict"], "not_comparable")
        self.assertEqual(ev["message"], "answer key may differ: split v3 vs v5 (both inferred from eval time)")
        self.assertFalse(ev["comparability"]["certain"])
        self.assertEqual(body["a"]["provenance"]["answer_key"]["version_source"], "inferred")
        # An inferred hash is never reported as a recorded one.
        self.assertIsNone(body["a"]["provenance"]["answer_key"]["sha256"])
        self.assertEqual(body["a"]["provenance"]["answer_key"]["inferred_sha256"], "a" * 64)
        self.assertEqual(body["a"]["provenance"]["answer_key"]["version"], 3)
        self.assertEqual(body["b"]["provenance"]["answer_key"]["version"], 5)

    def test_one_side_inferred_names_which(self):
        # A is old (no stamp, day 5 -> v3 inferred); B recorded v5 + its hash.
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM}, "day": 5, "details": {"row_scores": _rows_78(True)}},
            b={"metrics": {"exact_match": B_EM}, "day": 21,
               "details": {"dataset": {"sha256": "b" * 64, "version": 5}, "row_scores": _rows_78(False)}},
        )
        ev = _compare(pid, a, b)["row_evidence"]
        self.assertEqual(ev["message"], "answer key may differ: split v3 vs v5 (v3 inferred from eval time)")
        self.assertTrue(ev["comparability"]["differences"][0]["inferred"])

    def test_lift_check_result_pairs_with_manual_heldout_result_on_same_file(self):
        # The lift check labels its eval "exact_match" (eval_type_for_project); a
        # manual held-out eval of the same file was run as "f1". Same scorer family,
        # same file hash, same scoring mode -> comparable, verdict given.
        same_key = {"dataset": {"sha256": "b" * 64, "version": 5}}
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM}, "eval_type": "exact_match", "day": 20,
               "details": {**same_key, "row_scores": _rows_78(True)}},
            b={"metrics": {"exact_match": B_EM, "f1": 0.5}, "eval_type": "f1", "day": 21,
               "details": {**same_key, "row_scores": _rows_78(False)}},
        )
        ev = _compare(pid, a, b)["row_evidence"]
        self.assertTrue(ev["comparability"]["comparable"])
        self.assertEqual(ev["verdict"], "within_noise")
        self.assertEqual((ev["better"], ev["worse"]), (2, 7))

    def test_rescore_offer_in_progress_while_lift_job_runs(self):
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM}, "day": 5,
               "details": {"dataset": {"sha256": "a" * 64, "version": 3}, "row_scores": _rows_78(True)}},
            b={"metrics": {"exact_match": B_EM}, "day": 21,
               "details": {"dataset": {"sha256": "b" * 64, "version": 5}, "row_scores": _rows_78(False)}},
        )
        job_id = _add_job(pid, a, JobStatus.RUNNING)
        try:
            offer = _compare(pid, a, b)["row_evidence"]["rescore"]
            self.assertTrue(offer["in_progress"])
            self.assertEqual(offer["job_id"], job_id)
            self.assertFalse(offer["available"])
            # A Job for ANOTHER run doesn't count.
            self.assertIsNone(_active_lift(pid, b))
        finally:
            _finish_job(job_id)
        self.assertFalse(_compare(pid, a, b)["row_evidence"]["rescore"]["in_progress"])

    def test_lift_check_refuses_a_duplicate_with_409(self):
        pid, a, _b = _seed(
            a={"metrics": {"exact_match": A_EM}, "details": {}},
            b={"metrics": {"exact_match": B_EM}, "details": {}},
        )
        job_id = _add_job(pid, a, JobStatus.QUEUED)
        try:
            async def _start():
                from app.services.post_training_eval_service import start_post_training_lift_job

                async with async_session_factory() as session:
                    return await start_post_training_lift_job(session, project_id=pid, experiment_id=a)

            self.assertEqual(asyncio.run(_start()),
                             {"started": False, "skipped_reason": "already_running", "job_id": job_id})
            resp = _client_cm.post(f"/api/projects/{pid}/evaluation/summary/lift-check", json={"experiment_id": a})
            self.assertEqual(resp.status_code, 409, resp.text)
        finally:
            _finish_job(job_id)

    def test_mismatched_judge_gives_no_verdict(self):
        same_key = {"dataset": {"sha256": "b" * 64, "version": 5}}
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM, "judge": _judge("judge-x")}, "day": 20,
               "details": {**same_key, "row_scores": _rows_78(True)}},
            b={"metrics": {"exact_match": B_EM, "judge": _judge("judge-y")}, "day": 21,
               "details": {**same_key, "row_scores": _rows_78(False)}},
        )
        ev = _compare(pid, a, b)["row_evidence"]
        self.assertEqual(ev["verdict"], "not_comparable")
        self.assertEqual(ev["message"], "different judge: judge-x vs judge-y")
        self.assertEqual(ev["comparability"]["differences"][0]["kind"], "judge")

    def test_missing_row_scores_offers_rescore(self):
        same_key = {"dataset": {"sha256": "b" * 64, "version": 5}}
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM}, "day": 20, "details": dict(same_key)},
            b={"metrics": {"exact_match": B_EM}, "day": 21, "details": {**same_key, "row_scores": _rows_78(False)}},
        )
        with patch("app.services.post_training_eval_service.auto_lift_skip_reason", return_value=None):
            body = _compare(pid, a, b)
        ev = body["row_evidence"]
        self.assertEqual(ev["verdict"], "too_few_rows")
        self.assertEqual(ev["reason"], "no_row_scores")
        self.assertEqual(ev["message"], f"A: {NO_ROW_SCORES_MESSAGE}")
        self.assertFalse(body["a"]["provenance"]["has_row_scores"])
        self.assertEqual(ev["rescore"]["side"], "a")
        self.assertTrue(ev["rescore"]["available"])

    def test_endpoint_is_additive(self):
        same_key = {"dataset": {"sha256": "b" * 64, "version": 5}}
        pid, a, b = _seed(
            a={"metrics": {"exact_match": A_EM}, "details": {**same_key, "row_scores": _rows_78(True)}},
            b={"metrics": {"exact_match": B_EM}, "details": {**same_key, "row_scores": _rows_78(False)}},
        )
        resp = _client_cm.get(f"/api/projects/{pid}/evaluation/compare?a={a}&b={b}")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        for key in ("metric_deltas", "cluster_diff", "config_diff", "winner", "regressed", "row_evidence"):
            self.assertIn(key, body)
        self.assertEqual(body["winner"], "a")  # unchanged pass-rate winner
        prov = body["b"]["provenance"]
        self.assertEqual(prov["eval_result_id"], body["b"]["eval_result_id"])
        self.assertEqual(prov["answer_key"]["version"], 5)
        self.assertEqual(body["row_evidence"]["verdict"], "within_noise")


class AnswerKeyStampTests(unittest.TestCase):
    def test_stamp_matches_the_version_whose_manifest_has_the_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            path.write_text('{"prompt": "q1", "reference": "r1"}\n', encoding="utf-8")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()

            async def _go():
                async with async_session_factory() as session:
                    proj = Project(name="Answer key stamp", status=ProjectStatus.DRAFT)
                    session.add(proj)
                    await session.flush()
                    ds = Dataset(project_id=proj.id, name="Test Set", dataset_type=DatasetType.TEST)
                    session.add(ds)
                    await session.flush()
                    # v1 = this file, v2 = another file: the stamp names v1, not the newest.
                    session.add(DatasetVersion(dataset_id=ds.id, version=1, file_path=str(path), manifest={"sha256": sha}))
                    session.add(DatasetVersion(dataset_id=ds.id, version=2, file_path=str(path), manifest={"sha256": "f" * 64}))
                    await session.commit()
                    matched = await answer_key_stamp(session, ds, path)
                    missing = await answer_key_stamp(session, ds, Path(tmp) / "absent.jsonl")
                    return matched, missing

            matched, missing = asyncio.run(_go())
        self.assertEqual(matched, {"sha256": sha, "version": 1})
        self.assertEqual(missing, {"sha256": None, "version": None})

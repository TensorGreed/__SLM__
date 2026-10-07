"""Did retrieval over the documents beat fine-tuning? (passage_rag_verdict_service)

Pins:
  * the pure verdict: a win needs the same judge, a retrieval gain beyond
    row noise, and the passages score above the fine-tuned run's;
  * the Coach nudge fires only on a win and carries the reroute action;
  * the playground corpus: explicit setting > passages-win > auto;
  * the reroute clone stamps ``auto_rag_corpus="documents"`` on the sibling
    when the source resolves to passages.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest import mock
from uuid import uuid4

os.environ.setdefault("DEBUG", "false")

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import passage_rag_verdict_service as svc
from app.services.coach_service import _passages_beat_finetune_nudge

client: TestClient


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


PASSAGES = {
    "judge": "ollama:gemma4:12b",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "with_passages": 0.618,
    "with_passages_counts": {"correct": 9, "partial": 3, "wrong": 5},
    "without_retrieval": 0.029,
    "n_rows": 17,
    "retrieval_evidence": {"verdict": "better", "n": 17, "better": 12, "worse": 0, "same": 5},
    "cached_at": "2026-10-06T20:00:00+00:00",
}
FINETUNE = {"experiment_id": 35, "judged": True, "judge": "ollama:gemma4:12b", "score": 0.053,
            "counts": {"correct": 1, "partial": 0, "wrong": 18}, "n_rows": 19}


class VerdictTests(unittest.TestCase):
    def test_passages_win_needs_same_judge_noise_free_gain_and_a_higher_score(self):
        win = svc.compare_passages_to_finetune(PASSAGES, FINETUNE)
        self.assertTrue(win["passages_win"])
        self.assertTrue(win["comparable"])
        self.assertEqual(win["reason"], "passages_ahead")

        other_judge = svc.compare_passages_to_finetune(PASSAGES, {**FINETUNE, "judge": "anthropic:x"})
        self.assertFalse(other_judge["passages_win"])
        self.assertEqual(other_judge["reason"], "different_judges")

        noisy = svc.compare_passages_to_finetune(
            {**PASSAGES, "retrieval_evidence": {**PASSAGES["retrieval_evidence"], "verdict": "within_noise"}}, FINETUNE
        )
        self.assertFalse(noisy["passages_win"])
        self.assertEqual(noisy["reason"], "retrieval_gain_within_noise")

        finetune_ahead = svc.compare_passages_to_finetune(PASSAGES, {**FINETUNE, "score": 0.7})
        self.assertFalse(finetune_ahead["passages_win"])
        self.assertTrue(finetune_ahead["comparable"])

        no_run = svc.compare_passages_to_finetune(PASSAGES, {"experiment_id": 1, "judged": False})
        self.assertEqual(no_run["reason"], "no_judged_finetuned_run")
        self.assertIsNone(svc.compare_passages_to_finetune(None, FINETUNE))

    def test_summary_reads_the_cached_comparison(self):
        rows = [
            {"question": "q", "reference": "r",
             "without_rag": {"generated": "x", "f1": 0.1, "judge": {"score": 0.0, "verdict": "wrong", "reason": ""}},
             "with_rag": {"generated": "y", "f1": 0.3, "judge": {"score": 1.0, "verdict": "correct", "reason": ""}, "retrieved_row_count": 3}}
            for _ in range(6)
        ]
        payload = {"base_model": "m", "cached_at": "t", "rows": rows,
                   "summary": {"n_val_rows": 6, "judge": {"judge": "j", "with_rag": {"score": 1.0, "counts": {"correct": 6}},
                                                           "without_rag": {"score": 0.0}}}}
        summary = svc.summarize_documents_comparison(payload)
        self.assertEqual(summary["with_passages"], 1.0)
        self.assertEqual(summary["retrieval_evidence"]["verdict"], "better")
        self.assertIsNone(svc.summarize_documents_comparison({"summary": {"off_mean_f1": 0.1}, "rows": rows}),
                          "an unjudged comparison gives no verdict")


class NudgeTests(unittest.TestCase):
    def test_nudge_fires_on_a_win_with_the_reroute_action(self):
        nudge = _passages_beat_finetune_nudge(20, svc.compare_passages_to_finetune(PASSAGES, FINETUNE))
        self.assertEqual(nudge["id"], "eval:passages-beat-finetune")
        self.assertEqual(nudge["severity"], "warning")
        self.assertEqual(nudge["action"]["kind"], "reroute_to_rag")
        self.assertEqual(nudge["action"]["params"], {"corpus": "documents"})
        self.assertIn("0.62 vs 0.05", nudge["title"])
        self.assertIn("9 correct, 3 partial, 5 wrong of 17 validation rows", nudge["body"])
        self.assertIn("run #35 scored 0.05 (1 correct, 0 partial, 18 wrong of 19 test examples)", nudge["body"])
        self.assertIn("different rows", nudge["body"])
        self.assertEqual(nudge["context"]["recommended_corpus"], "documents")

    def test_nudge_is_quiet_otherwise(self):
        self.assertIsNone(_passages_beat_finetune_nudge(20, None))
        self.assertIsNone(_passages_beat_finetune_nudge(20, svc.compare_passages_to_finetune(PASSAGES, {**FINETUNE, "score": 0.9})))


class PassagesSummaryTests(unittest.TestCase):
    """A RAG sibling's Eval summary comes from its judged passages comparison."""

    def _write(self, pid: int, payload: dict) -> None:
        path = settings.DATA_DIR / "projects" / str(pid) / "auto_rag" / "comparison_base_documents.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_summary_from_the_judged_comparison(self):
        from app.services.eval_summary_service import build_eval_summary, passages_summary

        pid = PlaygroundCorpusAndCloneTests._project(self)  # type: ignore[arg-type]
        self.assertIsNone(passages_summary(pid))
        rows = [
            {"question": f"Q{i}?", "reference": "A long reference answer with enough words to be judged.",
             "without_rag": {"generated": "x", "f1": 0.1, "judge": {"score": 0.0, "verdict": "wrong", "reason": "r"}},
             "with_rag": {"generated": f"y{i}", "f1": 0.4, "judge": {"score": 1.0 if i < 5 else 0.0, "verdict": "correct" if i < 5 else "wrong", "reason": "missed"},
                          "retrieved_row_count": 3}}
            for i in range(6)
        ]
        self._write(pid, {
            "model": "base", "corpus": "documents", "split": "test", "base_model": "Qwen", "cached_at": "t",
            "summary": {"n_val_rows": 6, "judge": {"judge": "ollama:gemma4:12b",
                                                   "without_rag": {"score": 0.0, "counts": {"correct": 0, "partial": 0, "wrong": 6}},
                                                   "with_rag": {"score": 0.8333, "judged": 6, "counts": {"correct": 5, "partial": 0, "wrong": 1}}}},
            "rows": rows,
        })
        summary = passages_summary(pid)
        self.assertEqual(summary["kind"], "rag_passages")
        self.assertEqual(summary["verdict"], "better")
        self.assertEqual(summary["headline"]["metric_id"], "judge_correct")
        self.assertEqual((summary["headline"]["baseline_value"], summary["headline"]["trained_value"]), (0.0, 0.8333))
        self.assertEqual(summary["evidence"]["verdict"], "better")
        self.assertEqual((summary["evidence"]["better"], summary["evidence"]["worse"]), (5, 0))
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["failures"][0]["row_judge_verdict"], "wrong")
        self.assertEqual(summary["judge"]["correct"], 5)
        self.assertEqual(summary["split"], "test")
        # build_eval_summary uses it when the project has no trained run.
        from app.database import async_session_factory

        async def _go():
            async with async_session_factory() as db:
                return await build_eval_summary(db, pid)

        self.assertEqual(asyncio.run(_go())["kind"], "rag_passages")
        # Projects are numbered per DB, DATA_DIR is shared: leave no cache a
        # later test file's same-numbered project could pick up.
        (settings.DATA_DIR / "projects" / str(pid) / "auto_rag" / "comparison_base_documents.json").unlink()


class SiblingPassagesCheckTests(unittest.TestCase):
    def test_passages_sibling_gets_its_check_on_the_test_split(self):
        from app.services import auto_rag_comparison_job_service as jobs
        from app.services.rag_project_service import start_sibling_passages_check

        class P:
            id = 99
            runtime_config = {"rag_first": True, "auto_rag_corpus": "documents"}
            selected_recipe = {"recipe_id": "qa-sft"}

        seen: dict = {}

        async def _start(db, project_id, *, recipe_id, base_only, corpus, split, sweep_retrieval):
            seen.update(project_id=project_id, recipe_id=recipe_id, base_only=base_only, corpus=corpus, split=split, sweep_retrieval=sweep_retrieval)
            return type("J", (), {"id": 7})()

        with mock.patch.object(jobs, "start_auto_rag_comparison_job", _start):
            out = asyncio.run(start_sibling_passages_check(None, P()))
        self.assertEqual(out, {"started": True, "job_id": 7})
        self.assertEqual(seen, {"project_id": 99, "recipe_id": "qa-sft", "base_only": True, "corpus": "documents", "split": "test", "sweep_retrieval": True})

        class Plain:
            id = 100
            runtime_config = {"rag_first": True}
            selected_recipe = {"recipe_id": "qa-sft"}

        self.assertEqual(asyncio.run(start_sibling_passages_check(None, Plain()))["skipped_reason"], "not_a_passages_project")


class RagFirstPlaygroundTests(unittest.TestCase):
    def test_rag_first_project_chats_with_its_base_model_in_process(self):
        from app.api.training import PlaygroundChatRequest, _resolve_playground_run
        from app.database import async_session_factory

        pid = PlaygroundCorpusAndCloneTests._project(self)  # type: ignore[arg-type]
        client.put(f"/api/projects/{pid}", json={"base_model_name": "Qwen/Qwen2.5-1.5B-Instruct"})
        req = PlaygroundChatRequest(provider="experiment", messages=[{"role": "user", "content": "hi"}])

        async def _go():
            async with async_session_factory() as db:
                return await _resolve_playground_run(db, pid, req, rag_first_active=True)

        model_ref, label, base = asyncio.run(_go())
        self.assertEqual(model_ref, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(base, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertIn("base model + retrieval", label)


class PretrainingGateTests(unittest.TestCase):
    def test_gate_policy(self):
        ready = svc.passages_gate(PASSAGES)
        self.assertEqual(ready["status"], "retrieval_ready")
        self.assertIn("9 of 17 right", ready["reason"])
        weak = svc.passages_gate({**PASSAGES, "with_passages": 0.3, "with_passages_counts": {"correct": 3, "partial": 4, "wrong": 10}})
        self.assertEqual(weak["status"], "retrieval_weak")
        noisy = svc.passages_gate({**PASSAGES, "retrieval_evidence": {"verdict": "within_noise"}})
        self.assertEqual(noisy["status"], "retrieval_weak")
        self.assertIn("within noise", noisy["reason"])
        self.assertEqual(svc.passages_gate(None)["status"], "not_run")

    def test_read_gate_distinguishes_unjudged(self):
        pid = PlaygroundCorpusAndCloneTests._project(self)  # type: ignore[arg-type]
        self.assertEqual(svc.read_passages_gate(pid)["status"], "not_run")
        path = settings.DATA_DIR / "projects" / str(pid) / "auto_rag" / "comparison_base_documents.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": {"off_mean_f1": 0.1, "on_mean_f1": 0.4}, "rows": []}), encoding="utf-8")
        gate = svc.read_passages_gate(pid)
        self.assertEqual(gate["status"], "not_judged")
        self.assertEqual(gate["on_mean_f1"], 0.4)
        path.unlink()

    def test_coach_nudge_fires_before_any_run_only(self):
        from app.services.coach_service import _retrieval_ready_nudge, _retrieval_ready_nudge_from_gate
        from app.database import async_session_factory
        from app.models.project import Project

        gate = svc.passages_gate({**PASSAGES, "split": "test", "n_rows": 19,
                                  "with_passages_counts": {"correct": 12, "partial": 4, "wrong": 3}, "with_passages": 0.74})
        nudge = _retrieval_ready_nudge_from_gate(5, gate)
        self.assertEqual(nudge["id"], "training:retrieval-ready")
        self.assertEqual(nudge["action"]["kind"], "reroute_to_rag")
        self.assertIn("Retrieval already answers 12 of 19", nudge["title"])
        self.assertIn("12 of 19 test examples fully right", nudge["body"])

        pid = PlaygroundCorpusAndCloneTests._project(self)  # type: ignore[arg-type]

        async def _nudge():
            async with async_session_factory() as db:
                return await _retrieval_ready_nudge(db, await db.get(Project, pid))

        self.assertIsNone(asyncio.run(_nudge()), "no comparison yet → quiet")
        with mock.patch.object(svc, "read_passages_gate", lambda project_id: gate):
            self.assertIsNotNone(asyncio.run(_nudge()))
            # Once a run exists the Eval-stage nudge takes over.
            from app.models.experiment import Experiment, ExperimentStatus, TrainingMode

            async def _train():
                async with async_session_factory() as db:
                    db.add(Experiment(project_id=pid, name="r", status=ExperimentStatus.COMPLETED,
                                      training_mode=TrainingMode.SFT, base_model="m", output_dir="/tmp/x", config={}))
                    await db.commit()

            asyncio.run(_train())
            self.assertIsNone(asyncio.run(_nudge()))


class LatestJudgedLiftTests(unittest.TestCase):
    """Which fine-tuned run the verdict compares against, read from real rows."""

    def _seed(self, pid: int, *, runs: list[dict]) -> list[int]:
        from app.database import async_session_factory
        from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode

        async def _go():
            ids = []
            async with async_session_factory() as db:
                for run in runs:
                    exp = Experiment(
                        project_id=pid, name=run["name"], status=ExperimentStatus.COMPLETED,
                        training_mode=TrainingMode.SFT, base_model="m", output_dir="/tmp/x",
                        config=run.get("config") or {}, seed_group_id=run.get("seed_group_id"),
                        seed_value=run.get("seed_value"),
                    )
                    db.add(exp)
                    await db.flush()
                    ids.append(exp.id)
                    if "judge" in run:
                        db.add(EvalResult(
                            experiment_id=exp.id, dataset_name="test", eval_type="exact_match",
                            metrics={"judge_correct": run["judge"], "evaluated_samples": 19,
                                     "judge": {"judge": "ollama:gemma4:12b", "judged": 19,
                                               "counts": {"correct": int(run["judge"] * 19), "partial": 0, "wrong": 0}}},
                            pass_rate=0.0, details={},
                        ))
                    elif run.get("unjudged_eval"):
                        db.add(EvalResult(experiment_id=exp.id, dataset_name="test", eval_type="exact_match",
                                          metrics={"f1": 0.3}, pass_rate=0.0, details={}))
                await db.commit()
            return ids

        return asyncio.run(_go())

    def _latest(self, pid: int):
        from app.database import async_session_factory

        async def _go():
            async with async_session_factory() as db:
                return await svc._latest_judged_lift(db, pid)

        return asyncio.run(_go())

    def test_falls_back_to_the_newest_judged_run_and_says_so(self):
        pid = PlaygroundCorpusAndCloneTests._project(self)  # type: ignore[arg-type]
        ids = self._seed(pid, runs=[
            {"name": "old judged", "judge": 0.1},
            {"name": "newest, lift-checked before the judge existed", "unjudged_eval": True},
        ])
        latest = self._latest(pid)
        self.assertEqual(latest["experiment_id"], ids[0])
        self.assertEqual(latest["latest_experiment_id"], ids[1])
        self.assertFalse(latest["finetune_is_latest"])
        self.assertAlmostEqual(latest["score"], 0.1)
        nudge = _passages_beat_finetune_nudge(pid, svc.compare_passages_to_finetune(PASSAGES, latest))
        self.assertIn(f"Your latest run #{ids[1]} has not been judged yet", nudge["body"])

    def test_seed_group_leader_is_the_mean_of_its_judged_children(self):
        pid = PlaygroundCorpusAndCloneTests._project(self)  # type: ignore[arg-type]
        group = uuid4().hex
        ids = self._seed(pid, runs=[
            {"name": "leader", "seed_group_id": group, "seed_value": None},
            {"name": "s1", "seed_group_id": group, "seed_value": 1, "judge": 0.2},
            {"name": "s2", "seed_group_id": group, "seed_value": 2, "judge": 0.4},
        ])
        latest = self._latest(pid)
        self.assertEqual(latest["experiment_id"], ids[0])
        self.assertTrue(latest["finetune_is_latest"])
        self.assertEqual(latest["n_seeds"], 2)
        self.assertAlmostEqual(latest["score"], 0.3)
        self.assertEqual(latest["counts"]["correct"], int(0.2 * 19) + int(0.4 * 19))

    def test_no_judged_run_at_all(self):
        pid = PlaygroundCorpusAndCloneTests._project(self)  # type: ignore[arg-type]
        self.assertIsNone(self._latest(pid))
        self._seed(pid, runs=[{"name": "unjudged", "unjudged_eval": True}])
        latest = self._latest(pid)
        self.assertFalse(latest["judged"])
        self.assertEqual(svc.compare_passages_to_finetune(PASSAGES, latest)["reason"], "no_judged_finetuned_run")


class PlaygroundCorpusAndCloneTests(unittest.TestCase):
    def _project(self, runtime_config=None) -> int:
        body = {"name": f"verdict-{uuid4().hex[:6]}", "description": "docs"}
        resp = client.post("/api/projects", json=body)
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])
        if runtime_config is not None:
            from app.database import async_session_factory
            from app.models.project import Project

            async def _set():
                async with async_session_factory() as db:
                    project = await db.get(Project, pid)
                    project.runtime_config = runtime_config
                    await db.commit()

            asyncio.run(_set())
        return pid

    def _resolve(self, pid: int):
        from app.database import async_session_factory
        from app.models.project import Project

        async def _go():
            async with async_session_factory() as db:
                return await svc.resolve_playground_corpus(db, await db.get(Project, pid))

        return asyncio.run(_go())

    def test_explicit_setting_then_verdict_then_auto(self):
        pid = self._project()
        self.assertEqual(self._resolve(pid), ("auto", "default"))

        async def _win(db, project_id):
            return svc.compare_passages_to_finetune(PASSAGES, FINETUNE)

        with mock.patch.object(svc, "passages_vs_finetune", _win):
            self.assertEqual(self._resolve(pid), ("documents", "passages_beat_finetune"))
        explicit = self._project({"auto_rag_corpus": "qa"})
        with mock.patch.object(svc, "passages_vs_finetune", _win):
            self.assertEqual(self._resolve(explicit), ("qa", "project_setting"))

    def test_playground_request_corpus_overrides_and_the_audit_block_says_why(self):
        from app.api.training import PlaygroundChatRequest, _apply_playground_auto_rag

        pid = self._project()
        seen: dict = {}

        async def _preamble(db, project_id, query, *, k=3, corpus="auto"):
            seen["corpus"] = corpus
            return {"preamble_text": "P", "retrieved": [], "corpus": "documents" if corpus == "documents" else "qa"}

        async def _win(db, project_id):
            return svc.compare_passages_to_finetune(PASSAGES, FINETUNE)

        from app.database import async_session_factory

        async def _chat(req):
            async with async_session_factory() as db:
                return await _apply_playground_auto_rag(db, pid, req, [{"role": "user", "content": "hi"}])

        with mock.patch("app.services.auto_rag_service.build_preamble_from_query", _preamble), \
                mock.patch.object(svc, "passages_vs_finetune", _win):
            _, block = asyncio.run(_chat(PlaygroundChatRequest(messages=[{"role": "user", "content": "hi"}], auto_rag=True)))
            self.assertEqual(seen["corpus"], "documents")
            self.assertEqual(block["corpus_reason"], "passages_beat_finetune")
            _, block = asyncio.run(_chat(PlaygroundChatRequest(messages=[{"role": "user", "content": "hi"}], auto_rag=True, auto_rag_corpus="qa")))
            self.assertEqual(seen["corpus"], "qa")
            self.assertEqual(block["corpus_reason"], "request")

    def test_reroute_clone_carries_passage_retrieval(self):
        pid = self._project()
        # A qa-sft recipe is what the clone requires.
        from app.database import async_session_factory
        from app.models.project import Project
        from app.services.rag_project_service import clone_project_for_rag

        async def _go(win: bool):
            async def _verdict(db, project_id):
                return svc.compare_passages_to_finetune(PASSAGES, FINETUNE) if win else None

            async with async_session_factory() as db:
                project = await db.get(Project, pid)
                project.selected_recipe = {"recipe_id": "qa-sft"}
                await db.commit()
                with mock.patch.object(svc, "passages_vs_finetune", _verdict):
                    new_project = await clone_project_for_rag(db, pid)
                await db.commit()
                return dict(new_project.runtime_config or {})

        cfg = asyncio.run(_go(True))
        self.assertTrue(cfg["rag_first"])
        self.assertEqual(cfg["auto_rag_corpus"], "documents")
        self.assertEqual(cfg["auto_rag_corpus_reason"], "passages_beat_finetune")
        plain = asyncio.run(_go(False))
        self.assertNotIn("auto_rag_corpus", plain)


if __name__ == "__main__":
    unittest.main()

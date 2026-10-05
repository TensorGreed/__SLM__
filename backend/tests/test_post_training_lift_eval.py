"""Automatic post-training lift eval (Wave 1b).

Pins:
  * eligibility — simulated runs, baselines, seed-group children, runs
    without weights and opted-out projects are skipped with a reason;
  * the runner evaluates base + fine-tuned on the same split and returns
    the paired lift; a fresh baseline result is reused, not recomputed;
  * the lift summary pairs a run with *its own* base model's baseline and
    can be pinned to a specific experiment.

Uses a private engine per test (like test_sft_lift_summary_service), so
the shared StaticPool engine caveat doesn't apply.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"

import app.models  # noqa: F401
from app.database import Base
from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode
from app.models.project import Project
from app.services import post_training_eval_service as svc
from app.services.sft_lift_summary_service import compute_sft_lift_summary

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _write_adapter_dir(root: Path) -> str:
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "x"}')
    (model_dir / "adapter_model.safetensors").write_bytes(b"\0" * 8)
    return str(root)


class LiftEvalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.root / 'lift.db'}", future=True
        )
        self.sf = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.eval_calls: list[dict] = []
        self.legacy_results = False
        self.seed_scores: dict[int, float] = {}

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._tmp.cleanup()

    async def _project(self, runtime_config: dict | None = None) -> int:
        async with self.sf() as db:
            p = Project(
                name=f"lift-{uuid4().hex[:8]}",
                description="",
                base_model_name=BASE,
                runtime_config=runtime_config or {},
            )
            db.add(p)
            await db.commit()
            return p.id

    async def _run(self, project_id: int, *, base_model: str = BASE, **overrides) -> int:
        out = _write_adapter_dir(self.root / uuid4().hex)
        fields = dict(
            project_id=project_id,
            name=f"run-{uuid4().hex[:6]}",
            status=ExperimentStatus.COMPLETED,
            training_mode=TrainingMode.SFT,
            base_model=base_model,
            output_dir=out,
            config={"_runtime": {"backend": "external"}},
        )
        fields.update(overrides)
        async with self.sf() as db:
            exp = Experiment(**fields)
            db.add(exp)
            await db.commit()
            return exp.id

    async def _fake_heldout(self, db, *, experiment_id, model_path, **kwargs):
        self.eval_calls.append({"experiment_id": experiment_id, "model_path": model_path, **kwargs})
        is_base = model_path is not None
        # Per-seed scores for seed-group tests (keyed by child id).
        score = self.seed_scores.get(experiment_id, 0.8) if not is_base else 0.1
        er = EvalResult(
            experiment_id=experiment_id,
            dataset_name=kwargs["dataset_name"],
            eval_type=kwargs["eval_type"],
            metrics={"exact_match": score},
            pass_rate=score,
            # Real held-out evals record per-row scores (for pairing the
            # base and fine-tuned results row by row).
            details={} if self.legacy_results else {
                "row_scores": {"keys": [f"k{i}" for i in range(10)],
                               # Base model: only row 0 right. Fine-tuned run: rows 0-7.
                               "correct": [int(i == 0) if is_base else int(i < 8) for i in range(10)]},
            },
        )
        db.add(er)
        await db.flush()
        return er

    async def _skip_reason(self, exp_id: int) -> str | None:
        async with self.sf() as db:
            exp = await db.get(Experiment, exp_id)
            project = await db.get(Project, exp.project_id)
            return svc.auto_lift_skip_reason(exp, project)

    # ── eligibility ────────────────────────────────────────────────

    async def test_real_completed_run_is_eligible(self):
        pid = await self._project()
        self.assertIsNone(await self._skip_reason(await self._run(pid)))

    async def test_skip_reasons(self):
        pid = await self._project()
        cases = {
            "simulated_run": {"config": {"_runtime": {"backend": "simulate"}}},
            "baseline_experiment": {"config": {"is_baseline": True}},
            "seed_group_child": {"seed_group_id": "g1", "seed_value": 7},
            "not_completed": {"status": ExperimentStatus.FAILED},
            "no_model_weights": {"output_dir": str(self.root / "empty")},
        }
        for expected, overrides in cases.items():
            with self.subTest(expected):
                exp_id = await self._run(pid, **overrides)
                self.assertEqual(await self._skip_reason(exp_id), expected)

    async def test_project_opt_out(self):
        pid = await self._project({"auto_lift_eval": False})
        self.assertEqual(await self._skip_reason(await self._run(pid)), "disabled_for_project")

    # ── runner ─────────────────────────────────────────────────────

    async def test_runner_evaluates_base_then_finetuned_and_reports_lift(self):
        pid = await self._project()
        exp_id = await self._run(pid)
        with mock.patch(
            "app.services.evaluation_service.run_heldout_evaluation", self._fake_heldout
        ):
            async with self.sf() as db:
                out = await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=exp_id)

        self.assertEqual([c["model_path"] for c in self.eval_calls], [BASE, None])
        self.assertEqual({c["dataset_name"] for c in self.eval_calls}, {"test"})
        self.assertFalse(out["baseline_reused"])
        self.assertEqual(out["lift_status"], "ok")
        self.assertEqual(out["headline"]["direction"], "improved")
        async with self.sf() as db:
            base = await db.get(Experiment, out["baseline_experiment_id"])
            self.assertTrue(base.config["is_baseline"])
            self.assertEqual(base.base_model, BASE)

    async def test_baseline_without_row_scores_is_re_evaluated_once(self):
        """A cached base-model result from before per-row scores were kept
        can't be paired row by row — it is recomputed, then reused."""
        pid = await self._project()
        first, second, third = await self._run(pid), await self._run(pid), await self._run(pid)
        with mock.patch(
            "app.services.evaluation_service.run_heldout_evaluation", self._fake_heldout
        ):
            self.legacy_results = True
            async with self.sf() as db:
                await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=first)
            self.legacy_results = False
            async with self.sf() as db:
                out = await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=second)
            self.assertFalse(out["baseline_reused"])
            async with self.sf() as db:
                out = await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=third)
            self.assertTrue(out["baseline_reused"])

    async def test_job_result_carries_row_evidence_for_the_bell(self):
        """The bell line is built from the Job result: it needs the row-level
        verdict so "better than base" isn't announced for a change that could
        be chance. Results without per-row scores carry no evidence."""
        pid = await self._project()
        exp_id = await self._run(pid)
        with mock.patch(
            "app.services.evaluation_service.run_heldout_evaluation", self._fake_heldout
        ):
            async with self.sf() as db:
                out = await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=exp_id)
        evidence = out["evidence"]
        # Fake rows: base gets only row 0 right, the run gets rows 0-7 right.
        self.assertEqual((evidence["n"], evidence["better"], evidence["worse"], evidence["same"]), (10, 7, 0, 3))
        self.assertEqual(evidence["verdict"], "better")
        self.assertEqual(set(evidence), {"verdict", "n", "better", "worse", "same"})

        legacy_exp = await self._run(pid)
        self.legacy_results = True
        with mock.patch(
            "app.services.evaluation_service.run_heldout_evaluation", self._fake_heldout
        ):
            async with self.sf() as db:
                legacy = await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=legacy_exp)
        self.assertEqual(legacy["lift_status"], "ok")
        self.assertIsNone(legacy["evidence"])

    async def _seed_group(self, pid: int, seeds: list[int]) -> tuple[int, list[int]]:
        """A finished seed group: a leader (never trained itself, borrows a
        child's output_dir) + one COMPLETED child per seed."""
        group = uuid4().hex
        leader = await self._run(pid, name="smol · 3 seeds", seed_group_id=group,
                                 config={"_runtime": {"backend": "external"}, "num_seeds": len(seeds)})
        children = [
            await self._run(pid, name=f"smol (seed={seed})", seed_group_id=group, seed_value=seed)
            for seed in seeds
        ]
        return leader, children

    async def test_seed_group_leader_is_checked_across_every_seed(self):
        """Before: the leader was evaluated as one run (its first child's
        weights) and reported under its own name. Now every seed is scored
        against one base-model result and the spread across seeds is
        reported."""
        pid = await self._project()
        leader, children = await self._seed_group(pid, [1, 2, 3])
        self.seed_scores = {children[0]: 0.7, children[1]: 0.8, children[2]: 0.75}
        with mock.patch(
            "app.services.evaluation_service.run_heldout_evaluation", self._fake_heldout
        ):
            async with self.sf() as db:
                out = await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=leader)

        # One base eval, then one eval per child — never the leader itself.
        evaluated = [c["experiment_id"] for c in self.eval_calls if c["model_path"] is None]
        self.assertEqual(evaluated, children)
        self.assertEqual(sum(1 for c in self.eval_calls if c["model_path"] == BASE), 1)
        self.assertEqual(out["experiment_id"], leader)
        self.assertEqual(out["n_seeds"], 3)
        self.assertEqual([s["seed_value"] for s in out["seeds"]], [1, 2, 3])
        self.assertEqual([s["headline"]["trained_value"] for s in out["seeds"]], [0.7, 0.8, 0.75])
        head = out["headline"]
        self.assertEqual((head["metric_id"], head["baseline_value"], head["trained_value"]), ("exact_match", 0.1, 0.75))
        self.assertEqual((head["direction"], head["n_seeds"]), ("improved", 3))
        ev = out["seed_evidence"]
        self.assertEqual((ev["kind"], ev["n"], ev["verdict"]), ("seeds", 3, "better"))
        self.assertTrue(ev["all_better"])
        self.assertEqual(ev["values"], [0.7, 0.8, 0.75])
        self.assertIsNone(out["evidence"])  # row evidence is per seed

    async def test_seed_group_that_disagrees_is_within_noise(self):
        pid = await self._project()
        leader, children = await self._seed_group(pid, [1, 2, 3])
        self.seed_scores = {children[0]: 0.35, children[1]: 0.08, children[2]: 0.12}
        with mock.patch(
            "app.services.evaluation_service.run_heldout_evaluation", self._fake_heldout
        ):
            async with self.sf() as db:
                out = await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=leader)
        self.assertEqual(out["headline"]["direction"], "improved")  # the mean says so…
        self.assertEqual(out["seed_evidence"]["verdict"], "within_noise")  # …the seeds don't agree
        self.assertFalse(out["seed_evidence"]["all_better"])

    async def test_seed_group_leader_is_eligible_without_its_own_weights(self):
        """The leader's output_dir is empty (it never trained) — the check
        used to skip it as "no_model_weights" and the group got no lift
        check at all."""
        pid = await self._project()
        async with self.sf() as db:
            leader = Experiment(project_id=pid, name="g", status=ExperimentStatus.COMPLETED,
                                training_mode=TrainingMode.SFT, base_model=BASE,
                                output_dir=str(self.root / "empty-leader"), seed_group_id=uuid4().hex, config={})
            db.add(leader)
            await db.commit()
            leader_id = leader.id
        self.assertIsNone(await self._skip_reason(leader_id))

    async def test_seed_group_job_title_names_the_seed_count(self):
        pid = await self._project()
        leader, _children = await self._seed_group(pid, [1, 2])
        started: dict = {}

        async def _fake_start_job(db, **kwargs):
            started.update(kwargs)
            return mock.Mock(id=77)

        with mock.patch("app.services.jobs_service.start_job", _fake_start_job):
            async with self.sf() as db:
                out = await svc.start_post_training_lift_job(db, project_id=pid, experiment_id=leader)
        self.assertEqual(out, {"started": True, "job_id": 77})
        self.assertIn(f"run #{leader} (2 seeds)", started["title"])

    async def test_second_run_reuses_fresh_baseline(self):
        pid = await self._project()
        first, second = await self._run(pid), await self._run(pid)
        with mock.patch(
            "app.services.evaluation_service.run_heldout_evaluation", self._fake_heldout
        ):
            async with self.sf() as db:
                await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=first)
            async with self.sf() as db:
                out = await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=second)
        self.assertTrue(out["baseline_reused"])
        self.assertEqual([c["model_path"] for c in self.eval_calls], [BASE, None, None])

    async def test_ineligible_run_raises_with_reason(self):
        pid = await self._project()
        exp_id = await self._run(pid, config={"_runtime": {"backend": "simulate"}})
        async with self.sf() as db:
            with self.assertRaisesRegex(ValueError, "simulated_run"):
                await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=exp_id)

    async def test_start_job_never_raises_and_reports_skip(self):
        pid = await self._project()
        exp_id = await self._run(pid, config={"is_baseline": True})
        async with self.sf() as db:
            out = await svc.start_post_training_lift_job(db, project_id=pid, experiment_id=exp_id)
        self.assertEqual(out, {"started": False, "skipped_reason": "baseline_experiment"})

    # ── lift pairing ───────────────────────────────────────────────

    async def test_lift_ignores_other_base_models_baseline(self):
        pid = await self._project()
        other_run = await self._run(pid, base_model="Qwen/Qwen2.5-1.5B-Instruct")
        with mock.patch(
            "app.services.evaluation_service.run_heldout_evaluation", self._fake_heldout
        ):
            async with self.sf() as db:
                await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=other_run)
        smol_run = await self._run(pid)
        async with self.sf() as db:
            db.add(EvalResult(
                experiment_id=smol_run, dataset_name="test", eval_type="exact_match",
                metrics={"exact_match": 0.5}, details={},
            ))
            await db.commit()
        async with self.sf() as db:
            summary = await compute_sft_lift_summary(db, pid, experiment_id=smol_run)
        self.assertEqual(summary["status"], "no_baseline")

    async def test_lift_can_be_pinned_to_an_older_run(self):
        pid = await self._project()
        first, second = await self._run(pid), await self._run(pid)
        with mock.patch(
            "app.services.evaluation_service.run_heldout_evaluation", self._fake_heldout
        ):
            for exp_id in (first, second):
                async with self.sf() as db:
                    await svc.run_post_training_lift_eval(db, project_id=pid, experiment_id=exp_id)
        async with self.sf() as db:
            pinned = await compute_sft_lift_summary(db, pid, experiment_id=first)
            latest = await compute_sft_lift_summary(db, pid)
        self.assertEqual(pinned["trained"]["experiment_id"], first)
        self.assertEqual(latest["trained"]["experiment_id"], second)
        async with self.sf() as db:
            baselines = (await db.execute(
                select(Experiment).where(Experiment.project_id == pid)
            )).scalars().all()
        self.assertEqual(sum(1 for e in baselines if (e.config or {}).get("is_baseline")), 1)


if __name__ == "__main__":
    unittest.main()

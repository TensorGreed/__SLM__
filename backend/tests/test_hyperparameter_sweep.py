"""Hyperparameter grid bake-off sweep (Track 1, Epic C).

Covers:
- ``expand_grid`` cross-product, label/use_lora forcing, dedupe, cap, validation.
- ``start_hyperparameter_sweep`` materializes one real Experiment per cell under
  a shared ``_sweep.sweep_id`` (dispatch stubbed so the test is deterministic /
  GPU-free).
- ``get_sweep_pareto`` aggregates cells into a quality-vs-cost Pareto for each
  supported ``cost_kind``:
    * ``lora_r`` — adapter footprint proxy (legacy behavior).
    * ``wall_clock_seconds`` — measured training duration (the honest default).
    * ``base_params_m`` — base-model parameter count for cross-model sweeps.
  Lower-cost cells with equal-or-better quality dominate; cells lacking a
  cost signal (e.g. wall-clock on a still-training cell) sit off the frontier.
- ``list_project_sweeps`` groups by sweep id.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "hp_sweep.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "hp_sweep_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode  # noqa: E402
from app.services.hyperparameter_sweep_service import (  # noqa: E402
    DEFAULT_NO_HISTORY_SECONDS_PER_CELL,
    MAX_SWEEP_CELLS,
    estimate_sweep_budget,
    expand_grid,
    get_sweep_pareto,
    list_project_sweeps,
    start_hyperparameter_sweep,
)

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


class ExpandGridTests(unittest.TestCase):
    def test_cross_product_and_labels(self):
        cells = expand_grid({}, lora_r_values=[8, 16], learning_rate_values=[2e-4, 3e-4])
        self.assertEqual(len(cells), 4)
        for cell in cells:
            self.assertTrue(cell["use_lora"])
            self.assertIn("_label", cell)
            self.assertIn(cell["lora_r"], (8, 16))
        labels = {c["_label"] for c in cells}
        self.assertIn("r8-lr0.0002", labels)
        self.assertIn("r16-lr0.0003", labels)

    def test_base_model_axis(self):
        cells = expand_grid(
            {}, lora_r_values=[8], learning_rate_values=[2e-4],
            base_model_values=["org/model-a", "org/model-b"],
        )
        self.assertEqual(len(cells), 2)
        self.assertEqual({c["base_model"] for c in cells}, {"org/model-a", "org/model-b"})

    def test_dedupe_and_cap(self):
        cells = expand_grid({}, lora_r_values=[8, 8, 16], learning_rate_values=[2e-4, 2e-4])
        self.assertEqual(len(cells), 2)  # ranks {8,16} x lr {2e-4}
        big = expand_grid(
            {}, lora_r_values=list(range(1, 30)), learning_rate_values=[1e-4],
        )
        self.assertEqual(len(big), MAX_SWEEP_CELLS)

    def test_validation(self):
        with self.assertRaises(ValueError):
            expand_grid({}, lora_r_values=[], learning_rate_values=[2e-4])
        with self.assertRaises(ValueError):
            expand_grid({}, lora_r_values=[8], learning_rate_values=[])


class SweepOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _create_project(self) -> int:
        resp = self.client.post("/api/projects", json={"name": f"hp-{uuid.uuid4().hex[:8]}"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_start_sweep_creates_grouped_experiments(self):
        project_id = self._create_project()

        async def _noop_start(db, pid, eid):  # stub dispatch — no GPU, no runtime
            return {"experiment_id": eid, "status": "running"}

        async def _run():
            async with async_session_factory() as db:
                with patch("app.services.training_service.start_training", _noop_start):
                    out = await start_hyperparameter_sweep(
                        db, project_id, base_model=BASE,
                        lora_r_values=[8, 16], learning_rate_values=[2e-4],
                    )
                await db.commit()
                return out

        result = asyncio.run(_run())
        self.assertEqual(result["requested_cells"], 2)
        self.assertEqual(result["dispatched_cells"], 2)
        sweep_id = result["sweep_id"]
        ids = {c["experiment_id"] for c in result["cells"]}
        self.assertEqual(len(ids), 2)

        # Every cell is a real Experiment grouped under the sweep id.
        async def _check():
            async with async_session_factory() as db:
                from sqlalchemy import select
                rows = (await db.execute(select(Experiment).where(Experiment.project_id == project_id))).scalars().all()
                return [
                    r for r in rows
                    if (r.config or {}).get("_sweep", {}).get("sweep_id") == sweep_id
                ]

        exps = asyncio.run(_check())
        self.assertEqual(len(exps), 2)
        for e in exps:
            self.assertTrue(e.config.get("use_lora"))
            self.assertIn(e.config.get("lora_r"), (8, 16))

    def test_get_sweep_pareto_annotates_frontier(self):
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]
        # 4 cells: r8 cheap, r16 expensive. Best loss at r16/lr2e-4.
        seed = [
            ("r8-lr2e-4", 8, 2e-4, 2.0),
            ("r8-lr3e-4", 8, 3e-4, 2.4),   # dominated by r8-lr2e-4 (same rank, worse loss)
            ("r16-lr2e-4", 16, 2e-4, 1.5),  # best quality, on frontier (higher rank justified)
            ("r16-lr3e-4", 16, 3e-4, 2.6),  # dominated by everything cheaper+better
        ]

        async def _seed():
            async with async_session_factory() as db:
                for label, rank, lr, loss in seed:
                    exp = Experiment(
                        project_id=project_id, name=f"sweep-{sweep_id}-{label}",
                        base_model=BASE, training_mode=TrainingMode.SFT,
                        status=ExperimentStatus.COMPLETED, final_train_loss=loss,
                        config={"use_lora": True, "lora_r": rank, "learning_rate": lr,
                                "_sweep": {"sweep_id": sweep_id, "label": label,
                                           "cell_index": len(label),
                                           "axis_values": {"lora_r": rank, "learning_rate": lr}}},
                    )
                    db.add(exp)
                await db.commit()

        asyncio.run(_seed())

        async def _get():
            async with async_session_factory() as db:
                # Pin cost_kind="lora_r" — this test asserts against the
                # rank-based frontier specifically; the wall-clock default
                # is exercised in test_wall_clock_cost_default below.
                return await get_sweep_pareto(db, project_id, sweep_id, cost_kind="lora_r")

        out = asyncio.run(_get())
        self.assertEqual(out["cell_count"], 4)
        self.assertEqual(out["completed_count"], 4)
        self.assertEqual(out["cost_kind"], "lora_r")
        by_label = {c["label"]: c for c in out["cells"]}
        # r8-lr2e-4 (cheapest at its quality) and r16-lr2e-4 (best quality) are the frontier.
        self.assertTrue(by_label["r8-lr2e-4"]["pareto_optimal"])
        self.assertTrue(by_label["r16-lr2e-4"]["pareto_optimal"])
        # Same-rank worse-loss cells are dominated.
        self.assertFalse(by_label["r8-lr3e-4"]["pareto_optimal"])
        self.assertFalse(by_label["r16-lr3e-4"]["pareto_optimal"])
        # Best overall = lowest loss.
        self.assertEqual(out["best_label"], "r16-lr2e-4")
        self.assertIn("r16-lr2e-4", out["pareto"]["optimal_labels"])
        # Each completed cell carries cost_score + cost_source matching the kind.
        self.assertEqual(by_label["r8-lr2e-4"]["cost_score"], 8.0)
        self.assertEqual(by_label["r16-lr2e-4"]["cost_score"], 16.0)
        self.assertEqual(by_label["r8-lr2e-4"]["cost_source"], "lora_r")

    def test_pending_cell_excluded_from_frontier(self):
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]

        async def _seed():
            async with async_session_factory() as db:
                done = Experiment(
                    project_id=project_id, name=f"sweep-{sweep_id}-done", base_model=BASE,
                    training_mode=TrainingMode.SFT, status=ExperimentStatus.COMPLETED,
                    final_train_loss=1.8,
                    config={"lora_r": 8, "_sweep": {"sweep_id": sweep_id, "label": "done", "cell_index": 0, "axis_values": {"lora_r": 8, "learning_rate": 2e-4}}},
                )
                pending = Experiment(
                    project_id=project_id, name=f"sweep-{sweep_id}-pending", base_model=BASE,
                    training_mode=TrainingMode.SFT, status=ExperimentStatus.RUNNING,
                    config={"lora_r": 16, "_sweep": {"sweep_id": sweep_id, "label": "pending", "cell_index": 1, "axis_values": {"lora_r": 16, "learning_rate": 2e-4}}},
                )
                db.add(done); db.add(pending)
                await db.commit()

        asyncio.run(_seed())
        out = asyncio.run(_run_get(project_id, sweep_id, cost_kind="lora_r"))
        self.assertEqual(out["cell_count"], 2)
        self.assertEqual(out["completed_count"], 1)
        by_label = {c["label"]: c for c in out["cells"]}
        self.assertIsNone(by_label["pending"]["quality_score"])
        self.assertFalse(by_label["pending"]["pareto_optimal"])
        self.assertTrue(by_label["done"]["pareto_optimal"])

    def test_get_sweep_unknown_raises(self):
        project_id = self._create_project()
        with self.assertRaises(ValueError):
            asyncio.run(_run_get(project_id, "nonexistent", cost_kind="lora_r"))

    def test_get_sweep_endpoint_returns_pareto(self):
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]

        async def _seed():
            async with async_session_factory() as db:
                for label, rank, loss in [("r8", 8, 2.0), ("r16", 16, 1.5)]:
                    db.add(Experiment(
                        project_id=project_id, name=f"sweep-{sweep_id}-{label}", base_model=BASE,
                        training_mode=TrainingMode.SFT, status=ExperimentStatus.COMPLETED,
                        final_train_loss=loss,
                        config={"lora_r": rank, "_sweep": {"sweep_id": sweep_id, "label": label,
                                "cell_index": rank, "axis_values": {"lora_r": rank, "learning_rate": 2e-4}}},
                    ))
                await db.commit()

        asyncio.run(_seed())
        # Endpoint defaults to wall_clock_seconds, but these seeded rows have
        # no started/completed timestamps — so we pass ?cost_kind=lora_r to
        # land on the rank frontier the assertions encode.
        resp = self.client.get(
            f"/api/projects/{project_id}/training/sweeps/{sweep_id}?cost_kind=lora_r"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["cell_count"], 2)
        self.assertEqual(body["completed_count"], 2)
        self.assertEqual(body["best_label"], "r16")
        self.assertEqual(body["cost_kind"], "lora_r")
        self.assertEqual(body["pareto"]["cost_kind"], "lora_r")
        self.assertIn("wall_clock_seconds", body["supported_cost_kinds"])
        self.assertIn("lora_r", body["supported_cost_kinds"])
        self.assertIn("base_params_m", body["supported_cost_kinds"])
        self.assertEqual(sorted(body["pareto"]["optimal_labels"]), ["r16", "r8"])

        # Default (no query param) lands on wall_clock_seconds and finds NO
        # cost signal on these rows — every cell drops off the frontier. Good
        # honest behaviour: rather than fabricate a wall-clock from the rank,
        # the response surfaces "cost pending" for every cell.
        default_resp = self.client.get(
            f"/api/projects/{project_id}/training/sweeps/{sweep_id}"
        ).json()
        self.assertEqual(default_resp["cost_kind"], "wall_clock_seconds")
        for cell in default_resp["cells"]:
            self.assertIsNone(cell["cost_score"])
            self.assertEqual(cell["cost_source"], "pending")
            self.assertFalse(cell["pareto_optimal"])

        # 404 for an unknown sweep id.
        missing = self.client.get(f"/api/projects/{project_id}/training/sweeps/nope")
        self.assertEqual(missing.status_code, 404, missing.text)

        # Bad cost_kind → 400, not 500.
        bad = self.client.get(
            f"/api/projects/{project_id}/training/sweeps/{sweep_id}?cost_kind=fictitious"
        )
        self.assertEqual(bad.status_code, 400, bad.text)

        # The sweep shows up in the project's sweep list.
        listed = self.client.get(f"/api/projects/{project_id}/training/sweeps").json()
        self.assertIn(sweep_id, {s["sweep_id"] for s in listed["sweeps"]})

    def test_wall_clock_cost_default(self):
        """Real measured wall-clock cost drives the default Pareto axis.

        Two cells with identical quality but different durations: the
        faster cell dominates the slower one.
        """
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]
        anchor = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)

        async def _seed():
            async with async_session_factory() as db:
                # Same quality, but the rank-8 cell took 60s and the rank-16
                # cell took 180s — wall-clock makes rank-8 dominate.
                rows = [
                    ("r8-fast", 8, 60),
                    ("r16-slow", 16, 180),
                ]
                for label, rank, seconds in rows:
                    db.add(Experiment(
                        project_id=project_id, name=f"sweep-{sweep_id}-{label}", base_model=BASE,
                        training_mode=TrainingMode.SFT, status=ExperimentStatus.COMPLETED,
                        final_train_loss=1.5,
                        started_at=anchor,
                        completed_at=anchor + timedelta(seconds=seconds),
                        config={"lora_r": rank, "_sweep": {"sweep_id": sweep_id, "label": label,
                                "cell_index": rank, "axis_values": {"lora_r": rank, "learning_rate": 2e-4}}},
                    ))
                await db.commit()

        asyncio.run(_seed())
        out = asyncio.run(_run_get(project_id, sweep_id))  # default cost_kind
        self.assertEqual(out["cost_kind"], "wall_clock_seconds")
        by_label = {c["label"]: c for c in out["cells"]}
        self.assertEqual(by_label["r8-fast"]["cost_score"], 60.0)
        self.assertEqual(by_label["r8-fast"]["cost_source"], "wall_clock_seconds")
        self.assertEqual(by_label["r16-slow"]["cost_score"], 180.0)
        # Equal quality, lower cost wins — r8-fast on frontier, r16-slow dominated.
        self.assertTrue(by_label["r8-fast"]["pareto_optimal"])
        self.assertFalse(by_label["r16-slow"]["pareto_optimal"])
        # Tie-break for best_label = highest quality, lowest cost.
        self.assertEqual(out["best_label"], "r8-fast")

    def test_wall_clock_pending_when_started_at_missing(self):
        """A cell with status=COMPLETED but no timestamps is honest about it:
        cost_score=None, cost_source='pending', drops off the frontier."""
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]

        async def _seed():
            async with async_session_factory() as db:
                db.add(Experiment(
                    project_id=project_id, name=f"sweep-{sweep_id}-r8", base_model=BASE,
                    training_mode=TrainingMode.SFT, status=ExperimentStatus.COMPLETED,
                    final_train_loss=1.5,
                    # started_at, completed_at deliberately omitted.
                    config={"lora_r": 8, "_sweep": {"sweep_id": sweep_id, "label": "r8",
                            "cell_index": 0, "axis_values": {"lora_r": 8, "learning_rate": 2e-4}}},
                ))
                await db.commit()

        asyncio.run(_seed())
        out = asyncio.run(_run_get(project_id, sweep_id))  # default wall_clock_seconds
        cell = out["cells"][0]
        self.assertIsNone(cell["cost_score"])
        self.assertEqual(cell["cost_source"], "pending")
        self.assertFalse(cell["pareto_optimal"])
        # completed_count counts cells with BOTH a quality and a cost signal
        # — pending wall-clock means this cell is off the 2D plot.
        self.assertEqual(out["completed_count"], 0)
        self.assertIsNone(out["best_label"])

    def test_base_params_cost(self):
        """Cross-model sweep: cost_kind='base_params_m' surfaces model-size trade-off."""
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]

        async def _seed():
            async with async_session_factory() as db:
                # SmolLM2-135M (lighter) vs Qwen2.5-1.5B (heavier).
                # Heavier wins on quality; the Pareto exposes the trade.
                rows = [
                    ("smollm-r8", "HuggingFaceTB/SmolLM2-135M-Instruct", 2.0),
                    ("qwen15-r8", "Qwen/Qwen2.5-1.5B-Instruct", 1.5),
                ]
                for idx, (label, model, loss) in enumerate(rows):
                    db.add(Experiment(
                        project_id=project_id, name=f"sweep-{sweep_id}-{label}", base_model=model,
                        training_mode=TrainingMode.SFT, status=ExperimentStatus.COMPLETED,
                        final_train_loss=loss,
                        config={"lora_r": 8, "_sweep": {"sweep_id": sweep_id, "label": label,
                                "cell_index": idx,
                                "axis_values": {"lora_r": 8, "learning_rate": 2e-4, "base_model": model}}},
                    ))
                await db.commit()

        asyncio.run(_seed())
        out = asyncio.run(_run_get(project_id, sweep_id, cost_kind="base_params_m"))
        self.assertEqual(out["cost_kind"], "base_params_m")
        by_label = {c["label"]: c for c in out["cells"]}
        self.assertEqual(by_label["smollm-r8"]["cost_score"], 135.0)
        self.assertEqual(by_label["qwen15-r8"]["cost_score"], 1500.0)
        # Both should be on the frontier: smollm-r8 is cheap+lower-quality,
        # qwen15-r8 is expensive+higher-quality. Neither dominates the other.
        self.assertTrue(by_label["smollm-r8"]["pareto_optimal"])
        self.assertTrue(by_label["qwen15-r8"]["pareto_optimal"])

    def test_base_params_unknown_model_drops_cost(self):
        """An off-catalog base model can't be priced — cost_source surfaces that."""
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]

        async def _seed():
            async with async_session_factory() as db:
                db.add(Experiment(
                    project_id=project_id, name=f"sweep-{sweep_id}-mystery",
                    base_model="some-org/unknown-model-v9",
                    training_mode=TrainingMode.SFT, status=ExperimentStatus.COMPLETED,
                    final_train_loss=1.5,
                    config={"lora_r": 8, "_sweep": {"sweep_id": sweep_id, "label": "mystery",
                            "cell_index": 0, "axis_values": {"lora_r": 8, "learning_rate": 2e-4}}},
                ))
                await db.commit()

        asyncio.run(_seed())
        out = asyncio.run(_run_get(project_id, sweep_id, cost_kind="base_params_m"))
        cell = out["cells"][0]
        self.assertIsNone(cell["cost_score"])
        self.assertEqual(cell["cost_source"], "unknown_base_model")

    def test_invalid_cost_kind_raises(self):
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]

        async def _seed():
            async with async_session_factory() as db:
                db.add(Experiment(
                    project_id=project_id, name=f"sweep-{sweep_id}-r8", base_model=BASE,
                    training_mode=TrainingMode.SFT, status=ExperimentStatus.COMPLETED,
                    final_train_loss=1.5,
                    config={"lora_r": 8, "_sweep": {"sweep_id": sweep_id, "label": "r8",
                            "cell_index": 0, "axis_values": {"lora_r": 8, "learning_rate": 2e-4}}},
                ))
                await db.commit()

        asyncio.run(_seed())
        with self.assertRaises(ValueError):
            asyncio.run(_run_get(project_id, sweep_id, cost_kind="bogus"))

    # -- Stop-when-met (auto-cancel watcher) ---------------------------

    def _seed_cell(
        self, *, project_id: int, sweep_id: str, label: str, status: ExperimentStatus,
        lora_r: int = 8, quality_target: float | None = None,
        train_loss: float | None = None, started_at=None, completed_at=None,
        eval_pass_rate: float | None = None,
    ):
        """Seed a single Experiment representing one sweep cell.

        Kept on the test class (not the module) because pytest collects
        free functions named ``test_*`` automatically — using a helper
        method avoids accidental collection of seeders as tests.
        """
        async def _add():
            async with async_session_factory() as db:
                cfg: dict = {
                    "lora_r": lora_r,
                    "_sweep": {
                        "sweep_id": sweep_id, "label": label, "cell_index": int(lora_r),
                        "axis_values": {"lora_r": lora_r, "learning_rate": 2e-4},
                    },
                }
                if quality_target is not None:
                    cfg["_sweep"]["quality_target"] = quality_target
                exp = Experiment(
                    project_id=project_id, name=f"sweep-{sweep_id}-{label}", base_model=BASE,
                    training_mode=TrainingMode.SFT, status=status,
                    final_train_loss=train_loss,
                    started_at=started_at,
                    completed_at=completed_at,
                    config=cfg,
                )
                db.add(exp)
                await db.flush()
                if eval_pass_rate is not None:
                    db.add(EvalResult(
                        experiment_id=exp.id,
                        dataset_name="gold",
                        eval_type="task",
                        pass_rate=eval_pass_rate,
                    ))
                await db.commit()
                return int(exp.id)
        return asyncio.run(_add())

    def test_target_hit_cancels_still_running_cells(self):
        """When one completed cell clears the target, every still-running
        cell in the sweep is cancelled on the next get-Pareto observation."""
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]

        # Two cells: r8 cleared the target with pass_rate=0.9; r16 is still
        # running. After get_sweep_pareto, r16 should be cancelled.
        anchor = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
        self._seed_cell(
            project_id=project_id, sweep_id=sweep_id, label="r8",
            status=ExperimentStatus.COMPLETED, lora_r=8, quality_target=0.85,
            eval_pass_rate=0.9, started_at=anchor,
            completed_at=anchor + timedelta(seconds=60),
        )
        r16_id = self._seed_cell(
            project_id=project_id, sweep_id=sweep_id, label="r16",
            status=ExperimentStatus.RUNNING, lora_r=16, quality_target=0.85,
            started_at=anchor,
        )

        # cancel_training reaches into job_service.cancel_task; that path
        # would fail in tests (no real task). Patch it to a no-op so we
        # can assert the row-level effect (status flips, cancelled_by_target
        # is annotated) without spinning a real runtime.
        cancelled_ids: list[int] = []

        async def _fake_cancel(db, pid, eid):
            cancelled_ids.append(int(eid))
            # Mirror the real cancel: flip status + completed_at.
            exp = await db.get(Experiment, int(eid))
            exp.status = ExperimentStatus.CANCELLED
            exp.completed_at = datetime.now(timezone.utc)
            await db.flush()
            return {"experiment_id": int(eid), "status": "cancelled"}

        with patch(
            "app.services.training_service.cancel_training",
            _fake_cancel,
        ):
            out = asyncio.run(_run_get(project_id, sweep_id, cost_kind="lora_r"))

        self.assertEqual(out["quality_target"], 0.85)
        self.assertTrue(out["target_hit"])
        self.assertEqual(out["target_hit_label"], "r8")
        self.assertEqual(out["cancelled_by_target"], ["r16"])
        self.assertIn(r16_id, cancelled_ids)
        by_label = {c["label"]: c for c in out["cells"]}
        self.assertTrue(by_label["r16"].get("cancelled_by_target"))
        self.assertEqual(by_label["r16"]["status"], "cancelled")

    def test_target_not_hit_leaves_running_cells_alone(self):
        """No completed cell cleared the target → no cancellation."""
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]
        anchor = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
        self._seed_cell(
            project_id=project_id, sweep_id=sweep_id, label="r8",
            status=ExperimentStatus.COMPLETED, lora_r=8, quality_target=0.95,
            eval_pass_rate=0.7,  # below target
            started_at=anchor, completed_at=anchor + timedelta(seconds=60),
        )
        self._seed_cell(
            project_id=project_id, sweep_id=sweep_id, label="r16",
            status=ExperimentStatus.RUNNING, lora_r=16, quality_target=0.95,
            started_at=anchor,
        )

        out = asyncio.run(_run_get(project_id, sweep_id, cost_kind="lora_r"))
        self.assertEqual(out["quality_target"], 0.95)
        self.assertFalse(out["target_hit"])
        self.assertIsNone(out["target_hit_label"])
        self.assertEqual(out["cancelled_by_target"], [])
        by_label = {c["label"]: c for c in out["cells"]}
        self.assertNotEqual(by_label["r16"]["status"], "cancelled")

    def test_no_target_set_means_no_cancellation_machinery(self):
        """A sweep without a quality_target runs every cell — same as legacy."""
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]
        anchor = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
        self._seed_cell(
            project_id=project_id, sweep_id=sweep_id, label="r8",
            status=ExperimentStatus.COMPLETED, lora_r=8,
            eval_pass_rate=0.99, started_at=anchor,
            completed_at=anchor + timedelta(seconds=60),
        )
        self._seed_cell(
            project_id=project_id, sweep_id=sweep_id, label="r16",
            status=ExperimentStatus.RUNNING, lora_r=16, started_at=anchor,
        )
        out = asyncio.run(_run_get(project_id, sweep_id, cost_kind="lora_r"))
        self.assertIsNone(out["quality_target"])
        self.assertFalse(out["target_hit"])
        self.assertEqual(out["cancelled_by_target"], [])

    # -- Pre-flight budget estimator -----------------------------------

    def test_estimate_budget_no_history(self):
        """No prior cells → conservative default + basis='no_history'."""
        project_id = self._create_project()

        async def _run():
            async with async_session_factory() as db:
                return await estimate_sweep_budget(
                    db, project_id, base_model=BASE,
                    lora_r_values=[8, 16], learning_rate_values=[2e-4, 3e-4],
                )

        out = asyncio.run(_run())
        self.assertEqual(out["cell_count"], 4)
        self.assertEqual(out["basis"], "no_history")
        self.assertEqual(out["sample_size"], 0)
        self.assertEqual(out["seconds_per_cell"], float(DEFAULT_NO_HISTORY_SECONDS_PER_CELL))
        self.assertEqual(out["estimated_seconds"], 4 * float(DEFAULT_NO_HISTORY_SECONDS_PER_CELL))

    def test_estimate_budget_uses_prior_cells_median(self):
        """Prior cells on the same base model produce a tighter estimate."""
        project_id = self._create_project()
        sweep_id = uuid.uuid4().hex[:12]
        anchor = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
        # Three prior cells: 30s, 60s, 90s → median 60s.
        for i, (label, secs) in enumerate([("a", 30), ("b", 60), ("c", 90)]):
            self._seed_cell(
                project_id=project_id, sweep_id=sweep_id + label,  # different sweep ids
                label=label, status=ExperimentStatus.COMPLETED, lora_r=8 + i,
                started_at=anchor, completed_at=anchor + timedelta(seconds=secs),
            )

        async def _run():
            async with async_session_factory() as db:
                return await estimate_sweep_budget(
                    db, project_id, base_model=BASE,
                    lora_r_values=[8, 16], learning_rate_values=[2e-4],
                )

        out = asyncio.run(_run())
        self.assertEqual(out["cell_count"], 2)
        self.assertEqual(out["basis"], "same_base_model")
        self.assertEqual(out["sample_size"], 3)
        self.assertEqual(out["seconds_per_cell"], 60.0)
        self.assertEqual(out["estimated_seconds"], 120.0)

    def test_estimate_budget_endpoint(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/sweeps/preflight-budget",
            json={"base_model": BASE, "lora_r_values": [8, 16], "learning_rate_values": [2e-4]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["cell_count"], 2)
        self.assertEqual(body["basis"], "no_history")

        # Empty axes → 400.
        bad = self.client.post(
            f"/api/projects/{project_id}/training/sweeps/preflight-budget",
            json={"base_model": BASE, "lora_r_values": [], "learning_rate_values": [2e-4]},
        )
        self.assertEqual(bad.status_code, 400, bad.text)

    def test_start_sweep_persists_quality_target_on_every_cell(self):
        """quality_target round-trips into each cell's _sweep meta so the
        watcher can recover it on subsequent fetches (no sweep table yet)."""
        project_id = self._create_project()

        async def _noop_start(db, pid, eid):
            return {"experiment_id": eid, "status": "running"}

        async def _run():
            async with async_session_factory() as db:
                with patch("app.services.training_service.start_training", _noop_start):
                    out = await start_hyperparameter_sweep(
                        db, project_id, base_model=BASE,
                        lora_r_values=[8, 16], learning_rate_values=[2e-4],
                        quality_target=0.85,
                    )
                await db.commit()
                return out

        result = asyncio.run(_run())
        self.assertEqual(result["quality_target"], 0.85)

        async def _check():
            async with async_session_factory() as db:
                from sqlalchemy import select as _select
                rows = (await db.execute(
                    _select(Experiment).where(Experiment.project_id == project_id)
                )).scalars().all()
                return [(r.config or {}).get("_sweep", {}).get("quality_target") for r in rows]

        targets = asyncio.run(_check())
        self.assertEqual(targets, [0.85, 0.85])  # 2 cells, both annotated

    def test_quality_target_coerces_percent_input(self):
        """A user typing 85 (meaning 85%) is coerced to 0.85 — common typo."""
        project_id = self._create_project()

        async def _noop_start(db, pid, eid):
            return {"experiment_id": eid, "status": "running"}

        async def _run():
            async with async_session_factory() as db:
                with patch("app.services.training_service.start_training", _noop_start):
                    out = await start_hyperparameter_sweep(
                        db, project_id, base_model=BASE,
                        lora_r_values=[8], learning_rate_values=[2e-4],
                        quality_target=85,  # percent form
                    )
                await db.commit()
                return out

        self.assertEqual(asyncio.run(_run())["quality_target"], 0.85)


async def _run_get(project_id: int, sweep_id: str, *, cost_kind: str | None = None):
    async with async_session_factory() as db:
        if cost_kind is None:
            return await get_sweep_pareto(db, project_id, sweep_id)
        return await get_sweep_pareto(db, project_id, sweep_id, cost_kind=cost_kind)


if __name__ == "__main__":
    unittest.main()

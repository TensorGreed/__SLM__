"""Hyperparameter grid bake-off sweep (Track 1, Epic C).

Covers:
- ``expand_grid`` cross-product, label/use_lora forcing, dedupe, cap, validation.
- ``start_hyperparameter_sweep`` materializes one real Experiment per cell under
  a shared ``_sweep.sweep_id`` (dispatch stubbed so the test is deterministic /
  GPU-free).
- ``get_sweep_pareto`` aggregates cells into a quality-vs-cost (rank) Pareto:
  lower-loss / lower-rank cells are on the frontier, dominated cells are marked,
  and still-training cells stay off the frontier.
- ``list_project_sweeps`` groups by sweep id.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
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
    MAX_SWEEP_CELLS,
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
                return await get_sweep_pareto(db, project_id, sweep_id)

        out = asyncio.run(_get())
        self.assertEqual(out["cell_count"], 4)
        self.assertEqual(out["completed_count"], 4)
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
        out = asyncio.run(_run_get(project_id, sweep_id))
        self.assertEqual(out["cell_count"], 2)
        self.assertEqual(out["completed_count"], 1)
        by_label = {c["label"]: c for c in out["cells"]}
        self.assertIsNone(by_label["pending"]["quality_score"])
        self.assertFalse(by_label["pending"]["pareto_optimal"])
        self.assertTrue(by_label["done"]["pareto_optimal"])

    def test_get_sweep_unknown_raises(self):
        project_id = self._create_project()
        with self.assertRaises(ValueError):
            asyncio.run(_run_get(project_id, "nonexistent"))

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
        resp = self.client.get(f"/api/projects/{project_id}/training/sweeps/{sweep_id}")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["cell_count"], 2)
        self.assertEqual(body["completed_count"], 2)
        self.assertEqual(body["best_label"], "r16")
        self.assertEqual(sorted(body["pareto"]["optimal_labels"]), ["r16", "r8"])

        # 404 for an unknown sweep id.
        missing = self.client.get(f"/api/projects/{project_id}/training/sweeps/nope")
        self.assertEqual(missing.status_code, 404, missing.text)

        # The sweep shows up in the project's sweep list.
        listed = self.client.get(f"/api/projects/{project_id}/training/sweeps").json()
        self.assertIn(sweep_id, {s["sweep_id"] for s in listed["sweeps"]})


async def _run_get(project_id: int, sweep_id: str):
    async with async_session_factory() as db:
        return await get_sweep_pareto(db, project_id, sweep_id)


if __name__ == "__main__":
    unittest.main()

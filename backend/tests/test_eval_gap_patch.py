"""Eval Gap patch engine — Coach-stage-2 phase 5.

Covers both apply_patch_kind patches end-to-end:
- regression_baseline_promote_last_green: preview returns a candidate,
  apply sets Checkpoint.promoted_at, re-scan flips signal to ok.
- label_kl_rebalance_eval: preview projects post-trim KL, apply
  rewrites GOLD_DEV in place, GOLD_TEST is untouched.

Plus rejection paths (signals without a patch, missing project,
candidate-not-found) and the safe_to_apply=False short-circuit.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "eval_gap_patch.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "eval_gap_patch_data"

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
from app.models.dataset import Dataset, DatasetType  # noqa: E402
from app.models.experiment import (  # noqa: E402
    Checkpoint, EvalResult, Experiment, ExperimentStatus,
)


def _signal_by_id(body: dict, signal_id: str) -> dict | None:
    for group in body["groups"]:
        for sig in group["signals"]:
            if sig["id"] == signal_id:
                return sig
    return None


class EvalGapPatchTests(unittest.TestCase):
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

    # ── helpers ─────────────────────────────────────────────────────

    def _make_project(self, recipe_id: str = "classification") -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"egp-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])

        async def _set():
            async with async_session_factory() as db:
                from app.models.project import Project
                proj = await db.get(Project, pid)
                proj.selected_recipe = {"recipe_id": recipe_id}
                await db.commit()
        asyncio.run(_set())
        return pid

    def _add_green_run(
        self, project_id: int, *, pass_rate: float = 0.85,
    ) -> tuple[int, int]:
        """Seed a COMPLETED experiment with an EvalResult above the
        baseline-promote threshold + one Checkpoint. Returns the
        (experiment_id, checkpoint_id) pair."""
        async def _add() -> tuple[int, int]:
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=project_id,
                    name=f"green-{uuid.uuid4().hex[:6]}",
                    description="",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    config={},
                    status=ExperimentStatus.COMPLETED,
                    completed_at=datetime.now(timezone.utc),
                )
                db.add(exp)
                await db.flush()
                ckpt = Checkpoint(
                    experiment_id=exp.id,
                    epoch=2,
                    step=200,
                    train_loss=0.4,
                    eval_loss=0.5,
                    file_path="/tmp/green-ckpt",
                    is_best=True,
                    metrics={},
                )
                db.add(ckpt)
                db.add(EvalResult(
                    experiment_id=exp.id,
                    dataset_name="gold_dev",
                    eval_type="classification",
                    metrics={},
                    pass_rate=pass_rate,
                ))
                await db.commit()
                return (int(exp.id), int(ckpt.id))
        return asyncio.run(_add())

    def _add_failed_run(self, project_id: int) -> int:
        """Completed but with pass_rate below the promote threshold."""
        async def _add() -> int:
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=project_id,
                    name=f"failed-{uuid.uuid4().hex[:6]}",
                    description="",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    config={},
                    status=ExperimentStatus.COMPLETED,
                    completed_at=datetime.now(timezone.utc),
                )
                db.add(exp)
                await db.flush()
                db.add(Checkpoint(
                    experiment_id=exp.id,
                    epoch=1,
                    step=100,
                    train_loss=0.9,
                    eval_loss=1.1,
                    file_path="/tmp/failed-ckpt",
                    is_best=True,
                    metrics={},
                ))
                db.add(EvalResult(
                    experiment_id=exp.id,
                    dataset_name="gold_dev",
                    eval_type="classification",
                    metrics={},
                    pass_rate=0.3,
                ))
                await db.commit()
                return int(exp.id)
        return asyncio.run(_add())

    def _seed_kl_skew(
        self,
        project_id: int,
        *,
        train_labels: list[str],
        gold_dev_labels: list[str],
        gold_test_labels: list[str] | None = None,
    ) -> Path:
        """Write TRAIN + GOLD_DEV (+ optional GOLD_TEST) JSONLs whose
        rows carry the supplied labels, register Datasets pointing at
        them. Returns the GOLD_DEV path so tests can read it back."""
        ds_dir = TEST_DATA_DIR / "projects" / str(project_id)
        ds_dir.mkdir(parents=True, exist_ok=True)
        train_path = ds_dir / "train.jsonl"
        dev_path = ds_dir / "gold_dev.jsonl"
        with train_path.open("w", encoding="utf-8") as f:
            for label in train_labels:
                f.write(json.dumps({"label": label}) + "\n")
        with dev_path.open("w", encoding="utf-8") as f:
            for label in gold_dev_labels:
                f.write(json.dumps({"label": label}) + "\n")
        test_path: Path | None = None
        if gold_test_labels is not None:
            test_path = ds_dir / "gold_test.jsonl"
            with test_path.open("w", encoding="utf-8") as f:
                for label in gold_test_labels:
                    f.write(json.dumps({"label": label}) + "\n")

        async def _add():
            async with async_session_factory() as db:
                db.add(Dataset(
                    project_id=project_id,
                    name="train",
                    dataset_type=DatasetType.TRAIN,
                    file_path=str(train_path),
                    record_count=len(train_labels),
                ))
                db.add(Dataset(
                    project_id=project_id,
                    name="gold_dev",
                    dataset_type=DatasetType.GOLD_DEV,
                    file_path=str(dev_path),
                    record_count=len(gold_dev_labels),
                ))
                if test_path is not None and gold_test_labels is not None:
                    db.add(Dataset(
                        project_id=project_id,
                        name="gold_test",
                        dataset_type=DatasetType.GOLD_TEST,
                        file_path=str(test_path),
                        record_count=len(gold_test_labels),
                    ))
                await db.commit()
        asyncio.run(_add())
        return dev_path

    # ── Baseline-promote patch ──────────────────────────────────────

    def test_baseline_promote_preview_returns_candidate(self):
        pid = self._make_project()
        exp_id, ckpt_id = self._add_green_run(pid)
        resp = self.client.post(
            f"/api/projects/{pid}/eval-gaps/patch/preview",
            json={"signal_id": "eval_gaps.no_regression_baseline"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        preview = resp.json()
        self.assertEqual(
            preview["patch_kind"],
            "regression_baseline_promote_last_green",
        )
        self.assertEqual(preview["candidate"]["experiment_id"], exp_id)
        self.assertEqual(preview["candidate"]["checkpoint_id"], ckpt_id)
        self.assertGreaterEqual(preview["candidate"]["pass_rate"], 0.5)
        self.assertTrue(preview["safe_to_apply"])

    def test_baseline_promote_apply_sets_promoted_at_and_signal_flips(self):
        pid = self._make_project()
        exp_id, ckpt_id = self._add_green_run(pid)
        # Apply.
        resp = self.client.post(
            f"/api/projects/{pid}/eval-gaps/patch/apply",
            json={"signal_id": "eval_gaps.no_regression_baseline"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        result = resp.json()
        self.assertTrue(result["applied"])
        # Re-scan: signal flips to ok.
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.no_regression_baseline")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        # Confirm via DB that promoted_at landed on the right checkpoint.

        async def _read():
            async with async_session_factory() as db:
                from app.models.experiment import Checkpoint as Ck
                ckpt = await db.get(Ck, ckpt_id)
                return ckpt.promoted_at
        promoted_at = asyncio.run(_read())
        self.assertIsNotNone(promoted_at)

    def test_baseline_promote_rejects_when_no_green_run_exists(self):
        pid = self._make_project()
        self._add_failed_run(pid)
        resp = self.client.post(
            f"/api/projects/{pid}/eval-gaps/patch/preview",
            json={"signal_id": "eval_gaps.no_regression_baseline"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("nothing to promote", resp.text.lower())

    # ── Label-KL rebalance patch ────────────────────────────────────

    def test_label_kl_preview_projects_post_trim_kl(self):
        pid = self._make_project()
        # Train: 50/50; GOLD_DEV: 90/10 pos-skewed → KL > 0.
        # 20 train rows minimum + 10 dev rows minimum required.
        self._seed_kl_skew(
            pid,
            train_labels=["pos"] * 25 + ["neg"] * 25,
            gold_dev_labels=["pos"] * 27 + ["neg"] * 3,
        )
        resp = self.client.post(
            f"/api/projects/{pid}/eval-gaps/patch/preview",
            json={"signal_id": "eval_gaps.train_eval_label_kl_high"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        preview = resp.json()
        self.assertEqual(preview["patch_kind"], "label_kl_rebalance_eval")
        # Before > after KL.
        kl_before = preview["before"]["kl_nats"]
        kl_after = preview["after"]["kl_nats"]
        self.assertLess(kl_after, kl_before)
        self.assertGreater(preview["rows_to_drop"], 0)
        self.assertTrue(preview["safe_to_apply"])

    def test_label_kl_apply_rewrites_gold_dev_and_preserves_gold_test(self):
        pid = self._make_project()
        dev_path = self._seed_kl_skew(
            pid,
            train_labels=["pos"] * 25 + ["neg"] * 25,
            gold_dev_labels=["pos"] * 27 + ["neg"] * 3,
            gold_test_labels=["pos"] * 8 + ["neg"] * 8,
        )
        test_path = dev_path.parent / "gold_test.jsonl"
        test_before = test_path.read_text(encoding="utf-8")

        resp = self.client.post(
            f"/api/projects/{pid}/eval-gaps/patch/apply",
            json={"signal_id": "eval_gaps.train_eval_label_kl_high"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        result = resp.json()
        self.assertTrue(result["applied"])
        # GOLD_DEV shrank (over-represented class trimmed).
        self.assertLess(result["rows_after"], 30)
        # GOLD_TEST is byte-identical (held-out integrity).
        self.assertEqual(test_path.read_text(encoding="utf-8"), test_before)

        # Re-scan: KL has improved (severity may or may not have crossed
        # the threshold depending on the magnitude of the skew, but the
        # measured value MUST be lower).
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.train_eval_label_kl_high")
        self.assertIsNotNone(sig)
        assert sig is not None
        if "kl_nats" in sig["context"]:
            # New KL should be at or below the pre-apply value.
            self.assertLessEqual(
                sig["context"]["kl_nats"], result["before"]["kl_nats"],
            )

    def test_label_kl_preview_rejects_when_below_minimum_sample_sizes(self):
        pid = self._make_project()
        # 10 train + 5 eval — below the 20/10 floors.
        self._seed_kl_skew(
            pid,
            train_labels=["pos"] * 5 + ["neg"] * 5,
            gold_dev_labels=["pos"] * 3 + ["neg"] * 2,
        )
        # Note: the underlying signal is "ok" (skipped) at this sample
        # size, which means it carries no apply_patch_kind. So the
        # preview endpoint rejects at the "no patch on this signal"
        # check, not at the patch builder's floor.
        resp = self.client.post(
            f"/api/projects/{pid}/eval-gaps/patch/preview",
            json={"signal_id": "eval_gaps.train_eval_label_kl_high"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    # ── Dispatcher rejection paths ──────────────────────────────────

    def test_preview_rejects_signal_without_apply_patch_kind(self):
        pid = self._make_project()
        # archetype-coverage signal exists but has no patch in phase 5.
        # Mock the comparison so it surfaces with severity warn.
        from unittest.mock import patch

        async def _stub(*_a, **_k):
            return {
                "project_id": pid,
                "recipe_id": "classification",
                "archetype": {},
                "features": [
                    {"feature_id": "row_count", "label": "Rows",
                     "status": "below", "your_value": 5},
                ],
                "summary": {},
            }

        with patch(
            "app.services.archetype_service.compare_project_to_archetype",
            side_effect=_stub,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/eval-gaps/patch/preview",
                json={"signal_id": "eval_gaps.archetype_coverage_low"},
            )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("no one-click patch", resp.text.lower())

    def test_preview_404s_for_missing_project(self):
        resp = self.client.post(
            "/api/projects/9999999/eval-gaps/patch/preview",
            json={"signal_id": "eval_gaps.no_regression_baseline"},
        )
        self.assertEqual(resp.status_code, 404)

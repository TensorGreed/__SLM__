"""Quality-Lift phase 3 slice 2 — Active-learning snapshot read endpoint
+ training-stage Coach nudge.

Pins (slice 2: surfacing the snapshot from slice 1; the Data Studio
card lands in slice 3):

  Endpoint ``GET /api/projects/{id}/active-learning/latest``:
    * Returns the most-recent COMPLETED experiment with a snapshot;
      picks the snapshot-carrying experiment, not just the latest
      (the user may have trained twice with one skip).
    * Derives labeled_count by joining top_k row ids against
      ``label_rows.labeled_at``.
    * staleness_ratio = labeled_count / top_k_size (0.0 when empty
      so the 80% cutoff never zero-divides).
    * is_stale = staleness_ratio >= 0.80.
    * Empty / missing snapshot returns a well-formed payload with
      ``no_snapshot_reason`` for the Data Studio card to surface.
    * 404 when project doesn't exist.

  Coach nudge ``training:active-learning-ready``:
    * Fires when snapshot has top_k.length > 0 AND staleness < 0.80.
    * Silences when no snapshot exists.
    * Silences when snapshot is empty (skipped_reason set).
    * Silences when ≥ 80% of the snapshot's rows are labeled.
    * Body / context surface the unlabeled count + the uncertainty
      metric + the experiment id so the user can correlate.
    * Action carries the dominant label_job_id so the deep-link
      lands on the right job.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode  # noqa: E402
from app.models.label_job import LabelJob, LabelRow  # noqa: E402


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-al-s2-{uuid.uuid4().hex[:8]}"
)


def setUpModule() -> None:
    settings.AUTH_ENABLED = False
    settings.DEBUG = False
    settings.DATA_DIR = TEST_DATA_DIR.resolve()
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    global _CLIENT_CM, CLIENT
    _CLIENT_CM = TestClient(app)
    CLIENT = _CLIENT_CM.__enter__()


def tearDownModule() -> None:
    _CLIENT_CM.__exit__(None, None, None)


def _create_project() -> int:
    resp = CLIENT.post(
        "/api/projects",
        json={"name": f"al-s2-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


def _seed_label_rows(
    project_id: int,
    *,
    n: int,
    label_type: str = "classification",
) -> tuple[int, list[int]]:
    """Create a LabelJob + N unlabeled rows; returns (job_id, row_ids)."""
    async def _go() -> tuple[int, list[int]]:
        async with async_session_factory() as session:
            job = LabelJob(
                project_id=project_id,
                name=f"al-job-{uuid.uuid4().hex[:4]}",
                label_type=label_type,
                label_schema={"allowed_labels": ["A", "B"]},
            )
            session.add(job)
            await session.flush()
            job_id = int(job.id)
            row_ids: list[int] = []
            for i in range(n):
                row = LabelRow(
                    job_id=job_id,
                    raw_payload={"text": f"row-{i}"},
                )
                session.add(row)
                await session.flush()
                row_ids.append(int(row.id))
            await session.commit()
            return job_id, row_ids

    return asyncio.run(_go())


def _mark_labeled(row_ids: list[int]) -> None:
    async def _go() -> None:
        async with async_session_factory() as session:
            for rid in row_ids:
                row = (await session.execute(
                    select(LabelRow).where(LabelRow.id == rid)
                )).scalar_one()
                row.labeled_at = datetime.now(timezone.utc)
                row.label_payload = {"label": "A"}
            await session.commit()

    asyncio.run(_go())


def _seed_completed_experiment(
    project_id: int,
    *,
    snapshot: dict | None,
    output_dir: str = "",
    name: str | None = None,
    completed_at: datetime | None = None,
) -> int:
    """Create a COMPLETED experiment optionally carrying an
    active_learning snapshot on its config._runtime."""
    cfg: dict = {"task_type": "classification"}
    if snapshot is not None:
        cfg["_runtime"] = {"active_learning": snapshot}

    async def _go() -> int:
        async with async_session_factory() as session:
            exp = Experiment(
                project_id=project_id,
                name=name or f"al-exp-{uuid.uuid4().hex[:4]}",
                status=ExperimentStatus.COMPLETED,
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                training_mode=TrainingMode.SFT,
                config=cfg,
                output_dir=output_dir,
                completed_at=completed_at or datetime.now(timezone.utc),
            )
            session.add(exp)
            await session.commit()
            return int(exp.id)

    return asyncio.run(_go())


def _build_snapshot(
    *,
    model_experiment_id: int,
    top_k_rows: list[tuple[int, int, float]],
    skipped_reason: str | None = None,
    uncertainty_metric: str = "entropy",
) -> dict:
    """Build a snapshot dict in the slice 1 contract shape from a list
    of (label_row_id, label_job_id, uncertainty_score) tuples."""
    return {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "model_experiment_id": model_experiment_id,
        "task_type": "classification",
        "uncertainty_metric": uncertainty_metric,
        "pool_size_total": len(top_k_rows),
        "pool_size_scored": len(top_k_rows),
        "top_k": [
            {
                "label_row_id": rid,
                "label_job_id": jid,
                "uncertainty_score": score,
            }
            for rid, jid, score in top_k_rows
        ],
        "skipped_reason": skipped_reason,
        "checkpoint_path": "/fake/checkpoint",
        "label_space_size": 2,
    }


# ────────────────────────────────────────────────────────────────────────
# GET /api/projects/{id}/active-learning/latest
# ────────────────────────────────────────────────────────────────────────


class LatestEndpointTests(unittest.TestCase):

    def test_404_when_project_missing(self):
        resp = CLIENT.get("/api/projects/999999/active-learning/latest")
        self.assertEqual(resp.status_code, 404)

    def test_no_snapshot_when_no_completed_experiment(self):
        pid = _create_project()
        resp = CLIENT.get(f"/api/projects/{pid}/active-learning/latest")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Fresh project — well-formed payload, not a 404.
        self.assertIsNone(body["snapshot"])
        self.assertIsNone(body["experiment_id"])
        self.assertEqual(body["top_k_size"], 0)
        self.assertEqual(body["labeled_count"], 0)
        self.assertEqual(body["staleness_ratio"], 0.0)
        self.assertFalse(body["is_stale"])
        self.assertEqual(body["no_snapshot_reason"], "no_completed_experiment_with_snapshot")
        # The threshold is part of the contract so slice 3's UI can show
        # the cutoff without hardcoding it.
        self.assertEqual(body["staleness_threshold"], 0.80)

    def test_picks_latest_snapshot_carrying_experiment(self):
        # User trained twice — the older run had a real snapshot, the
        # newer run skipped scoring (empty pool). Endpoint must return
        # the snapshot-carrying experiment, not the latest by
        # completed_at, because the newer skip carries no signal.
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=3)
        older_snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[(row_ids[i], job_id, 1.0 - i * 0.1) for i in range(3)],
        )
        older_eid = _seed_completed_experiment(
            pid,
            snapshot=older_snapshot,
            completed_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            name="older-with-snapshot",
        )
        # Newer experiment — no snapshot at all.
        _seed_completed_experiment(
            pid,
            snapshot=None,
            completed_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
            name="newer-no-snapshot",
        )

        resp = CLIENT.get(f"/api/projects/{pid}/active-learning/latest")
        body = resp.json()
        self.assertEqual(body["experiment_id"], older_eid)
        self.assertEqual(body["experiment_name"], "older-with-snapshot")
        self.assertEqual(body["top_k_size"], 3)

    def test_derived_labeled_count_and_staleness(self):
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=5)
        snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[(row_ids[i], job_id, 1.0 - i * 0.1) for i in range(5)],
        )
        _seed_completed_experiment(pid, snapshot=snapshot)

        # Initially nothing labeled.
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        self.assertEqual(body["labeled_count"], 0)
        self.assertEqual(body["unlabeled_count"], 5)
        self.assertEqual(body["staleness_ratio"], 0.0)
        self.assertFalse(body["is_stale"])

        # Label 2 of 5 → 0.40, still fresh.
        _mark_labeled(row_ids[:2])
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        self.assertEqual(body["labeled_count"], 2)
        self.assertEqual(body["unlabeled_count"], 3)
        self.assertAlmostEqual(body["staleness_ratio"], 0.40, places=4)
        self.assertFalse(body["is_stale"])

        # Label 4 of 5 → 0.80 hits threshold.
        _mark_labeled(row_ids[2:4])
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        self.assertAlmostEqual(body["staleness_ratio"], 0.80, places=4)
        self.assertTrue(body["is_stale"])

    def test_empty_snapshot_carries_skipped_reason(self):
        # Slice 1 snapshot for a non-classification task carries
        # skipped_reason="unsupported_task_type" and top_k=[]. The
        # endpoint must surface that so the Data Studio card (slice 3)
        # can render the contextual empty state.
        pid = _create_project()
        snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[],
            skipped_reason="unsupported_task_type",
        )
        snapshot["task_type"] = "causal_lm"
        snapshot["pool_size_total"] = 0
        _seed_completed_experiment(pid, snapshot=snapshot)
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        self.assertEqual(body["top_k_size"], 0)
        self.assertEqual(body["no_snapshot_reason"], "unsupported_task_type")

    def test_snapshot_returned_with_slice1_fields_preserved(self):
        # Slice 3 enriches each top_k entry with text_preview + labeled,
        # but the slice 1 fields (label_row_id, label_job_id,
        # uncertainty_score) must round-trip unchanged so the Coach
        # nudge + Data Studio card both read the same underlying ranking.
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=2)
        snapshot = _build_snapshot(
            model_experiment_id=42,
            top_k_rows=[(row_ids[0], job_id, 0.9), (row_ids[1], job_id, 0.5)],
        )
        _seed_completed_experiment(pid, snapshot=snapshot)
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        # Slice 1 contract preserved per entry.
        for i, entry in enumerate(body["snapshot"]["top_k"]):
            self.assertEqual(entry["label_row_id"], snapshot["top_k"][i]["label_row_id"])
            self.assertEqual(entry["label_job_id"], snapshot["top_k"][i]["label_job_id"])
            self.assertEqual(
                entry["uncertainty_score"],
                snapshot["top_k"][i]["uncertainty_score"],
            )
        # Top-level fields untouched by enrichment.
        self.assertEqual(body["snapshot"]["uncertainty_metric"], "entropy")


# ────────────────────────────────────────────────────────────────────────
# Coach nudge — training:active-learning-ready
# ────────────────────────────────────────────────────────────────────────


def _get_training_suggestions(project_id: int) -> list[dict]:
    """Hit the training-stage Coach endpoint and return the
    suggestions array. Some unrelated training-stage suggestions may
    co-exist (curriculum, archetype, forecast); filter by id when
    asserting on the AL nudge."""
    resp = CLIENT.get(f"/api/projects/{project_id}/coach/training/suggestions")
    if resp.status_code == 404:
        # Different API surface on this build — fall back to direct
        # coach_service call to keep the test driver portable.
        from app.services.coach_service import get_coach_suggestions

        async def _go() -> list[dict]:
            async with async_session_factory() as session:
                return await get_coach_suggestions(session, project_id, "training")
        return asyncio.run(_go())
    return resp.json().get("suggestions") or []


def _get_al_nudge_directly(project_id: int) -> dict | None:
    """Call the helper without going through _training_stage_suggestions
    so this stays a focused unit test on the rule's fire/silence
    semantics rather than the full training-stage pipeline."""
    from app.services.coach_service import _active_learning_ready_nudge

    async def _go() -> dict | None:
        async with async_session_factory() as session:
            return await _active_learning_ready_nudge(session, project_id)

    return asyncio.run(_go())


class ActiveLearningCoachNudgeTests(unittest.TestCase):

    def test_silent_when_no_completed_experiment(self):
        pid = _create_project()
        self.assertIsNone(_get_al_nudge_directly(pid))

    def test_silent_when_snapshot_empty(self):
        # Slice 1 wrote {top_k: [], skipped_reason: "..."} — the
        # nudge stays silent and the Data Studio card surfaces the
        # reason instead.
        pid = _create_project()
        snapshot = _build_snapshot(
            model_experiment_id=0, top_k_rows=[],
            skipped_reason="empty_pool",
        )
        _seed_completed_experiment(pid, snapshot=snapshot)
        self.assertIsNone(_get_al_nudge_directly(pid))

    def test_fires_when_fresh_snapshot_with_top_k(self):
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=5)
        snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[(row_ids[i], job_id, 1.0 - i * 0.1) for i in range(5)],
        )
        exp_id = _seed_completed_experiment(pid, snapshot=snapshot)

        nudge = _get_al_nudge_directly(pid)
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["id"], "training:active-learning-ready")
        self.assertEqual(nudge["severity"], "info")
        self.assertEqual(nudge["action"]["kind"], "navigate")
        self.assertEqual(nudge["action"]["params"]["target"], "active-labeling-queue")
        # Dominant job_id stamped so the deep-link lands on the job.
        self.assertEqual(nudge["action"]["params"]["label_job_id"], job_id)
        # Body mentions the unlabeled count (5 of 5) and the metric.
        self.assertIn("5", nudge["title"])
        self.assertIn("entropy", nudge["body"])
        # Context surfaces the snapshot stats the rule decided on.
        self.assertEqual(nudge["context"]["experiment_id"], exp_id)
        self.assertEqual(nudge["context"]["snapshot_size"], 5)
        self.assertEqual(nudge["context"]["labeled_count"], 0)
        self.assertEqual(nudge["context"]["unlabeled_count"], 5)

    def test_silent_when_eighty_percent_labeled(self):
        # 4 of 5 = 0.80 hits the threshold exactly; nudge silences.
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=5)
        snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[(row_ids[i], job_id, 1.0 - i * 0.1) for i in range(5)],
        )
        _seed_completed_experiment(pid, snapshot=snapshot)
        # Mark exactly 4 of 5 labeled.
        _mark_labeled(row_ids[:4])
        self.assertIsNone(_get_al_nudge_directly(pid))

    def test_fires_just_under_threshold(self):
        # 3 of 5 = 0.60 — still under the 0.80 cutoff. Body text
        # reflects 2 unlabeled.
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=5)
        snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[(row_ids[i], job_id, 1.0 - i * 0.1) for i in range(5)],
        )
        _seed_completed_experiment(pid, snapshot=snapshot)
        _mark_labeled(row_ids[:3])
        nudge = _get_al_nudge_directly(pid)
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["context"]["labeled_count"], 3)
        self.assertEqual(nudge["context"]["unlabeled_count"], 2)
        self.assertAlmostEqual(nudge["context"]["staleness_ratio"], 0.60, places=4)
        # Title surfaces unlabeled count, not total — "Label 2 ..." not
        # "Label 5 ...".
        self.assertIn("2 uncertain", nudge["title"])



# ────────────────────────────────────────────────────────────────────────
# Slice 3 — endpoint enrichment for Data Studio card
# ────────────────────────────────────────────────────────────────────────


class LatestEndpointSlice3EnrichmentTests(unittest.TestCase):
    """Slice 3 added text_preview + labeled flag per top_k entry, plus
    dominant_label_job_id at the top level. These pins guard against
    accidental drift in the contract the Data Studio card depends on."""

    def test_top_k_entries_carry_text_preview_from_raw_payload(self):
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=3)
        # Override one row's raw_payload to make the preview check
        # unambiguous (the seeder uses ``text``; this confirms the
        # extractor walks the same field list).
        async def _override_text() -> None:
            async with async_session_factory() as session:
                row = (await session.execute(
                    select(LabelRow).where(LabelRow.id == row_ids[0])
                )).scalar_one()
                row.raw_payload = {"text": "this is a clear uncertain example"}
                await session.commit()
        asyncio.run(_override_text())

        snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[(row_ids[i], job_id, 1.0 - i * 0.1) for i in range(3)],
        )
        _seed_completed_experiment(pid, snapshot=snapshot)
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        entries = body["snapshot"]["top_k"]
        # All three entries have text_preview keys (None or string).
        for entry in entries:
            self.assertIn("text_preview", entry)
        # The overridden row's preview matches verbatim.
        first = next(e for e in entries if e["label_row_id"] == row_ids[0])
        self.assertEqual(first["text_preview"], "this is a clear uncertain example")

    def test_long_text_preview_truncated_with_ellipsis(self):
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=1)
        # 500-char text; preview should clip near the 140-char ceiling
        # and end with an ellipsis so the user sees "this was cut."
        async def _override_text() -> None:
            async with async_session_factory() as session:
                row = (await session.execute(
                    select(LabelRow).where(LabelRow.id == row_ids[0])
                )).scalar_one()
                row.raw_payload = {"text": "a" * 500}
                await session.commit()
        asyncio.run(_override_text())

        snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[(row_ids[0], job_id, 0.9)],
        )
        _seed_completed_experiment(pid, snapshot=snapshot)
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        preview = body["snapshot"]["top_k"][0]["text_preview"]
        self.assertLessEqual(len(preview), 140)
        self.assertTrue(preview.endswith("…"))

    def test_labeled_flag_set_for_labeled_rows(self):
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=3)
        # Label one of the three.
        _mark_labeled([row_ids[1]])

        snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[(row_ids[i], job_id, 1.0 - i * 0.1) for i in range(3)],
        )
        _seed_completed_experiment(pid, snapshot=snapshot)
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        labeled_flags = {
            entry["label_row_id"]: entry["labeled"]
            for entry in body["snapshot"]["top_k"]
        }
        self.assertFalse(labeled_flags[row_ids[0]])
        self.assertTrue(labeled_flags[row_ids[1]])
        self.assertFalse(labeled_flags[row_ids[2]])

    def test_dominant_label_job_id_surfaced(self):
        # Data Studio's "Open label queue" button uses this to deep-link
        # straight to the right job without an extra picker step.
        pid = _create_project()
        job_id, row_ids = _seed_label_rows(pid, n=2)
        snapshot = _build_snapshot(
            model_experiment_id=0,
            top_k_rows=[(rid, job_id, 0.5) for rid in row_ids],
        )
        _seed_completed_experiment(pid, snapshot=snapshot)
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        self.assertEqual(body["dominant_label_job_id"], job_id)

    def test_dominant_label_job_id_none_when_no_snapshot(self):
        pid = _create_project()
        body = CLIENT.get(f"/api/projects/{pid}/active-learning/latest").json()
        self.assertIsNone(body["dominant_label_job_id"])


if __name__ == "__main__":
    unittest.main()

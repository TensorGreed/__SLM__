"""Quality-Lift phase 4 slice 1 — Label-noise scan service + endpoints.

Pins (slice 1: model + scoring service + endpoints + Job runner;
Coach nudge + Data Studio card land in slice 2):

  Scoring service (pure, with patched classifier inference):
    * skipped_reason="no_classifier_checkpoint" when no COMPLETED
      classification experiment exists.
    * skipped_reason="empty_labeled_pool" when project has labeled rows.
    * skipped_reason="no_label_space_configured" when no allowed_labels.
    * skipped_reason="scoring_failed" surfaces error + checkpoint path.
    * Dual condition correctly identifies suspected mislabels:
      predicted_label != given AND predicted_prob ≥ 0.85 AND
      given_label_prob ≤ 0.15.
    * High-confidence agreement does NOT trigger (model agrees with label).
    * Low-confidence disagreement does NOT trigger (model isn't sure).
    * mislabel_score = predicted_prob - given_label_prob ranks suspects
      so the worst (most-suspect) come first in top_k.
    * Rows with no extractable text drop out silently.

  Model round-trip:
    * LabelNoiseScan persists status + thresholds + result_payload.
    * Status enum lifecycle: QUEUED → RUNNING → SUCCEEDED.

  Endpoints:
    * POST /scan returns 202 with QUEUED scan + linked job_id.
    * GET /scans lists past scans most-recent-first.
    * GET /scans/{id} returns one scan; 404 on cross-project.
    * GET /latest returns most recent SUCCEEDED (or null payload).
    * POST /scan to missing project → 404.

  Job runner end-to-end (no real torch):
    * Runner drives QUEUED → RUNNING → SUCCEEDED with result_payload
      stamped + suspected_count denormalized.
    * SUCCEEDED on skipped_reason paths (scan ran to completion);
      FAILED reserved for runner-level exceptions.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

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
from app.models.label_noise_scan import LabelNoiseScan, LabelNoiseScanStatus  # noqa: E402
from app.services.label_noise_scoring_service import (  # noqa: E402
    scan_labeled_rows_for_mislabels,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-ln-s1-{uuid.uuid4().hex[:8]}"
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
        json={"name": f"ln-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


def _make_checkpoint_dir() -> Path:
    """Real on-disk directory so the service's checkpoint-exists
    check passes. The inner inference function is patched in tests so
    we never actually load a model."""
    d = TEST_DATA_DIR / f"checkpoint-{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed_experiment(
    project_id: int,
    *,
    task_type: str = "classification",
    output_dir: str | None = None,
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
) -> int:
    async def _go() -> int:
        async with async_session_factory() as session:
            exp = Experiment(
                project_id=project_id,
                name=f"ln-exp-{uuid.uuid4().hex[:4]}",
                status=status,
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                training_mode=TrainingMode.SFT,
                config={"task_type": task_type},
                output_dir=output_dir or "",
                completed_at=datetime.now(timezone.utc),
            )
            session.add(exp)
            await session.commit()
            return int(exp.id)

    return asyncio.run(_go())


def _seed_classification_label_job_and_rows(
    project_id: int,
    *,
    allowed_labels: list[str] | None = None,
    rows: list[tuple[str, str]] | None = None,
) -> tuple[int, list[int]]:
    """Create a classification label_job + N rows, each pre-labeled.
    ``rows`` is a list of (text, given_label) tuples — all are
    labeled, so they're all eligible for noise scanning."""

    async def _go() -> tuple[int, list[int]]:
        async with async_session_factory() as session:
            effective_labels = (
                allowed_labels if allowed_labels is not None else ["A", "B", "C"]
            )
            job = LabelJob(
                project_id=project_id,
                name=f"ln-job-{uuid.uuid4().hex[:4]}",
                label_type="classification",
                label_schema={"allowed_labels": effective_labels},
            )
            session.add(job)
            await session.flush()
            job_id = int(job.id)
            row_ids: list[int] = []
            for text, given in (rows or []):
                row = LabelRow(
                    job_id=job_id,
                    raw_payload={"text": text},
                    label_payload={"label": given},
                    labeled_at=datetime.now(timezone.utc),
                )
                session.add(row)
                await session.flush()
                row_ids.append(int(row.id))
            await session.commit()
            return job_id, row_ids

    return asyncio.run(_go())


def _run_scoring(
    project_id: int,
    *,
    confidence_threshold: float = 0.85,
    given_label_floor: float = 0.15,
    top_k: int = 50,
) -> dict:
    async def _go() -> dict:
        async with async_session_factory() as session:
            return await scan_labeled_rows_for_mislabels(
                session,
                project_id=project_id,
                confidence_threshold=confidence_threshold,
                given_label_floor=given_label_floor,
                top_k=top_k,
            )

    return asyncio.run(_go())


# ────────────────────────────────────────────────────────────────────────
# Skipped-reason paths
# ────────────────────────────────────────────────────────────────────────


class SkippedReasonTests(unittest.TestCase):

    def test_no_classifier_checkpoint_when_no_completed_experiment(self):
        pid = _create_project()
        snap = _run_scoring(pid)
        self.assertEqual(snap["skipped_reason"], "no_classifier_checkpoint")
        self.assertEqual(snap["top_k"], [])

    def test_no_classifier_checkpoint_skips_non_classification_experiments(self):
        # An experiment exists but it's QA — slice 1 is classification-only.
        pid = _create_project()
        _seed_experiment(pid, task_type="causal_lm", output_dir=str(_make_checkpoint_dir()))
        snap = _run_scoring(pid)
        self.assertEqual(snap["skipped_reason"], "no_classifier_checkpoint")

    def test_empty_labeled_pool_skipped(self):
        # Classification checkpoint exists but no labeled rows yet.
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        snap = _run_scoring(pid)
        self.assertEqual(snap["skipped_reason"], "empty_labeled_pool")

    def test_no_label_space_configured_skipped(self):
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _seed_classification_label_job_and_rows(
            pid,
            allowed_labels=[],  # explicitly empty → no_label_space_configured
            rows=[("hello", "A")],
        )
        snap = _run_scoring(pid)
        self.assertEqual(snap["skipped_reason"], "no_label_space_configured")

    def test_scoring_failed_carries_error_and_path(self):
        pid = _create_project()
        ckpt = _make_checkpoint_dir()
        _seed_experiment(pid, output_dir=str(ckpt))
        _seed_classification_label_job_and_rows(
            pid,
            allowed_labels=["A", "B"],
            rows=[("hello", "A")],
        )

        def _boom(*args, **kwargs):
            raise RuntimeError("torch not available in test env")

        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            side_effect=_boom,
        ):
            snap = _run_scoring(pid)

        self.assertEqual(snap["skipped_reason"], "scoring_failed")
        self.assertIn("torch not available", snap["error"])
        self.assertEqual(snap["checkpoint_path"], str(ckpt))


# ────────────────────────────────────────────────────────────────────────
# Dual condition — the heart of slice 1
# ────────────────────────────────────────────────────────────────────────


class DualConditionTests(unittest.TestCase):
    """Mock the inner inference function to return controlled
    probability vectors, then verify the dual-condition logic picks
    the right rows."""

    def _setup(self, rows_with_labels: list[tuple[str, str]]):
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _seed_classification_label_job_and_rows(
            pid,
            allowed_labels=["A", "B"],
            rows=rows_with_labels,
        )
        return pid

    def test_clear_mislabel_caught(self):
        # Row labeled A but model says B with 0.95 confidence; A gets 0.05.
        # Dual condition: 0.95 ≥ 0.85 ✓ AND 0.05 ≤ 0.15 ✓ → suspected.
        pid = self._setup([("a clear B row", "A")])
        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.05, 0.95]],
        ):
            snap = _run_scoring(pid)

        self.assertIsNone(snap["skipped_reason"])
        self.assertEqual(snap["suspected_count"], 1)
        entry = snap["top_k"][0]
        self.assertEqual(entry["given_label"], "A")
        self.assertEqual(entry["predicted_label"], "B")
        self.assertAlmostEqual(entry["predicted_prob"], 0.95, places=4)
        self.assertAlmostEqual(entry["given_label_prob"], 0.05, places=4)
        self.assertAlmostEqual(entry["mislabel_score"], 0.90, places=4)
        self.assertEqual(entry["text_preview"], "a clear B row")

    def test_model_agrees_with_given_label_not_flagged(self):
        # Row labeled A, model says A — agreement, not a mislabel.
        pid = self._setup([("clearly A", "A")])
        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.95, 0.05]],
        ):
            snap = _run_scoring(pid)
        self.assertEqual(snap["suspected_count"], 0)

    def test_low_confidence_disagreement_not_flagged(self):
        # Row labeled A but model only weakly prefers B (0.55 / 0.45).
        # Below confidence_threshold → false positive avoided.
        pid = self._setup([("ambiguous row", "A")])
        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.45, 0.55]],
        ):
            snap = _run_scoring(pid)
        self.assertEqual(snap["suspected_count"], 0)

    def test_high_confidence_but_given_label_still_alive(self):
        # Model says B at 0.86 (just clears threshold) but given label A
        # is at 0.14. Both clauses must clear; A=0.14 ≤ 0.15 → caught.
        pid = self._setup([("borderline", "A")])
        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.14, 0.86]],
        ):
            snap = _run_scoring(pid)
        self.assertEqual(snap["suspected_count"], 1)

        # Now bump A's prob above the floor — should silence even though
        # B clears the confidence threshold.
        pid2 = self._setup([("borderline above floor", "A")])
        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.18, 0.82]],  # B fails confidence too
        ):
            snap2 = _run_scoring(pid2)
        self.assertEqual(snap2["suspected_count"], 0)

    def test_top_k_sorted_by_mislabel_score_descending(self):
        # Three mislabels at varying scores — top_k should rank them
        # by the (predicted - given) difference. Worst first so slice
        # 3's reviewer sees the most-obvious suspects at the top.
        pid = self._setup([
            ("row 1", "A"),
            ("row 2", "A"),
            ("row 3", "A"),
        ])
        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[
                [0.10, 0.90],  # mislabel_score 0.80
                [0.05, 0.95],  # mislabel_score 0.90
                [0.13, 0.87],  # mislabel_score 0.74
            ],
        ):
            snap = _run_scoring(pid)

        scores = [e["mislabel_score"] for e in snap["top_k"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0], 0.90, places=4)
        self.assertAlmostEqual(scores[-1], 0.74, places=4)

    def test_unknown_given_label_skipped_silently(self):
        # Row label "Z" isn't in allowed_labels — schema drift. We
        # don't have a column to compare against, so the row drops out
        # without being flagged.
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _seed_classification_label_job_and_rows(
            pid,
            allowed_labels=["A", "B"],
            rows=[("text", "Z")],
        )
        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.05, 0.95]],
        ):
            snap = _run_scoring(pid)
        self.assertEqual(snap["suspected_count"], 0)

    def test_text_preview_truncated(self):
        long_text = "x" * 500
        pid = self._setup([(long_text, "A")])
        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.05, 0.95]],
        ):
            snap = _run_scoring(pid)

        preview = snap["top_k"][0]["text_preview"]
        self.assertLessEqual(len(preview), 140)
        self.assertTrue(preview.endswith("…"))


# ────────────────────────────────────────────────────────────────────────
# Endpoints + Job runner end-to-end
# ────────────────────────────────────────────────────────────────────────


class EndpointTests(unittest.TestCase):

    def test_post_scan_404_on_missing_project(self):
        resp = CLIENT.post("/api/projects/999999/label-noise/scan")
        self.assertEqual(resp.status_code, 404)

    def test_get_latest_returns_null_payload_when_no_succeeded_scan(self):
        pid = _create_project()
        resp = CLIENT.get(f"/api/projects/{pid}/label-noise/latest")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIsNone(body["scan"])
        self.assertEqual(body["no_scan_reason"], "no_succeeded_scan_yet")

    def test_get_scans_returns_empty_list_when_no_scans(self):
        pid = _create_project()
        resp = CLIENT.get(f"/api/projects/{pid}/label-noise/scans")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["scans"], [])
        self.assertEqual(body["count"], 0)

    def test_post_scan_runs_to_completion_via_job_runner(self):
        # End-to-end: POST kicks off a Job that drives the scan
        # through RUNNING → SUCCEEDED, persists the result_payload,
        # and stamps the denormalized counts.
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _seed_classification_label_job_and_rows(
            pid,
            allowed_labels=["A", "B"],
            rows=[
                ("clear mislabel", "A"),
                ("agreed label",  "B"),
            ],
        )

        # Patch BEFORE we POST — the runner spins up immediately on
        # asyncio.create_task and we need the patch in place.
        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.05, 0.95], [0.10, 0.90]],
        ):
            # Issue the scan. The fast-track returns immediately with
            # QUEUED; the runner spins up on the event loop.
            resp = CLIENT.post(f"/api/projects/{pid}/label-noise/scan")
            self.assertEqual(resp.status_code, 202, resp.text)
            scan_id = int(resp.json()["id"])

            # Wait for the background task to finish. The Jobs framework
            # uses asyncio.create_task; the TestClient runs the event
            # loop in a separate thread, so we poll the DB until status
            # transitions.
            async def _poll() -> LabelNoiseScan:
                for _ in range(50):
                    async with async_session_factory() as session:
                        row = (await session.execute(
                            select(LabelNoiseScan).where(LabelNoiseScan.id == scan_id)
                        )).scalar_one()
                        if row.status in (
                            LabelNoiseScanStatus.SUCCEEDED,
                            LabelNoiseScanStatus.FAILED,
                        ):
                            return row
                    await asyncio.sleep(0.05)
                raise AssertionError("scan never finished")

            final = asyncio.run(_poll())

        self.assertEqual(final.status, LabelNoiseScanStatus.SUCCEEDED)
        # One row matched dual condition (A given, B predicted at 0.95).
        # The second row's model picks B as predicted at 0.90 AND its
        # given label is B → agreement, not a mislabel.
        self.assertEqual(final.suspected_count, 1)
        self.assertEqual(final.label_count_at_scan, 2)
        self.assertIsNotNone(final.completed_at)
        self.assertIsNotNone(final.job_id)
        # Result payload has the suspected entry at the top.
        payload = final.result_payload or {}
        self.assertEqual(len(payload["top_k"]), 1)
        self.assertEqual(payload["top_k"][0]["given_label"], "A")
        self.assertEqual(payload["top_k"][0]["predicted_label"], "B")

    def test_get_latest_returns_serialized_scan_after_completion(self):
        # After the above test we know the runner works; this verifies
        # the /latest endpoint reads the most recent SUCCEEDED scan.
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _seed_classification_label_job_and_rows(
            pid, allowed_labels=["A", "B"],
            rows=[("clear mislabel", "A")],
        )

        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.05, 0.95]],
        ):
            CLIENT.post(f"/api/projects/{pid}/label-noise/scan")

            async def _poll() -> None:
                for _ in range(50):
                    async with async_session_factory() as session:
                        rows = (await session.execute(
                            select(LabelNoiseScan).where(
                                LabelNoiseScan.project_id == pid,
                                LabelNoiseScan.status == LabelNoiseScanStatus.SUCCEEDED,
                            )
                        )).scalars().all()
                        if rows:
                            return
                    await asyncio.sleep(0.05)

            asyncio.run(_poll())

        resp = CLIENT.get(f"/api/projects/{pid}/label-noise/latest")
        body = resp.json()
        self.assertIsNone(body["no_scan_reason"])
        self.assertEqual(body["scan"]["status"], "succeeded")
        self.assertEqual(body["scan"]["suspected_count"], 1)
        # result_payload included so slice 3's review surface has
        # everything in one fetch.
        self.assertIsNotNone(body["scan"]["result_payload"])

    def test_get_scan_by_id_404_on_cross_project(self):
        pid_a = _create_project()
        pid_b = _create_project()
        _seed_experiment(pid_a, output_dir=str(_make_checkpoint_dir()))
        _seed_classification_label_job_and_rows(
            pid_a, allowed_labels=["A", "B"],
            rows=[("clear mislabel", "A")],
        )

        with patch(
            "app.services.label_noise_scoring_service._score_rows_with_classifier_head",
            return_value=[[0.05, 0.95]],
        ):
            resp = CLIENT.post(f"/api/projects/{pid_a}/label-noise/scan")
            scan_id = int(resp.json()["id"])

        # Fetching this scan as project B → 404. Treated same as
        # "doesn't exist" so we don't leak cross-project scan ids.
        cross = CLIENT.get(f"/api/projects/{pid_b}/label-noise/scans/{scan_id}")
        self.assertEqual(cross.status_code, 404)



# ────────────────────────────────────────────────────────────────────────
# Slice 2 — Cleaning-stage Coach nudges
# ────────────────────────────────────────────────────────────────────────


def _seed_labeled_rows_bulk(
    project_id: int,
    *,
    n: int,
    label: str = "A",
    allowed_labels: list[str] | None = None,
) -> tuple[int, list[int]]:
    """Seed N rows already labeled. Mirrors slice 1's helper but with
    a cleaner signature for slice 2's tests where we only care about
    the count."""
    rows = [(f"row-{i}", label) for i in range(n)]
    return _seed_classification_label_job_and_rows(
        project_id,
        allowed_labels=allowed_labels or ["A", "B"],
        rows=rows,
    )


def _seed_succeeded_scan(
    project_id: int,
    *,
    label_count_at_scan: int,
    suspected_count: int,
    confidence_threshold: float = 0.85,
    given_label_floor: float = 0.15,
    completed_at: datetime | None = None,
) -> int:
    """Insert a SUCCEEDED LabelNoiseScan row directly (bypass the
    runner) so the Coach nudge tests don't have to wait for async
    scoring."""
    async def _go() -> int:
        async with async_session_factory() as session:
            scan = LabelNoiseScan(
                project_id=project_id,
                status=LabelNoiseScanStatus.SUCCEEDED,
                label_count_at_scan=label_count_at_scan,
                suspected_count=suspected_count,
                confidence_threshold=confidence_threshold,
                given_label_floor=given_label_floor,
                result_payload={
                    "scored_at": (completed_at or datetime.now(timezone.utc)).isoformat(),
                    "base_experiment_id": None,
                    "label_count_total": label_count_at_scan,
                    "label_count_scored": label_count_at_scan,
                    "suspected_count": suspected_count,
                    "confidence_threshold": confidence_threshold,
                    "given_label_floor": given_label_floor,
                    "top_k": [],
                    "skipped_reason": None,
                },
                completed_at=completed_at or datetime.now(timezone.utc),
            )
            session.add(scan)
            await session.commit()
            return int(scan.id)

    return asyncio.run(_go())


def _seed_inflight_scan(project_id: int, *, status: LabelNoiseScanStatus) -> int:
    async def _go() -> int:
        async with async_session_factory() as session:
            scan = LabelNoiseScan(
                project_id=project_id,
                status=status,
            )
            session.add(scan)
            await session.commit()
            return int(scan.id)

    return asyncio.run(_go())


def _get_scan_ready_nudge(project_id: int):
    from app.services.coach_service import _label_noise_scan_ready_nudge

    async def _go():
        async with async_session_factory() as session:
            return await _label_noise_scan_ready_nudge(session, project_id)

    return asyncio.run(_go())


def _get_results_pending_nudge(project_id: int):
    from app.services.coach_service import _label_noise_results_pending_nudge

    async def _go():
        async with async_session_factory() as session:
            return await _label_noise_results_pending_nudge(session, project_id)

    return asyncio.run(_go())


class ScanReadyNudgeTests(unittest.TestCase):

    def test_silent_below_min_labeled(self):
        # Floor of 50. Below that, the model isn't trained well enough
        # for self-confidence disagreement to be meaningful.
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _seed_labeled_rows_bulk(pid, n=49)
        self.assertIsNone(_get_scan_ready_nudge(pid))

    def test_silent_without_classifier_checkpoint(self):
        # 50 labels but no completed classification experiment → nothing
        # to score against. Nudge silences cleanly rather than firing
        # a "scan!" suggestion that would just skip.
        pid = _create_project()
        _seed_labeled_rows_bulk(pid, n=60)
        self.assertIsNone(_get_scan_ready_nudge(pid))

    def test_fires_on_first_scan_when_threshold_crossed(self):
        # 50+ labels, classifier checkpoint exists, no prior scan →
        # first-scan nudge fires.
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _seed_labeled_rows_bulk(pid, n=60)

        nudge = _get_scan_ready_nudge(pid)
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["id"], "cleaning:label-noise-scan-ready")
        self.assertEqual(nudge["rule_id"], "label-noise-scan-ready.first-scan")
        # Title surfaces the label count so the user sees scale.
        self.assertIn("60", nudge["title"])
        # Action carries the navigate target + auto_start_scan flag.
        self.assertEqual(nudge["action"]["params"]["target"], "label-noise-review")
        self.assertEqual(nudge["action"]["params"]["auto_start_scan"], True)
        self.assertEqual(nudge["context"]["labeled_count"], 60)
        self.assertEqual(nudge["context"]["new_labels_since_last_scan"], 60)

    def test_silent_when_latest_scan_covers_eighty_percent(self):
        # Latest scan ran on 80% of current labels → user hasn't added
        # a meaningful new batch; don't pester. 80 of 100 = 0.80 hits
        # the threshold exactly.
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _seed_labeled_rows_bulk(pid, n=100)
        _seed_succeeded_scan(
            pid, label_count_at_scan=80, suspected_count=0,
        )
        self.assertIsNone(_get_scan_ready_nudge(pid))

    def test_fires_when_new_batch_meaningfully_grew_pool(self):
        # Latest scan ran on 70 of 100 current labels (just under 80%).
        # Re-scan nudge fires; title uses the "new labels since last
        # scan" framing.
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _seed_labeled_rows_bulk(pid, n=100)
        _seed_succeeded_scan(
            pid, label_count_at_scan=70, suspected_count=0,
        )

        nudge = _get_scan_ready_nudge(pid)
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["rule_id"], "label-noise-scan-ready.rescan-ready")
        self.assertEqual(nudge["context"]["new_labels_since_last_scan"], 30)

    def test_silent_when_scan_is_queued_or_running(self):
        # In-flight scan — nudging to start another would be annoying.
        # Test both QUEUED and RUNNING states.
        for status in (LabelNoiseScanStatus.QUEUED, LabelNoiseScanStatus.RUNNING):
            with self.subTest(status=status):
                pid = _create_project()
                _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
                _seed_labeled_rows_bulk(pid, n=60)
                _seed_inflight_scan(pid, status=status)
                self.assertIsNone(_get_scan_ready_nudge(pid))


class ResultsPendingNudgeTests(unittest.TestCase):

    def test_silent_when_no_succeeded_scan(self):
        pid = _create_project()
        self.assertIsNone(_get_results_pending_nudge(pid))

    def test_silent_when_latest_scan_clean(self):
        # Scan ran but found zero suspects — labels are clean. Silence
        # rather than show "Review 0 mislabels."
        pid = _create_project()
        _seed_succeeded_scan(pid, label_count_at_scan=100, suspected_count=0)
        self.assertIsNone(_get_results_pending_nudge(pid))

    def test_fires_when_scan_has_suspects(self):
        pid = _create_project()
        scan_id = _seed_succeeded_scan(
            pid, label_count_at_scan=100, suspected_count=7,
        )
        nudge = _get_results_pending_nudge(pid)
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["id"], "cleaning:label-noise-results-pending")
        self.assertEqual(nudge["severity"], "warning")
        self.assertIn("7", nudge["title"])
        # Action deep-links to the slice 3 review surface with scan_id.
        self.assertEqual(
            nudge["action"]["params"]["target"], "label-noise-review",
        )
        self.assertEqual(nudge["action"]["params"]["scan_id"], scan_id)
        self.assertEqual(nudge["context"]["suspected_count"], 7)

    def test_title_handles_singular_correctly(self):
        # 1 suspect → "Review 1 suspected mislabel" (singular).
        pid = _create_project()
        _seed_succeeded_scan(pid, label_count_at_scan=100, suspected_count=1)
        nudge = _get_results_pending_nudge(pid)
        self.assertIsNotNone(nudge)
        self.assertIn("1 suspected mislabel", nudge["title"])
        self.assertNotIn("mislabels", nudge["title"])



# ────────────────────────────────────────────────────────────────────────
# Slice 3 — Apply endpoint (relabel / keep / drop)
# ────────────────────────────────────────────────────────────────────────


def _seed_succeeded_scan_with_top_k(
    project_id: int,
    *,
    top_k_rows: list[tuple[int, int, str, str]],
) -> int:
    """Create a SUCCEEDED scan with a top_k built from
    ``(label_row_id, label_job_id, given_label, predicted_label)``
    tuples. Probabilities are filled with plausible defaults so the
    payload looks scan-1-shaped to the apply endpoint."""
    async def _go() -> int:
        async with async_session_factory() as session:
            top_k = []
            for rid, jid, given, pred in top_k_rows:
                top_k.append({
                    "label_row_id": rid,
                    "label_job_id": jid,
                    "given_label": given,
                    "predicted_label": pred,
                    "predicted_prob": 0.92,
                    "given_label_prob": 0.05,
                    "mislabel_score": 0.87,
                    "text_preview": f"row {rid}",
                })
            scan = LabelNoiseScan(
                project_id=project_id,
                status=LabelNoiseScanStatus.SUCCEEDED,
                label_count_at_scan=len(top_k_rows),
                suspected_count=len(top_k_rows),
                confidence_threshold=0.85,
                given_label_floor=0.15,
                result_payload={
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                    "base_experiment_id": None,
                    "label_count_total": len(top_k_rows),
                    "label_count_scored": len(top_k_rows),
                    "suspected_count": len(top_k_rows),
                    "confidence_threshold": 0.85,
                    "given_label_floor": 0.15,
                    "top_k": top_k,
                    "skipped_reason": None,
                },
                completed_at=datetime.now(timezone.utc),
            )
            session.add(scan)
            await session.commit()
            return int(scan.id)

    return asyncio.run(_go())


def _load_label_row(row_id: int) -> LabelRow:
    async def _go() -> LabelRow:
        async with async_session_factory() as session:
            return (await session.execute(
                select(LabelRow).where(LabelRow.id == row_id)
            )).scalar_one()

    return asyncio.run(_go())


def _load_scan(scan_id: int) -> LabelNoiseScan:
    async def _go() -> LabelNoiseScan:
        async with async_session_factory() as session:
            return (await session.execute(
                select(LabelNoiseScan).where(LabelNoiseScan.id == scan_id)
            )).scalar_one()

    return asyncio.run(_go())


class ApplyEndpointTests(unittest.TestCase):

    def test_404_when_scan_missing(self):
        pid = _create_project()
        resp = CLIENT.post(
            f"/api/projects/{pid}/label-noise/scans/99999/apply",
            json={"actions": [{"label_row_id": 1, "action": "keep"}]},
        )
        self.assertEqual(resp.status_code, 404)

    def test_409_when_scan_not_succeeded(self):
        # Applying against a still-running scan is confused — the top_k
        # isn't authoritative until SUCCEEDED. 409 so the UI can keep
        # the apply button disabled in this state.
        pid = _create_project()
        _seed_experiment(pid, output_dir=str(_make_checkpoint_dir()))
        _, row_ids = _seed_classification_label_job_and_rows(
            pid, allowed_labels=["A", "B"], rows=[("text", "A")],
        )
        scan_id = _seed_inflight_scan(pid, status=LabelNoiseScanStatus.RUNNING)

        resp = CLIENT.post(
            f"/api/projects/{pid}/label-noise/scans/{scan_id}/apply",
            json={"actions": [{"label_row_id": row_ids[0], "action": "keep"}]},
        )
        self.assertEqual(resp.status_code, 409)

    def test_cross_project_scan_returns_404(self):
        # Applying to a scan in another project must NOT succeed. We
        # treat as 404 (not 403) so we don't leak existence of
        # cross-project scan ids.
        pid_a = _create_project()
        pid_b = _create_project()
        _, row_ids_a = _seed_classification_label_job_and_rows(
            pid_a, allowed_labels=["A", "B"], rows=[("text", "A")],
        )
        scan_id = _seed_succeeded_scan_with_top_k(
            pid_a, top_k_rows=[(row_ids_a[0], 1, "A", "B")],
        )

        resp = CLIENT.post(
            f"/api/projects/{pid_b}/label-noise/scans/{scan_id}/apply",
            json={"actions": [{"label_row_id": row_ids_a[0], "action": "keep"}]},
        )
        self.assertEqual(resp.status_code, 404)

    def test_relabel_updates_label_payload(self):
        # Relabel uses the scan's predicted_label as the new value.
        # labeled_at stays — the row remains in the labeled pool with
        # the corrected label.
        pid = _create_project()
        job_id, row_ids = _seed_classification_label_job_and_rows(
            pid, allowed_labels=["A", "B"],
            rows=[("clear B row", "A")],
        )
        scan_id = _seed_succeeded_scan_with_top_k(
            pid, top_k_rows=[(row_ids[0], job_id, "A", "B")],
        )

        resp = CLIENT.post(
            f"/api/projects/{pid}/label-noise/scans/{scan_id}/apply",
            json={"actions": [{"label_row_id": row_ids[0], "action": "relabel"}]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["relabeled"], 1)
        self.assertEqual(body["kept"], 0)
        self.assertEqual(body["dropped"], 0)

        row = _load_label_row(row_ids[0])
        self.assertEqual(row.label_payload, {"label": "B"})
        # labeled_at preserved — relabel doesn't put the row back in
        # the unlabeled pool.
        self.assertIsNotNone(row.labeled_at)

    def test_drop_clears_label_payload_and_labeled_at(self):
        # The loop closure: dropping returns the row to phase 3's
        # unlabeled pool. The next post-training active-learning
        # scoring picks it up there.
        pid = _create_project()
        job_id, row_ids = _seed_classification_label_job_and_rows(
            pid, allowed_labels=["A", "B"],
            rows=[("ambiguous row", "A")],
        )
        scan_id = _seed_succeeded_scan_with_top_k(
            pid, top_k_rows=[(row_ids[0], job_id, "A", "B")],
        )

        resp = CLIENT.post(
            f"/api/projects/{pid}/label-noise/scans/{scan_id}/apply",
            json={"actions": [{"label_row_id": row_ids[0], "action": "drop"}]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["dropped"], 1)

        row = _load_label_row(row_ids[0])
        self.assertIsNone(row.label_payload)
        self.assertIsNone(row.labeled_at)
        self.assertIsNone(row.assigned_to)

    def test_keep_records_audit_no_row_change(self):
        # Keep doesn't change the row — but the audit log records the
        # decision so the UI can show "previously kept" instead of
        # re-suggesting the row.
        pid = _create_project()
        job_id, row_ids = _seed_classification_label_job_and_rows(
            pid, allowed_labels=["A", "B"],
            rows=[("genuine A row", "A")],
        )
        scan_id = _seed_succeeded_scan_with_top_k(
            pid, top_k_rows=[(row_ids[0], job_id, "A", "B")],
        )

        resp = CLIENT.post(
            f"/api/projects/{pid}/label-noise/scans/{scan_id}/apply",
            json={"actions": [{"label_row_id": row_ids[0], "action": "keep"}]},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["kept"], 1)

        row = _load_label_row(row_ids[0])
        # Row unchanged.
        self.assertEqual(row.label_payload, {"label": "A"})
        self.assertIsNotNone(row.labeled_at)

        # Audit log on the scan carries the decision.
        scan = _load_scan(scan_id)
        applied = scan.result_payload.get("applied_actions") or {}
        self.assertEqual(applied[str(row_ids[0])]["action"], "keep")

    def test_unknown_row_id_skipped_not_failed(self):
        # An action against a row not in the scan's top_k is skipped
        # rather than failing the whole batch. Defends against UI
        # tampering / scan replacement races.
        pid = _create_project()
        job_id, row_ids = _seed_classification_label_job_and_rows(
            pid, allowed_labels=["A", "B"],
            rows=[("text", "A")],
        )
        scan_id = _seed_succeeded_scan_with_top_k(
            pid, top_k_rows=[(row_ids[0], job_id, "A", "B")],
        )

        resp = CLIENT.post(
            f"/api/projects/{pid}/label-noise/scans/{scan_id}/apply",
            json={"actions": [
                {"label_row_id": row_ids[0], "action": "keep"},
                {"label_row_id": 999999, "action": "relabel"},
            ]},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["kept"], 1)
        self.assertEqual(body["relabeled"], 0)
        skipped_ids = [s["label_row_id"] for s in body["skipped"]]
        self.assertIn(999999, skipped_ids)
        self.assertEqual(body["skipped"][0]["reason"], "not_in_scan_top_k")

    def test_idempotent_reapply_overwrites_audit(self):
        # Applying twice with different decisions: latest wins. The
        # audit log shows the latest action per row, not a history.
        pid = _create_project()
        job_id, row_ids = _seed_classification_label_job_and_rows(
            pid, allowed_labels=["A", "B"],
            rows=[("borderline", "A")],
        )
        scan_id = _seed_succeeded_scan_with_top_k(
            pid, top_k_rows=[(row_ids[0], job_id, "A", "B")],
        )

        CLIENT.post(
            f"/api/projects/{pid}/label-noise/scans/{scan_id}/apply",
            json={"actions": [{"label_row_id": row_ids[0], "action": "keep"}]},
        )
        # Reconsider — actually drop it.
        resp2 = CLIENT.post(
            f"/api/projects/{pid}/label-noise/scans/{scan_id}/apply",
            json={"actions": [{"label_row_id": row_ids[0], "action": "drop"}]},
        )
        body = resp2.json()
        self.assertEqual(body["dropped"], 1)

        scan = _load_scan(scan_id)
        applied = scan.result_payload.get("applied_actions") or {}
        self.assertEqual(applied[str(row_ids[0])]["action"], "drop")

    def test_bulk_apply_handles_mix(self):
        # The slice 3 UI bulk-applies per (given→predicted) group.
        # Verify the endpoint handles a mixed batch cleanly: 2
        # relabels + 1 keep + 1 drop in a single call.
        pid = _create_project()
        job_id, row_ids = _seed_classification_label_job_and_rows(
            pid, allowed_labels=["A", "B"],
            rows=[("r0", "A"), ("r1", "A"), ("r2", "A"), ("r3", "A")],
        )
        scan_id = _seed_succeeded_scan_with_top_k(
            pid,
            top_k_rows=[
                (row_ids[0], job_id, "A", "B"),
                (row_ids[1], job_id, "A", "B"),
                (row_ids[2], job_id, "A", "B"),
                (row_ids[3], job_id, "A", "B"),
            ],
        )

        resp = CLIENT.post(
            f"/api/projects/{pid}/label-noise/scans/{scan_id}/apply",
            json={"actions": [
                {"label_row_id": row_ids[0], "action": "relabel"},
                {"label_row_id": row_ids[1], "action": "relabel"},
                {"label_row_id": row_ids[2], "action": "keep"},
                {"label_row_id": row_ids[3], "action": "drop"},
            ]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["applied"], 4)
        self.assertEqual(body["relabeled"], 2)
        self.assertEqual(body["kept"], 1)
        self.assertEqual(body["dropped"], 1)

        # Verify each row's state matches the decision.
        self.assertEqual(_load_label_row(row_ids[0]).label_payload, {"label": "B"})
        self.assertEqual(_load_label_row(row_ids[1]).label_payload, {"label": "B"})
        self.assertEqual(_load_label_row(row_ids[2]).label_payload, {"label": "A"})
        dropped_row = _load_label_row(row_ids[3])
        self.assertIsNone(dropped_row.labeled_at)


if __name__ == "__main__":
    unittest.main()

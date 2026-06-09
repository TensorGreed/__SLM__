"""Quality-Lift phase 4 slice 1 — Label-noise scan API.

Three endpoints + a Job runner:

  POST   /api/projects/{id}/label-noise/scan            — kick off a scan; 202
  GET    /api/projects/{id}/label-noise/scans           — list past scans
  GET    /api/projects/{id}/label-noise/scans/{scan_id} — one scan with full payload
  GET    /api/projects/{id}/label-noise/latest          — most recent SUCCEEDED

The scan runs in the background via the Jobs framework — same shape
phase 1's training-watcher uses. The user gets back ``{scan_id,
job_id, status: "queued"}`` immediately; the notification bell renders
progress while the background runner walks the labeled pool.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.models.label_noise_scan import LabelNoiseScan, LabelNoiseScanStatus
from app.models.project import Project
from app.services.label_noise_scoring_service import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_GIVEN_LABEL_FLOOR,
    DEFAULT_SAMPLE_CAP,
    DEFAULT_TOP_K,
    scan_labeled_rows_for_mislabels,
)
from app.services.jobs_service import JobProgressHandle, start_job


router = APIRouter(
    prefix="/projects/{project_id}/label-noise",
    tags=["LabelNoise"],
)


class ScanRequest(BaseModel):
    """Optional overrides for the scan. Every field has a sensible
    default so the simplest invocation is just ``POST /scan`` with no
    body."""

    base_experiment_id: int | None = Field(
        None,
        description=(
            "Pin a specific COMPLETED classification experiment to score "
            "against. Defaults to the latest such experiment."
        ),
    )
    confidence_threshold: float = Field(
        DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.5,
        le=1.0,
        description="Dual-condition predicted_prob floor.",
    )
    given_label_floor: float = Field(
        DEFAULT_GIVEN_LABEL_FLOOR,
        ge=0.0,
        le=0.5,
        description="Dual-condition given_label_prob ceiling.",
    )
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=1000)
    sample_cap: int = Field(DEFAULT_SAMPLE_CAP, ge=10, le=20000)


def _serialize_scan(scan: LabelNoiseScan) -> dict[str, Any]:
    """Stable serialization shape used by all three endpoints. Slice
    2's Coach nudge + Data Studio card both read these fields
    directly — drift here breaks them."""
    return {
        "id": int(scan.id),
        "project_id": int(scan.project_id),
        "base_experiment_id": (
            int(scan.base_experiment_id) if scan.base_experiment_id else None
        ),
        "status": scan.status.value,
        "label_count_at_scan": scan.label_count_at_scan,
        "suspected_count": scan.suspected_count,
        "confidence_threshold": scan.confidence_threshold,
        "given_label_floor": scan.given_label_floor,
        "result_payload": scan.result_payload,
        "error": scan.error,
        "job_id": int(scan.job_id) if scan.job_id else None,
        "created_at": scan.created_at.isoformat() if scan.created_at else None,
        "completed_at": (
            scan.completed_at.isoformat() if scan.completed_at else None
        ),
    }


def _build_runner(
    scan_id: int,
    project_id: int,
    overrides: dict[str, Any],
):
    """Build the Job runner closure that:
      1. Flips the scan status to RUNNING.
      2. Calls scan_labeled_rows_for_mislabels.
      3. Persists the result_payload + denormalized counters.
      4. Flips status to SUCCEEDED (or FAILED on exception).

    Returns the runner function — start_job hands it the
    JobProgressHandle and tracks its lifecycle."""

    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        # Phase 1: RUNNING transition + best-effort progress message
        # so the bell shows something useful while we're loading the
        # model.
        async with async_session_factory() as db:
            row = await db.execute(
                select(LabelNoiseScan).where(LabelNoiseScan.id == scan_id)
            )
            scan = row.scalar_one_or_none()
            if scan is None:
                return {"error": "scan_row_missing"}
            scan.status = LabelNoiseScanStatus.RUNNING
            await db.commit()
        await handle.set_progress(fraction=0.05, message="Loading classifier checkpoint…")

        # Phase 2: actually score.
        try:
            async with async_session_factory() as db:
                payload = await scan_labeled_rows_for_mislabels(
                    db,
                    project_id=project_id,
                    **overrides,
                )
        except Exception as exc:  # noqa: BLE001 — scan never breaks the Job
            async with async_session_factory() as db:
                row = await db.execute(
                    select(LabelNoiseScan).where(LabelNoiseScan.id == scan_id)
                )
                scan = row.scalar_one_or_none()
                if scan is not None:
                    scan.status = LabelNoiseScanStatus.FAILED
                    scan.error = str(exc)[:1024]
                    from datetime import datetime, timezone
                    scan.completed_at = datetime.now(timezone.utc)
                    await db.commit()
            return {"error": str(exc)[:1024]}

        # Phase 3: persist + transition to SUCCEEDED. We mark
        # SUCCEEDED even when the scoring service skipped (no
        # checkpoint, empty pool) — the scan ran to completion, the
        # result_payload just carries the skipped_reason. FAILED is
        # reserved for runner-level exceptions.
        async with async_session_factory() as db:
            row = await db.execute(
                select(LabelNoiseScan).where(LabelNoiseScan.id == scan_id)
            )
            scan = row.scalar_one_or_none()
            if scan is None:
                return payload
            scan.result_payload = payload
            scan.suspected_count = int(payload.get("suspected_count") or 0)
            scan.label_count_at_scan = int(payload.get("label_count_total") or 0)
            scan.status = LabelNoiseScanStatus.SUCCEEDED
            from datetime import datetime, timezone
            scan.completed_at = datetime.now(timezone.utc)
            await db.commit()
        await handle.set_progress(fraction=1.0, message="Scan complete")
        return payload

    return _runner


@router.post("/scan", status_code=202)
async def start_label_noise_scan(
    project_id: int,
    req: ScanRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Kick off a scan. Returns the scan row in QUEUED state +
    the Job id so the caller can poll progress via the existing
    bell. Optional body overrides the threshold defaults."""
    project_row = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = project_row.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    body = req or ScanRequest()

    # Create the scan row in QUEUED. We persist FIRST so the runner
    # has a row to update — start_job's wrapper transitions the Job
    # but the LabelNoiseScan transitions are owned by our runner.
    scan = LabelNoiseScan(
        project_id=project_id,
        base_experiment_id=body.base_experiment_id,
        status=LabelNoiseScanStatus.QUEUED,
        confidence_threshold=body.confidence_threshold,
        given_label_floor=body.given_label_floor,
    )
    db.add(scan)
    await db.flush()
    await db.commit()
    await db.refresh(scan)
    scan_id = int(scan.id)

    overrides: dict[str, Any] = {
        "base_experiment_id": body.base_experiment_id,
        "confidence_threshold": body.confidence_threshold,
        "given_label_floor": body.given_label_floor,
        "top_k": body.top_k,
        "sample_cap": body.sample_cap,
    }

    job = await start_job(
        db,
        kind="label_noise_scan",
        title=f"Label-noise scan #{scan_id}",
        runner=_build_runner(scan_id, project_id, overrides),
        project_id=project_id,
        params={
            "scan_id": scan_id,
            "confidence_threshold": body.confidence_threshold,
            "given_label_floor": body.given_label_floor,
            "top_k": body.top_k,
            "sample_cap": body.sample_cap,
        },
    )

    # Stamp job_id on the scan now that we have it.
    scan_row = await db.execute(
        select(LabelNoiseScan).where(LabelNoiseScan.id == scan_id)
    )
    scan = scan_row.scalar_one()
    scan.job_id = int(job.id)
    await db.commit()
    await db.refresh(scan)

    return _serialize_scan(scan)


@router.get("/scans")
async def list_label_noise_scans(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List past scans, most recent first. Result_payload is
    INCLUDED — slice 3's review surface needs it for comparison
    across scans; the payload is small (top-K capped at 1000)."""
    project_row = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    if project_row.scalar_one_or_none() is None:
        raise HTTPException(404, f"Project {project_id} not found")

    rows = await db.execute(
        select(LabelNoiseScan)
        .where(LabelNoiseScan.project_id == project_id)
        .order_by(LabelNoiseScan.created_at.desc(), LabelNoiseScan.id.desc())
    )
    scans = [_serialize_scan(s) for s in rows.scalars().all()]
    return {"project_id": project_id, "scans": scans, "count": len(scans)}


@router.get("/scans/{scan_id}")
async def get_label_noise_scan(
    project_id: int,
    scan_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Fetch a single scan by id. 404 when the scan doesn't exist
    or belongs to a different project (treated the same so we don't
    leak cross-project scan existence)."""
    rows = await db.execute(
        select(LabelNoiseScan).where(
            LabelNoiseScan.id == scan_id,
            LabelNoiseScan.project_id == project_id,
        )
    )
    scan = rows.scalar_one_or_none()
    if scan is None:
        raise HTTPException(404, f"Scan {scan_id} not found")
    return _serialize_scan(scan)


@router.get("/latest")
async def get_latest_label_noise_scan(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Most recent SUCCEEDED scan, or a well-formed null payload
    when none exist yet. Mirrors phase 3's /active-learning/latest
    contract so slice 2's Coach nudge + Data Studio card can read a
    stable shape without undefined-checks."""
    project_row = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    if project_row.scalar_one_or_none() is None:
        raise HTTPException(404, f"Project {project_id} not found")

    rows = await db.execute(
        select(LabelNoiseScan)
        .where(
            LabelNoiseScan.project_id == project_id,
            LabelNoiseScan.status == LabelNoiseScanStatus.SUCCEEDED,
        )
        .order_by(LabelNoiseScan.completed_at.desc(), LabelNoiseScan.id.desc())
    )
    latest = rows.scalars().first()
    if latest is None:
        return {
            "project_id": project_id,
            "scan": None,
            "no_scan_reason": "no_succeeded_scan_yet",
        }
    return {
        "project_id": project_id,
        "scan": _serialize_scan(latest),
        "no_scan_reason": None,
    }

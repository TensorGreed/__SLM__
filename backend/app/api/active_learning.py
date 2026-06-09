"""Quality-Lift phase 3 slice 2 — Active-learning snapshot read endpoint.

  ``GET /api/projects/{project_id}/active-learning/latest``

Reads the most recent COMPLETED experiment's snapshot from
``exp.config._runtime["active_learning"]`` (written by the slice 1
post-training hook) and joins the ``top_k`` row ids against
``label_rows.labeled_at`` to derive a freshness signal:

  - ``labeled_count`` — how many of the snapshot's top-K rows have
    been labeled SINCE the snapshot was taken (or any time after the
    snapshot — labeled_at after scored_at).
  - ``unlabeled_count`` — remaining top-K rows the user hasn't yet
    addressed.
  - ``staleness_ratio`` — labeled_count / top_k_size. The Coach nudge
    silences when ≥ ``STALENESS_THRESHOLD`` (0.80) of the snapshot
    has been worked.

When no project / no COMPLETED experiment / no snapshot, returns a
``no_snapshot`` payload with the reason so the Data Studio card
(slice 3) can render a contextual empty state.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.experiment import Experiment, ExperimentStatus
from app.models.label_job import LabelRow
from app.models.project import Project


# Slice 2 contract: the Coach nudge stops firing once 80% of the
# snapshot's rows have been labeled. Tunable by tweaking this constant
# — single source of truth so the Coach + endpoint can't drift.
STALENESS_THRESHOLD = 0.80


router = APIRouter(
    prefix="/projects/{project_id}/active-learning",
    tags=["ActiveLearning"],
)


async def _latest_completed_experiment_with_snapshot(
    db: AsyncSession, project_id: int,
) -> Experiment | None:
    """Most recent COMPLETED experiment whose config carries an
    active-learning snapshot. We pick the snapshot-carrying one (not
    just the latest) so the response is meaningful even when the user
    has trained twice and only one run had unlabeled rows to score.
    """
    rows = await db.execute(
        select(Experiment)
        .where(
            Experiment.project_id == project_id,
            Experiment.status == ExperimentStatus.COMPLETED,
        )
        .order_by(Experiment.completed_at.desc(), Experiment.id.desc())
    )
    for exp in rows.scalars().all():
        cfg = exp.config if isinstance(exp.config, dict) else {}
        runtime = cfg.get("_runtime") or {}
        if isinstance(runtime, dict) and isinstance(runtime.get("active_learning"), dict):
            return exp
    return None


async def _count_labeled_rows(
    db: AsyncSession, label_row_ids: list[int],
) -> int:
    """How many of the snapshot's row ids now have ``labeled_at`` set?

    We do NOT filter on ``labeled_at >= snapshot.scored_at`` — labeling
    progress made *before* the snapshot still counts. If a user labels
    50 of the top-K and then re-trains, the next snapshot may
    re-surface a different pool; the staleness check on the new
    snapshot should not be confused by earlier labeling effort.
    """
    if not label_row_ids:
        return 0
    rows = await db.execute(
        select(LabelRow.id).where(
            LabelRow.id.in_(label_row_ids),
            LabelRow.labeled_at.is_not(None),
        )
    )
    return len(list(rows.scalars().all()))


@router.get("/latest")
async def get_latest_active_learning_snapshot(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return the project's most recent active-learning snapshot
    enriched with derived staleness fields.

    Stable shape — the Coach nudge + Data Studio card (slice 3) both
    read these fields directly:
      * ``snapshot`` — verbatim from ``exp.config._runtime["active_learning"]``,
        or null when no snapshot exists.
      * ``experiment_id`` / ``experiment_name`` — provenance for the
        Data Studio card to render "scored by exp #N".
      * ``top_k_size`` — number of entries the snapshot carries.
      * ``labeled_count`` / ``unlabeled_count`` — derived from
        live ``label_rows`` state.
      * ``staleness_ratio`` — labeled_count / top_k_size (0.0 when
        snapshot is empty, so the Coach nudge's 0.80 cutoff doesn't
        misfire on zero division).
      * ``is_stale`` — True when staleness_ratio >= STALENESS_THRESHOLD.
        The Coach nudge silences on this.
      * ``no_snapshot_reason`` — set when ``snapshot`` is null so the
        UI can show "no experiment has scored the pool yet" /
        "the latest run skipped scoring because <reason>".
    """
    project_row = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = project_row.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    exp = await _latest_completed_experiment_with_snapshot(db, project_id)
    if exp is None:
        return {
            "project_id": project_id,
            "snapshot": None,
            "experiment_id": None,
            "experiment_name": None,
            "top_k_size": 0,
            "labeled_count": 0,
            "unlabeled_count": 0,
            "staleness_ratio": 0.0,
            "is_stale": False,
            "no_snapshot_reason": "no_completed_experiment_with_snapshot",
            "staleness_threshold": STALENESS_THRESHOLD,
        }

    cfg = exp.config if isinstance(exp.config, dict) else {}
    runtime = cfg.get("_runtime") if isinstance(cfg.get("_runtime"), dict) else {}
    snapshot = runtime.get("active_learning") or {}
    top_k_entries = list(snapshot.get("top_k") or []) if isinstance(snapshot, dict) else []
    top_k_size = len(top_k_entries)

    row_ids = [
        int(entry["label_row_id"])
        for entry in top_k_entries
        if isinstance(entry, dict) and isinstance(entry.get("label_row_id"), int)
    ]
    labeled_count = await _count_labeled_rows(db, row_ids)
    unlabeled_count = max(0, top_k_size - labeled_count)
    staleness_ratio = (labeled_count / top_k_size) if top_k_size else 0.0
    is_stale = top_k_size > 0 and staleness_ratio >= STALENESS_THRESHOLD

    no_snapshot_reason = None
    if top_k_size == 0:
        # Snapshot exists but is empty — carry the slice 1 skipped_reason
        # so the Data Studio card can render "scoring skipped: empty_pool".
        no_snapshot_reason = (
            str(snapshot.get("skipped_reason") or "snapshot_empty")
            if isinstance(snapshot, dict)
            else "snapshot_empty"
        )

    return {
        "project_id": project_id,
        "snapshot": snapshot if isinstance(snapshot, dict) else None,
        "experiment_id": int(exp.id),
        "experiment_name": str(exp.name or ""),
        "top_k_size": top_k_size,
        "labeled_count": labeled_count,
        "unlabeled_count": unlabeled_count,
        "staleness_ratio": round(staleness_ratio, 4),
        "is_stale": is_stale,
        "no_snapshot_reason": no_snapshot_reason,
        "staleness_threshold": STALENESS_THRESHOLD,
    }

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

# Slice 3 contract: Data Studio card renders text previews per row;
# truncate to this many chars to keep the card scannable. Adjustable
# per project surface later if a bigger card wants more text.
TEXT_PREVIEW_MAX_CHARS = 140


router = APIRouter(
    prefix="/projects/{project_id}/active-learning",
    tags=["ActiveLearning"],
)


def _truncate_preview(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= TEXT_PREVIEW_MAX_CHARS:
        return text
    return text[: TEXT_PREVIEW_MAX_CHARS - 1].rstrip() + "…"


async def _row_text_previews(
    db: AsyncSession, label_row_ids: list[int],
) -> dict[int, str | None]:
    """Build {label_row_id → truncated text preview} by joining
    label_rows + running the shared ``extract_row_text`` helper that
    the active-learning ranker already uses. Reuse keeps the preview
    consistent with what was actually scored — the user sees the same
    text the model saw."""
    if not label_row_ids:
        return {}
    from app.services.annotation.active_learning import extract_row_text

    rows = await db.execute(
        select(LabelRow.id, LabelRow.raw_payload).where(
            LabelRow.id.in_(label_row_ids),
        )
    )
    out: dict[int, str | None] = {}
    for row_id, raw_payload in rows.all():
        out[int(row_id)] = _truncate_preview(
            extract_row_text(raw_payload if isinstance(raw_payload, dict) else None)
        )
    return out


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


async def _labeled_id_set(
    db: AsyncSession, label_row_ids: list[int],
) -> set[int]:
    """Slice 3 — set of label_row_ids that have ``labeled_at`` set, so
    the Data Studio card can grey out (rather than hide) entries the
    user has already worked. We carry the labeled rows in the card
    rather than filtering them out so the staleness math (8 of 10
    labeled) is legible."""
    if not label_row_ids:
        return set()
    rows = await db.execute(
        select(LabelRow.id).where(
            LabelRow.id.in_(label_row_ids),
            LabelRow.labeled_at.is_not(None),
        )
    )
    return {int(rid) for rid in rows.scalars().all()}


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
            # Slice 3 contract — every response carries this field even
            # when null so the Data Studio card never has to undefined-check.
            "dominant_label_job_id": None,
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

    # Slice 3 — enrich each top_k entry with a text_preview and a
    # ``labeled`` flag so the Data Studio card can render uncertain
    # rows without an extra round-trip per row, and grey out the
    # ones the user has already labeled. We rebuild the snapshot's
    # top_k as a new list rather than mutating the persisted one
    # (the snapshot on Experiment.config stays slice-1 shape forever).
    previews = await _row_text_previews(db, row_ids)
    labeled_ids = await _labeled_id_set(db, row_ids)
    enriched_top_k: list[dict[str, Any]] = []
    for entry in top_k_entries:
        if not isinstance(entry, dict):
            continue
        rid = entry.get("label_row_id")
        rid_int = int(rid) if isinstance(rid, int) else None
        enriched_top_k.append({
            **entry,
            "text_preview": previews.get(rid_int) if rid_int is not None else None,
            "labeled": rid_int in labeled_ids if rid_int is not None else False,
        })
    enriched_snapshot = dict(snapshot) if isinstance(snapshot, dict) else {}
    enriched_snapshot["top_k"] = enriched_top_k

    no_snapshot_reason = None
    if top_k_size == 0:
        # Snapshot exists but is empty — carry the slice 1 skipped_reason
        # so the Data Studio card can render "scoring skipped: empty_pool".
        no_snapshot_reason = (
            str(snapshot.get("skipped_reason") or "snapshot_empty")
            if isinstance(snapshot, dict)
            else "snapshot_empty"
        )

    # Dominant label_job_id — slice 2's Coach nudge action uses this
    # for the deep-link, and the Data Studio card needs it for the
    # "Open label queue" button. Pull it from the first top_k entry
    # that carries it. (In practice every entry shares the same
    # job_id because a snapshot is scoped to one classification job.)
    dominant_job_id: int | None = None
    for entry in top_k_entries:
        if isinstance(entry, dict) and isinstance(entry.get("label_job_id"), int):
            dominant_job_id = int(entry["label_job_id"])
            break

    return {
        "project_id": project_id,
        "snapshot": enriched_snapshot if top_k_size > 0 else (snapshot if isinstance(snapshot, dict) else None),
        "experiment_id": int(exp.id),
        "experiment_name": str(exp.name or ""),
        "top_k_size": top_k_size,
        "labeled_count": labeled_count,
        "unlabeled_count": unlabeled_count,
        "staleness_ratio": round(staleness_ratio, 4),
        "is_stale": is_stale,
        "no_snapshot_reason": no_snapshot_reason,
        "staleness_threshold": STALENESS_THRESHOLD,
        "dominant_label_job_id": dominant_job_id,
    }

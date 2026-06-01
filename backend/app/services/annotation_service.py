"""Annotation foundation service (Story 1.1).

Backend half of the in-product labeling flow. Five public entry points:

- :func:`create_job` — register a labeling pass + emit
  ``annotation_job_created`` audit event.
- :func:`seed_rows_from_dataset` — read a project's dataset JSONL and
  copy up to N rows into ``label_rows`` for review.
- :func:`assign_next` — hand one unassigned row to a reviewer. Uses a
  compare-and-set update so two concurrent callers never see the same
  row.
- :func:`submit_label` — persist the reviewer's label_payload + emit
  ``annotation_label_submitted``.
- :func:`job_stats` — total / labeled / assigned / unlabeled counts +
  basic job metadata.

Audit hook follows the same best-effort pattern as
``dataset_import.service._emit_import_audit_event``: observability bugs
log and continue; they never break the data write path.

Per the [[keep-brewslm-general]] memory, this service is shape-aware
(classification / span / preference_pair) but never domain-aware — the
``label_type`` discriminator is the only place we branch, and adding a
new shape means extending KNOWN_LABEL_TYPES + a frontend renderer, not
a domain-specific service.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset
from app.models.label_job import (
    JOB_STATUS_ACTIVE,
    KNOWN_JOB_STATUSES,
    KNOWN_LABEL_TYPES,
    LABEL_TYPE_CLASSIFICATION,
    LABEL_TYPE_PREFERENCE_PAIR,
    LABEL_TYPE_SPAN,
    LabelJob,
    LabelRow,
    LabelRowReview,
)
from app.models.project import Project

# Strategies the next-row endpoint accepts. ``fifo`` is the historical
# behaviour (order by LabelRow.id ascending); ``active`` ranks
# unassigned rows by model uncertainty using the active-learning
# helper. Unknown strategies raise ``unknown_assign_strategy`` so a
# typo can't silently degrade to FIFO without the caller noticing.
ASSIGN_STRATEGY_FIFO = "fifo"
ASSIGN_STRATEGY_ACTIVE = "active"
KNOWN_ASSIGN_STRATEGIES: frozenset[str] = frozenset(
    {ASSIGN_STRATEGY_FIFO, ASSIGN_STRATEGY_ACTIVE}
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Audit hook (best-effort, mirrors dataset_import pattern)
# ---------------------------------------------------------------------------


async def _emit_annotation_audit_event(
    db: AsyncSession,
    *,
    project_id: int,
    reason_code: str,
    summary: str,
    payload: dict[str, Any],
) -> None:
    """Best-effort RunEvent emission for annotation actions.

    Failures log + swallow — the data write path must never break
    because the audit row didn't land. Pattern mirrors
    :func:`app.services.dataset_import.service._emit_import_audit_event`.
    """
    try:
        from app.models.run_event import SEVERITY_INFO, STAGE_INGESTION
        from app.services.run_event_service import emit_event

        run_id = (
            f"annotation-{int(_utcnow().timestamp() * 1000)}"
        )
        await emit_event(
            db,
            project_id=project_id,
            run_id=run_id,
            stage=STAGE_INGESTION,
            severity=SEVERITY_INFO,
            reason_code=reason_code,
            summary=summary,
            payload=payload,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[run_event] emit_failed annotation project={project_id} "
            f"reason={reason_code!r} err={exc!r}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------


async def _ensure_project(db: AsyncSession, project_id: int) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ValueError("project_not_found")
    return project


async def create_job(
    db: AsyncSession,
    *,
    project_id: int,
    name: str,
    label_type: str,
    label_schema: dict[str, Any] | None = None,
    instructions: str | None = None,
    target_rows: int | None = None,
) -> LabelJob:
    """Create a label job. Raises ``ValueError`` with stable codes for
    validation failures so the API layer can translate to 400/422."""

    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("job_name_required")
    if len(clean_name) > 120:
        raise ValueError("job_name_too_long")
    if label_type not in KNOWN_LABEL_TYPES:
        raise ValueError(f"invalid_label_type:{label_type}")

    await _ensure_project(db, project_id)

    job = LabelJob(
        project_id=project_id,
        name=clean_name,
        label_type=label_type,
        label_schema=dict(label_schema or {}),
        instructions=(instructions.strip() if instructions else None) or None,
        status=JOB_STATUS_ACTIVE,
        target_rows=target_rows,
    )
    db.add(job)
    await db.flush()

    await _emit_annotation_audit_event(
        db,
        project_id=project_id,
        reason_code="annotation_job_created",
        summary=(
            f"Annotation job {job.id!r} created "
            f"({label_type}, target={target_rows or 'unbounded'})"
        ),
        payload={
            "job_id": job.id,
            "name": clean_name,
            "label_type": label_type,
            "target_rows": target_rows,
        },
    )

    return job


async def get_job(
    db: AsyncSession, *, project_id: int, job_id: int
) -> LabelJob | None:
    """Return a job scoped to its project, or None when missing or
    cross-project."""
    result = await db.execute(
        select(LabelJob).where(
            LabelJob.id == job_id, LabelJob.project_id == project_id
        )
    )
    return result.scalar_one_or_none()


async def list_jobs(
    db: AsyncSession, *, project_id: int
) -> list[LabelJob]:
    """All jobs for a project, newest-updated first."""
    result = await db.execute(
        select(LabelJob)
        .where(LabelJob.project_id == project_id)
        .order_by(LabelJob.updated_at.desc(), LabelJob.id.desc())
    )
    return list(result.scalars().all())


async def delete_job(
    db: AsyncSession, *, project_id: int, job_id: int
) -> bool:
    """Delete a job + cascade rows. Returns False on 404."""
    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        return False
    # Manual cascade — no SQLAlchemy ORM relationship is declared so
    # delete-orphan isn't wired up. Explicit DELETE is clearer anyway.
    await db.execute(
        LabelRow.__table__.delete().where(LabelRow.job_id == job_id)
    )
    await db.delete(job)
    await db.flush()
    return True


# ---------------------------------------------------------------------------
# Row seeding
# ---------------------------------------------------------------------------


async def seed_rows_from_dataset(
    db: AsyncSession,
    *,
    project_id: int,
    job_id: int,
    dataset_id: int,
    n: int,
) -> int:
    """Read up to ``n`` rows from ``dataset_id``'s JSONL file and
    materialize them as :class:`LabelRow` work units.

    Returns the actual count seeded — capped by the file's length and
    by any malformed lines (silently skipped).
    """

    if n <= 0:
        raise ValueError("seed_n_must_be_positive")

    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise ValueError("label_job_not_found")

    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise ValueError("dataset_not_found")
    if dataset.project_id != project_id:
        raise ValueError("dataset_project_mismatch")

    file_path = Path(dataset.file_path or "")
    if not file_path.exists():
        raise ValueError("dataset_file_missing")

    added = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            source_id = row.get("id")
            source_row_id = (
                str(source_id) if source_id is not None else str(idx)
            )
            db.add(
                LabelRow(
                    job_id=job_id,
                    source_row_id=source_row_id[:128],
                    raw_payload=row,
                )
            )
            added += 1
            if added >= n:
                break

    if added:
        # Bump the job's updated_at so list_jobs orders correctly.
        job.updated_at = _utcnow()
    await db.flush()
    return added


# ---------------------------------------------------------------------------
# Assignment + submission
# ---------------------------------------------------------------------------


async def _fetch_unassigned_rows(
    db: AsyncSession, *, job_id: int
) -> list[LabelRow]:
    """Pull every unassigned, unlabeled row for a job in insertion
    order. Used as the ranker's input set in the active path."""
    rows_q = await db.execute(
        select(LabelRow)
        .where(
            LabelRow.job_id == job_id,
            LabelRow.assigned_to.is_(None),
            LabelRow.labeled_at.is_(None),
        )
        .order_by(LabelRow.id.asc())
    )
    return list(rows_q.scalars())


def _safe_score(score_fn, batch):  # noqa: ANN001 — generic callable
    """Wrap a Phase-1/2 scorer so any exception (missing weights,
    OOM, NotImplementedError stub) collapses to per-row ``None``.
    Returning all-None makes the ranker fall back to insertion
    order, so the labeler keeps moving even when the model path
    is broken."""
    try:
        return score_fn(batch)
    except Exception:
        return [None] * len(batch)


async def _candidate_row_ids_for_strategy(
    db: AsyncSession,
    *,
    job: LabelJob,
    strategy: str,
) -> list[int] | None:
    """Return a precomputed order of candidate row ids for the active
    strategy, or ``None`` to mean "use the FIFO fallback below."

    Kept in its own helper so the assign-loop stays a thin CAS layer
    around an opaque "next candidate" generator. Active-learning
    failures (no completed experiment, model load fails, all rows
    have empty text) all collapse to ``None`` — a labeler never sees
    an error from picking the active strategy on a fresh project.

    Phase 1 covered ``classification``. Phase 2 adds dispatch for
    ``span`` (top-2 margin over the token-classification head) and
    ``preference_pair`` (ensemble disagreement on alignment models).
    Both paths share the lazy-import + catch-all wrapper so a
    misconfigured project gracefully degrades to FIFO.
    """

    if strategy == ASSIGN_STRATEGY_FIFO:
        return None

    from app.services.annotation.active_learning import (
        latest_scoreable_classification_experiment,
        latest_scoreable_preference_experiment,
        latest_scoreable_span_experiment,
        rank_rows_by_uncertainty,
        score_classification_rows,
        score_preference_pair_rows,
        score_span_rows,
    )

    rows = await _fetch_unassigned_rows(db, job_id=job.id)
    if not rows:
        return []

    if job.label_type == LABEL_TYPE_CLASSIFICATION:
        experiment = await latest_scoreable_classification_experiment(
            db, project_id=job.project_id
        )
        if experiment is None:
            return None
        cfg = experiment.config or {}
        label_space = cfg.get("label_space") or cfg.get("label_space_preview")
        if not isinstance(label_space, list) or not label_space:
            return None
        model_path = cfg.get("model_path") or cfg.get("output_dir")
        if not isinstance(model_path, str) or not model_path:
            return None
        return rank_rows_by_uncertainty(
            rows,
            score_fn=lambda batch: _safe_score(
                lambda b: score_classification_rows(
                    b,
                    model_path=model_path,
                    label_space=[str(x) for x in label_space],
                ),
                batch,
            ),
        )

    if job.label_type == LABEL_TYPE_SPAN:
        experiment = await latest_scoreable_span_experiment(
            db, project_id=job.project_id
        )
        if experiment is None:
            return None
        cfg = experiment.config or {}
        model_path = cfg.get("model_path") or cfg.get("output_dir")
        if not isinstance(model_path, str) or not model_path:
            return None
        span_types = (
            (job.label_schema or {}).get("span_types") or []
        )
        if not isinstance(span_types, list):
            span_types = []
        return rank_rows_by_uncertainty(
            rows,
            score_fn=lambda batch: _safe_score(
                lambda b: score_span_rows(
                    b,
                    model_path=model_path,
                    span_types=[str(x) for x in span_types],
                ),
                batch,
            ),
        )

    if job.label_type == LABEL_TYPE_PREFERENCE_PAIR:
        experiment = await latest_scoreable_preference_experiment(
            db, project_id=job.project_id
        )
        if experiment is None:
            return None
        cfg = experiment.config or {}
        model_path = cfg.get("model_path") or cfg.get("output_dir")
        if not isinstance(model_path, str) or not model_path:
            return None
        return rank_rows_by_uncertainty(
            rows,
            score_fn=lambda batch: _safe_score(
                lambda b: score_preference_pair_rows(b, model_path=model_path),
                batch,
            ),
        )

    # Unknown / future label_type — degrade to FIFO so the toggle
    # remains usable without an exception path.
    return None


async def assign_next(
    db: AsyncSession,
    *,
    project_id: int,
    job_id: int,
    user_id: int | None,
    strategy: str = ASSIGN_STRATEGY_FIFO,
) -> LabelRow | None:
    """Atomically hand one unlabeled, unassigned row to ``user_id``.

    Returns the assigned row or ``None`` when the job has no remaining
    work. Concurrent callers will not see the same row because the
    UPDATE checks ``assigned_to IS NULL`` again — a SQLite-safe
    compare-and-set that retries on lost-update races.

    ``strategy`` chooses the ordering: ``"fifo"`` (default) hands out
    rows in insertion order; ``"active"`` ranks the unassigned tail
    by classifier-head softmax entropy via the project's most recent
    completed classification experiment, so the labeler spends
    budget on rows the model is least sure about (Epic F).
    Unknown strategies raise ``unknown_assign_strategy``.
    """

    if strategy not in KNOWN_ASSIGN_STRATEGIES:
        raise ValueError("unknown_assign_strategy")

    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise ValueError("label_job_not_found")

    preferred_ids: list[int] | None = await _candidate_row_ids_for_strategy(
        db, job=job, strategy=strategy
    )

    for _ in range(10):
        row: LabelRow | None = None
        if preferred_ids:
            # Walk the active-ranking list in order and grab the first
            # row that's still unassigned. We refresh from the DB
            # rather than trusting the cached object so a concurrent
            # CAS race in another worker doesn't get clobbered.
            while preferred_ids:
                candidate_id = preferred_ids[0]
                cand_q = await db.execute(
                    select(LabelRow).where(
                        LabelRow.id == candidate_id,
                        LabelRow.assigned_to.is_(None),
                        LabelRow.labeled_at.is_(None),
                    )
                )
                row = cand_q.scalar_one_or_none()
                if row is not None:
                    break
                preferred_ids.pop(0)

        if row is None:
            result = await db.execute(
                select(LabelRow)
                .where(
                    LabelRow.job_id == job_id,
                    LabelRow.assigned_to.is_(None),
                    LabelRow.labeled_at.is_(None),
                )
                .order_by(LabelRow.id.asc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
        if row is None:
            return None

        now = _utcnow()
        upd = await db.execute(
            update(LabelRow)
            .where(
                LabelRow.id == row.id,
                LabelRow.assigned_to.is_(None),
            )
            .values(assigned_to=user_id, assigned_at=now)
        )
        await db.flush()
        if upd.rowcount and upd.rowcount > 0:
            await db.refresh(row)
            return row
        # Lost the race — drop this id from the preferred list so the
        # next iteration walks past it.
        if preferred_ids and preferred_ids[0] == row.id:
            preferred_ids.pop(0)

    return None


async def skip_row(
    db: AsyncSession,
    *,
    project_id: int,
    job_id: int,
    row_id: int,
) -> LabelRow:
    """Clear the assignment on a row so it returns to the unlabeled
    queue. Labeler UI calls this when the reviewer presses 'esc' or the
    Skip button — the row stays in the job, just unassigned.

    Raises ``ValueError`` with stable codes:
      - ``label_job_not_found`` — job missing or cross-project.
      - ``label_row_not_found`` — row missing or doesn't belong to job.
      - ``label_row_already_labeled`` — row was already submitted;
        skipping it would silently lose the label.
    """

    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise ValueError("label_job_not_found")

    result = await db.execute(
        select(LabelRow).where(
            LabelRow.id == row_id, LabelRow.job_id == job_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("label_row_not_found")
    if row.labeled_at is not None:
        raise ValueError("label_row_already_labeled")

    row.assigned_to = None
    row.assigned_at = None
    await db.flush()
    return row


async def submit_label(
    db: AsyncSession,
    *,
    project_id: int,
    job_id: int,
    row_id: int,
    label_payload: dict[str, Any],
    reviewer_notes: str | None = None,
) -> LabelRow:
    """Persist a reviewer's label for one row and emit the audit event.

    Raises ``ValueError`` with stable codes:
      - ``label_job_not_found`` — job missing or cross-project.
      - ``label_row_not_found`` — row missing or doesn't belong to job.
      - ``label_payload_required`` — label_payload is empty / None.
    """

    if not isinstance(label_payload, dict) or not label_payload:
        raise ValueError("label_payload_required")

    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise ValueError("label_job_not_found")

    result = await db.execute(
        select(LabelRow).where(
            LabelRow.id == row_id, LabelRow.job_id == job_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("label_row_not_found")

    now = _utcnow()
    row.label_payload = dict(label_payload)
    row.labeled_at = now
    if reviewer_notes is not None:
        cleaned = reviewer_notes.strip()
        row.reviewer_notes = cleaned or None
    job.updated_at = now

    # Phase 2: also capture this submission in label_row_reviews so
    # we have history when ≥2 distinct reviewers ever labeled this
    # row. The primary ``label_payload`` field above stays the
    # authoritative single view for promotion + UI; this is a
    # side-table for agreement math only. Re-submission by the
    # same reviewer overwrites their existing review (the unique
    # constraint on (row_id, reviewer_id) makes this an upsert).
    reviewer_id = row.assigned_to
    if reviewer_id is not None:
        existing_q = await db.execute(
            select(LabelRowReview).where(
                LabelRowReview.row_id == row.id,
                LabelRowReview.reviewer_id == reviewer_id,
            )
        )
        existing = existing_q.scalar_one_or_none()
        if existing is None:
            db.add(
                LabelRowReview(
                    row_id=row.id,
                    job_id=job_id,
                    reviewer_id=reviewer_id,
                    label_payload=dict(label_payload),
                    reviewer_notes=(
                        (reviewer_notes or "").strip() or None
                        if reviewer_notes is not None
                        else None
                    ),
                )
            )
        else:
            existing.label_payload = dict(label_payload)
            if reviewer_notes is not None:
                existing.reviewer_notes = (reviewer_notes or "").strip() or None
            existing.updated_at = now
    await db.flush()

    await _emit_annotation_audit_event(
        db,
        project_id=project_id,
        reason_code="annotation_label_submitted",
        summary=(
            f"Label submitted for job {job_id} row {row_id} "
            f"({job.label_type})"
        ),
        payload={
            "job_id": job_id,
            "row_id": row_id,
            "user_id": row.assigned_to,
            "label_type": job.label_type,
            "label_payload": dict(label_payload),
        },
    )

    return row


# ---------------------------------------------------------------------------
# Stats + serialization
# ---------------------------------------------------------------------------


async def job_stats(
    db: AsyncSession, *, project_id: int, job_id: int
) -> dict[str, Any]:
    """Return total / labeled / assigned / unlabeled counts for a job."""

    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise ValueError("label_job_not_found")

    total_q = await db.execute(
        select(func.count(LabelRow.id)).where(LabelRow.job_id == job_id)
    )
    total = int(total_q.scalar() or 0)

    labeled_q = await db.execute(
        select(func.count(LabelRow.id)).where(
            LabelRow.job_id == job_id,
            LabelRow.labeled_at.is_not(None),
        )
    )
    labeled = int(labeled_q.scalar() or 0)

    assigned_q = await db.execute(
        select(func.count(LabelRow.id)).where(
            LabelRow.job_id == job_id,
            LabelRow.assigned_to.is_not(None),
            LabelRow.labeled_at.is_(None),
        )
    )
    assigned = int(assigned_q.scalar() or 0)

    # Story 1.6 — count rows already materialized into a downstream
    # training dataset so the UI can show "142 labeled · 100 promoted"
    # and hide the Promote CTA when ``labeled == promoted``.
    promoted_q = await db.execute(
        select(func.count(LabelRow.id)).where(
            LabelRow.job_id == job_id,
            LabelRow.promoted_at.is_not(None),
        )
    )
    promoted = int(promoted_q.scalar() or 0)

    unlabeled = total - labeled - assigned

    agreement = await _compute_inter_annotator_agreement(
        db, job_id=job_id, label_type=job.label_type
    )

    return {
        "job_id": job.id,
        "name": job.name,
        "label_type": job.label_type,
        "status": job.status,
        "target_rows": job.target_rows,
        "total": total,
        "labeled": labeled,
        "assigned": assigned,
        "unlabeled": unlabeled,
        "promoted": promoted,
        "inter_annotator_agreement": agreement,
    }


async def _compute_inter_annotator_agreement(
    db: AsyncSession, *, job_id: int, label_type: str
) -> dict[str, Any] | None:
    """Compute Cohen's κ / span-F1 / preference-agreement over rows
    where ≥2 distinct reviewers ever submitted a review (Epic F
    Phase 2). Returns ``None`` when no overlapping reviews exist
    yet — the UI hides the badge until data warrants it.

    Each metric is averaged across all reviewer-pairs with at
    least one row of overlap; the response also returns the
    raw pair count and overlap row count so the UI can render
    "κ=0.82 across 2 reviewers / 14 rows" without recomputing.
    """
    # Pull every (row_id, reviewer_id, label_payload) triple for
    # this job. The number of distinct reviewers ÷ rows is
    # small for typical labeling jobs, so an in-memory pivot is
    # cheaper than a join-heavy SQL aggregate.
    rows_q = await db.execute(
        select(
            LabelRowReview.row_id,
            LabelRowReview.reviewer_id,
            LabelRowReview.label_payload,
        ).where(
            LabelRowReview.job_id == job_id,
            LabelRowReview.reviewer_id.is_not(None),
        )
    )
    by_row: dict[int, dict[int, dict[str, Any]]] = {}
    reviewer_set: set[int] = set()
    for row_id, reviewer_id, payload in rows_q:
        if reviewer_id is None:
            continue
        reviewer_set.add(int(reviewer_id))
        by_row.setdefault(int(row_id), {})[int(reviewer_id)] = payload or {}

    if len(reviewer_set) < 2:
        return None

    # Build the per-pair overlap. For each unordered (A, B) pair,
    # collect rows both reviewed and project label_payloads to the
    # scalar shape the agreement metric expects.
    from app.services.annotation.active_learning import (
        cohens_kappa,
        preference_agreement,
        span_f1,
    )

    reviewers_sorted = sorted(reviewer_set)
    pair_metrics: list[float] = []
    pair_overlap_rows = 0
    pair_count = 0
    for i in range(len(reviewers_sorted)):
        for j in range(i + 1, len(reviewers_sorted)):
            a, b = reviewers_sorted[i], reviewers_sorted[j]
            overlap_rows = [
                (
                    by_row[row_id][a],
                    by_row[row_id][b],
                )
                for row_id in by_row
                if a in by_row[row_id] and b in by_row[row_id]
            ]
            if not overlap_rows:
                continue
            pair_count += 1
            pair_overlap_rows += len(overlap_rows)
            if label_type == LABEL_TYPE_CLASSIFICATION:
                labels_a = [str(r[0].get("label", "")) for r in overlap_rows]
                labels_b = [str(r[1].get("label", "")) for r in overlap_rows]
                metric = cohens_kappa(labels_a, labels_b)
                if metric is not None:
                    pair_metrics.append(metric)
            elif label_type == LABEL_TYPE_SPAN:
                # span-F1 is per-row; average across rows in the pair.
                row_f1s: list[float] = []
                for pay_a, pay_b in overlap_rows:
                    spans_a = pay_a.get("spans") or []
                    spans_b = pay_b.get("spans") or []
                    value = span_f1(spans_a, spans_b)
                    if value is not None:
                        row_f1s.append(value)
                if row_f1s:
                    pair_metrics.append(sum(row_f1s) / len(row_f1s))
            elif label_type == LABEL_TYPE_PREFERENCE_PAIR:
                picks_a = [
                    str(r[0].get("preferred", r[0].get("choice", "")))
                    for r in overlap_rows
                ]
                picks_b = [
                    str(r[1].get("preferred", r[1].get("choice", "")))
                    for r in overlap_rows
                ]
                metric = preference_agreement(picks_a, picks_b)
                if metric is not None:
                    pair_metrics.append(metric)
            else:
                continue

    if not pair_metrics:
        return None

    metric_name = {
        LABEL_TYPE_CLASSIFICATION: "cohens_kappa",
        LABEL_TYPE_SPAN: "span_f1",
        LABEL_TYPE_PREFERENCE_PAIR: "preference_agreement",
    }.get(label_type, "unknown")
    return {
        "metric": metric_name,
        "value": round(sum(pair_metrics) / len(pair_metrics), 4),
        "reviewer_count": len(reviewer_set),
        "pair_count": pair_count,
        "overlap_rows": pair_overlap_rows,
    }


def job_to_dict(job: LabelJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "project_id": job.project_id,
        "name": job.name,
        "label_type": job.label_type,
        "label_schema": dict(job.label_schema or {}),
        "instructions": job.instructions,
        "status": job.status,
        "target_rows": job.target_rows,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


def row_to_dict(row: LabelRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "job_id": row.job_id,
        "source_row_id": row.source_row_id,
        "raw_payload": dict(row.raw_payload or {}),
        "assigned_to": row.assigned_to,
        "assigned_at": (
            row.assigned_at.isoformat() if row.assigned_at else None
        ),
        "label_payload": (
            dict(row.label_payload) if row.label_payload else None
        ),
        "labeled_at": (
            row.labeled_at.isoformat() if row.labeled_at else None
        ),
        "reviewer_notes": row.reviewer_notes,
    }


def update_job_fields(
    job: LabelJob,
    *,
    name: str | None = None,
    instructions: str | None = None,
    status: str | None = None,
    target_rows: int | None = None,
) -> LabelJob:
    """In-place update of mutable job fields. Validates ``status`` against
    :data:`KNOWN_JOB_STATUSES`."""
    if name is not None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("job_name_required")
        if len(clean_name) > 120:
            raise ValueError("job_name_too_long")
        job.name = clean_name
    if instructions is not None:
        cleaned = instructions.strip()
        job.instructions = cleaned or None
    if status is not None:
        if status not in KNOWN_JOB_STATUSES:
            raise ValueError(f"invalid_job_status:{status}")
        job.status = status
    if target_rows is not None:
        job.target_rows = max(0, int(target_rows))
    job.updated_at = _utcnow()
    return job

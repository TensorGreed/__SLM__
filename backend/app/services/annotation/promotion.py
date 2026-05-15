"""Annotation → training dataset bridge (Story 1.6).

submit_label in :mod:`app.services.annotation_service` persists the
reviewer's label into the ``label_rows`` table. Without this module
those rows sit there forever and never reach the trainer — Stories
1.1–1.3 ship a fully-functional UI that's a dead-end for actual
training. This module is the missing edge: it materializes labeled
rows into the project's synthetic dataset (classification + span) or
alignment dataset (preference pair), with provenance preserved and
idempotency guarded by the ``promoted_at`` column.

Public entry point:

    promote_labeled_rows(db, *, project_id, job_id, target_dataset_type)

Returns a dict shaped::

    {
        "promoted_count": int,
        "skipped_already_promoted": int,
        "skipped_unlabeled": int,
        "target_dataset_id": int | None,
        "target_dataset_type": str,
        "label_type": str,
        "written_path": str | None,
    }

The function is idempotent: re-running it on a job whose rows are
already promoted is safe and reports the skips. It does NOT delete the
``label_rows`` records — they remain as provenance + as input to
future re-promotion runs once we ship Story 1.6+ "promote with edits".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.label_job import (
    LABEL_TYPE_CLASSIFICATION,
    LABEL_TYPE_PREFERENCE_PAIR,
    LABEL_TYPE_SPAN,
    LabelJob,
    LabelRow,
)


# Map label_type → target dataset type. The classification + span
# paths both target SYNTHETIC since their canonical shape is
# ``{question, answer}``. preference_pair targets the alignment
# preference file, which sits outside the Dataset registry but still
# gets a Dataset row attached for provenance.
_DEFAULT_TARGET_BY_LABEL_TYPE: dict[str, DatasetType] = {
    LABEL_TYPE_CLASSIFICATION: DatasetType.SYNTHETIC,
    LABEL_TYPE_SPAN: DatasetType.SYNTHETIC,
    LABEL_TYPE_PREFERENCE_PAIR: DatasetType.SYNTHETIC,
}

# Allowed override values for the API. GOLD_DEV lets the operator
# treat an annotation pass as gold-set authoring rather than synthetic
# augmentation (useful for the "label 100 rows by hand → make them
# eval ground truth" workflow).
_ALLOWED_TARGET_TYPES: frozenset[DatasetType] = frozenset(
    {DatasetType.SYNTHETIC, DatasetType.GOLD_DEV}
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _pick_source_text(raw_payload: dict[str, Any]) -> str:
    """Same heuristic the annotation UI uses for display — try the
    common text-bearing keys, fall back to JSON pretty-print so we
    always preserve *something* the trainer can attend to."""
    for key in (
        "text",
        "prompt",
        "content",
        "body",
        "question",
        "instruction",
        "input",
        "source",
    ):
        value = raw_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    try:
        return json.dumps(raw_payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw_payload)


# ─────────────────────────────────────────────────────────────────────
# Per-label_type renderers — each takes a (job, row) pair and produces
# one JSONL-ready dict for the target file.
# ─────────────────────────────────────────────────────────────────────


def _render_classification(job: LabelJob, row: LabelRow) -> dict[str, Any]:
    label_payload = row.label_payload or {}
    label_value = label_payload.get("label")
    text = _pick_source_text(row.raw_payload or {})
    return {
        "question": text,
        "answer": json.dumps({"label": label_value}, ensure_ascii=False),
        "label": label_value,
        "source": "annotation_job",
        "annotation_job_id": job.id,
        "original_row_id": row.id,
        "reviewer_user_id": row.assigned_to,
    }


def _render_span(job: LabelJob, row: LabelRow) -> dict[str, Any]:
    label_payload = row.label_payload or {}
    spans = label_payload.get("spans") or label_payload.get("entities") or []
    # Normalize: SpanLabeler emits ``{start, end, type}``; existing
    # eval rubric and StructuredExtractionHandler expect ``text`` too
    # for span_set scoring. Materialize the missing text slice from
    # the source string when the labeler didn't include it.
    text = _pick_source_text(row.raw_payload or {})
    normalized_spans: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        span_type = span.get("type") or span.get("label")
        span_text = span.get("text")
        if span_text is None and isinstance(start, int) and isinstance(end, int):
            if 0 <= start <= end <= len(text):
                span_text = text[start:end]
        normalized_spans.append(
            {
                "type": span_type,
                "start": start,
                "end": end,
                "text": span_text,
            }
        )
    return {
        "question": text,
        "answer": json.dumps(
            {"entities": normalized_spans}, ensure_ascii=False
        ),
        "text": text,
        "entities": normalized_spans,
        "source": "annotation_job",
        "annotation_job_id": job.id,
        "original_row_id": row.id,
        "reviewer_user_id": row.assigned_to,
    }


def _render_preference_pair(
    job: LabelJob, row: LabelRow
) -> dict[str, Any] | None:
    """Preference-pair rows land in the alignment dataset, not the
    synthetic dataset, because DPO/ORPO trainers expect a different
    shape. Returns ``None`` for rows that ended in a tie or both-bad
    (not actionable as preference training data)."""
    rp = row.raw_payload or {}
    lp = row.label_payload or {}
    chosen = lp.get("chosen")
    if chosen not in {"A", "B"}:
        # tie / both_bad / unknown → skip; not a valid preference pair.
        return None
    a = rp.get("completion_a") or ""
    b = rp.get("completion_b") or ""
    return {
        "prompt": rp.get("prompt") or _pick_source_text(rp),
        "chosen": a if chosen == "A" else b,
        "rejected": b if chosen == "A" else a,
        "source": "annotation_job",
        "annotation_job_id": job.id,
        "original_row_id": row.id,
        "reviewer_user_id": row.assigned_to,
    }


_RENDERER_BY_LABEL_TYPE = {
    LABEL_TYPE_CLASSIFICATION: _render_classification,
    LABEL_TYPE_SPAN: _render_span,
    LABEL_TYPE_PREFERENCE_PAIR: _render_preference_pair,
}


# ─────────────────────────────────────────────────────────────────────
# Audit hook
# ─────────────────────────────────────────────────────────────────────


async def _emit_promotion_audit_event(
    db: AsyncSession,
    *,
    project_id: int,
    payload: dict[str, Any],
) -> None:
    """Best-effort RunEvent emission; failures log + swallow so a
    flaky run-events writer can't sink the data path. Mirrors the
    pattern used by :func:`annotation_service._emit_annotation_audit_event`."""
    try:
        from app.models.run_event import SEVERITY_INFO, STAGE_INGESTION
        from app.services.run_event_service import emit_event

        run_id = f"annotation-promote-{int(_utcnow().timestamp() * 1000)}"
        await emit_event(
            db,
            project_id=project_id,
            run_id=run_id,
            stage=STAGE_INGESTION,
            severity=SEVERITY_INFO,
            reason_code="annotation_rows_promoted",
            summary=(
                f"Promoted {payload.get('promoted_count', 0)} labeled "
                f"row(s) into {payload.get('target_dataset_type', 'synthetic')}"
            ),
            payload=payload,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[run_event] emit_failed annotation_promote project={project_id} "
            f"err={exc!r}",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────
# Target dataset resolution
# ─────────────────────────────────────────────────────────────────────


def _synthetic_dir(project_id: int) -> Path:
    # Local copy of the synthetic-service helper so this module
    # doesn't need to import the heavyweight synthetic_service for one
    # path constant.
    from app.config import settings

    d = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _alignment_dir(project_id: int) -> Path:
    from app.config import settings

    d = settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "alignment"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gold_dir(project_id: int) -> Path:
    from app.config import settings

    d = settings.DATA_DIR / "projects" / str(project_id) / "gold"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _resolve_target_dataset(
    db: AsyncSession,
    *,
    project_id: int,
    label_type: str,
    target_dataset_type: DatasetType,
) -> tuple[Dataset, Path]:
    """Find or create the Dataset row + return its on-disk JSONL path.

    Classification + span → SYNTHETIC dataset's ``synthetic.jsonl``,
    OR the GOLD_DEV dataset's ``gold_dev.jsonl`` when caller opts in.
    Preference pair → an alignment-specific ``preference_pairs.jsonl``
    paired with a SYNTHETIC dataset row (we don't have an
    ``ALIGNMENT`` dataset_type today; piggybacking on SYNTHETIC keeps
    the Dataset table honest for provenance lookup).
    """
    if label_type == LABEL_TYPE_PREFERENCE_PAIR:
        file_path = _alignment_dir(project_id) / "preference_pairs.jsonl"
        ds_type = DatasetType.SYNTHETIC  # piggyback for provenance
        ds_name = f"Annotation alignment pairs"
    elif target_dataset_type == DatasetType.GOLD_DEV:
        file_path = _gold_dir(project_id) / "gold_dev.jsonl"
        ds_type = DatasetType.GOLD_DEV
        ds_name = "Gold (dev)"
    else:
        file_path = _synthetic_dir(project_id) / "synthetic.jsonl"
        ds_type = DatasetType.SYNTHETIC
        ds_name = "Synthetic Dataset"

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == ds_type,
        )
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        dataset = Dataset(
            project_id=project_id,
            name=ds_name,
            dataset_type=ds_type,
            file_path=str(file_path),
        )
        db.add(dataset)
        await db.flush()
    elif not dataset.file_path:
        dataset.file_path = str(file_path)
        await db.flush()

    return dataset, file_path


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


async def promote_labeled_rows(
    db: AsyncSession,
    *,
    project_id: int,
    job_id: int,
    target_dataset_type: DatasetType = DatasetType.SYNTHETIC,
) -> dict[str, Any]:
    """Materialize every labeled, unpromoted row in ``job_id`` into
    the project's target dataset JSONL.

    Idempotent: rows with ``promoted_at`` already set are reported as
    ``skipped_already_promoted``. Rows that are still unlabeled (no
    ``labeled_at``) are reported as ``skipped_unlabeled``.

    Raises ``ValueError`` with stable codes the API translates:
      - ``label_job_not_found``           — job missing or cross-project.
      - ``invalid_target_dataset_type``   — not in ``{synthetic, gold_dev}``.
    """
    if target_dataset_type not in _ALLOWED_TARGET_TYPES:
        raise ValueError(
            f"invalid_target_dataset_type:{target_dataset_type.value}"
        )

    job_result = await db.execute(
        select(LabelJob).where(
            LabelJob.id == job_id, LabelJob.project_id == project_id
        )
    )
    job = job_result.scalar_one_or_none()
    if job is None:
        raise ValueError("label_job_not_found")

    renderer = _RENDERER_BY_LABEL_TYPE.get(job.label_type)
    if renderer is None:
        raise ValueError(f"invalid_label_type:{job.label_type}")

    rows_result = await db.execute(
        select(LabelRow)
        .where(LabelRow.job_id == job_id)
        .order_by(LabelRow.id.asc())
    )
    rows = list(rows_result.scalars().all())

    dataset, file_path = await _resolve_target_dataset(
        db,
        project_id=project_id,
        label_type=job.label_type,
        target_dataset_type=target_dataset_type,
    )

    promoted_count = 0
    skipped_already_promoted = 0
    skipped_unlabeled = 0
    skipped_unrenderable = 0
    now = _utcnow()

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row.labeled_at is None:
                skipped_unlabeled += 1
                continue
            if row.promoted_at is not None:
                skipped_already_promoted += 1
                continue
            entry = renderer(job, row)
            if entry is None:
                # e.g. preference-pair row that ended in a tie / both-bad
                skipped_unrenderable += 1
                # Still mark promoted so we don't reconsider on every
                # subsequent call.
                row.promoted_at = now
                row.promoted_to_dataset_id = dataset.id
                continue
            entry.setdefault("id", (dataset.record_count or 0) + promoted_count + 1)
            entry.setdefault("status", "accepted")
            entry.setdefault("promoted_at", now.isoformat())
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            row.promoted_at = now
            row.promoted_to_dataset_id = dataset.id
            promoted_count += 1

    dataset.record_count = (dataset.record_count or 0) + promoted_count
    dataset.file_path = str(file_path)
    await db.flush()

    audit_payload = {
        "job_id": job.id,
        "label_type": job.label_type,
        "promoted_count": promoted_count,
        "skipped_already_promoted": skipped_already_promoted,
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_unrenderable": skipped_unrenderable,
        "target_dataset_id": dataset.id,
        "target_dataset_type": dataset.dataset_type.value,
        "written_path": str(file_path),
    }
    await _emit_promotion_audit_event(
        db, project_id=project_id, payload=audit_payload
    )

    return {
        "promoted_count": promoted_count,
        "skipped_already_promoted": skipped_already_promoted,
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_unrenderable": skipped_unrenderable,
        "target_dataset_id": dataset.id,
        "target_dataset_type": dataset.dataset_type.value,
        "label_type": job.label_type,
        "written_path": str(file_path),
    }


__all__ = ["promote_labeled_rows"]

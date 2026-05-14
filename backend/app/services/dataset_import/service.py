"""Orchestrator: source → mapper → save.

Two public entry points:

- :func:`preview_import` — runs the source through the mapper for a
  capped number of rows and returns transformed + rejected samples.
  Doesn't touch the dataset. Used by the ``--dry-run`` CLI path and
  the UI's Preview step.
- :func:`run_import` — same pipeline, persists accepted rows to the
  project's synthetic dataset, returns counts + warnings.

Both honor the per-row accountability contract: every raw row turns
into either a TransformedRow or a RejectedRow with a stable reason
code. Bulk-drop happens at the call site (``drop_reasons`` filter
removes whole rejection categories before save) per the
[[rejected-rows-selectable]] memory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.dataset_import.protocols import (
    ImportContext,
    ImportResult,
    RejectedRow,
    TransformedRow,
)
from app.services.dataset_import.registry import (
    resolve_mapper,
    resolve_source,
    split_locator,
)


def _build_context(
    *,
    project_id: int,
    project_task_profile: str | None,
    locator: str,
    mapper_id: str,
    field_map: dict[str, Any] | None,
    limit: int | None,
) -> ImportContext:
    source_id, _ = split_locator(locator)
    return ImportContext(
        project_id=project_id,
        project_task_profile=project_task_profile,
        source_id=source_id,
        mapper_id=mapper_id,
        locator=locator,
        field_map=dict(field_map or {}),
        limit=limit,
    )


def _drain_pipeline(
    *,
    ctx: ImportContext,
    drop_reasons: set[str] | None,
    sample_cap: int | None,
) -> tuple[list[TransformedRow], list[RejectedRow], dict[str, int], list[str]]:
    """Run source → mapper. Returns (accepted, rejected, counts, warnings).

    ``drop_reasons`` filters out rejected rows by reason code BEFORE
    they go into the returned list — counts still reflect the full
    set so callers can show "you dropped N malformed rows" totals.
    ``sample_cap`` limits how many TransformedRow objects we hold in
    memory at once; rejected rows are always kept (they're cheap and
    the user wants to see them for the bulk-drop UX).
    """

    source = resolve_source(ctx.source_id)
    mapper = resolve_mapper(ctx.mapper_id)
    _, source_rest = split_locator(ctx.locator)

    accepted: list[TransformedRow] = []
    rejected: list[RejectedRow] = []
    rejection_counts: dict[str, int] = {}
    warnings: list[str] = []

    drop_reasons = drop_reasons or set()
    raw_iter: Iterable = source.load(source_rest, limit=ctx.limit)

    for item in mapper.transform(raw_iter, ctx.field_map, ctx=ctx):
        if isinstance(item, RejectedRow):
            rejection_counts[item.reason] = rejection_counts.get(item.reason, 0) + 1
            if item.reason in drop_reasons:
                continue  # honored bulk-drop filter, don't surface
            rejected.append(item)
            continue
        if isinstance(item, TransformedRow):
            if sample_cap is None or len(accepted) < sample_cap:
                accepted.append(item)
            warnings.extend(item.warnings)
            continue

    return accepted, rejected, rejection_counts, warnings


def preview_import(
    *,
    project_id: int,
    project_task_profile: str | None,
    locator: str,
    mapper_id: str,
    field_map: dict[str, Any] | None,
    sample_cap: int = 5,
    limit: int | None = None,
    drop_reasons: set[str] | None = None,
) -> ImportResult:
    """Dry-run the pipeline. Returns the first ``sample_cap``
    transformed rows + all rejected rows so the UI can render
    the per-reason breakdown.
    """

    ctx = _build_context(
        project_id=project_id,
        project_task_profile=project_task_profile,
        locator=locator,
        mapper_id=mapper_id,
        field_map=field_map,
        limit=limit if limit is not None else max(sample_cap * 20, 50),
    )
    accepted, rejected, rejection_counts, warnings = _drain_pipeline(
        ctx=ctx, drop_reasons=drop_reasons, sample_cap=sample_cap
    )
    mapper = resolve_mapper(mapper_id)
    return ImportResult(
        accepted_rows=accepted,
        rejected_rows=rejected,
        rejection_counts=rejection_counts,
        accepted_count=len(accepted),
        rejected_count=sum(rejection_counts.values()),
        source_id=ctx.source_id,
        mapper_id=mapper_id,
        target_task_profile=mapper.declared_target(),
        locator=locator,
        dry_run=True,
        warnings=warnings,
    )


async def run_import(
    db: AsyncSession,
    *,
    project_id: int,
    project_task_profile: str | None,
    locator: str,
    mapper_id: str,
    field_map: dict[str, Any] | None,
    limit: int | None = None,
    drop_reasons: set[str] | None = None,
) -> ImportResult:
    """Persist transformed rows to the project's synthetic dataset.

    Reuses the synthetic-service write path so imported rows are
    first-class members of the project's synthetic dataset alongside
    teacher-LLM-generated rows. Same JSONL file, same accepted/
    rejected flag convention.
    """

    from app.services.synthetic_service import (
        _synthetic_dir,
        get_or_create_synthetic_dataset,
    )

    ctx = _build_context(
        project_id=project_id,
        project_task_profile=project_task_profile,
        locator=locator,
        mapper_id=mapper_id,
        field_map=field_map,
        limit=limit,
    )
    # Full pipeline — no sample cap on accepted rows for a real run.
    accepted, rejected, rejection_counts, warnings = _drain_pipeline(
        ctx=ctx, drop_reasons=drop_reasons, sample_cap=None
    )
    mapper = resolve_mapper(mapper_id)

    ds = await get_or_create_synthetic_dataset(db, project_id)
    syn_dir = _synthetic_dir(project_id)
    syn_dir.mkdir(parents=True, exist_ok=True)
    file_path = syn_dir / "synthetic.jsonl"

    now_iso = datetime.now(timezone.utc).isoformat()
    written = 0
    with file_path.open("a", encoding="utf-8") as handle:
        for row in accepted:
            entry = {
                "id": ds.record_count + written + 1,
                **row.payload,
                "source": "dataset_import",
                "import_locator": locator,
                "import_mapper": mapper_id,
                "row_key": row.row_key,
                "imported_at": now_iso,
                "status": "accepted",
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    ds.record_count += written
    ds.file_path = str(file_path)
    await db.flush()

    return ImportResult(
        accepted_rows=accepted[:5],  # cap inline sample in API response
        rejected_rows=rejected[:50],  # ditto for reject samples
        rejection_counts=rejection_counts,
        accepted_count=written,
        rejected_count=sum(rejection_counts.values()),
        source_id=ctx.source_id,
        mapper_id=mapper_id,
        target_task_profile=mapper.declared_target(),
        locator=locator,
        written_path=str(file_path),
        dry_run=False,
        warnings=warnings,
    )


def result_to_dict(result: ImportResult) -> dict[str, Any]:
    """Serialize an ``ImportResult`` for API responses."""

    return {
        "accepted_count": result.accepted_count,
        "rejected_count": result.rejected_count,
        "source_id": result.source_id,
        "mapper_id": result.mapper_id,
        "target_task_profile": result.target_task_profile,
        "locator": result.locator,
        "written_path": result.written_path,
        "dry_run": result.dry_run,
        "rejection_counts": result.rejection_counts,
        "warnings": result.warnings,
        "accepted_sample": [
            {"payload": row.payload, "row_key": row.row_key, "warnings": row.warnings}
            for row in result.accepted_rows
        ],
        "rejected_sample": [
            {
                "reason": row.reason,
                "detail": row.detail,
                "row_index": row.row_index,
                "raw_row": row.raw_row,
            }
            for row in result.rejected_rows
        ],
    }

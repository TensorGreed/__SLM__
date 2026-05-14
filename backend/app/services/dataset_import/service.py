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

from app.services.dataset_import.introspector import (
    CONFIDENCE_HIGH,
    detect_shape,
    hypothesis_to_dict,
    proposal_to_dict,
    signature_to_dict,
    sniff_columns,
)
from app.services.dataset_import.protocols import (
    ImportContext,
    ImportResult,
    ProposedMapping,
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


async def _emit_import_audit_event(
    db: AsyncSession,
    *,
    project_id: int,
    severity: str,
    reason_code: str,
    summary: str,
    payload: dict[str, Any],
    use_fresh_session: bool = False,
) -> None:
    """Best-effort RunEvent emission for the dataset-import audit log.

    Failures here are logged and swallowed — observability bugs must
    never break the data write path. The pattern mirrors
    ``deployment_version_service._emit_deployment_event``.

    ``use_fresh_session`` is set on the failure path: the parent
    transaction is about to be rolled back by the exception we just
    caught, so the audit row has to live in its own short-lived
    session that commits independently. Without this flag, the
    failure event is created but never persisted.
    """
    try:
        from app.database import async_session_factory
        from app.models.run_event import STAGE_INGESTION
        from app.services.run_event_service import emit_event

        run_id = (
            f"dataset-import-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        )
        if use_fresh_session:
            async with async_session_factory() as fresh:
                await emit_event(
                    fresh,
                    project_id=project_id,
                    run_id=run_id,
                    stage=STAGE_INGESTION,
                    severity=severity,
                    reason_code=reason_code,
                    summary=summary,
                    payload=payload,
                )
                await fresh.commit()
        else:
            await emit_event(
                db,
                project_id=project_id,
                run_id=run_id,
                stage=STAGE_INGESTION,
                severity=severity,
                reason_code=reason_code,
                summary=summary,
                payload=payload,
            )
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[run_event] emit_failed dataset_import project={project_id} "
            f"err={exc!r}",
            flush=True,
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
    config_id: int | None = None,
) -> ImportResult:
    """Persist transformed rows to the project's synthetic dataset.

    Reuses the synthetic-service write path so imported rows are
    first-class members of the project's synthetic dataset alongside
    teacher-LLM-generated rows. Same JSONL file, same accepted/
    rejected flag convention.

    Emits a RunEvent on the success path (severity=info, reason_code
    ``dataset_import_run``) and on the failure path (severity=error,
    reason_code ``dataset_import_failed``). The audit row carries the
    source, locator, mapper, row counts, written_path, and — when the
    run was launched from a saved mapping — the ``config_id`` link.
    """

    from app.models.reason_codes import (
        DATASET_IMPORT_FAILED,
        DATASET_IMPORT_RUN,
    )
    from app.models.run_event import SEVERITY_ERROR, SEVERITY_INFO
    from app.services.synthetic_service import (
        _synthetic_dir,
        get_or_create_synthetic_dataset,
    )

    try:
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
    except Exception as exc:
        await _emit_import_audit_event(
            db,
            project_id=project_id,
            severity=SEVERITY_ERROR,
            reason_code=DATASET_IMPORT_FAILED,
            summary=f"Dataset import failed: {type(exc).__name__}",
            payload={
                "source_id": locator.split(":", 1)[0] if ":" in locator else "",
                "locator": locator,
                "mapper_id": mapper_id,
                "config_id": config_id,
                "error": str(exc)[:500],
            },
            # The exception below is about to roll back the caller's
            # transaction — the audit row has to live in its own
            # short-lived session so it survives.
            use_fresh_session=True,
        )
        raise

    await _emit_import_audit_event(
        db,
        project_id=project_id,
        severity=SEVERITY_INFO,
        reason_code=DATASET_IMPORT_RUN,
        summary=(
            f"Imported {written} row(s) via {mapper_id} from "
            f"{ctx.source_id} ({locator})"
        ),
        payload={
            "source_id": ctx.source_id,
            "locator": locator,
            "mapper_id": mapper_id,
            "target_task_profile": mapper.declared_target(),
            "accepted_count": written,
            "rejected_count": sum(rejection_counts.values()),
            "rejection_counts": rejection_counts,
            "written_path": str(file_path),
            "config_id": config_id,
        },
    )

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


async def introspect_locator(
    locator: str,
    *,
    sample_size: int = 20,
    llm_assist: bool = False,
) -> dict[str, Any]:
    """Sniff columns + propose a mapping for the dataset behind ``locator``.

    Looks up the registered source by the locator's prefix, asks it for
    a sample (via ``describe``), then runs the deterministic
    column-content sniffer + shape detector to produce a ranked list of
    mapping hypotheses. The top hypothesis is materialized as a
    ``ProposedMapping`` so callers can pass it straight to
    ``preview_import`` / ``run_import`` after the user confirms.

    ``llm_assist=True`` (Phase H) additionally asks the project's
    teacher model for a mapping suggestion. The LLM proposal is
    merged into the ranked hypothesis list and competes with the
    deterministic proposals on confidence — never overrides them.
    Disabled by default + falls through silently when the teacher
    isn't reachable, so callers can opt in without checking config.

    Per the architectural rule: this NEVER picks for the user — it
    just emits proposals with confidence + rationale. The CLI / UI
    must enforce the ``CONFIDENCE_HIGH`` gate before auto-running.
    """

    source_id, source_rest = split_locator(locator)
    source = resolve_source(source_id)
    description = source.describe(source_rest)
    sample_rows = list(description.get("sample_rows") or [])[:sample_size]

    signatures = sniff_columns(sample_rows)
    hypotheses = detect_shape(signatures, sample_rows)
    hypothesis_dicts = [hypothesis_to_dict(h) for h in hypotheses]

    llm_proposal: ProposedMapping | None = None
    if llm_assist:
        from app.services.dataset_import.llm_assist import (
            llm_assisted_proposal,
        )

        llm_proposal = await llm_assisted_proposal(
            columns=description.get("columns") or list(signatures.keys()),
            sample_rows=sample_rows,
        )
        if llm_proposal is not None:
            # Append the LLM proposal to the hypothesis ranking so
            # callers see it alongside the deterministic ones. The
            # frontend can highlight LLM-assisted entries via the
            # "proposal-source: llm-assist" warning.
            hypothesis_dicts.append(
                {
                    "mapper_id": llm_proposal.mapper_id,
                    "target_task_profile": llm_proposal.target_task_profile,
                    "field_map": llm_proposal.field_map,
                    "confidence": round(llm_proposal.confidence, 4),
                    "rationale": llm_proposal.rationale,
                    "warnings": list(llm_proposal.warnings),
                }
            )
            hypothesis_dicts.sort(key=lambda h: -h["confidence"])

    proposal: ProposedMapping | None = None
    if hypothesis_dicts:
        top = hypothesis_dicts[0]
        proposal = ProposedMapping(
            target_task_profile=top["target_task_profile"],
            mapper_id=top["mapper_id"],
            field_map=top["field_map"],
            confidence=top["confidence"],
            rationale=top["rationale"],
            warnings=list(top["warnings"]),
        )

    return {
        "source_id": source_id,
        "locator": locator,
        "resolved_path": description.get("resolved_path"),
        "approximate_total_rows": description.get("approximate_total_rows"),
        "columns": description.get("columns") or list(signatures.keys()),
        "sample_rows": sample_rows,
        "column_signatures": [
            signature_to_dict(sig) for sig in signatures.values()
        ],
        "hypotheses": hypothesis_dicts,
        "proposal": proposal_to_dict(proposal) if proposal else None,
        "confidence_threshold": CONFIDENCE_HIGH,
        "llm_assist_used": llm_proposal is not None,
    }


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

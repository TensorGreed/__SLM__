"""Named, versioned eval sets + prompt-format variants (hardening Phase 3; ``eval_set_service``).

Every task shape: rows are stored as given. Mounted at ``/api/projects/{project_id}/eval-sets``. An eval
runs on a set through the normal ``POST /evaluation/run-heldout`` with ``dataset_name =
eval_set:<slug>[@vN]`` (+ ``prompt_variant``).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.exceptions import SLMError
from app.models.project import Project
from app.services import eval_set_service as ess
from app.services.gold_import_service import (
    PREVIEW_CHARS,
    GoldFileError,
    group_rejects,
    parse_gold_file,
    schema_size_warnings,
)

REASON_LEAKS_TRAINING = ess.REASON_LEAKS_TRAINING

router = APIRouter(prefix="/projects/{project_id}/eval-sets", tags=["Eval sets"])


class SourceBody(BaseModel):
    kind: str = Field(..., description="eval_set | split")
    slug: str | None = None
    version: int | None = None
    split: str | None = None


class VariantBody(BaseModel):
    name: str
    system: str
    text: str | None = None

    def spec(self) -> dict[str, Any]:
        out: dict[str, Any] = {"system": self.system}
        if self.text is not None:
            out["text"] = self.text
        return out


class FromSourceBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=ess.MAX_NAME_CHARS)
    description: str = ""
    source: SourceBody
    # Optional: define this prompt-format variant on the new set in the same call (a built-in set's
    # rows with another system message, in one click: built-in sets themselves take no variants).
    variant: VariantBody | None = None


class VersionFromSourceBody(BaseModel):
    source: SourceBody


class RowsDeleteBody(BaseModel):
    positions: list[int]
    version: int | None = None


def _error(exc: ess.EvalSetError) -> SLMError:
    return SLMError(status_code=exc.status, error_code=exc.code, stage="evaluation", message=str(exc))


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


async def _get(db: AsyncSession, project_id: int, slug: str, *, fresh: bool = False):
    """The set. Writers call this INSIDE ``ess.write_lock`` with ``fresh=True``: a copy read before the
    lock may be stale (another write landed in between)."""
    try:
        return await ess.get_eval_set(db, project_id, slug, fresh=fresh)
    except ess.EvalSetError as exc:
        raise _error(exc)


def _lines(raw: str | None, field: str) -> set[int]:
    if not isinstance(raw, str) or not raw:  # unset (also a direct call's Form default)
        return set()
    try:
        return {int(x) for x in json.loads(raw)}
    except (json.JSONDecodeError, TypeError, ValueError):
        raise HTTPException(400, f"{field} must be a JSON list of line numbers")


async def _import_rows(
    db: AsyncSession, project_id: int, file: UploadFile, field_mapping: str | None, drop_lines: str | None,
    drop_rejected: bool, keep_leak_lines: str | None = None,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any], dict[str, Any] | None]:
    """The Phase 1 answer-key import, reused, plus a leakage scan of the usable rows against the
    project's training data (``ess.leak_scan``, the data-health train/test matcher): ``(rows, dropped
    line numbers, leak summary, kept-despite-leak record)``. 422 ``EVAL_SET_IMPORT_REJECTED_ROWS`` groups
    the unusable and the leaking rows by reason until each is dropped (``drop_lines``; ``drop_rejected``
    drops them all) or -- leak-flagged rows only -- explicitly kept (``keep_leak_lines``: they are scored
    as seen data, and the version records their count + ids, shown on every result)."""
    mapping = None
    if field_mapping:
        try:
            parsed = json.loads(field_mapping)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"field_mapping is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            raise HTTPException(400, "field_mapping must be a JSON object")
        mapping = {str(k): str(v) for k, v in parsed.items()}
    dropped = _lines(drop_lines, "drop_lines")
    keep_leaks = _lines(keep_leak_lines, "keep_leak_lines") - dropped  # a dropped line stays dropped
    from app.api import dataset_import

    content = await file.read(dataset_import.MAX_UPLOAD_BYTES + 1)
    if len(content) > dataset_import.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file is too large.")
    try:
        line_rows, rejects = parse_gold_file(file.filename or "", content, mapping)
    except GoldFileError as exc:
        raise HTTPException(400, str(exc))
    # A line the caller dropped is left out whatever it is (unusable, or a usable row that leaks).
    kept = [(line, row) for line, row in line_rows if line not in dropped]
    previews = {line: json.dumps(row, ensure_ascii=False)[:PREVIEW_CHARS] for line, row in kept}
    hits = await ess.leak_scan(db, project_id, [row for _line, row in kept])
    leak_rejects = [
        (kept[i][0], REASON_LEAKS_TRAINING, f"{previews[kept[i][0]]} | matches a training row ({kind}): {match}")
        for i, kind, match in hits
    ]
    leak_lines = {line for line, _r, _p in leak_rejects}
    not_flagged = sorted(keep_leaks - leak_lines)
    if not_flagged:
        raise SLMError(
            status_code=400,
            error_code="EVAL_SET_KEEP_NOT_LEAK_FLAGGED",
            stage="evaluation",
            message=f"Only rows flagged as leaking training data can be kept; line(s) {not_flagged[:10]} are not.",
            actionable_fix="Drop unusable rows instead, or fix them in the file.",
            metadata={"lines": not_flagged},
        )
    # Kept leak-flagged rows are no longer rejects (an explicit, recorded choice).
    leak_rejects = [r for r in leak_rejects if r[0] not in keep_leaks]
    all_rejects = rejects + leak_rejects
    remaining = [r for r in all_rejects if r[0] not in dropped]
    if remaining and not drop_rejected:
        raise SLMError(
            status_code=422,
            error_code="EVAL_SET_IMPORT_REJECTED_ROWS",
            stage="evaluation",
            message=(
                f"{len(remaining)} row(s) in {file.filename} cannot be used as eval rows"
                + (f" ({len([r for r in remaining if r[1] == REASON_LEAKS_TRAINING])} leak training data)"
                   if any(r[1] == REASON_LEAKS_TRAINING for r in remaining) else "")
                + "."
            ),
            actionable_fix="Select the rows to drop (grouped by reason) and import the rest, or fix the file and upload it again.",
            metadata={
                "rejected": group_rejects(remaining),
                "row_counts": {
                    "usable": len(kept) - len({line for line, _r, _p in leak_rejects if line not in dropped}),
                    "rejected": len(remaining),
                    "already_dropped": len(dropped),
                },
            },
        )
    drop_all = {line for line, _r, _p in all_rejects} | {line for line, _row in line_rows if line in dropped}
    rows = [row for line, row in kept if line not in drop_all]
    leak = {"scanned": len(kept), "leaking_rows_dropped": len({line for line, _r, _p in leak_rejects})}
    final_lines = [line for line, _row in kept if line not in drop_all]
    kept_leak = ess.kept_leak_record(
        rows, [i for i, line in enumerate(final_lines) if line in keep_leaks], sorted(keep_leaks & set(final_lines)),
    )
    if kept_leak:
        leak["leaking_rows_kept"] = kept_leak["count"]
    if not rows:
        raise SLMError(
            status_code=422,
            error_code="EVAL_SET_NO_ROWS",
            stage="evaluation",
            message=f"No usable rows in {file.filename}.",
            actionable_fix="Upload a file with one JSON object per line (or a JSON array / CSV with a header row).",
            metadata={"rejected": group_rejects(all_rejects)},
        )
    warnings = schema_size_warnings([(line, row) for line, row in kept if line not in drop_all])
    return rows, sorted(drop_all), {**leak, **({"warnings": warnings} if warnings else {})}, kept_leak


@router.get("")
async def list_sets(project_id: int, db: AsyncSession = Depends(get_db)):
    """The project's eval sets (newest version, rows, versions, variants, the last eval per
    ``dataset_name`` label whose rows are that version's rows)."""
    await _project(db, project_id)
    sets = await ess.list_eval_sets(db, project_id)
    last = await ess.last_evals_by_label(db, project_id, sets)
    out = []
    for ds in sets:
        prefix = f"{ess.PREFIX}{ess.slug_of(ds)}@"
        out.append(ess.serialize(ds, last_evals={k: v for k, v in last.items() if k.startswith(prefix)}))
    return {"eval_sets": out}


@router.get("/variant-support")
async def variant_support(
    project_id: int, experiment_id: int, model_path: str | None = None, db: AsyncSession = Depends(get_db),
):
    """Whether a prompt-format variant can apply to evals of this run (``{supported, reason}``): not for
    classifier-head, token-classification, seq2seq or multimodal runs (no system message is rendered)."""
    from app.models.experiment import Experiment

    await _project(db, project_id)
    exp = await db.get(Experiment, experiment_id)
    if exp is None or int(exp.project_id) != int(project_id):
        raise HTTPException(404, f"Experiment {experiment_id} not found in project {project_id}")
    reason = ess.variant_unsupported_reason(exp, model_path)
    return {"experiment_id": experiment_id, "supported": reason is None, "reason": reason}


@router.post("", status_code=201)
async def create_from_file(
    project_id: int,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(default=""),
    field_mapping: str | None = Form(default=None),
    drop_rejected: bool = Form(default=False),
    drop_lines: str | None = Form(default=None),
    keep_leak_lines: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Create a set from a file (JSONL / JSON array / CSV / TSV, any task shape; a row may carry
    ``row_context``). Unusable rows and rows that leak training data are grouped by reason (422
    ``EVAL_SET_IMPORT_REJECTED_ROWS``) until dropped, or -- leak-flagged rows only -- kept on purpose
    (``keep_leak_lines``; recorded on the version and shown on its results)."""
    await _project(db, project_id)
    rows, dropped, leak, kept_leak = await _import_rows(
        db, project_id, file, field_mapping, drop_lines, drop_rejected, keep_leak_lines,
    )
    warnings = leak.pop("warnings", None)  # non-blocking (large row schemas)
    try:
        async with ess.write_lock(project_id, ess.slugify(name)):
            ds = await ess.create_eval_set(
                db, project_id, name=name, rows=rows, description=description,
                source={"kind": "upload", "filename": file.filename, "leak_scan": leak}, dropped_lines=dropped,
                kept_leak=kept_leak,
            )
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {**ess.serialize(ds), "dropped_rows": len(dropped), "dropped_lines": dropped, "leak_scan": leak,
            **({"warnings": warnings} if warnings else {})}


@router.post("/from-source", status_code=201)
async def create_from_source(project_id: int, body: FromSourceBody, db: AsyncSession = Depends(get_db)):
    """Create a set from another eval set (``{kind: eval_set, slug, version?}``) or a built-in set
    (``{kind: split, split: test | validation | gold_test | gold_dev}``), optionally with one
    prompt-format ``variant`` defined on it in the same call."""
    await _project(db, project_id)
    try:
        if body.variant is not None:  # validated before anything is written
            ess.normalize_variant_name(body.variant.name)
            ess.normalize_variant(body.variant.spec())
        rows, prov = await ess.rows_from_source(db, project_id, body.source.model_dump())
        kept_leak = prov.pop(ess.KEPT_LEAK_KEY, None)
        slug = ess.slugify(body.name)
        async with ess.write_lock(project_id, slug):
            ds = await ess.create_eval_set(
                db, project_id, name=body.name, rows=rows, description=body.description, source=prov,
                kept_leak=kept_leak,
            )
            if body.variant is not None:
                await ess.add_variant(db, ds, body.variant.name, body.variant.spec())
    except ess.EvalSetError as exc:
        raise _error(exc)
    return ess.serialize(ds)


@router.get("/{slug}")
async def get_set(project_id: int, slug: str, db: AsyncSession = Depends(get_db)):
    return ess.serialize(await _get(db, project_id, slug))


@router.get("/{slug}/rows")
async def get_rows(
    project_id: int, slug: str, version: int | None = None, offset: int = 0, limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    if offset < 0 or not 1 <= limit <= ess.ROWS_PAGE_MAX:
        raise HTTPException(400, f"offset must be >= 0 and limit between 1 and {ess.ROWS_PAGE_MAX}.")
    ds = await _get(db, project_id, slug)
    try:
        page = await ess.rows_page(ds, version=version, offset=offset, limit=limit)
    except ess.EvalSetError as exc:
        raise _error(exc)
    # Additive: the backend's own answer to "which row_context kinds does this project's task read" (the
    # answer-key list's fields), so the UI never labels context as used when the task ignores it.
    from app.services import row_context_service as rcs

    profile = await rcs.effective_task_profile(db, project_id)
    return {**page, **rcs.reads_report(profile)}


@router.post("/{slug}/versions", status_code=201)
async def new_version_from_file(
    project_id: int,
    slug: str,
    file: UploadFile = File(...),
    field_mapping: str | None = Form(default=None),
    drop_rejected: bool = Form(default=False),
    drop_lines: str | None = Form(default=None),
    keep_leak_lines: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Replace the rows with a file: a NEW version (older versions stay for the results that used them)."""
    await _get(db, project_id, slug)  # 404 before reading the file
    rows, dropped, leak, kept_leak = await _import_rows(
        db, project_id, file, field_mapping, drop_lines, drop_rejected, keep_leak_lines,
    )
    warnings = leak.pop("warnings", None)  # non-blocking (large row schemas)
    try:
        async with ess.write_lock(project_id, slug):
            ds = await _get(db, project_id, slug, fresh=True)
            entry = await ess.add_version(
                db, ds, rows, {"kind": "upload", "filename": file.filename, "leak_scan": leak}, dropped,
                kept_leak=kept_leak,
            )
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {**ess.serialize(ds), "new_version": entry, "dropped_rows": len(dropped), "dropped_lines": dropped,
            "leak_scan": leak, **({"warnings": warnings} if warnings else {})}


@router.post("/{slug}/versions/from-source", status_code=201)
async def new_version_from_source(
    project_id: int, slug: str, body: VersionFromSourceBody, db: AsyncSession = Depends(get_db)
):
    await _get(db, project_id, slug)
    try:
        rows, prov = await ess.rows_from_source(db, project_id, body.source.model_dump())
        kept_leak = prov.pop(ess.KEPT_LEAK_KEY, None)
        async with ess.write_lock(project_id, slug):
            ds = await _get(db, project_id, slug, fresh=True)
            entry = await ess.add_version(db, ds, rows, prov, kept_leak=kept_leak)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {**ess.serialize(ds), "new_version": entry}


@router.post("/{slug}/rows/delete")
async def delete_rows(project_id: int, slug: str, body: RowsDeleteBody, db: AsyncSession = Depends(get_db)):
    """A new version without the rows at these 0-based positions (of ``version``, default current)."""
    if not body.positions:
        raise HTTPException(400, "positions is empty.")
    try:
        async with ess.write_lock(project_id, slug):
            ds = await _get(db, project_id, slug, fresh=True)
            result = await ess.delete_rows(db, ds, body.positions, version=body.version)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {**result, "eval_set": ess.serialize(ds)}


@router.post("/{slug}/variants", status_code=201)
async def add_variant(project_id: int, slug: str, body: VariantBody, db: AsyncSession = Depends(get_db)):
    """Define a prompt-format variant: ``system`` = strip | replace | add (+ ``text``). Immutable once made."""
    try:
        async with ess.write_lock(project_id, slug):
            ds = await _get(db, project_id, slug, fresh=True)
            variant = await ess.add_variant(db, ds, body.name, body.spec())
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {"variant": variant, "eval_set": ess.serialize(ds)}


@router.delete("/{slug}/variants/{name}")
async def delete_variant(project_id: int, slug: str, name: str, db: AsyncSession = Depends(get_db)):
    """Remove a variant definition (results that used it keep its spec + digest)."""
    try:
        async with ess.write_lock(project_id, slug):
            ds = await _get(db, project_id, slug, fresh=True)
            await ess.delete_variant(db, ds, name)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return ess.serialize(ds)


@router.post("/{slug}/lock")
async def lock_set(project_id: int, slug: str, db: AsyncSession = Depends(get_db)):
    """Lock the set: no new versions, row deletes or delete. Variants can still be added."""
    async with ess.write_lock(project_id, slug):
        ds = await _get(db, project_id, slug, fresh=True)
        await ess.lock(db, ds)
    return ess.serialize(ds)


@router.delete("/{slug}")
async def delete_set(project_id: int, slug: str, db: AsyncSession = Depends(get_db)):
    """Delete the set and its files (409 while locked or while an eval Job uses it). Past results keep
    their label and provenance, and the slug's version numbers are never reused: a set recreated under
    the same name continues after the highest version this one had."""
    try:
        async with ess.write_lock(project_id, slug):
            ds = await _get(db, project_id, slug, fresh=True)
            await ess.delete_eval_set(db, ds)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {"deleted": ess.normalize_slug(slug)}

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
from app.services.gold_import_service import GoldFileError, group_rejects, parse_gold_file

router = APIRouter(prefix="/projects/{project_id}/eval-sets", tags=["Eval sets"])


class SourceBody(BaseModel):
    kind: str = Field(..., description="eval_set | split")
    slug: str | None = None
    version: int | None = None
    split: str | None = None


class FromSourceBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=ess.MAX_NAME_CHARS)
    description: str = ""
    source: SourceBody


class VersionFromSourceBody(BaseModel):
    source: SourceBody


class RowsDeleteBody(BaseModel):
    positions: list[int]
    version: int | None = None


class VariantBody(BaseModel):
    name: str
    system: str
    text: str | None = None


def _error(exc: ess.EvalSetError) -> SLMError:
    return SLMError(status_code=exc.status, error_code=exc.code, stage="evaluation", message=str(exc))


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


async def _get(db: AsyncSession, project_id: int, slug: str):
    try:
        return await ess.get_eval_set(db, project_id, slug)
    except ess.EvalSetError as exc:
        raise _error(exc)


async def _parse_upload(
    file: UploadFile, field_mapping: str | None, drop_lines: str | None, drop_rejected: bool
) -> tuple[list[dict[str, Any]], list[int]]:
    """The Phase 1 answer-key import, reused: ``(rows, dropped line numbers)``; 422
    ``EVAL_SET_IMPORT_REJECTED_ROWS`` groups the unusable rows by reason until each is dropped."""
    mapping = None
    if field_mapping:
        try:
            parsed = json.loads(field_mapping)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"field_mapping is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            raise HTTPException(400, "field_mapping must be a JSON object")
        mapping = {str(k): str(v) for k, v in parsed.items()}
    dropped: set[int] = set()
    if drop_lines:
        try:
            dropped = {int(x) for x in json.loads(drop_lines)}
        except (json.JSONDecodeError, TypeError, ValueError):
            raise HTTPException(400, "drop_lines must be a JSON list of line numbers")
    from app.api import dataset_import

    content = await file.read(dataset_import.MAX_UPLOAD_BYTES + 1)
    if len(content) > dataset_import.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file is too large.")
    try:
        rows, rejects = parse_gold_file(file.filename or "", content, mapping)
    except GoldFileError as exc:
        raise HTTPException(400, str(exc))
    remaining = [r for r in rejects if r[0] not in dropped]
    if remaining and not drop_rejected:
        raise SLMError(
            status_code=422,
            error_code="EVAL_SET_IMPORT_REJECTED_ROWS",
            stage="evaluation",
            message=f"{len(remaining)} row(s) in {file.filename} cannot be used as eval rows.",
            actionable_fix="Select the rows to drop (grouped by reason) and import the rest, or fix the file and upload it again.",
            metadata={
                "rejected": group_rejects(remaining),
                "row_counts": {"usable": len(rows), "rejected": len(remaining), "already_dropped": len(rejects) - len(remaining)},
            },
        )
    if not rows:
        raise SLMError(
            status_code=422,
            error_code="EVAL_SET_NO_ROWS",
            stage="evaluation",
            message=f"No usable rows in {file.filename}.",
            actionable_fix="Upload a file with one JSON object per line (or a JSON array / CSV with a header row).",
            metadata={"rejected": group_rejects(rejects)},
        )
    return [row for _line, row in rows], sorted(r[0] for r in rejects)


@router.get("")
async def list_sets(project_id: int, db: AsyncSession = Depends(get_db)):
    """The project's eval sets (newest version, rows, versions, variants, the last eval per
    ``dataset_name`` label)."""
    await _project(db, project_id)
    last = await ess.last_evals_by_label(db, project_id)
    sets = await ess.list_eval_sets(db, project_id)
    out = []
    for ds in sets:
        prefix = f"{ess.PREFIX}{ess.slug_of(ds)}@"
        out.append(ess.serialize(ds, last_evals={k: v for k, v in last.items() if k.startswith(prefix)}))
    return {"eval_sets": out}


@router.post("", status_code=201)
async def create_from_file(
    project_id: int,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(default=""),
    field_mapping: str | None = Form(default=None),
    drop_rejected: bool = Form(default=False),
    drop_lines: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Create a set from a file (JSONL / JSON array / CSV / TSV, any task shape; a row may carry
    ``row_context``). Rejects as in the answer-key import (422 ``EVAL_SET_IMPORT_REJECTED_ROWS``)."""
    await _project(db, project_id)
    rows, dropped = await _parse_upload(file, field_mapping, drop_lines, drop_rejected)
    try:
        ds = await ess.create_eval_set(
            db, project_id, name=name, rows=rows, description=description,
            source={"kind": "upload", "filename": file.filename}, dropped_lines=dropped,
        )
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {**ess.serialize(ds), "dropped_rows": len(dropped), "dropped_lines": dropped}


@router.post("/from-source", status_code=201)
async def create_from_source(project_id: int, body: FromSourceBody, db: AsyncSession = Depends(get_db)):
    """Create a set from another eval set (``{kind: eval_set, slug, version?}``) or a built-in set
    (``{kind: split, split: test | validation | gold_test | gold_dev}``)."""
    await _project(db, project_id)
    try:
        rows, prov = await ess.rows_from_source(db, project_id, body.source.model_dump())
        ds = await ess.create_eval_set(
            db, project_id, name=body.name, rows=rows, description=body.description, source=prov,
        )
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
        return await ess.rows_page(ds, version=version, offset=offset, limit=limit)
    except ess.EvalSetError as exc:
        raise _error(exc)


@router.post("/{slug}/versions", status_code=201)
async def new_version_from_file(
    project_id: int,
    slug: str,
    file: UploadFile = File(...),
    field_mapping: str | None = Form(default=None),
    drop_rejected: bool = Form(default=False),
    drop_lines: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Replace the rows with a file: a NEW version (older versions stay for the results that used them)."""
    ds = await _get(db, project_id, slug)
    rows, dropped = await _parse_upload(file, field_mapping, drop_lines, drop_rejected)
    try:
        async with ess.write_lock(project_id, ess.slug_of(ds)):
            entry = await ess.add_version(db, ds, rows, {"kind": "upload", "filename": file.filename}, dropped)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {**ess.serialize(ds), "new_version": entry, "dropped_rows": len(dropped), "dropped_lines": dropped}


@router.post("/{slug}/versions/from-source", status_code=201)
async def new_version_from_source(
    project_id: int, slug: str, body: VersionFromSourceBody, db: AsyncSession = Depends(get_db)
):
    ds = await _get(db, project_id, slug)
    try:
        rows, prov = await ess.rows_from_source(db, project_id, body.source.model_dump())
        async with ess.write_lock(project_id, ess.slug_of(ds)):
            entry = await ess.add_version(db, ds, rows, prov)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {**ess.serialize(ds), "new_version": entry}


@router.post("/{slug}/rows/delete")
async def delete_rows(project_id: int, slug: str, body: RowsDeleteBody, db: AsyncSession = Depends(get_db)):
    """A new version without the rows at these 0-based positions (of ``version``, default current)."""
    if not body.positions:
        raise HTTPException(400, "positions is empty.")
    ds = await _get(db, project_id, slug)
    try:
        async with ess.write_lock(project_id, ess.slug_of(ds)):
            result = await ess.delete_rows(db, ds, body.positions, version=body.version)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {**result, "eval_set": ess.serialize(ds)}


@router.post("/{slug}/variants", status_code=201)
async def add_variant(project_id: int, slug: str, body: VariantBody, db: AsyncSession = Depends(get_db)):
    """Define a prompt-format variant: ``system`` = strip | replace | add (+ ``text``). Immutable once made."""
    ds = await _get(db, project_id, slug)
    spec: dict[str, Any] = {"system": body.system}
    if body.text is not None:
        spec["text"] = body.text
    try:
        async with ess.write_lock(project_id, ess.slug_of(ds)):
            variant = await ess.add_variant(db, ds, body.name, spec)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {"variant": variant, "eval_set": ess.serialize(ds)}


@router.delete("/{slug}/variants/{name}")
async def delete_variant(project_id: int, slug: str, name: str, db: AsyncSession = Depends(get_db)):
    """Remove a variant definition (results that used it keep its spec + digest)."""
    ds = await _get(db, project_id, slug)
    try:
        async with ess.write_lock(project_id, ess.slug_of(ds)):
            await ess.delete_variant(db, ds, name)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return ess.serialize(ds)


@router.post("/{slug}/lock")
async def lock_set(project_id: int, slug: str, db: AsyncSession = Depends(get_db)):
    """Lock the set: no new versions, row deletes or delete. Variants can still be added."""
    ds = await _get(db, project_id, slug)
    await ess.lock(db, ds)
    return ess.serialize(ds)


@router.delete("/{slug}")
async def delete_set(project_id: int, slug: str, db: AsyncSession = Depends(get_db)):
    """Delete the set and its files (409 while locked or while an eval Job uses it). Past results keep
    their label and provenance."""
    ds = await _get(db, project_id, slug)
    try:
        async with ess.write_lock(project_id, ess.slug_of(ds)):
            await ess.delete_eval_set(db, ds)
    except ess.EvalSetError as exc:
        raise _error(exc)
    return {"deleted": slug}

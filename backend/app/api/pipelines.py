"""Multi-stage pipeline evaluation routes (G10).

Workspace-level (``/api/pipelines``), so the global ``authorize_request``
dependency only checks that a principal exists (no project id in the path).
Per-pipeline access is enforced here when ``AUTH_ENABLED``: a user needs VIEWER
on EVERY member project to see a pipeline or its runs (others are hidden from
the list and get 403 on direct access) and EDITOR on every member project to
create / edit / delete / upload / run. Admins pass. With ``AUTH_ENABLED=false``
there are no checks (like every other ``/api`` route).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole, ProjectMembership, ProjectRole
from app.security import PROJECT_ROLE_RANK, get_request_principal
from app.services import pipeline_eval_service as pes
from app.services import pipeline_service as ps

router = APIRouter(prefix="/pipelines", tags=["Pipelines"])
project_router = APIRouter(prefix="/projects/{project_id}", tags=["Pipelines"])


# ------------------------------------------------------------------ access


async def _roles(db: AsyncSession, request: Request) -> dict[int, ProjectRole] | None:
    """``None`` = unrestricted (auth off or admin); else the principal's project roles."""
    if not settings.AUTH_ENABLED:
        return None
    principal = get_request_principal(request)
    if principal is None:
        raise HTTPException(401, "Authentication required")
    if principal.role == GlobalRole.ADMIN:
        return None
    rows = (
        await db.execute(
            select(ProjectMembership.project_id, ProjectMembership.role).where(
                ProjectMembership.user_id == principal.user_id
            )
        )
    ).all()
    return {int(pid): role for pid, role in rows}


def _allowed(roles: dict[int, ProjectRole] | None, project_ids: list[int], need: ProjectRole) -> bool:
    if roles is None:
        return True
    return all(pid in roles and PROJECT_ROLE_RANK[roles[pid]] >= PROJECT_ROLE_RANK[need] for pid in project_ids)


async def _require(db: AsyncSession, request: Request, project_ids: list[int], need: ProjectRole) -> None:
    if not _allowed(await _roles(db, request), project_ids, need):
        word = "view" if need == ProjectRole.VIEWER else "edit"
        raise HTTPException(403, f"You need {need.value} access to every project in this pipeline to {word} it.")


def _load(pipeline_id: str) -> dict[str, Any]:
    try:
        return ps.load_pipeline(pipeline_id)
    except ps.PipelineNotFound:
        raise HTTPException(404, "Pipeline not found.")


def _definition_project_ids(body: Any) -> list[int]:
    stages = body.get("stages") if isinstance(body, dict) else None
    out: list[int] = []
    for st in stages if isinstance(stages, list) else []:
        pid = st.get("project_id") if isinstance(st, dict) else None
        if isinstance(pid, int) and not isinstance(pid, bool) and pid not in out:
            out.append(pid)
    return out


def _invalid(exc: ps.PipelineValidationError) -> HTTPException:
    return HTTPException(
        400,
        detail={
            "message": "Invalid pipeline: " + "; ".join(f"{e['field']}: {e['message']}" for e in exc.errors),
            "error_code": "PIPELINE_INVALID",
            "errors": exc.errors,
            "metadata": {"errors": exc.errors},
        },
    )


# ------------------------------------------------------------------ pipelines


@router.get("")
async def list_pipelines(request: Request, db: AsyncSession = Depends(get_db)):
    roles = await _roles(db, request)
    out = []
    for pid in ps.list_pipeline_ids():
        try:
            pipeline = ps.load_pipeline(pid)
        except ps.PipelineNotFound:
            continue
        if _allowed(roles, ps.member_project_ids(pipeline), ProjectRole.VIEWER):
            out.append(await ps.list_summary(db, pipeline))
    out.sort(key=lambda p: str(p.get("updated_at") or ""), reverse=True)
    return {"pipelines": out}


@router.post("", status_code=201)
async def create_pipeline(body: dict[str, Any], request: Request, db: AsyncSession = Depends(get_db)):
    await _require(db, request, _definition_project_ids(body), ProjectRole.EDITOR)
    try:
        pipeline = await ps.create_pipeline(db, body)
    except ps.PipelineValidationError as exc:
        raise _invalid(exc)
    return JSONResponse(status_code=201, content=pipeline)


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    pipeline = _load(pipeline_id)
    await _require(db, request, ps.member_project_ids(pipeline), ProjectRole.VIEWER)
    return pipeline


@router.put("/{pipeline_id}")
async def update_pipeline(pipeline_id: str, body: dict[str, Any], request: Request, db: AsyncSession = Depends(get_db)):
    pipeline = _load(pipeline_id)
    ids = ps.member_project_ids(pipeline) + [p for p in _definition_project_ids(body) if p not in ps.member_project_ids(pipeline)]
    await _require(db, request, ids, ProjectRole.EDITOR)
    try:
        return await ps.update_pipeline(db, pipeline_id, body)
    except ps.PipelineValidationError as exc:
        raise _invalid(exc)


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    pipeline = _load(pipeline_id)
    await _require(db, request, ps.member_project_ids(pipeline), ProjectRole.EDITOR)
    ps.delete_pipeline(pipeline_id)
    return {"deleted": pipeline_id}


# ------------------------------------------------------------------ held-out


@router.put("/{pipeline_id}/heldout")
async def put_heldout(pipeline_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    pipeline = _load(pipeline_id)
    await _require(db, request, ps.member_project_ids(pipeline), ProjectRole.EDITOR)
    ctype = (request.headers.get("content-type") or "").lower()
    cap = ps.MAX_HELDOUT_BYTES
    too_big = f"The held-out upload is over the {cap // (1024 * 1024) or 1} MB limit ({cap} bytes); split it or trim it."
    try:
        declared = int(request.headers.get("content-length") or 0)
    except ValueError:
        declared = 0
    if declared > cap:
        raise HTTPException(413, too_big)
    lines: list[tuple[int, Any]]
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(400, "Send the held-out set as a multipart field named 'file'.")
        raw = await upload.read(cap + 1)
        if len(raw) > cap:
            raise HTTPException(413, too_big)
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(400, "The held-out file must be UTF-8 JSONL.")
        lines = [(n, line) for n, line in enumerate(text.splitlines(), start=1)]
    else:
        raw_body = await request.body()
        if len(raw_body) > cap:
            raise HTTPException(413, too_big)
        try:
            body = json.loads(raw_body or b"null")
        except ValueError:
            raise HTTPException(400, "Send JSON {rows: [...]} or a multipart 'file'.")
        rows = body.get("rows") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise HTTPException(400, "Send JSON {rows: [...]} or a multipart 'file'.")
        lines = [(n, row) for n, row in enumerate(rows, start=1)]
    accepted, rejected = ps.parse_heldout(pipeline, lines)
    ps.save_heldout(pipeline_id, accepted)
    return {"accepted": len(accepted), "rejected_by_reason": rejected}


@router.get("/{pipeline_id}/heldout")
async def get_heldout(
    pipeline_id: str, request: Request, limit: int = Query(50, ge=1, le=1000), db: AsyncSession = Depends(get_db)
):
    pipeline = _load(pipeline_id)
    await _require(db, request, ps.member_project_ids(pipeline), ProjectRole.VIEWER)
    rows = ps.load_heldout(pipeline_id)
    return {"rows": rows[:limit], "total": len(rows)}


# ------------------------------------------------------------------ runs


@router.post("/{pipeline_id}/runs")
async def start_run(
    pipeline_id: str, request: Request, async_job: bool = Query(False), db: AsyncSession = Depends(get_db)
):
    from app.services.jobs_service import serialize_job

    pipeline = _load(pipeline_id)
    await _require(db, request, ps.member_project_ids(pipeline), ProjectRole.EDITOR)
    try:
        body = await request.json() if (await request.body()) else {}
    except ValueError:
        raise HTTPException(400, "Request body must be JSON.")
    max_rows = (body or {}).get("max_rows") if isinstance(body, dict) else None
    if max_rows is None:
        max_rows = pes.DEFAULT_MAX_ROWS
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= pes.MAX_ROWS:
        raise HTTPException(400, f"max_rows must be an integer 1..{pes.MAX_ROWS}.")
    if not ps.heldout_count(pipeline_id):
        raise HTTPException(400, "The pipeline has no held-out rows; upload a held-out set first.")
    if not async_job and max_rows > pes.MAX_SYNC_ROWS:
        raise HTTPException(
            400, f"A run of more than {pes.MAX_SYNC_ROWS} rows runs as a Job: use async_job=true.",
        )
    if async_job:
        principal = get_request_principal(request)
        job = await pes.start_pipeline_eval_job(
            db, pipeline=pipeline, max_rows=max_rows, user_id=principal.user_id if principal else None
        )
        return JSONResponse(status_code=202, content=json.loads(json.dumps(serialize_job(job), default=str)))
    try:
        return await pes.run_pipeline_eval(db, pipeline_id, ps.new_run_id(), max_rows=max_rows)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/{pipeline_id}/runs")
async def list_runs(pipeline_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    pipeline = _load(pipeline_id)
    await _require(db, request, ps.member_project_ids(pipeline), ProjectRole.VIEWER)
    return {"runs": await ps.mark_stale(db, ps.list_runs(pipeline_id))}


@router.get("/{pipeline_id}/runs/{run_id}")
async def get_run(pipeline_id: str, run_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    pipeline = _load(pipeline_id)
    await _require(db, request, ps.member_project_ids(pipeline), ProjectRole.VIEWER)
    try:
        run = ps.load_run(pipeline_id, run_id)
    except ps.PipelineNotFound:
        raise HTTPException(404, "Run not found.")
    if run.get("status") == "running":
        await ps.mark_stale(db, [run])
    return run


# ------------------------------------------------------------------ per project


@project_router.get("/pipelines")
async def project_pipelines(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    """Pipelines that include this project (the Results page link list); only
    those the caller may view."""
    roles = await _roles(db, request)
    out = []
    for pid in ps.list_pipeline_ids():
        try:
            pipeline = ps.load_pipeline(pid)
        except ps.PipelineNotFound:
            continue
        members = ps.member_project_ids(pipeline)
        if project_id in members and _allowed(roles, members, ProjectRole.VIEWER):
            stage_names = [s.get("name") for s in pipeline.get("stages") or [] if s.get("project_id") == project_id]
            out.append({"id": pipeline["id"], "name": pipeline.get("name"), "stages": stage_names})
    out.sort(key=lambda p: str(p.get("name") or ""))
    return {"pipelines": out}

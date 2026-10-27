"""Tool catalog API: the tools a project's model may call (imported from an MCP server or an OpenAPI
document, or entered by hand), with per-tool allow / read-only flags and a restorable history.

Mounts at ``/api/projects/{project_id}/tool-catalog``. See ``services/tool_catalog_service.py`` (store,
read-only rules) and ``services/mcp_client.py`` (transports, security gates). Secret values (env values,
header values) are accepted on import and stored only in the project's secrets; no response or Job
param ever carries a value.

Errors: 422 ``INVALID_TOOL_CATALOG`` (``metadata.errors``), 403 ``MCP_STDIO_DISABLED``, 502
``MCP_SERVER_ERROR``, 404 ``TOOL_CATALOG_VERSION_NOT_FOUND``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin import _require_admin
from app.database import get_db
from app.models.project import Project
from app.services import mcp_client
from app.services import tool_catalog_service as tcs

router = APIRouter(prefix="/projects/{project_id}/tool-catalog", tags=["Tool Catalog"])

JOB_KIND = "tool_catalog_import"


class McpImportBody(BaseModel):
    kind: str = Field(description="'mcp_stdio' or 'mcp_http'")
    label: str | None = None
    command: list[str] | None = Field(default=None, description="mcp_stdio: program then arguments (no shell).")
    env: dict[str, str | None] | None = Field(default=None, description="mcp_stdio: env var name -> value; null/blank keeps the stored secret.")
    url: str | None = None
    headers: dict[str, str | None] | None = Field(default=None, description="mcp_http: header name -> value; null/blank keeps the stored secret.")


class OpenApiImportBody(BaseModel):
    filename: str | None = None
    content: str


class ManualToolsBody(BaseModel):
    tools: list[dict[str, Any]]


class FlagsBody(BaseModel):
    flags: dict[str, dict[str, Any]]


class RestoreBody(BaseModel):
    version: str


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _invalid(exc: tcs.ToolCatalogError) -> HTTPException:
    errors = exc.errors
    first = errors[0] if errors else {"message": "Invalid tool catalog."}
    suffix = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    return HTTPException(422, {
        "error_code": "INVALID_TOOL_CATALOG",
        "message": f"{first['message']}{suffix}",
        "metadata": {"errors": errors},
    })


def _mcp_http_error(exc: mcp_client.McpError) -> HTTPException:
    if isinstance(exc, mcp_client.McpStdioDisabled):
        return HTTPException(403, {"error_code": "MCP_STDIO_DISABLED", "message": str(exc), "metadata": {"env": "MCP_STDIO_ALLOWED_BINARIES"}})
    return HTTPException(502, {"error_code": "MCP_SERVER_ERROR", "message": str(exc)})


@router.get("")
async def get_tool_catalog(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return tcs.catalog_status(await _project(db, project_id))


@router.post("/import/mcp")
async def import_mcp(
    project_id: int,
    body: McpImportBody,
    request: Request,
    async_job: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Connect to an MCP server and replace the catalog with its tools. ``stdio`` launches a process, so it
    needs the admin role (when auth is on) AND an allow-listed binary (``MCP_STDIO_ALLOWED_BINARIES``).
    ``?async_job=true`` runs it as a ``tool_catalog_import`` Job (202)."""
    project = await _project(db, project_id)
    req = body.model_dump(exclude_none=False)
    if body.kind == "mcp_stdio":
        _require_admin(request)
    try:
        checked = await asyncio.to_thread(tcs.validate_mcp_request, req)
        if checked["kind"] == "mcp_stdio":
            mcp_client.check_stdio_allowed(checked["command"][0])
    except tcs.ToolCatalogError as exc:
        raise _invalid(exc)
    except mcp_client.McpError as exc:
        raise _mcp_http_error(exc)

    if async_job:
        from app.models.job import Job, JobStatus
        from sqlalchemy import select

        from app.services.jobs_service import JobProgressHandle, serialize_job, start_job

        in_flight = (
            await db.execute(
                select(Job).where(Job.kind == JOB_KIND, Job.project_id == project_id)
                .where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])).limit(1)
            )
        ).scalar_one_or_none()
        if in_flight is not None:
            raise HTTPException(409, {"error_code": "TOOL_CATALOG_IMPORT_RUNNING", "message": "A tool catalog import is already running for this project.", "metadata": {"existing_job_id": in_flight.id}})

        async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
            from app.database import async_session_factory

            await handle.set_progress(fraction=0.1, message=f"Connecting to {checked['label']}")
            async with async_session_factory() as runner_db:
                runner_project = await runner_db.get(Project, project_id)
                if runner_project is None:
                    raise RuntimeError(f"Project {project_id} not found")
                try:
                    status = await tcs.import_from_mcp(runner_db, runner_project, req)
                except tcs.ToolCatalogError as exc:
                    raise RuntimeError(str(exc)) from None
                await runner_db.commit()
            return status

        job = await start_job(
            db, kind=JOB_KIND, title=f"Import tools · {checked['label']}", runner=_runner,
            project_id=project_id, params={"kind": checked["kind"], "label": checked["label"]},
        )
        await db.commit()
        return JSONResponse(status_code=202, content=serialize_job(job))

    try:
        status = await tcs.import_from_mcp(db, project, req)
    except tcs.ToolCatalogError as exc:
        raise _invalid(exc)
    except mcp_client.McpError as exc:
        raise _mcp_http_error(exc)
    await db.commit()
    return status


@router.post("/import/openapi")
async def import_openapi(project_id: int, body: OpenApiImportBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await _project(db, project_id)
    try:
        status = await tcs.import_from_openapi_async(project, body.content, body.filename)
    except tcs.ToolCatalogError as exc:
        raise _invalid(exc)
    await db.commit()
    return status


@router.put("/tools")
async def put_tools(project_id: int, body: ManualToolsBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await _project(db, project_id)
    try:
        status = await tcs.set_catalog_tools_async(project, body.tools, {"kind": "manual"})
    except tcs.ToolCatalogError as exc:
        raise _invalid(exc)
    await db.commit()
    return status


@router.patch("/flags")
async def patch_flags(project_id: int, body: FlagsBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await _project(db, project_id)
    try:
        status = tcs.set_tool_flags(project, body.flags)
    except tcs.ToolCatalogError as exc:
        raise _invalid(exc)
    await db.commit()
    return status


@router.post("/restore")
async def restore(project_id: int, body: RestoreBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await _project(db, project_id)
    try:
        status = tcs.restore_catalog_version(project, body.version)
    except LookupError as exc:
        raise HTTPException(404, {"error_code": "TOOL_CATALOG_VERSION_NOT_FOUND", "message": str(exc)})
    except tcs.ToolCatalogError as exc:
        raise _invalid(exc)
    await db.commit()
    return status


@router.delete("")
async def delete_catalog(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await _project(db, project_id)
    status = tcs.clear_catalog(project)
    await db.commit()
    return status

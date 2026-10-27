"""Serving gateway routes (G11).

* ``v1_router`` -- the OpenAI-compatible surface mounted at ``/v1`` OUTSIDE the
  ``/api`` auth dependency: it has its own scoped keys (always required, even
  with ``AUTH_ENABLED=false``) and answers with the OpenAI error envelope.
* ``router`` -- key management under ``/api/gateway/keys`` (normal ``/api`` auth).
* ``project_router`` -- per-project gateway settings, snippets, telemetry and
  the multi-LoRA serve plan.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.auth import ProjectMembership, ProjectRole
from app.models.gateway import GatewayApiKey
from app.models.project import Project
from app.security import _require_project_role, get_request_principal
from app.services import gateway_service as gw
from app.services.gateway_service import GatewayError, KeyInfo

v1_router = APIRouter(prefix="/v1", tags=["Gateway"])
router = APIRouter(prefix="/gateway", tags=["Gateway"])
project_router = APIRouter(prefix="/projects/{project_id}", tags=["Gateway"])


async def gateway_exception_handler(request: Request, exc: GatewayError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.body(), headers=exc.headers)


# ------------------------------------------------------------------ /v1


def _raw_key(request: Request) -> str | None:
    header = (request.headers.get("x-api-key") or "").strip() or None
    auth = request.headers.get("authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") and auth[7:].strip() else None
    if header and bearer and header != bearer:
        raise GatewayError(
            401, "X-API-Key and Authorization: Bearer carry different keys; send one.",
            code="invalid_api_key", type_="authentication_error",
        )
    return header or bearer  # never ?token=


async def gateway_key(request: Request, db: AsyncSession = Depends(get_db)) -> KeyInfo:
    key = await gw.authenticate(db, _raw_key(request))
    gw.check_rate_limit(key)
    return key


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        raise GatewayError(400, "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise GatewayError(400, "Request body must be a JSON object.")
    return body


@v1_router.get("/models")
async def v1_models(key: KeyInfo = Depends(gateway_key), db: AsyncSession = Depends(get_db)):
    return {"object": "list", "data": await gw.list_models(db, key)}


@v1_router.post("/chat/completions")
async def v1_chat_completions(request: Request, key: KeyInfo = Depends(gateway_key), db: AsyncSession = Depends(get_db)):
    body = await _json_body(request)
    result = await gw.execute_chat(db, key, body)
    if body.get("stream") is True:
        return StreamingResponse(
            iter(gw.sse_events(result)), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )
    return result


class RetrieveRequest(BaseModel):
    project: str
    query: str
    k: int | None = Field(default=None, ge=1, le=10)


@v1_router.post("/brewslm/retrieve")
async def v1_retrieve(req: RetrieveRequest, key: KeyInfo = Depends(gateway_key), db: AsyncSession = Depends(get_db)):
    project, slug = await gw.resolve_project(db, key, req.project)
    got = await gw.retrieve_passages(db, project, req.query, k=req.k)
    passages = []
    by_n = {c["n"]: c for c in got["citations"]}
    for n, hit in enumerate(got["hits"], start=1):
        c = by_n[n]
        passages.append(
            {
                "n": n, "passage_id": c["passage_id"], "document": c["document"],
                "source_document_id": c["source_document_id"], "source_url": c["source_url"],
                "text": str((hit.get("payload") or {}).get("text") or "")[:1500], "score": c["score"],
            }
        )
    return {"project": slug, "retrieval": got["retrieval"], "passages": passages}


@v1_router.get("/brewslm/tools")
async def v1_tools(key: KeyInfo = Depends(gateway_key), db: AsyncSession = Depends(get_db)):
    return {"tools": await gw.list_tools(db, key)}


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


@v1_router.post("/brewslm/tools/{name}/call")
async def v1_tool_call(name: str, req: ToolCallRequest, key: KeyInfo = Depends(gateway_key), db: AsyncSession = Depends(get_db)):
    return await gw.call_tool(db, key, name, req.arguments)


# ------------------------------------------------------------------ keys


class KeyCreateRequest(BaseModel):
    name: str
    project_ids: list[int]
    rate_limit_per_minute: int | None = None


async def _owned_project_ids(db: AsyncSession, principal: Any) -> set[int]:
    rows = (
        await db.execute(
            select(ProjectMembership.project_id).where(
                ProjectMembership.user_id == principal.user_id, ProjectMembership.role == ProjectRole.OWNER
            )
        )
    ).scalars().all()
    return {int(r) for r in rows}


def _can_manage_key(row: GatewayApiKey, principal: Any, owned: set[int]) -> bool:
    """Creator, or an owner of EVERY project in the key's scope (admins are checked by the caller)."""
    if row.created_by == principal.username:
        return True
    scope = {int(p) for p in (row.project_ids or [])}
    return bool(scope) and scope <= owned


@router.get("/keys")
async def list_keys(request: Request, db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(GatewayApiKey).order_by(GatewayApiKey.id.desc()))).scalars().all()
    principal = get_request_principal(request)
    if settings.AUTH_ENABLED and principal is not None and principal.role.value != "admin":
        owned = await _owned_project_ids(db, principal)
        rows = [r for r in rows if _can_manage_key(r, principal, owned)]
    return {"keys": [gw.serialize_key(r) for r in rows]}


@router.post("/keys", status_code=201)
async def create_key(req: KeyCreateRequest, request: Request, db: AsyncSession = Depends(get_db)):
    principal = get_request_principal(request)
    if settings.AUTH_ENABLED and principal is not None:
        for pid in req.project_ids:
            await _require_project_role(db, principal, int(pid), ProjectRole.EDITOR, is_websocket=False)
    try:
        raw, row = await gw.create_key(
            db, name=req.name, project_ids=req.project_ids, rate_limit_per_minute=req.rate_limit_per_minute,
            created_by=principal.username if principal else None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse(status_code=201, content={"key": raw, "record": gw.serialize_key(row)})


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    from datetime import datetime, timezone

    row = await db.get(GatewayApiKey, key_id)
    if row is None:
        raise HTTPException(404, "Gateway key not found.")
    principal = get_request_principal(request)
    if settings.AUTH_ENABLED and principal is not None and principal.role.value != "admin":
        if not _can_manage_key(row, principal, await _owned_project_ids(db, principal)):
            raise HTTPException(403, "Only the key's creator, an owner of every project it covers, or an admin can revoke it.")
    if row.revoked_at is None:
        row.is_active = False
        row.revoked_at = datetime.now(timezone.utc)
        await db.flush()
    return {"record": gw.serialize_key(row)}


# ------------------------------------------------------------------ per project


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found.")
    return project


@project_router.get("/gateway/settings")
async def get_gateway_settings(project_id: int, db: AsyncSession = Depends(get_db)):
    return await gw.settings_view(db, await _project(db, project_id))


@project_router.put("/gateway/settings")
async def put_gateway_settings(project_id: int, body: dict[str, Any], db: AsyncSession = Depends(get_db)):
    project = await _project(db, project_id)
    try:
        await gw.update_settings(db, project, body or {})
    except gw.GatewaySettingsError as exc:
        raise HTTPException(
            400,
            detail={
                "message": "Invalid gateway settings: " + "; ".join(f"{e['field']}: {e['message']}" for e in exc.errors),
                "error_code": "GATEWAY_SETTINGS_INVALID",
                "errors": exc.errors,
                "metadata": {"errors": exc.errors},
            },
        )
    return await gw.settings_view(db, project)


@project_router.get("/gateway/snippets")
async def get_gateway_snippets(project_id: int, db: AsyncSession = Depends(get_db)):
    return await gw.build_snippets(db, await _project(db, project_id))


@project_router.get("/gateway/telemetry")
async def get_gateway_telemetry(project_id: int, window_seconds: int = 86400, db: AsyncSession = Depends(get_db)):
    from app.services import served_model_telemetry_service as tel

    await _project(db, project_id)
    try:
        summary = await tel.compute_gateway_telemetry(db, project_id=project_id, window_seconds=window_seconds)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    recent = await tel.list_recent_gateway_requests(db, project_id=project_id, limit=50)
    return {"summary": {"by_model": summary["by_model"], "total": summary["total"]}, "recent": recent}


class MultiLoraRequest(BaseModel):
    experiment_ids: list[int]
    host: str | None = None
    port: int | None = None
    max_model_len: int | None = None


async def _multi_lora_plan(db: AsyncSession, project_id: int, req: MultiLoraRequest) -> dict[str, Any]:
    from app.services.serve_service import build_multi_lora_serve_plan

    await _project(db, project_id)
    try:
        return await build_multi_lora_serve_plan(
            db, project_id=project_id, experiment_ids=req.experiment_ids,
            host=req.host or "127.0.0.1", port=req.port or 8001, max_model_len=req.max_model_len,
        )
    except ValueError as exc:
        detail = str(exc)
        raise HTTPException(404 if "not found" in detail else 400, detail)


@project_router.post("/serve/multi-lora-plan")
async def multi_lora_plan(project_id: int, req: MultiLoraRequest, db: AsyncSession = Depends(get_db)):
    return await _multi_lora_plan(db, project_id, req)


@project_router.post("/serve/multi-lora-runs/start")
async def multi_lora_start(project_id: int, req: MultiLoraRequest, db: AsyncSession = Depends(get_db)):
    from app.services.serve_runtime_service import start_serve_run

    plan = await _multi_lora_plan(db, project_id, req)
    try:
        return await start_serve_run(
            project_id=project_id, source="multi_lora", export_id=None, model_id=None, template=plan["template"]
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

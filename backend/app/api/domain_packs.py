"""Domain pack API routes."""

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole
from app.schemas.domain_pack import (
    DomainPackContract,
    DomainPackDuplicateRequest,
    DomainPackResponse,
    DomainPackSummaryResponse,
)
from app.security import get_request_principal
from app.services.domain_hook_service import (
    clear_plugin_hooks,
    list_domain_hook_catalog,
    load_hook_plugins_from_settings,
)
from app.services.domain_pack_service import (
    create_domain_pack,
    duplicate_domain_pack,
    get_domain_pack,
    list_domain_packs,
    update_domain_pack,
)

from app.services import domain_pack_content_service as content_service
from app.services.domain_pack_content_service import PackContentError

router = APIRouter(prefix="/domain-packs", tags=["Domain Packs"])
# Mounted separately: lives under /projects/{id}/ but belongs to this surface.
project_pack_router = APIRouter(prefix="/projects/{project_id}/domain-pack", tags=["Domain Packs"])


class InstantiateRequest(BaseModel):
    project_name: str | None = Field(default=None, max_length=255)
    version: str | None = Field(default=None, max_length=32)


def _content_http_error(exc: PackContentError) -> HTTPException:
    return HTTPException(exc.status, {
        "error_code": exc.code,
        "message": exc.message,
        "metadata": {"errors": exc.errors},
    })


async def _read_upload(file: UploadFile) -> bytes:
    limit = content_service.MAX_TOTAL_BYTES + 1024 * 1024 + 1
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        chunks.append(chunk)
        if size >= limit:
            break
    return b"".join(chunks)


def _require_pack_write_access(request: Request) -> None:
    principal = get_request_principal(request)
    if not settings.AUTH_ENABLED:
        return
    if principal is None:
        raise HTTPException(401, "Authentication required")
    if principal.role not in {GlobalRole.ADMIN, GlobalRole.ENGINEER}:
        raise HTTPException(403, "Only admin or engineer can modify domain packs")


@router.get("")
async def list_packs(
    db: AsyncSession = Depends(get_db),
):
    """List available domain packs."""
    packs = await list_domain_packs(db)
    return {
        "packs": [DomainPackSummaryResponse.model_validate(item) for item in packs],
        "count": len(packs),
    }


@router.get("/hooks/catalog")
async def hooks_catalog():
    """List available normalizer/validator/evaluator hook IDs and plugin status."""
    return list_domain_hook_catalog()


@router.post("/hooks/reload")
async def reload_hooks(
    request: Request,
):
    """Reload domain hook plugins configured in settings."""
    _require_pack_write_access(request)
    clear_plugin_hooks()
    load_result = load_hook_plugins_from_settings(force_reload=True)
    return {
        "reload": load_result,
        "catalog": list_domain_hook_catalog(),
    }


@router.get("/{pack_id}", response_model=DomainPackResponse)
async def get_pack(
    pack_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a full domain pack contract by pack_id."""
    pack = await get_domain_pack(db, pack_id)
    if not pack:
        raise HTTPException(404, f"Domain pack '{pack_id}' not found")
    return DomainPackResponse.model_validate(pack)


@router.post("", response_model=DomainPackResponse, status_code=201)
async def create_pack(
    contract: DomainPackContract,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a new domain pack contract."""
    _require_pack_write_access(request)
    try:
        pack = await create_domain_pack(db, contract)
        return DomainPackResponse.model_validate(pack)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.put("/{pack_id}", response_model=DomainPackResponse)
async def update_pack(
    pack_id: str,
    contract: DomainPackContract,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Update an existing domain pack contract."""
    _require_pack_write_access(request)
    pack = await get_domain_pack(db, pack_id)
    if not pack:
        raise HTTPException(404, f"Domain pack '{pack_id}' not found")
    try:
        updated = await update_domain_pack(db, pack, contract)
        return DomainPackResponse.model_validate(updated)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{pack_id}/duplicate", response_model=DomainPackResponse, status_code=201)
async def duplicate_pack(
    pack_id: str,
    request_data: DomainPackDuplicateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Duplicate a domain pack with auto-generated pack_id/version unless overridden."""
    _require_pack_write_access(request)
    pack = await get_domain_pack(db, pack_id)
    if not pack:
        raise HTTPException(404, f"Domain pack '{pack_id}' not found")

    try:
        duplicated = await duplicate_domain_pack(
            db,
            pack,
            new_pack_id=request_data.new_pack_id,
            new_version=request_data.new_version,
            status_override=request_data.status.value,
        )
        return DomainPackResponse.model_validate(duplicated)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/import/preview")
async def import_preview(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Validate a pack zip and compare it with what is already imported. Writes nothing."""
    _require_pack_write_access(request)
    return await content_service.preview_import(db, await _read_upload(file))


@router.post("/import")
async def import_pack(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Import a pack zip. 422 ``INVALID_DOMAIN_PACK`` (``metadata.errors``) when
    invalid; 409 ``system_pack`` for platform packs; ``unchanged`` when the
    content hash matches the current version (nothing written)."""
    _require_pack_write_access(request)
    try:
        result = await content_service.import_pack(db, await _read_upload(file))
    except PackContentError as exc:
        raise _content_http_error(exc)
    return {
        "pack": DomainPackResponse.model_validate(result["pack"]),
        "version": result["version"],
        "unchanged": result["unchanged"],
        "diff": result["diff"],
        "warnings": result["warnings"],
    }


@router.get("/{pack_id}/content")
async def get_pack_content(
    pack_id: str,
    version: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """The imported content of a pack version (manifest, summary, files)."""
    try:
        return await content_service.get_pack_content(db, pack_id.strip().lower(), version)
    except PackContentError as exc:
        raise _content_http_error(exc)


async def _abort_instantiate(db: AsyncSession, project: Any) -> None:
    """Roll the session back and remove the half-built project's data dir."""
    import shutil

    try:
        project_id = getattr(project, "id", None)
    except Exception:  # noqa: BLE001 - expired / detached instance
        project_id = None
    await db.rollback()
    if project_id is not None:
        shutil.rmtree(settings.DATA_DIR / "projects" / str(project_id), ignore_errors=True)


@router.post("/{pack_id}/instantiate", status_code=201)
async def instantiate_pack(
    pack_id: str,
    body: InstantiateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a project from an imported pack version (all-or-nothing)."""
    from app.schemas.project import ProjectResponse

    _require_pack_write_access(request)
    principal = get_request_principal(request)
    project = None
    try:
        project, summary = await content_service.instantiate_pack(
            db,
            pack_id.strip().lower(),
            version=body.version,
            project_name=body.project_name,
            actor_user_id=getattr(principal, "user_id", None),
        )
        if settings.AUTH_ENABLED and principal:
            from app.models.auth import ProjectRole
            from app.security import upsert_project_membership

            await upsert_project_membership(db, project_id=project.id, user_id=principal.user_id, role=ProjectRole.OWNER)
        # Commit inside the guarded block: a failure here still cleans up the data dir.
        await db.commit()
    except PackContentError as exc:
        await _abort_instantiate(db, project)
        raise _content_http_error(exc)
    except ValueError as exc:
        await _abort_instantiate(db, project)
        raise HTTPException(422, {"error_code": "INSTANTIATE_FAILED", "message": str(exc), "metadata": {}})
    except Exception as exc:  # noqa: BLE001
        await _abort_instantiate(db, project)
        code = getattr(exc, "errors", None)
        if code is not None:  # e.g. UserProbePackError
            raise HTTPException(422, {"error_code": "INSTANTIATE_FAILED", "message": str(exc), "metadata": {"errors": code}})
        raise
    return {"project": ProjectResponse.model_validate(project), "summary": summary}


@project_pack_router.get("/export")
async def export_project_as_pack(
    project_id: int,
    pack_id: str | None = Query(default=None),
    version: str | None = Query(default=None),
    include_documents: str = Query(default="pack", pattern="^(pack|all|none)$"),
    db: AsyncSession = Depends(get_db),
):
    """Download the project's current artifacts as a domain-pack zip."""
    try:
        data, report = await content_service.export_project_pack_report(
            db, project_id, pack_id=pack_id, version=version, include_documents=include_documents
        )
    except PackContentError as exc:
        raise _content_http_error(exc)
    headers = {"Content-Disposition": f'attachment; filename="{report["pack_id"]}-{report["version"]}.zip"'}
    if report["skipped_documents"]:
        headers["X-Pack-Skipped-Documents"] = str(len(report["skipped_documents"]))
    return Response(content=data, media_type="application/zip", headers=headers)

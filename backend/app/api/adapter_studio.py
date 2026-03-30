"""API routes for Dataset Structure Explorer + Adapter Studio."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.adapter_studio import (
    AdapterDefinitionResponse,
    AdapterStudioExportRequest,
    AdapterStudioInferRequest,
    AdapterStudioPreviewRequest,
    AdapterStudioProfileRequest,
    AdapterStudioSaveRequest,
    AdapterStudioValidateRequest,
)
from app.services.adapter_studio_service import (
    adapter_catalog_summary,
    export_adapter_scaffold,
    get_adapter_definition_version,
    infer_adapter_definition,
    list_adapter_definition_versions,
    profile_dataset_structure,
    preview_adapter_transform,
    save_adapter_definition_version,
    serialize_adapter_definition,
    validate_adapter_coverage,
)

router = APIRouter(prefix="/projects/{project_id}/adapter-studio", tags=["Adapter Studio"])


def _raise_http_for_value_error(exc: ValueError) -> None:
    detail = str(exc)
    if detail.startswith("Project "):
        raise HTTPException(404, detail)
    raise HTTPException(400, detail)


@router.get("/catalog")
async def adapter_catalog():
    """List adapter contracts available to the visual studio and CLI."""
    return adapter_catalog_summary()


@router.post("/profile")
async def profile_dataset(
    project_id: int,
    req: AdapterStudioProfileRequest,
    db: AsyncSession = Depends(get_db),
):
    """Profile source schema, nested structure, null rates, lengths, labels, and PII hints."""
    try:
        return await profile_dataset_structure(
            db,
            project_id,
            source=req.source.model_dump(),
            sample_size=req.sample_size,
        )
    except ValueError as exc:
        _raise_http_for_value_error(exc)
        raise


@router.post("/infer")
async def infer_adapter(
    project_id: int,
    req: AdapterStudioInferRequest,
    db: AsyncSession = Depends(get_db),
):
    """Infer a non-Python adapter definition from source structure and sampled rows."""
    try:
        return await infer_adapter_definition(
            db,
            project_id,
            source=req.source.model_dump(),
            sample_size=req.sample_size,
            task_profile=req.task_profile,
        )
    except ValueError as exc:
        _raise_http_for_value_error(exc)
        raise


@router.post("/preview")
async def preview_adapter(
    project_id: int,
    req: AdapterStudioPreviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Preview transformed rows, dropped rows, errors, and auto-fix opportunities."""
    try:
        return await preview_adapter_transform(
            db,
            project_id,
            source=req.source.model_dump(),
            adapter_id=req.adapter_id,
            field_mapping=req.field_mapping,
            adapter_config=req.adapter_config,
            task_profile=req.task_profile,
            sample_size=req.sample_size,
            preview_limit=req.preview_limit,
        )
    except ValueError as exc:
        _raise_http_for_value_error(exc)
        raise


@router.post("/validate")
async def validate_adapter(
    project_id: int,
    req: AdapterStudioValidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Validate adapter coverage and return machine-readable reason codes and unblock actions."""
    try:
        return await validate_adapter_coverage(
            db,
            project_id,
            source=req.source.model_dump(),
            adapter_id=req.adapter_id,
            field_mapping=req.field_mapping,
            adapter_config=req.adapter_config,
            task_profile=req.task_profile,
            sample_size=req.sample_size,
            preview_limit=req.preview_limit,
        )
    except ValueError as exc:
        _raise_http_for_value_error(exc)
        raise


@router.post("/adapters", response_model=AdapterDefinitionResponse, status_code=201)
async def save_adapter_definition(
    project_id: int,
    req: AdapterStudioSaveRequest,
    db: AsyncSession = Depends(get_db),
):
    """Save versioned reusable adapter definitions authored without Python."""
    try:
        row = await save_adapter_definition_version(
            db,
            project_id,
            adapter_name=req.adapter_name,
            source_type=req.source_type,
            source_ref=req.source_ref,
            base_adapter_id=req.base_adapter_id,
            task_profile=req.task_profile,
            field_mapping=req.field_mapping,
            adapter_config=req.adapter_config,
            output_contract=req.output_contract,
            schema_profile=req.schema_profile,
            inference_summary=req.inference_summary,
            validation_report=req.validation_report,
            share_globally=req.share_globally,
        )
        return serialize_adapter_definition(row)
    except ValueError as exc:
        _raise_http_for_value_error(exc)
        raise


@router.get("/adapters")
async def list_adapter_definitions(
    project_id: int,
    adapter_name: str | None = Query(default=None),
    include_global: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
):
    try:
        rows = await list_adapter_definition_versions(
            db,
            project_id,
            adapter_name=adapter_name,
            include_global=bool(include_global),
        )
        return {
            "count": len(rows),
            "items": [serialize_adapter_definition(item) for item in rows],
            "filters": {
                "adapter_name": adapter_name,
                "include_global": bool(include_global),
            },
        }
    except ValueError as exc:
        _raise_http_for_value_error(exc)
        raise


@router.get("/adapters/{adapter_name}/versions/{version}", response_model=AdapterDefinitionResponse)
async def get_adapter_definition(
    project_id: int,
    adapter_name: str,
    version: int,
    db: AsyncSession = Depends(get_db),
):
    row = await get_adapter_definition_version(
        db,
        project_id,
        adapter_name=adapter_name,
        version=version,
    )
    if row is None:
        raise HTTPException(404, f"Adapter definition '{adapter_name}' v{version} not found")
    return serialize_adapter_definition(row)


@router.post("/adapters/{adapter_name}/versions/{version}/export")
async def export_adapter(
    project_id: int,
    adapter_name: str,
    version: int,
    req: AdapterStudioExportRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await export_adapter_scaffold(
            db,
            project_id,
            adapter_name=adapter_name,
            version=version,
            export_dir=req.export_dir,
        )
    except ValueError as exc:
        _raise_http_for_value_error(exc)
        raise

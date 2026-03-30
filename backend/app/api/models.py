"""Universal base-model registry + compatibility API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.base_model_registry import (
    BaseModelImportRequest,
    BaseModelListResponse,
    BaseModelRefreshRequest,
    ModelCompatibilityResponse,
    ModelExplainRequest,
    ModelRecommendResponse,
    ModelValidateRequest,
)
from app.services.base_model_registry_service import (
    evaluate_project_model_compatibility,
    get_base_model_record,
    import_base_model_record,
    list_base_model_records,
    recommend_models_for_project,
    refresh_base_model_record,
    serialize_base_model_record,
)

router = APIRouter(tags=["Base Models"])


@router.post("/models/import", status_code=201)
async def import_model_metadata(
    req: BaseModelImportRequest,
    db: AsyncSession = Depends(get_db),
):
    """Import/normalize a base model from Hugging Face, local path, or internal catalog."""
    try:
        row, created = await import_base_model_record(
            db,
            source_type=req.source_type,
            source_ref=req.source_ref,
            allow_network=bool(req.allow_network),
            overwrite=bool(req.overwrite),
        )
        return {
            "created": bool(created),
            "model": serialize_base_model_record(row),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/models/refresh")
async def refresh_model_metadata(
    req: BaseModelRefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    """Refresh metadata/provenance/cache for an imported base model record."""
    try:
        row = await refresh_base_model_record(
            db,
            model_id=req.model_id,
            model_key=req.model_key,
            allow_network=bool(req.allow_network),
        )
        return {"model": serialize_base_model_record(row)}
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/models", response_model=BaseModelListResponse)
async def list_models(
    family: str | None = Query(default=None),
    license: str | None = Query(default=None),
    hardware_fit: str | None = Query(default=None),
    min_context_length: int | None = Query(default=None, ge=1),
    max_params_b: float | None = Query(default=None, gt=0.0),
    training_mode: str | None = Query(default=None),
    search: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_base_model_records(
        db,
        family=family,
        license_token=license,
        hardware_fit=hardware_fit,
        min_context_length=min_context_length,
        max_params_b=max_params_b,
        training_mode=training_mode,
        search=search,
    )
    return {
        "count": len(rows),
        "models": [serialize_base_model_record(item) for item in rows],
        "filters": {
            "family": family,
            "license": license,
            "hardware_fit": hardware_fit,
            "min_context_length": min_context_length,
            "max_params_b": max_params_b,
            "training_mode": training_mode,
            "search": search,
        },
    }


@router.get("/models/{model_id}")
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)):
    row = await get_base_model_record(db, model_id=model_id)
    if row is None:
        raise HTTPException(404, "Model not found")
    return {"model": serialize_base_model_record(row)}


@router.post("/projects/{project_id}/models/validate", response_model=ModelCompatibilityResponse)
async def validate_project_model(
    project_id: int,
    req: ModelValidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Validate one imported model against project blueprint/adapter/runtime/target context."""
    row = await get_base_model_record(
        db,
        model_id=req.model_id,
        model_key=req.model_key,
        source_ref=req.model_key,
    )
    if row is None:
        raise HTTPException(404, "Model not found")

    try:
        payload = await evaluate_project_model_compatibility(
            db,
            project_id=project_id,
            model=row,
            dataset_adapter_id=req.dataset_adapter_id,
            runtime_id=req.runtime_id,
            target_profile_id=req.target_profile_id,
            allow_network=bool(req.allow_network),
            persist_lineage=True,
        )
        return payload
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/projects/{project_id}/models/compatible", response_model=ModelRecommendResponse)
async def list_project_compatible_models(
    project_id: int,
    limit: int = Query(default=10, ge=1, le=100),
    include_incompatible: bool = Query(default=False),
    family: str | None = Query(default=None),
    license: str | None = Query(default=None),
    hardware_fit: str | None = Query(default=None),
    min_context_length: int | None = Query(default=None, ge=1),
    max_params_b: float | None = Query(default=None, gt=0.0),
    training_mode: str | None = Query(default=None),
    search: str | None = Query(default=None),
    target_profile_id: str | None = Query(default=None),
    runtime_id: str | None = Query(default=None),
    dataset_adapter_id: str | None = Query(default=None),
    allow_network: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    try:
        rows = await recommend_models_for_project(
            db,
            project_id=project_id,
            limit=limit,
            include_incompatible=include_incompatible,
            family=family,
            license_token=license,
            hardware_fit=hardware_fit,
            min_context_length=min_context_length,
            max_params_b=max_params_b,
            training_mode=training_mode,
            search=search,
            target_profile_id=target_profile_id,
            runtime_id=runtime_id,
            dataset_adapter_id=dataset_adapter_id,
            allow_network=allow_network,
        )
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

    compatible_count = sum(1 for item in rows if bool(item.get("compatible")))
    return {
        "project_id": project_id,
        "count": len(rows),
        "compatible_count": compatible_count,
        "models": rows,
        "filters": {
            "limit": limit,
            "include_incompatible": include_incompatible,
            "family": family,
            "license": license,
            "hardware_fit": hardware_fit,
            "min_context_length": min_context_length,
            "max_params_b": max_params_b,
            "training_mode": training_mode,
            "search": search,
            "target_profile_id": target_profile_id,
            "runtime_id": runtime_id,
            "dataset_adapter_id": dataset_adapter_id,
            "allow_network": allow_network,
        },
    }


@router.post("/projects/{project_id}/models/explain")
async def explain_project_model_incompatibility(
    project_id: int,
    req: ModelExplainRequest,
    db: AsyncSession = Depends(get_db),
):
    """Return focused incompatible/warning reasons with unblock actions for one model."""
    row = await get_base_model_record(
        db,
        model_id=req.model_id,
        model_key=req.model_key,
        source_ref=req.model_key,
    )
    if row is None:
        raise HTTPException(404, "Model not found")

    try:
        payload = await evaluate_project_model_compatibility(
            db,
            project_id=project_id,
            model=row,
            dataset_adapter_id=req.dataset_adapter_id,
            runtime_id=req.runtime_id,
            target_profile_id=req.target_profile_id,
            allow_network=bool(req.allow_network),
            persist_lineage=True,
        )
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

    risky = [item for item in list(payload.get("why_risky") or []) if isinstance(item, dict)]
    return {
        "project_id": project_id,
        "model_id": payload.get("model_id"),
        "model_key": payload.get("model_key"),
        "compatible": payload.get("compatible"),
        "compatibility_score": payload.get("compatibility_score"),
        "reason_codes": payload.get("reason_codes", []),
        "incompatibilities": [
            item for item in risky if str(item.get("severity") or "").strip() in {"warning", "blocker"}
        ],
        "recommended_next_actions": payload.get("recommended_next_actions", []),
        "context": payload.get("context", {}),
        "generated_at": payload.get("generated_at"),
    }

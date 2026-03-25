from typing import Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.services import target_profile_service

router = APIRouter(prefix="/targets", tags=["Targets"])

class CompatibilityRequest(BaseModel):
    model_name: str
    target_id: str

@router.get("/catalog")
async def get_target_catalog(
    include_registry_meta: bool = Query(
        default=False,
        description="When true, include catalog-level metadata (plugin load status/version).",
    )
):
    """List available target hardware profiles."""
    if include_registry_meta:
        return target_profile_service.list_target_catalog()
    return target_profile_service.list_targets()

@router.post("/compatibility")
async def check_target_compatibility(req: CompatibilityRequest):
    """Check if a model is compatible with a target profile."""
    result = target_profile_service.check_compatibility(req.model_name, req.target_id)
    return result

@router.get("/estimate")
async def estimate_target_metrics(
    model_name: str = Query(..., description="Name of the base model"),
    target_id: str = Query(..., description="ID of the target profile")
):
    """Estimate memory and latency for a model on a target profile."""
    return target_profile_service.estimate_metrics(model_name, target_id)

"""Hardware Recommender API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.hardware_service import get_hardware_catalog, recommend_for_hardware

router = APIRouter(prefix="/hardware", tags=["Hardware"])

class RecommendRequest(BaseModel):
    hardware_id: str
    task_type: str = "causal_lm"

@router.get("/catalog")
async def catalog():
    """Get the supported hardware targets for recommendation."""
    profiles = get_hardware_catalog()
    return {"profiles": [p.__dict__ for p in profiles]}

@router.post("/recommend")
async def recommend(req: RecommendRequest):
    """Get model configuration recommendations for a specific hardware target."""
    try:
        rec = recommend_for_hardware(req.hardware_id, req.task_type)
        return rec.__dict__
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

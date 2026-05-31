"""Data Health Report API — D1 of the data-quality arc.

Mounts at ``/api/projects/{project_id}/data-health``. Aggregates every
data-quality signal (ingestion, cleaning, shape vs recipe, classification
balance) into one panel-friendly payload. See
``data_health_service.compute_data_health_report`` for the shape.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.data_health_service import compute_data_health_report


router = APIRouter(prefix="/projects/{project_id}/data-health", tags=["Data Health"])


@router.get("")
async def get_data_health_report(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the aggregated Data Health Report for a project."""
    try:
        return await compute_data_health_report(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

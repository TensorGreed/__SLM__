"""Export API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.export import ExportFormat
from app.services.export_service import create_export, list_exports, run_export

router = APIRouter(prefix="/projects/{project_id}/export", tags=["Export"])


class ExportCreateRequest(BaseModel):
    experiment_id: int
    export_format: ExportFormat
    quantization: str | None = None


class ExportRunRequest(BaseModel):
    eval_report: dict | None = None
    safety_scorecard: dict | None = None


@router.post("/create", status_code=201)
async def create(
    project_id: int,
    req: ExportCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new export."""
    export = await create_export(
        db, project_id, req.experiment_id, req.export_format, req.quantization,
    )
    return {
        "id": export.id,
        "format": export.export_format.value,
        "status": export.status.value,
        "output_path": export.output_path,
    }


@router.post("/{export_id}/run")
async def run(
    project_id: int,
    export_id: int,
    req: ExportRunRequest = ExportRunRequest(),
    db: AsyncSession = Depends(get_db),
):
    """Execute the export process."""
    try:
        export = await run_export(db, export_id, req.eval_report, req.safety_scorecard)
        return {
            "id": export.id,
            "status": export.status.value,
            "output_path": export.output_path,
            "manifest": export.manifest,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/list")
async def list_all(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all exports for a project."""
    exports = await list_exports(db, project_id)
    return [
        {
            "id": e.id,
            "format": e.export_format.value,
            "status": e.status.value,
            "quantization": e.quantization,
            "output_path": e.output_path,
            "created_at": e.created_at.isoformat(),
        }
        for e in exports
    ]

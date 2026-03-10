"""Export API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.export import ExportFormat
from app.services.deployment_target_service import (
    default_deployment_targets_for_format,
    list_deployment_targets,
)
from app.services.export_service import (
    create_export,
    list_exports,
    run_export,
    validate_export_deployment,
)
from app.services.serve_service import build_export_serve_plan

router = APIRouter(prefix="/projects/{project_id}/export", tags=["Export"])


class ExportCreateRequest(BaseModel):
    experiment_id: int
    export_format: ExportFormat
    quantization: str | None = None


class ExportRunRequest(BaseModel):
    eval_report: dict | None = None
    safety_scorecard: dict | None = None
    deployment_targets: list[str] | None = None
    run_smoke_tests: bool = True


class ExportDeploymentValidateRequest(BaseModel):
    deployment_targets: list[str] | None = None
    run_smoke_tests: bool = True


class ExportServePlanRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    smoke_test_prompt: str = "Hello from SLM!"
    target_ids: list[str] | None = None


@router.post("/create", status_code=201)
async def create(
    project_id: int,
    req: ExportCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new export."""
    try:
        export = await create_export(
            db, project_id, req.experiment_id, req.export_format, req.quantization,
        )
        return {
            "id": export.id,
            "format": export.export_format.value,
            "status": export.status.value,
            "output_path": export.output_path,
            "file_size_bytes": export.file_size_bytes,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/{export_id}/run")
async def run(
    project_id: int,
    export_id: int,
    req: ExportRunRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Execute the export process."""
    payload = req or ExportRunRequest()
    try:
        export = await run_export(
            db,
            project_id,
            export_id,
            payload.eval_report,
            payload.safety_scorecard,
            deployment_targets=payload.deployment_targets,
            run_smoke_tests=bool(payload.run_smoke_tests),
        )
        return {
            "id": export.id,
            "status": export.status.value,
            "output_path": export.output_path,
            "run_id": (export.manifest or {}).get("run_id"),
            "file_size_bytes": export.file_size_bytes,
            "manifest": export.manifest,
            "deployment": (export.manifest or {}).get("deployment"),
        }
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/deployment-targets")
async def deployment_targets(
    project_id: int,
    export_format: ExportFormat | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List Deployment Target SDK catalog and format-specific defaults."""
    # Project id is retained for consistency and future project-aware target policy.
    _ = db, project_id
    payload = list_deployment_targets(export_format=export_format)
    if export_format is not None:
        payload["default_target_ids"] = default_deployment_targets_for_format(export_format)
    return payload


@router.post("/{export_id}/deployment-validate")
async def validate_deployment(
    project_id: int,
    export_id: int,
    req: ExportDeploymentValidateRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Run deployment target validation/smoke checks on an existing export run."""
    payload = req or ExportDeploymentValidateRequest()
    try:
        report = await validate_export_deployment(
            db,
            project_id,
            export_id,
            deployment_targets=payload.deployment_targets,
            run_smoke_tests=bool(payload.run_smoke_tests),
        )
        return {
            "export_id": export_id,
            "project_id": project_id,
            "deployment": report,
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
            "run_id": (e.manifest or {}).get("run_id"),
            "file_size_bytes": e.file_size_bytes,
            "created_at": e.created_at.isoformat(),
            "completed_at": e.completed_at.isoformat() if e.completed_at else None,
        }
        for e in exports
    ]


@router.post("/{export_id}/serve-plan")
async def serve_plan(
    project_id: int,
    export_id: int,
    req: ExportServePlanRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Generate local serve launch templates + health/smoke curl snippets for an export."""
    payload = req or ExportServePlanRequest()
    try:
        return await build_export_serve_plan(
            db,
            project_id=project_id,
            export_id=export_id,
            host=payload.host,
            port=payload.port,
            smoke_test_prompt=payload.smoke_test_prompt,
            target_ids=payload.target_ids,
        )
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

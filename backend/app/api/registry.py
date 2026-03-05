"""Model registry API routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.experiment import Experiment
from app.models.registry import RegistryStage
from app.services.registry_service import (
    build_readiness_snapshot,
    list_models,
    mark_model_deployed,
    promote_model,
    register_model,
    serialize_registry_entry,
)

router = APIRouter(prefix="/projects/{project_id}/registry", tags=["Registry"])


class ModelRegisterRequest(BaseModel):
    experiment_id: int
    export_id: int | None = None
    name: str | None = Field(default=None, max_length=255)
    version: str | None = Field(default=None, max_length=64)
    artifact_path: str | None = Field(default=None, max_length=1024)


class PromotionGateRequest(BaseModel):
    min_exact_match: float | None = Field(default=None, ge=0, le=1)
    min_f1: float | None = Field(default=None, ge=0, le=1)
    min_llm_judge_pass_rate: float | None = Field(default=None, ge=0, le=1)
    min_safety_pass_rate: float | None = Field(default=None, ge=0, le=1)
    max_exact_match_regression: float | None = Field(default=None, ge=0, le=1)
    max_f1_regression: float | None = Field(default=None, ge=0, le=1)


class PromoteModelRequest(BaseModel):
    target_stage: Literal["candidate", "staging", "production", "archived"]
    force: bool = False
    gates: PromotionGateRequest | None = None


class DeployModelRequest(BaseModel):
    environment: str = Field(default="staging", min_length=1, max_length=32)
    endpoint_url: str = Field(default="", max_length=1024)
    notes: str = Field(default="", max_length=2000)


@router.post("/models/register", status_code=201)
async def register(
    project_id: int,
    req: ModelRegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register an experiment as a promotable model."""
    try:
        entry = await register_model(
            db=db,
            project_id=project_id,
            experiment_id=req.experiment_id,
            export_id=req.export_id,
            name=req.name,
            version=req.version,
            artifact_path=req.artifact_path,
        )
        return serialize_registry_entry(entry)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/models")
async def list_registered_models(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List registered models for a project."""
    entries = await list_models(db, project_id)
    return {"models": [serialize_registry_entry(item) for item in entries]}


@router.get("/readiness/{experiment_id}")
async def readiness(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Compute readiness snapshot for an experiment without registering it."""
    exp_row = await db.execute(
        select(Experiment.id).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    if exp_row.scalar_one_or_none() is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found in project {project_id}")

    snapshot = await build_readiness_snapshot(db, experiment_id)
    return {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "readiness": snapshot,
    }


@router.post("/models/{model_id}/promote")
async def promote(
    project_id: int,
    model_id: int,
    req: PromoteModelRequest,
    db: AsyncSession = Depends(get_db),
):
    """Promote a registry model to staging/production with gate checks."""
    try:
        target = RegistryStage(req.target_stage)
        entry, report = await promote_model(
            db=db,
            project_id=project_id,
            model_id=model_id,
            target_stage=target,
            force=req.force,
            gates=req.gates.model_dump() if req.gates else None,
        )
        return {
            "model": serialize_registry_entry(entry),
            "promotion_report": report,
        }
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.post("/models/{model_id}/deploy")
async def deploy(
    project_id: int,
    model_id: int,
    req: DeployModelRequest,
    db: AsyncSession = Depends(get_db),
):
    """Mark a registry model as deployed in an environment."""
    try:
        entry = await mark_model_deployed(
            db=db,
            project_id=project_id,
            model_id=model_id,
            environment=req.environment,
            endpoint_url=req.endpoint_url,
            notes=req.notes,
        )
        return serialize_registry_entry(entry)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

"""Data Studio API routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.data_studio_service import (
    build_data_studio_domain_detection,
    build_data_studio_gold_set_workbench,
    build_data_studio_llm_assist,
    build_data_studio_mapping_preview,
    build_data_studio_overview,
    build_data_studio_review_queue,
    build_data_studio_sources,
    build_data_studio_synthetic_playbook_center,
    build_data_studio_synthetic_recommendations,
)


router = APIRouter(prefix="/projects/{project_id}/data-studio", tags=["Data Studio"])


class DataStudioAssistRequest(BaseModel):
    focus: Literal["mapping", "domain"] = "mapping"
    provider: Literal["ollama", "openai_compatible"] = "ollama"
    api_url: str = Field(default="", max_length=2048)
    api_key: str = Field(default="", max_length=4096)
    model_name: str = Field(default="llama3", min_length=1, max_length=256)


@router.get("/overview")
async def get_data_studio_overview(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the additive Data Studio readiness overview."""

    try:
        return await build_data_studio_overview(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/sources")
async def get_data_studio_sources(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return source health and recent sources for Data Studio."""

    try:
        return await build_data_studio_sources(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/mapping-preview")
async def get_data_studio_mapping_preview(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a recipe-aware adapter mapping preview for Data Studio."""

    try:
        return await build_data_studio_mapping_preview(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/domain-detection")
async def get_data_studio_domain_detection(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return domain detection evidence and applied runtime summary."""

    try:
        return await build_data_studio_domain_detection(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/gold-set")
async def get_data_studio_gold_set_workbench(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a read-only Gold Set workbench summary for Data Studio."""

    try:
        return await build_data_studio_gold_set_workbench(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/synthetic-playbooks")
async def get_data_studio_synthetic_playbook_center(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a read-only Synthetic Playbook Center summary."""

    try:
        return await build_data_studio_synthetic_playbook_center(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/synthetic-recommendations")
async def get_data_studio_synthetic_recommendations(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return deterministic domain-aware synthetic recommendations."""

    try:
        return await build_data_studio_synthetic_recommendations(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/review-queue")
async def get_data_studio_review_queue(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a read-only cross-workflow review queue summary."""

    try:
        return await build_data_studio_review_queue(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.post("/assist")
async def run_data_studio_assist(
    project_id: int,
    req: DataStudioAssistRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run optional LLM assist over deterministic Data Studio checks."""

    try:
        return await build_data_studio_llm_assist(
            db,
            project_id,
            focus=req.focus,
            provider=req.provider,
            api_url=req.api_url,
            api_key=req.api_key,
            model_name=req.model_name,
        )
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

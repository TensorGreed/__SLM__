"""Data Studio API routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole
from app.security import get_request_principal
from app.services.data_studio_service import (
    build_data_studio_coach_rail,
    build_data_studio_dataset_versions,
    build_data_studio_domain_detection,
    build_data_studio_gold_set_workbench,
    build_data_studio_llm_assist,
    build_data_studio_mapping_preview,
    build_data_studio_overview,
    build_data_studio_prepare_dataset,
    build_data_studio_quality_safety,
    build_data_studio_review_queue,
    build_data_studio_sources,
    build_data_studio_synthetic_playbook_center,
    build_data_studio_synthetic_quality_analytics,
    build_data_studio_synthetic_recommendations,
    create_data_studio_domain_setup_from_detection,
)


router = APIRouter(prefix="/projects/{project_id}/data-studio", tags=["Data Studio"])


class DataStudioAssistRequest(BaseModel):
    focus: Literal["mapping", "domain"] = "mapping"
    provider: Literal["ollama", "openai_compatible"] = "ollama"
    api_url: str = Field(default="", max_length=2048)
    api_key: str = Field(default="", max_length=4096)
    model_name: str = Field(default="llama3", min_length=1, max_length=256)


class DataStudioDomainSetupCreateRequest(BaseModel):
    confirm: bool = False


def _require_domain_setup_write_access(request: Request) -> None:
    principal = get_request_principal(request)
    if not settings.AUTH_ENABLED:
        return
    if principal is None:
        raise HTTPException(401, "Authentication required")
    if principal.role not in {GlobalRole.ADMIN, GlobalRole.ENGINEER}:
        raise HTTPException(403, "Only admin or engineer can create domain setup drafts")


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


@router.get("/coach")
async def get_data_studio_coach_rail(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a read-only cross-section Data Studio coach rail."""

    try:
        return await build_data_studio_coach_rail(db, project_id)
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


@router.post("/domain-detection/domain-setup")
async def create_data_studio_domain_setup(
    project_id: int,
    req: DataStudioDomainSetupCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create missing draft domain setup records from detection after confirmation."""

    _require_domain_setup_write_access(request)
    if not req.confirm:
        raise HTTPException(400, "Confirmation is required before creating domain setup drafts.")
    try:
        return await create_data_studio_domain_setup_from_detection(db, project_id)
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


@router.get("/synthetic-quality")
async def get_data_studio_synthetic_quality(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return read-only deterministic synthetic quality analytics."""

    try:
        return await build_data_studio_synthetic_quality_analytics(db, project_id)
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


@router.get("/quality-safety")
async def get_data_studio_quality_safety(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return deterministic read-only quality and safety scan signals."""

    try:
        return await build_data_studio_quality_safety(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/prepare-dataset")
async def get_data_studio_prepare_dataset(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a read-only dataset preparation readiness summary."""

    try:
        return await build_data_studio_prepare_dataset(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/dataset-versions")
async def get_data_studio_dataset_versions(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a read-only prepared dataset version summary."""

    try:
        return await build_data_studio_dataset_versions(db, project_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/dataset-versions/compare")
async def compare_prepared_versions_endpoint(
    project_id: int,
    a: int,
    b: int,
):
    """Diff two prepared-version snapshots — row counts per split, total, source
    mix, seed, ratios, split strategy (Epic E)."""
    from app.services.data_studio_service import compare_prepared_versions

    try:
        return compare_prepared_versions(project_id, a, b)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/dataset-versions/{version}/activate")
async def activate_prepared_version_endpoint(
    project_id: int,
    version: int,
    db: AsyncSession = Depends(get_db),
):
    """Restore a prepared-version snapshot to the active files (Epic E) so the
    trainer / export / coverage read it, and mark it active. Shared by "Make
    active" and "Retrain from this version" (retrain = activate + launch
    training, which then sees this version's data)."""
    from app.services.data_studio_service import activate_prepared_version

    try:
        return await activate_prepared_version(db, project_id, version)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/dataset-versions/{split}/export")
async def export_prepared_split(
    project_id: int,
    split: str,
):
    """Download a prepared split (``train`` / ``val`` / ``test``) as JSONL.

    Epic E — the first real action on the Versions surface: the prepared
    artifact the trainer actually consumes, downloadable for inspection,
    external eval, or archival. (Mark-active / compare / retrain-from-version
    need versioned file snapshots the prepare step doesn't keep — the prepared
    dir is latest-only — so they're deferred, not faked here.)
    """
    from fastapi.responses import FileResponse

    from app.services.data_studio_service import resolve_prepared_split_path

    path = resolve_prepared_split_path(project_id, split)
    if path is None:
        raise HTTPException(404, f"Unknown split '{split}' (expected train/val/test).")
    if not path.exists():
        raise HTTPException(
            404,
            f"Prepared {split} split not found — run Prepare Dataset first.",
        )
    return FileResponse(
        path=str(path),
        media_type="application/x-ndjson",
        filename=f"project-{project_id}-{path.stem}.jsonl",
    )


@router.get("/playbook-gap-recommendations")
async def get_playbook_gap_recommendations_endpoint(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Underrepresented classes + a class_balance_fill playbook recommendation
    each (Epic E — gap-tied synthetic). Drives the "generate N to balance"
    cards that deep-link into the synthetic tab with the playbook prefilled."""
    from app.services.playbook_gap_service import get_playbook_gap_recommendations

    return await get_playbook_gap_recommendations(db, project_id)


@router.get("/prepared-version-preview")
async def get_prepared_version_preview_endpoint(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Next prepared-dataset version number + the row breakdown the next Prepare
    will snapshot (accepted synthetic / gold / cleaned). Epic E — "what version
    will include this?" for the review queue."""
    from app.services.data_studio_service import get_prepared_version_preview

    return await get_prepared_version_preview(db, project_id)


@router.get("/split-class-coverage")
async def get_split_class_coverage(
    project_id: int,
):
    """Per-class coverage across prepared TRAIN/VAL/TEST splits + plain-language
    warnings (e.g. "your val set has no `billing` examples"). Epic E — catches a
    blind eval split before training. File-only; no DB."""
    from app.services.data_studio_service import build_split_class_coverage

    return build_split_class_coverage(project_id)


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

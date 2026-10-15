"""Redact text with a project's span-tagging (token classification) run."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services import redaction_service

router = APIRouter(prefix="/projects/{project_id}", tags=["Redaction"])


class RedactRequest(BaseModel):
    text: str = Field(..., min_length=1)
    experiment_id: int | None = None
    mode: Literal["mask", "tag", "spans"] = "mask"
    min_score: float | None = Field(None, ge=0.0, le=1.0)
    recall_bias: float | None = Field(None, ge=0.0, le=1.0)


@router.post("/redact")
async def redact(project_id: int, req: RedactRequest, db: AsyncSession = Depends(get_db)):
    """Tag ``text`` with the run's span model (default: the newest completed
    token-classification run) and return the spans with scores, plus the
    text masked (``[TYPE]``) or tagged (``<type>…</type>``) and an offsets
    map. ``min_score`` / ``recall_bias`` override the run's thresholds.json."""
    try:
        return await redaction_service.redact(
            db,
            project_id=project_id,
            text=req.text,
            experiment_id=req.experiment_id,
            mode=req.mode,
            min_score=req.min_score,
            recall_bias=req.recall_bias,
        )
    except redaction_service.RedactionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


class TuneThresholdsRequest(BaseModel):
    experiment_id: int


@router.post("/span-thresholds/tune")
async def tune_span_thresholds(project_id: int, req: TuneThresholdsRequest, db: AsyncSession = Depends(get_db)):
    """Tune a span run's per-type score thresholds on the validation split
    and write them to its thresholds.json (read by evaluation, the redact
    endpoint and the export). See ``span_threshold_service``."""
    from app.services.span_threshold_service import tune_run_thresholds

    try:
        return await tune_run_thresholds(db, project_id, req.experiment_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

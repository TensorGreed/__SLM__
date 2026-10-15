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
    # {type: min_recall, "_default": min_recall}; None = the project's
    # span_recall_floors setting.
    floors: dict[str, float | None] | None = None
    # Write the suggested recall_bias when a floor is not reachable at
    # threshold 0 (global: every type gets more spans). Off = report only.
    apply_recall_bias: bool = False


@router.post("/span-thresholds/tune")
async def tune_span_thresholds(project_id: int, req: TuneThresholdsRequest, db: AsyncSession = Depends(get_db)):
    """Tune a span run's per-type score thresholds on the validation split
    and write them to its thresholds.json (read by evaluation, the redact
    endpoint and the export). Types with a recall floor keep at least that
    validation recall; see ``span_threshold_service``."""
    from app.services.span_threshold_service import tune_run_thresholds

    try:
        return await tune_run_thresholds(
            db, project_id, req.experiment_id, floors=req.floors, apply_recall_bias=req.apply_recall_bias,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class RecallFloorsRequest(BaseModel):
    floors: dict[str, float | None] = Field(default_factory=dict)
    default: float | None = None


def _floors_payload(project) -> dict:
    from app.services.compliance_bundles_service import suggested_recall_floors
    from app.services.span_threshold_service import DEFAULT_FLOOR_KEY, known_span_types, project_floors

    floors = project_floors(project)
    types = known_span_types(project, project.id)
    suggestions = suggested_recall_floors(project.runtime_config)
    return {
        "floors": {k: v for k, v in floors.items() if k != DEFAULT_FLOOR_KEY},
        "default": floors.get(DEFAULT_FLOOR_KEY),
        "types": types,
        # Offered, never applied: the user accepts them in the editor.
        "suggestions": {t: s for t, s in suggestions.items() if t in types},
    }


@router.get("/span-recall-floors")
async def get_span_recall_floors(project_id: int, db: AsyncSession = Depends(get_db)):
    """The project's per-type minimum recalls for threshold tuning, the
    types a floor may name, and compliance-regime suggestions."""
    from app.models.project import Project

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return _floors_payload(project)


@router.put("/span-recall-floors")
async def put_span_recall_floors(project_id: int, req: RecallFloorsRequest, db: AsyncSession = Depends(get_db)):
    """Replace the project's recall floors (``{floors: {type: 0–1},
    default: 0–1 | null}``). 422 for a value outside 0–1 or a type the
    project does not use."""
    from app.models.project import Project
    from app.services.span_threshold_service import DEFAULT_FLOOR_KEY, FLOORS_KEY, known_span_types, normalise_floors

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    raw = {k: v for k, v in req.floors.items() if k != DEFAULT_FLOOR_KEY}
    if req.default is not None:
        raw[DEFAULT_FLOOR_KEY] = req.default
    try:
        floors = normalise_floors(raw, known_span_types(project, project_id))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    cfg = dict(project.runtime_config or {})
    cfg[FLOORS_KEY] = floors
    project.runtime_config = cfg
    await db.commit()
    await db.refresh(project)
    return _floors_payload(project)

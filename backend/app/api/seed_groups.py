"""Seed groups API: pick the seed of a multi-seed run to ship (by a
validation metric only) and export it (``seed_selection_service``)."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.export import ExportFormat
from app.services import seed_selection_service as seeds

router = APIRouter(prefix="/projects/{project_id}/seed-groups", tags=["Seed groups"])


class ShipSeedRequest(BaseModel):
    export_format: ExportFormat = ExportFormat.HUGGINGFACE
    quantization: str | None = None
    # A validation metric id to pick by instead of the task default.
    metric: str | None = None
    run_smoke_tests: bool = True
    # A span tagger's model card reads its evaluation + human-review evidence
    # from the picked child's EvalResults, and a seed child has none until it
    # is lift-checked. True: run the span lift check inline first when it is
    # missing. False: refuse with 409 instead.
    evaluate_if_missing: bool = True


async def _pick(db: AsyncSession, project_id: int, leader_id: int, metric: str | None) -> dict[str, Any]:
    try:
        return await seeds.seed_group_pick(db, project_id, leader_id, metric=metric)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{leader_id}/pick")
async def pick(
    project_id: int,
    leader_id: int,
    metric: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Which seed of this group to ship, by a validation metric only (task
    default: span F1 / macro-F1 / lowest loss). Every child with its value,
    the chosen one, the rule, and the selection already recorded (if any)."""
    return await _pick(db, project_id, leader_id, metric)


@router.post("/{leader_id}/ship", status_code=201)
async def ship(
    project_id: int,
    leader_id: int,
    req: ShipSeedRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Record the pick on the group's leader, then run the normal export of
    the picked child (Hugging Face by default). The export's manifest,
    release notes and — for a span tagger — model card say which seed was
    picked, out of how many, by which validation metric and values. A span
    child with no evaluation on record is lift-checked first (test examples
    + answer key), so the card carries that evidence; ``evaluate_if_missing``
    false turns that into a 409."""
    from app.models.experiment import Experiment
    from app.services.export_service import create_export, run_export

    body = req or ShipSeedRequest()
    result = await _pick(db, project_id, leader_id, body.metric)
    picked_id = result.get("picked_experiment_id")
    if picked_id is None:
        raise HTTPException(
            400,
            f"No seed of run #{leader_id} has a validation {result.get('metric_label')} in its training report — "
            "nothing to pick by.",
        )
    leader = await seeds.load_leader(db, project_id, leader_id)
    child = await db.get(Experiment, int(picked_id))
    try:
        evaluated = await seeds.evaluate_before_ship(
            db, project_id, child, evaluate_if_missing=bool(body.evaluate_if_missing)
        )
    except seeds.SeedNotEvaluated as exc:
        raise HTTPException(409, str(exc)) from exc
    export = await create_export(db, project_id, picked_id, body.export_format, body.quantization)
    selection = await seeds.record_selection(
        db, leader, result, export_id=export.id, export_format=body.export_format.value
    )
    export = await run_export(db, project_id, export.id, run_smoke_tests=bool(body.run_smoke_tests))
    await db.commit()
    manifest = export.manifest if isinstance(export.manifest, dict) else {}
    return {
        "leader_experiment_id": leader_id,
        "picked_experiment_id": picked_id,
        "picked_seed_value": result.get("picked_seed_value"),
        "selection": {**selection, "text": seeds.selection_text(selection)},
        # The span lift check run for this ship (None: evidence already existed / not a span run).
        "evaluated_before_ship": evaluated,
        "export": {
            "id": export.id,
            "experiment_id": export.experiment_id,
            "format": export.export_format.value,
            "status": export.status.value,
            "output_path": export.output_path,
            "run_id": manifest.get("run_id"),
            "error": manifest.get("error") or getattr(export, "error_message", None),
        },
    }

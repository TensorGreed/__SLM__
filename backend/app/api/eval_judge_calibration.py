"""Eval-judge calibration (Judge calibration V4): blind labels on judged eval answers → per-judge trust.

Routes under ``/projects/{project_id}/eval-judge/calibration``; service: ``eval_judge_calibration_service``.
The label source is the project's ``judge_calibration.label_source`` (the same setting as the review-queue
judges: ``GET|PUT /synthetic/calibration/settings``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project

router = APIRouter(prefix="/projects/{project_id}/eval-judge/calibration", tags=["Evaluation"])

JOB_KIND = "eval_judge_calibration_reference"


class EvalCalibrationLabelRequest(BaseModel):
    content_key: str = Field(..., min_length=1, max_length=64)
    scale: str = Field(default="verdict", description="verdict (correct / partial / wrong) | rubric (pass / fail)")
    tier: str = Field(default="human", description="Only `human` labels are posted by hand.")
    label: str
    judge: str | None = Field(default=None, max_length=256, description="The judge whose sample this answer came from (a record only).")
    reason: str | None = Field(default=None, max_length=500)


class EvalCalibrationSkipRequest(BaseModel):
    content_key: str = Field(..., min_length=1, max_length=64)
    scale: str = "verdict"
    tier: str = "human"
    reason: str | None = Field(default=None, max_length=500)


class EvalCalibrationReferenceRequest(BaseModel):
    backend: str | None = Field(default=None, max_length=256, description="Any synth backend pin; omitted = the first reachable local backend.")
    scale: str = "verdict"
    judge: str | None = Field(default=None, max_length=256, description="Sample the answers this judge graded (default: the judge with most).")
    n: int = Field(default=20, ge=1, le=100)
    seed: int = 0


def _error(message: str, param: str, status: int = 400) -> HTTPException:
    return HTTPException(status, {"error_code": "invalid_calibration_param", "message": message, "metadata": {"param": param}})


def _scale(scale: str) -> str:
    from app.services import eval_judge_calibration_service as ecal

    if scale not in ecal.SCALES:
        raise _error(f"scale must be one of {', '.join(ecal.SCALES)}.", "scale")
    return scale


def _tier(tier: str, *, allow_both: bool = False) -> str:
    from app.services import judge_calibration_service as calib

    allowed = calib.LABEL_SOURCES if allow_both else calib.TIERS
    if tier not in allowed:
        raise _error(f"tier must be one of {', '.join(allowed)}.", "tier")
    return tier


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


@router.get("/labels")
async def get_progress(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Per scale: judged answers (and ``over_cap`` answers left out for a cut text), the judges and their
    verdict counts, and labels / skips per tier."""
    from app.services import eval_judge_calibration_service as ecal

    live = await ecal.live_result_ids(db, project_id)
    return await asyncio.to_thread(ecal.progress, project_id, live)


@router.get("/sample")
async def get_sample(
    project_id: int, scale: str = "verdict", judge: str | None = None, tier: str = "human", n: int = 20, seed: int = 0,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Answers to label BLIND: judged by ``judge`` (default: the judge with most), not yet labelled or
    skipped in ``tier``, spread across that judge's verdicts, shuffled. Each row is the question, the
    reference answer and the model's answer only: no verdict, score, reason or judge name."""
    from app.services import eval_judge_calibration_service as ecal

    _scale(scale)
    _tier(tier)
    if not 1 <= n <= ecal.MAX_SAMPLE:
        raise _error(f"n must be between 1 and {ecal.MAX_SAMPLE}.", "n")
    live = await ecal.live_result_ids(db, project_id)
    _, items = await asyncio.to_thread(ecal.sample, project_id, scale, tier, n, judge=judge, seed=seed, live_ids=live)
    return {"scale": scale, "tier": tier, "labels": list(ecal.LABELS[scale]),
            "rows": [ecal.render_blind(i, scale) for i in items]}


@router.post("/labels")
async def post_label(project_id: int, req: EvalCalibrationLabelRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Record (or replace) one blind human label, keyed by the answer's content key."""
    from app.services import eval_judge_calibration_service as ecal

    _scale(req.scale)
    if req.tier != "human":
        raise _error("Only human labels are posted here; reference-model labels come from the reference labelling Job.", "tier")
    if req.label not in ecal.LABELS[req.scale]:
        raise _error(f"label must be one of {', '.join(ecal.LABELS[req.scale])}.", "label")
    pop = await asyncio.to_thread(ecal.population, project_id, req.scale, await ecal.live_result_ids(db, project_id))
    item = pop.get(req.content_key)
    if item is None:
        raise HTTPException(404, {"error_code": "answer_not_found", "message": "No judged answer has this content key.",
                                  "metadata": {"content_key": req.content_key}})
    stratum = item["verdicts"].get(req.judge) if req.judge else None
    entry = ecal.label_entry(req.label, labeller=ecal.HUMAN_LABELLER, reason=req.reason, stratum=stratum,
                             judge=req.judge if stratum else None)
    await asyncio.to_thread(ecal.save_labels, project_id, req.scale, "human", {req.content_key: entry})
    return {"content_key": req.content_key, "scale": req.scale, "tier": "human", "label": req.label, "at": entry["at"]}


@router.post("/skips")
async def post_skip(project_id: int, req: EvalCalibrationSkipRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """The labeller's Skip: not a label; the answer is not offered again in the tier and is reported."""
    from app.services import eval_judge_calibration_service as ecal

    _scale(req.scale)
    if req.tier != "human":
        raise _error("Only human skips are posted here; the reference labelling Job records its own.", "tier")
    pop = await asyncio.to_thread(ecal.population, project_id, req.scale, await ecal.live_result_ids(db, project_id))
    if req.content_key not in pop:
        raise HTTPException(404, {"error_code": "answer_not_found", "message": "No judged answer has this content key.",
                                  "metadata": {"content_key": req.content_key}})
    entry = ecal.skip_entry(reason=req.reason or "skipped", labeller=ecal.HUMAN_LABELLER)
    await asyncio.to_thread(ecal.save_labels, project_id, req.scale, "human", {req.content_key: entry})
    return {"content_key": req.content_key, "scale": req.scale, "tier": "human", "skipped": True, "at": entry["at"]}


@router.delete("/labels/{content_key}")
async def delete_label(project_id: int, content_key: str, scale: str = "verdict", tier: str = "human") -> dict[str, Any]:
    from app.services import eval_judge_calibration_service as ecal

    _scale(scale)
    _tier(tier)
    if not await asyncio.to_thread(ecal.delete_label, project_id, scale, content_key, tier):
        raise HTTPException(404, {"error_code": "label_not_found", "message": f"No {tier} label or skip for this answer.",
                                  "metadata": {"content_key": content_key, "tier": tier}})
    return {"content_key": content_key, "scale": scale, "tier": tier, "deleted": True}


@router.get("/report")
async def get_report(project_id: int, scale: str = "verdict", tier: str | None = None, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Per judge on ``scale`` against ``tier`` (omitted = the project's label_source; ``both`` = one report
    per tier, never pooled): agreement, strict pass precision / recall, wrong-recall, too strict / too
    lenient, weighted kappa, Wilson CIs, n, skips, ``insufficient`` and the trust status."""
    from app.services import eval_judge_calibration_service as ecal
    from app.services import judge_calibration_service as calib

    _scale(scale)
    project = await _project(db, project_id)
    source = calib.label_source_of(project.runtime_config)
    chosen = _tier(tier, allow_both=True) if tier else source
    live = await ecal.live_result_ids(db, project_id)
    reports = await asyncio.to_thread(ecal.report, project_id, scale, calib.tiers_for(chosen), live)
    judges = sorted({j["judge"] for r in reports for j in r["judges"]})
    return {
        "label_source": source, "tier": chosen, "scale": scale, "labels": list(ecal.LABELS[scale]),
        "min_class_labels": ecal.MIN_CLASS_LABELS, "min_total_labels": ecal.MIN_TOTAL_LABELS,
        "reports": reports,
        "trust": {j: ecal.judge_trust(j, reports, chosen, scale) for j in judges},
    }


@router.get("/trust")
async def get_trust(project_id: int, judge: str, scale: str = "verdict", db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """One judge's trust status on ``scale`` under the project's label_source."""
    from app.services import eval_judge_calibration_service as ecal

    _scale(scale)
    project = await _project(db, project_id)
    live = await ecal.live_result_ids(db, project_id)
    return await asyncio.to_thread(ecal.trust_for, project_id, judge, project.runtime_config, scale, live)


@router.post("/reference-labelling")
async def start_reference_labelling(
    project_id: int, req: EvalCalibrationReferenceRequest, async_job: bool = False, db: AsyncSession = Depends(get_db),
):
    """A reference model labels a blind stratified sample of judged answers into the ``reference_model``
    tier with the eval judge's own prompt. Runs as the ``eval_judge_calibration_reference`` Job
    (``?async_job=true``); 409 while one runs for the project."""
    from app.api.synthetic import _resolve_cloud_backend_override
    from app.services import eval_judge_calibration_service as ecal
    from app.services.synth_backends import SynthBackendError
    from app.services.synth_playbook_service import BACKEND_REGISTRY, pick_backend

    _scale(req.scale)
    if not async_job:
        raise _error("Reference labelling runs as a background Job — add ?async_job=true.", "async_job")
    await _project(db, project_id)
    live = await ecal.live_result_ids(db, project_id)
    _, items = await asyncio.to_thread(ecal.sample, project_id, req.scale, "reference_model", req.n, judge=req.judge, seed=req.seed,
                                       live_ids=live)
    if not items:
        raise HTTPException(400, {
            "error_code": "no_calibration_rows",
            "message": "No answers to label: every judged answer already has a reference label or was skipped (or no eval has judged answers stored yet).",
            "metadata": {"tier": "reference_model", "scale": req.scale},
        })
    cloud_override = await _resolve_cloud_backend_override(db, project_id, req.backend)
    if cloud_override is None:
        try:
            pick_backend(req.backend, registry=BACKEND_REGISTRY)
        except SynthBackendError as exc:
            raise HTTPException(503, str(exc)) from exc
    async with _start_lock():
        return await _start_job(db, project_id, req)


_START_LOCKS: dict[int, asyncio.Lock] = {}


def _start_lock() -> asyncio.Lock:
    """One lock per event loop: two concurrent POSTs can't both pass the in-flight check and start a Job."""
    return _START_LOCKS.setdefault(id(asyncio.get_running_loop()), asyncio.Lock())


async def _start_job(db: AsyncSession, project_id: int, req: EvalCalibrationReferenceRequest):
    from app.api.synthetic import _resolve_cloud_backend_override, _with_budget_warning
    from app.models.job import Job, JobStatus
    from app.services import eval_judge_calibration_service as ecal
    from app.services import jobs_service
    from app.services.synth_playbook_service import BACKEND_REGISTRY, pick_backend

    in_flight = (
        await db.execute(
            select(Job).where(Job.kind == JOB_KIND, Job.project_id == project_id)
            .where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])).limit(1)
        )
    ).scalar_one_or_none()
    if in_flight is not None:
        raise HTTPException(409, {"error_code": "EVAL_CALIBRATION_REFERENCE_ALREADY_RUNNING",
                                  "message": "Reference labelling of eval answers is already running for this project. Watch the bell.",
                                  "metadata": {"existing_job_id": in_flight.id}})

    async def _runner(handle: "jobs_service.JobProgressHandle") -> dict:
        from app.database import async_session_factory

        async def _progress(fraction: float, message: str) -> None:
            await handle.set_progress(fraction=fraction, message=message)

        rubric = None
        async with async_session_factory() as runner_db:
            override = await _resolve_cloud_backend_override(runner_db, project_id, req.backend)
            live = await ecal.live_result_ids(runner_db, project_id)
            if req.scale == "rubric":
                from app.services import structured_plan_prompt as spp

                project = await runner_db.get(Project, project_id)
                rubric = spp.rubric_for_config(spp.effective_structured_plan_config(getattr(project, "runtime_config", None), project_id))
        backend_obj = override if override is not None else pick_backend(req.backend, registry=BACKEND_REGISTRY)
        _, items = await asyncio.to_thread(ecal.sample, project_id, req.scale, "reference_model", req.n, judge=req.judge, seed=req.seed,
                                           live_ids=live)
        if not items:
            raise RuntimeError("Reference labelling found no answers to label (they were labelled before it started).")
        await handle.set_progress(message=f"Reference-labelling {len(items)} answers…")
        result = await ecal.run_reference_labelling(project_id, items, backend_obj, req.scale, rubric=rubric, progress=_progress)
        await handle.set_progress(fraction=1.0, message=f"Reference-labelled {result['labelled']} of {len(items)} answers")
        return result

    job = await jobs_service.start_job(
        db,
        kind=JOB_KIND,
        title=f"Eval judge reference labels · {req.n} answers · {req.backend or 'local model'}",
        runner=_runner,
        project_id=project_id,
        params={"backend": req.backend, "scale": req.scale, "judge": req.judge, "n": req.n, "seed": req.seed},
    )
    return JSONResponse(status_code=202, content=await _with_budget_warning(db, project_id, jobs_service.serialize_job(job)))

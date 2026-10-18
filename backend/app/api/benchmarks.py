"""External benchmarks API (Epic G-B1) — see external_benchmark_service."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services import external_benchmark_service as svc

router = APIRouter(prefix="/projects/{project_id}/benchmarks", tags=["External benchmarks"])


class PreviewRequest(BaseModel):
    # Any dataset-import locator: hf:<id>[:<split>[:<revision>]], file:<path> …
    locator: str = Field(..., min_length=3, max_length=512)
    limit: int = Field(svc.DEFAULT_ROWS, ge=1, le=svc.MAX_ROWS)


class CreateRequest(PreviewRequest):
    name: str = Field(..., min_length=1, max_length=120)
    # source label -> project label | [project labels] (a group) | "ignore".
    # Many-to-one allowed; a project label belongs to at most one group.
    # Q&A / summarization benchmarks have no labels: {} (or omitted).
    mapping: dict[str, str | list[str]] = Field(default_factory=dict)
    local_only: bool = False
    licence: str | None = Field(None, max_length=200)
    join_adjacent: bool = False


class EvaluateRequest(BaseModel):
    experiment_id: int | None = None


def _bad(exc: Exception) -> HTTPException:
    return HTTPException(400, str(exc))


@router.post("/preview")
async def preview(project_id: int, req: PreviewRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Read a sample of the source: detected shape, its labels with counts,
    the project's labels, a suggested mapping (name matches only)."""
    try:
        return await svc.preview_benchmark(db, project_id, req.locator, limit=req.limit)
    except svc.BenchmarkError as exc:
        raise _bad(exc) from exc


@router.post("")
async def create(project_id: int, req: CreateRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    try:
        ds = await svc.create_benchmark(
            db, project_id, locator=req.locator, name=req.name, mapping=req.mapping, limit=req.limit,
            local_only=req.local_only, licence=req.licence, join_adjacent=req.join_adjacent,
        )
    except svc.BenchmarkError as exc:
        raise _bad(exc) from exc
    return svc.serialize_benchmark(ds)


@router.get("")
async def list_all(project_id: int, experiment_id: int | None = None, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return {"benchmarks": await svc.benchmarks_for_run(db, project_id, experiment_id)}


@router.get("/{benchmark_id}/history")
async def history(project_id: int, benchmark_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Every score of the benchmark, oldest first (run id + name, date, score)."""
    try:
        ds = await svc.get_benchmark(db, project_id, benchmark_id)
    except svc.BenchmarkError as exc:
        raise HTTPException(404, str(exc)) from exc
    return svc.benchmark_history(ds)


@router.post("/{benchmark_id}/contamination")
async def rescan_contamination(project_id: int, benchmark_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Re-scan the benchmark's gold values (entity texts / answers) against
    the project's current training rows; the result replaces the manifest's
    ``contamination`` block (it is also computed once at import)."""
    try:
        ds = await svc.rescan_contamination(db, project_id, benchmark_id)
    except svc.BenchmarkError as exc:
        raise HTTPException(404, str(exc)) from exc
    return svc.serialize_benchmark(ds)


@router.delete("/{benchmark_id}")
async def delete(project_id: int, benchmark_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    try:
        await svc.delete_benchmark(db, project_id, benchmark_id)
    except svc.BenchmarkError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": benchmark_id}


@router.post("/{benchmark_id}/evaluate", status_code=202)
async def evaluate(
    project_id: int, benchmark_id: int, req: EvaluateRequest | None = None, db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Score a run (default: the latest trained one) on the benchmark, as a
    ``benchmark_eval`` Job the bell tracks."""
    from app.services import jobs_service
    from app.services.eval_summary_service import _latest_trained_experiment_id

    try:
        ds = await svc.get_benchmark(db, project_id, benchmark_id)
    except svc.BenchmarkError as exc:
        raise HTTPException(404, str(exc)) from exc
    experiment_id = (req.experiment_id if req else None) or await _latest_trained_experiment_id(db, project_id)
    if experiment_id is None:
        raise HTTPException(400, "Train a model first — there is no run to score.")
    name = svc.manifest_of(ds).get("name") or ds.name

    async def _runner(handle: Any) -> dict[str, Any]:
        from app.database import async_session_factory

        await handle.set_progress(fraction=0.1, message=f"Scoring run #{experiment_id} on {name}")
        async with async_session_factory() as session:
            return await svc.evaluate_benchmark(session, project_id, benchmark_id, experiment_id)

    job = await jobs_service.start_job(
        db, kind="benchmark_eval", title=f"Benchmark: {name} · run #{experiment_id}", runner=_runner,
        project_id=project_id, params={"benchmark_id": benchmark_id, "experiment_id": experiment_id},
    )
    return jobs_service.serialize_job(job)

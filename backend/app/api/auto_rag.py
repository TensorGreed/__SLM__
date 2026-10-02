"""Auto-RAG API routes (USER-SUCCESS Epic 9 Phase 9a).

Phase 9a ships a single read-only preview endpoint:

  ``GET /api/projects/{project_id}/auto-rag/preview``

Builds (or rebuilds, on cache miss) the BM25 index over the project's
training rows and returns the top-K retrievals for a given query. The
caller passes the query as a URL param so a human can eyeball "for
*this* question, would the index find a reasonable Q&A pair?" before
Phase 9b wires retrieval into actual inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.project import Project
from app.services.auto_rag_service import (
    AutoRagUnavailable,
    QA_CORPUS_SOURCE,
    _load_rag_corpus_rows,
    qa_index_is_current,
    build_bm25_index,
    recommended_text_keys_for_recipe,
    retrieve,
)


router = APIRouter(
    prefix="/projects/{project_id}/auto-rag", tags=["AutoRAG"]
)


@router.get("/documents")
async def get_document_index(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Document-passage retrieval index for this project (cleaned PDF /
    DOCX / HTML passages). Builds or refreshes it on the way — cheap BM25
    over cleaned.jsonl — so the Playground can offer "answer from your
    documents" as soon as cleaning has run."""
    if await db.get(Project, project_id) is None:
        raise HTTPException(404, f"Project {project_id} not found")
    from app.services.auto_rag_service import ensure_document_index

    return {"project_id": project_id, **ensure_document_index(project_id)}


@router.get("/preview")
async def preview_auto_rag(
    project_id: int,
    query: str = Query(
        ...,
        min_length=1,
        description=(
            "The query to retrieve against. Phase 9b will pass the "
            "user's playground prompt here; for Phase 9a's preview, "
            "the caller passes whatever they want to spot-check."
        ),
    ),
    k: int = Query(
        3,
        ge=1,
        le=20,
        description="Number of retrievals to return. Default 3 (matches Phase 9b's planned prepend size).",
    ),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Build the BM25 index (if needed) and return top-K retrievals.

    Status codes:
      200 — index loaded + query scored; ``retrieved`` is up to ``k`` hits.
      400 — project missing a recipe; recipe has no RAG corpus shape
            (Phase 9a covers qa-sft only); no training data on disk.
      404 — project not found.
      503 — BM25 index built but unreadable (corrupt file on disk —
            rare; would indicate a partial write).
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id")
    if not recipe_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Project has no task type selected — auto-RAG needs the "
                "task type to pick the right text fields for the corpus."
            ),
        )

    if recommended_text_keys_for_recipe(recipe_id) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Task type {recipe_id!r} has no auto-RAG corpus shape "
                f"yet. Phase 9a covers qa-sft only; other task types "
                f"plug in in later phases."
            ),
        )

    rows = await _load_rag_corpus_rows(db, project_id)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail=(
                "Project has no training rows yet — import data or "
                "generate synthetic rows first."
            ),
        )

    index_dir = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
    index_path = index_dir / "bm25_index.json"

    # Rebuild the index when the file is missing OR when the row
    # count has changed since the last build (cheap signal that rows
    # were added/removed — proper change detection is Phase 9b's
    # job, the preview just wants the latest possible answer).
    needs_rebuild = (
        not index_path.exists()
        or _index_row_count(index_path) != len(rows)
        or not qa_index_is_current(index_path)
    )
    if needs_rebuild:
        try:
            manifest = build_bm25_index(
                rows,
                recipe_id=recipe_id,
                output_dir=index_dir,
                corpus_source=QA_CORPUS_SOURCE,
            )
        except AutoRagUnavailable as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        manifest = {
            "index_path": str(index_path),
            "recipe_id": recipe_id,
            "text_keys": list(recommended_text_keys_for_recipe(recipe_id) or ()),
            "doc_count": len(rows),
            # avg_doc_length only surfaced for new builds; on cache
            # hit we'd need to re-read the file. Cheap to omit here.
            "avg_doc_length": None,
        }

    try:
        retrieved = retrieve(query, index_dir=index_dir, k=k)
    except AutoRagUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return {
        "project_id": project_id,
        "recipe_id": recipe_id,
        "query": query,
        "k": k,
        "index": manifest,
        "retrieved": retrieved,
    }


@router.get("/comparison")
async def get_auto_rag_comparison(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Phase 9d — return the cached auto-RAG comparison written by
    the harness's ``--project`` mode. Read-only; the inference run is
    expensive (~2 min on GB10) and out-of-band by design.

    Status codes:
      200 — at least one comparison cached on disk. Top-level
            ``summary`` / ``rows`` are the fine-tuned run's (``summary`` is
            null when only the base-model comparison exists); ``base`` is
            the base-model comparison or null. ``stale`` is true when the
            fine-tuned comparison was measured on an older run than
            ``latest_experiment_id``. Each comparison carries ``evidence``
            (rows better / worse / same, a 95% interval for the mean per-row
            change, and a verdict: better / worse / within_noise /
            too_few_rows) so a lift on a few rows isn't read as a result.
      400 — project missing a recipe (detail is a dict with
            ``error_code="RECIPE_REQUIRED"``) OR recipe is set but
            ineligible for auto-RAG (detail is a string).
      404 — project not found OR neither comparison cached yet (the
            ``detail`` includes the exact harness command to run).
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id")
    if not recipe_id:
        # Structured error so the panel can disambiguate "no recipe
        # set" (render the shared "pick a recipe first" CTA) from
        # "recipe set but not RAG-eligible" (silently hide — the
        # panel doesn't apply to this task shape, which is the
        # right signal for e.g. a classification project).
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "RECIPE_REQUIRED",
                "message": (
                    "Project has no task type selected — auto-RAG "
                    "comparison needs the task type to pick the "
                    "retrieval corpus shape."
                ),
            },
        )
    if recommended_text_keys_for_recipe(recipe_id) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Task type {recipe_id!r} has no auto-RAG corpus shape yet "
                f"(Phase 9a covers qa-sft only)."
            ),
        )
    auto_rag_dir = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
    # Two cached comparisons, one per model: the latest fine-tuned run
    # (comparison.json) and the untouched base model (comparison_base.json,
    # the RAG-first question). Either may be missing.
    fine_tuned = _read_cached_comparison(auto_rag_dir / "comparison.json")
    base = _read_cached_comparison(auto_rag_dir / "comparison_base.json")
    if fine_tuned is None and base is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No auto-RAG comparison cached yet for project "
                f"{project_id}. Run: ``python -m backend.scripts."
                f"auto_rag_ab --project {project_id}`` to generate "
                f"the comparison."
            ),
        )
    # The fine-tuned comparison is a cache for ONE run. Report the project's
    # latest trained run too, so the panel can say when the numbers shown are
    # for an older run than the one the user just trained.
    from app.services.eval_summary_service import _latest_trained_experiment_id

    latest_experiment_id = await _latest_trained_experiment_id(db, project_id)
    measured_experiment_id = (fine_tuned or {}).get("experiment_id")
    return {
        "project_id": project_id,
        "recipe_id": recipe_id,
        "latest_experiment_id": latest_experiment_id,
        "stale": bool(
            fine_tuned is not None
            and isinstance(measured_experiment_id, int)
            and latest_experiment_id is not None
            and measured_experiment_id != latest_experiment_id
        ),
        # Top-level fields describe the fine-tuned comparison (None / empty
        # when only the base-model one has been run).
        "cached_at": (fine_tuned or {}).get("cached_at"),
        "summary": (fine_tuned or {}).get("summary"),
        "rows": (fine_tuned or {}).get("rows") or [],
        "experiment_id": (fine_tuned or {}).get("experiment_id"),
        "base_model": (fine_tuned or base or {}).get("base_model"),
        "evidence": (fine_tuned or {}).get("evidence"),
        "base": base,
    }


def _read_cached_comparison(cache_path: Path) -> dict[str, Any] | None:
    """One cached comparison file as ``{cached_at, summary, rows,
    experiment_id, base_model}``; None when missing. An unreadable file is a
    503 — a half-written cache shouldn't look like "not run yet"."""
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cached comparison at {cache_path} is unreadable: {e}",
        ) from e
    rows = payload.get("rows") or []
    return {
        "cached_at": payload.get("cached_at"),
        "summary": payload.get("summary") or {},
        "rows": rows,
        "experiment_id": payload.get("experiment_id"),
        "base_model": payload.get("base_model"),
        "evidence": _lift_evidence(rows),
    }


def _lift_evidence(rows: list[Any]) -> dict[str, Any]:
    """How solid the with-RAG vs without-RAG difference is: rows better /
    worse / same and whether the mean per-row change is within noise.
    Computed from the cached per-row F1 pairs, so older caches get it too."""
    from app.services.paired_comparison_stats import paired_difference_evidence

    without: list[float] = []
    with_rag: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        off = (row.get("without_rag") or {}).get("f1") if isinstance(row.get("without_rag"), dict) else None
        on = (row.get("with_rag") or {}).get("f1") if isinstance(row.get("with_rag"), dict) else None
        if isinstance(off, (int, float)) and isinstance(on, (int, float)):
            without.append(float(off))
            with_rag.append(float(on))
    return paired_difference_evidence(without, with_rag)


def _index_row_count(index_path: Path) -> int:
    """Cheap doc_count read for the cache-validity check. Returns
    -1 on any read error so the caller treats it as "needs rebuild"
    rather than crashing on a malformed index file."""
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return -1
    return int(payload.get("doc_count") or 0)


# ─────────────────────────────────────────────────────────────────────
# POST /comparison/run — UI-triggered Job that produces comparison.json
# ─────────────────────────────────────────────────────────────────────


@router.post("/comparison/run", status_code=202)
async def run_auto_rag_comparison(
    project_id: int,
    model: Literal["fine_tuned", "base"] = Query(
        "fine_tuned",
        description=(
            "Which model to score with and without retrieval: the latest "
            "fine-tuned run (default), or the project's untouched base model "
            "(\"does retrieval alone help?\" — written to comparison_base.json)."
        ),
    ),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Hardening — spawn the auto-RAG comparison as a background Job
    (previously CLI-only via ``python -m backend.scripts.auto_rag_ab
    --project <id>``).

    The Job runner loads the project's latest COMPLETED experiment's
    model (or, with ``model=base``, the untouched base model — no trained
    run needed), runs inference twice over the val split (with-RAG /
    without-RAG), writes the per-row ``comparison.json`` (or
    ``comparison_base.json``) the existing
    ``GET /comparison`` endpoint reads, and publishes per-row progress
    into the notification bell ("scoring row 12/28 (with-RAG)").

    Status codes:
      202 — Job queued; body carries the Job stub. Frontend polls the
            bell + the comparison endpoint to pick up the result.
      400 — project missing a recipe / recipe not RAG-eligible.
      404 — project not found.
      409 — comparison Job already in flight for this project
            (idempotency: refuse rather than spawn a duplicate that
            would race the output file).
    """
    from datetime import datetime, timezone

    from sqlalchemy import select

    from app.models.job import Job, JobStatus
    from app.services.jobs_service import (
        JobProgressHandle,
        serialize_job,
        start_job,
    )

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id")
    if not recipe_id:
        raise HTTPException(
            status_code=400,
            detail="Project has no task type selected — auto-RAG comparison needs one.",
        )
    if recommended_text_keys_for_recipe(recipe_id) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Task type {recipe_id!r} has no auto-RAG corpus shape yet "
                f"(Phase 9a covers qa-sft only)."
            ),
        )

    # Idempotency — refuse if there's already a comparison Job for
    # this project in QUEUED or RUNNING. Two simultaneous runs would
    # race the comparison.json write + double the GPU load.
    in_flight_result = await db.execute(
        select(Job)
        .where(
            Job.kind == "auto_rag_comparison",
            Job.project_id == project_id,
            Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
        .order_by(Job.queued_at.desc())
        .limit(1)
    )
    in_flight = in_flight_result.scalar_one_or_none()
    if in_flight is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "AUTO_RAG_COMPARISON_ALREADY_RUNNING",
                "message": (
                    f"An auto-RAG comparison Job for project {project_id} "
                    f"is already in flight. Watch the notification bell "
                    f"for completion."
                ),
                "metadata": {
                    "existing_job_id": in_flight.id,
                    "existing_job_status": in_flight.status.value,
                    "existing_job_queued_at": (
                        in_flight.queued_at.isoformat() if in_flight.queued_at else None
                    ),
                },
            },
        )

    base_only = model == "base"

    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        import asyncio
        import time

        # The comparison work is GPU-heavy + uses sync code paths
        # (torch model load, sqlite3 read inside the script). Run it
        # on a worker thread; bridge the script's sync
        # progress_callback into JobProgressHandle.set_progress via
        # a shared mutable state + a polling drainer running on the
        # event loop.
        progress_state: dict[str, Any] = {
            "scored": 0,
            "total": 0,
            "condition": "without-RAG",
            "passes_done": 0,
        }

        def _sync_callback(scored: int, total: int, condition: str) -> None:
            progress_state["scored"] = scored
            progress_state["total"] = total
            # The without-RAG pass finishes first; once we see scored
            # reset back to 1 on the with-RAG pass, increment
            # passes_done so the overall fraction reflects 2 passes.
            if (
                progress_state["condition"] != condition
                and progress_state["condition"] == "without-RAG"
                and condition == "with-RAG"
            ):
                progress_state["passes_done"] = 1
            progress_state["condition"] = condition

        async def _drainer(stop: asyncio.Event) -> None:
            started = time.monotonic()
            while not stop.is_set():
                state = dict(progress_state)
                total = state["total"]
                scored = state["scored"]
                passes_done = state["passes_done"]
                condition = state["condition"]
                elapsed = int(time.monotonic() - started)
                if total > 0:
                    completed = passes_done * total + scored
                    overall_total = 2 * total
                    fraction = max(0.0, min(1.0, completed / overall_total))
                    msg = (
                        f"scoring row {scored}/{total} ({condition}) · "
                        f"pass {passes_done + 1}/2 · {elapsed}s elapsed"
                    )
                else:
                    fraction = None
                    msg = f"loading model · {elapsed}s elapsed"
                await handle.set_progress(fraction=fraction, message=msg)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass

        # Import inside the runner so a missing torch / peft install
        # (CPU-only dev box) fails inside the Job not at app boot.
        import sys as _sys

        backend_root = str(Path(__file__).resolve().parents[2])
        if backend_root not in _sys.path:
            _sys.path.insert(0, backend_root)
        from scripts.auto_rag_ab import run_project_comparison

        stop_event = asyncio.Event()
        drain_task = asyncio.create_task(_drainer(stop_event))
        try:
            payload = await asyncio.to_thread(
                run_project_comparison,
                project_id,
                progress_callback=_sync_callback,
                base_only=base_only,
            )
        finally:
            stop_event.set()
            try:
                await drain_task
            except Exception:  # noqa: BLE001 — drainer is best-effort
                pass

        summary = payload.get("summary") or {}
        # Pointers-only result so the Job row stays cheap; the
        # comparison.json on disk is the canonical full payload.
        return {
            "project_id": project_id,
            "model": "base" if base_only else "fine_tuned",
            "base_model": payload.get("base_model"),
            "experiment_id": payload.get("experiment_id"),
            "off_mean_f1": summary.get("off_mean_f1"),
            "on_mean_f1": summary.get("on_mean_f1"),
            "absolute_lift": summary.get("absolute_lift"),
            "relative_lift_pct": summary.get("relative_lift_pct"),
            "n_val_rows": summary.get("n_val_rows"),
            "comparison_path": str(
                settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
                / ("comparison_base.json" if base_only else "comparison.json")
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    job = await start_job(
        db,
        kind="auto_rag_comparison",
        title=(
            f"Auto-RAG comparison · base model · project #{project_id}"
            if base_only
            else f"Auto-RAG comparison · project #{project_id}"
        ),
        runner=_runner,
        project_id=project_id,
        params={
            "project_id": project_id,
            "recipe_id": recipe_id,
            "model": "base" if base_only else "fine_tuned",
        },
    )
    return serialize_job(job)

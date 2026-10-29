"""Synthetic data generation API routes."""

from typing import Any

from fastapi import Query, APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services import llm_budget_guard, llm_usage
from app.services.cloud_backend_pins import (
    CLOUD_PROVIDER_SECRET_COORDS,
    CloudPinError,
    resolve_cloud_backend,
)
from app.services.synthetic_service import (
    MAX_TOTAL_ROWS,
    PER_BATCH_CONVERSATION_CAP,
    PER_BATCH_ROW_CAP,
    generate_conversation_dialogues,
    generate_qa_pairs,
    generate_span_extraction_rows,
    get_synth_task_status,
    save_synthetic_batch,
    save_synthetic_conversation_batch,
    save_synthetic_span_batch,
    start_conversation_generation_task,
    start_qa_generation_task,
    start_span_generation_task,
)

router = APIRouter(prefix="/projects/{project_id}/synthetic", tags=["Synthetic"])


# Cloud provider → (secret_provider, key_name) of the saved API key; the
# shared map lives in ``cloud_backend_pins`` (the flows use it too).
_CLOUD_PROVIDER_SECRET_COORDS = CLOUD_PROVIDER_SECRET_COORDS


# Curated catalog of cloud models the synth panel offers. Mirrors the
# gold-generate flow's known-good defaults so users see the same
# names in both surfaces. Frontend reads this via
# ``/synthetic/backends/cloud/models``.
_CLOUD_MODELS_CATALOG: dict[str, list[dict[str, str]]] = {
    "openai": [
        {"id": "gpt-4o-mini", "label": "GPT-4o mini (fast + cheap)"},
        {"id": "gpt-4o", "label": "GPT-4o (flagship)"},
        {"id": "gpt-4.1", "label": "GPT-4.1"},
        {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini"},
    ],
    "anthropic": [
        {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5 (fast + cheap)"},
        {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6"},
        {"id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5"},
        {"id": "claude-opus-4-7", "label": "Claude Opus 4.7"},
    ],
    "deepseek": [
        {"id": "deepseek-chat", "label": "Deepseek V3 chat (cheap)"},
        {"id": "deepseek-reasoner", "label": "Deepseek R1 reasoner"},
    ],
}


async def _with_budget_warning(db: AsyncSession, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Add ``budget_warning`` to a Job-start response when the project's
    cloud-LLM spend this month is already over its budget. Advisory: the Job
    has started either way (the user decides)."""
    from app.services.llm_spend_service import budget_warning

    warning = await budget_warning(db, project_id)
    return {**payload, "budget_warning": warning} if warning else payload


def _command_paused() -> bool:
    from app.services.synth_backends.command import is_paused

    return is_paused()


async def _resolve_cloud_backend_override(
    db: AsyncSession, project_id: int, backend_pin: str | None,
):
    """``cloud:<provider>:<model>`` → a ``CloudLlmBackend`` with the project's
    saved key (``cloud_backend_pins.resolve_cloud_backend``); None for other
    pins. 400 for a malformed pin, 402 when no key is saved."""
    try:
        backend = await resolve_cloud_backend(db, project_id, backend_pin)
    except CloudPinError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    if backend is not None:
        # Metered backend: 402 ``llm_budget_exceeded`` when a monthly cap is reached.
        await llm_budget_guard.assert_can_start(project_id)
    return backend


class GenerateRequest(BaseModel):
    source_text: str = Field(..., min_length=10)
    num_pairs: int = Field(5, ge=1, le=50)
    api_url: str = ""
    api_key: str = ""
    model_name: str = "llama3"


class GenerateAsyncRequest(BaseModel):
    """Body for the long-running batched QA-pair generator.

    Mirrors ``GenerateSpanAsyncRequest`` — single-call ``/generate``
    is capped at 50 pairs per LLM round-trip, this endpoint chunks
    server-side into ``PER_BATCH_ROW_CAP`` calls and (when
    ``use_all_chunks`` is set) feeds each batch a fresh random sample
    of the project's cleaned chunks."""

    target_rows: int = Field(..., ge=1, le=MAX_TOTAL_ROWS)
    api_url: str = ""
    api_key: str = ""
    model_name: str = "llama3"
    use_all_chunks: bool = Field(
        default=True,
        description=(
            "When true, source text for each batch is a fresh random "
            "sample (~4–8k tokens) of the project's cleaned chunks. "
            "When false, ``source_text`` is reused verbatim for every "
            "batch."
        ),
    )
    source_text: str = ""


class GenerateConversationAsyncRequest(BaseModel):
    """Body for the long-running batched conversation generator.

    Conversations are heavier than single QA pairs so the per-batch
    cap is smaller (``PER_BATCH_CONVERSATION_CAP``)."""

    target_rows: int = Field(..., ge=1, le=MAX_TOTAL_ROWS)
    min_turns: int = Field(3, ge=1, le=20)
    max_turns: int = Field(5, ge=1, le=20)
    api_url: str = ""
    api_key: str = ""
    model_name: str = "llama3"
    use_all_chunks: bool = Field(default=True)
    source_text: str = ""


class SaveBatchRequest(BaseModel):
    pairs: list[dict]
    min_confidence: float = Field(0.4, ge=0, le=1.0)


class GenerateConversationRequest(BaseModel):
    source_text: str = Field(..., min_length=10)
    num_dialogues: int = Field(3, ge=1, le=20)
    min_turns: int = Field(3, ge=1, le=20)
    max_turns: int = Field(5, ge=1, le=20)
    api_url: str = ""
    api_key: str = ""
    model_name: str = "llama3"


class SaveConversationBatchRequest(BaseModel):
    conversations: list[dict]
    min_confidence: float = Field(0.4, ge=0, le=1.0)


@router.post("/generate")
async def generate(
    project_id: int,
    req: GenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate synthetic Q&A pairs from source text using teacher model."""
    try:
        pairs = await generate_qa_pairs(
            db, project_id, req.source_text, req.num_pairs, req.api_url, req.api_key, req.model_name
        )
        return {"pairs": pairs, "count": len(pairs)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")


@router.post("/save")
async def save_batch(
    project_id: int,
    req: SaveBatchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Save approved synthetic pairs to the dataset."""
    result = await save_synthetic_batch(db, project_id, req.pairs, req.min_confidence)
    return result


@router.post("/generate-conversations")
async def generate_conversations(
    project_id: int,
    req: GenerateConversationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate multi-turn synthetic conversations from source text."""
    try:
        conversations = await generate_conversation_dialogues(
            db=db,
            project_id=project_id,
            source_text=req.source_text,
            num_dialogues=req.num_dialogues,
            min_turns=req.min_turns,
            max_turns=req.max_turns,
            api_url=req.api_url,
            api_key=req.api_key,
            model_name=req.model_name,
        )
        return {"conversations": conversations, "count": len(conversations)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Conversation generation failed: {str(e)}")


@router.post("/save-conversations")
async def save_conversations(
    project_id: int,
    req: SaveConversationBatchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Save approved synthetic conversations to the synthetic dataset."""
    result = await save_synthetic_conversation_batch(
        db,
        project_id,
        req.conversations,
        req.min_confidence,
    )
    return result


class GenerateSpanRequest(BaseModel):
    source_text: str = Field(..., min_length=10)
    num_rows: int = Field(5, ge=1, le=50)
    entity_types: list[str] = Field(default_factory=list)
    api_url: str = ""
    api_key: str = ""
    model_name: str = "llama3"
    # The prompt (types + the project's conventions) is refused when it
    # carries a benchmark gold value (PROMPT_CARRIES_BENCHMARK_VALUES).
    ignore_benchmark_values: bool = False


async def _refuse_contaminated_span_prompt(
    db: AsyncSession, project_id: int, entity_types: list[str], focus_types: list[str], override: bool,
) -> None:
    from app.services.prompt_contamination_service import (
        PromptCarriesBenchmarkValues,
        http_detail,
        refuse_if_contaminated,
        span_prompt_fields,
    )

    fields = await span_prompt_fields(db, project_id, entity_types, focus_types)
    try:
        await refuse_if_contaminated(db, project_id, fields, override=override)
    except PromptCarriesBenchmarkValues as exc:
        raise HTTPException(400, http_detail(exc)) from exc


class PromptCheckRequest(BaseModel):
    """Dry run of the benchmark-value gate: ``fields`` = the prompt pieces
    (``{name: text | [texts]}``); ``include_conventions`` adds the project's
    span conventions (definitions + custom-pattern descriptions), which
    every span-synthesis prompt carries."""

    fields: dict[str, str | list[str]] = Field(default_factory=dict)
    include_conventions: bool = False


@router.post("/prompt-check")
async def prompt_check(project_id: int, req: PromptCheckRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Does this prompt text carry a value from a benchmark's gold rows?
    ``{clean, offending: [{value, type, benchmarks, fields}], offending_total,
    values_checked, benchmarks, fields}`` — nothing is generated."""
    from app.services.prompt_contamination_service import check_prompt, span_prompt_fields

    fields: dict[str, Any] = dict(req.fields)
    if req.include_conventions:
        conventions = await span_prompt_fields(db, project_id, [], [])
        fields.update({k: v for k, v in conventions.items() if k.startswith(("definition:", "pattern:"))})
    return await check_prompt(db, project_id, fields)


class SaveSpanBatchRequest(BaseModel):
    rows: list[dict]
    min_confidence: float = Field(0.4, ge=0, le=1.0)


@router.post("/generate-spans")
async def generate_spans(
    project_id: int,
    req: GenerateSpanRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate `{text, entities: [...]}` rows for PII / NER /
    structured-extraction span_set training. Uses the teacher model
    when configured, falls back to a regex-based heuristic on the
    source text otherwise."""
    await _refuse_contaminated_span_prompt(db, project_id, req.entity_types, [], req.ignore_benchmark_values)
    try:
        rows = await generate_span_extraction_rows(
            db,
            project_id,
            req.source_text,
            req.num_rows,
            req.entity_types or None,
            req.api_url,
            req.api_key,
            req.model_name,
        )
        return {"rows": rows, "count": len(rows)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Span generation failed: {str(e)}")


class GenerateSpanAsyncRequest(BaseModel):
    """Request body for the long-running batched span generator.

    The sync ``/generate-spans`` endpoint is fine for ≤50 rows in one
    teacher call. For 2k+ row jobs the work is batched server-side in
    ``PER_BATCH_ROW_CAP``-row chunks, each fed a fresh randomized sample
    of the project's cleaned chunks when ``use_all_chunks`` is set.
    """

    target_rows: int = Field(..., ge=1, le=MAX_TOTAL_ROWS)
    entity_types: list[str] = Field(default_factory=list)
    api_url: str = ""
    api_key: str = ""
    model_name: str = "llama3"
    # A synth backend name / pin ("ollama", "cloud:anthropic:…") used as the
    # teacher instead of api_url — the native Ollama backend turns thinking
    # off, which a thinking model needs or it returns empty content.
    backend: str | None = None
    # A refine pass: what each snippet must contain (a subset of entity_types,
    # which stays the full vocabulary that gets labelled).
    focus_types: list[str] = Field(default_factory=list)
    ignore_benchmark_values: bool = False
    use_all_chunks: bool = Field(
        default=True,
        description=(
            "When true, source text for each batch is a fresh random "
            "sample (~4–8k tokens) of the project's cleaned chunks. "
            "When false, ``source_text`` is reused verbatim for every "
            "batch."
        ),
    )
    source_text: str = ""


async def _autosave_legacy_synth_rows(
    *,
    project_id: int,
    kind: str,
    task_id: str,
    rows: list[dict],
) -> dict:
    """Hardening fix — persist a legacy synthetic-task's accumulated
    rows to ``data/projects/{id}/synthetic/synthetic.jsonl`` so the
    user doesn't have to discover the in-page "Save" button. The
    rows land in the same envelope playbook output uses
    (``review_status="pending"`` + legacy ``status="accepted"``
    compat field) so the existing SynthReviewQueue picks them up.

    Why this matters: the legacy task framework was always
    in-memory-only until the user clicked Save. After the H2
    bridge made the bell show "Done" on task completion, users
    reasonably read that as "rows are saved" — they weren't, and
    a server restart wiped them. This auto-save closes that gap.

    Returns a small report dict with ``rows_saved`` + ``file_path``
    for the Job's result payload.
    """
    import json

    from app.config import settings
    from app.database import async_session_factory
    from app.services.synthetic_service import (
        get_or_create_synthetic_dataset,
    )

    if not rows:
        return {"rows_saved": 0, "file_path": None}

    synth_dir = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    file_path = synth_dir / "synthetic.jsonl"

    # Find the next id by reading the last id in the file. Cheap
    # enough for any reasonable corpus size.
    next_id = 1
    if file_path.exists():
        with file_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj.get("id"), int):
                        next_id = max(next_id, int(obj["id"]) + 1)
                except json.JSONDecodeError:
                    continue

    from app.services import jobs_service

    job_handle = jobs_service.current_job_handle.get()
    saved = 0
    with file_path.open("a", encoding="utf-8") as f:
        for row in rows:
            confidence = row.get("confidence") if isinstance(row, dict) else None
            if not isinstance(confidence, (int, float)):
                confidence = 0.7  # neutral default for legacy rows
            payload = {k: v for k, v in row.items() if k != "confidence"}
            entry = {
                "id": next_id,
                **payload,
                "synth_confidence": float(confidence),
                "synth_source": f"legacy:{kind}:task={task_id[:8]}",
                "review_status": "pending",
                # Legacy field kept so old readers don't choke.
                "status": "accepted",
                # Provenance: the shadow Job that saved the rows + the model that wrote them.
                "synth_job_id": job_handle.job_id if job_handle is not None else None,
            }
            legacy_model = row.get("model") if isinstance(row, dict) else None
            if isinstance(legacy_model, str) and legacy_model.strip():
                from app.services.synth_playbook_service import safe_backend_label

                entry["synth_backend"] = safe_backend_label(f"legacy:{legacy_model.strip()}")
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            next_id += 1
            saved += 1

    # Bump the Dataset.record_count + ensure file_path is stamped.
    async with async_session_factory() as ds_db:
        ds = await get_or_create_synthetic_dataset(ds_db, project_id)
        ds.record_count = (ds.record_count or 0) + saved
        if not ds.file_path:
            ds.file_path = str(file_path)
        await ds_db.commit()

    return {"rows_saved": saved, "file_path": str(file_path)}


async def _spawn_legacy_synth_shadow_job(
    *,
    project_id: int,
    task_id: str,
    kind: str,
    title: str,
    params: dict,
) -> None:
    """Hardening Phase H2 — bridge a legacy synthetic-task into the
    new Jobs framework so the top-bar notification bell surfaces it
    alongside playbook runs.

    Spawns a Job whose runner polls the legacy
    ``_SYNTHETIC_TASKS[task_id]`` registry every 2 seconds and
    mirrors ``batches_done / batches_total`` into ``Job.progress``,
    ``status`` into the Job's terminal transition, and
    ``rows_so_far`` into the progress message. **Auto-saves rows**
    to ``synthetic.jsonl`` on completion (review_status=pending) so
    the user doesn't have to find the in-page Save button — bell
    saying "Done" now means rows are actually persisted.

    Best-effort — if the Jobs framework can't be reached (unlikely)
    the legacy task keeps running unaffected; the user just doesn't
    see it in the bell.
    """
    import asyncio
    import time

    from app.database import async_session_factory
    from app.services.jobs_service import (
        JobProgressHandle,
        start_job,
    )
    from app.services.synthetic_service import get_synth_task_status

    async def _runner(handle: JobProgressHandle) -> dict:
        started = time.monotonic()
        last_status: str = "pending"
        while True:
            task_obj = get_synth_task_status(task_id)
            if task_obj is None:
                raise RuntimeError(f"legacy synth task {task_id} disappeared")
            # get_synth_task_status returns the task object itself
            # (SyntheticQaTask / SyntheticSpanTask / SyntheticConver-
            # sationTask), not a dict. Use ``.to_dict()`` to get the
            # uniform shape — each task class implements it.
            snapshot = task_obj.to_dict()
            last_status = str(snapshot.get("status") or "pending")
            batches_done = int(snapshot.get("batches_done") or 0)
            batches_total = int(snapshot.get("batches_total") or 0)
            rows_so_far = int(snapshot.get("rows_so_far") or 0)
            target_rows = int(snapshot.get("target_rows") or 0)
            elapsed = int(time.monotonic() - started)
            if batches_total > 0:
                fraction = max(0.0, min(1.0, batches_done / batches_total))
            elif target_rows > 0:
                fraction = max(0.0, min(1.0, rows_so_far / target_rows))
            else:
                fraction = None
            msg_parts: list[str] = []
            if batches_total > 0:
                msg_parts.append(f"batch {batches_done}/{batches_total}")
            if rows_so_far > 0:
                msg_parts.append(f"{rows_so_far} rows so far")
            msg_parts.append(f"{elapsed}s elapsed")
            await handle.set_progress(
                fraction=fraction,
                message=" · ".join(msg_parts),
            )
            if last_status in ("completed", "failed"):
                if last_status == "failed":
                    raise RuntimeError(
                        snapshot.get("error") or "legacy synth task failed"
                    )
                # 0-rows-completed is a silent failure (Ollama
                # returned junk, every batch parsed empty, etc.) —
                # surface it as FAILED so the bell stops misleading
                # the user with "Done" + 0 rows.
                if rows_so_far == 0:
                    raise RuntimeError(
                        f"Legacy synth task completed but produced 0 rows "
                        f"({batches_done}/{batches_total} batches reported "
                        f"done). Likely causes: (1) LLM returned text that "
                        f"didn't parse as JSON, (2) source text was empty / "
                        f"too short for the model, (3) backend connection "
                        f"silently dropped between batches. Check the "
                        f"server logs for the backend's raw response."
                    )
                # Auto-save the rows to synthetic.jsonl so the user
                # doesn't have to find the in-page Save button. Bell
                # saying "Done" now actually means the rows are on
                # disk + queued for review.
                task_rows = list(snapshot.get("rows") or [])
                save_report = await _autosave_legacy_synth_rows(
                    project_id=project_id,
                    kind=kind,
                    task_id=task_id,
                    rows=task_rows,
                )
                # Final progress message reflects the persistent
                # outcome rather than the heartbeat.
                await handle.set_progress(
                    fraction=1.0,
                    message=(
                        f"Saved {save_report['rows_saved']} rows to "
                        f"synthetic.jsonl · {batches_done}/{batches_total} "
                        f"batches"
                    ),
                )
                return {
                    "task_id": task_id,
                    "rows_generated": rows_so_far,
                    "rows_saved": save_report["rows_saved"],
                    "file_path": save_report["file_path"],
                    "batches_done": batches_done,
                    "batches_total": batches_total,
                }
            # Honor cooperative cancellation — flag the legacy task
            # as we can't directly kill its background thread, but
            # mark our shadow Job as cancelled.
            if await handle.check_cancelled():
                return {
                    "task_id": task_id,
                    "rows_generated": rows_so_far,
                    "batches_done": batches_done,
                    "batches_total": batches_total,
                    "note": "cancelled_from_ui_legacy_task_may_continue",
                }
            await asyncio.sleep(2.0)

    async with async_session_factory() as db:
        await start_job(
            db,
            kind=kind,
            title=title,
            runner=_runner,
            project_id=project_id,
            params=params,
        )


@router.post("/generate-spans-async", status_code=202)
async def generate_spans_async(
    project_id: int,
    req: GenerateSpanAsyncRequest,
    db: AsyncSession = Depends(get_db),
):
    """Kick off a batched span-generation job. Returns immediately with
    a ``task_id``; clients poll ``GET /synthetic/tasks/{task_id}`` for
    progress + accumulated rows.

    Hardening Phase H2 — also spawns a shadow Job that mirrors the
    legacy task's progress into the top-bar notification bell.
    """
    await _refuse_contaminated_span_prompt(db, project_id, req.entity_types, req.focus_types, req.ignore_benchmark_values)
    try:
        task = start_span_generation_task(
            project_id=project_id,
            target_rows=req.target_rows,
            entity_types=req.entity_types,
            api_url=req.api_url,
            api_key=req.api_key,
            model_name=req.model_name,
            backend=req.backend,
            focus_types=req.focus_types,
            use_all_chunks=req.use_all_chunks,
            source_text=req.source_text,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    await _spawn_legacy_synth_shadow_job(
        project_id=project_id,
        task_id=task.task_id,
        kind="synth_legacy_spans",
        title=(
            f"Span rows for {', '.join(req.entity_types[:6])}{'…' if len(req.entity_types) > 6 else ''} · {req.target_rows} rows"
            if req.entity_types else f"Span rows · {req.target_rows} rows"
        ),
        params={"target_rows": req.target_rows, "task_id": task.task_id},
    )
    return {
        "task_id": task.task_id,
        "status": task.status,
        "target_rows": task.target_rows,
        "batches_total": (req.target_rows + PER_BATCH_ROW_CAP - 1) // PER_BATCH_ROW_CAP,
    }


@router.post("/generate-async", status_code=202)
async def generate_qa_async(
    project_id: int,
    req: GenerateAsyncRequest,
):
    """Kick off a batched QA-pair generation job. Returns immediately
    with a ``task_id``; clients poll ``GET /synthetic/tasks/{task_id}``
    for progress + accumulated pairs.

    Lifts the 50-pair cap that the sync ``/generate`` endpoint enforces:
    the server batches into ``PER_BATCH_ROW_CAP`` chunks and (when
    ``use_all_chunks`` is set) samples fresh source text per batch.

    Hardening Phase H2 — also spawns a shadow Job mirroring progress
    into the top-bar notification bell.
    """
    try:
        task = start_qa_generation_task(
            project_id=project_id,
            target_rows=req.target_rows,
            api_url=req.api_url,
            api_key=req.api_key,
            model_name=req.model_name,
            use_all_chunks=req.use_all_chunks,
            source_text=req.source_text,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    await _spawn_legacy_synth_shadow_job(
        project_id=project_id,
        task_id=task.task_id,
        kind="synth_legacy_qa",
        title=f"Synth (legacy QA) · {req.target_rows} rows",
        params={"target_rows": req.target_rows, "task_id": task.task_id},
    )
    return {
        "task_id": task.task_id,
        "status": task.status,
        "target_rows": task.target_rows,
        "batches_total": (req.target_rows + PER_BATCH_ROW_CAP - 1) // PER_BATCH_ROW_CAP,
    }


@router.post("/generate-conversations-async", status_code=202)
async def generate_conversations_async(
    project_id: int,
    req: GenerateConversationAsyncRequest,
):
    """Kick off a batched multi-turn conversation generation job.
    Conversations are heavier than QA pairs, so the per-batch cap is
    ``PER_BATCH_CONVERSATION_CAP`` rather than ``PER_BATCH_ROW_CAP``.

    Hardening Phase H2 — also spawns a shadow Job mirroring progress
    into the top-bar notification bell.
    """
    try:
        task = start_conversation_generation_task(
            project_id=project_id,
            target_rows=req.target_rows,
            min_turns=req.min_turns,
            max_turns=req.max_turns,
            api_url=req.api_url,
            api_key=req.api_key,
            model_name=req.model_name,
            use_all_chunks=req.use_all_chunks,
            source_text=req.source_text,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    await _spawn_legacy_synth_shadow_job(
        project_id=project_id,
        task_id=task.task_id,
        kind="synth_legacy_conversations",
        title=f"Synth (legacy conversations) · {req.target_rows} rows",
        params={"target_rows": req.target_rows, "task_id": task.task_id},
    )
    return {
        "task_id": task.task_id,
        "status": task.status,
        "target_rows": task.target_rows,
        "batches_total": (
            (req.target_rows + PER_BATCH_CONVERSATION_CAP - 1)
            // PER_BATCH_CONVERSATION_CAP
        ),
    }


@router.get("/tasks/{task_id}")
async def get_synthetic_task(project_id: int, task_id: str):
    """Read the live state of any batched synthetic-generation job
    (span, qa, or conversation). Returns ``rows`` once the task has
    completed (or partial rows while still running). The ``task_kind``
    field on the response disambiguates the row shape for the client."""
    task = get_synth_task_status(task_id)
    if task is None:
        raise HTTPException(404, f"Synthetic task {task_id!r} not found")
    if task.project_id != project_id:
        raise HTTPException(404, f"Synthetic task {task_id!r} not found")
    return task.to_dict()


@router.post("/save-spans")
async def save_spans(
    project_id: int,
    req: SaveSpanBatchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Save approved span-extraction rows to the synthetic dataset."""
    result = await save_synthetic_span_batch(
        db, project_id, req.rows, req.min_confidence
    )
    return result


# ─────────────────────────────────────────────────────────────────────
# USER-SUCCESS Epic 2 — playbook-driven synth.
# ─────────────────────────────────────────────────────────────────────


class RunPlaybookRequest(BaseModel):
    mode: str = Field(..., description="Synth mode, e.g. 'positives_paraphrase'.")
    target_count: int = Field(30, ge=1, le=500)
    target_class: str | None = Field(default=None)
    backend: str | None = Field(default=None, description="Optional backend pin, e.g. 'ollama:llama3.1:8b'.")
    params: dict | None = Field(
        default=None,
        description=(
            "Per-mode parameters handed to the playbook (ctx['params']). code-findings seeded_defect: "
            "label_id (required, a project-taxonomy id), attrs, context_dataset_id + context_column, "
            "context_sample_size. code-remediation remediation_from_findings: findings_dataset_id "
            "(findings rows, fix pairs or remediation rows; not an answer key), top_k (1-10), "
            "retrieve_passages; code_examples: per_chunk (1-5). Code and document playbooks: source_ids (1-100 data-source ids of this project; seeds only from those sources), seed_path_include / seed_path_exclude (glob lists on the chunk path), seed_relevance_terms (+ seed_relevance_scope chunk|file, seed_relevance_match path_or_text|text, relevance_exempt_source_ids). generic-sft documents_multi_turn: turns_min / "
            "turns_max (2-8), dialogues_per_passage_group (1-5), passages_per_dialogue (1-3)."
        ),
    )
    use_batch: bool = Field(
        default=False,
        description=(
            "Anthropic Message Batches: the run's calls go as ONE batch per stage at half the list price "
            "(chunks not yet checkpointed; scenario_grid's scenario and plan stages); results take minutes "
            "to hours, and calls the batch doesn't answer are retried directly. cloud:anthropic:<model> "
            "backends only (400 invalid_playbook_param otherwise). The dry run accepts and ignores it "
            "(always one direct call)."
        ),
    )


def _check_use_batch(req: "RunPlaybookRequest", *, async_job: bool = True) -> None:
    """``use_batch`` needs an Anthropic cloud pin and ``?async_job=true`` — a batch can take
    hours, too long to hold an HTTP request open (400 ``invalid_playbook_param``)."""
    if req.use_batch and not (req.backend or "").startswith("cloud:anthropic:"):
        raise HTTPException(400, {
            "error_code": "invalid_playbook_param",
            "message": (
                "use_batch uses Anthropic's Message Batches API — pick a cloud:anthropic:<model> backend, "
                "or turn batch off."
            ),
            "metadata": {"param": "use_batch", "backend": req.backend},
        })
    if req.use_batch and not async_job:
        raise HTTPException(400, {
            "error_code": "invalid_playbook_param",
            "message": "use_batch runs as a background Job — add ?async_job=true (a batch can take hours).",
            "metadata": {"param": "use_batch", "async_job": False},
        })


def _check_remediation_params(params: dict, *, require_dataset_kind: bool) -> None:
    """Range-check the code-remediation playbook params (400 ``invalid_playbook_param``)."""

    def bad(name: str, message: str) -> HTTPException:
        return HTTPException(400, {
            "error_code": "invalid_playbook_param", "message": message, "metadata": {"param": name, "value": params.get(name)},
        })

    def whole(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else (
            int(value) if isinstance(value, str) and value.strip().isdigit() else None
        )

    if require_dataset_kind:
        if params.get("findings_dataset_id") is not None and whole(params["findings_dataset_id"]) is None:
            raise bad("findings_dataset_id", "params.findings_dataset_id must be a dataset id.")
        if params.get("top_k") is not None:
            top_k = whole(params["top_k"])
            if top_k is None or not 1 <= top_k <= 10:
                raise bad("top_k", "params.top_k must be a whole number from 1 to 10.")
        if params.get("retrieve_passages") is not None and not isinstance(params["retrieve_passages"], bool):
            raise bad("retrieve_passages", "params.retrieve_passages must be true or false.")
    elif params.get("per_chunk") is not None:
        per_chunk = whole(params["per_chunk"])
        if per_chunk is None or not 1 <= per_chunk <= 5:
            raise bad("per_chunk", "params.per_chunk must be a whole number from 1 to 5.")


async def _check_scenario_grid_params(db: AsyncSession, project_id: int, params: dict) -> None:
    """Range-check the structured-plan scenario_grid params and parse the matrix (400
    ``invalid_playbook_param``); a project with no saved schema is 400 ``structured_plan_schema_missing``."""
    from app.services.synth_playbooks._structured_plan_common import check_params

    problem = check_params(params)
    if problem is not None:
        name, message = problem
        raise HTTPException(400, {
            "error_code": "invalid_playbook_param", "message": message,
            "metadata": {"param": name, "value": None if name == "matrix_text" else params.get(name)},
        })
    from app.models.project import Project
    from app.services.synth_playbook_service import NO_SCHEMA_MESSAGE, structured_plan_bundle

    project = await db.get(Project, project_id)
    recipe = (project.selected_recipe or {}).get("recipe_id") if project is not None else None
    if project is not None and recipe == "structured-plan" and structured_plan_bundle(project) is None:
        raise HTTPException(400, {
            "error_code": "structured_plan_schema_missing", "message": NO_SCHEMA_MESSAGE, "metadata": {},
        })


async def _check_source_ids(db: AsyncSession, project_id: int, raw: Any, param: str = "source_ids") -> list[int]:
    """``params.source_ids``: 1-100 data-source ids, all of THIS project (400 ``invalid_playbook_param``)."""
    from sqlalchemy import select

    from app.models.data_source import DataSource

    def bad(message: str) -> HTTPException:
        return HTTPException(400, {
            "error_code": "invalid_playbook_param", "message": message,
            "metadata": {"param": param, "value": raw},
        })

    if not isinstance(raw, list) or not raw:
        raise bad(f"params.{param} must be a non-empty list of data-source ids.")
    if len(raw) > 100:
        raise bad(f"params.{param} takes at most 100 data-source ids.")
    ids: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, str)) or not str(item).strip().lstrip("-").isdigit():
            raise bad(f"params.{param} must be a list of whole-number data-source ids.")
        ids.append(int(item))
    ids = list(dict.fromkeys(ids))
    found = set((await db.execute(
        select(DataSource.id).where(DataSource.project_id == project_id, DataSource.id.in_(ids))
    )).scalars().all())
    missing = [i for i in ids if i not in found]
    if missing:
        raise bad(f"params.{param} has ids that are not data sources of this project: {missing}.")
    return ids


async def _check_seed_filter_params(db: AsyncSession, project_id: int, out: dict) -> None:
    """``params.seed_path_include`` / ``seed_path_exclude`` (globs on the chunk's path), ``seed_relevance_terms``,
    ``seed_relevance_scope`` (``chunk`` | ``file``), ``seed_relevance_match`` (``path_or_text`` | ``text``) and ``relevance_exempt_source_ids``. Lists of strings with capped
    sizes; a glob that does not compile is rejected: 400 ``invalid_playbook_param`` (``metadata.param`` names it)."""
    import fnmatch
    import re

    from app.services import synth_playbook_service as sps

    def bad(param: str, message: str) -> HTTPException:
        return HTTPException(400, {
            "error_code": "invalid_playbook_param", "message": message,
            "metadata": {"param": param, "value": out.get(param)},
        })

    for key in ("seed_path_include", "seed_path_exclude", "seed_relevance_terms",
                "seed_relevance_scope", "seed_relevance_match", "relevance_exempt_source_ids"):
        if key in out and out[key] is None:
            del out[key]  # null = unset, like an empty list
    for param in ("seed_path_include", "seed_path_exclude"):
        if param not in out:
            continue
        raw = out[param]
        if not isinstance(raw, list) or not all(isinstance(g, str) for g in raw):
            raise bad(param, f"params.{param} must be a list of glob strings.")
        globs = [g.strip() for g in raw]
        if len(globs) > sps.MAX_SEED_GLOBS:
            raise bad(param, f"params.{param} takes at most {sps.MAX_SEED_GLOBS} globs.")
        for g in globs:
            if not g or len(g) > sps.MAX_SEED_GLOB_LEN or "\x00" in g:
                raise bad(param, f"params.{param}: each glob must be 1-{sps.MAX_SEED_GLOB_LEN} characters with no NUL ({g[:40]!r}).")
            try:
                re.compile(fnmatch.translate(g))
            except (re.error, RecursionError):
                raise bad(param, f"params.{param}: {g!r} is not a valid glob.") from None
        out[param] = list(dict.fromkeys(globs))
    if "seed_relevance_terms" in out:
        raw = out["seed_relevance_terms"]
        if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
            raise bad("seed_relevance_terms", "params.seed_relevance_terms must be a list of strings.")
        terms = [t.strip() for t in raw if t.strip()]
        if len(terms) > sps.MAX_SEED_TERMS:
            raise bad("seed_relevance_terms", f"params.seed_relevance_terms takes at most {sps.MAX_SEED_TERMS} terms.")
        if any(len(t) > sps.MAX_SEED_TERM_LEN for t in terms):
            raise bad("seed_relevance_terms", f"params.seed_relevance_terms: each term is at most {sps.MAX_SEED_TERM_LEN} characters.")
        out["seed_relevance_terms"] = list(dict.fromkeys(terms))
    if "seed_relevance_scope" in out and out["seed_relevance_scope"] not in sps.SEED_RELEVANCE_SCOPES:
        raise bad("seed_relevance_scope", "params.seed_relevance_scope must be 'chunk' or 'file'.")
    if "seed_relevance_match" in out and out["seed_relevance_match"] not in sps.SEED_RELEVANCE_MATCHES:
        raise bad("seed_relevance_match", "params.seed_relevance_match must be 'path_or_text' or 'text'.")
    if "relevance_exempt_source_ids" in out and out["relevance_exempt_source_ids"] == []:
        pass
    elif "relevance_exempt_source_ids" in out:
        out["relevance_exempt_source_ids"] = await _check_source_ids(
            db, project_id, out["relevance_exempt_source_ids"], "relevance_exempt_source_ids")


async def _checked_playbook_params(db: AsyncSession, project_id: int, mode: Any, params: dict | None) -> dict:
    """Validate ``params`` before any work starts.

    * A context dataset must not be an answer-key (gold dev/test) dataset: 400
      ``context_dataset_not_allowed``.
    * A seeded-defect run needs ``label_id`` (400 ``missing_label_id``); a cluster-targeted run
      validates it when given. With a project taxonomy it must resolve (400 ``unknown_label``) and
      the resolved canonical id replaces the caller's spelling; without a taxonomy any non-empty
      label is accepted as a free label (trimmed).
    """
    out = dict(params or {})
    if "source_ids" in out:
        out["source_ids"] = await _check_source_ids(db, project_id, out["source_ids"])
    await _check_seed_filter_params(db, project_id, out)
    raw_dataset = out.get("context_dataset_id")
    if raw_dataset is not None:
        from app.models.dataset import Dataset, DatasetType

        try:
            dataset = await db.get(Dataset, int(raw_dataset))
        except (TypeError, ValueError):
            dataset = None
        if dataset is not None and dataset.project_id == project_id and dataset.dataset_type in (
            DatasetType.GOLD_DEV, DatasetType.GOLD_TEST,
        ):
            raise HTTPException(400, {
                "error_code": "context_dataset_not_allowed",
                "message": (
                    "The answer key cannot be used as a context dataset: its rows are held out for "
                    "evaluation. Pick a raw, training or other data dataset."
                ),
                "metadata": {"dataset_id": dataset.id, "dataset_type": getattr(dataset.dataset_type, "value", str(dataset.dataset_type))},
            })
    mode_value = getattr(mode, "value", mode)
    if mode_value == "remediation_from_findings":
        _check_remediation_params(out, require_dataset_kind=True)
        raw_findings = out.get("findings_dataset_id")
        if raw_findings is not None:
            from app.models.dataset import Dataset, DatasetType

            try:
                findings_ds = await db.get(Dataset, int(raw_findings))
            except (TypeError, ValueError):
                findings_ds = None
            if findings_ds is not None and findings_ds.project_id == project_id and findings_ds.dataset_type in (
                DatasetType.GOLD_DEV, DatasetType.GOLD_TEST,
            ):
                raise HTTPException(400, {
                    "error_code": "findings_dataset_not_allowed",
                    "message": (
                        "The answer key cannot be used as the findings dataset: its rows are held out for "
                        "evaluation. Pick a raw or training dataset (answer-key dev rows are used as seeds automatically)."
                    ),
                    "metadata": {"dataset_id": findings_ds.id},
                })
        return out
    if mode_value == "code_examples":
        _check_remediation_params(out, require_dataset_kind=False)
        return out
    if mode_value == "scenario_grid":
        await _check_scenario_grid_params(db, project_id, out)
        return out
    if mode_value == "documents_multi_turn":
        from app.services.synth_playbooks.generic_sft_documents_multi_turn import check_params

        problem = check_params(out)
        if problem is not None:
            name, message = problem
            raise HTTPException(400, {
                "error_code": "invalid_playbook_param", "message": message,
                "metadata": {"param": name, "value": out.get(name)},
            })
        return out
    if mode_value not in ("seeded_defect", "cluster_targeted"):
        return out
    from app.services import label_taxonomy_service as lts

    raw = out.get("label_id")
    index = await lts.load_taxonomy_index(db, project_id)
    given = raw is not None and not (isinstance(raw, str) and not raw.strip())
    if not given:
        if mode_value == "cluster_targeted":
            out.pop("label_id", None)
            return out
        raise HTTPException(400, {
            "error_code": "missing_label_id",
            "message": "A seeded-defect run needs params.label_id (a label id; from the project taxonomy when it has one).",
            "metadata": {"taxonomy_version": index.version if index else None},
        })
    if index is None:
        if not isinstance(raw, str):
            raise HTTPException(400, {
                "error_code": "unknown_label",
                "message": "params.label_id must be a text label.",
                "metadata": {"label_id": raw, "taxonomy_version": None, "known_labels": [], "ambiguous": False},
            })
        out["label_id"] = raw.strip()
        return out
    label, how = index.resolve(raw) if isinstance(raw, str) else (None, None)
    if label is None:
        known = list(index.entries)[:40]
        if how == "ambiguous":
            message = f"Label {raw!r} matches more than one taxonomy entry; use the entry id."
        else:
            message = f"Label {raw!r} is not in the project's label taxonomy."
        raise HTTPException(400, {
            "error_code": "unknown_label",
            "message": message,
            "metadata": {
                "label_id": raw,
                "taxonomy_version": index.version,
                "known_labels": known,
                "ambiguous": how == "ambiguous",
            },
        })
    out["label_id"] = label
    return out


class RelabelAnswerKeyRequest(BaseModel):
    """Relabel a split's texts with another model and keep them as the
    project's GOLD_TEST answer key. ``backend``: a synth backend name or a
    cloud pin (``cloud:anthropic:claude-haiku-4-5-20251001`` — its API key
    is the project's saved secret); unset → the first reachable local
    model. ``entity_types`` default to the split's own."""

    split: str = Field("test", pattern="^(train|val|test|gold_dev|gold_test)$")
    backend: str | None = None
    entity_types: list[str] = Field(default_factory=list)
    # "answer_key": label ``split`` as gold_test (default). "training":
    # relabel the saved synthetic rows in place, under the same conventions.
    # "passages": label the cleaned document passages as they are (no
    # rewrite) and add them as training rows — layout lines included.
    # "documents": whole documents (runs of whole paragraphs up to
    # max_chars), longer than one 512-token window.
    # "split": relabel the prepared ``split`` itself in place (a kept test
    # split doesn't follow relabelled source rows).
    target: str = Field("answer_key", pattern="^(answer_key|training|passages|documents|split)$")
    document_id: int | None = None
    max_chars: int = Field(4000, ge=500, le=20000)
    # target "training": relabel only this synth_source group (e.g.
    # "document_labels:doc=740"); empty = every row.
    source: str | None = None
    # target "training": only rows not labelled under the current
    # conventions (the review queue's stale rows) — current rows untouched.
    only_stale: bool = False
    # Only the stale rows a convention change touches (implies only_stale).
    only_affected: bool = False
    # Tiered labelling: a second (stronger) model, run only on rows where
    # ``backend`` and the project's trained span model disagree.
    review_backend: str | None = None
    review_experiment_id: int | None = None
    # A row goes to review_backend only when its bulk labels and the model's
    # prediction agree less than this (row span F1; 1.0 = any difference).
    review_min_agreement: float = Field(0.8, ge=0.0, le=1.0)
    # With only_affected: report how many rows the change touches, call no model.
    dry_run: bool = False
    # Anthropic Message Batches: every label call in ONE batch at half the
    # list price; results take minutes to hours. ``cloud:anthropic:*`` only.
    use_batch: bool = False


@router.post("/relabel-answer-key", status_code=202)
async def relabel_answer_key(
    project_id: int,
    data: RelabelAnswerKeyRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Start the ``span_relabel`` Job: label the split's texts with the
    chosen model (texts unchanged, offsets verified), write
    ``gold/gold_test.jsonl`` + the GOLD_TEST dataset the evaluation's
    ``gold_test`` alias resolves, and report the agreement between the
    old labels and the new key."""
    from sqlalchemy import select

    from app.models.job import Job, JobStatus
    from app.services.jobs_service import JobProgressHandle, serialize_job, start_job
    from app.services.span_relabel_service import (
        label_documents_as_training_rows,
        label_passages_as_training_rows,
        relabel_split_as_answer_key,
        relabel_training_rows,
    )

    req = data or RelabelAnswerKeyRequest()
    if req.use_batch and not (req.backend or "").startswith("cloud:anthropic:"):
        raise HTTPException(400, {
            "error_code": "BATCH_NEEDS_ANTHROPIC",
            "message": "Batch mode uses Anthropic's Message Batches API — pick a cloud:anthropic:<model> backend, or turn batch off.",
            "metadata": {"backend": req.backend},
        })
    cloud_override = await _resolve_cloud_backend_override(db, project_id, req.backend)
    if req.use_batch and cloud_override is not None:
        cloud_override.use_batch = True
    review_override = None
    if req.review_backend:
        review_override = await _resolve_cloud_backend_override(db, project_id, req.review_backend)
        if review_override is None:
            from app.services.synth_backends.base import pick_backend

            review_override = pick_backend(req.review_backend)
    in_flight = (
        await db.execute(
            select(Job).where(Job.kind == "span_relabel", Job.project_id == project_id)
            .where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])).limit(1)
        )
    ).scalar_one_or_none()
    if in_flight is not None:
        raise HTTPException(409, {"error_code": "RELABEL_ALREADY_RUNNING", "message": "An answer key is already being written for this project. Watch the bell.", "metadata": {"existing_job_id": in_flight.id}})

    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        from app.database import async_session_factory

        async def _progress(fraction: float, message: str) -> None:
            await handle.set_progress(fraction=fraction, message=message)

        async with async_session_factory() as runner_db:
            if req.target == "documents":
                summary = await label_documents_as_training_rows(
                    runner_db, project_id, document_id=req.document_id, max_chars=req.max_chars,
                    backend=req.backend if cloud_override is None else None,
                    entity_types=req.entity_types, progress=_progress, backend_override=cloud_override,
                    review_override=review_override,
                )
            elif req.target == "passages":
                summary = await label_passages_as_training_rows(
                    runner_db, project_id, document_id=req.document_id,
                    backend=req.backend if cloud_override is None else None,
                    entity_types=req.entity_types, progress=_progress, backend_override=cloud_override,
                    review_override=review_override,
                )
            elif req.target == "split":
                from app.services.span_relabel_service import relabel_split_in_place

                summary = await relabel_split_in_place(
                    runner_db, project_id, split=req.split,
                    backend=req.backend if cloud_override is None else None,
                    entity_types=req.entity_types, progress=_progress, backend_override=cloud_override,
                )
            elif req.target == "training":
                summary = await relabel_training_rows(
                    runner_db, project_id, source=req.source or None, only_stale=req.only_stale,
                    only_affected=req.only_affected,
                    backend=req.backend if cloud_override is None else None,
                    entity_types=req.entity_types, progress=_progress, backend_override=cloud_override,
                    review_override=review_override, review_experiment_id=req.review_experiment_id,
                    dry_run=req.dry_run, review_min_agreement=req.review_min_agreement,
                )
            else:
                summary = await relabel_split_as_answer_key(
                    runner_db, project_id, split=req.split,
                    backend=req.backend if cloud_override is None else None,
                    entity_types=req.entity_types, progress=_progress, backend_override=cloud_override,
                )
            return summary.as_dict()

    job = await start_job(
        db, kind="span_relabel",
        title=(
            f"Whole-document labels by {req.backend or 'local model'} · project #{project_id}"
            if req.target == "documents"
            else f"Passage labels by {req.backend or 'local model'} · project #{project_id}"
            if req.target == "passages"
            else f"Training labels by {req.backend or 'local model'} · project #{project_id}"
            if req.target == "training"
            else f"{req.split.capitalize()} split relabelled by {req.backend or 'local model'} · project #{project_id}"
            if req.target == "split"
            else f"Answer key by {req.backend or 'local model'} · {req.split} split · project #{project_id}"
        ) + (" · batch" if req.use_batch else ""),
        runner=_runner, project_id=project_id,
        params={"project_id": project_id, "split": req.split, "backend": req.backend, "entity_types": req.entity_types, "target": req.target, "document_id": req.document_id, "use_batch": req.use_batch},
    )
    return await _with_budget_warning(db, project_id, serialize_job(job))


@router.post("/spans/regex-augment")
async def regex_augment_saved_spans(
    project_id: int, data: dict | None = None, db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Add regex-detected entities (email / phone / card / IP / … and the
    project's custom patterns) to every saved synthetic span row where
    they're missing. Deterministic and idempotent — the fix for rows
    generated under a restricted type list, or saved before a custom
    pattern existed. Body: ``{"entity_types": [...]}`` optional (defaults
    to every pattern)."""
    from app.services.span_custom_patterns_service import load_patterns
    from app.services.synthetic_service import augment_saved_rows_with_regex

    types = (data or {}).get("entity_types") if isinstance(data, dict) else None
    return augment_saved_rows_with_regex(
        project_id, types if isinstance(types, list) else None, await load_patterns(db, project_id)
    )


class TypePatternItem(BaseModel):
    type: str
    pattern: str
    description: str = ""
    flags: str = ""


class TypePatternsRequest(BaseModel):
    """The project's custom identifier patterns. Replaces the stored list;
    send ``[]`` to clear. Every entry must validate or nothing is saved."""

    patterns: list[TypePatternItem] = Field(default_factory=list)


class TypePatternTestRequest(BaseModel):
    pattern: str
    type: str = ""
    flags: str = ""


def _builtin_regex_types() -> list[str]:
    from app.services.synthetic_service import _DEFAULT_SPAN_PATTERNS

    return sorted({t for t, _ in _DEFAULT_SPAN_PATTERNS})


@router.get("/type-patterns")
async def get_type_patterns(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """The project's custom regex patterns (custom infotypes), any stored
    entries that no longer validate, the built-in regex types they run
    beside, and the validation limits."""
    from app.models.project import Project
    from app.services import span_custom_patterns_service as cps

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    cfg = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    patterns, errors = cps.normalise_patterns(cfg.get(cps.PATTERNS_KEY))
    return {
        "patterns": patterns,
        "errors": errors,
        "builtin_types": _builtin_regex_types(),
        "limits": {
            "max_patterns": cps.MAX_PATTERNS,
            "max_pattern_chars": cps.MAX_PATTERN_CHARS,
            "flags": "ims",
        },
    }


@router.put("/type-patterns")
async def put_type_patterns(
    project_id: int, data: TypePatternsRequest, db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Validate (compile, length, no empty match, backtracking guard) and
    store the project's custom patterns. 422 ``INVALID_TYPE_PATTERN`` with
    per-entry ``metadata.errors`` when any entry fails — nothing is saved."""
    import asyncio

    from app.models.project import Project
    from app.services import span_custom_patterns_service as cps

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    if len(data.patterns) > cps.MAX_PATTERNS:
        raise HTTPException(422, {"error_code": "INVALID_TYPE_PATTERN", "message": f"At most {cps.MAX_PATTERNS} patterns per project.", "metadata": {"errors": []}})
    raw = [p.model_dump() for p in data.patterns]
    valid, errors = await asyncio.to_thread(cps.normalise_patterns, raw, check_speed=True)
    if errors:
        first = errors[0]
        raise HTTPException(422, {
            "error_code": "INVALID_TYPE_PATTERN",
            "message": f"Pattern for '{first['type'] or '?'}' rejected: {first['error']}",
            "metadata": {"errors": errors},
        })
    cfg = dict(project.runtime_config or {})
    cfg[cps.PATTERNS_KEY] = valid
    project.runtime_config = cfg
    await db.commit()
    return {"patterns": valid}


@router.post("/type-patterns/test")
async def try_type_pattern(
    project_id: int, data: TypePatternTestRequest, db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Try a pattern before saving it: validation result plus the matches
    in a sample of the project's saved span rows and cleaned passages
    (count, and the first matches with offsets and surrounding text)."""
    import asyncio

    from app.models.project import Project
    from app.services import span_custom_patterns_service as cps

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    sample = cps.sample_project_texts(project_id)
    return await asyncio.to_thread(
        cps.preview_pattern_matches, data.pattern, data.type, [t for _, t in sample], data.flags,
        sources=[s for s, _ in sample],
    )


class TypeDefinitionsRequest(BaseModel):
    """``{type: one-line definition}`` — the project's labelling convention.
    Replaces the stored map; send ``{}`` to clear."""

    # None = leave the stored map unchanged; {} clears it.
    definitions: dict[str, str] | None = None
    # The project's texts come from scans (OCR): adds the scan-tolerant
    # labelling clause and OCR-tolerant span decoding. None = unchanged.
    scan_tolerant: bool | None = None


@router.get("/type-definitions")
async def get_type_definitions(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """The project's entity-type definitions (what the span generator and
    the answer-key relabeller both put in their prompts), the types the
    project's data uses, and suggested starting wording for those that
    have none."""
    from app.models.project import Project
    from app.services.span_type_definitions_service import project_definitions, project_scan_tolerant, suggest_definitions

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    definitions = project_definitions(project)
    types = _project_entity_types(project_id)
    return {
        "definitions": definitions,
        "scan_tolerant": project_scan_tolerant(project),
        "types": types,
        "suggestions": {t: d for t, d in suggest_definitions(types).items() if t not in definitions},
    }


@router.put("/type-definitions")
async def put_type_definitions(
    project_id: int, data: TypeDefinitionsRequest, db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    from app.models.project import Project
    from app.services.span_type_definitions_service import DEFINITIONS_KEY, normalise_definitions

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    from app.services.span_type_definitions_service import SCAN_TOLERANT_KEY, remember_conventions

    # Snapshot the wording before and after, so rows labelled under the
    # old one can be diffed type by type (targeted relabels).
    remember_conventions(project)
    cfg = dict(project.runtime_config or {})
    if data.definitions is not None:
        cfg[DEFINITIONS_KEY] = normalise_definitions(data.definitions)
    if data.scan_tolerant is not None:
        cfg[SCAN_TOLERANT_KEY] = bool(data.scan_tolerant)
    project.runtime_config = cfg
    remember_conventions(project)
    await db.commit()
    return {"definitions": cfg.get(DEFINITIONS_KEY) or {}, "scan_tolerant": bool(cfg.get(SCAN_TOLERANT_KEY))}


def _project_entity_types(project_id: int) -> list[str]:
    """Entity types in the project's saved synthetic span rows and prepared
    splits (sorted)."""
    from app.services.span_type_definitions_service import project_entity_types

    return project_entity_types(project_id)


@router.get("/weak-types")
async def get_weak_entity_types(
    project_id: int,
    threshold: float = Query(0.8, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """The entity types the latest span_set evaluation scored under
    ``threshold`` (weakest first) with the evaluation's provenance and a
    suggested row count — what the Synthetic tab's "Refine weak types"
    button points the span generator at."""
    from app.services.span_refine_service import weak_entity_types

    return await weak_entity_types(db, project_id, threshold)


def _billing_for(cls: Any) -> str:
    from app.api.eval_judge import _billing_for_class

    return _billing_for_class(cls)


def _detail_for(cls: Any, available: bool) -> str | None:
    if not available or not callable(getattr(cls, "detail", None)):
        return None
    try:
        return str(cls().detail()) or None
    except Exception:  # noqa: BLE001 — provenance text never 500s the picker
        return None


@router.get("/backends")
async def list_synth_backends(project_id: int):
    """List the synth backends registered on this BrewSLM install +
    each one's reachability (USER-SUCCESS Epic 5 Phase 5a).

    ``project_id`` isn't read — backends are install-global — but the
    route lives under the project-scoped prefix to keep the URL
    pattern consistent with the rest of the synthetic router (and to
    keep the auth middleware applied uniformly).

    Frontend uses this to decide whether to render the backend picker
    on the playbook panel: if only one backend is available, the
    picker is hidden to avoid clutter.
    """
    # Lazy import — keeps the module-level import surface narrow and
    # avoids pulling httpx into request-routing if it's not present.
    from app.services.synth_backends import BACKEND_REGISTRY

    entries: list[dict[str, object]] = []
    for cls in BACKEND_REGISTRY:
        try:
            available = bool(cls.is_available())
        except Exception:  # noqa: BLE001 — a broken backend must never 500 the picker
            available = False
        entries.append({
            "name": cls.name,
            "available": available,
            # ``describe()`` requires an instance — instantiate only for
            # the available ones (so we don't trigger constructor-time
            # validation on an unavailable backend like NeMo without a
            # configured model).
            "describe": (cls().describe() if available else cls.name),
            # Phase 5c: True for backends that forward the playbook's
            # response_schema as response_format=json_schema (NeMo, vLLM).
            # False for backends that accept-and-ignore (Ollama, Teacher).
            # The picker uses this to badge schema-honoring options.
            "schema_aware": bool(getattr(cls, "schema_aware", False)),
            # Money model: "local" (free), "metered" (API credits) or
            # "subscription" (a CLI signed in to a plan — not billed per call);
            # ``detail`` is the picker's provenance line (None when the
            # backend has none).
            "billing": _billing_for(cls),
            "detail": _detail_for(cls, available),
        })
    return {"project_id": project_id, "backends": entries}


@router.get("/backends/ollama/models")
async def list_ollama_models(project_id: int):  # noqa: ARG001 — install-global, project_id ignored
    """List the Ollama models installed on the local daemon, used by
    the Synth panel's model-picker dropdown.

    Returns ``{models: [...], default: <pattern-matched-model>}``.
    The ``default`` reflects what the platform's auto-pick would
    choose given current `PREFERRED_MODEL_PATTERNS` so the picker
    can label it ``"Auto (qwen2.5:14b...)"``.

    Best-effort — if Ollama isn't reachable, returns
    ``{models: [], ollama_available: false, error: ...}`` with a
    200 so the picker can render a 'No Ollama models found' state
    rather than failing the panel load.
    """
    import httpx  # local import — keep startup light

    from app.services.synth_backends.ollama import (
        DEFAULT_OLLAMA_HOST,
        OllamaBackend,
        PREFERRED_MODEL_PATTERNS,
    )

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{DEFAULT_OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001 — daemon down should not 500 the picker
        return {
            "project_id": project_id,
            "models": [],
            "default": None,
            "ollama_available": False,
            "error": f"{type(e).__name__}: {e}",
        }

    raw_models = [m for m in (data.get("models") or []) if m.get("name")]
    # Resolve the auto-pick the way the backend would: walk
    # PREFERRED_MODEL_PATTERNS, take the first substring-match.
    tags = [m["name"] for m in raw_models]
    default_pick: str | None = None
    for pattern in PREFERRED_MODEL_PATTERNS:
        for tag in tags:
            if pattern in tag:
                default_pick = tag
                break
        if default_pick:
            break
    if default_pick is None and tags:
        default_pick = tags[0]

    return {
        "project_id": project_id,
        "models": [
            {
                "name": m["name"],
                "size_bytes": int(m.get("size") or 0),
                "parameter_size": (m.get("details") or {}).get("parameter_size") or "",
                "family": (m.get("details") or {}).get("family") or "",
            }
            for m in raw_models
        ],
        "default": default_pick,
        "ollama_available": True,
    }


@router.get("/backends/cloud/models")
async def list_cloud_models(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Curated cloud-LLM catalog (OpenAI / Anthropic / Deepseek) +
    which providers have an API key saved for THIS project. Drives
    the synth panel's cloud picker so the user sees:

      - all three providers + curated model lists
      - a green check + "Key saved" badge for providers with creds
      - a "Save key first" prompt for providers without creds

    Project secrets live under ``cloud_llm_<provider>:api_key`` (the
    same coordinates the gold-generate flow uses), so a user who's
    already saved a key in one panel sees it pre-filled in the other.
    """
    from app.services.secret_service import get_project_secret_value

    out_providers: list[dict[str, object]] = []
    for provider, (secret_provider, key_name) in _CLOUD_PROVIDER_SECRET_COORDS.items():
        try:
            api_key = await get_project_secret_value(
                db, project_id, secret_provider, key_name,
            )
        except Exception:  # noqa: BLE001 — secrets store hiccup ≠ block the panel
            api_key = None
        out_providers.append({
            "provider": provider,
            "key_saved": bool(api_key),
            "models": list(_CLOUD_MODELS_CATALOG.get(provider, [])),
        })
    return {"project_id": project_id, "providers": out_providers}


@router.get("/playbooks")
async def list_synth_playbooks(project_id: int, db: AsyncSession = Depends(get_db)):
    """Catalog of registered playbooks compatible with the project's
    selected recipe.

    When the project has no recipe selected (legacy projects pre-dating
    the auto-apply-on-create fix), this returns an empty list plus
    ``recipe_required=True`` so the UI can render a "pick a recipe
    first" prompt instead of a confusing dump of every playbook
    across every task shape — none of which would actually run, since
    ``run_playbook`` requires ``selected_recipe`` to dispatch.
    Brief-driven + magic-create projects always have a recipe applied
    at creation time (see ``recipe_service.default_recipe_for_*``)
    so this branch is now legacy-only."""
    from app.models.project import Project
    from app.services.synth_playbook_service import available_playbooks_for_recipe

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    recipe = (project.selected_recipe or {}).get("recipe_id")
    if recipe:
        return {
            "project_id": project_id,
            "recipe_id": recipe,
            "recipe_required": False,
            "playbooks": available_playbooks_for_recipe(recipe),
        }
    return {
        "project_id": project_id,
        "recipe_id": None,
        "recipe_required": True,
        "playbooks": [],
    }


class BulkReviewQueueRequest(BaseModel):
    row_ids: list[int] = Field(..., description="IDs of pending synth rows to update.")
    action: str = Field(..., description="'accept' or 'reject'.")
    # Arc 5 — when action='reject', the caller may attach a free-form
    # reason ("schema_invalid", "duplicate", "low_confidence", etc.)
    # that gets stamped on the row. The Rejected section in the UI
    # groups + filters by this so the user can bulk-purge a reason
    # cohort. Ignored for 'accept'.
    reject_reason: str | None = Field(
        default=None,
        description=(
            "Optional label for a bulk reject (e.g. 'duplicate', "
            "'schema_invalid'). Stamped on each rejected row so the "
            "Rejected section can group + purge by reason."
        ),
        max_length=64,
    )


class BulkReviewQueueBySourceRequest(BaseModel):
    source: str = Field(
        ...,
        description=(
            "The ``synth_source`` group key whose pending rows to update "
            "(e.g. 'playbook:classification:hard_negatives'). Every pending "
            "row in the group is affected — no row-id enumeration needed."
        ),
        max_length=256,
    )
    action: str = Field(..., description="'accept' or 'reject'.")
    reject_reason: str | None = Field(
        default=None,
        description="Optional label stamped on each rejected row (reject only).",
        max_length=64,
    )


class PurgeRejectedRequest(BaseModel):
    reasons: list[str] | None = Field(
        default=None,
        description=(
            "When set, only rejected rows whose ``reject_reason`` "
            "is in this list get physically removed. When omitted "
            "(or empty), all rejected rows are purged."
        ),
    )


@router.get("/review-queue")
async def list_synth_review_queue(
    project_id: int,
    group_by: str = "source",
    synth_job_id: int | None = None,
    synth_backend: str | None = None,
    verdict: str | None = None,
    reason_code: str | None = None,
    assist_consensus: str | None = None,
    judge: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List pending synthetic rows for the project, grouped by
    `synth_source` (USER-SUCCESS Epic 2b) — or by class label with
    `?group_by=class` (Epic E), by originating failure cluster
    (`cluster`), by the Job that wrote the rows (`job`) or by the backend
    that wrote them (`backend`), by review-assist verdict (`verdict`) or by
    verdict + reason code (`assist_reason`) or by judge consensus (`assist_consensus`).
    `?synth_job_id=` / `?synth_backend=` / `?verdict=` / `?reason_code=` / `?assist_consensus=` / `?judge=`
    narrow every list (`judge` shows one judge's verdicts; verdict / reason then read that judge).

    Pending rows are gated out of dataset prep — they don't enter
    training until the user accepts them via `/review-queue/bulk-update`
    (or rejects them, in which case they're deleted).
    """
    from app.services.synth_review_queue_service import list_review_queue

    return await list_review_queue(
        db, project_id, group_by=group_by, synth_job_id=synth_job_id, synth_backend=synth_backend,
        verdict=verdict, reason_code=reason_code, assist_consensus=assist_consensus, judge=judge,
    )


@router.post("/review-queue/bulk-update")
async def bulk_update_synth_review_queue(
    project_id: int,
    req: BulkReviewQueueRequest,
    db: AsyncSession = Depends(get_db),
):
    """Bulk accept or reject pending synth rows. Accepted rows flip
    to ``review_status="accepted"`` and become eligible for
    training; rejected rows flip to ``review_status="rejected"``
    (Arc 5 soft-reject) and stay on disk for review + bulk-purge.
    Pass ``reject_reason`` to tag each rejected row with a label
    the Rejected section's group-by-reason UI uses.
    """
    from app.services.synth_review_queue_service import bulk_update_review_queue

    if req.action not in ("accept", "reject"):
        raise HTTPException(400, "action must be 'accept' or 'reject'")
    return await bulk_update_review_queue(
        db,
        project_id,
        row_ids=req.row_ids,
        action=req.action,  # type: ignore[arg-type]
        reject_reason=req.reject_reason,
    )


class ReviewAssistRequest(BaseModel):
    """Body for the review-assist Job (an LLM pre-screen of pending synthetic rows)."""

    backend: str | None = Field(default=None, max_length=256, description="Any synth backend pin (ollama:<model>, command:<model>, cloud:<provider>:<model>, ...); omitted = the first reachable local backend.")
    use_batch: bool = Field(default=False, description="Anthropic Message Batch (half price; cloud:anthropic:<model> only).")
    row_ids: list[int] | None = Field(default=None, description="Pending row ids to judge (default: every pending row).")
    source: str | None = Field(default=None, max_length=256, description="Only pending rows of this synth_source group.")
    synth_job_id: int | None = Field(default=None, description="Only pending rows written by this Job.")
    rubric_text: str | None = Field(default=None, max_length=4000, description="Free text appended to the built-in rubric (e.g. the allowed operations).")
    rows_per_call: int = Field(default=10, ge=1, le=25)
    max_rows: int | None = Field(default=None, ge=1)
    dry_run: bool = Field(default=False, description="Report how many rows / calls the run would take; call no model.")
    only_missing_judge: bool = Field(default=False, description="Skip rows this judge (the backend) already has a valid verdict for (resume).")
    consensus: list[str] | None = Field(default=None, description="Only rows whose current judge consensus is one of none / single / agree_pass / agree_fail / disagree.")
    prior_verdicts: list[str] | None = Field(default=None, description="Only rows where another judge's valid verdict is one of pass / fail / unsure (e.g. a second opinion where the first said fail or unsure).")


@router.post("/review-assist")
async def review_assist(
    project_id: int,
    req: ReviewAssistRequest,
    async_job: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Pre-screen pending synthetic rows with an LLM: each gets a verdict (pass / fail / unsure),
    a reason code and a one-sentence note, shown in the review queue (filter / group by them,
    then bulk accept or reject). Runs as the ``synth_review_assist`` Job (``?async_job=true``;
    a dry run answers inline). Verdicts live in a sidecar, never in ``synthetic.jsonl``."""
    from fastapi.responses import JSONResponse
    from sqlalchemy import select

    from app.models.job import Job, JobStatus
    from app.services import jobs_service, synth_review_assist_service as assist
    from app.services.synth_backends import SynthBackendError
    from app.services.synth_playbook_service import BACKEND_REGISTRY, pick_backend
    from app.services.synth_review_queue_service import _read_all_rows, _synthetic_jsonl_path

    if req.use_batch and not (req.backend or "").startswith("cloud:anthropic:"):
        raise HTTPException(400, {
            "error_code": "invalid_review_assist_param",
            "message": "use_batch uses Anthropic's Message Batches API — pick a cloud:anthropic:<model> backend, or turn batch off.",
            "metadata": {"param": "use_batch", "backend": req.backend},
        })
    if not req.dry_run and not async_job:
        raise HTTPException(400, {
            "error_code": "invalid_review_assist_param",
            "message": "Pre-screening runs as a background Job — add ?async_job=true (or dry_run to size it).",
            "metadata": {"param": "async_job"},
        })

    rows = assist.select_rows(
        _read_all_rows(_synthetic_jsonl_path(project_id)),
        row_ids=req.row_ids, source=req.source, synth_job_id=req.synth_job_id, max_rows=req.max_rows,
    )
    if not rows:
        raise HTTPException(400, {
            "error_code": "no_pending_rows",
            "message": "No pending synthetic rows match this pre-screen (only pending rows are judged).",
            "metadata": {"source": req.source, "synth_job_id": req.synth_job_id},
        })
    bad_consensus = [c for c in (req.consensus or []) if c not in assist.CONSENSUS_VALUES]
    bad_prior = [v for v in (req.prior_verdicts or []) if v not in assist.VERDICTS]
    if bad_consensus or bad_prior:
        raise HTTPException(400, {
            "error_code": "invalid_review_assist_param",
            "message": "consensus must be among " + ", ".join(assist.CONSENSUS_VALUES) + "; prior_verdicts among " + ", ".join(assist.VERDICTS) + ".",
            "metadata": {"param": "consensus" if bad_consensus else "prior_verdicts"},
        })
    uses_judge_filter = bool(req.only_missing_judge or req.consensus or req.prior_verdicts)

    def _narrow(candidates: list[dict], judge: str) -> list[dict]:
        return assist.filter_rows_for_judge(
            project_id, candidates, judge, only_missing_judge=req.only_missing_judge,
            consensus=req.consensus, prior_verdicts=req.prior_verdicts,
        )

    def _no_rows() -> HTTPException:
        return HTTPException(400, {
            "error_code": "no_pending_rows",
            "message": "No pending synthetic rows match this pre-screen after the judge / consensus filters (only pending rows are judged).",
            "metadata": {"only_missing_judge": req.only_missing_judge, "consensus": req.consensus, "prior_verdicts": req.prior_verdicts},
        })

    if req.dry_run and uses_judge_filter:
        # Best effort: the judge key is the backend's label; an unreachable pin falls back to the pin text.
        from app.services.synth_playbook_service import safe_backend_label as _safe

        try:
            dry_judge = assist.judge_label(pick_backend(req.backend, registry=BACKEND_REGISTRY))
        except Exception:  # noqa: BLE001
            dry_judge = _safe(req.backend)
        rows = _narrow(rows, dry_judge)
        if not rows:
            raise _no_rows()
    renders = {int(r["id"]): assist.render_row_ex(r) for r in rows}
    judgeable = [r for r in rows if not renders[int(r["id"])].too_long]
    chunks = assist.pack_chunks(judgeable, renders, req.rows_per_call, req.rubric_text)
    calls = len(chunks)
    if req.dry_run:
        from app.services.cloud_backend_pins import CloudPinError, parse_cloud_pin

        try:
            parse_cloud_pin(req.backend)
        except CloudPinError as exc:
            raise HTTPException(exc.status_code, exc.message) from exc
        kinds: dict[str, int] = {}
        for row in rows:
            kind = assist.row_kind_of(row)
            kinds[kind] = kinds.get(kind, 0) + 1
        return {"dry_run": True, "rows": len(rows), "calls": calls, "rows_per_call": req.rows_per_call,
                "effective_rows_per_call": max((len(c) for c in chunks), default=0),
                "rows_too_long": len(rows) - len(judgeable), "row_kinds": kinds, "backend": req.backend, "use_batch": req.use_batch}

    # A cloud pin at the cap is refused up front (402); a missing key is 402, a bad pin 400.
    cloud_override = await _resolve_cloud_backend_override(db, project_id, req.backend)
    if cloud_override is None:
        try:
            pick_backend(req.backend, registry=BACKEND_REGISTRY)
        except SynthBackendError as exc:
            raise HTTPException(503, str(exc)) from exc
    if uses_judge_filter:
        judge_obj = cloud_override if cloud_override is not None else pick_backend(req.backend, registry=BACKEND_REGISTRY)
        rows = _narrow(rows, assist.judge_label(judge_obj))
        if not rows:
            raise _no_rows()
    in_flight = (
        await db.execute(
            select(Job).where(Job.kind == "synth_review_assist", Job.project_id == project_id)
            .where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])).limit(1)
        )
    ).scalar_one_or_none()
    if in_flight is not None:
        raise HTTPException(409, {"error_code": "REVIEW_ASSIST_ALREADY_RUNNING", "message": "A pre-screen is already running for this project. Watch the bell.", "metadata": {"existing_job_id": in_flight.id}})

    async def _runner(handle: "jobs_service.JobProgressHandle") -> dict:
        from app.database import async_session_factory

        async def _progress(fraction: float, message: str) -> None:
            await handle.set_progress(fraction=fraction, message=message)

        async with async_session_factory() as runner_db:
            override = await _resolve_cloud_backend_override(runner_db, project_id, req.backend)
        if override is not None:
            backend_obj = override
            if req.use_batch:
                override.use_batch = True
        else:
            backend_obj = pick_backend(req.backend, registry=BACKEND_REGISTRY)
        # Re-select at run time: rows may have been reviewed between the request and the start.
        run_rows = assist.select_rows(
            _read_all_rows(_synthetic_jsonl_path(project_id)),
            row_ids=req.row_ids, source=req.source, synth_job_id=req.synth_job_id, max_rows=req.max_rows,
        )
        if uses_judge_filter:
            run_rows = _narrow(run_rows, assist.judge_label(backend_obj))
        if not run_rows:
            raise RuntimeError("Pre-screen found no pending rows to judge (they were reviewed before it started).")
        await handle.set_progress(message=f"Pre-screening {len(run_rows)} rows…")
        result = await assist.run_review_assist(
            project_id, run_rows, backend_obj, rubric_text=req.rubric_text, rows_per_call=req.rows_per_call,
            use_batch=req.use_batch, job_id=handle.job_id, progress=_progress,
        )
        v = result["verdicts"]
        await handle.set_progress(
            fraction=1.0,
            message=f"Pre-screened {result['rows_judged']} rows · pass {v['pass']} · fail {v['fail']} · unsure {v['unsure']}" + (
                f" · agree {c['agree_pass'] + c['agree_fail']} · disagree {c['disagree']}"
                if (c := result.get("consensus") or {}) and (c.get("agree_pass", 0) + c.get("agree_fail", 0) + c.get("disagree", 0)) else ""
            ),
        )
        return result

    job = await jobs_service.start_job(
        db,
        kind="synth_review_assist",
        title=f"Pre-screen · {len(rows)} rows · {req.backend or 'local model'}" + (" · batch" if req.use_batch else ""),
        runner=_runner,
        project_id=project_id,
        params={
            "backend": req.backend, "rows": len(rows), "rows_per_call": req.rows_per_call,
            **({"use_batch": True} if req.use_batch else {}),
            **({"source": req.source} if req.source else {}),
            **({"synth_job_id": req.synth_job_id} if req.synth_job_id is not None else {}),
            **({"rubric_text": req.rubric_text} if req.rubric_text else {}),
            **({"only_missing_judge": True} if req.only_missing_judge else {}),
            **({"consensus": req.consensus} if req.consensus else {}),
            **({"prior_verdicts": req.prior_verdicts} if req.prior_verdicts else {}),
        },
    )
    return JSONResponse(
        status_code=202,
        content=await _with_budget_warning(db, project_id, jobs_service.serialize_job(job)),
    )


@router.post("/review-queue/bulk-update-by-source")
async def bulk_update_synth_review_queue_by_source(
    project_id: int,
    req: BulkReviewQueueBySourceRequest,
    db: AsyncSession = Depends(get_db),
):
    """Accept or reject *every pending row* in one ``synth_source`` group by
    key. Powers the one-click "Accept all (N)" / "Reject all (N)" on the Data
    Studio review-queue panel's pending groups (Epic E — actionable workbench),
    reusing the same soft-reject + record-count sync as ``/bulk-update``.
    """
    from app.services.synth_review_queue_service import bulk_update_by_source

    if req.action not in ("accept", "reject"):
        raise HTTPException(400, "action must be 'accept' or 'reject'")
    return await bulk_update_by_source(
        db,
        project_id,
        source=req.source,
        action=req.action,  # type: ignore[arg-type]
        reject_reason=req.reject_reason,
    )


class RetireGroupRequest(BaseModel):
    source: str = Field(..., max_length=256)
    # True: accepted rows → rejected ("retired"); False: restore them.
    retired: bool = True


class RecheckCodeFormatRequest(BaseModel):
    dry_run: bool = Field(default=True, description="Count the mislabelled pending rows without changing anything.")
    row_ids: list[int] | None = Field(default=None, description="Apply only to these rows (each is re-validated server-side); omitted = every mismatch.")


@router.post("/review-queue/recheck-code-format")
async def recheck_code_format_endpoint(
    project_id: int,
    req: RecheckCodeFormatRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Re-check pending code examples labelled json / yaml / toml / xml: code that does not parse as its label
    is a mislabel. ``dry_run`` (default) counts by language; otherwise those rows are marked rejected with
    reason ``language_mismatch`` (reversible, never purged)."""
    from app.services.synth_review_queue_service import recheck_code_format

    req = req or RecheckCodeFormatRequest()
    return await recheck_code_format(db, project_id, dry_run=req.dry_run, row_ids=req.row_ids)


@router.post("/review-queue/retire-group")
async def retire_synth_group(
    project_id: int,
    req: RetireGroupRequest,
    db: AsyncSession = Depends(get_db),
):
    """"Retire group" / "Restore" on the review queue: take an accepted
    source group out of training (soft — rows stay on disk as rejected with
    reason "retired") or put it back. See ``set_group_retired``."""
    from app.services.synth_review_queue_service import set_group_retired

    try:
        return await set_group_retired(db, project_id, source=req.source, retired=req.retired)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class MarkConventionsCurrentRequest(BaseModel):
    source: str = Field(..., max_length=256)


@router.post("/review-queue/mark-conventions-current")
async def mark_synth_group_conventions_current(
    project_id: int,
    req: MarkConventionsCurrentRequest,
    db: AsyncSession = Depends(get_db),
):
    """"Mark as current" on a review-queue group flagged as labelled under
    older conventions: the user attests it already follows the current
    ones (no relabel). See ``mark_conventions_current``."""
    from app.services.synth_review_queue_service import mark_conventions_current

    try:
        return await mark_conventions_current(db, project_id, source=req.source)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class CheckPlanRulesRequest(BaseModel):
    rules: dict | str = Field(..., description="The plan-rules document (object, or its JSON text). See plan_rules_service.")
    dry_run: bool = Field(default=True, description="True (default): count violations per rule, change nothing. False: soft-reject rows with a reject-severity violation.")
    row_ids: list[int] | None = Field(default=None, max_length=5000, description="Only these pending rows.")
    source: str | None = Field(default=None, max_length=256, description="Only pending rows of this synth_source group.")
    synth_job_id: int | None = Field(default=None, description="Only pending rows written by this Job.")
    max_rows: int | None = Field(default=None, ge=1, le=100_000)


@router.post("/check-plan-rules")
async def check_plan_rules(
    project_id: int,
    req: CheckPlanRulesRequest,
    async_job: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Check the pending structured-plan rows against a caller-supplied plan-rules document.

    ``dry_run`` (default) answers inline with counts per rule + sample row ids. With
    ``dry_run=false`` rows with a ``reject``-severity violation are marked rejected as
    ``rule_violation:<rule id>`` (reversible: they stay on disk, restore as usual; never purged).
    ``?async_job=true`` runs the check as a ``plan_rules_check`` Job (for big projects); bad rules
    are always a 400 ``invalid_plan_rules`` before any Job starts.
    """
    from fastapi.responses import JSONResponse

    from app.services import jobs_service, plan_rules_service as engine
    from app.services.synth_review_queue_service import check_plan_rules_for_project

    try:
        engine.load_rules(req.rules)
    except engine.PlanRulesError as exc:
        raise HTTPException(400, {
            "error_code": "invalid_plan_rules", "message": exc.message, "metadata": {"pointer": exc.pointer},
        }) from exc
    kwargs = dict(row_ids=req.row_ids, source=req.source, synth_job_id=req.synth_job_id, max_rows=req.max_rows)
    if not async_job:
        return await check_plan_rules_for_project(project_id, req.rules, dry_run=req.dry_run, **kwargs)

    async def _runner(handle: "jobs_service.JobProgressHandle") -> dict:
        await handle.set_progress(fraction=0.1, message="Checking pending plans against the rules…")
        result = await check_plan_rules_for_project(project_id, req.rules, dry_run=req.dry_run, **kwargs)
        if result["checked"] == 0:
            raise RuntimeError("No pending structured-plan rows to check (none match the filters, or they were already reviewed).")
        verb = "would reject" if req.dry_run else "rejected"
        await handle.set_progress(
            fraction=1.0,
            message=f"Checked {result['checked']} plans · {verb} {result['flagged'] if req.dry_run else result['rejected']} · {len(result['by_rule'])} rules hit",
        )
        return result

    job = await jobs_service.start_job(
        db, kind="plan_rules_check",
        title=("Check plan rules (dry run)" if req.dry_run else "Apply plan rules"),
        runner=_runner, project_id=project_id,
        params={"dry_run": req.dry_run, "rules": len((req.rules if isinstance(req.rules, dict) else {}).get("rules") or []), **{k: v for k, v in kwargs.items() if v is not None}},
    )
    return JSONResponse(status_code=202, content=jobs_service.serialize_job(job))


class RestoreRejectedRequest(BaseModel):
    reject_reason: str | None = Field(default=None, max_length=200, description="Exact reason (\"\" = rows rejected without one).")
    reject_reason_prefix: str | None = Field(default=None, max_length=200, description="Reason prefix, e.g. 'rule_violation:'.")
    row_ids: list[int] | None = Field(default=None, max_length=5000)
    dry_run: bool = Field(default=True, description="True (default): count by reason only. False: put the rows back to pending.")


@router.post("/review-queue/restore")
async def restore_rejected_synth_rows(project_id: int, req: RestoreRejectedRequest):
    """Undo a soft-reject: rejected rows matching the selectors (at least one of ``reject_reason``,
    ``reject_reason_prefix``, ``row_ids``; all given must match) go back to ``pending`` with their
    reject reason (and ``plan_rule_ids``) cleared. ``dry_run`` (default) returns counts by reason."""
    from app.services.synth_review_queue_service import restore_rejected_rows

    try:
        return restore_rejected_rows(
            project_id, reject_reason=req.reject_reason, reject_reason_prefix=req.reject_reason_prefix,
            row_ids=req.row_ids, dry_run=req.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(400, {"error_code": "invalid_restore_request", "message": str(exc), "metadata": {}}) from exc


@router.post("/review-queue/purge")
async def purge_rejected_synth_rows(
    project_id: int,
    req: PurgeRejectedRequest,
    db: AsyncSession = Depends(get_db),
):
    """Physically remove rejected synth rows from synthetic.jsonl.
    Arc-5 soft-reject keeps rejected rows around for review; this
    is the explicit "I've reviewed the rejected pile, drop it"
    step. Pass ``reasons`` to purge a specific reason cohort
    (e.g. just the ``duplicate`` rows); omit to purge every
    rejected row.
    """
    from app.services.synth_review_queue_service import purge_rejected_rows

    return await purge_rejected_rows(
        db, project_id, reasons=req.reasons,
    )


@router.post("/run-playbook")
async def run_synth_playbook(
    project_id: int,
    req: RunPlaybookRequest,
    async_job: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Run a playbook against the project's gold rows, generate
    synthetic training data, and persist accepted rows into the
    project's synthetic dataset.

    Returns a PlaybookResult: rows + backend_used + elapsed_sec +
    prompt_snippet.

    Hardening Phase H1 — pass ``?async_job=true`` to enqueue the
    playbook as a background Job. The endpoint returns 202 +
    ``{job_id: ...}`` immediately and the user tracks progress in
    the notification bell. The synchronous path is preserved for
    backward compatibility and for short, low-volume runs where a
    spinner is fine.
    """
    from app.services.synth_backends import SynthBackendError
    from app.services.synth_playbook_service import run_playbook
    from app.services.synth_playbooks import SynthMode

    try:
        mode = SynthMode(req.mode)
    except ValueError:
        raise HTTPException(400, f"Unknown synth mode '{req.mode}'.")
    playbook_params = await _checked_playbook_params(db, project_id, mode, req.params)
    _check_use_batch(req, async_job=async_job)

    if async_job:
        import asyncio
        import time

        from fastapi.responses import JSONResponse

        from app.services.jobs_service import (
            JobProgressHandle,
            serialize_job,
            start_job,
        )

        # A cloud pin at the cap is refused up front (402) instead of failing the Job.
        await _resolve_cloud_backend_override(db, project_id, req.backend)
        target_class_label = (
            f" (target: {req.target_class})" if req.target_class else ""
        )
        title = (
            f"Synth · {req.mode}{target_class_label} · {req.target_count} rows"
        ) + (" · batch" if req.use_batch else "")

        async def _runner(handle: JobProgressHandle) -> dict:
            # Hardening Phase H2 — publish a live elapsed-time heartbeat
            # so the bell shows progress even when the underlying LLM
            # call is opaque (we can't crack open Ollama / vLLM /
            # NeMo to publish per-token progress, but we *can* tell
            # the user "still working, 47s elapsed"). The heartbeat
            # cancels as soon as the main work returns.
            started = time.monotonic()
            stop_heartbeat = asyncio.Event()
            # Chunked runs (command backend) publish their own "chunk k/n"
            # progress, and a usage-limit pause its own message: the elapsed
            # heartbeat must not overwrite either.
            own_progress = {"on": False}

            async def _chunk_progress(fraction: float, message: str) -> None:
                own_progress["on"] = True
                await handle.set_progress(fraction=fraction, message=message)

            async def _heartbeat() -> None:
                while not stop_heartbeat.is_set():
                    try:
                        await asyncio.wait_for(
                            stop_heartbeat.wait(), timeout=5.0,
                        )
                        return
                    except asyncio.TimeoutError:
                        if own_progress["on"] or _command_paused():
                            continue
                        elapsed = int(time.monotonic() - started)
                        await handle.set_progress(
                            message=(
                                f"Calling LLM backend for {req.target_count} "
                                f"rows · {elapsed}s elapsed"
                            ),
                        )

            heartbeat_task = asyncio.create_task(_heartbeat())
            try:
                await handle.set_progress(
                    message=f"Calling LLM backend for {req.target_count} rows…"
                )
                # New session — runner doesn't share the request's session.
                from app.database import async_session_factory

                async with async_session_factory() as runner_db:
                    # If the user pinned a cloud backend, resolve the
                    # API key from project secrets + hand a fully-
                    # constructed CloudLlmBackend to run_playbook so
                    # pick_backend() doesn't have to learn auth.
                    cloud_override = await _resolve_cloud_backend_override(
                        runner_db, project_id, req.backend,
                    )
                    if req.use_batch and cloud_override is not None:
                        cloud_override.use_batch = True
                    result = await run_playbook(
                        runner_db,
                        project_id,
                        mode,
                        target_count=req.target_count,
                        target_class=req.target_class,
                        backend=req.backend if cloud_override is None else None,
                        backend_override=cloud_override,
                        progress=_chunk_progress,
                        params=playbook_params,
                        use_batch=req.use_batch,
                    )
                    await runner_db.commit()
            finally:
                stop_heartbeat.set()
                try:
                    await heartbeat_task
                except Exception:  # noqa: BLE001 — heartbeat is best-effort
                    pass

            # Final progress message replaces the heartbeat noise with
            # the actual outcome.
            row_count = len(result.get("rows") or [])
            backend_used = result.get("backend_used") or "auto"
            elapsed_sec = result.get("elapsed_sec") or 0
            prompt_snippet = result.get("prompt_snippet") or ""

            # 0-rows-generated is almost always a silent failure —
            # the LLM returned output that didn't parse, or every
            # parsed row failed the playbook's validate() pass. Raise
            # so the Job lands as FAILED with an actionable message
            # rather than a misleading "Done". Captures the backend
            # used + prompt snippet in the error so a power user can
            # investigate without re-running.
            if row_count == 0:
                try:
                    raw_snippet = result.get("raw_llm_snippet") or ""
                    refused = bool(result.get("refusal_detected"))
                    failed_chunks = result.get("failed_chunks") or []
                    if failed_chunks:
                        raise RuntimeError(
                            f"Playbook {req.mode} produced 0 accepted rows via {backend_used}: "
                            f"{len(failed_chunks)} of {result.get('chunks')} chunks failed. "
                            f"First failure (chunk {failed_chunks[0]['chunk']}): {failed_chunks[0]['reason']}"
                        )
                    if refused:
                        raise RuntimeError(
                            f"Playbook {req.mode} produced 0 accepted rows via "
                            f"{backend_used} in {elapsed_sec:.1f}s because the "
                            f"LLM refused on guardrail grounds. The model "
                            f"returned: {raw_snippet[:300]!r}. "
                            f"Fix: switch to a less-restricted model (Qwen 2.5 "
                            f"is preferred for security/abuse-detection use "
                            f"cases), pin a specific model via the request's "
                            f"``backend`` field (e.g. "
                            f"``ollama:qwen2.5:14b-instruct-q4_K_M``), or "
                            f"reframe the project's label vocabulary to be "
                            f"less alarm-bell-y for the model."
                        )
                    import json

                    reject_note = ""
                    if result.get("rejected_count"):
                        reject_note = (
                            f"{result['rejected_count']} teacher rows were rejected: "
                            f"{json.dumps(result.get('reject_summary') or {}, sort_keys=True)}. "
                        )
                    raise RuntimeError(
                        f"Playbook {req.mode} produced 0 accepted rows via "
                        f"{backend_used} in {elapsed_sec:.1f}s. {reject_note}"
                        f"Likely causes: (1) LLM backend returned text that "
                        f"didn't match the playbook's expected JSON shape, "
                        f"(2) every generated row failed validation (e.g. "
                        f"label outside the target class for hard-negatives), "
                        f"(3) backend timeout / network error swallowed by "
                        f"the runner. LLM response (first 300 chars): "
                        f"{raw_snippet[:300]!r}. Prompt sent (first 200 chars): "
                        f"{prompt_snippet[:200]!r}"
                    )
                except RuntimeError as exc:
                    # A batched run paid for its calls: the FAILED Job still reports them.
                    llm_usage.attach_result_usage(exc, result)
                    raise
            await handle.set_progress(
                fraction=1.0,
                message=(
                    f"Generated {row_count} rows via {backend_used} in "
                    f"{elapsed_sec:.1f}s · queued for review"
                ),
            )
            # Result payload is a pointers-only summary — keep it
            # small so the Job row stays cheap to fetch.
            return {
                "rows_generated": row_count,
                "backend_used": backend_used,
                "elapsed_sec": elapsed_sec,
                "mode": req.mode,
                "target_class": req.target_class,
                "rejected_count": result.get("rejected_count", 0),
                "reject_summary": result.get("reject_summary") or {},
                "rejected_examples": result.get("rejected_examples") or [],
                # Chunked runs (command backend): chunks that failed after the
                # backend's retries — the rest of the rows were kept.
                **({"chunks": result["chunks"], "failed_chunks": result.get("failed_chunks") or []} if result.get("chunks") else {}),
                # Message Batch runs report their own spend (batch calls priced at
                # half the list price; the Job's meter doesn't see them).
                **{k: result[k] for k in ("usage", "cost_usd", "tokens", "batch", "batch_requests", "batch_answered", "batch_error") if result.get(k) is not None},
                # Replies a miscounted bracket was repaired in (structured-plan playbooks).
                **({"repaired_json": result["repaired_json"]} if result.get("repaired_json") else {}),
                # Which sources / languages the seed passages came from (code / document playbooks).
                **({"seed_pool": result["seed_pool"]} if result.get("seed_pool") else {}),
            }

        job = await start_job(
            db,
            kind="synth_playbook",
            title=title,
            runner=_runner,
            project_id=project_id,
            params={
                "mode": req.mode,
                "target_count": req.target_count,
                "target_class": req.target_class,
                "backend": req.backend,
                **({"params": playbook_params} if playbook_params else {}),
                **({"use_batch": True} if req.use_batch else {}),
            },
        )
        return JSONResponse(
            status_code=202,
            content=await _with_budget_warning(db, project_id, serialize_job(job)),
        )

    try:
        cloud_override = await _resolve_cloud_backend_override(
            db, project_id, req.backend,
        )
        if req.use_batch and cloud_override is not None:
            cloud_override.use_batch = True
        result = await run_playbook(
            db,
            project_id,
            mode,
            target_count=req.target_count,
            target_class=req.target_class,
            backend=req.backend if cloud_override is None else None,
            backend_override=cloud_override,
            params=playbook_params,
            use_batch=req.use_batch,
        )
    except SynthBackendError as e:
        raise HTTPException(503, str(e))
    except llm_budget_guard.LlmBudgetExceeded:
        raise  # -> 402 llm_budget_exceeded (main.py handler)
    except ValueError as e:
        message = str(e)
        if "not found" in message.lower():
            raise HTTPException(404, message)
        raise HTTPException(400, message)
    except Exception as e:  # noqa: BLE001 — last-resort wrap
        # Anything that wasn't caught above is most likely an LLM
        # transport failure that didn't get wrapped at the backend
        # layer (httpx error from a custom backend, JSON-decode error,
        # etc.). Surface as a 503 with the type + message so the
        # frontend can render an actionable error instead of a
        # generic "network error" 500.
        raise HTTPException(
            503,
            f"Synthetic generation failed ({type(e).__name__}): {e}",
        )

    return result


@router.post("/run-playbook/dry-run")
async def dry_run_synth_playbook(
    project_id: int,
    req: RunPlaybookRequest,
    db: AsyncSession = Depends(get_db),
):
    """Pre-flight a playbook with target_count=1 against the chosen
    model — **without persisting** any rows. The frontend calls this
    before kicking the real async job so a refusal or empty-output
    failure surfaces in <10s inline on the panel, instead of after a
    60-180s job that ends in a notification-bell error.

    Returns the same envelope as ``/run-playbook`` plus:

      - ``accepted_count``: number of rows that passed validation.
      - ``refusal_detected``: True when the LLM returned a short
        non-JSON apology that ``_looks_like_refusal`` recognised.
      - ``raw_llm_snippet``: first ~280 chars of the model's response
        so the user can see exactly what came back.
      - ``ok``: convenience flag — True only when accepted_count >= 1
        AND refusal_detected is False.

    Never raises for refusals or 0-row outputs — those land as
    ``ok: False`` in a 200 response so the frontend can render an
    inline error + retry-with-different-model affordance. 4xx is
    reserved for genuine input errors (unknown mode, missing recipe,
    missing gold set) and 5xx for backend transport failures.
    """
    from app.services.synth_backends import SynthBackendError
    from app.services.synth_playbook_service import run_playbook
    from app.services.synth_playbooks import SynthMode

    try:
        mode = SynthMode(req.mode)
    except ValueError:
        raise HTTPException(400, f"Unknown synth mode '{req.mode}'.")
    playbook_params = await _checked_playbook_params(db, project_id, mode, req.params)

    try:
        cloud_override = await _resolve_cloud_backend_override(
            db, project_id, req.backend,
        )
        # Force target_count=1 — the user's chosen count is for the
        # real run, not the pre-flight. ``use_batch`` is ignored: the
        # pre-flight is always one direct call (the override never batches).
        result = await run_playbook(
            db, project_id, mode,
            target_count=1,
            target_class=req.target_class,
            backend=req.backend if cloud_override is None else None,
            backend_override=cloud_override,
            dry_run=True,
            params=playbook_params,
        )
    except HTTPException:
        # Re-raise the 402-no-key / 400-malformed-pin errors from
        # _resolve_cloud_backend_override — don't 503-wrap them.
        raise
    except ValueError as e:
        message = str(e)
        if "not found" in message.lower():
            raise HTTPException(404, message)
        raise HTTPException(400, message)
    except SynthBackendError as e:
        # Backend isn't available — surface as a 200 with ok=False so
        # the panel can render "no backend available" inline rather
        # than as a generic toast.
        return {
            "ok": False,
            "rows": [],
            "accepted_count": 0,
            "refusal_detected": False,
            "raw_llm_snippet": "",
            "backend_used": req.backend or "auto",
            "elapsed_sec": 0.0,
            "prompt_snippet": "",
            "error": f"Backend unavailable: {e}",
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            503,
            f"Dry-run failed ({type(e).__name__}): {e}",
        )

    accepted_count = len(result.get("rows") or [])
    refusal = bool(result.get("refusal_detected"))
    return {
        "ok": accepted_count >= 1 and not refusal,
        "rows": result.get("rows") or [],
        "accepted_count": accepted_count,
        "refusal_detected": refusal,
        "raw_llm_snippet": result.get("raw_llm_snippet", ""),
        "backend_used": result.get("backend_used", req.backend or "auto"),
        "elapsed_sec": result.get("elapsed_sec", 0.0),
        "prompt_snippet": result.get("prompt_snippet", ""),
        "rejected_count": result.get("rejected_count", 0),
        "reject_summary": result.get("reject_summary") or {},
        "rejected_examples": result.get("rejected_examples") or [],
        **({"repaired_json": result["repaired_json"]} if result.get("repaired_json") else {}),
        **({"seed_pool": result["seed_pool"]} if result.get("seed_pool") else {}),
    }

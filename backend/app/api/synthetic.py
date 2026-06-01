"""Synthetic data generation API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
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


# Cloud provider → (secret_provider, key_name) for resolving saved
# API keys before instantiating a CloudLlmBackend. Mirrors the
# gold-generate flow's ``_PROVIDER_SECRET_MAP`` exactly so users only
# enter their key once per project.
_CLOUD_PROVIDER_SECRET_COORDS: dict[str, tuple[str, str]] = {
    "openai": ("cloud_llm_openai", "api_key"),
    "anthropic": ("cloud_llm_anthropic", "api_key"),
    "deepseek": ("cloud_llm_deepseek", "api_key"),
}


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
        {"id": "claude-opus-4-7", "label": "Claude Opus 4.7"},
    ],
    "deepseek": [
        {"id": "deepseek-chat", "label": "Deepseek V3 chat (cheap)"},
        {"id": "deepseek-reasoner", "label": "Deepseek R1 reasoner"},
    ],
}


async def _resolve_cloud_backend_override(
    db: AsyncSession, project_id: int, backend_pin: str | None,
):
    """If ``backend_pin`` looks like ``cloud:<provider>:<model>``,
    resolve the project's saved API key and return a constructed
    ``CloudLlmBackend``. Returns ``None`` for all other pin shapes
    so the existing ``pick_backend()`` path handles them.

    Raises ``HTTPException(400)`` for malformed pins and
    ``HTTPException(402)`` (payment required — semantically closest)
    when no API key is saved for the requested provider. The 402
    distinction matters: the frontend renders a "save key first"
    affordance for 402 and a generic error toast for 400/500."""
    if not backend_pin or not backend_pin.startswith("cloud:"):
        return None
    parts = backend_pin.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise HTTPException(
            400,
            f"Malformed cloud backend pin {backend_pin!r}. Expected "
            f"'cloud:<provider>:<model>' (e.g. 'cloud:openai:gpt-4o-mini').",
        )
    _, provider, model = parts
    coords = _CLOUD_PROVIDER_SECRET_COORDS.get(provider)
    if coords is None:
        raise HTTPException(
            400,
            f"Unknown cloud provider {provider!r}. Supported: "
            f"{', '.join(_CLOUD_PROVIDER_SECRET_COORDS.keys())}.",
        )
    secret_provider, key_name = coords
    from app.services.secret_service import get_project_secret_value

    api_key = await get_project_secret_value(
        db, project_id, secret_provider, key_name,
    )
    if not api_key:
        raise HTTPException(
            402,
            f"No {provider} API key saved for this project. Save one "
            f"under Project Settings → Secrets (or via the gold "
            f"generator's 'Save key for this project' option) and "
            f"retry.",
        )
    from app.services.synth_backends import CloudLlmBackend

    return CloudLlmBackend(provider=provider, model=model, api_key=api_key)


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
            }
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
):
    """Kick off a batched span-generation job. Returns immediately with
    a ``task_id``; clients poll ``GET /synthetic/tasks/{task_id}`` for
    progress + accumulated rows.

    Hardening Phase H2 — also spawns a shadow Job that mirrors the
    legacy task's progress into the top-bar notification bell.
    """
    try:
        task = start_span_generation_task(
            project_id=project_id,
            target_rows=req.target_rows,
            entity_types=req.entity_types,
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
        kind="synth_legacy_spans",
        title=f"Synth (legacy spans) · {req.target_rows} rows",
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


@router.get("/review-queue")
async def list_synth_review_queue(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List pending synthetic rows for the project, grouped by
    `synth_source` (USER-SUCCESS Epic 2b).

    Pending rows are gated out of dataset prep — they don't enter
    training until the user accepts them via `/review-queue/bulk-update`
    (or rejects them, in which case they're deleted).
    """
    from app.services.synth_review_queue_service import list_review_queue

    return await list_review_queue(db, project_id)


@router.post("/review-queue/bulk-update")
async def bulk_update_synth_review_queue(
    project_id: int,
    req: BulkReviewQueueRequest,
    db: AsyncSession = Depends(get_db),
):
    """Bulk accept or reject pending synth rows. Accepted rows flip
    to `review_status="accepted"` and become eligible for training;
    rejected rows are removed from synthetic.jsonl permanently.
    """
    from app.services.synth_review_queue_service import bulk_update_review_queue

    if req.action not in ("accept", "reject"):
        raise HTTPException(400, "action must be 'accept' or 'reject'")
    return await bulk_update_review_queue(
        db,
        project_id,
        row_ids=req.row_ids,
        action=req.action,  # type: ignore[arg-type]
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

    if async_job:
        import asyncio
        import time

        from fastapi.responses import JSONResponse

        from app.services.jobs_service import (
            JobProgressHandle,
            serialize_job,
            start_job,
        )

        target_class_label = (
            f" (target: {req.target_class})" if req.target_class else ""
        )
        title = (
            f"Synth · {req.mode}{target_class_label} · {req.target_count} rows"
        )

        async def _runner(handle: JobProgressHandle) -> dict:
            # Hardening Phase H2 — publish a live elapsed-time heartbeat
            # so the bell shows progress even when the underlying LLM
            # call is opaque (we can't crack open Ollama / vLLM /
            # NeMo to publish per-token progress, but we *can* tell
            # the user "still working, 47s elapsed"). The heartbeat
            # cancels as soon as the main work returns.
            started = time.monotonic()
            stop_heartbeat = asyncio.Event()

            async def _heartbeat() -> None:
                while not stop_heartbeat.is_set():
                    try:
                        await asyncio.wait_for(
                            stop_heartbeat.wait(), timeout=5.0,
                        )
                        return
                    except asyncio.TimeoutError:
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
                    result = await run_playbook(
                        runner_db,
                        project_id,
                        mode,
                        target_count=req.target_count,
                        target_class=req.target_class,
                        backend=req.backend if cloud_override is None else None,
                        backend_override=cloud_override,
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
                raw_snippet = result.get("raw_llm_snippet") or ""
                refused = bool(result.get("refusal_detected"))
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
                raise RuntimeError(
                    f"Playbook {req.mode} produced 0 accepted rows via "
                    f"{backend_used} in {elapsed_sec:.1f}s. "
                    f"Likely causes: (1) LLM backend returned text that "
                    f"didn't match the playbook's expected JSON shape, "
                    f"(2) every generated row failed validation (e.g. "
                    f"label outside the target class for hard-negatives), "
                    f"(3) backend timeout / network error swallowed by "
                    f"the runner. LLM response (first 300 chars): "
                    f"{raw_snippet[:300]!r}. Prompt sent (first 200 chars): "
                    f"{prompt_snippet[:200]!r}"
                )
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
            },
        )
        return JSONResponse(
            status_code=202,
            content=serialize_job(job),
        )

    try:
        cloud_override = await _resolve_cloud_backend_override(
            db, project_id, req.backend,
        )
        result = await run_playbook(
            db,
            project_id,
            mode,
            target_count=req.target_count,
            target_class=req.target_class,
            backend=req.backend if cloud_override is None else None,
            backend_override=cloud_override,
        )
    except SynthBackendError as e:
        raise HTTPException(503, str(e))
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

    try:
        cloud_override = await _resolve_cloud_backend_override(
            db, project_id, req.backend,
        )
        # Force target_count=1 — the user's chosen count is for the
        # real run, not the pre-flight.
        result = await run_playbook(
            db, project_id, mode,
            target_count=1,
            target_class=req.target_class,
            backend=req.backend if cloud_override is None else None,
            backend_override=cloud_override,
            dry_run=True,
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
    }

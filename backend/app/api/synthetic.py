"""Synthetic data generation API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.synthetic_service import (
    MAX_TOTAL_ROWS,
    generate_conversation_dialogues,
    generate_qa_pairs,
    generate_span_extraction_rows,
    get_span_task_status,
    save_synthetic_batch,
    save_synthetic_conversation_batch,
    save_synthetic_span_batch,
    start_span_generation_task,
)

router = APIRouter(prefix="/projects/{project_id}/synthetic", tags=["Synthetic"])


class GenerateRequest(BaseModel):
    source_text: str = Field(..., min_length=10)
    num_pairs: int = Field(5, ge=1, le=50)
    api_url: str = ""
    api_key: str = ""
    model_name: str = "llama3"


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


@router.post("/generate-spans-async", status_code=202)
async def generate_spans_async(
    project_id: int,
    req: GenerateSpanAsyncRequest,
):
    """Kick off a batched span-generation job. Returns immediately with
    a ``task_id``; clients poll ``GET /synthetic/tasks/{task_id}`` for
    progress + accumulated rows."""
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
    return {
        "task_id": task.task_id,
        "status": task.status,
        "target_rows": task.target_rows,
        "batches_total": (req.target_rows + 49) // 50,
    }


@router.get("/tasks/{task_id}")
async def get_synthetic_task(project_id: int, task_id: str):
    """Read the live state of a batched span-generation job. Returns
    ``rows`` once the task has completed (or partial rows while still
    running)."""
    task = get_span_task_status(task_id)
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


@router.get("/playbooks")
async def list_synth_playbooks(project_id: int, db: AsyncSession = Depends(get_db)):
    """Catalog of registered playbooks; if the project has a selected
    recipe, filter to playbooks compatible with that recipe."""
    from app.models.project import Project
    from app.services.synth_playbook_service import available_playbooks_for_recipe
    from app.services.synth_playbooks import list_playbooks

    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    recipe = (project.selected_recipe or {}).get("recipe_id")
    if recipe:
        return {
            "project_id": project_id,
            "recipe_id": recipe,
            "playbooks": available_playbooks_for_recipe(recipe),
        }
    # No recipe — return the full catalog so the user can preview.
    return {
        "project_id": project_id,
        "recipe_id": None,
        "playbooks": list_playbooks(),
    }


@router.post("/run-playbook")
async def run_synth_playbook(
    project_id: int,
    req: RunPlaybookRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run a playbook against the project's gold rows, generate
    synthetic training data, and persist accepted rows into the
    project's synthetic dataset.

    Returns a PlaybookResult: rows + backend_used + elapsed_sec +
    prompt_snippet.
    """
    from app.services.synth_backends import SynthBackendError
    from app.services.synth_playbook_service import run_playbook
    from app.services.synth_playbooks import SynthMode

    try:
        mode = SynthMode(req.mode)
    except ValueError:
        raise HTTPException(400, f"Unknown synth mode '{req.mode}'.")

    try:
        result = await run_playbook(
            db,
            project_id,
            mode,
            target_count=req.target_count,
            target_class=req.target_class,
            backend=req.backend,
        )
    except SynthBackendError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        message = str(e)
        if "not found" in message.lower():
            raise HTTPException(404, message)
        raise HTTPException(400, message)

    return result

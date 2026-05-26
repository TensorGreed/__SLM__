"""Gold evaluation dataset API routes."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import DatasetType
from app.services.cloud_llm_service import CloudLlmError
from app.services.gold_llm_service import (
    GoldGenerationError,
    generate_gold_qa_via_llm,
)
from app.services.gold_service import (
    add_qa_pair,
    get_gold_entries,
    import_qa_pairs,
    lock_gold_dataset,
)
from app.services.secret_service import get_project_secret_value

router = APIRouter(prefix="/projects/{project_id}/gold", tags=["Gold Dataset"])


# Project-secret coordinates for the per-project API keys. Both
# clients accept an inline key on the request (one-shot, doesn't
# persist) OR fall back to the stored secret. Matches the existing
# synth path's (provider, key_name) convention.
_OPENAI_SECRET = ("cloud_llm_openai", "api_key")
_ANTHROPIC_SECRET = ("cloud_llm_anthropic", "api_key")


class QAPairCreate(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    dataset_type: str = "gold_dev"
    difficulty: str = "medium"
    criticality: str = "normal"
    is_hallucination_trap: bool = False


class QAPairBatchImport(BaseModel):
    pairs: list[dict]
    dataset_type: str = "gold_dev"


@router.post("/add", status_code=201)
async def add_pair(
    project_id: int,
    data: QAPairCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add a Q&A pair to the gold dataset."""
    ds_type = DatasetType.GOLD_DEV if data.dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    try:
        entry = await add_qa_pair(
            db, project_id, data.question, data.answer,
            ds_type, data.difficulty, data.criticality, data.is_hallucination_trap,
        )
        return entry
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/import")
async def import_pairs(
    project_id: int,
    data: QAPairBatchImport,
    db: AsyncSession = Depends(get_db),
):
    """Import multiple Q&A pairs."""
    ds_type = DatasetType.GOLD_DEV if data.dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    try:
        result = await import_qa_pairs(db, project_id, data.pairs, ds_type)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── LLM-assisted generation ──────────────────────────────────────────


class GenerateViaLlmRequest(BaseModel):
    provider: Literal["openai", "anthropic"] = Field(
        ...,
        description="Cloud LLM provider. 'openai' also covers Deepseek + "
                    "any OpenAI-compatible endpoint via the optional api_url.",
    )
    model: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Model name (e.g. 'gpt-4o-mini', 'claude-haiku-4-5-20251001').",
    )
    count: int = Field(default=10, ge=1, le=50)
    focus_hint: str = Field(
        default="",
        max_length=1000,
        description="Optional user direction — 'focus on edge cases', "
                    "'cover refund policy questions', etc.",
    )
    api_key: str | None = Field(
        default=None,
        max_length=200,
        description="Inline API key (one-shot, not persisted). When omitted, "
                    "falls back to the stored project secret for this provider.",
    )
    api_url: str | None = Field(
        default=None,
        max_length=300,
        description="Override for OpenAI-compatible endpoints "
                    "(e.g. Deepseek). Ignored for Anthropic.",
    )


@router.post("/generate-via-llm")
async def generate_via_llm(
    project_id: int,
    req: GenerateViaLlmRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate Q&A pairs via a flagship cloud LLM (OpenAI or Anthropic)
    using the project's domain blueprint + selected recipe as context.

    Returns the generated rows for **preview** — does NOT persist.
    The caller renders the rows for user review and writes accepted
    rows via the existing ``POST /gold/import`` endpoint.

    Status codes:
      * 200 — rows generated + returned in the payload.
      * 400 — structured error (RECIPE_REQUIRED, RECIPE_NOT_SUPPORTED,
        API_KEY_REQUIRED, COUNT_OUT_OF_RANGE, LLM_RESPONSE_UNPARSEABLE,
        LLM_RESPONSE_COUNT_MISMATCH).
      * 502 — upstream provider failure (rate limit, bad key, model
        not found). Body has the provider's error text.
    """
    # Resolve the API key: inline > stored project secret.
    api_key = (req.api_key or "").strip()
    if not api_key:
        secret_provider, key_name = (
            _OPENAI_SECRET if req.provider == "openai" else _ANTHROPIC_SECRET
        )
        stored = await get_project_secret_value(
            db, project_id, secret_provider, key_name,
        )
        api_key = stored or ""
    if not api_key:
        raise HTTPException(
            400,
            detail={
                "error_code": "API_KEY_REQUIRED",
                "message": (
                    f"No {req.provider} API key found. Either include "
                    "``api_key`` in the request or store it in this "
                    "project's secrets first."
                ),
            },
        )

    try:
        result = await generate_gold_qa_via_llm(
            db,
            project_id=project_id,
            provider=req.provider,
            model=req.model,
            api_key=api_key,
            count=req.count,
            focus_hint=req.focus_hint,
            api_url=req.api_url,
        )
    except GoldGenerationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": exc.error_code, "message": str(exc)},
        )
    except CloudLlmError as exc:
        # Upstream LLM failure — surface as 502 (bad gateway) so the
        # UI can distinguish "you broke" (400) from "OpenAI broke" (502).
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "rows": [
            {
                "question": r.question,
                "answer": r.answer,
                "rationale": r.rationale,
            }
            for r in result.rows
        ],
        "provider": result.provider,
        "model": result.model,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
        "prompt_preview": result.prompt_preview,
    }


@router.get("/entries")
async def list_entries(
    project_id: int,
    dataset_type: str = "gold_dev",
    db: AsyncSession = Depends(get_db),
):
    """List all gold dataset entries."""
    ds_type = DatasetType.GOLD_DEV if dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    entries = await get_gold_entries(db, project_id, ds_type)
    return {"entries": entries, "total": len(entries)}


@router.post("/lock")
async def lock_dataset(
    project_id: int,
    dataset_type: str = "gold_dev",
    db: AsyncSession = Depends(get_db),
):
    """Lock a gold dataset (make immutable)."""
    ds_type = DatasetType.GOLD_DEV if dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    ds = await lock_gold_dataset(db, project_id, ds_type)
    return {"id": ds.id, "name": ds.name, "is_locked": ds.is_locked, "record_count": ds.record_count}

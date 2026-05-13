"""Synthetic data generation API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.synthetic_service import (
    generate_conversation_dialogues,
    generate_qa_pairs,
    generate_span_extraction_rows,
    save_synthetic_batch,
    save_synthetic_conversation_batch,
    save_synthetic_span_batch,
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

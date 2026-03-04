"""Compression API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.compression_service import (
    benchmark_model,
    merge_lora,
    quantize_model,
)

router = APIRouter(prefix="/projects/{project_id}/compression", tags=["Compression"])


class QuantizeRequest(BaseModel):
    model_path: str
    bits: int = Field(4, ge=2, le=8)
    output_format: str = "gguf"


class MergeLoRARequest(BaseModel):
    base_model_path: str
    lora_adapter_path: str


class BenchmarkRequest(BaseModel):
    model_path: str
    num_samples: int = Field(100, ge=1, le=1000)


@router.post("/quantize")
async def quantize(project_id: int, req: QuantizeRequest):
    """Quantize a model."""
    try:
        return await quantize_model(project_id, req.model_path, req.bits, req.output_format)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/merge-lora")
async def merge(project_id: int, req: MergeLoRARequest):
    """Merge LoRA adapter with base model."""
    try:
        return await merge_lora(project_id, req.base_model_path, req.lora_adapter_path)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/benchmark")
async def benchmark(project_id: int, req: BenchmarkRequest):
    """Benchmark model size and performance."""
    try:
        return await benchmark_model(project_id, req.model_path, req.num_samples)
    except ValueError as e:
        raise HTTPException(400, str(e))

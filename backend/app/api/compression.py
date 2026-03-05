"""Compression API routes."""
import asyncio

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.config import settings
from app.services.compression_service import (
    benchmark_model,
    get_compression_job_status,
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


@router.get("/jobs/status")
async def job_status(
    project_id: int,
    report_path: str = Query(..., min_length=1, max_length=2048),
):
    """Check compression job status by report path returned from queue calls."""
    try:
        return get_compression_job_status(project_id, report_path)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.websocket("/ws/logs")
async def compression_logs(websocket: WebSocket, project_id: int):
    """Stream compression worker log lines for a project."""
    await websocket.accept()
    redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = redis_client.pubsub()
    channel = f"log:compression:project:{project_id}"
    await pubsub.subscribe(channel)
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("type") == "message":
                await websocket.send_json({"type": "log", "text": str(message.get("data", ""))})
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
            except TimeoutError:
                pass
            except WebSocketDisconnect:
                break
    finally:
        await pubsub.unsubscribe(channel)
        await redis_client.aclose()

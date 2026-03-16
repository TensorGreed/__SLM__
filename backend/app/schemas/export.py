from pydantic import BaseModel
from typing import Any, List, Optional

class OptimizationRequest(BaseModel):
    target_id: str

class OptimizationMetric(BaseModel):
    latency_ms: float
    memory_gb: float
    quality_score: float

class OptimizationCandidate(BaseModel):
    id: str
    name: str
    quantization: str
    runtime_template: str
    metrics: OptimizationMetric
    is_recommended: bool = False
    reasons: List[str] = []

class OptimizationResponse(BaseModel):
    project_id: int
    target_id: str
    candidates: List[OptimizationCandidate]

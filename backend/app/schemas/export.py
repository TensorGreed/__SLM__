from typing import Any

from pydantic import BaseModel, Field


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
    reasons: list[str] = Field(default_factory=list)
    metric_source: str = "estimated"
    metric_sources: dict[str, str] = Field(default_factory=dict)
    measurement: dict[str, Any] | None = None
    confidence: dict[str, Any] | None = None


class OptimizationRunEvidence(BaseModel):
    run_id: str
    created_at: str
    run_path: str
    run_hash: str
    prompt_set_id: str
    prompt_set_hash: str
    candidate_count: int
    measured_candidate_count: int
    estimated_candidate_count: int
    mixed_candidate_count: int = 0
    recommended_candidate_id: str | None = None


class OptimizationResponse(BaseModel):
    project_id: int
    target_id: str
    candidates: list[OptimizationCandidate]
    optimization_run: OptimizationRunEvidence | None = None


class OptimizationMatrixStartRequest(BaseModel):
    target_ids: list[str] | None = None
    max_probe_candidates_per_target: int = Field(default=3, ge=1, le=10)


class OptimizationMatrixRunResponse(BaseModel):
    run_id: str
    project_id: int
    status: str
    run_hash: str | None = None
    run_path: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    target_ids: list[str] = Field(default_factory=list)
    prompt_set: dict[str, Any] = Field(default_factory=dict)
    runtime_fingerprint: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    targets: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None

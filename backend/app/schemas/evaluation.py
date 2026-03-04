"""Pydantic schemas for evaluation APIs."""

from datetime import datetime
from pydantic import BaseModel, Field


class EvalRunRequest(BaseModel):
    experiment_id: int
    dataset_names: list[str] = Field(default=["gold_dev", "gold_test"])
    eval_types: list[str] = Field(
        default=["exact_match", "f1", "hallucination", "safety"]
    )

class JudgePrediction(BaseModel):
    prompt: str = ""
    reference: str = Field(default="", min_length=0, max_length=20000)
    prediction: str = Field(..., min_length=1, max_length=20000)


class LLMJudgeRequest(BaseModel):
    experiment_id: int
    dataset_name: str = Field(..., min_length=1, max_length=255)
    judge_model: str = Field(default="meta-llama/Meta-Llama-3-70B-Instruct", min_length=1, max_length=255)
    predictions: list[JudgePrediction] = Field(..., min_length=1, max_length=5000)


class EvalMetricResponse(BaseModel):
    metric_name: str
    value: float
    details: dict | None = None


class EvalResultResponse(BaseModel):
    id: int
    experiment_id: int
    dataset_name: str
    eval_type: str
    metrics: dict
    pass_rate: float | None
    risk_severity: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SafetyScorecardResponse(BaseModel):
    experiment_id: int
    prompt_injection_pass_rate: float | None = None
    secret_extraction_pass_rate: float | None = None
    pii_regurgitation_pass_rate: float | None = None
    jailbreak_pass_rate: float | None = None
    unknown_answer_pass_rate: float | None = None
    overall_risk: str = "unknown"
    red_flags: list[str] = []


class RegressionComparisonResponse(BaseModel):
    experiment_id: int
    base_model_metrics: dict
    finetuned_metrics: dict
    improvements: dict
    regressions: dict

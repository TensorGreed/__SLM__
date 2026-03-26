"""Closed-loop remediation planning for evaluation failures."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.experiment import EvalResult, Experiment
from app.services.artifact_registry_service import (
    get_latest_artifact,
    list_artifacts,
    publish_artifact,
    serialize_artifact,
)
from app.services.evaluation_service import exact_match, f1_score

_REMEDIATION_ARTIFACT_PREFIX = "evaluation.remediation_plan"
_EVAL_RESULT_ARTIFACT_PREFIX = "evaluation.result"
_REMEDIATION_SCHEMA_REF = "slm.evaluation.remediation-plan/v1"
_EVAL_RESULT_SCHEMA_REF = "slm.evaluation-result/v1"

_ROOT_CAUSE_ORDER = {
    "safety_failure": 0,
    "hallucination": 1,
    "coverage_gap": 2,
    "formatting_mismatch": 3,
}


class RemediationPlanBlockedError(ValueError):
    """Raised when a remediation plan cannot be generated safely."""

    def __init__(
        self,
        message: str,
        *,
        actionable_fix: str,
        metadata: dict[str, Any] | None = None,
        error_code: str = "REMEDIATION_BLOCKED",
        status_code: int = 409,
    ):
        super().__init__(message)
        self.status_code = int(status_code)
        self.detail = {
            "error_code": str(error_code or "REMEDIATION_BLOCKED"),
            "stage": "evaluation",
            "message": str(message or "Remediation plan generation is blocked."),
            "actionable_fix": str(actionable_fix or "Run evaluation with failure examples and retry."),
            "docs_url": "/docs/evaluation/troubleshooting",
            "metadata": dict(metadata or {}),
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _project_eval_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "evaluation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _remediation_dir(project_id: int) -> Path:
    path = _project_eval_dir(project_id) / "remediation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_spaces(value).lower())


def _strip_wrapper_tokens(value: str) -> str:
    token = _normalize_spaces(value)
    token = re.sub(r"^```[a-z0-9_\-]*", "", token, flags=re.IGNORECASE).strip()
    token = re.sub(r"```$", "", token).strip()
    lowered = token.lower()
    for prefix in ("answer:", "response:", "output:", "final:"):
        if lowered.startswith(prefix):
            return token[len(prefix):].strip()
    return token


def _token_set(value: str) -> set[str]:
    return {item for item in re.findall(r"[a-z0-9]+", _normalize_spaces(value).lower()) if item}


def _is_unknown_answer(prediction: str) -> bool:
    normalized = _normalize_spaces(prediction).lower()
    if not normalized:
        return True
    unknown_markers = [
        "i don't know",
        "i do not know",
        "not sure",
        "cannot answer",
        "can't answer",
        "unknown",
        "no idea",
        "n/a",
    ]
    return any(marker in normalized for marker in unknown_markers)


def _looks_like_formatting_mismatch(prediction: str, reference: str) -> bool:
    pred = _clean_text(prediction)
    ref = _clean_text(reference)
    if not pred or not ref:
        return False

    pred_norm = _normalize_alnum(pred)
    ref_norm = _normalize_alnum(ref)
    if pred_norm and pred_norm == ref_norm and _normalize_spaces(pred) != _normalize_spaces(ref):
        return True

    stripped_pred = _strip_wrapper_tokens(pred)
    stripped_norm = _normalize_alnum(stripped_pred)
    if stripped_norm and stripped_norm == ref_norm and _normalize_spaces(stripped_pred) != _normalize_spaces(ref):
        return True

    pred_lower = pred.lower()
    if (
        "```" in pred_lower
        or pred_lower.startswith("{")
        or pred_lower.startswith("[")
        or pred_lower.startswith("answer:")
        or pred_lower.startswith("response:")
    ):
        ref_tokens = _token_set(ref)
        pred_tokens = _token_set(stripped_pred)
        if ref_tokens and ref_tokens.issubset(pred_tokens):
            return True

    return False


def _prompt_length_bucket(prompt: str) -> str:
    token_count = len(_token_set(prompt))
    if token_count <= 12:
        return "short"
    if token_count <= 40:
        return "medium"
    return "long"


def _slice_key(failure: dict[str, Any]) -> str:
    modality = str(failure.get("input_modality") or "text").strip().lower() or "text"
    prompt = _clean_text(failure.get("prompt"))
    bucket = _prompt_length_bucket(prompt)
    test_type = str(failure.get("test_type") or "").strip().lower()
    if test_type:
        return f"{modality}:{bucket}:{test_type}"
    return f"{modality}:{bucket}"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_primary_metric(eval_result: EvalResult) -> tuple[str, float | None]:
    metrics = dict(eval_result.metrics or {})
    if "pass_rate" in metrics:
        return "pass_rate", _safe_float(metrics.get("pass_rate"))
    if eval_result.pass_rate is not None:
        return "pass_rate", _safe_float(eval_result.pass_rate)
    if "exact_match" in metrics:
        return "exact_match", _safe_float(metrics.get("exact_match"))
    if "f1" in metrics:
        return "f1", _safe_float(metrics.get("f1"))
    if "average_score" in metrics:
        return "average_score", _safe_float(metrics.get("average_score"))
    return "pass_rate", _safe_float(eval_result.pass_rate)


def _extract_failures_from_eval_result(
    eval_result: EvalResult,
    *,
    max_failures: int,
) -> list[dict[str, Any]]:
    eval_type = str(eval_result.eval_type or "").strip().lower()
    metrics = dict(eval_result.metrics or {})
    details = dict(eval_result.details or {})
    failures: list[dict[str, Any]] = []

    if eval_type == "llm_judge":
        scored_predictions = [
            item
            for item in list(metrics.get("scored_predictions") or [])
            if isinstance(item, dict)
        ]
        for row in scored_predictions:
            score = int(_safe_float(row.get("judge_score")) or 0)
            if score >= 4:
                continue
            failures.append(
                {
                    "prompt": _clean_text(row.get("prompt")),
                    "reference": _clean_text(row.get("reference")),
                    "prediction": _clean_text(row.get("prediction")),
                    "judge_score": score,
                    "judge_rationale": _clean_text(row.get("judge_rationale")),
                    "input_modality": str(row.get("input_modality") or "text").strip().lower() or "text",
                    "eval_type": eval_type,
                    "source": "metrics.scored_predictions",
                }
            )
            if len(failures) >= max_failures:
                return failures

    if eval_type in {"exact_match", "f1"} and len(failures) < max_failures:
        preview_rows = [
            item
            for item in list(details.get("predictions_preview") or [])
            if isinstance(item, dict)
        ]
        for row in preview_rows:
            prompt = _clean_text(row.get("prompt"))
            reference = _clean_text(row.get("reference"))
            prediction = _clean_text(row.get("prediction"))
            if not reference and not prediction:
                continue

            if eval_type == "exact_match":
                if exact_match(prediction, reference) >= 1.0:
                    continue
            else:
                if f1_score(prediction, reference) >= 0.75:
                    continue

            failures.append(
                {
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": prediction,
                    "input_modality": str(row.get("input_modality") or "text").strip().lower() or "text",
                    "eval_type": eval_type,
                    "source": "details.predictions_preview",
                }
            )
            if len(failures) >= max_failures:
                return failures

    if eval_type == "safety" and len(failures) < max_failures:
        failed = int(_safe_float(metrics.get("failed")) or 0)
        pass_rate = _safe_float(metrics.get("pass_rate")) or _safe_float(eval_result.pass_rate) or 0.0
        fallback_count = failed if failed > 0 else (1 if pass_rate < 0.99 else 0)
        for _ in range(min(max_failures - len(failures), max(0, fallback_count))):
            failures.append(
                {
                    "prompt": "",
                    "reference": "",
                    "prediction": "",
                    "input_modality": "text",
                    "eval_type": eval_type,
                    "test_type": "safety",
                    "source": "metrics.aggregate",
                }
            )

    if not failures:
        metric_name, metric_value = _find_primary_metric(eval_result)
        fail_threshold = 0.95
        if metric_name == "average_score":
            fail_threshold = 4.0
        if metric_value is None or metric_value >= fail_threshold:
            return []
        failures.append(
            {
                "prompt": "",
                "reference": "",
                "prediction": "",
                "input_modality": "text",
                "eval_type": eval_type,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "source": "metrics.aggregate",
            }
        )

    return failures[:max_failures]


def _classify_failure(failure: dict[str, Any]) -> dict[str, Any]:
    eval_type = str(failure.get("eval_type") or "").strip().lower()
    prediction = _clean_text(failure.get("prediction"))
    reference = _clean_text(failure.get("reference"))

    if eval_type == "safety" or str(failure.get("test_type") or "").strip().lower() in {
        "safety",
        "prompt_injection",
        "secret_extraction",
        "jailbreak",
        "pii_regurgitation",
    }:
        return {
            "root_cause": "safety_failure",
            "confidence": 0.92,
            "reason": "Safety-focused evaluation failures indicate policy/alignment gaps.",
        }

    if _is_unknown_answer(prediction) and bool(reference):
        return {
            "root_cause": "coverage_gap",
            "confidence": 0.86,
            "reason": "Model response suggests missing domain coverage for this slice.",
        }

    if _looks_like_formatting_mismatch(prediction, reference):
        return {
            "root_cause": "formatting_mismatch",
            "confidence": 0.9,
            "reason": "Prediction content overlaps reference but formatting/schema is inconsistent.",
        }

    overlap = f1_score(prediction, reference) if reference else 0.0
    pred_tokens = _token_set(prediction)
    ref_tokens = _token_set(reference)

    if reference and overlap <= 0.15 and len(pred_tokens) >= max(6, len(ref_tokens) + 2):
        return {
            "root_cause": "hallucination",
            "confidence": 0.78,
            "reason": "Low reference overlap with verbose output suggests fabricated details.",
        }

    if reference and overlap < 0.45:
        return {
            "root_cause": "coverage_gap",
            "confidence": 0.67,
            "reason": "Partial mismatch suggests insufficient examples for this topic slice.",
        }

    metric_value = _safe_float(failure.get("metric_value"))
    if metric_value is not None:
        return {
            "root_cause": "coverage_gap",
            "confidence": 0.6,
            "reason": "Aggregate metric shortfall without row-level failures points to coverage gaps.",
        }

    return {
        "root_cause": "hallucination",
        "confidence": 0.55,
        "reason": "Failure pattern is low-confidence; treat as factuality mismatch first.",
    }


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _cluster_failures(failures: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in failures:
        classification = _classify_failure(row)
        root_cause = str(classification.get("root_cause") or "coverage_gap")
        confidence = float(classification.get("confidence") or 0.5)
        reason = str(classification.get("reason") or "Failure requires additional remediation.")
        slice_value = _slice_key(row)

        enriched_row = dict(row)
        enriched_row["root_cause"] = root_cause
        enriched_row["slice"] = slice_value
        enriched_row["confidence"] = round(confidence, 3)
        enriched_row["reason"] = reason
        enriched.append(enriched_row)

        grouped.setdefault((root_cause, slice_value), []).append(enriched_row)

    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            _ROOT_CAUSE_ORDER.get(item[0][0], 99),
            item[0][0],
            item[0][1],
        ),
    )

    clusters: list[dict[str, Any]] = []
    for idx, ((root_cause, slice_value), rows) in enumerate(ordered, start=1):
        avg_conf = sum(float(item.get("confidence") or 0.0) for item in rows) / max(1, len(rows))
        examples = [
            {
                "prompt": _clean_text(item.get("prompt"))[:240],
                "reference": _clean_text(item.get("reference"))[:240],
                "prediction": _clean_text(item.get("prediction"))[:240],
                "judge_score": item.get("judge_score"),
                "metric_name": item.get("metric_name"),
                "metric_value": item.get("metric_value"),
                "reason": _clean_text(item.get("reason")),
                "source": _clean_text(item.get("source")),
            }
            for item in rows[:3]
        ]
        clusters.append(
            {
                "cluster_id": f"cluster-{idx}",
                "root_cause": root_cause,
                "error_type": root_cause,
                "slice": slice_value,
                "failure_count": len(rows),
                "confidence": {
                    "score": round(avg_conf, 3),
                    "label": _confidence_label(avg_conf),
                },
                "reason": _clean_text(rows[0].get("reason")) if rows else "",
                "examples": examples,
            }
        )

    return clusters, enriched


def _impact_template(root_cause: str, eval_type: str) -> dict[str, Any]:
    metric = "pass_rate"
    if eval_type in {"exact_match", "f1", "llm_judge"}:
        metric = "pass_rate"
    if root_cause == "formatting_mismatch":
        return {
            "metric": metric,
            "direction": "increase",
            "estimate": "2-6%",
            "horizon": "next training iteration",
        }
    if root_cause == "coverage_gap":
        return {
            "metric": metric,
            "direction": "increase",
            "estimate": "4-10%",
            "horizon": "1-2 iterations",
        }
    if root_cause == "hallucination":
        return {
            "metric": metric,
            "direction": "increase",
            "estimate": "3-8%",
            "horizon": "1-2 iterations",
        }
    return {
        "metric": metric,
        "direction": "increase",
        "estimate": "5-12%",
        "horizon": "2 iterations",
    }


def _recommendation_templates(root_cause: str, slices: list[str], eval_type: str) -> dict[str, Any]:
    if root_cause == "formatting_mismatch":
        return {
            "title": "Enforce output format consistency",
            "why": "Failures preserve core content but violate the expected output schema/format.",
            "data_operations": [
                {
                    "operation": "augment",
                    "description": "Add format-constrained examples with the exact output schema.",
                    "target_slices": slices,
                },
                {
                    "operation": "filter",
                    "description": "Filter rows with inconsistent reference formatting before training.",
                    "target_slices": slices,
                },
            ],
            "training_config_changes": [
                {
                    "field": "response_template",
                    "suggested_value": "Set explicit answer template/prefix for generation.",
                    "reason": "Reduces schema drift at inference time.",
                },
                {
                    "field": "max_new_tokens",
                    "suggested_value": "Reduce by 10-20%",
                    "reason": "Limits verbose wrapper text around short structured outputs.",
                },
            ],
            "expected_impact": _impact_template(root_cause, eval_type),
        }

    if root_cause == "hallucination":
        return {
            "title": "Improve grounding and factuality",
            "why": "Predictions diverge strongly from references with fabricated details.",
            "data_operations": [
                {
                    "operation": "collect",
                    "description": "Collect grounded examples with explicit evidence in references.",
                    "target_slices": slices,
                },
                {
                    "operation": "augment",
                    "description": "Add negative examples where unsupported facts should be refused.",
                    "target_slices": slices,
                },
                {
                    "operation": "filter",
                    "description": "Remove low-quality/noisy rows that teach speculative answers.",
                    "target_slices": slices,
                },
            ],
            "training_config_changes": [
                {
                    "field": "temperature",
                    "suggested_value": "0.0-0.3 for eval and safety-critical inference",
                    "reason": "Lower sampling variance reduces fabricated continuations.",
                },
                {
                    "field": "validation_frequency",
                    "suggested_value": "Increase eval cadence during fine-tuning",
                    "reason": "Detect hallucination regressions earlier in training.",
                },
            ],
            "expected_impact": _impact_template(root_cause, eval_type),
        }

    if root_cause == "safety_failure":
        return {
            "title": "Harden safety behavior",
            "why": "Safety checks indicate policy bypass or unsafe completions.",
            "data_operations": [
                {
                    "operation": "collect",
                    "description": "Collect adversarial prompts matching failed safety categories.",
                    "target_slices": slices,
                },
                {
                    "operation": "augment",
                    "description": "Augment refusal and safe-completion demonstrations for risky prompts.",
                    "target_slices": slices,
                },
            ],
            "training_config_changes": [
                {
                    "field": "alignment_recipe",
                    "suggested_value": "Enable DPO/ORPO safety alignment pass",
                    "reason": "Improves refusal and policy consistency.",
                },
                {
                    "field": "safety_weight",
                    "suggested_value": "Increase safety loss weighting by 10-20%",
                    "reason": "Prioritizes safe behavior under adversarial inputs.",
                },
            ],
            "expected_impact": _impact_template(root_cause, eval_type),
        }

    return {
        "title": "Expand domain coverage",
        "why": "Model misses expected reference content for specific slices.",
        "data_operations": [
            {
                "operation": "collect",
                "description": "Collect more examples for the failing slices from production-like data.",
                "target_slices": slices,
            },
            {
                "operation": "augment",
                "description": "Generate paraphrases and edge cases for under-covered intents.",
                "target_slices": slices,
            },
            {
                "operation": "filter",
                "description": "Filter ambiguous rows with inconsistent target answers.",
                "target_slices": slices,
            },
        ],
        "training_config_changes": [
            {
                "field": "num_train_epochs",
                "suggested_value": "+1 epoch (bounded)",
                "reason": "Provides additional exposure to expanded coverage slices.",
            },
            {
                "field": "learning_rate",
                "suggested_value": "Reduce by 10-20%",
                "reason": "Stabilizes adaptation when introducing new domain data.",
            },
        ],
        "expected_impact": _impact_template(root_cause, eval_type),
    }


def _build_recommendations(
    *,
    clusters: list[dict[str, Any]],
    eval_result: EvalResult,
) -> list[dict[str, Any]]:
    by_root: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        root_cause = str(cluster.get("root_cause") or "coverage_gap")
        bucket = by_root.setdefault(
            root_cause,
            {
                "root_cause": root_cause,
                "failure_count": 0,
                "slice_set": set(),
                "confidence_scores": [],
            },
        )
        bucket["failure_count"] = int(bucket["failure_count"]) + int(cluster.get("failure_count") or 0)
        bucket["slice_set"].add(str(cluster.get("slice") or "text:short"))
        confidence_score = _safe_float((cluster.get("confidence") or {}).get("score")) or 0.5
        bucket["confidence_scores"].append(float(confidence_score))

    ordered_roots = sorted(
        by_root.values(),
        key=lambda item: (
            -int(item.get("failure_count") or 0),
            _ROOT_CAUSE_ORDER.get(str(item.get("root_cause") or "coverage_gap"), 99),
            str(item.get("root_cause") or "coverage_gap"),
        ),
    )

    recommendations: list[dict[str, Any]] = []
    for idx, root_payload in enumerate(ordered_roots, start=1):
        root_cause = str(root_payload.get("root_cause") or "coverage_gap")
        slices = sorted(str(item) for item in set(root_payload.get("slice_set") or set()))
        template = _recommendation_templates(root_cause, slices, str(eval_result.eval_type or ""))
        confidence_scores = [float(item) for item in list(root_payload.get("confidence_scores") or [])]
        avg_conf = sum(confidence_scores) / max(1, len(confidence_scores))
        count = int(root_payload.get("failure_count") or 0)
        confidence_score = min(0.95, max(0.45, avg_conf + min(0.2, count / 50.0)))
        recommendation = {
            "recommendation_id": f"rec-{idx}",
            "priority": idx,
            "root_cause": root_cause,
            "title": str(template.get("title") or "Remediation action"),
            "why": str(template.get("why") or ""),
            "failure_count": count,
            "target_slices": slices,
            "data_operations": list(template.get("data_operations") or []),
            "training_config_changes": list(template.get("training_config_changes") or []),
            "expected_impact": dict(template.get("expected_impact") or {}),
            "confidence": {
                "score": round(confidence_score, 3),
                "label": _confidence_label(confidence_score),
            },
        }
        recommendations.append(recommendation)

    return recommendations


async def _get_experiment_for_project(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
) -> Experiment | None:
    result = await db.execute(
        select(Experiment).where(
            Experiment.project_id == project_id,
            Experiment.id == experiment_id,
        )
    )
    return result.scalar_one_or_none()


async def _resolve_eval_result(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    evaluation_result_id: int | None,
) -> EvalResult:
    exp = await _get_experiment_for_project(db, project_id=project_id, experiment_id=experiment_id)
    if exp is None:
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")

    query = (
        select(EvalResult)
        .join(Experiment, Experiment.id == EvalResult.experiment_id)
        .where(
            Experiment.project_id == project_id,
            EvalResult.experiment_id == experiment_id,
        )
        .order_by(EvalResult.created_at.desc(), EvalResult.id.desc())
    )
    if evaluation_result_id is not None:
        query = query.where(EvalResult.id == evaluation_result_id)

    result = await db.execute(query.limit(1))
    row = result.scalar_one_or_none()
    if row is None:
        if evaluation_result_id is not None:
            raise ValueError(
                f"Evaluation result {evaluation_result_id} not found for experiment {experiment_id} in project {project_id}"
            )
        raise ValueError(f"No evaluation results found for experiment {experiment_id} in project {project_id}")
    return row


async def _ensure_eval_result_artifact_link(
    db: AsyncSession,
    *,
    project_id: int,
    eval_result: EvalResult,
) -> dict[str, Any]:
    artifact_key = f"{_EVAL_RESULT_ARTIFACT_PREFIX}.{int(eval_result.id)}"
    existing = await get_latest_artifact(
        db,
        project_id=project_id,
        artifact_key=artifact_key,
    )
    if existing is not None:
        return serialize_artifact(existing)

    record = await publish_artifact(
        db=db,
        project_id=project_id,
        artifact_key=artifact_key,
        uri="",
        schema_ref=_EVAL_RESULT_SCHEMA_REF,
        producer_stage="evaluation",
        producer_run_id=f"evaluation-result-{eval_result.id}",
        producer_step_id=f"experiment:{eval_result.experiment_id}",
        metadata={
            "evaluation_result_id": int(eval_result.id),
            "experiment_id": int(eval_result.experiment_id),
            "eval_type": str(eval_result.eval_type or ""),
            "dataset_name": str(eval_result.dataset_name or ""),
            "created_at": eval_result.created_at.isoformat() if eval_result.created_at else None,
        },
    )
    return serialize_artifact(record)


def _persist_plan_file(
    *,
    project_id: int,
    plan_id: str,
    payload: dict[str, Any],
) -> Path:
    target = _remediation_dir(project_id) / f"{plan_id}.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return target


def _load_plan_from_uri(uri: str | None) -> dict[str, Any] | None:
    path = Path(str(uri or "").strip()).expanduser()
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _build_plan_summary(
    *,
    clusters: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    failures_count: int,
) -> dict[str, Any]:
    dominant_root_cause = str(clusters[0].get("root_cause") or "") if clusters else ""
    return {
        "total_failures_analyzed": int(failures_count),
        "cluster_count": len(clusters),
        "recommendation_count": len(recommendations),
        "dominant_root_cause": dominant_root_cause or None,
    }


async def generate_remediation_plan(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    evaluation_result_id: int | None = None,
    max_failures: int = 200,
) -> dict[str, Any]:
    eval_result = await _resolve_eval_result(
        db,
        project_id=project_id,
        experiment_id=experiment_id,
        evaluation_result_id=evaluation_result_id,
    )

    failures = _extract_failures_from_eval_result(
        eval_result,
        max_failures=max(1, min(int(max_failures or 200), 1000)),
    )
    if not failures:
        metric_name, metric_value = _find_primary_metric(eval_result)
        raise RemediationPlanBlockedError(
            "Evaluation does not contain failing samples to remediate.",
            error_code="REMEDIATION_NOT_REQUIRED",
            actionable_fix=(
                "Run a broader held-out or LLM-judge evaluation with failure examples, then regenerate remediation."
            ),
            metadata={
                "experiment_id": int(experiment_id),
                "evaluation_result_id": int(eval_result.id),
                "eval_type": str(eval_result.eval_type or ""),
                "primary_metric": metric_name,
                "primary_metric_value": metric_value,
            },
            status_code=409,
        )

    clusters, enriched_failures = _cluster_failures(failures)
    recommendations = _build_recommendations(
        clusters=clusters,
        eval_result=eval_result,
    )

    evidence_payload = {
        "experiment_id": int(experiment_id),
        "evaluation_result_id": int(eval_result.id),
        "eval_type": str(eval_result.eval_type or ""),
        "dataset_name": str(eval_result.dataset_name or ""),
        "failure_count": len(enriched_failures),
        "clusters": [
            {
                "root_cause": cluster.get("root_cause"),
                "slice": cluster.get("slice"),
                "failure_count": cluster.get("failure_count"),
            }
            for cluster in clusters
        ],
    }
    run_hash = _hash_payload(evidence_payload)[:24]
    plan_id = f"remediation-{int(experiment_id)}-{int(eval_result.id)}-{run_hash}"

    eval_artifact = await _ensure_eval_result_artifact_link(
        db,
        project_id=project_id,
        eval_result=eval_result,
    )

    summary = _build_plan_summary(
        clusters=clusters,
        recommendations=recommendations,
        failures_count=len(enriched_failures),
    )

    plan_payload: dict[str, Any] = {
        "plan_id": plan_id,
        "project_id": int(project_id),
        "experiment_id": int(experiment_id),
        "evaluation_result_id": int(eval_result.id),
        "created_at": _utcnow_iso(),
        "analysis_version": "remediation.v1",
        "source_evaluation": {
            "id": int(eval_result.id),
            "experiment_id": int(eval_result.experiment_id),
            "eval_type": str(eval_result.eval_type or ""),
            "dataset_name": str(eval_result.dataset_name or ""),
            "pass_rate": _safe_float(eval_result.pass_rate),
            "risk_severity": _clean_text(eval_result.risk_severity) or None,
            "created_at": eval_result.created_at.isoformat() if eval_result.created_at else None,
        },
        "summary": summary,
        "clusters": clusters,
        "recommendations": recommendations,
        "analysis_evidence": {
            "failures_analyzed": len(enriched_failures),
            "max_failures": max(1, min(int(max_failures or 200), 1000)),
            "failure_preview": [
                {
                    "root_cause": row.get("root_cause"),
                    "slice": row.get("slice"),
                    "prompt": _clean_text(row.get("prompt"))[:200],
                    "reference": _clean_text(row.get("reference"))[:200],
                    "prediction": _clean_text(row.get("prediction"))[:200],
                    "source": _clean_text(row.get("source")),
                }
                for row in enriched_failures[:10]
            ],
        },
        "linked_artifacts": {
            "evaluation_result_artifact": eval_artifact,
        },
    }

    plan_path = _persist_plan_file(
        project_id=project_id,
        plan_id=plan_id,
        payload=plan_payload,
    )

    remediation_artifact_key = f"{_REMEDIATION_ARTIFACT_PREFIX}.{int(eval_result.id)}"
    remediation_artifact = await publish_artifact(
        db=db,
        project_id=project_id,
        artifact_key=remediation_artifact_key,
        uri=str(plan_path),
        schema_ref=_REMEDIATION_SCHEMA_REF,
        producer_stage="evaluation",
        producer_run_id=plan_id,
        producer_step_id=f"evaluation_result:{eval_result.id}",
        metadata={
            "plan_id": plan_id,
            "experiment_id": int(experiment_id),
            "evaluation_result_id": int(eval_result.id),
            "eval_type": str(eval_result.eval_type or ""),
            "dataset_name": str(eval_result.dataset_name or ""),
            "summary": summary,
            "root_causes": sorted({str(item.get("root_cause") or "") for item in clusters if item.get("root_cause")}),
            "linked_evaluation_artifact_key": eval_artifact.get("artifact_key"),
            "linked_evaluation_artifact_id": eval_artifact.get("id"),
        },
    )
    remediation_artifact_payload = serialize_artifact(remediation_artifact)

    plan_payload["linked_artifacts"]["remediation_plan_artifact"] = remediation_artifact_payload
    _persist_plan_file(
        project_id=project_id,
        plan_id=plan_id,
        payload=plan_payload,
    )

    return plan_payload


def _serialize_remediation_plan_index_row(artifact_row) -> dict[str, Any]:
    metadata = dict(artifact_row.metadata_ or {})
    summary = dict(metadata.get("summary") or {})
    return {
        "plan_id": str(metadata.get("plan_id") or artifact_row.producer_run_id or f"artifact-{artifact_row.id}"),
        "artifact_id": int(artifact_row.id),
        "artifact_key": str(artifact_row.artifact_key or ""),
        "artifact_version": int(artifact_row.version),
        "created_at": artifact_row.created_at.isoformat() if artifact_row.created_at else None,
        "experiment_id": int(metadata.get("experiment_id") or 0) if metadata.get("experiment_id") is not None else None,
        "evaluation_result_id": (
            int(metadata.get("evaluation_result_id") or 0)
            if metadata.get("evaluation_result_id") is not None
            else None
        ),
        "eval_type": str(metadata.get("eval_type") or "") or None,
        "dataset_name": str(metadata.get("dataset_name") or "") or None,
        "root_causes": [
            str(item)
            for item in list(metadata.get("root_causes") or [])
            if str(item).strip()
        ],
        "summary": {
            "total_failures_analyzed": summary.get("total_failures_analyzed"),
            "cluster_count": summary.get("cluster_count"),
            "recommendation_count": summary.get("recommendation_count"),
            "dominant_root_cause": summary.get("dominant_root_cause"),
        },
        "uri": str(artifact_row.uri or ""),
    }


async def list_remediation_plan_index(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int | None = None,
    evaluation_result_id: int | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    fetch_limit = max(1, min(int(limit or 20) * 10, 500))
    rows = await list_artifacts(db, project_id, artifact_key=None, limit=fetch_limit)
    output: list[dict[str, Any]] = []
    for row in rows:
        key = str(getattr(row, "artifact_key", "") or "")
        if not key.startswith(_REMEDIATION_ARTIFACT_PREFIX):
            continue
        metadata = dict(getattr(row, "metadata_", {}) or {})
        row_experiment_id = metadata.get("experiment_id")
        row_eval_id = metadata.get("evaluation_result_id")
        if experiment_id is not None and int(row_experiment_id or -1) != int(experiment_id):
            continue
        if evaluation_result_id is not None and int(row_eval_id or -1) != int(evaluation_result_id):
            continue
        output.append(_serialize_remediation_plan_index_row(row))
        if len(output) >= max(1, min(int(limit or 20), 200)):
            break
    return output


async def get_remediation_plan(
    db: AsyncSession,
    *,
    project_id: int,
    plan_id: str,
) -> dict[str, Any] | None:
    token = str(plan_id or "").strip()
    if not token:
        return None

    rows = await list_artifacts(db, project_id, artifact_key=None, limit=500)
    for row in rows:
        key = str(getattr(row, "artifact_key", "") or "")
        if not key.startswith(_REMEDIATION_ARTIFACT_PREFIX):
            continue
        metadata = dict(getattr(row, "metadata_", {}) or {})
        candidate_plan_id = str(metadata.get("plan_id") or row.producer_run_id or "").strip()
        if candidate_plan_id != token:
            continue

        payload = _load_plan_from_uri(str(getattr(row, "uri", "") or ""))
        if payload is None:
            payload = {
                "plan_id": candidate_plan_id,
                "project_id": project_id,
                "experiment_id": metadata.get("experiment_id"),
                "evaluation_result_id": metadata.get("evaluation_result_id"),
                "summary": dict(metadata.get("summary") or {}),
                "clusters": [],
                "recommendations": [],
                "analysis_evidence": {"load_warning": "Plan file missing or invalid; returning artifact metadata."},
            }

        linked_artifacts = dict(payload.get("linked_artifacts") or {})
        linked_artifacts["remediation_plan_artifact"] = serialize_artifact(row)
        payload["linked_artifacts"] = linked_artifacts
        return payload

    return None

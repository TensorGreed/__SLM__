"""Newbie autopilot intent mapping for zero-knowledge training UX."""

from __future__ import annotations

import re
from statistics import mean, median, pstdev
from pathlib import Path
from typing import Any

from app.config import settings
from app.services import target_profile_service


_INTENT_PRESETS: list[dict[str, Any]] = [
    {
        "preset_id": "autopilot.support_qa_safe",
        "label": "Support Q&A Assistant",
        "description": "Answer customer/support questions with grounded, concise replies.",
        "task_profile": "qa",
        "task_type": "causal_lm",
        "chat_template": "llama3",
        "keywords": (
            "support",
            "helpdesk",
            "ticket",
            "faq",
            "customer question",
            "answer questions",
            "qa",
        ),
    },
    {
        "preset_id": "autopilot.structured_extraction_safe",
        "label": "Structured Extraction",
        "description": "Extract key fields from contracts/forms/invoices into consistent outputs.",
        "task_profile": "structured_extraction",
        "task_type": "causal_lm",
        "chat_template": "chatml",
        "keywords": (
            "extract",
            "extraction",
            "invoice",
            "contract",
            "form",
            "fields",
            "json",
            "structured",
        ),
    },
    {
        "preset_id": "autopilot.summarization_safe",
        "label": "Summarization Assistant",
        "description": "Summarize long content into concise, useful outputs.",
        "task_profile": "summarization",
        "task_type": "causal_lm",
        "chat_template": "llama3",
        "keywords": (
            "summarize",
            "summary",
            "tldr",
            "brief",
            "digest",
            "notes",
            "recap",
        ),
    },
    {
        "preset_id": "autopilot.classification_safe",
        "label": "Text Classifier",
        "description": "Classify text into labels with conservative defaults.",
        "task_profile": "classification",
        "task_type": "classification",
        "chat_template": "chatml",
        "keywords": (
            "classify",
            "classification",
            "categorize",
            "category",
            "sentiment",
            "label",
            "routing",
        ),
    },
]

_DEFAULT_PRESET: dict[str, Any] = {
    "preset_id": "autopilot.general_chat_safe",
    "label": "General Assistant",
    "description": "General-purpose assistant behavior with safe starter defaults.",
    "task_profile": "instruction_sft",
    "task_type": "causal_lm",
    "chat_template": "llama3",
    "keywords": (),
}

_PLAN_PROFILE_ALIASES: dict[str, str] = {
    "fastest": "safe",
    "best_quality": "max_quality",
}

_PLAN_PROFILES: set[str] = {"safe", "balanced", "max_quality"}

_TARGET_BASE_TIME_PER_1K_ROWS: dict[str, float] = {
    "mobile_cpu": 1200.0,
    "browser_webgpu": 900.0,
    "edge_gpu": 220.0,
    "vllm_server": 60.0,
}

_PLAN_PROFILE_MULTIPLIER: dict[str, float] = {
    "safe": 0.5,
    "balanced": 1.0,
    "max_quality": 2.0,
}

_COHORT_CONFIDENCE_WEIGHT: dict[str, float] = {
    "target+profile+task": 1.0,
    "target+profile": 0.9,
    "profile+task": 0.82,
    "target+task": 0.78,
    "profile": 0.72,
    "target": 0.66,
    "global": 0.52,
    "none": 0.3,
}


def _normalize_intent(intent: str) -> str:
    token = str(intent or "").strip().lower()
    token = re.sub(r"\s+", " ", token)
    return token


def _normalize_target_profile(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "vllm_server"
    return token


def _safe_defaults_for_target(target_profile_id: str) -> dict[str, Any]:
    target = target_profile_service.get_target_by_id(target_profile_id)
    if not target:
        return {"batch_size": 1, "max_seq_length": 512}
    
    # Heuristics based on target
    if target_profile_id == "mobile_cpu":
        return {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 1024,
            "num_epochs": 2,
        }
    if target_profile_id == "vllm_server":
        return {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "num_epochs": 3,
        }
    return {
        "batch_size": 2,
        "gradient_accumulation_steps": 6,
        "max_seq_length": 1536,
        "num_epochs": 2,
    }


def _score_preset(intent: str, preset: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    matched: list[str] = []
    for raw_keyword in list(preset.get("keywords") or []):
        keyword = str(raw_keyword or "").strip().lower()
        if not keyword:
            continue
        if keyword in intent:
            score += 1
            matched.append(keyword)
    return score, matched


def _confidence_band(confidence: float) -> str:
    value = max(0.0, min(float(confidence), 1.0))
    if value >= 0.8:
        return "high"
    if value >= 0.6:
        return "medium"
    return "low"


def _normalize_plan_profile(value: Any, *, default: str | None = None) -> str | None:
    token = str(value or "").strip().lower()
    if not token:
        return default
    token = _PLAN_PROFILE_ALIASES.get(token, token)
    if token in _PLAN_PROFILES:
        return token
    return default


def _normalize_task_profile(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    alias = {
        "chat_sft": "instruction_sft",
        "instruction": "instruction_sft",
        "qa_assistant": "qa",
    }
    return alias.get(token, token)


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _coerce_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _heuristic_seconds_estimate(
    *,
    plan_profile: str,
    target_profile_id: str,
    dataset_size_rows: int,
) -> int:
    base_time_per_1k_rows = float(_TARGET_BASE_TIME_PER_1K_ROWS.get(target_profile_id, 300.0))
    multiplier = float(_PLAN_PROFILE_MULTIPLIER.get(plan_profile, 1.0))
    estimated_seconds = int(base_time_per_1k_rows * (max(1, int(dataset_size_rows)) / 1000.0) * multiplier)
    return max(30, estimated_seconds)


def _cohort_rows(
    *,
    rows: list[dict[str, Any]],
    target_profile_id: str,
    plan_profile: str,
    task_profile: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    with_metric = [row for row in rows if _coerce_positive_float(row.get("seconds_per_1k_rows"))]
    if not with_metric:
        return "none", []

    task_token = _normalize_task_profile(task_profile)
    selectors: list[tuple[str, Any]] = []
    if task_token:
        selectors.extend(
            [
                (
                    "target+profile+task",
                    lambda row: (
                        row.get("target_profile_id") == target_profile_id
                        and row.get("plan_profile") == plan_profile
                        and row.get("task_profile") == task_token
                    ),
                ),
                (
                    "target+profile",
                    lambda row: (
                        row.get("target_profile_id") == target_profile_id
                        and row.get("plan_profile") == plan_profile
                    ),
                ),
                (
                    "profile+task",
                    lambda row: (
                        row.get("plan_profile") == plan_profile
                        and row.get("task_profile") == task_token
                    ),
                ),
                (
                    "target+task",
                    lambda row: (
                        row.get("target_profile_id") == target_profile_id
                        and row.get("task_profile") == task_token
                    ),
                ),
            ]
        )
    selectors.extend(
        [
            (
                "profile",
                lambda row: row.get("plan_profile") == plan_profile,
            ),
            (
                "target",
                lambda row: row.get("target_profile_id") == target_profile_id,
            ),
            (
                "global",
                lambda row: True,
            ),
        ]
    )

    for label, selector in selectors:
        subset = [row for row in with_metric if selector(row)]
        if len(subset) >= 2:
            return label, subset
    for label, selector in selectors:
        subset = [row for row in with_metric if selector(row)]
        if subset:
            return label, subset
    return "none", []


def _estimate_hourly_cost_usd(target_profile_id: str) -> tuple[float, str, dict[str, Any]]:
    normalized_target = _normalize_target_profile(target_profile_id)
    if normalized_target in {"mobile_cpu", "browser_webgpu"}:
        return 0.0, "local_runtime", {"reason": "Target runs locally; cloud pricing is not applied."}

    if normalized_target == "vllm_server":
        try:
            from app.services.cloud_burst_service import list_cloud_burst_catalog

            catalog = dict(list_cloud_burst_catalog() or {})
            gpu_rows = [item for item in list(catalog.get("gpu_skus") or []) if isinstance(item, dict)]
            preferred = next(
                (
                    item
                    for item in gpu_rows
                    if str(item.get("gpu_sku") or "").strip().lower() == "a10g.24gb"
                ),
                gpu_rows[0] if gpu_rows else {},
            )
            hourly_map = dict(preferred.get("hourly_usd") or {})
            rates = [float(value) for value in hourly_map.values() if _coerce_positive_float(value) is not None]
            if rates:
                hourly = round(sum(rates) / float(len(rates)), 4)
                return hourly, "cloud_burst_catalog_avg", {
                    "gpu_sku": str(preferred.get("gpu_sku") or "a10g.24gb"),
                    "provider_count": len(rates),
                }
        except Exception:
            pass
        return 1.0, "fallback_server_heuristic", {"reason": "Cloud burst pricing catalog unavailable."}

    if normalized_target == "edge_gpu":
        return 0.15, "edge_gpu_heuristic", {"reason": "Estimated local edge-GPU operating cost."}

    return 0.2, "generic_heuristic", {"reason": "No target-specific pricing available."}


def _intent_rewrite_suggestions_for_task_profile(
    *,
    task_profile: str | None,
) -> list[dict[str, Any]]:
    profile = str(task_profile or "").strip().lower() or "instruction_sft"
    if profile == "structured_extraction":
        return [
            {
                "id": "rewrite.structured_extraction.json",
                "label": "Extract key fields as JSON",
                "rewritten_intent": (
                    "Extract invoice number, vendor name, due date, and total amount from each "
                    "document and return strict JSON only."
                ),
                "reason": "Defines explicit fields and output format.",
                "recommended": True,
            },
            {
                "id": "rewrite.structured_extraction.qa",
                "label": "Contract clause extraction",
                "rewritten_intent": (
                    "Extract liability and payment clauses from contract text and return a concise "
                    "JSON object with clause type and evidence sentence."
                ),
                "reason": "Clarifies extraction target and expected structure.",
                "recommended": False,
            },
        ]
    if profile == "summarization":
        return [
            {
                "id": "rewrite.summarization.support",
                "label": "Ticket summary with actions",
                "rewritten_intent": (
                    "Summarize each support ticket into 3 bullet points plus next actions and urgency level."
                ),
                "reason": "Specifies summary format and decision-support fields.",
                "recommended": True,
            },
            {
                "id": "rewrite.summarization.exec",
                "label": "Executive brief summary",
                "rewritten_intent": (
                    "Generate a short executive summary of each thread with key risks and recommended actions."
                ),
                "reason": "Clarifies audience and desired output style.",
                "recommended": False,
            },
        ]
    if profile == "classification":
        return [
            {
                "id": "rewrite.classification.routing",
                "label": "Message routing labels",
                "rewritten_intent": (
                    "Classify each incoming message into exactly one label: billing, technical, or cancellation."
                ),
                "reason": "Defines fixed labels and single-label behavior.",
                "recommended": True,
            },
            {
                "id": "rewrite.classification.sentiment",
                "label": "Customer sentiment",
                "rewritten_intent": (
                    "Classify customer sentiment as positive, neutral, or negative and return only the label."
                ),
                "reason": "Specifies label set and concise output.",
                "recommended": False,
            },
        ]
    if profile == "qa":
        return [
            {
                "id": "rewrite.qa.support_answer",
                "label": "Support Q&A answer drafting",
                "rewritten_intent": (
                    "Answer customer support questions in concise plain language with one recommended next step."
                ),
                "reason": "Clarifies response objective and style.",
                "recommended": True,
            },
            {
                "id": "rewrite.qa.faq",
                "label": "FAQ-style response",
                "rewritten_intent": (
                    "Read a support question and draft a short FAQ-style answer with clear troubleshooting steps."
                ),
                "reason": "Specifies answer format for consistency.",
                "recommended": False,
            },
        ]
    return [
        {
            "id": "rewrite.general.task_output",
            "label": "Define output behavior clearly",
            "rewritten_intent": (
                "Given an input text, produce a concise and reliable output in a consistent format with examples."
            ),
            "reason": "Makes the task and output constraints explicit.",
            "recommended": True,
        },
        {
            "id": "rewrite.general.helpdesk",
            "label": "General assistant for support text",
            "rewritten_intent": (
                "Read user requests and generate safe, helpful responses that are short, accurate, and actionable."
            ),
            "reason": "Improves clarity around desired assistant behavior.",
            "recommended": False,
        },
    ]


def build_newbie_autopilot_intent_clarification(
    *,
    intent: str,
    confidence: float,
    task_profile: str | None = None,
    matched_keywords: list[str] | None = None,
) -> dict[str, Any]:
    normalized_intent = _normalize_intent(intent)
    matched = [str(item).strip() for item in list(matched_keywords or []) if str(item).strip()]
    word_count = len([item for item in normalized_intent.split(" ") if item])
    band = _confidence_band(confidence)

    requires_clarification = False
    reasons: list[str] = []
    if band == "low":
        requires_clarification = True
        reasons.append("intent confidence is low")
    if word_count < 6:
        requires_clarification = True
        reasons.append("intent text is very short")
    if not matched:
        requires_clarification = True
        reasons.append("no clear task keywords were detected")

    questions = [
        "What exact output should the model produce?",
        "What are 2-3 example inputs and expected answers?",
        "Should responses be concise summaries, labels, or structured JSON?",
    ]
    suggested_intent_examples = [
        "Summarize support tickets into 3 bullet points and next actions.",
        "Extract invoice number, due date, vendor, and total into JSON.",
        "Classify incoming messages into billing, technical, or cancellation.",
    ]
    rewrite_suggestions = _intent_rewrite_suggestions_for_task_profile(task_profile=task_profile)
    reason = "; ".join(reasons) if reasons else ""
    return {
        "required": requires_clarification,
        "confidence_band": band,
        "reason": reason or None,
        "matched_keywords": matched,
        "questions": questions if requires_clarification else [],
        "suggested_intent_examples": suggested_intent_examples if requires_clarification else [],
        "rewrite_suggestions": rewrite_suggestions if requires_clarification else [],
    }


def _project_prepared_train_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"


def evaluate_newbie_autopilot_dataset_readiness(
    *,
    project_id: int,
    min_rows: int = 20,
) -> dict[str, Any]:
    safe_min_rows = max(1, int(min_rows))
    train_path = _project_prepared_train_path(project_id)
    blockers: list[str] = []
    warnings: list[str] = []
    hints: list[str] = []
    auto_fixes: list[dict[str, str]] = []
    row_count = 0
    valid_json_rows = 0
    invalid_json_rows = 0

    if not train_path.exists():
        blockers.append("Prepared training split is missing (`prepared/train.jsonl`).")
        hints.append("Go to Data pipeline tab and run dataset split/prep before training.")
        auto_fixes.append(
            {
                "id": "open_data_pipeline",
                "label": "Open Data Prep",
                "description": "Open data prep tab and create prepared train/val splits.",
                "action_type": "navigate",
                "navigate_to": f"/project/{project_id}/pipeline/data",
            }
        )
    else:
        try:
            with open(train_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    token = str(line or "").strip()
                    if not token:
                        continue
                    row_count += 1
                    if token.startswith("{") and token.endswith("}"):
                        valid_json_rows += 1
                    else:
                        invalid_json_rows += 1
        except Exception:
            blockers.append("Prepared train split could not be read.")
            hints.append("Check file permissions and regenerate dataset split.")

    if row_count == 0 and not blockers:
        blockers.append("Prepared train split has zero usable rows.")
    elif 0 < row_count < safe_min_rows:
        warnings.append(
            f"Prepared train split is very small ({row_count} rows). Target at least {safe_min_rows} rows."
        )
        hints.append("Add more examples or generate synthetic data before one-click run.")
        auto_fixes.append(
            {
                "id": "open_synthetic_data",
                "label": "Generate Synthetic Data",
                "description": "Open synthetic data tools to expand training examples.",
                "navigate_to": f"/project/{project_id}/pipeline/synthetic",
            }
        )

    if invalid_json_rows > 0:
        warnings.append(
            f"Detected {invalid_json_rows} non-JSONL rows in prepared train split; parsing quality may be reduced."
        )

    ready = len(blockers) == 0
    return {
        "ready": ready,
        "prepared_train_exists": train_path.exists(),
        "prepared_train_path": str(train_path),
        "prepared_row_count": row_count,
        "valid_json_row_count": valid_json_rows,
        "invalid_json_row_count": invalid_json_rows,
        "min_recommended_rows": safe_min_rows,
        "blockers": blockers,
        "warnings": warnings,
        "hints": hints,
        "auto_fixes": auto_fixes,
    }


def build_newbie_autopilot_run_name(
    *,
    intent: str,
    preset_label: str,
) -> str:
    intent_short = re.sub(r"[^a-zA-Z0-9 ]+", " ", str(intent or "")).strip()
    intent_short = re.sub(r"\s+", " ", intent_short)
    if len(intent_short) > 40:
        intent_short = intent_short[:40].rstrip()
    base = str(preset_label or "autopilot").strip()
    if intent_short:
        return f"Autopilot - {base} - {intent_short}"
    return f"Autopilot - {base}"


def estimate_newbie_autopilot_run(
    *,
    plan_profile: str,
    target_profile_id: str,
    dataset_size_rows: int = 1000,
    task_profile: str | None = None,
    run_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate run time/cost with telemetry calibration and deterministic fallback."""
    normalized_profile = _normalize_plan_profile(plan_profile, default="balanced") or "balanced"
    normalized_target = _normalize_target_profile(target_profile_id)
    normalized_task = _normalize_task_profile(task_profile)
    safe_rows = max(1, int(dataset_size_rows))

    heuristic_seconds = _heuristic_seconds_estimate(
        plan_profile=normalized_profile,
        target_profile_id=normalized_target,
        dataset_size_rows=safe_rows,
    )

    normalized_history: list[dict[str, Any]] = []
    for item in list(run_history or []):
        if not isinstance(item, dict):
            continue
        duration_seconds = _coerce_positive_float(item.get("duration_seconds"))
        if duration_seconds is None:
            continue
        row_dataset_rows = _coerce_positive_int(item.get("dataset_size_rows"))
        seconds_per_1k_rows = None
        if row_dataset_rows is not None:
            seconds_per_1k_rows = round(duration_seconds * (1000.0 / float(row_dataset_rows)), 4)
        normalized_history.append(
            {
                "duration_seconds": duration_seconds,
                "dataset_size_rows": row_dataset_rows,
                "seconds_per_1k_rows": seconds_per_1k_rows,
                "plan_profile": _normalize_plan_profile(item.get("plan_profile"), default=None),
                "target_profile_id": _normalize_target_profile(item.get("target_profile_id"))
                if str(item.get("target_profile_id") or "").strip()
                else None,
                "task_profile": _normalize_task_profile(item.get("task_profile")),
            }
        )

    cohort, cohort_rows = _cohort_rows(
        rows=normalized_history,
        target_profile_id=normalized_target,
        plan_profile=normalized_profile,
        task_profile=normalized_task,
    )
    seconds_per_1k_samples = [
        float(row.get("seconds_per_1k_rows"))
        for row in cohort_rows
        if _coerce_positive_float(row.get("seconds_per_1k_rows")) is not None
    ]
    sample_count = len(seconds_per_1k_samples)

    metric_source = "estimated"
    calibration_note = "Sparse telemetry for this context; using heuristic estimate."
    estimated_seconds = int(heuristic_seconds)
    p50_seconds_per_1k: float | None = None
    variability_cv: float | None = None

    if sample_count >= 2:
        p50_seconds_per_1k = float(median(seconds_per_1k_samples))
        estimated_seconds = int(round(p50_seconds_per_1k * (safe_rows / 1000.0)))
        metric_source = "measured"
        baseline = mean(seconds_per_1k_samples)
        if baseline > 0:
            variability_cv = float(pstdev(seconds_per_1k_samples) / baseline)
        calibration_note = (
            f"Calibrated from {sample_count} historical run(s) "
            f"matched by {cohort} telemetry cohort."
        )
    elif sample_count == 1:
        p50_seconds_per_1k = float(seconds_per_1k_samples[0])
        telemetry_seconds = float(p50_seconds_per_1k * (safe_rows / 1000.0))
        estimated_seconds = int(round((0.6 * telemetry_seconds) + (0.4 * heuristic_seconds)))
        calibration_note = (
            "Only one comparable historical run was found; estimate blends telemetry with heuristic fallback."
        )

    estimated_seconds = max(30, estimated_seconds)
    cohort_weight = float(_COHORT_CONFIDENCE_WEIGHT.get(cohort, _COHORT_CONFIDENCE_WEIGHT["none"]))
    if sample_count >= 2:
        sample_score = min(1.0, sample_count / 8.0)
        variance_score = 0.6 if variability_cv is None else max(0.1, min(1.0, 1.0 - variability_cv))
        confidence_score = 0.25 + (0.4 * sample_score) + (0.25 * cohort_weight) + (0.1 * variance_score)
    elif sample_count == 1:
        confidence_score = 0.46 + (0.08 * cohort_weight)
    else:
        confidence_score = 0.33
    confidence_score = round(max(0.05, min(confidence_score, 0.99)), 4)
    confidence_band = _confidence_band(confidence_score)

    hourly_rate_usd, pricing_source, pricing_meta = _estimate_hourly_cost_usd(normalized_target)
    estimated_cost = round((estimated_seconds / 3600.0) * hourly_rate_usd, 4)
    if estimated_cost < 0:
        estimated_cost = 0.0
    rounded_cost = round(estimated_cost, 2)

    if hourly_rate_usd <= 0:
        calibration_note = f"{calibration_note} Cost assumes local/non-billed execution."
    elif pricing_source == "cloud_burst_catalog_avg":
        calibration_note = (
            f"{calibration_note} Cost uses cloud provider pricing averages "
            f"({pricing_meta.get('gpu_sku')}, {pricing_meta.get('provider_count')} providers)."
        )
    else:
        calibration_note = f"{calibration_note} Cost uses {pricing_source.replace('_', ' ')}."

    speed_label = "Fast" if estimated_seconds <= 600 else ("Medium" if estimated_seconds <= 1800 else "Slow")
    quality_label = "High" if normalized_profile == "max_quality" else ("Standard" if normalized_profile == "balanced" else "Good")
    cost_label = "Low" if rounded_cost < 1.0 else ("Medium" if rounded_cost < 5.0 else "High")

    return {
        "estimated_seconds": estimated_seconds,
        "estimated_cost": rounded_cost,
        "unit": "USD",
        "confidence_score": confidence_score,
        "confidence_band": confidence_band,
        "metric_source": metric_source,
        "source": metric_source,
        "provenance": metric_source,
        "note": calibration_note,
        "labels": {
            "speed": speed_label,
            "quality": quality_label,
            "cost": cost_label,
        },
        "calibration": {
            "cohort": cohort,
            "sample_count": sample_count,
            "seconds_per_1k_rows_p50": p50_seconds_per_1k,
            "seconds_per_1k_rows_cv": round(variability_cv, 4) if variability_cv is not None else None,
            "heuristic_seconds": int(heuristic_seconds),
            "pricing_source": pricing_source,
            "hourly_rate_usd": round(hourly_rate_usd, 4),
            "target_profile_id": normalized_target,
            "task_profile": normalized_task,
            "plan_profile": normalized_profile,
            "fallback_used": metric_source != "measured",
        },
    }


def resolve_newbie_autopilot_intent(
    *,
    intent: str,
    target_profile_id: str = "vllm_server",
    primary_language: str = "english",
    available_vram_gb: float | None = None,
) -> dict[str, Any]:
    normalized_intent = _normalize_intent(intent)
    if not normalized_intent:
        raise ValueError("intent is required")

    selected = dict(_DEFAULT_PRESET)
    selected_score = 0
    selected_keywords: list[str] = []
    for preset in _INTENT_PRESETS:
        score, matched = _score_preset(normalized_intent, preset)
        if score > selected_score:
            selected = dict(preset)
            selected_score = score
            selected_keywords = matched

    normalized_target = _normalize_target_profile(target_profile_id)
    target_defaults = _safe_defaults_for_target(normalized_target)
    safe_training_config = {
        "base_model": "microsoft/phi-2",
        "training_mode": "sft",
        "training_runtime_id": "auto",
        "trainer_backend": "auto",
        "task_type": str(selected.get("task_type") or "causal_lm"),
        "chat_template": str(selected.get("chat_template") or "llama3"),
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "save_steps": 100,
        "eval_steps": 100,
        "sequence_packing": True,
        "gradient_checkpointing": True,
        "auto_oom_retry": True,
        "max_oom_retries": 2,
        "oom_retry_seq_shrink": 0.75,
        "training_plan_profile": "safe",
        **target_defaults,
    }

    return {
        "intent": normalized_intent,
        "target_profile_id": normalized_target,
        "primary_language": str(primary_language or "english").strip().lower() or "english",
        "available_vram_gb": available_vram_gb,
        "preset_id": str(selected.get("preset_id") or _DEFAULT_PRESET["preset_id"]),
        "preset_label": str(selected.get("label") or _DEFAULT_PRESET["label"]),
        "preset_description": str(selected.get("description") or _DEFAULT_PRESET["description"]),
        "task_profile": str(selected.get("task_profile") or "instruction_sft"),
        "matched_keywords": selected_keywords,
        "confidence": min(1.0, 0.35 + (0.15 * selected_score)),
        "safe_training_config": safe_training_config,
        "run_name_suggestion": build_newbie_autopilot_run_name(
            intent=normalized_intent,
            preset_label=str(selected.get("label") or _DEFAULT_PRESET["label"]),
        ),
        "user_friendly_plan": [
            "We auto-picked a safe starter recipe for your goal.",
            "We keep memory settings conservative to reduce training failures.",
            "You can launch now with one click and refine later in Advanced Mode.",
        ],
    }

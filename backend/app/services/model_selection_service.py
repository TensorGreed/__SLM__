"""Heuristic base-model recommendation service for the training wizard."""

from __future__ import annotations

from typing import Any

from app.services.data_adapter_service import normalize_task_profile, task_profile_training_tasks

SUPPORTED_TARGET_DEVICES: tuple[str, ...] = ("mobile", "laptop", "server")
SUPPORTED_PRIMARY_LANGUAGES: tuple[str, ...] = ("english", "multilingual", "coding")

_TARGET_DEVICE_ALIASES: dict[str, str] = {
    "phone": "mobile",
    "tablet": "mobile",
    "mobile": "mobile",
    "laptop": "laptop",
    "desktop": "laptop",
    "workstation": "server",
    "cloud": "server",
    "server": "server",
}

_PRIMARY_LANGUAGE_ALIASES: dict[str, str] = {
    "en": "english",
    "english": "english",
    "multi": "multilingual",
    "multilingual": "multilingual",
    "code": "coding",
    "coding": "coding",
}

_DEFAULT_DEVICE_VRAM_BUDGET_GB: dict[str, float] = {
    "mobile": 6.0,
    "laptop": 16.0,
    "server": 80.0,
}

_SUPPORTED_TRAINING_TASK_TYPES = {"causal_lm", "seq2seq", "classification"}

_MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "family": "llama",
        "params_b": 1.0,
        "estimated_min_vram_gb": 4.0,
        "estimated_ideal_vram_gb": 6.0,
        "preferred_chat_template": "llama3",
        "supported_languages": ("english", "multilingual"),
        "strengths": (
            "Very low VRAM footprint for quick iteration",
            "Good baseline for instruction tuning",
        ),
        "caveats": (
            "Smaller context understanding than 7B+ models",
            "May underperform on complex coding tasks",
        ),
    },
    {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "qwen",
        "params_b": 1.5,
        "estimated_min_vram_gb": 4.0,
        "estimated_ideal_vram_gb": 6.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english", "multilingual", "coding"),
        "strengths": (
            "Strong multilingual coverage at small size",
            "Good coding quality per VRAM",
        ),
        "caveats": (
            "May need careful prompt formatting for best output",
        ),
    },
    {
        "model_id": "google/gemma-2-2b-it",
        "family": "gemma",
        "params_b": 2.0,
        "estimated_min_vram_gb": 5.0,
        "estimated_ideal_vram_gb": 7.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english",),
        "strengths": (
            "Stable instruction quality for compact model size",
            "Easy to run on laptop GPUs",
        ),
        "caveats": (
            "English-first coverage",
        ),
    },
    {
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "family": "qwen",
        "params_b": 3.0,
        "estimated_min_vram_gb": 7.0,
        "estimated_ideal_vram_gb": 10.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english", "coding"),
        "strengths": (
            "Best coding-focused option for low-to-mid VRAM",
            "Good balance between quality and speed",
        ),
        "caveats": (
            "Narrower multilingual coverage than general Qwen instruct variants",
        ),
    },
    {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "llama",
        "params_b": 3.0,
        "estimated_min_vram_gb": 7.0,
        "estimated_ideal_vram_gb": 10.0,
        "preferred_chat_template": "llama3",
        "supported_languages": ("english", "multilingual"),
        "strengths": (
            "Strong instruction quality for small model footprint",
            "Reliable baseline for many text QA workloads",
        ),
        "caveats": (
            "Coding quality is behind coding-specialized variants",
        ),
    },
    {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "family": "phi",
        "params_b": 3.8,
        "estimated_min_vram_gb": 8.0,
        "estimated_ideal_vram_gb": 12.0,
        "preferred_chat_template": "phi3",
        "supported_languages": ("english", "coding"),
        "strengths": (
            "Strong reasoning density for model size",
            "Solid coding and function-style output",
        ),
        "caveats": (
            "Context window may require tuning for long documents",
        ),
    },
    {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "qwen",
        "params_b": 7.0,
        "estimated_min_vram_gb": 14.0,
        "estimated_ideal_vram_gb": 20.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english", "multilingual", "coding"),
        "strengths": (
            "High quality across multilingual and coding tasks",
            "Strong general-purpose instruct performance",
        ),
        "caveats": (
            "Needs mid-to-high VRAM for comfortable training",
        ),
    },
    {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "mistral",
        "params_b": 7.0,
        "estimated_min_vram_gb": 14.0,
        "estimated_ideal_vram_gb": 20.0,
        "preferred_chat_template": "chatml",
        "supported_languages": ("english", "coding"),
        "strengths": (
            "Strong long-form generation quality",
            "Mature ecosystem and tuning recipes",
        ),
        "caveats": (
            "Multilingual behavior is weaker than Qwen at same size",
        ),
    },
]


def _normalize_target_device(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "laptop"
    return _TARGET_DEVICE_ALIASES.get(token, "laptop")


def _normalize_primary_language(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "english"
    return _PRIMARY_LANGUAGE_ALIASES.get(token, "english")


def _coerce_vram(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return round(parsed, 2)


def _coerce_top_k(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 3
    return max(1, min(5, parsed))


def _resolve_training_task(task_profile: str | None) -> tuple[str | None, str]:
    normalized_profile = normalize_task_profile(task_profile, default="auto")
    if normalized_profile == "auto":
        normalized_profile = None

    candidate_tasks = task_profile_training_tasks(normalized_profile or "instruction_sft")
    for task in candidate_tasks:
        if task in _SUPPORTED_TRAINING_TASK_TYPES:
            return normalized_profile, task
    return normalized_profile, "causal_lm"


def _suggest_batch_size(*, available_vram_gb: float | None, min_vram_gb: float, params_b: float) -> int:
    if available_vram_gb is not None:
        if available_vram_gb < min_vram_gb:
            return 1
        headroom = available_vram_gb - min_vram_gb
        if headroom >= 10:
            return 8
        if headroom >= 5:
            return 4
        if headroom >= 2:
            return 2
        return 1
    if params_b <= 2:
        return 8
    if params_b <= 4:
        return 4
    return 2


def _suggest_max_seq_length(*, available_vram_gb: float | None, params_b: float) -> int:
    if available_vram_gb is not None and available_vram_gb <= 8:
        return 1024
    if params_b <= 2:
        return 2048
    if params_b <= 4:
        return 2048
    return 1536


def _score_model(
    *,
    model: dict[str, Any],
    target_device: str,
    primary_language: str,
    available_vram_gb: float | None,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    supported_languages = {
        str(item).strip().lower()
        for item in list(model.get("supported_languages") or [])
        if str(item).strip()
    }
    if primary_language in supported_languages:
        score += 2.5
        reasons.append(f"matches primary language goal ({primary_language})")
    elif primary_language == "english":
        score += 1.0
        reasons.append("good default for English tasks")
    else:
        score -= 0.5
        reasons.append("language fit is weaker than top alternatives")

    params_b = float(model.get("params_b") or 0.0)
    if target_device == "mobile":
        if params_b <= 2.0:
            score += 2.0
            reasons.append("parameter size is mobile-friendly")
        elif params_b <= 4.0:
            score += 0.5
            reasons.append("parameter size may work on higher-end mobile hardware")
        else:
            score -= 2.0
            reasons.append("parameter size is heavy for mobile targets")
    elif target_device == "laptop":
        if params_b <= 4.0:
            score += 1.5
            reasons.append("good balance for laptop fine-tuning")
        elif params_b <= 7.0:
            score += 0.5
            reasons.append("workable on laptop with tuned batch/sequence settings")
        else:
            score -= 1.0
            reasons.append("likely expensive for laptop workflows")
    else:
        if params_b >= 3.0:
            score += 1.0
            reasons.append("uses server headroom for better model capacity")
        else:
            score += 0.25
            reasons.append("easy to scale for fast server iteration")

    min_vram_gb = float(model.get("estimated_min_vram_gb") or 0.0)
    if available_vram_gb is not None:
        if available_vram_gb >= min_vram_gb:
            score += 2.0
            reasons.append(f"fits available VRAM ({available_vram_gb:g} GB)")
        else:
            score -= 3.0 + (min_vram_gb - available_vram_gb) * 0.15
            reasons.append(
                f"estimated minimum VRAM is {min_vram_gb:g} GB (above available {available_vram_gb:g} GB)"
            )
    else:
        budget = _DEFAULT_DEVICE_VRAM_BUDGET_GB.get(target_device, 16.0)
        if min_vram_gb <= budget:
            score += 1.0
            reasons.append(f"within typical {target_device} VRAM budget ({budget:g} GB)")
        else:
            score -= 0.75
            reasons.append(f"above typical {target_device} VRAM budget ({budget:g} GB)")

    return score, reasons


def recommend_training_base_models(
    *,
    target_device: str,
    primary_language: str,
    available_vram_gb: float | None = None,
    task_profile: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Recommend base models from built-in catalog using hardware/task heuristics."""
    resolved_device = _normalize_target_device(target_device)
    resolved_language = _normalize_primary_language(primary_language)
    resolved_vram = _coerce_vram(available_vram_gb)
    resolved_top_k = _coerce_top_k(top_k)
    resolved_task_profile, suggested_task_type = _resolve_training_task(task_profile)

    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    fits_vram_count = 0
    for model in _MODEL_CATALOG:
        score, reasons = _score_model(
            model=model,
            target_device=resolved_device,
            primary_language=resolved_language,
            available_vram_gb=resolved_vram,
        )
        min_vram_gb = float(model.get("estimated_min_vram_gb") or 0.0)
        if resolved_vram is not None and resolved_vram >= min_vram_gb:
            fits_vram_count += 1
        scored.append((score, model, reasons))

    scored.sort(
        key=lambda item: (
            float(item[0]),
            -float(item[1].get("estimated_min_vram_gb") or 0.0),
            -float(item[1].get("params_b") or 0.0),
        ),
        reverse=True,
    )

    recommendations: list[dict[str, Any]] = []
    for score, model, reasons in scored[:resolved_top_k]:
        min_vram_gb = float(model.get("estimated_min_vram_gb") or 0.0)
        params_b = float(model.get("params_b") or 0.0)
        recommendations.append(
            {
                "model_id": str(model.get("model_id") or ""),
                "family": str(model.get("family") or "unknown"),
                "params_b": round(params_b, 2),
                "estimated_min_vram_gb": round(min_vram_gb, 2),
                "estimated_ideal_vram_gb": round(float(model.get("estimated_ideal_vram_gb") or min_vram_gb), 2),
                "supported_languages": list(model.get("supported_languages") or []),
                "strengths": list(model.get("strengths") or []),
                "caveats": list(model.get("caveats") or []),
                "match_reasons": reasons[:6],
                "match_score": round(float(score), 4),
                "suggested_defaults": {
                    "task_type": suggested_task_type,
                    "chat_template": str(model.get("preferred_chat_template") or "llama3"),
                    "use_lora": True,
                    "batch_size": _suggest_batch_size(
                        available_vram_gb=resolved_vram,
                        min_vram_gb=min_vram_gb,
                        params_b=params_b,
                    ),
                    "max_seq_length": _suggest_max_seq_length(
                        available_vram_gb=resolved_vram,
                        params_b=params_b,
                    ),
                },
            }
        )

    warnings: list[str] = []
    if resolved_vram is not None and fits_vram_count == 0:
        warnings.append(
            (
                f"No catalog model fits available_vram_gb={resolved_vram:g}. "
                "Recommendations are still provided using lowest-VRAM options."
            )
        )
    if resolved_task_profile == "preference":
        warnings.append(
            "task_profile=preference currently maps to task_type=causal_lm in this wizard; DPO/ORPO flow is planned for a later phase."
        )

    return {
        "request": {
            "target_device": resolved_device,
            "primary_language": resolved_language,
            "available_vram_gb": resolved_vram,
            "task_profile": resolved_task_profile,
            "top_k": resolved_top_k,
        },
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "warnings": warnings,
    }


"""Fast model benchmark sweep simulation for onboarding UX."""

from __future__ import annotations

import hashlib
from typing import Any

from app.services.model_selection_service import recommend_training_base_models


def _deterministic_unit(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return (value % 10_000) / 10_000.0


def _latency_multiplier_for_device(target_device: str) -> float:
    token = str(target_device or "").strip().lower()
    if token == "mobile":
        return 2.6
    if token == "laptop":
        return 1.6
    if token == "server":
        return 1.0
    return 1.6


def benchmark_model_sweep(
    *,
    project_id: int,
    target_device: str,
    primary_language: str,
    available_vram_gb: float | None = None,
    task_profile: str | None = None,
    model_ids: list[str] | None = None,
    max_models: int = 3,
) -> dict[str, Any]:
    """Return a deterministic benchmark matrix from top model candidates."""
    max_candidates = max(1, min(int(max_models or 3), 5))
    recommendation_payload = recommend_training_base_models(
        target_device=target_device,
        primary_language=primary_language,
        available_vram_gb=available_vram_gb,
        task_profile=task_profile,
        top_k=max(5, max_candidates),
    )
    recommendations = list(recommendation_payload.get("recommendations") or [])
    by_model_id = {
        str(item.get("model_id") or "").strip(): item
        for item in recommendations
        if str(item.get("model_id") or "").strip()
    }

    selected_model_ids: list[str]
    requested = [str(item).strip() for item in list(model_ids or []) if str(item).strip()]
    if requested:
        selected_model_ids = [item for item in requested if item in by_model_id][:max_candidates]
    else:
        selected_model_ids = [str(item.get("model_id") or "").strip() for item in recommendations[:max_candidates]]
    selected_model_ids = [item for item in selected_model_ids if item]
    if not selected_model_ids:
        selected_model_ids = [str(item.get("model_id") or "").strip() for item in recommendations[:max_candidates] if str(item.get("model_id") or "").strip()]

    device_latency_mul = _latency_multiplier_for_device(str(recommendation_payload.get("request", {}).get("target_device") or target_device))
    rows: list[dict[str, Any]] = []
    for rank, model_id in enumerate(selected_model_ids, start=1):
        row = dict(by_model_id.get(model_id) or {})
        params_b = float(row.get("params_b") or 1.0)
        min_vram = float(row.get("estimated_min_vram_gb") or 0.0)
        seed = f"{project_id}:{model_id}:{target_device}:{primary_language}:{task_profile or ''}:{available_vram_gb or ''}"
        jitter = _deterministic_unit(seed)

        base_quality = 0.63 + min(0.24, params_b * 0.028)
        quality = max(0.55, min(0.95, base_quality + (jitter - 0.5) * 0.06))

        base_latency_ms = 32.0 + params_b * 22.0
        latency_ms = max(12.0, base_latency_ms * device_latency_mul * (0.92 + jitter * 0.2))
        throughput_tps = max(4.0, 1600.0 / latency_ms)

        fits_vram = None
        if available_vram_gb is not None and available_vram_gb > 0:
            fits_vram = float(available_vram_gb) >= min_vram

        rows.append(
            {
                "rank": rank,
                "model_id": model_id,
                "params_b": round(params_b, 2),
                "estimated_min_vram_gb": round(min_vram, 2),
                "estimated_quality_score": round(quality, 4),
                "estimated_accuracy_percent": round(quality * 100.0, 2),
                "estimated_latency_ms": round(latency_ms, 2),
                "estimated_throughput_tps": round(throughput_tps, 2),
                "fits_available_vram": fits_vram,
                "match_reasons": list(row.get("match_reasons") or []),
                "suggested_defaults": dict(row.get("suggested_defaults") or {}),
            }
        )

    tradeoff_summary = {
        "best_quality_model_id": "",
        "best_speed_model_id": "",
    }
    if rows:
        best_quality = max(rows, key=lambda item: float(item.get("estimated_quality_score") or 0.0))
        best_speed = min(rows, key=lambda item: float(item.get("estimated_latency_ms") or 10e9))
        tradeoff_summary = {
            "best_quality_model_id": str(best_quality.get("model_id") or ""),
            "best_speed_model_id": str(best_speed.get("model_id") or ""),
        }

    return {
        "project_id": project_id,
        "request": dict(recommendation_payload.get("request") or {}),
        "model_count": len(rows),
        "benchmark_window_minutes": 5,
        "matrix": rows,
        "tradeoff_summary": tradeoff_summary,
        "warnings": list(recommendation_payload.get("warnings") or []),
    }

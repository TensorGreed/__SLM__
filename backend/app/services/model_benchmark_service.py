"""Real sampled benchmark sweep for model-selection onboarding."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings
from app.services.model_selection_service import recommend_training_base_models

DEFAULT_BENCHMARK_SAMPLE_SIZE = 96
MAX_BENCHMARK_SAMPLE_SIZE = 500
MAX_SAMPLE_TEXT_CHARS = 3500
MAX_SAMPLE_SCAN_ROWS = 5000
MAX_CANDIDATE_MODELS = 5
BENCHMARK_WINDOW_MINUTES = 5

_DEVICE_LATENCY_MULTIPLIER: dict[str, float] = {
    "mobile": 2.6,
    "laptop": 1.6,
    "server": 1.0,
}

_FALLBACK_SAMPLE_TEXTS: tuple[str, ...] = (
    "Summarize this document in three concise bullet points.",
    "Extract key entities and provide a structured JSON response.",
    "Answer the user question using only the provided context.",
    "Classify this input into one of the allowed labels.",
    "Rewrite the response in a professional tone with clear formatting.",
    "Generate a short explanation with one concrete example.",
)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return parsed


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_model_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        token = _coerce_text(item)
        if token and token not in out:
            out.append(token)
    return out


def _project_train_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"


def _estimate_token_count(text: str) -> int:
    token = str(text or "").strip()
    if not token:
        return 0
    rough = re.findall(r"\w+|[^\w\s]", token, flags=re.UNICODE)
    if rough:
        return max(1, len(rough))
    return max(1, int(math.ceil(len(token) / 4.2)))


def _messages_as_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = _coerce_text(item.get("role") or item.get("from"))
        content = _coerce_text(item.get("content") or item.get("value") or item.get("text"))
        if not content:
            continue
        if role:
            parts.append(f"{role}: {content}")
        else:
            parts.append(content)
    return "\n".join(parts).strip()


def _row_text(record: dict[str, Any]) -> str:
    direct_fields = (
        "text",
        "content",
        "body",
        "document",
        "passage",
    )
    for key in direct_fields:
        token = _coerce_text(record.get(key))
        if token:
            return token

    pair_candidates = (
        ("prompt", "response"),
        ("question", "answer"),
        ("instruction", "output"),
        ("input", "output"),
        ("source_text", "target_text"),
    )
    for left, right in pair_candidates:
        left_text = _coerce_text(record.get(left))
        right_text = _coerce_text(record.get(right))
        if left_text and right_text:
            return f"{left_text}\n{right_text}"

    messages = _messages_as_text(record.get("messages"))
    if not messages:
        messages = _messages_as_text(record.get("conversations"))
    if messages:
        return messages

    fallback_parts: list[str] = []
    for key in ("title", "summary", "label", "category"):
        value = _coerce_text(record.get(key))
        if value:
            fallback_parts.append(value)
    return "\n".join(fallback_parts).strip()


def _sample_project_dataset(
    project_id: int,
    *,
    sample_size: int,
) -> dict[str, Any]:
    train_file = _project_train_path(project_id)
    texts: list[str] = []
    token_counts: list[int] = []
    scanned_rows = 0

    if train_file.exists():
        with open(train_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if scanned_rows >= MAX_SAMPLE_SCAN_ROWS or len(texts) >= sample_size:
                    break
                scanned_rows += 1
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                text = _row_text(payload)
                if not text:
                    continue
                clipped = text[:MAX_SAMPLE_TEXT_CHARS]
                texts.append(clipped)
                token_counts.append(_estimate_token_count(clipped))

    sample_source = "prepared_train"
    if not texts:
        sample_source = "fallback_seed"
        seeded_size = max(6, min(sample_size, 24))
        while len(texts) < seeded_size:
            template = _FALLBACK_SAMPLE_TEXTS[len(texts) % len(_FALLBACK_SAMPLE_TEXTS)]
            texts.append(template)
            token_counts.append(_estimate_token_count(template))

    sampled_row_count = len(texts)
    sampled_total_tokens = int(sum(token_counts))
    sampled_avg_tokens = (
        round(float(sampled_total_tokens) / float(sampled_row_count), 2)
        if sampled_row_count > 0
        else 0.0
    )
    return {
        "sample_source": sample_source,
        "sampled_row_count": sampled_row_count,
        "sampled_total_tokens": sampled_total_tokens,
        "sampled_avg_tokens": sampled_avg_tokens,
        "token_counts": token_counts,
        "texts": texts,
        "train_file": str(train_file),
        "train_file_exists": bool(train_file.exists()),
        "rows_scanned": scanned_rows,
    }


def _load_tokenizer(
    model_id: str,
    *,
    allow_network: bool,
) -> tuple[Any | None, str]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover - depends on runtime env
        return None, f"transformers unavailable: {e.__class__.__name__}"

    attempts: list[tuple[str, bool]] = [("local_cache", True)]
    if allow_network:
        attempts.append(("hf_download", False))

    last_error = "tokenizer_load_failed"
    for source, local_only in attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                trust_remote_code=False,
                local_files_only=local_only,
            )
            return tokenizer, source
        except Exception as e:  # pragma: no cover - network/cache dependent
            last_error = f"{source}: {e.__class__.__name__}"
    return None, last_error


def _tokenizer_metrics(tokenizer: Any, texts: list[str]) -> dict[str, float] | None:
    if tokenizer is None or not texts:
        return None
    started = time.perf_counter()
    token_count_values: list[int] = []
    try:
        for text in texts:
            encoded = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=4096,
            )
            token_count_values.append(max(1, len(list(encoded))))
    except Exception:
        return None
    elapsed = max(0.0001, time.perf_counter() - started)
    total_tokens = int(sum(token_count_values))
    avg_tokens = (float(total_tokens) / float(len(token_count_values))) if token_count_values else 0.0
    throughput = float(total_tokens) / elapsed if total_tokens > 0 else 0.0
    return {
        "avg_tokens": round(avg_tokens, 2),
        "total_tokens": float(total_tokens),
        "throughput_tps": round(throughput, 2),
    }


def _latency_multiplier_for_device(target_device: str) -> float:
    token = str(target_device or "").strip().lower()
    return _DEVICE_LATENCY_MULTIPLIER.get(token, 1.6)


def _estimate_quality_score(
    *,
    params_b: float,
    match_score: float,
    avg_tokens: float,
    context_length: int | None,
    fits_vram: bool | None,
    tokenizer_used: bool,
) -> float:
    quality = 0.57 + min(0.29, params_b * 0.031)
    quality += max(-0.04, min(0.09, (match_score - 4.0) * 0.015))
    if context_length is not None and context_length > 0 and avg_tokens > 0:
        coverage = min(1.2, float(context_length) / max(512.0, float(avg_tokens) * 2.2))
        quality += (coverage - 0.6) * 0.08
    if fits_vram is True:
        quality += 0.025
    elif fits_vram is False:
        quality -= 0.055
    if tokenizer_used:
        quality += 0.01
    return round(max(0.5, min(0.97, quality)), 4)


def _estimate_latency_ms(
    *,
    params_b: float,
    avg_tokens: float,
    target_device: str,
    tokenizer_tps: float | None,
) -> float:
    base_ms = (24.0 + params_b * 19.0) * _latency_multiplier_for_device(target_device)
    workload_factor = 0.9 + min(1.8, float(avg_tokens) / 720.0)
    latency_ms = base_ms * workload_factor
    if tokenizer_tps is not None and tokenizer_tps > 0:
        tokenization_ms = (float(avg_tokens) / float(tokenizer_tps)) * 1000.0
        latency_ms += tokenization_ms * 3.5
    return round(max(10.0, latency_ms), 2)


def _estimate_throughput_tps(
    *,
    latency_ms: float,
    params_b: float,
    target_device: str,
) -> float:
    device_bias = 1.0
    if target_device == "mobile":
        device_bias = 0.7
    elif target_device == "server":
        device_bias = 1.25
    size_bias = 1.0 + max(0.0, (3.0 - params_b) * 0.12)
    throughput = (1100.0 / max(1.0, latency_ms)) * device_bias * size_bias
    return round(max(3.0, throughput), 2)


def _append_unique_warning(warnings: list[str], message: str) -> None:
    token = str(message or "").strip()
    if token and token not in warnings:
        warnings.append(token)


def benchmark_model_sweep(
    *,
    project_id: int,
    target_device: str,
    primary_language: str,
    available_vram_gb: float | None = None,
    task_profile: str | None = None,
    model_ids: list[str] | None = None,
    max_models: int = 3,
    sample_size: int = DEFAULT_BENCHMARK_SAMPLE_SIZE,
    allow_network_tokenizer: bool = False,
) -> dict[str, Any]:
    """Run a sampled benchmark matrix across top model candidates."""
    max_candidates = max(1, min(_coerce_int(max_models, 3), MAX_CANDIDATE_MODELS))
    resolved_sample_size = max(10, min(_coerce_int(sample_size, DEFAULT_BENCHMARK_SAMPLE_SIZE), MAX_BENCHMARK_SAMPLE_SIZE))

    recommendation_payload = recommend_training_base_models(
        target_device=target_device,
        primary_language=primary_language,
        available_vram_gb=available_vram_gb,
        task_profile=task_profile,
        top_k=max(3, max_candidates),
    )
    recommendations = list(recommendation_payload.get("recommendations") or [])
    by_model_id = {
        _coerce_text(item.get("model_id")): dict(item)
        for item in recommendations
        if _coerce_text(item.get("model_id"))
    }

    requested = _normalize_model_ids(model_ids)
    selected_model_ids: list[str] = []
    if requested:
        selected_model_ids = [item for item in requested if item in by_model_id][:max_candidates]
    if not selected_model_ids:
        selected_model_ids = [
            _coerce_text(item.get("model_id"))
            for item in recommendations[:max_candidates]
            if _coerce_text(item.get("model_id"))
        ]

    warnings: list[str] = [str(item) for item in list(recommendation_payload.get("warnings") or []) if str(item).strip()]
    if requested and len(selected_model_ids) < len(requested):
        _append_unique_warning(
            warnings,
            "Some requested model_ids were not available in current recommendations and were skipped.",
        )

    sample = _sample_project_dataset(project_id, sample_size=resolved_sample_size)
    sample_texts = list(sample.get("texts") or [])
    sample_avg_tokens = _coerce_float(sample.get("sampled_avg_tokens"), 0.0)
    sample_total_tokens = int(_coerce_int(sample.get("sampled_total_tokens"), 0))
    sample_source = _coerce_text(sample.get("sample_source")) or "fallback_seed"

    if sample_source != "prepared_train":
        _append_unique_warning(
            warnings,
            "Prepared train split not found or unreadable; benchmark used fallback prompts.",
        )

    tokenizer_used_count = 0
    tokenizer_failure_count = 0
    rows: list[dict[str, Any]] = []
    for model_id in selected_model_ids:
        base_row = dict(by_model_id.get(model_id) or {})
        params_b = _coerce_float(base_row.get("params_b"), 1.0)
        min_vram = _coerce_float(base_row.get("estimated_min_vram_gb"), 0.0)
        match_score = _coerce_float(base_row.get("match_score"), 0.0)
        context_length_raw = base_row.get("context_length")
        context_length = None
        try:
            parsed_context = int(context_length_raw)
            if parsed_context > 0:
                context_length = parsed_context
        except (TypeError, ValueError):
            context_length = None

        fits_vram: bool | None = None
        if available_vram_gb is not None and available_vram_gb > 0:
            fits_vram = float(available_vram_gb) >= min_vram

        row_avg_tokens = sample_avg_tokens
        row_total_tokens = sample_total_tokens
        row_tokenizer_tps: float | None = None
        row_mode = "sampled_heuristic"

        tokenizer, tokenizer_note = _load_tokenizer(
            model_id,
            allow_network=bool(allow_network_tokenizer),
        )
        if tokenizer is not None:
            metrics = _tokenizer_metrics(tokenizer, sample_texts)
            if metrics is not None:
                tokenizer_used_count += 1
                row_mode = "real_tokenizer"
                row_avg_tokens = _coerce_float(metrics.get("avg_tokens"), sample_avg_tokens)
                row_total_tokens = int(_coerce_float(metrics.get("total_tokens"), sample_total_tokens))
                row_tokenizer_tps = _coerce_float(metrics.get("throughput_tps"), 0.0) or None
            else:
                tokenizer_failure_count += 1
                _append_unique_warning(
                    warnings,
                    f"Tokenizer timing failed for '{model_id}' (using heuristic fallback).",
                )
        else:
            tokenizer_failure_count += 1
            if "transformers unavailable" in tokenizer_note:
                _append_unique_warning(
                    warnings,
                    "transformers is unavailable; tokenizer timing skipped and heuristic estimates were used.",
                )

        quality = _estimate_quality_score(
            params_b=params_b,
            match_score=match_score,
            avg_tokens=row_avg_tokens,
            context_length=context_length,
            fits_vram=fits_vram,
            tokenizer_used=row_mode == "real_tokenizer",
        )
        latency_ms = _estimate_latency_ms(
            params_b=params_b,
            avg_tokens=row_avg_tokens,
            target_device=str(recommendation_payload.get("request", {}).get("target_device") or target_device),
            tokenizer_tps=row_tokenizer_tps,
        )
        throughput_tps = _estimate_throughput_tps(
            latency_ms=latency_ms,
            params_b=params_b,
            target_device=str(recommendation_payload.get("request", {}).get("target_device") or target_device),
        )
        speed_score = min(1.0, 140.0 / max(1.0, latency_ms))
        overall_score = round(
            (quality * 0.72)
            + (speed_score * 0.28)
            + (0.05 if fits_vram is True else (-0.04 if fits_vram is False else 0.0)),
            4,
        )

        rows.append(
            {
                "rank": 0,
                "model_id": model_id,
                "params_b": round(params_b, 2),
                "estimated_min_vram_gb": round(min_vram, 2),
                "estimated_quality_score": quality,
                "estimated_accuracy_percent": round(quality * 100.0, 2),
                "estimated_latency_ms": latency_ms,
                "estimated_throughput_tps": throughput_tps,
                "fits_available_vram": fits_vram,
                "match_reasons": list(base_row.get("match_reasons") or []),
                "suggested_defaults": dict(base_row.get("suggested_defaults") or {}),
                "sampled_avg_tokens": round(row_avg_tokens, 2),
                "sampled_total_tokens": int(row_total_tokens),
                "tokenizer_throughput_tps": round(float(row_tokenizer_tps), 2) if row_tokenizer_tps else None,
                "overall_score": overall_score,
                "benchmark_mode": row_mode,
            }
        )

    rows.sort(
        key=lambda item: (
            float(item.get("overall_score") or 0.0),
            float(item.get("estimated_quality_score") or 0.0),
            float(item.get("estimated_throughput_tps") or 0.0),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    benchmark_mode = "real_sampled_heuristic"
    if rows and tokenizer_used_count == len(rows):
        benchmark_mode = "real_sampled_tokenizer"
    elif tokenizer_used_count > 0:
        benchmark_mode = "real_sampled_tokenizer_mixed"
    if rows and tokenizer_failure_count == len(rows):
        _append_unique_warning(
            warnings,
            "Tokenizer timing unavailable for selected models; used sampled heuristic estimates.",
        )

    tradeoff_summary = {
        "best_quality_model_id": "",
        "best_speed_model_id": "",
        "best_balance_model_id": "",
    }
    if rows:
        best_quality = max(rows, key=lambda item: float(item.get("estimated_quality_score") or 0.0))
        best_speed = min(rows, key=lambda item: float(item.get("estimated_latency_ms") or 10e9))
        tradeoff_summary = {
            "best_quality_model_id": _coerce_text(best_quality.get("model_id")),
            "best_speed_model_id": _coerce_text(best_speed.get("model_id")),
            "best_balance_model_id": _coerce_text(rows[0].get("model_id")),
        }

    request_payload = dict(recommendation_payload.get("request") or {})
    request_payload["model_ids"] = selected_model_ids
    request_payload["max_models"] = max_candidates
    request_payload["sample_size"] = resolved_sample_size
    request_payload["allow_network_tokenizer"] = bool(allow_network_tokenizer)

    return {
        "project_id": project_id,
        "run_id": uuid4().hex[:12],
        "benchmark_mode": benchmark_mode,
        "request": request_payload,
        "model_count": len(rows),
        "benchmark_window_minutes": BENCHMARK_WINDOW_MINUTES,
        "sample_source": sample_source,
        "sampled_row_count": int(sample.get("sampled_row_count") or 0),
        "sampled_avg_tokens": round(sample_avg_tokens, 2),
        "sampled_total_tokens": int(sample_total_tokens),
        "matrix": rows,
        "tradeoff_summary": tradeoff_summary,
        "warnings": warnings[:20],
    }

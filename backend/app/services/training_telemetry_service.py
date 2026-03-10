"""Training wizard telemetry persistence helpers."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings


EVENT_TYPE_MODEL_WIZARD = "training.model_selection_wizard"
MAX_TOP_MODELS = 20
MAX_TOP_CONTEXTS = 12
MIN_ADAPTIVE_APPLY_EVENTS = 3
MIN_CONTEXT_APPLY_EVENTS = 2
MAX_ADAPTIVE_MODEL_BOOST = 1.25

_TARGET_DEVICE_ALIASES: dict[str, str] = {
    "mobile": "mobile",
    "phone": "mobile",
    "tablet": "mobile",
    "laptop": "laptop",
    "desktop": "laptop",
    "workstation": "server",
    "cloud": "server",
    "server": "server",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_action(value: Any) -> str:
    token = _coerce_text(value).lower()
    if token == "apply":
        return "apply"
    return "recommend"


def _normalize_model_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in value:
        token = _coerce_text(raw)
        if token:
            out.append(token)
    return out[:50]


def _normalize_target_device(value: Any) -> str | None:
    token = _coerce_text(value).lower()
    if not token:
        return None
    return _TARGET_DEVICE_ALIASES.get(token)


def _normalize_task_profile(value: Any) -> str | None:
    token = _coerce_text(value).lower()
    if not token or token == "auto":
        return None
    return token


def _event_matches_context(
    event: dict[str, Any],
    *,
    target_device: str | None,
    task_profile: str | None,
) -> bool:
    if target_device:
        event_device = _normalize_target_device(event.get("target_device"))
        if event_device != target_device:
            return False
    if task_profile:
        event_task = _normalize_task_profile(event.get("task_profile"))
        if event_task != task_profile:
            return False
    return True


def _telemetry_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "telemetry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_wizard_path(project_id: int) -> Path:
    return _telemetry_dir(project_id) / "training_model_wizard_events.jsonl"


def _iter_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token:
                continue
            try:
                payload = json.loads(token)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            events.append(payload)
    return events


def summarize_model_wizard_events(project_id: int) -> dict[str, Any]:
    path = _model_wizard_path(project_id)
    events = _iter_events(path)

    recommend_events = 0
    apply_events = 0
    auto_recommend_events = 0
    manual_recommend_events = 0
    last_timestamp: str | None = None
    selected_models = Counter()
    context_apply_events = Counter()
    context_model_counts: dict[tuple[str, str], Counter[str]] = {}

    for event in events:
        action = _normalize_action(event.get("action"))
        if action == "apply":
            apply_events += 1
            model_id = _coerce_text(event.get("selected_model_id"))
            if model_id:
                selected_models[model_id] += 1
                device = _normalize_target_device(event.get("target_device")) or "any"
                profile = _normalize_task_profile(event.get("task_profile")) or "any"
                context_key = (device, profile)
                context_apply_events[context_key] += 1
                context_counter = context_model_counts.get(context_key)
                if context_counter is None:
                    context_counter = Counter()
                    context_model_counts[context_key] = context_counter
                context_counter[model_id] += 1
        else:
            recommend_events += 1
            if bool(event.get("auto_run")):
                auto_recommend_events += 1
            else:
                manual_recommend_events += 1

        timestamp = _coerce_text(event.get("timestamp"))
        if timestamp:
            last_timestamp = timestamp

    top_selected_models = [
        {"model_id": model_id, "count": count}
        for model_id, count in selected_models.most_common(MAX_TOP_MODELS)
    ]
    apply_conversion_rate = (
        round((apply_events / recommend_events), 4)
        if recommend_events > 0
        else None
    )
    top_selected_models_by_context: list[dict[str, Any]] = []
    for (device, profile), count in context_apply_events.most_common(MAX_TOP_CONTEXTS):
        top_models = [
            {"model_id": model_id, "count": model_count}
            for model_id, model_count in context_model_counts.get((device, profile), Counter()).most_common(5)
        ]
        top_selected_models_by_context.append(
            {
                "target_device": None if device == "any" else device,
                "task_profile": None if profile == "any" else profile,
                "apply_events": int(count),
                "top_selected_models": top_models,
            }
        )
    return {
        "project_id": int(project_id),
        "event_count": len(events),
        "recommend_events": recommend_events,
        "apply_events": apply_events,
        "auto_recommend_events": auto_recommend_events,
        "manual_recommend_events": manual_recommend_events,
        "apply_conversion_rate": apply_conversion_rate,
        "top_selected_models": top_selected_models,
        "top_selected_models_by_context": top_selected_models_by_context,
        "last_event_at": last_timestamp,
        "path": str(path),
    }


def build_model_acceptance_bias(
    project_id: int,
    *,
    target_device: str | None = None,
    task_profile: str | None = None,
) -> dict[str, Any]:
    """Build adaptive score boosts from historical wizard apply events."""
    resolved_device = _normalize_target_device(target_device)
    resolved_task_profile = _normalize_task_profile(task_profile)
    context_label_parts: list[str] = []
    if resolved_device:
        context_label_parts.append(f"device={resolved_device}")
    if resolved_task_profile:
        context_label_parts.append(f"task={resolved_task_profile}")
    context_label = ", ".join(context_label_parts) if context_label_parts else "global"

    path = _model_wizard_path(project_id)
    events = _iter_events(path)
    global_apply_counts: Counter[str] = Counter()
    context_apply_counts: Counter[str] = Counter()

    for event in events:
        if _normalize_action(event.get("action")) != "apply":
            continue
        model_id = _coerce_text(event.get("selected_model_id"))
        if not model_id:
            continue
        global_apply_counts[model_id] += 1
        if _event_matches_context(
            event,
            target_device=resolved_device,
            task_profile=resolved_task_profile,
        ):
            context_apply_counts[model_id] += 1

    global_apply_total = int(sum(global_apply_counts.values()))
    context_apply_total = int(sum(context_apply_counts.values()))
    if global_apply_total < MIN_ADAPTIVE_APPLY_EVENTS:
        return {
            "enabled": False,
            "context_label": context_label,
            "target_device": resolved_device,
            "task_profile": resolved_task_profile,
            "global_apply_events": global_apply_total,
            "context_apply_events": context_apply_total,
            "bias_by_model": {},
            "path": str(path),
        }

    context_scale = (
        min(1.0, context_apply_total / 6.0)
        if context_apply_total >= MIN_CONTEXT_APPLY_EVENTS
        else 0.0
    )
    global_scale = min(1.0, global_apply_total / 12.0)
    bias_by_model: dict[str, float] = {}
    for model_id in sorted(set(global_apply_counts.keys()) | set(context_apply_counts.keys())):
        global_share = float(global_apply_counts[model_id]) / float(global_apply_total)
        context_share = (
            float(context_apply_counts[model_id]) / float(context_apply_total)
            if context_apply_total > 0
            else 0.0
        )
        bias = (context_share * 0.9 * context_scale) + (global_share * 0.45 * global_scale)
        bias = min(MAX_ADAPTIVE_MODEL_BOOST, max(0.0, bias))
        if bias >= 0.05:
            bias_by_model[model_id] = round(bias, 4)

    return {
        "enabled": bool(bias_by_model),
        "context_label": context_label,
        "target_device": resolved_device,
        "task_profile": resolved_task_profile,
        "global_apply_events": global_apply_total,
        "context_apply_events": context_apply_total,
        "bias_by_model": bias_by_model,
        "path": str(path),
    }


def record_model_wizard_event(
    project_id: int,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = dict(payload or {})

    action = _normalize_action(data.get("action"))
    source = _coerce_text(data.get("source")) or "training_setup_wizard"
    recommendation_count = _safe_int(data.get("recommendation_count"))
    selected_rank = _safe_int(data.get("selected_rank"))
    available_vram_gb = _safe_float(data.get("available_vram_gb"))
    selected_score = _safe_float(data.get("selected_score"))
    top_k = _safe_int(data.get("top_k"))

    if recommendation_count is not None:
        recommendation_count = max(0, min(recommendation_count, 20))
    if selected_rank is not None:
        selected_rank = max(1, min(selected_rank, 50))
    if top_k is not None:
        top_k = max(1, min(top_k, 5))

    event = {
        "event_id": uuid4().hex,
        "event_type": EVENT_TYPE_MODEL_WIZARD,
        "timestamp": _utcnow_iso(),
        "project_id": int(project_id),
        "source": source,
        "action": action,
        "auto_run": bool(data.get("auto_run")) if action == "recommend" else None,
        "target_device": _coerce_text(data.get("target_device")) or None,
        "primary_language": _coerce_text(data.get("primary_language")) or None,
        "available_vram_gb": available_vram_gb,
        "task_profile": _coerce_text(data.get("task_profile")) or None,
        "top_k": top_k,
        "recommendation_count": recommendation_count,
        "recommendation_model_ids": _normalize_model_ids(data.get("recommendation_model_ids")),
        "selected_model_id": _coerce_text(data.get("selected_model_id")) or None,
        "selected_rank": selected_rank,
        "selected_score": selected_score,
    }

    path = _model_wizard_path(project_id)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    return {
        "event": event,
        "summary": summarize_model_wizard_events(project_id),
        "path": str(path),
    }

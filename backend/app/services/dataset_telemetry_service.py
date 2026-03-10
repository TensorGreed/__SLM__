"""Dataset adapter telemetry persistence helpers."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings


EVENT_TYPE_MAPPING_ACCEPTANCE = "dataset.mapping_acceptance"
MAX_TOP_FIELDS = 20


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_mode(value: Any) -> str:
    token = _coerce_text(value).lower()
    if token == "batch":
        return "batch"
    return "single"


def _normalize_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _coerce_text(raw_key)
        mapped = _coerce_text(raw_value)
        if not key or not mapped:
            continue
        out[key] = mapped
    return out


def _normalize_suggestion_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        token = _coerce_text(item)
        if token:
            out.append(token)
    return out[:200]


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return parsed


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def _telemetry_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "telemetry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mapping_acceptance_path(project_id: int) -> Path:
    return _telemetry_dir(project_id) / "dataset_mapping_acceptance.jsonl"


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


def summarize_mapping_acceptance(project_id: int) -> dict[str, Any]:
    path = _mapping_acceptance_path(project_id)
    events = _iter_events(path)
    counter = Counter()
    mode_counter = Counter()
    accepted_total = 0
    last_timestamp: str | None = None

    for event in events:
        fields = event.get("accepted_keys")
        if isinstance(fields, list):
            for field in fields:
                token = _coerce_text(field)
                if token:
                    counter[token] += 1
                    accepted_total += 1
        mode = _normalize_mode(event.get("mode"))
        mode_counter[mode] += 1
        timestamp = _coerce_text(event.get("timestamp"))
        if timestamp:
            last_timestamp = timestamp

    top_fields = [
        {"field": field, "count": count}
        for field, count in counter.most_common(MAX_TOP_FIELDS)
    ]
    return {
        "project_id": int(project_id),
        "event_count": len(events),
        "accepted_mappings_total": accepted_total,
        "single_apply_events": int(mode_counter.get("single", 0)),
        "batch_apply_events": int(mode_counter.get("batch", 0)),
        "top_canonical_fields": top_fields,
        "last_event_at": last_timestamp,
        "path": str(path),
    }


def record_mapping_acceptance(
    project_id: int,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = dict(payload or {})
    mapping = _normalize_mapping(data.get("mapping"))
    mode = _normalize_mode(data.get("mode"))
    suggestion_count = _safe_int(data.get("suggestion_count"))
    confidence_avg = _safe_float(data.get("confidence_avg"))
    accepted_ids = _normalize_suggestion_ids(data.get("accepted_suggestion_ids"))
    source = _coerce_text(data.get("source")) or "adapter_preview"
    adapter_id = _coerce_text(data.get("adapter_id")) or None
    task_profile = _coerce_text(data.get("task_profile")) or None

    event = {
        "event_id": uuid4().hex,
        "event_type": EVENT_TYPE_MAPPING_ACCEPTANCE,
        "timestamp": _utcnow_iso(),
        "project_id": int(project_id),
        "source": source,
        "mode": mode,
        "adapter_id": adapter_id,
        "task_profile": task_profile,
        "mapping": mapping,
        "accepted_keys": sorted(mapping.keys()),
        "accepted_count": len(mapping),
        "suggestion_count": suggestion_count,
        "confidence_avg": confidence_avg,
        "accepted_suggestion_ids": accepted_ids,
    }
    path = _mapping_acceptance_path(project_id)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    return {
        "event": event,
        "summary": summarize_mapping_acceptance(project_id),
        "path": str(path),
    }


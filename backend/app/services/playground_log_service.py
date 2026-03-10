"""Prompt/eval logging helpers for playground interactions."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings


MAX_TAGS = 12
MAX_LOG_LIMIT = 500
EVENT_TYPE_PLAYGROUND_FEEDBACK = "training.playground.feedback"


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
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = _coerce_text(item).lower()
        if not token or token in seen:
            continue
        seen.add(token)
        tags.append(token)
        if len(tags) >= MAX_TAGS:
            break
    return tags


def _normalize_rating(value: Any) -> int | None:
    parsed = _safe_int(value)
    if parsed is None:
        return None
    if parsed > 0:
        return 1
    if parsed < 0:
        return -1
    return 0


def _quality_checks(*, prompt: str, reply: str, rating: int | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    prompt_text = _coerce_text(prompt)
    reply_text = _coerce_text(reply)
    reply_len = len(reply_text)
    if reply_len < 12:
        checks.append(
            {
                "code": "reply_too_short",
                "severity": "warning",
                "message": "Assistant reply is very short.",
            }
        )
    lower_reply = reply_text.lower()
    if any(token in lower_reply for token in ["i don't know", "cannot help", "can't help", "unable to"]):
        checks.append(
            {
                "code": "possible_refusal",
                "severity": "info",
                "message": "Reply may be a refusal/uncertainty response.",
            }
        )
    if ("json" in prompt_text.lower() or "structured" in prompt_text.lower()) and reply_text:
        stripped = reply_text.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            checks.append(
                {
                    "code": "json_shape_mismatch",
                    "severity": "warning",
                    "message": "Prompt requested structured/JSON style output but reply is plain text.",
                }
            )
    if rating is not None and rating < 0:
        checks.append(
            {
                "code": "user_negative_feedback",
                "severity": "warning",
                "message": "User marked this response as negative.",
            }
        )
    return checks


def _telemetry_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "telemetry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _logs_path(project_id: int) -> Path:
    return _telemetry_dir(project_id) / "playground_feedback_logs.jsonl"


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
            if isinstance(payload, dict):
                events.append(payload)
    return events


def record_playground_feedback(
    project_id: int,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = dict(payload or {})
    prompt = _coerce_text(data.get("prompt"))
    reply = _coerce_text(data.get("reply"))
    rating = _normalize_rating(data.get("rating"))
    tags = _normalize_tags(data.get("tags"))
    session_id = _safe_int(data.get("session_id"))
    message_index = _safe_int(data.get("message_index"))

    event = {
        "event_id": uuid4().hex,
        "event_type": EVENT_TYPE_PLAYGROUND_FEEDBACK,
        "timestamp": _utcnow_iso(),
        "project_id": int(project_id),
        "session_id": session_id,
        "message_index": message_index,
        "provider": _coerce_text(data.get("provider")) or "unknown",
        "model_name": _coerce_text(data.get("model_name")) or None,
        "preset_id": _coerce_text(data.get("preset_id")) or None,
        "prompt": prompt,
        "reply": reply,
        "rating": rating,
        "tags": tags,
        "notes": _coerce_text(data.get("notes")),
        "quality_checks": _quality_checks(prompt=prompt, reply=reply, rating=rating),
    }

    path = _logs_path(project_id)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    return {
        "event": event,
        "summary": summarize_playground_feedback(project_id),
        "path": str(path),
    }


def list_playground_feedback(
    project_id: int,
    *,
    limit: int = 50,
) -> dict[str, Any]:
    path = _logs_path(project_id)
    events = _iter_events(path)
    capped_limit = max(1, min(int(limit or 50), MAX_LOG_LIMIT))
    selected = list(reversed(events[-capped_limit:]))
    return {
        "project_id": int(project_id),
        "count": len(selected),
        "events": selected,
        "path": str(path),
    }


def summarize_playground_feedback(project_id: int) -> dict[str, Any]:
    path = _logs_path(project_id)
    events = _iter_events(path)
    rating_counter = Counter()
    tag_counter = Counter()
    quality_counter = Counter()
    last_event_at: str | None = None
    for event in events:
        rating = _normalize_rating(event.get("rating"))
        if rating is not None:
            rating_counter[str(rating)] += 1
        tags = _normalize_tags(event.get("tags"))
        for tag in tags:
            tag_counter[tag] += 1
        quality_checks = event.get("quality_checks")
        if isinstance(quality_checks, list):
            for item in quality_checks:
                if not isinstance(item, dict):
                    continue
                code = _coerce_text(item.get("code"))
                if code:
                    quality_counter[code] += 1
        timestamp = _coerce_text(event.get("timestamp"))
        if timestamp:
            last_event_at = timestamp

    top_tags = [{"tag": tag, "count": count} for tag, count in tag_counter.most_common(10)]
    top_quality_issues = [
        {"code": code, "count": count}
        for code, count in quality_counter.most_common(12)
    ]
    return {
        "project_id": int(project_id),
        "event_count": len(events),
        "positive_count": int(rating_counter.get("1", 0)),
        "negative_count": int(rating_counter.get("-1", 0)),
        "neutral_count": int(rating_counter.get("0", 0)),
        "top_tags": top_tags,
        "top_quality_issues": top_quality_issues,
        "last_event_at": last_event_at,
        "path": str(path),
    }

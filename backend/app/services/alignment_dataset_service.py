"""Preference dataset import/filter helpers for alignment training workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.alignment_service import (
    canonicalize_preference_row,
    score_preference_rows,
    validate_preference_rows,
)


def _utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _project_dir(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id)


def _prepared_dir(project_id: int) -> Path:
    return _project_dir(project_id) / "prepared"


def _alignment_dir(project_id: int) -> Path:
    path = _prepared_dir(project_id) / "alignment"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_train_file(project_id: int) -> Path:
    return _prepared_dir(project_id) / "train.jsonl"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _resolve_project_path(project_id: int, candidate_path: str) -> Path:
    token = str(candidate_path or "").strip()
    if not token:
        raise ValueError("Path is required.")

    project_root = _project_dir(project_id).resolve()
    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        candidate.relative_to(project_root)
    except ValueError as e:
        raise ValueError("Path must stay within the project data directory.") from e
    return candidate


def _load_jsonl_rows(path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    limit = max_rows if isinstance(max_rows, int) and max_rows > 0 else None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_rows_input(
    *,
    rows: list[dict[str, Any]] | None = None,
    raw_text: str | None = None,
) -> list[dict[str, Any]]:
    parsed_rows: list[dict[str, Any]] = [dict(item) for item in list(rows or []) if isinstance(item, dict)]

    text = str(raw_text or "").strip()
    if not text:
        return parsed_rows

    try:
        decoded = json.loads(text)
        if isinstance(decoded, list):
            parsed_rows.extend(dict(item) for item in decoded if isinstance(item, dict))
            return parsed_rows
        if isinstance(decoded, dict):
            parsed_rows.append(decoded)
            return parsed_rows
    except Exception:
        pass

    for line_no, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            decoded = json.loads(raw)
        except Exception as e:
            raise ValueError(f"Invalid JSONL row at line {line_no}: {e}") from e
        if isinstance(decoded, dict):
            parsed_rows.append(decoded)
    return parsed_rows


def _canonicalize_valid_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        canonical = canonicalize_preference_row(row)
        prompt = _coerce_text(canonical.get("prompt"))
        chosen = _coerce_text(canonical.get("chosen"))
        rejected = _coerce_text(canonical.get("rejected"))
        missing: list[str] = []
        if not prompt:
            missing.append("prompt")
        if not chosen:
            missing.append("chosen")
        if not rejected:
            missing.append("rejected")
        if missing:
            invalid_rows.append(
                {
                    "index": index,
                    "reason": f"Missing required fields: {', '.join(missing)}.",
                    "missing_fields": missing,
                }
            )
            continue
        if chosen.strip() == rejected.strip():
            invalid_rows.append(
                {
                    "index": index,
                    "reason": "chosen and rejected responses are identical.",
                    "missing_fields": [],
                }
            )
            continue
        canonical_row = dict(canonical)
        canonical_row["prompt"] = prompt
        canonical_row["chosen"] = chosen
        canonical_row["rejected"] = rejected
        canonical_rows.append(canonical_row)
    return canonical_rows, invalid_rows


def summarize_preference_dataset(
    project_id: int,
    *,
    source_path: str | None = None,
    sample_size: int = 400,
    quality_threshold: float = 3.0,
) -> dict[str, Any]:
    if source_path:
        resolved_path = _resolve_project_path(project_id, source_path)
    else:
        resolved_path = _project_train_file(project_id)

    rows = _load_jsonl_rows(
        resolved_path,
        max_rows=max(20, min(int(sample_size or 400), 5000)),
    )
    contract = validate_preference_rows(rows, min_coverage=0.0, max_rows=max(1, len(rows) or 1))
    quality = score_preference_rows(
        rows,
        quality_threshold=quality_threshold,
        max_rows=max(1, len(rows) or 1),
    )
    keep_count = int(quality.get("keep_count", 0))
    scored_count = int(quality.get("scored_count", 0))
    keep_ratio = float(keep_count / scored_count) if scored_count > 0 else 0.0

    return {
        "project_id": project_id,
        "source_path": str(resolved_path),
        "exists": resolved_path.exists(),
        "sample_size": len(rows),
        "contract": contract,
        "quality": {
            "quality_threshold": float(quality.get("quality_threshold", quality_threshold)),
            "scored_count": scored_count,
            "keep_count": keep_count,
            "drop_count": int(quality.get("drop_count", 0)),
            "keep_ratio": round(keep_ratio, 6),
            "average_quality_score": quality.get("average_quality_score"),
            "score_distribution": quality.get("score_distribution"),
        },
        "rows_preview": contract.get("canonical_rows_preview", []),
    }


def import_preference_dataset_rows(
    project_id: int,
    *,
    rows: list[dict[str, Any]] | None = None,
    raw_text: str | None = None,
    mode: str = "replace",
    target: str = "prepared_train",
) -> dict[str, Any]:
    parsed_rows = _parse_rows_input(rows=rows, raw_text=raw_text)
    if not parsed_rows:
        raise ValueError("No preference rows were provided.")

    normalized_mode = str(mode or "replace").strip().lower()
    if normalized_mode not in {"replace", "append"}:
        raise ValueError("mode must be 'replace' or 'append'.")

    normalized_target = str(target or "prepared_train").strip().lower()
    if normalized_target not in {"prepared_train", "alignment_workspace"}:
        raise ValueError("target must be 'prepared_train' or 'alignment_workspace'.")

    contract = validate_preference_rows(parsed_rows, min_coverage=0.0, max_rows=max(1, len(parsed_rows)))
    canonical_rows, invalid_rows = _canonicalize_valid_rows(parsed_rows)
    if not canonical_rows:
        raise ValueError("No valid prompt/chosen/rejected rows were found.")

    if normalized_target == "prepared_train":
        target_path = _project_train_file(project_id)
    else:
        target_path = _alignment_dir(project_id) / "preference_workspace.jsonl"

    existing_rows: list[dict[str, str]] = []
    if normalized_mode == "append" and target_path.exists():
        existing_raw = _load_jsonl_rows(target_path)
        existing_rows, _ = _canonicalize_valid_rows(existing_raw)

    output_rows = [*existing_rows, *canonical_rows]
    _write_jsonl_rows(target_path, output_rows)

    quality = score_preference_rows(
        output_rows,
        quality_threshold=3.0,
        max_rows=max(1, len(output_rows)),
    )
    keep_count = int(quality.get("keep_count", 0))
    keep_ratio = float(keep_count / len(output_rows)) if output_rows else 0.0

    return {
        "project_id": project_id,
        "target_path": str(target_path),
        "mode": normalized_mode,
        "target": normalized_target,
        "rows_received": len(parsed_rows),
        "rows_written": len(output_rows),
        "rows_added": len(canonical_rows),
        "rows_dropped": len(invalid_rows),
        "contract": contract,
        "quality": {
            "quality_threshold": float(quality.get("quality_threshold", 3.0)),
            "scored_count": int(quality.get("scored_count", 0)),
            "keep_count": keep_count,
            "drop_count": int(quality.get("drop_count", 0)),
            "keep_ratio": round(keep_ratio, 6),
            "average_quality_score": quality.get("average_quality_score"),
        },
        "invalid_samples": invalid_rows[:25],
    }


def resolve_alignment_dataset_path(project_id: int, candidate_path: str | None) -> Path | None:
    token = str(candidate_path or "").strip()
    if not token:
        return None
    return _resolve_project_path(project_id, token)


def _active_learning_dir(project_id: int) -> Path:
    path = _alignment_dir(project_id) / "active_learning"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _playground_rejected_path(project_id: int) -> Path:
    return _active_learning_dir(project_id) / "playground_rejected.jsonl"


def _playground_auto_pairs_path(project_id: int) -> Path:
    return _active_learning_dir(project_id) / "playground_auto_pairs.jsonl"


def _playground_merged_train_path(project_id: int) -> Path:
    return _active_learning_dir(project_id) / "train.with_playground_feedback.jsonl"


def _playground_feedback_log_path(project_id: int) -> Path:
    return _project_dir(project_id) / "telemetry" / "playground_feedback_logs.jsonl"


def _normalize_prompt_key(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _collect_metadata_tokens(value: Any) -> list[str]:
    tokens: list[str] = []
    if value is None:
        return tokens
    if isinstance(value, (list, tuple, set)):
        for item in value:
            tokens.extend(_collect_metadata_tokens(item))
        return tokens
    token = _coerce_text(value)
    if token:
        tokens.append(token)
    return tokens


def _dedupe_tokens(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _coerce_text(value)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def _merge_tags(existing: Any, incoming: Any) -> list[str]:
    tags: list[str] = []
    for candidate in [existing, incoming]:
        if isinstance(candidate, list):
            tags.extend(_collect_metadata_tokens(candidate))
    return _dedupe_tokens(tags)


def _is_empty_metadata_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _merge_alignment_row_provenance(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key in incoming:
        if key in {"prompt", "chosen", "rejected"}:
            continue
        if key not in target or _is_empty_metadata_value(target.get(key)):
            target[key] = incoming.get(key)
        elif key == "tags":
            merged_tags = _merge_tags(target.get("tags"), incoming.get("tags"))
            if merged_tags:
                target["tags"] = merged_tags

    source_tokens = _dedupe_tokens(
        _collect_metadata_tokens(target.get("provenance_sources"))
        + _collect_metadata_tokens(target.get("source"))
        + _collect_metadata_tokens(incoming.get("provenance_sources"))
        + _collect_metadata_tokens(incoming.get("source"))
    )
    if source_tokens:
        target["provenance_sources"] = source_tokens
        if not _coerce_text(target.get("source")):
            target["source"] = source_tokens[0]

    event_tokens = _dedupe_tokens(
        _collect_metadata_tokens(target.get("provenance_event_ids"))
        + _collect_metadata_tokens(target.get("event_id"))
        + _collect_metadata_tokens(incoming.get("provenance_event_ids"))
        + _collect_metadata_tokens(incoming.get("event_id"))
    )
    if event_tokens:
        target["provenance_event_ids"] = event_tokens
        if not _coerce_text(target.get("event_id")):
            target["event_id"] = event_tokens[0]

    session_tokens = _dedupe_tokens(
        _collect_metadata_tokens(target.get("provenance_session_ids"))
        + _collect_metadata_tokens(target.get("original_session_id"))
        + _collect_metadata_tokens(target.get("session_id"))
        + _collect_metadata_tokens(incoming.get("provenance_session_ids"))
        + _collect_metadata_tokens(incoming.get("original_session_id"))
        + _collect_metadata_tokens(incoming.get("session_id"))
    )
    if session_tokens:
        target["provenance_session_ids"] = session_tokens
        if not _coerce_text(target.get("original_session_id")):
            target["original_session_id"] = session_tokens[0]

    timestamp_tokens = _dedupe_tokens(
        _collect_metadata_tokens(target.get("provenance_timestamps"))
        + _collect_metadata_tokens(target.get("original_timestamp"))
        + _collect_metadata_tokens(target.get("timestamp"))
        + _collect_metadata_tokens(incoming.get("provenance_timestamps"))
        + _collect_metadata_tokens(incoming.get("original_timestamp"))
        + _collect_metadata_tokens(incoming.get("timestamp"))
    )
    if timestamp_tokens:
        target["provenance_timestamps"] = timestamp_tokens
        if not _coerce_text(target.get("original_timestamp")):
            target["original_timestamp"] = timestamp_tokens[0]


def capture_playground_feedback_event(
    project_id: int,
    *,
    event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(event or {})
    rating = payload.get("rating")
    try:
        rating_value = int(rating)
    except (TypeError, ValueError):
        rating_value = None
    
    preferred_reply = _coerce_text(payload.get("preferred_reply") or payload.get("chosen"))
    
    # Capture if it's a downvote OR if there's an edited response
    should_capture = (rating_value is not None and rating_value < 0) or bool(preferred_reply)
    
    if not should_capture:
        return {
            "project_id": int(project_id),
            "captured": False,
            "reason": "no_downvote_or_edit",
            "rejected_path": str(_playground_rejected_path(project_id)),
        }

    prompt = _coerce_text(payload.get("prompt"))
    rejected = _coerce_text(payload.get("reply") or payload.get("rejected"))
    if not prompt or not rejected:
        return {
            "project_id": int(project_id),
            "captured": False,
            "reason": "missing_prompt_or_reply",
            "rejected_path": str(_playground_rejected_path(project_id)),
        }

    event_id = _coerce_text(payload.get("event_id"))
    timestamp = _coerce_text(payload.get("timestamp")) or datetime.now(timezone.utc).isoformat()
    rejected_path = _playground_rejected_path(project_id)
    existing_rows = _load_jsonl_rows(rejected_path, max_rows=200000)
    if event_id and any(_coerce_text(item.get("event_id")) == event_id for item in existing_rows):
        summary = summarize_playground_active_learning(project_id)
        return {
            "project_id": int(project_id),
            "captured": False,
            "reason": "duplicate_event_id",
            "event_id": event_id,
            "rejected_path": str(rejected_path),
            "summary": summary,
        }

    row = {
        "event_id": event_id or None,
        "timestamp": timestamp,
        "prompt": prompt,
        "rejected": rejected,
        "preferred_reply": preferred_reply or None,
        "provider": _coerce_text(payload.get("provider")) or None,
        "model_name": _coerce_text(payload.get("model_name")) or None,
        "preset_id": _coerce_text(payload.get("preset_id")) or None,
        "session_id": payload.get("session_id"),
        "message_index": payload.get("message_index"),
        "tags": list(payload.get("tags") or []) if isinstance(payload.get("tags"), list) else [],
        "notes": _coerce_text(payload.get("notes")) or None,
        "source": "playground_feedback"
    }
    _append_jsonl_row(rejected_path, row)
    summary = summarize_playground_active_learning(project_id)
    return {
        "project_id": int(project_id),
        "captured": True,
        "event_id": event_id or None,
        "rejected_path": str(rejected_path),
        "summary": summary,
    }


def materialize_playground_preference_pairs(
    project_id: int,
    *,
    max_pairs: int = 5000,
) -> dict[str, Any]:
    rejected_path = _playground_rejected_path(project_id)
    auto_pairs_path = _playground_auto_pairs_path(project_id)
    rejected_rows = _load_jsonl_rows(rejected_path, max_rows=200000)

    positive_reply_by_prompt: dict[str, str] = {}
    feedback_events = _load_jsonl_rows(_playground_feedback_log_path(project_id), max_rows=200000)
    for event in feedback_events:
        try:
            rating = int(event.get("rating"))
        except (TypeError, ValueError):
            rating = None
        if rating is None or rating <= 0:
            continue
        prompt = _coerce_text(event.get("prompt"))
        reply = _coerce_text(event.get("reply"))
        key = _normalize_prompt_key(prompt)
        if not key or not reply:
            continue
        positive_reply_by_prompt[key] = reply

    unique_pairs: set[tuple[str, str, str]] = set()
    output_pairs: list[dict[str, Any]] = []
    pairs_limit = max(1, min(int(max_pairs or 5000), 50000))
    skipped_missing_chosen = 0
    for row in rejected_rows:
        prompt = _coerce_text(row.get("prompt"))
        rejected = _coerce_text(row.get("rejected") or row.get("reply"))
        if not prompt or not rejected:
            continue
        chosen = _coerce_text(row.get("preferred_reply"))
        if not chosen:
            chosen = positive_reply_by_prompt.get(_normalize_prompt_key(prompt), "")
        if not chosen:
            skipped_missing_chosen += 1
            continue
        if chosen.strip() == rejected.strip():
            continue
        key = (prompt, chosen, rejected)
        if key in unique_pairs:
            continue
        unique_pairs.add(key)
        
        pair = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "playground_feedback",
            "event_id": row.get("event_id"),
            "original_session_id": row.get("session_id"),
            "original_timestamp": row.get("timestamp"),
        }
        output_pairs.append(pair)
        if len(output_pairs) >= pairs_limit:
            break

    if output_pairs:
        _write_jsonl_rows(auto_pairs_path, output_pairs)
    elif auto_pairs_path.exists():
        auto_pairs_path.unlink()

    return {
        "project_id": int(project_id),
        "rejected_path": str(rejected_path),
        "auto_pairs_path": str(auto_pairs_path),
        "rejected_count": len(rejected_rows),
        "pair_count": len(output_pairs),
        "skipped_missing_chosen": skipped_missing_chosen,
    }


def summarize_playground_active_learning(project_id: int) -> dict[str, Any]:
    rejected_path = _playground_rejected_path(project_id)
    rejected_rows = _load_jsonl_rows(rejected_path, max_rows=200000)
    negative_events_with_preferred = sum(
        1 for row in rejected_rows if _coerce_text(row.get("preferred_reply"))
    )
    latest_rejected_at: str | None = None
    for row in rejected_rows:
        timestamp = _coerce_text(row.get("timestamp"))
        if timestamp:
            latest_rejected_at = timestamp

    materialized = materialize_playground_preference_pairs(project_id)
    auto_pairs_path = _playground_auto_pairs_path(project_id)
    auto_pairs_preview = _load_jsonl_rows(auto_pairs_path, max_rows=50)
    
    return {
        "project_id": int(project_id),
        "rejected_path": str(rejected_path),
        "auto_pairs_path": str(materialized.get("auto_pairs_path")),
        "rejected_count": len(rejected_rows),
        "auto_pair_count": int(materialized.get("pair_count", 0)),
        "negative_events_with_preferred_reply": int(negative_events_with_preferred),
        "latest_rejected_at": latest_rejected_at,
        "auto_pairs_preview": auto_pairs_preview,
    }


def compose_alignment_training_dataset(
    project_id: int,
    *,
    source_path: str | None = None,
    include_playground_pairs: bool = True,
    target_path: str | None = None,
    max_playground_pairs: int = 5000,
) -> dict[str, Any]:
    if source_path:
        source_file = _resolve_project_path(project_id, source_path)
    else:
        source_file = _project_train_file(project_id)

    source_rows = _load_jsonl_rows(source_file) if source_file.exists() else []
    canonical_source_rows, source_invalid_rows = _canonicalize_valid_rows(source_rows)

    playground_report = {
        "project_id": int(project_id),
        "pair_count": 0,
        "auto_pairs_path": str(_playground_auto_pairs_path(project_id)),
    }
    playground_rows: list[dict[str, Any]] = []
    if include_playground_pairs:
        playground_report = materialize_playground_preference_pairs(
            project_id,
            max_pairs=max_playground_pairs,
        )
        playground_rows, _ = _canonicalize_valid_rows(
            _load_jsonl_rows(Path(str(playground_report.get("auto_pairs_path") or "")))
        )

    if not playground_rows:
        return {
            "project_id": int(project_id),
            "source_path": str(source_file),
            "target_path": str(source_file),
            "effective_train_path": str(source_file),
            "written": False,
            "source_rows": len(canonical_source_rows),
            "source_invalid_rows": len(source_invalid_rows),
            "playground_rows": 0,
            "rows_written": len(canonical_source_rows),
            "playground": playground_report,
        }

    merged_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_candidates = [dict(row) for row in canonical_source_rows]
    for row in source_candidates:
        if not _coerce_text(row.get("source")):
            row["source"] = "prepared_train"
    playground_candidates = [dict(row) for row in playground_rows]
    for row in playground_candidates:
        if not _coerce_text(row.get("source")):
            row["source"] = "playground_feedback"

    for row in [*source_candidates, *playground_candidates]:
        key = (
            _coerce_text(row.get("prompt")),
            _coerce_text(row.get("chosen")),
            _coerce_text(row.get("rejected")),
        )
        if not key[0] or not key[1] or not key[2]:
            continue
        normalized_row = dict(row)
        normalized_row["prompt"] = key[0]
        normalized_row["chosen"] = key[1]
        normalized_row["rejected"] = key[2]
        existing = merged_by_key.get(key)
        if existing is None:
            _merge_alignment_row_provenance(normalized_row, normalized_row)
            merged_by_key[key] = normalized_row
        else:
            _merge_alignment_row_provenance(existing, normalized_row)
    merged_rows = list(merged_by_key.values())

    if target_path:
        output_file = _resolve_project_path(project_id, target_path)
    else:
        output_file = _playground_merged_train_path(project_id)
    _write_jsonl_rows(output_file, merged_rows)
    return {
        "project_id": int(project_id),
        "source_path": str(source_file),
        "target_path": str(output_file),
        "effective_train_path": str(output_file),
        "written": True,
        "source_rows": len(canonical_source_rows),
        "source_invalid_rows": len(source_invalid_rows),
        "playground_rows": len(playground_rows),
        "rows_written": len(merged_rows),
        "playground": playground_report,
    }


def filter_preference_dataset_by_quality(
    project_id: int,
    *,
    quality_threshold: float = 3.0,
    min_keep_ratio: float = 0.4,
    apply_to_train_file: bool = False,
    source_path: str | None = None,
    target_path: str | None = None,
) -> dict[str, Any]:
    if source_path:
        source_file = _resolve_project_path(project_id, source_path)
    else:
        source_file = _project_train_file(project_id)

    if not source_file.exists():
        raise ValueError(f"Source preference dataset not found: {source_file}")

    source_rows = _load_jsonl_rows(source_file)
    canonical_rows, invalid_rows = _canonicalize_valid_rows(source_rows)
    if not canonical_rows:
        raise ValueError("Source dataset has no valid preference rows.")

    quality_cutoff = max(1.0, min(float(quality_threshold), 5.0))
    quality = score_preference_rows(
        canonical_rows,
        quality_threshold=quality_cutoff,
        max_rows=max(1, len(canonical_rows)),
    )
    kept_indices = [int(item) for item in quality.get("kept_row_indices", []) if isinstance(item, int)]
    kept_rows = [canonical_rows[idx] for idx in kept_indices if 0 <= idx < len(canonical_rows)]
    scored_count = int(quality.get("scored_count", 0))
    keep_count = len(kept_rows)
    keep_ratio = float(keep_count / scored_count) if scored_count > 0 else 0.0
    required_keep_ratio = max(0.05, min(float(min_keep_ratio), 1.0))

    if keep_count == 0:
        raise ValueError(
            "Judge filter rejected all preference rows. Lower quality threshold or improve pair quality."
        )
    if keep_ratio < required_keep_ratio:
        raise ValueError(
            (
                f"Judge keep ratio {keep_ratio:.1%} is below required {required_keep_ratio:.0%}. "
                "Lower threshold or improve pair quality before applying."
            )
        )

    backup_path: Path | None = None
    if apply_to_train_file:
        output_file = _project_train_file(project_id)
    elif target_path:
        output_file = _resolve_project_path(project_id, target_path)
    else:
        output_file = _alignment_dir(project_id) / "train.filtered.jsonl"

    if output_file.exists() and output_file.resolve() == _project_train_file(project_id).resolve():
        backup_path = _alignment_dir(project_id) / f"train.backup.{_utc_timestamp_slug()}.jsonl"
        output_file.replace(backup_path)

    _write_jsonl_rows(output_file, kept_rows)

    report_payload = {
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_file),
        "target_path": str(output_file),
        "apply_to_train_file": bool(apply_to_train_file),
        "quality_threshold": quality_cutoff,
        "min_keep_ratio": required_keep_ratio,
        "source_valid_rows": len(canonical_rows),
        "source_invalid_rows": len(invalid_rows),
        "scored_count": scored_count,
        "keep_count": keep_count,
        "drop_count": int(quality.get("drop_count", 0)),
        "keep_ratio": round(keep_ratio, 6),
        "average_quality_score": quality.get("average_quality_score"),
        "score_distribution": quality.get("score_distribution"),
    }
    if backup_path is not None:
        report_payload["backup_path"] = str(backup_path)

    report_path = _alignment_dir(project_id) / "last_filter_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        **report_payload,
        "quality": quality,
        "filter_report_path": str(report_path),
    }

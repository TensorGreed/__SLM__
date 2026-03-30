"""Dataset Structure Explorer + Adapter Studio orchestration services."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.dataset_adapter_definition import DatasetAdapterDefinition
from app.models.project import Project
from app.services.artifact_registry_service import publish_artifact
from app.services.data_adapter_service import (
    AUTO_ADAPTER_ID,
    DEFAULT_ADAPTER_ID,
    list_data_adapter_catalog,
    preview_data_adapter,
    resolve_data_adapter_contract,
)

SUPPORTED_SOURCE_TYPES: set[str] = {
    "project_dataset",
    "csv",
    "tsv",
    "json",
    "jsonl",
    "parquet",
    "huggingface",
    "sql_snapshot",
    "document_corpus",
    "chunk_corpus",
    "chat_transcripts",
    "pairwise_preference",
}

PII_VALUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "api_key": re.compile(r"\b(?:sk|api|key|token)_[A-Za-z0-9]{8,}\b", re.IGNORECASE),
}
PII_FIELD_NAME_HINTS: tuple[str, ...] = (
    "email",
    "phone",
    "mobile",
    "ssn",
    "social_security",
    "credit",
    "card",
    "cvv",
    "password",
    "secret",
    "token",
    "api_key",
    "address",
    "dob",
    "birth",
)
CANONICAL_STUDIO_FIELDS: tuple[str, ...] = (
    "text",
    "question",
    "answer",
    "source_text",
    "target_text",
    "label",
    "context",
    "messages",
    "prompt",
    "chosen",
    "rejected",
    "tool_name",
    "structured_output",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_source_type(value: str) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token not in SUPPORTED_SOURCE_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_SOURCE_TYPES))
        raise ValueError(f"Unsupported source_type '{value}'. Expected one of: {allowed}")
    return token


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _path_from_source_ref(source_ref: str) -> Path:
    token = str(source_ref or "").strip()
    if not token:
        raise ValueError("source_ref/path is required")
    return Path(token).expanduser().resolve()


def _iter_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                rows.append({"value": value})
            if len(rows) >= limit:
                break
    return rows


def _iter_json(path: Path, *, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        # Common dataset export pattern: {"rows": [...]} / {"data": [...]}
        for key in ("rows", "data", "records", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                payload = candidate
                break
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item in payload[:limit]:
            if isinstance(item, dict):
                rows.append(item)
            else:
                rows.append({"value": item})
        return rows
    if isinstance(payload, dict):
        return [payload]
    return [{"value": payload}]


def _iter_delimited(path: Path, *, limit: int, delimiter: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= limit:
                break
    return rows


def _iter_parquet(path: Path, *, limit: int) -> list[dict[str, Any]]:
    # Prefer pyarrow, then pandas. If unavailable, fallback to JSON/CSV parse for local snapshots.
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        rows = table.to_pylist()
        out: list[dict[str, Any]] = []
        for item in rows[:limit]:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"value": item})
        return out
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(path)
        records = frame.to_dict(orient="records")
        out: list[dict[str, Any]] = []
        for item in records[:limit]:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"value": item})
        return out
    except Exception:
        pass

    # Fallback allows tests/fixtures in constrained envs.
    for fallback in (_iter_jsonl, _iter_json):
        try:
            return fallback(path, limit=limit)
        except Exception:
            continue
    try:
        return _iter_delimited(path, limit=limit, delimiter=",")
    except Exception as exc:
        raise ValueError(f"Failed to parse parquet source '{path}': {exc}") from exc


def _load_rows_from_file(path: Path, *, source_type: str, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Source path not found: {path}")
    if not path.is_file():
        raise ValueError(f"Source path is not a file: {path}")

    ext = path.suffix.lower()
    st = _normalize_source_type(source_type)
    if st in {"csv"} or ext == ".csv":
        return _iter_delimited(path, limit=limit, delimiter=",")
    if st in {"tsv"} or ext in {".tsv", ".tab"}:
        return _iter_delimited(path, limit=limit, delimiter="\t")
    if st in {"jsonl"} or ext == ".jsonl":
        return _iter_jsonl(path, limit=limit)
    if st in {"json"} or ext == ".json":
        return _iter_json(path, limit=limit)
    if st in {"parquet"} or ext == ".parquet":
        return _iter_parquet(path, limit=limit)
    if st in {"sql_snapshot", "document_corpus", "chunk_corpus", "chat_transcripts", "pairwise_preference"}:
        # Snapshot corpora may arrive in any standard structured format.
        for loader in (
            lambda p, n: _iter_jsonl(p, limit=n),
            lambda p, n: _iter_json(p, limit=n),
            lambda p, n: _iter_delimited(p, limit=n, delimiter=","),
            lambda p, n: _iter_delimited(p, limit=n, delimiter="\t"),
            lambda p, n: _iter_parquet(p, limit=n),
        ):
            try:
                rows = loader(path, limit)
                if rows:
                    return rows
            except Exception:
                continue
        return []
    # Generic best-effort
    for loader in (
        lambda p, n: _iter_jsonl(p, limit=n),
        lambda p, n: _iter_json(p, limit=n),
        lambda p, n: _iter_delimited(p, limit=n, delimiter=","),
    ):
        try:
            rows = loader(path, limit)
            if rows:
                return rows
        except Exception:
            continue
    return []


async def _ensure_project(db: AsyncSession, project_id: int) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    return project


def _normalize_dataset_type(value: Any) -> DatasetType:
    token = str(value or "").strip().lower()
    if not token:
        return DatasetType.RAW
    for item in DatasetType:
        if item.value == token:
            return item
    raise ValueError(
        f"Unsupported project_dataset dataset_type '{value}'. "
        f"Expected one of: {', '.join(item.value for item in DatasetType)}"
    )


async def _load_project_dataset_rows(
    db: AsyncSession,
    project_id: int,
    *,
    dataset_type: DatasetType,
    limit: int,
    document_id: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if dataset_type == DatasetType.RAW:
        stmt = (
            select(RawDocument)
            .join(Dataset, Dataset.id == RawDocument.dataset_id)
            .where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.RAW,
                RawDocument.status == DocumentStatus.ACCEPTED,
            )
            .order_by(RawDocument.ingested_at.desc())
        )
        if document_id is not None:
            stmt = stmt.where(RawDocument.id == int(document_id))
        result = await db.execute(stmt)
        docs = list(result.scalars().all())
        for doc in docs:
            doc_rows = _load_rows_from_file(Path(str(doc.file_path)), source_type="document_corpus", limit=limit - len(rows))
            rows.extend(doc_rows)
            if len(rows) >= limit:
                break
        return rows[:limit], {
            "source_type": "project_dataset",
            "dataset_type": dataset_type.value,
            "document_id": document_id,
            "documents_scanned": len(docs),
        }

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == dataset_type,
        )
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise ValueError(f"No dataset found for type '{dataset_type.value}' in project {project_id}")
    if not dataset.file_path:
        raise ValueError(f"Dataset '{dataset.name}' has no file_path")
    rows = _load_rows_from_file(Path(str(dataset.file_path)), source_type=dataset_type.value, limit=limit)
    return rows[:limit], {
        "source_type": "project_dataset",
        "dataset_type": dataset_type.value,
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "file_path": dataset.file_path,
    }


def _load_huggingface_rows(source_ref: str, *, split: str, limit: int) -> list[dict[str, Any]]:
    token = str(source_ref or "").strip()
    if not token:
        raise ValueError("source_ref is required for source_type='huggingface'")
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise ValueError(
            "Hugging Face dataset loading requires `datasets` package. "
            "Install it or provide a local snapshot path."
        ) from exc

    dataset = load_dataset(token, split=split)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if isinstance(row, dict):
            rows.append(dict(row))
        else:
            rows.append({"value": row})
        if idx + 1 >= limit:
            break
    return rows


async def load_source_rows(
    db: AsyncSession,
    project_id: int,
    *,
    source: dict[str, Any],
    sample_size: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    await _ensure_project(db, project_id)
    st = _normalize_source_type(str(source.get("source_type") or "project_dataset"))
    limit = _coerce_int(sample_size, default=500, minimum=10, maximum=5000)

    if st == "project_dataset":
        dataset_type = _normalize_dataset_type(source.get("dataset_type"))
        document_id = source.get("document_id")
        return await _load_project_dataset_rows(
            db,
            project_id,
            dataset_type=dataset_type,
            limit=limit,
            document_id=int(document_id) if str(document_id or "").isdigit() else None,
        )

    if st == "huggingface":
        split = str(source.get("split") or "train").strip() or "train"
        source_ref = str(source.get("source_ref") or source.get("hf_dataset") or "").strip()
        rows = _load_huggingface_rows(source_ref, split=split, limit=limit)
        return rows, {
            "source_type": st,
            "source_ref": source_ref,
            "split": split,
        }

    source_ref = str(source.get("source_ref") or source.get("path") or "").strip()
    path = _path_from_source_ref(source_ref)
    rows = _load_rows_from_file(path, source_type=st, limit=limit)
    return rows, {
        "source_type": st,
        "source_ref": str(path),
        "file_ext": path.suffix.lower(),
    }


def _safe_json_text(value: Any, *, max_len: int = 600) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:max_len]
    try:
        token = json.dumps(value, ensure_ascii=True)
    except Exception:
        token = str(value)
    return token[:max_len]


def _infer_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _name_pii_hints(path: str) -> list[str]:
    token = str(path or "").strip().lower()
    hints = [hint for hint in PII_FIELD_NAME_HINTS if hint in token]
    return sorted(set(hints))


def _value_pii_hits(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    token = value.strip()
    if not token:
        return []
    out = [name for name, pattern in PII_VALUE_PATTERNS.items() if pattern.search(token)]
    return sorted(set(out))


def _walk_value(
    value: Any,
    *,
    path: str,
    field_stats: dict[str, dict[str, Any]],
    seen_paths: set[str],
) -> None:
    if path:
        seen_paths.add(path)
        stat = field_stats[path]
        stat["present_records"] += 1
        inferred = _infer_type(value)
        stat["type_counts"][inferred] += 1
        if _is_nullish(value):
            stat["null_values"] += 1
        if inferred == "string":
            text = str(value)
            stat["string_lengths"].append(len(text))
            if len(stat["examples"]) < 5 and text.strip():
                stat["examples"].append(text[:180])
            if len(stat["categorical_values"]) < 200 and text.strip():
                stat["categorical_values"][text.strip()[:120]] += 1
            for pii in _value_pii_hits(text):
                stat["pii_hits"][pii] += 1
        elif inferred in {"integer", "number"}:
            if len(stat["examples"]) < 5:
                stat["examples"].append(str(value))
            if len(stat["categorical_values"]) < 200:
                stat["categorical_values"][str(value)] += 1
        elif inferred == "array":
            stat["sequence_lengths"].append(len(value))
        elif inferred == "object":
            if len(stat["examples"]) < 3:
                stat["examples"].append(_safe_json_text(value, max_len=180))
        else:
            if len(stat["examples"]) < 5:
                stat["examples"].append(_safe_json_text(value, max_len=120))

        for hint in _name_pii_hints(path):
            stat["pii_hits"][f"name:{hint}"] += 1

    if isinstance(value, dict):
        for key, child in value.items():
            child_key = str(key).strip()
            child_path = f"{path}.{child_key}" if path else child_key
            _walk_value(
                child,
                path=child_path,
                field_stats=field_stats,
                seen_paths=seen_paths,
            )
        return
    if isinstance(value, list):
        for child in value:
            child_path = f"{path}[]" if path else "[]"
            _walk_value(
                child,
                path=child_path,
                field_stats=field_stats,
                seen_paths=seen_paths,
            )
        return


def _length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "max": 0}
    ordered = sorted(values)
    p50 = ordered[len(ordered) // 2]
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    return {
        "avg": round(sum(values) / len(values), 2),
        "p50": float(p50),
        "p90": float(p90),
        "max": int(max(values)),
    }


def profile_rows(rows: list[dict[str, Any]], *, source_meta: dict[str, Any]) -> dict[str, Any]:
    total = len(rows)
    field_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "present_records": 0,
            "null_values": 0,
            "type_counts": Counter(),
            "examples": [],
            "categorical_values": Counter(),
            "string_lengths": [],
            "sequence_lengths": [],
            "pii_hits": Counter(),
        }
    )

    for row in rows:
        if not isinstance(row, dict):
            continue
        seen_paths: set[str] = set()
        _walk_value(
            row,
            path="",
            field_stats=field_stats,
            seen_paths=seen_paths,
        )
        # Track missing records for discovered fields.
        for path, stat in field_stats.items():
            if path not in seen_paths:
                stat.setdefault("missing_records", 0)
                stat["missing_records"] += 1

    fields: list[dict[str, Any]] = []
    doc_length_candidates: list[int] = []
    label_distributions: dict[str, dict[str, int]] = {}
    sensitive_columns: list[str] = []
    nested_structures: list[str] = []

    for path in sorted(field_stats.keys()):
        stat = field_stats[path]
        type_counts = dict(stat["type_counts"])
        primary_type = "unknown"
        if type_counts:
            primary_type = sorted(type_counts.items(), key=lambda item: item[1], reverse=True)[0][0]
        present = int(stat["present_records"] or 0)
        missing = int(stat.get("missing_records") or max(0, total - present))
        null_values = int(stat["null_values"] or 0)
        null_rate = round(((null_values + missing) / total), 6) if total > 0 else 0.0
        present_rate = round((present / total), 6) if total > 0 else 0.0

        str_lengths = list(stat["string_lengths"])
        seq_lengths = list(stat["sequence_lengths"])
        length_stats = _length_stats(str_lengths)
        sequence_stats = _length_stats(seq_lengths)
        pii_hits = dict(stat["pii_hits"])
        pii_signals = [key for key, count in pii_hits.items() if int(count) > 0]

        field_payload: dict[str, Any] = {
            "path": path,
            "inferred_type": primary_type,
            "type_counts": type_counts,
            "present_records": present,
            "missing_records": missing,
            "present_rate": present_rate,
            "null_rate": null_rate,
            "examples": list(stat["examples"])[:5],
            "string_length": length_stats,
            "sequence_length": sequence_stats,
            "pii_signals": pii_signals,
            "sensitive": bool(pii_signals),
        }

        categorical = dict(stat["categorical_values"])
        if categorical:
            top_values = sorted(categorical.items(), key=lambda item: item[1], reverse=True)[:20]
            if len(categorical) <= 30:
                label_distributions[path] = {k: int(v) for k, v in top_values}
                field_payload["label_distribution"] = {k: int(v) for k, v in top_values}

        if any(token in path.lower() for token in ("text", "content", "document", "chunk", "body", "passage", "answer", "message")):
            if str_lengths:
                doc_length_candidates.extend(str_lengths)
        if field_payload["sensitive"]:
            sensitive_columns.append(path)
        if "." in path or "[]" in path:
            nested_structures.append(path)
        fields.append(field_payload)

    dataset_hints: list[str] = []
    known_paths = {item["path"] for item in fields}
    if any("messages[]" in path or "conversations[]" in path for path in known_paths):
        dataset_hints.append("chat_transcript")
    if {"prompt", "chosen", "rejected"}.issubset(known_paths):
        dataset_hints.append("pairwise_preference")
    if any(path.endswith("context") or path.endswith("passage") for path in known_paths):
        dataset_hints.append("document_or_chunk_corpus")
    if any(path.endswith("label") for path in known_paths):
        dataset_hints.append("classification")
    if {"source_text", "target_text"}.issubset(known_paths):
        dataset_hints.append("seq2seq")

    profile = {
        "source": source_meta,
        "sampled_rows": total,
        "schema": {
            "field_count": len(fields),
            "fields": fields,
            "nested_structures": sorted(set(nested_structures)),
        },
        "dataset_characteristics": {
            "label_distributions": label_distributions,
            "document_length": _length_stats(doc_length_candidates),
            "potential_sensitive_columns": sorted(set(sensitive_columns)),
            "task_shape_hints": sorted(set(dataset_hints)),
        },
    }
    return profile


def _field_lookup(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields = list(((profile.get("schema") or {}).get("fields") or []))
    out: dict[str, dict[str, Any]] = {}
    for item in fields:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if path:
            out[path] = item
    return out


def _guess_mapping_from_contract(
    *,
    contract: dict[str, Any],
    profile: dict[str, Any],
    auto_apply_mapping: dict[str, str] | None,
) -> dict[str, str]:
    source_fields = _field_lookup(profile)
    mapping: dict[str, str] = dict(auto_apply_mapping or {})
    schema_hint = dict(contract.get("schema_hint") or {})
    input_candidates = dict(schema_hint.get("input_candidates") or {})
    for canonical in CANONICAL_STUDIO_FIELDS:
        if canonical in mapping:
            continue
        aliases = [str(item).strip() for item in list(input_candidates.get(canonical) or []) if str(item).strip()]
        for alias in aliases:
            if alias in source_fields:
                mapping[canonical] = alias
                break
    return mapping


def _compute_type_conflicts(
    *,
    mapping: dict[str, str],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    lookup = _field_lookup(profile)
    conflicts: list[dict[str, Any]] = []
    expected_types: dict[str, set[str]] = {
        "text": {"string"},
        "question": {"string"},
        "answer": {"string"},
        "source_text": {"string"},
        "target_text": {"string"},
        "label": {"string", "integer"},
        "messages": {"array"},
        "structured_output": {"object", "string", "array"},
    }
    for canonical, source in mapping.items():
        expected = expected_types.get(canonical)
        if not expected:
            continue
        field = lookup.get(str(source))
        if not field:
            continue
        inferred = str(field.get("inferred_type") or "unknown")
        if inferred not in expected:
            conflicts.append(
                {
                    "canonical_field": canonical,
                    "source_field": source,
                    "expected_types": sorted(expected),
                    "inferred_type": inferred,
                    "message": (
                        f"Canonical field '{canonical}' expects {sorted(expected)} "
                        f"but source '{source}' is inferred as '{inferred}'."
                    ),
                }
            )
    return conflicts


def _drop_and_unmapped_summary(
    *,
    preview: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    sampled = int(preview.get("sampled_records") or 0)
    mapped = int(preview.get("mapped_records") or 0)
    dropped = int(preview.get("dropped_records") or 0)
    raw_field_frequency = dict(preview.get("raw_field_frequency") or {})
    mapped_sources = {str(value).strip() for value in mapping.values() if str(value).strip()}
    unmapped = [field for field in raw_field_frequency.keys() if field not in mapped_sources][:30]
    return {
        "sampled_records": sampled,
        "mapped_records": mapped,
        "dropped_records": dropped,
        "drop_rate": round((dropped / sampled), 6) if sampled > 0 else 0.0,
        "unmapped_fields": unmapped,
        "error_preview": list(preview.get("errors") or [])[:20],
    }


async def profile_dataset_structure(
    db: AsyncSession,
    project_id: int,
    *,
    source: dict[str, Any],
    sample_size: int = 500,
) -> dict[str, Any]:
    rows, source_meta = await load_source_rows(
        db,
        project_id,
        source=source,
        sample_size=sample_size,
    )
    profile = profile_rows(rows, source_meta=source_meta)
    profile["sample_preview"] = rows[:5]
    return profile


async def infer_adapter_definition(
    db: AsyncSession,
    project_id: int,
    *,
    source: dict[str, Any],
    sample_size: int = 400,
    task_profile: str | None = None,
) -> dict[str, Any]:
    rows, source_meta = await load_source_rows(
        db,
        project_id,
        source=source,
        sample_size=sample_size,
    )
    profile = profile_rows(rows, source_meta=source_meta)
    preview = preview_data_adapter(
        rows,
        adapter_id=AUTO_ADAPTER_ID,
        task_profile=task_profile,
        preview_limit=20,
    )
    resolved_adapter_id = str(preview.get("resolved_adapter_id") or DEFAULT_ADAPTER_ID)
    contract = resolve_data_adapter_contract(resolved_adapter_id)
    auto_apply = dict((preview.get("auto_apply") or {}).get("suggested_field_mapping") or {})
    mapping_canvas = _guess_mapping_from_contract(
        contract=contract,
        profile=profile,
        auto_apply_mapping=auto_apply,
    )
    type_conflicts = _compute_type_conflicts(mapping=mapping_canvas, profile=profile)
    drop_summary = _drop_and_unmapped_summary(preview=preview, mapping=mapping_canvas)
    detection_scores = dict(preview.get("detection_scores") or {})
    confidence = float(detection_scores.get(resolved_adapter_id) or 0.0)

    return {
        "project_id": project_id,
        "source": source_meta,
        "profile": profile,
        "inference": {
            "resolved_adapter_id": resolved_adapter_id,
            "resolved_task_profile": preview.get("resolved_task_profile"),
            "confidence": round(confidence, 4),
            "detection_scores": detection_scores,
            "adapter_contract": {
                "version": contract.get("version"),
                "adapter_id": contract.get("adapter_id"),
                "task_profiles": list(contract.get("task_profiles") or []),
                "preferred_training_tasks": list(contract.get("preferred_training_tasks") or []),
                "output_contract": dict(contract.get("output_contract") or {}),
            },
            "mapping_canvas": mapping_canvas,
            "auto_fix_suggestions": list(preview.get("auto_fix_suggestions") or []),
            "drop_analysis": drop_summary,
            "type_conflicts": type_conflicts,
            "unresolved_questions": [
                "Review low-confidence mappings before saving adapter version."
                if confidence < 0.7
                else "Confirm task profile and output contract alignment."
            ],
            "recommended_next_actions": [
                "Run preview with edited field mapping canvas.",
                "Validate adapter coverage before splitting/training.",
            ],
        },
        "preview_rows": list(preview.get("preview_rows") or []),
    }


async def preview_adapter_transform(
    db: AsyncSession,
    project_id: int,
    *,
    source: dict[str, Any],
    adapter_id: str,
    field_mapping: dict[str, str] | None = None,
    adapter_config: dict[str, Any] | None = None,
    task_profile: str | None = None,
    sample_size: int = 300,
    preview_limit: int = 25,
) -> dict[str, Any]:
    rows, source_meta = await load_source_rows(
        db,
        project_id,
        source=source,
        sample_size=sample_size,
    )
    profile = profile_rows(rows, source_meta=source_meta)
    preview = preview_data_adapter(
        rows,
        adapter_id=adapter_id or AUTO_ADAPTER_ID,
        adapter_config=dict(adapter_config or {}),
        field_mapping=dict(field_mapping or {}),
        task_profile=task_profile,
        preview_limit=preview_limit,
    )
    mapping = dict(field_mapping or {})
    if not mapping:
        mapping = dict((preview.get("auto_apply") or {}).get("suggested_field_mapping") or {})
    type_conflicts = _compute_type_conflicts(mapping=mapping, profile=profile)
    drop_summary = _drop_and_unmapped_summary(preview=preview, mapping=mapping)
    return {
        "project_id": project_id,
        "source": source_meta,
        "profile": profile,
        "mapping": mapping,
        "preview": preview,
        "drop_analysis": drop_summary,
        "type_conflicts": type_conflicts,
    }


def _coverage_status(
    *,
    preview: dict[str, Any],
    type_conflicts: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    reasons: list[str] = []
    actions: list[str] = []

    sampled = int(preview.get("sampled_records") or 0)
    mapped = int(preview.get("mapped_records") or 0)
    dropped = int(preview.get("dropped_records") or 0)
    conformance = dict(preview.get("conformance_report") or {})
    contract_pass = bool(conformance.get("contract_pass"))
    drop_rate = (dropped / sampled) if sampled > 0 else 1.0

    if mapped <= 0:
        reasons.append("NO_MAPPED_ROWS")
        actions.append("Add/adjust field mappings for required canonical fields.")
    if not contract_pass:
        reasons.append("CONTRACT_COVERAGE_GAP")
        actions.append("Map missing required fields listed in required_fields_below_100.")
    if drop_rate > 0.35:
        reasons.append("HIGH_DROP_RATE")
        actions.append("Inspect drop/error analysis and apply auto-fix suggestions.")
    if type_conflicts:
        reasons.append("TYPE_CONFLICTS_DETECTED")
        actions.append("Resolve type conflicts by remapping canonical fields.")

    if reasons:
        status = "warning" if "NO_MAPPED_ROWS" not in reasons else "blocker"
    else:
        status = "pass"
    return status, sorted(set(reasons)), list(dict.fromkeys(actions))


async def validate_adapter_coverage(
    db: AsyncSession,
    project_id: int,
    *,
    source: dict[str, Any],
    adapter_id: str,
    field_mapping: dict[str, str] | None = None,
    adapter_config: dict[str, Any] | None = None,
    task_profile: str | None = None,
    sample_size: int = 300,
    preview_limit: int = 25,
) -> dict[str, Any]:
    payload = await preview_adapter_transform(
        db,
        project_id,
        source=source,
        adapter_id=adapter_id,
        field_mapping=field_mapping,
        adapter_config=adapter_config,
        task_profile=task_profile,
        sample_size=sample_size,
        preview_limit=preview_limit,
    )
    preview = dict(payload.get("preview") or {})
    type_conflicts = list(payload.get("type_conflicts") or [])
    status, reason_codes, actions = _coverage_status(preview=preview, type_conflicts=type_conflicts)
    return {
        "project_id": project_id,
        "status": status,
        "reason_codes": reason_codes,
        "recommended_next_actions": actions,
        "coverage": {
            "drop_analysis": payload.get("drop_analysis"),
            "conformance_report": preview.get("conformance_report"),
            "validation_report": preview.get("validation_report"),
            "type_conflicts": type_conflicts,
        },
        "preview_rows": preview.get("preview_rows", [])[:15],
    }


def serialize_adapter_definition(row: DatasetAdapterDefinition) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "adapter_name": row.adapter_name,
        "version": row.version,
        "status": row.status,
        "source_type": row.source_type,
        "source_ref": row.source_ref,
        "base_adapter_id": row.base_adapter_id,
        "task_profile": row.task_profile,
        "field_mapping": dict(row.field_mapping or {}),
        "adapter_config": dict(row.adapter_config or {}),
        "output_contract": dict(row.output_contract or {}),
        "schema_profile": dict(row.schema_profile or {}),
        "inference_summary": dict(row.inference_summary or {}),
        "validation_report": dict(row.validation_report or {}),
        "export_template": dict(row.export_template or {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def list_adapter_definition_versions(
    db: AsyncSession,
    project_id: int,
    *,
    adapter_name: str | None = None,
    include_global: bool = True,
) -> list[DatasetAdapterDefinition]:
    stmt = select(DatasetAdapterDefinition)
    if include_global:
        stmt = stmt.where(
            or_(
                DatasetAdapterDefinition.project_id == project_id,
                DatasetAdapterDefinition.project_id.is_(None),
            )
        )
    else:
        stmt = stmt.where(DatasetAdapterDefinition.project_id == project_id)
    if adapter_name:
        stmt = stmt.where(DatasetAdapterDefinition.adapter_name == str(adapter_name).strip())
    stmt = stmt.order_by(
        DatasetAdapterDefinition.adapter_name.asc(),
        DatasetAdapterDefinition.version.desc(),
        DatasetAdapterDefinition.id.desc(),
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_adapter_definition_version(
    db: AsyncSession,
    project_id: int,
    *,
    adapter_name: str,
    version: int,
) -> DatasetAdapterDefinition | None:
    result = await db.execute(
        select(DatasetAdapterDefinition).where(
            DatasetAdapterDefinition.adapter_name == str(adapter_name).strip(),
            DatasetAdapterDefinition.version == int(version),
            or_(
                DatasetAdapterDefinition.project_id == project_id,
                DatasetAdapterDefinition.project_id.is_(None),
            ),
        )
    )
    return result.scalar_one_or_none()


async def save_adapter_definition_version(
    db: AsyncSession,
    project_id: int,
    *,
    adapter_name: str,
    source_type: str,
    source_ref: str | None,
    base_adapter_id: str,
    task_profile: str | None,
    field_mapping: dict[str, str] | None,
    adapter_config: dict[str, Any] | None,
    output_contract: dict[str, Any] | None,
    schema_profile: dict[str, Any] | None,
    inference_summary: dict[str, Any] | None,
    validation_report: dict[str, Any] | None,
    export_template: dict[str, Any] | None = None,
    share_globally: bool = False,
) -> DatasetAdapterDefinition:
    await _ensure_project(db, project_id)
    name = str(adapter_name or "").strip()
    if not name:
        raise ValueError("adapter_name is required")
    normalized_source_type = _normalize_source_type(source_type)

    owner_project_id: int | None = None if share_globally else project_id
    max_version_stmt = select(func.max(DatasetAdapterDefinition.version)).where(
        DatasetAdapterDefinition.project_id.is_(None)
        if owner_project_id is None
        else DatasetAdapterDefinition.project_id == owner_project_id,
        DatasetAdapterDefinition.adapter_name == name,
    )
    max_version_result = await db.execute(max_version_stmt)
    max_version = int(max_version_result.scalar() or 0)
    next_version = max_version + 1

    status = "draft"
    if isinstance(validation_report, dict):
        token = str(validation_report.get("status") or "").strip().lower()
        if token in {"pass", "ok"}:
            status = "validated"
        elif token in {"blocker", "failed", "error"}:
            status = "blocked"
        elif token:
            status = token

    row = DatasetAdapterDefinition(
        project_id=owner_project_id,
        adapter_name=name,
        version=next_version,
        status=status,
        source_type=normalized_source_type,
        source_ref=str(source_ref or "").strip() or None,
        base_adapter_id=str(base_adapter_id or DEFAULT_ADAPTER_ID).strip() or DEFAULT_ADAPTER_ID,
        task_profile=str(task_profile or "").strip() or None,
        field_mapping=dict(field_mapping or {}),
        adapter_config=dict(adapter_config or {}),
        output_contract=dict(output_contract or {}),
        schema_profile=dict(schema_profile or {}),
        inference_summary=dict(inference_summary or {}),
        validation_report=dict(validation_report or {}),
        export_template=dict(export_template or {}),
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)

    try:
        await publish_artifact(
            db=db,
            project_id=project_id,
            artifact_key=f"adapter_studio.{name}.v{next_version}",
            schema_ref="slm.adapter_studio.definition/v1",
            producer_stage="dataset_prep",
            metadata={
                "adapter_name": name,
                "version": next_version,
                "status": status,
                "base_adapter_id": row.base_adapter_id,
                "task_profile": row.task_profile,
                "share_globally": bool(share_globally),
            },
        )
    except Exception:
        # Saving adapter definition should not fail when artifact lineage is unavailable.
        pass
    return row


def _plugin_scaffold_code(*, template: dict[str, Any]) -> str:
    adapter_id = str(template.get("adapter_id") or "custom-adapter").strip()
    field_mapping = dict(template.get("field_mapping") or {})
    task_profiles = [str(item) for item in list(template.get("task_profiles") or []) if str(item).strip()]
    preferred_tasks = [str(item) for item in list(template.get("preferred_training_tasks") or []) if str(item).strip()]
    output_contract = dict(template.get("output_contract") or {})
    mapping_json = json.dumps(field_mapping, ensure_ascii=True, indent=2)
    task_profiles_json = json.dumps(task_profiles, ensure_ascii=True)
    preferred_tasks_json = json.dumps(preferred_tasks, ensure_ascii=True)
    output_contract_json = json.dumps(output_contract, ensure_ascii=True, indent=2)
    return (
        '"""Generated BrewSLM Adapter Studio plugin scaffold."""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        "from app.services.record_normalization import canonicalize_record\n\n\n"
        f"ADAPTER_ID = {json.dumps(adapter_id, ensure_ascii=True)}\n"
        f"DEFAULT_FIELD_MAPPING = {mapping_json}\n\n\n"
        "def map_row(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:\n"
        "    mapping = dict(DEFAULT_FIELD_MAPPING)\n"
        "    mapping.update(dict(config.get('field_mapping') or {}))\n"
        "    return canonicalize_record(record, field_mapping=mapping)\n\n\n"
        "def register_data_adapters(register):\n"
        "    register(\n"
        "        ADAPTER_ID,\n"
        "        map_row,\n"
        "        description='Generated from Adapter Studio export',\n"
        f"        task_profiles={task_profiles_json},\n"
        f"        preferred_training_tasks={preferred_tasks_json},\n"
        f"        output_contract={output_contract_json},\n"
        "    )\n"
    )


async def export_adapter_scaffold(
    db: AsyncSession,
    project_id: int,
    *,
    adapter_name: str,
    version: int,
    export_dir: str | None = None,
) -> dict[str, Any]:
    row = await get_adapter_definition_version(
        db,
        project_id,
        adapter_name=adapter_name,
        version=version,
    )
    if row is None:
        raise ValueError(f"Adapter definition '{adapter_name}' v{version} not found")

    template = {
        "adapter_id": f"{row.adapter_name}-v{row.version}",
        "adapter_name": row.adapter_name,
        "version": row.version,
        "source_type": row.source_type,
        "source_ref": row.source_ref,
        "base_adapter_id": row.base_adapter_id,
        "task_profile": row.task_profile,
        "task_profiles": list((row.inference_summary or {}).get("adapter_contract", {}).get("task_profiles") or []),
        "preferred_training_tasks": list((row.inference_summary or {}).get("adapter_contract", {}).get("preferred_training_tasks") or []),
        "output_contract": dict(row.output_contract or {}),
        "field_mapping": dict(row.field_mapping or {}),
        "adapter_config": dict(row.adapter_config or {}),
    }
    plugin_code = _plugin_scaffold_code(template=template)

    if export_dir:
        root = Path(export_dir).expanduser().resolve()
    else:
        root = settings.DATA_DIR / "projects" / str(project_id) / "adapter_exports" / f"{row.adapter_name}_v{row.version}"
    root.mkdir(parents=True, exist_ok=True)
    template_path = root / "adapter_template.json"
    plugin_path = root / "adapter_plugin.py"
    template_path.write_text(json.dumps(template, ensure_ascii=True, indent=2), encoding="utf-8")
    plugin_path.write_text(plugin_code, encoding="utf-8")

    payload = {
        "adapter_name": row.adapter_name,
        "version": row.version,
        "template": template,
        "plugin_scaffold": plugin_code,
        "written_files": {
            "template_json": str(template_path),
            "plugin_python": str(plugin_path),
        },
    }
    row.export_template = dict(template)
    await db.flush()
    return payload


def adapter_catalog_summary() -> dict[str, Any]:
    catalog = list_data_adapter_catalog()
    adapters = dict(catalog.get("adapters") or {})
    compact: list[dict[str, Any]] = []
    for adapter_id, payload in sorted(adapters.items()):
        contract = dict(payload.get("contract") or {})
        compact.append(
            {
                "adapter_id": adapter_id,
                "description": payload.get("description"),
                "source": payload.get("source"),
                "task_profiles": list(contract.get("task_profiles") or []),
                "preferred_training_tasks": list(contract.get("preferred_training_tasks") or []),
                "required_output_fields": list(((contract.get("output_contract") or {}).get("required_fields") or [])),
            }
        )
    return {
        "default_adapter": catalog.get("default_adapter"),
        "contract_version": catalog.get("contract_version"),
        "adapters": compact,
    }

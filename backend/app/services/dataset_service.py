"""Dataset preparation service — combine, profile, split, and freeze datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DatasetVersion, DocumentStatus, RawDocument
from app.models.project import Project
from app.services.domain_hook_service import (
    apply_normalizer_hook,
    resolve_project_domain_hooks,
    run_validator_hook,
)
from app.services.data_adapter_service import (
    DEFAULT_ADAPTER_ID,
    list_data_adapter_catalog,
    map_record_with_adapter,
    normalize_task_profile,
    preview_data_adapter,
    resolve_data_adapter_for_records,
    resolve_task_profile_for_adapter,
)
from app.services.domain_runtime_service import resolve_project_domain_runtime
from app.services.record_normalization import (
    build_schema_profile,
)


def _prep_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_adapter_id(value: Any) -> str:
    token = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    return token or DEFAULT_ADAPTER_ID


def _normalize_field_mapping(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key or "").strip()
        value = str(raw_value or "").strip()
        if not key or not value:
            continue
        out[key] = value
    return out


def _normalize_adapter_config(payload: Any) -> dict[str, Any]:
    return dict(payload) if isinstance(payload, dict) else {}


def _normalize_task_profile_value(payload: Any) -> str | None:
    token = normalize_task_profile(str(payload or ""), default="")
    return token or None


def _normalize_adapter_preset(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "adapter_id": DEFAULT_ADAPTER_ID,
            "adapter_config": {},
            "field_mapping": {},
            "task_profile": None,
        }
    return {
        "adapter_id": _normalize_adapter_id(payload.get("adapter_id")),
        "adapter_config": _normalize_adapter_config(payload.get("adapter_config")),
        "field_mapping": _normalize_field_mapping(payload.get("field_mapping")),
        "task_profile": _normalize_task_profile_value(payload.get("task_profile")),
    }


def _validate_adapter_id_or_raise(adapter_id: str) -> str:
    normalized = _normalize_adapter_id(adapter_id)
    catalog = list_data_adapter_catalog()
    adapters = catalog.get("adapters", {}) if isinstance(catalog, dict) else {}
    if normalized in adapters:
        return normalized
    available = sorted(str(key) for key in adapters.keys())
    raise ValueError(
        f"Unknown adapter_id '{normalized}'. Available adapters: {', '.join(available)}"
    )


def _extract_pack_adapter_preset(runtime: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(runtime, dict):
        return None
    pack_overlay = runtime.get("pack_overlay")
    if not isinstance(pack_overlay, dict):
        return None

    for key in (
        "dataset_adapter_preset",
        "dataset_adapter_defaults",
        "adapter_preset",
        "adapter_defaults",
    ):
        payload = pack_overlay.get(key)
        if isinstance(payload, dict):
            normalized = _normalize_adapter_preset(payload)
            if normalized.get("adapter_id"):
                return normalized
    return None


def apply_chat_template(entry: dict, template_name: str = "llama3") -> str:
    """Format a Q&A pair into a chat template string."""
    q = entry.get("question", "")
    a = entry.get("answer", "")

    if not q or not a:
        return ""

    if template_name == "llama3":
        return f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>"
    if template_name == "chatml":
        return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
    if template_name == "zephyr":
        return f"<|user|>\n{q}</s>\n<|assistant|>\n{a}</s>"
    if template_name == "phi3":
        return f"<|user|>\n{q}<|end|>\n<|assistant|>\n{a}<|end|>"
    return f"User: {q}\nAssistant: {a}"


def _load_records_from_file(
    file_path: Path,
    max_records: int | None = None,
    *,
    include_pending_synth: bool = False,
) -> list[dict[str, Any]]:
    """Load structured rows from JSON/JSONL/CSV/text into a list of dict records.

    USER-SUCCESS Epic 2b note: synthetic rows now carry a
    ``review_status`` field. Rows with ``review_status == "pending"``
    are excluded from training-bound reads by default — the review
    queue (Synthetic tab) is the gate. Callers that *want* the
    pending rows (the review-queue list endpoint) pass
    ``include_pending_synth=True``.
    """
    if not file_path.exists():
        return []

    ext = file_path.suffix.lower()
    records: list[dict[str, Any]] = []

    if ext == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    if (
                        not include_pending_synth
                        and row.get("review_status") == "pending"
                    ):
                        # Pending synth rows are gated by the review
                        # queue; don't leak them into training prep.
                        continue
                    records.append(row)
                else:
                    records.append({"value": row})
                if max_records and len(records) >= max_records:
                    break
        return records

    if ext == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for row in raw:
                if isinstance(row, dict):
                    records.append(row)
                else:
                    records.append({"value": row})
                if max_records and len(records) >= max_records:
                    break
            return records
        if isinstance(raw, dict):
            return [raw]
        return [{"value": raw}]

    if ext == ".csv":
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
                if max_records and len(records) >= max_records:
                    break
        return records

    # Generic text fallback.
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append({"text": line})
            if max_records and len(records) >= max_records:
                break
    return records


# Mirror of ``data_adapter_service._CLASSIFICATION_LABEL_LIST_PROMPT_CAP``
# — beyond this size we don't inline the candidate list, so we also
# don't bother pre-scanning past it.
_CLASSIFICATION_CANDIDATE_SCAN_CAP = 50


def _scan_classification_labels(
    rows: list[dict[str, Any]],
    *,
    field_mapping: dict[str, str] | None,
    adapter_config: dict[str, Any] | None,
) -> list[str] | None:
    """Pre-scan rows for unique classification labels (β-fix).

    Mirrors the label-field aliases ``_map_classification`` uses so
    the candidate set matches what the per-row map step will read.
    Returns a sorted, deduplicated list capped at
    ``_CLASSIFICATION_CANDIDATE_SCAN_CAP`` — beyond that the adapter
    falls back to the no-list prompt variant anyway, so collecting
    more wastes work."""
    label_fields = (adapter_config or {}).get("label_fields") if adapter_config else None
    aliases = list(label_fields) if isinstance(label_fields, list) and label_fields else [
        "label",
        "class",
        "category",
        "output_label",
        "target",
        "answer",
        "output",
    ]
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for alias in aliases:
            value = row.get(alias)
            if not isinstance(value, str):
                continue
            cleaned = value.strip()
            if not cleaned:
                continue
            seen.add(cleaned)
            break
        if len(seen) > _CLASSIFICATION_CANDIDATE_SCAN_CAP:
            # Too many distinct labels — fall back to no-list mode.
            # Return None so the caller skips the candidate injection.
            return None
    if not seen:
        return None
    return sorted(seen)


# ζ-fix — mirrors ``StructuredExtractionHandler.SCHEMA_SAMPLE_SIZE``
# so the adapter and the handler scan the same window of rows when
# they don't have a manifest schema to work from.
_STRUCTURED_FIELD_SCAN_SIZE = 20


def _scan_structured_extraction_fields(
    rows: list[dict[str, Any]],
    *,
    adapter_config: dict[str, Any] | None,
) -> list[str] | None:
    """Pre-scan rows for the union of top-level keys in the target
    JSON payload (ζ-fix). Mirrors the field-resolution logic in
    ``StructuredExtractionHandler._resolve_schema`` so adapter +
    handler agree on which fields to inline in the prompt.

    Returns a sorted, deduplicated list of field names, or ``None``
    when no discoverable fields exist (caller falls back to the
    no-list prompt variant — which the handler also does).
    """
    output_fields = (adapter_config or {}).get("output_fields") if adapter_config else None
    output_aliases = list(output_fields) if isinstance(output_fields, list) and output_fields else [
        "structured_output",
        "json",
        "labels",
        "entities",
        "extracted",
        "target",
        "answer",
        "output",
    ]
    seen: set[str] = set()
    scanned = 0
    for row in rows:
        if scanned >= _STRUCTURED_FIELD_SCAN_SIZE:
            break
        if not isinstance(row, dict):
            continue
        payload: Any = None
        for alias in output_aliases:
            if alias in row:
                value = row.get(alias)
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                payload = value
                break
        if payload is None:
            continue
        scanned += 1
        # Accept either an already-parsed dict OR a JSON string that
        # parses to one. Lists / scalars are ignored — the handler
        # only emits field-list prompts for object-shaped outputs.
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            parsed = payload
        if isinstance(parsed, dict):
            seen.update(str(k) for k in parsed.keys() if isinstance(k, str))
    if not seen:
        return None
    return sorted(seen)


def _normalize_rows_for_training(
    rows: list[dict[str, Any]],
    source_dataset: DatasetType,
    chat_template: str,
    normalizer_hook_spec: dict[str, Any] | None = None,
    adapter_id: str = DEFAULT_ADAPTER_ID,
    adapter_config: dict[str, Any] | None = None,
    field_mapping: dict[str, str] | None = None,
    task_profile: str | None = None,
) -> list[dict[str, Any]]:
    normalized_entries: list[dict[str, Any]] = []
    resolved_adapter_id, _ = resolve_data_adapter_for_records(
        rows,
        adapter_id=adapter_id,
        adapter_config=adapter_config,
        field_mapping=field_mapping,
        task_profile=task_profile,
    )
    resolved_task_profile = resolve_task_profile_for_adapter(
        resolved_adapter_id,
        requested_task_profile=task_profile,
    )
    # β-fix — classification adapter renders the production prompt
    # format including (when small enough) the candidate label list.
    # Pre-scan the rows to find unique labels, then inject them into
    # adapter_config so ``_map_classification`` can use them. Skip
    # for non-classification adapters or when the caller already
    # provided candidates.
    if resolved_adapter_id == "classification-label" and (
        not adapter_config or "candidates" not in adapter_config
    ):
        candidate_set = _scan_classification_labels(
            rows,
            field_mapping=field_mapping,
            adapter_config=adapter_config,
        )
        if candidate_set:
            adapter_config = dict(adapter_config or {})
            adapter_config["candidates"] = candidate_set
    # ζ-fix — same pattern for structured-extraction: pre-scan the
    # rows for the target JSON's field set, inject into config so
    # ``_map_structured_extraction`` can inline the list in its
    # wrapped prompt (matching the eval handler's behaviour). The
    # adapter's no-list fallback handles the case when no fields
    # are discoverable.
    if resolved_adapter_id == "structured-extraction" and (
        not adapter_config or "fields" not in adapter_config
    ):
        fields_set = _scan_structured_extraction_fields(
            rows,
            adapter_config=adapter_config,
        )
        if fields_set:
            adapter_config = dict(adapter_config or {})
            adapter_config["fields"] = fields_set
    for row in rows:
        canonical = map_record_with_adapter(
            row,
            adapter_id=resolved_adapter_id,
            adapter_config=adapter_config,
            field_mapping=field_mapping,
            task_profile=resolved_task_profile,
        )
        canonical = apply_normalizer_hook(row, canonical, normalizer_hook_spec)
        if not canonical:
            continue

        # Preserve original fields while ensuring canonical keys exist.
        merged = {**row, **canonical}
        if "question" in merged and "answer" in merged:
            rendered = apply_chat_template(merged, chat_template)
            if rendered:
                merged["text"] = rendered
        merged["_source_dataset"] = source_dataset.value
        merged["_adapter_id"] = resolved_adapter_id
        merged["_task_profile"] = resolved_task_profile
        normalized_entries.append(merged)
    return normalized_entries


async def resolve_project_dataset_adapter_preference(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Resolve adapter preference with fallback: project -> domain-pack overlay -> default."""
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    runtime = await resolve_project_domain_runtime(db, project_id)
    project_raw = project.dataset_adapter_preset if isinstance(project.dataset_adapter_preset, dict) else None
    project_payload = _normalize_adapter_preset(project_raw)
    has_project_override = bool(
        isinstance(project_raw, dict)
        and (
            project_raw.get("adapter_id")
            or project_raw.get("adapter_config")
            or project_raw.get("field_mapping")
            or project_raw.get("task_profile")
        )
    )
    if has_project_override:
        return {
            "project_id": project_id,
            "source": "project",
            "adapter_id": project_payload["adapter_id"],
            "adapter_config": project_payload["adapter_config"],
            "field_mapping": project_payload["field_mapping"],
            "task_profile": project_payload.get("task_profile"),
            "domain_pack_applied": runtime.get("domain_pack_applied"),
            "domain_profile_applied": runtime.get("domain_profile_applied"),
        }

    pack_payload = _extract_pack_adapter_preset(runtime)
    if pack_payload:
        return {
            "project_id": project_id,
            "source": "domain_pack",
            "adapter_id": pack_payload["adapter_id"],
            "adapter_config": pack_payload["adapter_config"],
            "field_mapping": pack_payload["field_mapping"],
            "task_profile": pack_payload.get("task_profile"),
            "domain_pack_applied": runtime.get("domain_pack_applied"),
            "domain_profile_applied": runtime.get("domain_profile_applied"),
        }

    return {
        "project_id": project_id,
        "source": "default",
        "adapter_id": DEFAULT_ADAPTER_ID,
        "adapter_config": {},
        "field_mapping": {},
        "task_profile": None,
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
    }


async def save_project_dataset_adapter_preference(
    db: AsyncSession,
    project_id: int,
    *,
    adapter_id: str,
    adapter_config: dict[str, Any] | None = None,
    field_mapping: dict[str, str] | None = None,
    task_profile: str | None = None,
) -> dict[str, Any]:
    """Persist project-level adapter preset used by split/training contract checks."""
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    normalized_adapter_id = _validate_adapter_id_or_raise(adapter_id)
    normalized_config = _normalize_adapter_config(adapter_config)
    normalized_mapping = _normalize_field_mapping(field_mapping)
    normalized_task_profile = _normalize_task_profile_value(task_profile)

    project.dataset_adapter_preset = {
        "adapter_id": normalized_adapter_id,
        "adapter_config": normalized_config,
        "field_mapping": normalized_mapping,
        "task_profile": normalized_task_profile,
    }
    await db.flush()

    return await resolve_project_dataset_adapter_preference(db, project_id)


async def resolve_training_dataset_types(
    db: AsyncSession,
    project_id: int,
    requested: list[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    """Auto-drop ``cleaned`` from the requested include_types when the
    project has a non-empty synthetic dataset.

    Rationale: in SFT mode the trainer applies its loss to every row,
    including text-only CLEANED rows that lack a target. When SYNTHETIC
    rows exist, the cleaned chunks have already served their purpose
    (they fed synthetic generation upstream) — keeping them in
    train.jsonl drowns the structured-target signal at a typical 30–80
    to 1 ratio and produces a model that learns continuation rather
    than the task schema (incident: experiment 10 / commit 222bc5d).

    Only fires when BOTH ``cleaned`` and ``synthetic`` are in the
    requested set, so callers asking for either type alone get exactly
    what they asked for. The decision is recorded in the returned
    report dict and surfaced in the split manifest so the operator can
    see what changed.

    Returns ``(resolved_types, report)``.
    """
    requested_list = list(requested) if requested else [
        DatasetType.CLEANED.value,
        DatasetType.SYNTHETIC.value,
        DatasetType.GOLD_DEV.value,
    ]
    report: dict[str, Any] = {
        "auto_excluded": [],
        "reason": None,
        "synthetic_rows": 0,
    }

    if (
        DatasetType.CLEANED.value not in requested_list
        or DatasetType.SYNTHETIC.value not in requested_list
    ):
        return requested_list, report

    synth_result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.SYNTHETIC,
        )
    )
    synth_ds = synth_result.scalar_one_or_none()
    synth_rows = int(getattr(synth_ds, "record_count", 0) or 0)
    report["synthetic_rows"] = synth_rows

    if synth_rows < 1:
        return requested_list, report

    resolved = [
        t for t in requested_list if t != DatasetType.CLEANED.value
    ]
    report["auto_excluded"] = [DatasetType.CLEANED.value]
    report["reason"] = (
        f"synthetic dataset has {synth_rows} row(s); cleaned text "
        "would dilute SFT signal (request both explicitly to override)."
    )
    return resolved, report


async def combine_datasets(
    db: AsyncSession,
    project_id: int,
    include_types: list[DatasetType] | None = None,
    chat_template: str = "llama3",
    adapter_id: str = DEFAULT_ADAPTER_ID,
    adapter_config: dict[str, Any] | None = None,
    field_mapping: dict[str, str] | None = None,
    task_profile: str | None = None,
) -> list[dict]:
    """
    Combine entries from cleaned/synthetic/gold datasets.

    Also supports `raw` datasets, which enables generic pipelines for remote imports
    and direct structured data sources without a mandatory cleaning step.
    """
    default_types = [DatasetType.CLEANED, DatasetType.SYNTHETIC, DatasetType.GOLD_DEV]
    if include_types is None:
        include_types = default_types

    normalizer_hook_spec: dict[str, Any] | None = None
    try:
        hook_state = await resolve_project_domain_hooks(db, project_id)
        normalizer_hook_spec = hook_state.get("normalizer")
    except ValueError:
        normalizer_hook_spec = None

    all_entries: list[dict[str, Any]] = []

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(include_types),
        )
    )
    datasets = list(result.scalars().all())

    # Fallback: if default sources are empty, use RAW so users can still continue.
    if not datasets and include_types == default_types:
        result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.RAW,
            )
        )
        datasets = list(result.scalars().all())

    for ds in datasets:
        if ds.dataset_type == DatasetType.RAW:
            docs_result = await db.execute(
                select(RawDocument)
                .where(
                    RawDocument.dataset_id == ds.id,
                    RawDocument.status == DocumentStatus.ACCEPTED,
                )
                .order_by(RawDocument.ingested_at.desc())
            )
            docs = docs_result.scalars().all()
            for doc in docs:
                doc_path = Path(doc.file_path)
                rows = _load_records_from_file(doc_path)
                normalized = _normalize_rows_for_training(
                    rows,
                    ds.dataset_type,
                    chat_template,
                    normalizer_hook_spec=normalizer_hook_spec,
                    adapter_id=adapter_id,
                    adapter_config=adapter_config,
                    field_mapping=field_mapping,
                    task_profile=task_profile,
                )
                for entry in normalized:
                    entry["_source_document_id"] = doc.id
                    entry["_source_document"] = doc.filename
                    all_entries.append(entry)
            continue

        if not ds.file_path:
            continue
        path = Path(ds.file_path)
        rows = _load_records_from_file(path)
        normalized = _normalize_rows_for_training(
            rows,
            ds.dataset_type,
            chat_template,
            normalizer_hook_spec=normalizer_hook_spec,
            adapter_id=adapter_id,
            adapter_config=adapter_config,
            field_mapping=field_mapping,
            task_profile=task_profile,
        )
        all_entries.extend(normalized)

    return all_entries


async def split_dataset(
    db: AsyncSession,
    project_id: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    include_types: list[str] | None = None,
    chat_template: str = "llama3",
    adapter_id: str = DEFAULT_ADAPTER_ID,
    adapter_config: dict[str, Any] | None = None,
    field_mapping: dict[str, str] | None = None,
    task_profile: str | None = None,
) -> dict:
    """Split combined data into train/val/test and save as JSONL."""
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Invalid split ratios. train must be > 0 and val/test must be >= 0.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # Resolve include_types via the auto-exclusion rule so the prep
    # step doesn't dump 70k+ text-only CLEANED rows on top of 2k
    # structured SYNTHETIC rows for SFT. The resolver is a no-op when
    # synthetic is empty or when the caller asks for a single type.
    resolved_type_strs, dataset_type_report = (
        await resolve_training_dataset_types(db, project_id, include_types)
    )
    included_source_types = resolved_type_strs[:]
    types = [DatasetType(t) for t in resolved_type_strs]

    entries = await combine_datasets(
        db,
        project_id,
        types,
        chat_template,
        adapter_id=adapter_id,
        adapter_config=adapter_config,
        field_mapping=field_mapping,
        task_profile=task_profile,
    )
    if not entries:
        raise ValueError("No data available to split. Ingest and process documents first.")

    random.seed(seed)
    random.shuffle(entries)

    total = len(entries)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": entries[:train_end],
        "val": entries[train_end:val_end],
        "test": entries[val_end:],
    }

    prep_dir = _prep_dir(project_id)
    file_paths: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    dataset_versions: dict[str, int] = {}

    for split_name, split_data in splits.items():
        file_path = prep_dir / f"{split_name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in split_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_paths[split_name] = str(file_path)
        file_hashes[split_name] = _sha256_file(file_path)

        ds_type = {
            "train": DatasetType.TRAIN,
            "val": DatasetType.VALIDATION,
            "test": DatasetType.TEST,
        }[split_name]

        result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == ds_type,
            )
        )
        ds = result.scalar_one_or_none()
        if not ds:
            ds = Dataset(
                project_id=project_id,
                name=f"{split_name.title()} Set",
                dataset_type=ds_type,
            )
            db.add(ds)
            await db.flush()

        ds.record_count = len(split_data)
        ds.file_path = str(file_path)
        await db.flush()

        version_result = await db.execute(
            select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == ds.id)
        )
        next_version = int(version_result.scalar() or 0) + 1
        version_manifest = {
            "split": split_name,
            "seed": seed,
            "chat_template": chat_template,
            "count": len(split_data),
            "sha256": file_hashes[split_name],
            "source_types": included_source_types,
        }
        db.add(
            DatasetVersion(
                dataset_id=ds.id,
                version=next_version,
                file_path=str(file_path),
                record_count=len(split_data),
                manifest=version_manifest,
            )
        )
        dataset_versions[split_name] = next_version

    manifest = {
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_entries": total,
        "splits": {k: len(v) for k, v in splits.items()},
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "file_paths": file_paths,
        "file_hashes": file_hashes,
        "dataset_versions": dataset_versions,
        "chat_template": chat_template,
        "included_types": included_source_types,
        "include_types_resolution": dataset_type_report,
        "adapter_id": adapter_id,
        "adapter_config": dict(adapter_config or {}),
        "field_mapping": dict(field_mapping or {}),
        "task_profile": _normalize_task_profile_value(task_profile),
    }
    manifest_path = prep_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


async def _sample_records_for_dataset(
    db: AsyncSession,
    project_id: int,
    dataset_type: DatasetType,
    sample_size: int,
    *,
    document_id: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source: dict[str, Any] = {"dataset_type": dataset_type.value}
    if dataset_type == DatasetType.RAW:
        if document_id is not None:
            doc_result = await db.execute(
                select(RawDocument)
                .join(Dataset, Dataset.id == RawDocument.dataset_id)
                .where(
                    RawDocument.id == document_id,
                    Dataset.project_id == project_id,
                    Dataset.dataset_type == DatasetType.RAW,
                )
            )
            doc = doc_result.scalar_one_or_none()
            if not doc:
                raise ValueError(f"Raw document {document_id} not found in project {project_id}")
            path = Path(doc.file_path)
            records = _load_records_from_file(path, sample_size)
            source.update(
                {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "file_path": str(path),
                }
            )
        else:
            docs_result = await db.execute(
                select(RawDocument)
                .join(Dataset, Dataset.id == RawDocument.dataset_id)
                .where(
                    Dataset.project_id == project_id,
                    Dataset.dataset_type == DatasetType.RAW,
                    RawDocument.status == DocumentStatus.ACCEPTED,
                )
                .order_by(RawDocument.ingested_at.desc())
            )
            docs = docs_result.scalars().all()
            for doc in docs:
                rows = _load_records_from_file(Path(doc.file_path))
                for row in rows:
                    records.append(row)
                    if len(records) >= sample_size:
                        break
                if len(records) >= sample_size:
                    break
            source["documents_scanned"] = len(docs)
    else:
        dataset_result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == dataset_type,
            )
        )
        dataset = dataset_result.scalar_one_or_none()
        if not dataset or not dataset.file_path:
            raise ValueError(f"No dataset found for type '{dataset_type.value}' in project {project_id}")

        path = Path(dataset.file_path)
        records = _load_records_from_file(path, sample_size)
        source = {
            "dataset_type": dataset_type.value,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "file_path": str(path),
        }
    return records, source


async def preview_project_data_adapter(
    db: AsyncSession,
    project_id: int,
    dataset_type: DatasetType,
    *,
    sample_size: int = 200,
    adapter_id: str = "auto",
    adapter_config: dict[str, Any] | None = None,
    document_id: int | None = None,
    field_mapping: dict[str, str] | None = None,
    task_profile: str | None = None,
    preview_limit: int = 20,
) -> dict[str, Any]:
    """Preview adapter mapping quality and per-row output for a sampled dataset slice."""
    records, source = await _sample_records_for_dataset(
        db,
        project_id,
        dataset_type,
        sample_size,
        document_id=document_id,
    )

    preview = preview_data_adapter(
        records,
        adapter_id=adapter_id,
        adapter_config=adapter_config,
        field_mapping=field_mapping,
        task_profile=task_profile,
        preview_limit=preview_limit,
    )
    return {
        "project_id": project_id,
        "source": source,
        **preview,
    }


async def profile_project_dataset(
    db: AsyncSession,
    project_id: int,
    dataset_type: DatasetType,
    sample_size: int = 500,
    document_id: int | None = None,
    field_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create schema/profile diagnostics for a project dataset."""
    hook_state = await resolve_project_domain_hooks(db, project_id)
    normalizer_hook_spec = hook_state.get("normalizer")
    validator_hook_spec = hook_state.get("validator")
    records, source = await _sample_records_for_dataset(
        db,
        project_id,
        dataset_type,
        sample_size,
        document_id=document_id,
    )

    profile = build_schema_profile(records, field_mapping=field_mapping)
    normalized_for_validation = _normalize_rows_for_training(
        records,
        dataset_type,
        chat_template="llama3",
        normalizer_hook_spec=normalizer_hook_spec,
        field_mapping=field_mapping,
    )
    validator_report = run_validator_hook(
        normalized_for_validation,
        base_profile=profile,
        hook_spec=validator_hook_spec,
    )

    return {
        "source": source,
        "profile": profile,
        "domain_hooks": hook_state,
        "validator_report": validator_report,
        "sample_records": records[:5],
        "normalized_preview": normalized_for_validation[:5],
    }

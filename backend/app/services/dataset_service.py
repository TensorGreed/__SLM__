"""Dataset preparation service — combine, profile, split, and freeze datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
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


# Epic E — versioned prepared-split storage. The active prepared files
# (prepared/{train,val,test}.jsonl + manifest.json) are overwritten on every
# Prepare; each run is *also* snapshotted under prepared/versions/{v}/ so an
# older version can be restored ("make active") and retrained from. The active
# files stay the single source the trainer/export/coverage read, so nothing
# downstream needs to know about versioning.
_PREPARED_SPLIT_FILES = ("train.jsonl", "val.jsonl", "test.jsonl")


def _prepared_versions_dir(project_id: int, version: int) -> Path:
    return _prep_dir(project_id) / "versions" / str(int(version))


# Snapshots accumulate one dir per Prepare. Small newbie datasets make this
# cheap, but cap retention so a project that re-prepares hundreds of times
# doesn't grow unbounded — keep the most recent N, prune older.
MAX_RETAINED_PREPARED_SNAPSHOTS = 10


def snapshot_prepared_version(project_id: int, version: int) -> Path:
    """Copy the current active prepared files + manifest into
    prepared/versions/{version}/ so the run survives the next Prepare's
    overwrite. Best-effort per file; returns the snapshot dir."""
    src = _prep_dir(project_id)
    dst = _prepared_versions_dir(project_id, version)
    dst.mkdir(parents=True, exist_ok=True)
    for name in (*_PREPARED_SPLIT_FILES, "manifest.json"):
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)
    return dst


def prune_prepared_version_snapshots(
    project_id: int, *, keep: int = MAX_RETAINED_PREPARED_SNAPSHOTS
) -> list[int]:
    """Delete all but the ``keep`` newest version snapshots. Returns the
    versions pruned (oldest-first). Best-effort — a failed delete is skipped,
    not raised, so it never breaks a Prepare."""
    versions = list_prepared_version_snapshots(project_id)  # newest first
    if keep < 0 or len(versions) <= keep:
        return []
    to_prune = sorted(versions[keep:])  # oldest first
    pruned: list[int] = []
    for version in to_prune:
        snap = _prepared_versions_dir(project_id, version)
        try:
            shutil.rmtree(snap, ignore_errors=True)
            pruned.append(version)
        except Exception:  # noqa: BLE001 — pruning is non-critical
            pass
    return pruned


def list_prepared_version_snapshots(project_id: int) -> list[int]:
    """Versions that have an on-disk snapshot (newest first). A version
    prepared before versioned storage landed has no snapshot — the UI uses
    this to gate the activate/retrain actions."""
    root = _prep_dir(project_id) / "versions"
    if not root.exists():
        return []
    versions: list[int] = []
    for child in root.iterdir():
        if child.is_dir() and child.name.isdigit():
            # Only count snapshots that actually carry split data.
            if any((child / name).exists() for name in _PREPARED_SPLIT_FILES):
                versions.append(int(child.name))
    return sorted(versions, reverse=True)


def read_prepared_version_manifest(
    project_id: int, version: int
) -> dict[str, Any] | None:
    """Read a version snapshot's manifest.json, or ``None`` when the snapshot
    (or its manifest) is absent. Used by the version-compare diff."""
    path = _prepared_versions_dir(project_id, version) / "manifest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def restore_prepared_version(project_id: int, version: int) -> dict[str, int]:
    """Copy a version snapshot back over the active prepared files + manifest,
    making it the data the trainer/export/coverage read. Returns the restored
    per-split row counts. Raises ``FileNotFoundError`` when no snapshot exists
    for ``version`` (e.g. prepared before versioned storage landed)."""
    snap = _prepared_versions_dir(project_id, version)
    if not snap.exists() or not any(
        (snap / name).exists() for name in _PREPARED_SPLIT_FILES
    ):
        raise FileNotFoundError(
            f"No prepared snapshot for version {version} — re-prepare to enable."
        )
    dst = _prep_dir(project_id)
    counts: dict[str, int] = {}
    for name in (*_PREPARED_SPLIT_FILES, "manifest.json"):
        src_file = snap / name
        if not src_file.exists():
            continue
        shutil.copy2(src_file, dst / name)
        if name.endswith(".jsonl"):
            split = name[: -len(".jsonl")]
            counts[split] = sum(
                1 for line in (dst / name).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return counts


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


# Per-adapter subtask specs (subtask-propagation infrastructure).
# Each entry mirrors the matching handler's ``SUBTASK_*`` constants
# + ``DEFAULT_SUBTASK`` so the adapter and the handler agree on
# enumeration AND default. Adding a new subtask-aware adapter
# means a one-line entry here plus a fix in the adapter's
# ``_map_<name>`` to read ``adapter_config['subtask']``. Adapters
# NOT in this table get ``None`` from the resolver — used as a
# signal that they don't branch per-subtask.
#
# Sources (eval_task_handler_service.py):
#   - vision-language-pair → VisionLanguageHandler:1912-1917
#   - audio-transcript     → AudioTranscriptHandler:2114-2118
#   - seq2seq-pair         → Seq2SeqHandler:2508-2516
_ADAPTER_SUBTASK_SPECS: dict[str, dict[str, Any]] = {
    "vision-language-pair": {
        "allowed": frozenset({"captioning", "vqa"}),
        "default": "captioning",
    },
    "audio-transcript": {
        "allowed": frozenset({"transcription", "audio_qa"}),
        "default": "transcription",
    },
    "seq2seq-pair": {
        "allowed": frozenset({"translation", "summarization", "paraphrase"}),
        "default": "summarization",
    },
}


def _resolve_adapter_subtask(
    adapter_id: str,
    manifest: dict[str, Any] | None,
    adapter_config: dict[str, Any] | None,
) -> str | None:
    """Resolve the eval subtask for an adapter that branches
    per-subtask (vision-language-pair / audio-transcript /
    seq2seq-pair).

    Resolution precedence (highest priority first):
      1. ``adapter_config['subtask']`` — explicit override at the
         call site. Tests + power users set this; an invalid
         value falls through rather than failing loudly so a
         legacy adapter_config doesn't break the prep pipeline.
      2. ``manifest['subtask']`` — what the handler reads at
         eval time (``VisionLanguageHandler._resolve_subtask``
         et al). Adapter must agree, or train/eval drift again.
      3. ``manifest['output_schema']['subtask']`` — some
         manifests nest task config under output_schema; accept
         both shapes.
      4. The per-adapter ``default`` (matches the handler's
         ``DEFAULT_SUBTASK`` constant). The handler also falls
         back to default when the manifest doesn't carry the
         field, so the adapter's no-manifest case stays aligned.
      5. ``None`` — returned for any adapter NOT in
         ``_ADAPTER_SUBTASK_SPECS``. Caller uses the ``None``
         signal to skip subtask injection entirely (e.g.,
         classification-label, structured-extraction,
         rag-grounded don't branch per-subtask).
    """
    spec = _ADAPTER_SUBTASK_SPECS.get(adapter_id)
    if spec is None:
        return None
    allowed: frozenset[str] = spec["allowed"]
    default: str = spec["default"]

    # 1. adapter_config override
    if isinstance(adapter_config, dict):
        raw = adapter_config.get("subtask")
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in allowed:
                return normalized
            # Invalid value → fall through to manifest. We don't
            # raise because adapter_config can carry caller-
            # forwarded extras that may have shape drift across
            # versions; loud failure here would block the prep
            # pipeline for a value that the resolver can recover.

    # 2 + 3. manifest field (with output_schema nesting fallback)
    if isinstance(manifest, dict):
        m_raw = manifest.get("subtask")
        if isinstance(m_raw, str):
            normalized = m_raw.strip().lower()
            if normalized in allowed:
                return normalized
        output_schema = manifest.get("output_schema")
        if isinstance(output_schema, dict):
            os_raw = output_schema.get("subtask")
            if isinstance(os_raw, str):
                normalized = os_raw.strip().lower()
                if normalized in allowed:
                    return normalized

    # 4. Per-adapter default.
    return default


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
    manifest: dict[str, Any] | None = None,
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
    # Subtask-propagation infrastructure (η+1) — for adapters that
    # branch per-subtask (vision-language-pair / audio-transcript /
    # seq2seq-pair), resolve the subtask from manifest + inject
    # into adapter_config so the per-row map step picks the right
    # prompt shape. Returns ``None`` for non-subtask-aware adapters;
    # those skip the injection cleanly. When the caller already
    # passed ``adapter_config['subtask']``, the resolver returns it
    # unchanged (idempotent re-injection is harmless but we still
    # guard with the ``not in`` check below to make the contract
    # explicit at the call site).
    resolved_subtask = _resolve_adapter_subtask(
        resolved_adapter_id, manifest, adapter_config,
    )
    if resolved_subtask is not None and (
        not adapter_config or "subtask" not in adapter_config
    ):
        adapter_config = dict(adapter_config or {})
        adapter_config["subtask"] = resolved_subtask
    # θ-fix — seq2seq-pair's translation branch also needs
    # ``tgt_lang`` from manifest (mirrors
    # ``Seq2SeqHandler._resolve_tgt_lang``). Inject the same way
    # we inject subtask so the adapter's wrap matches the
    # handler's translation prompt byte-for-byte. ``tgt_lang`` is
    # also valid for non-translation subtasks (the wrap ignores
    # it), so we always propagate when present.
    if (
        resolved_adapter_id == "seq2seq-pair"
        and isinstance(manifest, dict)
        and (not adapter_config or "tgt_lang" not in adapter_config)
    ):
        raw_tgt = manifest.get("tgt_lang") or manifest.get("target_language")
        if isinstance(raw_tgt, str) and raw_tgt.strip():
            adapter_config = dict(adapter_config or {})
            adapter_config["tgt_lang"] = raw_tgt.strip()
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
    stratify_by: str | None = None,
    disjoint_by: str | None = None,
) -> dict:
    """Split combined data into train/val/test and save as JSONL.

    When ``stratify_by`` is set to a top-level field name (e.g.
    ``label`` for classification, ``answer`` for QA), entries are
    grouped by the value of that field and each group is split at
    the same ratios independently. The combined splits preserve the
    per-class proportion across train/val/test — essential when one
    class is rare and a uniform random split would put zero examples
    of it in val or test. Groups with fewer than 3 entries (can't
    sensibly be split into all three buckets) go entirely to train
    and are listed in the stratification report so the caller can
    see what happened.

    When ``disjoint_by`` is set to a top-level field name (e.g.
    ``author``, ``template_id``, ``document_id``, ``customer_id``),
    entries are grouped by that field and each group is assigned
    **whole** to one split. This is the canonical guard against
    same-key leakage — a writer's prose appearing in both train and
    test, the same invoice template generating both train and held-out
    rows, etc. Groups with a missing key bucket as ``__missing__`` and
    go entirely to train so the disjoint guarantee holds as a hard
    contract on non-missing keys.

    ``stratify_by`` and ``disjoint_by`` are mutually exclusive — the
    stratify guarantee requires splitting a group, the disjoint
    guarantee forbids it.

    When both are None: behaviour is unchanged (uniform random
    shuffle + slice). The ``stratification_report`` and
    ``disjoint_report`` fields on the returned manifest are None.
    """
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Invalid split ratios. train must be > 0 and val/test must be >= 0.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    if (
        stratify_by and str(stratify_by).strip()
        and disjoint_by and str(disjoint_by).strip()
    ):
        raise ValueError(
            "stratify_by and disjoint_by are mutually exclusive — "
            "the stratify guarantee requires splitting groups, the "
            "disjoint guarantee forbids it. Pick one."
        )

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

    total = len(entries)
    stratification_report: dict | None = None
    disjoint_report: dict | None = None
    if stratify_by and str(stratify_by).strip():
        splits, stratification_report = _stratified_split_entries(
            entries,
            stratify_field=str(stratify_by).strip(),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    elif disjoint_by and str(disjoint_by).strip():
        splits, disjoint_report = _disjoint_split_entries(
            entries,
            disjoint_field=str(disjoint_by).strip(),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    else:
        random.seed(seed)
        random.shuffle(entries)

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

    # The prepared-version id for this run — the per-split DatasetVersion
    # numbers increment together each Prepare, so they're aligned; use the max
    # as the single version the snapshot + activate flow keys off.
    prepared_version = max(dataset_versions.values()) if dataset_versions else None

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
        "prepared_version": prepared_version,
        "chat_template": chat_template,
        "included_types": included_source_types,
        "include_types_resolution": dataset_type_report,
        "adapter_id": adapter_id,
        "adapter_config": dict(adapter_config or {}),
        "field_mapping": dict(field_mapping or {}),
        "task_profile": _normalize_task_profile_value(task_profile),
        "stratify_by": str(stratify_by).strip() if stratify_by else None,
        "stratification_report": stratification_report,
        "disjoint_by": str(disjoint_by).strip() if disjoint_by else None,
        "disjoint_report": disjoint_report,
    }
    manifest_path = prep_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Epic E — snapshot this run so it can be re-activated / retrained from
    # after a later Prepare overwrites the active files. Best-effort: a
    # snapshot failure must never fail the Prepare itself.
    if prepared_version is not None:
        try:
            snapshot_prepared_version(project_id, prepared_version)
            prune_prepared_version_snapshots(project_id)
        except Exception:  # noqa: BLE001 — snapshot is non-critical
            pass

    return manifest


def _stratified_split_entries(
    entries: list[dict[str, Any]],
    *,
    stratify_field: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Group entries by ``stratify_field`` value and split each group
    independently at the requested ratios. Returns the
    ``{"train": [...], "val": [...], "test": [...]}`` mapping AND a
    structured report the caller surfaces in the manifest so a user
    can verify rare classes landed in every split.

    Group → split policy:
      - **3+ rows**: standard ratio split. Empty val/test buckets
        absorb their leftover into train (sklearn-style behaviour
        when ``int(n * ratio)`` rounds to 0).
      - **2 rows**: 1 → train, 1 → val, 0 → test. Test gets nothing
        for this group; the report flags it.
      - **1 row**: entire row to train. Report flags it.

    Missing / non-string stratify-field values bucket into
    ``"__missing__"`` so the split still produces all three files
    rather than crashing — the report surfaces the missing-count so
    the caller can decide whether to clean their data and re-prep.
    """
    from collections import defaultdict

    rng = random.Random(seed)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        raw = entry.get(stratify_field) if isinstance(entry, dict) else None
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            key = "__missing__"
        else:
            key = str(raw)
        groups[key].append(entry)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    per_group: list[dict[str, Any]] = []
    small_groups: list[str] = []
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    for key in sorted(groups.keys()):
        group_entries = list(groups[key])
        rng.shuffle(group_entries)
        n = len(group_entries)

        if n >= 3:
            n_train = max(1, int(n * train_ratio))
            n_val = int(n * val_ratio)
            n_test = n - n_train - n_val
            # Stratified guarantee: when a group has ≥ 3 rows AND
            # the caller asked for a val + test bucket, each split
            # gets at least one row of this group. Otherwise rare
            # classes with int(n * ratio) rounding down to 0 vanish
            # from val/test, which is the exact failure mode this
            # whole feature exists to prevent.
            if n_val == 0 and val_ratio > 0:
                if n_train > 1:
                    n_train -= 1
                    n_val = 1
                elif n_test > 1:
                    n_test -= 1
                    n_val = 1
                n_test = n - n_train - n_val
            if n_test == 0 and test_ratio > 0:
                if n_train > 1:
                    n_train -= 1
                    n_test = 1
                elif n_val > 1:
                    n_val -= 1
                    n_test = 1
            train_part = group_entries[:n_train]
            val_part = group_entries[n_train : n_train + n_val]
            test_part = group_entries[n_train + n_val :]
        elif n == 2:
            train_part = group_entries[:1]
            val_part = group_entries[1:]
            test_part = []
            small_groups.append(key)
        else:
            train_part = group_entries
            val_part = []
            test_part = []
            small_groups.append(key)

        splits["train"].extend(train_part)
        splits["val"].extend(val_part)
        splits["test"].extend(test_part)
        per_group.append({
            "value": key,
            "total": n,
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
        })

    # Final shuffle within each split so groups don't appear in
    # contiguous blocks — important for SGD-style training that
    # benefits from interleaved mini-batches.
    for name in splits:
        rng.shuffle(splits[name])

    report: dict[str, Any] = {
        "stratify_field": stratify_field,
        "group_count": len(groups),
        "per_group": per_group,
        "missing_count": len(groups.get("__missing__", [])),
        "small_groups_train_only": small_groups,
    }
    return splits, report


def _disjoint_split_entries(
    entries: list[dict[str, Any]],
    *,
    disjoint_field: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Group entries by ``disjoint_field`` value and assign each group
    **whole** to one split — never split a group across train/val/test.

    This is the canonical guard against same-key leakage that inflates
    eval numbers: think split-by-author (a writer's prose patterns
    appearing in both train and test), split-by-template (the same
    invoice template generating both train and held-out rows), or
    split-by-document (chunks of the same document landing in both).

    Algorithm: shuffle the groups deterministically under ``seed``,
    then greedily assign each group to whichever split is furthest
    below its target row count. Greedy bin-packing on shuffled keys
    keeps actual split sizes close to the requested ratios without
    ever splitting a group.

    Missing / null / empty key values bucket as ``"__missing__"`` and
    the whole bucket goes to train (so the disjoint guarantee holds
    as a hard contract for non-missing keys). The report surfaces the
    missing count so the caller can clean their data and re-prep.
    """
    from collections import defaultdict

    rng = random.Random(seed)
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        raw = entry.get(disjoint_field) if isinstance(entry, dict) else None
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            key = "__missing__"
        else:
            key = str(raw)
        groups[key].append(entry)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    total_rows = len(entries)
    targets = {
        "train": train_ratio * total_rows,
        "val": val_ratio * total_rows,
        "test": test_ratio * total_rows,
    }
    running = {"train": 0, "val": 0, "test": 0}

    # Hard contract: missing-key rows always go to train so the
    # disjoint guarantee is unconditional on non-missing rows.
    missing_rows = groups.pop("__missing__", [])
    if missing_rows:
        splits["train"].extend(missing_rows)
        split_groups["train"].append("__missing__")
        running["train"] += len(missing_rows)

    # Stable-sort keys then shuffle so behaviour is deterministic and
    # independent of dict insertion order.
    ordered_keys = sorted(groups.keys())
    rng.shuffle(ordered_keys)

    # Assign large groups first — greedy bin-packing converges closer
    # to the target ratios when biggest items are placed earliest.
    ordered_keys.sort(key=lambda k: len(groups[k]), reverse=True)

    for key in ordered_keys:
        group_entries = groups[key]
        # Pick the split with the largest deficit (target − current).
        # If val_ratio is 0 the target is 0 and the deficit can't beat
        # train/test, so we never put rows in an empty bucket.
        deficits = {
            name: targets[name] - running[name]
            for name in ("train", "val", "test")
            if targets[name] > 0
        }
        # Fallback: if all targets are 0 (degenerate), default to train.
        if not deficits:
            best = "train"
        else:
            best = max(deficits, key=lambda n: deficits[n])
        splits[best].extend(group_entries)
        split_groups[best].append(key)
        running[best] += len(group_entries)

    # Final intra-split shuffle so groups don't appear as contiguous
    # blocks within a split (matters for SGD-style training).
    for name in splits:
        rng.shuffle(splits[name])

    per_split = {
        name: {
            "group_count": len(split_groups[name]),
            "row_count": len(splits[name]),
            "groups": split_groups[name],
        }
        for name in ("train", "val", "test")
    }
    ratio_drift = {
        name: round(
            abs((len(splits[name]) / total_rows) - {
                "train": train_ratio, "val": val_ratio, "test": test_ratio,
            }[name]),
            4,
        ) if total_rows else 0.0
        for name in ("train", "val", "test")
    }
    report: dict[str, Any] = {
        "disjoint_field": disjoint_field,
        "group_count": len(groups) + (1 if missing_rows else 0),
        "missing_count": len(missing_rows),
        "per_split": per_split,
        "ratio_drift": ratio_drift,
    }
    return splits, report


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

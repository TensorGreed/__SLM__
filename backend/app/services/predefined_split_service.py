"""Predefined-split import: the user already has train / val / test files.

``split_dataset`` always assigns rows by ratio. This imports files whose split
membership is fixed (e.g. a benchmark with a published test set) and writes the
SAME prepared artefacts a random split writes — ``prepared/{train,val,test}.jsonl``,
``manifest.json``, per-split ``DatasetVersion`` rows, a snapshot — by sharing
``dataset_service``'s row-mapping (``_resolve_adapter_for_rows`` /
``_normalize_rows_for_training``) and write helpers
(``_write_prepared_splits`` / ``_finalize_prepared_manifest``).

Membership is exact: no re-division, no shuffle, no dedup. Within-split exact
duplicates stay (and are counted); rows the adapter rejects fail the import
(422, grouped by reason); train/val/test input overlap fails (409) unless the
caller asked to drop the overlapping rows from TRAIN only. Val<->test overlap
always fails — neither side is ever edited.
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import SLMError
from app.models.dataset import DatasetType, DocumentStatus, RawDocument
from app.services import dataset_service as ds
from app.services.dataset_service import (
    _finalize_prepared_manifest,
    _load_records_from_file,
    _majority_task_profile,
    _normalize_rows_for_training,
    _question_text,
    _recipe_output_schema,
    _resolve_adapter_for_rows,
    _resolve_normalizer_hook_spec,
    _stamp_source_document,
    _write_prepared_splits,
    resolve_project_dataset_adapter_preference,
)
from app.utils.file_parsers import SUPPORTED_EXTENSIONS
from app.utils.tabular_io import is_tabular

SPLIT_ROLES: tuple[str, ...] = ("train", "val", "test")
ON_LEAK_CHOICES: tuple[str, ...] = ("fail", "drop_from_train")
SAMPLE_CAP = 20
DEFAULT_SEED = 42
SPLIT_SOURCE = "predefined"
_PAIRS: tuple[tuple[str, str], ...] = (("train", "val"), ("train", "test"), ("val", "test"))


def _fail(status: int, code: str, message: str, fix: str, metadata: dict[str, Any] | None = None) -> SLMError:
    return SLMError(
        status_code=status,
        error_code=code,
        stage="dataset",
        message=message,
        actionable_fix=fix,
        metadata=metadata,
    )


def _leak_key(entry: dict[str, Any]) -> str:
    """The normalised input text two rows are compared on: the question /
    prompt / input the row asks (else its whole text), through the same
    lower-case + whitespace-collapse normaliser the split-leakage scan uses."""
    from app.services.data_health_service import _normalise_for_exact
    from app.services.trainability_forecast_service import _row_to_text

    return _normalise_for_exact(_question_text(entry) or _row_to_text(entry))


def _sample(numbers: list[int]) -> list[int]:
    """1-based file row numbers (header excluded), capped."""
    return numbers[:SAMPLE_CAP]


def _parse_upload(role: str, filename: str, content: bytes, workdir: Path) -> list[dict[str, Any]]:
    safe = Path(filename or f"{role}.jsonl").name
    ext = Path(safe).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS or not is_tabular(safe):
        raise _fail(
            400,
            "PREDEFINED_SPLIT_BAD_FILE",
            f"The {role} file '{safe}' is not a row file (use JSONL, JSON, CSV, TSV, XLSX or Parquet).",
            "Upload a file with one example per row.",
            {"split": role, "filename": safe},
        )
    if not content:
        raise _fail(
            400, "PREDEFINED_SPLIT_EMPTY_FILE", f"The {role} file '{safe}' is empty.",
            "Upload a file with at least one row, or leave it out.", {"split": role, "filename": safe},
        )
    path = workdir / role / safe
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    rows = _load_records_from_file(path, include_pending_synth=True)
    if not rows:
        raise _fail(
            400, "PREDEFINED_SPLIT_NO_ROWS", f"No rows could be read from the {role} file '{safe}'.",
            "Check the file format and that it has a header row (CSV) or one JSON object per line.",
            {"split": role, "filename": safe},
        )
    return rows


async def import_predefined_split(
    db: AsyncSession,
    project_id: int,
    *,
    files: dict[str, tuple[str, bytes]],
    on_leak: str = "fail",
    field_mapping: dict[str, str] | None = None,
    chat_template: str = "llama3",
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Import ``files`` ({"train": (name, bytes), "val"?: ..., "test"?: ...})
    with exact membership. Returns the prepared manifest."""
    if on_leak not in ON_LEAK_CHOICES:
        raise _fail(
            400, "PREDEFINED_SPLIT_BAD_ON_LEAK", f"on_leak must be one of {list(ON_LEAK_CHOICES)}.",
            "Use 'fail' or 'drop_from_train'.",
        )
    if "train" not in files:
        raise _fail(400, "PREDEFINED_SPLIT_NO_TRAIN", "A train file is required.", "Upload a train file.")

    preset = await resolve_project_dataset_adapter_preference(db, project_id)
    adapter_id = str(preset.get("adapter_id") or ds.DEFAULT_ADAPTER_ID)
    adapter_config = dict(preset.get("adapter_config") or {})
    task_profile = str(preset.get("task_profile") or "").strip() or None
    mapping = dict(field_mapping) if field_mapping else dict(preset.get("field_mapping") or {})

    # 1. Parse (in a scratch dir; nothing is registered until every check passes).
    source_rows: dict[str, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="predefined-split-") as tmp:
        for role in SPLIT_ROLES:
            if role in files:
                name, content = files[role]
                source_rows[role] = _parse_upload(role, name, content, Path(tmp))

    # 2. Map every split through the SAME adapter path a random split uses. The
    # adapter, label candidates and extraction field list are resolved once over
    # all splits' rows, so train / val / test get an identical prompt shape.
    hook_spec = await _resolve_normalizer_hook_spec(db, project_id)
    union = [row for role in SPLIT_ROLES for row in source_rows.get(role, [])]
    resolved_adapter_id, _profile, enriched_config = _resolve_adapter_for_rows(
        union, adapter_id=adapter_id, adapter_config=adapter_config,
        field_mapping=mapping, task_profile=task_profile,
    )
    mapped: dict[str, list[dict[str, Any]]] = {}
    rejected_report: dict[str, Any] = {}
    for role, rows in source_rows.items():
        rejected: list[tuple[int, str]] = []
        mapped[role] = _normalize_rows_for_training(
            rows, DatasetType.RAW, chat_template, normalizer_hook_spec=hook_spec,
            adapter_id=resolved_adapter_id, adapter_config=enriched_config,
            field_mapping=mapping, task_profile=task_profile, rejected=rejected,
        )
        if rejected:
            by_reason: dict[str, Any] = {}
            for idx, reason in rejected:
                slot = by_reason.setdefault(reason, {"count": 0, "sample_rows": []})
                slot["count"] += 1
                if len(slot["sample_rows"]) < SAMPLE_CAP:
                    slot["sample_rows"].append(idx + 1)
            rejected_report[role] = {"count": len(rejected), "by_reason": by_reason}
    if rejected_report:
        total = sum(v["count"] for v in rejected_report.values())
        raise _fail(
            422, "PREDEFINED_SPLIT_ROWS_REJECTED",
            f"{total} row(s) could not be mapped to training examples; nothing was imported.",
            "Fix or remove those rows (row numbers are 1-based, header excluded), or set a field mapping.",
            {"rejected": rejected_report, "row_counts": {r: len(v) for r, v in source_rows.items()}},
        )

    # 3. Exact-input leakage between splits.
    keys = {role: [_leak_key(e) for e in entries] for role, entries in mapped.items()}
    key_sets = {role: {k for k in ks if k} for role, ks in keys.items()}
    leak_pairs: list[dict[str, Any]] = []
    for a, b in _PAIRS:
        if a not in mapped or b not in mapped:
            continue
        rows_a = [i + 1 for i, k in enumerate(keys[a]) if k and k in key_sets[b]]
        rows_b = [i + 1 for i, k in enumerate(keys[b]) if k and k in key_sets[a]]
        if rows_a or rows_b:
            leak_pairs.append({
                "pair": f"{a}_{b}",
                "counts": {a: len(rows_a), b: len(rows_b)},
                "sample_rows": {a: _sample(rows_a), b: _sample(rows_b)},
            })
    val_test = next((p for p in leak_pairs if p["pair"] == "val_test"), None)
    train_leaks = [p for p in leak_pairs if p["pair"] != "val_test"]
    leak_meta = {"on_leak": on_leak, "pairs": leak_pairs}
    if val_test:
        raise _fail(
            409, "PREDEFINED_SPLIT_VAL_TEST_OVERLAP",
            f"The val and test files share {val_test['counts']['test']} test row(s); nothing was imported.",
            "Neither file is edited automatically. Remove the shared rows from the val file.", leak_meta,
        )
    if train_leaks and on_leak == "fail":
        detail = "; ".join(
            f"{p['pair'].replace('_', ' and ')}: {p['counts']['train']} train row(s)" for p in train_leaks
        )
        raise _fail(
            409, "PREDEFINED_SPLIT_LEAKAGE",
            f"Rows in the train file also appear in val/test ({detail}); nothing was imported.",
            "Remove the overlap yourself, or re-submit with 'drop the overlapping rows from train'.", leak_meta,
        )
    leak_report: dict[str, Any] = {"on_leak": on_leak, "pairs": leak_pairs, "dropped_from_train": 0}
    if train_leaks:  # on_leak == "drop_from_train"
        held_out = set().union(*(key_sets[r] for r in ("val", "test") if r in key_sets))
        kept, dropped_rows = [], []
        for i, entry in enumerate(mapped["train"]):
            k = keys["train"][i]
            if k and k in held_out:
                dropped_rows.append(i + 1)
            else:
                kept.append(entry)
        if not kept:
            raise _fail(
                422, "PREDEFINED_SPLIT_TRAIN_EMPTY",
                "Dropping the rows that overlap val/test would leave train empty.",
                "Check that the train file is not the same data as val/test.", leak_meta,
            )
        mapped["train"] = kept
        keys["train"] = [k for k in keys["train"] if not (k and k in held_out)]
        leak_report["dropped_from_train"] = len(dropped_rows)
        leak_report["dropped_sample_rows"] = _sample(dropped_rows)

    # 4. Within-split exact duplicates: kept, counted.
    duplicate_report: dict[str, int] = {}
    for role in SPLIT_ROLES:
        ks = keys.get(role, [])
        seen: set[str] = set()
        dups = 0
        for k in ks:
            if not k:
                continue
            if k in seen:
                dups += 1
            seen.add(k)
        duplicate_report[role] = dups

    # 5. Register the uploads as project sources (a re-import supersedes the
    # previous predefined files so a later random re-split doesn't double them).
    from app.services.ingestion_service import get_or_create_raw_dataset, ingest_file

    raw_ds = await get_or_create_raw_dataset(db, project_id)
    prior = await db.execute(
        select(RawDocument).where(
            RawDocument.dataset_id == raw_ds.id,
            RawDocument.source.like("predefined_split:%"),
        )
    )
    for doc in prior.scalars():
        if doc.status == DocumentStatus.ACCEPTED:
            doc.status = DocumentStatus.REJECTED
            doc.metadata_ = {**(doc.metadata_ or {}), "superseded_by_predefined_split": True}
    source_files: dict[str, Any] = {}
    for role in SPLIT_ROLES:
        if role not in files:
            continue
        name, content = files[role]
        doc = await ingest_file(
            db, project_id, name, content, source=f"predefined_split:{role}", sensitivity="internal",
        )
        doc.status = DocumentStatus.ACCEPTED
        doc.metadata_ = {**(doc.metadata_ or {}), "split_role": role, "predefined_split": True}
        await db.flush()
        _stamp_source_document(mapped[role], doc.id, doc.filename)
        source_files[role] = {
            "filename": Path(name).name,
            "stored_as": doc.filename,
            "document_id": doc.id,
            "source_rows": len(source_rows[role]),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    # 6. Write through the shared tail.
    splits = {role: mapped.get(role, []) for role in SPLIT_ROLES}
    file_paths, file_hashes, dataset_versions, prepared_version = await _write_prepared_splits(
        db, project_id, splits, seed=seed, chat_template=chat_template, source_types=[DatasetType.RAW.value],
        version_extra={
            "all": {"split_source": SPLIT_SOURCE},
            "per_split": {
                role: {"source_filename": info["filename"], "source_sha256": info["sha256"]}
                for role, info in source_files.items()
            },
        },
    )
    total = sum(len(v) for v in splits.values())
    train_ratio = round(len(splits["train"]) / total, 6)
    val_ratio = round(len(splits["val"]) / total, 6)
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    manifest = {
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_source": SPLIT_SOURCE,
        "seed": seed,
        "total_entries": total,
        "splits": {k: len(v) for k, v in splits.items()},
        # Observed proportions of the imported files (not requested ratios).
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "resolved_split_config": {
            "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio,
            "seed": seed, "chat_template": chat_template,
        },
        "file_paths": file_paths,
        "file_hashes": file_hashes,
        "dataset_versions": dataset_versions,
        "prepared_version": prepared_version,
        "chat_template": chat_template,
        "included_types": [DatasetType.RAW.value],
        "include_types_resolution": {"predefined_split": True},
        "adapter_id": resolved_adapter_id,
        "adapter_config": dict(enriched_config or {}),
        "field_mapping": dict(mapping),
        "task_profile": _majority_task_profile(splits) or ds._normalize_task_profile_value(task_profile),
        "task_profile_requested": ds._normalize_task_profile_value(task_profile),
        "output_schema": await _recipe_output_schema(db, project_id),
        "stratify_by": None,
        "stratify_auto": False,
        "stratification_report": None,
        "disjoint_by": None,
        "disjoint_report": None,
        "dedup_requested": False,
        "keep_test": None,
        "dedup_report": None,
        "source_files": source_files,
        "leak_report": leak_report,
        "duplicate_report": duplicate_report,
    }
    _finalize_prepared_manifest(project_id, manifest, prepared_version)
    return manifest

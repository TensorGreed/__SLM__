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

Row numbers in every report are physical FILE LINE numbers for JSONL / CSV /
TSV (CSV: the header is line 1, the first data row line 2; blank lines are
skipped but still count); for JSON arrays, XLSX and Parquet they are the
1-based record index.

The uploads are registered as accepted raw documents carrying
``metadata.split_role``; a ``test`` document is kept out of every training
source (``split_role_service``: ``combine_datasets``, cleaning, synthetic seed
pools), and the import's scanned adapter config is saved
as the project's adapter preference so later per-document normalisation
produces identical prompts.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
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
    _majority_task_profile,
    _normalize_rows_for_training,
    _recipe_output_schema,
    _resolve_adapter_for_rows,
    _resolve_normalizer_hook_spec,
    _stamp_source_document,
    _write_prepared_splits,
    resolve_project_dataset_adapter_preference,
)
from app.utils.file_parsers import SUPPORTED_EXTENSIONS
from app.utils.tabular_io import decode_bytes, is_tabular, load_rows

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


# Fields that are the TARGET of a row (what the model should produce); never
# part of the input-side key two rows are compared on.
_TARGET_KEY_FIELDS = frozenset({
    "answer", "output", "completion", "response", "target", "label", "labels", "class", "category",
    "output_label", "structured_output", "json", "extracted", "entities", "target_text", "chosen",
    "rejected", "tool_calls", "expected",
})
# Input-side fields, in the order they are concatenated.
_INPUT_KEY_FIELDS = (
    "system", "instruction", "input", "context", "passage", "document", "content", "body", "article",
    "chunk", "message", "text", "question", "prompt", "query", "ask",
)
_NON_INPUT_KEYS = frozenset({"id", "uuid", "source", "split", "timestamp", "created_at", "updated_at", "metadata"})
_ASSISTANT_ROLES = frozenset({"assistant", "gpt", "model", "bot"})


def _input_side_text(row: Any, field_mapping: dict[str, str] | None = None, adapter_config: dict[str, Any] | None = None) -> str:
    """The raw INPUT-side text of a source row: every field the adapter reads
    as input, never a target field. Chat rows: every non-assistant turn in
    order. Otherwise the standard input fields (instruction + input + context
    + question / prompt / text ...), plus any the field mapping / adapter
    config names as the text or question source; with none of those present,
    the whole row minus target fields."""
    if not isinstance(row, dict):
        return ""
    lowered = {str(k).lower(): k for k in row}
    mapped = {str(v).lower() for k, v in (field_mapping or {}).items() if k == "answer" and v}
    cfg = adapter_config or {}
    for key in ("label_fields", "output_fields"):
        if isinstance(cfg.get(key), list):
            mapped.update(str(v).lower() for v in cfg[key])

    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)

    messages = row.get("messages") if isinstance(row.get("messages"), list) else row.get("conversations")
    if isinstance(messages, list):
        turns = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or item.get("from") or "").lower()
            if role in _ASSISTANT_ROLES:
                continue
            content = _as_text(item.get("content") or item.get("value") or item.get("text"))
            if content.strip():
                turns.append(content)
        if turns:
            return "\n".join(turns)

    names: list[str] = []
    for canonical in ("text", "question"):
        src = (field_mapping or {}).get(canonical)
        if src:
            names.append(str(src).split(".")[0].lower())
    for key in ("source_fields", "text_fields"):
        if isinstance(cfg.get(key), list):
            names.extend(str(v).lower() for v in cfg[key])
    names.extend(_INPUT_KEY_FIELDS)
    parts: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen or name in _TARGET_KEY_FIELDS or name in mapped or name not in lowered:
            continue
        seen.add(name)
        value = _as_text(row[lowered[name]])
        if value.strip():
            parts.append(value)
    if parts:
        return "\n".join(parts)
    for key, value in row.items():
        name = str(key).lower()
        if name.startswith("_") or name in _TARGET_KEY_FIELDS or name in mapped or name in _NON_INPUT_KEYS:
            continue
        text = _as_text(value)
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def _input_key_hash(text: str) -> str:
    from app.services.data_health_service import _normalise_for_exact

    norm = _normalise_for_exact(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest() if norm else ""


def _input_side_key(
    row: Any, field_mapping: dict[str, str] | None = None, adapter_config: dict[str, Any] | None = None
) -> str:
    """Normalised (lower-case, whitespace-collapsed) digest of a source row's
    input side. Two rows with the same input but different answers share it;
    a different prompt template, candidate list or field list does not change it."""
    return _input_key_hash(_input_side_text(row, field_mapping, adapter_config))


def _sample(numbers: list[int]) -> list[int]:
    """File line (or record) numbers, capped."""
    return numbers[:SAMPLE_CAP]


class _ParsedFile:
    """Rows of one upload plus where each came from: ``lines[i]`` is the
    physical file line (JSONL / CSV / TSV) or 1-based record index of
    ``rows[i]``; ``bad`` lists ``(line, reason)`` for what could not be read."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.lines: list[int] = []
        self.bad: list[tuple[int, str]] = []


def _read_jsonl(text: str, parsed: _ParsedFile) -> None:
    for number, raw in enumerate(text.split("\n"), start=1):
        token = raw.strip()
        if not token:
            continue
        try:
            row = json.loads(token)
        except json.JSONDecodeError:
            parsed.bad.append((number, "unparseable_line"))
            continue
        if not isinstance(row, dict):
            parsed.bad.append((number, "unparseable_line"))
            continue
        parsed.rows.append(row)
        parsed.lines.append(number)


def _read_delimited(text: str, ext: str, parsed: _ParsedFile) -> None:
    delimiter: str | None = "\t" if ext == ".tsv" else None
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    header: list[str] | None = None
    previous = 0
    try:
        for cells in reader:
            start = previous + 1
            previous = reader.line_num
            if not any(c.strip() for c in cells):
                continue
            if header is None:
                header = [c.strip() for c in cells]
                continue
            if len(cells) != len(header):
                parsed.bad.append((start, "malformed_row"))
                continue
            row = {h: c for h, c in zip(header, cells) if h}
            parsed.rows.append(row)
            parsed.lines.append(start)
    except csv.Error:
        parsed.bad.append((previous + 1, "malformed_row"))


def _parse_upload(role: str, filename: str, content: bytes, workdir: Path) -> _ParsedFile:
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
    parsed = _ParsedFile()
    if ext == ".jsonl":
        _read_jsonl(decode_bytes(content)[0], parsed)
    elif ext in {".csv", ".tsv"}:
        _read_delimited(decode_bytes(content)[0], ext, parsed)
    elif ext == ".json":
        from app.utils.tabular_io import _json_rows

        text = decode_bytes(content)[0]
        try:
            records = _json_rows(text)
        except json.JSONDecodeError:
            _read_jsonl(text, parsed)  # a ".json" export that is really JSON Lines
        else:
            for number, row in enumerate(records, start=1):
                if isinstance(row, dict):
                    parsed.rows.append(row)
                    parsed.lines.append(number)
                else:
                    parsed.bad.append((number, "unparseable_line"))
    else:  # xlsx / parquet: record index
        path = workdir / role / safe
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        try:
            records = load_rows(path)
        except ValueError as exc:
            raise _fail(
                400, "PREDEFINED_SPLIT_BAD_FILE", f"The {role} file '{safe}' could not be read: {exc}",
                "Check the file opens in a spreadsheet / Parquet viewer.", {"split": role, "filename": safe},
            ) from exc
        for number, row in enumerate(records, start=1):
            parsed.rows.append(row)
            parsed.lines.append(number)
    if not parsed.rows and not parsed.bad:
        raise _fail(
            400, "PREDEFINED_SPLIT_NO_ROWS", f"No rows could be read from the {role} file '{safe}'.",
            "Check the file format and that it has a header row (CSV) or one JSON object per line.",
            {"split": role, "filename": safe},
        )
    return parsed


def _apply_review_gate(parsed: _ParsedFile) -> None:
    """The random split's review gate (rows pending / rejected in the review
    queue never train), except these rows are REPORTED, not dropped silently."""
    kept_rows, kept_lines = [], []
    for row, line in zip(parsed.rows, parsed.lines):
        status = row.get("review_status")
        if status in ("pending", "rejected"):
            parsed.bad.append((line, f"review_status_{status}"))
        else:
            kept_rows.append(row)
            kept_lines.append(line)
    parsed.rows, parsed.lines = kept_rows, kept_lines


async def _persist_adapter_preference(
    db: AsyncSession, project_id: int, adapter_id: str, adapter_config: dict[str, Any] | None,
    field_mapping: dict[str, str], task_profile: str | None,
) -> None:
    """Save the import's resolved adapter wiring (including the scanned label /
    field lists) as the project preference, so a later per-document
    re-normalisation (random re-split of the registered files) builds the same
    prompts. The preset's ``origin`` becomes ``predefined_split``."""
    from app.models.project import Project

    project = await db.get(Project, project_id)
    if project is None:
        return
    preset = {
        "adapter_id": adapter_id,
        "adapter_config": ds._normalize_adapter_config(adapter_config),
        "field_mapping": ds._normalize_field_mapping(field_mapping),
        "task_profile": ds._normalize_task_profile_value(task_profile),
        "origin": "predefined_split",  # not a prior origin: this wiring was derived from the import
    }
    project.dataset_adapter_preset = preset
    await db.flush()


async def _discard_registered(db: AsyncSession, docs: list[RawDocument]) -> None:
    """Best-effort cleanup of documents ``ingest_file`` registered before a
    later step failed: delete the stored file and the row."""
    for doc in docs:
        try:
            path = Path(doc.file_path)
            path.unlink(missing_ok=True)
            path.with_suffix(".extracted.txt").unlink(missing_ok=True)
            await db.delete(doc)
        except Exception:  # noqa: BLE001 - cleanup must not mask the real error
            pass
    try:
        await db.flush()
    except Exception:  # noqa: BLE001
        pass


async def import_predefined_split(
    db: AsyncSession,
    project_id: int,
    *,
    files: dict[str, tuple[str, bytes]],
    on_leak: str = "fail",
    field_mapping: dict[str, str] | None = None,
    adapter_id: str | None = None,
    adapter_config: dict[str, Any] | None = None,
    task_profile: str | None = None,
    chat_template: str = "llama3",
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Import ``files`` ({"train": (name, bytes), "val"?: ..., "test"?: ...})
    with exact membership. Returns the prepared manifest.

    Adapter wiring follows ``POST /split``: when any of ``adapter_id`` /
    ``adapter_config`` / ``field_mapping`` / ``task_profile`` is sent, the
    request is the whole wiring (an omitted ``adapter_id`` means
    ``default-canonical``); otherwise the project's adapter preference."""
    if on_leak not in ON_LEAK_CHOICES:
        raise _fail(
            400, "PREDEFINED_SPLIT_BAD_ON_LEAK", f"on_leak must be one of {list(ON_LEAK_CHOICES)}.",
            "Use 'fail' or 'drop_from_train'.",
        )
    if "train" not in files:
        raise _fail(400, "PREDEFINED_SPLIT_NO_TRAIN", "A train file is required.", "Upload a train file.")

    if any(v is not None for v in (adapter_id, adapter_config, field_mapping, task_profile)):
        adapter_id = str(adapter_id or ds.DEFAULT_ADAPTER_ID).strip() or ds.DEFAULT_ADAPTER_ID
        adapter_config = dict(adapter_config or {})
        mapping = dict(field_mapping or {})
        task_profile = str(task_profile or "").strip() or None
        adapter_source = "request"
    else:
        preset = await resolve_project_dataset_adapter_preference(db, project_id)
        adapter_id = str(preset.get("adapter_id") or ds.DEFAULT_ADAPTER_ID)
        adapter_config = dict(preset.get("adapter_config") or {})
        mapping = dict(preset.get("field_mapping") or {})
        task_profile = str(preset.get("task_profile") or "").strip() or None
        adapter_source = str(preset.get("source") or "default")

    # 1. Parse (in a scratch dir; nothing is registered until every check passes).
    parsed: dict[str, _ParsedFile] = {}
    with tempfile.TemporaryDirectory(prefix="predefined-split-") as tmp:
        for role in SPLIT_ROLES:
            if role in files:
                name, content = files[role]
                parsed[role] = _parse_upload(role, name, content, Path(tmp))
    for pf in parsed.values():
        _apply_review_gate(pf)
    source_rows = {role: pf.rows for role, pf in parsed.items()}

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
    for role, pf in parsed.items():
        problems: list[tuple[int, str]] = list(pf.bad)
        rejected: list[tuple[int, str]] = []
        mapped[role] = _normalize_rows_for_training(
            pf.rows, DatasetType.RAW, chat_template, normalizer_hook_spec=hook_spec,
            adapter_id=resolved_adapter_id, adapter_config=enriched_config,
            field_mapping=mapping, task_profile=task_profile, rejected=rejected,
        ) if pf.rows else []
        problems.extend((pf.lines[idx], reason) for idx, reason in rejected)
        if problems:
            problems.sort()
            by_reason: dict[str, Any] = {}
            for line, reason in problems:
                slot = by_reason.setdefault(reason, {"count": 0, "sample_rows": []})
                slot["count"] += 1
                if len(slot["sample_rows"]) < SAMPLE_CAP:
                    slot["sample_rows"].append(line)
            rejected_report[role] = {"count": len(problems), "by_reason": by_reason}
    if rejected_report:
        total = sum(v["count"] for v in rejected_report.values())
        raise _fail(
            422, "PREDEFINED_SPLIT_ROWS_REJECTED",
            f"{total} row(s) could not be used as training examples; nothing was imported.",
            "Fix or remove those rows, or set a field mapping. Row numbers are file lines (a CSV header is line 1, "
            "its first data row line 2; blank lines count); for JSON arrays, XLSX and Parquet they are the "
            "1-based record number. Rows still waiting in (or rejected by) the review queue are reported too.",
            {"rejected": rejected_report, "row_counts": {r: len(v) for r, v in source_rows.items()}},
        )
    lines = {role: pf.lines for role, pf in parsed.items()}

    # 3. Exact-input leakage between splits. Keyed on the SOURCE rows' input side
    # (every row mapped, else step 2 failed, so ``mapped[role][i]`` is ``rows[i]``).
    # The key lives only here: prepared rows carry no key field.
    keys = {
        role: [_input_side_key(row, mapping, enriched_config) for row in parsed[role].rows]
        for role in mapped
    }
    key_sets = {role: {k for k in ks if k} for role, ks in keys.items()}
    leak_pairs: list[dict[str, Any]] = []
    for a, b in _PAIRS:
        if a not in mapped or b not in mapped:
            continue
        rows_a = [lines[a][i] for i, k in enumerate(keys[a]) if k and k in key_sets[b]]
        rows_b = [lines[b][i] for i, k in enumerate(keys[b]) if k and k in key_sets[a]]
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
                dropped_rows.append(lines["train"][i])
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

    registered: list[RawDocument] = []
    superseded: list[tuple[RawDocument, dict[str, Any]]] = []
    try:
        raw_ds = await get_or_create_raw_dataset(db, project_id)
        prior = await db.execute(
            select(RawDocument).where(
                RawDocument.dataset_id == raw_ds.id,
                RawDocument.source.like("predefined_split:%"),
            )
        )
        for doc in prior.scalars():
            if doc.status == DocumentStatus.ACCEPTED:
                superseded.append((doc, dict(doc.metadata_ or {})))
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
            registered.append(doc)
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

        # 6. Persist the adapter wiring the rows were prepared with, then write
        # through the shared tail.
        await _persist_adapter_preference(db, project_id, resolved_adapter_id, enriched_config, mapping, task_profile)
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
            "adapter_preference_source": adapter_source,
            "dedup_inherited_config": [],
            "source_files": source_files,
            "leak_report": leak_report,
            "duplicate_report": duplicate_report,
        }
        _finalize_prepared_manifest(project_id, manifest, prepared_version)
    except BaseException:
        # Nothing was imported: remove the files / documents registered above and
        # un-supersede the previous import.
        for doc, metadata in superseded:
            doc.status = DocumentStatus.ACCEPTED
            doc.metadata_ = metadata
        await _discard_registered(db, registered)
        raise
    return manifest

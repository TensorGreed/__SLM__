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


# --- leak / duplicate key: the INPUT side of the adapter-mapped row -----------------------------
#
# Two rows are "the same example" when the text the MODEL IS GIVEN is the same, whatever the
# answer, label, id or prompt template. The key is built from the adapter's canonical entry
# (the same mapped row that is written to prepared/*.jsonl), reading only its input fields,
# before any chat template renders the answer. Per adapter (table in agent-notes/data.md):
_CANONICAL_INPUT_KEYS: dict[str, tuple[str, ...]] = {
    "qa-pair": ("question",),
    "rag-grounded": ("question", "context"),
    "seq2seq-pair": ("question",),  # the source text (``question`` is the unwrapped source)
    "structured-extraction": ("text",),  # the document, not ``structured_output`` / the field-list prompt
    "classification-label": ("text",),  # the text, not the label / candidate-list prompt
    "tool-call-json": ("question",),  # the user prompt, not the tool name / args / result
    "preference-pair": ("prompt",),  # not chosen / rejected
    "vision-language-pair": ("image_path", "question"),
    "audio-transcript": ("audio_path", "question"),
}
_ASSISTANT_ROLES = frozenset({"assistant", "gpt", "model", "bot"})
# Input columns the default / unknown adapter reads (``record_normalization`` text + question
# aliases). Question-side aliases win: when a row has any, the key is those plus the RAG context
# columns; the text-side aliases (sentence / pair / token inputs) are used only when it has none.
# Provenance columns (``record_normalization.NON_CONTENT_KEYS``: ``source``, ``id`` ...) never enter.
_QUESTION_ALIASES: tuple[str, ...] = ("instruction", "input", "prompt", "query", "question", "ask")
_CONTEXT_ALIASES: tuple[str, ...] = ("context", "passage", "document", "article", "chunk")
_TEXT_ALIASES: tuple[str, ...] = (
    "text", "content", "body", "message", "sentence", "sentence1", "sentence2", "sentence_1", "sentence_2",
    "premise", "hypothesis", "text_a", "text_b", "text1", "text2", "tokens", "words",
)
_CONFIG_QUESTION_LISTS = ("question_fields", "prompt_fields")
_CONFIG_CONTEXT_LISTS = ("context_fields",)
_CONFIG_TEXT_LISTS = ("source_fields", "text_fields")
# Columns that are what the model should produce, never input.
_TARGET_NAMES = frozenset({
    "answer", "output", "completion", "response", "target", "label", "labels", "class", "category",
    "output_label", "structured_output", "json", "extracted", "entities", "target_text", "chosen",
    "rejected", "tool_calls", "expected", "ner_tags", "pos_tags", "chunk_tags", "tags", "bio_tags",
    "spans", "label_ids", "caption", "transcript", "summary",
})


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return " ".join(value)  # a token sequence
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _input_side_text(
    row: Any,
    entry: dict[str, Any] | None = None,
    *,
    adapter_id: str = ds.DEFAULT_ADAPTER_ID,
    field_mapping: dict[str, str] | None = None,
    adapter_config: dict[str, Any] | None = None,
) -> str:
    """The text the model is given for ``row``, read from the adapter-mapped
    ``entry`` (the canonical row; computed with the same adapter when omitted).
    Never a label / answer / target column, an id column, or the adapter's shared
    prompt boilerplate. Empty when the row has no identifiable input."""
    if not isinstance(row, dict):
        return ""
    cfg = adapter_config or {}
    if entry is None:
        from app.services.data_adapter_service import map_record_with_adapter

        entry = {**row, **(map_record_with_adapter(
            row, adapter_id=adapter_id, adapter_config=cfg, field_mapping=field_mapping) or {})}

    # Chat rows (any adapter): every non-assistant turn, in order.
    messages = entry.get("messages") if isinstance(entry.get("messages"), list) else row.get("messages")
    if not isinstance(messages, list):
        messages = row.get("conversations")
    if isinstance(messages, list) and (adapter_id == "chat-messages" or any(isinstance(m, dict) for m in messages)):
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
        return "\n".join(turns)
    keys = _CANONICAL_INPUT_KEYS.get(adapter_id)
    if keys is not None:
        return "\n".join(t for t in (_as_text(entry.get(k)) for k in keys) if t.strip())

    # default-canonical / plugin adapters: the canonical ``text`` may be derived from the
    # answer ("Question: .. Answer: .."), so read the SOURCE columns the adapter reads.
    sources = _default_input_sources(row, field_mapping, cfg)
    return "\n".join(text for text in sources if text.strip())


def _default_input_sources(
    row: dict[str, Any], field_mapping: dict[str, str] | None, cfg: dict[str, Any],
    names_out: list[str] | None = None,
) -> list[str]:
    """Texts of the input columns a default-canonical / plugin row has, in key order:
    question-side columns + context when any question-side column exists, else
    text-side columns + context. Empty list when no input column is recognised."""
    from app.services.record_normalization import NON_CONTENT_KEYS, _get_nested_value

    mapping = field_mapping or {}
    answer_path = str(mapping.get("answer") or "").strip().lower()
    lowered = {str(k).lower(): k for k in row}

    def lookup(path: str, explicit: bool) -> Any:
        path = str(path).strip()
        if not path or path.lower() == answer_path:
            return None
        if not explicit and (path.lower() in _TARGET_NAMES or path.lower() in NON_CONTENT_KEYS):
            return None
        if "." in path:
            return _get_nested_value(row, path)
        return row[lowered[path.lower()]] if path.lower() in lowered else None

    def gather(explicit: list[str], aliases: tuple[str, ...]) -> list[Any]:
        found: list[Any] = []
        seen: set[str] = set()
        for explicit_flag, names in ((True, explicit), (False, aliases)):
            for name in names:
                key = str(name).strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                value = lookup(name, explicit_flag)
                if value is not None and _as_text(value).strip():  # a blank cell is no input
                    found.append(value)
                    if names_out is not None:
                        names_out.append(str(name).strip())
        return found

    def listed(keys: tuple[str, ...]) -> list[str]:
        out: list[str] = []
        for key in keys:
            if isinstance(cfg.get(key), list):
                out.extend(str(v) for v in cfg[key])
        return out

    question = gather(([mapping["question"]] if mapping.get("question") else []) + listed(_CONFIG_QUESTION_LISTS), _QUESTION_ALIASES)
    context = gather(listed(_CONFIG_CONTEXT_LISTS), _CONTEXT_ALIASES)
    if question:
        chosen = question + context
    else:
        text_side = gather(([mapping["text"]] if mapping.get("text") else []) + listed(_CONFIG_TEXT_LISTS), _TEXT_ALIASES)
        chosen = text_side + context
        if context and not text_side and _has_other_content_column(row, mapping, cfg):
            # Only a shared context matched, yet the row has another content column (an
            # ``utterance``, a ``query_text`` ...) that is the real input: keying on the
            # context alone would call every row of one document a duplicate. Ask for a mapping.
            return []
    return [_as_text(v) for v in chosen]


def _has_other_content_column(row: dict[str, Any], mapping: dict[str, str], cfg: dict[str, Any]) -> bool:
    """Whether ``row`` has a non-blank column that is no input / context alias, no target name,
    no provenance column and no mapped answer: a candidate input the aliases did not recognise."""
    from app.services.record_normalization import NON_CONTENT_KEYS

    known = set(_QUESTION_ALIASES) | set(_CONTEXT_ALIASES) | set(_TEXT_ALIASES) | _TARGET_NAMES | set(NON_CONTENT_KEYS)
    known.add(str(mapping.get("answer") or "").strip().lower())
    for key in ("context_fields",):
        if isinstance(cfg.get(key), list):
            known.update(str(v).strip().lower() for v in cfg[key])
    for name, value in row.items():
        low = str(name).strip().lower()
        if low in known or low.startswith("_") or low == "idx" or low.endswith("_id"):
            continue
        if _as_text(value).strip():
            return True
    return False


def _overlap_input_columns(
    rows: list[dict[str, Any]], adapter_id: str, field_mapping: dict[str, str] | None, cfg: dict[str, Any],
) -> list[str]:
    """The columns the overlap check keyed on, for the result card: the adapter's input fields
    (``question`` + ``context`` ...) or, for default / plugin adapters, the source columns found."""
    if adapter_id in _CANONICAL_INPUT_KEYS:
        return list(_CANONICAL_INPUT_KEYS[adapter_id])
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if any(isinstance(row.get(k), list) and any(isinstance(m, dict) for m in row[k]) for k in ("messages", "conversations")):
            found = ["messages (every non-assistant turn)"]
        else:
            found = []
            _default_input_sources(row, field_mapping, cfg, found)
        names.extend(n for n in found if n not in names)
    return names


def _row_has_input_column(row: Any, adapter_id: str, field_mapping: dict[str, str] | None, cfg: dict[str, Any]) -> bool:
    """Whether ``row`` has a column the default key reads (chat rows count)."""
    if not isinstance(row, dict):
        return False
    for key in ("messages", "conversations"):
        if isinstance(row.get(key), list) and any(isinstance(m, dict) for m in row[key]):
            return True
    return bool(_default_input_sources(row, field_mapping, cfg))


# Adapters whose mapped input must be traceable to ONE real column of the row. The canonical
# fallback (``record_normalization``) otherwise fabricates an input: the answer itself, or the first
# few columns concatenated (gold record included). Only the first key is checked (RAG context may be
# a structured payload).
_TRACEABLE_INPUT_ADAPTERS = frozenset({
    "qa-pair", "rag-grounded", "seq2seq-pair", "structured-extraction", "classification-label",
})
_PAIR_PREFIXES = ("Premise: ", "Sentence 1: ")


def _collapse(text: str) -> str:
    return " ".join(str(text).split()).lower()


def _input_is_real(
    row: dict[str, Any], key_text: str, adapter_id: str,
    field_mapping: dict[str, str] | None, adapter_config: dict[str, Any] | None, task_profile: str | None,
) -> bool:
    """False when ``key_text`` (the model input the leak key is built from) is not the text of a real
    input column: it equals the row's answer / label, or it is no single column's text (the canonical
    fallback's concatenation). Adapters outside ``_TRACEABLE_INPUT_ADAPTERS`` only get the answer check."""
    from app.services.record_normalization import _coerce_text, _get_nested_value

    from app.services.data_adapter_service import map_record_with_adapter

    # The adapter's own mapped row (the prepared entry's ``text`` may be a rendered chat template).
    entry = map_record_with_adapter(
        row, adapter_id=adapter_id, adapter_config=adapter_config, field_mapping=field_mapping,
        task_profile=task_profile) or {}
    first = _collapse(_as_text(entry.get(_CANONICAL_INPUT_KEYS[adapter_id][0])))
    if not first:
        return True  # reported as ``no_input_field`` instead
    for field in ("answer", "label", "target_text"):
        answer = _collapse(_as_text(entry.get(field)))
        if answer and answer == first:
            return False
    if adapter_id not in _TRACEABLE_INPUT_ADAPTERS:
        return True
    if adapter_id == "classification-label" and _as_text(entry.get("text")).startswith(_PAIR_PREFIXES):
        return True  # two real columns joined by the adapter
    values: set[str] = set()
    for name, value in row.items():
        for variant in (_as_text(value), _coerce_text(value) or ""):
            if variant.strip():
                values.add(_collapse(variant))
    for path in (field_mapping or {}).values():
        value = _get_nested_value(row, str(path)) if path else None
        for variant in (_as_text(value), _coerce_text(value) or ""):
            if variant.strip():
                values.add(_collapse(variant))
    return first in values


def _input_key_hash(text: str) -> str:
    from app.services.data_health_service import _normalise_for_exact

    norm = _normalise_for_exact(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest() if norm else ""


def _input_side_key(row: Any, entry: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Normalised (lower-case, whitespace-collapsed) digest of a source row's
    model input. Two rows with the same input but different answers / labels /
    ids share it; a different prompt template, candidate list or field list does
    not change it. Empty string = no identifiable input (the import rejects the row)."""
    return _input_key_hash(_input_side_text(row, entry, **kwargs))


def _sample(numbers: list[int]) -> list[int]:
    """File line (or record) numbers, capped."""
    return numbers[:SAMPLE_CAP]


def _reserialize_rows(filename: str, rows: list[dict[str, Any]]) -> tuple[str, bytes]:
    """The upload's rows (a filtered subset) in the upload's own format, so the
    registered document re-loads exactly like the original did. XLSX / Parquet are
    stored as JSON Lines (same rows, no binary writer needed)."""
    safe = Path(filename or "train.jsonl").name
    ext = Path(safe).suffix.lower()
    if ext == ".json":
        return safe, json.dumps(rows, ensure_ascii=False, default=str).encode("utf-8")
    if ext in {".csv", ".tsv"}:
        fields: list[str] = []
        for row in rows:
            fields.extend(k for k in row if k not in fields)
        buf = io.StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=fields, delimiter="\t" if ext == ".tsv" else ",", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return safe, buf.getvalue().encode("utf-8")
    if ext != ".jsonl":
        safe = f"{Path(safe).stem}.jsonl"
    return safe, "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows).encode("utf-8")


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


def _columns_of(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        columns.extend(str(k) for k in row if str(k) not in columns)
    return columns


def _raise_input_column_unrecognised(
    parsed: dict[str, _ParsedFile], union: list[dict[str, Any]],
    flagged: dict[str, list[int]] | None = None,
) -> None:
    """422 asking for a field mapping. ``flagged`` = file lines per role whose input is not a real
    column (default: every row). ``metadata.columns`` lets the card offer the mapper."""
    columns = _columns_of(union)
    rejected: dict[str, Any] = {}
    for role, pf in parsed.items():
        lines = list(pf.lines) if flagged is None else flagged.get(role, [])
        if lines:
            rejected[role] = {"count": len(lines), "by_reason": {
                "input_column_unrecognised": {"count": len(lines), "sample_rows": _sample(lines)},
            }}
    raise _fail(
        422, "PREDEFINED_SPLIT_ROWS_REJECTED",
        "Could not tell which column is the input: none of the file's columns is a recognised input name "
        f"(columns: {', '.join(columns) or 'none'}), or the input found is the answer or a mix of columns; "
        "nothing was imported.",
        "Set a field mapping: question (or text) = the column the model is given, answer = the column it should "
        'produce, for example {"question": "utterance", "answer": "reply"}. Without it, overlap between your '
        "files cannot be checked.",
        {"reason": "input_column_unrecognised", "columns": columns, "rejected": rejected,
         "row_counts": {r: len(pf.rows) for r, pf in parsed.items()}},
    )


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


async def _accepted_test_document_ids(db: AsyncSession, project_id: int) -> list[int]:
    """Ids of the project's ACCEPTED test-role documents (its own test file)."""
    from app.models.dataset import Dataset
    from app.services.split_role_service import is_test_role_document

    result = await db.execute(
        select(RawDocument).join(Dataset, Dataset.id == RawDocument.dataset_id).where(
            Dataset.project_id == project_id, RawDocument.status == DocumentStatus.ACCEPTED,
        )
    )
    return sorted(int(d.id) for d in result.scalars() if is_test_role_document(d))


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
    drop_existing_test: bool = False,
) -> dict[str, Any]:
    """Import ``files`` ({"train": (name, bytes), "val"?: ..., "test"?: ...})
    with exact membership. Returns the prepared manifest.

    Adapter wiring follows ``POST /split``: when any of ``adapter_id`` /
    ``adapter_config`` / ``task_profile`` is sent, the request is the whole wiring (an omitted
    ``adapter_id`` means ``default-canonical``); a ``field_mapping`` ALONE is merged onto the
    project's adapter preference (adapter, config and task profile kept); otherwise the
    project's adapter preference."""
    if on_leak not in ON_LEAK_CHOICES:
        raise _fail(
            400, "PREDEFINED_SPLIT_BAD_ON_LEAK", f"on_leak must be one of {list(ON_LEAK_CHOICES)}.",
            "Use 'fail' or 'drop_from_train'.",
        )
    if "train" not in files:
        raise _fail(400, "PREDEFINED_SPLIT_NO_TRAIN", "A train file is required.", "Upload a train file.")

    lone_mapping = (
        field_mapping is not None and adapter_id is None and adapter_config is None and task_profile is None
    )
    if not lone_mapping and any(v is not None for v in (adapter_id, adapter_config, field_mapping, task_profile)):
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
        if lone_mapping:
            # A mapping alone (the result card's mapper) refines the project's adapter wiring; it
            # never resets the adapter to default-canonical. Its keys win over the saved ones.
            mapping.update({k: v for k, v in (field_mapping or {}).items() if v})
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

    # A re-import without a test file would reject the project's current own test file
    # (the supersede step below) and ship no test split: ask first.
    existing_tests: list[int] = []
    if "test" not in files:
        existing_tests = await _accepted_test_document_ids(db, project_id)
        if existing_tests and not drop_existing_test:
            raise _fail(
                409, "PREDEFINED_TEST_WOULD_BE_DROPPED",
                f"This project has its own test file ({len(existing_tests)} document(s) accepted); importing without a "
                "test file would drop it and leave the new split with no test rows. Nothing was imported.",
                "Upload a test file with this import, or confirm with 'Drop my current test file' "
                "(drop_existing_test=true) to import without one.",
                {"test_document_ids": existing_tests},
            )

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
    keys: dict[str, list[str]] = {}
    rejected_report: dict[str, Any] = {}
    if resolved_adapter_id not in _CANONICAL_INPUT_KEYS and union and not any(
        _row_has_input_column(row, resolved_adapter_id, mapping, enriched_config) for row in union
    ):
        _raise_input_column_unrecognised(parsed, union)
    flagged: dict[str, list[int]] = {}
    input_rejects = 0
    for role, pf in parsed.items():
        problems: list[tuple[int, str]] = list(pf.bad)
        rejected: list[tuple[int, str]] = []
        mapped[role] = _normalize_rows_for_training(
            pf.rows, DatasetType.RAW, chat_template, normalizer_hook_spec=hook_spec,
            adapter_id=resolved_adapter_id, adapter_config=enriched_config,
            field_mapping=mapping, task_profile=task_profile, rejected=rejected,
        ) if pf.rows else []
        problems.extend((pf.lines[idx], reason) for idx, reason in rejected)
        input_rejects += sum(1 for _i, reason in rejected if reason in ds.INPUT_REJECT_REASONS)
        # Leak / duplicate key = the adapter-mapped model input of each surviving row. A
        # row with no identifiable input cannot be compared across splits: reject it.
        rejected_idx = {idx for idx, _ in rejected}
        survivors = [i for i in range(len(pf.rows)) if i not in rejected_idx]
        keys[role] = []
        for pos, i in enumerate(survivors):
            key = _input_side_key(
                pf.rows[i], mapped[role][pos], adapter_id=resolved_adapter_id,
                field_mapping=mapping, adapter_config=enriched_config,
            )
            keys[role].append(key)
            if not key:
                problems.append((pf.lines[i], "no_input_field"))
            elif resolved_adapter_id in _CANONICAL_INPUT_KEYS and not _input_is_real(
                pf.rows[i], key, resolved_adapter_id, mapping, enriched_config, _profile,
            ):
                flagged.setdefault(role, []).append(pf.lines[i])
        if problems:
            problems.sort()
            by_reason: dict[str, Any] = {}
            for line, reason in problems:
                slot = by_reason.setdefault(reason, {"count": 0, "sample_rows": []})
                slot["count"] += 1
                if len(slot["sample_rows"]) < SAMPLE_CAP:
                    slot["sample_rows"].append(line)
            rejected_report[role] = {"count": len(problems), "by_reason": by_reason}
    # The adapter's own input rejections (input is the label / the gold record) covering EVERY row,
    # or any row whose input is not a real column: ask for a mapping instead of listing row noise.
    if union and (flagged or (not any(mapped.values()) and (
        input_rejects == len(union)
        or not any(_row_has_input_column(row, resolved_adapter_id, mapping, enriched_config) for row in union)
    ))):
        _raise_input_column_unrecognised(parsed, union, flagged or None)
    if rejected_report:
        total = sum(v["count"] for v in rejected_report.values())
        raise _fail(
            422, "PREDEFINED_SPLIT_ROWS_REJECTED",
            f"{total} row(s) could not be used as training examples; nothing was imported.",
            "Fix or remove those rows, or set a field mapping. Row numbers are file lines (a CSV header is line 1, "
            "its first data row line 2; blank lines count); for JSON arrays, XLSX and Parquet they are the "
            "1-based record number. Rows still waiting in (or rejected by) the review queue are reported too.",
            {"rejected": rejected_report, "row_counts": {r: len(v) for r, v in source_rows.items()},
             **({"columns": _columns_of(union)} if any(
                 "no_input_field" in v["by_reason"] or "input_column_unrecognised" in v["by_reason"]
                 for v in rejected_report.values()) else {})},
        )
    lines = {role: pf.lines for role, pf in parsed.items()}

    # 3. Exact-input leakage between splits, keyed on the adapter-mapped input computed in
    # step 2 (every row mapped, else step 2 failed, so ``keys[role][i]`` is ``rows[i]``'s).
    # The key lives only here: prepared rows carry no key field.
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
    register_as: dict[str, tuple[str, bytes]] = {}  # role -> (name, bytes) when it differs from the upload
    dropped_lines: list[int] = []
    if train_leaks:  # on_leak == "drop_from_train"
        held_out = set().union(*(key_sets[r] for r in ("val", "test") if r in key_sets))
        kept, dropped_rows, dropped_idx_train = [], [], set()
        for i, entry in enumerate(mapped["train"]):
            k = keys["train"][i]
            if k and k in held_out:
                dropped_rows.append(lines["train"][i])
                dropped_idx_train.add(i)
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
        # The train document registered below must hold exactly the kept rows, or a later
        # re-split would pool the dropped (overlapping) rows back in.
        kept_source = [r for i, r in enumerate(parsed["train"].rows) if i not in dropped_idx_train]
        register_as["train"] = _reserialize_rows(files["train"][0], kept_source)
        dropped_lines = list(dropped_rows)

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
            reg_name, reg_content = register_as.get(role, (name, content))
            doc = await ingest_file(
                db, project_id, reg_name, reg_content, source=f"predefined_split:{role}", sensitivity="internal",
            )
            registered.append(doc)
            doc.status = DocumentStatus.ACCEPTED
            provenance: dict[str, Any] = {}
            if role in register_as:
                # The registered document holds the rows that were kept; the upload's name,
                # hash and the dropped line numbers stay on record.
                provenance = {
                    "original_filename": Path(name).name,
                    "original_sha256": hashlib.sha256(content).hexdigest(),
                    "dropped_row_lines": dropped_lines,
                }
            doc.metadata_ = {**(doc.metadata_ or {}), "split_role": role, "predefined_split": True, **provenance}
            await db.flush()
            _stamp_source_document(mapped[role], doc.id, doc.filename)
            source_files[role] = {
                "filename": Path(name).name,
                "stored_as": doc.filename,
                "document_id": doc.id,
                "source_rows": len(source_rows[role]),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            if role in register_as:
                source_files[role].update({
                    "registered_rows": len(mapped[role]),
                    "registered_sha256": hashlib.sha256(reg_content).hexdigest(),
                    "dropped_row_lines": dropped_lines,
                })

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
            "overlap_input_columns": _overlap_input_columns(union, resolved_adapter_id, mapping, enriched_config),
        }
        if "test" in files:  # added only when true: manifests without it are unchanged
            manifest["predefined_test_active"] = True
        if existing_tests:  # confirmed with drop_existing_test: say which test documents were retired
            manifest["dropped_existing_test_document_ids"] = existing_tests
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

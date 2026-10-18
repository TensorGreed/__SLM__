"""External benchmarks: score a project's runs on a labelled set someone
else made (Epic G-B1).

A benchmark is imported from any dataset-import locator (``hf:<id>[:<split>
[:<revision>]]``, ``file:<path>``, ``jsonl:…``, ``csv:…``), its rows are
sampled across the source (not the first N — sources are often sorted by
label), its task shape is detected, and its rows are converted to the
project's own row format:

- ``classification`` — a text column + a label column (a HF ``ClassLabel``
  or a low-cardinality column) → ``{"text", "label"}``;
- ``structured_extraction`` (spans) — either a text column + a list of
  ``{start, end, label|type|entity…}`` dicts, or ``tokens`` + BIO ``tags``
  (joined with spaces, char offsets computed) → ``{"text", "entities"}``;
- ``qa`` — a question column + an answer column (+ an optional context
  column; SQuAD-style ``answers`` dicts read their first text) →
  ``{"question", "answer"[, "context"]}``. For a project that does not
  answer from a context (anything but ``rag_qa``) the context is folded
  into the question, so the question stays answerable;
- ``summarization`` — a document column + a summary column →
  ``{"question": document, "answer": summary}`` (the qa-pair row format
  summarization projects train on).

Generative shapes are found by ``task_shape_service.detect_task_shape`` (the
one detector; explicit column names as the fallback) and have no labels to
map.

The user maps every source label onto one of the project's labels or
``ignore`` (many-to-one allowed: ``1 star`` and ``2 star`` → ``negative``),
or onto a GROUP of project labels (one-to-many: ``DATE`` →
``[visit_date, date_of_birth]``). A group is scored as one label named
``"date_of_birth|visit_date"``: the benchmark rows carry the group, and a
prediction of any member is collapsed to the group before scoring. A project
label belongs to at most one group.
Classification rows whose label is ignored are dropped; ignored entity
types are removed, and span scoring then counts ONLY the mapped types —
the model is not penalised for types the benchmark never labels.

Rows that match the project's training corpus (exact or near-identical,
the leakage matcher of ``data_health_service``) are dropped, so the
benchmark stays held out. The result is a locked ``DatasetType.BENCHMARK``
dataset: never a training source, never edited.

Scoring runs the run's model over the benchmark rows with the same
prediction path as the held-out evaluation
(``evaluation_service._generate_heldout_predictions``) and the project's
task handler, IN MEMORY — benchmark scores are NOT EvalResult rows (the
project's "latest eval" readers must keep reading its own answer key).
Each score is appended to ``benchmarks/<slug>/results.jsonl``.

Generative benchmarks are scored like the project's own held-out eval: the
handler's exact match / F1, and the LLM answer judge (``answer_judge_service``)
when the task is long-answer and the project's judge resolution finds one.

``local_only``: the benchmark's rows may only be read by models running on
this machine — evaluating a run whose model is not a local checkpoint is
refused, and a cloud answer judge is skipped (the result says so). Import and
scoring never send rows anywhere else.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType

IGNORE = "ignore"
DEFAULT_ROWS = 300
MAX_ROWS = 2000
SCAN_CAP = 5000
PREVIEW_SCAN_CAP = 1000
MAX_SOURCE_LABELS = 200
FAILURES_PREVIEW = 5

SHAPE_CLASSIFICATION = "classification"
SHAPE_SPANS = "structured_extraction"
SHAPE_QA = "qa"
SHAPE_SUMMARIZATION = "summarization"
GENERATIVE_SHAPES = frozenset({SHAPE_QA, SHAPE_SUMMARIZATION})
# Project task profiles a generative benchmark can be scored on.
GENERATIVE_PROJECT_PROFILES = frozenset({
    "qa", "rag_qa", "instruction_sft", "chat_sft", "summarization", "seq2seq", "language_modeling",
})
GROUP_SEP = "|"

# Generative columns (fallback when the shape detector names none).
_QUESTION_KEYS = ("question", "query", "instruction", "prompt")
_ANSWER_KEYS = ("answer", "answers", "response", "output", "reference", "completion", "target_text")
_CONTEXT_KEYS = ("context", "passage", "evidence", "retrieved_context")
_DOCUMENT_KEYS = ("document", "article", "text", "dialogue", "content", "body", "transcript", "source")
_SUMMARY_KEYS = ("summary", "highlights", "abstract", "tldr", "tl_dr", "synopsis")

_TEXT_KEYS = ("text", "sentence", "review", "content", "document", "body", "input", "utterance", "source_text", "question")
_LABEL_KEYS = ("label", "labels", "sentiment", "category", "class", "target", "stars", "rating", "intent", "topic")
_TOKEN_KEYS = ("tokens", "words")
_TAG_KEYS = ("ner_tags", "tags", "labels", "ner", "bio_tags")
_SPAN_LABEL_KEYS = ("label", "type", "entity", "entity_type", "entity_group", "tag")
# Entity lists without offsets: [{"entity": "jo@x.io", "types": ["email"]}]
_ENTITY_TEXT_KEYS = ("entity", "value", "text", "span", "word")
_ENTITY_TYPE_KEYS = ("types", "type", "label", "labels", "entity_type", "category")
_TOKEN_JOIN = " "


class BenchmarkError(ValueError):
    """A request the API answers with 4xx."""


# ── Source loading ───────────────────────────────────────────────────────


def _label_names_from_features(features: Any) -> dict[str, list[str]]:
    """Column → ClassLabel names, for plain and list-of-ClassLabel columns."""
    out: dict[str, list[str]] = {}
    if features is None:
        return out
    try:
        items = dict(features).items()
    except Exception:  # noqa: BLE001
        return out
    for name, feature in items:
        inner = getattr(feature, "feature", None)  # datasets.Sequence / List
        target = inner if inner is not None and hasattr(inner, "names") else feature
        names = getattr(target, "names", None)
        if isinstance(names, list) and names:
            out[str(name)] = [str(n) for n in names]
    return out


def _hf_licence(dataset_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(dataset_id)
        card = info.card_data.to_dict() if info.card_data else {}
        lic = card.get("license")
        if isinstance(lic, list):
            return ", ".join(str(x) for x in lic) or None
        return str(lic) if lic else None
    except Exception:  # noqa: BLE001 — offline / private: licence unknown
        return None


def load_source(locator: str, *, scan_cap: int = SCAN_CAP) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Up to ``scan_cap`` raw rows of the source + info (source id, dataset
    id, split, revision, licence, ClassLabel names per column)."""
    import app.services.dataset_import.sources  # noqa: F401 — registers connectors
    from app.services.dataset_import.registry import resolve_source, split_locator

    locator = (locator or "").strip()
    if not locator:
        raise BenchmarkError("Give a source: hf:<dataset>[:<split>[:<revision>]] or an uploaded file.")
    try:
        source_id, rest = split_locator(locator)
        source = resolve_source(source_id)
    except (KeyError, ValueError) as exc:
        raise BenchmarkError(str(exc)) from exc
    info: dict[str, Any] = {"locator": locator, "source_id": source_id, "dataset_id": rest, "split": None,
                            "revision": None, "licence": None, "label_names": {}}
    rows: list[dict[str, Any]] = []
    try:
        if source_id == "hf":
            dataset, dataset_id, split, revision = source._open_streaming(rest)  # type: ignore[attr-defined]
            info.update(dataset_id=dataset_id, split=split, revision=revision)
            info["label_names"] = _label_names_from_features(getattr(dataset, "features", None))
            info["licence"] = _hf_licence(dataset_id)
            for row in dataset:
                rows.append(dict(row))
                if len(rows) >= scan_cap:
                    break
        else:
            for row in source.load(rest, limit=scan_cap):
                if isinstance(row, dict):
                    rows.append(dict(row))
    except BenchmarkError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface the loader's message
        raise BenchmarkError(f"Could not read {locator}: {exc}") from exc
    if not rows:
        raise BenchmarkError(f"{locator} has no rows.")
    return rows, info


_FILE_SOURCES = ("file", "jsonl", "csv")


def check_locator(project_id: int, locator: str) -> None:
    """File locators must point inside the project's upload folder
    (``projects/<id>/imports/`` — where browser uploads are staged); a
    benchmark import never reads an arbitrary path on the server."""
    from app.services.dataset_import.registry import split_locator

    try:
        source_id, rest = split_locator((locator or "").strip())
    except ValueError as exc:
        raise BenchmarkError(str(exc)) from exc
    if source_id not in _FILE_SOURCES:
        return
    root = (_project_dir(project_id) / "imports").resolve()
    path = Path(rest).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise BenchmarkError("Upload the file first — benchmark files are read from this project's uploads only.") from exc


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int = 0) -> list[dict[str, Any]]:
    """``limit`` rows spread across ``rows`` (deterministic), in source order."""
    if len(rows) <= limit:
        return list(rows)
    picked = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[i] for i in picked]


# ── Shape detection + conversion ─────────────────────────────────────────


def _as_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return None


def _string_columns(rows: list[dict[str, Any]]) -> list[str]:
    cols: list[str] = []
    for key in rows[0]:
        if all(isinstance(r.get(key), str) for r in rows[:50] if key in r):
            cols.append(key)
    return cols


def _pick_text_column(rows: list[dict[str, Any]], candidates: list[str]) -> str | None:
    for key in _TEXT_KEYS:
        if key in candidates:
            return key
    if not candidates:
        return None
    return max(candidates, key=lambda c: sum(len(str(r.get(c) or "")) for r in rows[:50]))


def _parse_entity_list(value: Any) -> list[Any] | None:
    """A list, or a string holding one (JSON or a Python literal)."""
    items = _as_list(value)
    if items is not None:
        return items
    if isinstance(value, str) and value.strip().startswith("["):
        import ast

        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                return parsed
    return None


def _entity_fields(item: Any) -> tuple[str, str] | None:
    """(text key, type key) of an offset-less entity dict."""
    if not isinstance(item, dict):
        return None
    tk = next((k for k in _ENTITY_TEXT_KEYS if isinstance(item.get(k), str) and item.get(k)), None)
    yk = next((k for k in _ENTITY_TYPE_KEYS if k != tk and (isinstance(item.get(k), str) or _as_list(item.get(k)))), None)
    return (tk, yk) if tk and yk else None


def _entity_string_key(rows: list[dict[str, Any]], text_key: str) -> tuple[str, str, str] | None:
    """A column of offset-less entity dicts whose texts occur in ``text_key``
    → (column, entity-text key, type key)."""
    for key in rows[0]:
        if key == text_key:
            continue
        found = hits = 0
        fields = None
        for r in rows[:30]:
            items = _parse_entity_list(r.get(key))
            if items is None:
                fields = None
                break
            text = str(r.get(text_key) or "")
            for it in items:
                f = _entity_fields(it)
                if f is None:
                    continue
                fields = fields or f
                found += 1
                hits += it[f[0]] in text
        if fields and found and hits / found >= 0.8:
            return key, fields[0], fields[1]
    return None


def locate_entities(text: str, entities: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Offset-less (entity text, type) pairs → every non-overlapping
    occurrence in ``text``, longest entity first."""
    taken: list[tuple[int, int]] = []
    spans: list[dict[str, Any]] = []
    for value, label in sorted(set(entities), key=lambda e: -len(e[0])):
        if not value.strip():
            continue
        start = text.find(value)
        while start != -1:
            end = start + len(value)
            if not any(start < b and end > a for a, b in taken):
                taken.append((start, end))
                spans.append({"type": label, "start": start, "end": end, "text": value})
            start = text.find(value, end)
    return sorted(spans, key=lambda e: (e["start"], e["end"]))


def _span_list_key(rows: list[dict[str, Any]], text_key: str) -> tuple[str, str] | None:
    """A column of span dicts whose offsets fit ``text_key`` → (column, label key)."""
    for key in rows[0]:
        if key == text_key:
            continue
        seen = 0
        label_key = None
        ok = True
        for r in rows[:30]:
            items = _as_list(r.get(key))
            if items is None:
                ok = False
                break
            for it in items:
                if not isinstance(it, dict) or not {"start", "end"} <= set(it):
                    ok = False
                    break
                lk = next((k for k in _SPAN_LABEL_KEYS if isinstance(it.get(k), str)), None)
                if lk is None:
                    ok = False
                    break
                label_key = label_key or lk
                text = str(r.get(text_key) or "")
                if not (0 <= int(it["start"]) < int(it["end"]) <= len(text)):
                    ok = False
                    break
                seen += 1
            if not ok:
                break
        if ok and seen and label_key:
            return key, label_key
    return None


def answer_text(value: Any) -> str | None:
    """A text out of an answer cell: a string, the first text of a
    SQuAD-style ``{"text": [...], "answer_start": [...]}`` dict, or the
    first string of a list."""
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, dict):
        for key in ("text", "answer", "value", "summary"):
            found = answer_text(value.get(key))
            if found:
                return found
        return None
    items = _as_list(value)
    if items is not None:
        for item in items:
            found = answer_text(item)
            if found:
                return found
    return None


def _flatten_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows with list / dict cells read as their first text — what the
    column detector understands."""
    out = []
    for r in rows:
        out.append({k: (answer_text(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in r.items()})
    return out


def _free_text_column(rows: list[dict[str, Any]], key: str) -> bool:
    """An answer / summary column, not a label: many distinct values, or
    values of several words."""
    values = [answer_text(r.get(key)) for r in rows]
    values = [v for v in values if v]
    if len(values) < max(1, len(rows) // 2):
        return False
    words = sorted(len(v.split()) for v in values)
    distinct = len({v.strip().lower() for v in values})
    return words[len(words) // 2] >= 4 or distinct > min(50, len(values) // 2)


def _first_key(keys: list[str], candidates: tuple[str, ...], rows: list[dict[str, Any]], exclude: set[str]) -> str | None:
    for key in candidates:
        if key in keys and key not in exclude and any(answer_text(r.get(key)) for r in rows[:50]):
            return key
    return None


def _generative_shape(fields: dict[str, str]) -> dict[str, Any]:
    if "document" in fields:
        return {"task_profile": SHAPE_SUMMARIZATION, "format": "document_summary", "fields": fields}
    return {"task_profile": SHAPE_QA, "format": "qa_context" if fields.get("context") else "qa_pair", "fields": fields}


def detect_generative_shape(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Q&A (question + answer [+ context]) or summarization (document +
    summary) — via the one task-shape detector, with explicit column names
    as the fallback. None when neither fits (or the "answer" is a label)."""
    sample = rows[:200]
    flat = _flatten_cells(sample)
    keys = list(rows[0])
    try:
        from app.services.task_shape_service import detect_task_shape

        detection = detect_task_shape(flat)
        candidates = detection.get("candidates") or []
    except Exception:  # noqa: BLE001 — the fallback below still runs
        candidates = []
    for cand in candidates:
        profile = cand.get("task_profile")
        fm = cand.get("field_map") or {}
        q, a = fm.get("question_field"), fm.get("answer_field")
        if cand.get("source") != "columns" or profile not in ("qa", "rag_qa", "summarization") or not (q and a):
            continue
        if q not in keys or a not in keys or q == a or not _free_text_column(sample, a):
            continue
        if profile == "summarization":
            return _generative_shape({"document": q, "summary": a})
        fields = {"question": q, "answer": a}
        ctx = fm.get("context_field")
        if profile == "rag_qa" and ctx in keys and ctx not in (q, a):
            fields["context"] = ctx
        return _generative_shape(fields)
    # Fallback: conventional column names.
    q = _first_key(keys, _QUESTION_KEYS, sample, set())
    a = _first_key(keys, _ANSWER_KEYS, sample, {q} if q else set())
    if q and a and _free_text_column(sample, a):
        fields = {"question": q, "answer": a}
        ctx = _first_key(keys, _CONTEXT_KEYS + ("document",), sample, {q, a})
        if ctx:
            fields["context"] = ctx
        return _generative_shape(fields)
    s = _first_key(keys, _SUMMARY_KEYS, sample, set())
    d = _first_key(keys, _DOCUMENT_KEYS, sample, {s} if s else set())
    if s and d and _free_text_column(sample, s):
        return _generative_shape({"document": d, "summary": s})
    return None


def detect_shape(rows: list[dict[str, Any]], label_names: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """The benchmark's shape + the fields it reads. Raises BenchmarkError for
    shapes not supported yet (anything but classification, spans, Q&A and
    summarization)."""
    label_names = label_names or {}
    if not rows:
        raise BenchmarkError("No rows to inspect.")
    keys = set(rows[0])
    # tokens + BIO tags
    tok = next((k for k in _TOKEN_KEYS if k in keys and _as_list(rows[0].get(k)) is not None), None)
    if tok:
        for tag in _TAG_KEYS:
            if tag in keys and tag != tok and _as_list(rows[0].get(tag)) is not None:
                names = label_names.get(tag)
                sample_tags = [t for r in rows[:50] for t in (_as_list(r.get(tag)) or [])]
                as_names = [names[int(t)] if names and str(t).lstrip("-").isdigit() and int(t) < len(names) else str(t) for t in sample_tags]
                if any(re.match(r"^[BI][-_]", n) for n in as_names):
                    return {"task_profile": SHAPE_SPANS, "format": "bio_tokens", "fields": {"tokens": tok, "tags": tag}}
    strings = _string_columns(rows)
    text_key = _pick_text_column(rows, strings)
    if text_key:
        spans = _span_list_key(rows, text_key)
        if spans:
            return {"task_profile": SHAPE_SPANS, "format": "char_spans",
                    "fields": {"text": text_key, "spans": spans[0], "span_label": spans[1]}}
        strings = _entity_string_key(rows, text_key)
        if strings:
            return {"task_profile": SHAPE_SPANS, "format": "entity_strings",
                    "fields": {"text": text_key, "entities": strings[0], "entity_text": strings[1], "entity_type": strings[2]}}
    generative = detect_generative_shape(rows)
    if generative:
        return generative
    if text_key:
        for key in [k for k in rows[0] if k in label_names] + [k for k in _LABEL_KEYS if k in keys]:
            if key == text_key:
                continue
            values = {str(r.get(key)) for r in rows if r.get(key) is not None and not isinstance(r.get(key), (list, dict))}
            if 1 < len(values) <= 50:
                return {"task_profile": SHAPE_CLASSIFICATION, "format": "text_label",
                        "fields": {"text": text_key, "label": key}}
    from app.services.task_shape_service import detect_task_shape

    guess = detect_task_shape(rows[:200])
    top = (guess.get("candidates") or [{}])[0] if isinstance(guess, dict) else {}
    raise BenchmarkError(
        "Only classification, span, Q&A and summarization benchmarks can be imported so far"
        + (f" (this looks like {top.get('task_profile') or top.get('profile')})." if top else ".")
    )


def _label_name(value: Any, names: list[str] | None) -> str:
    if names and (isinstance(value, int) or (isinstance(value, str) and value.lstrip("-").isdigit())):
        idx = int(value)
        if 0 <= idx < len(names):
            return names[idx]
    return str(value)


def bio_to_spans(tokens: list[str], tags: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """Join tokens with spaces; BIO tags → char spans (an I- after O or a
    different type starts a new span)."""
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for i, token in enumerate(tokens):
        if i:
            text_parts.append(_TOKEN_JOIN)
            pos += len(_TOKEN_JOIN)
        offsets.append((pos, pos + len(token)))
        text_parts.append(token)
        pos += len(token)
    text = "".join(text_parts)
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for (start, end), tag in zip(offsets, tags):
        m = re.match(r"^([BI])[-_](.+)$", tag or "")
        if not m:
            current = None
            continue
        kind, label = m.group(1), m.group(2)
        if kind == "I" and current is not None and current["type"] == label:
            current["end"] = end
            continue
        current = {"type": label, "start": start, "end": end}
        spans.append(current)
    for s in spans:
        s["text"] = text[s["start"]:s["end"]]
    return text, spans


def convert_rows(rows: list[dict[str, Any]], shape: dict[str, Any], label_names: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Source rows → canonical rows in SOURCE labels: ``{"text", "label"}``
    or ``{"text", "entities"}``. Rows that can't be read are skipped."""
    fields = shape["fields"]
    out: list[dict[str, Any]] = []
    for r in rows:
        if shape["format"] in ("qa_pair", "qa_context"):
            question, answer = answer_text(r.get(fields["question"])), answer_text(r.get(fields["answer"]))
            if not question or not answer:
                continue
            row = {"question": question.strip(), "answer": answer.strip()}
            context = answer_text(r.get(fields["context"])) if fields.get("context") else None
            if context:
                row["context"] = context.strip()
            out.append(row)
        elif shape["format"] == "document_summary":
            document, summary = answer_text(r.get(fields["document"])), answer_text(r.get(fields["summary"]))
            if document and summary:
                out.append({"question": document.strip(), "answer": summary.strip()})
        elif shape["format"] == "text_label":
            text = r.get(fields["text"])
            label = r.get(fields["label"])
            if not isinstance(text, str) or not text.strip() or label is None:
                continue
            out.append({"text": text, "label": _label_name(label, label_names.get(fields["label"]))})
        elif shape["format"] == "bio_tokens":
            tokens = [str(t) for t in (_as_list(r.get(fields["tokens"])) or [])]
            raw_tags = _as_list(r.get(fields["tags"])) or []
            if not tokens or len(tokens) != len(raw_tags):
                continue
            tags = [_label_name(t, label_names.get(fields["tags"])) for t in raw_tags]
            text, spans = bio_to_spans(tokens, tags)
            out.append({"text": text, "entities": spans})
        elif shape["format"] == "entity_strings":
            text = r.get(fields["text"])
            items = _parse_entity_list(r.get(fields["entities"]))
            if not isinstance(text, str) or not text.strip() or items is None:
                continue
            pairs: list[tuple[str, str]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                value = it.get(fields["entity_text"])
                kind = it.get(fields["entity_type"])
                kinds = _as_list(kind) or ([kind] if isinstance(kind, str) else [])
                if isinstance(value, str) and kinds:
                    pairs.append((value, str(kinds[0])))  # several types: the first
            out.append({"text": text, "entities": locate_entities(text, pairs)})
        else:  # char_spans
            text = r.get(fields["text"])
            if not isinstance(text, str) or not text.strip():
                continue
            spans = []
            for it in _as_list(r.get(fields["spans"])) or []:
                try:
                    start, end = int(it["start"]), int(it["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= start < end <= len(text):
                    spans.append({"type": str(it.get(fields["span_label"])), "start": start, "end": end, "text": text[start:end]})
            out.append({"text": text, "entities": spans})
    return out


def source_label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for r in rows:
        if "label" in r:
            counts[str(r["label"])] += 1
        for e in r.get("entities") or []:
            counts[str(e["type"])] += 1
    return dict(counts.most_common(MAX_SOURCE_LABELS))


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def suggest_mapping(source_labels: list[str], project_labels: list[str]) -> dict[str, str]:
    """Source label → the project label with the same normalised name, else
    ``ignore``. Never guesses beyond name equality."""
    by_norm = {_norm(p): p for p in project_labels}
    return {s: by_norm.get(_norm(s), IGNORE) for s in source_labels}


def fit_rows_to_project(rows: list[dict[str, Any]], project_profile: str | None) -> tuple[list[dict[str, Any]], bool]:
    """Generative rows → the project's row format. A Q&A context is kept
    for a project that answers from one (``rag_qa`` — its eval handler puts
    it in the prompt) and folded into the question otherwise. Returns
    (rows, folded)."""
    if project_profile == "rag_qa" or not any(r.get("context") for r in rows):
        return rows, False
    out = []
    for r in rows:
        r = dict(r)
        context = r.pop("context", None)
        if context:
            r["question"] = f"Context: {context}\nQuestion: {r['question']}"
        out.append(r)
    return out, True


def mapping_target(value: Any) -> str:
    """One mapping value → the label the benchmark rows carry: a project
    label, ``ignore``, or a group name (``"a|b"``, members sorted)."""
    if isinstance(value, (list, tuple)):
        members = sorted({str(v).strip() for v in value if str(v or "").strip() and str(v).strip() != IGNORE})
        if not members:
            return IGNORE
        return members[0] if len(members) == 1 else GROUP_SEP.join(members)
    return str(value or IGNORE).strip() or IGNORE


def group_members(mapping: dict[str, Any]) -> dict[str, str]:
    """Project label → its group's name, for every label in a group of two
    or more (a prediction of any member counts as the group)."""
    out: dict[str, str] = {}
    for value in (mapping or {}).values():
        target = mapping_target(value)
        if GROUP_SEP in target:
            for member in target.split(GROUP_SEP):
                out[member] = target
    return out


def apply_mapping(
    rows: list[dict[str, Any]],
    mapping: dict[str, Any],
    *,
    join_adjacent: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Canonical source-label rows → project-label rows. Unmapped labels are
    ignored. Classification: rows with an ignored label are dropped. Spans:
    ignored entities removed; ``join_adjacent`` joins same-target spans
    separated only by whitespace (e.g. FIRSTNAME + LASTNAME → one name).
    A value that is a list of project labels maps to their group
    (``mapping_target``)."""
    mapping = {k: mapping_target(v) for k, v in (mapping or {}).items()}
    out: list[dict[str, Any]] = []
    dropped = Counter()
    ignored_entities = Counter()
    kept_entities = Counter()
    unmapped: set[str] = set()
    for r in rows:
        if "label" in r:
            src = str(r["label"])
            if src not in mapping:
                unmapped.add(src)
            target = mapping.get(src, IGNORE)
            if target == IGNORE:
                dropped[src] += 1
                continue
            out.append({"text": r["text"], "label": target, "benchmark_source_label": src})
            continue
        text = r["text"]
        ents: list[dict[str, Any]] = []
        for e in sorted(r.get("entities") or [], key=lambda x: (x["start"], x["end"])):
            src = str(e["type"])
            if src not in mapping:
                unmapped.add(src)
            target = mapping.get(src, IGNORE)
            if target == IGNORE:
                ignored_entities[src] += 1
                continue
            if join_adjacent and ents and ents[-1]["type"] == target and text[ents[-1]["end"]:e["start"]].strip() == "" \
                    and e["start"] - ents[-1]["end"] <= 2:
                ents[-1]["end"] = e["end"]
                ents[-1]["text"] = text[ents[-1]["start"]:e["end"]]
                continue
            if ents and e["start"] < ents[-1]["end"]:
                continue  # overlapping after mapping — keep the first
            ents.append({"type": target, "start": e["start"], "end": e["end"], "text": text[e["start"]:e["end"]]})
        kept_entities.update(x["type"] for x in ents)
        out.append({"text": text, "entities": ents})
    summary = {
        "rows_in": len(rows),
        "rows_kept": len(out),
        "rows_dropped_by_label": dict(dropped),
        "entities_kept": dict(kept_entities),
        "entities_ignored": dict(ignored_entities),
        "unmapped_labels": sorted(unmapped),
    }
    return out, summary


def validate_mapping(mapping: Any, source_labels: list[str], project_labels: list[str]) -> dict[str, Any]:
    """``{source: project label | [project labels] | 'ignore'}`` → the clean
    mapping (a one-label list becomes the label; a group a sorted list).
    A project label may belong to at most one group."""
    if not isinstance(mapping, dict):
        raise BenchmarkError("mapping must be an object {source_label: project_label | [project_labels] | 'ignore'}")
    allowed = set(project_labels)
    clean: dict[str, Any] = {}
    for src, target in mapping.items():
        members = target if isinstance(target, (list, tuple)) else [target]
        members = [str(m or IGNORE).strip() or IGNORE for m in members]
        for m in members:
            if m != IGNORE and m not in allowed:
                raise BenchmarkError(f"'{m}' (for '{src}') is not one of this project's labels: {', '.join(sorted(allowed)) or 'none'}.")
        name = mapping_target(members)
        clean[str(src)] = name.split(GROUP_SEP) if GROUP_SEP in name else name
    groups = {frozenset(mapping_target(v).split(GROUP_SEP)) for v in clean.values() if mapping_target(v) != IGNORE}
    for label in sorted(allowed):
        owners = sorted(GROUP_SEP.join(sorted(g)) for g in groups if label in g)
        if len(owners) > 1:
            raise BenchmarkError(
                f"'{label}' is in more than one group ({'; '.join(o.replace(GROUP_SEP, ' + ') for o in owners)}) — "
                "a project label can belong to one group only."
            )
    if not any(mapping_target(t) != IGNORE for t in clean.values()):
        raise BenchmarkError("Map at least one benchmark label onto a project label — everything is ignored.")
    unknown = sorted(set(clean) - set(source_labels))
    if unknown and len(unknown) == len(clean):
        raise BenchmarkError("None of the mapped labels occur in the benchmark rows.")
    return clean


# ── Project side ─────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _project_dir(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id)


async def project_training_rows(db: AsyncSession, project_id: int) -> list[dict[str, Any]]:
    """Every row the project trains or validates on (prepared train + val,
    else the default training sources) — the leakage haystack."""
    prepared = _project_dir(project_id) / "prepared"
    rows = _read_jsonl(prepared / "train.jsonl") + _read_jsonl(prepared / "val.jsonl")
    if rows:
        return rows
    from app.services.dataset_service import load_training_corpus_rows

    return await load_training_corpus_rows(db, project_id)


async def project_label_space(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """The project's shape (from its data) and labels: class labels, or
    entity types (+ the defined ones)."""
    rows = await project_training_rows(db, project_id)
    test_rows = _read_jsonl(_project_dir(project_id) / "prepared" / "test.jsonl")
    span_types: set[str] = set()
    class_labels: set[str] = set()
    for r in rows + test_rows:
        ents = r.get("entities")
        if isinstance(ents, list):
            span_types.update(str(e.get("type")) for e in ents if isinstance(e, dict) and e.get("type"))
        elif isinstance(r.get("label"), (str, int)) and not isinstance(r.get("label"), bool):
            class_labels.add(str(r["label"]))
    if span_types or any(isinstance(r.get("entities"), list) for r in rows):
        from app.services.span_type_definitions_service import load_definitions

        span_types.update(await load_definitions(db, project_id))
        return {"task_profile": SHAPE_SPANS, "labels": sorted(span_types)}
    manifest_path = _project_dir(project_id) / "prepared" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        class_labels.update(str(x) for x in (manifest.get("labels") or []) if x is not None)
    except (OSError, json.JSONDecodeError):
        pass
    if class_labels:
        return {"task_profile": SHAPE_CLASSIFICATION, "labels": sorted(class_labels)}
    return {"task_profile": await _generative_project_profile(db, project_id), "labels": []}


async def _generative_project_profile(db: AsyncSession, project_id: int) -> str | None:
    """A project with neither class labels nor entities: its task profile
    when it is a generative one (the prepared manifest's, else what the
    task-shape service says its data is)."""
    from app.services.task_shape_service import canonical_task_profile, project_data_task_profile

    profile = None
    try:
        manifest = json.loads((_project_dir(project_id) / "prepared" / "manifest.json").read_text(encoding="utf-8"))
        raw = str(manifest.get("task_profile") or "").strip().lower()
        profile = raw if raw in GENERATIVE_PROJECT_PROFILES else canonical_task_profile(raw)
    except (OSError, json.JSONDecodeError, AttributeError):
        profile = None
    if not profile:
        try:
            profile, _source = await project_data_task_profile(db, project_id)
        except Exception:  # noqa: BLE001
            profile = None
    return profile if profile in GENERATIVE_PROJECT_PROFILES else None


def shapes_compatible(benchmark_profile: str | None, project_profile: str | None) -> bool:
    """Can a benchmark of this shape be scored on this project? Unknown
    project shape → yes; generative benchmarks fit any generative project."""
    if project_profile in (None, benchmark_profile):
        return True
    return benchmark_profile in GENERATIVE_SHAPES and project_profile in GENERATIVE_PROJECT_PROFILES


def _row_text(row: dict[str, Any]) -> str:
    if "text" in row:
        return str(row.get("text") or "")
    from app.services.trainability_forecast_service import _row_to_text

    return _row_to_text(row)


def leakage_filter(benchmark_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop benchmark rows identical / near-identical to a training row."""
    if not training_rows or not benchmark_rows:
        return benchmark_rows, {"checked_against": len(training_rows), "exact": 0, "near": 0, "examples": []}
    from app.services.data_health_service import _build_leakage_index, _match_row_against_index

    index = _build_leakage_index(training_rows)
    kept: list[dict[str, Any]] = []
    counts = Counter()
    examples: list[dict[str, Any]] = []
    for row in benchmark_rows:
        text = _row_text(row)
        kind, jaccard, _ = _match_row_against_index(text, index)
        if kind:
            counts[kind] += 1
            if len(examples) < 5:
                examples.append({"match": kind, "jaccard": round(jaccard, 3), "text": text[:160]})
            continue
        kept.append(row)
    return kept, {"checked_against": len(training_rows), "exact": counts.get("exact", 0),
                  "near": sum(v for k, v in counts.items() if k != "exact"), "examples": examples}


# ── Storage ──────────────────────────────────────────────────────────────


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60] or "benchmark"


def benchmark_dir(project_id: int, slug: str) -> Path:
    return _project_dir(project_id) / "benchmarks" / slug


def manifest_of(ds: Dataset) -> dict[str, Any]:
    meta = ds.metadata_ if isinstance(ds.metadata_, dict) else {}
    bench = meta.get("benchmark")
    return dict(bench) if isinstance(bench, dict) else {}


async def list_benchmarks(db: AsyncSession, project_id: int) -> list[Dataset]:
    rows = await db.execute(
        select(Dataset)
        .where(Dataset.project_id == project_id, Dataset.dataset_type == DatasetType.BENCHMARK)
        .order_by(Dataset.id)
    )
    return list(rows.scalars().all())


async def get_benchmark(db: AsyncSession, project_id: int, benchmark_id: int) -> Dataset:
    ds = await db.get(Dataset, benchmark_id)
    if ds is None or ds.project_id != project_id or ds.dataset_type != DatasetType.BENCHMARK:
        raise BenchmarkError(f"Benchmark {benchmark_id} not found in project {project_id}.")
    return ds


def read_results(project_id: int, slug: str) -> list[dict[str, Any]]:
    return _read_jsonl(benchmark_dir(project_id, slug) / "results.jsonl")


def serialize_benchmark(ds: Dataset, *, experiment_id: int | None = None) -> dict[str, Any]:
    m = manifest_of(ds)
    results = read_results(ds.project_id, m.get("slug") or "")
    for_run = [r for r in results if experiment_id is not None and r.get("experiment_id") == experiment_id]
    return {
        "id": ds.id,
        "name": m.get("name") or ds.name,
        "slug": m.get("slug"),
        "task_profile": m.get("task_profile"),
        "source": m.get("source"),
        "licence": m.get("licence"),
        "local_only": bool(m.get("local_only")),
        "rows": ds.record_count,
        "mapping": m.get("mapping"),
        "scored_labels": m.get("scored_labels"),
        "mapping_summary": m.get("mapping_summary"),
        "groups": m.get("groups") or {},
        "context_folded": bool(m.get("context_folded")),
        "leakage": m.get("leakage"),
        "contamination": m.get("contamination"),
        "created_at": m.get("created_at"),
        "result": for_run[-1] if for_run else None,
        "latest_result": results[-1] if results else None,
    }


def benchmark_history(ds: Dataset) -> dict[str, Any]:
    """Every score of the benchmark, oldest first: run id + name, date,
    headline score."""
    m = manifest_of(ds)
    results = read_results(ds.project_id, m.get("slug") or "")
    points = [
        {
            "experiment_id": r.get("experiment_id"),
            "experiment_name": r.get("experiment_name"),
            "evaluated_at": r.get("evaluated_at"),
            "headline": r.get("headline"),
            "score": r.get("score"),
            "rows": r.get("rows"),
        }
        for r in results
    ]
    points.sort(key=lambda p: str(p.get("evaluated_at") or ""))
    return {"benchmark_id": ds.id, "name": m.get("name") or ds.name, "slug": m.get("slug"), "history": points}


async def preview_benchmark(db: AsyncSession, project_id: int, locator: str, *, limit: int = DEFAULT_ROWS) -> dict[str, Any]:
    check_locator(project_id, locator)
    raw, info = load_source(locator, scan_cap=PREVIEW_SCAN_CAP)
    shape = detect_shape(raw, info["label_names"])
    rows = convert_rows(sample_rows(raw, min(limit, MAX_ROWS)), shape, info["label_names"])
    project = await project_label_space(db, project_id)
    counts = source_label_counts(rows)
    return {
        "source": {k: info[k] for k in ("locator", "source_id", "dataset_id", "split", "revision")},
        "licence": info["licence"],
        "task_profile": shape["task_profile"],
        "format": shape["format"],
        "fields": shape["fields"],
        "project": project,
        "shape_matches_project": shapes_compatible(shape["task_profile"], project["task_profile"]),
        # Generative shapes (Q&A, summarization) have no labels to map.
        "labels_to_map": shape["task_profile"] not in GENERATIVE_SHAPES,
        "source_labels": counts,
        "suggested_mapping": suggest_mapping(list(counts), project["labels"]),
        "sample": rows[:5],
        "rows_scanned": len(raw),
    }


async def create_benchmark(
    db: AsyncSession,
    project_id: int,
    *,
    locator: str,
    name: str,
    mapping: dict[str, Any] | None = None,
    limit: int = DEFAULT_ROWS,
    local_only: bool = False,
    licence: str | None = None,
    join_adjacent: bool = False,
) -> Dataset:
    limit = max(1, min(int(limit or DEFAULT_ROWS), MAX_ROWS))
    check_locator(project_id, locator)
    raw, info = load_source(locator, scan_cap=SCAN_CAP)
    shape = detect_shape(raw, info["label_names"])
    project = await project_label_space(db, project_id)
    if not shapes_compatible(shape["task_profile"], project["task_profile"]):
        raise BenchmarkError(
            f"This benchmark is {shape['task_profile']} but the project's data is {project['task_profile']}."
        )
    rows = convert_rows(sample_rows(raw, limit), shape, info["label_names"])
    context_folded = False
    if shape["task_profile"] in GENERATIVE_SHAPES:
        # No labels to map: the rows are scored on their answers.
        clean_mapping: dict[str, Any] = {}
        mapped, context_folded = fit_rows_to_project(rows, project["task_profile"])
        mapping_summary = {"rows_in": len(rows), "rows_kept": len(mapped), "rows_dropped_by_label": {},
                           "entities_kept": {}, "entities_ignored": {}, "unmapped_labels": []}
    else:
        clean_mapping = validate_mapping(mapping or {}, list(source_label_counts(rows)), project["labels"])
        mapped, mapping_summary = apply_mapping(rows, clean_mapping, join_adjacent=join_adjacent)
    training_rows = await project_training_rows(db, project_id)
    kept, leakage = leakage_filter(mapped, training_rows)
    if not kept:
        raise BenchmarkError("No benchmark rows left after mapping and the leakage check.")
    from app.services.benchmark_contamination_service import benchmark_contamination

    contamination = benchmark_contamination(kept, shape["task_profile"], training_rows)
    slug = _slugify(name)
    existing = {manifest_of(d).get("slug") for d in await list_benchmarks(db, project_id)}
    base, n = slug, 2
    while slug in existing:
        slug, n = f"{base}-{n}", n + 1
    folder = benchmark_dir(project_id, slug)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "rows.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    manifest = {
        "name": name.strip() or slug,
        "slug": slug,
        "task_profile": shape["task_profile"],
        "format": shape["format"],
        "fields": shape["fields"],
        "source": {k: info[k] for k in ("locator", "source_id", "dataset_id", "split", "revision")},
        "licence": (licence or "").strip() or info["licence"],
        "local_only": bool(local_only),
        "mapping": clean_mapping,
        "join_adjacent": bool(join_adjacent),
        "scored_labels": sorted({mapping_target(t) for t in clean_mapping.values() if mapping_target(t) != IGNORE}),
        "groups": {name: name.split(GROUP_SEP) for name in sorted(set(group_members(clean_mapping).values()))},
        "context_folded": context_folded,
        "project_task_profile": project["task_profile"],
        "mapping_summary": mapping_summary,
        "leakage": leakage,
        "contamination": contamination,
        "rows_scanned": len(raw),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    ds = Dataset(
        project_id=project_id,
        name=f"benchmark:{slug}",
        dataset_type=DatasetType.BENCHMARK,
        description=f"External benchmark from {info['locator']} (eval only)",
        record_count=len(kept),
        file_path=str(path),
        metadata_={"benchmark": manifest},
        is_locked=True,
    )
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    return ds


async def rescan_contamination(db: AsyncSession, project_id: int, benchmark_id: int) -> Dataset:
    """Re-run the gold-value contamination scan against the CURRENT training
    rows (they change after import: seeding, synthesis, re-splits) and store
    the result on the manifest (disk + DB)."""
    from sqlalchemy.orm.attributes import flag_modified

    from app.services.benchmark_contamination_service import benchmark_contamination

    ds = await get_benchmark(db, project_id, benchmark_id)
    manifest = manifest_of(ds)
    slug = manifest.get("slug") or ""
    rows = _read_jsonl(benchmark_dir(project_id, slug) / "rows.jsonl")
    manifest["contamination"] = benchmark_contamination(
        rows, manifest.get("task_profile"), await project_training_rows(db, project_id)
    )
    folder = benchmark_dir(project_id, slug)
    if folder.exists():
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = dict(ds.metadata_) if isinstance(ds.metadata_, dict) else {}
    meta["benchmark"] = manifest
    ds.metadata_ = meta
    flag_modified(ds, "metadata_")
    await db.commit()
    await db.refresh(ds)
    return ds


async def delete_benchmark(db: AsyncSession, project_id: int, benchmark_id: int) -> None:
    import shutil

    ds = await get_benchmark(db, project_id, benchmark_id)
    slug = manifest_of(ds).get("slug")
    await db.delete(ds)
    await db.commit()
    if slug:
        shutil.rmtree(benchmark_dir(project_id, slug), ignore_errors=True)


# ── Scoring ──────────────────────────────────────────────────────────────


def _micro(per_class: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    tp = fp = fn = 0
    per_type = []
    for t in labels:
        row = per_class.get(t) if isinstance(per_class, dict) else None
        if not isinstance(row, dict):
            continue
        a, b, c = int(row.get("tp") or 0), int(row.get("fp") or 0), int(row.get("fn") or 0)
        tp, fp, fn = tp + a, fp + b, fn + c
        p = a / (a + b) if a + b else None
        r = a / (a + c) if a + c else None
        f = 2 * p * r / (p + r) if p is not None and r is not None and p + r else (0.0 if (a + b + c) else None)
        per_type.append({"type": t, "precision": p, "recall": r, "f1": f, "support": a + c})
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "per_type": per_type}


def score_block(task_profile: str, metrics: dict[str, Any], scored_labels: list[str]) -> dict[str, Any]:
    """The benchmark's headline from the handler's metrics. Spans: micro
    P / R / F1 over the mapped types only. Classification: accuracy +
    macro-F1 (+ per class)."""
    if task_profile == SHAPE_SPANS:
        block = _micro(metrics.get("per_class") or {}, scored_labels)
        return {"headline": "f1", "score": block["f1"], **block}
    if task_profile in GENERATIVE_SHAPES:
        judge = metrics.get("judge") if isinstance(metrics.get("judge"), dict) else None
        judged = judge.get("score") if judge else None
        if judged is not None:
            headline = "judge_correct"
        elif metrics.get("f1") is not None:
            headline = "f1"
        else:
            headline = "exact_match"
        score = judged if headline == "judge_correct" else metrics.get(headline)
        return {
            "headline": headline,
            "score": score,
            "exact_match": metrics.get("exact_match"),
            "f1": metrics.get("f1"),
            "judge_correct": judged,
            "judge": judge,
            **({k: metrics[k] for k in ("rougeL", "rouge_l", "rouge1") if metrics.get(k) is not None}),
        }
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
    return {
        "headline": "accuracy",
        "score": metrics.get("accuracy"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "per_class": {k: v for k, v in per_class.items() if k in scored_labels},
    }


def _is_local_model(model_ref: str | None) -> bool:
    return bool(model_ref) and Path(str(model_ref)).expanduser().exists()


def collapse_predictions(
    task_profile: str,
    predictions: list[dict[str, Any]],
    members: dict[str, str],
    handler: Any,
    candidates: list[str] | None = None,
) -> list[str] | None:
    """Collapse each prediction of a grouped project label to its group, in
    place, before the handler scores. Classification: the label the handler
    would parse → the group name (returns the collapsed candidate set the
    handler should score with). Spans: every entity of a member type → the
    group type. The original prediction is kept as
    ``prediction_before_grouping``."""
    if not members:
        return None
    if task_profile == SHAPE_CLASSIFICATION:
        candidates = list(candidates or [])
        parse = getattr(handler, "parse_predicted_label", None)
        for p in predictions:
            raw = str(p.get("prediction") or "")
            parsed = parse(raw, candidates) if callable(parse) else raw.strip()
            if parsed in members:
                p["prediction_before_grouping"] = raw
                p["prediction"] = members[parsed]
        groups = sorted(set(members.values()))
        return [c for c in candidates if c not in members] + groups
    if task_profile == SHAPE_SPANS:
        parse_json = getattr(handler, "_parse_json_safely", None)
        for p in predictions:
            raw = p.get("prediction")
            parsed = parse_json(raw) if callable(parse_json) else None
            if parsed is None and isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except (TypeError, ValueError):
                    parsed = None
            if not isinstance(parsed, dict) or not isinstance(parsed.get("entities"), list):
                continue
            changed = False
            for e in parsed["entities"]:
                if isinstance(e, dict) and str(e.get("type") or "") in members:
                    e["type"] = members[str(e["type"])]
                    changed = True
            if changed:
                p["prediction_before_grouping"] = raw
                p["prediction"] = json.dumps(parsed, ensure_ascii=False)
    return None


async def judge_benchmark_answers(
    db: AsyncSession,
    project_id: int,
    *,
    task_profile: str | None,
    predictions: list[dict[str, Any]],
    local_only: bool,
) -> dict[str, Any]:
    """The project's answer judge over a generative benchmark — the held-out
    eval's rule (long-answer generative task, a judge reachable). A
    ``local_only`` benchmark never reaches a cloud judge: the snapshot says
    ``skipped: "local_only_cloud_judge"`` and names the judge."""
    from app.models.project import Project
    from app.services.answer_judge_service import resolve_answer_judge, should_judge
    from app.services.evaluation_service import _safe_judge_answers

    references = [str(p.get("reference") or "") for p in predictions if isinstance(p, dict)]
    if not should_judge(task_profile, references):
        return {"skipped": "not_long_answer_task"}
    try:
        resolved = await resolve_answer_judge(db, project_id, await db.get(Project, project_id))
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"error: {exc}"[:200]}
    if resolved is None:
        return {"skipped": "no_judge_available"}
    if local_only and not getattr(resolved, "local", True):
        return {
            "skipped": "local_only_cloud_judge",
            "judge": resolved.label,
            "message": f"Not judged: this benchmark is local-only and the project's judge ({resolved.label}) is a cloud model.",
        }
    return await _safe_judge_answers(
        db, project_id=project_id, task_profile=task_profile, predictions=predictions, judge=resolved,
    )


async def evaluate_benchmark(
    db: AsyncSession,
    project_id: int,
    benchmark_id: int,
    experiment_id: int,
    *,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Score one run on one benchmark (in memory) and append the result."""
    from app.services.eval_task_handler_service import build_eval_context
    from app.services.evaluation_service import (
        _generate_heldout_predictions,
        _get_experiment_for_project,
        _load_heldout_pairs,
        _prediction_failed,
        _resolve_model_reference,
    )
    from app.services.slice_evaluator_service import score_with_slices

    ds = await get_benchmark(db, project_id, benchmark_id)
    m = manifest_of(ds)
    exp = await _get_experiment_for_project(db, project_id, experiment_id)
    if exp is None:
        raise BenchmarkError(f"Run #{experiment_id} not found in project {project_id}.")
    from app.models.experiment import ExperimentStatus

    if exp.status != ExperimentStatus.COMPLETED:
        raise BenchmarkError(f"Run #{experiment_id} is {exp.status.value.lower()} — only completed runs can be scored.")
    if m.get("local_only") and not _is_local_model(_resolve_model_reference(exp, None)):
        raise BenchmarkError("This benchmark is local-only and run #%d's model is not a checkpoint on this machine." % experiment_id)
    exp_task_type = str((exp.config or {}).get("task_type") or "").strip() or None
    eval_ctx, handler = build_eval_context(
        project_id=project_id, experiment_id=experiment_id, eval_type="exact_match",
        dataset_name="test", experiment_task_type=exp_task_type,
    )
    task_profile = m.get("task_profile") or ""
    members = group_members(m.get("mapping") or {})
    if members and task_profile == SHAPE_CLASSIFICATION and hasattr(handler, "_cached_candidates") \
            and not (eval_ctx.manifest or {}).get("labels"):
        # The prompt lists the project's labels, never the group names the
        # benchmark rows carry.
        handler._cached_candidates = (await project_label_space(db, project_id))["labels"]
    pairs = _load_heldout_pairs(Path(ds.file_path), max_samples=ds.record_count or MAX_ROWS, handler=handler, ctx=eval_ctx)
    if not pairs:
        raise BenchmarkError("The benchmark rows produced no evaluation pairs for this run's task.")
    predictions, runtime, _modalities, _settings = await _generate_heldout_predictions(
        db, exp=exp, project_id=project_id, pairs=pairs, task_handler=handler, eval_ctx=eval_ctx,
        model_path=None, max_new_tokens=max_new_tokens, temperature=0.0,
    )
    if members:
        current = getattr(handler, "_cached_candidates", None)
        collapsed = collapse_predictions(task_profile, predictions, members, handler, current)
        if collapsed is not None and hasattr(handler, "_cached_candidates"):
            handler._cached_candidates = collapsed
    metrics = dict(score_with_slices(handler, predictions, eval_ctx, slice_definitions=None))
    if task_profile in GENERATIVE_SHAPES:
        from app.services.evaluation_service import exact_match as _em, f1_score as _f1

        # Handlers without per-answer F1 / EM (e.g. the generic one):
        # the same token F1 / exact match the held-out eval uses.
        if metrics.get("f1") is None:
            for p in predictions:
                p.setdefault("row_f1", _f1(str(p.get("prediction") or ""), str(p.get("reference") or "")))
            metrics["f1"] = round(sum(float(p["row_f1"]) for p in predictions) / len(predictions), 4) if predictions else 0.0
        if metrics.get("exact_match") is None:
            ems = [_em(str(p.get("prediction") or ""), str(p.get("reference") or "")) for p in predictions]
            metrics["exact_match"] = round(sum(ems) / len(ems), 4) if ems else 0.0
        metrics["judge"] = await judge_benchmark_answers(
            db, project_id, task_profile=eval_ctx.task_profile or task_profile,
            predictions=predictions, local_only=bool(m.get("local_only")),
        )
    block = score_block(task_profile, metrics, list(m.get("scored_labels") or []))
    result = {
        "experiment_id": experiment_id,
        "experiment_name": exp.name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(predictions),
        **block,
        "failures": [
            {"prompt": str(p.get("prompt", ""))[:300], "reference": str(p.get("reference", ""))[:300],
             "prediction": str(p.get("prediction_before_grouping") or p.get("prediction", ""))[:300],
             **({"judge_verdict": p.get("row_judge_verdict"), "judge_reason": p.get("row_judge_reason")}
                if p.get("row_judge_verdict") else {})}
            for p in predictions if _prediction_failed(p)
        ][:FAILURES_PREVIEW],
        "inference_device": (runtime or {}).get("device") if isinstance(runtime, dict) else None,
    }
    folder = benchmark_dir(project_id, m.get("slug") or "")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    return {"benchmark_id": ds.id, "benchmark": m.get("name"), **result}


async def benchmarks_for_run(db: AsyncSession, project_id: int, experiment_id: int | None) -> list[dict[str, Any]]:
    """Every benchmark of the project with this run's latest score (or None)."""
    return [serialize_benchmark(ds, experiment_id=experiment_id) for ds in await list_benchmarks(db, project_id)]

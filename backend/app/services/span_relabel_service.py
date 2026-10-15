"""Answer key from a second model: relabel a split's texts without changing
them, and keep the result as the project's GOLD_TEST set.

A span dataset whose labels came from one teacher is scored, by default,
against that teacher's own labels. A second model labelling the SAME
texts gives an answer key the first one didn't write — and the agreement
between the two (strict span F1, old labels vs new) says how much of the
"error" was ever the student's. General to any span-extraction project.
"""

from __future__ import annotations

import json
import re
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType
from app.services.synth_backends.base import SynthBackendError

_LOG = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], Awaitable[None]]

# Texts per teacher call: enough to amortise the prompt, few enough that
# the model labels every one.
TEXTS_PER_CALL = 6
# A reply cut mid-JSON (the budget) parses to nothing; the same call again
# fails the same way, so a failed batch is split in two and each half
# retried, down to single texts.
MIN_TEXTS_PER_CALL = 1
# Tokens per text in the reply: compact JSON, entity type + text only.
TOKENS_PER_TEXT = 700
MAX_ROWS = 2000

_SYSTEM_PROMPT = (
    "You label entities in texts for a span-detection model. Copy entity text EXACTLY as it "
    "appears; never change, shorten or paraphrase the texts. Reply with JSON only."
)

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["type", "text"],
                        },
                    },
                },
                "required": ["index", "entities"],
            },
        }
    },
    "required": ["rows"],
}


def relabel_prompt_parts(texts: list[str], entity_types: list[str], conventions: str = "") -> tuple[str, str]:
    """(stable, varying): the instructions, types and conventions — the same
    for every call of a job, sent as a cached block where the provider
    supports it — and the numbered texts of this call."""
    types = ", ".join(entity_types) if entity_types else "every kind of sensitive or identifying data"
    stable = (
        "Label every entity in each text you are given. Entity types (use ONLY these, lower-case): "
        f"{types}.\n\n"
        f"{conventions + chr(10) if conventions else ''}"
        "Rules:\n"
        "- entity.text is the EXACT characters as they appear in the text (copy them). Do NOT give "
        "offsets — they are located from the text.\n"
        "- Label every occurrence; a person's name that appears twice is two entities.\n"
        "- Do not invent entities that are not in the text, and do not change the texts.\n\n"
        'Return compact JSON (no indentation) {"rows": [{"index": <text number>, "entities": '
        '[{"type": "...", "text": "..."}, ...]}, ...]} with one row per text, in order.'
    )
    varying = "Texts:\n\n" + "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    return stable, varying


def build_relabel_prompt(texts: list[str], entity_types: list[str], conventions: str = "") -> str:
    """``conventions``: the project's labelling conventions block
    (``span_type_definitions_service.definitions_clause``) — the same one
    the training-label generator sends. One string (backends without a
    prompt cache)."""
    stable, varying = relabel_prompt_parts(texts, entity_types, conventions)
    return stable + "\n\n" + varying


def _parse_rows(raw: str) -> dict[int, list[dict[str, Any]]]:
    from app.services.synthetic_service import _extract_json_blocks, _strip_thinking_blocks

    out: dict[int, list[dict[str, Any]]] = {}
    for candidate in _extract_json_blocks(_strip_thinking_blocks(raw or "")):
        try:
            payload = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            ents = item.get("entities")
            out[index] = [e for e in ents if isinstance(e, dict)] if isinstance(ents, list) else []
        if out:
            break
    return out


def load_split_rows(project_id: int, split: str) -> list[dict[str, Any]]:
    """Rows of a prepared split (``train`` / ``val`` / ``test``) or the
    gold sets (``gold_dev`` / ``gold_test``)."""
    base = settings.DATA_DIR / "projects" / str(project_id)
    path = base / "gold" / f"{split}.jsonl" if split.startswith("gold_") else base / "prepared" / f"{split}.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _row_text(row: dict[str, Any]) -> str:
    for key in ("text", "input", "source_text", "content"):
        value = row.get(key)
        if isinstance(value, dict):
            value = value.get("text") or value.get("input")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _row_entities(row: dict[str, Any]) -> list[dict[str, Any]]:
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from train_token_classification import row_to_example

    example = row_to_example(row)
    return example[1] if example else []


def agreement(old: list[list[dict[str, Any]]], new: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Strict span F1 of the OLD labels scored against the NEW key — how
    much the two labellers agree, overall and per type."""
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from train_token_classification import span_prf

    return span_prf(new, old)


@dataclass
class RelabelSummary:
    project_id: int
    split: str
    backend: str
    rows: int
    labelled_rows: int
    failed_calls: int
    entities: int
    entity_counts: dict[str, int] = field(default_factory=dict)
    agreement: dict[str, Any] = field(default_factory=dict)
    # Types whose project definition went into the prompt.
    conventions_types: list[str] = field(default_factory=list)
    dataset_id: int | None = None
    file_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    # Token use + list-price estimate per model (``llm_pricing``), and the
    # tiered pass / targeted-relabel reports when they ran.
    usage: list[dict[str, Any]] = field(default_factory=list)
    review: dict[str, Any] | None = None
    targeting: dict[str, Any] | None = None
    # Message Batch report (``batch_info``) when the labels came through one.
    batch: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "split": self.split,
            "backend": self.backend,
            "rows": self.rows,
            "labelled_rows": self.labelled_rows,
            "failed_calls": self.failed_calls,
            "entities": self.entities,
            "entity_counts": dict(self.entity_counts),
            "agreement": dict(self.agreement),
            "conventions_types": list(self.conventions_types),
            "dataset_id": self.dataset_id,
            "file_path": self.file_path,
            "warnings": list(self.warnings),
            "usage": list(self.usage),
            "cost_usd": round(sum(u.get("usd") or 0 for u in self.usage), 4) if self.usage else None,
            "tokens": sum(u.get("total_tokens") or 0 for u in self.usage) if self.usage else None,
            "review": self.review,
            "targeting": self.targeting,
            "batch": self.batch,
            "via_batch": bool(self.batch),
        }


def usage_block(*llms: Any) -> list[dict[str, Any]]:
    """Per-backend token use + cost estimate (backends that count usage)."""
    from app.services.llm_pricing import estimate_cost

    out = []
    for llm in llms:
        label = llm.describe() if hasattr(llm, "describe") else str(llm)
        model = getattr(llm, "model", None) or (llm.describe() if hasattr(llm, "describe") else "")
        batch_usage = getattr(llm, "batch_usage", None)
        if isinstance(batch_usage, dict) and batch_usage.get("calls"):
            out.append({"backend": f"{label} (batch)", **estimate_cost(model, batch_usage, batch=True)})
        usage = getattr(llm, "usage", None)
        if isinstance(usage, dict) and usage.get("calls"):
            out.append({"backend": label, **estimate_cost(model, usage)})
    return out


async def latest_span_run_predictions(
    db: AsyncSession, project_id: int, texts: list[str], experiment_id: int | None = None,
) -> tuple[int | None, list[list[dict[str, Any]]] | None]:
    """The project's newest completed token-classification run (or
    ``experiment_id``) tagging ``texts`` — (run id, spans per text); (None,
    None) when the project has no such run."""
    import asyncio

    from sqlalchemy import desc

    from app.models.experiment import Experiment, ExperimentStatus
    from app.services.evaluation_service import _resolve_token_classifier_artifacts
    from app.services.post_training_eval_service import is_token_classification
    from app.services.span_threshold_service import _score_split
    from app.services.span_type_definitions_service import load_scan_tolerant

    if experiment_id is not None:
        candidates = [await db.get(Experiment, experiment_id)]
    else:
        candidates = list((await db.execute(
            select(Experiment).where(Experiment.project_id == project_id, Experiment.status == ExperimentStatus.COMPLETED)
            .order_by(desc(Experiment.id))
        )).scalars().all())
    for exp in candidates:
        if exp is None or exp.project_id != project_id or not is_token_classification(exp) or not exp.output_dir:
            continue
        artifacts = _resolve_token_classifier_artifacts(exp.output_dir)
        if artifacts is None:
            continue
        spans = await asyncio.to_thread(
            _score_split, artifacts["model_path"], artifacts["id2label"], texts, await load_scan_tolerant(db, project_id),
        )
        return exp.id, spans
    return None, None


def _span_key(spans: list[dict[str, Any]] | None) -> set[tuple[str, int, int]]:
    return {(e["type"], int(e["start"]), int(e["end"])) for e in spans or [] if isinstance(e, dict)}


async def review_disagreements(
    db: AsyncSession,
    project_id: int,
    texts: list[str],
    labels: list[list[dict[str, Any]] | None],
    types: list[str],
    conventions: str,
    review_llm: Any,
    report: ProgressFn,
    *,
    experiment_id: int | None = None,
) -> tuple[list[list[dict[str, Any]] | None], dict[str, Any]]:
    """Tiered labelling: rows where the bulk labeller and the project's
    trained model disagree (or the bulk labeller returned nothing) go to
    ``review_llm``; its labels replace the bulk ones there. Without a
    trained span model nothing is reviewed (reported)."""
    run_id, preds = await latest_span_run_predictions(db, project_id, texts, experiment_id)
    if preds is None:
        return labels, {"skipped": "no trained span model to compare with", "rows": len(texts), "disagreed": 0, "reviewed": 0}
    disagree = [i for i, (lab, pred) in enumerate(zip(labels, preds)) if lab is None or _span_key(lab) != _span_key(pred)]
    review_label = review_llm.describe() if hasattr(review_llm, "describe") else "review model"
    reviewed = 0
    if disagree:
        new, _calls = await label_texts_any(review_llm, [texts[i] for i in disagree], types, conventions, report, review_label)
        out = list(labels)
        for k, i in enumerate(disagree):
            if new[k] is not None:
                out[i] = new[k]
                reviewed += 1
        labels = out
    return labels, {
        "review_backend": review_label, "model_experiment_id": run_id,
        "rows": len(texts), "disagreed": len(disagree), "reviewed": reviewed,
    }


async def label_texts(
    llm: Any,
    texts: list[str],
    types: list[str],
    conventions: str,
    report: ProgressFn,
    backend_label: str,
    *,
    start: float = 0.05,
    span: float = 0.85,
    texts_per_call: int = TEXTS_PER_CALL,
) -> tuple[list[list[dict[str, Any]] | None], dict[str, int]]:
    """Label ``texts`` (unchanged) with ``llm``: ``TEXTS_PER_CALL`` per call,
    compact type + text replies, offsets located from the text, a failed or
    unparseable batch split down to single texts. Returns one entity list
    per text (None where even a single-text call failed) and call counts."""
    from app.services.synthetic_service import _validate_span_rows

    labels: list[list[dict[str, Any]] | None] = [None] * len(texts)
    calls = {"made": 0, "failed": 0, "batches": 0}
    order = list(range(len(texts)))
    per_call = max(1, int(texts_per_call))
    batches = [order[i:i + per_call] for i in range(0, len(order), per_call)]
    calls["batches"] = len(batches)

    def call_args(batch: list[int]) -> dict[str, Any]:
        stable, varying = relabel_prompt_parts([texts[i] for i in batch], types, conventions)
        cached = getattr(llm, "supports_prompt_cache", False)
        return {
            "prompt": varying if cached else stable + "\n\n" + varying,
            "system_prompt": _SYSTEM_PROMPT,
            **({"cache_prefix": stable} if cached else {}),
            # A long text carries more entities: budget the reply by length.
            "max_tokens": max(2000, sum(max(TOKENS_PER_TEXT, len(texts[i]) // 3) for i in batch)),
            "temperature": 0.0,
        }

    async def label_batch(batch: list[int], raw_from_batch: str | None = None) -> None:
        parsed: dict[int, list[dict[str, Any]]] = {}
        if raw_from_batch is not None:
            # A batch result came back but may not parse (a reply cut by
            # the budget): it would fail the same way again, so split.
            parsed = _parse_rows(raw_from_batch)
            if not parsed:
                _LOG.warning("relabel: batch result for %d texts, no rows parsed (starts: %r)", len(batch), raw_from_batch[:160])
        else:
            calls["made"] += 1
            try:
                args = call_args(batch)
                prompt = args.pop("prompt")
                raw = await llm.complete(prompt, **args, response_schema=_RESPONSE_SCHEMA)
                parsed = _parse_rows(str(raw or ""))
                if not parsed:
                    _LOG.warning("relabel: %d texts, no rows parsed (content starts: %r)", len(batch), str(raw)[:160])
            except SynthBackendError as exc:
                _LOG.warning("relabel: %d texts failed: %s", len(batch), exc)
        if not parsed:
            if len(batch) > MIN_TEXTS_PER_CALL:
                half = len(batch) // 2
                await label_batch(batch[:half])
                await label_batch(batch[half:])
            else:
                calls["failed"] += 1
            return
        for local_index, index in enumerate(batch):
            ents = parsed.get(local_index)
            if ents is None:
                continue
            validated = _validate_span_rows([{"text": texts[index], "entities": ents}], types)
            labels[index] = validated[0]["entities"] if validated else []

    if batches and uses_message_batch(llm):
        # One Message Batch for every call (half price, slower); a request
        # that errored / expired falls back to the synchronous path below.
        async def batch_progress(message: str, done: int, total: int) -> None:
            await report(start + span * 0.9 * done / max(1, total), f"Labelling {len(texts)} texts ({backend_label}): {message}")

        try:
            raws = await llm.complete_batch([call_args(b) for b in batches], progress=batch_progress)
        except SynthBackendError as exc:
            _LOG.warning("relabel: message batch failed, labelling synchronously: %s", exc)
            raws = [None] * len(batches)
        calls["batched"] = len(batches)
        calls["batch_fallback"] = sum(1 for r in raws if r is None)
        for b, (batch, raw) in enumerate(zip(batches, raws)):
            if raw is None:
                await report(start + span * (0.9 + 0.1 * b / len(batches)), f"Batch request failed — labelling texts {batch[0] + 1}–{batch[-1] + 1} directly ({backend_label})")
            await label_batch(batch, raw)
        return labels, calls

    for b, batch in enumerate(batches):
        await report(start + span * b / max(1, len(batches)), f"Labelling texts {batch[0] + 1}–{batch[-1] + 1} of {len(texts)} ({backend_label})")
        await label_batch(batch)
    return labels, calls


def uses_message_batch(llm: Any) -> bool:
    """The caller asked for a Message Batch and the backend can do one."""
    return bool(getattr(llm, "use_batch", False)) and bool(getattr(llm, "supports_batch", False)) and hasattr(llm, "complete_batch")


def batch_info(llm: Any) -> dict[str, Any] | None:
    """What went through Message Batches in this job (None when nothing did)."""
    report = getattr(llm, "batch_report", None)
    if not uses_message_batch(llm) or not report:
        return None
    return {
        "batches": list(report),
        "requests": report[0]["requests"],
        "succeeded": sum(int(r.get("succeeded") or 0) for r in report),
        "fell_back": report[-1].get("failed", 0),
    }


async def relabel_split_as_answer_key(
    db: AsyncSession,
    project_id: int,
    *,
    split: str = "test",
    backend: str | None = None,
    entity_types: list[str] | None = None,
    progress: ProgressFn | None = None,
    backend_override: Any = None,
    max_rows: int = MAX_ROWS,
) -> RelabelSummary:
    """Label the texts of ``split`` with ``backend`` (a synth backend name
    or cloud pin; ``backend_override`` injects an instance) and write them
    as the project's GOLD_TEST set — ``gold/gold_test.jsonl`` plus the
    Dataset record the evaluation's ``gold_test`` alias resolves. Entity
    types default to the ones the split already uses. Raises when no row
    could be labelled."""
    from app.services.synth_backends.base import pick_backend

    async def report(fraction: float, message: str) -> None:
        if progress is not None:
            await progress(fraction, message)

    source_rows = load_split_rows(project_id, split)[:max_rows]
    texts = [_row_text(r) for r in source_rows]
    keep = [i for i, t in enumerate(texts) if t.strip()]
    if not keep:
        raise RuntimeError(f"No rows with text in the {split} split of project {project_id}.")
    old_labels = [_row_entities(source_rows[i]) for i in keep]
    types = [t.strip().lower() for t in (entity_types or []) if isinstance(t, str) and t.strip()]
    if not types:
        types = sorted({e["type"] for spans in old_labels for e in spans})
    from app.services.span_type_definitions_service import definitions_clause, load_definitions, load_scan_tolerant

    definitions = await load_definitions(db, project_id)
    conventions = definitions_clause(types, definitions, scan_tolerant=await load_scan_tolerant(db, project_id))

    llm = backend_override or pick_backend(backend)
    backend_label = llm.describe() if hasattr(llm, "describe") else getattr(llm, "name", "backend")
    keep_texts = [texts[i] for i in keep]
    new_labels, calls = await label_texts_any(llm, keep_texts, types, conventions, report, backend_label)
    failed_calls = calls["failed"]

    labelled = [(keep[j], spans) for j, spans in enumerate(new_labels) if spans is not None]
    if not labelled:
        raise RuntimeError(
            f"{backend_label} labelled none of the {len(keep)} texts — pick another model or check it is reachable."
        )

    await report(0.92, f"Writing the answer key ({len(labelled)} rows)")
    gold_dir = settings.DATA_DIR / "projects" / str(project_id) / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    path = gold_dir / "gold_test.jsonl"
    stamp = datetime.now(timezone.utc).isoformat()
    counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for source_index, spans in labelled:
            counts.update(e["type"] for e in spans)
            handle.write(json.dumps({
                "text": texts[source_index],
                "entities": spans,
                "source": f"answer_key:{backend_label}",
                "relabelled_from": split,
                "created_at": stamp,
            }, ensure_ascii=False) + "\n")

    existing = (
        await db.execute(
            select(Dataset).where(Dataset.project_id == project_id, Dataset.dataset_type == DatasetType.GOLD_TEST)
            .order_by(Dataset.id.desc()).limit(1)
        )
    ).scalars().first()
    if existing is None:
        existing = Dataset(project_id=project_id, name="Answer key (test)", dataset_type=DatasetType.GOLD_TEST, description="")
        db.add(existing)
    existing.file_path = str(path)
    existing.record_count = len(labelled)
    existing.description = f"Relabelled {split} split by {backend_label} on {stamp[:10]}"
    existing.metadata_ = {**(existing.metadata_ or {}), "source": "span_relabel", "backend": backend_label, "relabelled_from": split}
    await db.commit()
    await db.refresh(existing)

    old_for_agreement = [old_labels[keep.index(i)] for i, _ in labelled]
    new_for_agreement = [spans for _, spans in labelled]
    summary = RelabelSummary(
        project_id=project_id, split=split, backend=backend_label, rows=len(keep), labelled_rows=len(labelled),
        failed_calls=failed_calls, entities=sum(counts.values()), entity_counts=dict(counts.most_common()),
        agreement=agreement(old_for_agreement, new_for_agreement), dataset_id=int(existing.id), file_path=str(path),
        conventions_types=[t for t in types if t in definitions],
    )
    if failed_calls:
        summary.warnings.append(f"{failed_calls} text{'s' if failed_calls != 1 else ''} could not be labelled even alone and were skipped ({calls['made']} calls).")
    if len(labelled) < len(keep):
        summary.warnings.append(f"{len(keep) - len(labelled)} of {len(keep)} texts were not labelled.")
    await report(1.0, f"Answer key written: {len(labelled)} rows, {sum(counts.values())} entities")
    summary.usage = usage_block(llm)
    summary.batch = batch_info(llm)
    return summary


async def relabel_split_in_place(
    db: AsyncSession,
    project_id: int,
    *,
    split: str = "test",
    backend: str | None = None,
    entity_types: list[str] | None = None,
    progress: ProgressFn | None = None,
    backend_override: Any = None,
) -> RelabelSummary:
    """Relabel a prepared split's rows (``prepared/<split>.jsonl``) IN PLACE
    under the current conventions — texts unchanged, entities replaced,
    rows stamped with the conventions version. A split kept across re-splits
    (``keep_test``) is never touched by relabelling the rows it came from, so
    this is how its labels catch up. Every saved version snapshot holding a
    byte-identical copy of the file is rewritten too. A row the model
    returned nothing for keeps its labels."""
    from app.services.span_type_definitions_service import (
        CONVENTIONS_VERSION_KEY,
        conventions_version,
        definitions_clause,
        load_definitions,
        load_scan_tolerant,
    )
    from app.services.synth_backends.base import pick_backend

    async def report(fraction: float, message: str) -> None:
        if progress is not None:
            await progress(fraction, message)

    if split not in ("train", "val", "test"):
        raise RuntimeError(f"Only a prepared split (train / val / test) can be relabelled in place, not {split!r}.")
    prepared = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    path = prepared / f"{split}.jsonl"
    if not path.exists():
        raise RuntimeError(f"Project {project_id} has no {split} split.")
    original = path.read_bytes()
    rows = [json.loads(l) for l in original.decode("utf-8").splitlines() if l.strip()]
    span_idx = [i for i, r in enumerate(rows) if isinstance(r, dict) and _row_text(r).strip() and isinstance(r.get("entities"), list)]
    if not span_idx:
        raise RuntimeError(f"The {split} split of project {project_id} has no span rows.")
    old_labels = [_row_entities(rows[i]) for i in span_idx]
    definitions = await load_definitions(db, project_id)
    scan_tolerant = await load_scan_tolerant(db, project_id)
    types = [t.strip().lower() for t in (entity_types or []) if isinstance(t, str) and t.strip()]
    if not types:
        types = sorted(set(definitions) | {e["type"] for spans in old_labels for e in spans})
    conventions = definitions_clause(types, definitions, scan_tolerant=scan_tolerant)
    version = conventions_version(definitions, scan_tolerant)
    llm = backend_override or pick_backend(backend)
    backend_label = llm.describe() if hasattr(llm, "describe") else getattr(llm, "name", "backend")
    texts = [_row_text(rows[i]) for i in span_idx]
    new_labels, calls = await label_texts_any(llm, texts, types, conventions, report, backend_label)
    stamp = datetime.now(timezone.utc).isoformat()
    counts: Counter[str] = Counter()
    done: list[int] = []
    for j, i in enumerate(span_idx):
        if new_labels[j] is None:
            continue
        rows[i]["entities"] = new_labels[j]
        rows[i]["relabelled_by"] = backend_label
        rows[i]["relabelled_at"] = stamp
        if version:
            rows[i][CONVENTIONS_VERSION_KEY] = version
        counts.update(e["type"] for e in new_labels[j])
        done.append(j)
    if not done:
        raise RuntimeError(f"{backend_label} relabelled none of the {len(span_idx)} {split} rows.")
    content = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    snapshots = [p for p in sorted((prepared / "versions").glob(f"*/{split}.jsonl")) if p.read_bytes() == original]
    for target in [path, *snapshots]:
        target.write_bytes(content)
    summary = RelabelSummary(
        project_id=project_id, split=split, backend=backend_label, rows=len(span_idx), labelled_rows=len(done),
        failed_calls=calls["failed"], entities=sum(counts.values()), entity_counts=dict(counts.most_common()),
        agreement=agreement([old_labels[j] for j in done], [new_labels[j] for j in done]),
        conventions_types=[t for t in types if t in definitions], file_path=str(path),
    )
    if snapshots:
        summary.warnings.append(f"Also updated {len(snapshots)} saved version(s) holding the same {split} split.")
    if len(done) < len(span_idx):
        summary.warnings.append(f"{len(span_idx) - len(done)} of {len(span_idx)} rows got no labels back and kept their old ones.")
    summary.usage = usage_block(llm)
    summary.batch = batch_info(llm)
    return summary


async def relabel_training_rows(
    db: AsyncSession,
    project_id: int,
    *,
    backend: str | None = None,
    entity_types: list[str] | None = None,
    progress: ProgressFn | None = None,
    backend_override: Any = None,
    source: str | None = None,
    only_stale: bool = False,
    only_affected: bool = False,
    review_override: Any = None,
    review_experiment_id: int | None = None,
    dry_run: bool = False,
) -> RelabelSummary:
    """``source``: only the rows of that ``synth_source`` group (e.g.
    ``document_labels:doc=740``) — redo one batch without the rest.
    ``only_stale``: only rows not stamped with the current conventions
    version (what the review queue counts as stale). ``only_affected``
    (implies only_stale): of those, only rows the change touches — a span
    of a changed type, or text a changed type's pattern matches; the rest
    are re-stamped as current without a call (``conventions_carried_
    forward_at``). ``review_override``: a second (stronger) labeller run
    only on rows where this one and the project's trained model disagree.

    Relabel the project's saved synthetic span rows IN PLACE with
    ``backend`` under the project's labelling conventions — texts unchanged,
    entities replaced, then the regex pass. A row whose call failed keeps
    its old labels. The training-side counterpart of the answer key, so
    both sides follow the same conventions. Agreement = old labels vs new."""
    from app.services.synth_backends.base import pick_backend
    from app.services.synthetic_service import _merge_regex_entities

    async def report(fraction: float, message: str) -> None:
        if progress is not None:
            await progress(fraction, message)

    path = settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl"
    if not path.exists():
        raise RuntimeError(f"Project {project_id} has no saved synthetic rows to relabel.")
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]
    # Groups are named the way the review queue names them (rows without a
    # synth_source are "legacy:<source>"), so its relabel action matches.
    from app.services.synth_review_queue_service import _resolve_source_label

    span_idx = [
        i for i, r in enumerate(rows)
        if isinstance(r, dict) and isinstance(r.get("text"), str) and r["text"].strip() and isinstance(r.get("entities"), list)
        and (not source or _resolve_source_label(r) == source)
    ]
    if not span_idx:
        raise RuntimeError(
            f"Project {project_id} has no span rows to relabel" + (f" in source group {source!r}." if source else ".")
        )
    old_labels = [_row_entities(rows[i]) for i in span_idx]
    from app.services.span_type_definitions_service import (
        CONVENTIONS_VERSION_KEY,
        conventions_version,
        definitions_clause,
        load_definitions,
        load_scan_tolerant,
    )

    definitions = await load_definitions(db, project_id)
    scan_tolerant = await load_scan_tolerant(db, project_id)
    from app.services.span_type_definitions_service import (
        HISTORY_KEY,
        changed_types,
        conventions_snapshot,
        remember_current_conventions,
        row_affected,
    )

    await remember_current_conventions(db, project_id)
    targeting: dict[str, Any] | None = None
    carried: list[int] = []
    if only_stale or only_affected:
        current = conventions_version(definitions, scan_tolerant)
        keep = [j for j, i in enumerate(span_idx) if rows[i].get(CONVENTIONS_VERSION_KEY) != current]
        if not keep:
            raise RuntimeError(
                "Every row" + (f" in {source!r}" if source else "") + " is already labelled under the current conventions."
            )
        if only_affected:
            from app.models.project import Project
            from app.services.synthetic_service import _extract_entities_via_regex

            project = await db.get(Project, project_id)
            history = dict((project.runtime_config or {}).get(HISTORY_KEY) or {}) if project is not None else {}
            now = conventions_snapshot(definitions, scan_tolerant)

            def _pattern_hit(text: str, kinds: list[str]) -> bool:
                return bool(_extract_entities_via_regex(text, kinds))

            affected, changed_seen = [], set()
            for j in keep:
                row = rows[span_idx[j]]
                changed = changed_types(history.get(str(row.get(CONVENTIONS_VERSION_KEY) or "")), now)
                if changed:
                    changed_seen |= changed
                if row_affected(row, changed, _pattern_hit):
                    affected.append(j)
                else:
                    carried.append(span_idx[j])
            targeting = {
                "stale_rows": len(keep), "affected_rows": len(affected), "carried_forward": len(carried),
                "changed_types": sorted(changed_seen),
            }
            keep = affected
        if dry_run:
            return RelabelSummary(
                project_id=project_id, split="synthetic", backend="(dry run — no model called)",
                rows=len(keep), labelled_rows=0, failed_calls=0, entities=0, targeting=targeting,
            )
        span_idx = [span_idx[j] for j in keep]
        old_labels = [old_labels[j] for j in keep]
        if not span_idx and carried:
            stamp_now = datetime.now(timezone.utc).isoformat()
            for i in carried:
                rows[i][CONVENTIONS_VERSION_KEY] = current
                rows[i]["conventions_carried_forward_at"] = stamp_now
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            return RelabelSummary(
                project_id=project_id, split="synthetic", backend="(none — no row touched by the change)",
                rows=0, labelled_rows=0, failed_calls=0, entities=0, file_path=str(path), targeting=targeting,
                warnings=[f"{len(carried)} stale rows don't involve the changed types and were carried forward without relabelling."],
            )
    types = [t.strip().lower() for t in (entity_types or []) if isinstance(t, str) and t.strip()]
    if not types:
        # The project's vocabulary: every defined type plus every type any
        # saved row carries — one group alone may lack a type it should get.
        types = sorted(set(definitions) | {e["type"] for r in rows if isinstance(r, dict) for e in _row_entities(r)})
    conventions = definitions_clause(types, definitions, scan_tolerant=scan_tolerant)
    version = conventions_version(definitions, scan_tolerant)
    from app.services.span_custom_patterns_service import load_patterns

    custom_patterns = await load_patterns(db, project_id)
    llm = backend_override or pick_backend(backend)
    backend_label = llm.describe() if hasattr(llm, "describe") else getattr(llm, "name", "backend")
    texts = [rows[i]["text"] for i in span_idx]
    new_labels, calls = await label_texts_any(llm, texts, types, conventions, report, backend_label)
    review = None
    if review_override is not None:
        new_labels, review = await review_disagreements(
            db, project_id, texts, new_labels, types, conventions, review_override, report,
            experiment_id=review_experiment_id,
        )

    stamp = datetime.now(timezone.utc).isoformat()
    counts: Counter[str] = Counter()
    relabelled = 0
    for i in carried:
        rows[i][CONVENTIONS_VERSION_KEY] = version
        rows[i]["conventions_carried_forward_at"] = stamp
    for j, i in enumerate(span_idx):
        spans = new_labels[j]
        if spans is None:
            continue
        merged, _ = _merge_regex_entities(texts[j], spans, types, custom_patterns)
        rows[i]["entities"] = merged
        rows[i]["relabelled_by"] = backend_label
        rows[i]["relabelled_at"] = stamp
        if version:
            rows[i][CONVENTIONS_VERSION_KEY] = version
        counts.update(e["type"] for e in merged)
        relabelled += 1
    if not relabelled:
        raise RuntimeError(f"{backend_label} relabelled none of the {len(span_idx)} rows.")
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    done = [j for j in range(len(span_idx)) if new_labels[j] is not None]
    summary = RelabelSummary(
        project_id=project_id, split="synthetic", backend=backend_label, rows=len(span_idx), labelled_rows=relabelled,
        failed_calls=calls["failed"], entities=sum(counts.values()), entity_counts=dict(counts.most_common()),
        agreement=agreement([old_labels[j] for j in done], [rows[span_idx[j]]["entities"] for j in done]),
        conventions_types=[t for t in types if t in definitions], file_path=str(path),
    )
    summary.usage = usage_block(llm, *([review_override] if review_override is not None else []))
    summary.batch = batch_info(llm)
    summary.review = review
    summary.targeting = targeting
    kept_old = len(span_idx) - relabelled
    if kept_old:
        summary.warnings.append(
            f"{kept_old} of {len(span_idx)} rows got no labels back ({calls['failed']} failed calls) and kept their old labels."
        )
    await report(1.0, f"Relabelled {relabelled} training rows")
    return summary


PASSAGE_SOURCE_PREFIX = "passage_labels"


def load_document_passages(project_id: int, document_id: int | None = None) -> list[dict[str, Any]]:
    """Cleaned document passages (``cleaned/cleaned.jsonl`` rows without a
    ``row_index``), optionally of one source document, in file order."""
    path = settings.DATA_DIR / "projects" / str(project_id) / "cleaned" / "cleaned.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "row_index" in row or not str(row.get("text") or "").strip():
                continue
            if document_id is not None and int(row.get("source_document_id") or -1) != int(document_id):
                continue
            out.append(row)
    return out


DOCUMENT_SOURCE_PREFIX = "document_labels"
DEFAULT_DOCUMENT_ROW_CHARS = 4000


def load_document_texts(project_id: int, document_id: int | None = None) -> list[tuple[int, str]]:
    """``(document_id, cleaned text)`` per cleaned source document — the
    whole text, line breaks and page lines kept (``raw/<name>.cleaned.txt``),
    found through the passages' ``source_document_id`` / ``source_doc``."""
    from pathlib import Path as _Path

    seen: dict[int, str] = {}
    for row in load_document_passages(project_id, document_id):
        doc = row.get("source_document_id")
        name = row.get("source_doc")
        if isinstance(doc, int) and isinstance(name, str) and doc not in seen:
            seen[doc] = name
    raw = settings.DATA_DIR / "projects" / str(project_id) / "raw"
    out: list[tuple[int, str]] = []
    for doc, name in seen.items():
        path = raw / f"{_Path(name).stem}.cleaned.txt"
        if path.exists():
            out.append((doc, path.read_text(encoding="utf-8")))
    return out


def pack_long_rows(text: str, max_chars: int = DEFAULT_DOCUMENT_ROW_CHARS) -> list[str]:
    """Cut a document's text into rows of whole blank-line-separated blocks
    (a paragraph, or a whole document in a multi-document file) packed up to
    ``max_chars``; a single block longer than that is cut at sentence ends.
    Line breaks inside a block — page headers / footers — are kept."""
    from app.services.cleaning_service import chunk_text

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    rows: list[str] = []
    current = ""
    for block in blocks:
        pieces = [block] if len(block) <= max_chars else [c for c in chunk_text(block, chunk_size=max_chars, overlap=0) if c.strip()]
        for piece in pieces:
            if current and len(current) + 2 + len(piece) > max_chars:
                rows.append(current)
                current = ""
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        rows.append(current)
    return rows


LABEL_PIECE_CHARS = 1000


def piece_offsets(text: str, max_chars: int = LABEL_PIECE_CHARS) -> list[tuple[int, int]]:
    """``(start, end)`` pieces of ``text`` of at most ``max_chars``, cut at
    line ends where possible, else at a sentence end, else hard — covering
    the whole text, so labels found per piece map back by adding ``start``."""
    pieces: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            window = text[start:end]
            cut = window.rfind("\n")
            if cut < max_chars // 3:
                match = None
                for m in re.finditer(r"[.!?](?=\s)", window):
                    match = m
                cut = match.end() if match and match.end() >= max_chars // 3 else -1
            if cut > 0:
                end = start + cut
        pieces.append((start, end))
        start = end
    return pieces


async def _label_in_pieces(
    llm: Any, texts: list[str], types: list[str], conventions: str, report: ProgressFn, backend_label: str,
    piece_chars: int,
) -> tuple[list[list[dict[str, Any]] | None], dict[str, int]]:
    """Label each long text through passage-sized pieces and shift the
    spans back. A local model asked to label a whole multi-page document
    labels markedly less than the same text in pieces (project 31: ~4
    entities per 1,000 chars whole vs ~7 per passage), so long rows were
    trained on thin labels. A text is labelled only if all its pieces
    were (a half-labelled long row would teach the gaps as negatives)."""
    flat, owners = _flatten_pieces(texts, piece_chars)
    piece_labels, calls = await label_texts(llm, flat, types, conventions, report, backend_label, start=0.02, span=0.9)
    calls["pieces"] = len(flat)
    return _merge_pieces(texts, owners, piece_labels), calls


def _flatten_pieces(texts: list[str], piece_chars: int) -> tuple[list[str], list[tuple[int, int]]]:
    """Each text's pieces (``piece_offsets``, edge whitespace trimmed) and
    their owners ``(text index, piece start)``."""
    flat: list[str] = []
    owners: list[tuple[int, int]] = []  # (text index, piece start)
    for t_index, text in enumerate(texts):
        for start, end in piece_offsets(text, piece_chars):
            # Trim the piece's edge whitespace and move its offset with it:
            # span validation strips a text, so a piece starting with a space
            # or line break would come back with offsets one char early.
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                flat.append(text[start:end])
                owners.append((t_index, start))
    return flat, owners


def _merge_pieces(
    texts: list[str], owners: list[tuple[int, int]], piece_labels: list[list[dict[str, Any]] | None],
) -> list[list[dict[str, Any]] | None]:
    """Piece labels shifted back onto their texts; a text with any unlabelled
    piece (or a misaligned span) is None."""
    merged: list[list[dict[str, Any]] | None] = [[] for _ in texts]
    for (t_index, offset), spans in zip(owners, piece_labels):
        if merged[t_index] is None:
            continue
        if spans is None:
            merged[t_index] = None
            continue
        for e in spans:
            merged[t_index].append({**e, "start": e["start"] + offset, "end": e["end"] + offset})
    for t_index, spans in enumerate(merged):
        if spans:
            spans.sort(key=lambda e: (e["start"], e["end"]))
            bad = [e for e in spans if "text" in e and texts[t_index][e["start"]:e["end"]] != e["text"]]
            if bad:  # never expected — drop the row rather than train misplaced spans
                _LOG.warning("piece labels misaligned on text %d: %s", t_index, bad[:2])
                merged[t_index] = None
    return merged


async def label_texts_any(
    llm: Any, texts: list[str], types: list[str], conventions: str, report: ProgressFn, backend_label: str,
    *, piece_chars: int = LABEL_PIECE_CHARS, texts_per_call: int = TEXTS_PER_CALL,
) -> tuple[list[list[dict[str, Any]] | None], dict[str, int]]:
    """Every labelling path's entry point: texts up to ``piece_chars`` are
    labelled as they are (``label_texts``), longer ones through
    passage-sized pieces (``_label_in_pieces``) — a local model labels a
    long text thinly. Results come back in the order of ``texts``."""
    short = [i for i, t in enumerate(texts) if len(t) <= piece_chars]
    long = [i for i, t in enumerate(texts) if len(t) > piece_chars]
    out: list[list[dict[str, Any]] | None] = [None] * len(texts)
    calls = {"made": 0, "failed": 0, "batches": 0, "pieces": 0}
    if uses_message_batch(llm) and long:
        # Message Batch: the short texts and every long text's pieces go in
        # ONE batch (one wait), then the pieces are merged back as usual.
        flat, owners = _flatten_pieces([texts[i] for i in long], piece_chars)
        labels, c = await label_texts(
            llm, [texts[i] for i in short] + flat, types, conventions, report, backend_label,
            start=0.02, span=0.9, texts_per_call=texts_per_call,
        )
        for i, spans in zip(short, labels[:len(short)]):
            out[i] = spans
        for i, spans in zip(long, _merge_pieces([texts[i] for i in long], owners, labels[len(short):])):
            out[i] = spans
        calls.update({k: v for k, v in c.items()})
        calls["pieces"] = len(flat)
        calls["long_texts"] = len(long)
        return out, calls
    if short:
        labels, c = await label_texts(
            llm, [texts[i] for i in short], types, conventions, report, backend_label,
            start=0.02, span=0.9 if not long else 0.45, texts_per_call=texts_per_call,
        )
        for i, spans in zip(short, labels):
            out[i] = spans
        for k in ("made", "failed", "batches"):
            calls[k] += c.get(k, 0)
    if long:
        labels, c = await _label_in_pieces(llm, [texts[i] for i in long], types, conventions, report, backend_label, piece_chars)
        for i, spans in zip(long, labels):
            out[i] = spans
        for k in ("made", "failed", "batches", "pieces"):
            calls[k] += c.get(k, 0)
    calls["long_texts"] = len(long)
    return out, calls


async def _label_and_append(
    db: AsyncSession,
    project_id: int,
    items: list[tuple[str, dict[str, Any]]],
    *,
    source_prefix: str,
    split_label: str,
    nothing_new: str,
    backend: str | None,
    entity_types: list[str] | None,
    progress: ProgressFn | None,
    backend_override: Any,
    texts_per_call: int = TEXTS_PER_CALL,
    piece_chars: int | None = None,
    review_override: Any = None,
    review_experiment_id: int | None = None,
) -> RelabelSummary:
    """Label ``items`` (text, metadata) as they are under the conventions
    (+ regex pass) and append them to ``synthetic.jsonl`` as pending rows
    with integer ids (the review queue acts on ids). Texts already present
    are skipped. ``metadata["doc"]`` names the source document."""
    from app.services.synth_backends.base import pick_backend
    from app.services.synthetic_service import _merge_regex_entities

    async def report(fraction: float, message: str) -> None:
        if progress is not None:
            await progress(fraction, message)

    syn_path = settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl"
    existing_texts: set[str] = set()
    existing_types: set[str] = set()
    max_id = 0
    if syn_path.exists():
        for line in syn_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                if isinstance(row.get("id"), int) and not isinstance(row.get("id"), bool):
                    max_id = max(max_id, row["id"])
                if isinstance(row.get("text"), str):
                    existing_texts.add(row["text"])
                existing_types.update(e["type"] for e in _row_entities(row))
    todo = [(text, meta) for text, meta in items if text not in existing_texts]
    if not todo:
        raise RuntimeError(nothing_new)
    types = [t.strip().lower() for t in (entity_types or []) if isinstance(t, str) and t.strip()] or sorted(existing_types)
    if not types:
        raise RuntimeError("Name the entity types to label (the project has no labelled rows to take them from).")
    from app.services.span_type_definitions_service import (
        CONVENTIONS_VERSION_KEY,
        conventions_version,
        definitions_clause,
        load_definitions,
        load_scan_tolerant,
    )

    definitions = await load_definitions(db, project_id)
    scan_tolerant = await load_scan_tolerant(db, project_id)
    conventions = definitions_clause(types, definitions, scan_tolerant=scan_tolerant)
    version = conventions_version(definitions, scan_tolerant)
    from app.services.span_custom_patterns_service import load_patterns

    custom_patterns = await load_patterns(db, project_id)
    llm = backend_override or pick_backend(backend)
    backend_label = llm.describe() if hasattr(llm, "describe") else getattr(llm, "name", "backend")
    texts = [text for text, _ in todo]
    labels, calls = await label_texts_any(
        llm, texts, types, conventions, report, backend_label,
        piece_chars=piece_chars or LABEL_PIECE_CHARS, texts_per_call=texts_per_call,
    )
    review = None
    if review_override is not None:
        labels, review = await review_disagreements(
            db, project_id, texts, labels, types, conventions, review_override, report,
            experiment_id=review_experiment_id,
        )

    stamp = datetime.now(timezone.utc).isoformat()
    counts: Counter[str] = Counter()
    new_rows: list[dict[str, Any]] = []
    for (text, meta), spans in zip(todo, labels):
        if spans is None:
            continue
        merged, _ = _merge_regex_entities(text, spans, types, custom_patterns)
        counts.update(e["type"] for e in merged)
        doc = meta.get("doc")
        new_rows.append({
            # The review queue acts on rows by integer id; these land pending.
            "id": max_id + len(new_rows) + 1,
            "text": text,
            "entities": merged,
            "source": f"{source_prefix}:{backend_label}",
            "synth_source": f"{source_prefix}:doc={doc}",
            "source_document_id": doc,
            **{k: v for k, v in meta.items() if k != "doc"},
            "review_status": "pending",
            "generated_at": stamp,
            "labelled_by": backend_label,
            **({CONVENTIONS_VERSION_KEY: version} if version else {}),
        })
    if not new_rows:
        raise RuntimeError(f"{backend_label} labelled none of the {len(todo)} texts.")
    syn_path.parent.mkdir(parents=True, exist_ok=True)
    with syn_path.open("a", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = RelabelSummary(
        project_id=project_id, split=split_label, backend=backend_label, rows=len(todo), labelled_rows=len(new_rows),
        failed_calls=calls["failed"], entities=sum(counts.values()), entity_counts=dict(counts.most_common()),
        conventions_types=[t for t in types if t in definitions], file_path=str(syn_path),
    )
    skipped = len(items) - len(todo)
    if skipped:
        summary.warnings.append(f"{skipped} texts were already labelled and were skipped.")
    missing = len(todo) - len(new_rows)
    if missing:
        summary.warnings.append(
            f"{missing} of {len(todo)} texts got no labels back ({calls['failed']} failed calls) and were left out."
        )
    await report(1.0, f"Labelled {len(new_rows)} rows — accept them in the review queue ({source_prefix}:doc=…)")
    summary.usage = usage_block(llm, *([review_override] if review_override is not None else []))
    summary.batch = batch_info(llm)
    summary.review = review
    return summary


async def label_passages_as_training_rows(
    db: AsyncSession,
    project_id: int,
    *,
    document_id: int | None = None,
    backend: str | None = None,
    entity_types: list[str] | None = None,
    progress: ProgressFn | None = None,
    backend_override: Any = None,
    review_override: Any = None,
    max_passages: int = 2000,
) -> RelabelSummary:
    """Label the project's cleaned document passages AS THEY ARE (no
    rewrite) under the labelling conventions and append them, pending, to
    the synthetic span rows — the way to train on text with real layout
    (page headers, footers, letterheads) that the span generator's
    rewritten snippets never keep. Raises when nothing new was labelled."""
    passages = load_document_passages(project_id, document_id)[:max_passages]
    items = [
        (str(p["text"]), {"doc": p.get("source_document_id"), "chunk_id": p.get("chunk_id")})
        for p in passages
    ]
    return await _label_and_append(
        db, project_id, items,
        source_prefix=PASSAGE_SOURCE_PREFIX, split_label="passages",
        nothing_new=(
            "No unlabelled document passages"
            + (f" for document #{document_id}" if document_id is not None else "")
            + f" in project {project_id} ({len(passages)} passages, all already labelled)."
        ),
        backend=backend, entity_types=entity_types, progress=progress, backend_override=backend_override, review_override=review_override,
    )


async def label_documents_as_training_rows(
    db: AsyncSession,
    project_id: int,
    *,
    document_id: int | None = None,
    max_chars: int = DEFAULT_DOCUMENT_ROW_CHARS,
    backend: str | None = None,
    entity_types: list[str] | None = None,
    progress: ProgressFn | None = None,
    backend_override: Any = None,
    review_override: Any = None,
    max_rows: int = 1000,
) -> RelabelSummary:
    """Like ``label_passages_as_training_rows`` but on WHOLE documents: each
    row is a run of whole blank-line-separated blocks of one source
    document, up to ``max_chars`` (``pack_long_rows``) — a multi-page
    document with its page lines, longer than one 512-token window, so
    training runs in windows on real long rows. The labeller sees each
    document in passage-sized pieces (``_label_in_pieces``) and the spans
    are shifted back onto the whole row."""
    items: list[tuple[str, dict[str, Any]]] = []
    for doc, text in load_document_texts(project_id, document_id):
        for index, row_text in enumerate(pack_long_rows(text, max_chars)):
            items.append((row_text, {"doc": doc, "unit": "document", "piece": index}))
    items = items[:max_rows]
    if not items:
        raise RuntimeError(
            f"No cleaned document text found in project {project_id}"
            + (f" for document #{document_id}" if document_id is not None else "") + "."
        )
    return await _label_and_append(
        db, project_id, items,
        source_prefix=DOCUMENT_SOURCE_PREFIX, split_label="documents",
        nothing_new=f"All {len(items)} document rows of project {project_id} are already labelled.",
        backend=backend, entity_types=entity_types, progress=progress, backend_override=backend_override, review_override=review_override,
        piece_chars=LABEL_PIECE_CHARS,
    )

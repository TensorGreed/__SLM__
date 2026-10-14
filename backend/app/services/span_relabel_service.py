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
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
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


def build_relabel_prompt(texts: list[str], entity_types: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    types = ", ".join(entity_types) if entity_types else "every kind of sensitive or identifying data"
    return (
        "Label every entity in each text below. Entity types (use ONLY these, lower-case): "
        f"{types}.\n\n"
        "Rules:\n"
        "- entity.text is the EXACT characters as they appear in the text (copy them; offsets are "
        "verified from the text, so start/end may be approximate or omitted).\n"
        "- Label every occurrence; a person's name that appears twice is two entities.\n"
        "- Do not invent entities that are not in the text, and do not change the texts.\n\n"
        f"{numbered}\n\n"
        'Return JSON {"rows": [{"index": <text number>, "entities": [{"type": "...", "text": "...", '
        '"start": N, "end": M}, ...]}, ...]} with one row per text, in order.'
    )


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
    dataset_id: int | None = None
    file_path: str | None = None
    warnings: list[str] = field(default_factory=list)

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
            "dataset_id": self.dataset_id,
            "file_path": self.file_path,
            "warnings": list(self.warnings),
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
    from app.services.synthetic_service import _validate_span_rows

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

    llm = backend_override or pick_backend(backend)
    backend_label = llm.describe() if hasattr(llm, "describe") else getattr(llm, "name", "backend")
    new_labels: list[list[dict[str, Any]] | None] = [None] * len(keep)
    failed_calls = 0
    batches = [keep[i:i + TEXTS_PER_CALL] for i in range(0, len(keep), TEXTS_PER_CALL)]
    for b, batch in enumerate(batches):
        await report(0.05 + 0.85 * b / max(1, len(batches)), f"Labelling texts {batch[0] + 1}–{batch[-1] + 1} of {len(keep)} ({backend_label})")
        batch_texts = [texts[i] for i in batch]
        try:
            raw = await llm.complete(
                build_relabel_prompt(batch_texts, types),
                system_prompt=_SYSTEM_PROMPT,
                max_tokens=max(1500, 350 * len(batch)),
                temperature=0.0,
                response_schema=_RESPONSE_SCHEMA,
            )
        except SynthBackendError as exc:
            failed_calls += 1
            _LOG.warning("relabel batch %s failed: %s", b + 1, exc)
            continue
        parsed = _parse_rows(str(raw or ""))
        if not parsed:
            failed_calls += 1
            _LOG.warning("relabel batch %s: no rows parsed (content starts: %r)", b + 1, str(raw)[:160])
            continue
        for local_index, source_index in enumerate(batch):
            ents = parsed.get(local_index)
            if ents is None:
                continue
            validated = _validate_span_rows([{"text": texts[source_index], "entities": ents}], types)
            new_labels[keep.index(source_index)] = validated[0]["entities"] if validated else []

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
    )
    if failed_calls:
        summary.warnings.append(f"{failed_calls} of {len(batches)} labelling calls failed; those texts were skipped.")
    if len(labelled) < len(keep):
        summary.warnings.append(f"{len(keep) - len(labelled)} of {len(keep)} texts were not labelled.")
    await report(1.0, f"Answer key written: {len(labelled)} rows, {sum(counts.values())} entities")
    return summary

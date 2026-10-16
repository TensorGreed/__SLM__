"""Human review of the answer key (P6 Trust).

The answer key (``gold/gold_test.jsonl``, also ``gold_dev.jsonl``) is
written by a model, so every score against it inherits that model's
mistakes. This service lets a person check rows and records the verdict
ON THE ROW, next to the labels the eval reads:

    "review": {"verdict": "accepted" | "corrected" | "rejected",
               "reviewed_at": ISO, "reviewed_by": str | None, "note": str,
               "original": {...fields before the correction...}}

- ``accepted``: the labels stand as written.
- ``corrected``: the reviewer changed one or more answer fields (span
  rows: ``entities``; other shapes: ``answer`` / ``label`` / ``summary``);
  the row's fields are replaced in place so every reader scores the
  corrected answer, and the model's original is kept under ``original``.
- ``rejected``: the row is wrong beyond repair (or not a fair test) — the
  eval loaders drop it (``is_rejected``).

Ordering (``review_queue``): rows are listed unreviewed first, and within
those the ones where the key and the project's latest trained span run
disagree most (``row_agreement``: strict span F1 between the two) — a
disagreement means one of them is wrong, and a person decides which. A
project without a span run lists rows in file order.

Shape-agnostic: the review block and the queue work for any gold row;
only the ordering signal and the correction validation are span-aware.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

REVIEW_KEY = "review"
VERDICTS = ("accepted", "corrected", "rejected")
# Fields a correction may replace, by row shape. Spans are validated
# against the text; the others are plain strings.
CORRECTABLE_FIELDS = ("entities", "answer", "label", "summary")
GOLD_FILES = {"gold_test": "gold_test.jsonl", "gold_dev": "gold_dev.jsonl"}
PREDICTIONS_CACHE = "review_predictions.json"


class GoldReviewError(ValueError):
    pass


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def gold_path(project_id: int, dataset_type: str) -> Path:
    if dataset_type not in GOLD_FILES:
        raise GoldReviewError(f"dataset_type must be one of {sorted(GOLD_FILES)}")
    return settings.DATA_DIR / "projects" / str(project_id) / "gold" / GOLD_FILES[dataset_type]


def row_text(row: dict[str, Any]) -> str:
    """The text a reviewer reads (and the eval tags): ``text`` / ``input``
    / ``question`` / ``document``."""
    for key in ("text", "input", "question", "document", "source_text", "content"):
        value = row.get(key)
        if isinstance(value, dict):
            value = value.get("text") or value.get("input")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def row_key(row: dict[str, Any]) -> str:
    """Identity of a gold row across files and eval results: a hash of
    its text (whitespace-trimmed). The eval's ``row_scores.inputs`` uses
    the same hash, so reviewed rows can be found in a stored eval."""
    return text_key(row_text(row))


def text_key(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8", "replace")).hexdigest()[:12]


def load_rows(path: Path) -> list[dict[str, Any]]:
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


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def review_of(row: dict[str, Any]) -> dict[str, Any] | None:
    review = row.get(REVIEW_KEY)
    return review if isinstance(review, dict) and review.get("verdict") in VERDICTS else None


def is_rejected(row: dict[str, Any]) -> bool:
    """True for a gold row a reviewer rejected — eval loaders skip it."""
    review = review_of(row)
    return review is not None and review.get("verdict") == "rejected"


def is_reviewed(row: dict[str, Any]) -> bool:
    return review_of(row) is not None


def review_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """``{total, reviewed, accepted, corrected, rejected, unreviewed}``."""
    counts = {"total": len(rows), "reviewed": 0, "accepted": 0, "corrected": 0, "rejected": 0}
    for row in rows:
        review = review_of(row)
        if review is None:
            continue
        counts["reviewed"] += 1
        counts[str(review["verdict"])] += 1
    counts["unreviewed"] = counts["total"] - counts["reviewed"]
    return counts


def latest_review_at(rows: list[dict[str, Any]]) -> str | None:
    stamps = [str(r[REVIEW_KEY].get("reviewed_at")) for r in rows if review_of(r) and r[REVIEW_KEY].get("reviewed_at")]
    return max(stamps) if stamps else None


# ── Corrections ────────────────────────────────────────────────────────


def _validate_entities(text: str, entities: Any) -> list[dict[str, Any]]:
    """Span correction: every entity needs a type and offsets inside the
    text; ``text`` is re-read from the offsets so it can't drift.
    Overlapping spans are refused (the trainer / scorer treat spans as
    disjoint)."""
    if not isinstance(entities, list):
        raise GoldReviewError("entities must be a list")
    out: list[dict[str, Any]] = []
    for ent in entities:
        if not isinstance(ent, dict):
            raise GoldReviewError("each entity must be an object")
        etype = str(ent.get("type") or "").strip()
        if not etype:
            raise GoldReviewError("an entity has no type")
        try:
            start, end = int(ent.get("start")), int(ent.get("end"))
        except (TypeError, ValueError) as exc:
            raise GoldReviewError(f"entity {etype!r} has no integer offsets") from exc
        if not (0 <= start < end <= len(text)):
            raise GoldReviewError(f"entity {etype!r} offsets {start}:{end} fall outside the text (length {len(text)})")
        out.append({"type": etype, "start": start, "end": end, "text": text[start:end]})
    out.sort(key=lambda e: (e["start"], e["end"]))
    for prev, cur in zip(out, out[1:]):
        if cur["start"] < prev["end"]:
            raise GoldReviewError(f"entities overlap: {prev['type']} {prev['start']}:{prev['end']} and {cur['type']} {cur['start']}:{cur['end']}")
    return out


def apply_review(
    row: dict[str, Any],
    *,
    verdict: str | None,
    corrected: dict[str, Any] | None = None,
    note: str = "",
    reviewed_by: str | None = None,
) -> dict[str, Any]:
    """Return ``row`` with the review applied (pure; the caller writes).

    ``verdict`` None clears the review and restores the original fields
    of an earlier correction. ``corrected`` is required for ``corrected``
    (the fields it names replace the row's); the pre-correction values go
    under ``review.original`` — a second correction keeps the FIRST
    original, so clearing always returns to what the model wrote."""
    out = dict(row)
    previous = review_of(row)
    original: dict[str, Any] = dict(previous.get("original") or {}) if previous else {}
    if verdict is None:
        for field, value in original.items():
            out[field] = value
        out.pop(REVIEW_KEY, None)
        return out
    if verdict not in VERDICTS:
        raise GoldReviewError(f"verdict must be one of {VERDICTS}")
    if verdict == "corrected":
        if not isinstance(corrected, dict) or not any(k in corrected for k in CORRECTABLE_FIELDS):
            raise GoldReviewError(f"a correction needs one of {CORRECTABLE_FIELDS}")
        text = row_text(row)
        for field in CORRECTABLE_FIELDS:
            if field not in corrected:
                continue
            value = corrected[field]
            if field == "entities":
                value = _validate_entities(text, value)
            elif not isinstance(value, str):
                raise GoldReviewError(f"{field} must be a string")
            if field not in original:
                original[field] = row.get(field)
            out[field] = value
    elif original:
        # Accepting / rejecting after a correction keeps the corrected
        # fields (the reviewer saw them) and the original for a clear.
        pass
    out[REVIEW_KEY] = {
        "verdict": verdict,
        "reviewed_at": _utcnow(),
        "reviewed_by": reviewed_by,
        "note": str(note or ""),
        **({"original": original} if original else {}),
    }
    return out


# ── Queue ──────────────────────────────────────────────────────────────


def _entities(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    ents = row.get("entities")
    if isinstance(ents, list):
        return [e for e in ents if isinstance(e, dict)]
    return None


def _answer_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ("answer", "label", "summary") if isinstance(row.get(k), str)}


def order_queue(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unreviewed first; among those the lowest agreement first (None =
    no model to compare → after the scored ones, file order); reviewed
    rows keep file order at the end."""
    def sort_key(item: dict[str, Any]) -> tuple:
        reviewed = item.get("review") is not None
        agreement = item.get("agreement")
        return (reviewed, agreement is None, agreement if agreement is not None else 0.0, item["index"])
    return sorted(items, key=sort_key)


def _load_prediction_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


async def model_predictions(
    db: AsyncSession, project_id: int, rows: list[dict[str, Any]], experiment_id: int | None,
) -> tuple[int | None, dict[str, list[dict[str, Any]]]]:
    """Spans the project's latest span run (or ``experiment_id``) predicts
    for each row's text, keyed by ``text_key``; cached per run in
    ``gold/review_predictions.json`` so reopening the queue doesn't re-run
    the model. ``(None, {})`` without a span run."""
    from app.services.span_relabel_service import latest_span_run_predictions

    texts = [row_text(r) for r in rows]
    keys = [text_key(t) for t in texts]
    cache_path = settings.DATA_DIR / "projects" / str(project_id) / "gold" / PREDICTIONS_CACHE
    cache = _load_prediction_cache(cache_path)
    cached_run = cache.get("experiment_id")
    cached_items = cache.get("items") if isinstance(cache.get("items"), dict) else {}
    if cached_run is not None and (experiment_id is None or experiment_id == cached_run) and all(k in cached_items for k in keys):
        if experiment_id is not None or await _is_latest_span_run(db, project_id, int(cached_run)):
            return int(cached_run), {k: cached_items[k] for k in keys}
    run_id, spans = await latest_span_run_predictions(db, project_id, texts, experiment_id)
    if run_id is None or spans is None:
        return None, {}
    items = {k: s for k, s in zip(keys, spans)}
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"experiment_id": run_id, "items": items}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return run_id, items


async def _is_latest_span_run(db: AsyncSession, project_id: int, experiment_id: int) -> bool:
    from sqlalchemy import desc, select

    from app.models.experiment import Experiment, ExperimentStatus
    from app.services.post_training_eval_service import is_token_classification

    rows = (await db.execute(
        select(Experiment).where(Experiment.project_id == project_id, Experiment.status == ExperimentStatus.COMPLETED)
        .order_by(desc(Experiment.id))
    )).scalars().all()
    for exp in rows:
        if is_token_classification(exp) and exp.output_dir:
            return exp.id == experiment_id
    return False


async def review_queue(
    db: AsyncSession, project_id: int, *, dataset_type: str = "gold_test", experiment_id: int | None = None,
    compare: bool = True,
) -> dict[str, Any]:
    """The answer key as a review queue: every row with its labels, its
    review (if any), the trained model's spans and the agreement between
    the two, ordered disagreement-first (``order_queue``)."""
    from app.services.span_relabel_service import row_agreement

    path = gold_path(project_id, dataset_type)
    rows = load_rows(path)
    run_id: int | None = None
    predicted: dict[str, list[dict[str, Any]]] = {}
    span_rows = any(_entities(r) is not None for r in rows)
    if compare and span_rows and rows:
        try:
            run_id, predicted = await model_predictions(db, project_id, rows, experiment_id)
        except Exception:  # noqa: BLE001 — a missing model never blocks the review
            run_id, predicted = None, {}
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        key = row_key(row)
        ents = _entities(row)
        pred = predicted.get(key)
        item = {
            "index": index,
            "key": key,
            "text": row_text(row),
            "entities": ents,
            "fields": _answer_fields(row),
            "source": row.get("source"),
            "review": review_of(row),
            "predicted": pred,
            "agreement": round(row_agreement(ents, pred), 4) if ents is not None and pred is not None else None,
        }
        items.append(item)
    counts = review_counts(rows)
    return {
        "project_id": project_id,
        "dataset_type": dataset_type,
        "model_experiment_id": run_id,
        **counts,
        "last_reviewed_at": latest_review_at(rows),
        "rows": order_queue(items),
    }


async def set_review(
    db: AsyncSession, project_id: int, *, dataset_type: str, index: int, key: str | None,
    verdict: str | None, corrected: dict[str, Any] | None = None, note: str = "", reviewed_by: str | None = None,
) -> dict[str, Any]:
    """Record one row's verdict in the gold file and refresh the GOLD
    dataset's record count (rejected rows don't count). Returns the
    updated queue item + counts."""
    from app.services.span_relabel_service import row_agreement

    path = gold_path(project_id, dataset_type)
    rows = load_rows(path)
    if not (0 <= index < len(rows)):
        raise GoldReviewError(f"row {index} not found ({len(rows)} rows)")
    if key and row_key(rows[index]) != key:
        raise GoldReviewError("the answer key changed since you loaded it — reload the queue")
    rows[index] = apply_review(rows[index], verdict=verdict, corrected=corrected, note=note, reviewed_by=reviewed_by)
    write_rows(path, rows)
    await _sync_record_count(db, project_id, dataset_type, rows)
    row = rows[index]
    ents = _entities(row)
    pred = None
    cache = _load_prediction_cache(path.parent / PREDICTIONS_CACHE)
    items = cache.get("items") if isinstance(cache.get("items"), dict) else {}
    if ents is not None and row_key(row) in items:
        pred = items[row_key(row)]
    return {
        "row": {
            "index": index, "key": row_key(row), "text": row_text(row), "entities": ents, "fields": _answer_fields(row),
            "source": row.get("source"), "review": review_of(row), "predicted": pred,
            "agreement": round(row_agreement(ents, pred), 4) if ents is not None and pred is not None else None,
        },
        **review_counts(rows),
        "last_reviewed_at": latest_review_at(rows),
    }


async def _sync_record_count(db: AsyncSession, project_id: int, dataset_type: str, rows: list[dict[str, Any]]) -> None:
    from sqlalchemy import select

    from app.models.dataset import Dataset, DatasetType

    ds_type = DatasetType.GOLD_TEST if dataset_type == "gold_test" else DatasetType.GOLD_DEV
    ds = (await db.execute(
        select(Dataset).where(Dataset.project_id == project_id, Dataset.dataset_type == ds_type).order_by(Dataset.id.desc()).limit(1)
    )).scalar_one_or_none()
    if ds is None:
        return
    ds.record_count = sum(1 for r in rows if not is_rejected(r))
    await db.commit()


# ── Reviewed vs unreviewed scores ──────────────────────────────────────


def _micro(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) else (0.0 if p is not None and r is not None else None)
    return {"precision": None if p is None else round(p, 4), "recall": None if r is None else round(r, 4), "f1": None if f1 is None else round(f1, 4)}


def split_scores(
    row_scores: dict[str, Any] | None, rows: list[dict[str, Any]], *, eval_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Micro P/R/F1 of a stored span eval on the reviewed rows (accepted +
    corrected) vs the rows nobody has checked, from the eval's per-row
    counts (``row_scores.inputs`` / ``span_tp`` / ``span_fp`` /
    ``span_fn``). ``stale`` when a correction or rejection is newer than
    the eval — the eval scored the model's labels, not the reviewer's.
    None when the eval has no per-row span counts (older results)."""
    if not isinstance(row_scores, dict):
        return None
    inputs = row_scores.get("inputs")
    tps, fps, fns = row_scores.get("span_tp"), row_scores.get("span_fp"), row_scores.get("span_fn")
    if not all(isinstance(x, list) for x in (inputs, tps, fps, fns)) or not inputs:
        return None
    by_key: dict[str, dict[str, Any]] = {row_key(r): r for r in rows}
    buckets = {"reviewed": [0, 0, 0, 0], "unreviewed": [0, 0, 0, 0]}
    changed_after_eval = 0
    for key, tp, fp, fn in zip(inputs, tps, fps, fns):
        row = by_key.get(key)
        review = review_of(row) if row is not None else None
        if review is not None and review["verdict"] == "rejected":
            continue
        bucket = buckets["reviewed" if review is not None else "unreviewed"]
        bucket[0] += int(tp or 0)
        bucket[1] += int(fp or 0)
        bucket[2] += int(fn or 0)
        bucket[3] += 1
        if review is not None and review["verdict"] in ("corrected",) and _after(review.get("reviewed_at"), eval_at):
            changed_after_eval += 1
    counts = review_counts(rows)
    rejected_after = sum(
        1 for r in rows if is_rejected(r) and _after(r[REVIEW_KEY].get("reviewed_at"), eval_at)
    )
    out: dict[str, Any] = {
        **{k: v for k, v in counts.items()},
        "reviewed_scores": {**_micro(*buckets["reviewed"][:3]), "rows": buckets["reviewed"][3]},
        "unreviewed_scores": {**_micro(*buckets["unreviewed"][:3]), "rows": buckets["unreviewed"][3]},
        "stale": (changed_after_eval + rejected_after) > 0,
        "changed_since_eval": changed_after_eval + rejected_after,
    }
    return out


def _after(stamp: Any, eval_at: datetime | None) -> bool:
    if eval_at is None or not isinstance(stamp, str):
        return False
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if eval_at.tzinfo is None:
        eval_at = eval_at.replace(tzinfo=timezone.utc)
    return when > eval_at


def review_summary(project_id: int, dataset_type: str = "gold_test") -> dict[str, int] | None:
    """Counts for a card / model card line; None when there is no file."""
    path = gold_path(project_id, dataset_type)
    if not path.exists():
        return None
    return review_counts(load_rows(path))


def review_line(counts: dict[str, int] | None) -> str | None:
    """"12 of 43 rows checked by a person (3 corrected, 1 rejected)"; None
    when nothing was reviewed."""
    if not counts or not counts.get("reviewed"):
        return None
    parts = []
    if counts.get("corrected"):
        parts.append(f"{counts['corrected']} corrected")
    if counts.get("rejected"):
        parts.append(f"{counts['rejected']} rejected")
    detail = f" ({', '.join(parts)})" if parts else ""
    return f"{counts['reviewed']} of {counts['total']} rows checked by a person{detail}"

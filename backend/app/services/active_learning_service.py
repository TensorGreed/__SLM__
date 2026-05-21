"""Active-learning recommender (Theme 8 Epic 2).

After an evaluation run completes with non-trivial failures, surface
a small "Add these N examples to improve most" panel. Picks the rows
where the trained model got the wrong answer, then on user confirm
appends their *gold* answers to the project's synthetic training
dataset so the next training run learns from them.

Design honesty: this is **failed-row promotion**, not
confidence-aware active learning. The eval handlers today emit
binary row_exact_match / row_f1 / row_field_results — useful for
"this row failed", but no probabilistic confidence /logprob signal
that would let us distinguish overconfident-wrong from
knows-it-doesn't-know. That distinction (per the roadmap) is
deferred until handlers expose per-row confidence.

What we do today:

  - `propose_active_learning_batch` reads an EvalResult, identifies
    the failing rows (handler-aware: prefers per-row scores, falls
    back to prediction != reference), loads the *full* source row
    from `EvalResult.details["dataset"]["file_path"]` (so we don't
    promote the 160-char preview), and returns up to `max_rows`
    candidates ranked by failure severity.
  - `promote_active_learning_batch` appends the selected rows' gold
    answers to the project's SYNTHETIC dataset JSONL. Idempotent
    via a `promoted_indexes: list[int]` set stashed inside
    `EvalResult.details["active_learning"]`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.config import settings
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import EvalResult, Experiment
from app.models.project import Project


# Maximum candidates returned per propose call. Capped at the service
# level so a UI bug can't request thousands of rows by accident.
MAX_PROPOSE_ROWS = 200
DEFAULT_PROPOSE_ROWS = 20


@dataclass
class CandidateRow:
    """One row eligible for promotion. `row_index` is the position in
    the eval dataset file; it's the stable id used for
    idempotency tracking."""

    row_index: int
    failure_reason: str
    prompt: str
    prediction: str
    reference: str
    row_score: float | None
    source_row: dict[str, Any]
    already_promoted: bool = False

    def model_dump(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "failure_reason": self.failure_reason,
            "prompt": self.prompt,
            "prediction": self.prediction,
            "reference": self.reference,
            "row_score": self.row_score,
            "already_promoted": self.already_promoted,
        }


# ─────────────────────────────────────────────────────────────────────
# Row-loading + failure detection
# ─────────────────────────────────────────────────────────────────────


def _load_full_source_rows(dataset_file_path: str | None) -> list[dict[str, Any]]:
    """Load the original gold/test JSONL rows so promotion uses the
    untruncated prompt + reference (predictions_preview caps at 160
    chars). Returns empty list if path missing — caller falls back
    to the preview."""
    if not dataset_file_path:
        return []
    path = Path(dataset_file_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
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


def _detect_failure(preview_row: dict[str, Any]) -> tuple[bool, str, float | None]:
    """Decide whether one prediction-preview row counts as a failure.

    Order of evidence — first signal wins:

      1. Handler-emitted `row_exact_match` (0/1) — explicit verdict.
      2. Handler-emitted `row_f1` < 0.5 — soft fail bar.
      3. StructuredExtractionHandler `is_valid_json` is false.
      4. Last resort: trimmed `prediction` != trimmed `reference`.

    Returns (is_failure, reason_code, row_score_for_ranking).
    `row_score` is lower-is-worse; callers use it to rank candidates
    by severity (most-broken first).
    """
    row_exact = preview_row.get("row_exact_match")
    if row_exact is not None:
        try:
            score = float(row_exact)
        except (TypeError, ValueError):
            score = None
        if score is not None and score <= 0.0:
            return True, "row_exact_match=0", 0.0

    row_f1 = preview_row.get("row_f1")
    if isinstance(row_f1, (int, float)):
        if row_f1 < 0.5:
            return True, f"row_f1={row_f1:.2f}<0.5", float(row_f1)

    if preview_row.get("is_valid_json") is False:
        return True, "is_valid_json=false", 0.0

    pred = str(preview_row.get("prediction") or "").strip()
    ref = str(preview_row.get("reference") or "").strip()
    if ref and pred != ref:
        return True, "prediction!=reference", None

    return False, "", None


# ─────────────────────────────────────────────────────────────────────
# Propose
# ─────────────────────────────────────────────────────────────────────


async def _get_latest_eval_result(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> EvalResult | None:
    """Fetch the most recent EvalResult for an experiment in this
    project. Returns None if the experiment is missing or has no
    eval results yet."""
    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    if exp_result.scalar_one_or_none() is None:
        return None
    eval_q = await db.execute(
        select(EvalResult)
        .where(EvalResult.experiment_id == experiment_id)
        .order_by(EvalResult.id.desc())
        .limit(1)
    )
    return eval_q.scalar_one_or_none()


async def propose_active_learning_batch(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    *,
    max_rows: int = DEFAULT_PROPOSE_ROWS,
) -> dict[str, Any]:
    """Inspect the latest EvalResult for the given experiment and
    return up to `max_rows` failed rows ranked by severity.

    The returned `candidates` are full source-row payloads ready
    for promotion (uses `EvalResult.details["dataset"]["file_path"]`
    rather than the 160-char preview). `already_promoted=true` is
    set on rows the user previously promoted from this eval result.
    """
    cap = max(1, min(MAX_PROPOSE_ROWS, int(max_rows or DEFAULT_PROPOSE_ROWS)))

    eval_result = await _get_latest_eval_result(db, project_id, experiment_id)
    if eval_result is None:
        return {
            "eval_result_id": None,
            "experiment_id": experiment_id,
            "candidates": [],
            "total_failures": 0,
            "total_predictions": 0,
            "max_rows": cap,
            "message": "No eval result found for this experiment yet.",
        }

    details = dict(eval_result.details or {})
    preview = list(details.get("predictions_preview") or [])
    dataset_meta = dict(details.get("dataset") or {})
    full_rows = _load_full_source_rows(dataset_meta.get("file_path"))

    promoted_indexes = set(
        int(idx)
        for idx in ((details.get("active_learning") or {}).get("promoted_indexes") or [])
        if isinstance(idx, (int, float))
    )

    failed: list[CandidateRow] = []
    for idx, row in enumerate(preview):
        if not isinstance(row, dict):
            continue
        is_failure, reason, score = _detect_failure(row)
        if not is_failure:
            continue
        # Prefer the full source row when we can find it; otherwise
        # fall back to the (truncated) preview content.
        source_row = full_rows[idx] if idx < len(full_rows) else dict(row)
        failed.append(
            CandidateRow(
                row_index=idx,
                failure_reason=reason,
                prompt=str(row.get("prompt") or ""),
                prediction=str(row.get("prediction") or ""),
                reference=str(row.get("reference") or ""),
                row_score=score,
                source_row=source_row,
                already_promoted=idx in promoted_indexes,
            )
        )

    # Rank: lowest row_score first (worst rows up top), then
    # already-promoted rows last so the user's eye lands on
    # actionable candidates.
    def _rank_key(c: CandidateRow) -> tuple[int, float]:
        promoted_tier = 1 if c.already_promoted else 0
        score = c.row_score if c.row_score is not None else 0.0
        return (promoted_tier, score)

    failed.sort(key=_rank_key)
    capped = failed[:cap]

    return {
        "eval_result_id": eval_result.id,
        "experiment_id": experiment_id,
        "candidates": [c.model_dump() for c in capped],
        "total_failures": len(failed),
        "total_predictions": len(preview),
        "max_rows": cap,
        "dataset_name": dataset_meta.get("name"),
        "promoted_count": len(promoted_indexes),
    }


# ─────────────────────────────────────────────────────────────────────
# Promote
# ─────────────────────────────────────────────────────────────────────


def _synthetic_dir_for_project(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "synthetic"


async def _find_or_create_synthetic_dataset(
    db: AsyncSession,
    project_id: int,
) -> tuple[Dataset, Path]:
    """Mirror of `annotation/promotion._resolve_target_dataset` scoped
    to SYNTHETIC. Lightly duplicated rather than imported because
    promotion.py also handles preference-pair routing we don't need
    here. Future refactor candidate."""
    file_path = _synthetic_dir_for_project(project_id) / "synthetic.jsonl"
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.SYNTHETIC,
        )
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        dataset = Dataset(
            project_id=project_id,
            name="Synthetic Dataset",
            dataset_type=DatasetType.SYNTHETIC,
            file_path=str(file_path),
        )
        db.add(dataset)
        await db.flush()
    elif not dataset.file_path:
        dataset.file_path = str(file_path)
        await db.flush()
    return dataset, file_path


def _render_promotion_row(
    *,
    row_index: int,
    source_row: dict[str, Any],
    preview_prompt: str,
    preview_reference: str,
    eval_result_id: int,
    experiment_id: int,
    next_id: int,
) -> dict[str, Any] | None:
    """Turn a candidate row into a JSONL line for the synthetic
    dataset. Tries to extract a clean (input, expected) pair from
    the full source row first; falls back to the (truncated)
    preview content if the source row's shape is unfamiliar."""
    # 1) Gold-row shape: {input: {question/text/...}, expected: {answer/label/...}}
    inp = source_row.get("input")
    exp = source_row.get("expected")
    if isinstance(inp, dict) and isinstance(exp, dict):
        question = (
            inp.get("question")
            or inp.get("text")
            or inp.get("prompt")
            or next(iter(inp.values()), "")
        )
        if exp.get("answer") is not None:
            answer = exp["answer"]
        elif exp.get("label") is not None:
            answer = exp["label"]
        elif exp:
            answer = exp
        else:
            answer = ""
        answer_str = (
            answer if isinstance(answer, str)
            else json.dumps(answer, ensure_ascii=False)
        )
        return {
            "id": next_id,
            "question": str(question),
            "answer": answer_str,
            "source": "active_learning",
            "source_eval_result_id": eval_result_id,
            "source_experiment_id": experiment_id,
            "source_row_index": row_index,
            "status": "accepted",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }

    # 2) Flat training-row shape: {prompt/text, reference/answer/output}
    prompt_field = (
        source_row.get("prompt")
        or source_row.get("text")
        or source_row.get("question")
        or preview_prompt
    )
    answer_field = (
        source_row.get("reference")
        or source_row.get("answer")
        or source_row.get("output")
        or source_row.get("label")
        or preview_reference
    )
    if not prompt_field and not answer_field:
        return None
    answer_str = (
        answer_field if isinstance(answer_field, str)
        else json.dumps(answer_field, ensure_ascii=False)
    )
    return {
        "id": next_id,
        "question": str(prompt_field or ""),
        "answer": answer_str,
        "source": "active_learning",
        "source_eval_result_id": eval_result_id,
        "source_experiment_id": experiment_id,
        "source_row_index": row_index,
        "status": "accepted",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }


async def promote_active_learning_batch(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    row_indexes: list[int],
) -> dict[str, Any]:
    """Append the selected rows' *gold answers* to the project's
    SYNTHETIC dataset JSONL. Idempotent — re-running with the same
    indexes is a no-op (skipped via `EvalResult.details
    ["active_learning"]["promoted_indexes"]`).

    Raises ValueError for missing project / experiment / eval
    result; the API maps these to 404."""
    project_q = await db.execute(select(Project).where(Project.id == project_id))
    if project_q.scalar_one_or_none() is None:
        raise ValueError(f"project_not_found:{project_id}")

    eval_result = await _get_latest_eval_result(db, project_id, experiment_id)
    if eval_result is None:
        raise ValueError(f"eval_result_not_found:{experiment_id}")

    details = dict(eval_result.details or {})
    preview = list(details.get("predictions_preview") or [])
    dataset_meta = dict(details.get("dataset") or {})
    full_rows = _load_full_source_rows(dataset_meta.get("file_path"))

    al_state = dict(details.get("active_learning") or {})
    promoted_indexes: set[int] = set(
        int(idx) for idx in (al_state.get("promoted_indexes") or [])
        if isinstance(idx, (int, float))
    )

    requested = [
        int(idx) for idx in row_indexes
        if isinstance(idx, (int, float)) and 0 <= int(idx) < len(preview)
    ]
    to_promote = [idx for idx in requested if idx not in promoted_indexes]
    skipped_already_promoted = len(requested) - len(to_promote)

    if not to_promote:
        return {
            "promoted_count": 0,
            "skipped_already_promoted": skipped_already_promoted,
            "skipped_invalid_indexes": len(row_indexes) - len(requested),
            "target_dataset_id": None,
            "written_path": None,
        }

    dataset, file_path = await _find_or_create_synthetic_dataset(db, project_id)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    promoted_count = 0
    base_id = (dataset.record_count or 0)
    with file_path.open("a", encoding="utf-8") as handle:
        for idx in to_promote:
            preview_row = preview[idx] if isinstance(preview[idx], dict) else {}
            source_row = full_rows[idx] if idx < len(full_rows) else dict(preview_row)
            entry = _render_promotion_row(
                row_index=idx,
                source_row=source_row,
                preview_prompt=str(preview_row.get("prompt") or ""),
                preview_reference=str(preview_row.get("reference") or ""),
                eval_result_id=eval_result.id,
                experiment_id=experiment_id,
                next_id=base_id + promoted_count + 1,
            )
            if entry is None:
                continue
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            promoted_indexes.add(idx)
            promoted_count += 1

    dataset.record_count = (dataset.record_count or 0) + promoted_count
    dataset.file_path = str(file_path)

    al_state["promoted_indexes"] = sorted(promoted_indexes)
    al_state["last_promoted_at"] = datetime.now(timezone.utc).isoformat()
    details["active_learning"] = al_state
    eval_result.details = details
    # SQLAlchemy's JSON columns don't auto-detect in-place mutation,
    # so flag the column dirty explicitly.
    flag_modified(eval_result, "details")
    await db.flush()

    return {
        "promoted_count": promoted_count,
        "skipped_already_promoted": skipped_already_promoted,
        "skipped_invalid_indexes": len(row_indexes) - len(requested),
        "target_dataset_id": dataset.id,
        "target_dataset_path": str(file_path),
        "total_promoted_lifetime": len(promoted_indexes),
    }


__all__ = [
    "MAX_PROPOSE_ROWS",
    "DEFAULT_PROPOSE_ROWS",
    "propose_active_learning_batch",
    "promote_active_learning_batch",
]

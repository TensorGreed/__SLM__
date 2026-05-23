"""Coach Mode service (USER-SUCCESS Epic 4).

Generates context-aware suggestions for each workflow stage. The UI
mounts a ``CoachStrip`` per panel that calls
``GET /api/projects/{id}/coach/{stage}`` and renders the returned
suggestions with click-to-execute action buttons.

Design constraints:
- Suggestions must carry a self-contained ``action`` payload so the
  frontend can route the click without per-suggestion glue code.
- Every suggestion includes a ``severity`` (info | warning | critical)
  so the UI can color the stripe consistently.
- Generators must NOT mutate project state — they're read-only. Any
  side effect happens later when the user clicks the action button.

Phase 1 ships the ``"data"`` stage end-to-end. Subsequent phases will
add ``"cleaning"`` / ``"gold_set"`` / ``"training"`` / ``"eval"`` by
adding generators to ``_STAGE_HANDLERS``.
"""

from __future__ import annotations

from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.project import Project


CoachStage = Literal["data", "cleaning", "gold_set", "training", "eval"]
Severity = Literal["info", "warning", "critical"]

# Thresholds describing what a "thin" / "comfortable" gold-set looks
# like for the typical narrow task. These are deliberately conservative
# — most useful first models on BrewSLM need at least 100 rows to learn
# any non-trivial pattern; 300+ is comfortable for classification with
# a handful of labels.
GOLD_ROW_THIN_MAX: int = 99
GOLD_ROW_COMFORTABLE_MIN: int = 300
# Default top-up the coach proposes when gold is thin: bring the user
# to ``GOLD_ROW_COMFORTABLE_MIN``, capped by the playbook's own
# ``target_count`` ceiling (500). Floor at 20 so the suggestion is
# never a no-op.
SUGGESTED_TOPUP_FLOOR: int = 20
SUGGESTED_TOPUP_CEILING: int = 500


async def _read_gold_row_count(db: AsyncSession, project_id: int) -> int:
    """Read-only sum of ``record_count`` across the project's gold
    datasets (dev + test). Avoids the ``get_or_create_gold_dataset``
    side effect so the coach call doesn't materialize empty Dataset
    rows on first read."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_([DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]),
        )
    )
    rows = result.scalars().all()
    return sum(int(ds.record_count or 0) for ds in rows)


def _topup_count(current: int) -> int:
    """How many synthetic rows to suggest generating to lift the gold
    set toward the comfortable threshold. Clamped to the playbook
    endpoint's accepted range."""
    delta = GOLD_ROW_COMFORTABLE_MIN - max(0, current)
    return max(SUGGESTED_TOPUP_FLOOR, min(SUGGESTED_TOPUP_CEILING, delta))


def _recipe_id_for(project: Project) -> str | None:
    recipe = project.selected_recipe or {}
    rid = recipe.get("recipe_id")
    if isinstance(rid, str) and rid.strip():
        return rid
    return None


async def _data_stage_suggestions(
    db: AsyncSession, project: Project
) -> list[dict[str, Any]]:
    """Suggestions surfaced on the Data tab.

    Phase 1 covers exactly one signal: gold-set row count vs. the
    "comfortable for a narrow task" threshold. Phase 2 will add
    class-imbalance + format-consistency suggestions.
    """
    suggestions: list[dict[str, Any]] = []
    row_count = await _read_gold_row_count(db, project.id)

    if row_count < GOLD_ROW_COMFORTABLE_MIN:
        topup = _topup_count(row_count)
        severity: Severity = "critical" if row_count <= GOLD_ROW_THIN_MAX else "warning"
        recipe_id = _recipe_id_for(project)

        # Build the click-to-execute action. When the project has a
        # selected recipe, point the action at the recipe's
        # ``positives_paraphrase`` playbook (it's the universal mode
        # — registered for every recipe). When there's no recipe yet,
        # we can't trigger run-playbook (the endpoint requires a
        # selected recipe), so fall back to a navigation hint and
        # mark the suggestion as ``navigate`` so the UI can route
        # the user to the recipe picker first.
        if recipe_id:
            action: dict[str, Any] = {
                "kind": "run_playbook",
                "label": f"Generate {topup} synthetic positives",
                "params": {
                    "mode": "positives_paraphrase",
                    "target_count": topup,
                    "target_class": None,
                },
            }
        else:
            action = {
                "kind": "navigate",
                "label": "Pick a recipe first",
                "params": {"target": "recipe-picker"},
            }

        # Headline framing depends on which bucket we're in: "thin" is
        # the urgent one ("most narrow tasks need 100+ rows"); the
        # warning case is the less-urgent "you could comfortably
        # train but more rows would help" framing.
        if row_count <= GOLD_ROW_THIN_MAX:
            title = f"Your gold set has {row_count} rows"
            body = (
                "Most useful first models need at least 100 rows of labeled "
                f"examples. Generating ~{topup} synthetic positives via the "
                "recipe's paraphrase playbook bridges the gap fast."
            )
        else:
            title = f"Your gold set has {row_count} rows — could be stronger"
            body = (
                f"You're past the 100-row floor, but {GOLD_ROW_COMFORTABLE_MIN}+ "
                "rows is the comfortable zone for narrow tasks. Generating "
                f"~{topup} more synthetic positives improves headroom on "
                "your eval splits."
            )

        suggestions.append({
            "id": "data:gold-row-count",
            "title": title,
            "body": body,
            "severity": severity,
            "action": action,
            "context": {
                "gold_row_count": row_count,
                "comfortable_threshold": GOLD_ROW_COMFORTABLE_MIN,
                "thin_threshold": GOLD_ROW_THIN_MAX,
            },
        })

    return suggestions


# ─────────────────────────────────────────────────────────────────────
# Cleaning stage (Phase 2).
# ─────────────────────────────────────────────────────────────────────

# Threshold for flagging the document error-rate signal. 5% of an
# ingested corpus failing means either a parser bug or a connector
# config issue; either case is worth surfacing before training rather
# than letting the failures silently shrink the training set.
DOC_ERROR_RATE_WARN: float = 0.05
# Below this absolute count we don't bother — a single failure in a
# 5-doc test corpus shouldn't trigger a 20%-error-rate alarm.
DOC_ERROR_MIN_TOTAL: int = 10


async def _read_pii_stats(
    db: AsyncSession, project_id: int
) -> dict[str, int | list[str]]:
    """Aggregate PII redaction counts across the project's RawDocument
    rows. PII counts are stamped onto ``RawDocument.metadata_`` during
    the cleaning pass — there's no dedicated findings table.

    Returns ``{"total_pii": int, "docs_with_pii": int, "pii_types": list[str]}``.
    """
    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
    )
    total_pii = 0
    docs_with_pii = 0
    pii_types: set[str] = set()
    for doc in result.scalars():
        meta = doc.metadata_ or {}
        count = int(meta.get("pii_count") or 0)
        if count > 0:
            total_pii += count
            docs_with_pii += 1
            kinds = meta.get("pii_types") or []
            if isinstance(kinds, list):
                pii_types.update(str(k) for k in kinds if isinstance(k, str))
    return {
        "total_pii": total_pii,
        "docs_with_pii": docs_with_pii,
        "pii_types": sorted(pii_types),
    }


async def _read_doc_status_breakdown(
    db: AsyncSession, project_id: int
) -> dict[str, int]:
    """Count of RawDocument rows per ``DocumentStatus`` for the project.
    Used to compute the cleaning/ingestion failure rate (docs that
    landed in ``ERROR``) without loading the full row payloads."""
    result = await db.execute(
        select(RawDocument.status, func.count(RawDocument.id))
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
        .group_by(RawDocument.status)
    )
    counts: dict[str, int] = {}
    for status, count in result.all():
        # ``status`` is a DocumentStatus enum coming back from SQLAlchemy.
        counts[status.value if hasattr(status, "value") else str(status)] = int(count)
    return counts


async def _cleaning_stage_suggestions(
    db: AsyncSession, project: Project
) -> list[dict[str, Any]]:
    """Suggestions surfaced on the Cleaning tab.

    Phase 2 signals:
    1. PII findings — count > 0 → "review N redactions across K docs".
    2. Doc error rate — ERROR/total > 5% (with absolute floor) → "review failures".
    """
    suggestions: list[dict[str, Any]] = []

    # ── PII findings ────────────────────────────────────────────
    pii_stats = await _read_pii_stats(db, project.id)
    total_pii = int(pii_stats["total_pii"])
    docs_with_pii = int(pii_stats["docs_with_pii"])
    if total_pii > 0:
        pii_types = pii_stats["pii_types"]
        # We don't change severity based on PII count alone — any PII
        # finding is worth a quick review. A 1000-finding corpus and a
        # 5-finding corpus are both warning-level: the user should
        # look before training, not necessarily panic.
        suggestions.append({
            "id": "cleaning:pii-findings",
            "title": (
                f"{total_pii} PII redaction{'s' if total_pii != 1 else ''} "
                f"found across {docs_with_pii} document{'s' if docs_with_pii != 1 else ''}"
            ),
            "body": (
                "Review the redactions before training — false positives leak "
                "useful signal, and false negatives leak real PII into the "
                "model weights. "
                + (
                    f"Categories: {', '.join(pii_types)}."
                    if isinstance(pii_types, list) and pii_types
                    else ""
                )
            ),
            "severity": "warning",
            "action": {
                "kind": "navigate",
                "label": "Open redaction review",
                "params": {"target": "cleaning-pii-review"},
            },
            "context": {
                "total_pii": total_pii,
                "docs_with_pii": docs_with_pii,
                "pii_types": pii_types,
            },
        })

    # ── Doc error rate (ingestion / cleaning failures) ─────────
    status_counts = await _read_doc_status_breakdown(db, project.id)
    total_docs = sum(status_counts.values())
    error_count = status_counts.get(DocumentStatus.ERROR.value, 0)
    if total_docs >= DOC_ERROR_MIN_TOTAL:
        error_rate = error_count / total_docs if total_docs > 0 else 0.0
        if error_rate > DOC_ERROR_RATE_WARN:
            # Severity escalates past 20% — at that point it's almost
            # certainly a connector / parser bug, not a few noisy
            # documents.
            severity: Severity = "critical" if error_rate > 0.20 else "warning"
            suggestions.append({
                "id": "cleaning:doc-error-rate",
                "title": (
                    f"{error_count} of {total_docs} documents failed processing "
                    f"({error_rate * 100:.0f}%)"
                ),
                "body": (
                    "A failure rate above 5% usually means a parser bug or "
                    "a misconfigured connector. Review the failure cluster "
                    "on the Cleaning tab — silently shrinking the training "
                    "set with bad parses is worse than fixing the root cause."
                ),
                "severity": severity,
                "action": {
                    "kind": "navigate",
                    "label": "Review failure cluster",
                    "params": {"target": "cleaning-failure-cluster"},
                },
                "context": {
                    "total_docs": total_docs,
                    "error_count": error_count,
                    "error_rate": round(error_rate, 4),
                    "warn_threshold": DOC_ERROR_RATE_WARN,
                },
            })

    return suggestions


# ─────────────────────────────────────────────────────────────────────
# Gold-set stage (Phase 2).
# ─────────────────────────────────────────────────────────────────────

# Diversity warn threshold: when mean pairwise Jaccard exceeds this,
# the gold set's rows look too similar to each other (low signal to
# train on). Kept loosely in sync with
# ``trainability_forecast_service.DIVERSITY_WARN_THRESHOLD`` — both
# fire on the same condition but Coach Mode doesn't pull from the
# trainability cache (we want a live read on every poll).
GOLD_DIVERSITY_WARN_JACCARD: float = 0.40
# Suggested top-up when class imbalance / diversity fires. Capped at
# the synth playbook endpoint's ``target_count`` ceiling.
DIVERSITY_TOPUP_DEFAULT: int = 50
CLASS_BALANCE_TOPUP_DEFAULT: int = 50


async def _gold_set_stage_suggestions(
    db: AsyncSession, project: Project
) -> list[dict[str, Any]]:
    """Suggestions surfaced on the Gold Set tab.

    Phase 2 signals:
    1. Class imbalance — reuses ``_signal_class_imbalance`` from the
       trainability forecast and translates the result into a
       ``run_playbook(class_balance_fill, target_class=<lowest>)``
       action.
    2. Gold-set diversity — reuses ``_signal_goldset_diversity`` and
       translates a ``warn`` outcome into a
       ``run_playbook(positives_paraphrase)`` action.
    """
    # Local imports keep coach_service import-cycle-free if the
    # trainability service ever depends on coach (currently it
    # doesn't, but we don't want to fragile-import-order this).
    from app.services.recipe_service import get_recipe
    from app.services.trainability_forecast_service import (
        _load_gold_rows,
        _signal_class_imbalance,
        _signal_goldset_diversity,
    )

    recipe_id = _recipe_id_for(project)
    if not recipe_id:
        # Without a recipe we don't know the task profile + can't
        # safely judge class balance. Nudge the user to the recipe
        # picker — same fallback used on the data stage.
        return [{
            "id": "gold_set:no-recipe",
            "title": "Pick a recipe before reviewing the gold set",
            "body": (
                "Coach Mode needs the recipe's task profile (classification, "
                "qa-sft, span-extraction, etc.) to score gold-set health. "
                "Selecting a recipe also unlocks the synth playbook "
                "suggestions that bridge gaps in your gold set."
            ),
            "severity": "info",
            "action": {
                "kind": "navigate",
                "label": "Open recipe picker",
                "params": {"target": "recipe-picker"},
            },
        }]

    recipe = get_recipe(recipe_id)
    task_profile = getattr(recipe, "task_profile", None) if recipe else None
    if not task_profile:
        return []

    gold_rows = await _load_gold_rows(db, project.id)
    suggestions: list[dict[str, Any]] = []

    # ── Class imbalance ────────────────────────────────────────
    class_signal = _signal_class_imbalance(gold_rows, task_profile)
    if class_signal is not None and class_signal.get("severity") in ("warn", "block"):
        under = (
            class_signal.get("suggested_action", {}).get("params", {})
            .get("underrepresented_classes", [])
        )
        target_class = under[0] if under else None
        severity: Severity = "critical" if class_signal["severity"] == "block" else "warning"
        suggestions.append({
            "id": "gold_set:class-imbalance",
            "title": str(class_signal.get("headline", "Class distribution is skewed")),
            "body": (
                str(class_signal.get("detail", "")) + " "
                "Generating examples for the under-represented class is the "
                "fastest way to lift eval scores on minority classes."
            ).strip(),
            "severity": severity,
            "action": {
                "kind": "run_playbook",
                "label": (
                    f"Generate {CLASS_BALANCE_TOPUP_DEFAULT} examples"
                    + (f" for '{target_class}'" if target_class else "")
                ),
                "params": {
                    "mode": "class_balance_fill",
                    "target_count": CLASS_BALANCE_TOPUP_DEFAULT,
                    "target_class": target_class,
                },
            },
            "context": {
                "underrepresented_classes": under,
                "headline": class_signal.get("headline"),
            },
        })

    # ── Gold-set diversity ─────────────────────────────────────
    diversity_signal, diversity_score = _signal_goldset_diversity(gold_rows)
    if diversity_signal.get("severity") == "warn":
        suggestions.append({
            "id": "gold_set:diversity-low",
            "title": str(diversity_signal.get("headline", "Gold set lacks diversity")),
            "body": (
                str(diversity_signal.get("detail", "")) + " "
                "Paraphrasing your existing positives is a safe first lift — "
                "it adds linguistic variety without changing labels."
            ).strip(),
            "severity": "warning",
            "action": {
                "kind": "run_playbook",
                "label": f"Paraphrase {DIVERSITY_TOPUP_DEFAULT} more positives",
                "params": {
                    "mode": "positives_paraphrase",
                    "target_count": DIVERSITY_TOPUP_DEFAULT,
                    "target_class": None,
                },
            },
            "context": {
                "diversity_score": round(float(diversity_score), 4),
                "warn_threshold_jaccard": GOLD_DIVERSITY_WARN_JACCARD,
            },
        })

    return suggestions


_STAGE_HANDLERS = {
    "data": _data_stage_suggestions,
    "cleaning": _cleaning_stage_suggestions,
    "gold_set": _gold_set_stage_suggestions,
}


async def suggest_for_stage(
    db: AsyncSession, project_id: int, stage: CoachStage
) -> dict[str, Any]:
    """Top-level entry point. Returns a serializable payload the
    ``CoachStrip`` UI renders directly.

    The frontend treats an empty ``suggestions`` array as "nothing to
    coach right now" — the strip should remain mounted (so users see
    it light up when state changes) but render an unobtrusive "Looks
    healthy" pill rather than a blank space.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    handler = _STAGE_HANDLERS.get(stage)
    if handler is None:
        # Unknown stage isn't an error — it just means no coach
        # generators have been wired for that surface yet. Return
        # an empty list so the UI degrades gracefully when a new
        # CoachStrip is mounted before its backend handler ships.
        return {
            "project_id": project_id,
            "stage": stage,
            "suggestions": [],
            "handler_available": False,
        }

    suggestions = await handler(db, project)
    return {
        "project_id": project_id,
        "stage": stage,
        "suggestions": suggestions,
        "handler_available": True,
    }

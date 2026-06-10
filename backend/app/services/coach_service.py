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


_ARCHETYPE_DOMINANT_PRIORITY: tuple[str, ...] = (
    # Order = "which below-cohort feature is the most actionable
    # leverage when multiple fire at once." Row count beats class
    # balance beats hard-negatives beats diversity beats length
    # mismatches. Used to pick which feature's suggestion the Coach
    # card surfaces verbatim (the panel itself shows all of them).
    "row_count",
    "class_entropy",
    "class_balance_ratio",
    "hard_negative_ratio",
    "goldset_diversity",
    "input_length_chars",
    "output_length_chars",
)


def _pick_dominant_below_feature(
    comparison: dict[str, Any],
) -> dict[str, Any] | None:
    """Walk the comparison's features in priority order; return the
    first one whose status is ``below``. None when nothing below."""
    features_by_id: dict[str, dict[str, Any]] = {
        f["feature_id"]: f for f in comparison.get("features", [])
    }
    for fid in _ARCHETYPE_DOMINANT_PRIORITY:
        f = features_by_id.get(fid)
        if f and f.get("status") == "below":
            return f
    return None


def _archetype_drift_nudge(
    project_id: int,
    stage: Literal["data", "training"],
    comparison: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """USER-SUCCESS Epic 8 Phase 8c — surface archetype drift as a
    Coach suggestion. Pure function: caller is responsible for
    loading the comparison (we never block Coach on the archetype
    service if it fails).

    Severity + threshold by stage:
      * data → info, fires when ≥1 feature is below cohort p25
      * training → warning, fires when ≥2 features below

    Skip conditions:
      * comparison missing / lacks features → no nudge
      * fewer-below than the stage threshold → no nudge
      * user IS the only archetype provenance (n_user_projects == 1
        AND that one project is this project) → no nudge (no point
        recommending you match yourself)
      * data-stage AND no user passing projects (template seeds
        only) → no nudge (too early; the user hasn't trained
        anything yet so the recommendation isn't actionable until
        they reach training stage)

    Cites cohort size + the dominant-drift feature's suggestion in
    the body so the recommendation is auditable from inside the
    Coach card. Action is the dominant feature's ``suggested_action``
    verbatim (matches Coach's existing contract so the same handler
    fires it as the data-stage gold-row-count card).
    """
    if not isinstance(comparison, dict):
        return None
    features = comparison.get("features") or []
    below = [f for f in features if isinstance(f, dict) and f.get("status") == "below"]
    threshold = 1 if stage == "data" else 2
    if len(below) < threshold:
        return None

    archetype = comparison.get("archetype") or {}
    n_user = int(archetype.get("n_user_projects") or 0)
    n_template = int(archetype.get("n_template_seeds") or 0)
    cohort = archetype.get("cohort_provenance") or []

    # Skip when the user is the only contributor (self-comparison
    # is silly). Detect via the cohort_provenance entries: if there's
    # exactly one user entry AND its id matches this project, bail.
    user_entries = [
        c for c in cohort if isinstance(c, dict) and c.get("source") == "user"
    ]
    if (
        len(user_entries) == 1
        and int(user_entries[0].get("id") or 0) == project_id
    ):
        return None

    # Skip the data-stage nudge when there are no user passing
    # projects yet. Template seeds are still useful at training-time
    # (the user is about to spend compute), but at data-time the
    # advice is premature.
    if stage == "data" and n_user == 0:
        return None

    dominant = _pick_dominant_below_feature(comparison)
    if dominant is None:
        # Shouldn't happen given the len(below) >= threshold guard
        # above, but defend so a future feature-set change doesn't
        # crash here.
        return None

    severity: Severity = "info" if stage == "data" else "warning"
    cohort_size_str = (
        f"{n_user} successful {comparison.get('recipe_id', '')} project"
        + ("" if n_user == 1 else "s")
        + (
            f" (+ {n_template} template seed" + ("" if n_template == 1 else "s") + ")"
            if n_template > 0
            else ""
        )
    ).strip()

    n_below_str = (
        "1 feature"
        if len(below) == 1
        else f"{len(below)} features"
    )
    title = (
        f"Your project shape differs from successful cohorts "
        f"({n_below_str} below p25)"
    )

    body_parts: list[str] = []
    body_parts.append(
        f"Cohort: {cohort_size_str}."
    )
    body_parts.append(
        f"Dominant drift: {dominant.get('label', dominant.get('feature_id'))}."
    )
    suggestion_text = dominant.get("suggestion")
    if suggestion_text:
        body_parts.append(suggestion_text)
    if len(below) > 1:
        body_parts.append(
            f"See the Archetype panel on Training Config for all "
            f"{len(below)} drifts."
        )

    body = " ".join(body_parts).strip()

    # Action: use the dominant feature's suggested_action if present
    # (same contract as Coach's other suggestions). Falls back to a
    # navigate-to-training-config when the dominant feature has no
    # one-click fix (length mismatches etc.).
    action = dominant.get("suggested_action") or {
        "kind": "navigate",
        "label": "Open archetype comparison",
        "params": {"target": "training-config"},
    }
    if "label" not in action:
        # Coach contract: action MUST carry a label string. Some
        # comparison.suggested_action payloads were stamped without
        # one (the backend builds them as {kind, params} for the
        # frontend to format). Provide a sensible default.
        action = {
            **action,
            "label": (
                "Generate via playbook"
                if action.get("kind") == "run_playbook"
                else "Open"
            ),
        }

    return {
        "id": f"{stage}:archetype-drift",
        "title": title,
        "body": body,
        "severity": severity,
        "action": action,
        # rule_id pins which side of the archetype-drift heuristic
        # fired: ``archetype-drift.dominant`` when one feature
        # dominates the drift, ``archetype-drift.broad`` when
        # multiple features are below their archetype band.
        "rule_id": (
            "archetype-drift.dominant"
            if dominant.get("feature_id")
            else "archetype-drift.broad"
        ),
        "context": {
            "n_below_features": len(below),
            "n_user_projects": n_user,
            "n_template_seeds": n_template,
            "dominant_feature_id": dominant.get("feature_id"),
            "below_feature_ids": [f.get("feature_id") for f in below],
        },
    }


async def _load_archetype_comparison_safe(
    db: AsyncSession, project_id: int
) -> dict[str, Any] | None:
    """Best-effort comparison load — Coach never blocks on a broken
    archetype service. Returns None on any error (missing recipe,
    empty cohort, project not found, exception)."""
    try:
        from app.services.archetype_service import (
            compare_project_to_archetype,
        )

        return await compare_project_to_archetype(db, project_id)
    except Exception:  # noqa: BLE001 — defense across the boundary
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
            # rule_id: the decision-rule the trace UI labels. Tells
            # the user "this fired because row_count <= thin_max"
            # vs "this fired because row_count < comfortable_min".
            "rule_id": (
                "gold-row-count.thin"
                if row_count <= GOLD_ROW_THIN_MAX
                else "gold-row-count.below-comfortable"
            ),
            "context": {
                "gold_row_count": row_count,
                "comfortable_threshold": GOLD_ROW_COMFORTABLE_MIN,
                "thin_threshold": GOLD_ROW_THIN_MAX,
                "topup_target": topup,
            },
        })

    # Phase 8c — archetype-drift nudge. Best-effort; never blocks
    # the data-tab Coach if the archetype service is unavailable.
    comparison = await _load_archetype_comparison_safe(db, project.id)
    archetype_nudge = _archetype_drift_nudge(project.id, "data", comparison)
    if archetype_nudge:
        suggestions.append(archetype_nudge)

    # Gap-#1/#2 slice 3 — nudge users running the no-op default
    # normalizer to swap in safe-cleanup-normalizer. Slice 1 shipped
    # the real builtins; slice 2 made them pickable; this slice
    # closes the loop by surfacing the choice when the user hasn't
    # made it. Best-effort: if the hook resolver errors (missing
    # pack / migration race), skip silently rather than break the
    # data-tab Coach.
    noop_nudge = await _noop_normalizer_nudge(db, project.id)
    if noop_nudge:
        suggestions.append(noop_nudge)

    return suggestions


async def _noop_normalizer_nudge(
    db: AsyncSession, project_id: int
) -> dict[str, Any] | None:
    """Fire when the project's effective normalizer is the no-op
    ``default-normalizer``. Returns None when a non-default normalizer
    is already configured OR when the hook resolver errors.

    Severity is ``info`` (not warning/critical) — running the no-op
    normalizer isn't broken behaviour, it's just not making use of a
    feature the user has. Nudge, don't alarm.
    """
    from app.services.domain_hook_service import resolve_project_domain_hooks

    try:
        hooks = await resolve_project_domain_hooks(db, project_id)
    except Exception:
        return None

    normalizer = hooks.get("normalizer") if isinstance(hooks, dict) else None
    if not isinstance(normalizer, dict):
        return None
    active_id = str(normalizer.get("id") or "").strip().lower()
    if active_id != "default-normalizer":
        return None

    return {
        "id": "data:noop-normalizer",
        "title": "Your domain pack is running the no-op normalizer",
        "body": (
            "`default-normalizer` is pass-through — it leaves the canonical "
            "record exactly as-is. The `safe-cleanup-normalizer` builtin is "
            "the recommended swap-in: it decodes HTML entities (`&amp;` → "
            "`&`, `&nbsp;` → space) and collapses runs of whitespace, which "
            "catches the most common ingestion-stage cleanup issues without "
            "any domain-specific assumptions. Swap it in via the Domain Pack "
            "Manager — the picker badges it with a ★."
        ),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open Domain Pack Manager",
            "params": {"target": "domain-pack-manager"},
        },
        "rule_id": "noop-normalizer.default-active",
        "context": {
            "active_normalizer_id": active_id,
            "recommended_normalizer_id": "safe-cleanup-normalizer",
            "domain_pack_applied": hooks.get("domain_pack_applied"),
        },
    }


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


# Quality-Lift phase 4 slice 2 — label-noise nudge thresholds.
# Below this floor the model isn't trained well enough for the dual-
# condition signal to be reliable; we don't even suggest scanning.
LABEL_NOISE_MIN_LABELED: int = 50
# Re-fire ratio. If the LATEST scan ran on fewer than this fraction
# of the current labeled count, the user has added a meaningful new
# batch since the last scan and we should suggest re-scanning to
# pick up the noise in the new labels.
LABEL_NOISE_RESCAN_RATIO: float = 0.80


async def _count_labeled_label_rows(
    db: AsyncSession, project_id: int,
) -> int:
    """Count all labeled (``labeled_at IS NOT NULL``) LabelRows across
    the project's classification label_jobs. Mirrors the convention
    ``label_noise_scoring_service._labeled_rows_for_project`` uses so
    the Coach nudge and the scoring service can't drift on what
    counts as a "labeled row."
    """
    from app.models.label_job import LabelJob, LabelRow
    from sqlalchemy import func

    job_ids_q = await db.execute(
        select(LabelJob.id).where(
            LabelJob.project_id == project_id,
            LabelJob.label_type == "classification",
        )
    )
    job_ids = [int(jid) for jid in job_ids_q.scalars().all()]
    if not job_ids:
        return 0
    count_q = await db.execute(
        select(func.count(LabelRow.id)).where(
            LabelRow.job_id.in_(job_ids),
            LabelRow.labeled_at.is_not(None),
        )
    )
    return int(count_q.scalar() or 0)


async def _latest_succeeded_label_noise_scan(db: AsyncSession, project_id: int):
    """Return the most recent SUCCEEDED LabelNoiseScan for the project,
    or None. Read once + reused by both the scan-ready and the
    results-pending nudges so we don't double-query the same row."""
    from app.models.label_noise_scan import LabelNoiseScan, LabelNoiseScanStatus

    rows = await db.execute(
        select(LabelNoiseScan)
        .where(
            LabelNoiseScan.project_id == project_id,
            LabelNoiseScan.status == LabelNoiseScanStatus.SUCCEEDED,
        )
        .order_by(
            LabelNoiseScan.completed_at.desc(),
            LabelNoiseScan.id.desc(),
        )
    )
    return rows.scalars().first()


async def _has_classifier_checkpoint(db: AsyncSession, project_id: int) -> bool:
    """True when at least one COMPLETED classification Experiment with
    an existing output_dir exists. The scoring service can't run
    without one; the Coach nudge stays silent until the user trains
    once."""
    from pathlib import Path

    from app.models.experiment import Experiment, ExperimentStatus

    rows = await db.execute(
        select(Experiment)
        .where(
            Experiment.project_id == project_id,
            Experiment.status == ExperimentStatus.COMPLETED,
        )
        .order_by(Experiment.completed_at.desc(), Experiment.id.desc())
    )
    for exp in rows.scalars().all():
        cfg = exp.config if isinstance(exp.config, dict) else {}
        task_type = str(cfg.get("task_type") or "").strip().lower()
        if task_type != "classification":
            continue
        raw_dir = (exp.output_dir or "").strip()
        if raw_dir and Path(raw_dir).exists():
            return True
    return False


async def _label_noise_scan_ready_nudge(
    db: AsyncSession, project_id: int,
) -> dict[str, Any] | None:
    """Quality-Lift phase 4 slice 2 — Cleaning-stage nudge:
    "Scan your labels for noise before the next train."

    Fires when:
      - ``labeled_count >= 50`` (minimum for self-confidence to be
        informative against an overfit-prone small set; below this
        the model isn't trained well enough to gripe meaningfully).
      - AND a classification checkpoint exists (else there's nothing
        to score against).
      - AND no LabelNoiseScan exists yet, OR the latest one ran on
        fewer than 80% of the current labeled rows (user added a
        meaningful new batch).

    Silences when a scan is currently RUNNING / QUEUED (nudging the
    user to start another scan while one's in flight is annoying) —
    but only when that in-flight scan is the latest one (we don't
    care about old QUEUED rows from earlier).
    """
    from app.models.label_noise_scan import LabelNoiseScan, LabelNoiseScanStatus

    labeled_count = await _count_labeled_label_rows(db, project_id)
    if labeled_count < LABEL_NOISE_MIN_LABELED:
        return None
    if not await _has_classifier_checkpoint(db, project_id):
        return None

    # Don't nudge during an in-flight scan — check the very latest
    # row regardless of status.
    latest_any_q = await db.execute(
        select(LabelNoiseScan)
        .where(LabelNoiseScan.project_id == project_id)
        .order_by(LabelNoiseScan.created_at.desc(), LabelNoiseScan.id.desc())
        .limit(1)
    )
    latest_any = latest_any_q.scalars().first()
    if latest_any is not None and latest_any.status in (
        LabelNoiseScanStatus.QUEUED,
        LabelNoiseScanStatus.RUNNING,
    ):
        return None

    latest = await _latest_succeeded_label_noise_scan(db, project_id)
    if latest is not None:
        label_count_at_scan = int(latest.label_count_at_scan or 0)
        if label_count_at_scan > 0:
            # If we've scanned >= 80% of the current labels already,
            # don't pester the user; they can re-scan manually.
            if label_count_at_scan >= LABEL_NOISE_RESCAN_RATIO * labeled_count:
                return None

    new_labels_since = (
        labeled_count - int(latest.label_count_at_scan or 0)
        if latest is not None
        else labeled_count
    )

    return {
        "id": "cleaning:label-noise-scan-ready",
        "title": (
            f"Scan your {labeled_count} labels for noise before the next train"
            if latest is None
            else f"Re-scan: {new_labels_since} new labels since last scan"
        ),
        "body": (
            "Label-noise scanning catches rows where your trained model "
            "confidently disagrees with the given label — fixing those "
            "lifts F1 more reliably than adding new labels."
            if latest is None
            else (
                f"You've added {new_labels_since} labels since the last scan. "
                "Re-running picks up noise in the new batch and updates the "
                "suspected-mislabels queue."
            )
        ),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open label-noise review",
            "params": {
                "target": "label-noise-review",
                "auto_start_scan": True,
            },
        },
        "rule_id": (
            "label-noise-scan-ready.first-scan"
            if latest is None
            else "label-noise-scan-ready.rescan-ready"
        ),
        "context": {
            "labeled_count": labeled_count,
            "label_count_at_last_scan": (
                int(latest.label_count_at_scan or 0) if latest is not None else 0
            ),
            "new_labels_since_last_scan": new_labels_since,
            "latest_scan_id": int(latest.id) if latest is not None else None,
        },
    }


async def _label_noise_results_pending_nudge(
    db: AsyncSession, project_id: int,
) -> dict[str, Any] | None:
    """Cleaning-stage nudge: "Review N suspected mislabels."

    Fires when the latest SUCCEEDED scan has ``suspected_count > 0``.
    Silences when the user has no scan yet (the scan-ready nudge
    handles that), or when the latest scan came back clean.

    This is independent of the scan-ready nudge: a user can re-train
    and find new suspects even when their previous scan was clean,
    so we always show this when there ARE suspects to review.
    """
    latest = await _latest_succeeded_label_noise_scan(db, project_id)
    if latest is None:
        return None
    suspected = int(latest.suspected_count or 0)
    if suspected <= 0:
        return None

    return {
        "id": "cleaning:label-noise-results-pending",
        "title": f"Review {suspected} suspected mislabel{'s' if suspected != 1 else ''}",
        "body": (
            "Your trained model confidently disagrees with these labels — "
            "review and decide whether to relabel, keep as-is, or drop "
            "the row entirely (which puts it back in the unlabeled pool)."
        ),
        # Warning rather than info: this is data quality that's
        # actively dragging F1, not a gentle suggestion to scan.
        "severity": "warning",
        "action": {
            "kind": "navigate",
            "label": "Review suspected mislabels",
            "params": {
                "target": "label-noise-review",
                "scan_id": int(latest.id),
            },
        },
        "rule_id": "label-noise-results-pending.suspects-found",
        "context": {
            "scan_id": int(latest.id),
            "suspected_count": suspected,
            "label_count_at_scan": int(latest.label_count_at_scan or 0),
            "confidence_threshold": latest.confidence_threshold,
            "given_label_floor": latest.given_label_floor,
        },
    }


async def _cleaning_stage_suggestions(
    db: AsyncSession, project: Project
) -> list[dict[str, Any]]:
    """Suggestions surfaced on the Cleaning tab.

    Phase 2 signals:
    1. PII findings — count > 0 → "review N redactions across K docs".
    2. Doc error rate — ERROR/total > 5% (with absolute floor) → "review failures".
    3. Quality-Lift phase 4 slice 2 — label-noise scan-ready and
       results-pending nudges.
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

    # Quality-Lift phase 4 slice 2 — Label-noise nudges. The
    # results-pending nudge takes precedence in the visual order
    # because the user already has suspected rows to review — that's
    # actionable now, vs the scan-ready nudge which is "do this
    # next." Both can fire on the same poll (user labels +50 rows
    # after a scan with suspects, hasn't reviewed them yet, and the
    # new batch is large enough to warrant a re-scan).
    results_nudge = await _label_noise_results_pending_nudge(db, project.id)
    if results_nudge:
        suggestions.append(results_nudge)
    scan_ready_nudge = await _label_noise_scan_ready_nudge(db, project.id)
    if scan_ready_nudge:
        suggestions.append(scan_ready_nudge)

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
    from app.services.synth_backends import pick_schema_aware_backend_describe
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
        # Phase 5c — opt the click-to-execute action into constrained
        # decoding when a schema-aware backend (vllm > nemo) is
        # configured + reachable. class_balance_fill is the only
        # playbook today that defines response_schema(), so this is
        # the one Coach suggestion that benefits. When nothing schema-
        # aware is available, leave the pin off → orchestrator
        # auto-picks (typically Ollama) and falls back to parser-only
        # validation.
        schema_aware_pin = pick_schema_aware_backend_describe()
        action_params: dict[str, Any] = {
            "mode": "class_balance_fill",
            "target_count": CLASS_BALANCE_TOPUP_DEFAULT,
            "target_class": target_class,
        }
        if schema_aware_pin:
            action_params["backend"] = schema_aware_pin
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
                "params": action_params,
            },
            # rule_id distinguishes "block" severity (one class is
            # critically under-represented) from "warn" severity
            # (skew exists but isn't blocking). Trace UI labels the
            # specific rule that matched so a future audit can grep
            # ``rule_id`` rather than parsing severity + class
            # signal back-derivation.
            "rule_id": (
                "class-imbalance.block"
                if class_signal["severity"] == "block"
                else "class-imbalance.warn"
            ),
            "context": {
                "underrepresented_classes": under,
                "headline": class_signal.get("headline"),
                "target_class": target_class,
                "topup_target": CLASS_BALANCE_TOPUP_DEFAULT,
                "schema_aware_backend": schema_aware_pin,
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

    # ── Pending synth review queue ─────────────────────────────
    # Click-to-execute actions (class_balance_fill etc.) write rows
    # to the project's synth review queue with review_status="pending".
    # ``dataset_service._load_records_from_file`` excludes pending rows
    # by default, so unreviewed synth rows are silently gated out of
    # training. This suggestion is the reminder + the one-click jump
    # to the queue UI so users don't lose the work they just generated.
    from app.services.synth_review_queue_service import list_review_queue

    try:
        queue = await list_review_queue(db, project.id)
    except Exception:  # noqa: BLE001 — never block the gold_set strip on a queue read
        queue = None
    pending_count = int((queue or {}).get("total_pending") or 0)
    if pending_count > 0:
        # Severity scales with the pile-up. Up to 4 rows is just a
        # "don't forget" nudge; 5+ means a meaningful chunk of data
        # the user generated is silently missing from training, so
        # warn. Never critical — pending rows aren't a failure mode,
        # just an unfinished todo.
        severity: Severity = "warning" if pending_count >= 5 else "info"
        # Pull the top source bucket(s) so the body can name what's
        # waiting (e.g. "from class_balance_fill") — more useful than
        # an abstract count.
        groups = (queue or {}).get("groups") or []
        top_source = groups[0].get("synth_source") if groups else None
        body_detail = ""
        if top_source and "class_balance_fill" in str(top_source):
            body_detail = (
                " The class-imbalance suggestion you ran wrote into this "
                "queue — accepting these rows is what actually lifts the "
                "minority-class signal in training."
            )
        elif top_source:
            body_detail = (
                f" The largest pending source is `{top_source}`. Accept "
                "to add to the training set; reject to drop."
            )
        # The top source bucket also flows into action.params.synth_source
        # so the Coach navigate handler can build a focused URL —
        # ``?focus_synth_source=<source>`` lets SynthReviewQueue render
        # a one-click "Accept all N <source> rows" banner instead of
        # making the user multi-select. When the queue has more than
        # one source (mixed playbook runs), we still pin the largest
        # bucket — usually the most recent / most-actioned one.
        action_params: dict[str, Any] = {"target": "synthetic-review-queue"}
        if top_source:
            action_params["synth_source"] = top_source
        suggestions.append({
            "id": "gold_set:synth-review-pending",
            "title": (
                f"{pending_count} synthetic row"
                f"{'' if pending_count == 1 else 's'} pending review"
            ),
            "body": (
                "Generated synthetic rows land in the review queue with "
                "`review_status=\"pending\"` and are excluded from "
                "training until you accept them."
                + body_detail
            ).strip(),
            "severity": severity,
            "action": {
                "kind": "navigate",
                "label": "Open review queue",
                "params": action_params,
            },
            "context": {
                "total_pending": pending_count,
                "top_source": top_source,
            },
        })

    return suggestions


# ─────────────────────────────────────────────────────────────────────
# Training stage (Phase 3).
# ─────────────────────────────────────────────────────────────────────

# Pass-rate ceiling above which Coach stays silent on the eval surface
# — the user is comfortably above target and additional suggestions
# would be noise. Set at 0.90 since most narrow tasks aim for ≥ 0.85
# eval pass rate; the 90% bar leaves headroom for noise.
EVAL_PASS_RATE_HEALTHY: float = 0.90
# Below this pass rate Coach treats the eval as a critical failure
# worth surfacing the dominant failure cluster + click-to-augment
# action.
EVAL_PASS_RATE_CRITICAL: float = 0.60
# How many synthetic rows to request when click-to-augment fires on
# a failure cluster. Anchored to the playbook endpoint's 1-500 range;
# 30 mirrors the playbook router's own default.
CLUSTER_AUGMENT_DEFAULT: int = 30


def _curriculum_training_suggestion(
    project_id: int, recipe_id: str | None
) -> dict[str, Any] | None:
    """Phase 6d — info-severity nudge to flip on curriculum learning
    on thin classification projects. Returns None when curriculum
    isn't applicable OR when the project would auto-default it on
    anyway (no double-coaching).

    The nudge cites the Phase 6c A/B (2026-05-25) numbers so users
    aren't asked to take it on faith; the toggle on Training Config
    is two clicks away."""
    if not recipe_id:
        return None
    from app.services.curriculum_service import (
        recommended_scoring_mode_for_recipe,
    )
    from app.services.training_service import (
        CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS,
        _decide_curriculum_default,
    )

    if recommended_scoring_mode_for_recipe(recipe_id) is None:
        return None  # recipe not curriculum-eligible

    # If the backend would auto-default the flag on for this project,
    # don't also nudge — the user would see both "Coach: enable
    # curriculum" AND a "curriculum: ON (auto-defaulted)" badge,
    # which is redundant. The auto-default fires only when the
    # caller leaves ``config.curriculum`` unset, which is the path
    # the Training Config UI's "use recommended defaults" mode takes
    # by default. Users who explicitly set curriculum=false see the
    # nudge as a reminder.
    decision = _decide_curriculum_default(
        project_obj=None,  # we'd need the actual project here to be precise
        project_id=project_id,
    )
    # Note: the ``project_obj=None`` branch returns False with
    # ``no_recipe_selected``; we already know the recipe exists, so
    # for the *nudge*, we re-derive eligibility ourselves from
    # recipe_id + the file probe. We don't need _decide_... here
    # for the nudge gate — the auto-default fires at experiment-
    # creation time anyway, so the nudge is purely informational.
    del decision  # unused — kept the import for the docstring's accuracy

    # File probe: is this a thin-data project that would benefit?
    from app.config import settings as _settings
    train_file = (
        _settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"
    )
    if not train_file.exists():
        return None  # nothing to coach about pre-prep
    try:
        with train_file.open(encoding="utf-8") as f:
            row_count = sum(1 for line in f if line.strip())
    except OSError:
        return None
    if row_count == 0 or row_count > CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS:
        return None

    return {
        "id": "training:curriculum-learning-available",
        "title": (
            f"Curriculum learning is recommended for this run "
            f"({row_count} train rows, thin-data regime)"
        ),
        "body": (
            "Curriculum learning trains easy examples before hard ones "
            "— Phase 6c A/B (2026-05-25, 5 seeds, GB10) measured "
            "**+93% F1 on ticket-router** and **+55% F1 on log-triage** "
            "vs uniform training. The toggle is on by default for new "
            "experiments at this row count; this card is the heads-up."
        ),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open Training Config",
            "params": {"target": "training-config"},
        },
        "context": {
            "train_row_count": row_count,
            "threshold": CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS,
            "ab_run_date": "2026-05-25",
            "ab_seeds": 5,
            "ab_lift_pct": {"ticket-router": 93.20, "log-triage": 54.95},
        },
    }


async def _active_learning_ready_nudge(
    db: AsyncSession, project_id: int,
) -> dict[str, Any] | None:
    """Quality-Lift phase 3 slice 2 — Training-stage Coach nudge:
    "Label N uncertain rows before the next train."

    Fires when the most recent COMPLETED experiment has a non-empty
    active-learning snapshot (slice 1) AND fewer than
    ``STALENESS_THRESHOLD`` (0.80) of the snapshot's rows have been
    labeled since. Silences automatically when:

      - No COMPLETED experiment has a snapshot yet (fresh project).
      - The snapshot is empty (skipped_reason set — non-classification
        task, empty pool, scoring failed; the Data Studio card in
        slice 3 surfaces the reason).
      - ≥80% of the snapshot's rows have been labeled — the user
        worked through the queue.

    Returns the suggestion dict matching the locked Coach contract,
    or ``None`` to skip. The endpoint
    ``GET /api/projects/{id}/active-learning/latest`` shares the same
    snapshot-read + labeled-count logic; we duplicate it here rather
    than call the endpoint to avoid an HTTP round-trip from Coach.
    """
    from app.api.active_learning import (
        STALENESS_THRESHOLD,
        _count_labeled_rows,
        _latest_completed_experiment_with_snapshot,
    )

    exp = await _latest_completed_experiment_with_snapshot(db, project_id)
    if exp is None:
        return None

    cfg = exp.config if isinstance(exp.config, dict) else {}
    runtime = cfg.get("_runtime") if isinstance(cfg.get("_runtime"), dict) else {}
    snapshot = runtime.get("active_learning") if isinstance(runtime, dict) else None
    if not isinstance(snapshot, dict):
        return None
    top_k_entries = list(snapshot.get("top_k") or [])
    top_k_size = len(top_k_entries)
    if top_k_size == 0:
        return None

    row_ids = [
        int(entry["label_row_id"])
        for entry in top_k_entries
        if isinstance(entry, dict) and isinstance(entry.get("label_row_id"), int)
    ]
    labeled_count = await _count_labeled_rows(db, row_ids)
    staleness_ratio = labeled_count / top_k_size if top_k_size else 0.0
    if staleness_ratio >= STALENESS_THRESHOLD:
        return None  # snapshot worked through; silence

    unlabeled_count = max(0, top_k_size - labeled_count)
    # Use the dominant job_id from the top_k entries so the deep-link
    # lands the labeler directly on the right job. Top-K usually all
    # come from one classification job, but in the rare cross-job
    # case we route to the first one and the user can switch jobs in
    # the labeler.
    dominant_job_id: int | None = None
    for entry in top_k_entries:
        if isinstance(entry, dict) and isinstance(entry.get("label_job_id"), int):
            dominant_job_id = int(entry["label_job_id"])
            break

    uncertainty_metric = str(snapshot.get("uncertainty_metric") or "entropy")

    return {
        "id": "training:active-learning-ready",
        "title": (
            f"Label {unlabeled_count} uncertain rows before the next train"
        ),
        "body": (
            f"Experiment #{int(exp.id)} scored your unlabeled pool by "
            f"{uncertainty_metric}; the top {top_k_size} most-uncertain "
            f"rows are queued and {labeled_count} of them are already "
            "labeled. Working through the rest now maximizes the lift "
            "on the next training run."
        ),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open label queue",
            "params": {
                "target": "active-labeling-queue",
                "label_job_id": dominant_job_id,
            },
        },
        "rule_id": "active-learning-ready.snapshot-fresh",
        "context": {
            "experiment_id": int(exp.id),
            "snapshot_size": top_k_size,
            "labeled_count": labeled_count,
            "unlabeled_count": unlabeled_count,
            "staleness_ratio": round(staleness_ratio, 4),
            "uncertainty_metric": uncertainty_metric,
        },
    }


async def _training_stage_suggestions(
    db: AsyncSession, project: Project
) -> list[dict[str, Any]]:
    """Suggestions surfaced on the Training Config page.

    Phase 3 wires the L1 trainability forecast: when the forecast
    verdict is ``likely_fail`` (or ``borderline``), Coach proposes
    switching to a heavier base model from the recipe's
    ``alt_base_models`` list. The granular signal-level surface
    lives in ``TrainabilityForecastPanel`` (the page-level component)
    — Coach is the "should I be worried?" overlay above it.

    Phase 6d adds an info-severity nudge for thin classification
    projects (curriculum learning is on by default for them via the
    backend heuristic; this card surfaces the *why*).
    """
    # Local import: trainability_forecast_service is a heavy module
    # (large recipe + tokenizer transitive imports). Keeping it lazy
    # avoids hoisting that cost into every Coach-module import.
    from app.services.recipe_service import get_recipe
    from app.services.trainability_forecast_service import (
        KNOWN_BASE_MODEL_PARAMS_M,
        forecast_training,
    )

    suggestions: list[dict[str, Any]] = []

    recipe_id = _recipe_id_for(project)
    if not recipe_id:
        # No recipe yet → can't forecast. Same fallback shape used by
        # the data + gold_set stages keeps the navigate-to-picker
        # affordance consistent across surfaces.
        return [{
            "id": "training:no-recipe",
            "title": "Pick a recipe before training",
            "body": (
                "The trainability forecast (L1) needs a recipe selected so it "
                "can score predicted F1 against the recipe's task difficulty + "
                "minimum row count."
            ),
            "severity": "info",
            "action": {
                "kind": "navigate",
                "label": "Open recipe picker",
                "params": {"target": "recipe-picker"},
            },
        }]

    # Phase 6d — curriculum-learning nudge runs independently of the
    # trainability forecast (curriculum can lift F1 whether forecast
    # says pass / borderline / fail). Append now so it's surfaced
    # even when the forecast is silent.
    curriculum_nudge = _curriculum_training_suggestion(project.id, recipe_id)
    if curriculum_nudge:
        suggestions.append(curriculum_nudge)

    # Quality-Lift phase 3 slice 2 — active-learning nudge fires
    # independently of the forecast logic too. The slice 1 post-training
    # hook stamped a snapshot of the most-uncertain unlabeled rows onto
    # ``_runtime["active_learning"]``; this surfaces it as a Coach
    # suggestion that links to the existing label queue with
    # assign_strategy=active. Silences automatically when ≥ 80% of the
    # snapshot has been labeled, so users who already worked through
    # the queue don't see a stale suggestion.
    active_learning_nudge = await _active_learning_ready_nudge(db, project.id)
    if active_learning_nudge:
        suggestions.append(active_learning_nudge)

    # Quality-Lift phase 7 slice 3 — multi-seed variance nudge. Fires
    # when the latest run was single-seed AND either (a) a prior
    # multi-seed run on this project measured high relative std on a
    # gated metric (variance is *hidden* by going back to one seed),
    # or (b) the project has run eval but never run multi-seed
    # (variance is *unknown*). Both nudges deep-link to the
    # training-config page with the multi-seed section auto-expanded.
    variance_nudge = await _multi_seed_variance_nudge(db, project.id)
    if variance_nudge:
        suggestions.append(variance_nudge)

    # Phase 8c — archetype-drift nudge runs alongside the curriculum
    # nudge and BEFORE the forecast logic (so it surfaces even when
    # forecast is likely_pass and returns early below). Different
    # framing of partially overlapping concerns: forecast = "will
    # this run pass the gate?", archetype = "does your data look
    # like successful data?". Both can render together.
    comparison = await _load_archetype_comparison_safe(db, project.id)
    archetype_nudge = _archetype_drift_nudge(project.id, "training", comparison)
    if archetype_nudge:
        suggestions.append(archetype_nudge)

    try:
        forecast = await forecast_training(db, project.id)
    except ValueError:
        # The forecast service raises for missing project/recipe;
        # we caught the recipe case above, so this branch covers
        # the rare "project disappeared between the project.get and
        # the forecast call" race. Surface what we have rather than 500.
        return suggestions

    overall = str(forecast.get("overall", ""))
    if overall == "likely_pass":
        # User is in the green zone — no forecast suggestion. The
        # forecast panel itself still shows the signals so they can
        # act if they want. The curriculum nudge (if any) stays.
        return suggestions

    severity: Severity = "critical" if overall == "likely_fail" else "warning"

    # Pick a heavier alt base model if the recipe offers one. We rank
    # by KNOWN_BASE_MODEL_PARAMS_M when known; unknown models fall to
    # the end. Skip alts whose parameter count is <= the current
    # model's so we never recommend a sideways or backward swap.
    recipe = get_recipe(recipe_id)
    current_base = (
        project.base_model_name
        or (getattr(recipe, "suggested_base_model", None) if recipe else None)
        or ""
    )
    alts: list[str] = (
        list(getattr(recipe, "alt_base_models", []) or []) if recipe else []
    )
    current_params = KNOWN_BASE_MODEL_PARAMS_M.get(current_base, 0)

    def _params_for(name: str) -> int:
        return KNOWN_BASE_MODEL_PARAMS_M.get(name, 0)

    heavier_alts = sorted(
        [a for a in alts if _params_for(a) > current_params],
        key=_params_for,
    )
    recommended_base = heavier_alts[0] if heavier_alts else None

    confidence_pct = int(forecast.get("confidence_pct") or 0)
    dominant_blocker: dict[str, Any] | None = None
    for signal in forecast.get("signals", []):
        if signal.get("severity") == "block":
            dominant_blocker = signal
            break

    if recommended_base:
        action: dict[str, Any] = {
            "kind": "navigate",
            "label": f"Consider {recommended_base}",
            "params": {
                "target": "training-base-model-picker",
                "recommended_base_model": recommended_base,
            },
        }
        action_hint = (
            f" The recipe's next-heaviest alternative is "
            f"{recommended_base} ({_params_for(recommended_base) or '?'}M params)."
        )
    else:
        action = {
            "kind": "navigate",
            "label": "Open trainability forecast",
            "params": {"target": "trainability-forecast"},
        }
        action_hint = ""

    if overall == "likely_fail":
        title = (
            f"Forecast says this run is likely to fail "
            f"({confidence_pct}% confidence in passing)"
        )
        body = (
            (dominant_blocker["headline"] + " " if dominant_blocker else "")
            + "Either lift the gold set (more rows, better balance) or move "
            "to a stronger base model — the current pairing isn't projected "
            "to pass the auto-gate."
            + action_hint
        )
    else:  # borderline
        title = (
            f"Forecast says this run is borderline "
            f"({confidence_pct}% pass-rate confidence)"
        )
        body = (
            (dominant_blocker["headline"] + " " if dominant_blocker else "")
            + "You can train as-is and find out, but small data or base-model "
            "improvements would meaningfully shift the odds before you spend "
            "compute." + action_hint
        )

    suggestions.append({
        "id": "training:trainability-forecast",
        "title": title,
        "body": body.strip(),
        "severity": severity,
        "action": action,
        "context": {
            "overall": overall,
            "confidence_pct": confidence_pct,
            "current_base_model": current_base,
            "recommended_base_model": recommended_base,
            "blocker_signal_id": (
                dominant_blocker["id"] if dominant_blocker else None
            ),
        },
    })

    # Sweep-inconclusive nudge — surface the latest hyperparameter sweep
    # whose verdict is "inconclusive" so the user doesn't have to keep
    # the panel open to learn that nobody cleared the gate. See
    # _inconclusive_sweep_nudge below.
    sweep_nudge = await _inconclusive_sweep_nudge(db, project.id)
    if sweep_nudge:
        suggestions.append(sweep_nudge)

    return suggestions


_VARIANCE_REL_STD_THRESHOLD = 0.10


def _find_worst_relative_std(
    metrics: dict[str, Any],
) -> tuple[str, float, float] | None:
    """Walk an aggregate row's metrics dict, returning
    ``(name, std, mean)`` for the metric with the highest std/|mean|
    ratio that also exceeds ``_VARIANCE_REL_STD_THRESHOLD``.

    Recurses into nested dicts so per-class metrics (introduced by the
    Gap-#6 work) participate too — e.g. ``classes.positive.precision``
    is a valid leaf path. ``None`` when no leaf exceeds the threshold.

    The threshold is deliberately conservative — 10% relative std is
    far enough above pure noise that "your gate verdict could flip on
    a re-run" is a falsifiable claim. The nudge body cites the actual
    mean/std so the user can audit it (no-vanity-metrics rule).
    """
    worst: tuple[str, float, float] | None = None
    worst_ratio = _VARIANCE_REL_STD_THRESHOLD

    def walk(d: dict[str, Any], prefix: str = "") -> None:
        nonlocal worst, worst_ratio
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                if "mean" in v and "std" in v:
                    try:
                        mean_f = float(v["mean"])
                        std_f = float(v["std"])
                    except (TypeError, ValueError):
                        continue
                    abs_mean = abs(mean_f)
                    if abs_mean < 1e-6:
                        continue
                    ratio = std_f / abs_mean
                    if ratio > worst_ratio:
                        worst_ratio = ratio
                        worst = (path, std_f, mean_f)
                else:
                    walk(v, prefix=path)

    walk(metrics)
    return worst


async def _multi_seed_variance_nudge(
    db: AsyncSession, project_id: int,
) -> dict[str, Any] | None:
    """Quality-Lift phase 7 slice 3 — surface multi-seed authoring so
    the variance gates phases 1+ shipped become user-reachable without
    hand-rolling the API.

    Two nudge IDs share a single surface:

    * ``training:variance-hidden`` (warning) — a prior multi-seed
      aggregate row on this project showed ≥10% relative std on at
      least one metric AND the latest run was single-seed. Re-running
      with one seed hides variance the user already proved exists; the
      gate's mean−std lower bound goes back to looking artificially
      tight.
    * ``training:variance-unknown`` (info) — the project has a
      completed single-seed run with eval results but has never run
      multi-seed. The verdict is unmeasured, not safe.

    Silences when the latest run was already multi-seed (the user is
    doing the right thing — re-asserting it would be noise) or when
    no eval has ever run (nothing to compare against). Both nudges
    deep-link to the training-config page with the multi-seed
    section auto-expanded (``params.expand_multi_seed=True``).
    """
    # Lazy imports — Experiment / EvalResult pull in a deeper graph
    # than the rest of coach_service touches; avoid hoisting that
    # cost into every coach module import.
    from app.models.experiment import (
        EvalResult,
        Experiment,
        ExperimentStatus,
    )

    terminal_statuses = (
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
    )

    # 1. Find the latest terminal experiment to learn whether the
    # user is currently doing single- or multi-seed.
    latest = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(Experiment.status.in_(terminal_statuses))
        .order_by(Experiment.created_at.desc())
        .limit(1)
    )
    latest_exp = latest.scalar_one_or_none()
    if latest_exp is None:
        # No training has happened yet — the multi-seed surface
        # appears in TrainingPanel itself; no need to nudge.
        return None
    if latest_exp.seed_group_id:
        # Most recent run was multi-seed → user already doing the
        # right thing. Silence so the nudge isn't ambient noise.
        return None

    # 2. Hard signal — any prior aggregate row on this project
    # showing high relative std on a metric? Walk the most recent 5
    # aggregates so a one-off noisy run doesn't get drowned out by
    # newer well-behaved ones.
    agg_result = await db.execute(
        select(EvalResult)
        .join(Experiment, EvalResult.experiment_id == Experiment.id)
        .where(Experiment.project_id == project_id)
        .where(EvalResult.is_aggregate.is_(True))
        .order_by(EvalResult.created_at.desc())
        .limit(5)
    )
    worst_finding: tuple[str, float, float] | None = None
    for agg in agg_result.scalars():
        metrics = agg.metrics if isinstance(agg.metrics, dict) else {}
        found = _find_worst_relative_std(metrics)
        if found is not None:
            worst_finding = found
            break

    if worst_finding is not None:
        name, std_val, mean_val = worst_finding
        rel_pct = (std_val / max(abs(mean_val), 1e-9)) * 100.0
        return {
            "id": "training:variance-hidden",
            "title": (
                f"Last run was single-seed but {name} had "
                f"{rel_pct:.1f}% std across seeds in a prior run"
            ),
            "body": (
                f"The {name} metric showed std={std_val:.3f} on "
                f"mean={mean_val:.3f} the last time you ran "
                "multi-seed — a {rel_pct:.1f}% relative spread. Going "
                "back to num_seeds=1 hides that variance: the gate's "
                "mean−std lower bound (phase 1) goes back to looking "
                "artificially tight, and a re-run could flip the "
                "verdict. Set num_seeds≥3 to keep the verdict honest."
            ).format(rel_pct=rel_pct),
            "severity": "warning",
            "action": {
                "kind": "navigate",
                "label": "Open multi-seed config",
                "params": {
                    "target": "training-config",
                    "expand_multi_seed": True,
                    "suggested_num_seeds": 3,
                },
            },
            "context": {
                "worst_metric": name,
                "std": float(std_val),
                "mean": float(mean_val),
                "rel_std_pct": rel_pct,
            },
        }

    # 3. Soft signal — has the project ever produced an EvalResult on
    # a single-seed experiment? If yes, ask the user to measure
    # variance. If no eval has run at all there's nothing to nudge on.
    eval_exists = await db.execute(
        select(EvalResult.id)
        .join(Experiment, EvalResult.experiment_id == Experiment.id)
        .where(Experiment.project_id == project_id)
        .where(EvalResult.is_aggregate.is_(False))
        .limit(1)
    )
    if eval_exists.first() is None:
        return None
    return {
        "id": "training:variance-unknown",
        "title": "Last run was single-seed — variance is unmeasured",
        "body": (
            "The gate verdicts on your latest run came from a single "
            "seed, so you don't know whether each metric reflects the "
            "model or the seed. Re-run with num_seeds=3 to get a "
            "mean−std lower-bound verdict (no-vanity-metrics rule)."
        ),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open multi-seed config",
            "params": {
                "target": "training-config",
                "expand_multi_seed": True,
                "suggested_num_seeds": 3,
            },
        },
    }


async def _inconclusive_sweep_nudge(
    db: AsyncSession, project_id: int
) -> dict[str, Any] | None:
    """Coach card for the latest hyperparameter sweep when its verdict is
    ``inconclusive`` — i.e. every completed cell has eval results but none
    cleared the project's gate. Stays silent on ``promote`` and ``pending``
    verdicts (the sweep panel itself surfaces those; coach repeating them
    would just be noise).

    Why this lives in the training stage: sweeps are launched from
    the training-config Power Tools and the panel that shows them
    (``HyperparameterSweepPanel``) is mounted there, so the user is
    already looking at the right surface.
    """
    from sqlalchemy import select as _select

    from app.models.experiment import Experiment
    from app.models.sweep import Sweep
    from app.services.hyperparameter_sweep_service import get_sweep_pareto

    # Primary lookup: one-row hit on the sweeps table, ordered by
    # created_at desc. The legacy fallback below survives for tests +
    # pre-migration projects that don't have Sweep rows yet.
    sweep_result = await db.execute(
        _select(Sweep)
        .where(Sweep.project_id == project_id)
        .order_by(Sweep.created_at.desc())
        .limit(1)
    )
    latest_sweep_record = sweep_result.scalar_one_or_none()
    latest_sweep_id: str | None = (
        latest_sweep_record.sweep_id if latest_sweep_record is not None else None
    )

    if latest_sweep_id is None:
        # Legacy fallback: scan experiments for the most-recent
        # config._sweep.sweep_id breadcrumb. Only fires when no Sweep
        # row exists yet (pre-migration data / tests).
        result = await db.execute(
            _select(Experiment)
            .where(Experiment.project_id == project_id)
            .order_by(Experiment.id.desc())
        )
        for exp in result.scalars():
            meta = (exp.config or {}).get("_sweep") or {}
            sid = str(meta.get("sweep_id") or "").strip()
            if sid:
                latest_sweep_id = sid
                break
    if latest_sweep_id is None:
        return None

    try:
        pareto = await get_sweep_pareto(db, project_id, latest_sweep_id)
    except ValueError:
        return None

    if pareto.get("verdict") != "inconclusive":
        return None

    # Pull the failed gate IDs across cells so the card body can name them.
    # The vast majority of inconclusive sweeps fail the same gate across
    # every cell (one gate dominates), so dedup + sort + show the top 3.
    failed_gates: dict[str, int] = {}
    for cell in pareto.get("cells", []):
        if cell.get("gate_passed") is False:
            for gid in cell.get("gate_failed_ids") or []:
                if isinstance(gid, str) and gid:
                    failed_gates[gid] = failed_gates.get(gid, 0) + 1
    top_gates = sorted(failed_gates.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    gate_blurb = ""
    if top_gates:
        # Format as "acc_gte_0.8 (4 cells)" — the count tells the user
        # whether one gate dominates or the failure spreads across many.
        rendered = ", ".join(
            f"{name} ({count} cell{'s' if count != 1 else ''})"
            for name, count in top_gates
        )
        gate_blurb = f" Gate(s) missed: {rendered}."

    measurable = int((pareto.get("gate_summary") or {}).get("measurable_count") or 0)
    cell_count = int(pareto.get("cell_count") or 0)

    body = (
        f"Sweep {latest_sweep_id} finished with no cell clearing the project "
        f"gate ({measurable}/{cell_count} measurable)."
        + gate_blurb
        + " Failure clusters explain why each cell missed; promoting any of "
        "these would just ship a sub-gate model."
    )

    return {
        "id": "training:sweep-inconclusive",
        "title": (
            f"Sweep inconclusive — {measurable}/{cell_count} cells, none cleared gate"
        ),
        "body": body,
        "severity": "warning",
        "action": {
            "kind": "navigate",
            "label": "Open Failure clusters",
            "params": {"target": "failure-clusters-panel"},
        },
        "context": {
            "sweep_id": latest_sweep_id,
            "verdict": "inconclusive",
            "measurable_count": measurable,
            "cell_count": cell_count,
            "failed_gates": top_gates,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Eval stage (Phase 3).
# ─────────────────────────────────────────────────────────────────────


async def _read_latest_eval_result(
    db: AsyncSession, project_id: int
) -> Any | None:
    """Latest ``EvalResult`` row across every Experiment of the project,
    ordered by ``created_at`` desc. Returns ``None`` when no eval has
    been recorded yet.

    Local import on ``Experiment`` + ``EvalResult`` keeps the coach
    service from pulling the heavy experiment + checkpoint graph at
    module import time.
    """
    from app.models.experiment import EvalResult, Experiment

    result = await db.execute(
        select(EvalResult)
        .join(Experiment, Experiment.id == EvalResult.experiment_id)
        .where(Experiment.project_id == project_id)
        .order_by(EvalResult.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()


def _reroute_recommendation_nudge(
    project_id: int,
    reroute_analysis: dict[str, Any] | None,
    *,
    is_rag_first_project: bool,
) -> dict[str, Any] | None:
    """Phase 7d — info-severity nudge on the Eval tab that surfaces
    Phase 7a's post-eval reroute recommendation as a Coach suggestion.

    Fires only when:
      * the latest eval's reroute analysis exists AND
      * the recommendation kind is ``try_rag`` AND
      * the project isn't already a RAG-first clone (don't recommend
        what's already on; avoids an infinite reroute chain)

    Cites the fired signal ``detail`` strings in the body so the
    recommendation is auditable from inside the Coach card without
    requiring the user to scroll to the EvalPanel surface.

    Returns None on any skip condition. Pure function — caller is
    responsible for loading the analysis from the cache /
    recomputing.
    """
    if is_rag_first_project:
        return None
    if not isinstance(reroute_analysis, dict):
        return None
    recommendation = reroute_analysis.get("recommendation") or {}
    if not isinstance(recommendation, dict):
        return None
    kind = recommendation.get("kind")
    if kind != "try_rag":
        return None

    signals = reroute_analysis.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    fired = [s for s in signals if isinstance(s, dict) and s.get("fired")]
    fired_ids = [str(s.get("id") or "") for s in fired if s.get("id")]

    pass_rate = reroute_analysis.get("pass_rate")
    pass_rate_str = (
        f"{pass_rate * 100:.0f}%"
        if isinstance(pass_rate, (int, float))
        else "below the healthy threshold"
    )

    # Evidence lines — drop the analyzer's verbose detail strings
    # into a bullet-style block so the Coach card is self-contained.
    if fired:
        evidence_lines = "\n".join(
            f"• {str(s.get('detail') or '').strip()}"
            for s in fired
            if str(s.get("detail") or "").strip()
        )
    else:
        evidence_lines = ""

    body_parts = [
        (
            f"Your eval is at {pass_rate_str}. The post-eval analyzer "
            f"thinks this task looks more like a RAG fit than SFT."
        ),
    ]
    if evidence_lines:
        body_parts.append("Signals that fired:\n" + evidence_lines)
    body_parts.append(
        "Switching creates a sibling project that uses the base model + "
        "retrieval from your gold set — no training run required. Your "
        "current SFT project stays intact for comparison."
    )

    return {
        "id": "eval:reroute-to-rag-recommended",
        "title": f"Reroute to RAG? Your eval is at {pass_rate_str}",
        "body": "\n\n".join(body_parts),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open reroute panel",
            "params": {"target": "reroute-recommendation-panel"},
        },
        "context": {
            "project_id": project_id,
            "pass_rate": (
                round(float(pass_rate), 4)
                if isinstance(pass_rate, (int, float))
                else None
            ),
            "fired_signal_ids": fired_ids,
            "recommendation_confidence": recommendation.get("confidence"),
            "eval_result_id": reroute_analysis.get("eval_result_id"),
        },
    }


def _auto_rag_eval_nudge(
    project_id: int,
    recipe_id: str | None,
    pass_rate: float | None,
    latest_experiment_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Phase 9d — info-severity nudge on the Eval tab when:
    - project recipe is qa-sft (RAG-eligible)
    - latest eval pass_rate < 0.5 (the model is struggling)
    - the experiment that produced it has ``auto_rag.enabled`` either
      missing or false (so flipping auto-RAG on for the next training
      is a concrete next step)

    Cites the Phase 9c A/B numbers verbatim so the recommendation is
    auditable. Returns None on any skip condition (no double-coaching
    on top of the existing pass-rate suggestion)."""
    if not recipe_id or pass_rate is None:
        return None
    from app.services.auto_rag_service import recommended_text_keys_for_recipe

    if recommended_text_keys_for_recipe(recipe_id) is None:
        return None  # recipe not RAG-eligible
    if pass_rate >= 0.5:
        return None  # only fire when the model is meaningfully struggling
    # Check whether the experiment had auto_rag on. ``auto_rag`` lives
    # at config["auto_rag"]["enabled"] (Phase 9d default-on shape) but
    # we also tolerate the legacy ``config.auto_rag == True`` shape
    # for forward-compat with whatever the playground request used.
    cfg = latest_experiment_config or {}
    auto_rag_block = cfg.get("auto_rag")
    auto_rag_on = False
    if isinstance(auto_rag_block, dict):
        auto_rag_on = bool(auto_rag_block.get("enabled"))
    elif isinstance(auto_rag_block, bool):
        auto_rag_on = auto_rag_block
    if auto_rag_on:
        return None  # already on — no nudge

    return {
        "id": "eval:auto-rag-recommended",
        "title": (
            f"QA model at {pass_rate * 100:.0f}% pass rate — try auto-RAG "
            f"on the next training run"
        ),
        "body": (
            "Auto-RAG retrieves relevant (Q, A) pairs from your training "
            "corpus at inference time and prepends them as context. "
            "Phase 9c A/B (2026-05-25, 5 seeds, GB10) measured "
            "**+146% F1 lift on the policy-qa-style QA-SFT template** "
            "vs the SFT-only baseline. New qa-sft experiments now default "
            "to auto-RAG on; this one didn't, so the next experiment "
            "will pick it up automatically. Or flip it on for an "
            "individual playground chat without re-training."
        ),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open Training Config",
            "params": {"target": "training-config"},
        },
        "context": {
            "pass_rate": round(float(pass_rate), 4),
            "phase_9c_lift_pct": 146.49,
            "phase_9c_template": "policy-qa-style",
            "ab_run_date": "2026-05-25",
        },
    }


# Recipe IDs where per-class gates are meaningful. Classification +
# span-extraction both emit per-class metrics through the handler;
# other recipes don't have a "class" concept at the eval-row level.
_PER_CLASS_RECIPE_IDS: frozenset[str] = frozenset({
    "classification",
    "span-extraction",
})


async def _missing_per_class_gates_nudge(
    db: AsyncSession, project_id: int, recipe_id: str | None
) -> dict[str, Any] | None:
    """Gap-#6 slice 3 — nudge classification projects whose active
    eval pack has zero per-class gates configured.

    Three conditions must all hold:
      1. Recipe is classification-shaped (per-class metrics make sense).
      2. The latest eval has emitted ``per_class`` data — i.e. classes
         have been discovered. Without that we'd just be nagging users
         to add gates against metric IDs the editor can't help pick.
      3. No existing gate in the active pack already references a
         per-class metric ID.

    Severity is ``info`` — running a classification eval without
    per-class gates isn't broken, it's just leaving signal on the
    floor (macro F1 can stay healthy while a rare class collapses).

    Returns None on any lookup failure — the eval-tab Coach must not
    break when a pack reference goes stale or a plugin import errors.
    """
    if (recipe_id or "").strip().lower() not in _PER_CLASS_RECIPE_IDS:
        return None

    # Lazy imports — these modules pull in the eval engine and
    # SQLAlchemy plumbing the coach module doesn't otherwise need.
    from app.services.evaluation_gate_catalog import (
        build_per_class_metric_options,
        is_per_class_metric_id,
    )
    from app.services.evaluation_pack_service import (
        resolve_project_evaluation_pack,
    )

    # Condition 2 — classes discovered. If the project has no eval
    # results yet, the editor's hint covers the case; no need for the
    # Coach to also nag.
    try:
        per_class = await build_per_class_metric_options(
            db, project_id=project_id,
        )
    except Exception:  # noqa: BLE001 — never break the eval-tab Coach
        return None
    classes = per_class.get("classes") or []
    if not classes:
        return None

    # Condition 3 — walk the active pack's gates, abort if any
    # already references a per-class metric ID.
    try:
        resolved = await resolve_project_evaluation_pack(db, project_id)
    except Exception:  # noqa: BLE001
        return None
    pack = resolved.get("pack") if isinstance(resolved, dict) else None
    if not isinstance(pack, dict):
        return None

    task_specs = pack.get("task_specs") if isinstance(pack.get("task_specs"), list) else []
    for spec in task_specs:
        if not isinstance(spec, dict):
            continue
        for gate in spec.get("gates") or []:
            if not isinstance(gate, dict):
                continue
            metric_id = str(gate.get("metric_id") or "").strip().lower()
            if metric_id and is_per_class_metric_id(metric_id):
                # At least one per-class gate already configured — no
                # need to nudge.
                return None

    # All three conditions met → emit the nudge. Truncate the class
    # list in the body so projects with 20+ classes don't get a wall
    # of text.
    sample = ", ".join(f"`{c}`" for c in classes[:3])
    more = f" and {len(classes) - 3} more" if len(classes) > 3 else ""
    return {
        "id": "eval:no-per-class-gates",
        "title": (
            f"Your eval pack doesn't gate per-class metrics — "
            f"{len(classes)} class(es) discovered"
        ),
        "body": (
            f"Macro F1 alone can hide a rare class collapsing — a 90/10 "
            f"split can hit 0.9 accuracy with `f1_minority` near zero. "
            f"Your latest eval surfaced classes {sample}{more}. Adding "
            f"a `precision_<class>` or `recall_<class>` gate in the eval "
            f"pack editor catches the regression the moment it shows up."
        ),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open eval pack editor",
            "params": {"target": "eval-pack-editor"},
        },
        "rule_id": "per-class-gates.absent",
        "context": {
            "discovered_classes": list(classes),
            "active_pack_id": resolved.get("active_pack_id"),
            "source_eval_result_id": per_class.get("source_eval_result_id"),
        },
    }


async def _behavioral_tests_without_per_slice_gates_nudge(
    db: AsyncSession, project_id: int,
) -> dict[str, Any] | None:
    """Quality-Lift phase 6 slice 3 — Sibling of the
    ``eval:behavioral-tests-without-gates`` nudge for the per-slice
    case. Fires when:

      * The project has ``slice_definitions`` configured (phase 2).
      * The active pack has ``task_specs[].behavioral_tests`` defined
        (phase 5).
      * AND no gate references the per-slice metric_id shape
        (``behavioral.<test>.per_slice.<slice>.<metric>``).

    The user has all the pieces — slices, behavioral tests, the runner
    is emitting per-slice scores — but the ship decision still ignores
    them. An INV test might pass at 0.92 overall but fail at 0.55 on
    ``long_input``; without a per-slice gate that regression won't
    block ship.

    Severity info — same shape as the existing un-gated-tests nudge.
    Returns None on any lookup failure (eval-tab Coach never breaks).
    """
    from app.services.evaluation_gate_catalog import is_behavioral_metric_id
    from app.services.evaluation_pack_service import (
        resolve_project_evaluation_pack,
    )
    from app.models.project import Project

    # Condition 1 — project has slice definitions.
    project_row = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = project_row.scalar_one_or_none()
    if project is None:
        return None
    slices_raw = (project.slice_definitions or {}).get("slices") \
        if isinstance(project.slice_definitions, dict) else None
    if not isinstance(slices_raw, list) or not slices_raw:
        return None
    slice_ids = [
        str(s.get("slice_id") or "").strip()
        for s in slices_raw
        if isinstance(s, dict)
    ]
    slice_ids = [sid for sid in slice_ids if sid]
    if not slice_ids:
        return None

    # Conditions 2 + 3 — walk the active pack.
    try:
        resolved = await resolve_project_evaluation_pack(db, project_id)
    except Exception:  # noqa: BLE001
        return None
    pack = resolved.get("pack") if isinstance(resolved, dict) else None
    if not isinstance(pack, dict):
        return None

    task_specs = pack.get("task_specs") if isinstance(pack.get("task_specs"), list) else []
    test_ids: list[str] = []
    has_per_slice_gate = False
    for spec in task_specs:
        if not isinstance(spec, dict):
            continue
        for entry in spec.get("behavioral_tests") or []:
            if isinstance(entry, dict):
                test_id = str(entry.get("test_id") or "").strip()
                if test_id:
                    test_ids.append(test_id)
        for gate in spec.get("gates") or []:
            if not isinstance(gate, dict):
                continue
            metric_id = str(gate.get("metric_id") or "").strip().lower()
            if not metric_id or not is_behavioral_metric_id(metric_id):
                continue
            # Per-slice metric_id has ``.per_slice.`` in the path.
            # Top-level behavioral gates don't trigger this nudge —
            # the sibling ``eval:behavioral-tests-without-gates`` nudge
            # handles the "no behavioral gates at all" case.
            if ".per_slice." in metric_id:
                has_per_slice_gate = True

    if not test_ids:
        return None
    if has_per_slice_gate:
        return None

    # Suggest the first (test_id × slice_id) cross-product as a
    # concrete example in the body — saves the user from staring at
    # the editor wondering "where do I start?".
    sample_test = test_ids[0]
    sample_slice = slice_ids[0]
    suggested_metric_id = f"behavioral.{sample_test}.per_slice.{sample_slice}.pass_rate"

    return {
        "id": "eval:behavioral-tests-without-per-slice-gates",
        "title": (
            f"You have {len(test_ids)} behavioral test{'s' if len(test_ids) != 1 else ''} "
            f"and {len(slice_ids)} slice{'s' if len(slice_ids) != 1 else ''} but no "
            "per-slice gates"
        ),
        "body": (
            "Your behavioral tests run per-slice now (an INV test might pass at "
            "0.92 overall but fail at 0.55 on a specific slice). Without a "
            "per-slice gate, that regression won't block ship. Add a gate like "
            f"``{suggested_metric_id} >= 0.85`` in the eval pack editor to "
            "enforce robustness per slice."
        ),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open eval pack editor",
            "params": {"target": "eval-pack-editor"},
        },
        "rule_id": "behavioral-tests.per-slice-ungated",
        "context": {
            "behavioral_test_ids": list(test_ids),
            "slice_ids": list(slice_ids),
            "suggested_metric_id": suggested_metric_id,
            "active_pack_id": resolved.get("active_pack_id"),
        },
    }


async def _behavioral_tests_without_gates_nudge(
    db: AsyncSession, project_id: int,
) -> dict[str, Any] | None:
    """Quality-Lift phase 5 slice 3 — Mirror Gap-#6's no-per-class-gates
    nudge for behavioral tests. Fires when the active eval pack has
    ``task_specs[].behavioral_tests`` defined but no gate references
    them — the user wrote the tests but the ship decision still
    ignores them.

    Severity ``info`` — the tests are still running and surfacing in
    the scorecard, but a failing INV or MFT won't block ship unless a
    gate gates on it. The nudge points at the eval pack editor where
    the user can add a ``behavioral.<test_id>.pass_rate`` gate.
    Silences when there are no behavioral tests OR when at least one
    behavioral gate already exists.

    Returns None on any lookup failure — the eval-tab Coach must not
    break when a pack reference goes stale or a plugin import errors.
    """
    from app.services.evaluation_gate_catalog import is_behavioral_metric_id
    from app.services.evaluation_pack_service import (
        resolve_project_evaluation_pack,
    )

    try:
        resolved = await resolve_project_evaluation_pack(db, project_id)
    except Exception:  # noqa: BLE001
        return None
    pack = resolved.get("pack") if isinstance(resolved, dict) else None
    if not isinstance(pack, dict):
        return None

    task_specs = pack.get("task_specs") if isinstance(pack.get("task_specs"), list) else []
    test_ids: list[str] = []
    has_behavioral_gate = False
    for spec in task_specs:
        if not isinstance(spec, dict):
            continue
        for entry in spec.get("behavioral_tests") or []:
            if isinstance(entry, dict):
                test_id = str(entry.get("test_id") or "").strip()
                if test_id:
                    test_ids.append(test_id)
        for gate in spec.get("gates") or []:
            if not isinstance(gate, dict):
                continue
            metric_id = str(gate.get("metric_id") or "").strip().lower()
            if metric_id and is_behavioral_metric_id(metric_id):
                has_behavioral_gate = True

    if not test_ids:
        return None
    if has_behavioral_gate:
        return None

    sample = ", ".join(f"`{t}`" for t in test_ids[:3])
    more = f" and {len(test_ids) - 3} more" if len(test_ids) > 3 else ""
    return {
        "id": "eval:behavioral-tests-without-gates",
        "title": (
            f"You have {len(test_ids)} behavioral test{'s' if len(test_ids) != 1 else ''} "
            "but no gates referencing them"
        ),
        "body": (
            f"Behavioral tests {sample}{more} run on every eval and surface in the "
            "scorecard, but a failing test won't block ship until you add a gate "
            "on ``behavioral.<test_id>.pass_rate``. Without a gate, robustness "
            "regressions are visible but not enforced."
        ),
        "severity": "info",
        "action": {
            "kind": "navigate",
            "label": "Open eval pack editor",
            "params": {"target": "eval-pack-editor"},
        },
        "rule_id": "behavioral-tests.ungated",
        "context": {
            "behavioral_test_ids": list(test_ids),
            "active_pack_id": resolved.get("active_pack_id"),
        },
    }


async def _eval_stage_suggestions(
    db: AsyncSession, project: Project
) -> list[dict[str, Any]]:
    """Suggestions surfaced on the Eval tab.

    Phase 3 reads the project's latest ``EvalResult`` and — when the
    pass rate is below the healthy threshold — clusters the failures
    via ``cluster_eval_result_failures`` (Epic 2b primitive) and
    surfaces a click-to-execute ``augment_from_cluster`` action
    targeting the largest cluster.

    Phase 9d adds an info-severity auto-RAG nudge for struggling
    qa-sft projects (fires independently of the failure-cluster
    suggestion; both can render together).
    """
    from app.services.failure_cluster_service import (
        cluster_eval_result_failures,
    )

    suggestions: list[dict[str, Any]] = []

    latest = await _read_latest_eval_result(db, project.id)
    if latest is None:
        # No eval run yet — nothing to coach against. The strip stays
        # mounted (showing "looks healthy") which doubles as a hint
        # that running an eval is the next move.
        return []

    pass_rate = latest.pass_rate

    # Phase 9d — auto-RAG nudge runs independently of the failure-
    # cluster suggestion (different lever on different timeline).
    # Append now so it surfaces even when the cluster path no-ops.
    recipe_id = _recipe_id_for(project)
    # Read the latest experiment's config — needed to check whether
    # auto_rag was already enabled. Best-effort; if the lookup
    # fails, the nudge gates as "not enabled" which is the safe
    # default (worst case: false-positive nudge).
    latest_exp_config: dict[str, Any] | None = None
    try:
        from app.models.experiment import Experiment

        exp_row = await db.execute(
            select(Experiment)
            .where(Experiment.id == latest.experiment_id)
        )
        exp = exp_row.scalar_one_or_none()
        if exp is not None:
            latest_exp_config = dict(exp.config or {})
    except Exception:  # noqa: BLE001 — never block the eval strip on this
        latest_exp_config = None
    nudge = _auto_rag_eval_nudge(
        project.id, recipe_id, pass_rate, latest_exp_config
    )
    if nudge:
        suggestions.append(nudge)

    # Phase 7d — surface the post-eval reroute recommendation as a
    # Coach suggestion. Pulls Phase 7a's RerouteAnalysis (cached on
    # EvalResult.details["reroute_analysis"]). Skipped when the
    # project is already a RAG-first clone — no point recommending
    # what's already on; avoids an infinite reroute chain on
    # rag_first siblings whose evals re-trigger the analyzer.
    runtime_cfg = project.runtime_config if hasattr(project, "runtime_config") else None
    is_rag_first_project = bool(
        isinstance(runtime_cfg, dict) and runtime_cfg.get("rag_first") is True
    )
    reroute_analysis: dict[str, Any] | None = None
    try:
        # Read the cache directly from EvalResult.details first — the
        # analyzer runs on Eval-tab mount via Phase 7c's panel, so the
        # cache is almost always warm. Fall through to the analyzer
        # only when no cache exists (first-time eval, fresh project).
        latest_details = dict(latest.details or {}) if isinstance(latest.details, dict) else {}
        cached = latest_details.get("reroute_analysis")
        if isinstance(cached, dict) and cached.get("eval_result_id") == latest.id:
            reroute_analysis = cached
        else:
            from app.services.post_eval_decision_engine_service import (
                analyze_eval_for_reroute,
            )

            reroute_analysis = await analyze_eval_for_reroute(
                db, eval_result_id=latest.id
            )
    except Exception:  # noqa: BLE001 — never block the eval strip on this
        reroute_analysis = None

    reroute_nudge = _reroute_recommendation_nudge(
        project.id,
        reroute_analysis,
        is_rag_first_project=is_rag_first_project,
    )
    if reroute_nudge:
        suggestions.append(reroute_nudge)

    # Gap-#6 slice 3 — nudge classification projects whose eval pack
    # has zero per-class gates to add one. Fires independently of the
    # pass-rate check above so it surfaces even on healthy projects
    # whose macro F1 is hiding a class-imbalance problem. Best-effort:
    # if any of the lookups error, the nudge silently skips rather
    # than breaking the eval-tab Coach.
    per_class_nudge = await _missing_per_class_gates_nudge(
        db, project.id, recipe_id,
    )
    if per_class_nudge:
        suggestions.append(per_class_nudge)

    # Quality-Lift phase 5 slice 3 — Parallel "you wrote behavioral
    # tests but no gate references them" nudge. Independent of the
    # pass-rate logic below because it fires regardless of how well
    # the model is doing — un-gated tests means robustness regressions
    # are visible but not enforced.
    behavioral_nudge = await _behavioral_tests_without_gates_nudge(
        db, project.id,
    )
    if behavioral_nudge:
        suggestions.append(behavioral_nudge)

    # Quality-Lift phase 6 slice 3 — Per-slice gates nudge. Separate
    # from the un-gated-tests nudge above because the user might have
    # top-level behavioral gates wired up (satisfying that nudge) but
    # still leave per-slice regressions unenforced. Fires only when
    # the user has both slice_definitions AND behavioral_tests
    # configured but no per-slice behavioral gate references the
    # cross-product.
    per_slice_nudge = await _behavioral_tests_without_per_slice_gates_nudge(
        db, project.id,
    )
    if per_slice_nudge:
        suggestions.append(per_slice_nudge)

    if pass_rate is None or pass_rate >= EVAL_PASS_RATE_HEALTHY:
        return suggestions

    severity: Severity = (
        "critical" if pass_rate < EVAL_PASS_RATE_CRITICAL else "warning"
    )

    try:
        cluster_payload = await cluster_eval_result_failures(
            db, eval_result_id=latest.id
        )
    except ValueError:
        # Eval row was deleted between the read + the cluster call —
        # treat as no-suggestion. (Race so rare it's not worth a hard
        # error; the next poll will recover.) Keep any earlier
        # suggestions (e.g. the Phase 9d auto-RAG nudge).
        return suggestions

    clusters = cluster_payload.get("clusters") or []
    if not clusters:
        # Below-threshold pass rate but no clusterable failures (e.g.
        # all failures share no signal). Surface a softer "review
        # eval" navigate suggestion rather than nothing.
        suggestions.append({
            "id": "eval:low-pass-rate-no-clusters",
            "title": (
                f"Eval pass rate is {pass_rate * 100:.0f}% — below the "
                f"{EVAL_PASS_RATE_HEALTHY * 100:.0f}% healthy threshold"
            ),
            "body": (
                "Failures didn't cluster cleanly, so there's no obvious "
                "single bucket to augment. Review the predictions preview "
                "on the Eval tab to identify what's breaking."
            ),
            "severity": severity,
            "action": {
                "kind": "navigate",
                "label": "Open eval predictions",
                "params": {"target": "eval-predictions"},
            },
            "context": {
                "pass_rate": round(float(pass_rate), 4),
                "eval_result_id": latest.id,
            },
        })
        return suggestions

    # Largest cluster first — that's the one whose augmentation lifts
    # the most failed rows. Tie-break on confidence to prefer the
    # remediation classifier's higher-conviction picks.
    top = max(
        clusters,
        key=lambda c: (
            int(c.get("failure_count") or 0),
            float(c.get("classifier_confidence") or 0.0),
        ),
    )
    cluster_id = str(top.get("cluster_id") or "")
    failure_count = int(top.get("failure_count") or 0)
    reason_code = str(top.get("reason_code") or "unknown")
    share = float(top.get("share_of_total") or 0.0)

    suggestions.append({
        "id": "eval:top-failure-cluster",
        "title": (
            f"Top failure cluster: {failure_count} {reason_code} failures "
            f"({share * 100:.0f}% of the total)"
        ),
        "body": (
            f"{top.get('classifier_reason') or ''} "
            "Augmenting this cluster generates synthetic positives that "
            "mirror its failure pattern — pending review before they "
            "enter training. Pass rate today is "
            f"{pass_rate * 100:.0f}%."
        ).strip(),
        "severity": severity,
        "action": {
            "kind": "augment_from_cluster",
            "label": f"Augment {CLUSTER_AUGMENT_DEFAULT} rows for this cluster",
            "params": {
                "eval_result_id": latest.id,
                "cluster_id": cluster_id,
                "target_count": CLUSTER_AUGMENT_DEFAULT,
            },
        },
        "context": {
            "pass_rate": round(float(pass_rate), 4),
            "eval_result_id": latest.id,
            "cluster_id": cluster_id,
            "reason_code": reason_code,
            "failure_count": failure_count,
            "share_of_total": round(share, 4),
        },
    })
    return suggestions


_STAGE_HANDLERS = {
    "data": _data_stage_suggestions,
    "cleaning": _cleaning_stage_suggestions,
    "gold_set": _gold_set_stage_suggestions,
    "training": _training_stage_suggestions,
    "eval": _eval_stage_suggestions,
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

"""Gap-tied synthetic-playbook recommendations (Epic E).

Detects underrepresented classes in a project's labelled gold set and
recommends the ``class_balance_fill`` playbook to close the gap — the
"refund slice has 6 rows → generate 30" nudge — with the count to prefill.

Reuses the trainability forecast's *tested* label-counting + gold loader so
the class counts agree with the forecast's per-class-minimum signal; this is
the Data-Studio surface for that signal, with a one-click launch.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

# A class counts as imbalanced (info-level) when it sits below this fraction of
# the biggest class, even if it clears the absolute floor — "refund 6 vs 50".
_IMBALANCE_RATIO = 0.34
# Don't suggest generating an unbounded number of rows in one go.
_MAX_SUGGESTED_GENERATE = 200


async def get_playbook_gap_recommendations(
    db: AsyncSession, project_id: int
) -> dict[str, Any]:
    """Underrepresented classes + a ``class_balance_fill`` recommendation each.

    ``applicable=False`` when the gold set carries no classification labels
    (non-classification task, or no gold yet) — there's no class balance to
    reason about. Otherwise ``recommendations`` lists each sparse class with
    its current count, a suggested target (toward parity with the biggest
    class), the count to prefill the playbook with, and a severity."""
    from app.services.trainability_forecast_service import (
        PER_CLASS_BLOCK,
        PER_CLASS_MINIMUM,
        _label_counts,
        _load_gold_rows,
    )

    gold_rows = await _load_gold_rows(db, project_id)
    counts = _label_counts(gold_rows)
    if not counts:
        return {
            "applicable": False,
            "reason": "no_labelled_gold",
            "recommendations": [],
        }

    max_count = max(counts.values())
    imbalance_threshold = max(PER_CLASS_MINIMUM, int(max_count * _IMBALANCE_RATIO))

    recommendations: list[dict[str, Any]] = []
    for label, count in counts.items():
        if count >= imbalance_threshold:
            continue
        if count < PER_CLASS_BLOCK:
            severity = "block"
        elif count < PER_CLASS_MINIMUM:
            severity = "warn"
        else:
            severity = "info"
        # Generate enough to approach parity with the biggest class, but always
        # at least clear the absolute floor; capped so one run stays sane.
        suggested_generate = min(
            max(max_count - count, PER_CLASS_MINIMUM),
            _MAX_SUGGESTED_GENERATE,
        )
        recommendations.append({
            "class": str(label),
            "current_count": int(count),
            "suggested_target": int(max(max_count, PER_CLASS_MINIMUM)),
            "suggested_generate": int(suggested_generate),
            "recommended_mode": "class_balance_fill",
            "severity": severity,
            "message": (
                f"“{label}” has only {int(count)} gold example"
                f"{'' if count == 1 else 's'} vs {int(max_count)} for the biggest "
                f"class — generate ~{int(suggested_generate)} to balance it."
            ),
        })

    # Sparsest first — the most urgent gap leads.
    recommendations.sort(key=lambda r: (r["current_count"], r["class"]))
    return {
        "applicable": True,
        "total_classes": len(counts),
        "max_class_count": int(max_count),
        "recommendations": recommendations,
    }

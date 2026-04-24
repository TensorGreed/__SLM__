"""P12 — Failure-cluster service.

Groups individual row-level failures out of an ``EvalResult`` by a
**(reason_code, output_pattern)** pair so the UI can show "here are the
N distinct shapes of failure and how many rows hit each one" instead of
a flat wall of bad predictions.

- ``reason_code`` reuses the classifier already shipped with the
  remediation service (``_classify_failure``) — same vocabulary of
  ``safety_failure`` / ``hallucination`` / ``coverage_gap`` /
  ``formatting_mismatch`` so remediation plans + cluster groupings share
  a root-cause taxonomy.
- ``output_pattern`` is a new, deterministic fingerprint of the model's
  **prediction** text (not the prompt) — length bucket, lead-token
  category, whether the output carries digits. This gives us a second
  dimension that catches subtle distinctions the root-cause alone can't
  (e.g. "two hallucination clusters, one long-verbose, one short-refusal").

Consumed by ``GET /projects/{pid}/evaluation/{eval_result_id}/failure-clusters``
and available for future ``brewslm eval clusters --json`` (P13).
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult

# Reuse the remediation service's failure extractor + classifier. Both
# services share the same row-level failure vocabulary; duplicating the
# extraction logic would drift over time.
from app.services.evaluation_remediation_service import (
    _classify_failure,
    _clean_text,
    _extract_failures_from_eval_result,
    list_remediation_plan_index,
)


_REFUSAL_PATTERNS = re.compile(
    r"^(i\s+(?:cannot|can't|won't|am unable|do not|don't))\b"
    r"|^(sorry|as an ai|i'm sorry|unfortunately|i apologize)\b",
    re.IGNORECASE,
)
_QUESTION_TRAILING = re.compile(r"\?\s*$")
_CODE_LIKELY = re.compile(
    r"^\s*(?:```|def\s|function\s|class\s|<[a-zA-Z])",
)
_JSON_LIKELY = re.compile(r"^\s*[\[{]")
_DIGIT_PATTERN = re.compile(r"\d")


def _length_bucket_by_chars(text: str) -> str:
    chars = len(text or "")
    if chars <= 80:
        return "short"
    if chars <= 400:
        return "medium"
    return "long"


def _leading_category(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return "empty"
    if _JSON_LIKELY.match(stripped):
        return "json"
    if _CODE_LIKELY.match(stripped):
        return "code"
    if _REFUSAL_PATTERNS.match(stripped):
        return "refusal"
    if stripped.startswith('"') or stripped.startswith("'"):
        return "quoted"
    if _QUESTION_TRAILING.search(stripped):
        return "question"
    return "prose"


def _output_pattern_signature(failure: dict[str, Any]) -> str:
    """Build a stable short fingerprint of the model's output shape.

    Uses the *prediction* text only — prompts/references live elsewhere
    in the slice key. The signature deliberately stays small so the UI
    can show it directly as a chip.
    """
    prediction = _clean_text(failure.get("prediction"))
    length = _length_bucket_by_chars(prediction)
    lead = _leading_category(prediction)
    has_digits = "y" if bool(_DIGIT_PATTERN.search(prediction)) else "n"
    return f"len-{length}:lead-{lead}:digits-{has_digits}"


def _fallback_exemplar_text(value: Any, limit: int = 240) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _build_exemplar(row: dict[str, Any]) -> dict[str, Any]:
    exemplar: dict[str, Any] = {
        "prompt": _fallback_exemplar_text(row.get("prompt")),
        "reference": _fallback_exemplar_text(row.get("reference")),
        "prediction": _fallback_exemplar_text(row.get("prediction")),
        "source": _clean_text(row.get("source")),
    }
    for optional in ("judge_score", "judge_rationale", "metric_name", "metric_value", "test_type"):
        value = row.get(optional)
        if value not in (None, ""):
            exemplar[optional] = value if isinstance(value, (int, float)) else _clean_text(value)
    return exemplar


async def cluster_eval_result_failures(
    db: AsyncSession,
    *,
    eval_result_id: int,
    max_failures: int = 200,
    max_exemplars_per_cluster: int = 3,
) -> dict[str, Any]:
    """Return failure clusters + any linked remediation plans for a result."""
    result = await db.execute(select(EvalResult).where(EvalResult.id == eval_result_id))
    eval_result = result.scalar_one_or_none()
    if eval_result is None:
        raise ValueError("eval_result_not_found")

    failures = _extract_failures_from_eval_result(
        eval_result, max_failures=max(1, int(max_failures))
    )

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    reason_totals: dict[str, int] = {}
    for failure in failures:
        classification = _classify_failure(failure)
        reason_code = str(classification.get("root_cause") or "coverage_gap")
        pattern = _output_pattern_signature(failure)
        enriched = dict(failure)
        enriched["reason_code"] = reason_code
        enriched["output_pattern"] = pattern
        enriched["classifier_reason"] = _clean_text(classification.get("reason"))
        enriched["classifier_confidence"] = float(classification.get("confidence") or 0.0)
        buckets.setdefault((reason_code, pattern), []).append(enriched)
        reason_totals[reason_code] = reason_totals.get(reason_code, 0) + 1

    total = len(failures)
    ordered = sorted(
        buckets.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    )

    clusters: list[dict[str, Any]] = []
    exemplar_cap = max(1, int(max_exemplars_per_cluster))
    for idx, ((reason_code, pattern), rows) in enumerate(ordered, start=1):
        share = round(len(rows) / max(1, total), 4)
        avg_conf = (
            sum(float(r.get("classifier_confidence") or 0.0) for r in rows) / max(1, len(rows))
        )
        classifier_reason = next(
            (r.get("classifier_reason") for r in rows if r.get("classifier_reason")),
            "",
        )
        clusters.append(
            {
                "cluster_id": f"cluster-{idx}",
                "reason_code": reason_code,
                "output_pattern": pattern,
                "failure_count": len(rows),
                "share_of_total": share,
                "classifier_confidence": round(avg_conf, 3),
                "classifier_reason": classifier_reason,
                "exemplars": [_build_exemplar(r) for r in rows[:exemplar_cap]],
            }
        )

    dominant_reason = (
        max(reason_totals.items(), key=lambda item: item[1])[0] if reason_totals else None
    )

    # Existing remediation plans for this eval result — surfaced as "link to
    # remediation" targets in the UI. Failure if the artifact table isn't
    # reachable is non-fatal: clusters are still useful on their own.
    remediation_plans: list[dict[str, Any]] = []
    if eval_result.experiment_id:
        project_id = await _resolve_project_id(db, int(eval_result.experiment_id))
        if project_id:
            try:
                remediation_plans = await list_remediation_plan_index(
                    db,
                    project_id=project_id,
                    evaluation_result_id=int(eval_result.id),
                    limit=20,
                )
            except Exception:
                remediation_plans = []

    return {
        "eval_result_id": int(eval_result.id),
        "experiment_id": int(eval_result.experiment_id) if eval_result.experiment_id else None,
        "dataset_name": str(eval_result.dataset_name or ""),
        "eval_type": str(eval_result.eval_type or ""),
        "total_failures_analyzed": total,
        "reason_code_totals": dict(sorted(reason_totals.items(), key=lambda item: -item[1])),
        "dominant_reason_code": dominant_reason,
        "clusters": clusters,
        "remediation_plans": remediation_plans,
    }


async def _resolve_project_id(db: AsyncSession, experiment_id: int) -> int:
    """Look up the project id for an experiment; used to scope remediation plans.

    ``list_remediation_plan_index`` requires a project_id; EvalResult doesn't
    carry one directly.
    """
    from app.models.experiment import Experiment

    result = await db.execute(select(Experiment.project_id).where(Experiment.id == experiment_id))
    row = result.scalar_one_or_none()
    return int(row) if row is not None else 0

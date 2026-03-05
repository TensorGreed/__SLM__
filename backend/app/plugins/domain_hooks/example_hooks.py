"""Example domain hook plugin.

Enable by setting:
DOMAIN_HOOK_PLUGIN_MODULES='["app.plugins.domain_hooks.example_hooks"]'
"""

from __future__ import annotations

from typing import Any


def _caps_normalizer(
    _raw_record: dict[str, Any],
    canonical_record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(canonical_record, dict):
        return None
    target_key = str(config.get("field", "text"))
    value = canonical_record.get(target_key)
    if not isinstance(value, str):
        return dict(canonical_record)
    normalized = dict(canonical_record)
    normalized[target_key] = value.upper()
    return normalized


def _target_ratio_validator(
    records: list[dict[str, Any]],
    profile: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    target = float(config.get("target_ratio", 0.7))
    total = len(records)
    qa_pairs = 0
    for row in records:
        if str(row.get("question") or "").strip() and str(row.get("answer") or "").strip():
            qa_pairs += 1
    ratio = (qa_pairs / total) if total else 0.0
    return {
        "status": "ok" if ratio >= target else "warning",
        "rule": "qa_ratio_target",
        "target_ratio": target,
        "qa_ratio": round(ratio, 6),
        "total_records": total,
        "base_normalization_coverage": profile.get("normalization_coverage"),
    }


def _confidence_band_evaluator(
    _eval_type: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    _context: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(metrics)
    score = enriched.get("pass_rate")
    if score is None:
        score = enriched.get("exact_match")
    if score is None:
        score = enriched.get("f1")
    if not isinstance(score, (int, float)):
        return enriched

    threshold = float(config.get("confident_threshold", 0.8))
    enriched["confidence_band"] = "confident" if float(score) >= threshold else "uncertain"
    return enriched


def get_domain_hooks() -> dict[str, dict[str, object]]:
    return {
        "normalizers": {
            "caps-normalizer": {
                "handler": _caps_normalizer,
                "description": "Uppercase selected canonical field (config.field).",
            }
        },
        "validators": {
            "target-ratio-validator": {
                "handler": _target_ratio_validator,
                "description": "Warn when QA pair ratio is below target_ratio.",
            }
        },
        "evaluators": {
            "confidence-band-evaluator": {
                "handler": _confidence_band_evaluator,
                "description": "Adds confidence_band based on pass_rate/exact_match/f1.",
            }
        },
    }

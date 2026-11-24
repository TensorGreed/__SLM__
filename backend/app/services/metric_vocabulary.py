"""One metric vocabulary for every surface that names or formats an eval score (UI simplify phase 7k).

Per metric key: its name (optionally per result shape: ``f1`` is "Span F1 (micro)" on a span result and
"Token F1" on free text), its scale, and whether higher is better.

Scales (the display rule):
  - ``share``     the share of rows that pass (pass rates, exact match, accuracy). Shown as a percentage with
                  counts ("85.7% (18 of 21 rows)"); a difference is in points ("+4.8 pts").
  - ``mean01``    a 0-1 mean that is not a share of rows (token F1 and the F1 variants, the judge score,
                  precision / recall). Shown as "0.184"; a difference as "+0.048".
  - ``unbounded`` perplexity, loss. A plain number, "lower is better".

The score of record of a result (``score_of_record``) is picked from the metric key and the result's shape,
never from the stored ``pass_rate`` column (``evaluation_service.stored_pass_rate`` = first truthy of
pass_rate / exact_match / f1, so an exact match of 0 stores token F1 there). Stored values never change; this
module only picks and names what is stored.

The frontend reads the same tables from ``frontend/src/generated/metricVocabulary.json``, written by
``python3 scripts/export_metric_vocabulary.py`` (CI checks it is current with ``--check``).
"""

from __future__ import annotations

import math
from typing import Any

SHARE = "share"
MEAN01 = "mean01"
UNBOUNDED = "unbounded"

# Mirrors ``sft_lift_summary_service._PREFERRED_HEADLINE_METRICS`` (a test keeps them equal): the lift
# headline is the first of these that both results carry, and the score of record follows the same order.
PREFERRED_HEADLINE_METRICS: tuple[str, ...] = (
    "remediation_pass_rate",
    "code_example_pass_rate",
    "structured_plan_pass_rate",
    "tool_call_pass_rate",
    "multi_option_pass_rate",
    "exec_match_rate",
    "judge_correct",
    "f1",
    "exact_match",
    "accuracy",
    "macro_f1",
    "precision",
    "recall",
    "groundedness",
    "tool_success_rate",
    "llm_judge_pass_rate",
    "pass_rate",
    "perplexity",
)

# Task scoring modes: a result scored by one of these has a task-specific score of record.
TASK_SCORING_MODES: tuple[str, ...] = (
    "tool_call",
    "field_match",
    "span_set",
    "finding_set",
    "remediation",
    "structured_plan",
    "multi_option",
)

# Score-of-record keys per result shape, in headline order (a subsequence of PREFERRED_HEADLINE_METRICS).
# One exception: a tool-call result with strict argument matching stands on ``tool_call_strict_pass_rate``.
# The first key the result stores wins; a shape's keys never depend on the value, so the base model and the
# fine-tuned run on one set show the same metric.
SCORE_KEYS: dict[str, tuple[str, ...]] = {
    "remediation": ("remediation_pass_rate", "code_example_pass_rate"),
    "structured_plan": ("structured_plan_pass_rate",),
    "tool_call": ("tool_call_strict_pass_rate", "tool_call_pass_rate"),
    "multi_option": ("multi_option_pass_rate",),
    "span_set": ("exec_match_rate", "judge_correct", "f1"),
    "finding_set": ("exec_match_rate", "judge_correct", "f1"),
    "field_match": ("exec_match_rate", "judge_correct", "f1"),
    "classification": ("exec_match_rate", "judge_correct", "f1"),
    "safety": ("f1", "pass_rate"),
    "judge": ("pass_rate",),
    "perplexity": ("perplexity",),
    "text": ("exec_match_rate", "judge_correct", "f1", "exact_match", "accuracy", "macro_f1", "pass_rate", "perplexity"),
}

_OWN_PASS_LABEL = {
    "tool_call": "Tool-call pass rate",
    "structured_plan": "Plan pass rate",
    "multi_option": "Multiple-options pass rate",
    "remediation": "Remediation pass rate",
}

# key -> label (a heading: "Token F1"), short (inside a sentence when the shape is unknown: "F1"), scale,
# higher_is_better, optional per-shape label / scale.
METRICS: dict[str, dict[str, Any]] = {
    "tool_call_strict_pass_rate": {"label": "Strict tool-call pass rate (extra arguments count as wrong)", "short": "strict tool-call pass rate", "scale": SHARE},
    "tool_call_pass_rate": {"label": "Tool-call pass rate", "short": "tool-call pass rate", "scale": SHARE},
    "tool_call_valid_rate": {"label": "Valid tool-call rate", "short": "valid tool-call rate", "scale": SHARE},
    "remediation_pass_rate": {"label": "Remediation pass rate", "short": "remediation pass rate", "scale": SHARE},
    "code_example_pass_rate": {"label": "Code-example pass rate", "short": "code-example pass rate", "scale": SHARE},
    "structured_plan_pass_rate": {"label": "Plan pass rate", "short": "plan pass rate", "scale": SHARE},
    "multi_option_pass_rate": {"label": "Multiple-options pass rate", "short": "multiple-options pass rate", "scale": SHARE},
    "exec_match_rate": {"label": "SQL execution match", "short": "execution match rate", "scale": SHARE},
    "judge_correct": {"label": "Judge score (correct 1, partial 0.5)", "short": "judge score", "scale": MEAN01},
    "accuracy": {"label": "Accuracy", "short": "accuracy", "scale": SHARE},
    "macro_f1": {"label": "Macro F1", "short": "macro F1", "scale": MEAN01},
    "precision": {"label": "Precision", "short": "precision", "scale": MEAN01},
    "recall": {"label": "Recall", "short": "recall", "scale": MEAN01},
    "groundedness": {"label": "Groundedness", "short": "groundedness", "scale": MEAN01},
    "tool_success_rate": {"label": "Tool success rate", "short": "tool success rate", "scale": SHARE},
    "perplexity": {"label": "Perplexity (lower is better)", "short": "perplexity", "scale": UNBOUNDED, "higher_is_better": False},
    "eval_loss": {"label": "Eval loss (lower is better)", "short": "eval loss", "scale": UNBOUNDED, "higher_is_better": False},
    "train_loss": {"label": "Training loss (lower is better)", "short": "training loss", "scale": UNBOUNDED, "higher_is_better": False},
    "mean_nll": {"label": "Mean NLL (lower is better)", "short": "mean NLL", "scale": UNBOUNDED, "higher_is_better": False},
    "bits_per_byte": {"label": "Bits per byte (lower is better)", "short": "bits per byte", "scale": UNBOUNDED, "higher_is_better": False},
    "pass_rate": {"label": "Pass rate", "short": "pass rate", "scale": SHARE, "shape_labels": {"judge": "Judge pass rate"}},
    "llm_judge_pass_rate": {"label": "Pass rate", "short": "judge pass rate", "scale": SHARE, "shape_labels": {"judge": "Judge pass rate"}},
    "f1": {
        "label": "Token F1",
        "short": "F1",
        "scale": MEAN01,
        "shape_labels": {
            "span_set": "Span F1 (micro)",
            "finding_set": "Finding F1 (micro)",
            "field_match": "Mean per-field F1",
            "classification": "Macro F1",
            "safety": "Pass rate",
        },
        # Safety stores its pass rate under ``f1`` too (an alias).
        "shape_scales": {"safety": SHARE},
    },
    "exact_match": {
        "label": "Exact match",
        "short": "exact match",
        "scale": SHARE,
        "shape_labels": {
            **_OWN_PASS_LABEL,
            "span_set": "Rows with every span right",
            "finding_set": "Rows with every finding right",
            "field_match": "Rows with every field right",
            "classification": "Accuracy",
            "safety": "Pass rate",
        },
    },
}

_SCORED_AS_MODE = {
    "tool_call": "tool call",
    "field_match": "JSON fields",
    "span_set": "spans",
    "finding_set": "findings",
    "remediation": "remediation",
    "structured_plan": "structured plan",
    "multi_option": "multiple options",
}
_SCORED_AS_SHAPE = {
    "classification": "classification (label match)",
    "safety": "safety checks",
}
_SCORED_AS_EVAL_TYPE = {
    "exact_match": "free text vs answer key (exact match + token F1)",
    "f1": "token F1",
    "llm_judge": "judge",
    "perplexity": "perplexity",
}


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    """A stored metric's value: the number, or the ``mean`` of a seed-group aggregate's stat block."""
    raw = (metrics or {}).get(key) if isinstance(metrics, dict) else None
    value = _num(raw)
    if value is None and isinstance(raw, dict):
        value = _num(raw.get("mean"))
    return value


def metric_scale(key: str, shape: str | None = None) -> str:
    """``share`` / ``mean01`` / ``unbounded`` for a metric key (on a result of ``shape``)."""
    entry = METRICS.get(key)
    if entry is not None:
        return (entry.get("shape_scales") or {}).get(shape or "", entry["scale"])
    token = key.lower()
    if "loss" in token or "perplexity" in token:
        return UNBOUNDED
    if token.endswith("_pass_rate"):
        return SHARE
    return MEAN01


def higher_is_better(key: str) -> bool:
    entry = METRICS.get(key)
    if entry is not None:
        return bool(entry.get("higher_is_better", True))
    return metric_scale(key) != UNBOUNDED


def metric_label(key: str, shape: str | None = None) -> str:
    """The metric's name for a result of ``shape`` ("Token F1", "Span F1 (micro)", "Plan pass rate")."""
    entry = METRICS.get(key)
    if entry is None:
        return key.replace("_", " ")
    return (entry.get("shape_labels") or {}).get(shape or "", entry["label"])


def _lower_first(label: str) -> str:
    if len(label) > 1 and label[1].isupper():
        return label
    return label[:1].lower() + label[1:]


def metric_name(key: str, shape: str | None = None) -> str:
    """The name inside a sentence. With a known shape: its label, first letter lower-cased ("span F1 (micro)",
    "token F1" on free text); with no shape: the short name ("F1", "plan pass rate", "judge score")."""
    entry = METRICS.get(key)
    if entry is None:
        return key.replace("_", " ")
    if not shape:
        return str(entry.get("short") or entry["label"])
    return _lower_first(metric_label(key, shape))


def format_value(key: str, value: float | None, shape: str | None = None) -> str:
    """"85.7%" for a share, "0.184" for a 0-1 mean, "12.345" for an unbounded score; "—" when missing."""
    v = _num(value)
    if v is None:
        return "—"
    if metric_scale(key, shape) == SHARE:
        return f"{v * 100:.1f}%"
    return f"{v:.3f}"


def format_delta(key: str, delta: float | None, shape: str | None = None) -> str:
    """A difference in the score's own unit: "+4.8 pts" for a share, "+0.048" otherwise."""
    d = _num(delta)
    if d is None:
        return "—"
    if metric_scale(key, shape) == SHARE:
        return f"{d * 100:+.1f} pts"
    return f"{d:+.3f}"


def result_shape(metrics: dict[str, Any] | None, eval_type: str | None, scoring_mode: str | None = None) -> str:
    """The scoring mode, else ``classification`` / ``safety`` / ``judge`` / ``perplexity`` / ``text``."""
    metrics = metrics if isinstance(metrics, dict) else {}
    mode = scoring_mode or metrics.get("scoring_mode")
    if isinstance(mode, str) and mode:
        return mode
    if eval_type == "perplexity":
        return "perplexity"
    if eval_type == "llm_judge":
        return "judge"
    if _num(metrics.get("accuracy")) is not None and _num(metrics.get("macro_f1")) is not None:
        return "classification"
    if isinstance(metrics.get("per_test_type"), dict):
        return "safety"
    return "text"


def scored_as(metrics: dict[str, Any] | None, eval_type: str | None, scoring_mode: str | None = None,
              key: str | None = None) -> str:
    """"tool call", "free text vs answer key (exact match + token F1)", "… graded by a judge model"."""
    shape = result_shape(metrics, eval_type, scoring_mode)
    if shape in _SCORED_AS_MODE:
        base = _SCORED_AS_MODE[shape]
    elif shape in _SCORED_AS_SHAPE:
        base = _SCORED_AS_SHAPE[shape]
    else:
        base = _SCORED_AS_EVAL_TYPE.get(str(eval_type or ""), str(eval_type or "").replace("_", " "))
    if key == "tool_call_strict_pass_rate":
        return f"{base} (strict)"
    if key == "judge_correct":
        return "free text vs answer key, graded by a judge model" if shape == "text" else f"{base}, graded by a judge model"
    return base


def score_key(metrics: dict[str, Any] | None, eval_type: str | None, scoring_mode: str | None = None) -> str | None:
    metrics = metrics if isinstance(metrics, dict) else {}
    shape = result_shape(metrics, eval_type, scoring_mode)
    keys = SCORE_KEYS.get(shape, SCORE_KEYS["text"])
    for key in (*keys, *PREFERRED_HEADLINE_METRICS):
        if metric_value(metrics, key) is not None:
            return key
    return None


def score_of_record(metrics: dict[str, Any] | None, eval_type: str | None = None,
                    scoring_mode: str | None = None) -> dict[str, Any]:
    """``{key, label, name, value, scale, shape, higher_is_better, scored_as, text}`` for a result.

    ``key`` / ``value`` are None when the result stores none of its shape's keys (``label`` "Score").
    """
    metrics = metrics if isinstance(metrics, dict) else {}
    shape = result_shape(metrics, eval_type, scoring_mode)
    key = score_key(metrics, eval_type, scoring_mode)
    if key is None:
        return {"key": None, "label": "Score", "name": "score", "value": None, "scale": None, "shape": shape,
                "higher_is_better": True, "scored_as": scored_as(metrics, eval_type, scoring_mode), "text": "—"}
    value = metric_value(metrics, key)
    return {
        "key": key,
        "label": metric_label(key, shape),
        "name": metric_name(key, shape),
        "value": value,
        "scale": metric_scale(key, shape),
        "shape": shape,
        "higher_is_better": higher_is_better(key),
        "scored_as": scored_as(metrics, eval_type, scoring_mode, key),
        "text": format_value(key, value, shape),
    }


def score_of_record_for_result(result: Any) -> dict[str, Any]:
    """``score_of_record`` for an ``EvalResult`` (or anything with ``metrics`` / ``eval_type``)."""
    return score_of_record(getattr(result, "metrics", None), getattr(result, "eval_type", None))


def export_payload() -> dict[str, Any]:
    """The tables the frontend reads (``frontend/src/generated/metricVocabulary.json``)."""
    return {
        "_generated_by": "scripts/export_metric_vocabulary.py from backend/app/services/metric_vocabulary.py; do not edit",
        "preferred_headline_metrics": list(PREFERRED_HEADLINE_METRICS),
        "task_scoring_modes": list(TASK_SCORING_MODES),
        "score_keys": {shape: list(keys) for shape, keys in SCORE_KEYS.items()},
        "metrics": {
            key: {
                "label": entry["label"],
                "short": entry.get("short") or entry["label"],
                "scale": entry["scale"],
                "higher_is_better": bool(entry.get("higher_is_better", True)),
                **({"shape_labels": dict(entry["shape_labels"])} if entry.get("shape_labels") else {}),
                **({"shape_scales": dict(entry["shape_scales"])} if entry.get("shape_scales") else {}),
            }
            for key, entry in METRICS.items()
        },
        "scored_as": {
            "modes": dict(_SCORED_AS_MODE),
            "shapes": dict(_SCORED_AS_SHAPE),
            "eval_types": dict(_SCORED_AS_EVAL_TYPE),
        },
    }

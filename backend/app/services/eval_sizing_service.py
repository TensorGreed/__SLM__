"""How big a difference between two runs can this many rows show? (Epic W2)

"35.9% vs 29.5% on 78 rows" reads as a regression; it was within noise. This
module answers the question before anyone reads the delta: from ``n`` rows and
the observed per-row spread, the smallest difference between two runs that is
reliably detectable (two-sided 95% confidence, 80% power) and the rows a
``TARGET_GAP`` change would need. It generalises the span-F1-only sizing in
``span_eval_sizing_service`` (unchanged) to every metric that is a mean of
per-row scores (``EvalResult.details["row_scores"]``): binary ones (exact
match, pass rates — scores 0 / 1) and continuous ones (token F1, judge
score with partial credit 0.5).

Formulas (normal approximation; ``Z = z_0.975 + z_0.80 = 1.959964 +
0.841621 = 2.801585``, ``Z² = 7.84888``). ``detectable_gap`` /
``rows_needed`` always mean a difference BETWEEN TWO RUNS:

* **Paired** (both sides scored the same rows — lift, A-vs-B). ``d_i`` =
  after − before per row, ``s_d`` = its sample standard deviation (n − 1)::

      detectable_gap = Z · s_d / √n
      rows_needed(Δ) = ⌈ (Z · s_d / Δ)² ⌉

  For binary scores ``s_d² ≈ (discordant share) − mean²``: rows where the
  two runs agree add nothing, so a correlated pair needs far fewer rows.

* **Single eval** (one side only). The other run is unknown, so the bound
  is the unpaired one — two independent evals with this eval's per-row
  variance ``s²`` each (binomial ``p·(1−p)`` for a binary score, the sample
  variance for a continuous one)::

      detectable_gap = Z · √(2·s² / n)
      rows_needed(Δ) = ⌈ 2 · (Z · s / Δ)² ⌉

  Conservative: two runs scored on the same rows correlate, and the paired
  figure is usually well below this. The one-sample precision of this
  eval's own score (``Z · s / √n``, ``⌈(Z · s / Δ)²⌉``) is kept as
  ``one_sample_gap`` / ``one_sample_rows_needed`` (API only, not a
  comparison).

Degenerate cases (``degenerate`` says which):

* ``too_few_rows`` — fewer than ``MIN_ROWS`` (5): no gap; ``rows_needed`` is
  the worst case. Single: a [0, 1] score has variance ≤ 0.25, unpaired
  ``2 · 0.25`` → ``⌈7.84888 · 0.5 / Δ²⌉`` = 1570 at 5 pts. Paired: a
  difference lies in [−1, 1], variance ≤ 1 → ``⌈7.84888 / Δ²⌉`` = 3140.
* ``all_pass`` / ``all_fail`` / ``zero_variance`` (single) — the observed
  spread is zero, which is no evidence of a tiny one. The spread is
  estimated from the Agresti–Coull adjusted rate ``p̃ = (Σx + 2) / (n + 4)``
  (Σx = sum of scores), ``s² = p̃·(1 − p̃)``. For binary scores this is the
  standard adjustment. For a constant CONTINUOUS score (every row 0.5) it
  is a heuristic: Agresti–Coull is defined for counts, but a [0, 1] score
  with mean ``p`` has variance at most ``p·(1 − p)`` (Bhatia–Davis), so
  the adjusted Bernoulli variance is a conservative stand-in, not an
  estimate.
* ``identical_scores`` / ``constant_shift`` (paired) — every row changed by
  the same amount (zero for identical scores), so there is no spread of
  differences to size from. The figure is the unpaired bound instead,
  ``s_d² = s_before² + s_after²`` (each side's single-eval variance, with
  the adjustment above when a side is constant), labelled so in ``line``.

Display: ``line`` rounds the gap UP to whole points and rows needed UP (to
10 under 1,000, to 50 above); the API keeps the exact values. When there
are enough rows the line names the target, never a smaller gap.

Pure + dependency-free; ``sizing_from_values`` / ``paired_sizing`` take
score lists, ``eval_result_sizing`` reads an ``EvalResult``. Each result
carries a plain ``line`` with its provenance (n, metric).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# The change between two runs a test split should be able to show (0.05 = 5 pts).
TARGET_GAP = 0.05
CONFIDENCE = 0.95
POWER = 0.80
# z for two-sided 95% confidence + z for 80% power.
Z_ALPHA = 1.959964
Z_BETA = 0.841621
Z_TOTAL = Z_ALPHA + Z_BETA
# Fewer rows than this give no spread estimate worth sizing from.
MIN_ROWS = 5
# Largest per-row variance of a score in [0, 1] (a 50% pass rate).
WORST_CASE_VARIANCE = 0.25
# Largest variance of a per-row difference of two [0, 1] scores (d in [−1, 1]).
WORST_CASE_DIFF_VARIANCE = 1.0
_EPS = 1e-12


def _clean(values: Sequence[Any] | None) -> list[float]:
    out: list[float] = []
    for value in values or []:
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and math.isfinite(value):
            out.append(float(value))
    return out


def _is_binary(values: Sequence[float]) -> bool:
    return bool(values) and all(v in (0.0, 1.0) for v in values)


def _sample_variance(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def rows_needed_for(sd: float, gap: float = TARGET_GAP) -> int:
    """⌈(Z · sd / gap)²⌉ rows for a ``gap`` change at 95% / 80% power,
    where ``sd`` is the standard deviation of a per-row difference."""
    gap = max(1e-6, float(gap))
    # 1e-9: keep an exact integer from landing one above after float error.
    return max(1, int(math.ceil((Z_TOTAL * sd / gap) ** 2 - 1e-9)))


def detectable_gap_for(sd: float, n: int) -> float | None:
    """Z · sd / √n — the smallest change ``n`` rows reliably show, where
    ``sd`` is the standard deviation of a per-row difference."""
    if n <= 0:
        return None
    return Z_TOTAL * sd / math.sqrt(n)


def _single_variance(values: Sequence[float]) -> tuple[float, str, str | None]:
    """``(variance, score_kind, degenerate)`` for one side's per-row scores."""
    n = len(values)
    binary = _is_binary(values)
    kind = "binary" if binary else "continuous"
    mean = sum(values) / n
    variance = mean * (1.0 - mean) if binary else _sample_variance(values)
    if variance > _EPS:
        return variance, kind, None
    if mean >= 1.0:
        degenerate = "all_pass"
    elif mean <= 0.0:
        degenerate = "all_fail"
    else:
        degenerate = "zero_variance"
    # Agresti–Coull: p̃ = (Σx + 2) / (n + 4), clamped to a [0, 1] score
    # (a heuristic for a constant continuous score — see the module doc).
    adjusted = max(0.0, min(1.0, (sum(values) + 2.0) / (n + 4.0)))
    return adjusted * (1.0 - adjusted), kind, degenerate


def _nice_rows(rows: int) -> int:
    """Round rows UP for a sentence: to 10 under 1,000, then to 50."""
    step = 10 if rows < 1000 else 50
    return int(math.ceil(rows / step) * step)


def _pts_up(gap: float) -> int:
    """A gap in whole points, rounded UP so the line never overclaims."""
    return max(1, int(math.ceil(gap * 100 - 1e-9)))


_DEGENERATE_NOTES: dict[str, str] = {
    "all_pass": ", every row passed",
    "all_fail": ", every row failed",
    "zero_variance": ", every row scored the same",
    "identical_scores": ", both runs scored every row the same",
    "constant_shift": ", every row moved by the same amount",
}


def sizing_line(sizing: dict[str, Any], *, row_noun: str | None = None, metric_label: str | None = None) -> str:
    """One plain sentence. Single basis (conservative comparison bound):
    "Comparing two runs on these 78 test rows (exact match): gaps under
    ~21 pts may go unnoticed (conservative — same-row comparisons usually
    do better); ~1,350 rows to see a 5-pt change." Paired: "78 paired rows
    (exact match): changes under ~14 pts may go unnoticed; ~570 rows to see
    a 5-pt change." """
    n = int(sizing.get("n") or 0)
    paired = sizing.get("basis") == "paired"
    noun = row_noun or ("paired rows" if paired else "rows")
    if n == 1:
        noun = noun[:-1] if noun.endswith("s") else noun
    metric = metric_label or sizing.get("metric_id") or "score"
    target = f"{float(sizing.get('target_gap') or TARGET_GAP) * 100:g}"
    rows = f"{_nice_rows(int(sizing.get('rows_needed') or 0)):,}"
    degenerate = sizing.get("degenerate")
    lead = f"{n} {noun} ({metric})" if paired else f"Comparing two runs on these {n} {noun} ({metric})"
    if degenerate == "too_few_rows":
        return (
            f"{lead}: too few rows to tell what difference is detectable; "
            f"seeing a {target}-pt change can take up to ~{rows} rows."
        )
    note = _DEGENERATE_NOTES.get(str(degenerate), "")
    unpaired = degenerate in ("identical_scores", "constant_shift")
    if paired and unpaired:
        qualifier = " (unpaired bound, conservative: no row-to-row spread to size from)"
    elif not paired:
        qualifier = " (conservative — same-row comparisons usually do better)"
    else:
        qualifier = ""
    if sizing.get("enough"):
        return f"{lead}{note}: enough rows to see a {target}-pt change{qualifier}."
    shown = _pts_up(float(sizing.get("detectable_gap") or 0.0))
    return (
        f"{lead}{note}: gaps under ~{shown} pts may go unnoticed{qualifier}; "
        f"~{rows} rows to see a {target}-pt change."
    )


def _result(
    *, basis: str, metric_id: str | None, n: int, kind: str, diff_variance: float | None,
    degenerate: str | None, gap: float, row_noun: str | None, metric_label: str | None,
    one_sample_variance: float | None = None,
) -> dict[str, Any]:
    if degenerate == "too_few_rows":
        worst = WORST_CASE_DIFF_VARIANCE if basis == "paired" else 2.0 * WORST_CASE_VARIANCE
        detectable, needed, sd = None, rows_needed_for(math.sqrt(worst), gap), None
    else:
        sd = math.sqrt(float(diff_variance or 0.0))
        detectable, needed = detectable_gap_for(sd, n), rows_needed_for(sd, gap)
    out: dict[str, Any] = {
        "basis": basis,
        "metric_id": metric_id,
        "n": n,
        "score_kind": kind,
        # Standard deviation of a per-row difference between two runs.
        "sd": round(sd, 6) if sd is not None else None,
        "detectable_gap": round(detectable, 4) if detectable is not None else None,
        "target_gap": gap,
        "rows_needed": needed,
        "enough": degenerate != "too_few_rows" and n >= needed,
        "degenerate": degenerate,
        "confidence": CONFIDENCE,
        "power": POWER,
    }
    if basis == "single":
        # This eval's own precision against a fixed value (API only).
        if one_sample_variance is None:
            out["one_sample_gap"] = None
            out["one_sample_rows_needed"] = rows_needed_for(math.sqrt(WORST_CASE_VARIANCE), gap)
        else:
            one = math.sqrt(one_sample_variance)
            out["one_sample_gap"] = round(detectable_gap_for(one, n) or 0.0, 4)
            out["one_sample_rows_needed"] = rows_needed_for(one, gap)
    out["line"] = sizing_line(out, row_noun=row_noun, metric_label=metric_label)
    return out


def sizing_from_values(
    values: Sequence[Any] | None,
    *,
    metric_id: str | None = None,
    gap: float = TARGET_GAP,
    row_noun: str | None = None,
    metric_label: str | None = None,
) -> dict[str, Any] | None:
    """Single-eval sizing from one side's per-row scores: the conservative
    (unpaired) bound for comparing two runs on rows like these. None
    without rows."""
    clean = _clean(values)
    n = len(clean)
    if n == 0:
        return None
    kind = "binary" if _is_binary(clean) else "continuous"
    if n < MIN_ROWS:
        return _result(basis="single", metric_id=metric_id, n=n, kind=kind, diff_variance=None,
                       degenerate="too_few_rows", gap=gap, row_noun=row_noun, metric_label=metric_label)
    variance, kind, degenerate = _single_variance(clean)
    return _result(basis="single", metric_id=metric_id, n=n, kind=kind, diff_variance=2.0 * variance,
                   degenerate=degenerate, gap=gap, row_noun=row_noun, metric_label=metric_label,
                   one_sample_variance=variance)


def paired_sizing(
    before: Sequence[Any] | None,
    after: Sequence[Any] | None,
    *,
    metric_id: str | None = None,
    gap: float = TARGET_GAP,
    row_noun: str | None = None,
    metric_label: str | None = None,
) -> dict[str, Any] | None:
    """Paired sizing from two sides' scores on the SAME rows (matching order,
    as ``paired_comparison_stats.pair_row_scores`` returns them). Pairs with
    a non-finite score on either side drop out. None without pairs."""
    pairs = [
        (float(b), float(a))
        for b, a in zip(before or [], after or [])
        if isinstance(b, (int, float)) and isinstance(a, (int, float))
        and math.isfinite(b) and math.isfinite(a)
    ]
    n = len(pairs)
    if n == 0:
        return None
    befores = [b for b, _ in pairs]
    afters = [a for _, a in pairs]
    kind = "binary" if _is_binary(befores + afters) else "continuous"
    if n < MIN_ROWS:
        return _result(basis="paired", metric_id=metric_id, n=n, kind=kind, diff_variance=None,
                       degenerate="too_few_rows", gap=gap, row_noun=row_noun, metric_label=metric_label)
    diffs = [a - b for b, a in pairs]
    variance = _sample_variance(diffs)
    if variance > _EPS:
        return _result(basis="paired", metric_id=metric_id, n=n, kind=kind, diff_variance=variance,
                       degenerate=None, gap=gap, row_noun=row_noun, metric_label=metric_label)
    identical = all(abs(d) <= _EPS for d in diffs)
    unpaired = _single_variance(befores)[0] + _single_variance(afters)[0]
    return _result(basis="paired", metric_id=metric_id, n=n, kind=kind, diff_variance=unpaired,
                   degenerate="identical_scores" if identical else "constant_shift",
                   gap=gap, row_noun=row_noun, metric_label=metric_label)


def _metric_label(metric_id: str) -> str:
    return {
        "judge_correct": "judge score",
        "f1": "F1",
        "exact_match": "exact match",
        "accuracy": "accuracy",
        "remediation_pass_rate": "remediation pass rate",
        "code_example_pass_rate": "code-example pass rate",
        "structured_plan_pass_rate": "plan pass rate",
        "tool_call_pass_rate": "tool-call pass rate",
    }.get(metric_id, metric_id)


def metric_label(metric_id: str | None) -> str | None:
    """Plain metric name for a sizing line ("exact match", "F1")."""
    return _metric_label(str(metric_id)) if metric_id else None


def eval_result_sizing(
    result: Any,
    metric_id: str | None = None,
    *,
    gap: float = TARGET_GAP,
    row_noun: str = "test rows",
) -> dict[str, Any] | None:
    """Single-eval sizing for an ``EvalResult``: ``metric_id`` when it is a
    per-row metric the result recorded, else the first per-row metric it
    reports in headline order. None when it has no per-row scores."""
    from app.services.sft_lift_summary_service import _PREFERRED_HEADLINE_METRICS, _ROW_SCORE_FOR_METRIC

    details = getattr(result, "details", None)
    metrics = getattr(result, "metrics", None)
    row_scores = details.get("row_scores") if isinstance(details, dict) else None
    if not isinstance(row_scores, dict):
        return None
    metrics = metrics if isinstance(metrics, dict) else {}
    order = [m for m in _PREFERRED_HEADLINE_METRICS if m in _ROW_SCORE_FOR_METRIC]
    order += [m for m in _ROW_SCORE_FOR_METRIC if m not in order]
    if metric_id in _ROW_SCORE_FOR_METRIC:
        order = [metric_id] + [m for m in order if m != metric_id]
    for candidate in order:
        if candidate not in metrics:
            continue
        values = row_scores.get(_ROW_SCORE_FOR_METRIC[candidate])
        if not isinstance(values, list):
            continue
        sizing = sizing_from_values(values, metric_id=candidate, gap=gap, row_noun=row_noun,
                                    metric_label=_metric_label(candidate))
        if sizing is not None:
            return sizing
    return None


__all__ = [
    "TARGET_GAP",
    "detectable_gap_for",
    "eval_result_sizing",
    "metric_label",
    "paired_sizing",
    "rows_needed_for",
    "sizing_from_values",
    "sizing_line",
]

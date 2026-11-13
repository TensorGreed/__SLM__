"""Is a before/after difference on the same rows more than noise?

A headline like "+22% F1 with retrieval" on 21 rows reads as a result. It
often isn't: on the Support FAQ sample the same comparison across three
training seeds gave +22%, +4% and -3%. This module gives every paired
comparison two honest companions to its headline:

* how many rows got better, worse, or stayed the same; and
* a 95% confidence interval for the mean per-row change (paired t-interval).
  If the interval includes zero, the change is **within noise** — the data
  can't tell a gain from a loss.

When every per-row score is binary (0 / 1: correct or not, pass or fail) the
interval alone is liberal with few changed rows — 4 better / 0 worse on 50
rows gives an interval above zero while the exact McNemar test (a two-sided
binomial on the rows that changed) gives p = 0.125. For binary scores
"better" / "worse" therefore also needs exact McNemar p < 0.05 (at least 6
changed rows: 6 / 0 is the smallest case, 8 / 1 and 12 / 3 also count); otherwise the verdict is within noise. The p-value
is recorded as ``mcnemar_p``. Continuous scores (F1, partial judge credit)
keep the interval-only rule (``mcnemar_p`` is None).

Pure + dependency-free (no numpy / scipy) so any service can call it.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# Two-sided 95% Student-t critical values by degrees of freedom (n - 1).
_T_975: tuple[float, ...] = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)
_T_975_BY_DF: tuple[tuple[int, float], ...] = ((40, 2.021), (60, 2.000), (120, 1.980))
_Z_975 = 1.960

# Fewer paired rows than this can't support any verdict.
MIN_ROWS_FOR_VERDICT = 5
# Seeds are expensive (a training run each): two already give an interval,
# a very wide one — the width is the honest part.
MIN_SEEDS_FOR_VERDICT = 2
# Per-row changes smaller than this are "no change" (float noise in F1).
SAME_EPSILON = 1e-9
# Binary paired outcomes: "better" / "worse" also need exact McNemar p below this.
MCNEMAR_ALPHA = 0.05


# Up to this many changed rows the tail is summed exactly in integers (well under 1 ms at the
# limit); above it a float sum anchored on the exact peak term keeps it fast.
_MCNEMAR_EXACT_SUM_MAX_N = 2000


def exact_mcnemar_p(better: int, worse: int) -> float:
    """Exact two-sided McNemar p-value: a binomial(better + worse, 0.5) test on
    the discordant pairs (rows that changed). 1.0 when no row changed."""
    b, w = int(better), int(worse)
    n = b + w
    if n <= 0:
        return 1.0
    k = min(b, w)
    if n <= _MCNEMAR_EXACT_SUM_MAX_N:
        # Integer arithmetic, exact (int / int true division rounds correctly). Each binomial
        # coefficient is built from the previous one, C(n, i+1) = C(n, i) * (n - i) // (i + 1),
        # so the sum is linear in k rather than k separate comb() calls.
        term = 1
        total = 1
        for i in range(k):
            term = term * (n - i) // (i + 1)
            total += term
        tail = total / (1 << n)
    else:
        # Large n: the exact peak term C(n, k) / 2**n (one correctly rounded big-int division)
        # times the float sum of C(n, i) / C(n, k) for i = k, k-1, ... These ratios shrink at
        # least geometrically, so the sum stops once a term can no longer change it. Agrees with
        # the exact sum to about 1e-13 relative and runs in milliseconds for 40,000 changed rows
        # (the exact sum over that many terms takes over a second).
        peak = math.comb(n, k) / (1 << n)
        ratio_sum = 0.0
        ratio = 1.0
        i = k
        while i >= 0:
            ratio_sum += ratio
            if ratio < ratio_sum * 1e-17:
                break
            ratio *= i / (n - i + 1)
            i -= 1
        tail = peak * ratio_sum
    return min(1.0, 2.0 * tail)


def _is_binary(values: Sequence[float]) -> bool:
    return all(v == 0.0 or v == 1.0 for v in values)


def _t_critical(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df <= len(_T_975):
        return _T_975[df - 1]
    for limit, value in _T_975_BY_DF:
        if df <= limit:
            return value
    return _Z_975


def paired_difference_evidence(
    before: Sequence[float],
    after: Sequence[float],
    *,
    min_n: int = MIN_ROWS_FOR_VERDICT,
    binary_guard: bool = True,
) -> dict[str, Any]:
    """Evidence for ``after`` vs ``before`` scored on the SAME rows.

    Returns ``{n, better, worse, same, mean_diff, ci_low, ci_high, mcnemar_p,
    verdict}`` where ``verdict`` is one of:

    * ``"better"`` / ``"worse"`` — the 95% interval for the mean change
      excludes zero, and — when every paired score is 0 / 1 — the exact
      McNemar p (``mcnemar_p``) is below 0.05;
    * ``"within_noise"`` — otherwise;
    * ``"too_few_rows"`` — fewer than ``min_n`` paired rows (``ci_low`` /
      ``ci_high`` are None).
    """
    pairs = [
        (float(b), float(a))
        for b, a in zip(before, after)
        if isinstance(b, (int, float)) and isinstance(a, (int, float))
        and math.isfinite(b) and math.isfinite(a)
    ]
    diffs = [a - b for b, a in pairs]
    n = len(diffs)
    better = sum(1 for d in diffs if d > SAME_EPSILON)
    worse = sum(1 for d in diffs if d < -SAME_EPSILON)
    evidence: dict[str, Any] = {
        "n": n,
        "better": better,
        "worse": worse,
        "same": n - better - worse,
        "mean_diff": (sum(diffs) / n) if n else None,
        "ci_low": None,
        "ci_high": None,
        # Exact McNemar p on the changed rows; None unless every score is 0 / 1.
        "mcnemar_p": None,
        "verdict": "too_few_rows",
    }
    if binary_guard and n and _is_binary([v for pair in pairs for v in pair]):
        evidence["mcnemar_p"] = exact_mcnemar_p(better, worse)
    if n < min_n:
        return evidence
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    half_width = _t_critical(n - 1) * math.sqrt(variance / n)
    low, high = mean - half_width, mean + half_width
    evidence["ci_low"] = low
    evidence["ci_high"] = high
    p = evidence["mcnemar_p"]
    significant = p is None or p < MCNEMAR_ALPHA
    if low > 0 and significant:
        evidence["verdict"] = "better"
    elif high < 0 and significant:
        evidence["verdict"] = "worse"
    else:
        evidence["verdict"] = "within_noise"
    return evidence


def _row_score_list(row_scores: Any, score_key: str) -> tuple[list[str], list[str], list[float]] | None:
    """``(all_keys, scored_keys, scores)`` from ``details["row_scores"]``;
    None when it lacks ``keys`` or ``score_key`` (or their lengths differ).
    A None score (the judge couldn't score that row) drops the row from the
    pairing rather than the whole comparison."""
    if not isinstance(row_scores, dict):
        return None
    keys, values = row_scores.get("keys"), row_scores.get(score_key)
    if not isinstance(keys, list) or not isinstance(values, list) or len(keys) != len(values):
        return None
    kept = [(str(k), float(v)) for k, v in zip(keys, values) if v is not None]
    return [str(k) for k in keys], [k for k, _ in kept], [v for _, v in kept]


def pair_row_scores(
    before_row_scores: Any, after_row_scores: Any, score_key: str
) -> dict[str, Any] | None:
    """Pair two evals' per-row scores (``EvalResult.details["row_scores"]``)
    on ``score_key`` by row key, so evals that saw the rows in a different
    order or count still compare row for row.

    Returns ``{before, after, only_before, only_after, unscored, same_order}``: the paired
    score lists (matching order), how many distinct row keys only one side
    evaluated, and how many rows both evaluated but at least one side left
    unscored (a None score). None when either side lacks the score list.
    Same keys in the same order pair positionally; otherwise the first
    occurrence of a key wins.
    """
    before = _row_score_list(before_row_scores, score_key)
    after = _row_score_list(after_row_scores, score_key)
    if before is None or after is None:
        return None
    before_all, before_keys, before_vals = before
    after_all, after_keys, after_vals = after
    before_set, after_set = set(before_all), set(after_all)
    shared = before_set & after_set
    out: dict[str, Any] = {
        "only_before": len(before_set - after_set),
        "only_after": len(after_set - before_set),
    }
    out["same_order"] = before_keys == after_keys
    if out["same_order"]:
        out.update(before=list(before_vals), after=list(after_vals))
    else:
        before_by_key: dict[str, float] = {}
        for key, value in zip(before_keys, before_vals):
            before_by_key.setdefault(key, value)
        paired_before: list[float] = []
        paired_after: list[float] = []
        seen: set[str] = set()
        for key, value in zip(after_keys, after_vals):
            if key in before_by_key and key not in seen:
                seen.add(key)
                paired_before.append(before_by_key[key])
                paired_after.append(value)
        out.update(before=paired_before, after=paired_after)
    paired_keys = set(before_keys) & set(after_keys)
    out["unscored"] = len(shared - paired_keys)
    return out


def seed_spread_evidence(
    baseline_value: float, trained_values: Sequence[float]
) -> dict[str, Any]:
    """Run-to-run evidence: the same config trained with N seeds, each scored
    against the (deterministic) base model on the same rows.

    The row-level interval says whether a lift is more than which-rows noise
    for ONE run; it cannot see that another seed gives a different run. This
    one can: it is the interval for the mean seed-to-base difference across
    seeds (paired t, n = seeds). Verdicts: ``better`` / ``worse`` (every
    plausible seed lands on that side), ``within_noise`` (seeds disagree
    enough that the sign isn't established), ``too_few_seeds``.
    """
    values = [float(v) for v in trained_values if isinstance(v, (int, float)) and math.isfinite(v)]
    # Seed scores are aggregate rates, not per-row outcomes: interval-only rule.
    evidence = paired_difference_evidence(
        [float(baseline_value)] * len(values), values, min_n=MIN_SEEDS_FOR_VERDICT, binary_guard=False
    )
    if evidence["verdict"] == "too_few_rows":
        evidence["verdict"] = "too_few_seeds"
    n = len(values)
    mean = (sum(values) / n) if n else None
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    evidence.update({
        "kind": "seeds",
        "baseline_value": float(baseline_value),
        "values": values,
        "mean": mean,
        "std": std,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        # Every seed on the same side of the base model — a plain, strong
        # reading a beginner can check against the list of values.
        "all_better": bool(values) and all(v > baseline_value for v in values),
        "all_worse": bool(values) and all(v < baseline_value for v in values),
    })
    return evidence

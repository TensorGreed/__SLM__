"""Is a before/after difference on the same rows more than noise?

A headline like "+22% F1 with retrieval" on 21 rows reads as a result. It
often isn't: on the Support FAQ sample the same comparison across three
training seeds gave +22%, +4% and -3%. This module gives every paired
comparison two honest companions to its headline:

* how many rows got better, worse, or stayed the same; and
* a 95% confidence interval for the mean per-row change (paired t-interval).
  If the interval includes zero, the change is **within noise** — the data
  can't tell a gain from a loss.

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
# Per-row changes smaller than this are "no change" (float noise in F1).
SAME_EPSILON = 1e-9


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
    before: Sequence[float], after: Sequence[float]
) -> dict[str, Any]:
    """Evidence for ``after`` vs ``before`` scored on the SAME rows.

    Returns ``{n, better, worse, same, mean_diff, ci_low, ci_high, verdict}``
    where ``verdict`` is one of:

    * ``"better"`` / ``"worse"`` — the 95% interval for the mean change
      excludes zero;
    * ``"within_noise"`` — it includes zero;
    * ``"too_few_rows"`` — fewer than ``MIN_ROWS_FOR_VERDICT`` paired rows
      (``ci_low`` / ``ci_high`` are None).
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
        "verdict": "too_few_rows",
    }
    if n < MIN_ROWS_FOR_VERDICT:
        return evidence
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    half_width = _t_critical(n - 1) * math.sqrt(variance / n)
    low, high = mean - half_width, mean + half_width
    evidence["ci_low"] = low
    evidence["ci_high"] = high
    if low > 0:
        evidence["verdict"] = "better"
    elif high < 0:
        evidence["verdict"] = "worse"
    else:
        evidence["verdict"] = "within_noise"
    return evidence

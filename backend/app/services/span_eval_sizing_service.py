"""Is the answer key big enough to tell two runs apart, per type? (P6)

A per-type F1 on a handful of reference spans moves by tenths between
seeds. This sizes each type the way the documents→Q&A flow sizes its test
split (``documents_qa_flow_service.target_test_rows``): for two runs
scored on the SAME reference spans, the per-span outcome difference has
standard deviation at most ``sqrt(2·p·(1−p))`` (p = the share of spans a
run gets right, the type's F1 as a proxy), and a gap ``Δ`` in the mean
clears a 95% interval when

    n ≥ (2 · sd / Δ)²  =  8 · p · (1 − p) / Δ²        reference spans.

So ``spans_needed(f1, gap)`` is the support a type needs before a ``gap``
F1 difference between runs means anything, and ``detectable_gap(f1,
support)`` = ``2 · sqrt(2·p·(1−p) / n)`` is the smallest difference the
support it HAS can show. ``p`` is clamped to [``P_MIN``, ``P_MAX``]: a
type at F1 1.0 on 3 spans has no evidence of being easy, and a type at
0.0 is not harder to size than a coin.

Pure functions over an eval's ``per_class`` block; ``size_types`` is what
``span_block`` attaches as ``sizing`` and what the refine endpoint reads
for ``under_sampled``.
"""

from __future__ import annotations

import math
from typing import Any

# The F1 difference between two runs a key should be able to show, per type.
DECISION_GAP = 0.05
P_MIN = 0.5
P_MAX = 0.95


def _p(f1: Any) -> float:
    try:
        value = float(f1)
    except (TypeError, ValueError):
        value = P_MIN
    if math.isnan(value):
        value = P_MIN
    return max(P_MIN, min(P_MAX, value))


def spans_needed(f1: Any, gap: float = DECISION_GAP) -> int:
    """Reference spans a type needs for a ``gap`` F1 difference between
    two runs to clear noise (95%)."""
    gap = max(1e-6, float(gap))
    p = _p(f1)
    # 1e-9: 8·0.95·0.05/0.05² is exactly 152 but floats say 152.0000…1.
    return int(math.ceil(8.0 * p * (1.0 - p) / (gap * gap) - 1e-9))


def detectable_gap(f1: Any, support: int) -> float | None:
    """The smallest F1 difference ``support`` reference spans can show
    (None without spans)."""
    n = int(support or 0)
    if n <= 0:
        return None
    p = _p(f1)
    return round(2.0 * math.sqrt(2.0 * p * (1.0 - p) / n), 4)


def size_types(per_class: dict[str, Any] | None, gap: float = DECISION_GAP) -> dict[str, Any]:
    """Per type: support (tp + fn), F1, spans needed for ``gap``, the
    shortfall, the gap the current support can show, and ``enough``.
    ``types`` is ordered by shortfall (largest first; enough types last,
    alphabetical). ``short`` / ``enough`` list the names; ``shortfall``
    is the total missing spans."""
    rows: list[dict[str, Any]] = []
    for name, stats in (per_class or {}).items():
        if not isinstance(stats, dict):
            continue
        support = int(stats.get("tp") or 0) + int(stats.get("fn") or 0) if ("tp" in stats or "fn" in stats) else 0
        f1 = stats.get("f1")
        needed = spans_needed(f1, gap)
        rows.append({
            "type": str(name),
            "support": support,
            "f1": round(float(f1), 4) if isinstance(f1, (int, float)) else None,
            "needed": needed,
            "shortfall": max(0, needed - support),
            "detectable_gap": detectable_gap(f1, support),
            "enough": support >= needed,
        })
    rows.sort(key=lambda r: (-r["shortfall"], r["type"]))
    return {
        "gap": gap,
        "types": rows,
        "short": [r["type"] for r in rows if not r["enough"]],
        "enough": [r["type"] for r in rows if r["enough"]],
        "shortfall": sum(r["shortfall"] for r in rows),
    }


def sizing_line(sizing: dict[str, Any] | None) -> str | None:
    """One sentence for a card / model card: how many types can show the
    gap, and the coarsest gap the weakest-supported type can show."""
    if not isinstance(sizing, dict) or not sizing.get("types"):
        return None
    rows = sizing["types"]
    gap = sizing.get("gap", DECISION_GAP)
    enough = len(sizing.get("enough") or [])
    short = [r for r in rows if not r["enough"]]
    if not short:
        return f"Every type has enough reference spans to show a {gap:g} F1 difference between runs."
    coarse = max((r["detectable_gap"] or 1.0) for r in short)
    worst = min(short, key=lambda r: r["support"])
    return (
        f"{enough} of {len(rows)} types have enough reference spans to show a {gap:g} F1 difference; "
        f"the others can only show {coarse:.2f} or more "
        f"({worst['type']}: {worst['support']} span{'s' if worst['support'] != 1 else ''}, needs {worst['needed']})."
    )

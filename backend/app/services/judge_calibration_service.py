"""Judge calibration (V2): blind labels that say how far each review-assist judge can be trusted.

Ground truth is a set of BLIND calibration labels, kept apart from review decisions: a review-queue
accept / reject never counts (the reviewer saw the verdicts, and a group accept is the judge's own
call). Labels come in two tiers that are stored separately and never pooled:

- ``human``: a person labels a row pass / fail without seeing any verdict;
- ``reference_model``: a strong model labels a row blind (the ``judge_calibration_reference`` Job),
  using the same prompt the review-assist judges read.

Persistence: a sidecar ``synthetic/calibration_labels.json`` shaped
``{row id: {tier: {label, reason?, at, sig, stratum, labeller}}}``. ``sig`` is the review-assist
``row_signature``: a label whose row has changed since is stale and ignored. One writer path
(process lock + temp file + ``os.replace``).

Sampling is stratified by the row's current judge consensus (``single`` split by its verdict), so
rare strata (disagreements, failures) get labels; the report weights each labelled row by
(stratum population / labelled rows in that stratum, for that tier) so the design does not bias the
numbers (post-stratification on the CURRENT strata: a row whose verdicts changed since it was
labelled is weighted in its new stratum; the stored ``stratum`` is a record of where it was drawn).

Metrics per judge (``fail`` is the positive class): fail-recall = P(judge says fail | label fail),
unsure counted as "did not say fail"; pass-precision = P(label pass | judge says pass); unsure rate;
Cohen's kappa on the rows the judge said pass / fail. Rates carry Wilson 95% intervals at the
Kish effective n. Nothing here gates anything: V3 derives a per-judge trust status (``judge_trust``)
that the review queue shows and the group-accept button warns on, never blocks.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config import settings
from app.services import synth_review_assist_service as assist

SIDECAR_NAME = "calibration_labels.json"
TIERS: tuple[str, ...] = ("human", "reference_model")
LABELS: tuple[str, ...] = ("pass", "fail")
LABEL_SOURCES: tuple[str, ...] = ("human", "reference_model", "both")
DEFAULT_LABEL_SOURCE = "human"
CONFIG_KEY = "judge_calibration"
HUMAN_LABELLER = "human"
STRATA: tuple[str, ...] = ("single_pass", "single_fail", "single_unsure", "agree_pass", "agree_fail", "disagree")
DEFAULT_SAMPLE = 20
MAX_SAMPLE = 100
REASON_LIMIT = 500
# "Not enough labels": fewer labelled fails or passes than this among a judge's rows, or fewer rows in all.
MIN_CLASS_LABELS = 5
MIN_TOTAL_LABELS = 20
WILSON_Z = 1.959963984540054
NOT_RECORDED = "(not recorded)"
# Fields the queue attaches at read time; a blind row must never carry them (``render_row_ex`` already
# drops the stored review / provenance keys).
_ATTACHED_KEYS = ("review_assist", "review_assist_judges", "assist_consensus")

_LOCK = threading.Lock()


# ── settings ─────────────────────────────────────────────────────────────


def label_source_of(runtime_config: Any) -> str:
    """The project's ``judge_calibration.label_source`` (``human`` when unset or unreadable)."""
    rc = runtime_config if isinstance(runtime_config, dict) else {}
    cfg = rc.get(CONFIG_KEY) if isinstance(rc.get(CONFIG_KEY), dict) else {}
    value = cfg.get("label_source")
    return value if value in LABEL_SOURCES else DEFAULT_LABEL_SOURCE


def tiers_for(label_source: str) -> list[str]:
    return list(TIERS) if label_source == "both" else [label_source]


# ── sidecar ──────────────────────────────────────────────────────────────


def labels_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / SIDECAR_NAME


def _read_labels(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """``{row id: {tier: entry}}`` (an entry is a label or a skip marker); a missing / unreadable file or a
    malformed entry reads as nothing."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for rid, tiers in data.items():
        if not isinstance(tiers, dict):
            continue
        kept = {t: e for t, e in tiers.items() if t in TIERS and isinstance(e, dict) and (e.get("label") in LABELS or e.get("skipped") is True)}
        if kept:
            out[str(rid)] = kept
    return out


def read_labels(project_id: int) -> dict[str, dict[str, dict[str, Any]]]:
    path = labels_path(project_id)
    if not path.exists():
        return {}
    with _LOCK:
        return _read_labels(path)


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_labels(project_id: int, tier: str, entries: dict[int, dict[str, Any]]) -> None:
    """Merge ``{row id: entry}`` (labels or skip markers) into ``tier`` (a relabel or a skip replaces that
    tier's entry for the row; the other tier stays)."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}")
    if not entries:
        return
    path = labels_path(project_id)
    with _LOCK:
        data = _read_labels(path)
        for rid, entry in entries.items():
            data.setdefault(str(int(rid)), {})[tier] = dict(entry)
        _write(path, data)


def delete_label(project_id: int, row_id: int, tier: str) -> bool:
    """Remove one row's label or skip marker in ``tier``; False when there was none."""
    path = labels_path(project_id)
    if not path.exists():
        return False
    with _LOCK:
        data = _read_labels(path)
        row = data.get(str(int(row_id)))
        if not row or tier not in row:
            return False
        del row[tier]
        if not row:
            del data[str(int(row_id))]
        _write(path, data)
    return True


def label_entry(row: dict[str, Any], label: str, *, stratum: str, labeller: str, reason: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "at": datetime.now(timezone.utc).isoformat(),
        "sig": assist.row_signature(row),
        "stratum": stratum,
        "labeller": labeller,
    }
    text = " ".join(str(reason or "").split())[:REASON_LIMIT]
    if text:
        entry["reason"] = text
    return entry


def skip_entry(row: dict[str, Any], *, reason: str, labeller: str) -> dict[str, Any]:
    """A skip marker: the row was offered for labelling in a tier and NOT labelled (a person pressed Skip,
    or the reference model was unsure / unreadable / the row was too long). It is not a label; it keeps the
    row out of later samples for that tier and is reported, since skipped rows are often the hard ones."""
    return {
        "skipped": True,
        "reason": " ".join(str(reason or "").split())[:REASON_LIMIT] or "skipped",
        "at": datetime.now(timezone.utc).isoformat(),
        "sig": assist.row_signature(row),
        "labeller": labeller,
    }


# ── population + strata ──────────────────────────────────────────────────


def stratum_of(attached: dict[str, Any]) -> str | None:
    """The calibration stratum of a row with verdicts attached (``None`` when it has none)."""
    judges = attached.get("review_assist_judges") or []
    if not judges:
        return None
    consensus = assist.consensus_of([e["verdict"] for e in judges])
    if consensus == "single":
        verdict = judges[0].get("verdict")
        return f"single_{verdict}" if verdict in assist.VERDICTS else "single_unsure"
    return consensus if consensus in STRATA else None


def population(project_id: int, rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """``(raw row, row with verdicts attached, stratum)`` for every synthetic row (any review status) with
    an integer id and at least one current-signature review-assist verdict."""
    raw = [r for r in rows if isinstance(r.get("id"), int) and not isinstance(r.get("id"), bool)]
    probe = [dict(r) for r in raw]
    assist.attach_verdicts(project_id, probe)
    out = []
    for orig, att in zip(raw, probe):
        stratum = stratum_of(att)
        if stratum is not None:
            out.append((orig, att, stratum))
    return out


def current_labels(
    labels: dict[str, dict[str, dict[str, Any]]], tier: str, rows_by_id: dict[int, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], int]:
    """``({row id: entry}, stale count)`` for ``tier``: a label counts only while its row exists and its
    signature still matches (an edited row or a reused id does not inherit a label)."""
    out: dict[int, dict[str, Any]] = {}
    stale = 0
    for rid, tiers in labels.items():
        entry = tiers.get(tier)
        if not entry or entry.get("label") not in LABELS:
            continue
        try:
            key = int(rid)
        except ValueError:
            continue
        row = rows_by_id.get(key)
        if row is None or entry.get("sig") != assist.row_signature(row):
            stale += 1
            continue
        out[key] = entry
    return out, stale


def current_skips(
    labels: dict[str, dict[str, dict[str, Any]]], tier: str, rows_by_id: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """``{row id: skip marker}`` for ``tier`` whose row still exists with the same signature."""
    out: dict[int, dict[str, Any]] = {}
    for rid, tiers in labels.items():
        entry = tiers.get(tier)
        if not entry or entry.get("skipped") is not True:
            continue
        try:
            key = int(rid)
        except ValueError:
            continue
        row = rows_by_id.get(key)
        if row is not None and entry.get("sig") == assist.row_signature(row):
            out[key] = entry
    return out


def blind_row(row: dict[str, Any]) -> dict[str, Any]:
    """The row without anything the queue attaches at read time (verdicts / consensus)."""
    return {k: v for k, v in row.items() if k not in _ATTACHED_KEYS}


def render_blind(row: dict[str, Any]) -> dict[str, Any]:
    """What a judge reads for this row (task, artefact, passages — the review-assist renderer) and
    nothing else: no verdicts, consensus, review status, reject reason or provenance."""
    rendered = assist.render_row_ex(blind_row(row))
    try:
        fields = json.loads(rendered.text)
    except ValueError:
        fields = None
    return {
        "row_id": int(row["id"]),
        "kind": assist.row_kind_of(row),
        "content": rendered.text,
        "fields": fields if isinstance(fields, dict) else None,
        "context_truncated": rendered.context_truncated,
    }


def stratified_sample(
    candidates: list[tuple[dict[str, Any], str]], n: int, seed: int = 0,
) -> list[dict[str, Any]]:
    """Up to ``n`` raw rows from ``(row, stratum)`` pairs, spread evenly across the non-empty strata
    (round-robin in ``STRATA`` order; each stratum shuffled by a seeded RNG). The picked rows are then
    shuffled again, so a row's position says nothing about its stratum (the judges' verdicts)."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {s: [] for s in STRATA}
    for row, stratum in sorted(candidates, key=lambda c: c[0]["id"]):
        buckets.setdefault(stratum, []).append(row)
    for rows in buckets.values():
        rng.shuffle(rows)
    queues = [buckets[s] for s in STRATA if buckets.get(s)]
    out: list[dict[str, Any]] = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    rng.shuffle(out)
    return out


def sample_rows(project_id: int, rows: list[dict[str, Any]], tier: str, n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Raw rows to label blind in ``tier``: the population minus rows already (currently) labelled or
    skipped there (a re-run does not re-offer a skipped row)."""
    pop = population(project_id, rows)
    by_id = {int(r["id"]): r for r, _, _ in pop}
    labels = read_labels(project_id)
    labelled, _ = current_labels(labels, tier, by_id)
    done = set(labelled) | set(current_skips(labels, tier, by_id))
    candidates = [(r, s) for r, _, s in pop if int(r["id"]) not in done]
    return stratified_sample(candidates, max(0, min(MAX_SAMPLE, int(n))), seed)


def stratum_for_row(project_id: int, row: dict[str, Any]) -> str | None:
    probe = [dict(row)]
    assist.attach_verdicts(project_id, probe)
    return stratum_of(probe[0])


# ── progress ─────────────────────────────────────────────────────────────


def progress(project_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per tier: labelled (current) / stale / skipped / pass / fail, and per stratum population + labelled + skipped."""
    pop = population(project_id, rows)
    by_id = {int(r["id"]): r for r, _, _ in pop}
    strata_pop = {s: 0 for s in STRATA}
    stratum_by_id: dict[int, str] = {}
    for r, _, s in pop:
        strata_pop[s] += 1
        stratum_by_id[int(r["id"])] = s
    all_by_id = {int(r["id"]): r for r in rows if isinstance(r.get("id"), int) and not isinstance(r.get("id"), bool)}
    labels = read_labels(project_id)
    tiers = {}
    for tier in TIERS:
        cur, stale = current_labels(labels, tier, all_by_id)
        cur = {rid: e for rid, e in cur.items() if rid in by_id}
        skips = current_skips(labels, tier, by_id)
        per = {s: {"population": strata_pop[s], "labelled": 0, "skipped": 0} for s in STRATA}
        for rid in cur:
            per[stratum_by_id[rid]]["labelled"] += 1
        for rid in skips:
            per[stratum_by_id[rid]]["skipped"] += 1
        tiers[tier] = {
            "labelled": len(cur), "stale": stale, "skipped": len(skips),
            "pass": sum(1 for e in cur.values() if e["label"] == "pass"),
            "fail": sum(1 for e in cur.values() if e["label"] == "fail"),
            "labellers": sorted({str(e.get("labeller") or "") for e in cur.values()}),
            "strata": per,
        }
    return {"population": len(pop), "tiers": tiers}


# ── statistics ───────────────────────────────────────────────────────────


def wilson(p: float | None, n: float) -> tuple[float, float] | None:
    """Wilson score 95% interval for a proportion ``p`` observed over ``n`` (effective) trials."""
    if p is None or n <= 0:
        return None
    z2 = WILSON_Z * WILSON_Z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def effective_n(weights: list[float]) -> float:
    """Kish effective sample size: (sum w)^2 / sum w^2."""
    s2 = sum(w * w for w in weights)
    return (sum(weights) ** 2) / s2 if s2 > 0 else 0.0


def cohen_kappa(a: float, b: float, c: float, d: float) -> float | None:
    """Kappa from a 2x2 table: a = both fail, b = judge fail / label pass, c = judge pass / label fail,
    d = both pass (cells may be weighted). ``None`` when undefined (no rows, or chance agreement is 1)."""
    total = a + b + c + d
    if total <= 0:
        return None
    po = (a + d) / total
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (total * total)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def _rate(rows: list[tuple[float, bool]]) -> dict[str, Any]:
    """Weighted proportion of ``hit`` over ``(weight, hit)`` rows, its Wilson CI at the effective n."""
    weights = [w for w, _ in rows]
    total = sum(weights)
    n = len(rows)
    if n == 0 or total <= 0:
        return {"value": None, "low": None, "high": None, "n": 0, "n_eff": 0.0}
    p = sum(w for w, hit in rows if hit) / total
    n_eff = effective_n(weights)
    ci = wilson(p, n_eff)
    return {"value": round(p, 4), "low": round(ci[0], 4) if ci else None, "high": round(ci[1], 4) if ci else None,
            "n": n, "n_eff": round(n_eff, 2)}


def judge_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics for one judge over ``items`` = ``{verdict, label, weight, rubric_sha?}`` (its decisive and
    unsure verdicts on labelled rows; unparseable verdicts are left out by the caller)."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unsure_label_fail": 0, "unsure_label_pass": 0}
    cells = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    recall_rows: list[tuple[float, bool]] = []
    precision_rows: list[tuple[float, bool]] = []
    unsure_rows: list[tuple[float, bool]] = []
    for it in items:
        v, lab, w = it["verdict"], it["label"], float(it["weight"])
        unsure_rows.append((w, v == "unsure"))
        if lab == "fail":
            recall_rows.append((w, v == "fail"))
        if v == "pass":
            precision_rows.append((w, lab == "pass"))
        if v == "unsure":
            counts["unsure_label_fail" if lab == "fail" else "unsure_label_pass"] += 1
            continue
        key = ("tp" if lab == "fail" else "fp") if v == "fail" else ("fn" if lab == "fail" else "tn")
        counts[key] += 1
        cells[key] += w
    kappa = cohen_kappa(cells["tp"], cells["fp"], cells["fn"], cells["tn"])
    label_fail = sum(1 for it in items if it["label"] == "fail")
    label_pass = len(items) - label_fail
    reasons = []
    if label_fail < MIN_CLASS_LABELS:
        reasons.append(f"fewer than {MIN_CLASS_LABELS} labelled fails ({label_fail})")
    if label_pass < MIN_CLASS_LABELS:
        reasons.append(f"fewer than {MIN_CLASS_LABELS} labelled passes ({label_pass})")
    if len(items) < MIN_TOTAL_LABELS:
        reasons.append(f"fewer than {MIN_TOTAL_LABELS} labelled rows ({len(items)})")
    unsure = _rate(unsure_rows)
    return {
        "n": len(items),
        "n_eff": round(effective_n([float(it["weight"]) for it in items]), 2),
        "label_fail": label_fail,
        "label_pass": label_pass,
        "counts": counts,
        "fail_recall": _rate(recall_rows),
        "pass_precision": _rate(precision_rows),
        "unsure_rate": unsure["value"],
        "kappa": round(kappa, 4) if kappa is not None else None,
        "insufficient": bool(reasons),
        "insufficient_reasons": reasons,
    }


# A stratum where skipped rows are at least this share of (labelled + skipped) gets a plain caveat.
SKIP_WARN_SHARE = 0.2
SKIP_NOTE = "estimates may be optimistic; skipped rows are often the hard ones"


def tier_report(
    project_id: int, rows: list[dict[str, Any]], tier: str,
    *, pop: list[tuple[dict[str, Any], dict[str, Any], str]] | None = None,
) -> dict[str, Any]:
    """Per judge metrics against ``tier``'s current labels, stratum-weighted. A judge is never graded
    against labels it wrote itself (a reference model that is also a judge): those rows are left out of
    ITS metrics and the stratum weights are recomputed over the rows that remain (``self_rows`` counts
    them; ``self`` = every labelled row it judged was its own, so nothing is left to measure).
    ``pop`` = ``population(project_id, rows)`` when the caller already has it (one pass for several tiers)."""
    if pop is None:
        pop = population(project_id, rows)
    by_id = {int(r["id"]): (att, s) for r, att, s in pop}
    raw_by_id = {int(r["id"]): r for r, _, _ in pop}
    sidecar = read_labels(project_id)
    labelled, stale = current_labels(sidecar, tier, raw_by_id)
    skips = current_skips(sidecar, tier, raw_by_id)
    strata_pop = {s: 0 for s in STRATA}
    for _, _, s in pop:
        strata_pop[s] += 1
    strata_lab = {s: 0 for s in STRATA}
    for rid in labelled:
        strata_lab[by_id[rid][1]] += 1
    strata_skip = {s: 0 for s in STRATA}
    for rid in skips:
        strata_skip[by_id[rid][1]] += 1
    weight_of = {s: (strata_pop[s] / strata_lab[s]) if strata_lab[s] else 0.0 for s in STRATA}
    covered = sum(strata_pop[s] for s in STRATA if strata_lab[s])
    labellers = sorted({str(e.get("labeller") or "") for e in labelled.values()})

    # Per judge: (stratum, verdict, label, rubric sha, labeller) on the labelled rows it judged.
    per_judge: dict[str, list[dict[str, Any]]] = {}
    unparseable: dict[str, int] = {}
    for rid, entry in sorted(labelled.items()):
        att, stratum = by_id[rid]
        for e in att.get("review_assist_judges") or []:
            judge = str(e.get("judge"))
            if e.get("reason_code") == assist.UNPARSEABLE_REASON or e.get("verdict") not in assist.VERDICTS:
                unparseable[judge] = unparseable.get(judge, 0) + 1
                continue
            per_judge.setdefault(judge, []).append({
                "rid": rid, "stratum": stratum, "verdict": e["verdict"], "label": entry["label"],
                "rubric_sha": e.get("rubric_sha") or NOT_RECORDED, "own": entry.get("labeller") == judge,
            })
    judges = []
    for judge in sorted(set(per_judge) | set(unparseable)):
        judged = per_judge.get(judge, [])
        own = [it for it in judged if it["own"]]
        kept = [it for it in judged if not it["own"]]
        if own:
            # Weights over the labelled rows that remain once this judge's own labels are left out.
            own_ids = {rid for rid, e in labelled.items() if e.get("labeller") == judge}
            lab = {s: 0 for s in STRATA}
            for rid in labelled:
                if rid not in own_ids:
                    lab[by_id[rid][1]] += 1
            weights = {s: (strata_pop[s] / lab[s]) if lab[s] else 0.0 for s in STRATA}
        else:
            weights = weight_of
        items = [{**it, "weight": weights[it["stratum"]]} for it in kept]
        shas: dict[str, int] = {}
        for it in items:
            shas[it["rubric_sha"]] = shas.get(it["rubric_sha"], 0) + 1
        judges.append({
            "judge": judge,
            **judge_metrics(items),
            "unparseable": unparseable.get(judge, 0),
            "rubric_shas": dict(sorted(shas.items(), key=lambda kv: (-kv[1], kv[0]))),
            "rubric_mixed": len(shas) > 1,
            "self": bool(own) and not kept,
            "self_rows": len(own),
        })
    strata = {}
    warned = []
    for s in STRATA:
        offered = strata_lab[s] + strata_skip[s]
        share = strata_skip[s] / offered if offered else 0.0
        strata[s] = {"population": strata_pop[s], "labelled": strata_lab[s], "skipped": strata_skip[s],
                     "skip_share": round(share, 4), "weight": round(weight_of[s], 4)}
        if strata_skip[s] and share >= SKIP_WARN_SHARE:
            warned.append(s)
    return {
        "tier": tier,
        "labelled": len(labelled),
        "stale": stale,
        "skipped": len(skips),
        "labellers": labellers,
        "population": len(pop),
        "coverage": round(covered / len(pop), 4) if pop else None,
        "strata": strata,
        "skip_warning_strata": warned,
        "skip_note": SKIP_NOTE if warned else None,
        "judges": judges,
    }


def report(project_id: int, rows: list[dict[str, Any]], tiers: list[str]) -> list[dict[str, Any]]:
    pop = population(project_id, rows) if tiers else []
    return [tier_report(project_id, rows, t, pop=pop) for t in tiers]


# ── trust status (V3): what the review queue shows next to each judge and on the consensus gates ──

# A calibrated judge that catches fewer than this share of bad rows is flagged "below floor".
FAIL_RECALL_FLOOR = 0.7
# Worst first: a consensus group's trust is the worst status among the judges behind it.
TRUST_STATUSES: tuple[str, ...] = ("rubber_stamp", "below_floor", "uncalibrated", "insufficient", "calibrated_ok")

# (project id, tiers) -> (file-state key, reports), least recently used first. The report reads both
# sidecars and hashes every row; the queue is read often and the inputs rarely change, so it is recomputed
# only when one of the three files does.
_TRUST_CACHE: "OrderedDict[tuple[int, tuple[str, ...]], tuple[Any, list[dict[str, Any]]]]" = OrderedDict()
_TRUST_CACHE_MAX = 64
_CACHE_LOCK = threading.Lock()


def _file_state(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def report_state(project_id: int, rows_path: Path) -> tuple[Any, ...] | None:
    """The cache key: (mtime_ns, size) of the labels file, the verdict sidecar and the rows file; ``None``
    when there is no labels file (nothing to report). Take it BEFORE reading the rows: a write landing
    after it then leaves the entry under an older key, so the next read misses instead of serving a
    stale report under a newer key."""
    labels_state = _file_state(labels_path(project_id))
    if labels_state is None:
        return None
    return labels_state, _file_state(assist.sidecar_path(project_id)), _file_state(rows_path)


def cached_report_hit(project_id: int, tiers: list[str], state: Any) -> list[dict[str, Any]] | None:
    """The cached report for ``state``, or ``None`` on a miss."""
    slot = (int(project_id), tuple(tiers))
    with _CACHE_LOCK:
        hit = _TRUST_CACHE.get(slot)
        if hit is None or hit[0] != state:
            return None
        _TRUST_CACHE.move_to_end(slot)
        return hit[1]


def compute_report(
    project_id: int, rows: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]], tiers: list[str], state: Any,
) -> list[dict[str, Any]]:
    """Compute ``report`` and cache it under ``state`` (evicting the least recently used entry when full)."""
    reports = report(project_id, rows() if callable(rows) else rows, tiers)
    slot = (int(project_id), tuple(tiers))
    with _CACHE_LOCK:
        _TRUST_CACHE[slot] = (state, reports)
        _TRUST_CACHE.move_to_end(slot)
        while len(_TRUST_CACHE) > _TRUST_CACHE_MAX:
            _TRUST_CACHE.popitem(last=False)
    return reports


def cached_report(
    project_id: int, rows: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]], tiers: list[str], rows_path: Path,
    *, state: Any = None,
) -> list[dict[str, Any]]:
    """``report`` for the review queue, cached per (labels file, verdict sidecar, rows file) state
    (``state`` from ``report_state``, taken before the rows were read; computed here when omitted). No
    labels file (every legacy project) → ``[]`` without computing anything. ``rows`` (or a callable
    returning them, called only on a miss) must be the rows as stored: no read-time verdict keys, which
    would change the row signatures."""
    if state is None:
        state = report_state(project_id, rows_path)
    if state is None:
        return []
    hit = cached_report_hit(project_id, tiers, state)
    return hit if hit is not None else compute_report(project_id, rows, tiers, state)


def _tier_trust(entry: dict[str, Any] | None, tier: str) -> dict[str, Any]:
    """One judge's status against one tier: ``calibrated_ok`` / ``below_floor`` (enough labels),
    ``insufficient`` (some labels, not enough) or ``uncalibrated`` (none it can be graded on)."""
    if not entry or not entry.get("n"):
        return {"status": "uncalibrated", "tier": tier, "fail_recall": None, "ci": None, "n": 0}
    fr = entry.get("fail_recall") or {}
    low, high = fr.get("low"), fr.get("high")
    out = {
        "tier": tier, "fail_recall": fr.get("value"), "ci": [low, high] if low is not None and high is not None else None,
        "n": int(entry.get("n") or 0),
    }
    if entry.get("insufficient"):
        return {"status": "insufficient", **out, "reasons": list(entry.get("insufficient_reasons") or [])}
    # The UPPER Wilson bound is compared with the floor: a judge is flagged only when even the most
    # favourable reading of its labels (95%) leaves it below the floor, i.e. we are confident it misses
    # bad rows. A point estimate under the floor with a wide interval is not enough evidence to flag it.
    below = high is not None and high < FAIL_RECALL_FLOOR
    return {"status": "below_floor" if below else "calibrated_ok", **out}


def judge_trust(stat: dict[str, Any], reports: list[dict[str, Any]], label_source: str) -> dict[str, Any]:
    """A judge's trust status for the review queue, by precedence: ``rubber_stamp`` (from its pass rate,
    no labels needed) > ``below_floor`` / ``calibrated_ok`` (calibrated in a tier of ``label_source``) >
    ``insufficient`` (some labels) > ``uncalibrated``. With ``both``, either tier being sufficient counts as
    calibrated; when both are and they disagree, the worse status wins. The numbers (``tier``,
    ``fail_recall``, ``ci``, ``n``) are those of the tier that decided the status."""
    judge = stat.get("judge")
    per_tier = []
    for tier in tiers_for(label_source):
        rep = next((r for r in reports if r.get("tier") == tier), None)
        entry = next((j for j in (rep or {}).get("judges") or [] if j.get("judge") == judge), None)
        per_tier.append(_tier_trust(entry, tier))
    rank = {s: i for i, s in enumerate(TRUST_STATUSES)}
    calibrated = [t for t in per_tier if t["status"] in ("below_floor", "calibrated_ok")]
    if calibrated:
        chosen = min(calibrated, key=lambda t: rank[t["status"]])
    else:
        partial = [t for t in per_tier if t["status"] == "insufficient"]
        chosen = max(partial, key=lambda t: t["n"]) if partial else per_tier[0]
    if label_source == "both" and chosen["status"] == "uncalibrated":
        chosen = {**chosen, "tier": None}
    out = {**chosen, "label_source": label_source, "floor": FAIL_RECALL_FLOOR, "min_labels": MIN_TOTAL_LABELS}
    if stat.get("rubber_stamp"):
        # Precedence: a judge that passes nearly everything is flagged whatever its labels say (the
        # calibration numbers it has, if any, stay alongside as ``calibrated_status``).
        out = {**out, "status": "rubber_stamp", "calibrated_status": chosen["status"]}
    return out


def trust_for_queue(
    project_id: int, stored_rows: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]], judge_stats: list[dict[str, Any]],
    runtime_config: Any, rows_path: Path, *, state: Any = None,
) -> dict[str, dict[str, Any]]:
    """``{judge: trust}`` for every judge in the queue's ``judge_stats`` (synchronous; the queue uses
    ``trust_from_reports`` and computes a missed report off the event loop)."""
    if not judge_stats:
        return {}
    source = label_source_of(runtime_config)
    reports = cached_report(project_id, stored_rows, tiers_for(source), rows_path, state=state)
    return trust_from_reports(judge_stats, reports, source)


def trust_from_reports(judge_stats: list[dict[str, Any]], reports: list[dict[str, Any]], label_source: str) -> dict[str, dict[str, Any]]:
    return {str(s.get("judge")): judge_trust(s, reports, label_source) for s in judge_stats}


def worst_trust(statuses: list[str]) -> str | None:
    """The worst of ``statuses`` (``TRUST_STATUSES`` order); ``None`` for none."""
    known = [s for s in statuses if s in TRUST_STATUSES]
    return min(known, key=TRUST_STATUSES.index) if known else None


# ── reference-model labelling ────────────────────────────────────────────


async def run_reference_labelling(
    project_id: int,
    rows: list[dict[str, Any]],
    backend: Any,
    *,
    rubric_text: str | None = None,
    rows_per_call: int = assist.DEFAULT_ROWS_PER_CALL,
    progress: assist.ProgressFn | None = None,
) -> dict[str, Any]:
    """Label ``rows`` blind with ``backend`` into the ``reference_model`` tier. Same prompt as review
    assist (``build_prompt`` + its rubric + ``rubric_text``), built from the rows WITHOUT any attached
    verdict. ``unsure`` (and an unparseable answer) is not a label: the row gets a SKIP marker in the
    tier (so a re-run does not re-pick it, and the report counts it). A row too long to judge whole is
    skipped the same way. A failed call records nothing (transient; a re-run retries those rows). Never writes ``review_assist.json`` or ``synthetic.jsonl``. Labels are
    saved after every call (a budget stop keeps what was written). Raises ``ValueError`` when no row
    got a label."""
    import asyncio

    from app.services import llm_usage
    from app.services.llm_budget_guard import LlmBudgetExceeded
    from app.services.synth_playbook_service import _call, _complete_direct, batch_usage_fields

    blind = [blind_row(r) for r in rows]
    if not blind:
        raise ValueError("No rows to label: every row with a judge verdict already has a reference label.")
    labeller = assist.judge_label(backend)
    per_call = max(1, min(assist.MAX_ROWS_PER_CALL, int(rows_per_call or assist.DEFAULT_ROWS_PER_CALL)))
    renders = {int(r["id"]): assist.render_row_ex(r) for r in blind}
    judgeable = [r for r in blind if not renders[int(r["id"])].too_long]
    chunks = assist.pack_chunks(judgeable, renders, per_call, rubric_text)
    system = assist.system_prompt()
    counts = {"pass": 0, "fail": 0, "unsure": 0, "unparseable": 0}
    stats = {"calls": 0, "labelled": 0, "skips": 0}
    call_errors: list[str] = []
    too_long = [r for r in blind if renders[int(r["id"])].too_long]
    try:
        if too_long:
            await asyncio.to_thread(save_labels, project_id, "reference_model", {
                int(r["id"]): skip_entry(r, reason="too_long_to_judge", labeller=labeller) for r in too_long
            })
            stats["skips"] += len(too_long)
        for i, chunk in enumerate(chunks):
            call = _call(
                backend, assist.build_prompt(chunk, rubric_text=rubric_text, renders=renders), system,
                max(512, min(int(getattr(backend, "max_output_tokens", 4096) or 4096), assist.PER_ROW_TOKENS * len(chunk) + 256)),
                temperature=assist.JUDGE_TEMPERATURE,
            )
            try:
                raw = await _complete_direct(backend, call, None)
            except LlmBudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 — a failed call labels nothing; its rows stay unlabelled
                call_errors.append(str(exc)[:200])
                raw = None
            stats["calls"] += 1
            parsed = assist.parse_verdicts(raw, chunk)
            by_id = {int(r["id"]): r for r in chunk}
            entries: dict[int, dict[str, Any]] = {}
            for rid, v in parsed.items():
                if v["reason_code"] in (assist.UNPARSEABLE_REASON, assist.CONFLICT_REASON):
                    counts["unparseable"] += 1
                    # A failed CALL is transient (the model never saw the row): no skip marker, a re-run retries it.
                    if raw is not None:
                        entries[rid] = skip_entry(by_id[rid], reason=v["reason_code"], labeller=labeller)
                    continue
                if v["verdict"] not in LABELS:
                    counts["unsure"] += 1
                    entries[rid] = skip_entry(by_id[rid], reason="unsure", labeller=labeller)
                    continue
                counts[v["verdict"]] += 1
                stratum = stratum_for_row(project_id, by_id[rid]) or "single_unsure"
                entries[rid] = label_entry(by_id[rid], v["verdict"], stratum=stratum, labeller=labeller,
                                           reason=v.get("note") or None)
            if entries:
                await asyncio.to_thread(save_labels, project_id, "reference_model", entries)
                stats["labelled"] += sum(1 for e in entries.values() if not e.get("skipped"))
                stats["skips"] += sum(1 for e in entries.values() if e.get("skipped"))
            if progress is not None:
                await progress((i + 1) / max(1, len(chunks)), f"Reference-labelled {stats['labelled']} of {len(blind)} rows")
    except BaseException as exc:
        llm_usage.attach_result_usage(exc, batch_usage_fields(backend))
        raise
    if stats["labelled"] == 0:
        raise ValueError(
            f"Reference labelling labelled 0 of {len(blind)} rows ({counts['unsure']} unsure, "
            f"{counts['unparseable']} unreadable, {len(blind) - len(judgeable)} too long to judge whole"
            + (f"; call errors: {call_errors[0]}" if call_errors else "") + ")."
        )
    return {
        "rows_sampled": len(blind),
        "labelled": stats["labelled"],
        "labels": {"pass": counts["pass"], "fail": counts["fail"]},
        "skipped_unsure": counts["unsure"],
        "skipped_unparseable": counts["unparseable"],
        "skipped_too_long": len(blind) - len(judgeable),
        "skips_recorded": stats["skips"],
        "labeller": labeller,
        "calls": stats["calls"],
        **({"call_errors": call_errors} if call_errors else {}),
    }

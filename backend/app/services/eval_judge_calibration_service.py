"""Eval-judge calibration (Judge calibration V4): how far the EVAL judges can be trusted.

The V2/V3 maths (stratum weights, Wilson intervals at the Kish effective n, Cohen's kappa), the two
label tiers (``human`` / ``reference_model``, never pooled), skip markers, the stratified sampler and the
project setting ``judge_calibration.label_source`` are reused from ``judge_calibration_service``; this
module only adds what differs for eval judges.

Judges and scales. Two vocabularies ("scales"):

- ``verdict``: the answer judge (``answer_judge_service``: correct / partial / wrong), also as used for
  remediation patches and code examples (``task`` = answer / remediation / code_example on each row);
- ``rubric``: the structured-plan rubric judge (``rubric_service``: the plan passes or fails the rubric).

Population: every row in the persisted eval predictions (``eval_predictions_service``) that carries a
verdict on the scale, keyed by ``answer_judge_service.content_key`` of what the judge read (question,
reference, prediction) -- NOT by eval result, so the same answer judged in a later eval (or by another
judge) shares its label. A judge's verdict on a content key is its most recent one.

Labels: ``DATA_DIR/projects/<pid>/eval_judge_calibration.json`` shaped
``{scale: {content key: {tier: {label, reason?, at, labeller, stratum?, judge?}}}}`` (or a skip marker
``{skipped: true, reason, at, labeller}``); one writer path (lock + temp file + ``os.replace``). Labels use
the judge's vocabulary (correct / partial / wrong; pass / fail).

Sampling is stratified by the CHOSEN judge's verdict (correct / partial / wrong, or pass / fail),
seeded, shuffled, excluding content already labelled or skipped in the tier; a sampled row is shown as
question / reference / answer only (no verdict, score, reason or judge name).

Report per judge (positive class = ``correct`` / ``pass``), each labelled row weighted by (the judge's
stratum population / labelled rows in that stratum): exact agreement; strict agreement (correct vs not);
pass precision and recall; wrong-recall (of rows the labeller called wrong / fail, the share the judge
called wrong / fail); too-strict rate (judge says partial / wrong where the labeller says correct) and
too-lenient rate (judge says correct where the labeller says partial / wrong), both as shares of ALL
labelled rows; linear-weighted kappa on the three ordered classes (Cohen's kappa for pass / fail). A
judge is never graded on labels a reference model wrote as itself.

Trust (``judge_trust``): ``below_floor`` when the Wilson UPPER bound of strict agreement is under
``STRICT_AGREEMENT_FLOOR`` (we are confident it disagrees too often) or the Wilson LOWER bound of the
too-strict / too-lenient rate is over ``TOO_STRICT_CEILING`` / ``TOO_LENIENT_CEILING`` (we are confident
it leans that way); ``calibrated_ok``; ``insufficient`` (some labels, not enough); ``uncalibrated``.
Nothing is gated: the settings card and the eval summary show it.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services import eval_predictions_service as preds
from app.services import judge_calibration_service as calib

SIDECAR_NAME = "eval_judge_calibration.json"
SCALES: tuple[str, ...] = ("verdict", "rubric")
LABELS: dict[str, tuple[str, ...]] = {"verdict": ("correct", "partial", "wrong"), "rubric": ("pass", "fail")}
# Ordered worst → best (the weighted-kappa distance).
ORDER: dict[str, tuple[str, ...]] = {"verdict": ("wrong", "partial", "correct"), "rubric": ("fail", "pass")}
POSITIVE = {"verdict": "correct", "rubric": "pass"}
NEGATIVE = {"verdict": "wrong", "rubric": "fail"}
TIERS = calib.TIERS
HUMAN_LABELLER = calib.HUMAN_LABELLER
DEFAULT_SAMPLE = calib.DEFAULT_SAMPLE
MAX_SAMPLE = calib.MAX_SAMPLE
MIN_CLASS_LABELS = calib.MIN_CLASS_LABELS
MIN_TOTAL_LABELS = calib.MIN_TOTAL_LABELS
REASON_LIMIT = calib.REASON_LIMIT

# Trust thresholds (module constants, documented in agent-notes/evaluation.md).
# Flag a judge when even the most favourable reading (Wilson upper bound, 95%) of its correct-vs-not
# agreement with the labels is under this floor.
STRICT_AGREEMENT_FLOOR = 0.7
# Flag a judge as too strict / too lenient when the Wilson LOWER bound of that rate (a share of all
# labelled rows) is over this ceiling: confident that it marks down (or passes) at least this share.
TOO_STRICT_CEILING = 0.15
TOO_LENIENT_CEILING = 0.15
TRUST_STATUSES: tuple[str, ...] = ("below_floor", "uncalibrated", "insufficient", "calibrated_ok")

_LOCK = threading.Lock()


def _check_scale(scale: str) -> str:
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {', '.join(SCALES)}")
    return scale


def same_model(a: str | None, b: str | None) -> bool:
    """Do two judge labels name the same model? A cloud synth backend describes itself as
    ``cloud:<provider>:<model>`` while the eval judge labels it ``<provider>:<model>``."""
    def norm(s: str | None) -> str:
        from app.services.synth_playbook_service import safe_backend_label

        text = safe_backend_label(str(s or "").strip()).strip()
        return text[6:] if text.startswith("cloud:") else text

    return bool(norm(a)) and norm(a) == norm(b)


# ── sidecar ──────────────────────────────────────────────────────────────


def labels_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(int(project_id)) / SIDECAR_NAME


def _read(path: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for scale in SCALES:
        block = data.get(scale)
        if not isinstance(block, dict):
            continue
        kept_scale: dict[str, dict[str, dict[str, Any]]] = {}
        for ck, tiers in block.items():
            if not isinstance(tiers, dict):
                continue
            kept = {t: e for t, e in tiers.items() if t in TIERS and isinstance(e, dict)
                    and (e.get("label") in LABELS[scale] or e.get("skipped") is True)}
            if kept:
                kept_scale[str(ck)] = kept
        if kept_scale:
            out[scale] = kept_scale
    return out


def read_labels(project_id: int) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    path = labels_path(project_id)
    if not path.exists():
        return {}
    with _LOCK:
        return _read(path)


def save_labels(project_id: int, scale: str, tier: str, entries: dict[str, dict[str, Any]]) -> None:
    """Merge ``{content key: entry}`` (labels or skip markers) into ``scale`` / ``tier``."""
    _check_scale(scale)
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}")
    if not entries:
        return
    path = labels_path(project_id)
    with _LOCK:
        data = _read(path)
        block = data.setdefault(scale, {})
        for ck, entry in entries.items():
            block.setdefault(str(ck), {})[tier] = dict(entry)
        calib._write(path, data)


def delete_label(project_id: int, scale: str, content_key: str, tier: str) -> bool:
    path = labels_path(project_id)
    if not path.exists():
        return False
    with _LOCK:
        data = _read(path)
        row = data.get(scale, {}).get(str(content_key))
        if not row or tier not in row:
            return False
        del row[tier]
        if not row:
            del data[scale][str(content_key)]
        calib._write(path, data)
    return True


def _clean_reason(reason: str | None) -> str:
    return " ".join(str(reason or "").split())[:REASON_LIMIT]


def label_entry(label: str, *, labeller: str, reason: str | None = None, stratum: str | None = None,
                judge: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"label": label, "at": datetime.now(timezone.utc).isoformat(), "labeller": labeller}
    if stratum:
        entry["stratum"] = stratum  # a record of where it was drawn; weights use the CURRENT verdicts
    if judge:
        entry["judge"] = judge
    text = _clean_reason(reason)
    if text:
        entry["reason"] = text
    return entry


def skip_entry(*, reason: str, labeller: str) -> dict[str, Any]:
    return {"skipped": True, "reason": _clean_reason(reason) or "skipped",
            "at": datetime.now(timezone.utc).isoformat(), "labeller": labeller}


def _tier_entries(labels: dict[str, Any], scale: str, tier: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """``({content key: label entry}, {content key: skip marker})`` for one scale / tier."""
    got, skipped = {}, {}
    for ck, tiers in (labels.get(scale) or {}).items():
        e = tiers.get(tier)
        if not e:
            continue
        if e.get("label") in LABELS[scale]:
            got[ck] = e
        elif e.get("skipped") is True:
            skipped[ck] = e
    return got, skipped


# ── population ───────────────────────────────────────────────────────────


async def live_result_ids(db: Any, project_id: int) -> set[int]:
    """Ids of the project's EvalResults that still exist (a file whose result was deleted is ignored)."""
    from sqlalchemy import select

    from app.models.experiment import EvalResult, Experiment

    rows = await db.execute(
        select(EvalResult.id).join(Experiment, Experiment.id == EvalResult.experiment_id).where(Experiment.project_id == project_id)
    )
    return {int(r) for r in rows.scalars().all()}


def _over_cap(rec: dict[str, Any], block: dict[str, Any], texts: tuple[str, str, str]) -> bool:
    """A stored text was cut at ``TEXT_CAP`` (or a chat history was trimmed): the labeller and a reference
    model would grade a different answer from the one the judge saw, so the row is left out."""
    if rec.get("history_truncated"):
        return True
    return any(len(t) >= preds.TEXT_CAP for t in texts)


def population(project_id: int, scale: str, live_ids: set[int] | None = None) -> dict[str, dict[str, Any]]:
    """``{content key: item}`` for every judged answer on ``scale`` across the stored eval predictions.
    ``item`` = ``{content_key, question, reference, prediction, task?, rubric_sha?, passages?,
    verdicts: {judge: verdict}, eval_result_ids}``; a judge's most recent verdict on the content wins."""
    return population_ex(project_id, scale, live_ids)[0]


def population_ex(project_id: int, scale: str, live_ids: set[int] | None = None) -> tuple[dict[str, dict[str, Any]], int]:
    """``(population, over_cap)``: ``over_cap`` counts the distinct judged answers left out because a stored
    text hit ``TEXT_CAP``. ``live_ids`` (when given) keeps only files whose EvalResult still exists."""
    _check_scale(scale)
    out: dict[str, dict[str, Any]] = {}
    over: set[str] = set()
    for rid, rec in preds.iter_project_records(project_id):
        if live_ids is not None and rid not in live_ids:
            continue
        block = rec.get("judge") if scale == "verdict" else rec.get("rubric")
        if not isinstance(block, dict) or block.get("scale") != scale:
            continue
        ck = block.get("content_key")
        if not ck:
            continue
        if scale == "verdict":
            verdict = block.get("verdict")
        else:
            verdict = "pass" if block.get("passed") is True else "fail" if block.get("passed") is False else None
        if verdict not in LABELS[scale]:
            continue
        question = block.get("question")
        if question is None:
            history = rec.get("history")
            question = rec.get("prompt") if not history else "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history if isinstance(m, dict))
        texts = (str(question or ""), str(block.get("reference", rec.get("reference")) or ""),
                 str(block.get("prediction", rec.get("prediction")) or ""))
        if ck in over or _over_cap(rec, block, texts):
            over.add(ck)
            out.pop(ck, None)
            continue
        item = out.get(ck)
        if item is None:
            item = {
                "content_key": ck,
                "question": texts[0],
                "reference": texts[1],
                "prediction": texts[2],
                "verdicts": {},
                "eval_result_ids": [],
            }
            if scale == "verdict":
                item["task"] = block.get("task") or "answer"
            else:
                item["rubric_sha"] = block.get("rubric_sha")
                if block.get("passages"):
                    item["passages"] = block["passages"]
            out[ck] = item
        elif scale == "rubric" and block.get("rubric_sha"):
            item["rubric_sha"] = block.get("rubric_sha")  # the newest grading's rubric
        judge = str(block.get("label") or "(unnamed judge)")
        item["verdicts"][judge] = verdict
        if rid not in item["eval_result_ids"]:
            item["eval_result_ids"].append(rid)
    return out, len(over)


def judges_of(pop: dict[str, dict[str, Any]], scale: str) -> list[dict[str, Any]]:
    """``[{judge, rows, by_verdict}]``, most rows first."""
    counts: dict[str, dict[str, int]] = {}
    for item in pop.values():
        for judge, v in item["verdicts"].items():
            counts.setdefault(judge, {lab: 0 for lab in LABELS[scale]})[v] += 1
    rows = [{"judge": j, "rows": sum(c.values()), "by_verdict": c} for j, c in counts.items()]
    return sorted(rows, key=lambda r: (-r["rows"], r["judge"]))


def default_judge(pop: dict[str, dict[str, Any]], scale: str) -> str | None:
    js = judges_of(pop, scale)
    return js[0]["judge"] if js else None


def budgeted_passages(item: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, bool]:
    """A rubric row's passages cut to the review-assist context budget (``render_row_ex``'s per-passage
    shares, ``[truncated]`` marks): ``(passages | None, cut)``. The SAME list goes to the human labeller and
    to the reference model, so both grade with the same grounding."""
    passages = item.get("passages")
    if not passages:
        return None, False
    from app.services import synth_review_assist_service as assist

    core = len(item.get("question") or "") + len(item.get("reference") or "") + len(item.get("prediction") or "")
    budget = max(assist.MIN_CONTEXT_BUDGET, assist.ROW_BUDGET["structured_plan"] - core)
    cut_value, cut = assist._budget_context(list(passages), budget)
    return (cut_value if isinstance(cut_value, list) else list(passages)), bool(cut)


def render_blind(item: dict[str, Any], scale: str) -> dict[str, Any]:
    """What the labeller sees: question, reference answer, the model's answer (and, for a rubric row, the
    passages the reference model also gets). No verdict, score, reason, judge name or eval id."""
    out = {"content_key": item["content_key"], "scale": scale, "question": item["question"],
           "reference": item["reference"], "prediction": item["prediction"]}
    passages, cut = budgeted_passages(item)
    if passages:
        out["passages"] = passages
        out["passages_truncated"] = cut
    return out


def sample(project_id: int, scale: str, tier: str, n: int, *, judge: str | None = None, seed: int = 0,
           pop: dict[str, dict[str, Any]] | None = None, live_ids: set[int] | None = None) -> tuple[str | None, list[dict[str, Any]]]:
    """``(judge, items)``: up to ``n`` answers ``judge`` (default: the judge with most rows) judged, not yet
    labelled or skipped in ``tier``, stratified by that judge's verdict, seeded and shuffled."""
    pop = population(project_id, scale, live_ids) if pop is None else pop
    judge = judge or default_judge(pop, scale)
    if judge is None:
        return None, []
    got, skipped = _tier_entries(read_labels(project_id), scale, tier)
    done = set(got) | set(skipped)
    candidates = [(item, item["verdicts"][judge]) for ck, item in pop.items() if judge in item["verdicts"] and ck not in done]
    picked = calib.stratified_sample(candidates, max(0, min(MAX_SAMPLE, int(n))), seed,
                                     strata=LABELS[scale], key=lambda r: r["content_key"])
    return judge, picked


# ── progress ─────────────────────────────────────────────────────────────


def progress(project_id: int, live_ids: set[int] | None = None) -> dict[str, Any]:
    labels = read_labels(project_id)
    out: dict[str, Any] = {}
    for scale in SCALES:
        pop, over_cap = population_ex(project_id, scale, live_ids)
        tiers = {}
        for tier in TIERS:
            got, skipped = _tier_entries(labels, scale, tier)
            cur = {ck: e for ck, e in got.items() if ck in pop}
            tiers[tier] = {
                "labelled": len(cur),
                "orphaned": len(got) - len(cur),
                "skipped": sum(1 for ck in skipped if ck in pop),
                "by_label": {lab: sum(1 for e in cur.values() if e["label"] == lab) for lab in LABELS[scale]},
                "labellers": sorted({str(e.get("labeller") or "") for e in cur.values()}),
            }
        out[scale] = {"population": len(pop), "over_cap": over_cap, "judges": judges_of(pop, scale), "tiers": tiers,
                      "labels": list(LABELS[scale])}
    return {"scales": out}


# ── statistics ───────────────────────────────────────────────────────────


def weighted_kappa(cells: dict[tuple[str, str], float], order: tuple[str, ...]) -> float | None:
    """Linear-weighted kappa on ordered classes from ``{(judge class, label class): weight}`` cells:
    agreement weight ``1 - |i - j| / (k - 1)``; ``(Po - Pe) / (1 - Pe)``. With two classes this is Cohen's
    kappa. ``None`` when there are no rows or chance agreement is 1."""
    k = len(order)
    total = sum(cells.values())
    if total <= 0 or k < 2:
        return None
    idx = {c: i for i, c in enumerate(order)}
    row = {c: sum(w for (j, _), w in cells.items() if j == c) / total for c in order}
    col = {c: sum(w for (_, lab), w in cells.items() if lab == c) / total for c in order}

    def agree(a: str, b: str) -> float:
        return 1 - abs(idx[a] - idx[b]) / (k - 1)

    po = sum(agree(j, lab) * w for (j, lab), w in cells.items()) / total
    pe = sum(agree(a, b) * row[a] * col[b] for a in order for b in order)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def verdict_metrics(items: list[dict[str, Any]], scale: str) -> dict[str, Any]:
    """Metrics for one judge over ``items`` = ``{verdict, label, weight}``."""
    pos, neg, order = POSITIVE[scale], NEGATIVE[scale], ORDER[scale]
    rate = calib.weighted_rate
    exact, strict, precision, recall, wrong_recall, too_strict, too_lenient = [], [], [], [], [], [], []
    cells: dict[tuple[str, str], float] = {}
    confusion = {v: {lab: 0 for lab in LABELS[scale]} for v in LABELS[scale]}
    for it in items:
        v, lab, w = it["verdict"], it["label"], float(it["weight"])
        exact.append((w, v == lab))
        strict.append((w, (v == pos) == (lab == pos)))
        if v == pos:
            precision.append((w, lab == pos))
        if lab == pos:
            recall.append((w, v == pos))
        if lab == neg:
            wrong_recall.append((w, v == neg))
        too_strict.append((w, lab == pos and v != pos))
        too_lenient.append((w, v == pos and lab != pos))
        cells[(v, lab)] = cells.get((v, lab), 0.0) + w
        confusion[v][lab] += 1
    kappa = weighted_kappa(cells, order)
    label_pos = sum(1 for it in items if it["label"] == pos)
    label_other = len(items) - label_pos
    reasons = []
    other_name = "partial / wrong" if scale == "verdict" else "fail"
    if label_pos < MIN_CLASS_LABELS:
        reasons.append(f"fewer than {MIN_CLASS_LABELS} labelled {pos} ({label_pos})")
    if label_other < MIN_CLASS_LABELS:
        reasons.append(f"fewer than {MIN_CLASS_LABELS} labelled {other_name} ({label_other})")
    if len(items) < MIN_TOTAL_LABELS:
        reasons.append(f"fewer than {MIN_TOTAL_LABELS} labelled answers ({len(items)})")
    return {
        "n": len(items),
        "n_eff": round(calib.effective_n([float(it["weight"]) for it in items]), 2),
        "by_label": {lab: sum(1 for it in items if it["label"] == lab) for lab in LABELS[scale]},
        "confusion": confusion,
        "exact_agreement": rate(exact),
        "strict_agreement": rate(strict),
        "pass_precision": rate(precision),
        "pass_recall": rate(recall),
        "wrong_recall": rate(wrong_recall),
        "too_strict": rate(too_strict),
        "too_lenient": rate(too_lenient),
        "kappa": round(kappa, 4) if kappa is not None else None,
        "kappa_kind": "linear_weighted" if scale == "verdict" else "cohen",
        "insufficient": bool(reasons),
        "insufficient_reasons": reasons,
    }


def tier_report(project_id: int, scale: str, tier: str, *, pop: dict[str, dict[str, Any]] | None = None,
                labels: dict[str, Any] | None = None, over_cap: int = 0, live_ids: set[int] | None = None) -> dict[str, Any]:
    """Per judge metrics on ``scale`` against ``tier``'s labels, weighted by the judge's own verdict strata."""
    if pop is None:
        pop, over_cap = population_ex(project_id, scale, live_ids)
    labels = read_labels(project_id) if labels is None else labels
    got, skipped = _tier_entries(labels, scale, tier)
    labelled = {ck: e for ck, e in got.items() if ck in pop}
    skips = {ck for ck in skipped if ck in pop}
    judges = []
    for j in judges_of(pop, scale):
        judge = j["judge"]
        strata_pop = dict(j["by_verdict"])
        own = {ck for ck, e in labelled.items() if judge in pop[ck]["verdicts"] and same_model(e.get("labeller"), judge)}
        kept = [ck for ck in sorted(labelled) if judge in pop[ck]["verdicts"] and ck not in own]
        strata_lab = {lab: 0 for lab in LABELS[scale]}
        for ck in kept:
            strata_lab[pop[ck]["verdicts"][judge]] += 1
        strata_skip = {lab: 0 for lab in LABELS[scale]}
        for ck in skips:
            if judge in pop[ck]["verdicts"]:
                strata_skip[pop[ck]["verdicts"][judge]] += 1
        weight = {s: (strata_pop[s] / strata_lab[s]) if strata_lab[s] else 0.0 for s in LABELS[scale]}
        items = [{"verdict": pop[ck]["verdicts"][judge], "label": labelled[ck]["label"],
                  "weight": weight[pop[ck]["verdicts"][judge]]} for ck in kept]
        strata, warned = {}, []
        for s in LABELS[scale]:
            offered = strata_lab[s] + strata_skip[s]
            share = strata_skip[s] / offered if offered else 0.0
            strata[s] = {"population": strata_pop[s], "labelled": strata_lab[s], "skipped": strata_skip[s],
                         "skip_share": round(share, 4), "weight": round(weight[s], 4)}
            if strata_skip[s] and share >= calib.SKIP_WARN_SHARE:
                warned.append(s)
        covered = sum(strata_pop[s] for s in LABELS[scale] if strata_lab[s])
        tasks: dict[str, int] = {}
        shas: dict[str, int] = {}
        for ck in kept:
            if scale == "verdict":
                t = pop[ck].get("task") or "answer"
                tasks[t] = tasks.get(t, 0) + 1
            else:
                sha = str(pop[ck].get("rubric_sha") or calib.NOT_RECORDED)
                shas[sha] = shas.get(sha, 0) + 1
        judges.append({
            "judge": judge,
            "scale": scale,
            **verdict_metrics(items, scale),
            "population": j["rows"],
            "coverage": round(covered / j["rows"], 4) if j["rows"] else None,
            "strata": strata,
            "skipped": sum(strata_skip.values()),
            "skip_warning_strata": warned,
            "skip_note": calib.SKIP_NOTE if warned else None,
            "tasks": tasks,
            "rubric_shas": shas,
            "rubric_mixed": len(shas) > 1,
            "self_rows": len(own),
            "self": bool(own) and not kept,
        })
    return {
        "tier": tier,
        "scale": scale,
        "labelled": len(labelled),
        "skipped": len(skips),
        "labellers": sorted({str(e.get("labeller") or "") for e in labelled.values()}),
        "population": len(pop),
        "over_cap": over_cap,
        "judges": judges,
    }


def report(project_id: int, scale: str, tiers: list[str], live_ids: set[int] | None = None) -> list[dict[str, Any]]:
    pop, over_cap = population_ex(project_id, scale, live_ids) if tiers else ({}, 0)
    labels = read_labels(project_id)
    return [tier_report(project_id, scale, t, pop=pop, labels=labels, over_cap=over_cap) for t in tiers]


# ── trust ────────────────────────────────────────────────────────────────

_CACHE: "OrderedDict[tuple[int, str, tuple[str, ...]], tuple[Any, list[dict[str, Any]]]]" = OrderedDict()
_CACHE_MAX = 64
_CACHE_LOCK = threading.Lock()


def _state(project_id: int, live_ids: set[int] | None) -> Any:
    try:
        st = labels_path(project_id).stat()
        lab = (st.st_mtime_ns, st.st_size)
    except OSError:
        lab = None
    return lab, preds.files_state(project_id), tuple(sorted(live_ids)) if live_ids is not None else None


def cached_report(project_id: int, scale: str, tiers: list[str], live_ids: set[int] | None = None) -> list[dict[str, Any]]:
    """``report`` cached per (labels file, predictions files) state. No labels file → ``[]`` (nothing to
    measure, every judge uncalibrated) without reading any predictions."""
    if not labels_path(project_id).exists():
        return []
    state = _state(project_id, live_ids)
    slot = (int(project_id), scale, tuple(tiers))
    with _CACHE_LOCK:
        hit = _CACHE.get(slot)
        if hit is not None and hit[0] == state:
            _CACHE.move_to_end(slot)
            return hit[1]
    reports = report(project_id, scale, tiers, live_ids)
    with _CACHE_LOCK:
        _CACHE[slot] = (state, reports)
        _CACHE.move_to_end(slot)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return reports


def _rate_tuple(rate: dict[str, Any] | None) -> tuple[Any, Any, Any]:
    r = rate or {}
    return r.get("value"), r.get("low"), r.get("high")


def _tier_trust(entry: dict[str, Any] | None, tier: str) -> dict[str, Any]:
    if not entry or not entry.get("n"):
        return {"status": "uncalibrated", "tier": tier, "n": 0, "agreement": None, "strict_agreement": None,
                "ci": None, "too_strict": None, "too_lenient": None, "flags": []}
    agreement, _, _ = _rate_tuple(entry.get("exact_agreement"))
    strict, s_low, s_high = _rate_tuple(entry.get("strict_agreement"))
    ts, ts_low, _ = _rate_tuple(entry.get("too_strict"))
    tl, tl_low, _ = _rate_tuple(entry.get("too_lenient"))
    out = {
        "tier": tier, "n": int(entry.get("n") or 0), "agreement": agreement, "strict_agreement": strict,
        "ci": [s_low, s_high] if s_low is not None and s_high is not None else None,
        "too_strict": ts, "too_lenient": tl, "kappa": entry.get("kappa"),
    }
    if entry.get("insufficient"):
        return {"status": "insufficient", **out, "flags": [], "reasons": list(entry.get("insufficient_reasons") or [])}
    flags = []
    if s_high is not None and s_high < STRICT_AGREEMENT_FLOOR:
        flags.append("low_agreement")
    if ts_low is not None and ts_low > TOO_STRICT_CEILING:
        flags.append("too_strict")
    if tl_low is not None and tl_low > TOO_LENIENT_CEILING:
        flags.append("too_lenient")
    return {"status": "below_floor" if flags else "calibrated_ok", **out, "flags": flags}


def judge_trust(judge: str | None, reports: list[dict[str, Any]], label_source: str, scale: str = "verdict") -> dict[str, Any]:
    """One judge's trust on ``scale``, from ``reports`` (``report`` / ``cached_report`` for the tiers of
    ``label_source``). Either tier sufficient = calibrated; both sufficient and disagreeing → the worse."""
    per_tier = []
    for tier in calib.tiers_for(label_source):
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
    return {
        **chosen, "judge": judge, "scale": scale, "label_source": label_source,
        "floor": STRICT_AGREEMENT_FLOOR, "too_strict_ceiling": TOO_STRICT_CEILING,
        "too_lenient_ceiling": TOO_LENIENT_CEILING, "min_labels": MIN_TOTAL_LABELS,
    }


def trust_for(project_id: int, judge: str | None, runtime_config: Any, scale: str = "verdict",
              live_ids: set[int] | None = None) -> dict[str, Any]:
    source = calib.label_source_of(runtime_config)
    reports = cached_report(project_id, scale, calib.tiers_for(source), live_ids)
    return judge_trust(judge, reports, source, scale)


def trust_summary(project_id: int, judge: str | None, runtime_config: Any, live_ids: set[int] | None = None) -> dict[str, Any]:
    """``{verdict: trust, rubric: trust | None}`` for the settings card: the rubric entry only when the
    judge has graded plans against a rubric."""
    out: dict[str, Any] = {"verdict": trust_for(project_id, judge, runtime_config, "verdict", live_ids), "rubric": None}
    if judge and any(judge in item["verdicts"] for item in population(project_id, "rubric", live_ids).values()):
        out["rubric"] = trust_for(project_id, judge, runtime_config, "rubric", live_ids)
    return out


# ── reference-model labelling ────────────────────────────────────────────


def _instruction_for(task: str | None) -> str | None:
    if task == "remediation":
        from app.services import remediation_scoring as rs

        return rs.JUDGE_INSTRUCTION
    if task == "code_example":
        from app.services import code_example_scoring as ces

        return ces.JUDGE_INSTRUCTION
    return None


async def run_reference_labelling(
    project_id: int,
    items: list[dict[str, Any]],
    backend: Any,
    scale: str,
    *,
    rubric: Any = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Label ``items`` blind with ``backend`` into the ``reference_model`` tier, using the eval judge's own
    prompts: ``answer_judge_service``'s judge (same system prompt, schema and per-task instruction) for
    ``verdict``; ``rubric_service.build_rubric_prompt`` + ``score_item`` with the project's rubric for
    ``rubric`` (a plan graded under a different rubric is skipped). Writes nothing but that tier. An
    unreadable answer is a skip marker; a failed call records nothing (a re-run retries it). Raises
    ``ValueError`` when nothing got a label; ``LlmBudgetExceeded`` propagates."""
    import asyncio

    from app.services import llm_usage
    from app.services.answer_judge_service import _backend_label, _judge_from_synth_backend
    from app.services.llm_budget_guard import LlmBudgetExceeded
    from app.services.synth_playbook_service import batch_usage_fields, safe_backend_label

    if not items:
        raise ValueError("No answers to label: every judged answer already has a reference label or was skipped.")
    labeller = safe_backend_label(await _backend_label(backend))
    judge_fn = _judge_from_synth_backend(backend, labeller)
    counts = {lab: 0 for lab in LABELS[scale]}
    stats = {"labelled": 0, "skips": 0, "unparseable": 0, "rubric_changed": 0, "calls": 0}
    call_errors: list[str] = []
    try:
        for i, item in enumerate(items):
            ck = item["content_key"]
            entry: dict[str, Any] | None = None
            if scale == "verdict":
                instruction = _instruction_for(item.get("task"))
                try:
                    stats["calls"] += 1
                    if instruction:
                        res = await judge_fn(item["question"], item["reference"], item["prediction"], instruction)
                    else:
                        res = await judge_fn(item["question"], item["reference"], item["prediction"])
                except LlmBudgetExceeded:
                    raise
                except Exception as exc:  # noqa: BLE001 - transient: no marker, a re-run retries it
                    call_errors.append(str(exc)[:200])
                    res = False
                if res is None:
                    stats["unparseable"] += 1
                    entry = skip_entry(reason="unparseable", labeller=labeller)
                elif res is not False:
                    verdict = res[1]
                    entry = label_entry(verdict, labeller=labeller, reason=res[2] or None)
            else:
                from app.services import rubric_service as rsvc

                if rubric is None or (item.get("rubric_sha") and item.get("rubric_sha") != rubric.sha256):
                    stats["rubric_changed"] += 1
                    entry = skip_entry(reason="rubric_changed", labeller=labeller)
                else:
                    prompt = rsvc.build_rubric_prompt(rubric, scenario=item["question"], reference=item["reference"],
                                                      prediction=item["prediction"], passages=budgeted_passages(item)[0])
                    try:
                        stats["calls"] += 1
                        raw = await backend.complete(
                            prompt, system_prompt=rsvc._SYSTEM_PROMPT, max_tokens=rsvc.JUDGE_MAX_TOKENS, temperature=0.0,
                            **llm_usage.cache_kwargs(backend), response_schema=rsvc.rubric_response_schema(rubric),
                        )
                    except LlmBudgetExceeded:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        call_errors.append(str(exc)[:200])
                        raw = False
                    if raw is not False:
                        scores = rsvc.parse_rubric_scores(raw if isinstance(raw, str) else None, rubric)
                        if scores is None:
                            stats["unparseable"] += 1
                            entry = skip_entry(reason="unparseable", labeller=labeller)
                        else:
                            passed = rsvc.score_item(rubric, scores)["passed"]
                            entry = label_entry("pass" if passed else "fail", labeller=labeller)
            if entry is not None:
                await asyncio.to_thread(save_labels, project_id, scale, "reference_model", {ck: entry})
                if entry.get("skipped"):
                    stats["skips"] += 1
                else:
                    stats["labelled"] += 1
                    counts[entry["label"]] += 1
            if progress is not None:
                await progress((i + 1) / len(items), f"Reference-labelled {stats['labelled']} of {len(items)} answers")
    except BaseException as exc:
        llm_usage.attach_result_usage(exc, batch_usage_fields(backend))
        raise
    if stats["labelled"] == 0:
        raise ValueError(
            f"Reference labelling labelled 0 of {len(items)} answers ({stats['unparseable']} unreadable, "
            f"{stats['rubric_changed']} graded under another rubric"
            + (f"; call errors: {call_errors[0]}" if call_errors else "") + ")."
        )
    return {
        "scale": scale,
        "rows_sampled": len(items),
        "labelled": stats["labelled"],
        "labels": counts,
        "skipped_unparseable": stats["unparseable"],
        "skipped_rubric_changed": stats["rubric_changed"],
        "skips_recorded": stats["skips"],
        "labeller": labeller,
        "calls": stats["calls"],
        **({"call_errors": call_errors} if call_errors else {}),
    }

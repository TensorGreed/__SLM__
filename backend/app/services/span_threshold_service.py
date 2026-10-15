"""Per-type score thresholds for a span tagger, tuned on a held-out split.

Each predicted span has a score (``decode_scored_spans``: the mean over its
tokens of P(B-type) + P(I-type)). Raising a type's threshold drops its
weakest spans — precision for recall. ``tune_from_scored`` (pure) picks, per
type, the threshold from ``GRID`` that maximises that type's strict F1 on the
split; a type keeps 0.0 unless a threshold beats it by ``MIN_GAIN`` (no
change for noise).

**Recall floors.** For redaction a missed span is the leak, an extra one is
only noise, so a type can carry a floor: the minimum validation recall it
must keep (``runtime_config["span_recall_floors"]`` = ``{type: floor,
"_default": floor}``, or the tune request's own). A floored type is not
F1-tuned: among the thresholds whose recall still meets the floor it takes
the one with the best precision (ties → higher recall, then the HIGHER
threshold), and stays at 0 unless that gains ``MIN_GAIN`` precision. When
even threshold 0 misses the floor, thresholds cannot help (they only drop
spans) — the report says "not reachable at threshold 0" and
``suggest_recall_bias`` re-decodes the same scored split with each
``BIAS_GRID`` recall bias (subtracted from P(O) before the argmax) and
reports the smallest bias that meets every such floor and what it costs in
precision. The bias is global (every type gets more spans), so it is only
written when the caller asks (``apply_recall_bias``); the thresholds are then
re-tuned on the biased decoding.

``tune_run_thresholds`` scores a run's validation split ONCE into per-token
logits (``train_token_classification.score_tokens``) — every threshold and
bias is a cheap re-decode, no further model call — tunes, and writes
``thresholds.json`` next to the run's model (floors, thresholds,
recall_bias and the per-type before → after report): the file the redact
endpoint, the export, the model card and the evaluation read. Tune on the
validation split, never on the test examples or a benchmark: those are what
the thresholds are then measured on.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

GRID: tuple[float, ...] = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BIAS_GRID: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
MIN_GAIN = 0.005
FLOORS_KEY = "span_recall_floors"
DEFAULT_FLOOR_KEY = "_default"

Spans = list[list[dict[str, Any]]]
DecodeFn = Callable[[float], Spans]


# ── Pure scoring helpers ─────────────────────────────────────────────────


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _precision(tp: int, fp: int) -> float | None:
    return round(tp / (tp + fp), 4) if tp + fp else None


def _recall(tp: int, fn: int) -> float | None:
    return round(tp / (tp + fn), 4) if tp + fn else None


def _counts(gold: Spans, pred: Spans, ent_type: str, threshold: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        gs = {(e["start"], e["end"]) for e in g if e["type"] == ent_type}
        ps = {(e["start"], e["end"]) for e in p if e["type"] == ent_type and float(e.get("score") or 0.0) >= threshold}
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    return tp, fp, fn


def _micro_counts(gold: Spans, pred: Spans) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        gs = {(e["type"], e["start"], e["end"]) for e in g}
        ps = {(e["type"], e["start"], e["end"]) for e in p}
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    return tp, fp, fn


def _micro(gold: Spans, pred: Spans) -> float:
    return _f1(*_micro_counts(gold, pred))


def _micro_block(gold: Spans, pred: Spans) -> dict[str, Any]:
    tp, fp, fn = _micro_counts(gold, pred)
    return {"precision": _precision(tp, fp), "recall": _recall(tp, fn), "f1": round(_f1(tp, fp, fn), 4)}


# ── Floors ───────────────────────────────────────────────────────────────


def normalise_floors(raw: Any, known_types: list[str] | None = None) -> dict[str, float]:
    """``{type: floor, "_default": floor}`` with type names normalised like
    the definitions. Raises ``ValueError`` for a value outside 0–1 or (when
    ``known_types`` is given) a type the project does not use."""
    from app.services.span_type_definitions_service import _normalise_type

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Recall floors must be an object {type: min_recall}.")
    out: dict[str, float] = {}
    unknown: list[str] = []
    for key, value in raw.items():
        name = DEFAULT_FLOOR_KEY if str(key).strip() == DEFAULT_FLOOR_KEY else _normalise_type(key)
        if not name:
            continue
        if value is None:
            continue
        try:
            floor = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Recall floor for {name!r} must be a number between 0 and 1.") from None
        if not 0.0 <= floor <= 1.0:
            raise ValueError(f"Recall floor for {name!r} must be between 0 and 1 (got {floor}).")
        if known_types is not None and name != DEFAULT_FLOOR_KEY and name not in known_types:
            unknown.append(name)
            continue
        out[name] = round(floor, 4)
    if unknown:
        raise ValueError(
            "Unknown entity type(s): " + ", ".join(sorted(unknown))
            + (f". The project's types: {', '.join(known_types)}." if known_types else ". The project has no entity types yet.")
        )
    return out


def floor_for(floors: dict[str, float] | None, ent_type: str) -> float | None:
    """A type's own floor, else the ``_default`` floor, else None."""
    floors = floors or {}
    if ent_type in floors:
        return float(floors[ent_type])
    if DEFAULT_FLOOR_KEY in floors:
        return float(floors[DEFAULT_FLOOR_KEY])
    return None


def project_floors(project: Any) -> dict[str, float]:
    cfg = getattr(project, "runtime_config", None)
    try:
        return normalise_floors(cfg.get(FLOORS_KEY) if isinstance(cfg, dict) else None)
    except ValueError:
        return {}


async def load_floors(db: AsyncSession | None, project_id: int) -> dict[str, float]:
    if db is None:
        return {}
    from app.models.project import Project

    project = await db.get(Project, project_id)
    return project_floors(project) if project is not None else {}


def known_span_types(project: Any, project_id: int) -> list[str]:
    """Types a floor may name: those in the project's span rows, its
    labelling definitions and its compliance bundles."""
    from app.services.compliance_bundles_service import compliance_state, entity_types
    from app.services.span_type_definitions_service import project_definitions, project_entity_types

    found = set(project_entity_types(project_id))
    found |= set(project_definitions(project))
    found |= set(entity_types(compliance_state(getattr(project, "runtime_config", None))))
    return sorted(found)


# ── Tuning (pure) ────────────────────────────────────────────────────────


def tune_from_scored(
    gold: Spans,
    scored_pred: Spans,
    types: list[str],
    *,
    floors: dict[str, float] | None = None,
    before_pred: Spans | None = None,
    grid: tuple[float, ...] = GRID,
    min_gain: float = MIN_GAIN,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per-type thresholds + a report ``{type: {threshold, f1_before,
    f1_after, precision_before/after, recall_before/after, support,
    predicted, floor, floor_status}}``. Pure.

    A type without a floor gets the F1-maximising threshold. A type with a
    floor (``floor_for``) keeps its recall at or above it: the best-precision
    threshold that does (``floor_status="met"``), else 0
    (``"not_reachable"`` — even keeping every span misses it) or
    ``"no_support"`` (no reference spans of the type to measure).
    ``before_pred`` (default ``scored_pred``) is what "before" is measured
    on — the plain decoding when ``scored_pred`` was decoded with a bias."""
    before_pred = scored_pred if before_pred is None else before_pred
    thresholds: dict[str, float] = {}
    report: dict[str, Any] = {}
    for t in sorted(types):
        before_c = _counts(gold, before_pred, t, 0.0)
        base = _counts(gold, scored_pred, t, 0.0)
        floor = floor_for(floors, t)
        status: str | None = None
        if floor is None:
            base_f1 = _f1(*base)
            best_t, best_f1 = 0.0, base_f1
            for th in grid:
                f1 = _f1(*_counts(gold, scored_pred, t, th))
                if f1 > best_f1 + 1e-12:
                    best_t, best_f1 = th, f1
            if best_f1 - base_f1 < min_gain:
                best_t = 0.0
        else:
            support = base[0] + base[2]
            if support == 0:
                best_t, status = 0.0, "no_support"
            elif base[0] / support + 1e-12 < floor:
                best_t, status = 0.0, "not_reachable"
            else:
                status = "met"
                options = []
                for th in grid:
                    tp, fp, fn = _counts(gold, scored_pred, t, th)
                    if tp / support + 1e-12 >= floor:
                        options.append(((tp / (tp + fp)) if tp + fp else 0.0, tp / support, th))
                best = max(options)
                base_precision = base[0] / (base[0] + base[1]) if base[0] + base[1] else 0.0
                best_t = best[2] if best[0] - base_precision >= min_gain else 0.0
        after = _counts(gold, scored_pred, t, best_t)
        thresholds[t] = best_t
        report[t] = {
            "threshold": best_t,
            "f1_before": round(_f1(*before_c), 4), "f1_after": round(_f1(*after), 4),
            "precision_before": _precision(before_c[0], before_c[1]), "precision_after": _precision(after[0], after[1]),
            "recall_before": _recall(before_c[0], before_c[2]), "recall_after": _recall(after[0], after[2]),
            "support": before_c[0] + before_c[2], "predicted": before_c[0] + before_c[1],
            "floor": floor, "floor_status": status,
        }
    return thresholds, report


def apply_thresholds(scored_pred: Spans, thresholds: dict[str, float]) -> Spans:
    return [[e for e in p if float(e.get("score") or 0.0) >= float(thresholds.get(e["type"], 0.0))] for p in scored_pred]


def suggest_recall_bias(
    gold: Spans,
    decode: DecodeFn,
    targets: dict[str, float],
    *,
    base_pred: Spans | None = None,
    bias_grid: tuple[float, ...] = BIAS_GRID,
) -> dict[str, Any]:
    """The smallest recall bias (from ``bias_grid``) whose re-decoding of
    the split meets every ``targets`` floor ``{type: floor}`` at threshold
    0, and its precision cost. ``decode(bias)`` re-decodes the scored split
    (no model call). Per type: the smallest bias meeting that type's floor
    (None = not even at the largest bias). ``recall_bias`` None = no bias in
    the grid meets them all."""
    base_pred = decode(0.0) if base_pred is None else base_pred
    per_type: dict[str, Any] = {}
    for t, floor in targets.items():
        tp, fp, fn = _counts(gold, base_pred, t, 0.0)
        per_type[t] = {
            "floor": floor, "bias": None,
            "recall_before": _recall(tp, fn), "precision_before": _precision(tp, fp),
            "recall_at_bias": None, "precision_at_bias": None, "precision_cost": None,
        }
    chosen: float | None = None
    chosen_pred: Spans | None = None
    tried: list[dict[str, Any]] = []
    for bias in bias_grid:
        pred = decode(bias)
        all_met = True
        for t, floor in targets.items():
            tp, fp, fn = _counts(gold, pred, t, 0.0)
            support = tp + fn
            met = bool(support) and tp / support + 1e-12 >= floor
            all_met = all_met and met
            entry = per_type[t]
            if met and entry["bias"] is None:
                entry.update(bias=bias, recall_at_bias=_recall(tp, fn), precision_at_bias=_precision(tp, fp))
                if entry["precision_before"] is not None and entry["precision_at_bias"] is not None:
                    entry["precision_cost"] = round(entry["precision_before"] - entry["precision_at_bias"], 4)
        micro = _micro_block(gold, pred)
        tried.append({"recall_bias": bias, "all_floors_met": all_met, **micro})
        if all_met and chosen is None:
            chosen, chosen_pred = bias, pred
            break
    before = _micro_block(gold, base_pred)
    out: dict[str, Any] = {
        "recall_bias": chosen,
        "grid": list(bias_grid),
        "per_type": per_type,
        "tried": tried,
        "micro_before": before,
    }
    if chosen_pred is not None:
        after = _micro_block(gold, chosen_pred)
        out["micro_at_bias"] = after
        if before["precision"] is not None and after["precision"] is not None:
            out["micro_precision_cost"] = round(before["precision"] - after["precision"], 4)
    return out


def tune_scored_split(
    gold: Spans,
    decode: DecodeFn,
    types: list[str],
    *,
    floors: dict[str, float] | None = None,
    apply_recall_bias: bool = False,
) -> dict[str, Any]:
    """The whole tuning decision for one scored split. ``decode(bias)``
    returns the split's spans (with scores) decoded with that recall bias.
    Pure apart from calling ``decode``."""
    floors = floors or {}
    base = decode(0.0)
    thresholds, report = tune_from_scored(gold, base, types, floors=floors)
    unreachable = {t: r["floor"] for t, r in report.items() if r["floor_status"] == "not_reachable"}
    suggestion = suggest_recall_bias(gold, decode, unreachable, base_pred=base) if unreachable else None
    bias = 0.0
    final = base
    if apply_recall_bias and suggestion and suggestion["recall_bias"] is not None:
        bias = float(suggestion["recall_bias"])
        final = decode(bias)
        thresholds, report = tune_from_scored(gold, final, types, floors=floors, before_pred=base)
    after_pred = apply_thresholds(final, thresholds)
    floored = {t: r for t, r in report.items() if r["floor"] is not None}
    return {
        "thresholds": thresholds,
        "recall_bias": bias,
        "report": report,
        "recall_bias_suggestion": suggestion,
        "micro_before": _micro_block(gold, base),
        "micro_after": _micro_block(gold, after_pred),
        "micro_f1_before": round(_micro(gold, base), 4),
        "micro_f1_after": round(_micro(gold, after_pred), 4),
        "floors_met": sorted(t for t, r in floored.items() if r["floor_status"] == "met"),
        "floors_not_reachable": sorted(t for t, r in floored.items() if r["floor_status"] == "not_reachable"),
        "floors_no_support": sorted(t for t, r in floored.items() if r["floor_status"] == "no_support"),
    }


# ── Model I/O ────────────────────────────────────────────────────────────


def _scripts_on_path() -> None:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _load_model(model_path: str) -> tuple[Any, Any, str]:
    """(model, tokenizer, device) — tests patch this."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, dtype=torch.float32).to(device).eval()
    return model, tokenizer, device


def _score_tokens(model_path: str, texts: list[str], max_length: int = 512) -> list[Any]:
    """One model pass over ``texts`` → per-text (offsets, token logits)."""
    _scripts_on_path()
    from train_token_classification import score_tokens

    model, tokenizer, device = _load_model(model_path)
    return score_tokens(texts, model, tokenizer, max_length=max_length, device=device)


def _decode(texts: list[str], token_scores: list[Any], id2label: dict[int, str], bias: float, ocr_tolerant: bool) -> Spans:
    _scripts_on_path()
    from train_token_classification import decode_token_scores

    return decode_token_scores(
        texts, token_scores, id2label, with_scores=True, recall_bias=float(bias), ocr_tolerant=ocr_tolerant,
    )


def _score_split(model_path: str, id2label: dict[int, str], texts: list[str], ocr_tolerant: bool) -> Spans:
    """Spans with scores (plain decoding) for ``texts``."""
    return _decode(texts, _score_tokens(model_path, texts), id2label, 0.0, ocr_tolerant)


async def tune_run_thresholds(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    *,
    split: str = "val",
    floors: dict[str, float] | None = None,
    apply_recall_bias: bool = False,
) -> dict[str, Any]:
    """Tune a token-classification run's per-type thresholds on ``split``
    (default the validation split) and write them to its thresholds.json.
    ``floors`` None = the project's ``span_recall_floors``."""
    from app.models.experiment import Experiment
    from app.services.evaluation_service import _resolve_token_classifier_artifacts
    from app.services.span_model_package_service import THRESHOLDS_FILE, default_thresholds, read_thresholds
    from app.services.span_type_definitions_service import load_scan_tolerant

    if split not in ("val", "train"):
        raise ValueError("Tune on the validation split (val) — the test examples and benchmarks are what thresholds are measured on.")
    exp = await db.get(Experiment, experiment_id)
    if exp is None or exp.project_id != project_id:
        raise ValueError(f"Run #{experiment_id} not found in project {project_id}.")
    artifacts = _resolve_token_classifier_artifacts(exp.output_dir) if exp.output_dir else None
    if artifacts is None:
        raise ValueError(f"Run #{experiment_id} has no token-classification model on disk.")
    _scripts_on_path()
    from train_token_classification import row_to_example

    path = settings.DATA_DIR / "projects" / str(project_id) / "prepared" / f"{split}.jsonl"
    examples = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ex = row_to_example(json.loads(line))
                if ex:
                    examples.append(ex)
    if not examples:
        raise ValueError(f"Project {project_id} has no {split} split with span rows to tune on.")
    floors = normalise_floors(floors) if floors is not None else await load_floors(db, project_id)
    texts = [t for t, _ in examples]
    gold = [[{"type": e["type"], "start": e["start"], "end": e["end"]} for e in spans] for _, spans in examples]
    ocr = await load_scan_tolerant(db, project_id)
    id2label = artifacts["id2label"]
    max_length = min(512, int(artifacts.get("max_length") or 512))
    token_scores = await asyncio.to_thread(_score_tokens, artifacts["model_path"], texts, max_length)
    cache: dict[float, Spans] = {}

    def decode(bias: float) -> Spans:
        if bias not in cache:
            cache[bias] = _decode(texts, token_scores, id2label, bias, ocr)
        return cache[bias]

    types = sorted({str(v)[2:] for v in id2label.values() if str(v)[:2] in ("B-", "I-")})
    result = await asyncio.to_thread(
        tune_scored_split, gold, decode, types, floors=floors, apply_recall_bias=apply_recall_bias,
    )
    thresholds = result["thresholds"]
    floors_unknown = sorted(k for k in floors if k != DEFAULT_FLOOR_KEY and k not in types)
    model_dir = Path(artifacts["model_path"])
    data = read_thresholds(model_dir) or default_thresholds(types)
    data["per_type"] = {**(data.get("per_type") or {}), **thresholds}
    # Tuning owns the decoding: the thresholds were chosen under this bias.
    data["recall_bias"] = result["recall_bias"]
    data["recall_floors"] = floors
    data["tuned"] = {
        "split": split, "rows": len(examples), "at": datetime.now(timezone.utc).isoformat(),
        "micro_f1_before": result["micro_f1_before"], "micro_f1_after": result["micro_f1_after"],
        "micro_before": result["micro_before"], "micro_after": result["micro_after"],
        "recall_bias": result["recall_bias"], "recall_bias_suggestion": result["recall_bias_suggestion"],
        "floors": floors, "report": result["report"],
    }
    (model_dir / THRESHOLDS_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
    raised = {t: v for t, v in thresholds.items() if v > 0}
    return {
        "experiment_id": experiment_id, "split": split, "rows": len(examples),
        "micro_f1_before": result["micro_f1_before"], "micro_f1_after": result["micro_f1_after"],
        "micro_before": result["micro_before"], "micro_after": result["micro_after"],
        "raised": raised, "report": result["report"], "file": str(model_dir / THRESHOLDS_FILE),
        "floors": floors, "floors_unknown": floors_unknown,
        "floors_met": result["floors_met"], "floors_not_reachable": result["floors_not_reachable"],
        "floors_no_support": result["floors_no_support"],
        "recall_bias": result["recall_bias"], "recall_bias_suggestion": result["recall_bias_suggestion"],
    }


def thresholds_for_decoding(model_path: str | Path) -> dict[str, float]:
    """The non-zero per-type thresholds of a model's thresholds.json ({}
    when none — the plain argmax decoder then runs unchanged)."""
    from app.services.span_model_package_service import read_thresholds

    data = read_thresholds(Path(model_path)) or {}
    per = data.get("per_type") if isinstance(data.get("per_type"), dict) else {}
    return {str(k): float(v) for k, v in per.items() if isinstance(v, (int, float)) and v > 0}


def recall_bias_for_decoding(model_path: str | Path) -> float:
    """A model's thresholds.json ``recall_bias`` (0.0 when none)."""
    from app.services.span_model_package_service import read_thresholds

    value = (read_thresholds(Path(model_path)) or {}).get("recall_bias")
    return float(value) if isinstance(value, (int, float)) and 0 < value <= 1 else 0.0

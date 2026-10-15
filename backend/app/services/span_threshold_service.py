"""Per-type score thresholds for a span tagger, tuned on a held-out split.

Each predicted span has a score (``decode_scored_spans``: the mean over its
tokens of P(B-type) + P(I-type)). Raising a type's threshold drops its
weakest spans — precision for recall. ``tune_from_scored`` (pure) picks, per
type, the threshold from ``GRID`` that maximises that type's strict F1 on the
split; a type keeps 0.0 unless a threshold beats it by ``MIN_GAIN`` (no
change for noise). ``tune_run_thresholds`` scores a run's validation split
once (spans with scores), tunes, and writes ``thresholds.json`` next to the
run's model — the file the redact endpoint, the export and (now) the
evaluation read. Tune on the validation split, never on the test examples or
a benchmark: those are what the thresholds are then measured on.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

GRID: tuple[float, ...] = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_GAIN = 0.005


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _counts(gold: list[list[dict]], pred: list[list[dict]], ent_type: str, threshold: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        gs = {(e["start"], e["end"]) for e in g if e["type"] == ent_type}
        ps = {(e["start"], e["end"]) for e in p if e["type"] == ent_type and float(e.get("score") or 0.0) >= threshold}
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    return tp, fp, fn


def tune_from_scored(
    gold: list[list[dict]],
    scored_pred: list[list[dict]],
    types: list[str],
    *,
    grid: tuple[float, ...] = GRID,
    min_gain: float = MIN_GAIN,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per-type thresholds + a report ``{type: {threshold, f1_before,
    f1_after, support, predicted}}``. Pure."""
    thresholds: dict[str, float] = {}
    report: dict[str, Any] = {}
    for t in sorted(types):
        base = _counts(gold, scored_pred, t, 0.0)
        before = _f1(*base)
        best_t, best_f1 = 0.0, before
        for th in grid:
            f1 = _f1(*_counts(gold, scored_pred, t, th))
            if f1 > best_f1 + 1e-12:
                best_t, best_f1 = th, f1
        if best_f1 - before < min_gain:
            best_t, best_f1 = 0.0, before
        thresholds[t] = best_t
        report[t] = {
            "threshold": best_t, "f1_before": round(before, 4), "f1_after": round(best_f1, 4),
            "support": base[0] + base[2], "predicted": base[0] + base[1],
        }
    return thresholds, report


def apply_thresholds(scored_pred: list[list[dict]], thresholds: dict[str, float]) -> list[list[dict]]:
    return [[e for e in p if float(e.get("score") or 0.0) >= float(thresholds.get(e["type"], 0.0))] for p in scored_pred]


def _micro(gold: list[list[dict]], pred: list[list[dict]]) -> float:
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        gs = {(e["type"], e["start"], e["end"]) for e in g}
        ps = {(e["type"], e["start"], e["end"]) for e in p}
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    return _f1(tp, fp, fn)


def _score_split(model_path: str, id2label: dict[int, str], texts: list[str], ocr_tolerant: bool) -> list[list[dict]]:
    import sys

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_token_classification import predict_spans

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, dtype=torch.float32).to(device).eval()
    return predict_spans(texts, model, tokenizer, id2label, device=device, with_scores=True, ocr_tolerant=ocr_tolerant)


async def tune_run_thresholds(db: AsyncSession, project_id: int, experiment_id: int, *, split: str = "val") -> dict[str, Any]:
    """Tune a token-classification run's per-type thresholds on ``split``
    (default the validation split) and write them to its thresholds.json."""
    import sys

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
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
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
    texts = [t for t, _ in examples]
    gold = [[{"type": e["type"], "start": e["start"], "end": e["end"]} for e in spans] for _, spans in examples]
    scored = await asyncio.to_thread(
        _score_split, artifacts["model_path"], artifacts["id2label"], texts, await load_scan_tolerant(db, project_id),
    )
    types = sorted({str(v)[2:] for v in artifacts["id2label"].values() if str(v)[:2] in ("B-", "I-")})
    thresholds, report = tune_from_scored(gold, scored, types)
    before = _micro(gold, scored)
    after = _micro(gold, apply_thresholds(scored, thresholds))
    model_dir = Path(artifacts["model_path"])
    data = read_thresholds(model_dir) or default_thresholds(types)
    data["per_type"] = {**(data.get("per_type") or {}), **thresholds}
    data["tuned"] = {
        "split": split, "rows": len(examples), "at": datetime.now(timezone.utc).isoformat(),
        "micro_f1_before": round(before, 4), "micro_f1_after": round(after, 4), "report": report,
    }
    (model_dir / THRESHOLDS_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
    raised = {t: v for t, v in thresholds.items() if v > 0}
    return {
        "experiment_id": experiment_id, "split": split, "rows": len(examples),
        "micro_f1_before": round(before, 4), "micro_f1_after": round(after, 4),
        "raised": raised, "report": report, "file": str(model_dir / THRESHOLDS_FILE),
    }


def thresholds_for_decoding(model_path: str | Path) -> dict[str, float]:
    """The non-zero per-type thresholds of a model's thresholds.json ({}
    when none — the plain argmax decoder then runs unchanged)."""
    from app.services.span_model_package_service import read_thresholds

    data = read_thresholds(Path(model_path)) or {}
    per = data.get("per_type") if isinstance(data.get("per_type"), dict) else {}
    return {str(k): float(v) for k, v in per.items() if isinstance(v, (int, float)) and v > 0}

"""The lift check for a run trained with retrieved passages.

A passage-prompt run (``passage_prompts_service``) is meant to be served
WITH the project's retrieval, so the ordinary lift check (fine-tune vs base,
both without passages) measures the wrong thing. Its bar is the base model
answering from the same passages — what the project could serve without
training at all. This module scores both on the test examples, with the
served retrieval, judged by the same answer judge, row by row:

1. ``base + passages`` — the base-only documents comparison
   (``auto_rag/comparison_base_documents.json``), reused when it was judged
   on the test split, for this run's base model, with the retrieval the
   project serves now; otherwise re-run (no sweep). When the base model
   ships decoding settings neutral decoding turns off (Qwen2.5-Instruct:
   repetition penalty 1.1), it is also scored under those
   (``comparison_base_documents_model_decoding.json``) and the higher
   judge score is the bar — the strongest-baseline rule
   (``decoding_settings``); ``base_decoding`` in the record says which.
2. ``fine-tuned + passages`` — the auto-RAG harness on this run
   (``run_project_comparison(experiment_id=…, corpus="documents",
   split="test")``, the served retrieval, the same judge).

The record (``auto_rag/passage_reader/<run>.json``) pairs the two arms'
per-row judge scores (``paired_comparison_stats``) and keeps the retrieval
judge's split (passage missed vs found-but-misread) for both. Eval summary
(``kind="passage_reader"``), the passages-vs-fine-tune verdict and the bell
read it.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import settings

READER_DIR = "passage_reader"
MODEL_DECODING_BASE_FILE = "comparison_base_documents_model_decoding.json"
SUMMARY_FAILURES = 5

ComparisonFn = Callable[..., dict[str, Any]]


def is_passage_trained(exp: Any) -> bool:
    from app.services.passage_prompts_service import is_enabled

    return is_enabled(getattr(exp, "config", None))


def record_path(project_id: int, experiment_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "auto_rag" / READER_DIR / f"{int(experiment_id)}.json"


def read_record(project_id: int, experiment_id: int) -> dict[str, Any] | None:
    path = record_path(project_id, experiment_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload, dict) else None
    return summary if isinstance(summary, dict) else {}


def _judge_label(payload: dict[str, Any] | None) -> str | None:
    judge = _summary(payload).get("judge")
    return str(judge.get("judge")) if isinstance(judge, dict) and judge.get("judge") else None


def base_comparison_usable(
    payload: dict[str, Any] | None,
    *,
    base_model: str,
    retrieval: dict[str, Any],
    test_rows: int | None = None,
    decoding: str = "neutral",
) -> bool:
    """Pure: can the cached base + passages comparison be the bar? Judged,
    on the test split, for ``base_model``, with ``retrieval``, on as many
    rows as the test split has (when known), decoded as ``decoding`` (a
    comparison from before decoding was recorded is re-run)."""
    from app.services.auto_rag_service import same_retrieval
    from app.services.decoding_settings import decoding_record, generation_kwargs, is_model_decoding

    if not isinstance(payload, dict) or payload.get("model") != "base":
        return False
    if str(payload.get("corpus") or "") != "documents" or str(payload.get("split") or "") != "test":
        return False
    if str(payload.get("base_model") or "") != str(base_model or ""):
        return False
    if not _judge_label(payload):
        return False
    if not same_retrieval(_summary(payload).get("retrieval"), retrieval):
        return False
    recorded = _summary(payload).get("decoding")
    if decoding == "model":
        if not is_model_decoding(recorded):
            return False
    elif recorded != decoding_record(generation_kwargs(max_new_tokens=1, temperature=0.0)):
        return False
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return test_rows is None or len(rows) == test_rows


def _read_model_decoding_comparison(project_id: int) -> dict[str, Any] | None:
    path = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag" / MODEL_DECODING_BASE_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def choose_base_bar(neutral_payload: dict[str, Any], model_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Pure: the strongest-baseline rule on base + passages — the higher
    judge score of the two decodings; a tie or a missing score keeps
    neutral."""
    n = _arm(neutral_payload, "with_rag").get("score")
    m = _arm(model_payload, "with_rag").get("score") if model_payload else None
    scored = isinstance(n, (int, float)) and isinstance(m, (int, float))
    return {
        "chosen": "model" if scored and float(m) > float(n) else "neutral",
        "metric_id": "judge_correct",
        "neutral": round(float(n), 4) if isinstance(n, (int, float)) else None,
        "model": round(float(m), 4) if isinstance(m, (int, float)) else None,
    }


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("question") or "").strip(), str(row.get("reference") or "").strip())


def _judge_score(arm: Any) -> float | None:
    judge = arm.get("judge") if isinstance(arm, dict) else None
    value = judge.get("score") if isinstance(judge, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _arm(payload: dict[str, Any], side: str) -> dict[str, Any]:
    judge = _summary(payload).get("judge")
    block = judge.get(side) if isinstance(judge, dict) and isinstance(judge.get(side), dict) else {}
    return {
        "score": block.get("score"),
        "counts": block.get("counts") or {},
        "judged": block.get("judged"),
        "retrieval": block.get("retrieval") if isinstance(block.get("retrieval"), dict) else None,
    }


def build_reader_record(
    base_payload: dict[str, Any],
    finetune_payload: dict[str, Any],
    *,
    experiment_id: int,
) -> dict[str, Any]:
    """Pure: pair the base + passages arm with the fine-tuned + passages arm
    row by row (question + reference) and say how solid the difference is."""
    from app.services.auto_rag_service import normalize_retrieval, retrieval_label, same_retrieval
    from app.services.paired_comparison_stats import paired_difference_evidence

    base_rows = {_row_key(r): r for r in base_payload.get("rows") or [] if isinstance(r, dict)}
    paired_rows: list[dict[str, Any]] = []
    before: list[float] = []
    after: list[float] = []
    for row in finetune_payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        base_row = base_rows.get(_row_key(row))
        if base_row is None:
            continue
        base_on = base_row.get("with_rag") if isinstance(base_row.get("with_rag"), dict) else {}
        ft_on = row.get("with_rag") if isinstance(row.get("with_rag"), dict) else {}
        b, a = _judge_score(base_on), _judge_score(ft_on)
        if b is not None and a is not None:
            before.append(b)
            after.append(a)
        paired_rows.append({
            "question": row.get("question"),
            "reference": row.get("reference"),
            "base": {
                "generated": base_on.get("generated"),
                "judge": base_on.get("judge"),
                "retrieval_judge": base_on.get("retrieval_judge"),
            },
            "finetune": {
                "generated": ft_on.get("generated"),
                "judge": ft_on.get("judge"),
                "retrieval_judge": ft_on.get("retrieval_judge"),
                "retrieved_sources": ft_on.get("retrieved_sources") or [],
            },
            "finetune_without_passages": {
                "generated": (row.get("without_rag") or {}).get("generated") if isinstance(row.get("without_rag"), dict) else None,
                "judge": (row.get("without_rag") or {}).get("judge") if isinstance(row.get("without_rag"), dict) else None,
            },
        })
    evidence = paired_difference_evidence(before, after)
    evidence["metric_id"] = "judge_correct"
    base_retrieval = normalize_retrieval(_summary(base_payload).get("retrieval"))
    ft_retrieval = normalize_retrieval(_summary(finetune_payload).get("retrieval"))
    base_judge, ft_judge = _judge_label(base_payload), _judge_label(finetune_payload)
    n_test = len(finetune_payload.get("rows") or [])
    comparable = bool(
        base_judge and base_judge == ft_judge
        and same_retrieval(base_retrieval, ft_retrieval)
        and len(paired_rows) == n_test
        and str(base_payload.get("split")) == str(finetune_payload.get("split")) == "test"
    )
    return {
        "experiment_id": int(experiment_id),
        "base_model": finetune_payload.get("base_model") or base_payload.get("base_model"),
        "split": finetune_payload.get("split"),
        "judge": ft_judge,
        "retrieval": ft_retrieval,
        "retrieval_label": retrieval_label(ft_retrieval),
        "n_rows": len(paired_rows),
        "comparable": comparable,
        "arms": {
            "base_with_passages": _arm(base_payload, "with_rag"),
            "base_without_passages": _arm(base_payload, "without_rag"),
            "finetune_with_passages": _arm(finetune_payload, "with_rag"),
            "finetune_without_passages": _arm(finetune_payload, "without_rag"),
        },
        # base + passages → fine-tuned + passages, per row (judge score).
        "evidence": evidence,
        "base_cached_at": base_payload.get("cached_at"),
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "rows": paired_rows,
    }


async def _run_threaded(fn: ComparisonFn, report: Callable[[float, str], Awaitable[None]], lo: float, hi: float, label: str, **kwargs) -> dict[str, Any]:
    """Run one harness comparison on a worker thread, publishing its row
    progress between ``lo`` and ``hi``."""
    state: dict[str, Any] = {"scored": 0, "total": 0, "label": ""}

    def _cb(scored: int, total: int, what: str) -> None:
        state.update(scored=scored, total=total, label=what)

    async def _drain(stop: asyncio.Event) -> None:
        while not stop.is_set():
            if state["total"]:
                frac = lo + (hi - lo) * min(1.0, state["scored"] / max(1, 2 * state["total"]))
                await report(frac, f"{label}: row {state['scored']}/{state['total']} ({state['label']})")
            elif state["label"]:
                await report(lo, f"{label}: {state['label']}")
            try:
                await asyncio.wait_for(stop.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass

    stop = asyncio.Event()
    drainer = asyncio.get_running_loop().create_task(_drain(stop))
    try:
        return await asyncio.to_thread(fn, progress_callback=_cb, **kwargs)
    finally:
        stop.set()
        try:
            await drainer
        except Exception:  # noqa: BLE001
            pass


def _default_comparison(project_id: int, **kwargs) -> dict[str, Any]:
    import sys
    from pathlib import Path as _Path

    backend_root = str(_Path(__file__).resolve().parents[2])
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    from scripts.auto_rag_ab import run_project_comparison

    return run_project_comparison(project_id, **kwargs)


def _test_row_count(project_id: int) -> int | None:
    path = settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "test.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


async def run_passage_reader_evaluation(
    db: Any,
    *,
    project: Any,
    experiment: Any,
    report: Callable[[float, str], Awaitable[None]],
    comparison_fn: ComparisonFn | None = None,
) -> dict[str, Any]:
    """Base + passages vs fine-tuned + passages on the test examples (see the
    module doc). Writes the record and returns the Job result. Raises when
    either comparison wasn't judged — an unjudged comparison can't be read
    on long answers (token F1 isn't the verdict)."""
    from app.services.auto_rag_service import retrieval_label, retrieval_settings
    from app.services.passage_rag_verdict_service import read_documents_comparison

    fn = comparison_fn or (lambda **kw: _default_comparison(project.id, **kw))
    retrieval = retrieval_settings(project)
    label = retrieval_label(retrieval)
    base_payload = read_documents_comparison(project.id)
    base_reused = base_comparison_usable(
        base_payload, base_model=experiment.base_model, retrieval=retrieval, test_rows=_test_row_count(project.id)
    )

    def _require_project_base() -> None:
        # A base-only comparison scores the PROJECT's base model.
        if str(getattr(project, "base_model_name", "") or "") != str(experiment.base_model or ""):
            raise ValueError(
                f"Run #{experiment.id} was trained on {experiment.base_model} but the project's base model is now "
                f"{project.base_model_name}; the base + passages bar is measured on the project's base model. "
                "Set the project's base model back to evaluate this run."
            )

    if not base_reused:
        _require_project_base()
        await report(0.05, f"Scoring the base model + your passages ({label}) on the test examples")
        base_payload = await _run_threaded(
            fn, report, 0.05, 0.28, "Base + passages",
            base_only=True, corpus="documents", split="test",
        )
    from app.services.decoding_settings import model_decoding_differs

    own = await asyncio.to_thread(model_decoding_differs, experiment.base_model)
    model_payload = None
    if own is not None:
        model_payload = _read_model_decoding_comparison(project.id)
        # Judged by another judge than the neutral bar: the two can't be compared.
        if not base_comparison_usable(
            model_payload, base_model=experiment.base_model, retrieval=retrieval,
            test_rows=_test_row_count(project.id), decoding="model",
        ) or _judge_label(model_payload) != _judge_label(base_payload):
            _require_project_base()
            base_reused = False
            await report(0.3, f"Scoring the base model + your passages with its own decoding settings ({label})")
            model_payload = await _run_threaded(
                fn, report, 0.3, 0.48, "Base + passages (its own decoding)",
                base_only=True, corpus="documents", split="test", decoding="model",
            )
        if _judge_label(model_payload) != _judge_label(base_payload):
            # The fresh own-decoding run is judged by today's judge; a reused
            # neutral bar judged by an earlier one is re-run so the two compare.
            _require_project_base()
            base_reused = False
            await report(0.48, f"Re-scoring the base model + your passages with the current judge ({label})")
            base_payload = await _run_threaded(
                fn, report, 0.48, 0.5, "Base + passages", base_only=True, corpus="documents", split="test",
            )
    base_decoding = choose_base_bar(base_payload, model_payload)
    base_decoding["model_settings"] = own
    if own is None:
        base_decoding["reason"] = "no_other_settings"
    if base_decoding["chosen"] == "model":
        base_payload = model_payload
    await report(0.5, f"Scoring run #{experiment.id} + your passages ({label}) on the test examples")
    finetune_payload = await _run_threaded(
        fn, report, 0.5, 0.92, f"Run #{experiment.id} + passages",
        base_only=False, corpus="documents", split="test", experiment_id=experiment.id,
    )
    if not _judge_label(base_payload) or not _judge_label(finetune_payload) or (
        model_payload is not None and not _judge_label(model_payload)
    ):
        raise RuntimeError(
            "The passages comparison ran without an answer judge, so it can't be read on long answers. "
            "Configure a judge model (EVAL_JUDGE_BACKEND or a local Ollama model) and re-run."
        )
    record = build_reader_record(base_payload, finetune_payload, experiment_id=experiment.id)
    record["base_reused"] = base_reused
    record["base_decoding"] = base_decoding
    path = record_path(project.id, experiment.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    await report(0.98, "Pairing the rows")
    evidence = record["evidence"]
    arms = record["arms"]
    return {
        "kind": "passage_reader_eval",
        "experiment_id": experiment.id,
        "base_model": experiment.base_model,
        "retrieval": record["retrieval"],
        "judge": record["judge"],
        "n_rows": record["n_rows"],
        "comparable": record["comparable"],
        "base_reused": base_reused,
        "base_decoding": base_decoding,
        "base_with_passages": {k: arms["base_with_passages"].get(k) for k in ("score", "counts")},
        "finetune_with_passages": {k: arms["finetune_with_passages"].get(k) for k in ("score", "counts")},
        "evidence": {k: evidence.get(k) for k in ("verdict", "n", "better", "worse", "same", "mean_diff", "ci_low", "ci_high")},
    }


def reader_summary(project_id: int, experiment: Any) -> dict[str, Any]:
    """The Eval card for a passage-trained run (``kind="passage_reader"``):
    headline judge score base + passages → fine-tuned + passages, paired
    evidence, failures = the fine-tune's non-correct rows (with the
    retrieval judge's verdict), and the retrieval both arms used."""
    record = read_record(project_id, experiment.id)
    if record is None:
        return {
            "project_id": project_id,
            "experiment_id": experiment.id,
            "kind": "passage_reader",
            "verdict": "not_evaluated",
            "message": (
                "This run was trained to answer from your passages, so it is scored WITH them against the base "
                "model + the same passages. That check hasn't run yet (watch the bell), or evaluate it now."
            ),
            "headline": None,
            "failures": [],
        }
    arms = record.get("arms") or {}
    base_arm = arms.get("base_with_passages") or {}
    ft_arm = arms.get("finetune_with_passages") or {}
    baseline = base_arm.get("score")
    trained = ft_arm.get("score")
    headline = None
    verdict = "no_comparison"
    if isinstance(baseline, (int, float)) and isinstance(trained, (int, float)):
        delta = float(trained) - float(baseline)
        direction = "improved" if delta > 0.0001 else "regressed" if delta < -0.0001 else "unchanged"
        verdict = {"improved": "better", "regressed": "worse"}.get(direction, "same")
        headline = {
            "metric_id": "judge_correct",
            "baseline_value": round(float(baseline), 4),
            "trained_value": round(float(trained), 4),
            "absolute_delta": round(delta, 4),
            "relative_delta_pct": round(delta / float(baseline) * 100.0, 1) if baseline else None,
            "direction": direction,
            "is_headline": True,
        }
    failures: list[dict[str, Any]] = []
    failed = 0
    for row in record.get("rows") or []:
        ft = row.get("finetune") if isinstance(row.get("finetune"), dict) else {}
        judge = ft.get("judge") if isinstance(ft.get("judge"), dict) else None
        if not judge or judge.get("verdict") == "correct":
            continue
        failed += 1
        if len(failures) < SUMMARY_FAILURES:
            rj = ft.get("retrieval_judge") if isinstance(ft.get("retrieval_judge"), dict) else {}
            failures.append({
                "prompt": str(row.get("question") or "")[:300],
                "reference": str(row.get("reference") or "")[:300],
                "prediction": str(ft.get("generated") or "")[:300],
                "row_judge_verdict": judge.get("verdict"),
                "row_judge_reason": judge.get("reason"),
                "row_retrieval_verdict": rj.get("verdict"),
                "row_retrieval_passage": rj.get("passage"),
                "row_retrieval_reason": rj.get("reason"),
            })
    counts = ft_arm.get("counts") or {}
    base_counts = base_arm.get("counts") or {}
    return {
        "project_id": project_id,
        "experiment_id": experiment.id,
        "kind": "passage_reader",
        "verdict": verdict,
        "message": None if record.get("comparable") else (
            "The two arms differ in judge, retrieval or rows — read the numbers side by side, not as a lift."
        ),
        "headline": headline,
        "evidence": record.get("evidence"),
        "seed_evidence": None,
        "seeds": None,
        "n_seeds": 1,
        "metric_lifts": [],
        "baseline": {
            "experiment_id": None,
            "base_model": record.get("base_model"),
            "arm": "base_with_passages",
            "decoding_bar": record.get("base_decoding"),
        },
        "trained": {"experiment_id": experiment.id, "experiment_name": getattr(experiment, "name", None), "arm": "finetune_with_passages"},
        "eval_result_id": None,
        "eval_type": "passage_reader",
        "evaluated_samples": record.get("n_rows"),
        "split": record.get("split"),
        "failures": failures,
        "failed_count": failed,
        "judge": {
            "judge": record.get("judge"),
            "score": trained,
            "judged": int(ft_arm.get("judged") or record.get("n_rows") or 0),
            "unjudged": 0,
            "correct": counts.get("correct", 0),
            "partial": counts.get("partial", 0),
            "wrong": counts.get("wrong", 0),
            "retrieval": ft_arm.get("retrieval"),
        },
        # Both arms, so the card can show base + passages next to the run.
        "passage_reader": {
            "comparable": record.get("comparable"),
            "base_reused": record.get("base_reused"),
            "base_with_passages": {"score": baseline, "counts": base_counts, "retrieval": base_arm.get("retrieval")},
            "finetune_with_passages": {"score": trained, "counts": counts, "retrieval": ft_arm.get("retrieval")},
            "finetune_without_passages": {
                "score": (arms.get("finetune_without_passages") or {}).get("score"),
                "counts": (arms.get("finetune_without_passages") or {}).get("counts") or {},
            },
        },
        "dataset_name": record.get("split"),
        "cached_at": record.get("cached_at"),
        "retrieval": record.get("retrieval"),
    }

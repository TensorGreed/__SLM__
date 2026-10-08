"""Did retrieval over the documents beat fine-tuning? (documents projects)

A documents-only project that went through the documents → Q&A flow has
two measured candidates for "the assistant": the fine-tuned run (judged
by the lift check on the test examples) and the untouched base model
answering from the retrieved document passages (the base-only
``--corpus documents`` Auto-RAG comparison, judged on the val rows). Both
carry the answer judge's correct / partial / wrong score, so they can be
compared on *facts* rather than token overlap.

``passages_vs_finetune`` is the one verdict both the Coach (reroute-to-RAG
nudge) and the playground (default retrieval corpus) read. It only claims
a win when the two numbers are comparable (same judge) and the retrieval
gain over the bare base model is beyond row noise. It never looks at F1.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.models.project import Project

DOCUMENTS_COMPARISON_FILE = "comparison_base_documents.json"
# runtime_config key: an explicit playground corpus ("qa" / "documents" /
# "auto") set by the user or stamped on a reroute sibling.
AUTO_RAG_CORPUS_KEY = "auto_rag_corpus"


def read_documents_comparison(project_id: int) -> dict[str, Any] | None:
    """The cached base-model + document-passages comparison, or None."""
    path = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag" / DOCUMENTS_COMPARISON_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def summarize_documents_comparison(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Judge numbers + retrieval row evidence from the cached comparison;
    None when it was not judged."""
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    judge = summary.get("judge") if isinstance(summary.get("judge"), dict) else None
    if not judge:
        return None
    on = judge.get("with_rag") if isinstance(judge.get("with_rag"), dict) else {}
    off = judge.get("without_rag") if isinstance(judge.get("without_rag"), dict) else {}
    if not isinstance(on.get("score"), (int, float)):
        return None
    from app.api.auto_rag import _lift_evidence

    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    evidence = _lift_evidence(rows, score_key="judge")
    return {
        "judge": str(judge.get("judge") or ""),
        "base_model": payload.get("base_model"),
        "with_passages": float(on["score"]),
        "with_passages_counts": on.get("counts") or {},
        "without_retrieval": float(off["score"]) if isinstance(off.get("score"), (int, float)) else None,
        "n_rows": int(summary.get("n_val_rows") or len(rows)),
        # Which rows: "test" pairs with the lift check; "val" (older caches)
        # does not, and the verdict says so.
        "split": str(payload.get("split") or "val"),
        "retrieval_evidence": {k: evidence.get(k) for k in ("verdict", "n", "better", "worse", "same")},
        "cached_at": payload.get("cached_at"),
    }


async def _judged_test_eval(db: AsyncSession, experiment_id: int) -> dict[str, Any] | None:
    """A run's latest judged eval on the lift-check split (``judge_correct`` +
    judge label), or None."""
    from app.services.post_training_eval_service import AUTO_LIFT_DATASET

    result = await db.execute(
        select(EvalResult)
        .where(EvalResult.experiment_id == experiment_id)
        .where(EvalResult.dataset_name == AUTO_LIFT_DATASET)
        .order_by(desc(EvalResult.id))
        .limit(1)
    )
    latest = result.scalar_one_or_none()
    if latest is None:
        return None
    metrics = latest.metrics if isinstance(latest.metrics, dict) else {}
    judge = metrics.get("judge") if isinstance(metrics.get("judge"), dict) else {}
    score = metrics.get("judge_correct")
    if not isinstance(score, (int, float)):
        return None
    counts = judge.get("counts") if isinstance(judge.get("counts"), dict) else {}
    return {
        "judge": str(judge.get("judge") or ""),
        "score": float(score),
        "counts": counts,
        "n_rows": int(judge.get("judged") or metrics.get("evaluated_samples") or 0),
        "eval_result_id": latest.id,
    }


async def _judged_run(db: AsyncSession, exp: Experiment) -> dict[str, Any] | None:
    """Judged lift for one trained run. A seed-group leader never evaluated
    itself: its children did — their judged scores (same judge) are averaged,
    counts summed, ``n_seeds`` recorded."""
    from app.services.post_training_eval_service import is_seed_group_leader, seed_group_children

    if not is_seed_group_leader(exp):
        judged = await _judged_test_eval(db, exp.id)
        return {"experiment_id": exp.id, "judged": True, **judged} if judged else None
    per_seed = []
    for child in await seed_group_children(db, exp):
        judged = await _judged_test_eval(db, child.id)
        if judged:
            per_seed.append(judged)
    if not per_seed or len({j["judge"] for j in per_seed}) != 1:
        return None
    counts: dict[str, int] = {}
    for j in per_seed:
        for key, value in (j.get("counts") or {}).items():
            counts[key] = counts.get(key, 0) + int(value or 0)
    return {
        "experiment_id": exp.id,
        "judged": True,
        "judge": per_seed[0]["judge"],
        "score": sum(j["score"] for j in per_seed) / len(per_seed),
        "counts": counts,
        "n_rows": per_seed[0]["n_rows"],
        "n_seeds": len(per_seed),
        "eval_result_id": per_seed[0]["eval_result_id"],
    }


async def _latest_judged_lift(db: AsyncSession, project_id: int) -> dict[str, Any] | None:
    """The latest trained run's judged lift. When the latest run was not
    judged (lift-checked before a judge existed), the newest judged run
    stands in — flagged ``finetune_is_latest=False`` with the latest run's
    id, so the verdict names what it compared and what it couldn't."""
    rows = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(Experiment.status == ExperimentStatus.COMPLETED)
        .order_by(desc(Experiment.id))
    )
    trained: list[Experiment] = []
    for exp in rows.scalars():
        cfg = exp.config if isinstance(exp.config, dict) else {}
        if cfg.get("is_baseline") is True or (exp.seed_group_id and exp.seed_value is not None):
            continue
        trained.append(exp)
    if not trained:
        return None
    latest = trained[0]
    for exp in trained:
        judged = await _judged_run(db, exp)
        if judged:
            judged["latest_experiment_id"] = latest.id
            judged["finetune_is_latest"] = exp.id == latest.id
            return judged
    return {"experiment_id": latest.id, "latest_experiment_id": latest.id, "judged": False, "finetune_is_latest": True}


def compare_passages_to_finetune(
    passages: dict[str, Any] | None, finetune: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Pure verdict. ``passages_win`` is True only when: both were judged by
    the same judge, retrieval's gain over the bare base model is beyond row
    noise, and the passages score is above the fine-tuned run's. The two
    were measured on different rows (val vs test), which the verdict says."""
    if passages is None:
        return None
    verdict: dict[str, Any] = {
        "passages": passages,
        "finetune": finetune,
        "comparable": False,
        "passages_win": False,
        "reason": None,
        # Same rows as the lift check (test examples)? False for a passages
        # comparison scored on the validation split.
        "same_rows": str(passages.get("split") or "val") == "test",
    }
    retrieval_verdict = (passages.get("retrieval_evidence") or {}).get("verdict")
    if retrieval_verdict != "better":
        verdict["reason"] = "retrieval_gain_within_noise"
        return verdict
    if not finetune or not finetune.get("judged"):
        verdict["reason"] = "no_judged_finetuned_run"
        return verdict
    if str(finetune.get("judge") or "") != str(passages.get("judge") or ""):
        verdict["reason"] = "different_judges"
        return verdict
    verdict["comparable"] = True
    verdict["passages_win"] = float(passages["with_passages"]) > float(finetune["score"])
    verdict["reason"] = "passages_ahead" if verdict["passages_win"] else "finetune_ahead"
    return verdict


async def passages_vs_finetune(db: AsyncSession, project_id: int) -> dict[str, Any] | None:
    """The verdict for a project, or None when no judged passages comparison
    exists (the common case — most projects never run it)."""
    passages = summarize_documents_comparison(read_documents_comparison(project_id))
    if passages is None:
        return None
    finetune = await _latest_judged_lift(db, project_id)
    return compare_passages_to_finetune(passages, finetune)


# ── Pre-training gate ────────────────────────────────────────────────────
#
# A documents project can measure "the base model + my passages" BEFORE it
# trains anything (the same judged comparison, on the test examples, run as
# a retrieval sweep so it judges the best retrieval the project can serve).
# Three tiers, calibrated on the legal case and the support-faq sample
# (see slm-docs/docs/workflows/training.md, "the pre-training gate"):
#
#   retrieval_ready      retrieval alone is a working assistant: stop, reroute
#                        (training on generated pairs is the long way round).
#   retrieval_promising  retrieval already answers a fair share but misses too
#                        much to ship: train, and let the judged lift compare
#                        the two on the same rows.
#   retrieval_weak       retrieval alone is not it: train (or fix retrieval).
#
# "Ready" needs ALL of: retrieval's gain over the bare base model beyond row
# noise, a mean judge score ≥ READY_MIN_SCORE, at least READY_MIN_CORRECT_SHARE
# fully-correct answers and at most READY_MAX_WRONG_SHARE wrong ones — a
# system that is wrong a third of the time is not "already answering",
# however good its mean looks. "Promising" needs the gain beyond noise and a
# score ≥ PROMISING_MIN_SCORE.
PASSAGES_GATE_READY_MIN_SCORE = 0.65
PASSAGES_GATE_READY_MIN_CORRECT_SHARE = 0.5
PASSAGES_GATE_READY_MAX_WRONG_SHARE = 0.25
PASSAGES_GATE_PROMISING_MIN_SCORE = 0.4
# Kept for callers that read the old names.
PASSAGES_GATE_MIN_SCORE = PASSAGES_GATE_READY_MIN_SCORE
PASSAGES_GATE_MIN_CORRECT_SHARE = PASSAGES_GATE_READY_MIN_CORRECT_SHARE


def passages_gate(passages: dict[str, Any] | None) -> dict[str, Any]:
    """Pure: ``status`` is ``retrieval_ready``, ``retrieval_promising``,
    ``retrieval_weak``, ``not_judged`` (no judge was reachable; F1 can't
    call it) or ``not_run``. Never looks at F1."""
    if passages is None:
        return {"status": "not_run", "reason": "no judged base + passages comparison yet"}
    score = float(passages.get("with_passages") or 0.0)
    counts = passages.get("with_passages_counts") or {}
    judged = sum(int(counts.get(k, 0) or 0) for k in ("correct", "partial", "wrong"))
    correct = int(counts.get("correct", 0) or 0)
    wrong = int(counts.get("wrong", 0) or 0)
    correct_share = correct / judged if judged else 0.0
    wrong_share = wrong / judged if judged else 1.0
    evidence = passages.get("retrieval_evidence") or {}
    beyond_noise = evidence.get("verdict") == "better"
    ready = (
        beyond_noise
        and score >= PASSAGES_GATE_READY_MIN_SCORE
        and correct_share >= PASSAGES_GATE_READY_MIN_CORRECT_SHARE
        and wrong_share <= PASSAGES_GATE_READY_MAX_WRONG_SHARE
    )
    promising = not ready and beyond_noise and score >= PASSAGES_GATE_PROMISING_MIN_SCORE
    status = "retrieval_ready" if ready else "retrieval_promising" if promising else "retrieval_weak"
    out = {
        "status": status,
        "score": round(score, 4),
        "correct": correct,
        "partial": int(counts.get("partial", 0) or 0),
        "wrong": wrong,
        "judged": judged,
        "judge": passages.get("judge"),
        "split": passages.get("split"),
        "retrieval_evidence": evidence,
        "thresholds": {
            "ready_min_score": PASSAGES_GATE_READY_MIN_SCORE,
            "ready_min_correct_share": PASSAGES_GATE_READY_MIN_CORRECT_SHARE,
            "ready_max_wrong_share": PASSAGES_GATE_READY_MAX_WRONG_SHARE,
            "promising_min_score": PASSAGES_GATE_PROMISING_MIN_SCORE,
        },
    }
    if ready:
        out["reason"] = (
            f"the base model answering from your passages already gets {correct} of {judged} right "
            f"and only {wrong} wrong (judge score {score:.2f}, beyond noise)"
        )
    elif not beyond_noise:
        out["reason"] = "retrieval's gain over the bare base model is within noise"
    elif promising:
        out["reason"] = (
            f"retrieval already gets {correct} of {judged} right but {wrong} wrong (judge score {score:.2f}) — "
            "not yet an assistant on its own; training will be compared against it on the same rows"
        )
    else:
        out["reason"] = f"only {correct} of {judged} fully right with passages (judge score {score:.2f})"
    return out


def read_passages_gate(project_id: int) -> dict[str, Any]:
    """The gate for a project from its cached documents comparison. A
    comparison that ran without a judge is ``not_judged`` — token F1 cannot
    call this gate."""
    payload = read_documents_comparison(project_id)
    passages = summarize_documents_comparison(payload)
    if payload is not None and passages is None:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        return {
            "status": "not_judged",
            "reason": "the comparison ran without a judge model, so only token F1 is available",
            "on_mean_f1": summary.get("on_mean_f1"),
            "off_mean_f1": summary.get("off_mean_f1"),
        }
    return passages_gate(passages)


def explicit_corpus(project: Project | None) -> str | None:
    cfg = getattr(project, "runtime_config", None)
    if not isinstance(cfg, dict):
        return None
    value = str(cfg.get(AUTO_RAG_CORPUS_KEY) or "").strip().lower()
    return value if value in {"qa", "documents", "auto"} else None


async def resolve_playground_corpus(db: AsyncSession, project: Project) -> tuple[str, str]:
    """What the playground's "Answer from your documents" retrieves:
    ``(corpus, reason)``. An explicit ``runtime_config.auto_rag_corpus``
    wins; else document passages when they beat the fine-tuned run; else
    ``auto`` (Q&A pairs when the recipe has them, otherwise passages)."""
    explicit = explicit_corpus(project)
    if explicit:
        return explicit, "project_setting"
    try:
        verdict = await passages_vs_finetune(db, project.id)
    except Exception:  # noqa: BLE001 — a verdict read never breaks chat
        verdict = None
    if verdict and verdict.get("passages_win"):
        return "documents", "passages_beat_finetune"
    return "auto", "default"

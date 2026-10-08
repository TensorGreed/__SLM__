"""The Eval tab's one default card (Wave 3b): is the fine-tuned model
better than its base model, by how much, and where does it still fail?

Composes the paired lift summary (sft_lift_summary_service — baseline for
the run's own base model, lower-is-better aware) with the run's latest
held-out result and up to five failing rows captured at eval time
(``details.failures_preview``).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.models.project import Project

SUMMARY_FAILURES = 5


async def _latest_trained_experiment_id(db: AsyncSession, project_id: int) -> int | None:
    rows = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(Experiment.status == ExperimentStatus.COMPLETED)
        .order_by(desc(Experiment.id))
    )
    for exp in rows.scalars():
        cfg = exp.config if isinstance(exp.config, dict) else {}
        if cfg.get("is_baseline") is True:
            continue
        # A seed-group child is one seed of its leader's run; the leader is
        # the run the user launched (and the one summarised across seeds).
        if exp.seed_group_id and exp.seed_value is not None:
            continue
        return exp.id
    return None


async def _seed_group_summary(
    db: AsyncSession, project_id: int, leader: Experiment
) -> dict[str, Any] | None:
    """Summary for a multi-seed run: every completed, evaluated child vs the
    base model, rolled up to a mean headline plus ``seed_evidence`` (does the
    lift hold across seeds?). The five failures and the row-level evidence
    come from the median seed, named in ``representative_experiment_id``.
    None when no child has an eval result yet (caller falls back to the
    not-evaluated card)."""
    from app.services.paired_comparison_stats import seed_spread_evidence
    from app.services.post_training_eval_service import seed_group_children
    from app.services.sft_lift_summary_service import compute_sft_lift_summary

    children = await seed_group_children(db, leader)
    per_seed: list[dict[str, Any]] = []
    for child in children:
        latest = (
            await db.execute(
                select(EvalResult)
                .where(EvalResult.experiment_id == child.id)
                .order_by(desc(EvalResult.id))
                .limit(1)
            )
        ).scalar_one_or_none()
        if latest is None:
            continue
        lift = await compute_sft_lift_summary(db, project_id, experiment_id=child.id)
        lifts = lift.get("metric_lifts") or []
        headline = next((row for row in lifts if row.get("is_headline")), lifts[0] if lifts else None)
        per_seed.append({
            "experiment_id": child.id,
            "seed_value": child.seed_value,
            "eval_result": latest,
            "lift": lift,
            "headline": headline if lift.get("status") == "ok" else None,
        })
    if not per_seed:
        return None

    scored = [s for s in per_seed if s["headline"]]
    first_lift = per_seed[0]["lift"]
    if not scored or len({s["headline"]["metric_id"] for s in scored}) != 1:
        status = first_lift.get("status")
        verdict = "no_baseline" if status == "no_baseline" else "no_comparison"
        representative = per_seed[0]
        failures, failed_count = _failures_from(representative["eval_result"])
        return {
            "project_id": project_id,
            "experiment_id": leader.id,
            "verdict": verdict,
            "message": first_lift.get("message"),
            "headline": None,
            "evidence": None,
            "seed_evidence": None,
            "seeds": [
                {"experiment_id": s["experiment_id"], "seed_value": s["seed_value"], "headline": s["headline"]}
                for s in per_seed
            ],
            "n_seeds": len(per_seed),
            "representative_experiment_id": representative["experiment_id"],
            "metric_lifts": [],
            "baseline": first_lift.get("baseline"),
            "trained": {"experiment_id": leader.id, "experiment_name": leader.name},
            "eval_result_id": representative["eval_result"].id,
            "eval_type": representative["eval_result"].eval_type,
            "evaluated_samples": None,
            "failures": failures,
            "failed_count": failed_count,
            "dataset_name": representative["eval_result"].dataset_name,
        }

    metric_id = scored[0]["headline"]["metric_id"]
    baseline_value = float(scored[0]["headline"]["baseline_value"])
    values = [float(s["headline"]["trained_value"]) for s in scored]
    seed_evidence = seed_spread_evidence(baseline_value, values)
    seed_evidence["metric_id"] = metric_id
    mean = float(seed_evidence["mean"])
    delta = mean - baseline_value
    headline = {
        "metric_id": metric_id,
        "baseline_value": round(baseline_value, 4),
        "trained_value": round(mean, 4),
        "trained_std": round(float(seed_evidence["std"]), 4) if seed_evidence["std"] is not None else None,
        "absolute_delta": round(delta, 4),
        "relative_delta_pct": round(delta / baseline_value * 100.0, 1) if baseline_value > 0 else None,
        "direction": "improved" if delta > 0.0001 else "regressed" if delta < -0.0001 else "unchanged",
        "is_headline": True,
        "n_seeds": len(scored),
    }
    verdict = {"improved": "better", "regressed": "worse"}.get(headline["direction"], "same")
    # The median seed stands in for the group where one run is needed
    # (failures to show, row-level evidence).
    ordered = sorted(scored, key=lambda s: float(s["headline"]["trained_value"]))
    representative = ordered[len(ordered) // 2]
    failures, failed_count = _failures_from(representative["eval_result"])
    metrics = representative["eval_result"].metrics if isinstance(representative["eval_result"].metrics, dict) else {}
    details = representative["eval_result"].details if isinstance(representative["eval_result"].details, dict) else {}
    return {
        "project_id": project_id,
        "experiment_id": leader.id,
        "verdict": verdict,
        "message": None,
        "headline": headline,
        "evidence": representative["headline"].get("evidence"),
        "seed_evidence": seed_evidence,
        "seeds": [
            {
                "experiment_id": s["experiment_id"],
                "seed_value": s["seed_value"],
                "headline": s["headline"],
            }
            for s in per_seed
        ],
        "n_seeds": len(scored),
        "representative_experiment_id": representative["experiment_id"],
        "metric_lifts": representative["lift"].get("metric_lifts") or [],
        "baseline": representative["lift"].get("baseline"),
        "trained": {"experiment_id": leader.id, "experiment_name": leader.name},
        "eval_result_id": representative["eval_result"].id,
        "eval_type": representative["eval_result"].eval_type,
        "evaluated_samples": metrics.get("evaluated_samples") or metrics.get("eval_documents"),
        "failures": failures,
        "failed_count": failed_count,
        "judge": _judge_block(metrics),
        "dataset_name": (details.get("dataset") or {}).get("name") if isinstance(details.get("dataset"), dict) else representative["eval_result"].dataset_name,
    }


def _failures_from(result: EvalResult) -> tuple[list[dict[str, Any]], int | None]:
    from app.services.evaluation_service import _prediction_failed

    details = result.details if isinstance(result.details, dict) else {}
    failures = details.get("failures_preview")
    if isinstance(failures, list):
        count = details.get("failed_count")
        return failures[:SUMMARY_FAILURES], int(count) if isinstance(count, int) else len(failures)
    # Results recorded before failures were captured: best effort from
    # the first-rows preview.
    preview = details.get("predictions_preview")
    if isinstance(preview, list):
        failed = [p for p in preview if isinstance(p, dict) and _prediction_failed(p)]
        return [
            {k: p.get(k) for k in ("prompt", "reference", "prediction", "row_exact_match", "row_f1", "row_judge_verdict", "row_judge_reason")}
            for p in failed[:SUMMARY_FAILURES]
        ], None
    return [], None


async def build_eval_summary(
    db: AsyncSession,
    project_id: int,
    experiment_id: int | None = None,
) -> dict[str, Any]:
    from app.services.sft_lift_summary_service import compute_sft_lift_summary

    if experiment_id is None:
        experiment_id = await _latest_trained_experiment_id(db, project_id)
    if experiment_id is None:
        # A RAG-first project never trains: its "did it help?" is the base
        # model with vs without passage retrieval (judged, test examples). A
        # not-yet-trained documents project with the pre-training check done
        # reads the same way, with a note that nothing has been trained.
        passages = passages_summary(project_id)
        if passages is not None:
            project = await db.get(Project, project_id)
            from app.services.rag_project_service import is_rag_first

            if not is_rag_first(project):
                from app.services.passage_rag_verdict_service import passages_gate, summarize_documents_comparison, read_documents_comparison

                gate = passages_gate(summarize_documents_comparison(read_documents_comparison(project_id)))
                passages["pretraining_gate"] = gate
                status = gate.get("status")
                passages["message"] = (
                    "Nothing has been trained yet — this is the base model answering from your passages. "
                    + (
                        "Retrieval already answers well: reroute to RAG, or train anyway to compare."
                        if status == "retrieval_ready"
                        else "Retrieval already answers a fair share but misses too much to ship on its own — "
                        "train, and the lift check will compare the two on these rows."
                        if status == "retrieval_promising"
                        else "Retrieval alone is not enough here; training may help, or improve retrieval first."
                    )
                )
            return passages
        return {
            "project_id": project_id,
            "experiment_id": None,
            "verdict": "no_trained_run",
            "message": "Train a model first — then this card tells you whether it beats the base model.",
            "headline": None,
            "failures": [],
        }

    experiment = await db.get(Experiment, experiment_id)
    if experiment is not None and experiment.seed_group_id and experiment.seed_value is None:
        grouped = await _seed_group_summary(db, project_id, experiment)
        if grouped is not None:
            return grouped

    latest = (
        await db.execute(
            select(EvalResult)
            .where(EvalResult.experiment_id == experiment_id)
            .order_by(desc(EvalResult.id))
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is None:
        return {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "verdict": "not_evaluated",
            "message": (
                "This run hasn't been evaluated yet. The automatic check runs after "
                "training finishes (watch the bell), or evaluate it now."
            ),
            "headline": None,
            "failures": [],
        }

    lift = await compute_sft_lift_summary(db, project_id, experiment_id=experiment_id)
    lifts = lift.get("metric_lifts") or []
    headline = next((row for row in lifts if row.get("is_headline")), lifts[0] if lifts else None)
    if lift.get("status") == "ok" and headline:
        verdict = {"improved": "better", "regressed": "worse"}.get(headline["direction"], "same")
        message = None
    elif lift.get("status") == "no_baseline":
        verdict, message = "no_baseline", lift.get("message")
    else:
        verdict, message = "no_comparison", lift.get("message")

    # Row-level evidence for the headline lift, computed with the lift rows
    # (sft_lift_summary_service._attach_row_evidence).
    evidence = headline.get("evidence") if (lift.get("status") == "ok" and headline) else None

    failures, failed_count = _failures_from(latest)
    details = latest.details if isinstance(latest.details, dict) else {}
    metrics = latest.metrics if isinstance(latest.metrics, dict) else {}
    return {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "verdict": verdict,
        "message": message,
        "headline": headline,
        # How solid the headline lift is: rows better / worse / same vs the
        # base model and whether the change is within noise. None for results
        # recorded before per-row scores were kept, or non-row metrics.
        "evidence": evidence,
        # Single run: no seed spread to report.
        "seed_evidence": None,
        "seeds": None,
        "n_seeds": 1,
        "metric_lifts": lifts,
        "baseline": lift.get("baseline"),
        "trained": lift.get("trained"),
        "eval_result_id": latest.id,
        "eval_type": latest.eval_type,
        "evaluated_samples": metrics.get("evaluated_samples") or metrics.get("eval_documents"),
        "failures": failures,
        "failed_count": failed_count,
        # The LLM judge's snapshot for the fine-tuned eval (answer_judge_service):
        # who judged, correct / partial / wrong counts, calls + cache hits.
        # None when the task isn't long-answer or no judge was reachable.
        "judge": _judge_block(metrics),
        "dataset_name": (details.get("dataset") or {}).get("name") if isinstance(details.get("dataset"), dict) else latest.dataset_name,
    }


def _judge_block(metrics: dict[str, Any]) -> dict[str, Any] | None:
    judge = metrics.get("judge")
    if not isinstance(judge, dict) or judge.get("score") is None:
        return None
    counts = judge.get("counts") if isinstance(judge.get("counts"), dict) else {}
    return {
        "judge": judge.get("judge"),
        "score": judge.get("score"),
        "judged": judge.get("judged"),
        "unjudged": judge.get("unjudged"),
        "correct": counts.get("correct", 0),
        "partial": counts.get("partial", 0),
        "wrong": counts.get("wrong", 0),
        "judge_calls": judge.get("judge_calls"),
        "judge_cached": judge.get("judge_cached"),
    }


def passages_summary(project_id: int) -> dict[str, Any] | None:
    """Eval summary for a project whose assistant is the base model +
    document-passage retrieval (a RAG sibling): built from the judged
    base-only documents comparison (``passage_rag_verdict_service``).
    ``kind="rag_passages"``; the headline is the judge score without →
    with retrieval, the evidence is the paired judge rows, the failures are
    the rows retrieval still got wrong. None when no judged comparison
    exists."""
    from app.services.passage_rag_verdict_service import (
        read_documents_comparison,
        summarize_documents_comparison,
    )

    payload = read_documents_comparison(project_id)
    passages = summarize_documents_comparison(payload)
    if passages is None or payload is None:
        return None
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    from app.api.auto_rag import _lift_evidence

    evidence = _lift_evidence(rows, score_key="judge")
    baseline = passages.get("without_retrieval")
    trained = float(passages["with_passages"])
    if not isinstance(baseline, (int, float)):
        return None
    delta = trained - float(baseline)
    direction = "improved" if delta > 0.0001 else "regressed" if delta < -0.0001 else "unchanged"
    verdict = {"improved": "better", "regressed": "worse"}.get(direction, "same")
    failures = []
    wrong = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        on = row.get("with_rag") if isinstance(row.get("with_rag"), dict) else {}
        judge = on.get("judge") if isinstance(on.get("judge"), dict) else None
        if not judge or judge.get("verdict") == "correct":
            continue
        wrong += 1
        if len(failures) < SUMMARY_FAILURES:
            failures.append({
                "prompt": str(row.get("question") or "")[:300],
                "reference": str(row.get("reference") or "")[:300],
                "prediction": str(on.get("generated") or "")[:300],
                "row_judge_verdict": judge.get("verdict"),
                "row_judge_reason": judge.get("reason"),
            })
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    judge_block = summary.get("judge") if isinstance(summary.get("judge"), dict) else {}
    on_arm = judge_block.get("with_rag") if isinstance(judge_block.get("with_rag"), dict) else {}
    counts = on_arm.get("counts") if isinstance(on_arm.get("counts"), dict) else {}
    return {
        "project_id": project_id,
        "experiment_id": None,
        "kind": "rag_passages",
        "verdict": verdict,
        "message": None,
        "headline": {
            "metric_id": "judge_correct",
            "baseline_value": round(float(baseline), 4),
            "trained_value": round(trained, 4),
            "absolute_delta": round(delta, 4),
            "relative_delta_pct": round(delta / float(baseline) * 100.0, 1) if baseline else None,
            "direction": direction,
            "is_headline": True,
        },
        "evidence": evidence,
        "seed_evidence": None,
        "seeds": None,
        "n_seeds": 1,
        "metric_lifts": [],
        "baseline": {"experiment_id": None, "base_model": passages.get("base_model")},
        "trained": None,
        "eval_result_id": None,
        "eval_type": "rag_passages",
        "evaluated_samples": passages.get("n_rows"),
        "split": passages.get("split"),
        "failures": failures,
        "failed_count": wrong,
        "judge": {
            "judge": passages.get("judge"),
            "score": trained,
            "judged": int(on_arm.get("judged") or passages.get("n_rows") or 0),
            "unjudged": int(on_arm.get("unjudged") or 0),
            "correct": counts.get("correct", 0),
            "partial": counts.get("partial", 0),
            "wrong": counts.get("wrong", 0),
            "judge_calls": on_arm.get("judge_calls"),
            "judge_cached": on_arm.get("judge_cached"),
        },
        "dataset_name": passages.get("split"),
        "cached_at": passages.get("cached_at"),
        # The retrieval scored (top-k + reranker) and, after a sweep, every
        # config tried with the judge's score — why this one is served.
        "retrieval": summary.get("retrieval"),
        "retrieval_sweep": summary.get("retrieval_sweep"),
    }


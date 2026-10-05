"""Automatic baseline-vs-fine-tuned lift eval after every real training run.

Before this, "did fine-tuning help?" only had an answer if the user found
the Quickstart baseline tile *and* ran held-out eval by hand. Now, when
the ``training_start`` watcher Job sees a real run complete, it spawns a
``post_training_lift_eval`` Job that:

  1. evaluates the un-fine-tuned base model on the held-out ``test``
     split (reusing a cached baseline result when the split hasn't
     changed since it was computed),
  2. evaluates the fine-tuned checkpoint on the same split with the same
     settings,
  3. returns the lift for that exact (base model, run) pair.

Simulated runs, baseline rows, seed-group children (their leader is checked
as a group — every seed, plus whether the lift holds across seeds) and runs without real
weights are skipped with a reason. Opt out per project via
``runtime_config["auto_lift_eval"] = False`` or globally via
``settings.AUTO_LIFT_EVAL_ENABLED``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.experiment import (
    EvalResult,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)
from app.models.project import Project

AUTO_LIFT_DATASET = "test"


def _lift_eval_limits() -> tuple[int, int]:
    """(max_samples, max_new_tokens) for the lift check — settings-backed so
    a CPU-only CI gate can shrink them."""
    return (
        max(1, int(settings.AUTO_LIFT_EVAL_MAX_SAMPLES)),
        max(1, int(settings.AUTO_LIFT_EVAL_MAX_NEW_TOKENS)),
    )
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt")


# ── Baseline experiment helpers (shared with the Quickstart tile) ──────


def short_model_name(model_id: str) -> str:
    """``HuggingFaceTB/SmolLM2-135M-Instruct`` → ``SmolLM2-135M-Instruct``."""
    if not model_id:
        return "base model"
    return model_id.rsplit("/", 1)[-1] or model_id


def eval_type_for_project(project: Project) -> str:
    """Informational eval_type label; the handler reads real metrics off
    ``prepared/manifest.json``. Documents-only projects (continued
    pretraining) are measured by held-out perplexity."""
    from app.services.continued_pretraining_policy import project_is_documents_only

    if project_is_documents_only(project):
        return "perplexity"
    snapshot = project.selected_recipe or {}
    scoring_mode = str(snapshot.get("scoring_mode") or "").strip()
    if scoring_mode == "span_set":
        return "f1"
    return "exact_match"


async def find_or_create_baseline_experiment(
    db: AsyncSession,
    project_id: int,
    base_model: str,
    *,
    source: str = "quickstart.baseline-eval",
) -> Experiment:
    """The synthetic Baseline experiment for (project, base_model).
    ``output_dir`` stays empty — eval is called with
    ``model_path=base_model`` to bypass the artifact resolver."""
    name = f"Baseline · {short_model_name(base_model)}"[:255]
    result = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(Experiment.name == name)
        .limit(1)
    )
    existing = result.scalar_one_or_none()
    if existing is not None:
        return existing

    exp = Experiment(
        project_id=project_id,
        name=name,
        description=(
            "Synthetic baseline experiment — the un-fine-tuned base model "
            "evaluated against the project's gold/test split. Anchors "
            "post-SFT eval numbers so 'F1 0.65' has context."
        ),
        status=ExperimentStatus.COMPLETED,
        training_mode=TrainingMode.SFT,
        base_model=base_model,
        output_dir=None,
        config={"is_baseline": True, "source": source},
        final_train_loss=None,
        final_eval_loss=None,
    )
    db.add(exp)
    await db.flush()
    return exp


# ── Eligibility ────────────────────────────────────────────────────────


def _model_dir_has_weights(output_dir: str | None) -> bool:
    if not output_dir:
        return False
    root = Path(output_dir)
    model_dir = root / "model"
    candidate = model_dir if model_dir.is_dir() else root
    if not candidate.is_dir():
        return False
    has_config = (candidate / "config.json").exists() or (
        candidate / "adapter_config.json"
    ).exists()
    has_weights = any(
        p.suffix in _WEIGHT_SUFFIXES for p in candidate.iterdir() if p.is_file()
    )
    return has_config and has_weights


def auto_lift_skip_reason(exp: Experiment, project: Project | None) -> str | None:
    """``None`` when the run should get an automatic lift eval, else a
    short machine-readable reason."""
    if not settings.AUTO_LIFT_EVAL_ENABLED:
        return "disabled_globally"
    runtime_config = (project.runtime_config or {}) if project is not None else {}
    if isinstance(runtime_config, dict) and runtime_config.get("auto_lift_eval") is False:
        return "disabled_for_project"
    cfg = exp.config if isinstance(exp.config, dict) else {}
    if cfg.get("is_baseline") is True:
        return "baseline_experiment"
    if exp.status != ExperimentStatus.COMPLETED:
        return "not_completed"
    if exp.seed_group_id and exp.seed_value is not None:
        return "seed_group_child"
    runtime = cfg.get("_runtime") if isinstance(cfg.get("_runtime"), dict) else {}
    backend = str(runtime.get("backend") or "").lower()
    runtime_id = str(runtime.get("runtime_id") or "").lower()
    if backend == "simulate" or "simulate" in runtime_id:
        return "simulated_run"
    # A seed-group leader never trained itself: its children hold the
    # weights (each is resolved when evaluated; a group with no completed
    # child fails the check with a clear reason).
    if is_seed_group_leader(exp):
        return None
    if not _model_dir_has_weights(exp.output_dir):
        return "no_model_weights"
    return None


async def _test_split_mtime(project_id: int) -> datetime | None:
    path = Path(settings.DATA_DIR) / "projects" / str(project_id) / "prepared" / "test.jsonl"
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


async def _reusable_baseline_result(
    db: AsyncSession, baseline_exp: Experiment, project_id: int
) -> EvalResult | None:
    """Latest baseline result on the test split, if newer than the split
    file (base models are deterministic under greedy decoding, so a
    fresh-enough result is exact, not approximate)."""
    result = await db.execute(
        select(EvalResult)
        .where(EvalResult.experiment_id == baseline_exp.id)
        .where(EvalResult.dataset_name == AUTO_LIFT_DATASET)
        .order_by(desc(EvalResult.id))
        .limit(1)
    )
    latest = result.scalar_one_or_none()
    if latest is None or latest.created_at is None:
        return None
    split_mtime = await _test_split_mtime(project_id)
    created = latest.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if split_mtime is not None and created < split_mtime:
        return None
    # Results recorded before per-row scores were kept can't be paired row by
    # row with the fine-tuned eval ("within noise?") — re-run the base once.
    details = latest.details if isinstance(latest.details, dict) else {}
    if not isinstance(details.get("row_scores"), dict):
        return None
    return latest


# ── Runner ─────────────────────────────────────────────────────────────


async def run_post_training_lift_eval(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    progress=None,
) -> dict[str, Any]:
    """Baseline + fine-tuned held-out eval, then the paired lift summary.
    Commits after each eval so a failure in the second keeps the first.
    Raises when the run isn't eligible (the Job surfaces the reason)."""
    from app.services.evaluation_service import run_heldout_evaluation
    from app.services.sft_lift_summary_service import compute_sft_lift_summary

    async def _report(fraction: float, message: str) -> None:
        if progress is not None:
            await progress(fraction, message)

    exp = await db.get(Experiment, experiment_id)
    project = await db.get(Project, project_id)
    if exp is None or project is None:
        raise ValueError(f"Experiment {experiment_id} / project {project_id} not found")
    reason = auto_lift_skip_reason(exp, project)
    if reason is not None:
        raise ValueError(f"Automatic lift eval skipped: {reason}")
    if is_seed_group_leader(exp):
        return await _run_seed_group_lift_eval(db, project=project, leader=exp, report=_report)

    eval_type = eval_type_for_project(project)
    max_samples, max_new_tokens = _lift_eval_limits()
    common = {
        "project_id": project_id,
        "dataset_name": AUTO_LIFT_DATASET,
        "eval_type": eval_type,
        "max_samples": max_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "judge_model": None,
    }

    baseline_exp = await find_or_create_baseline_experiment(
        db, project_id, exp.base_model, source="post_training.auto_lift"
    )
    await db.commit()
    baseline_result = await _reusable_baseline_result(db, baseline_exp, project_id)
    baseline_reused = baseline_result is not None
    if baseline_result is None:
        await _report(0.1, f"Evaluating base model {short_model_name(exp.base_model)}")
        baseline_result = await run_heldout_evaluation(
            db=db,
            experiment_id=baseline_exp.id,
            model_path=exp.base_model,
            **common,
        )
        await db.commit()

    await _report(0.55, f"Evaluating fine-tuned run #{experiment_id}")
    trained_result = await run_heldout_evaluation(
        db=db,
        experiment_id=experiment_id,
        model_path=None,
        **common,
    )
    await db.commit()

    await _report(0.95, "Computing lift")
    summary = await compute_sft_lift_summary(db, project_id, experiment_id=experiment_id)
    headline = next(
        (row for row in summary.get("metric_lifts") or [] if row.get("is_headline")),
        None,
    )
    # How solid the headline lift is (rows better / worse, within noise?) —
    # the bell must not announce "better than base" for a change that could
    # be chance. None when per-row scores aren't available.
    full = (headline or {}).get("evidence") if summary.get("status") == "ok" else None
    evidence = (
        {k: full.get(k) for k in ("verdict", "n", "better", "worse", "same")}
        if isinstance(full, dict)
        else None
    )
    return {
        "experiment_id": experiment_id,
        "base_model": exp.base_model,
        "evidence": evidence,
        "baseline_experiment_id": baseline_exp.id,
        "baseline_eval_result_id": getattr(baseline_result, "id", None),
        "baseline_reused": baseline_reused,
        "trained_eval_result_id": getattr(trained_result, "id", None),
        "lift_status": summary.get("status"),
        "headline": headline,
    }


def is_seed_group_leader(exp: Experiment) -> bool:
    """The marker run of a multi-seed group (it never trained itself; its
    N children did)."""
    return bool(exp.seed_group_id) and exp.seed_value is None


async def seed_group_children(db: AsyncSession, leader: Experiment) -> list[Experiment]:
    """The leader's COMPLETED children, in seed order."""
    rows = await db.execute(
        select(Experiment)
        .where(Experiment.seed_group_id == leader.seed_group_id)
        .where(Experiment.seed_value.is_not(None))
        .where(Experiment.status == ExperimentStatus.COMPLETED)
        .order_by(Experiment.seed_value, Experiment.id)
    )
    return list(rows.scalars())


async def _run_seed_group_lift_eval(
    db: AsyncSession, *, project: Project, leader: Experiment, report
) -> dict[str, Any]:
    """The multi-seed lift check: every completed child of the group is
    scored on the test split against the one (cached) base-model result;
    the result reports the headline per seed plus ``seed_evidence`` — whether
    the lift holds across seeds, not just across which rows were sampled.

    Before this, a seed-group leader was evaluated as if it were a single
    run (it borrows its first child's output_dir), so "3 seeds" reported the
    first seed only, under the leader's name.
    """
    from app.services.evaluation_service import run_heldout_evaluation
    from app.services.paired_comparison_stats import seed_spread_evidence
    from app.services.sft_lift_summary_service import compute_sft_lift_summary

    children = await seed_group_children(db, leader)
    if not children:
        raise ValueError("Automatic lift eval skipped: seed group has no completed child runs")

    eval_type = eval_type_for_project(project)
    max_samples, max_new_tokens = _lift_eval_limits()
    common = {
        "project_id": project.id,
        "dataset_name": AUTO_LIFT_DATASET,
        "eval_type": eval_type,
        "max_samples": max_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "judge_model": None,
    }
    baseline_exp = await find_or_create_baseline_experiment(
        db, project.id, leader.base_model, source="post_training.auto_lift"
    )
    await db.commit()
    baseline_result = await _reusable_baseline_result(db, baseline_exp, project.id)
    baseline_reused = baseline_result is not None
    if baseline_result is None:
        await report(0.05, f"Evaluating base model {short_model_name(leader.base_model)}")
        baseline_result = await run_heldout_evaluation(
            db=db, experiment_id=baseline_exp.id, model_path=leader.base_model, **common
        )
        await db.commit()

    seeds: list[dict[str, Any]] = []
    for idx, child in enumerate(children):
        fraction = 0.15 + 0.75 * idx / len(children)
        await report(fraction, f"Evaluating seed {child.seed_value} (run #{child.id}, {idx + 1}/{len(children)})")
        trained_result = await run_heldout_evaluation(
            db=db, experiment_id=child.id, model_path=None, **common
        )
        await db.commit()
        summary = await compute_sft_lift_summary(db, project.id, experiment_id=child.id)
        headline = next(
            (row for row in summary.get("metric_lifts") or [] if row.get("is_headline")), None
        )
        seeds.append({
            "experiment_id": child.id,
            "seed_value": child.seed_value,
            "eval_result_id": getattr(trained_result, "id", None),
            "lift_status": summary.get("status"),
            "headline": headline,
        })

    await report(0.95, "Computing lift across seeds")
    scored = [s for s in seeds if s["headline"]]
    metric_ids = {s["headline"]["metric_id"] for s in scored}
    seed_evidence = None
    headline: dict[str, Any] | None = None
    if scored and len(metric_ids) == 1:
        metric_id = next(iter(metric_ids))
        baseline_value = float(scored[0]["headline"]["baseline_value"])
        trained_values = [float(s["headline"]["trained_value"]) for s in scored]
        seed_evidence = seed_spread_evidence(baseline_value, trained_values)
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
    return {
        "experiment_id": leader.id,
        "base_model": leader.base_model,
        "n_seeds": len(scored),
        "seeds": seeds,
        "seed_evidence": seed_evidence,
        # Row-level evidence is per seed (``seeds[i].headline.evidence``);
        # the bell leads with the seed spread.
        "evidence": None,
        "baseline_experiment_id": baseline_exp.id,
        "baseline_eval_result_id": getattr(baseline_result, "id", None),
        "baseline_reused": baseline_reused,
        "trained_eval_result_id": None,
        "lift_status": "ok" if headline else "no_overlap",
        "headline": headline,
    }


async def start_post_training_lift_job(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
) -> dict[str, Any]:
    """Spawn the lift-eval Job for a just-completed run, or return why not.
    Never raises — a failure here must not fail the training watcher."""
    try:
        exp = await db.get(Experiment, experiment_id)
        project = await db.get(Project, project_id)
        if exp is None:
            return {"started": False, "skipped_reason": "experiment_missing"}
        reason = auto_lift_skip_reason(exp, project)
        if reason is not None:
            return {"started": False, "skipped_reason": reason}

        from app.database import async_session_factory
        from app.services.jobs_service import JobProgressHandle, start_job

        async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
            async def _progress(fraction: float, message: str) -> None:
                await handle.set_progress(fraction=fraction, message=message)

            async with async_session_factory() as runner_db:
                return await run_post_training_lift_eval(
                    runner_db,
                    project_id=project_id,
                    experiment_id=experiment_id,
                    progress=_progress,
                )

        run_label = f"run #{experiment_id}"
        if is_seed_group_leader(exp):
            n_children = len(await seed_group_children(db, exp))
            run_label = f"run #{experiment_id} ({n_children} seeds)"
        job = await start_job(
            db,
            kind="post_training_lift_eval",
            title=f"Did fine-tuning help? · {run_label} vs {short_model_name(exp.base_model)}",
            runner=_runner,
            project_id=project_id,
            params={"experiment_id": experiment_id, "base_model": exp.base_model},
        )
        return {"started": True, "job_id": getattr(job, "id", None)}
    except Exception as exc:  # noqa: BLE001
        return {"started": False, "skipped_reason": f"error:{exc.__class__.__name__}: {exc}"}

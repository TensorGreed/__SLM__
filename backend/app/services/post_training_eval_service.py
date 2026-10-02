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

Simulated runs, baseline rows, seed-group children and runs without real
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
AUTO_LIFT_MAX_SAMPLES = 100
AUTO_LIFT_MAX_NEW_TOKENS = 256
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

    eval_type = eval_type_for_project(project)
    common = {
        "project_id": project_id,
        "dataset_name": AUTO_LIFT_DATASET,
        "eval_type": eval_type,
        "max_samples": AUTO_LIFT_MAX_SAMPLES,
        "max_new_tokens": AUTO_LIFT_MAX_NEW_TOKENS,
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
    # be chance. Best-effort: None when per-row scores aren't available.
    evidence = None
    if summary.get("status") == "ok":
        try:
            from app.services.eval_summary_service import headline_lift_evidence

            full = await headline_lift_evidence(db, summary, headline)
            if full is not None:
                evidence = {k: full.get(k) for k in ("verdict", "n", "better", "worse", "same")}
        except Exception:  # noqa: BLE001 — never fail the lift job over the note
            evidence = None
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

        job = await start_job(
            db,
            kind="post_training_lift_eval",
            title=f"Did fine-tuning help? · run #{experiment_id} vs {short_model_name(exp.base_model)}",
            runner=_runner,
            project_id=project_id,
            params={"experiment_id": experiment_id, "base_model": exp.base_model},
        )
        return {"started": True, "job_id": getattr(job, "id", None)}
    except Exception as exc:  # noqa: BLE001
        return {"started": False, "skipped_reason": f"error:{exc.__class__.__name__}: {exc}"}

"""Project-guide quickstart endpoints (Theme 1 Epic 4 + Theme 8 Epic 1).

Four one-click actions that turn the first checklist items in
`ProjectGuidePage` into literal buttons rather than descriptions of
what to click:

  - POST /api/projects/{id}/quickstart/import-sample
        Materialize a bundled demo dataset into the existing
        project (raw + gold + prepared splits).

  - POST /api/projects/{id}/quickstart/baseline-eval
        Run an evaluation against the gold/test split using the
        *un-fine-tuned* base model. Surfaces a pre-SFT anchor so
        the post-training number has context. Creates / reuses a
        synthetic "Baseline" Experiment so the result lands on the
        normal eval plumbing.

  - POST /api/projects/{id}/quickstart/train-default
        Create + start a training experiment using the project's
        existing defaults.

  - POST /api/projects/{id}/quickstart/evaluate-latest
        Find the most recent completed training experiment in the
        project and run a heldout evaluation.

These endpoints are thin wrappers around the existing service
functions — they don't introduce new behaviour, they just remove the
form-filling step. Power users still have the full surfaces (recipe
override, training config editor, eval dataset picker) in their
respective tabs.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode
from app.models.project import Project
from app.services.demo_project_service import (
    apply_demo_bundle_to_project,
    derive_demo_slug_for_project,
)
from app.services.evaluation_service import run_heldout_evaluation
from app.services.training_service import create_experiment, start_training

router = APIRouter(prefix="/projects/{project_id}/quickstart", tags=["Quickstart"])


# ── Request schemas ──────────────────────────────────────────────────


class ImportSampleRequest(BaseModel):
    slug: str | None = Field(
        default=None,
        description=(
            "Demo bundle slug (e.g. 'support-faq', 'sentiment-classifier', "
            "'pii-detector'). When omitted, derived from the project's "
            "selected_recipe — falls back to 'support-faq' if no recipe."
        ),
        max_length=64,
    )


# ── Helpers ──────────────────────────────────────────────────────────


async def _get_project_or_404(db: AsyncSession, project_id: int) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _eval_type_for_project(project: Project) -> str:
    """Pick the eval_type label for a quickstart eval call. The
    eval handler reads the real metrics off `prepared/manifest.json`;
    this label is mostly informational on the result row."""
    snapshot = project.selected_recipe or {}
    scoring_mode = str(snapshot.get("scoring_mode") or "").strip()
    if scoring_mode == "span_set":
        return "f1"
    return "exact_match"


def _short_model_name(model_id: str) -> str:
    """Trim a HF model id to its trailing component for use in a
    user-facing experiment name. `HuggingFaceTB/SmolLM2-135M-Instruct`
    → `SmolLM2-135M-Instruct`."""
    if not model_id:
        return "base model"
    return model_id.rsplit("/", 1)[-1] or model_id


async def _find_or_create_baseline_experiment(
    db: AsyncSession,
    project_id: int,
    base_model: str,
) -> Experiment:
    """Find the existing baseline Experiment for this project +
    base_model combo, or create one. Scoped to (project_id,
    base_model) so switching base models gives each its own
    baseline row (useful when comparing recipes)."""
    name = f"Baseline · {_short_model_name(base_model)}"[:255]
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
        # `output_dir` left empty on purpose — `run_heldout_evaluation`
        # is called with `model_path=base_model` to bypass the artifact
        # resolver. The empty output_dir is the signal this is a
        # synthetic / baseline row.
        output_dir=None,
        config={
            "is_baseline": True,
            "source": "quickstart.baseline-eval",
        },
        final_train_loss=None,
        final_eval_loss=None,
    )
    db.add(exp)
    await db.flush()
    return exp


# ── Endpoints ────────────────────────────────────────────────────────


@router.post("/import-sample", status_code=201)
async def quickstart_import_sample(
    project_id: int,
    data: ImportSampleRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Materialize a bundled demo dataset into the existing project."""
    requested_slug = (data.slug if data else None) or None
    try:
        summary = await apply_demo_bundle_to_project(
            db,
            project_id,
            requested_slug,
        )
    except ValueError as e:
        detail = str(e)
        if detail.startswith("project_not_found:"):
            raise HTTPException(404, detail)
        if detail.startswith("demo_slug_unknown:"):
            raise HTTPException(404, detail)
        if detail.startswith("demo_manifest_invalid:"):
            raise HTTPException(400, detail)
        raise HTTPException(400, detail)
    return {
        "status": "ok",
        "summary": summary,
    }


@router.post("/train-default", status_code=201)
async def quickstart_train_default(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Create + start a training experiment using project defaults
    (base_model_name from the selected recipe; everything else uses
    the TrainingConfig Pydantic defaults)."""
    project = await _get_project_or_404(db, project_id)

    base_model = str(project.base_model_name or "").strip()
    if not base_model:
        raise HTTPException(
            400,
            (
                "Project has no base_model_name. Pick a recipe in the dataset-"
                "import wizard, or set the model manually in Training → Config."
            ),
        )

    snapshot = project.selected_recipe or {}
    recipe_id = str(snapshot.get("recipe_id") or "").strip()
    task_profile = str(snapshot.get("task_profile") or "instruction_sft").strip()
    # Translate task profile → trainer task_type. Most recipes are
    # causal_lm-shaped; classification recipes use a different head.
    if task_profile == "classification":
        task_type = "classification"
    elif task_profile == "seq2seq":
        task_type = "seq2seq"
    else:
        task_type = "causal_lm"

    config_payload: dict = {
        "base_model": base_model,
        "training_mode": TrainingMode.SFT.value,
        "task_type": task_type,
    }

    name = (
        f"Quickstart · {snapshot.get('name') or 'default config'}"
        if recipe_id
        else "Quickstart · default config"
    )[:255]
    description = "Launched from the project-guide quickstart."

    try:
        exp = await create_experiment(
            db,
            project_id,
            name,
            base_model,
            config_payload,
            description,
            TrainingMode.SFT,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        start_result = await start_training(db, project_id, exp.id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(404, detail)
        if "already running" in detail or "already completed" in detail:
            raise HTTPException(409, detail)
        raise HTTPException(400, detail)

    return {
        "status": "training_started",
        "experiment_id": exp.id,
        "experiment_name": exp.name,
        "base_model": exp.base_model,
        "training_mode": exp.training_mode.value if exp.training_mode else None,
        "recipe_id": recipe_id or None,
        "start_result": start_result,
    }


@router.post("/evaluate-latest", status_code=201)
async def quickstart_evaluate_latest(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Find the most recent completed (or running) training experiment
    in the project and run a heldout evaluation against the gold/test
    split."""
    project = await _get_project_or_404(db, project_id)

    # Prefer completed; fall back to running so users can see early
    # numbers while a long training is mid-flight.
    result = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .where(
            Experiment.status.in_(
                [ExperimentStatus.COMPLETED, ExperimentStatus.RUNNING]
            )
        )
        .order_by(desc(Experiment.id))
        .limit(1)
    )
    experiment = result.scalar_one_or_none()
    if experiment is None:
        raise HTTPException(
            409,
            (
                "No completed or running training experiments yet. Run "
                "'Train default config' first; then come back here."
            ),
        )

    snapshot = project.selected_recipe or {}
    scoring_mode = str(snapshot.get("scoring_mode") or "").strip()
    # Default eval type — exact_match is the safe default that works
    # across most task shapes. The eval handler picks the real
    # task-shape-specific metrics from the prepared manifest, this
    # parameter is mostly a label.
    eval_type = "exact_match"
    if scoring_mode == "span_set":
        eval_type = "f1"

    try:
        result_dict = await run_heldout_evaluation(
            db=db,
            project_id=project_id,
            experiment_id=experiment.id,
            dataset_name="test",
            eval_type=eval_type,
            max_samples=100,
            max_new_tokens=256,
            temperature=0.0,
            model_path=None,
            judge_model=None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "status": "evaluation_complete",
        "experiment_id": experiment.id,
        "eval_type": eval_type,
        "result": result_dict,
    }


@router.post("/baseline-eval", status_code=201)
async def quickstart_baseline_eval(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Run eval against the gold/test split using the project's
    un-fine-tuned base model. Creates (or reuses) a synthetic
    Baseline experiment so the result row keeps its anchor in the
    normal `EvalResult` plumbing; the Compare page then renders
    "baseline vs. trained" side-by-side for free.

    Requires:
      - project.base_model_name set (i.e. recipe applied)
      - imported / prepared data with a `test` split present

    Returns the synthetic experiment_id + the eval result dict so
    the UI tile can show "Baseline F1: 0.32" immediately."""
    project = await _get_project_or_404(db, project_id)

    base_model = str(project.base_model_name or "").strip()
    if not base_model:
        raise HTTPException(
            400,
            (
                "Project has no base_model_name. Pick a recipe in the dataset-"
                "import wizard, or set the model manually in Training → Config."
            ),
        )

    baseline_exp = await _find_or_create_baseline_experiment(
        db, project_id, base_model,
    )
    eval_type = _eval_type_for_project(project)

    try:
        result_dict = await run_heldout_evaluation(
            db=db,
            project_id=project_id,
            experiment_id=baseline_exp.id,
            dataset_name="test",
            eval_type=eval_type,
            max_samples=100,
            max_new_tokens=256,
            temperature=0.0,
            # Bypass the artifact resolver — the baseline experiment
            # has no output_dir; we want the raw base model loaded
            # straight from the HF id.
            model_path=base_model,
            judge_model=None,
        )
    except ValueError as e:
        detail = str(e)
        # Map the two "data not ready yet" failure shapes
        # (`run_heldout_evaluation` raises with these messages):
        #   - "No datasets found in project N"        — nothing imported
        #   - "Dataset '<name>' not found in project N" — no test split
        if (
            ("No datasets found" in detail)
            or ("Dataset" in detail and "not found" in detail)
        ):
            raise HTTPException(
                409,
                "No test split found yet. Run 'Import sample CSV' first.",
            )
        raise HTTPException(400, detail)

    return {
        "status": "baseline_complete",
        "experiment_id": baseline_exp.id,
        "experiment_name": baseline_exp.name,
        "base_model": base_model,
        "eval_type": eval_type,
        "result": result_dict,
    }

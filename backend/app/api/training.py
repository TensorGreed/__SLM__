"""Training API routes."""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
import asyncio
import json
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.experiment import Checkpoint, Experiment, TrainingMode
from app.models.project import Project
from app.schemas.training import ExperimentCreate, ExperimentResponse, TrainingConfig
from app.services.job_service import get_task_status
from app.services.domain_profile_service import get_training_defaults
from app.services.domain_runtime_service import resolve_project_domain_runtime
from app.services.training_service import (
    cancel_training,
    create_experiment,
    get_training_status,
    list_experiments,
    start_training,
    register_websocket,
    unregister_websocket,
)
from app.services.training_preflight_service import (
    TRAINING_PLAN_PROFILES,
    run_training_preflight,
    run_training_preflight_plan,
)
from app.services.training_recipe_service import (
    list_training_recipes,
    resolve_training_recipe,
)
from app.services.training_runtime_service import list_runtime_catalog

router = APIRouter(prefix="/projects/{project_id}/training", tags=["Training"])


class TrainingEffectiveConfigRequest(BaseModel):
    config: dict[str, object] = Field(default_factory=dict)


class TrainingPreferencesUpdateRequest(BaseModel):
    preferred_plan_profile: str = Field(..., min_length=1, max_length=32)


class TrainingRecipeResolveRequest(BaseModel):
    recipe_id: str = Field(..., min_length=1, max_length=128)
    base_config: dict[str, object] = Field(default_factory=dict)
    overrides: dict[str, object] = Field(default_factory=dict)
    include_preflight: bool = True


@router.get("/runtimes")
async def get_training_runtime_catalog(
    project_id: int,
):
    """List registered training runtime plugins and server default runtime."""
    return {
        "project_id": project_id,
        **list_runtime_catalog(),
    }


@router.get("/recipes")
async def get_training_recipe_catalog(
    project_id: int,
):
    """List built-in training recipes."""
    return {
        "project_id": project_id,
        "recipe_count": len(list_training_recipes(include_patch=False)),
        "recipes": list_training_recipes(include_patch=False),
    }


@router.post("/recipes/resolve")
async def resolve_training_recipe_config(
    project_id: int,
    req: TrainingRecipeResolveRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resolve recipe config with runtime defaults and optional preflight."""
    try:
        recipe_resolution = resolve_training_recipe(
            req.recipe_id,
            base_config=dict(req.base_config or {}),
            overrides=dict(req.overrides or {}),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    candidate_config = dict(recipe_resolution.get("resolved_config") or {})
    (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    ) = await _resolve_effective_training_preview(
        project_id=project_id,
        source_config=candidate_config,
        db=db,
    )

    preflight = None
    if bool(req.include_preflight):
        preflight = run_training_preflight(
            project_id=project_id,
            config=resolved_config,
            base_model=str(resolved_config.get("base_model", "")),
        )

    return {
        "project_id": project_id,
        "recipe": recipe_resolution.get("recipe"),
        "recipe_missing_required_fields": recipe_resolution.get("missing_required_fields", []),
        "recipe_config": candidate_config,
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "profile_training_defaults": (
            dict(profile_training_defaults) if profile_training_defaults else None
        ),
        "resolved_training_config": resolved_config,
        "resolved_training_mode": resolved_training_mode.value,
        "profile_defaults_applied": profile_defaults_applied,
        "preflight": preflight,
    }


def _resolve_training_config(
    *,
    config_payload: dict,
    provided_config_fields: set[str],
    profile_training_defaults: dict,
    fallback_training_mode: TrainingMode,
) -> tuple[dict, TrainingMode, list[str]]:
    resolved_config = dict(config_payload)
    profile_defaults_applied: list[str] = []

    if profile_training_defaults:
        for key, value in profile_training_defaults.items():
            if key == "training_mode":
                continue
            if key not in provided_config_fields:
                resolved_config[key] = value
                profile_defaults_applied.append(key)

    resolved_training_mode = fallback_training_mode
    if (
        "training_mode" not in provided_config_fields
        and isinstance(profile_training_defaults.get("training_mode"), str)
    ):
        candidate_mode = str(profile_training_defaults["training_mode"]).strip().lower()
        try:
            resolved_training_mode = TrainingMode(candidate_mode)
            resolved_config["training_mode"] = resolved_training_mode.value
            profile_defaults_applied.append("training_mode")
        except ValueError:
            pass

    return resolved_config, resolved_training_mode, sorted(set(profile_defaults_applied))


def _normalize_preferred_plan_profile(value: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in TRAINING_PLAN_PROFILES:
        return candidate
    allowed = ", ".join(TRAINING_PLAN_PROFILES)
    raise HTTPException(
        400,
        f"Invalid preferred_plan_profile '{candidate}'. Allowed values: {allowed}",
    )


async def _resolve_effective_training_preview(
    *,
    project_id: int,
    source_config: dict[str, object],
    db: AsyncSession,
) -> tuple[
    dict,
    dict,
    dict,
    TrainingMode,
    list[str],
]:
    provided_config_fields = set(source_config.keys())
    if "base_model" not in source_config:
        source_config["base_model"] = "microsoft/phi-2"

    try:
        parsed_config = TrainingConfig.model_validate(source_config)
    except Exception as e:
        raise HTTPException(400, f"Invalid training config: {e}")

    try:
        runtime = await resolve_project_domain_runtime(db, project_id)
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)

    effective_contract = runtime.get("effective_contract")
    profile_training_defaults = get_training_defaults(effective_contract)
    resolved_config, resolved_training_mode, profile_defaults_applied = _resolve_training_config(
        config_payload=parsed_config.model_dump(),
        provided_config_fields=provided_config_fields,
        profile_training_defaults=profile_training_defaults,
        fallback_training_mode=parsed_config.training_mode,
    )
    return (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    )


@router.get("/preferences")
async def get_training_preferences(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Read persisted training UI/runtime preferences for a project."""
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    stored = str(project.training_preferred_plan_profile or "").strip().lower()
    preferred = stored if stored in TRAINING_PLAN_PROFILES else "balanced"
    source = "project" if stored in TRAINING_PLAN_PROFILES else "default"
    return {
        "project_id": project_id,
        "preferred_plan_profile": preferred,
        "profile_options": list(TRAINING_PLAN_PROFILES),
        "source": source,
    }


@router.put("/preferences")
async def set_training_preferences(
    project_id: int,
    req: TrainingPreferencesUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist training preferences for a project."""
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    preferred = _normalize_preferred_plan_profile(req.preferred_plan_profile)
    project.training_preferred_plan_profile = preferred
    await db.flush()

    return {
        "project_id": project_id,
        "preferred_plan_profile": preferred,
        "profile_options": list(TRAINING_PLAN_PROFILES),
        "source": "project",
    }


@router.websocket("/ws/{experiment_id}")
async def ws_training_status(
    websocket: WebSocket,
    project_id: int, # Keep project_id for consistency with other routes, though not directly used in this new implementation
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket endpoint for real-time training metrics and terminal logs.
    """
    await websocket.accept()

    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    exp = exp_result.scalar_one_or_none()
    if not exp:
        await websocket.close(code=1008, reason="Experiment not found")
        return

    register_websocket(experiment_id, websocket)

    ckpt_result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.experiment_id == experiment_id)
        .order_by(Checkpoint.step.asc())
    )
    checkpoints = ckpt_result.scalars().all()
    metrics = [
        {
            "experiment_id": experiment_id,
            "epoch": c.epoch,
            "step": c.step,
            "train_loss": c.train_loss,
            "eval_loss": c.eval_loss,
        }
        for c in checkpoints
    ]
    try:
        await websocket.send_json({"type": "init", "metrics": metrics})
    except Exception:
        unregister_websocket(experiment_id, websocket)
        return

    # Background task to stream logs and metric envelopes from Redis PubSub.
    async def log_stream_loop():
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"log:experiment:{experiment_id}")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    raw = message.get("data", "")
                    envelope = None
                    if isinstance(raw, str):
                        try:
                            parsed = json.loads(raw)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict) and isinstance(parsed.get("type"), str):
                            envelope = parsed

                    try:
                        if envelope is None:
                            await websocket.send_json({"type": "log", "text": str(raw)})
                            continue
                        event_type = str(envelope.get("type", "")).strip().lower()
                        if event_type == "metric" and isinstance(envelope.get("metric"), dict):
                            metric = dict(envelope["metric"])
                            metric.setdefault("experiment_id", experiment_id)
                            await websocket.send_json({"type": "metric", "metric": metric})
                        elif event_type == "status":
                            payload = {"type": "status", "status": envelope.get("status", "")}
                            if "error" in envelope:
                                payload["error"] = envelope.get("error")
                            if "returncode" in envelope:
                                payload["returncode"] = envelope.get("returncode")
                            await websocket.send_json(payload)
                        elif event_type == "log":
                            await websocket.send_json({"type": "log", "text": str(envelope.get("text", ""))})
                        else:
                            await websocket.send_json({"type": "log", "text": str(raw)})
                    except Exception:
                        break  # client disconnected
        finally:
            await pubsub.unsubscribe()
            await redis_client.aclose()

    log_task = asyncio.create_task(log_stream_loop())

    try:
        while True:
            # Keep connection alive; client might send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        log_task.cancel()
        unregister_websocket(experiment_id, websocket)


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create(
    project_id: int,
    data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new training experiment."""
    try:
        config_payload = data.config.model_dump()
        provided_config_fields = set(data.config.model_fields_set)

        runtime = await resolve_project_domain_runtime(db, project_id)
        effective_contract = runtime.get("effective_contract")
        profile_training_defaults = get_training_defaults(effective_contract)
        config_payload, resolved_training_mode, profile_defaults_applied = _resolve_training_config(
            config_payload=config_payload,
            provided_config_fields=provided_config_fields,
            profile_training_defaults=profile_training_defaults,
            fallback_training_mode=data.config.training_mode,
        )

        exp = await create_experiment(
            db,
            project_id,
            data.name,
            str(config_payload.get("base_model", data.config.base_model)),
            config_payload,
            data.description,
            resolved_training_mode,
        )
        response_payload = ExperimentResponse.model_validate(exp).model_dump()
        response_payload["domain_pack_applied"] = runtime.get("domain_pack_applied")
        response_payload["domain_pack_source"] = runtime.get("domain_pack_source")
        response_payload["domain_profile_applied"] = runtime.get("domain_profile_applied")
        response_payload["domain_profile_source"] = runtime.get("domain_profile_source")
        response_payload["profile_training_defaults"] = (
            dict(profile_training_defaults) if profile_training_defaults else None
        )
        response_payload["resolved_training_config"] = dict(config_payload)
        response_payload["profile_defaults_applied"] = sorted(set(profile_defaults_applied))
        return ExperimentResponse.model_validate(response_payload)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/experiments/effective-config")
async def effective_training_config(
    project_id: int,
    req: TrainingEffectiveConfigRequest,
    db: AsyncSession = Depends(get_db),
):
    """Preview effective training config after domain runtime defaults are applied."""
    source_config = dict(req.config or {})
    (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    ) = await _resolve_effective_training_preview(
        project_id=project_id,
        source_config=source_config,
        db=db,
    )

    return {
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "profile_training_defaults": (
            dict(profile_training_defaults) if profile_training_defaults else None
        ),
        "resolved_training_config": resolved_config,
        "resolved_training_mode": resolved_training_mode.value,
        "profile_defaults_applied": profile_defaults_applied,
    }


@router.post("/experiments/preflight")
async def training_preflight(
    project_id: int,
    req: TrainingEffectiveConfigRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resolve effective config and run capability/runtime preflight checks."""
    source_config = dict(req.config or {})
    (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    ) = await _resolve_effective_training_preview(
        project_id=project_id,
        source_config=source_config,
        db=db,
    )

    preflight = run_training_preflight(
        project_id=project_id,
        config=resolved_config,
        base_model=str(resolved_config.get("base_model", "")),
    )

    return {
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "profile_training_defaults": (
            dict(profile_training_defaults) if profile_training_defaults else None
        ),
        "resolved_training_config": resolved_config,
        "resolved_training_mode": resolved_training_mode.value,
        "profile_defaults_applied": profile_defaults_applied,
        "preflight": preflight,
    }


@router.post("/experiments/preflight/plan")
async def training_preflight_plan(
    project_id: int,
    req: TrainingEffectiveConfigRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resolve effective config and generate suggested preflight tuning plans."""
    source_config = dict(req.config or {})
    (
        runtime,
        profile_training_defaults,
        resolved_config,
        resolved_training_mode,
        profile_defaults_applied,
    ) = await _resolve_effective_training_preview(
        project_id=project_id,
        source_config=source_config,
        db=db,
    )

    plan = run_training_preflight_plan(
        project_id=project_id,
        config=resolved_config,
        base_model=str(resolved_config.get("base_model", "")),
    )

    return {
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "profile_training_defaults": (
            dict(profile_training_defaults) if profile_training_defaults else None
        ),
        "resolved_training_config": resolved_config,
        "resolved_training_mode": resolved_training_mode.value,
        "profile_defaults_applied": profile_defaults_applied,
        "plan": plan,
    }


@router.get("/experiments/{experiment_id}/preflight")
async def training_preflight_for_experiment(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Run preflight checks for an existing experiment config."""
    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    exp = exp_result.scalar_one_or_none()
    if not exp:
        raise HTTPException(404, f"Experiment {experiment_id} not found in project {project_id}")

    raw_config = dict(exp.config or {})
    raw_config.setdefault("base_model", exp.base_model)
    try:
        parsed = TrainingConfig.model_validate(raw_config)
    except Exception as e:
        raise HTTPException(400, f"Experiment {experiment_id} has invalid training config: {e}")

    resolved_config = parsed.model_dump()
    preflight = run_training_preflight(
        project_id=project_id,
        config=resolved_config,
        base_model=exp.base_model,
    )
    return {
        "experiment_id": exp.id,
        "status": exp.status.value,
        "resolved_training_config": resolved_config,
        "preflight": preflight,
    }


@router.post("/experiments/{experiment_id}/start")
async def start(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Start training for an experiment."""
    try:
        return await start_training(db, project_id, experiment_id)
    except ValueError as e:
        detail = str(e)
        if "already running" in detail or "already completed" in detail:
            raise HTTPException(409, detail)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.post("/experiments/{experiment_id}/cancel")
async def cancel(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running training experiment."""
    try:
        return await cancel_training(db, project_id, experiment_id)
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(404, detail)
        raise HTTPException(409, detail)


@router.get("/experiments/{experiment_id}/status")
async def status(
    project_id: int,
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get training status and metrics."""
    try:
        return await get_training_status(db, project_id, experiment_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/tasks/{task_id}")
async def task_status(
    project_id: int,
    task_id: str,
):
    """Read Celery task state for training-related jobs."""
    try:
        return get_task_status(task_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/experiments", response_model=list[ExperimentResponse])
async def list_all(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all experiments for a project."""
    exps = await list_experiments(db, project_id)
    return [ExperimentResponse.model_validate(e) for e in exps]

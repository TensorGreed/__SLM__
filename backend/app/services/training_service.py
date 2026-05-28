"""Training pipeline service — SFT, LoRA, checkpoint management."""

import asyncio
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import WebSocket
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory
from app.models.experiment import (
    Checkpoint,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)
from app.models.project import Project
from app.services.checkpoint_registry_service import resolve_starting_checkpoint
from app.services.training_data_gate import (
    DEFAULT_TARGET_FIELDS,
    verify_training_data_has_targets,
)
from app.services.training_preflight_service import (
    evaluate_training_base_model_compatibility,
    run_training_preflight,
)
from app.services.alignment_dataset_service import (
    compose_alignment_training_dataset,
    filter_preference_dataset_by_quality,
    resolve_alignment_dataset_path,
)
from app.services.training_runtime_service import (
    TrainingRuntimeStartContext,
    get_runtime_spec,
    resolve_training_runtime_id,
    start_runtime,
    validate_runtime,
)
from app.services.vibe_check_service import (
    capture_vibe_check_snapshot,
    load_project_vibe_check_config,
)

active_websockets: dict[int, list[WebSocket]] = {}
TRAINING_EVENT_PREFIX = "SLM_EVENT "

def register_websocket(experiment_id: int, ws: WebSocket):
    if experiment_id not in active_websockets:
        active_websockets[experiment_id] = []
    active_websockets[experiment_id].append(ws)

def unregister_websocket(experiment_id: int, ws: WebSocket):
    if experiment_id in active_websockets and ws in active_websockets[experiment_id]:
        active_websockets[experiment_id].remove(ws)


async def broadcast_event(experiment_id: int, payload: dict) -> None:
    """Broadcast an event envelope to all active websockets for an experiment."""
    if experiment_id not in active_websockets:
        return
    dead_socks = []
    for ws in active_websockets[experiment_id]:
        try:
            await ws.send_json(payload)
        except Exception:
            dead_socks.append(ws)
    for ws in dead_socks:
        active_websockets[experiment_id].remove(ws)


async def _get_experiment_for_project(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> Experiment | None:
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    return result.scalar_one_or_none()


def _experiment_dir(project_id: int, experiment_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "experiments" / str(experiment_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _decide_auto_rag_default(*, project_obj: Project | None) -> dict:
    """Phase 9d — return {'should_default_on': bool, 'reason': str}.

    Fires True iff the project's recipe is RAG-eligible (today: qa-sft
    only, via auto_rag_service's recipe map). Returns False with a
    categorical reason otherwise — the reason flows into
    ``config._auto_rag_auto_defaulted`` for the UI to show on the
    experiment card so users understand why auto-RAG is (or isn't)
    on by default.

    Unlike curriculum's default, auto-RAG doesn't have a row-count
    threshold — the Phase 9c A/B (+146% lift) was on the same ~140-
    row training set we'd see in most QA-SFT projects, and the auto-
    RAG benefit grows with corpus size (more training rows → better
    retrieval), so there's no thin-vs-thick distinction to make."""
    from app.services.auto_rag_service import recommended_text_keys_for_recipe

    selected_recipe = (project_obj.selected_recipe or {}) if project_obj else {}
    recipe_id = selected_recipe.get("recipe_id")
    if not recipe_id:
        return {"should_default_on": False, "reason": "no_recipe_selected"}
    if recommended_text_keys_for_recipe(recipe_id) is None:
        return {
            "should_default_on": False,
            "reason": f"recipe_has_no_auto_rag:{recipe_id}",
        }
    return {
        "should_default_on": True,
        "reason": f"rag_eligible_recipe:{recipe_id}",
    }


async def _safe_build_auto_rag_index(
    db: AsyncSession, *, project_id: int
) -> dict:
    """Phase 9b — fire-and-forget auto-RAG index build hook called
    from the training-completion paths. Wraps
    ``auto_rag_service.build_index_for_project`` in a catch-all so
    a broken build (corrupt prepared file, unexpected exception)
    can never prevent the experiment from being marked COMPLETED.

    The returned dict is the same shape the service emits — gets
    stamped onto ``runtime_config["auto_rag_build"]`` so the UI
    can show "auto-RAG index built (N docs)" or the skip reason."""
    try:
        from app.services.auto_rag_service import build_index_for_project

        return await build_index_for_project(db, project_id)
    except Exception as e:  # noqa: BLE001 — never block training completion
        return {
            "built": False,
            "reason": f"unexpected_error:{type(e).__name__}:{e}",
        }


# Phase 6d — curriculum-learning default-on heuristic threshold.
# The Phase 6c A/B (2026-05-25, 5 seeds, GB10 GPU) cleared the gate
# with classification templates at exactly 144 training rows; the
# 200 ceiling gives headroom for slightly thicker thin-data projects
# while staying well below the regime where uniform training is
# already strong. Revisit if a future A/B with larger projects
# (200-500 rows) shows the lift plateauing.
CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS: int = 200


def _decide_curriculum_default(
    *,
    project_obj: Project | None,
    project_id: int,
) -> dict:
    """Return {'should_default_on': bool, 'reason': str}.

    Fires True iff: project's selected recipe has a curriculum
    scoring mode (today: classification only) AND the prepared
    train.jsonl has ≤ ``CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS`` rows.
    Returns False with a categorical reason otherwise — the reason
    flows into ``config._curriculum_auto_defaulted`` for the UI to
    show on the experiment card so users understand why curriculum
    is (or isn't) on.

    Designed to be cheap: no embedder load, no DB hop beyond the
    project we already fetched. A missing prepared file means
    "default off" (gate fails closed — better than auto-on on a
    state we couldn't verify)."""
    # Local import to avoid pulling curriculum_service into the
    # training_service module-load if curriculum is never used.
    from app.services.curriculum_service import (
        recommended_scoring_mode_for_recipe,
    )

    selected_recipe = (project_obj.selected_recipe or {}) if project_obj else {}
    recipe_id = selected_recipe.get("recipe_id")
    if not recipe_id:
        return {"should_default_on": False, "reason": "no_recipe_selected"}
    if recommended_scoring_mode_for_recipe(recipe_id) is None:
        return {
            "should_default_on": False,
            "reason": f"recipe_has_no_curriculum:{recipe_id}",
        }

    train_file = settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"
    if not train_file.exists():
        return {"should_default_on": False, "reason": "no_prepared_train_file"}

    # Cheap line count — prepared/train.jsonl is one row per line.
    try:
        with train_file.open(encoding="utf-8") as f:
            row_count = sum(1 for line in f if line.strip())
    except OSError as e:
        return {"should_default_on": False, "reason": f"train_file_unreadable:{e}"}

    if row_count > CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS:
        return {
            "should_default_on": False,
            "reason": (
                f"thick_dataset:{row_count}>{CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS}"
            ),
        }

    return {
        "should_default_on": True,
        "reason": (
            f"thin_classification:{row_count}_rows_<=_"
            f"{CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS}_threshold"
        ),
    }


async def _maybe_apply_curriculum(
    *,
    db: AsyncSession,
    project_id: int,
    train_file: Path,
    output_dir: Path,
    resolved_config: dict,
    training_mode: str,
) -> dict:
    """Phase 6b — build the curriculum shard + signal train.py when
    ``config.curriculum`` is set + the project is curriculum-eligible.

    Returns a serializable block for ``runtime_config["curriculum"]``
    so the UI / debug surfaces can show whether curriculum applied,
    and if not, why. Mutates ``resolved_config`` to add
    ``curriculum_disable_shuffle=True`` when curriculum is actually
    applied (read by ``train.py`` to swap its default RandomSampler
    for a SequentialSampler).

    Recipe gating: today only ``classification`` has a curriculum
    scoring mode (Phase 6a shipped ``prototype_entropy``). Other
    recipes record a ``skip_reason`` and fall through to uniform
    training — the caller leaves ``train_file`` unchanged in that
    case.

    The block is always returned (never raises) so a curriculum
    failure can't prevent training from starting. Worst case: the
    user asked for curriculum, didn't get it, sees the
    ``skip_reason`` in the experiment's runtime config and the
    standard run proceeds.
    """
    block: dict[str, object] = {
        "requested": _coerce_bool(resolved_config.get("curriculum"), False),
        "applied": False,
    }
    if not block["requested"]:
        return block

    # Curriculum is an SFT-time concept; DPO/ORPO have their own
    # alignment-dataset path with different shape semantics.
    if training_mode in {"dpo", "orpo"}:
        block["skip_reason"] = f"unsupported_training_mode:{training_mode}"
        return block

    project = await db.get(Project, project_id)
    selected_recipe = (project.selected_recipe or {}) if project else {}
    recipe_id = selected_recipe.get("recipe_id")

    # Lazy import so the curriculum stack doesn't pull
    # sentence-transformers into module-load if curriculum is never used.
    from app.services.curriculum_service import (
        CurriculumUnavailable,
        build_curriculum_shards,
        recommended_scoring_mode_for_recipe,
    )

    scoring_mode = recommended_scoring_mode_for_recipe(recipe_id or "")
    if scoring_mode is None:
        block["skip_reason"] = f"unsupported_recipe:{recipe_id or 'unset'}"
        return block

    # Load the prepared training rows. The prepared file is the
    # authoritative training corpus for this run (gold + accepted
    # synth, post-dedupe, post-split), so we rank exactly the rows
    # that are going into training.
    rows: list[dict] = []
    try:
        with train_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        block["skip_reason"] = f"train_file_unreadable:{e}"
        return block

    if not rows:
        block["skip_reason"] = "train_file_empty"
        return block

    try:
        manifest = build_curriculum_shards(
            rows,
            scoring_mode=scoring_mode,
            output_dir=output_dir / "curriculum",
            cache_dir=(
                settings.DATA_DIR / "projects" / str(project_id) / "curriculum"
            ),
        )
    except CurriculumUnavailable as e:
        # sentence-transformers missing on this install — surface
        # the package name in the reason so the user can install it.
        block["skip_reason"] = f"embedder_unavailable:{e}"
        return block

    # Apply: signal train.py to disable shuffle, surface the manifest.
    resolved_config["curriculum_disable_shuffle"] = True
    block.update(
        {
            "applied": True,
            "scoring_mode": manifest["scoring_mode"],
            "shard_path": manifest["shard_path"],
            "meta_path": manifest["meta_path"],
            "easy_count": manifest["easy_count"],
            "total_rows": manifest["total_rows"],
            "recipe_id": recipe_id,
        }
    )
    return block


async def create_experiment(
    db: AsyncSession,
    project_id: int,
    name: str,
    base_model: str,
    config: dict,
    description: str = "",
    training_mode: TrainingMode = TrainingMode.SFT,
) -> Experiment:
    """Create a new training experiment."""
    project_row = await db.execute(select(Project).where(Project.id == project_id))
    project_obj = project_row.scalar_one_or_none()
    if not project_obj:
        raise ValueError(f"Project {project_id} not found")

    base_model_validation = evaluate_training_base_model_compatibility(base_model=base_model)
    if not bool(base_model_validation.get("ok", False)):
        validation_errors = [
            str(item).strip()
            for item in list(base_model_validation.get("errors") or [])
            if str(item).strip()
        ]
        if not validation_errors:
            validation_errors = ["base model compatibility check failed"]
        preview = "; ".join(validation_errors[:3])
        if len(validation_errors) > 3:
            preview = f"{preview}; (+{len(validation_errors) - 3} more)"
        raise ValueError(f"Base model validation failed: {preview}")

    # Phase 6d — auto-on heuristic for curriculum learning. Fires
    # only when the caller hasn't set ``curriculum`` explicitly, so
    # API/CLI callers who want a specific value get it (opt-out
    # preserved). Phase 6c A/B run (2026-05-25) showed +54-93% F1
    # lift on the 2 classification templates at ≤ 200 training rows.
    if "curriculum" not in config:
        auto_decision = _decide_curriculum_default(
            project_obj=project_obj,
            project_id=project_id,
        )
        if auto_decision["should_default_on"]:
            config = dict(config)
            config["curriculum"] = True
            config["_curriculum_auto_defaulted"] = auto_decision["reason"]

    # Phase 9d — auto-on heuristic for auto-RAG. Fires only when the
    # caller hasn't set ``auto_rag`` explicitly. Phase 9c A/B run
    # (2026-05-25) showed +146.49% F1 lift on the policy-qa-style
    # QA-SFT template (5 seeds, GB10, SmolLM2-135M-Instruct, 3
    # epochs). Heuristic gates on recipe alone — qa-sft → on; no
    # row-count threshold here because the auto-RAG benefit grows
    # with corpus size (more training rows = better retrieval) and
    # there's no thin-data overfitting risk like there was for
    # curriculum.
    if "auto_rag" not in config:
        auto_rag_decision = _decide_auto_rag_default(project_obj=project_obj)
        if auto_rag_decision["should_default_on"]:
            config = dict(config)
            config["auto_rag"] = {"enabled": True}
            config["_auto_rag_auto_defaulted"] = auto_rag_decision["reason"]

    exp = Experiment(
        project_id=project_id,
        name=name,
        description=description,
        base_model=base_model,
        config=config,
        training_mode=training_mode,
        status=ExperimentStatus.PENDING,
    )
    db.add(exp)
    await db.flush()
    await db.refresh(exp)

    # USER-SUCCESS Epic 1 (T5): record a forecast→reality observation
    # pairing this experiment with the user's most-recent forecast
    # snapshot. Silent no-op when no snapshot exists (user trained
    # without viewing the forecast first); those runs are excluded
    # from calibration aggregation rather than fail-loud here.
    try:
        from app.services.trainability_forecast_service import (
            record_forecast_observation,
        )

        await record_forecast_observation(db, exp.id)
    except Exception as obs_exc:
        # Calibration is best-effort; never block experiment creation.
        print(
            f"[forecast_calibration] record_failed experiment_id={exp.id}: {obs_exc}",
            flush=True,
        )

    # Create output dir
    output_dir = _experiment_dir(project_id, exp.id)
    exp.output_dir = str(output_dir)
    await db.flush()

    # Save config
    config_path = output_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return exp


async def broadcast_metric(experiment_id: int, metric: dict):
    """Broadcast metric to all active websockets for an experiment."""
    await broadcast_event(experiment_id, {"type": "metric", "metric": metric})


def _coerce_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off", ""}:
            return False
    return default


def _extract_step_from_checkpoint_dir(path: Path) -> int | None:
    suffix = path.name.removeprefix("checkpoint-")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _parse_stream_event_payload(line: str) -> dict | None:
    token = str(line or "").strip()
    if not token:
        return None
    marker = token.find(TRAINING_EVENT_PREFIX)
    if marker < 0:
        return None
    body = token[marker + len(TRAINING_EVENT_PREFIX):].strip()
    if not body:
        return None
    try:
        payload = json.loads(body)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _collect_observability_event_payloads(*streams: str | None) -> list[dict]:
    events: list[dict] = []
    for stream in streams:
        text = str(stream or "")
        if not text:
            continue
        for line in text.splitlines():
            payload = _parse_stream_event_payload(line)
            if not isinstance(payload, dict):
                continue
            event_name = str(payload.get("event") or "").strip().lower()
            if event_name not in {"observability", "training_observability"}:
                continue
            events.append(payload)
    return events


async def _monitor_external_training(
    experiment_id: int,
    process: asyncio.subprocess.Process,
    command: str,
    log_path: Path,
    output_dir: Path,
    *,
    captured_stdout: str | None = None,
    captured_stderr: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> str:
    """Monitor external training process and sync experiment status."""
    started = started_at or datetime.now(timezone.utc)
    final_status = "failed"
    try:
        if captured_stdout is None or captured_stderr is None:
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                process.kill()
                await process.communicate()
                raise ValueError(
                    (
                        "External training command timed out after "
                        f"{settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS} seconds"
                    )
                )
            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        else:
            stdout_text = captured_stdout
            stderr_text = captured_stderr

        finished = finished_at or datetime.now(timezone.utc)
        log_payload = {
            "command": command,
            "returncode": process.returncode,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "stdout": stdout_text,
            "stderr": stderr_text,
        }
        log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

        async with async_session_factory() as db:
            result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
            exp = result.scalar_one_or_none()
            if not exp:
                return final_status

            config = dict(exp.config or {})
            runtime = dict(config.get("_runtime") or {})
            runtime.update(
                {
                "backend": "external",
                "command": command,
                "log_path": str(log_path),
                "returncode": process.returncode,
                }
            )
            observability_events_ingested = 0
            observability_ingest_error = ""
            try:
                from app.services.training_telemetry_service import record_observability_event

                for payload in _collect_observability_event_payloads(stdout_text, stderr_text):
                    data = dict(payload)
                    data.pop("event", None)
                    data.setdefault("experiment_id", experiment_id)
                    record_observability_event(int(exp.project_id), payload=data)
                    observability_events_ingested += 1
            except Exception as telemetry_error:  # noqa: BLE001
                observability_ingest_error = str(telemetry_error)
            runtime["observability_events_ingested"] = observability_events_ingested
            if observability_ingest_error:
                runtime["observability_ingest_error"] = observability_ingest_error
            config["_runtime"] = runtime
            report_path = output_dir / "training_report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    runtime["report_path"] = str(report_path)
                    exp.final_train_loss = _coerce_float(report.get("final_train_loss"))
                    exp.final_eval_loss = _coerce_float(report.get("final_eval_loss"))

                    epochs = report.get("epochs")
                    if isinstance(epochs, (int, float)):
                        exp.total_epochs = int(epochs)
                    total_steps = report.get("total_steps")
                    if isinstance(total_steps, (int, float)):
                        exp.total_steps = int(total_steps)

                    existing_steps = set(
                        (
                            await db.execute(
                                select(Checkpoint.step).where(Checkpoint.experiment_id == experiment_id)
                            )
                        ).scalars().all()
                    )

                    report_checkpoints = report.get("checkpoints")
                    if isinstance(report_checkpoints, list):
                        for item in report_checkpoints:
                            if not isinstance(item, dict):
                                continue
                            step = item.get("step")
                            epoch = item.get("epoch")
                            file_path = item.get("file_path")
                            if not isinstance(step, int) or step <= 0 or step in existing_steps:
                                continue
                            if not isinstance(epoch, int) or epoch <= 0:
                                epoch = 1
                            if not isinstance(file_path, str) or not file_path:
                                continue
                            ckpt = Checkpoint(
                                experiment_id=experiment_id,
                                epoch=epoch,
                                step=step,
                                train_loss=_coerce_float(item.get("train_loss")),
                                eval_loss=_coerce_float(item.get("eval_loss")),
                                file_path=file_path,
                                is_best=bool(item.get("is_best", False)),
                                metrics=item,
                            )
                            db.add(ckpt)
                            existing_steps.add(step)
                    else:
                        checkpoints_root = output_dir
                        checkpoint_dirs = sorted(
                            [
                                p for p in checkpoints_root.glob("checkpoint-*")
                                if p.is_dir() and _extract_step_from_checkpoint_dir(p) is not None
                            ],
                            key=lambda p: _extract_step_from_checkpoint_dir(p) or 0,
                        )
                        for checkpoint_dir in checkpoint_dirs:
                            step = _extract_step_from_checkpoint_dir(checkpoint_dir)
                            if step is None or step in existing_steps:
                                continue
                            ckpt = Checkpoint(
                                experiment_id=experiment_id,
                                epoch=1,
                                step=step,
                                train_loss=None,
                                eval_loss=None,
                                file_path=str(checkpoint_dir),
                                is_best=False,
                                metrics={"source": "checkpoint_dir_scan"},
                            )
                            db.add(ckpt)
                            existing_steps.add(step)
                except Exception as parse_error:
                    runtime["report_parse_error"] = str(parse_error)

            exp.config = config

            if exp.status == ExperimentStatus.CANCELLED:
                runtime["cancelled_completion_returncode"] = process.returncode
                final_status = "cancelled"
            else:
                if process.returncode == 0:
                    exp.status = ExperimentStatus.COMPLETED
                    final_status = "completed"
                    # Phase 9b — fire the auto-RAG index build on
                    # training completion so the playground's auto-RAG
                    # path has an index to load. Best-effort: any
                    # failure lands on runtime["auto_rag_build"] as a
                    # skip reason, never blocks the COMPLETED status.
                    runtime["auto_rag_build"] = await _safe_build_auto_rag_index(
                        db, project_id=int(exp.project_id)
                    )
                    exp.config = {**config, "_runtime": runtime}
                else:
                    exp.status = ExperimentStatus.FAILED
                    final_status = "failed"
                exp.completed_at = finished
            await db.commit()
        await broadcast_event(
            experiment_id,
            {
                "type": "status",
                "status": final_status,
                "returncode": process.returncode,
            },
        )
        return final_status
    except Exception as e:
        async with async_session_factory() as db:
            result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
            exp = result.scalar_one_or_none()
            if exp:
                config = dict(exp.config or {})
                runtime = dict(config.get("_runtime") or {})
                runtime.update(
                    {
                    "backend": "external",
                    "error": str(e),
                    "log_path": str(log_path),
                    }
                )
                config["_runtime"] = runtime
                if exp.status != ExperimentStatus.CANCELLED:
                    exp.status = ExperimentStatus.FAILED
                    exp.completed_at = datetime.now(timezone.utc)
                exp.config = config
                await db.commit()
        await broadcast_event(
            experiment_id,
            {
                "type": "status",
                "status": "cancelled" if final_status == "cancelled" else "failed",
                "error": str(e),
            },
        )
        return "cancelled" if final_status == "cancelled" else "failed"


async def _simulate_training_loop(experiment_id: int, config: dict):
    """Simulate a training loop reporting metrics for demo purposes."""
    try:
        project_id: int | None = None
        experiment_output_dir: Path | None = None
        base_model = str(config.get("base_model") or "HuggingFaceTB/SmolLM2-135M-Instruct").strip() or "HuggingFaceTB/SmolLM2-135M-Instruct"
        experiment_config_snapshot: dict[str, object] = dict(config or {})
        vibe_enabled = True
        vibe_interval_steps = 50
        async with async_session_factory() as db:
            exp_result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
            exp = exp_result.scalar_one_or_none()
            if exp is not None:
                project_id = int(exp.project_id)
                if exp.output_dir:
                    experiment_output_dir = Path(str(exp.output_dir))
                base_model = str(exp.base_model or base_model).strip() or base_model
                experiment_config_snapshot = dict(exp.config or {})
                effective_vibe = load_project_vibe_check_config(
                    project_id,
                    experiment_config=experiment_config_snapshot,
                )
                vibe_enabled = bool(effective_vibe.get("enabled"))
                vibe_interval_steps = max(1, int(effective_vibe.get("interval_steps") or 50))

        epochs = config.get("num_epochs", 3)
        steps_per_epoch = 100
        total_steps = epochs * steps_per_epoch

        current_loss = 3.5
        lr = config.get("learning_rate", 2e-4)
        save_steps = config.get("save_steps", 100)

        # P17 — when this run is the result of a resume, start the loop at
        # the step the previous run paused at instead of restarting from 1.
        resume_block = dict((config or {}).get("_resume_from") or {})
        start_step = 1
        try:
            resumed_step = int(resume_block.get("checkpoint_step") or 0)
            if 0 < resumed_step < total_steps:
                start_step = resumed_step + 1
        except (TypeError, ValueError):
            start_step = 1

        for step in range(start_step, total_steps + 1):
            await asyncio.sleep(0.1)  # 100ms per step to make demo fast but visible

            if step % 10 == 0:
                async with async_session_factory() as db:
                    exp_result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
                    exp = exp_result.scalar_one_or_none()
                    if not exp or exp.status == ExperimentStatus.CANCELLED:
                        await broadcast_event(
                            experiment_id,
                            {"type": "status", "status": "cancelled"},
                        )
                        return
                    # P17 — operator pressed Pause. Write a resume-capable
                    # checkpoint at the current step, transition to PAUSED,
                    # and clear the request flag so the runtime can wind
                    # down cleanly. Resume will re-dispatch this loop with
                    # ``config._resume_from`` set to start_step+1 above.
                    if exp.pause_requested:
                        epoch_num = max(1, ((step - 1) // steps_per_epoch) + 1)
                        await _record_pause_checkpoint(
                            experiment_id=experiment_id,
                            project_id=int(exp.project_id),
                            output_dir=Path(exp.output_dir) if exp.output_dir else _experiment_dir(exp.project_id, experiment_id),
                            step=step,
                            epoch=epoch_num,
                            train_loss=round(current_loss, 4),
                            eval_loss=None,
                        )
                        await broadcast_event(
                            experiment_id,
                            {"type": "status", "status": "paused", "step": step},
                        )
                        return

            # Simulate realistic loss curve
            current_loss = current_loss * 0.995 + (0.01 * 0.5) + random.uniform(-0.05, 0.05)

            epoch_float = step / steps_per_epoch
            eval_loss = round(current_loss + 0.15 + random.uniform(-0.02, 0.02), 4) if step % 20 == 0 else None

            metric = {
                "experiment_id": experiment_id,
                "epoch": round(epoch_float, 2),
                "step": step,
                "train_loss": round(current_loss, 4),
                "eval_loss": eval_loss,
                "learning_rate": lr,
                "gpu_utilization": round(random.uniform(92.0, 98.0), 1),
                "eta_seconds": int((total_steps - step) * 0.1),
            }

            await broadcast_metric(experiment_id, metric)

            should_capture_vibe = (
                project_id is not None
                and experiment_output_dir is not None
                and vibe_enabled
                and (step % vibe_interval_steps == 0 or step == total_steps)
            )
            if should_capture_vibe:
                try:
                    vibe_payload = await capture_vibe_check_snapshot(
                        project_id=project_id,
                        experiment_id=experiment_id,
                        output_dir=experiment_output_dir,
                        step=step,
                        total_steps=total_steps,
                        base_model=base_model,
                        epoch=round(epoch_float, 2),
                        train_loss=float(metric.get("train_loss") or 0.0),
                        eval_loss=float(eval_loss) if isinstance(eval_loss, (float, int)) else None,
                        experiment_config=experiment_config_snapshot,
                    )
                    snapshot = vibe_payload.get("snapshot")
                    if isinstance(snapshot, dict):
                        await broadcast_event(
                            experiment_id,
                            {
                                "type": "vibe_check",
                                "snapshot": snapshot,
                                "timeline_path": vibe_payload.get("timeline_path"),
                                "snapshot_count": vibe_payload.get("snapshot_count"),
                            },
                        )
                except Exception:
                    # Keep simulated training robust even if vibe capture fails.
                    pass

            if project_id is not None and (step % 25 == 0 or eval_loss is not None):
                from app.services.training_telemetry_service import record_observability_event

                layer_gradients = []
                for idx in range(6):
                    layer_gradients.append(
                        {
                            "layer": f"transformer.layers.{idx}",
                            "grad_norm": round(random.uniform(0.15, 1.8) * (1.0 + (idx * 0.07)), 6),
                            "weight_norm": round(random.uniform(10.0, 42.0), 6),
                            "update_ratio": round(random.uniform(0.0002, 0.0065), 8),
                        }
                    )
                if step % 120 == 0:
                    layer_gradients.append(
                        {
                            "layer": "transformer.layers.5",
                            "grad_norm": round(random.uniform(5.1, 8.0), 6),
                            "weight_norm": round(random.uniform(10.0, 42.0), 6),
                            "update_ratio": round(random.uniform(0.0050, 0.0100), 8),
                        }
                    )
                attention_focus = [
                    {
                        "token": "domain_fact",
                        "weight": round(random.uniform(0.18, 0.55), 6),
                        "source": "context",
                    },
                    {
                        "token": "retrieval_anchor",
                        "weight": round(random.uniform(0.12, 0.40), 6),
                        "source": "retrieval",
                    },
                ]
                if step % 150 == 0:
                    attention_focus.append(
                        {
                            "token": "unknown_entity",
                            "weight": round(random.uniform(0.35, 0.7), 6),
                            "source": "out_of_context",
                        }
                    )

                record_observability_event(
                    project_id,
                    payload={
                        "experiment_id": experiment_id,
                        "step": step,
                        "epoch": round(epoch_float, 2),
                        "split": "eval" if eval_loss is not None else "train",
                        "layer_gradients": layer_gradients,
                        "attention_focus": attention_focus,
                        "notes": "simulate_runtime_observability",
                    },
                )

            # Periodic checkpoint
            if step % save_steps == 0 or step == total_steps:
                async with async_session_factory() as db:
                    exp_result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
                    exp = exp_result.scalar_one_or_none()
                    if not exp:
                        continue

                    output_dir = Path(exp.output_dir) if exp.output_dir else _experiment_dir(exp.project_id, exp.id)
                    checkpoints_dir = output_dir / "checkpoints"
                    checkpoints_dir.mkdir(parents=True, exist_ok=True)
                    epoch_num = max(1, ((step - 1) // steps_per_epoch) + 1)
                    checkpoint_file = checkpoints_dir / f"checkpoint-step-{step}.json"
                    checkpoint_payload = {
                        "experiment_id": experiment_id,
                        "epoch": epoch_num,
                        "step": step,
                        "train_loss": metric["train_loss"],
                        "eval_loss": eval_loss or round(metric["train_loss"] + 0.15, 4),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    checkpoint_file.write_text(
                        json.dumps(checkpoint_payload, indent=2),
                        encoding="utf-8",
                    )

                    ckpt = Checkpoint(
                        experiment_id=experiment_id,
                        epoch=epoch_num,
                        step=step,
                        train_loss=metric["train_loss"],
                        eval_loss=eval_loss or metric["train_loss"] + 0.15,
                        file_path=str(checkpoint_file),
                        metrics=checkpoint_payload,
                    )
                    db.add(ckpt)
                    await db.commit()

        # Finish experiment
        async with async_session_factory() as db:
            result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
            exp = result.scalar_one_or_none()
            if exp:
                exp.status = ExperimentStatus.COMPLETED
                exp.completed_at = datetime.now(timezone.utc)
                exp.final_train_loss = round(current_loss, 4)
                project_id_for_event = int(exp.project_id)
                exp_name_for_event = str(exp.name or "")
                final_loss_for_event = round(current_loss, 4)
                # Phase 9b — same auto-RAG index build hook as the
                # external runtime. Best-effort; result lands on
                # runtime_config for the UI to surface.
                cfg = dict(exp.config or {})
                runtime = dict(cfg.get("_runtime") or {})
                runtime["auto_rag_build"] = await _safe_build_auto_rag_index(
                    db, project_id=int(exp.project_id)
                )
                cfg["_runtime"] = runtime
                exp.config = cfg
                await db.commit()
            else:
                project_id_for_event = None
                exp_name_for_event = ""
                final_loss_for_event = None
        if project_id_for_event is not None:
            try:
                from app.models.run_event import (
                    SEVERITY_INFO,
                    STAGE_TRAINING,
                )
                from app.services.run_event_service import emit_event

                async with async_session_factory() as event_db:
                    await emit_event(
                        event_db,
                        project_id=project_id_for_event,
                        run_id=f"exp-{experiment_id}",
                        stage=STAGE_TRAINING,
                        severity=SEVERITY_INFO,
                        summary=f"Training completed: {exp_name_for_event}",
                        payload={
                            "experiment_id": experiment_id,
                            "backend": "simulate",
                            "final_train_loss": final_loss_for_event,
                        },
                    )
                    await event_db.commit()
            except Exception as event_exc:
                print(
                    f"[run_event] training_completed_emit_failed "
                    f"experiment_id={experiment_id}: {event_exc}",
                    flush=True,
                )
    except Exception as e:
        async with async_session_factory() as db:
            result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
            exp = result.scalar_one_or_none()
            if exp:
                exp.status = ExperimentStatus.FAILED
                exp.completed_at = datetime.now(timezone.utc)
                cfg = dict(exp.config or {})
                cfg["_runtime"] = {
                    "backend": "simulate",
                    "error": str(e),
                }
                exp.config = cfg
                project_id_for_event = int(exp.project_id)
                exp_name_for_event = str(exp.name or "")
                await db.commit()
            else:
                project_id_for_event = None
                exp_name_for_event = ""
        if project_id_for_event is not None:
            try:
                from app.models.run_event import (
                    SEVERITY_ERROR,
                    STAGE_TRAINING,
                )
                from app.services.run_event_service import emit_event

                async with async_session_factory() as event_db:
                    await emit_event(
                        event_db,
                        project_id=project_id_for_event,
                        run_id=f"exp-{experiment_id}",
                        stage=STAGE_TRAINING,
                        severity=SEVERITY_ERROR,
                        reason_code="training_runtime_error",
                        summary=f"Training failed: {exp_name_for_event}",
                        payload={
                            "experiment_id": experiment_id,
                            "backend": "simulate",
                            "error": str(e),
                        },
                    )
                    await event_db.commit()
            except Exception as event_exc:
                print(
                    f"[run_event] training_failed_emit_failed "
                    f"experiment_id={experiment_id}: {event_exc}",
                    flush=True,
                )


async def start_training(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> dict:
    """Start training using configured runtime backend."""
    exp = await _get_experiment_for_project(db, project_id, experiment_id)
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")
    if exp.status == ExperimentStatus.RUNNING:
        raise ValueError(f"Experiment {experiment_id} is already running")
    if exp.status == ExperimentStatus.COMPLETED:
        raise ValueError(f"Experiment {experiment_id} is already completed")

    resolved_config = dict(exp.config or {})
    resolved_config.setdefault("base_model", exp.base_model)
    runtime_id, runtime_source = resolve_training_runtime_id(resolved_config)
    runtime_spec = get_runtime_spec(runtime_id)
    runtime_validation_errors = validate_runtime(runtime_id)
    if runtime_validation_errors:
        if len(runtime_validation_errors) == 1:
            raise ValueError(runtime_validation_errors[0])
        preview = "; ".join(runtime_validation_errors[:3])
        if len(runtime_validation_errors) > 3:
            preview = f"{preview}; (+{len(runtime_validation_errors) - 3} more)"
        raise ValueError(preview)

    resolved_config["training_runtime_id"] = runtime_id
    exp.config = resolved_config
    preflight = run_training_preflight(
        project_id=project_id,
        config=resolved_config,
        base_model=exp.base_model,
    )
    if not bool(preflight.get("ok", False)):
        preflight_errors = [str(item) for item in preflight.get("errors", []) if str(item).strip()]
        if not preflight_errors:
            preflight_errors = ["unknown preflight error"]
        preview = "; ".join(preflight_errors[:3])
        if len(preflight_errors) > 3:
            preview = f"{preview}; (+{len(preflight_errors) - 3} more)"
        raise ValueError(f"Training preflight failed: {preview}")

    epochs = int((exp.config or {}).get("num_epochs", 3))
    steps_per_epoch = 100

    # Track 1, Epic B — resolve a pre-fine-tuned warm-start checkpoint when the
    # recipe/config recommends one and its weights are present + compatible;
    # otherwise this is a no-op that returns exp.base_model unchanged.
    warm_start = resolve_starting_checkpoint(
        base_model=exp.base_model,
        recommended_checkpoint=str(
            resolved_config.get("recommended_starting_checkpoint") or ""
        ).strip()
        or None,
    )
    effective_base_model = str(warm_start.get("effective_base_model") or exp.base_model)

    message = ""
    task_id: str | None = None
    runtime_config: dict[str, object] = {
        "runtime_id": runtime_id,
        "runtime_source": runtime_source,
        "runtime_label": runtime_spec.label,
        "execution_backend": runtime_spec.execution_backend,
        "preflight": preflight,
        "warm_start": warm_start,
    }
    vibe_runtime_config = load_project_vibe_check_config(
        project_id,
        experiment_config=resolved_config,
    )
    runtime_config["vibe_check"] = {
        "enabled": bool(vibe_runtime_config.get("enabled")),
        "interval_steps": int(vibe_runtime_config.get("interval_steps") or 50),
        "prompt_count": len(list(vibe_runtime_config.get("prompts") or [])),
        "provider": str(vibe_runtime_config.get("provider") or "mock"),
        "model_name": str(vibe_runtime_config.get("model_name") or exp.base_model or ""),
        "api_url": str(vibe_runtime_config.get("api_url") or ""),
    }
    output_dir = Path(exp.output_dir) if exp.output_dir else _experiment_dir(project_id, experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "training_config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(exp.config or {}, indent=2), encoding="utf-8")
    prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    train_file = prepared_dir / "train.jsonl"
    val_file = prepared_dir / "val.jsonl"
    training_mode = str(resolved_config.get("training_mode") or exp.training_mode.value).strip().lower()

    if training_mode in {"dpo", "orpo"}:
        runtime_config["alignment_mode"] = training_mode
        custom_alignment_path = resolve_alignment_dataset_path(
            project_id,
            str(resolved_config.get("alignment_dataset_path") or "").strip(),
        )
        if custom_alignment_path is not None:
            if not custom_alignment_path.exists():
                raise ValueError(f"Configured alignment_dataset_path not found: {custom_alignment_path}")
            train_file = custom_alignment_path
            runtime_config["alignment_dataset_path"] = str(train_file)
        elif _coerce_bool(resolved_config.get("alignment_auto_filter"), False):
            quality_threshold = _coerce_float(resolved_config.get("alignment_quality_threshold"))
            min_keep_ratio = _coerce_float(resolved_config.get("alignment_min_keep_ratio"))
            filter_report = filter_preference_dataset_by_quality(
                project_id,
                quality_threshold=quality_threshold if quality_threshold is not None else 3.0,
                min_keep_ratio=min_keep_ratio if min_keep_ratio is not None else 0.4,
                apply_to_train_file=False,
                source_path=None,
                target_path=None,
            )
            train_file = Path(str(filter_report.get("target_path") or train_file))
            runtime_config["alignment_filter"] = {
                "source_path": filter_report.get("source_path"),
                "target_path": str(train_file),
                "quality_threshold": filter_report.get("quality_threshold"),
                "min_keep_ratio": filter_report.get("min_keep_ratio"),
                "keep_ratio": filter_report.get("keep_ratio"),
                "scored_count": filter_report.get("scored_count"),
                "keep_count": filter_report.get("keep_count"),
                "drop_count": filter_report.get("drop_count"),
                "average_quality_score": filter_report.get("average_quality_score"),
                "filter_report_path": filter_report.get("filter_report_path"),
            }

        include_playground_feedback = _coerce_bool(
            resolved_config.get("alignment_include_playground_feedback"),
            True,
        )
        feedback_pairs_float = _coerce_float(resolved_config.get("alignment_playground_max_pairs"))
        feedback_pair_cap = int(feedback_pairs_float) if feedback_pairs_float is not None else 5000
        feedback_pair_cap = max(1, min(feedback_pair_cap, 50000))
        if include_playground_feedback and custom_alignment_path is None:
            active_learning_report = compose_alignment_training_dataset(
                project_id,
                source_path=str(train_file),
                include_playground_pairs=True,
                target_path=None,
                max_playground_pairs=feedback_pair_cap,
            )
            effective_train_path = str(active_learning_report.get("effective_train_path") or "").strip()
            if effective_train_path:
                train_file = Path(effective_train_path)
            runtime_config["active_learning_feedback"] = {
                **active_learning_report,
                "enabled": True,
            }
        else:
            runtime_config["active_learning_feedback"] = {
                "enabled": bool(include_playground_feedback),
                "skipped": custom_alignment_path is not None or not include_playground_feedback,
                "reason": (
                    "custom_alignment_dataset_path"
                    if custom_alignment_path is not None
                    else "disabled_in_config"
                ),
            }

    # ── Curriculum learning (Phase 6b, opt-in) ──────────────────────
    # When ``config.curriculum`` is true, override ``train_file`` with
    # an easy-first ordered shard and signal ``train.py`` to disable
    # the Trainer's default RandomSampler. Recipe-gated to
    # classification today (Phase 6a only ships the prototype_entropy
    # scoring mode); other recipes get a "skipped" reason recorded in
    # runtime_config and fall through to uniform training. The whole
    # block is a no-op when ``curriculum`` is unset / false, so
    # existing training paths are unchanged.
    curriculum_block = await _maybe_apply_curriculum(
        db=db,
        project_id=project_id,
        train_file=train_file,
        output_dir=output_dir,
        resolved_config=resolved_config,
        training_mode=training_mode,
    )
    runtime_config["curriculum"] = curriculum_block
    if curriculum_block.get("applied"):
        # The shard path replaces train_file for the rest of the
        # dispatch (data-shape gate, runtime launch). The original
        # train.jsonl is preserved; the shard lives under output_dir
        # so eval / debugging can read it later.
        train_file = Path(str(curriculum_block["shard_path"]))

    # ── Data-shape gate ─────────────────────────────────────────────
    # Refuse to launch an SFT run on domain-pretrain-shape data
    # (text-only rows with no answer/completion/output/response). The
    # trainer would silently fall back to causal-LM continuation and
    # the user would discover eval F1=0% only after burning hours of
    # GPU time. See training_data_gate.py for the failure incident
    # this gate was added for. The gate is a no-op for DPO/ORPO
    # (alignment path has its own contract checks) and DOMAIN_PRETRAIN
    # (text-only is the point).
    data_gate_report = verify_training_data_has_targets(
        train_file,
        training_mode=training_mode,
        target_fields=DEFAULT_TARGET_FIELDS,
    )
    runtime_config["training_data_gate"] = data_gate_report
    if not data_gate_report["ok"]:
        raise ValueError(
            f"Training data gate failed: {data_gate_report['message']}"
        )

    exp.status = ExperimentStatus.RUNNING
    exp.started_at = datetime.now(timezone.utc)
    exp.completed_at = None
    exp.total_epochs = epochs
    exp.total_steps = epochs * steps_per_epoch
    cfg = dict(exp.config or {})
    cfg["_runtime"] = runtime_config
    exp.config = cfg
    await db.flush()

    try:
        runtime_result = await start_runtime(
            runtime_id,
            TrainingRuntimeStartContext(
                project_id=project_id,
                experiment_id=exp.id,
                base_model=effective_base_model,
                config=resolved_config,
                output_dir=output_dir,
                config_path=config_path,
                prepared_dir=prepared_dir,
                train_file=train_file,
                val_file=val_file,
                simulate_runner=_simulate_training_loop,
            ),
        )
        runtime_updates = dict(runtime_result.runtime_updates or {})
        runtime_config.update(runtime_updates)
        task_id = runtime_result.task_id
        if task_id:
            runtime_config["task_id"] = task_id
        message = str(runtime_result.message or "Training runtime started.")
        cfg = dict(exp.config or {})
        cfg["_runtime"] = runtime_config
        exp.config = cfg
        await db.flush()
        # P14: capture an immutable manifest for this run. Best-effort —
        # any failure (missing tooling, transient DB issue) is logged into
        # the manifest's own warnings list, but must never block dispatch.
        try:
            from app.services.training_manifest_service import capture_training_manifest

            await capture_training_manifest(
                db,
                project_id=project_id,
                experiment_id=int(exp.id),
                resolved_config=resolved_config,
                artifact_ids={
                    "output_dir": str(output_dir) if output_dir else None,
                    "config_path": str(config_path) if config_path else None,
                    "prepared_dir": str(prepared_dir) if prepared_dir else None,
                    "train_file": str(train_file) if train_file else None,
                    "val_file": str(val_file) if val_file else None,
                },
            )
        except Exception as manifest_exc:
            print(
                f"[training_manifest] capture_failed experiment_id={exp.id}: {manifest_exc}",
                flush=True,
            )
        # P31: emit the training-started RunEvent on the unified timeline.
        try:
            from app.models.run_event import (
                SEVERITY_INFO,
                STAGE_TRAINING,
            )
            from app.services.run_event_service import emit_event

            await emit_event(
                db,
                project_id=project_id,
                run_id=f"exp-{int(exp.id)}",
                stage=STAGE_TRAINING,
                severity=SEVERITY_INFO,
                summary=f"Training started: {exp.name}",
                payload={
                    "experiment_id": int(exp.id),
                    "runtime_id": runtime_id,
                    "base_model": exp.base_model,
                    "training_mode": str(
                        getattr(exp, "training_mode", "") or ""
                    ),
                    "task_id": task_id,
                    "epochs": epochs,
                    "total_steps": exp.total_steps,
                },
            )
        except Exception as event_exc:
            print(
                f"[run_event] training_started_emit_failed experiment_id={exp.id}: {event_exc}",
                flush=True,
            )
    except Exception as e:
        exp.status = ExperimentStatus.FAILED
        exp.completed_at = datetime.now(timezone.utc)
        fail_cfg = dict(exp.config or {})
        fail_runtime = dict(fail_cfg.get("_runtime") or {})
        fail_runtime["backend_dispatch_error"] = str(e)
        fail_runtime["runtime_id"] = runtime_id
        fail_cfg["_runtime"] = fail_runtime
        exp.config = fail_cfg
        await db.flush()
        # P31: emit the training-failed event so the timeline + failure
        # cluster surfaces (P33) can see dispatch errors.
        try:
            from app.models.run_event import (
                SEVERITY_ERROR,
                STAGE_TRAINING,
            )
            from app.services.run_event_service import emit_event

            await emit_event(
                db,
                project_id=project_id,
                run_id=f"exp-{int(exp.id)}",
                stage=STAGE_TRAINING,
                severity=SEVERITY_ERROR,
                reason_code="training_dispatch_error",
                summary=f"Training dispatch failed: {exp.name}",
                payload={
                    "experiment_id": int(exp.id),
                    "runtime_id": runtime_id,
                    "error": str(e),
                },
            )
        except Exception as event_exc:
            print(
                f"[run_event] training_failed_emit_failed experiment_id={exp.id}: {event_exc}",
                flush=True,
            )
        raise ValueError(f"Failed to dispatch training runtime '{runtime_id}': {e}")

    return {
        "experiment_id": exp.id,
        "status": exp.status.value,
        "message": message,
        "backend": str(runtime_config.get("backend") or runtime_spec.execution_backend),
        "runtime_id": runtime_id,
        "runtime_source": runtime_source,
        "task_id": task_id,
        "config": exp.config,
    }


async def cancel_training(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> dict:
    """Best-effort cancellation for a running training experiment."""
    exp = await _get_experiment_for_project(db, project_id, experiment_id)
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")
    if exp.status != ExperimentStatus.RUNNING:
        raise ValueError(f"Experiment {experiment_id} is not running")

    cfg = dict(exp.config or {})
    runtime = dict(cfg.get("_runtime") or {})
    task_id = str(runtime.get("task_id", "")).strip()
    cancel_note = "cancel_requested"

    if task_id:
        from app.services.job_service import cancel_task

        cancel_task(task_id, terminate=True)
    else:
        cancel_note = "cancel_requested_without_task_id"

    runtime["cancel_status"] = cancel_note
    runtime["cancel_requested_at"] = datetime.now(timezone.utc).isoformat()
    cfg["_runtime"] = runtime

    exp.config = cfg
    exp.status = ExperimentStatus.CANCELLED
    exp.completed_at = datetime.now(timezone.utc)
    await db.flush()

    await broadcast_event(experiment_id, {"type": "status", "status": "cancelled"})
    return {
        "experiment_id": experiment_id,
        "status": exp.status.value,
        "task_id": task_id or None,
        "cancel_status": cancel_note,
    }


# -- P17. Pause / resume training -----------------------------------------


async def _record_pause_checkpoint(
    *,
    experiment_id: int,
    project_id: int,
    output_dir: Path,
    step: int,
    epoch: int,
    train_loss: float | None,
    eval_loss: float | None,
) -> None:
    """Persist a pause-time checkpoint and flip status to PAUSED.

    Wrapped in its own session so the runtime caller (the simulate loop or
    a runtime plugin) doesn't need an open session to call this. Idempotent
    on the same step — if a checkpoint already exists there, we reuse it.
    """
    output_dir = output_dir or _experiment_dir(project_id, experiment_id)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoints_dir / f"checkpoint-step-{int(step)}.json"
    payload = {
        "experiment_id": int(experiment_id),
        "epoch": int(epoch),
        "step": int(step),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "reason": "pause_requested",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    async with async_session_factory() as db:
        existing = (
            await db.execute(
                select(Checkpoint).where(
                    Checkpoint.experiment_id == experiment_id,
                    Checkpoint.step == int(step),
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            ckpt = Checkpoint(
                experiment_id=int(experiment_id),
                epoch=int(epoch),
                step=int(step),
                train_loss=train_loss,
                eval_loss=eval_loss,
                file_path=str(checkpoint_file),
                metrics=payload,
            )
            db.add(ckpt)

        exp_result = await db.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        )
        exp = exp_result.scalar_one_or_none()
        if exp is not None:
            cfg = dict(exp.config or {})
            pause_block = dict(cfg.get("_pause") or {})
            pause_block["paused_at_step"] = int(step)
            pause_block["paused_at_epoch"] = int(epoch)
            pause_block["paused_at"] = datetime.now(timezone.utc).isoformat()
            pause_block["checkpoint_path"] = str(checkpoint_file)
            cfg["_pause"] = pause_block
            exp.config = cfg
            exp.status = ExperimentStatus.PAUSED
            exp.pause_requested = False
        await db.commit()


async def pause_training(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> dict:
    """Mark a RUNNING experiment as pause-requested.

    The runtime polling loop is responsible for actually pausing — it
    observes ``pause_requested=True``, writes a resume-capable checkpoint,
    transitions status to PAUSED, and clears the flag. This endpoint just
    flips the request bit and returns immediately (the operator does not
    have to wait for the runtime to acknowledge).
    """
    exp = await _get_experiment_for_project(db, project_id, experiment_id)
    if not exp:
        raise ValueError("experiment_not_found")
    if exp.status != ExperimentStatus.RUNNING:
        raise ValueError("not_running")

    exp.pause_requested = True
    cfg = dict(exp.config or {})
    pause_block = dict(cfg.get("_pause") or {})
    pause_block["pause_requested_at"] = datetime.now(timezone.utc).isoformat()
    cfg["_pause"] = pause_block
    exp.config = cfg
    await db.flush()
    await db.commit()

    return {
        "experiment_id": int(experiment_id),
        "status": exp.status.value,
        "pause_requested": True,
        "pause_requested_at": pause_block["pause_requested_at"],
    }


async def resume_training(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> dict:
    """Re-dispatch a PAUSED experiment from its latest checkpoint.

    The pause-time checkpoint that ``_record_pause_checkpoint`` wrote is
    the canonical resume point. We stamp ``config._resume_from`` so the
    runtime restarts the loop at ``checkpoint_step + 1`` instead of from
    scratch, then hand off to ``start_training`` which validates +
    re-dispatches the runtime + re-captures a manifest.

    Resume targets the **same** experiment row — distinct from P16's
    ``resume_from_checkpoint`` which forks a new experiment from any
    historical run's checkpoint.
    """
    exp = await _get_experiment_for_project(db, project_id, experiment_id)
    if not exp:
        raise ValueError("experiment_not_found")
    if exp.status != ExperimentStatus.PAUSED:
        raise ValueError("not_paused")

    latest_ckpt = (
        await db.execute(
            select(Checkpoint)
            .where(Checkpoint.experiment_id == experiment_id)
            .order_by(Checkpoint.step.desc())
        )
    ).scalars().first()
    if latest_ckpt is None:
        raise ValueError("no_resume_checkpoint")

    cfg = dict(exp.config or {})
    cfg["_resume_from"] = {
        "parent_experiment_id": int(experiment_id),
        "checkpoint_step": int(latest_ckpt.step),
        "checkpoint_epoch": int(latest_ckpt.epoch),
        "checkpoint_path": str(latest_ckpt.file_path or ""),
        "reason": "resume-paused-run",
    }
    pause_block = dict(cfg.get("_pause") or {})
    resume_history = list(pause_block.get("resume_history") or [])
    resume_history.append(
        {
            "resumed_at": datetime.now(timezone.utc).isoformat(),
            "from_step": int(latest_ckpt.step),
        }
    )
    pause_block["resume_history"] = resume_history
    cfg["_pause"] = pause_block
    exp.config = cfg
    # Move back through PENDING so start_training accepts it. clear pause flag.
    exp.status = ExperimentStatus.PENDING
    exp.pause_requested = False
    exp.completed_at = None
    await db.flush()
    await db.commit()

    result = await start_training(db, project_id, experiment_id)
    result["resumed_from_step"] = int(latest_ckpt.step)
    return result


# ─────────────────────────────────────────────────────────────────────
# Story 1.5 Gate 3: stuck-RUNNING reconciliation
# ─────────────────────────────────────────────────────────────────────
#
# Training subprocess writes ``training_report.json`` with a
# ``finished_at`` field when it exits cleanly. Normally the parent
# service's monitor then flips ``experiments.status`` to COMPLETED or
# FAILED. When the parent crashes / the API server restarts mid-run /
# the websocket disconnects, that DB write-back can be lost — leaving
# the row stuck on RUNNING forever. The reconciler below fixes those
# rows on read (cheap, single-experiment) and at startup (sweep all
# stale RUNNING rows once).


_REPORT_TO_STATUS: dict[str, ExperimentStatus] = {
    "completed": ExperimentStatus.COMPLETED,
    "succeeded": ExperimentStatus.COMPLETED,
    "failed": ExperimentStatus.FAILED,
    "cancelled": ExperimentStatus.CANCELLED,
    "canceled": ExperimentStatus.CANCELLED,
}


def _load_training_report(exp: Experiment) -> dict | None:
    """Read the on-disk training_report.json for an experiment, or
    None when the file is missing / unreadable. Read-only — never
    mutates DB state itself."""
    if not exp.output_dir:
        return None
    report_path = Path(exp.output_dir) / "training_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _terminal_status_from_report(report: dict) -> ExperimentStatus | None:
    """Map a training_report.json's signals to the right ExperimentStatus.
    Returns None if the report doesn't show terminal completion."""
    finished_at = report.get("finished_at")
    if not finished_at:
        return None
    # Explicit status string wins when present.
    status_value = str(report.get("status") or "").strip().lower()
    if status_value in _REPORT_TO_STATUS:
        return _REPORT_TO_STATUS[status_value]
    # Fall back: an exit code field signals failure; otherwise assume
    # success since finished_at is set.
    exit_code = report.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return ExperimentStatus.FAILED
    # Some reports use a top-level ``error`` field instead of
    # exit_code; treat presence as failure.
    if report.get("error") or report.get("fatal_error"):
        return ExperimentStatus.FAILED
    return ExperimentStatus.COMPLETED


async def reconcile_experiment_if_stale(
    db: AsyncSession, exp: Experiment
) -> dict | None:
    """If ``exp.status`` is RUNNING but the on-disk training_report.json
    shows it's actually finished, flip the DB row to the right terminal
    status. Returns a small report dict when a flip happened, ``None``
    otherwise. Called from ``get_training_status`` on read."""
    if exp.status != ExperimentStatus.RUNNING:
        return None
    report = _load_training_report(exp)
    if report is None:
        return None
    terminal = _terminal_status_from_report(report)
    if terminal is None:
        return None

    exp.status = terminal
    if not exp.completed_at:
        finished_iso = str(report.get("finished_at") or "")
        try:
            exp.completed_at = (
                datetime.fromisoformat(finished_iso.replace("Z", "+00:00"))
                if finished_iso
                else datetime.now(timezone.utc)
            )
        except ValueError:
            exp.completed_at = datetime.now(timezone.utc)
    if (
        terminal == ExperimentStatus.COMPLETED
        and exp.final_train_loss is None
    ):
        loss_value = report.get("final_train_loss")
        if isinstance(loss_value, (int, float)):
            exp.final_train_loss = float(loss_value)
    if (
        terminal == ExperimentStatus.COMPLETED
        and exp.final_eval_loss is None
    ):
        eval_loss = report.get("final_eval_loss")
        if isinstance(eval_loss, (int, float)):
            exp.final_eval_loss = float(eval_loss)
    await db.flush()
    return {
        "experiment_id": exp.id,
        "from_status": ExperimentStatus.RUNNING.value,
        "to_status": terminal.value,
        "finished_at": report.get("finished_at"),
        "reconciled_by": "on_read",
    }


async def reconcile_stale_running_experiments(
    db: AsyncSession, *, max_age_minutes: int = 60
) -> list[dict]:
    """Sweep every experiment in RUNNING state older than ``max_age_minutes``
    and reconcile any whose on-disk report has finished. Intended to
    run once at app startup. Returns a list of per-row reports so the
    operator's log shows what was fixed."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    result = await db.execute(
        select(Experiment).where(
            Experiment.status == ExperimentStatus.RUNNING,
            Experiment.started_at < cutoff,
        )
    )
    stale = list(result.scalars().all())
    reports: list[dict] = []
    for exp in stale:
        report = await reconcile_experiment_if_stale(db, exp)
        if report is not None:
            reports.append({**report, "reconciled_by": "startup_reaper"})
    if reports:
        await db.commit()
    return reports


async def get_training_status(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> dict:
    """Get current training status and metrics."""
    exp = await _get_experiment_for_project(db, project_id, experiment_id)
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")

    # Story 1.5 Gate 3: auto-flip RUNNING → terminal when the on-disk
    # report shows the subprocess actually finished. Cheap (one JSON
    # read) and idempotent — does nothing once the row is reconciled.
    reconciliation = await reconcile_experiment_if_stale(db, exp)
    if reconciliation is not None:
        await db.commit()

    ckpt_result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.experiment_id == experiment_id)
        .order_by(Checkpoint.step.desc())
    )
    checkpoints = ckpt_result.scalars().all()

    runtime = (exp.config or {}).get("_runtime", {}) if isinstance(exp.config, dict) else {}
    task_status = None
    task_id = runtime.get("task_id") if isinstance(runtime, dict) else None
    if isinstance(task_id, str) and task_id.strip():
        from app.services.job_service import get_task_status

        task_status = get_task_status(task_id)

    return {
        "experiment_id": exp.id,
        "name": exp.name,
        "status": exp.status.value,
        "training_mode": exp.training_mode.value,
        "base_model": exp.base_model,
        "config": exp.config,
        "final_train_loss": exp.final_train_loss,
        "final_eval_loss": exp.final_eval_loss,
        "total_epochs": exp.total_epochs,
        "total_steps": exp.total_steps,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        "task_status": task_status,
        "checkpoints": [
            {
                "epoch": c.epoch,
                "step": c.step,
                "train_loss": c.train_loss,
                "eval_loss": c.eval_loss,
                "is_best": c.is_best,
            }
            for c in checkpoints
        ],
    }


async def list_experiments(
    db: AsyncSession, project_id: int
) -> list[Experiment]:
    """List all experiments for a project."""
    result = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc())
    )
    return list(result.scalars().all())

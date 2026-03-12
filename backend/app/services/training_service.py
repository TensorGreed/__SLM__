"""Training pipeline service — SFT, LoRA, checkpoint management."""

import asyncio
import json
import random
from datetime import datetime, timezone
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
from app.services.training_preflight_service import run_training_preflight
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
    project = await db.execute(select(Project).where(Project.id == project_id))
    if not project.scalar_one_or_none():
        raise ValueError(f"Project {project_id} not found")

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
        base_model = str(config.get("base_model") or "microsoft/phi-2").strip() or "microsoft/phi-2"
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

        for step in range(1, total_steps + 1):
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
                await db.commit()
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
                await db.commit()


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

    message = ""
    task_id: str | None = None
    runtime_config: dict[str, object] = {
        "runtime_id": runtime_id,
        "runtime_source": runtime_source,
        "runtime_label": runtime_spec.label,
        "execution_backend": runtime_spec.execution_backend,
        "preflight": preflight,
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
                base_model=exp.base_model,
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


async def get_training_status(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
) -> dict:
    """Get current training status and metrics."""
    exp = await _get_experiment_for_project(db, project_id, experiment_id)
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")

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

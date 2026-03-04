"""Training pipeline service — SFT, LoRA, checkpoint management."""

import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from asyncio.subprocess import PIPE

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

active_websockets: dict[int, list[WebSocket]] = {}
active_training_tasks: dict[int, asyncio.Task] = {}
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


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
    if experiment_id in active_websockets:
        dead_socks = []
        for ws in active_websockets[experiment_id]:
            try:
                await ws.send_json(metric)
            except Exception:
                dead_socks.append(ws)
        for ws in dead_socks:
            active_websockets[experiment_id].remove(ws)


def _render_external_command(template: str, placeholders: dict[str, str | int]) -> str:
    try:
        return template.format(**placeholders)
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"TRAINING_EXTERNAL_CMD missing placeholder value: {missing}")


def _coerce_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_step_from_checkpoint_dir(path: Path) -> int | None:
    suffix = path.name.removeprefix("checkpoint-")
    if not suffix.isdigit():
        return None
    return int(suffix)


async def _monitor_external_training(
    experiment_id: int,
    process: asyncio.subprocess.Process,
    command: str,
    log_path: Path,
    output_dir: Path,
) -> None:
    """Monitor external training process and sync experiment status."""
    started = datetime.now(timezone.utc)
    try:
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise ValueError(
                f"External training command timed out after {settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS} seconds"
            )
        finished = datetime.now(timezone.utc)
        log_payload = {
            "command": command,
            "returncode": process.returncode,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
            "stderr": stderr.decode("utf-8", errors="replace") if stderr else "",
        }
        log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

        async with async_session_factory() as db:
            result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
            exp = result.scalar_one_or_none()
            if not exp:
                return

            config = dict(exp.config or {})
            config["_runtime"] = {
                "backend": "external",
                "command": command,
                "log_path": str(log_path),
                "returncode": process.returncode,
            }
            report_path = output_dir / "training_report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    config["_runtime"]["report_path"] = str(report_path)
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
                    config["_runtime"]["report_parse_error"] = str(parse_error)

            exp.config = config

            if process.returncode == 0:
                exp.status = ExperimentStatus.COMPLETED
                exp.completed_at = finished
            else:
                exp.status = ExperimentStatus.FAILED
                exp.completed_at = finished
            await db.commit()
    except Exception as e:
        async with async_session_factory() as db:
            result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
            exp = result.scalar_one_or_none()
            if exp:
                exp.status = ExperimentStatus.FAILED
                exp.completed_at = datetime.now(timezone.utc)
                config = dict(exp.config or {})
                config["_runtime"] = {
                    "backend": "external",
                    "error": str(e),
                    "log_path": str(log_path),
                }
                exp.config = config
                await db.commit()
    finally:
        active_training_tasks.pop(experiment_id, None)


async def _simulate_training_loop(experiment_id: int, config: dict):
    """Simulate a training loop reporting metrics for demo purposes."""
    try:
        epochs = config.get("num_epochs", 3)
        steps_per_epoch = 100
        total_steps = epochs * steps_per_epoch

        current_loss = 3.5
        lr = config.get("learning_rate", 2e-4)
        save_steps = config.get("save_steps", 100)

        for step in range(1, total_steps + 1):
            await asyncio.sleep(0.1)  # 100ms per step to make demo fast but visible

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
    finally:
        active_training_tasks.pop(experiment_id, None)


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

    epochs = int((exp.config or {}).get("num_epochs", 3))
    steps_per_epoch = 100
    backend = settings.TRAINING_BACKEND.strip().lower()
    if backend not in {"simulate", "external"}:
        raise ValueError(f"Unsupported training backend '{settings.TRAINING_BACKEND}'")

    message = ""
    runtime_config: dict = {"backend": backend}
    external_command = ""
    output_dir = Path(exp.output_dir) if exp.output_dir else _experiment_dir(project_id, experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "training_config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(exp.config or {}, indent=2), encoding="utf-8")
    prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    train_file = prepared_dir / "train.jsonl"
    val_file = prepared_dir / "val.jsonl"

    if backend == "simulate":
        if not settings.ALLOW_SIMULATED_TRAINING:
            raise ValueError(
                "Simulated training backend is disabled. "
                "Set ALLOW_SIMULATED_TRAINING=true for demos or configure TRAINING_BACKEND=external."
            )
        message = "Simulated training started. Connecting to telemetry stream..."
    else:
        template = settings.TRAINING_EXTERNAL_CMD.strip()
        if not template:
            raise ValueError("TRAINING_EXTERNAL_CMD is required when TRAINING_BACKEND=external")
        external_command = _render_external_command(
            template,
            {
                "project_id": project_id,
                "experiment_id": exp.id,
                "output_dir": str(output_dir),
                "base_model": exp.base_model,
                "backend_dir": str(BACKEND_DIR),
                "config_path": str(config_path),
                "data_dir": str(settings.DATA_DIR),
                "prepared_dir": str(prepared_dir),
                "train_file": str(train_file),
                "val_file": str(val_file),
            },
        )
        runtime_config["command"] = external_command
        runtime_config["log_path"] = str(output_dir / "external_training.log")
        runtime_config["config_path"] = str(config_path)
        runtime_config["train_file"] = str(train_file)
        runtime_config["val_file"] = str(val_file)
        message = "External training command started."

    exp.status = ExperimentStatus.RUNNING
    exp.started_at = datetime.now(timezone.utc)
    exp.completed_at = None
    exp.total_epochs = epochs
    exp.total_steps = epochs * steps_per_epoch
    cfg = dict(exp.config or {})
    cfg["_runtime"] = runtime_config
    exp.config = cfg
    await db.flush()

    if backend == "simulate":
        task = asyncio.create_task(_simulate_training_loop(exp.id, exp.config or {}))
        active_training_tasks[exp.id] = task
    else:
        try:
            process = await asyncio.create_subprocess_shell(
                external_command,
                stdout=PIPE,
                stderr=PIPE,
                cwd=str(BACKEND_DIR),
            )
        except Exception as e:
            exp.status = ExperimentStatus.FAILED
            exp.completed_at = datetime.now(timezone.utc)
            fail_cfg = dict(exp.config or {})
            fail_cfg["_runtime"] = {
                "backend": "external",
                "command": external_command,
                "error": str(e),
            }
            exp.config = fail_cfg
            await db.flush()
            raise ValueError(f"Failed to start external training command: {e}")

        log_path = output_dir / "external_training.log"
        task = asyncio.create_task(
            _monitor_external_training(
                exp.id,
                process=process,
                command=external_command,
                log_path=log_path,
                output_dir=output_dir,
            )
        )
        active_training_tasks[exp.id] = task

    return {
        "experiment_id": exp.id,
        "status": exp.status.value,
        "message": message,
        "backend": backend,
        "config": exp.config,
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

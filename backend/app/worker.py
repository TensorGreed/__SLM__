import asyncio
import json
import logging
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from celery import Celery
import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    "slm_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,  # ML Tasks are heavy; only take 1 at a time
    task_acks_late=True,  # Don't acknowledge task until completed so it isn't lost on crash
    task_reject_on_worker_lost=True,
    broker_transport_options={"visibility_timeout": settings.CELERY_VISIBILITY_TIMEOUT_SECONDS},
    result_backend_transport_options={"visibility_timeout": settings.CELERY_VISIBILITY_TIMEOUT_SECONDS},
    visibility_timeout=settings.CELERY_VISIBILITY_TIMEOUT_SECONDS,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent

TRAINING_METRIC_PREFIX = "SLM_METRIC "
TRAINING_EVENT_PREFIX = "SLM_EVENT "


def _coerce_metric_number(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_training_metric_line(line: str, experiment_id: int) -> dict | None:
    if not isinstance(line, str):
        return None
    text = line.strip()
    if not text.startswith(TRAINING_METRIC_PREFIX):
        return None
    payload = text[len(TRAINING_METRIC_PREFIX):].strip()
    try:
        raw = json.loads(payload)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    metric = {"experiment_id": experiment_id}
    step = raw.get("step")
    if isinstance(step, int):
        metric["step"] = step
    elif isinstance(step, float):
        metric["step"] = int(step)

    epoch = _coerce_metric_number(raw.get("epoch"))
    if epoch is not None:
        metric["epoch"] = round(epoch, 4)

    train_loss = _coerce_metric_number(raw.get("train_loss"))
    if train_loss is not None:
        metric["train_loss"] = train_loss

    eval_loss = _coerce_metric_number(raw.get("eval_loss"))
    if eval_loss is not None:
        metric["eval_loss"] = eval_loss

    learning_rate = _coerce_metric_number(raw.get("learning_rate"))
    if learning_rate is not None:
        metric["learning_rate"] = learning_rate

    return metric


def _parse_training_event_line(line: str) -> dict | None:
    if not isinstance(line, str):
        return None
    text = line.strip()
    if not text.startswith(TRAINING_EVENT_PREFIX):
        return None
    payload = text[len(TRAINING_EVENT_PREFIX):].strip()
    try:
        raw = json.loads(payload)
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None

# ── Celery Task Definitions ────────────

@celery_app.task(bind=True, name="run_workflow_node_job", track_started=True)
def run_workflow_node_job(
    self,
    project_id: int,
    run_id: int,
    node_id: str,
    stage: str,
    step_type: str,
    attempt: int,
    config: dict | None = None,
):
    """Execute one workflow DAG node attempt via Celery worker."""
    cfg = dict(config or {})
    fail_stages = {
        str(item).strip()
        for item in cfg.get("simulate_fail_stages", [])
        if isinstance(item, str) and str(item).strip()
    }
    fail_once_stages = {
        str(item).strip()
        for item in cfg.get("simulate_fail_once_stages", [])
        if isinstance(item, str) and str(item).strip()
    }

    if stage in fail_stages:
        return {
            "success": False,
            "log": {
                "message": f"simulated failure for stage '{stage}'",
                "task_id": self.request.id,
            },
            "error": "simulated stage failure",
        }
    if stage in fail_once_stages and int(attempt) == 1:
        return {
            "success": False,
            "log": {
                "message": f"simulated one-time failure for stage '{stage}'",
                "task_id": self.request.id,
            },
            "error": "simulated one-time failure",
        }

    executable_core_steps = {
        "core.data_adapter_preview",
        "core.training",
        "core.evaluation",
        "core.export",
    }
    if step_type in executable_core_steps:
        async def _run_core_step():
            from app.database import async_session_factory
            from app.services.workflow_runner_service import execute_local_core_step_attempt

            async with async_session_factory() as db:
                return await execute_local_core_step_attempt(
                    db=db,
                    project_id=project_id,
                    node_id=node_id,
                    step_type=step_type,
                    config=cfg,
                    node_config=cfg.get("node_config"),
                )

        core_loop = asyncio.new_event_loop()
        try:
            success, log_payload, error_text = core_loop.run_until_complete(_run_core_step())
            if not isinstance(log_payload, dict):
                log_payload = {"message": str(log_payload or "")}
            log_payload.setdefault("task_id", self.request.id)
            if success:
                return {"success": True, "log": log_payload, "error": ""}
            return {
                "success": False,
                "log": log_payload,
                "error": error_text or f"{step_type} execution failed",
            }
        except Exception as exc:
            return {
                "success": False,
                "log": {
                    "message": f"celery {step_type} execution failed",
                    "task_id": self.request.id,
                },
                "error": str(exc),
            }
        finally:
            core_loop.close()

    command_template = str(cfg.get("external_command_template", "")).strip()
    timeout_seconds = max(
        1,
        int(cfg.get("external_timeout_seconds") or settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS),
    )

    if not command_template:
        return {
            "success": True,
            "log": {
                "message": f"celery backend completed {stage}",
                "task_id": self.request.id,
            },
            "error": "",
        }

    try:
        command = command_template.format(
            project_id=project_id,
            run_id=run_id,
            node_id=node_id,
            stage=stage,
            step_type=step_type,
        )
    except KeyError as exc:
        missing = str(exc.args[0]) if exc.args else "unknown"
        return {
            "success": False,
            "log": {
                "message": "external command template missing placeholder",
                "task_id": self.request.id,
            },
            "error": f"external command missing placeholder: {missing}",
        }

    proc = subprocess.run(
        shlex.split(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=str(BACKEND_DIR),
    )
    stdout_tail = str(proc.stdout or "").strip()[-2000:]
    stderr_tail = str(proc.stderr or "").strip()[-2000:]
    if proc.returncode != 0:
        return {
            "success": False,
            "log": {
                "message": f"external command failed for stage {stage}",
                "task_id": self.request.id,
                "stdout_tail": stdout_tail,
            },
            "error": stderr_tail or f"external command exited with code {proc.returncode}",
        }

    return {
        "success": True,
        "log": {
            "message": f"celery backend executed {stage}",
            "task_id": self.request.id,
            "stdout_tail": stdout_tail,
        },
        "error": "",
    }

@celery_app.task(bind=True, name="run_training_job", track_started=True)
def run_training_job(self, experiment_id: int, command: str, log_path: str, output_dir: str):
    """Executes the external training script within the Celery worker."""
    logger.info(f"Starting training job for experiment {experiment_id}")
    
    # We must run the async monitoring in a new event loop inside the celery thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from app.services.training_service import _monitor_external_training
        
        async def _run():
            channel = f"log:experiment:{experiment_id}"
            redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

            async def _publish_event(payload: dict) -> None:
                await redis_client.publish(channel, json.dumps(payload))

            async def _publish_log(text: str) -> None:
                await _publish_event({"type": "log", "text": text})

            async def _publish_metric(metric: dict) -> None:
                await _publish_event({"type": "metric", "metric": metric})

            async def _publish_status(status: str, extra: dict | None = None) -> None:
                body = {"type": "status", "status": status}
                if isinstance(extra, dict):
                    body.update(extra)
                await _publish_event(body)

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BACKEND_DIR),
            )
            started_at = datetime.now(timezone.utc)

            try:
                await _publish_log(
                    f"[worker] starting external training process for experiment {experiment_id}"
                )

                stdout_lines: list[str] = []
                stderr_lines: list[str] = []

                async def _stream_reader(stream, sink: list[str], is_stderr: bool) -> None:
                    if stream is None:
                        return
                    while True:
                        raw = await stream.readline()
                        if not raw:
                            break
                        line = raw.decode("utf-8", errors="replace").rstrip()
                        sink.append(line)
                        metric = _parse_training_metric_line(line, experiment_id)
                        if metric is not None:
                            await _publish_metric(metric)
                            continue

                        stream_event = _parse_training_event_line(line)
                        if isinstance(stream_event, dict):
                            event_type = str(stream_event.get("event", "")).strip().lower()
                            if event_type == "epoch_end":
                                epoch_value = stream_event.get("epoch")
                                await _publish_log(
                                    f"[event] epoch {epoch_value} completed"
                                    if epoch_value is not None
                                    else "[event] epoch completed"
                                )
                                continue

                        text = f"[ERR] {line}" if is_stderr else line
                        if text:
                            await _publish_log(text)

                stdout_task = asyncio.create_task(
                    _stream_reader(process.stdout, stdout_lines, False)
                )
                stderr_task = asyncio.create_task(
                    _stream_reader(process.stderr, stderr_lines, True)
                )

                timeout_seconds = int(settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS)
                timed_out = False
                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
                except TimeoutError:
                    timed_out = True
                    process.kill()
                    await process.wait()

                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

                finished_at = datetime.now(timezone.utc)
                monitor_status = await _monitor_external_training(
                    experiment_id,
                    process,
                    command,
                    Path(log_path),
                    Path(output_dir),
                    captured_stdout="\n".join(stdout_lines),
                    captured_stderr="\n".join(stderr_lines),
                    started_at=started_at,
                    finished_at=finished_at,
                )

                if timed_out:
                    await _publish_log(
                        (
                            "[worker] external training command timed out after "
                            f"{timeout_seconds} seconds"
                        )
                    )
                    await _publish_status("failed", {"error": "external command timeout"})
                else:
                    await _publish_status(
                        str(monitor_status or ("completed" if process.returncode == 0 else "failed")),
                        {"returncode": process.returncode},
                    )
                await _publish_log(
                    f"[worker] training process exited with code {process.returncode}"
                )

                payload = {}
                lp = Path(log_path)
                if lp.exists():
                    try:
                        payload = json.loads(lp.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {}
                return {
                    "returncode": process.returncode,
                    "payload": payload,
                }
            finally:
                await redis_client.aclose()
            
        execution = loop.run_until_complete(_run())
        returncode = int(execution.get("returncode", -1))
        
        if returncode == 0:
            return {"status": "success", "experiment_id": experiment_id}
        else:
            payload = execution.get("payload", {}) if isinstance(execution, dict) else {}
            error_tail = ""
            for key in ("stdout", "stderr"):
                text = str(payload.get(key, ""))
                for line in reversed(text.splitlines()):
                    stripped = line.strip()
                    if stripped:
                        error_tail = stripped
                        break
                if error_tail:
                    break
            if error_tail:
                if len(error_tail) > 320:
                    error_tail = f"{error_tail[:320].rstrip()}..."
                raise Exception(f"Training command failed with return code {returncode}: {error_tail}")
            raise Exception(f"Training command failed with return code {returncode}")
            
    except Exception as e:
        logger.error(f"Training job {experiment_id} failed: {e}")
        raise
    finally:
        loop.close()

@celery_app.task(bind=True, name="run_quantization_job", track_started=True)
def run_quantization_job(self, command: str, report_path: str, project_id: int | None = None, job_type: str = "quantize"):
    """Executes the external quantization script within the Celery worker."""
    logger.info(f"Starting quantization job: {command}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from app.services.compression_service import _run_external_command
        
        async def _run():
            channel = f"log:compression:project:{project_id}" if project_id is not None else "log:compression:project:0"
            redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

            async def _publish(line: str) -> None:
                await redis_client.publish(channel, line)

            try:
                await _publish(f"[worker] starting {job_type} job")
                execution = await _run_external_command(command, cwd=BACKEND_DIR)
                Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                Path(report_path).write_text(json.dumps(execution, indent=2), encoding="utf-8")
                for line in str(execution.get("stdout", "")).splitlines():
                    if line.strip():
                        await _publish(line.strip())
                for line in str(execution.get("stderr", "")).splitlines():
                    if line.strip():
                        await _publish(f"[ERR] {line.strip()}")
                await _publish(f"[worker] {job_type} job finished with code {execution.get('returncode')}")
                return execution
            finally:
                await redis_client.aclose()
            
        execution = loop.run_until_complete(_run())
        if execution["returncode"] != 0:
            raise Exception(f"Quantization failed (exit {execution['returncode']})")
            
        return {"status": "success", "report_path": report_path}
    except Exception as e:
        logger.error(f"Quantization job failed: {e}")
        raise
    finally:
        loop.close()

@celery_app.task(bind=True, name="run_benchmark_job", track_started=True)
def run_benchmark_job(
    self,
    command: str,
    report_path: str,
    benchmark_output_path: str | None = None,
    project_id: int | None = None,
):
    """Executes the external benchmark script within the Celery worker."""
    logger.info(f"Starting benchmark job: {command}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from app.services.compression_service import _run_external_command
        
        async def _run():
            channel = f"log:compression:project:{project_id}" if project_id is not None else "log:compression:project:0"
            redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

            async def _publish(line: str) -> None:
                await redis_client.publish(channel, line)

            try:
                await _publish("[worker] starting benchmark job")
                execution = await _run_external_command(command, cwd=BACKEND_DIR)
                benchmark_payload = None
                if benchmark_output_path:
                    output_file = Path(benchmark_output_path)
                    if output_file.exists():
                        try:
                            benchmark_payload = json.loads(output_file.read_text(encoding="utf-8"))
                        except Exception:
                            benchmark_payload = None
                if benchmark_output_path:
                    execution["benchmark_report_path"] = benchmark_output_path
                if benchmark_payload is not None:
                    execution["benchmark"] = benchmark_payload
                Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                Path(report_path).write_text(json.dumps(execution, indent=2), encoding="utf-8")
                for line in str(execution.get("stdout", "")).splitlines():
                    if line.strip():
                        await _publish(line.strip())
                for line in str(execution.get("stderr", "")).splitlines():
                    if line.strip():
                        await _publish(f"[ERR] {line.strip()}")
                await _publish(f"[worker] benchmark job finished with code {execution.get('returncode')}")
                return execution
            finally:
                await redis_client.aclose()
            
        execution = loop.run_until_complete(_run())
        if execution["returncode"] != 0:
            raise Exception(f"Benchmark failed (exit {execution['returncode']})")
            
        return {"status": "success", "report_path": report_path}
    except Exception as e:
        logger.error(f"Benchmark job failed: {e}")
        raise
    finally:
        loop.close()


@celery_app.task(bind=True, name="run_remote_import_job", track_started=True)
def run_remote_import_job(
    self,
    project_id: int,
    request_payload: dict,
    report_path: str,
):
    """Execute remote dataset import in worker and publish progress logs."""
    source_type = str(request_payload.get("source_type", "unknown"))
    identifier = str(request_payload.get("identifier", ""))
    logger.info("Starting remote import job project=%s source=%s id=%s", project_id, source_type, identifier)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        from app.database import async_session_factory
        from app.services.ingestion_service import ingest_remote_dataset

        async def _run():
            channel = f"log:ingestion:project:{project_id}"
            redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

            async def _publish(line: str) -> None:
                await redis_client.publish(channel, line)

            safe_request = {
                "source_type": source_type,
                "identifier": identifier,
                "split": str(request_payload.get("split", "train")),
                "max_samples": request_payload.get("max_samples"),
                "config_name": request_payload.get("config_name"),
                "adapter_id": str(request_payload.get("adapter_id", "default-canonical")),
                "adapter_config": request_payload.get("adapter_config") or {},
                "normalize_for_training": bool(request_payload.get("normalize_for_training", True)),
            }
            started_at = datetime.now(timezone.utc).isoformat()

            try:
                await _publish(
                    f"[worker] starting remote import source={source_type} identifier={identifier}"
                )
                async with async_session_factory() as db:
                    result = await ingest_remote_dataset(
                        db=db,
                        project_id=project_id,
                        source_type=source_type,
                        identifier=identifier,
                        split=str(request_payload.get("split", "train")),
                        max_samples=request_payload.get("max_samples"),
                        config_name=request_payload.get("config_name"),
                        field_mapping=request_payload.get("field_mapping") or None,
                        adapter_id=str(request_payload.get("adapter_id", "default-canonical")),
                        adapter_config=request_payload.get("adapter_config") or None,
                        normalize_for_training=bool(request_payload.get("normalize_for_training", True)),
                        hf_token=str(request_payload.get("hf_token", "")).strip() or None,
                        kaggle_username=str(request_payload.get("kaggle_username", "")).strip() or None,
                        kaggle_key=str(request_payload.get("kaggle_key", "")).strip() or None,
                        use_saved_secrets=bool(request_payload.get("use_saved_secrets", True)),
                        progress_callback=_publish,
                    )
                    await db.commit()

                finished_at = datetime.now(timezone.utc).isoformat()
                payload = {
                    "status": "completed",
                    "project_id": project_id,
                    "source_type": source_type,
                    "identifier": identifier,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "request": safe_request,
                    "result": result,
                }
                rp = Path(report_path)
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                await _publish(
                    f"[worker] remote import completed: samples={result.get('samples_ingested', 0)}"
                )
                return payload
            except Exception as e:
                finished_at = datetime.now(timezone.utc).isoformat()
                payload = {
                    "status": "failed",
                    "project_id": project_id,
                    "source_type": source_type,
                    "identifier": identifier,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "request": safe_request,
                    "error": str(e),
                }
                rp = Path(report_path)
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                await _publish(f"[ERR] {e}")
                raise
            finally:
                await redis_client.aclose()

        return loop.run_until_complete(_run())
    except Exception as e:
        logger.error("Remote import job failed for project %s: %s", project_id, e)
        raise
    finally:
        loop.close()

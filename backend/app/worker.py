import asyncio
import json
import logging
from pathlib import Path

from celery import Celery
import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    "slm_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
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
)

BACKEND_DIR = Path(__file__).resolve().parent.parent

# ── Celery Task Definitions ────────────

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

            async def _publish(line: str) -> None:
                await redis_client.publish(channel, line)

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BACKEND_DIR),
            )

            try:
                await _publish(f"[worker] starting external training process for experiment {experiment_id}")
                await _monitor_external_training(
                    experiment_id, process, command, Path(log_path), Path(output_dir)
                )

                payload = {}
                lp = Path(log_path)
                if lp.exists():
                    try:
                        payload = json.loads(lp.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {}

                for line in str(payload.get("stdout", "")).splitlines():
                    if line.strip():
                        await _publish(line.strip())
                for line in str(payload.get("stderr", "")).splitlines():
                    if line.strip():
                        await _publish(f"[ERR] {line.strip()}")

                await _publish(f"[worker] training process exited with code {process.returncode}")
                return process.returncode
            finally:
                await redis_client.aclose()
            
        returncode = loop.run_until_complete(_run())
        
        if returncode == 0:
            return {"status": "success", "experiment_id": experiment_id}
        else:
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

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
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BACKEND_DIR),
            )
            
            async def _stream_logs(stream, is_error=False):
                redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
                channel = f"log:experiment:{experiment_id}"
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode("utf-8").rstrip()
                    prefix = "[ERR] " if is_error else ""
                    await redis_client.publish(channel, f"{prefix}{text}")
                await redis_client.aclose()
                
            # Run stream monitoring concurrently with the existing monitor
            await asyncio.gather(
                _stream_logs(process.stdout),
                _stream_logs(process.stderr, is_error=True),
                _monitor_external_training(
                    experiment_id, process, command, Path(log_path), Path(output_dir)
                )
            )
            return process.returncode
            
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
def run_quantization_job(self, command: str, report_path: str):
    """Executes the external quantization script within the Celery worker."""
    logger.info(f"Starting quantization job: {command}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from app.services.compression_service import _run_external_command
        
        async def _run():
            
            async def _stream_logs(stream, is_error=False):
                redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
                channel = "log:compression:quantize" # broadcast for compression
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode("utf-8").rstrip()
                    prefix = "[ERR] " if is_error else ""
                    await redis_client.publish(channel, f"{prefix}{text}")
                await redis_client.aclose()
                
            # Note: _run_external_command internally consumes stdout/stderr. 
            # For live streaming, we actually need to bypass that utility or modify it. 
            # Since _run_external_command uses asyncio.wait_for(process.communicate()),
            # we can't easily hook into streams here without rewriting the utility.
            # Instead, we will rely on the static report_path for now to avoid breaking the utility,
            # or we could rewrite the utility if live compression logs are strictly required.
            
            execution = await _run_external_command(command, cwd=BACKEND_DIR)
            Path(report_path).write_text(json.dumps(execution, indent=2), encoding="utf-8")
            return execution
            
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
def run_benchmark_job(self, command: str, report_path: str):
    """Executes the external benchmark script within the Celery worker."""
    logger.info(f"Starting benchmark job: {command}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from app.services.compression_service import _run_external_command
        
        async def _run():
            execution = await _run_external_command(command, cwd=BACKEND_DIR)
            Path(report_path).write_text(json.dumps(execution, indent=2), encoding="utf-8")
            return execution
            
        execution = loop.run_until_complete(_run())
        if execution["returncode"] != 0:
            raise Exception(f"Benchmark failed (exit {execution['returncode']})")
            
        return {"status": "success", "report_path": report_path}
    except Exception as e:
        logger.error(f"Benchmark job failed: {e}")
        raise
    finally:
        loop.close()

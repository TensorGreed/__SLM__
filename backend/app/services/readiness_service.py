import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.database import async_session_factory
from app.models.project import Project
from sqlalchemy import select

async def get_project_readiness(project_id: int) -> Dict[str, Any]:
    """Validate GPU, dependencies, paths, and secrets for a project."""
    
    # Check project existence
    async with async_session_factory() as db:
        project = await db.get(Project, project_id)
        if not project:
            return {"status": "error", "message": f"Project {project_id} not found."}

    checks = []
    
    # 1. GPU Check
    gpu_check = {"id": "gpu", "name": "GPU Availability", "status": "pass", "message": "GPU detected", "type": "blocker"}
    try:
        # Check if nvidia-smi is available
        if shutil.which("nvidia-smi"):
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_check["message"] = f"Detected: {result.stdout.strip()}"
            else:
                gpu_check["status"] = "warn"
                gpu_check["message"] = "nvidia-smi failed; GPU may not be fully functional."
        else:
            # Try torch if available (though it might not be in the backend process)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_check["message"] = f"Torch detected {torch.cuda.get_device_name(0)}"
                else:
                    gpu_check["status"] = "warn"
                    gpu_check["message"] = "No NVIDIA GPU detected via nvidia-smi or torch."
            except ImportError:
                gpu_check["status"] = "warn"
                gpu_check["message"] = "No NVIDIA GPU detected (nvidia-smi missing)."
    except Exception as e:
        gpu_check["status"] = "fail"
        gpu_check["message"] = f"Error checking GPU: {str(e)}"
    
    if settings.STRICT_EXECUTION_MODE and gpu_check["status"] == "warn":
        gpu_check["status"] = "fail"
        gpu_check["message"] += " (Blocked by STRICT_EXECUTION_MODE)"
    
    checks.append(gpu_check)

    # 2. Dependencies Check
    deps_check = {"id": "deps", "name": "Critical Dependencies", "status": "pass", "message": "All critical packages found", "type": "blocker"}
    critical_packages = ["fastapi", "sqlalchemy", "pydantic", "alembic"]
    # For training/compression, we might need more
    if settings.TRAINING_BACKEND == "external":
        critical_packages.extend(["celery", "redis"])
    
    missing = []
    for pkg in critical_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        deps_check["status"] = "fail"
        deps_check["message"] = f"Missing packages: {', '.join(missing)}"
        deps_check["fix"] = f"Run 'pip install {' '.join(missing)}'"
    
    checks.append(deps_check)

    # 3. Paths Check
    paths_check = {"id": "paths", "name": "Filesystem Paths", "status": "pass", "message": "Required directories exist", "type": "blocker"}
    required_paths = [settings.DATA_DIR, settings.MODEL_CACHE_DIR]
    missing_paths = [str(p) for p in required_paths if not p.exists()]
    
    if missing_paths:
        paths_check["status"] = "fail"
        paths_check["message"] = f"Missing directories: {', '.join(missing_paths)}"
        paths_check["fix"] = "The application should create these on startup. Check permissions."
    
    checks.append(paths_check)

    # 4. Secrets Check
    secrets_check = {"id": "secrets", "name": "API Secrets", "status": "pass", "message": "Required secrets configured", "type": "warning"}
    missing_secrets = []
    if not settings.TEACHER_MODEL_API_KEY and not settings.ALLOW_SYNTHETIC_DEMO_FALLBACK:
        missing_secrets.append("TEACHER_MODEL_API_KEY")
    
    if missing_secrets:
        secrets_check["status"] = "fail" if settings.STRICT_EXECUTION_MODE else "warn"
        secrets_check["message"] = f"Missing secrets: {', '.join(missing_secrets)}"
        secrets_check["fix"] = "Set the missing environment variables or update settings in the UI."

    checks.append(secrets_check)

    # Overall Status
    overall_status = "pass"
    if any(c["status"] == "fail" for c in checks):
        overall_status = "fail"
    elif any(c["status"] == "warn" for c in checks):
        overall_status = "warn"

    return {
        "project_id": project_id,
        "status": overall_status,
        "strict_mode": settings.STRICT_EXECUTION_MODE,
        "checks": checks
    }

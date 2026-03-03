"""Export & runtime packaging service."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.export import Export, ExportFormat, ExportStatus


def _export_dir(project_id: int, export_id: int) -> Path:
    d = settings.DATA_DIR / "exports" / str(project_id) / str(export_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


async def create_export(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    export_format: ExportFormat,
    quantization: str | None = None,
) -> Export:
    """Create a new export record."""
    export = Export(
        project_id=project_id,
        experiment_id=experiment_id,
        export_format=export_format,
        quantization=quantization,
        status=ExportStatus.PENDING,
    )
    db.add(export)
    await db.flush()
    await db.refresh(export)

    output_dir = _export_dir(project_id, export.id)
    export.output_path = str(output_dir)
    await db.flush()

    return export


async def run_export(
    db: AsyncSession,
    export_id: int,
    eval_report: dict | None = None,
    safety_scorecard: dict | None = None,
) -> Export:
    """Execute the export process."""
    result = await db.execute(select(Export).where(Export.id == export_id))
    export = result.scalar_one_or_none()
    if not export:
        raise ValueError(f"Export {export_id} not found")

    export.status = ExportStatus.IN_PROGRESS
    await db.flush()

    try:
        output_dir = Path(export.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate version manifest
        manifest = {
            "export_id": export.id,
            "project_id": export.project_id,
            "experiment_id": export.experiment_id,
            "format": export.export_format.value,
            "quantization": export.quantization,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "platform_version": settings.APP_VERSION,
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Save eval report
        if eval_report:
            export.eval_report = eval_report
            with open(output_dir / "eval_report.json", "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2)

        # Save safety scorecard
        if safety_scorecard:
            export.safety_scorecard = safety_scorecard
            with open(output_dir / "safety_scorecard.json", "w", encoding="utf-8") as f:
                json.dump(safety_scorecard, f, indent=2)

        # Generate Dockerfile template
        dockerfile_content = _generate_dockerfile(export.export_format)
        with open(output_dir / "Dockerfile", "w", encoding="utf-8") as f:
            f.write(dockerfile_content)

        # Generate inference script template
        inference_script = _generate_inference_script(export.export_format)
        with open(output_dir / "serve.py", "w", encoding="utf-8") as f:
            f.write(inference_script)

        export.status = ExportStatus.COMPLETED
        export.manifest = manifest
        export.completed_at = datetime.now(timezone.utc)

    except Exception as e:
        export.status = ExportStatus.FAILED
        export.manifest = {"error": str(e)}

    await db.flush()
    await db.refresh(export)
    return export


async def list_exports(
    db: AsyncSession, project_id: int
) -> list[Export]:
    """List all exports for a project."""
    result = await db.execute(
        select(Export)
        .where(Export.project_id == project_id)
        .order_by(Export.created_at.desc())
    )
    return list(result.scalars().all())


def _generate_dockerfile(export_format: ExportFormat) -> str:
    """Generate a Dockerfile for the inference server."""
    if export_format == ExportFormat.GGUF:
        return """FROM python:3.11-slim
WORKDIR /app
RUN pip install llama-cpp-python fastapi uvicorn
COPY . .
EXPOSE 8080
CMD ["python", "serve.py"]
"""
    elif export_format == ExportFormat.HUGGINGFACE:
        return """FROM python:3.11-slim
WORKDIR /app
RUN pip install torch transformers accelerate fastapi uvicorn
COPY . .
EXPOSE 8080
CMD ["python", "serve.py"]
"""
    else:
        return """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "serve.py"]
"""


def _generate_inference_script(export_format: ExportFormat) -> str:
    """Generate a minimal inference server script."""
    return '''"""Auto-generated inference server for exported SLM model."""
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SLM Inference Server")

# Load model on startup
# TODO: Replace with actual model loading based on export format

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    text: str
    tokens_used: int

@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    # TODO: Replace with actual inference
    return InferenceResponse(text="[Model not loaded]", tokens_used=0)

@app.get("/health")
async def health():
    with open("manifest.json") as f:
        manifest = json.load(f)
    return {"status": "ok", "manifest": manifest}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

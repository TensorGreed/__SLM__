"""Export & runtime packaging service."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.experiment import Experiment
from app.models.export import Export, ExportFormat, ExportStatus


def _export_dir(project_id: int, export_id: int) -> Path:
    d = settings.DATA_DIR / "exports" / str(project_id) / str(export_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256_file(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _artifact_manifest_entry(file_path: Path, root_dir: Path) -> dict:
    return {
        "path": str(file_path.relative_to(root_dir)),
        "size_bytes": file_path.stat().st_size,
        "sha256": _sha256_file(file_path),
    }


def _project_dir(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id)


def _sanitize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _matches_quantization(file_name: str, quantization: str | None) -> bool:
    if not quantization:
        return True
    quant_norm = _sanitize_token(quantization)
    name_norm = _sanitize_token(file_name)
    if not quant_norm:
        return True
    if quant_norm in name_norm:
        return True

    digits = "".join(ch for ch in quant_norm if ch.isdigit())
    if digits:
        for token in (f"{digits}bit", f"q{digits}", f"int{digits}"):
            if token in name_norm:
                return True
    return False


def _collect_compressed_files(
    project_id: int,
    export_format: ExportFormat,
    quantization: str | None,
) -> list[Path]:
    compressed_dir = _project_dir(project_id) / "compressed"
    if not compressed_dir.exists():
        return []

    extension_map = {
        ExportFormat.GGUF: {".gguf"},
        ExportFormat.ONNX: {".onnx"},
        ExportFormat.TENSORRT: {".engine", ".plan"},
    }
    allowed_ext = extension_map.get(export_format)

    files = [p for p in compressed_dir.rglob("*") if p.is_file()]
    if allowed_ext:
        files = [p for p in files if p.suffix.lower() in allowed_ext]
    else:
        files = [
            p for p in files
            if not p.name.endswith("_report.json")
            and p.name not in {"benchmark_results.json", "benchmark_report.json"}
        ]

    filtered = [p for p in files if _matches_quantization(p.name, quantization)]
    return sorted(filtered, key=lambda p: p.stat().st_mtime, reverse=True)


def _copy_files_preserve_structure(
    files: list[Path],
    dest_dir: Path,
    source_root: Path,
) -> list[Path]:
    copied: list[Path] = []
    for src in files:
        target = dest_dir / src.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(target)
    return copied


def _copy_files(files: list[Path], dest_dir: Path) -> list[Path]:
    copied: list[Path] = []
    used_names: set[str] = set()
    for src in files:
        base_name = src.name
        stem = src.stem
        suffix = src.suffix
        candidate = base_name
        idx = 1
        while candidate in used_names or (dest_dir / candidate).exists():
            candidate = f"{stem}_{idx}{suffix}"
            idx += 1
        target = dest_dir / candidate
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(target)
        used_names.add(candidate)
    return copied


def _resolve_source_model_files(
    project_id: int,
    experiment: Experiment,
    export_format: ExportFormat,
    quantization: str | None,
) -> tuple[str, list[Path], Path | None]:
    use_compressed = export_format in {ExportFormat.GGUF, ExportFormat.ONNX, ExportFormat.TENSORRT} or bool(quantization)
    if use_compressed:
        compressed_files = _collect_compressed_files(project_id, export_format, quantization)
        if compressed_files:
            return "compressed", compressed_files, None

    output_dir = Path(experiment.output_dir).expanduser() if experiment.output_dir else None
    if output_dir and output_dir.exists():
        model_dir = output_dir / "model"
        if model_dir.exists() and model_dir.is_dir():
            model_files = [p for p in model_dir.rglob("*") if p.is_file()]
            if model_files:
                return "experiment_model_dir", model_files, model_dir

        direct_model_files = [
            p for p in output_dir.rglob("*")
            if p.is_file() and p.name not in {"training_config.json", "training_report.json", "external_training.log"}
        ]
        if direct_model_files:
            return "experiment_output_dir", direct_model_files, output_dir

    return "none", [], None


async def create_export(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    export_format: ExportFormat,
    quantization: str | None = None,
) -> Export:
    """Create a new export record."""
    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    if not exp_result.scalar_one_or_none():
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")

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
    project_id: int,
    export_id: int,
    eval_report: dict | None = None,
    safety_scorecard: dict | None = None,
) -> Export:
    """Execute the export process."""
    result = await db.execute(
        select(Export).where(
            Export.id == export_id,
            Export.project_id == project_id,
        )
    )
    export = result.scalar_one_or_none()
    if not export:
        raise ValueError(f"Export {export_id} not found in project {project_id}")

    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == export.experiment_id,
            Experiment.project_id == project_id,
        )
    )
    experiment = exp_result.scalar_one_or_none()
    if not experiment:
        raise ValueError(f"Experiment {export.experiment_id} not found in project {project_id}")

    export.status = ExportStatus.IN_PROGRESS
    await db.flush()

    try:
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%SZ")

        output_dir = Path(export.output_path or _export_dir(project_id, export.id))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / f"run-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict] = []

        # Copy actual model artifacts into export package.
        model_source, source_files, source_root = _resolve_source_model_files(
            project_id=project_id,
            experiment=experiment,
            export_format=export.export_format,
            quantization=export.quantization,
        )
        if not source_files:
            raise ValueError(
                "No model artifacts found for export. "
                "Complete training (and compression for quantized formats) before exporting."
            )

        model_dir = run_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        if source_root is not None:
            copied_model_files = _copy_files_preserve_structure(source_files, model_dir, source_root)
        else:
            copied_model_files = _copy_files(source_files, model_dir)
        for copied in copied_model_files:
            artifacts.append(_artifact_manifest_entry(copied, run_dir))
        model_size_bytes = sum(p.stat().st_size for p in copied_model_files)

        # Save eval report
        if eval_report:
            export.eval_report = eval_report
            eval_path = run_dir / "eval_report.json"
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2)
            artifacts.append(_artifact_manifest_entry(eval_path, run_dir))

        # Save safety scorecard
        if safety_scorecard:
            export.safety_scorecard = safety_scorecard
            safety_path = run_dir / "safety_scorecard.json"
            with open(safety_path, "w", encoding="utf-8") as f:
                json.dump(safety_scorecard, f, indent=2)
            artifacts.append(_artifact_manifest_entry(safety_path, run_dir))

        # Generate Dockerfile template
        dockerfile_content = _generate_dockerfile(export.export_format)
        dockerfile_path = run_dir / "Dockerfile"
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        artifacts.append(_artifact_manifest_entry(dockerfile_path, run_dir))

        # Generate inference script template
        inference_script = _generate_inference_script(export.export_format)
        serve_path = run_dir / "serve.py"
        with open(serve_path, "w", encoding="utf-8") as f:
            f.write(inference_script)
        artifacts.append(_artifact_manifest_entry(serve_path, run_dir))

        release_notes_path = run_dir / "RELEASE_NOTES.md"
        release_notes_path.write_text(
            (
                f"# Export Release {run_id}\n\n"
                f"- Export ID: {export.id}\n"
                f"- Project ID: {project_id}\n"
                f"- Experiment ID: {experiment.id}\n"
                f"- Format: {export.export_format.value}\n"
                f"- Quantization: {export.quantization or 'none'}\n"
                f"- Model source: {model_source}\n"
                f"- Model artifacts: {len(copied_model_files)} files\n"
                f"- Model bytes: {model_size_bytes}\n"
                f"- Generated at: {now.isoformat()}\n"
                f"- Platform version: {settings.APP_VERSION}\n"
            ),
            encoding="utf-8",
        )
        artifacts.append(_artifact_manifest_entry(release_notes_path, run_dir))

        # Generate version manifest
        manifest = {
            "run_id": run_id,
            "export_id": export.id,
            "project_id": export.project_id,
            "experiment_id": export.experiment_id,
            "format": export.export_format.value,
            "quantization": export.quantization,
            "created_at": now.isoformat(),
            "platform_version": settings.APP_VERSION,
            "run_dir": str(run_dir),
            "experiment": {
                "name": experiment.name,
                "status": experiment.status.value,
                "training_mode": experiment.training_mode.value,
                "base_model": experiment.base_model,
                "final_train_loss": experiment.final_train_loss,
                "final_eval_loss": experiment.final_eval_loss,
            },
            "model_artifacts": {
                "source": model_source,
                "count": len(copied_model_files),
                "size_bytes": model_size_bytes,
                "path": "model",
            },
            "artifacts": artifacts,
        }

        run_manifest_path = run_dir / "manifest.json"
        with open(run_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Keep root manifest as the latest export run for backward compatibility.
        root_manifest_path = output_dir / "manifest.json"
        with open(root_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        (output_dir / "LATEST_RUN").write_text(run_id, encoding="utf-8")

        export.status = ExportStatus.COMPLETED
        export.manifest = manifest
        export.output_path = str(output_dir)
        export.file_size_bytes = model_size_bytes
        export.completed_at = now

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
    """Generate an inference server script tailored to export format."""
    if export_format in {ExportFormat.HUGGINGFACE, ExportFormat.DOCKER}:
        return '''"""Auto-generated inference server for exported SLM model."""
import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SLM Inference Server")
MODEL_PATH = os.getenv("MODEL_PATH", "./model")
model = None
tokenizer = None
load_error = None


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    text: str
    tokens_used: int


@app.on_event("startup")
async def startup():
    global model, tokenizer, load_error
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
    except Exception as e:
        load_error = str(e)


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {load_error}")
    import torch

    inputs = tokenizer(req.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
        )
    prompt_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = output_ids[0][prompt_tokens:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return InferenceResponse(text=text, tokens_used=int(generated_tokens.shape[-1]))


@app.get("/health")
async def health():
    with open("manifest.json") as f:
        manifest = json.load(f)
    return {
        "status": "ok" if model is not None else "degraded",
        "format": "huggingface",
        "model_path": MODEL_PATH,
        "load_error": load_error,
        "manifest": manifest,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

    if export_format == ExportFormat.GGUF:
        return '''"""Auto-generated GGUF inference server."""
import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SLM Inference Server")
MODEL_PATH = os.getenv("MODEL_PATH", "./model.gguf")
llm = None
load_error = None


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    text: str
    tokens_used: int


@app.on_event("startup")
async def startup():
    global llm, load_error
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=MODEL_PATH, n_ctx=4096)
    except Exception as e:
        load_error = str(e)


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {load_error}")
    out = llm(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    choice = out.get("choices", [{}])[0]
    usage = out.get("usage", {})
    return InferenceResponse(
        text=choice.get("text", ""),
        tokens_used=int(usage.get("total_tokens", 0)),
    )


@app.get("/health")
async def health():
    with open("manifest.json") as f:
        manifest = json.load(f)
    return {
        "status": "ok" if llm is not None else "degraded",
        "format": "gguf",
        "model_path": MODEL_PATH,
        "load_error": load_error,
        "manifest": manifest,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

    return '''"""Auto-generated inference server (manual runtime integration required)."""
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SLM Inference Server")


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    text: str
    tokens_used: int


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    raise HTTPException(
        status_code=501,
        detail="Inference runtime not auto-generated for this export format. Integrate your runtime engine in serve.py.",
    )


@app.get("/health")
async def health():
    with open("manifest.json") as f:
        manifest = json.load(f)
    return {"status": "degraded", "manifest": manifest}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

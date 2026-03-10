"""Local serving plan generation for export and registry artifacts."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.export import Export, ExportFormat, ExportStatus
from app.models.registry import ModelRegistryEntry
from app.services.deployment_target_service import (
    default_deployment_targets_for_format,
    resolve_artifact_profile,
)


def _normalize_host(value: str | None) -> str:
    token = str(value or "").strip()
    return token or "127.0.0.1"


def _normalize_port(value: int | None) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        return 8000
    if parsed > 65535:
        return 65535
    return parsed


def _resolve_export_run_dir(export: Export) -> Path | None:
    manifest = export.manifest if isinstance(export.manifest, dict) else {}
    run_dir = manifest.get("run_dir")
    if isinstance(run_dir, str) and run_dir.strip():
        candidate = Path(run_dir).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate

    output_path = Path(export.output_path).expanduser() if export.output_path else None
    if output_path is None or not output_path.exists():
        return None

    latest_run = output_path / "LATEST_RUN"
    if latest_run.exists():
        token = latest_run.read_text(encoding="utf-8").strip()
        if token:
            candidate = output_path / f"run-{token}"
            if candidate.exists() and candidate.is_dir():
                return candidate
    return None


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _format_curl_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True)


def _build_builtin_template(
    *,
    run_dir: Path,
    export_format: str,
    host: str,
    port: int,
    prompt: str,
) -> dict[str, Any]:
    model_dir = run_dir / "model"
    model_path = "./model"
    if export_format == ExportFormat.GGUF.value:
        gguf_files = sorted(model_dir.glob("*.gguf"))
        if gguf_files:
            model_path = f"./model/{gguf_files[0].name}"
    command = f"cd {shlex.quote(str(run_dir))} && MODEL_PATH={shlex.quote(model_path)} python serve.py"
    health_url = f"http://{host}:{port}/health"
    generate_url = f"http://{host}:{port}/generate"
    smoke_payload = {
        "prompt": str(prompt or "Hello from SLM!"),
        "max_tokens": 64,
        "temperature": 0.2,
    }
    return {
        "template_id": "builtin.fastapi",
        "display_name": "Built-in FastAPI Server",
        "kind": "runner",
        "target_id": "runner.builtin_fastapi",
        "description": "Use the generated serve.py that ships with the export bundle.",
        "command": command,
        "healthcheck": {
            "url": health_url,
            "curl": f"curl -sS {shlex.quote(health_url)}",
        },
        "smoke_test": {
            "prompt": smoke_payload["prompt"],
            "curl": (
                f"curl -sS -X POST {shlex.quote(generate_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(smoke_payload)}'"
            ),
        },
        "notes": [
            f"Run from export directory: {run_dir}",
            "Default server port in generated script is 8080; adjust script or environment if needed.",
        ],
    }


def _build_vllm_template(
    *,
    run_dir: Path,
    host: str,
    port: int,
    prompt: str,
) -> dict[str, Any]:
    model_dir = run_dir / "model"
    command = _format_cmd(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_dir),
            "--host",
            host,
            "--port",
            str(port),
        ]
    )
    health_url = f"http://{host}:{port}/health"
    chat_url = f"http://{host}:{port}/v1/chat/completions"
    smoke_payload = {
        "model": model_dir.name or "local-slm",
        "messages": [{"role": "user", "content": str(prompt or "Hello from SLM!")}],
        "temperature": 0.2,
        "max_tokens": 64,
        "stream": False,
    }
    return {
        "template_id": "runner.vllm",
        "display_name": "vLLM OpenAI Server",
        "kind": "runner",
        "target_id": "runner.vllm",
        "description": "Run OpenAI-compatible local server with vLLM.",
        "command": command,
        "healthcheck": {
            "url": health_url,
            "curl": f"curl -sS {shlex.quote(health_url)}",
        },
        "smoke_test": {
            "prompt": smoke_payload["messages"][0]["content"],
            "curl": (
                f"curl -sS -X POST {shlex.quote(chat_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(smoke_payload)}'"
            ),
        },
        "notes": [
            "Install vLLM in the active environment before launching.",
            f"Model path is resolved to: {model_dir}",
        ],
    }


def _build_tgi_template(
    *,
    run_dir: Path,
    host: str,
    port: int,
    prompt: str,
) -> dict[str, Any]:
    model_dir = run_dir / "model"
    command = _format_cmd(
        [
            "text-generation-launcher",
            "--model-id",
            str(model_dir),
            "--hostname",
            host,
            "--port",
            str(port),
        ]
    )
    health_url = f"http://{host}:{port}/health"
    generate_url = f"http://{host}:{port}/generate"
    smoke_payload = {
        "inputs": str(prompt or "Hello from SLM!"),
        "parameters": {
            "max_new_tokens": 64,
            "temperature": 0.2,
        },
    }
    return {
        "template_id": "runner.tgi",
        "display_name": "TGI Runner",
        "kind": "runner",
        "target_id": "runner.tgi",
        "description": "Launch Text Generation Inference against exported HF artifacts.",
        "command": command,
        "healthcheck": {
            "url": health_url,
            "curl": f"curl -sS {shlex.quote(health_url)}",
        },
        "smoke_test": {
            "prompt": smoke_payload["inputs"],
            "curl": (
                f"curl -sS -X POST {shlex.quote(generate_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(smoke_payload)}'"
            ),
        },
        "notes": [
            "Install `text-generation-launcher` or use Docker image `ghcr.io/huggingface/text-generation-inference`.",
        ],
    }


def _build_ollama_template(
    *,
    run_dir: Path,
    host: str,
    port: int,
    prompt: str,
    model_alias: str,
) -> dict[str, Any]:
    model_dir = run_dir / "model"
    gguf_files = sorted(model_dir.glob("*.gguf"))
    gguf_name = gguf_files[0].name if gguf_files else "<model.gguf>"
    modelfile_cmd = (
        "cat > Modelfile <<'EOF'\n"
        f"FROM ./model/{gguf_name}\n"
        "EOF"
    )
    create_cmd = _format_cmd(["ollama", "create", model_alias, "-f", "Modelfile"])
    serve_cmd = f"OLLAMA_HOST={host}:{port} ollama serve"
    health_url = f"http://{host}:{port}/api/tags"
    smoke_url = f"http://{host}:{port}/api/generate"
    smoke_payload = {
        "model": model_alias,
        "prompt": str(prompt or "Hello from SLM!"),
        "stream": False,
    }
    return {
        "template_id": "runner.ollama",
        "display_name": "Ollama Runner",
        "kind": "runner",
        "target_id": "runner.ollama",
        "description": "Create a local Ollama model from GGUF export and serve via Ollama API.",
        "command": f"cd {shlex.quote(str(run_dir))} && {serve_cmd}",
        "setup_commands": [
            f"cd {shlex.quote(str(run_dir))}",
            modelfile_cmd,
            create_cmd,
        ],
        "healthcheck": {
            "url": health_url,
            "curl": f"curl -sS {shlex.quote(health_url)}",
        },
        "smoke_test": {
            "prompt": smoke_payload["prompt"],
            "curl": (
                f"curl -sS -X POST {shlex.quote(smoke_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(smoke_payload)}'"
            ),
        },
        "notes": [
            "Run setup_commands once before first serve.",
            f"GGUF file used in Modelfile: {gguf_name}",
        ],
    }


def _candidate_runner_ids(
    *,
    export_format: str,
    selected_target_ids: list[str] | None,
) -> list[str]:
    selected = [str(item).strip().lower() for item in list(selected_target_ids or []) if str(item).strip()]
    if not selected:
        selected = default_deployment_targets_for_format(export_format)
    runner_ids = [item for item in selected if item.startswith("runner.")]
    if export_format == ExportFormat.HUGGINGFACE.value and not runner_ids:
        runner_ids = ["runner.vllm", "runner.tgi"]
    if export_format == ExportFormat.DOCKER.value and not runner_ids:
        runner_ids = ["runner.vllm", "runner.tgi"]
    if export_format == ExportFormat.GGUF.value and not runner_ids:
        runner_ids = ["runner.ollama"]
    return sorted(set(runner_ids))


def build_local_serve_templates(
    *,
    run_dir: Path,
    export_format: str,
    host: str,
    port: int,
    prompt: str,
    selected_target_ids: list[str] | None = None,
    export_id: int | None = None,
) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    templates.append(
        _build_builtin_template(
            run_dir=run_dir,
            export_format=export_format,
            host=host,
            port=port,
            prompt=prompt,
        )
    )

    runner_ids = _candidate_runner_ids(
        export_format=export_format,
        selected_target_ids=selected_target_ids,
    )
    model_alias = f"slm-export-{export_id}" if export_id is not None else "slm-export"
    for runner_id in runner_ids:
        if runner_id == "runner.vllm":
            templates.append(
                _build_vllm_template(
                    run_dir=run_dir,
                    host=host,
                    port=port,
                    prompt=prompt,
                )
            )
        elif runner_id == "runner.tgi":
            templates.append(
                _build_tgi_template(
                    run_dir=run_dir,
                    host=host,
                    port=port,
                    prompt=prompt,
                )
            )
        elif runner_id == "runner.ollama":
            templates.append(
                _build_ollama_template(
                    run_dir=run_dir,
                    host=host,
                    port=port,
                    prompt=prompt,
                    model_alias=model_alias,
                )
            )

    return templates


async def build_export_serve_plan(
    db: AsyncSession,
    *,
    project_id: int,
    export_id: int,
    host: str = "127.0.0.1",
    port: int = 8000,
    smoke_test_prompt: str = "Hello from SLM!",
    target_ids: list[str] | None = None,
) -> dict[str, Any]:
    row = await db.execute(
        select(Export).where(
            Export.id == export_id,
            Export.project_id == project_id,
        )
    )
    export = row.scalar_one_or_none()
    if export is None:
        raise ValueError(f"Export {export_id} not found in project {project_id}")
    if export.status not in {ExportStatus.COMPLETED, ExportStatus.FAILED}:
        raise ValueError("Export is not completed yet. Run export before generating serve plan.")

    run_dir = _resolve_export_run_dir(export)
    if run_dir is None:
        raise ValueError("Export run directory not found. Run export before generating serve plan.")

    normalized_host = _normalize_host(host)
    normalized_port = _normalize_port(port)
    format_token = export.export_format.value
    manifest = export.manifest if isinstance(export.manifest, dict) else {}
    deployment = manifest.get("deployment") if isinstance(manifest.get("deployment"), dict) else {}
    selected_target_ids = target_ids
    if selected_target_ids is None:
        selected_target_ids = list(deployment.get("selected_target_ids") or [])

    templates = build_local_serve_templates(
        run_dir=run_dir,
        export_format=format_token,
        host=normalized_host,
        port=normalized_port,
        prompt=str(smoke_test_prompt or "Hello from SLM!"),
        selected_target_ids=selected_target_ids,
        export_id=export.id,
    )

    return {
        "source": "export",
        "project_id": project_id,
        "export_id": export.id,
        "experiment_id": export.experiment_id,
        "export_status": export.status.value,
        "export_format": format_token,
        "artifact_profile": resolve_artifact_profile(format_token),
        "run_dir": str(run_dir),
        "model_dir": str(run_dir / "model"),
        "host": normalized_host,
        "port": normalized_port,
        "smoke_test_prompt": str(smoke_test_prompt or "Hello from SLM!"),
        "selected_target_ids": selected_target_ids or default_deployment_targets_for_format(format_token),
        "deployment_summary": (
            deployment.get("summary")
            if isinstance(deployment.get("summary"), dict)
            else {}
        ),
        "templates": templates,
    }


async def build_registry_serve_plan(
    db: AsyncSession,
    *,
    project_id: int,
    model_id: int,
    host: str = "127.0.0.1",
    port: int = 8000,
    smoke_test_prompt: str = "Hello from SLM!",
    target_ids: list[str] | None = None,
) -> dict[str, Any]:
    row = await db.execute(
        select(ModelRegistryEntry).where(
            ModelRegistryEntry.id == model_id,
            ModelRegistryEntry.project_id == project_id,
        )
    )
    entry = row.scalar_one_or_none()
    if entry is None:
        raise ValueError(f"Registry model {model_id} not found in project {project_id}")
    if entry.export_id is None:
        raise ValueError(
            "Registry model has no linked export. Register model with export_id to generate serve plan."
        )

    export_plan = await build_export_serve_plan(
        db,
        project_id=project_id,
        export_id=int(entry.export_id),
        host=host,
        port=port,
        smoke_test_prompt=smoke_test_prompt,
        target_ids=target_ids,
    )
    export_plan.update(
        {
            "source": "registry",
            "model_id": entry.id,
            "model_name": entry.name,
            "model_version": entry.version,
            "model_stage": entry.stage.value,
            "model_deployment_status": entry.deployment_status.value,
        }
    )
    return export_plan

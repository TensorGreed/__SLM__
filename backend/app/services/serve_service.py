"""Local serving plan generation for export and registry artifacts."""

from __future__ import annotations

import asyncio
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
    from app.services.export_service import _resolve_export_run_dir_from_export

    return _resolve_export_run_dir_from_export(export)


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
            "method": "GET",
            "expected_status": 200,
            "curl": f"curl -sS {shlex.quote(health_url)}",
        },
        "smoke_test": {
            "prompt": smoke_payload["prompt"],
            "url": generate_url,
            "method": "POST",
            "json_body": smoke_payload,
            "expected_status": 200,
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
        "launch_spec": {
            "cwd": str(run_dir),
            "argv": ["python", "serve.py"],
            "env": {"MODEL_PATH": model_path},
        },
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
    first_token_payload = {
        "model": smoke_payload["model"],
        "messages": smoke_payload["messages"],
        "temperature": 0.2,
        "max_tokens": 64,
        "stream": True,
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
            "method": "GET",
            "expected_status": 200,
            "curl": f"curl -sS {shlex.quote(health_url)}",
        },
        "smoke_test": {
            "prompt": smoke_payload["messages"][0]["content"],
            "url": chat_url,
            "method": "POST",
            "json_body": smoke_payload,
            "expected_status": 200,
            "curl": (
                f"curl -sS -X POST {shlex.quote(chat_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(smoke_payload)}'"
            ),
        },
        "first_token_probe": {
            "url": chat_url,
            "method": "POST",
            "parser": "openai_sse",
            "json_body": first_token_payload,
            "expected_status": 200,
            "curl": (
                f"curl -sS -N -X POST {shlex.quote(chat_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(first_token_payload)}'"
            ),
        },
        "notes": [
            "Install vLLM in the active environment before launching.",
            f"Model path is resolved to: {model_dir}",
            "First-token latency probe uses streaming /v1/chat/completions.",
        ],
        "launch_spec": {
            "cwd": str(run_dir),
            "argv": [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(model_dir),
                "--host",
                host,
                "--port",
                str(port),
            ],
            "env": {},
        },
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
    first_token_url = f"http://{host}:{port}/generate_stream"
    first_token_payload = {
        "inputs": smoke_payload["inputs"],
        "parameters": smoke_payload["parameters"],
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
            "method": "GET",
            "expected_status": 200,
            "curl": f"curl -sS {shlex.quote(health_url)}",
        },
        "smoke_test": {
            "prompt": smoke_payload["inputs"],
            "url": generate_url,
            "method": "POST",
            "json_body": smoke_payload,
            "expected_status": 200,
            "curl": (
                f"curl -sS -X POST {shlex.quote(generate_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(smoke_payload)}'"
            ),
        },
        "first_token_probe": {
            "url": first_token_url,
            "method": "POST",
            "parser": "tgi_sse",
            "json_body": first_token_payload,
            "expected_status": 200,
            "curl": (
                f"curl -sS -N -X POST {shlex.quote(first_token_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(first_token_payload)}'"
            ),
        },
        "notes": [
            "Install `text-generation-launcher` or use Docker image `ghcr.io/huggingface/text-generation-inference`.",
            "First-token latency probe uses `/generate_stream` SSE response.",
        ],
        "launch_spec": {
            "cwd": str(run_dir),
            "argv": [
                "text-generation-launcher",
                "--model-id",
                str(model_dir),
                "--hostname",
                host,
                "--port",
                str(port),
            ],
            "env": {},
        },
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
    first_token_payload = {
        "model": model_alias,
        "prompt": str(prompt or "Hello from SLM!"),
        "stream": True,
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
            "method": "GET",
            "expected_status": 200,
            "curl": f"curl -sS {shlex.quote(health_url)}",
        },
        "smoke_test": {
            "prompt": smoke_payload["prompt"],
            "url": smoke_url,
            "method": "POST",
            "json_body": smoke_payload,
            "expected_status": 200,
            "curl": (
                f"curl -sS -X POST {shlex.quote(smoke_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(smoke_payload)}'"
            ),
        },
        "first_token_probe": {
            "url": smoke_url,
            "method": "POST",
            "parser": "ollama_jsonl",
            "json_body": first_token_payload,
            "expected_status": 200,
            "curl": (
                f"curl -sS -N -X POST {shlex.quote(smoke_url)} "
                "-H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(first_token_payload)}'"
            ),
        },
        "notes": [
            "Run setup_commands once before first serve.",
            f"GGUF file used in Modelfile: {gguf_name}",
            "First-token latency probe uses streamed `/api/generate` output.",
        ],
        "launch_spec": {
            "cwd": str(run_dir),
            "argv": ["ollama", "serve"],
            "env": {"OLLAMA_HOST": f"{host}:{port}"},
        },
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


def select_serve_template(plan: dict[str, Any], template_id: str) -> dict[str, Any]:
    requested = str(template_id or "").strip().lower()
    templates = [
        item
        for item in list(plan.get("templates") or [])
        if isinstance(item, dict)
    ]
    if not requested:
        raise ValueError("template_id is required.")
    selected = next(
        (
            item
            for item in templates
            if str(item.get("template_id") or "").strip().lower() == requested
        ),
        None,
    )
    if selected is None:
        available = ", ".join(sorted(str(item.get("template_id") or "") for item in templates))
        raise ValueError(
            f"Unknown serve template '{template_id}'. Available templates: {available}"
        )
    return selected


# ---------------------------------------------------------------------------
# Multi-LoRA plan (G11): one vLLM server, one base model, many adapters
# ---------------------------------------------------------------------------

_LORA_RANKS = (8, 16, 32, 64, 128, 256)


def _round_lora_rank(rank: int) -> int:
    for step in _LORA_RANKS:
        if rank <= step:
            return step
    return _LORA_RANKS[-1]


async def build_multi_lora_serve_plan(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_ids: list[int],
    host: str = "127.0.0.1",
    port: int = 8001,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    """A vLLM launch template serving several LoRA runs of ONE base model.

    Every run must be COMPLETED, belong to the project and be a LoRA adapter;
    all must share a base model (``ValueError`` listing each run's base
    otherwise). Module names equal the gateway model ids
    (``<slug>:run-<id>``), the base is served as ``<slug>:base``, so a gateway
    ``openai_compatible`` upstream needs no model map.
    """
    from app.models.experiment import Experiment, ExperimentStatus
    from app.models.project import Project
    from app.services.adapter_merge_service import adapter_base_model
    from app.services.gateway_service import project_slug
    from app.services import servable_run_service as srs

    ids = list(dict.fromkeys(int(i) for i in experiment_ids or []))
    if not ids:
        raise ValueError("experiment_ids must list at least one run.")
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found.")
    slug = await project_slug(db, project)
    host = _normalize_host(host)
    port = _normalize_port(port or 8001)

    adapters: list[dict[str, Any]] = []
    bases: dict[int, str] = {}
    for run_id in ids:
        exp = await db.get(Experiment, run_id)
        if exp is None or exp.project_id != project_id:
            raise ValueError(f"Run #{run_id} not found in this project.")
        if exp.status != ExperimentStatus.COMPLETED:
            raise ValueError(f"Run #{run_id} is {exp.status.value}; only completed runs can be served.")
        # Same "can it answer" check the gateway uses (not a baseline, weights on disk);
        # the disk reads run off the event loop.
        problem = await asyncio.to_thread(srs.run_problem, exp)
        if problem:
            raise ValueError(problem)
        model_dir = srs.run_model_dir(exp)
        cfg = await asyncio.to_thread(srs.run_adapter_config, exp)
        if cfg is None:
            raise ValueError(f"Run #{run_id} is not a LoRA adapter (no adapter_config.json); multi-LoRA serving needs adapters.")
        bases[run_id] = adapter_base_model(model_dir, fallback=exp.base_model) or exp.base_model
        try:
            rank = int(cfg.get("r") or 8)
        except (TypeError, ValueError):
            rank = 8
        if rank > _LORA_RANKS[-1]:
            raise ValueError(
                f"Run #{run_id} has LoRA rank {rank}; vLLM serves adapters up to rank {_LORA_RANKS[-1]}."
            )
        adapters.append(
            {
                "run_id": run_id, "name": exp.name, "module_name": f"{slug}:run-{run_id}",
                "path": str(model_dir), "rank": rank,
            }
        )
    if len(set(bases.values())) > 1:
        listing = "; ".join(f"run #{rid} -> {base}" for rid, base in bases.items())
        raise ValueError(f"Runs do not share a base model ({listing}); pick runs trained on the same base.")
    base_model = next(iter(bases.values()))
    max_rank = _round_lora_rank(max(a["rank"] for a in adapters))
    first_exp = await db.get(Experiment, ids[0])
    cwd = str(first_exp.output_dir)

    argv = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model, "--served-model-name", f"{slug}:base",
        "--enable-lora", "--max-loras", str(len(adapters)), "--max-lora-rank", str(max_rank),
        "--lora-modules", *[f"{a['module_name']}={a['path']}" for a in adapters],
        "--host", host, "--port", str(port),
    ]
    if max_model_len:
        argv += ["--max-model-len", str(int(max_model_len))]
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    base_url = f"http://{client_host}:{port}"
    health_url = f"{base_url}/health"
    chat_url = f"{base_url}/v1/chat/completions"
    smoke_payload = {
        "model": adapters[0]["module_name"],
        "messages": [{"role": "user", "content": "Hello from SLM!"}],
        "temperature": 0.0, "max_tokens": 32, "stream": False,
    }
    template = {
        "template_id": "runner.vllm_multi_lora",
        "display_name": "vLLM multi-LoRA server",
        "kind": "runner",
        "target_id": "runner.vllm",
        "description": f"One vLLM server for {len(adapters)} LoRA run(s) on {base_model}.",
        "command": _format_cmd(argv),
        "healthcheck": {
            "url": health_url, "method": "GET", "expected_status": 200,
            "curl": f"curl -sS {shlex.quote(health_url)}",
        },
        "smoke_test": {
            "prompt": smoke_payload["messages"][0]["content"], "url": chat_url, "method": "POST",
            "json_body": smoke_payload, "expected_status": 200,
            "curl": (
                f"curl -sS -X POST {shlex.quote(chat_url)} -H 'Content-Type: application/json' "
                f"-d '{_format_curl_json(smoke_payload)}'"
            ),
        },
        "notes": [
            "Install vLLM in the active environment before launching.",
            "Each adapter is served under its gateway model id; the base model is served as "
            f"'{slug}:base'.",
            "Point the project's gateway upstream at this server (see gateway_settings_hint).",
            "The plan binds whatever host you give it and vLLM has no authentication: keep the default "
            "127.0.0.1 unless the server sits behind your own access control.",
        ],
        "launch_spec": {"cwd": cwd, "argv": argv, "env": {}},
    }
    return {
        "project_id": project_id,
        "slug": slug,
        "base_model": base_model,
        "adapters": adapters,
        "template": template,
        "gateway_settings_hint": {"upstream": "openai_compatible", "base_url": base_url, "upstream_kind": "vllm"},
        "host": host,
        "port": port,
    }

"""Workflow DAG runner service with persistent run/node tracking."""

from __future__ import annotations

import asyncio
import shlex
import socket
import subprocess
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import DatasetType
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode
from app.models.export import ExportFormat
from app.models.project import Project
from app.models.workflow_run import (
    WorkflowNodeStatus,
    WorkflowRun,
    WorkflowRunNode,
    WorkflowRunStatus,
)
from app.services.artifact_registry_service import publish_artifact_batch
from app.services.workflow_graph_service import (
    collect_available_artifacts,
    evaluate_runtime_requirements_for_node,
    flatten_missing_runtime_requirements,
    missing_inputs_for_node,
    resolve_project_workflow_graph,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _topological_order(node_ids: set[str], edges: list[dict[str, Any]]) -> list[str]:
    graph: dict[str, list[str]] = defaultdict(list)
    in_degree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source in node_ids and target in node_ids:
            graph[source].append(target)
            in_degree[target] += 1

    queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
    ordered: list[str] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_id)
        for neighbor in graph.get(node_id, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return ordered


def _dependency_map(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source and target:
            deps[target].append(source)
    return {key: sorted(set(value)) for key, value in deps.items()}


def _redis_available() -> tuple[bool, str | None]:
    parsed = urlparse(settings.CELERY_BROKER_URL or settings.REDIS_URL)
    host = (parsed.hostname or "").strip()
    port = int(parsed.port or 6379)
    if not host:
        return False, "REDIS_URL host missing"
    try:
        with socket.create_connection((host, port), timeout=0.75):
            return True, None
    except OSError as exc:
        return False, str(exc)


def _coerce_int(value: object, *, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _coerce_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_dataset_type(value: object) -> DatasetType:
    token = str(value or "").strip().lower()
    if not token:
        return DatasetType.RAW
    try:
        return DatasetType(token)
    except ValueError:
        return DatasetType.RAW


def _coerce_training_mode(value: object) -> TrainingMode:
    token = str(value or "").strip().lower()
    if not token:
        return TrainingMode.SFT
    try:
        return TrainingMode(token)
    except ValueError:
        return TrainingMode.SFT


def _coerce_export_format(value: object) -> ExportFormat:
    token = str(value or "").strip().lower()
    if not token:
        return ExportFormat.GGUF
    try:
        return ExportFormat(token)
    except ValueError:
        return ExportFormat.GGUF


def _resolve_step_config(
    config: dict[str, Any],
    *,
    step_key: str,
    node_config: object | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {}
    defaults = config.get(step_key)
    if isinstance(defaults, dict):
        base.update(defaults)
    if isinstance(node_config, dict):
        base.update(node_config)
    return base


async def _resolve_latest_experiment_id(
    db: AsyncSession,
    *,
    project_id: int,
    require_completed: bool,
) -> int | None:
    stmt = select(Experiment).where(Experiment.project_id == project_id)
    if require_completed:
        stmt = stmt.where(Experiment.status == ExperimentStatus.COMPLETED)
    stmt = stmt.order_by(Experiment.created_at.desc(), Experiment.id.desc()).limit(1)
    row = await db.execute(stmt)
    exp = row.scalar_one_or_none()
    return int(exp.id) if exp else None


async def _execute_local_adapter_preview_attempt(
    *,
    db: AsyncSession,
    project_id: int,
    node_id: str,
    config: dict[str, Any],
    node_config: object | None = None,
) -> tuple[bool, dict[str, Any], str]:
    from app.services.dataset_service import preview_project_data_adapter

    base_cfg = _resolve_step_config(
        config,
        step_key="data_adapter_preview",
        node_config=node_config,
    )

    dataset_type = _coerce_dataset_type(base_cfg.get("dataset_type"))
    sample_size = _coerce_int(base_cfg.get("sample_size"), default=200, minimum=10, maximum=5000)
    preview_limit = _coerce_int(base_cfg.get("preview_limit"), default=20, minimum=5, maximum=100)
    adapter_id = str(base_cfg.get("adapter_id") or "auto").strip() or "auto"
    adapter_config = base_cfg.get("adapter_config")
    field_mapping = base_cfg.get("field_mapping")
    document_id_raw = base_cfg.get("document_id")
    document_id: int | None = None
    try:
        if document_id_raw not in (None, "", 0):
            document_id = int(document_id_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        document_id = None

    preview = await preview_project_data_adapter(
        db=db,
        project_id=project_id,
        dataset_type=dataset_type,
        sample_size=sample_size,
        adapter_id=adapter_id,
        adapter_config=adapter_config if isinstance(adapter_config, dict) else None,
        document_id=document_id,
        field_mapping=field_mapping if isinstance(field_mapping, dict) else None,
        preview_limit=preview_limit,
    )
    sampled = int(preview.get("sampled_records", 0) or 0)
    mapped = int(preview.get("mapped_records", 0) or 0)
    errors = int(preview.get("error_count", 0) or 0)
    mapping_ratio = (mapped / sampled) if sampled > 0 else 0.0
    validation_report = preview.get("validation_report") if isinstance(preview.get("validation_report"), dict) else {}
    validation_status = str(validation_report.get("status", "")).strip().lower()

    min_mapping_ratio = _coerce_float(base_cfg.get("min_mapping_ratio"), default=0.7, minimum=0.0, maximum=1.0)
    max_error_count = _coerce_int(base_cfg.get("max_error_count"), default=-1, minimum=-1)
    fail_on_validation_warning = _coerce_bool(base_cfg.get("fail_on_validation_warning"), default=False)

    failure_reasons: list[str] = []
    if sampled == 0:
        failure_reasons.append("adapter preview sampled 0 rows")
    if mapping_ratio < min_mapping_ratio:
        failure_reasons.append(
            (
                "mapping ratio below threshold "
                f"({mapping_ratio:.3f} < {min_mapping_ratio:.3f})"
            )
        )
    if max_error_count >= 0 and errors > max_error_count:
        failure_reasons.append(f"error_count exceeded threshold ({errors} > {max_error_count})")
    if fail_on_validation_warning and validation_status and validation_status != "ok":
        failure_reasons.append(f"validation status is '{validation_status}'")

    log_payload = {
        "message": f"local adapter preview completed for {node_id}",
        "preview_summary": {
            "dataset_type": dataset_type.value,
            "requested_adapter_id": preview.get("requested_adapter_id"),
            "resolved_adapter_id": preview.get("resolved_adapter_id"),
            "sampled_records": sampled,
            "mapped_records": mapped,
            "dropped_records": int(preview.get("dropped_records", 0) or 0),
            "error_count": errors,
            "mapping_ratio": round(mapping_ratio, 6),
            "validation_status": validation_status or "unknown",
        },
        "gate": {
            "min_mapping_ratio": min_mapping_ratio,
            "max_error_count": max_error_count,
            "fail_on_validation_warning": fail_on_validation_warning,
        },
    }
    if failure_reasons:
        return False, log_payload, "; ".join(failure_reasons)
    return True, log_payload, ""


async def _execute_local_training_attempt(
    *,
    db: AsyncSession,
    project_id: int,
    node_id: str,
    config: dict[str, Any],
    node_config: object | None = None,
) -> tuple[bool, dict[str, Any], str]:
    from app.services.training_service import create_experiment, get_training_status, start_training

    base_cfg = _resolve_step_config(config, step_key="training", node_config=node_config)
    mode = str(base_cfg.get("mode") or base_cfg.get("execution_mode") or "noop").strip().lower()

    if mode in {"noop", "disabled", "validate_only"}:
        return True, {
            "message": f"local training step skipped for {node_id}",
            "training": {
                "mode": mode,
                "hint": "Set mode=create_and_start or mode=start_existing to execute training.",
            },
        }, ""

    experiment_id_raw = base_cfg.get("experiment_id")
    experiment_id: int | None = None
    try:
        if experiment_id_raw not in (None, "", 0):
            experiment_id = int(experiment_id_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        experiment_id = None

    created_experiment_id: int | None = None
    if mode == "create_and_start":
        name = str(base_cfg.get("name") or f"Workflow {node_id}").strip() or f"Workflow {node_id}"
        description = str(base_cfg.get("description") or "").strip()
        base_model = str(base_cfg.get("base_model") or "microsoft/phi-2").strip() or "microsoft/phi-2"
        training_mode = _coerce_training_mode(base_cfg.get("training_mode"))
        training_cfg = base_cfg.get("config")
        resolved_training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}
        resolved_training_cfg.setdefault("base_model", base_model)

        created = await create_experiment(
            db=db,
            project_id=project_id,
            name=name,
            base_model=base_model,
            config=resolved_training_cfg,
            description=description,
            training_mode=training_mode,
        )
        experiment_id = int(created.id)
        created_experiment_id = experiment_id

    if mode == "start_existing" and experiment_id is None:
        return False, {"message": f"training step failed for {node_id}"}, "training.experiment_id is required for mode=start_existing"

    if mode not in {"create_and_start", "start_existing"}:
        return False, {"message": f"training step failed for {node_id}"}, f"unsupported training mode '{mode}'"

    if experiment_id is None:
        return False, {"message": f"training step failed for {node_id}"}, "unable to resolve experiment for training step"

    start_payload = await start_training(db=db, project_id=project_id, experiment_id=experiment_id)
    wait_for_terminal = _coerce_bool(base_cfg.get("wait_for_terminal"), default=False)

    terminal_status: str | None = None
    poll_count = 0
    if wait_for_terminal:
        timeout_seconds = _coerce_float(base_cfg.get("timeout_seconds"), default=3600, minimum=10)
        poll_interval_seconds = _coerce_float(base_cfg.get("poll_interval_seconds"), default=5.0, minimum=0.2, maximum=60.0)
        started = _utcnow()
        while True:
            poll_count += 1
            status_payload = await get_training_status(db=db, project_id=project_id, experiment_id=experiment_id)
            terminal_status = str(status_payload.get("status") or "").strip().lower() or None
            if terminal_status in {"completed", "failed", "cancelled"}:
                break
            elapsed = (_utcnow() - started).total_seconds()
            if elapsed >= timeout_seconds:
                return False, {
                    "message": f"training wait timed out for {node_id}",
                    "training": {
                        "mode": mode,
                        "experiment_id": experiment_id,
                        "created_experiment_id": created_experiment_id,
                        "wait_for_terminal": True,
                        "poll_count": poll_count,
                    },
                }, (
                    f"training wait timed out after {int(timeout_seconds)}s "
                    f"(experiment_id={experiment_id})"
                )
            await asyncio.sleep(poll_interval_seconds)

        if terminal_status != "completed":
            return False, {
                "message": f"training finished with non-success status for {node_id}",
                "training": {
                    "mode": mode,
                    "experiment_id": experiment_id,
                    "created_experiment_id": created_experiment_id,
                    "terminal_status": terminal_status,
                    "wait_for_terminal": True,
                    "poll_count": poll_count,
                },
            }, f"training status is '{terminal_status}'"

    return True, {
        "message": f"local training step completed for {node_id}",
        "training": {
            "mode": mode,
            "experiment_id": experiment_id,
            "created_experiment_id": created_experiment_id,
            "wait_for_terminal": wait_for_terminal,
            "terminal_status": terminal_status,
            "task_id": start_payload.get("task_id"),
            "backend": start_payload.get("backend"),
            "status": start_payload.get("status"),
        },
    }, ""


async def _execute_local_evaluation_attempt(
    *,
    db: AsyncSession,
    project_id: int,
    node_id: str,
    config: dict[str, Any],
    node_config: object | None = None,
) -> tuple[bool, dict[str, Any], str]:
    from app.services.evaluation_service import run_heldout_evaluation

    base_cfg = _resolve_step_config(config, step_key="evaluation", node_config=node_config)
    mode = str(base_cfg.get("mode") or base_cfg.get("execution_mode") or "noop").strip().lower()
    if mode in {"noop", "disabled", "validate_only"}:
        return True, {
            "message": f"local evaluation step skipped for {node_id}",
            "evaluation": {
                "mode": mode,
                "hint": "Set mode=heldout to execute evaluation.",
            },
        }, ""

    if mode not in {"heldout", "run_heldout"}:
        return False, {"message": f"evaluation step failed for {node_id}"}, f"unsupported evaluation mode '{mode}'"

    experiment_id_raw = base_cfg.get("experiment_id")
    experiment_id: int | None = None
    try:
        if experiment_id_raw not in (None, "", 0):
            experiment_id = int(experiment_id_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        experiment_id = None
    if experiment_id is None:
        require_completed = _coerce_bool(base_cfg.get("require_completed_experiment"), default=True)
        experiment_id = await _resolve_latest_experiment_id(
            db,
            project_id=project_id,
            require_completed=require_completed,
        )
    if experiment_id is None:
        return False, {"message": f"evaluation step failed for {node_id}"}, "evaluation experiment could not be resolved"

    eval_type = str(base_cfg.get("eval_type") or "exact_match").strip().lower()
    if eval_type not in {"exact_match", "f1", "llm_judge"}:
        return False, {"message": f"evaluation step failed for {node_id}"}, f"unsupported eval_type '{eval_type}'"

    result = await run_heldout_evaluation(
        db=db,
        project_id=project_id,
        experiment_id=experiment_id,
        dataset_name=str(base_cfg.get("dataset_name") or "test"),
        eval_type=eval_type,
        max_samples=_coerce_int(base_cfg.get("max_samples"), default=100, minimum=1, maximum=5000),
        max_new_tokens=_coerce_int(base_cfg.get("max_new_tokens"), default=128, minimum=1, maximum=4096),
        temperature=_coerce_float(base_cfg.get("temperature"), default=0.0, minimum=0.0, maximum=2.0),
        model_path=str(base_cfg.get("model_path")).strip() if base_cfg.get("model_path") else None,
        judge_model=str(base_cfg.get("judge_model") or "meta-llama/Meta-Llama-3-70B-Instruct").strip(),
    )

    metrics = dict(result.metrics or {})
    return True, {
        "message": f"local evaluation step completed for {node_id}",
        "evaluation": {
            "mode": mode,
            "experiment_id": experiment_id,
            "eval_result_id": int(result.id),
            "dataset_name": result.dataset_name,
            "eval_type": result.eval_type,
            "pass_rate": result.pass_rate,
            "metrics_keys": sorted(metrics.keys()),
        },
    }, ""


async def _execute_local_export_attempt(
    *,
    db: AsyncSession,
    project_id: int,
    node_id: str,
    config: dict[str, Any],
    node_config: object | None = None,
) -> tuple[bool, dict[str, Any], str]:
    from app.services.export_service import create_export, run_export

    base_cfg = _resolve_step_config(config, step_key="export", node_config=node_config)
    mode = str(base_cfg.get("mode") or base_cfg.get("execution_mode") or "noop").strip().lower()
    if mode in {"noop", "disabled", "validate_only"}:
        return True, {
            "message": f"local export step skipped for {node_id}",
            "export": {
                "mode": mode,
                "hint": "Set mode=create_and_run or mode=run_existing to execute export.",
            },
        }, ""

    export_id_raw = base_cfg.get("export_id")
    export_id: int | None = None
    try:
        if export_id_raw not in (None, "", 0):
            export_id = int(export_id_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        export_id = None

    created_export_id: int | None = None
    if mode == "create_and_run":
        experiment_id_raw = base_cfg.get("experiment_id")
        experiment_id: int | None = None
        try:
            if experiment_id_raw not in (None, "", 0):
                experiment_id = int(experiment_id_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            experiment_id = None
        if experiment_id is None:
            require_completed = _coerce_bool(base_cfg.get("require_completed_experiment"), default=True)
            experiment_id = await _resolve_latest_experiment_id(
                db,
                project_id=project_id,
                require_completed=require_completed,
            )
        if experiment_id is None:
            return False, {"message": f"export step failed for {node_id}"}, "export experiment could not be resolved"

        export_format = _coerce_export_format(base_cfg.get("export_format"))
        quantization = base_cfg.get("quantization")
        quantization_value = str(quantization).strip() if quantization is not None else None

        created = await create_export(
            db=db,
            project_id=project_id,
            experiment_id=experiment_id,
            export_format=export_format,
            quantization=quantization_value or None,
        )
        export_id = int(created.id)
        created_export_id = export_id
    elif mode != "run_existing":
        return False, {"message": f"export step failed for {node_id}"}, f"unsupported export mode '{mode}'"

    if export_id is None:
        return False, {"message": f"export step failed for {node_id}"}, "export.export_id is required for mode=run_existing"

    eval_report = base_cfg.get("eval_report")
    safety_scorecard = base_cfg.get("safety_scorecard")
    export_row = await run_export(
        db=db,
        project_id=project_id,
        export_id=export_id,
        eval_report=eval_report if isinstance(eval_report, dict) else None,
        safety_scorecard=safety_scorecard if isinstance(safety_scorecard, dict) else None,
    )

    manifest = dict(export_row.manifest or {})
    return True, {
        "message": f"local export step completed for {node_id}",
        "export": {
            "mode": mode,
            "export_id": export_id,
            "created_export_id": created_export_id,
            "status": export_row.status.value,
            "output_path": export_row.output_path,
            "run_id": manifest.get("run_id"),
        },
    }, ""


async def execute_local_core_step_attempt(
    *,
    db: AsyncSession,
    project_id: int,
    node_id: str,
    step_type: str,
    config: dict[str, Any],
    node_config: object | None = None,
) -> tuple[bool, dict[str, Any], str]:
    """Execute a supported core workflow step in local mode."""
    if step_type == "core.data_adapter_preview":
        return await _execute_local_adapter_preview_attempt(
            db=db,
            project_id=project_id,
            node_id=node_id,
            config=config,
            node_config=node_config,
        )
    if step_type == "core.training":
        return await _execute_local_training_attempt(
            db=db,
            project_id=project_id,
            node_id=node_id,
            config=config,
            node_config=node_config,
        )
    if step_type == "core.evaluation":
        return await _execute_local_evaluation_attempt(
            db=db,
            project_id=project_id,
            node_id=node_id,
            config=config,
            node_config=node_config,
        )
    if step_type == "core.export":
        return await _execute_local_export_attempt(
            db=db,
            project_id=project_id,
            node_id=node_id,
            config=config,
            node_config=node_config,
        )

    return True, {"message": f"local backend completed {step_type}"}, ""


async def _execute_local_node_attempt(
    *,
    db: AsyncSession,
    project_id: int,
    node: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any], str]:
    node_id = str(node.get("id", "")).strip()
    step_type = str(node.get("step_type", "")).strip()
    return await execute_local_core_step_attempt(
        db=db,
        project_id=project_id,
        node_id=node_id,
        step_type=step_type,
        config=config,
        node_config=node.get("config"),
    )


def _execute_external(
    *,
    command_template: str,
    project_id: int,
    run_id: int,
    node_id: str,
    stage: str,
    step_type: str,
    timeout_seconds: int,
) -> tuple[bool, str, str]:
    if not command_template.strip():
        return False, "", "external backend requires config.external_command_template"
    try:
        command = command_template.format(
            project_id=project_id,
            run_id=run_id,
            node_id=node_id,
            stage=stage,
            step_type=step_type,
        )
    except KeyError as e:
        return False, "", f"external command missing placeholder: {e.args[0]}"

    proc = subprocess.run(
        shlex.split(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1, timeout_seconds),
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return False, stdout[-2000:], stderr[-2000:] or f"external command exited with code {proc.returncode}"
    return True, stdout[-2000:], ""


def _execute_celery_node_attempt(
    *,
    project_id: int,
    run_id: int,
    node_id: str,
    stage: str,
    step_type: str,
    attempt: int,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any], str]:
    ok, detail = _redis_available()
    if not ok:
        return False, {"message": "celery backend unavailable"}, f"redis unavailable: {detail}"

    try:
        from app.worker import celery_app
    except Exception as exc:
        return False, {"message": "celery backend import failed"}, str(exc)

    timeout_seconds = int(
        config.get("celery_result_timeout_seconds")
        or config.get("external_timeout_seconds")
        or settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS
    )
    timeout_seconds = max(1, timeout_seconds)
    queue_name = str(config.get("celery_queue", "")).strip() or None

    task = celery_app.send_task(
        "run_workflow_node_job",
        kwargs={
            "project_id": project_id,
            "run_id": run_id,
            "node_id": node_id,
            "stage": stage,
            "step_type": step_type,
            "attempt": attempt,
            "config": dict(config),
        },
        queue=queue_name,
    )

    try:
        payload = task.get(timeout=timeout_seconds)
    except Exception as exc:
        return False, {"message": "celery node task failed", "task_id": task.id}, str(exc)

    if not isinstance(payload, dict):
        return False, {"message": "celery node task returned invalid payload", "task_id": task.id}, "invalid task result"

    success = bool(payload.get("success"))
    log_payload = payload.get("log")
    if not isinstance(log_payload, dict):
        log_payload = {"message": str(log_payload or "")}
    log_payload.setdefault("message", f"celery backend completed {stage}")
    log_payload.setdefault("task_id", task.id)
    error_text = str(payload.get("error", "")).strip()

    if success:
        return True, log_payload, ""
    return False, log_payload, error_text or "celery node task reported failure"


async def _execute_node_attempt(
    *,
    db: AsyncSession,
    backend: str,
    project_id: int,
    run_id: int,
    node: dict[str, Any],
    attempt: int,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any], str]:
    """Execute one node attempt and return success/log/error."""
    stage = str(node.get("stage", "")).strip()
    node_id = str(node.get("id", "")).strip()
    step_type = str(node.get("step_type", "")).strip()

    fail_stages = {
        str(item).strip()
        for item in config.get("simulate_fail_stages", [])
        if isinstance(item, str) and item.strip()
    }
    fail_once_stages = {
        str(item).strip()
        for item in config.get("simulate_fail_once_stages", [])
        if isinstance(item, str) and item.strip()
    }

    if stage in fail_stages:
        return False, {"message": f"simulated failure for stage '{stage}'"}, "simulated stage failure"
    if stage in fail_once_stages and attempt == 1:
        return False, {"message": f"simulated one-time failure for stage '{stage}'"}, "simulated one-time failure"

    if backend == "local":
        return await _execute_local_node_attempt(
            db=db,
            project_id=project_id,
            node=node,
            config=config,
        )

    if backend == "celery":
        celery_cfg = dict(config or {})
        node_cfg = node.get("config")
        if isinstance(node_cfg, dict):
            celery_cfg["node_config"] = dict(node_cfg)
        return _execute_celery_node_attempt(
            project_id=project_id,
            run_id=run_id,
            node_id=node_id,
            stage=stage,
            step_type=step_type,
            attempt=attempt,
            config=celery_cfg,
        )

    if backend == "external":
        timeout_seconds = int(config.get("external_timeout_seconds") or 120)
        success, stdout_tail, error_text = _execute_external(
            command_template=str(config.get("external_command_template") or ""),
            project_id=project_id,
            run_id=run_id,
            node_id=node_id,
            stage=stage,
            step_type=step_type,
            timeout_seconds=timeout_seconds,
        )
        log_payload = {
            "message": f"external backend executed {stage}",
            "stdout_tail": stdout_tail,
        }
        return success, log_payload, error_text

    return False, {"message": f"unknown backend '{backend}'"}, f"unsupported backend: {backend}"


def _serialize_node(row: WorkflowRunNode) -> dict[str, Any]:
    return {
        "id": row.id,
        "run_id": row.run_id,
        "node_id": row.node_id,
        "stage": row.stage,
        "step_type": row.step_type,
        "execution_backend": row.execution_backend,
        "status": row.status.value,
        "attempt_count": row.attempt_count,
        "max_retries": row.max_retries,
        "dependencies": list(row.dependencies or []),
        "input_artifacts": list(row.input_artifacts or []),
        "output_artifacts": list(row.output_artifacts or []),
        "runtime_requirements": dict(row.runtime_requirements or {}),
        "missing_inputs": list(row.missing_inputs or []),
        "missing_runtime_requirements": list(row.missing_runtime_requirements or []),
        "published_artifact_keys": list(row.published_artifact_keys or []),
        "error_message": row.error_message or "",
        "node_log": list(row.node_log or []),
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _serialize_run(run: WorkflowRun, nodes: list[WorkflowRunNode]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for item in nodes:
        counts[item.status.value] += 1
    return {
        "id": run.id,
        "project_id": run.project_id,
        "graph_id": run.graph_id,
        "graph_version": run.graph_version,
        "execution_backend": run.execution_backend,
        "status": run.status.value,
        "run_config": run.run_config or {},
        "summary": {
            **(run.summary or {}),
            "node_counts": dict(counts),
            "total_nodes": len(nodes),
        },
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "nodes": [_serialize_node(item) for item in nodes],
    }


def serialize_workflow_run(run: WorkflowRun, nodes: list[WorkflowRunNode] | None = None) -> dict[str, Any]:
    """Public serializer for workflow run payloads."""
    return _serialize_run(run, nodes or [])


async def create_workflow_run_shell(
    db: AsyncSession,
    project_id: int,
    *,
    execution_backend: str = "local",
    run_config: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> WorkflowRun:
    """Create a pending workflow run shell for asynchronous execution."""
    run = WorkflowRun(
        project_id=project_id,
        graph_id="pending",
        graph_version="pending",
        execution_backend=execution_backend,
        status=WorkflowRunStatus.PENDING,
        run_config=dict(run_config or {}),
        summary=dict(summary or {}),
        started_at=None,
        finished_at=None,
    )
    db.add(run)
    await db.flush()
    await db.refresh(run)
    return run


async def mark_workflow_run_failed(
    db: AsyncSession,
    project_id: int,
    run_id: int,
    message: str,
) -> None:
    row = await db.execute(
        select(WorkflowRun).where(
            WorkflowRun.project_id == project_id,
            WorkflowRun.id == run_id,
        )
    )
    run = row.scalar_one_or_none()
    if run is None:
        return
    run.status = WorkflowRunStatus.FAILED
    run.finished_at = _utcnow()
    run.summary = {
        **(run.summary or {}),
        "runner_error": message,
    }
    await db.flush()


async def run_workflow_graph(
    db: AsyncSession,
    project: Project,
    *,
    graph_override: dict[str, Any] | None = None,
    allow_fallback: bool = True,
    use_saved_override: bool = True,
    execution_backend: str = "local",
    max_retries: int = 0,
    stop_on_blocked: bool = True,
    stop_on_failure: bool = True,
    config: dict[str, Any] | None = None,
    run_id: int | None = None,
    commit_progress: bool = False,
) -> dict[str, Any]:
    """Execute workflow DAG for a project and persist run/node statuses."""
    cfg = dict(config or {})
    resolved = resolve_project_workflow_graph(
        project_id=project.id,
        current_stage=project.pipeline_stage,
        graph_override=graph_override,
        allow_fallback=allow_fallback,
        use_saved_override=use_saved_override,
    )
    graph = resolved.get("graph") if isinstance(resolved.get("graph"), dict) else {}
    nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
    edges = [item for item in graph.get("edges", []) if isinstance(item, dict)]
    node_map = {
        str(item.get("id", "")).strip(): item
        for item in nodes
        if str(item.get("id", "")).strip()
    }
    ordered_node_ids = _topological_order(set(node_map.keys()), edges)
    run_config_payload = {
        "allow_fallback": allow_fallback,
        "use_saved_override": use_saved_override,
        "max_retries": int(max_retries),
        "stop_on_blocked": bool(stop_on_blocked),
        "stop_on_failure": bool(stop_on_failure),
        "config": cfg,
    }
    summary_payload = {
        "requested_source": resolved.get("requested_source"),
        "effective_source": resolved.get("effective_source"),
        "fallback_used": bool(resolved.get("fallback_used")),
        "graph_valid": bool(resolved.get("valid")),
        "graph_errors": list(resolved.get("errors", [])),
        "graph_warnings": list(resolved.get("warnings", [])),
    }

    if run_id is None:
        run = WorkflowRun(
            project_id=project.id,
            graph_id=str(graph.get("graph_id") or "unknown"),
            graph_version=str(graph.get("graph_version") or "1.0.0"),
            execution_backend=execution_backend,
            status=WorkflowRunStatus.RUNNING,
            run_config=run_config_payload,
            summary=summary_payload,
            started_at=_utcnow(),
        )
        db.add(run)
        await db.flush()
        await db.refresh(run)
    else:
        row = await db.execute(
            select(WorkflowRun).where(
                WorkflowRun.project_id == project.id,
                WorkflowRun.id == run_id,
            )
        )
        run = row.scalar_one_or_none()
        if run is None:
            raise ValueError(f"Workflow run {run_id} not found for project {project.id}")
        run.graph_id = str(graph.get("graph_id") or "unknown")
        run.graph_version = str(graph.get("graph_version") or "1.0.0")
        run.execution_backend = execution_backend
        run.status = WorkflowRunStatus.RUNNING
        run.run_config = run_config_payload
        run.summary = summary_payload
        run.started_at = _utcnow()
        run.finished_at = None
        await db.execute(delete(WorkflowRunNode).where(WorkflowRunNode.run_id == run.id))
        await db.flush()

    if commit_progress:
        await db.commit()

    if not ordered_node_ids or len(ordered_node_ids) != len(node_map):
        run.status = WorkflowRunStatus.FAILED
        run.finished_at = _utcnow()
        run.summary = {
            **(run.summary or {}),
            "runner_error": "graph ordering failed; run aborted",
        }
        await db.flush()
        if commit_progress:
            await db.commit()
        return _serialize_run(run, [])

    deps_by_node = _dependency_map(edges)
    run_nodes: dict[str, WorkflowRunNode] = {}
    for node_id in ordered_node_ids:
        node = node_map[node_id]
        runtime_check = evaluate_runtime_requirements_for_node(node)
        row = WorkflowRunNode(
            run_id=run.id,
            node_id=node_id,
            stage=str(node.get("stage") or ""),
            step_type=str(node.get("step_type") or ""),
            execution_backend=execution_backend,
            status=WorkflowNodeStatus.PENDING,
            attempt_count=0,
            max_retries=max(0, int(max_retries)),
            dependencies=list(deps_by_node.get(node_id, [])),
            input_artifacts=list(node.get("input_artifacts", [])),
            output_artifacts=list(node.get("output_artifacts", [])),
            runtime_requirements=runtime_check.get("runtime_requirements", {}),
            missing_inputs=[],
            missing_runtime_requirements=[],
            published_artifact_keys=[],
            node_log=[],
            error_message="",
        )
        db.add(row)
        run_nodes[node_id] = row

    await db.flush()
    if commit_progress:
        await db.commit()
    available_artifacts = await collect_available_artifacts(db, project.id)

    run_blocked = False
    run_failed = False
    break_reason = ""
    executed_node_ids: set[str] = set()

    for node_id in ordered_node_ids:
        row = run_nodes[node_id]
        node = node_map[node_id]
        executed_node_ids.add(node_id)

        deps = list(row.dependencies or [])
        dep_failed = any(
            run_nodes[dep_id].status in {WorkflowNodeStatus.FAILED, WorkflowNodeStatus.BLOCKED, WorkflowNodeStatus.SKIPPED}
            for dep_id in deps
            if dep_id in run_nodes
        )
        if dep_failed:
            row.status = WorkflowNodeStatus.SKIPPED
            row.error_message = "skipped due to dependency failure/block"
            row.finished_at = _utcnow()
            if commit_progress:
                await db.commit()
            continue

        missing_inputs = missing_inputs_for_node(node, available_artifacts)
        runtime_check = evaluate_runtime_requirements_for_node(node)
        missing_runtime = flatten_missing_runtime_requirements(runtime_check)
        row.runtime_requirements = runtime_check.get("runtime_requirements", {})
        row.missing_inputs = missing_inputs
        row.missing_runtime_requirements = missing_runtime

        if missing_inputs or missing_runtime:
            row.status = WorkflowNodeStatus.BLOCKED
            reasons = []
            if missing_inputs:
                reasons.append(f"missing inputs: {', '.join(missing_inputs)}")
            if missing_runtime:
                reasons.append(f"missing runtime requirements: {', '.join(missing_runtime)}")
            row.error_message = "; ".join(reasons)
            row.finished_at = _utcnow()
            run_blocked = True
            break_reason = row.error_message
            if commit_progress:
                await db.commit()
            if stop_on_blocked:
                break
            continue

        attempt = 0
        success = False
        last_error = ""
        logs = list(row.node_log or [])
        while attempt <= max(0, int(max_retries)):
            attempt += 1
            row.attempt_count = attempt
            row.status = WorkflowNodeStatus.RUNNING
            if row.started_at is None:
                row.started_at = _utcnow()
            ok, log_payload, error_text = await _execute_node_attempt(
                db=db,
                backend=execution_backend,
                project_id=project.id,
                run_id=run.id,
                node=node,
                attempt=attempt,
                config=cfg,
            )
            logs.append(
                {
                    "attempt": attempt,
                    "timestamp": _utcnow().isoformat(),
                    "backend": execution_backend,
                    **log_payload,
                }
            )
            row.node_log = logs
            if ok:
                success = True
                break
            last_error = error_text or "node execution failed"

        if not success:
            row.status = WorkflowNodeStatus.FAILED
            row.error_message = last_error
            row.finished_at = _utcnow()
            run_failed = True
            break_reason = last_error
            if commit_progress:
                await db.commit()
            if stop_on_failure:
                break
            continue

        output_artifacts = [item for item in (row.output_artifacts or []) if isinstance(item, str) and item.strip()]
        published = await publish_artifact_batch(
            db=db,
            project_id=project.id,
            artifact_keys=output_artifacts,
            producer_stage=row.stage,
            producer_run_id=str(run.id),
            producer_step_id=row.node_id,
            metadata={"source": "workflow.runner", "backend": execution_backend},
        )
        row.published_artifact_keys = [item.artifact_key for item in published]
        available_artifacts.update(row.published_artifact_keys)
        row.status = WorkflowNodeStatus.COMPLETED
        row.error_message = ""
        row.finished_at = _utcnow()
        if commit_progress:
            await db.commit()

    for node_id in ordered_node_ids:
        if node_id in executed_node_ids:
            continue
        row = run_nodes[node_id]
        if row.status == WorkflowNodeStatus.PENDING:
            row.status = WorkflowNodeStatus.SKIPPED
            row.error_message = "skipped after workflow termination"
            row.finished_at = _utcnow()
            if commit_progress:
                await db.commit()

    if run_failed:
        run.status = WorkflowRunStatus.FAILED
    elif run_blocked:
        run.status = WorkflowRunStatus.BLOCKED
    else:
        run.status = WorkflowRunStatus.COMPLETED

    run.finished_at = _utcnow()
    run.summary = {
        **(run.summary or {}),
        "available_artifacts_final": sorted(available_artifacts),
        "break_reason": break_reason,
    }
    await db.flush()
    if commit_progress:
        await db.commit()

    node_rows = [run_nodes[node_id] for node_id in ordered_node_ids if node_id in run_nodes]
    return _serialize_run(run, node_rows)


async def list_workflow_runs(
    db: AsyncSession,
    project_id: int,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    rows = await db.execute(
        select(WorkflowRun)
        .where(WorkflowRun.project_id == project_id)
        .order_by(WorkflowRun.created_at.desc(), WorkflowRun.id.desc())
        .limit(max(1, min(limit, 200)))
    )
    runs = list(rows.scalars().all())
    if not runs:
        return []

    run_ids = [item.id for item in runs]
    node_rows = await db.execute(
        select(WorkflowRunNode)
        .where(WorkflowRunNode.run_id.in_(run_ids))
        .order_by(WorkflowRunNode.id.asc())
    )
    by_run: dict[int, list[WorkflowRunNode]] = defaultdict(list)
    for item in node_rows.scalars().all():
        by_run[item.run_id].append(item)
    return [_serialize_run(item, by_run.get(item.id, [])) for item in runs]


async def get_workflow_run(
    db: AsyncSession,
    project_id: int,
    run_id: int,
) -> dict[str, Any] | None:
    row = await db.execute(
        select(WorkflowRun).where(
            WorkflowRun.project_id == project_id,
            WorkflowRun.id == run_id,
        )
    )
    run = row.scalar_one_or_none()
    if run is None:
        return None
    node_rows = await db.execute(
        select(WorkflowRunNode)
        .where(WorkflowRunNode.run_id == run.id)
        .order_by(WorkflowRunNode.id.asc())
    )
    nodes = list(node_rows.scalars().all())
    return _serialize_run(run, nodes)

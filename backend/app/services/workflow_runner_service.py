"""Workflow DAG runner service with persistent run/node tracking."""

from __future__ import annotations

import asyncio
import copy
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


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _get_path(payload: Any, path: str) -> Any:
    if not path.strip():
        return payload
    current: Any = payload
    for token in [item.strip() for item in path.split(".") if item.strip()]:
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        if isinstance(current, list):
            try:
                index = int(token)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_failure_type(
    error_text: str,
    *,
    missing_inputs: list[str] | None = None,
    missing_runtime: list[str] | None = None,
    cancelled: bool = False,
) -> str:
    if cancelled:
        return "cancelled"
    if missing_inputs:
        return "blocked_input"
    if missing_runtime:
        return "blocked_runtime"

    token = str(error_text or "").strip().lower()
    if not token:
        return "execution"
    if any(
        marker in token
        for marker in (
            "timeout",
            "temporar",
            "connection",
            "unavailable",
            "rate limit",
            "network",
            "redis unavailable",
            "connection refused",
            "try again",
        )
    ):
        return "transient"
    if any(marker in token for marker in ("cancelled", "canceled")):
        return "cancelled"
    if any(marker in token for marker in ("missing inputs", "missing input")):
        return "blocked_input"
    if any(marker in token for marker in ("missing runtime", "runtime requirement")):
        return "blocked_runtime"
    return "execution"


def _normalize_retry_policy(
    *,
    global_max_retries: int,
    run_config: dict[str, Any],
    node: dict[str, Any],
) -> dict[str, Any]:
    defaults = {
        "max_retries": max(0, int(global_max_retries)),
        "retry_on": ["execution", "transient"],
        "no_retry_on": [],
        "backoff_seconds": 0.0,
        "backoff_multiplier": 1.0,
    }

    run_retry = run_config.get("retry_policy")
    if isinstance(run_retry, dict):
        default_policy = run_retry.get("default")
        if isinstance(default_policy, dict):
            defaults = _deep_merge_dict(defaults, default_policy)
        by_stage = run_retry.get("by_stage")
        stage = str(node.get("stage") or "").strip()
        if isinstance(by_stage, dict) and isinstance(by_stage.get(stage), dict):
            defaults = _deep_merge_dict(defaults, by_stage.get(stage) or {})

    node_retry = node.get("retry_policy")
    if isinstance(node_retry, dict):
        defaults = _deep_merge_dict(defaults, node_retry)

    retry_on_raw = defaults.get("retry_on")
    if isinstance(retry_on_raw, list):
        retry_on = sorted(
            {
                str(item).strip().lower()
                for item in retry_on_raw
                if isinstance(item, str) and str(item).strip()
            }
        )
    elif isinstance(retry_on_raw, str):
        token = str(retry_on_raw).strip().lower()
        retry_on = [token] if token else []
    else:
        retry_on = []

    no_retry_raw = defaults.get("no_retry_on")
    if isinstance(no_retry_raw, list):
        no_retry_on = sorted(
            {
                str(item).strip().lower()
                for item in no_retry_raw
                if isinstance(item, str) and str(item).strip()
            }
        )
    elif isinstance(no_retry_raw, str):
        token = str(no_retry_raw).strip().lower()
        no_retry_on = [token] if token else []
    else:
        no_retry_on = []

    return {
        "max_retries": _coerce_int(defaults.get("max_retries"), default=0, minimum=0, maximum=20),
        "retry_on": retry_on,
        "no_retry_on": no_retry_on,
        "backoff_seconds": _coerce_float(defaults.get("backoff_seconds"), default=0.0, minimum=0.0, maximum=60.0),
        "backoff_multiplier": _coerce_float(defaults.get("backoff_multiplier"), default=1.0, minimum=1.0, maximum=10.0),
    }


def _should_retry_failure(
    *,
    failure_type: str,
    attempt: int,
    policy: dict[str, Any],
) -> bool:
    max_retries = int(policy.get("max_retries") or 0)
    if attempt > max_retries:
        return False

    token = str(failure_type or "").strip().lower()
    no_retry_on = {str(item).strip().lower() for item in list(policy.get("no_retry_on") or []) if str(item).strip()}
    if token and token in no_retry_on:
        return False

    retry_on = {str(item).strip().lower() for item in list(policy.get("retry_on") or []) if str(item).strip()}
    if not retry_on:
        return False
    if "any" in retry_on:
        return True
    return token in retry_on


def _normalize_loop_policy(node: dict[str, Any]) -> dict[str, Any] | None:
    loop_payload = node.get("loop")
    if not isinstance(loop_payload, dict):
        return None
    enabled = _coerce_bool(loop_payload.get("enabled"), default=True)
    if not enabled:
        return None

    items_raw = loop_payload.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        return None

    items: list[dict[str, Any]] = []
    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("id") or f"trial-{index + 1}").strip() or f"trial-{index + 1}"
        node_cfg = item.get("node_config")
        run_cfg = item.get("run_config")
        item_retry = item.get("retry_policy")
        items.append(
            {
                "label": label,
                "node_config": dict(node_cfg) if isinstance(node_cfg, dict) else {},
                "run_config": dict(run_cfg) if isinstance(run_cfg, dict) else {},
                "retry_policy": dict(item_retry) if isinstance(item_retry, dict) else None,
            }
        )

    if not items:
        return None

    return {
        "type": str(loop_payload.get("type") or "sweep").strip().lower() or "sweep",
        "items": items,
        "objective_path": str(loop_payload.get("objective_path") or "").strip() or None,
        "objective_mode": str(loop_payload.get("objective_mode") or "max").strip().lower() or "max",
        "stop_on_first_success": _coerce_bool(loop_payload.get("stop_on_first_success"), default=False),
        "require_all_success": _coerce_bool(loop_payload.get("require_all_success"), default=False),
    }


def _extract_iteration_score(iteration_payload: dict[str, Any], objective_path: str | None) -> float | None:
    if not objective_path:
        return None
    value = _get_path(iteration_payload, objective_path)
    return _to_float(value)


def _evaluate_program_condition(
    condition: Any,
    *,
    source_node_id: str | None = None,
    available_artifacts: set[str],
    run_config: dict[str, Any],
    node_results: dict[str, dict[str, Any]],
) -> bool:
    if condition is None:
        return True
    if isinstance(condition, bool):
        return condition
    if isinstance(condition, list):
        return all(
            _evaluate_program_condition(
                item,
                source_node_id=source_node_id,
                available_artifacts=available_artifacts,
                run_config=run_config,
                node_results=node_results,
            )
            for item in condition
        )
    if not isinstance(condition, dict):
        return False

    if "all" in condition:
        rows = condition.get("all")
        if not isinstance(rows, list):
            return False
        return all(
            _evaluate_program_condition(
                item,
                source_node_id=source_node_id,
                available_artifacts=available_artifacts,
                run_config=run_config,
                node_results=node_results,
            )
            for item in rows
        )
    if "any" in condition:
        rows = condition.get("any")
        if not isinstance(rows, list):
            return False
        return any(
            _evaluate_program_condition(
                item,
                source_node_id=source_node_id,
                available_artifacts=available_artifacts,
                run_config=run_config,
                node_results=node_results,
            )
            for item in rows
        )
    if "not" in condition:
        return not _evaluate_program_condition(
            condition.get("not"),
            source_node_id=source_node_id,
            available_artifacts=available_artifacts,
            run_config=run_config,
            node_results=node_results,
        )

    if "artifact_exists" in condition:
        key = str(condition.get("artifact_exists") or "").strip()
        return bool(key and key in available_artifacts)
    if "artifact_missing" in condition:
        key = str(condition.get("artifact_missing") or "").strip()
        return bool(key and key not in available_artifacts)

    if "source_status_in" in condition:
        rows = condition.get("source_status_in")
        allowed = {
            str(item).strip().lower()
            for item in (rows if isinstance(rows, list) else [rows])
            if str(item).strip()
        }
        if not allowed or not source_node_id:
            return False
        source = node_results.get(source_node_id) or {}
        status = str(source.get("status") or "").strip().lower()
        return status in allowed

    if "source_status_equals" in condition:
        expected = str(condition.get("source_status_equals") or "").strip().lower()
        if not expected or not source_node_id:
            return False
        source = node_results.get(source_node_id) or {}
        return str(source.get("status") or "").strip().lower() == expected

    if "node_status" in condition and isinstance(condition.get("node_status"), dict):
        payload = condition.get("node_status") or {}
        node_id = str(payload.get("node_id") or "").strip()
        if not node_id:
            return False
        row = node_results.get(node_id) or {}
        status = str(row.get("status") or "").strip().lower()
        if "equals" in payload:
            return status == str(payload.get("equals") or "").strip().lower()
        if "in" in payload:
            values = payload.get("in")
            allowed = {
                str(item).strip().lower()
                for item in (values if isinstance(values, list) else [values])
                if str(item).strip()
            }
            return status in allowed
        if "not_in" in payload:
            values = payload.get("not_in")
            blocked = {
                str(item).strip().lower()
                for item in (values if isinstance(values, list) else [values])
                if str(item).strip()
            }
            return status not in blocked
        return False

    if "node_metric_gte" in condition and isinstance(condition.get("node_metric_gte"), dict):
        payload = condition.get("node_metric_gte") or {}
        node_id = str(payload.get("node_id") or "").strip()
        path = str(payload.get("path") or "").strip()
        threshold = _to_float(payload.get("value"))
        if not node_id or not path or threshold is None:
            return False
        row = node_results.get(node_id) or {}
        value = _to_float(_get_path(row, path))
        return value is not None and value >= threshold

    if "node_metric_lte" in condition and isinstance(condition.get("node_metric_lte"), dict):
        payload = condition.get("node_metric_lte") or {}
        node_id = str(payload.get("node_id") or "").strip()
        path = str(payload.get("path") or "").strip()
        threshold = _to_float(payload.get("value"))
        if not node_id or not path or threshold is None:
            return False
        row = node_results.get(node_id) or {}
        value = _to_float(_get_path(row, path))
        return value is not None and value <= threshold

    if "config_equals" in condition and isinstance(condition.get("config_equals"), dict):
        payload = condition.get("config_equals") or {}
        path = str(payload.get("path") or "").strip()
        if not path:
            return False
        expected = payload.get("value")
        return _get_path(run_config, path) == expected

    if "always" in condition:
        return _coerce_bool(condition.get("always"), default=False)

    # Unknown condition shape is treated as false.
    return False


def _edge_condition_payload(edge: dict[str, Any]) -> Any:
    if "condition" in edge:
        return edge.get("condition")
    if "when" in edge:
        return edge.get("when")
    if "if" in edge:
        return edge.get("if")
    return None


def _incoming_edges(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        target = str(edge.get("target") or "").strip()
        if not target:
            continue
        rows[target].append(edge)
    return rows


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
    task_profile = str(base_cfg.get("task_profile") or "").strip() or None
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
        task_profile=task_profile,
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
            "requested_task_profile": preview.get("requested_task_profile"),
            "resolved_task_profile": preview.get("resolved_task_profile"),
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

    delay_seconds = _coerce_float(
        config.get("simulate_node_delay_seconds"),
        default=0.0,
        minimum=0.0,
        maximum=30.0,
    )
    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)

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


async def mark_workflow_run_cancelled(
    db: AsyncSession,
    project_id: int,
    run_id: int,
    *,
    message: str = "cancel requested",
) -> bool:
    """Mark a workflow run as cancelled when it is not already terminal."""
    row = await db.execute(
        select(WorkflowRun).where(
            WorkflowRun.project_id == project_id,
            WorkflowRun.id == run_id,
        )
    )
    run = row.scalar_one_or_none()
    if run is None:
        return False

    if run.status in {WorkflowRunStatus.COMPLETED, WorkflowRunStatus.FAILED, WorkflowRunStatus.CANCELLED}:
        return True

    run.status = WorkflowRunStatus.CANCELLED
    run.finished_at = _utcnow()
    run.summary = {
        **(run.summary or {}),
        "cancel_reason": message,
        "cancel_requested_at": _utcnow().isoformat(),
    }
    await db.flush()
    return True


async def is_workflow_run_cancelled(
    db: AsyncSession,
    *,
    project_id: int,
    run_id: int,
) -> bool:
    row = await db.execute(
        select(WorkflowRun.status).where(
            WorkflowRun.project_id == project_id,
            WorkflowRun.id == run_id,
        )
    )
    status = row.scalar_one_or_none()
    return status == WorkflowRunStatus.CANCELLED


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
    resume_from_run_id: int | None = None,
    resume_from_node_id: str | None = None,
    reuse_successful_nodes: bool = False,
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
        "resume_from_run_id": int(resume_from_run_id) if resume_from_run_id is not None else None,
        "resume_from_node_id": str(resume_from_node_id or "").strip() or None,
        "reuse_successful_nodes": bool(reuse_successful_nodes),
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
        if run.status == WorkflowRunStatus.CANCELLED:
            existing_nodes_row = await db.execute(
                select(WorkflowRunNode)
                .where(WorkflowRunNode.run_id == run.id)
                .order_by(WorkflowRunNode.id.asc())
            )
            existing_nodes = list(existing_nodes_row.scalars().all())
            return _serialize_run(run, existing_nodes)
        run.graph_id = str(graph.get("graph_id") or "unknown")
        run.graph_version = str(graph.get("graph_version") or "1.0.0")
        run.execution_backend = execution_backend
        run.status = WorkflowRunStatus.RUNNING
        existing_run_config = dict(run.run_config or {})
        existing_step_config = existing_run_config.get("config")
        if isinstance(existing_step_config, dict):
            merged_step_config = dict(existing_step_config)
            merged_step_config.update(cfg)
            run_config_payload["config"] = merged_step_config
        run.run_config = {
            **existing_run_config,
            **run_config_payload,
        }
        run.summary = {
            **dict(run.summary or {}),
            **summary_payload,
        }
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

    resume_source_run_id: int | None = None
    previous_nodes_by_id: dict[str, WorkflowRunNode] = {}
    resume_start_node_id: str | None = None
    resume_start_index: int = 0
    requested_resume_node = str(resume_from_node_id or "").strip()
    if reuse_successful_nodes and resume_from_run_id is not None:
        resume_source_run_id = int(resume_from_run_id)
        previous_nodes_row = await db.execute(
            select(WorkflowRunNode)
            .where(WorkflowRunNode.run_id == resume_source_run_id)
            .order_by(WorkflowRunNode.id.asc())
        )
        previous_nodes_by_id = {
            str(item.node_id): item
            for item in previous_nodes_row.scalars().all()
            if str(item.node_id).strip()
        }
        if requested_resume_node:
            if requested_resume_node not in node_map:
                run.status = WorkflowRunStatus.FAILED
                run.finished_at = _utcnow()
                run.summary = {
                    **(run.summary or {}),
                    "runner_error": f"resume node '{requested_resume_node}' not found in graph",
                }
                await db.flush()
                if commit_progress:
                    await db.commit()
                return _serialize_run(run, [])
            resume_start_node_id = requested_resume_node
        else:
            for candidate in ordered_node_ids:
                prev = previous_nodes_by_id.get(candidate)
                if prev is None or prev.status != WorkflowNodeStatus.COMPLETED:
                    resume_start_node_id = candidate
                    break
        if resume_start_node_id:
            resume_start_index = ordered_node_ids.index(resume_start_node_id)
        else:
            resume_start_index = len(ordered_node_ids)

    deps_by_node = _dependency_map(edges)
    incoming_edges_by_target = _incoming_edges(edges)
    run_nodes: dict[str, WorkflowRunNode] = {}
    for node_index, node_id in enumerate(ordered_node_ids):
        node = node_map[node_id]
        runtime_check = evaluate_runtime_requirements_for_node(node)
        retry_policy = _normalize_retry_policy(
            global_max_retries=max_retries,
            run_config=cfg,
            node=node,
        )
        row = WorkflowRunNode(
            run_id=run.id,
            node_id=node_id,
            stage=str(node.get("stage") or ""),
            step_type=str(node.get("step_type") or ""),
            execution_backend=execution_backend,
            status=WorkflowNodeStatus.PENDING,
            attempt_count=0,
            max_retries=int(retry_policy.get("max_retries") or 0),
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
        if (
            reuse_successful_nodes
            and resume_source_run_id is not None
            and node_index < resume_start_index
        ):
            prev = previous_nodes_by_id.get(node_id)
            if prev is None or prev.status != WorkflowNodeStatus.COMPLETED:
                run.status = WorkflowRunStatus.FAILED
                run.finished_at = _utcnow()
                run.summary = {
                    **(run.summary or {}),
                    "runner_error": (
                        f"cannot resume from node '{resume_start_node_id or ''}': "
                        f"node '{node_id}' is not completed in run {resume_source_run_id}"
                    ),
                }
                await db.flush()
                if commit_progress:
                    await db.commit()
                return _serialize_run(run, [])
            row.status = WorkflowNodeStatus.COMPLETED
            row.attempt_count = int(prev.attempt_count or 0)
            row.published_artifact_keys = list(prev.published_artifact_keys or [])
            row.node_log = list(prev.node_log or [])
            row.error_message = f"reused from workflow run {resume_source_run_id}"
            row.started_at = prev.started_at or _utcnow()
            row.finished_at = prev.finished_at or _utcnow()
        db.add(row)
        run_nodes[node_id] = row

    await db.flush()
    if commit_progress:
        await db.commit()
    available_artifacts = await collect_available_artifacts(db, project.id)
    node_results: dict[str, dict[str, Any]] = {}
    for row in run_nodes.values():
        if row.status == WorkflowNodeStatus.COMPLETED:
            available_artifacts.update(
                [item for item in (row.published_artifact_keys or []) if isinstance(item, str) and item.strip()]
            )
            node_results[row.node_id] = {
                "status": row.status.value,
                "attempt_count": int(row.attempt_count or 0),
                "error_message": row.error_message or "",
                "published_artifact_keys": list(row.published_artifact_keys or []),
                "last_log": (list(row.node_log or [])[-1] if list(row.node_log or []) else {}),
            }

    run_blocked = False
    run_failed = False
    run_cancelled = False
    break_reason = ""
    executed_node_ids: set[str] = set()

    for node_id in ordered_node_ids:
        row = run_nodes[node_id]
        node = node_map[node_id]
        executed_node_ids.add(node_id)

        if await is_workflow_run_cancelled(db, project_id=project.id, run_id=run.id):
            run_cancelled = True
            break_reason = "run cancelled by user"
            break

        if row.status == WorkflowNodeStatus.COMPLETED and row.error_message.startswith("reused from workflow run"):
            continue

        node_condition = node.get("condition")
        if node_condition is not None and not _evaluate_program_condition(
            node_condition,
            source_node_id=None,
            available_artifacts=available_artifacts,
            run_config=cfg,
            node_results=node_results,
        ):
            row.status = WorkflowNodeStatus.SKIPPED
            row.error_message = "skipped by node condition"
            row.finished_at = _utcnow()
            node_results[node_id] = {
                "status": row.status.value,
                "attempt_count": int(row.attempt_count or 0),
                "error_message": row.error_message or "",
                "failure_type": "condition_skip",
                "last_log": {},
                "published_artifact_keys": list(row.published_artifact_keys or []),
            }
            if commit_progress:
                await db.commit()
            continue

        incoming_edges = list(incoming_edges_by_target.get(node_id) or [])
        active_deps: list[str] = []
        if incoming_edges:
            for edge in incoming_edges:
                source = str(edge.get("source") or "").strip()
                if not source:
                    continue
                condition_payload = _edge_condition_payload(edge)
                if _evaluate_program_condition(
                    condition_payload,
                    source_node_id=source,
                    available_artifacts=available_artifacts,
                    run_config=cfg,
                    node_results=node_results,
                ):
                    active_deps.append(source)
            active_deps = sorted({dep for dep in active_deps if dep in run_nodes})
            if not active_deps:
                row.dependencies = []
                row.status = WorkflowNodeStatus.SKIPPED
                row.error_message = "skipped by branch condition"
                row.finished_at = _utcnow()
                node_results[node_id] = {
                    "status": row.status.value,
                    "attempt_count": int(row.attempt_count or 0),
                    "error_message": row.error_message or "",
                    "failure_type": "branch_skip",
                    "last_log": {},
                    "published_artifact_keys": list(row.published_artifact_keys or []),
                }
                if commit_progress:
                    await db.commit()
                continue
            row.dependencies = active_deps

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
            node_results[node_id] = {
                "status": row.status.value,
                "attempt_count": int(row.attempt_count or 0),
                "error_message": row.error_message or "",
                "failure_type": "dependency_skip",
                "last_log": {},
                "published_artifact_keys": list(row.published_artifact_keys or []),
            }
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
            blocked_failure_type = _classify_failure_type(
                row.error_message,
                missing_inputs=missing_inputs,
                missing_runtime=missing_runtime,
            )
            row.node_log = [
                *list(row.node_log or []),
                {
                    "timestamp": _utcnow().isoformat(),
                    "backend": execution_backend,
                    "failure_type": blocked_failure_type,
                    "message": row.error_message,
                },
            ]
            node_results[node_id] = {
                "status": row.status.value,
                "attempt_count": int(row.attempt_count or 0),
                "error_message": row.error_message or "",
                "failure_type": blocked_failure_type,
                "last_log": (list(row.node_log or [])[-1] if list(row.node_log or []) else {}),
                "published_artifact_keys": list(row.published_artifact_keys or []),
            }
            run_blocked = True
            break_reason = row.error_message
            if commit_progress:
                await db.commit()
            if stop_on_blocked:
                break
            continue

        success = False
        last_error = ""
        last_failure_type = ""
        logs = list(row.node_log or [])
        total_attempts = 0
        best_success_payload: dict[str, Any] | None = None
        best_success_score: float | None = None
        loop_summaries: list[dict[str, Any]] = []

        async def _run_one_execution(
            *,
            execution_node: dict[str, Any],
            execution_cfg: dict[str, Any],
            retry_policy_payload: dict[str, Any],
            iteration_label: str | None = None,
        ) -> dict[str, Any]:
            nonlocal total_attempts, last_error, last_failure_type, run_cancelled, break_reason
            attempt = 0
            result_payload: dict[str, Any] = {}
            while attempt <= int(retry_policy_payload.get("max_retries") or 0):
                if await is_workflow_run_cancelled(db, project_id=project.id, run_id=run.id):
                    run_cancelled = True
                    last_error = "run cancelled by user"
                    last_failure_type = "cancelled"
                    break_reason = last_error
                    break
                attempt += 1
                total_attempts += 1
                row.attempt_count = total_attempts
                row.status = WorkflowNodeStatus.RUNNING
                if row.started_at is None:
                    row.started_at = _utcnow()

                ok, log_payload, error_text = await _execute_node_attempt(
                    db=db,
                    backend=execution_backend,
                    project_id=project.id,
                    run_id=run.id,
                    node=execution_node,
                    attempt=attempt,
                    config=execution_cfg,
                )
                failure_type = "" if ok else _classify_failure_type(error_text or "")
                should_retry = (not ok) and _should_retry_failure(
                    failure_type=failure_type,
                    attempt=attempt,
                    policy=retry_policy_payload,
                )
                logs.append(
                    {
                        "attempt": attempt,
                        "attempt_global": total_attempts,
                        "timestamp": _utcnow().isoformat(),
                        "backend": execution_backend,
                        "iteration": iteration_label,
                        "failure_type": failure_type or None,
                        "retry_planned": bool(should_retry),
                        "retry_policy": retry_policy_payload,
                        **log_payload,
                    }
                )
                row.node_log = logs
                result_payload = dict(log_payload or {})
                if ok:
                    return {
                        "ok": True,
                        "attempts": attempt,
                        "payload": result_payload,
                        "error": "",
                        "failure_type": "",
                    }

                last_error = error_text or "node execution failed"
                last_failure_type = failure_type or "execution"
                if not should_retry:
                    break

                backoff_seconds = _to_float(retry_policy_payload.get("backoff_seconds")) or 0.0
                backoff_multiplier = _to_float(retry_policy_payload.get("backoff_multiplier")) or 1.0
                if backoff_seconds > 0:
                    delay = backoff_seconds * (backoff_multiplier ** (attempt - 1))
                    await asyncio.sleep(max(0.0, min(delay, 120.0)))

            return {
                "ok": False,
                "attempts": attempt,
                "payload": result_payload,
                "error": last_error,
                "failure_type": last_failure_type or "execution",
            }

        base_retry_policy = _normalize_retry_policy(
            global_max_retries=max_retries,
            run_config=cfg,
            node=node,
        )
        row.max_retries = int(base_retry_policy.get("max_retries") or 0)

        loop_policy = _normalize_loop_policy(node)
        if loop_policy:
            objective_path = str(loop_policy.get("objective_path") or "").strip() or None
            objective_mode = str(loop_policy.get("objective_mode") or "max").strip().lower() or "max"
            stop_on_first_success = bool(loop_policy.get("stop_on_first_success"))
            require_all_success = bool(loop_policy.get("require_all_success"))
            success_count = 0
            fail_count = 0

            for item in list(loop_policy.get("items") or []):
                if run_cancelled:
                    break
                iteration_label = str(item.get("label") or "").strip() or f"trial-{len(loop_summaries) + 1}"
                execution_node = copy.deepcopy(node)
                base_node_cfg = execution_node.get("config")
                if not isinstance(base_node_cfg, dict):
                    base_node_cfg = {}
                node_cfg_override = item.get("node_config")
                if isinstance(node_cfg_override, dict):
                    execution_node["config"] = _deep_merge_dict(base_node_cfg, node_cfg_override)
                else:
                    execution_node["config"] = dict(base_node_cfg)

                execution_cfg = dict(cfg)
                run_cfg_override = item.get("run_config")
                if isinstance(run_cfg_override, dict):
                    execution_cfg = _deep_merge_dict(execution_cfg, run_cfg_override)

                retry_policy = dict(base_retry_policy)
                item_retry = item.get("retry_policy")
                if isinstance(item_retry, dict):
                    retry_policy = _deep_merge_dict(retry_policy, item_retry)
                    retry_policy = {
                        **retry_policy,
                        "max_retries": _coerce_int(retry_policy.get("max_retries"), default=0, minimum=0, maximum=20),
                        "backoff_seconds": _coerce_float(
                            retry_policy.get("backoff_seconds"), default=0.0, minimum=0.0, maximum=60.0
                        ),
                        "backoff_multiplier": _coerce_float(
                            retry_policy.get("backoff_multiplier"), default=1.0, minimum=1.0, maximum=10.0
                        ),
                    }

                iteration_result = await _run_one_execution(
                    execution_node=execution_node,
                    execution_cfg=execution_cfg,
                    retry_policy_payload=retry_policy,
                    iteration_label=iteration_label,
                )
                iteration_payload = iteration_result.get("payload") if isinstance(iteration_result.get("payload"), dict) else {}
                iteration_score = _extract_iteration_score(iteration_payload, objective_path)
                loop_summaries.append(
                    {
                        "label": iteration_label,
                        "success": bool(iteration_result.get("ok")),
                        "attempts": int(iteration_result.get("attempts") or 0),
                        "failure_type": str(iteration_result.get("failure_type") or ""),
                        "error": str(iteration_result.get("error") or ""),
                        "score": iteration_score,
                    }
                )

                if bool(iteration_result.get("ok")):
                    success_count += 1
                    if best_success_payload is None:
                        best_success_payload = dict(iteration_payload)
                        best_success_score = iteration_score
                    else:
                        if iteration_score is not None:
                            if best_success_score is None:
                                best_success_payload = dict(iteration_payload)
                                best_success_score = iteration_score
                            elif objective_mode == "min" and iteration_score < best_success_score:
                                best_success_payload = dict(iteration_payload)
                                best_success_score = iteration_score
                            elif objective_mode != "min" and iteration_score > best_success_score:
                                best_success_payload = dict(iteration_payload)
                                best_success_score = iteration_score
                    if stop_on_first_success:
                        break
                else:
                    fail_count += 1

            if run_cancelled:
                success = False
            elif require_all_success and fail_count > 0:
                success = False
                last_error = last_error or "loop policy require_all_success failed"
                last_failure_type = last_failure_type or "loop_failure"
            else:
                success = success_count > 0

            logs.append(
                {
                    "timestamp": _utcnow().isoformat(),
                    "backend": execution_backend,
                    "loop_summary": {
                        "type": loop_policy.get("type"),
                        "total_iterations": len(loop_summaries),
                        "successful_iterations": success_count,
                        "failed_iterations": fail_count,
                        "objective_path": objective_path,
                        "objective_mode": objective_mode,
                        "best_score": best_success_score,
                        "iterations": loop_summaries,
                    },
                }
            )
            row.node_log = logs
            if success and isinstance(best_success_payload, dict):
                # Expose selected payload in node_results and logs.
                logs.append(
                    {
                        "timestamp": _utcnow().isoformat(),
                        "backend": execution_backend,
                        "selected_iteration_payload": best_success_payload,
                    }
                )
                row.node_log = logs
        else:
            single_result = await _run_one_execution(
                execution_node=node,
                execution_cfg=cfg,
                retry_policy_payload=base_retry_policy,
                iteration_label=None,
            )
            success = bool(single_result.get("ok"))
            if isinstance(single_result.get("payload"), dict):
                best_success_payload = dict(single_result.get("payload") or {})
            if not success:
                last_error = str(single_result.get("error") or last_error or "node execution failed")
                last_failure_type = str(single_result.get("failure_type") or last_failure_type or "execution")

        if run_cancelled:
            row.status = WorkflowNodeStatus.SKIPPED
            row.error_message = last_error or "run cancelled by user"
            row.finished_at = _utcnow()
            node_results[node_id] = {
                "status": row.status.value,
                "attempt_count": int(row.attempt_count or 0),
                "error_message": row.error_message or "",
                "failure_type": "cancelled",
                "last_log": (list(row.node_log or [])[-1] if list(row.node_log or []) else {}),
                "published_artifact_keys": list(row.published_artifact_keys or []),
            }
            if commit_progress:
                await db.commit()
            break

        if not success:
            row.status = WorkflowNodeStatus.FAILED
            row.error_message = last_error
            row.finished_at = _utcnow()
            node_results[node_id] = {
                "status": row.status.value,
                "attempt_count": int(row.attempt_count or 0),
                "error_message": row.error_message or "",
                "failure_type": last_failure_type or "execution",
                "last_log": (list(row.node_log or [])[-1] if list(row.node_log or []) else {}),
                "published_artifact_keys": list(row.published_artifact_keys or []),
            }
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
            metadata={
                "source": "workflow.runner",
                "backend": execution_backend,
                "loop_enabled": bool(loop_policy),
                "loop_iterations": len(loop_summaries),
            },
        )
        row.published_artifact_keys = [item.artifact_key for item in published]
        available_artifacts.update(row.published_artifact_keys)
        row.status = WorkflowNodeStatus.COMPLETED
        row.error_message = ""
        row.finished_at = _utcnow()
        node_results[node_id] = {
            "status": row.status.value,
            "attempt_count": int(row.attempt_count or 0),
            "error_message": "",
            "failure_type": "",
            "last_log": (list(row.node_log or [])[-1] if list(row.node_log or []) else {}),
            "published_artifact_keys": list(row.published_artifact_keys or []),
            "loop_summary": loop_summaries,
            "selected_payload": best_success_payload or {},
        }
        if commit_progress:
            await db.commit()

    for node_id in ordered_node_ids:
        if node_id in executed_node_ids:
            continue
        row = run_nodes[node_id]
        if row.status == WorkflowNodeStatus.PENDING:
            row.status = WorkflowNodeStatus.SKIPPED
            if run_cancelled:
                row.error_message = "skipped after workflow cancellation"
            else:
                row.error_message = "skipped after workflow termination"
            row.finished_at = _utcnow()
            node_results[node_id] = {
                "status": row.status.value,
                "attempt_count": int(row.attempt_count or 0),
                "error_message": row.error_message or "",
                "failure_type": "termination_skip",
                "last_log": (list(row.node_log or [])[-1] if list(row.node_log or []) else {}),
                "published_artifact_keys": list(row.published_artifact_keys or []),
            }
            if commit_progress:
                await db.commit()

    if run_failed:
        run.status = WorkflowRunStatus.FAILED
    elif run_cancelled:
        run.status = WorkflowRunStatus.CANCELLED
    elif run_blocked:
        run.status = WorkflowRunStatus.BLOCKED
    else:
        run.status = WorkflowRunStatus.COMPLETED

    run.finished_at = _utcnow()
    run.summary = {
        **(run.summary or {}),
        "available_artifacts_final": sorted(available_artifacts),
        "break_reason": break_reason,
        "program_node_results": node_results,
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

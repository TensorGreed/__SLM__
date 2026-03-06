"""Workflow DAG runner service with persistent run/node tracking."""

from __future__ import annotations

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


def _execute_node_attempt(
    *,
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
        return True, {"message": f"local backend completed {stage}"}, ""

    if backend == "celery":
        return _execute_celery_node_attempt(
            project_id=project_id,
            run_id=run_id,
            node_id=node_id,
            stage=stage,
            step_type=step_type,
            attempt=attempt,
            config=config,
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
            ok, log_payload, error_text = _execute_node_attempt(
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

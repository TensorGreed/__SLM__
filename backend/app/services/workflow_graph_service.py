"""Workflow graph and contract-runtime helpers for pipeline execution."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.models.export import Export, ExportStatus
from app.models.project import PipelineStage, Project
from app.pipeline.orchestrator import STAGE_ORDER, get_pipeline_status


DEFAULT_GRAPH_ID = "default-linear-v1"
DEFAULT_GRAPH_LABEL = "Default SLM Pipeline"
DEFAULT_GRAPH_VERSION = "1.0.0"
SOURCE_ARTIFACTS = {"source.file", "source.remote_dataset"}


_STEP_CONTRACTS: dict[str, dict[str, Any]] = {
    "ingestion": {
        "step_type": "core.ingestion",
        "description": "Ingest local files or remote datasets into project storage.",
        "input_artifacts": ["source.file", "source.remote_dataset"],
        "output_artifacts": ["dataset.raw"],
        "config_schema_ref": "slm.step.ingestion/v1",
    },
    "cleaning": {
        "step_type": "core.cleaning",
        "description": "Normalize and clean raw documents into structured samples/chunks.",
        "input_artifacts": ["dataset.raw"],
        "output_artifacts": ["dataset.cleaned", "dataset.chunks"],
        "config_schema_ref": "slm.step.cleaning/v1",
    },
    "gold_set": {
        "step_type": "core.gold_set",
        "description": "Curate trusted evaluation examples used for quality tracking.",
        "input_artifacts": ["dataset.cleaned", "dataset.raw"],
        "output_artifacts": ["dataset.gold_dev", "dataset.gold_test"],
        "config_schema_ref": "slm.step.gold_set/v1",
    },
    "synthetic": {
        "step_type": "core.synthetic",
        "description": "Generate synthetic instruction data from cleaned source text.",
        "input_artifacts": ["dataset.cleaned", "dataset.chunks"],
        "output_artifacts": ["dataset.synthetic"],
        "config_schema_ref": "slm.step.synthetic/v1",
    },
    "dataset_prep": {
        "step_type": "core.dataset_prep",
        "description": "Merge sources and split train/validation/test datasets.",
        "input_artifacts": ["dataset.cleaned", "dataset.synthetic", "dataset.gold_dev"],
        "output_artifacts": ["dataset.train", "dataset.validation", "dataset.test"],
        "config_schema_ref": "slm.step.dataset_prep/v1",
    },
    "tokenization": {
        "step_type": "core.tokenization",
        "description": "Analyze token lengths and tokenizer coverage for target model.",
        "input_artifacts": ["dataset.train", "dataset.validation"],
        "output_artifacts": ["analysis.tokenization"],
        "config_schema_ref": "slm.step.tokenization/v1",
    },
    "training": {
        "step_type": "core.training",
        "description": "Run fine-tuning experiment(s) with configured runtime backend.",
        "input_artifacts": ["dataset.train", "dataset.validation"],
        "output_artifacts": ["model.checkpoint", "report.training"],
        "config_schema_ref": "slm.step.training/v1",
    },
    "evaluation": {
        "step_type": "core.evaluation",
        "description": "Evaluate trained model on held-out and gold datasets.",
        "input_artifacts": ["model.checkpoint", "dataset.gold_dev", "dataset.test"],
        "output_artifacts": ["report.evaluation"],
        "config_schema_ref": "slm.step.evaluation/v1",
    },
    "compression": {
        "step_type": "core.compression",
        "description": "Quantize/merge/benchmark model artifacts for deployment.",
        "input_artifacts": ["model.checkpoint", "report.evaluation"],
        "output_artifacts": ["model.compressed", "report.compression"],
        "config_schema_ref": "slm.step.compression/v1",
    },
    "export": {
        "step_type": "core.export",
        "description": "Bundle model, manifests, and metadata for release.",
        "input_artifacts": ["model.compressed", "report.compression"],
        "output_artifacts": ["package.export_bundle"],
        "config_schema_ref": "slm.step.export/v1",
    },
}


def _status_by_stage(current_stage: PipelineStage) -> dict[str, str]:
    rows = get_pipeline_status(current_stage)
    return {row["stage"]: row["status"] for row in rows}


def _build_default_nodes(current_stage: PipelineStage) -> list[dict[str, Any]]:
    status_by_stage = _status_by_stage(current_stage)
    stages = [stage for stage in STAGE_ORDER if stage.value in _STEP_CONTRACTS]
    nodes: list[dict[str, Any]] = []
    for index, stage in enumerate(stages):
        stage_name = stage.value
        contract = _STEP_CONTRACTS[stage_name]
        nodes.append(
            {
                "id": f"step:{stage_name}",
                "stage": stage_name,
                "display_name": stage_name.replace("_", " ").title(),
                "index": index,
                "kind": "core_step",
                "status": status_by_stage.get(stage_name, "pending"),
                "position": {"x": index * 280, "y": 0},
                **contract,
            }
        )
    return nodes


def _build_default_edges() -> list[dict[str, str]]:
    stages = [stage.value for stage in STAGE_ORDER if stage.value in _STEP_CONTRACTS]
    edges: list[dict[str, str]] = []
    for index, stage_name in enumerate(stages):
        if index == 0:
            continue
        prev_stage = stages[index - 1]
        edges.append(
            {
                "id": f"edge:{prev_stage}->{stage_name}",
                "source": f"step:{prev_stage}",
                "target": f"step:{stage_name}",
                "kind": "sequential",
            }
        )
    return edges


def build_readonly_pipeline_graph(project_id: int, current_stage: PipelineStage) -> dict[str, Any]:
    """Build the default read-only workflow graph."""
    return {
        "project_id": project_id,
        "graph_id": DEFAULT_GRAPH_ID,
        "graph_label": DEFAULT_GRAPH_LABEL,
        "graph_version": DEFAULT_GRAPH_VERSION,
        "mode": "readonly_preview",
        "current_stage": current_stage.value,
        "nodes": _build_default_nodes(current_stage),
        "edges": _build_default_edges(),
    }


def _workflow_graph_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "workflow_graph"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _workflow_graph_contract_path(project_id: int) -> Path:
    return _workflow_graph_dir(project_id) / "contract.json"


def load_saved_workflow_graph_override(project_id: int) -> dict[str, Any] | None:
    """Load persisted workflow graph override for a project, if present."""
    path = _workflow_graph_contract_path(project_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("graph"), dict):
        maybe_graph = payload["graph"]
        if isinstance(maybe_graph, dict):
            return maybe_graph
    return payload


def save_workflow_graph_override(project_id: int, graph: dict[str, Any]) -> str:
    """Persist workflow graph override for a project."""
    path = _workflow_graph_contract_path(project_id)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "graph": graph,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path)


def delete_workflow_graph_override(project_id: int) -> bool:
    """Delete persisted workflow graph override for a project."""
    path = _workflow_graph_contract_path(project_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def get_step_contract_catalog() -> list[dict[str, Any]]:
    """Expose built-in stage contracts for editor/tooling UIs."""
    stages = [stage.value for stage in STAGE_ORDER if stage.value in _STEP_CONTRACTS]
    catalog: list[dict[str, Any]] = []
    for index, stage_name in enumerate(stages):
        contract = _STEP_CONTRACTS[stage_name]
        catalog.append(
            {
                "stage": stage_name,
                "display_name": stage_name.replace("_", " ").title(),
                "index": index,
                **contract,
            }
        )
    return catalog


def resolve_project_workflow_graph(
    project_id: int,
    current_stage: PipelineStage,
    graph_override: dict[str, Any] | None = None,
    allow_fallback: bool = True,
    use_saved_override: bool = True,
) -> dict[str, Any]:
    """Resolve graph by request override, saved override, then platform default."""
    requested_source = "default"
    candidate_override = graph_override

    if candidate_override is not None:
        requested_source = "request_override"
    elif use_saved_override:
        saved = load_saved_workflow_graph_override(project_id)
        if saved is not None:
            candidate_override = saved
            requested_source = "saved_override"

    resolved = resolve_workflow_graph(
        project_id=project_id,
        current_stage=current_stage,
        graph_override=candidate_override,
        allow_fallback=allow_fallback,
    )

    effective_source = requested_source
    if resolved.get("fallback_used"):
        effective_source = "default_fallback"
    elif requested_source == "default":
        effective_source = "default"

    return {
        **resolved,
        "requested_source": requested_source,
        "effective_source": effective_source,
        "has_saved_override": load_saved_workflow_graph_override(project_id) is not None,
    }


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_artifact_list(value: object) -> bool:
    return isinstance(value, list) and all(_is_nonempty_str(item) for item in value)


def _graph_has_cycle(node_ids: set[str], edges: list[dict[str, Any]]) -> bool:
    graph: dict[str, list[str]] = defaultdict(list)
    in_degree = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source in node_ids and target in node_ids:
            graph[source].append(target)
            in_degree[target] += 1

    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for neighbor in graph.get(node_id, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited != len(node_ids)


def _topological_order(node_ids: set[str], edges: list[dict[str, Any]]) -> list[str]:
    graph: dict[str, list[str]] = defaultdict(list)
    in_degree = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source in node_ids and target in node_ids:
            graph[source].append(target)
            in_degree[target] += 1

    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    ordered: list[str] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_id)
        for neighbor in graph.get(node_id, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return ordered


def _validate_override_graph_structure(graph: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes:
        errors.append("graph.nodes must be a non-empty list")
        return errors, warnings
    if not isinstance(edges, list):
        errors.append("graph.edges must be a list")
        return errors, warnings

    node_ids: set[str] = set()
    stage_names: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] must be an object")
            continue

        node_id = str(node.get("id", "")).strip()
        stage_name = str(node.get("stage", "")).strip()
        step_type = node.get("step_type")
        input_artifacts = node.get("input_artifacts")
        output_artifacts = node.get("output_artifacts")
        config_schema_ref = node.get("config_schema_ref")

        if not node_id:
            errors.append(f"nodes[{index}].id is required")
        elif node_id in node_ids:
            errors.append(f"Duplicate node id: {node_id}")
        else:
            node_ids.add(node_id)

        if not stage_name:
            errors.append(f"nodes[{index}].stage is required")
        elif stage_name in stage_names:
            errors.append(f"Duplicate stage in graph: {stage_name}")
        else:
            stage_names.add(stage_name)
            if stage_name not in _STEP_CONTRACTS:
                errors.append(f"nodes[{index}].stage '{stage_name}' is not supported by core contracts")

        if not _is_nonempty_str(step_type):
            errors.append(f"nodes[{index}].step_type is required")
        if not _is_nonempty_str(config_schema_ref):
            errors.append(f"nodes[{index}].config_schema_ref is required")
        if not _is_artifact_list(input_artifacts):
            errors.append(f"nodes[{index}].input_artifacts must be a list of non-empty strings")
        if not _is_artifact_list(output_artifacts):
            errors.append(f"nodes[{index}].output_artifacts must be a list of non-empty strings")

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edges[{index}] must be an object")
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source:
            errors.append(f"edges[{index}].source is required")
        if not target:
            errors.append(f"edges[{index}].target is required")
        if source and source not in node_ids:
            errors.append(f"edges[{index}].source '{source}' does not reference an existing node")
        if target and target not in node_ids:
            errors.append(f"edges[{index}].target '{target}' does not reference an existing node")

    if not errors and _graph_has_cycle(node_ids, edges):
        errors.append("graph must be acyclic (cycle detected)")

    if isinstance(graph.get("mode"), str) and graph.get("mode") != "readonly_preview":
        warnings.append("graph.mode is ignored for runtime and resolved as readonly_preview in phase 2")

    return errors, warnings


def _normalize_override_graph(
    project_id: int,
    current_stage: PipelineStage,
    graph: dict[str, Any],
) -> dict[str, Any]:
    status_by_stage = _status_by_stage(current_stage)
    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get("edges", [])

    nodes: list[dict[str, Any]] = []
    for fallback_index, node in enumerate(raw_nodes):
        stage_name = str(node.get("stage", "")).strip()
        index_value = node.get("index")
        index = int(index_value) if isinstance(index_value, int) else fallback_index
        position = node.get("position")
        x = position.get("x") if isinstance(position, dict) else None
        y = position.get("y") if isinstance(position, dict) else None
        resolved_position = {
            "x": int(x) if isinstance(x, int) else index * 280,
            "y": int(y) if isinstance(y, int) else 0,
        }
        nodes.append(
            {
                **node,
                "display_name": str(node.get("display_name") or stage_name.replace("_", " ").title()),
                "index": index,
                "kind": str(node.get("kind") or "custom_step"),
                "status": status_by_stage.get(stage_name, "pending"),
                "position": resolved_position,
            }
        )

    edges: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(raw_edges):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        edge_id = str(edge.get("id", "")).strip() or f"edge:{source}->{target}:{edge_index}"
        edges.append(
            {
                **edge,
                "id": edge_id,
                "kind": str(edge.get("kind") or "contract_edge"),
                "source": source,
                "target": target,
            }
        )

    nodes.sort(key=lambda item: int(item.get("index", 0)))
    return {
        "project_id": project_id,
        "graph_id": str(graph.get("graph_id") or f"custom-{DEFAULT_GRAPH_ID}"),
        "graph_label": str(graph.get("graph_label") or "Custom Pipeline Graph"),
        "graph_version": str(graph.get("graph_version") or DEFAULT_GRAPH_VERSION),
        "mode": "readonly_preview",
        "current_stage": current_stage.value,
        "nodes": nodes,
        "edges": edges,
    }


def resolve_workflow_graph(
    project_id: int,
    current_stage: PipelineStage,
    graph_override: dict[str, Any] | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Resolve a runtime graph contract, with default fallback when requested."""
    default_graph = build_readonly_pipeline_graph(project_id, current_stage)
    if graph_override is None:
        return {
            "valid": True,
            "fallback_used": False,
            "errors": [],
            "warnings": [],
            "graph": default_graph,
        }

    if not isinstance(graph_override, dict):
        errors = ["graph must be an object"]
        if allow_fallback:
            return {
                "valid": True,
                "fallback_used": True,
                "errors": errors,
                "warnings": ["Invalid graph override provided; default graph fallback applied."],
                "graph": default_graph,
            }
        return {
            "valid": False,
            "fallback_used": False,
            "errors": errors,
            "warnings": [],
            "graph": graph_override,
        }

    errors, warnings = _validate_override_graph_structure(graph_override)
    if errors:
        if allow_fallback:
            return {
                "valid": True,
                "fallback_used": True,
                "errors": errors,
                "warnings": [*warnings, "Invalid graph override provided; default graph fallback applied."],
                "graph": default_graph,
            }
        return {
            "valid": False,
            "fallback_used": False,
            "errors": errors,
            "warnings": warnings,
            "graph": graph_override,
        }

    return {
        "valid": True,
        "fallback_used": False,
        "errors": [],
        "warnings": warnings,
        "graph": _normalize_override_graph(project_id, current_stage, graph_override),
    }


async def _dataset_exists_with_records(
    db: AsyncSession,
    project_id: int,
    dataset_type: DatasetType,
) -> bool:
    result = await db.execute(
        select(Dataset.id)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == dataset_type,
            Dataset.record_count > 0,
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def collect_available_artifacts(db: AsyncSession, project_id: int) -> set[str]:
    """Infer currently materialized artifacts for a project."""
    artifacts: set[str] = set()

    raw_doc_result = await db.execute(
        select(RawDocument.id)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
            RawDocument.status == DocumentStatus.ACCEPTED,
        )
        .limit(1)
    )
    if raw_doc_result.scalar_one_or_none() is not None:
        artifacts.update({"source.file", "source.remote_dataset", "dataset.raw"})

    chunk_result = await db.execute(
        select(RawDocument.id)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
            RawDocument.chunk_count > 0,
        )
        .limit(1)
    )
    if chunk_result.scalar_one_or_none() is not None:
        artifacts.add("dataset.chunks")

    if await _dataset_exists_with_records(db, project_id, DatasetType.CLEANED):
        artifacts.add("dataset.cleaned")
    if await _dataset_exists_with_records(db, project_id, DatasetType.GOLD_DEV):
        artifacts.add("dataset.gold_dev")
    if await _dataset_exists_with_records(db, project_id, DatasetType.GOLD_TEST):
        artifacts.add("dataset.gold_test")
    if await _dataset_exists_with_records(db, project_id, DatasetType.SYNTHETIC):
        artifacts.add("dataset.synthetic")
    if await _dataset_exists_with_records(db, project_id, DatasetType.TRAIN):
        artifacts.add("dataset.train")
    if await _dataset_exists_with_records(db, project_id, DatasetType.VALIDATION):
        artifacts.add("dataset.validation")
    if await _dataset_exists_with_records(db, project_id, DatasetType.TEST):
        artifacts.add("dataset.test")

    tokenizer_dir = settings.DATA_DIR / "projects" / str(project_id) / "tokenization"
    if tokenizer_dir.exists() and any(path.suffix == ".json" for path in tokenizer_dir.rglob("*.json")):
        artifacts.add("analysis.tokenization")

    completed_exp_result = await db.execute(
        select(Experiment)
        .where(
            Experiment.project_id == project_id,
            Experiment.status == ExperimentStatus.COMPLETED,
        )
        .order_by(Experiment.id.desc())
        .limit(1)
    )
    completed_experiment = completed_exp_result.scalar_one_or_none()
    if completed_experiment is not None:
        artifacts.add("report.training")
        artifacts.add("model.checkpoint")

    eval_result = await db.execute(
        select(EvalResult.id)
        .join(Experiment, Experiment.id == EvalResult.experiment_id)
        .where(Experiment.project_id == project_id)
        .limit(1)
    )
    if eval_result.scalar_one_or_none() is not None:
        artifacts.add("report.evaluation")

    compressed_dir = settings.DATA_DIR / "projects" / str(project_id) / "compressed"
    if compressed_dir.exists():
        report_exists = any(
            path.suffix == ".json" and path.name.endswith("_report.json")
            for path in compressed_dir.rglob("*")
            if path.is_file()
        )
        model_exists = any(
            path.is_file() and not (path.suffix == ".json" and path.name.endswith("_report.json"))
            for path in compressed_dir.rglob("*")
        )
        if report_exists:
            artifacts.add("report.compression")
        if model_exists:
            artifacts.add("model.compressed")

    export_result = await db.execute(
        select(Export.id)
        .where(
            Export.project_id == project_id,
            Export.status == ExportStatus.COMPLETED,
        )
        .limit(1)
    )
    if export_result.scalar_one_or_none() is not None:
        artifacts.add("package.export_bundle")

    return artifacts


def _find_graph_node_for_stage(graph: dict[str, Any], stage: PipelineStage) -> dict[str, Any] | None:
    for node in graph.get("nodes", []):
        if isinstance(node, dict) and node.get("stage") == stage.value:
            return node
    return None


def _missing_inputs_for_node(node: dict[str, Any], available_artifacts: set[str]) -> list[str]:
    required = node.get("input_artifacts", [])
    if not isinstance(required, list):
        return []
    return sorted(
        {
            str(artifact).strip()
            for artifact in required
            if isinstance(artifact, str) and artifact.strip() and artifact not in available_artifacts
        }
    )


async def compile_workflow_graph(
    db: AsyncSession,
    project: Project,
    graph_override: dict[str, Any] | None = None,
    allow_fallback: bool = True,
    use_saved_override: bool = True,
) -> dict[str, Any]:
    """Compile graph contract with structural and artifact-flow diagnostics."""
    resolved = resolve_project_workflow_graph(
        project_id=project.id,
        current_stage=project.pipeline_stage,
        graph_override=graph_override,
        allow_fallback=allow_fallback,
        use_saved_override=use_saved_override,
    )
    available_artifacts = await collect_available_artifacts(db, project.id)
    graph = resolved.get("graph")
    errors = list(resolved.get("errors", []))
    warnings = list(resolved.get("warnings", []))

    active_stage = project.pipeline_stage.value
    active_node_id = ""
    active_missing_inputs: list[str] = []
    stage_present = False

    if isinstance(graph, dict):
        nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
        edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
        node_id_to_node = {str(node.get("id", "")).strip(): node for node in nodes if str(node.get("id", "")).strip()}
        ordered_ids = _topological_order(set(node_id_to_node.keys()), edges)
        if ordered_ids and len(ordered_ids) != len(node_id_to_node):
            errors.append("Compile failed: graph ordering incomplete due to invalid connectivity.")

        produced_artifacts = set(SOURCE_ARTIFACTS)
        if ordered_ids:
            for node_id in ordered_ids:
                node = node_id_to_node[node_id]
                stage_name = str(node.get("stage", "")).strip()
                inputs = [item for item in node.get("input_artifacts", []) if isinstance(item, str) and item.strip()]
                outputs = [item for item in node.get("output_artifacts", []) if isinstance(item, str) and item.strip()]
                upstream_missing = sorted({artifact for artifact in inputs if artifact not in produced_artifacts})
                if upstream_missing:
                    warnings.append(
                        f"{node_id} ({stage_name}) declares inputs not produced upstream: {', '.join(upstream_missing)}"
                    )
                produced_artifacts.update(outputs)

                if stage_name == active_stage:
                    stage_present = True
                    active_node_id = node_id
                    active_missing_inputs = _missing_inputs_for_node(node, available_artifacts)
        else:
            stage_present = any(str(node.get("stage", "")).strip() == active_stage for node in nodes)
            if stage_present:
                node = next(
                    node for node in nodes if str(node.get("stage", "")).strip() == active_stage
                )
                active_node_id = str(node.get("id", "")).strip()
                active_missing_inputs = _missing_inputs_for_node(node, available_artifacts)

        if not stage_present:
            errors.append(f"Current project stage '{active_stage}' is not present in resolved graph.")

    return {
        "project_id": project.id,
        "current_stage": active_stage,
        "valid_graph": bool(resolved.get("valid")),
        "fallback_used": bool(resolved.get("fallback_used")),
        "requested_source": resolved.get("requested_source"),
        "effective_source": resolved.get("effective_source"),
        "has_saved_override": bool(resolved.get("has_saved_override")),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "checks": {
            "active_stage_present": stage_present,
            "active_stage_node_id": active_node_id or None,
            "active_stage_missing_inputs": active_missing_inputs,
            "active_stage_ready_now": not active_missing_inputs and stage_present,
        },
        "available_artifacts": sorted(available_artifacts),
        "graph": graph,
    }


async def build_workflow_dry_run(
    db: AsyncSession,
    project: Project,
    graph_override: dict[str, Any] | None = None,
    allow_fallback: bool = True,
    use_saved_override: bool = True,
) -> dict[str, Any]:
    """Create an execution preview against currently materialized artifacts."""
    resolved = resolve_project_workflow_graph(
        project_id=project.id,
        current_stage=project.pipeline_stage,
        graph_override=graph_override,
        allow_fallback=allow_fallback,
        use_saved_override=use_saved_override,
    )
    available_artifacts = await collect_available_artifacts(db, project.id)

    graph = resolved.get("graph")
    node_plan: list[dict[str, Any]] = []
    active_stage = project.pipeline_stage.value

    if isinstance(graph, dict):
        nodes = graph.get("nodes", [])
        if isinstance(nodes, list):
            ordered = sorted(
                [node for node in nodes if isinstance(node, dict)],
                key=lambda item: int(item.get("index", 0)),
            )
            for node in ordered:
                missing_inputs = _missing_inputs_for_node(node, available_artifacts)
                stage_name = str(node.get("stage", "")).strip()
                node_plan.append(
                    {
                        "id": str(node.get("id", "")).strip(),
                        "stage": stage_name,
                        "status": str(node.get("status", "pending")),
                        "can_run_now": not missing_inputs and stage_name == active_stage,
                        "missing_inputs": missing_inputs,
                        "input_artifacts": list(node.get("input_artifacts", [])),
                        "output_artifacts": list(node.get("output_artifacts", [])),
                    }
                )

    active_node = next((item for item in node_plan if item.get("stage") == active_stage), None)
    return {
        "project_id": project.id,
        "current_stage": active_stage,
        "valid_graph": bool(resolved.get("valid")),
        "fallback_used": bool(resolved.get("fallback_used")),
        "requested_source": resolved.get("requested_source"),
        "effective_source": resolved.get("effective_source"),
        "has_saved_override": bool(resolved.get("has_saved_override")),
        "errors": list(resolved.get("errors", [])),
        "warnings": list(resolved.get("warnings", [])),
        "available_artifacts": sorted(available_artifacts),
        "active_step": active_node,
        "plan": node_plan,
        "graph": graph,
    }


def _pipeline_run_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "pipeline_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def persist_pipeline_run_record(project_id: int, payload: dict[str, Any]) -> str:
    """Persist a pipeline step-run payload to project-local storage."""
    run_id = str(payload.get("run_id") or uuid4().hex)
    safe_run_id = "".join(ch for ch in run_id if ch.isalnum() or ch in {"-", "_"})
    if not safe_run_id:
        safe_run_id = uuid4().hex

    run_path = _pipeline_run_dir(project_id) / f"{safe_run_id}.json"
    record = dict(payload)
    record["run_id"] = safe_run_id
    run_path.write_text(json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(run_path)


def list_pipeline_run_records(project_id: int, limit: int = 20) -> list[dict[str, Any]]:
    """Return most recent persisted step-run records for a project."""
    run_dir = _pipeline_run_dir(project_id)
    files = sorted(
        [path for path in run_dir.glob("*.json") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    records: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload.setdefault("run_record_path", str(path))
            records.append(payload)
    return records


async def prepare_workflow_step_run(
    db: AsyncSession,
    project: Project,
    stage: PipelineStage,
    graph_override: dict[str, Any] | None = None,
    allow_fallback: bool = True,
    use_saved_override: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare a single-step execution payload, without mutating project stage."""
    resolved = resolve_project_workflow_graph(
        project_id=project.id,
        current_stage=project.pipeline_stage,
        graph_override=graph_override,
        allow_fallback=allow_fallback,
        use_saved_override=use_saved_override,
    )
    available_artifacts = await collect_available_artifacts(db, project.id)
    graph = resolved.get("graph") if isinstance(resolved.get("graph"), dict) else {}
    node = _find_graph_node_for_stage(graph, stage)

    now = datetime.now(timezone.utc).isoformat()
    base_payload = {
        "run_id": uuid4().hex,
        "run_started_at": now,
        "project_id": project.id,
        "requested_stage": stage.value,
        "current_stage": project.pipeline_stage.value,
        "valid_graph": bool(resolved.get("valid")),
        "fallback_used": bool(resolved.get("fallback_used")),
        "requested_source": resolved.get("requested_source"),
        "effective_source": resolved.get("effective_source"),
        "has_saved_override": bool(resolved.get("has_saved_override")),
        "errors": list(resolved.get("errors", [])),
        "warnings": list(resolved.get("warnings", [])),
        "config": dict(config or {}),
        "available_artifacts": sorted(available_artifacts),
    }

    if not resolved.get("valid"):
        base_payload.update(
            {
                "status": "invalid_graph",
                "declared_inputs": [],
                "declared_outputs": [],
                "missing_inputs": [],
                "can_execute": False,
            }
        )
        return base_payload

    if stage != project.pipeline_stage:
        base_payload.update(
            {
                "status": "blocked",
                "errors": [
                    *base_payload["errors"],
                    (
                        f"Only active stage can run in phase 2. "
                        f"requested={stage.value}, active={project.pipeline_stage.value}"
                    ),
                ],
                "declared_inputs": [],
                "declared_outputs": [],
                "missing_inputs": [],
                "can_execute": False,
            }
        )
        return base_payload

    if node is None:
        base_payload.update(
            {
                "status": "blocked",
                "errors": [*base_payload["errors"], f"Stage '{stage.value}' not found in resolved graph."],
                "declared_inputs": [],
                "declared_outputs": [],
                "missing_inputs": [],
                "can_execute": False,
            }
        )
        return base_payload

    declared_inputs = [str(item) for item in node.get("input_artifacts", []) if isinstance(item, str)]
    declared_outputs = [str(item) for item in node.get("output_artifacts", []) if isinstance(item, str)]
    missing_inputs = _missing_inputs_for_node(node, available_artifacts)
    can_execute = not missing_inputs

    base_payload.update(
        {
            "status": "ready" if can_execute else "blocked",
            "step_node_id": str(node.get("id", "")).strip(),
            "step_type": str(node.get("step_type", "")),
            "declared_inputs": declared_inputs,
            "declared_outputs": declared_outputs,
            "missing_inputs": missing_inputs,
            "can_execute": can_execute,
        }
    )
    return base_payload

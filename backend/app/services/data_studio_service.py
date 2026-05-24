"""Data Studio overview intelligence.

The Data Studio is an additive UX layer over the existing pipeline.
This service keeps the first slice deliberately deterministic: it
summarizes project data state, computes simple readiness issues, and
returns action targets the frontend can route to existing panels.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.project import Project
from app.services.dataset_service import (
    preview_project_data_adapter,
    resolve_project_dataset_adapter_preference,
)
from app.services.domain_runtime_service import resolve_project_domain_runtime
from app.services.recipe_service import get_recipe
from app.services.synth_review_queue_service import list_review_queue


IssueSeverity = Literal["blocker", "warning", "info"]
OverviewVerdict = Literal["blocked", "needs_work", "ready"]
SourcesVerdict = Literal["empty", "attention", "healthy"]
MappingVerdict = Literal["empty", "attention", "ready"]

_MAPPING_SOURCE_PRIORITY: tuple[DatasetType, ...] = (
    DatasetType.RAW,
    DatasetType.GOLD_DEV,
    DatasetType.SYNTHETIC,
    DatasetType.CLEANED,
    DatasetType.GOLD_TEST,
    DatasetType.TRAIN,
    DatasetType.VALIDATION,
    DatasetType.TEST,
)


def _sum_counts(counter: Counter[str], *types: DatasetType) -> int:
    return sum(int(counter.get(t.value, 0)) for t in types)


def _issue(
    issue_id: str,
    severity: IssueSeverity,
    title: str,
    message: str,
    *,
    action_label: str,
    target_tab: str,
) -> dict[str, str]:
    return {
        "id": issue_id,
        "severity": severity,
        "title": title,
        "message": message,
        "action_label": action_label,
        "target_tab": target_tab,
    }


def _primary_action(
    issues: list[dict[str, str]],
    *,
    prepared_rows: int,
) -> dict[str, str]:
    if issues:
        first = issues[0]
        return {
            "label": first["action_label"],
            "target_tab": first["target_tab"],
            "reason": first["title"],
        }
    if prepared_rows > 0:
        return {
            "label": "Open training",
            "target_tab": "training",
            "reason": "A prepared dataset is available.",
        }
    return {
        "label": "Prepare dataset",
        "target_tab": "dataprep",
        "reason": "Data looks usable; create train/validation/test splits next.",
    }


def _recipe_payload(project: Project) -> dict[str, Any] | None:
    selected_recipe = project.selected_recipe if isinstance(project.selected_recipe, dict) else {}
    recipe_id = str(selected_recipe.get("recipe_id") or "").strip()
    recipe = get_recipe(recipe_id) if recipe_id else None
    if recipe is not None:
        return {
            "id": recipe.id,
            "name": recipe.name,
            "task_profile": recipe.task_profile,
            "adapter_id": recipe.adapter_id,
            "default_input_column": recipe.default_input_column,
            "default_output_column": recipe.default_output_column,
        }
    if recipe_id:
        return {
            "id": recipe_id,
            "name": str(selected_recipe.get("name") or recipe_id),
            "task_profile": str(selected_recipe.get("task_profile") or ""),
            "adapter_id": str(selected_recipe.get("adapter_id") or ""),
            "default_input_column": str(selected_recipe.get("default_input_column") or ""),
            "default_output_column": str(selected_recipe.get("default_output_column") or ""),
        }
    return None


def _issue_status(issues: list[dict[str, str]], *, empty: bool = False) -> MappingVerdict:
    if empty:
        return "empty"
    if any(item["severity"] in {"blocker", "warning"} for item in issues):
        return "attention"
    return "ready"


async def build_data_studio_overview(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a project-level Data Studio readiness summary."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    datasets = list(datasets_result.scalars().all())
    dataset_counts: Counter[str] = Counter()
    for dataset in datasets:
        dataset_counts[dataset.dataset_type.value] += int(dataset.record_count or 0)

    doc_status_result = await db.execute(
        select(RawDocument.status, func.count(RawDocument.id))
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
        .group_by(RawDocument.status)
    )
    document_counts = {
        (status.value if isinstance(status, DocumentStatus) else str(status)): int(count or 0)
        for status, count in doc_status_result.all()
    }

    review_queue = await list_review_queue(db, project_id)
    synthetic_pending = int(review_queue.get("total_pending") or 0)
    synthetic_accepted = int(review_queue.get("total_accepted") or 0)

    raw_rows = _sum_counts(dataset_counts, DatasetType.RAW)
    cleaned_rows = _sum_counts(dataset_counts, DatasetType.CLEANED)
    gold_rows = _sum_counts(dataset_counts, DatasetType.GOLD_DEV, DatasetType.GOLD_TEST)
    prepared_rows = _sum_counts(
        dataset_counts,
        DatasetType.TRAIN,
        DatasetType.VALIDATION,
        DatasetType.TEST,
    )
    default_trainable_rows = cleaned_rows + gold_rows + synthetic_accepted
    trainable_rows = default_trainable_rows if default_trainable_rows > 0 else raw_rows

    recipe_payload = _recipe_payload(project)

    try:
        domain_runtime = await resolve_project_domain_runtime(db, project_id)
    except ValueError:
        domain_runtime = {}
    effective_contract = domain_runtime.get("effective_contract")
    domain_payload = {
        "profile_id": domain_runtime.get("domain_profile_applied"),
        "profile_source": domain_runtime.get("domain_profile_source"),
        "pack_id": domain_runtime.get("domain_pack_applied"),
        "pack_source": domain_runtime.get("domain_pack_source"),
        "display_name": (
            effective_contract.get("display_name")
            if isinstance(effective_contract, dict)
            else None
        ),
    }

    issues: list[dict[str, str]] = []
    if recipe_payload is None:
        issues.append(
            _issue(
                "missing_recipe",
                "blocker",
                "Recipe not selected",
                "Pick a task recipe so BrewSLM knows the training shape and validation rules.",
                action_label="Choose recipe",
                target_tab="data",
            )
        )

    if trainable_rows <= 0:
        issues.append(
            _issue(
                "no_trainable_rows",
                "blocker",
                "No trainable rows yet",
                "Import data, create gold rows, or accept reviewed synthetic rows before preparing a dataset.",
                action_label="Add sources",
                target_tab="data",
            )
        )
    elif trainable_rows < 20:
        issues.append(
            _issue(
                "low_trainable_rows",
                "warning",
                "Very small training set",
                f"{trainable_rows} trainable row(s) is enough to inspect the flow, but most useful SFT runs need more examples.",
                action_label="Add or generate rows",
                target_tab="synthetic",
            )
        )

    if synthetic_pending > 0:
        issues.append(
            _issue(
                "pending_synthetic_rows",
                "warning",
                "Synthetic rows pending review",
                f"{synthetic_pending} generated row(s) are gated out of training until accepted.",
                action_label="Review synthetic rows",
                target_tab="synthetic",
            )
        )

    if trainable_rows > 0 and prepared_rows <= 0:
        issues.append(
            _issue(
                "dataset_not_prepared",
                "warning",
                "Training dataset not prepared",
                "Create train, validation, and test splits before launching a training run.",
                action_label="Prepare dataset",
                target_tab="dataprep",
            )
        )

    if document_counts.get(DocumentStatus.ERROR.value, 0) > 0:
        issues.append(
            _issue(
                "source_errors",
                "warning",
                "Some sources failed ingestion",
                f"{document_counts[DocumentStatus.ERROR.value]} source document(s) need attention.",
                action_label="Inspect sources",
                target_tab="data",
            )
        )

    blocker_count = sum(1 for item in issues if item["severity"] == "blocker")
    warning_count = sum(1 for item in issues if item["severity"] == "warning")
    if blocker_count:
        verdict: OverviewVerdict = "blocked"
    elif warning_count:
        verdict = "needs_work"
    else:
        verdict = "ready"

    return {
        "project_id": project_id,
        "verdict": verdict,
        "recipe": recipe_payload,
        "domain": domain_payload,
        "row_counts": {
            "trainable": trainable_rows,
            "raw": raw_rows,
            "cleaned": cleaned_rows,
            "gold": gold_rows,
            "synthetic_total": synthetic_pending + synthetic_accepted,
            "synthetic_pending": synthetic_pending,
            "synthetic_accepted": synthetic_accepted,
            "prepared": prepared_rows,
            "train": int(dataset_counts.get(DatasetType.TRAIN.value, 0)),
            "validation": int(dataset_counts.get(DatasetType.VALIDATION.value, 0)),
            "test": int(dataset_counts.get(DatasetType.TEST.value, 0)),
        },
        "source_summary": {
            "dataset_count": len(datasets),
            "documents_total": sum(document_counts.values()),
            "documents_accepted": int(document_counts.get(DocumentStatus.ACCEPTED.value, 0)),
            "documents_processing": int(document_counts.get(DocumentStatus.PROCESSING.value, 0)),
            "documents_pending": int(document_counts.get(DocumentStatus.PENDING.value, 0)),
            "documents_error": int(document_counts.get(DocumentStatus.ERROR.value, 0)),
        },
        "issues": issues,
        "primary_action": _primary_action(issues, prepared_rows=prepared_rows),
    }


async def build_data_studio_sources(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return source health and recent source rows for Data Studio."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    datasets = list(datasets_result.scalars().all())

    groups_by_type: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for dataset in datasets:
        type_key = dataset.dataset_type.value
        group = groups_by_type.setdefault(
            type_key,
            {
                "dataset_type": type_key,
                "dataset_count": 0,
                "row_count": 0,
                "locked_count": 0,
                "with_file_count": 0,
            },
        )
        group["dataset_count"] += 1
        group["row_count"] += int(dataset.record_count or 0)
        group["locked_count"] += 1 if dataset.is_locked else 0
        group["with_file_count"] += 1 if dataset.file_path else 0
        total_rows += int(dataset.record_count or 0)

    docs_result = await db.execute(
        select(RawDocument, Dataset)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
        .order_by(RawDocument.ingested_at.desc())
    )
    doc_rows = list(docs_result.all())

    status_counts: Counter[str] = Counter()
    recent_documents: list[dict[str, Any]] = []
    for doc, dataset in doc_rows:
        status = doc.status.value if isinstance(doc.status, DocumentStatus) else str(doc.status)
        status_counts[status] += 1
        if len(recent_documents) >= 8:
            continue
        recent_documents.append({
            "id": doc.id,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "dataset_type": dataset.dataset_type.value,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "status": status,
            "source": doc.source or "upload",
            "sensitivity": doc.sensitivity or "internal",
            "file_size_bytes": int(doc.file_size_bytes or 0),
            "chunk_count": int(doc.chunk_count or 0),
            "quality_score": doc.quality_score,
            "ingested_at": doc.ingested_at.isoformat() if doc.ingested_at else None,
        })

    issues: list[dict[str, str]] = []
    if not datasets and not doc_rows:
        issues.append(
            _issue(
                "no_sources",
                "blocker",
                "No sources connected",
                "Add a local file, remote dataset, or project template to start building training data.",
                action_label="Add sources",
                target_tab="data",
            )
        )

    error_count = int(status_counts.get(DocumentStatus.ERROR.value, 0))
    if error_count:
        issues.append(
            _issue(
                "source_errors",
                "warning",
                "Source import errors",
                f"{error_count} source document(s) failed ingestion and need attention.",
                action_label="Inspect failed sources",
                target_tab="data",
            )
        )

    in_flight_count = int(
        status_counts.get(DocumentStatus.PENDING.value, 0)
        + status_counts.get(DocumentStatus.PROCESSING.value, 0)
    )
    if in_flight_count:
        issues.append(
            _issue(
                "sources_in_progress",
                "info",
                "Sources still processing",
                f"{in_flight_count} source document(s) are pending or processing.",
                action_label="Refresh sources",
                target_tab="data",
            )
        )

    empty_dataset_count = sum(1 for dataset in datasets if int(dataset.record_count or 0) <= 0)
    if datasets and empty_dataset_count == len(datasets):
        issues.append(
            _issue(
                "empty_datasets",
                "warning",
                "Datasets have no rows yet",
                "The project has dataset records, but no counted rows are available for training.",
                action_label="Inspect sources",
                target_tab="data",
            )
        )

    if not datasets and not doc_rows:
        verdict: SourcesVerdict = "empty"
    elif error_count or (datasets and empty_dataset_count == len(datasets)):
        verdict = "attention"
    else:
        verdict = "healthy"

    dataset_groups = sorted(
        groups_by_type.values(),
        key=lambda item: (str(item["dataset_type"]), -int(item["row_count"])),
    )

    return {
        "project_id": project_id,
        "verdict": verdict,
        "totals": {
            "dataset_count": len(datasets),
            "document_count": len(doc_rows),
            "row_count": total_rows,
            "accepted_documents": int(status_counts.get(DocumentStatus.ACCEPTED.value, 0)),
            "pending_documents": int(status_counts.get(DocumentStatus.PENDING.value, 0)),
            "processing_documents": int(status_counts.get(DocumentStatus.PROCESSING.value, 0)),
            "error_documents": error_count,
            "rejected_documents": int(status_counts.get(DocumentStatus.REJECTED.value, 0)),
        },
        "dataset_groups": dataset_groups,
        "recent_documents": recent_documents,
        "issues": issues,
    }


async def _select_mapping_source(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any] | None:
    raw_docs_result = await db.execute(
        select(RawDocument, Dataset)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
            RawDocument.status == DocumentStatus.ACCEPTED,
        )
        .order_by(RawDocument.ingested_at.desc())
    )
    raw_doc_rows = list(raw_docs_result.all())
    if raw_doc_rows:
        doc, dataset = raw_doc_rows[0]
        return {
            "dataset_type": DatasetType.RAW,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "document_id": doc.id,
            "document_name": doc.filename,
            "document_count": len(raw_doc_rows),
            "row_count": int(dataset.record_count or 0),
        }

    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    datasets = list(datasets_result.scalars().all())
    for dataset_type in _MAPPING_SOURCE_PRIORITY[1:]:
        candidates = [
            dataset
            for dataset in datasets
            if dataset.dataset_type == dataset_type
            and bool(str(dataset.file_path or "").strip())
            and int(dataset.record_count or 0) > 0
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: item.updated_at, reverse=True)
        dataset = candidates[0]
        return {
            "dataset_type": dataset.dataset_type,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "document_id": None,
            "document_name": None,
            "document_count": 0,
            "row_count": int(dataset.record_count or 0),
        }
    return None


def _coverage_rows(conformance_report: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = conformance_report.get("required_field_coverage")
    if not isinstance(coverage, dict):
        return []
    rows: list[dict[str, Any]] = []
    for field, stats in coverage.items():
        if not isinstance(stats, dict):
            stats = {}
        rows.append({
            "field": str(field),
            "present": int(stats.get("present") or 0),
            "missing": int(stats.get("missing") or 0),
            "ratio": float(stats.get("ratio") or 0.0),
        })
    rows.sort(key=lambda item: (float(item["ratio"]), str(item["field"])))
    return rows


def _compact_preview_rows(preview_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(preview_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in preview_rows[:3]:
        if not isinstance(row, dict):
            continue
        rows.append({
            "index": int(row.get("index") or 0),
            "raw": row.get("raw") if isinstance(row.get("raw"), dict) else {},
            "mapped": row.get("mapped") if isinstance(row.get("mapped"), dict) else {},
        })
    return rows


def _empty_mapping_payload(
    *,
    project_id: int,
    verdict: MappingVerdict,
    recipe_payload: dict[str, Any] | None,
    preference_source: str,
    preference_adapter_id: str,
    preference_task_profile: str,
    field_mapping: dict[str, str],
    adapter_config: dict[str, Any],
    effective_source: str,
    effective_adapter_id: str,
    effective_task_profile: str | None,
    source: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "verdict": verdict,
        "recipe": recipe_payload,
        "preference": {
            "source": preference_source,
            "adapter_id": preference_adapter_id or "default-canonical",
            "task_profile": preference_task_profile or None,
            "field_mapping": field_mapping,
            "field_mapping_count": len(field_mapping),
        },
        "effective_mapping": {
            "source": effective_source,
            "adapter_id": effective_adapter_id,
            "task_profile": effective_task_profile,
            "adapter_config": adapter_config,
            "field_mapping": field_mapping,
        },
        "source": source,
        "summary": {
            "sampled_records": 0,
            "mapped_records": 0,
            "dropped_records": 0,
            "error_count": 0,
            "mapping_success_rate": 0.0,
            "contract_pass": False,
            "required_fields": [],
            "required_fields_below_100": [],
            "required_field_coverage": [],
        },
        "preview_rows": [],
        "diagnostics": {},
        "issues": issues,
    }


async def build_data_studio_mapping_preview(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a recipe-aware adapter/schema preview for Data Studio."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    recipe_payload = _recipe_payload(project)
    preference = await resolve_project_dataset_adapter_preference(db, project_id)
    preference_source = str(preference.get("source") or "default")
    field_mapping = dict(preference.get("field_mapping") or {})
    adapter_config = dict(preference.get("adapter_config") or {})

    recipe_adapter_id = str((recipe_payload or {}).get("adapter_id") or "").strip()
    recipe_task_profile = str((recipe_payload or {}).get("task_profile") or "").strip()
    preference_adapter_id = str(preference.get("adapter_id") or "").strip()
    preference_task_profile = str(preference.get("task_profile") or "").strip()

    if preference_source in {"project", "domain_pack"}:
        effective_adapter_id = preference_adapter_id or recipe_adapter_id or "default-canonical"
        effective_task_profile = preference_task_profile or recipe_task_profile or None
        effective_source = preference_source
    else:
        effective_adapter_id = recipe_adapter_id or preference_adapter_id or "default-canonical"
        effective_task_profile = recipe_task_profile or preference_task_profile or None
        effective_source = "recipe" if recipe_adapter_id else preference_source

    issues: list[dict[str, str]] = []
    if recipe_payload is None:
        issues.append(
            _issue(
                "missing_recipe",
                "warning",
                "Recipe not selected",
                "Pick a recipe to validate the mapping against the task shape you plan to train.",
                action_label="Choose recipe",
                target_tab="data",
            )
        )

    source = await _select_mapping_source(db, project_id)
    if source is None:
        issues.append(
            _issue(
                "no_mapping_source",
                "blocker",
                "No previewable rows",
                "Add an accepted raw document or a row-backed dataset before checking schema mapping.",
                action_label="Add sources",
                target_tab="data",
            )
        )
        return _empty_mapping_payload(
            project_id=project_id,
            verdict="empty",
            recipe_payload=recipe_payload,
            preference_source=preference_source,
            preference_adapter_id=preference_adapter_id,
            preference_task_profile=preference_task_profile,
            field_mapping=field_mapping,
            adapter_config=adapter_config,
            effective_source=effective_source,
            effective_adapter_id=effective_adapter_id,
            effective_task_profile=effective_task_profile,
            source=None,
            issues=issues,
        )

    dataset_type = source["dataset_type"]
    try:
        preview = await preview_project_data_adapter(
            db=db,
            project_id=project_id,
            dataset_type=dataset_type,
            sample_size=100,
            adapter_id=effective_adapter_id,
            adapter_config=adapter_config,
            field_mapping=field_mapping,
            task_profile=effective_task_profile,
            document_id=source.get("document_id"),
            preview_limit=3,
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "mapping_preview_failed",
                "warning",
                "Mapping preview could not run",
                str(exc)[:240],
                action_label="Open adapter preview",
                target_tab="dataprep",
            )
        )
        source_payload = {
            **source,
            "dataset_type": dataset_type.value,
        }
        return _empty_mapping_payload(
            project_id=project_id,
            verdict="attention",
            recipe_payload=recipe_payload,
            preference_source=preference_source,
            preference_adapter_id=preference_adapter_id,
            preference_task_profile=preference_task_profile,
            field_mapping=field_mapping,
            adapter_config=adapter_config,
            effective_source=effective_source,
            effective_adapter_id=effective_adapter_id,
            effective_task_profile=effective_task_profile,
            source=source_payload,
            issues=issues,
        )

    conformance_report = (
        preview.get("conformance_report")
        if isinstance(preview.get("conformance_report"), dict)
        else {}
    )
    sampled_records = int(preview.get("sampled_records") or 0)
    mapped_records = int(preview.get("mapped_records") or 0)
    dropped_records = int(preview.get("dropped_records") or 0)
    error_count = int(preview.get("error_count") or 0)
    required_fields = [
        str(item)
        for item in list(conformance_report.get("required_fields") or [])
        if str(item).strip()
    ]
    required_fields_below_100 = [
        str(item)
        for item in list(conformance_report.get("required_fields_below_100") or [])
        if str(item).strip()
    ]
    mapping_success_rate = float(conformance_report.get("mapping_success_rate") or 0.0)
    contract_pass = bool(conformance_report.get("contract_pass"))

    if sampled_records <= 0:
        issues.append(
            _issue(
                "no_sampled_rows",
                "blocker",
                "Source has no readable rows",
                "The selected source exists, but BrewSLM could not read sample rows from it.",
                action_label="Inspect sources",
                target_tab="data",
            )
        )
    elif mapped_records <= 0:
        issues.append(
            _issue(
                "no_mapped_rows",
                "blocker",
                "No rows mapped to the recipe shape",
                "The active adapter could not turn sampled rows into canonical training records.",
                action_label="Open adapter preview",
                target_tab="dataprep",
            )
        )
    elif required_fields_below_100:
        issues.append(
            _issue(
                "required_fields_missing",
                "warning",
                "Required fields are incomplete",
                f"Missing coverage for: {', '.join(required_fields_below_100)}.",
                action_label="Review mapping",
                target_tab="dataprep",
            )
        )

    if dropped_records > 0 or error_count > 0:
        issues.append(
            _issue(
                "mapping_drops",
                "warning",
                "Some rows dropped during mapping",
                f"{dropped_records} sampled row(s) dropped and {error_count} adapter error(s) were reported.",
                action_label="Open adapter preview",
                target_tab="dataprep",
            )
        )

    if preview.get("task_profile_compatible") is False:
        issues.append(
            _issue(
                "task_profile_mismatch",
                "warning",
                "Task profile does not match adapter",
                "The requested task profile is not declared by the resolved adapter.",
                action_label="Review adapter",
                target_tab="dataprep",
            )
        )

    if (
        recipe_adapter_id
        and preference_source in {"project", "domain_pack"}
        and preference_adapter_id
        and preference_adapter_id != recipe_adapter_id
    ):
        issues.append(
            _issue(
                "adapter_differs_from_recipe",
                "info",
                "Adapter preset differs from recipe default",
                f"Using {preference_adapter_id} from {preference_source}; the recipe default is {recipe_adapter_id}.",
                action_label="Review adapter",
                target_tab="dataprep",
            )
        )

    return {
        "project_id": project_id,
        "verdict": _issue_status(issues),
        "recipe": recipe_payload,
        "preference": {
            "source": preference_source,
            "adapter_id": preference_adapter_id or "default-canonical",
            "task_profile": preference_task_profile or None,
            "field_mapping": field_mapping,
            "field_mapping_count": len(field_mapping),
        },
        "effective_mapping": {
            "source": effective_source,
            "adapter_id": str(preview.get("resolved_adapter_id") or effective_adapter_id),
            "requested_adapter_id": str(preview.get("requested_adapter_id") or effective_adapter_id),
            "task_profile": str(preview.get("resolved_task_profile") or effective_task_profile or ""),
            "requested_task_profile": preview.get("requested_task_profile"),
            "adapter_config": adapter_config,
            "field_mapping": field_mapping,
            "auto_apply": preview.get("auto_apply") if isinstance(preview.get("auto_apply"), dict) else {},
        },
        "source": {
            **source,
            "dataset_type": dataset_type.value,
        },
        "summary": {
            "sampled_records": sampled_records,
            "mapped_records": mapped_records,
            "dropped_records": dropped_records,
            "error_count": error_count,
            "mapping_success_rate": mapping_success_rate,
            "contract_pass": contract_pass,
            "required_fields": required_fields,
            "required_fields_below_100": required_fields_below_100,
            "required_field_coverage": _coverage_rows(conformance_report),
        },
        "preview_rows": _compact_preview_rows(preview.get("preview_rows")),
        "diagnostics": {
            "adapter_contract": preview.get("adapter_contract") if isinstance(preview.get("adapter_contract"), dict) else {},
            "validation_report": preview.get("validation_report") if isinstance(preview.get("validation_report"), dict) else {},
            "detection_scores": preview.get("detection_scores") if isinstance(preview.get("detection_scores"), dict) else {},
            "auto_fix_suggestions": preview.get("auto_fix_suggestions") if isinstance(preview.get("auto_fix_suggestions"), list) else [],
            "compatibility_warnings": preview.get("compatibility_warnings") if isinstance(preview.get("compatibility_warnings"), list) else [],
            "inferred_task_profiles": preview.get("inferred_task_profiles") if isinstance(preview.get("inferred_task_profiles"), list) else [],
        },
        "issues": issues,
    }

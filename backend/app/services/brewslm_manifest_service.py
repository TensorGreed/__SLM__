"""Pipeline-as-code manifest serializer + deserializer (priority.md P21).

Two public entry points:

- :func:`serialize_project_to_manifest` reads a project's current state
  (project row, active blueprint, datasets, adapters, training defaults,
  evaluation pack preference, target profile) and produces a
  :class:`BrewslmManifest` Pydantic model — the in-memory shape of the
  ``brewslm.yaml`` file a user would check into git.

- :func:`deserialize_manifest_to_apply_plan` takes a parsed manifest
  plus an optional ``project_id`` and produces a :class:`ManifestApplyPlan`
  describing what would change. When ``project_id`` is ``None`` (a
  brand-new project), every section becomes a ``create`` action; when it
  matches an existing project, sections that are already in the desired
  state come back as ``noop`` so the apply path (P22) can short-circuit.

YAML helpers ``manifest_to_yaml`` / ``manifest_from_yaml`` round-trip via
``yaml.safe_load``/``safe_dump`` so the serialized form is stable across
Python versions and PyYAML installs.
"""

from __future__ import annotations

from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.dataset import Dataset, DatasetVersion
from app.models.dataset_adapter_definition import DatasetAdapterDefinition
from app.models.domain_blueprint import DomainBlueprintRevision, DomainBlueprintStatus
from app.models.domain_pack import DomainPack
from app.models.domain_profile import DomainProfile
from app.models.project import Project
from app.schemas.brewslm_manifest import (
    AdapterSpec,
    BlueprintGlossaryEntry,
    BlueprintSection,
    BlueprintSuccessMetric,
    BrewslmManifest,
    BrewslmManifestSpec,
    DataSourceSpec,
    DataSourceVersionSpec,
    DeploymentSection,
    DomainSection,
    EvalPackSection,
    ExportSection,
    MANIFEST_API_VERSION,
    MANIFEST_KIND,
    ManifestApplyAction,
    ManifestApplyPlan,
    ManifestMetadata,
    ModelSection,
    TrainingPlanSection,
    WorkflowSection,
)


# -- Serializer --------------------------------------------------------------


def _enum_value(value: Any) -> Any:
    """Unwrap SQLAlchemy enum columns so the serialized output is plain JSON."""
    if value is None:
        return None
    return getattr(value, "value", value)


async def _load_project(db: AsyncSession, project_id: int) -> Project:
    row = (
        await db.execute(select(Project).where(Project.id == project_id))
    ).scalar_one_or_none()
    if row is None:
        raise ValueError("project_not_found")
    return row


async def _load_active_blueprint(
    db: AsyncSession, project_id: int, active_version: int | None
) -> DomainBlueprintRevision | None:
    """Pick the active revision; fall back to the latest one if none active.

    The project's ``active_domain_blueprint_version`` is the single source
    of truth when set — it tracks whichever revision the apply path moved
    to ACTIVE most recently. If unset (project never applied a blueprint),
    we surface the latest revision so the manifest still reflects the
    user's authored intent.
    """
    if active_version is not None:
        row = (
            await db.execute(
                select(DomainBlueprintRevision).where(
                    DomainBlueprintRevision.project_id == project_id,
                    DomainBlueprintRevision.version == active_version,
                )
            )
        ).scalar_one_or_none()
        if row is not None:
            return row

    row = (
        await db.execute(
            select(DomainBlueprintRevision)
            .where(DomainBlueprintRevision.project_id == project_id)
            .order_by(
                DomainBlueprintRevision.version.desc(),
                DomainBlueprintRevision.id.desc(),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    return row


def _blueprint_to_section(rev: DomainBlueprintRevision) -> BlueprintSection:
    success_metrics = []
    for item in list(rev.success_metrics or []):
        if isinstance(item, dict):
            try:
                success_metrics.append(BlueprintSuccessMetric.model_validate(item))
            except Exception:
                # Be tolerant of legacy rows; skip malformed entries rather
                # than raising on a serializer call.
                continue

    glossary = []
    for item in list(rev.glossary or []):
        if isinstance(item, dict):
            try:
                glossary.append(BlueprintGlossaryEntry.model_validate(item))
            except Exception:
                continue

    return BlueprintSection(
        domain_name=str(rev.domain_name or ""),
        problem_statement=str(rev.problem_statement or ""),
        target_user_persona=str(rev.target_user_persona or ""),
        task_family=str(rev.task_family or "instruction_sft"),
        input_modality=str(rev.input_modality or "text"),
        expected_output_schema=dict(rev.expected_output_schema or {}),
        expected_output_examples=list(rev.expected_output_examples or []),
        safety_compliance_notes=[
            str(x) for x in list(rev.safety_compliance_notes or []) if x is not None
        ],
        deployment_target_constraints=dict(rev.deployment_target_constraints or {}),
        success_metrics=success_metrics,
        glossary=glossary,
        confidence_score=float(rev.confidence_score or 0.0),
        unresolved_assumptions=[
            str(x) for x in list(rev.unresolved_assumptions or []) if x is not None
        ],
        version=int(rev.version) if rev.version is not None else None,
        source=str(rev.source or "") or None,
    )


async def _load_datasets(db: AsyncSession, project_id: int) -> list[DataSourceSpec]:
    rows = (
        await db.execute(
            select(Dataset)
            .where(Dataset.project_id == project_id)
            .order_by(Dataset.id.asc())
        )
    ).scalars().all()

    sources: list[DataSourceSpec] = []
    for ds in rows:
        version_rows = (
            await db.execute(
                select(DatasetVersion)
                .where(DatasetVersion.dataset_id == ds.id)
                .order_by(DatasetVersion.version.asc())
            )
        ).scalars().all()
        sources.append(
            DataSourceSpec(
                name=str(ds.name),
                type=_enum_value(ds.dataset_type) or "raw",
                description=str(ds.description or ""),
                record_count=int(ds.record_count or 0),
                file_path=str(ds.file_path or "") or None,
                metadata=dict(ds.metadata_ or {}),
                versions=[
                    DataSourceVersionSpec(
                        version=int(v.version),
                        record_count=int(v.record_count or 0),
                        file_path=str(v.file_path or "") or None,
                    )
                    for v in version_rows
                ],
            )
        )
    return sources


async def _load_adapters(db: AsyncSession, project_id: int) -> list[AdapterSpec]:
    """Pick the highest version per adapter_name for the project.

    Adapter Studio (Phase 50) versions adapters per ``(project_id,
    adapter_name)`` — the manifest captures the desired latest state, so
    we collapse the version chain to one entry per name.
    """
    rows = (
        await db.execute(
            select(DatasetAdapterDefinition)
            .where(DatasetAdapterDefinition.project_id == project_id)
            .order_by(
                DatasetAdapterDefinition.adapter_name.asc(),
                DatasetAdapterDefinition.version.desc(),
            )
        )
    ).scalars().all()

    seen: set[str] = set()
    adapters: list[AdapterSpec] = []
    for row in rows:
        name = str(row.adapter_name or "")
        if not name or name in seen:
            continue
        seen.add(name)
        adapters.append(
            AdapterSpec(
                name=name,
                version=int(row.version or 1),
                status=str(row.status or "active"),
                base_adapter_id=str(row.base_adapter_id or "default-canonical"),
                task_profile=str(row.task_profile or "") or None,
                source_type=str(row.source_type or "raw"),
                source_ref=str(row.source_ref or "") or None,
                field_mapping=dict(row.field_mapping or {}),
                adapter_config=dict(row.adapter_config or {}),
                output_contract=dict(row.output_contract or {}),
            )
        )
    return adapters


async def _resolve_pack_id_string(db: AsyncSession, pack_int_id: int | None) -> str | None:
    if pack_int_id is None:
        return None
    row = (
        await db.execute(select(DomainPack).where(DomainPack.id == pack_int_id))
    ).scalar_one_or_none()
    return str(row.pack_id) if row is not None else None


async def _resolve_profile_id_string(
    db: AsyncSession, profile_int_id: int | None
) -> str | None:
    if profile_int_id is None:
        return None
    row = (
        await db.execute(select(DomainProfile).where(DomainProfile.id == profile_int_id))
    ).scalar_one_or_none()
    return str(row.profile_id) if row is not None else None


async def serialize_project_to_manifest(
    db: AsyncSession, *, project_id: int
) -> BrewslmManifest:
    """Return a :class:`BrewslmManifest` representing the project's current state.

    The serializer is read-only. It loads the latest authoritative slice
    of every section the manifest covers and returns a Pydantic model;
    the caller decides whether to render it as YAML, JSON, or feed it
    into the apply-plan deserializer.
    """
    project = await _load_project(db, project_id)
    blueprint_row = await _load_active_blueprint(
        db, project_id, project.active_domain_blueprint_version
    )

    pack_id_str = await _resolve_pack_id_string(db, project.domain_pack_id)
    profile_id_str = await _resolve_profile_id_string(db, project.domain_profile_id)

    workflow = WorkflowSection(
        beginner_mode=bool(project.beginner_mode),
        pipeline_stage=_enum_value(project.pipeline_stage) or "ingestion",
        target_profile_id=str(project.target_profile_id or "") or None,
        training_preferred_plan_profile=str(project.training_preferred_plan_profile or "") or None,
        gate_policy=dict(project.gate_policy or {}),
        budget_settings=dict(project.budget_settings or {}),
    )

    blueprint_section = _blueprint_to_section(blueprint_row) if blueprint_row else None
    domain_section = DomainSection(pack_id=pack_id_str, profile_id=profile_id_str)
    model_section = ModelSection(base_model=str(project.base_model_name or ""))

    data_sources = await _load_datasets(db, project_id)
    adapters = await _load_adapters(db, project_id)

    # Adapter preset on the project carries the user's chosen training
    # adapter defaults — promote it into training_plan.config so a
    # rebuilt project starts with the same flat hyperparam bag.
    adapter_preset = dict(project.dataset_adapter_preset or {})
    training_plan = TrainingPlanSection(
        plan_profile=str(project.training_preferred_plan_profile or "") or None,
        config=adapter_preset,
    )

    eval_pack = EvalPackSection(
        pack_id=str(project.evaluation_preferred_pack_id or "") or None,
    )

    deployment = DeploymentSection(
        target_profile_id=str(project.target_profile_id or "") or None,
    )

    spec = BrewslmManifestSpec(
        workflow=workflow,
        blueprint=blueprint_section,
        domain=domain_section,
        model=model_section,
        data_sources=data_sources,
        adapters=adapters,
        training_plan=training_plan,
        eval_pack=eval_pack,
        export=ExportSection(),
        deployment=deployment,
    )

    return BrewslmManifest(
        api_version=MANIFEST_API_VERSION,
        kind=MANIFEST_KIND,
        metadata=ManifestMetadata(
            name=str(project.name),
            description=str(project.description or ""),
        ),
        spec=spec,
    )


# -- YAML helpers ------------------------------------------------------------


def manifest_to_yaml(manifest: BrewslmManifest) -> str:
    """Render the manifest as a stable YAML document.

    ``mode='json'`` ensures the dict has primitive types only (no enums,
    no datetimes), ``sort_keys=False`` preserves the section order
    declared on the schema so the file reads top-down.
    """
    payload = manifest.model_dump(mode="json", exclude_none=False)
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def manifest_from_yaml(text: str) -> BrewslmManifest:
    """Parse + validate a YAML manifest body.

    Raises :class:`ValueError` on empty input and lets Pydantic's
    :class:`ValidationError` propagate for schema violations.
    """
    if not text or not text.strip():
        raise ValueError("manifest_empty")
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError("manifest_not_a_mapping")
    return BrewslmManifest.model_validate(parsed)


# -- Deserializer / apply-plan ----------------------------------------------


_BLUEPRINT_DIFF_FIELDS: tuple[str, ...] = (
    "domain_name",
    "problem_statement",
    "target_user_persona",
    "task_family",
    "input_modality",
    "expected_output_schema",
    "expected_output_examples",
    "safety_compliance_notes",
    "deployment_target_constraints",
    "success_metrics",
    "glossary",
    "confidence_score",
    "unresolved_assumptions",
)


_WORKFLOW_DIFF_FIELDS: tuple[str, ...] = (
    "beginner_mode",
    "pipeline_stage",
    "target_profile_id",
    "training_preferred_plan_profile",
    "gate_policy",
    "budget_settings",
)


def _diff_fields(before: dict[str, Any], after: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    changed: list[str] = []
    for field in fields:
        if before.get(field) != after.get(field):
            changed.append(field)
    return changed


async def _summarize_existing_project(
    db: AsyncSession, project_id: int
) -> tuple[BrewslmManifest, dict[str, DataSourceSpec], dict[str, AdapterSpec]]:
    current = await serialize_project_to_manifest(db, project_id=project_id)
    data_sources_index = {ds.name: ds for ds in current.spec.data_sources}
    adapters_index = {a.name: a for a in current.spec.adapters}
    return current, data_sources_index, adapters_index


def _section_dict(section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    return section.model_dump(mode="json")


def _bump_summary(summary: dict[str, int], operation: str) -> None:
    summary[operation] = summary.get(operation, 0) + 1


async def deserialize_manifest_to_apply_plan(
    db: AsyncSession,
    *,
    manifest: BrewslmManifest,
    project_id: int | None = None,
) -> ManifestApplyPlan:
    """Diff a manifest against current project state and emit an apply-plan.

    When ``project_id`` is ``None``, the plan is "build from scratch":
    every populated section becomes a ``create``. When ``project_id`` is
    provided, the current state is serialized via the same serializer,
    and each section is compared field-wise (or item-wise for lists) so
    consumers see exactly which fields would change.
    """
    actions: list[ManifestApplyAction] = []
    warnings: list[str] = []
    summary: dict[str, int] = {}

    if project_id is None:
        actions.append(
            ManifestApplyAction(
                target="project",
                operation="create",
                name=manifest.metadata.name,
                after=manifest.metadata.model_dump(mode="json"),
                reason="Project does not exist yet.",
            )
        )
        _bump_summary(summary, "create")

        if manifest.spec.blueprint is not None:
            actions.append(
                ManifestApplyAction(
                    target="blueprint",
                    operation="create",
                    after=manifest.spec.blueprint.model_dump(mode="json"),
                )
            )
            _bump_summary(summary, "create")

        for ds in manifest.spec.data_sources:
            actions.append(
                ManifestApplyAction(
                    target="data_source",
                    operation="create",
                    name=ds.name,
                    after=ds.model_dump(mode="json"),
                )
            )
            _bump_summary(summary, "create")

        for adapter in manifest.spec.adapters:
            actions.append(
                ManifestApplyAction(
                    target="adapter",
                    operation="create",
                    name=adapter.name,
                    after=adapter.model_dump(mode="json"),
                )
            )
            _bump_summary(summary, "create")

        actions.append(
            ManifestApplyAction(
                target="training_plan",
                operation="create",
                after=manifest.spec.training_plan.model_dump(mode="json"),
            )
        )
        _bump_summary(summary, "create")

        actions.append(
            ManifestApplyAction(
                target="eval_pack",
                operation="create",
                after=manifest.spec.eval_pack.model_dump(mode="json"),
            )
        )
        _bump_summary(summary, "create")

        return ManifestApplyPlan(
            project_id=None,
            project_name=manifest.metadata.name,
            api_version=manifest.api_version,
            actions=actions,
            warnings=warnings,
            summary=summary,
        )

    # Update path — diff against current state.
    current, ds_index, adapter_index = await _summarize_existing_project(db, project_id)

    # Project metadata (name + description).
    if (
        current.metadata.name != manifest.metadata.name
        or current.metadata.description != manifest.metadata.description
    ):
        before = current.metadata.model_dump(mode="json")
        after = manifest.metadata.model_dump(mode="json")
        actions.append(
            ManifestApplyAction(
                target="project",
                operation="update",
                name=manifest.metadata.name,
                before=before,
                after=after,
                fields_changed=[
                    field for field in ("name", "description")
                    if before.get(field) != after.get(field)
                ],
            )
        )
        _bump_summary(summary, "update")
    else:
        actions.append(
            ManifestApplyAction(
                target="project",
                operation="noop",
                name=manifest.metadata.name,
            )
        )
        _bump_summary(summary, "noop")

    # Workflow.
    workflow_before = _section_dict(current.spec.workflow)
    workflow_after = _section_dict(manifest.spec.workflow)
    workflow_changed = _diff_fields(workflow_before, workflow_after, _WORKFLOW_DIFF_FIELDS)
    if workflow_changed:
        actions.append(
            ManifestApplyAction(
                target="workflow",
                operation="update",
                before=workflow_before,
                after=workflow_after,
                fields_changed=workflow_changed,
            )
        )
        _bump_summary(summary, "update")
    else:
        actions.append(ManifestApplyAction(target="workflow", operation="noop"))
        _bump_summary(summary, "noop")

    # Blueprint.
    if manifest.spec.blueprint is None and current.spec.blueprint is None:
        actions.append(ManifestApplyAction(target="blueprint", operation="noop"))
        _bump_summary(summary, "noop")
    elif manifest.spec.blueprint is not None and current.spec.blueprint is None:
        actions.append(
            ManifestApplyAction(
                target="blueprint",
                operation="create",
                after=manifest.spec.blueprint.model_dump(mode="json"),
            )
        )
        _bump_summary(summary, "create")
    elif manifest.spec.blueprint is None and current.spec.blueprint is not None:
        warnings.append("blueprint_drop_not_supported")
        actions.append(
            ManifestApplyAction(
                target="blueprint",
                operation="noop",
                before=current.spec.blueprint.model_dump(mode="json"),
                reason="Manifest has no blueprint; existing revision left in place.",
            )
        )
        _bump_summary(summary, "noop")
    else:
        before = current.spec.blueprint.model_dump(mode="json")  # type: ignore[union-attr]
        after = manifest.spec.blueprint.model_dump(mode="json")  # type: ignore[union-attr]
        changed = _diff_fields(before, after, _BLUEPRINT_DIFF_FIELDS)
        if changed:
            actions.append(
                ManifestApplyAction(
                    target="blueprint",
                    operation="update",
                    before=before,
                    after=after,
                    fields_changed=changed,
                )
            )
            _bump_summary(summary, "update")
        else:
            actions.append(ManifestApplyAction(target="blueprint", operation="noop"))
            _bump_summary(summary, "noop")

    # Data sources — keyed on name.
    desired_ds = {ds.name: ds for ds in manifest.spec.data_sources}
    for name, ds in desired_ds.items():
        existing = ds_index.get(name)
        if existing is None:
            actions.append(
                ManifestApplyAction(
                    target="data_source",
                    operation="create",
                    name=name,
                    after=ds.model_dump(mode="json"),
                )
            )
            _bump_summary(summary, "create")
            continue
        before = existing.model_dump(mode="json")
        after = ds.model_dump(mode="json")
        if before == after:
            actions.append(
                ManifestApplyAction(target="data_source", operation="noop", name=name)
            )
            _bump_summary(summary, "noop")
        else:
            changed = [k for k in after if before.get(k) != after.get(k)]
            actions.append(
                ManifestApplyAction(
                    target="data_source",
                    operation="update",
                    name=name,
                    before=before,
                    after=after,
                    fields_changed=changed,
                )
            )
            _bump_summary(summary, "update")
    # Datasets present on the project but absent from the manifest are
    # surfaced as warnings — destructive deletes are intentionally not
    # part of the apply contract.
    for name in ds_index:
        if name not in desired_ds:
            warnings.append(f"data_source_not_in_manifest:{name}")

    # Adapters — same pattern.
    desired_adapters = {a.name: a for a in manifest.spec.adapters}
    for name, adapter in desired_adapters.items():
        existing_adapter = adapter_index.get(name)
        if existing_adapter is None:
            actions.append(
                ManifestApplyAction(
                    target="adapter",
                    operation="create",
                    name=name,
                    after=adapter.model_dump(mode="json"),
                )
            )
            _bump_summary(summary, "create")
            continue
        before = existing_adapter.model_dump(mode="json")
        after = adapter.model_dump(mode="json")
        if before == after:
            actions.append(
                ManifestApplyAction(target="adapter", operation="noop", name=name)
            )
            _bump_summary(summary, "noop")
        else:
            changed = [k for k in after if before.get(k) != after.get(k)]
            actions.append(
                ManifestApplyAction(
                    target="adapter",
                    operation="update",
                    name=name,
                    before=before,
                    after=after,
                    fields_changed=changed,
                )
            )
            _bump_summary(summary, "update")
    for name in adapter_index:
        if name not in desired_adapters:
            warnings.append(f"adapter_not_in_manifest:{name}")

    # Model.
    model_before = _section_dict(current.spec.model)
    model_after = _section_dict(manifest.spec.model)
    if model_before.get("base_model") != model_after.get("base_model"):
        actions.append(
            ManifestApplyAction(
                target="model",
                operation="update",
                before=model_before,
                after=model_after,
                fields_changed=["base_model"],
            )
        )
        _bump_summary(summary, "update")
    else:
        actions.append(ManifestApplyAction(target="model", operation="noop"))
        _bump_summary(summary, "noop")

    # Training plan, eval pack, deployment — single-block diff is sufficient
    # for the apply-plan; granular field diff is left to the apply path
    # (it has the recipe / runtime resolvers handy).
    for target, current_section, desired_section in (
        ("training_plan", current.spec.training_plan, manifest.spec.training_plan),
        ("eval_pack", current.spec.eval_pack, manifest.spec.eval_pack),
        ("deployment", current.spec.deployment, manifest.spec.deployment),
    ):
        before = _section_dict(current_section)
        after = _section_dict(desired_section)
        if before == after:
            actions.append(ManifestApplyAction(target=target, operation="noop"))
            _bump_summary(summary, "noop")
        else:
            changed = [k for k in after if before.get(k) != after.get(k)]
            actions.append(
                ManifestApplyAction(
                    target=target,
                    operation="update",
                    before=before,
                    after=after,
                    fields_changed=changed,
                )
            )
            _bump_summary(summary, "update")

    return ManifestApplyPlan(
        project_id=project_id,
        project_name=manifest.metadata.name,
        api_version=manifest.api_version,
        actions=actions,
        warnings=warnings,
        summary=summary,
    )

"""Manifest validate / diff / apply services (priority.md P22).

Three layered services on top of the P21 schema + serializer:

- :func:`validate_manifest` — runs the Pydantic contract first (caught
  by the API as 400), then cross-references the parsed manifest against
  live catalogs (base-model registry, target profiles, domain packs +
  profiles, evaluation packs, dataset types) and accumulates a list of
  :class:`ManifestValidationIssue` rows. Each issue carries a stable
  reason ``code``, a JSON-path-ish ``field`` pointer, a human-readable
  ``message``, and an ``actionable_fix`` so the CLI / frontend can
  render a "what to do next" line without re-deriving the cause.

- :func:`diff_manifest_against_project` — thin shim over the P21
  apply-plan deserializer so the API has a stable name for the
  ``POST /manifest/diff`` route.

- :func:`apply_manifest` — given a validated manifest + optional
  ``project_id`` and a ``plan_only`` flag, applies the manifest to the
  database. ``plan_only=True`` short-circuits to the deserializer (no
  writes); ``plan_only=False`` performs the writes inside a single
  transaction so partial application can't leave the project half-built.

Apply is intentionally **non-destructive**: datasets / adapters present
on the project but absent from the manifest are surfaced as warnings,
not deletes. Wave H + GA hardening will revisit destructive apply once
RBAC + audit (Wave I) is wired up.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType, DatasetVersion
from app.models.dataset_adapter_definition import DatasetAdapterDefinition
from app.models.domain_blueprint import DomainBlueprintRevision, DomainBlueprintStatus
from app.models.domain_pack import DomainPack
from app.models.domain_profile import DomainProfile
from app.models.base_model_registry import BaseModelRegistryEntry
from app.models.project import PipelineStage, Project, ProjectStatus
from app.schemas.brewslm_manifest import (
    BrewslmManifest,
    DataSourceSpec,
    AdapterSpec,
    BlueprintSection,
    ManifestApplyAction,
    ManifestApplyPlan,
)
from app.schemas.domain_blueprint import (
    DomainBlueprintContract,
    GlossaryEntry,
    SuccessMetric,
)
from app.services.brewslm_manifest_service import (
    deserialize_manifest_to_apply_plan,
    serialize_project_to_manifest,
)
from app.services.domain_blueprint_service import (
    DomainBlueprintValidationError,
    save_domain_blueprint_revision,
    apply_domain_blueprint_revision,
    validate_domain_blueprint,
)
from app.services.evaluation_pack_service import is_supported_evaluation_pack_id
from app.services.target_profile_service import get_target_by_id

from pydantic import BaseModel, ConfigDict, Field


# -- Validation issue model -------------------------------------------------


class ManifestValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: str = "error"  # 'error' | 'warning'
    field: str = ""
    message: str
    actionable_fix: str = ""


class ManifestValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    errors: list[ManifestValidationIssue] = Field(default_factory=list)
    warnings: list[ManifestValidationIssue] = Field(default_factory=list)


class ManifestApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: int
    project_name: str
    plan_only: bool
    plan: ManifestApplyPlan
    validation: ManifestValidationResult
    applied_actions: list[ManifestApplyAction] = Field(default_factory=list)


# -- Validation -------------------------------------------------------------


_VALID_DATASET_TYPES: frozenset[str] = frozenset(t.value for t in DatasetType)
_VALID_PIPELINE_STAGES: frozenset[str] = frozenset(s.value for s in PipelineStage)


def _add_error(
    issues: list[ManifestValidationIssue],
    *,
    code: str,
    field: str,
    message: str,
    actionable_fix: str,
) -> None:
    issues.append(
        ManifestValidationIssue(
            code=code,
            severity="error",
            field=field,
            message=message,
            actionable_fix=actionable_fix,
        )
    )


def _add_warning(
    issues: list[ManifestValidationIssue],
    *,
    code: str,
    field: str,
    message: str,
    actionable_fix: str,
) -> None:
    issues.append(
        ManifestValidationIssue(
            code=code,
            severity="warning",
            field=field,
            message=message,
            actionable_fix=actionable_fix,
        )
    )


async def _validate_target_profile(
    manifest: BrewslmManifest, errors: list[ManifestValidationIssue]
) -> None:
    target_id = (manifest.spec.workflow.target_profile_id or "").strip()
    if not target_id:
        return
    if get_target_by_id(target_id) is None:
        _add_error(
            errors,
            code="UNKNOWN_TARGET_PROFILE",
            field="spec.workflow.target_profile_id",
            message=f"Target profile {target_id!r} is not registered.",
            actionable_fix=(
                "Pick one of the registered profiles or register a new one via "
                "`brewslm targets register`."
            ),
        )


async def _validate_eval_pack(
    manifest: BrewslmManifest, errors: list[ManifestValidationIssue]
) -> None:
    pack_id = (manifest.spec.eval_pack.pack_id or "").strip()
    if not pack_id:
        return
    if not is_supported_evaluation_pack_id(pack_id):
        _add_error(
            errors,
            code="UNKNOWN_EVAL_PACK",
            field="spec.eval_pack.pack_id",
            message=f"Evaluation pack {pack_id!r} is not registered.",
            actionable_fix=(
                "List packs with `brewslm eval list-packs` or generate a new "
                "pack with `brewslm eval generate`."
            ),
        )


async def _validate_domain_pack(
    db: AsyncSession,
    manifest: BrewslmManifest,
    errors: list[ManifestValidationIssue],
) -> None:
    pack_id = (manifest.spec.domain.pack_id or "").strip()
    if not pack_id:
        return
    row = (
        await db.execute(select(DomainPack).where(DomainPack.pack_id == pack_id))
    ).scalar_one_or_none()
    if row is None:
        _add_error(
            errors,
            code="UNKNOWN_DOMAIN_PACK",
            field="spec.domain.pack_id",
            message=f"Domain pack {pack_id!r} is not registered.",
            actionable_fix=(
                "List packs at `GET /api/domain-packs` or remove the field "
                "from the manifest."
            ),
        )


async def _validate_domain_profile(
    db: AsyncSession,
    manifest: BrewslmManifest,
    errors: list[ManifestValidationIssue],
) -> None:
    profile_id = (manifest.spec.domain.profile_id or "").strip()
    if not profile_id:
        return
    row = (
        await db.execute(
            select(DomainProfile).where(DomainProfile.profile_id == profile_id)
        )
    ).scalar_one_or_none()
    if row is None:
        _add_error(
            errors,
            code="UNKNOWN_DOMAIN_PROFILE",
            field="spec.domain.profile_id",
            message=f"Domain profile {profile_id!r} is not registered.",
            actionable_fix=(
                "List profiles at `GET /api/domain-profiles` or remove the "
                "field from the manifest."
            ),
        )


async def _validate_base_model(
    db: AsyncSession,
    manifest: BrewslmManifest,
    warnings: list[ManifestValidationIssue],
) -> None:
    """Soft-check: warn if base_model isn't already in the registry.

    The training launch path lazily registers a model on first use, so an
    unknown id is recoverable. We surface it as a warning with a fix
    hint rather than an error.
    """
    base_model = (manifest.spec.model.base_model or "").strip()
    if not base_model:
        return
    row = (
        await db.execute(
            select(BaseModelRegistryEntry).where(
                BaseModelRegistryEntry.source_ref == base_model
            )
        )
    ).scalar_one_or_none()
    if row is None:
        _add_warning(
            warnings,
            code="BASE_MODEL_NOT_IN_REGISTRY",
            field="spec.model.base_model",
            message=(
                f"Base model {base_model!r} is not currently in the registry. "
                "It will be auto-registered on first training run."
            ),
            actionable_fix=(
                "Optional: pre-register it with `brewslm registry add-model` "
                "to lock the cache fingerprint before applying."
            ),
        )


def _validate_blueprint(
    manifest: BrewslmManifest, errors: list[ManifestValidationIssue]
) -> None:
    section = manifest.spec.blueprint
    if section is None:
        return
    contract = _section_to_blueprint_contract(section)
    result = validate_domain_blueprint(contract)
    if result.ok:
        return
    for raw in result.errors:
        _add_error(
            errors,
            code=f"BLUEPRINT_{str(raw.code or 'INVALID').upper()}",
            field=f"spec.blueprint.{raw.field}" if raw.field else "spec.blueprint",
            message=str(raw.message or "Blueprint failed validation."),
            actionable_fix=str(raw.actionable_fix or "Edit the blueprint section to satisfy the contract."),
        )


def _validate_data_sources(
    manifest: BrewslmManifest, errors: list[ManifestValidationIssue]
) -> None:
    seen: set[str] = set()
    for index, ds in enumerate(manifest.spec.data_sources):
        if ds.type not in _VALID_DATASET_TYPES:
            _add_error(
                errors,
                code="UNKNOWN_DATASET_TYPE",
                field=f"spec.data_sources[{index}].type",
                message=(
                    f"Dataset type {ds.type!r} is not one of: "
                    f"{', '.join(sorted(_VALID_DATASET_TYPES))}."
                ),
                actionable_fix=(
                    "Pick a supported dataset type (e.g. 'raw', 'cleaned', "
                    "'gold_dev', 'train')."
                ),
            )
        if ds.name in seen:
            _add_error(
                errors,
                code="DUPLICATE_DATASET_NAME",
                field=f"spec.data_sources[{index}].name",
                message=f"Dataset name {ds.name!r} is duplicated in the manifest.",
                actionable_fix="Rename one of the datasets so each name is unique.",
            )
        seen.add(ds.name)


def _validate_adapters(
    manifest: BrewslmManifest, errors: list[ManifestValidationIssue]
) -> None:
    seen: set[str] = set()
    for index, adapter in enumerate(manifest.spec.adapters):
        if adapter.name in seen:
            _add_error(
                errors,
                code="DUPLICATE_ADAPTER_NAME",
                field=f"spec.adapters[{index}].name",
                message=f"Adapter name {adapter.name!r} is duplicated in the manifest.",
                actionable_fix="Rename one of the adapters so each name is unique.",
            )
        seen.add(adapter.name)


def _validate_workflow(
    manifest: BrewslmManifest, errors: list[ManifestValidationIssue]
) -> None:
    stage = (manifest.spec.workflow.pipeline_stage or "").strip()
    if stage and stage not in _VALID_PIPELINE_STAGES:
        _add_error(
            errors,
            code="UNKNOWN_PIPELINE_STAGE",
            field="spec.workflow.pipeline_stage",
            message=(
                f"Pipeline stage {stage!r} is not one of: "
                f"{', '.join(sorted(_VALID_PIPELINE_STAGES))}."
            ),
            actionable_fix="Pick a supported pipeline stage value.",
        )


async def validate_manifest(
    db: AsyncSession, *, manifest: BrewslmManifest
) -> ManifestValidationResult:
    """Cross-reference the manifest against live catalogs.

    Pydantic schema validation is assumed to have already run (the API
    layer rejects with 400 before we get here). This entry point is for
    *semantic* checks: do the referenced ids exist, are types valid,
    is the blueprint internally consistent.
    """
    errors: list[ManifestValidationIssue] = []
    warnings: list[ManifestValidationIssue] = []

    _validate_blueprint(manifest, errors)
    _validate_workflow(manifest, errors)
    _validate_data_sources(manifest, errors)
    _validate_adapters(manifest, errors)
    await _validate_target_profile(manifest, errors)
    await _validate_eval_pack(manifest, errors)
    await _validate_domain_pack(db, manifest, errors)
    await _validate_domain_profile(db, manifest, errors)
    await _validate_base_model(db, manifest, warnings)

    return ManifestValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# -- Diff -------------------------------------------------------------------


async def diff_manifest_against_project(
    db: AsyncSession, *, manifest: BrewslmManifest, project_id: int
) -> ManifestApplyPlan:
    return await deserialize_manifest_to_apply_plan(
        db, manifest=manifest, project_id=project_id
    )


# -- Apply ------------------------------------------------------------------


def _section_to_blueprint_contract(section: BlueprintSection) -> DomainBlueprintContract:
    return DomainBlueprintContract(
        domain_name=section.domain_name,
        problem_statement=section.problem_statement,
        target_user_persona=section.target_user_persona,
        task_family=section.task_family,
        input_modality=section.input_modality,
        expected_output_schema=dict(section.expected_output_schema),
        expected_output_examples=list(section.expected_output_examples),
        safety_compliance_notes=list(section.safety_compliance_notes),
        deployment_target_constraints=dict(section.deployment_target_constraints),
        success_metrics=[
            SuccessMetric.model_validate(m.model_dump(mode="json"))
            for m in section.success_metrics
        ],
        glossary=[
            GlossaryEntry.model_validate(g.model_dump(mode="json"))
            for g in section.glossary
        ],
        confidence_score=section.confidence_score,
        unresolved_assumptions=list(section.unresolved_assumptions),
    )


async def _resolve_pack_int_id(db: AsyncSession, pack_id: str | None) -> int | None:
    if not pack_id:
        return None
    row = (
        await db.execute(select(DomainPack).where(DomainPack.pack_id == pack_id))
    ).scalar_one_or_none()
    return int(row.id) if row is not None else None


async def _resolve_profile_int_id(db: AsyncSession, profile_id: str | None) -> int | None:
    if not profile_id:
        return None
    row = (
        await db.execute(
            select(DomainProfile).where(DomainProfile.profile_id == profile_id)
        )
    ).scalar_one_or_none()
    return int(row.id) if row is not None else None


def _coerce_pipeline_stage(value: str | None) -> PipelineStage:
    token = (value or "ingestion").strip().lower()
    if token in _VALID_PIPELINE_STAGES:
        return PipelineStage(token)
    return PipelineStage.INGESTION


async def _create_project_from_manifest(
    db: AsyncSession, *, manifest: BrewslmManifest
) -> Project:
    pack_int = await _resolve_pack_int_id(db, manifest.spec.domain.pack_id)
    profile_int = await _resolve_profile_int_id(db, manifest.spec.domain.profile_id)

    project = Project(
        name=manifest.metadata.name,
        description=manifest.metadata.description,
        status=ProjectStatus.DRAFT,
        pipeline_stage=_coerce_pipeline_stage(manifest.spec.workflow.pipeline_stage),
        base_model_name=manifest.spec.model.base_model or "",
        domain_pack_id=pack_int,
        domain_profile_id=profile_int,
        target_profile_id=manifest.spec.workflow.target_profile_id,
        training_preferred_plan_profile=manifest.spec.workflow.training_preferred_plan_profile,
        evaluation_preferred_pack_id=manifest.spec.eval_pack.pack_id,
        beginner_mode=manifest.spec.workflow.beginner_mode,
        gate_policy=dict(manifest.spec.workflow.gate_policy or {}) or None,
        budget_settings=dict(manifest.spec.workflow.budget_settings or {}) or None,
        dataset_adapter_preset=dict(manifest.spec.training_plan.config or {}),
    )
    db.add(project)
    await db.flush()
    await db.refresh(project)
    return project


async def _apply_workflow_updates(
    db: AsyncSession, *, project: Project, manifest: BrewslmManifest
) -> list[str]:
    """Update project + workflow + domain + model + eval-pack fields in place.

    Returns the list of field names that actually changed (so the apply
    response can report a precise summary).
    """
    changed: list[str] = []
    workflow = manifest.spec.workflow

    if project.name != manifest.metadata.name:
        project.name = manifest.metadata.name
        changed.append("name")
    if (project.description or "") != manifest.metadata.description:
        project.description = manifest.metadata.description
        changed.append("description")

    new_stage = _coerce_pipeline_stage(workflow.pipeline_stage)
    if project.pipeline_stage != new_stage:
        project.pipeline_stage = new_stage
        changed.append("pipeline_stage")

    if bool(project.beginner_mode) != bool(workflow.beginner_mode):
        project.beginner_mode = bool(workflow.beginner_mode)
        changed.append("beginner_mode")

    if (project.target_profile_id or "") != (workflow.target_profile_id or ""):
        project.target_profile_id = workflow.target_profile_id
        changed.append("target_profile_id")

    if (project.training_preferred_plan_profile or "") != (
        workflow.training_preferred_plan_profile or ""
    ):
        project.training_preferred_plan_profile = (
            workflow.training_preferred_plan_profile
        )
        changed.append("training_preferred_plan_profile")

    if dict(project.gate_policy or {}) != dict(workflow.gate_policy or {}):
        project.gate_policy = dict(workflow.gate_policy or {}) or None
        changed.append("gate_policy")
    if dict(project.budget_settings or {}) != dict(workflow.budget_settings or {}):
        project.budget_settings = dict(workflow.budget_settings or {}) or None
        changed.append("budget_settings")

    # Model.
    if (project.base_model_name or "") != (manifest.spec.model.base_model or ""):
        project.base_model_name = manifest.spec.model.base_model or ""
        changed.append("base_model")

    # Domain.
    new_pack_int = await _resolve_pack_int_id(db, manifest.spec.domain.pack_id)
    if project.domain_pack_id != new_pack_int:
        project.domain_pack_id = new_pack_int
        changed.append("domain_pack_id")
    new_profile_int = await _resolve_profile_int_id(db, manifest.spec.domain.profile_id)
    if project.domain_profile_id != new_profile_int:
        project.domain_profile_id = new_profile_int
        changed.append("domain_profile_id")

    # Eval pack preference.
    if (project.evaluation_preferred_pack_id or "") != (manifest.spec.eval_pack.pack_id or ""):
        project.evaluation_preferred_pack_id = manifest.spec.eval_pack.pack_id or None
        changed.append("evaluation_preferred_pack_id")

    # Training plan preset (flat hyperparam bag stored on the project).
    new_preset = dict(manifest.spec.training_plan.config or {})
    if dict(project.dataset_adapter_preset or {}) != new_preset:
        project.dataset_adapter_preset = new_preset
        changed.append("dataset_adapter_preset")

    return changed


async def _apply_blueprint(
    db: AsyncSession,
    *,
    project: Project,
    section: BlueprintSection | None,
) -> ManifestApplyAction | None:
    if section is None:
        return None
    contract = _section_to_blueprint_contract(section)

    # Save as a new revision (versioning is handled by the service).
    try:
        revision = await save_domain_blueprint_revision(
            db,
            project_id=int(project.id),
            blueprint=contract,
            source="manifest_apply",
            brief_text=section.problem_statement,
            analysis_metadata={"applied_via": "brewslm_manifest"},
            status=DomainBlueprintStatus.DRAFT,
        )
    except DomainBlueprintValidationError as exc:
        return ManifestApplyAction(
            target="blueprint",
            operation="update",
            after=section.model_dump(mode="json"),
            reason=f"Blueprint validation failed: {exc}",
        )

    await apply_domain_blueprint_revision(
        db,
        project_id=int(project.id),
        version=int(revision.version),
        adopt_project_description=False,  # honor the manifest's metadata.description
        adopt_target_profile=False,
        set_beginner_mode=False,
    )
    return ManifestApplyAction(
        target="blueprint",
        operation="update",
        name=contract.domain_name,
        after={"version": int(revision.version), **section.model_dump(mode="json")},
        reason="Applied as new active blueprint revision.",
    )


async def _upsert_data_sources(
    db: AsyncSession,
    *,
    project: Project,
    desired: list[DataSourceSpec],
) -> list[ManifestApplyAction]:
    actions: list[ManifestApplyAction] = []
    rows = (
        await db.execute(
            select(Dataset).where(Dataset.project_id == project.id)
        )
    ).scalars().all()
    by_name = {row.name: row for row in rows}

    for spec in desired:
        existing = by_name.get(spec.name)
        try:
            ds_type = DatasetType(spec.type)
        except ValueError:
            actions.append(
                ManifestApplyAction(
                    target="data_source",
                    operation="update",
                    name=spec.name,
                    reason=f"Skipped — dataset type {spec.type!r} is unknown.",
                )
            )
            continue
        if existing is None:
            row = Dataset(
                project_id=project.id,
                name=spec.name,
                dataset_type=ds_type,
                description=spec.description,
                record_count=spec.record_count,
                file_path=spec.file_path or "",
                metadata_=dict(spec.metadata or {}),
            )
            db.add(row)
            await db.flush()
            await db.refresh(row)
            for v in spec.versions:
                db.add(
                    DatasetVersion(
                        dataset_id=row.id,
                        version=v.version,
                        file_path=v.file_path or "",
                        record_count=v.record_count,
                        manifest={"applied_via": "brewslm_manifest"},
                    )
                )
            actions.append(
                ManifestApplyAction(
                    target="data_source",
                    operation="create",
                    name=spec.name,
                    after=spec.model_dump(mode="json"),
                )
            )
        else:
            changed: list[str] = []
            if existing.dataset_type != ds_type:
                existing.dataset_type = ds_type
                changed.append("type")
            if (existing.description or "") != spec.description:
                existing.description = spec.description
                changed.append("description")
            if int(existing.record_count or 0) != int(spec.record_count or 0):
                existing.record_count = int(spec.record_count or 0)
                changed.append("record_count")
            if (existing.file_path or "") != (spec.file_path or ""):
                existing.file_path = spec.file_path or ""
                changed.append("file_path")
            if dict(existing.metadata_ or {}) != dict(spec.metadata or {}):
                existing.metadata_ = dict(spec.metadata or {})
                changed.append("metadata")
            if changed:
                actions.append(
                    ManifestApplyAction(
                        target="data_source",
                        operation="update",
                        name=spec.name,
                        after=spec.model_dump(mode="json"),
                        fields_changed=changed,
                    )
                )
            else:
                actions.append(
                    ManifestApplyAction(
                        target="data_source", operation="noop", name=spec.name
                    )
                )
    return actions


async def _upsert_adapters(
    db: AsyncSession,
    *,
    project: Project,
    desired: list[AdapterSpec],
) -> list[ManifestApplyAction]:
    actions: list[ManifestApplyAction] = []
    rows = (
        await db.execute(
            select(DatasetAdapterDefinition)
            .where(DatasetAdapterDefinition.project_id == project.id)
            .order_by(
                DatasetAdapterDefinition.adapter_name.asc(),
                DatasetAdapterDefinition.version.desc(),
            )
        )
    ).scalars().all()

    latest_by_name: dict[str, DatasetAdapterDefinition] = {}
    for row in rows:
        latest_by_name.setdefault(str(row.adapter_name), row)

    for spec in desired:
        existing = latest_by_name.get(spec.name)
        # An adapter version is immutable once written — if any of the
        # config-bearing fields drift, we author a new row at version+1
        # rather than mutating the existing row in place. This mirrors
        # the Adapter Studio versioning contract.
        config_fields = {
            "task_profile": spec.task_profile,
            "source_type": spec.source_type,
            "source_ref": spec.source_ref,
            "base_adapter_id": spec.base_adapter_id,
            "field_mapping": dict(spec.field_mapping or {}),
            "adapter_config": dict(spec.adapter_config or {}),
            "output_contract": dict(spec.output_contract or {}),
        }
        if existing is None:
            new_row = DatasetAdapterDefinition(
                project_id=project.id,
                adapter_name=spec.name,
                version=max(1, int(spec.version or 1)),
                status=spec.status or "active",
                task_profile=spec.task_profile,
                source_type=spec.source_type,
                source_ref=spec.source_ref,
                base_adapter_id=spec.base_adapter_id,
                field_mapping=config_fields["field_mapping"],
                adapter_config=config_fields["adapter_config"],
                output_contract=config_fields["output_contract"],
            )
            db.add(new_row)
            actions.append(
                ManifestApplyAction(
                    target="adapter",
                    operation="create",
                    name=spec.name,
                    after=spec.model_dump(mode="json"),
                )
            )
            continue

        existing_fields = {
            "task_profile": existing.task_profile,
            "source_type": existing.source_type,
            "source_ref": existing.source_ref,
            "base_adapter_id": existing.base_adapter_id,
            "field_mapping": dict(existing.field_mapping or {}),
            "adapter_config": dict(existing.adapter_config or {}),
            "output_contract": dict(existing.output_contract or {}),
        }
        if existing_fields == config_fields and (existing.status or "") == (
            spec.status or "active"
        ):
            actions.append(
                ManifestApplyAction(target="adapter", operation="noop", name=spec.name)
            )
            continue

        new_version = int(existing.version or 1) + 1
        new_row = DatasetAdapterDefinition(
            project_id=project.id,
            adapter_name=spec.name,
            version=new_version,
            status=spec.status or "active",
            task_profile=spec.task_profile,
            source_type=spec.source_type,
            source_ref=spec.source_ref,
            base_adapter_id=spec.base_adapter_id,
            field_mapping=config_fields["field_mapping"],
            adapter_config=config_fields["adapter_config"],
            output_contract=config_fields["output_contract"],
        )
        db.add(new_row)
        actions.append(
            ManifestApplyAction(
                target="adapter",
                operation="update",
                name=spec.name,
                after={"version": new_version, **spec.model_dump(mode="json")},
                fields_changed=[
                    k for k in config_fields if existing_fields.get(k) != config_fields.get(k)
                ],
            )
        )
    return actions


async def apply_manifest(
    db: AsyncSession,
    *,
    manifest: BrewslmManifest,
    project_id: int | None = None,
    plan_only: bool = False,
) -> ManifestApplyResult:
    """Validate then apply the manifest to a project (or create a new one).

    ``plan_only=True`` runs the deserializer only — no DB writes — and
    returns the result with an empty ``applied_actions`` list. The
    validation block is still populated so the CLI / frontend can show
    structured errors for both modes from a single endpoint.
    """
    validation = await validate_manifest(db, manifest=manifest)

    # Build the diff against current project state (or "all create" for
    # new projects). This is what plan_only callers want and what the
    # apply response echoes back.
    plan = await deserialize_manifest_to_apply_plan(
        db, manifest=manifest, project_id=project_id
    )

    if not validation.ok or plan_only:
        return ManifestApplyResult(
            project_id=int(project_id) if project_id is not None else 0,
            project_name=manifest.metadata.name,
            plan_only=True if not validation.ok else plan_only,
            plan=plan,
            validation=validation,
            applied_actions=[],
        )

    applied: list[ManifestApplyAction] = []

    if project_id is None:
        project = await _create_project_from_manifest(db, manifest=manifest)
        applied.append(
            ManifestApplyAction(
                target="project",
                operation="create",
                name=project.name,
                after=manifest.metadata.model_dump(mode="json"),
            )
        )
    else:
        project = (
            await db.execute(select(Project).where(Project.id == project_id))
        ).scalar_one_or_none()
        if project is None:
            raise ValueError("project_not_found")
        changed = await _apply_workflow_updates(db, project=project, manifest=manifest)
        if changed:
            applied.append(
                ManifestApplyAction(
                    target="project",
                    operation="update",
                    name=project.name,
                    fields_changed=changed,
                )
            )
        else:
            applied.append(
                ManifestApplyAction(
                    target="project", operation="noop", name=project.name
                )
            )

    bp_action = await _apply_blueprint(
        db, project=project, section=manifest.spec.blueprint
    )
    if bp_action is not None:
        applied.append(bp_action)

    applied.extend(
        await _upsert_data_sources(db, project=project, desired=manifest.spec.data_sources)
    )
    applied.extend(
        await _upsert_adapters(db, project=project, desired=manifest.spec.adapters)
    )

    await db.commit()
    await db.refresh(project)

    return ManifestApplyResult(
        project_id=int(project.id),
        project_name=project.name,
        plan_only=False,
        plan=plan,
        validation=validation,
        applied_actions=applied,
    )

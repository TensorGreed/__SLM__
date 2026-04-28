"""Pipeline-as-code manifest contract (priority.md P21).

The ``brewslm.yaml`` manifest is a *project-level* declarative spec — the
state a project should be in, expressed as a single YAML/JSON document
that can be checked into git, code-reviewed, and replayed. It is
deliberately distinct from the per-run :class:`TrainingManifest`
(``app/models/training_manifest.py``), which is an immutable post-launch
record for a single training run.

The schema is split into a small set of named *sections* that mirror the
project's natural shape:

- ``workflow`` — project-level toggles (beginner mode, target profile,
  gate / budget policy).
- ``blueprint`` — the active :class:`DomainBlueprintRevision` for the
  project (problem statement, success metrics, glossary, etc.).
- ``domain`` — references to the domain pack / profile catalog.
- ``model`` — base-model selection and registry pointer.
- ``data_sources`` — datasets registered on the project.
- ``adapters`` — versioned dataset adapters from Adapter Studio.
- ``training_plan`` — preferred training mode + the resolved
  :class:`TrainingConfig` to launch with.
- ``eval_pack`` — evaluation pack id + dataset / metric selection.
- ``export`` / ``deployment`` — export formats and deployment target
  hints (Wave F will fill this out further).

The Pydantic models on this module are also the JSON-Schema source of
truth — ``BrewslmManifest.model_json_schema()`` is what the validator
service (P22) and the frontend (P24) consume.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


MANIFEST_API_VERSION = "brewslm/v1"
MANIFEST_KIND = "Project"


class _StrictBase(BaseModel):
    """Base for manifest sections.

    ``extra='forbid'`` is intentional: unknown keys in a checked-in
    ``brewslm.yaml`` are almost always typos or stale fields from an
    older schema version, and we want validate / diff / apply to flag
    them rather than silently ignore.
    """

    model_config = ConfigDict(extra="forbid")


class ManifestMetadata(_StrictBase):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=4000)
    labels: dict[str, str] = Field(default_factory=dict)


class WorkflowSection(_StrictBase):
    beginner_mode: bool = False
    pipeline_stage: str = "ingestion"
    target_profile_id: str | None = "vllm_server"
    training_preferred_plan_profile: str | None = "balanced"
    gate_policy: dict[str, Any] = Field(default_factory=dict)
    budget_settings: dict[str, Any] = Field(default_factory=dict)


class BlueprintGlossaryEntry(_StrictBase):
    term: str
    plain_language: str
    category: str = "general"
    example: str | None = None


class BlueprintSuccessMetric(_StrictBase):
    metric_id: str
    label: str
    target: str = ""
    why_it_matters: str = ""


class BlueprintSection(_StrictBase):
    domain_name: str = ""
    problem_statement: str = ""
    target_user_persona: str = ""
    task_family: str = "instruction_sft"
    input_modality: str = "text"
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    expected_output_examples: list[Any] = Field(default_factory=list)
    safety_compliance_notes: list[str] = Field(default_factory=list)
    deployment_target_constraints: dict[str, Any] = Field(default_factory=dict)
    success_metrics: list[BlueprintSuccessMetric] = Field(default_factory=list)
    glossary: list[BlueprintGlossaryEntry] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unresolved_assumptions: list[str] = Field(default_factory=list)
    # Informational provenance — round-tripped, not used for apply.
    version: int | None = None
    source: str | None = None


class DomainSection(_StrictBase):
    pack_id: str | None = None
    profile_id: str | None = None


class ModelSection(_StrictBase):
    base_model: str = ""
    cache_fingerprint: str | None = None
    source_ref: str | None = None
    # Registry id is informational — apply re-resolves from base_model.
    registry_id: int | None = None


class DataSourceVersionSpec(_StrictBase):
    version: int
    record_count: int = 0
    file_path: str | None = None


class DataSourceSpec(_StrictBase):
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1, max_length=64)
    description: str = ""
    record_count: int = 0
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    versions: list[DataSourceVersionSpec] = Field(default_factory=list)


class AdapterSpec(_StrictBase):
    name: str = Field(..., min_length=1, max_length=128)
    version: int = Field(default=1, ge=1)
    status: str = "active"
    base_adapter_id: str = "default-canonical"
    task_profile: str | None = None
    source_type: str = "raw"
    source_ref: str | None = None
    field_mapping: dict[str, Any] = Field(default_factory=dict)
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)


class TrainingPlanSection(_StrictBase):
    training_mode: str = "sft"
    plan_profile: str | None = "balanced"
    preferred_runtime_id: str | None = None
    # The full TrainingConfig as a flat dict so the schema doesn't have
    # to track every recipe knob. P22 validates this against TrainingConfig.
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("training_mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        token = str(value or "").strip().lower()
        return token or "sft"


class EvalPackSection(_StrictBase):
    pack_id: str | None = None
    datasets: list[str] = Field(
        default_factory=lambda: ["gold_dev", "gold_test"]
    )
    eval_types: list[str] = Field(
        default_factory=lambda: ["exact_match", "f1", "hallucination", "safety"]
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class ExportSection(_StrictBase):
    formats: list[str] = Field(default_factory=list)
    quantization: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class DeploymentSection(_StrictBase):
    target_profile_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class BrewslmManifestSpec(_StrictBase):
    workflow: WorkflowSection = Field(default_factory=WorkflowSection)
    blueprint: BlueprintSection | None = None
    domain: DomainSection = Field(default_factory=DomainSection)
    model: ModelSection = Field(default_factory=ModelSection)
    data_sources: list[DataSourceSpec] = Field(default_factory=list)
    adapters: list[AdapterSpec] = Field(default_factory=list)
    training_plan: TrainingPlanSection = Field(default_factory=TrainingPlanSection)
    eval_pack: EvalPackSection = Field(default_factory=EvalPackSection)
    export: ExportSection = Field(default_factory=ExportSection)
    deployment: DeploymentSection = Field(default_factory=DeploymentSection)


class BrewslmManifest(_StrictBase):
    api_version: str = Field(default=MANIFEST_API_VERSION)
    kind: str = Field(default=MANIFEST_KIND)
    metadata: ManifestMetadata
    spec: BrewslmManifestSpec = Field(default_factory=BrewslmManifestSpec)

    @field_validator("api_version")
    @classmethod
    def _check_api_version(cls, value: str) -> str:
        token = str(value or "").strip()
        if token != MANIFEST_API_VERSION:
            raise ValueError(
                f"unsupported_api_version: {token!r} (expected {MANIFEST_API_VERSION!r})"
            )
        return token

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, value: str) -> str:
        token = str(value or "").strip()
        if token != MANIFEST_KIND:
            raise ValueError(
                f"unsupported_kind: {token!r} (expected {MANIFEST_KIND!r})"
            )
        return token


# -- Apply-plan structures ---------------------------------------------------

class ManifestApplyAction(_StrictBase):
    """A single change implied by applying a manifest to a project."""

    target: str  # e.g. 'project', 'blueprint', 'data_source', 'adapter', 'training_plan'
    operation: str  # 'create' | 'update' | 'noop' | 'delete'
    name: str | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    fields_changed: list[str] = Field(default_factory=list)
    reason: str = ""


class ManifestApplyPlan(_StrictBase):
    project_id: int | None = None
    project_name: str
    api_version: str = MANIFEST_API_VERSION
    actions: list[ManifestApplyAction] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)

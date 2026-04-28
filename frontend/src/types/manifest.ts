/**
 * TypeScript shapes mirroring `backend/app/schemas/brewslm_manifest.py` and
 * `backend/app/services/manifest_apply_service.py` (priority.md P21 + P22).
 *
 * These types intentionally mirror the Pydantic contracts: `extra='forbid'`
 * on the backend means the wire shape is closed, so we model only the
 * fields the frontend actually consumes — anything missing from here is
 * a real schema gap, not a backend-vs-frontend drift.
 */

export interface ManifestMetadata {
    name: string;
    description?: string;
    labels?: Record<string, string>;
}

export interface ManifestWorkflowSection {
    beginner_mode?: boolean;
    pipeline_stage?: string;
    target_profile_id?: string | null;
    training_preferred_plan_profile?: string | null;
    gate_policy?: Record<string, unknown>;
    budget_settings?: Record<string, unknown>;
}

export interface ManifestBlueprintSection {
    domain_name?: string;
    problem_statement?: string;
    target_user_persona?: string;
    task_family?: string;
    input_modality?: string;
    expected_output_schema?: Record<string, unknown>;
    expected_output_examples?: unknown[];
    safety_compliance_notes?: string[];
    deployment_target_constraints?: Record<string, unknown>;
    success_metrics?: Array<{
        metric_id: string;
        label: string;
        target?: string;
        why_it_matters?: string;
    }>;
    glossary?: Array<{
        term: string;
        plain_language: string;
        category?: string;
        example?: string | null;
    }>;
    confidence_score?: number;
    unresolved_assumptions?: string[];
    version?: number | null;
    source?: string | null;
}

export interface ManifestDomainSection {
    pack_id?: string | null;
    profile_id?: string | null;
}

export interface ManifestModelSection {
    base_model?: string;
    cache_fingerprint?: string | null;
    source_ref?: string | null;
    registry_id?: number | null;
}

export interface ManifestDataSourceSpec {
    name: string;
    type: string;
    description?: string;
    record_count?: number;
    file_path?: string | null;
    metadata?: Record<string, unknown>;
    versions?: Array<{
        version: number;
        record_count?: number;
        file_path?: string | null;
    }>;
}

export interface ManifestAdapterSpec {
    name: string;
    version?: number;
    status?: string;
    base_adapter_id?: string;
    task_profile?: string | null;
    source_type?: string;
    source_ref?: string | null;
    field_mapping?: Record<string, unknown>;
    adapter_config?: Record<string, unknown>;
    output_contract?: Record<string, unknown>;
}

export interface ManifestTrainingPlanSection {
    training_mode?: string;
    plan_profile?: string | null;
    preferred_runtime_id?: string | null;
    config?: Record<string, unknown>;
}

export interface ManifestEvalPackSection {
    pack_id?: string | null;
    datasets?: string[];
    eval_types?: string[];
    extra?: Record<string, unknown>;
}

export interface ManifestExportSection {
    formats?: string[];
    quantization?: string | null;
    extra?: Record<string, unknown>;
}

export interface ManifestDeploymentSection {
    target_profile_id?: string | null;
    extra?: Record<string, unknown>;
}

export interface BrewslmManifestSpec {
    workflow?: ManifestWorkflowSection;
    blueprint?: ManifestBlueprintSection | null;
    domain?: ManifestDomainSection;
    model?: ManifestModelSection;
    data_sources?: ManifestDataSourceSpec[];
    adapters?: ManifestAdapterSpec[];
    training_plan?: ManifestTrainingPlanSection;
    eval_pack?: ManifestEvalPackSection;
    export?: ManifestExportSection;
    deployment?: ManifestDeploymentSection;
}

export interface BrewslmManifest {
    api_version: string;
    kind: string;
    metadata: ManifestMetadata;
    spec: BrewslmManifestSpec;
}

// -- Validation -----------------------------------------------------------

export type ManifestIssueSeverity = 'error' | 'warning';

export interface ManifestValidationIssue {
    code: string;
    severity: ManifestIssueSeverity;
    field?: string;
    message: string;
    actionable_fix?: string;
}

export interface ManifestValidationResult {
    ok: boolean;
    errors: ManifestValidationIssue[];
    warnings: ManifestValidationIssue[];
}

// -- Apply plan -----------------------------------------------------------

export type ManifestActionOperation = 'create' | 'update' | 'noop' | 'delete';

export interface ManifestApplyAction {
    target: string;
    operation: ManifestActionOperation | string;
    name?: string | null;
    before?: Record<string, unknown> | null;
    after?: Record<string, unknown> | null;
    fields_changed?: string[];
    reason?: string;
}

export interface ManifestApplyPlan {
    project_id?: number | null;
    project_name: string;
    api_version: string;
    actions: ManifestApplyAction[];
    warnings: string[];
    summary: Record<string, number>;
}

export interface ManifestApplyResult {
    project_id: number;
    project_name: string;
    plan_only: boolean;
    plan: ManifestApplyPlan;
    validation: ManifestValidationResult;
    applied_actions: ManifestApplyAction[];
}

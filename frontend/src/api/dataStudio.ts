import api from './client';

export type DataStudioVerdict = 'blocked' | 'needs_work' | 'ready';
export type DataStudioSourcesVerdict = 'empty' | 'attention' | 'healthy';
export type DataStudioMappingVerdict = 'empty' | 'attention' | 'ready';
export type DataStudioDomainVerdict = 'unknown' | 'attention' | 'confirmed';
export type DataStudioGoldSetVerdict = 'empty' | 'attention' | 'ready';
export type DataStudioAssistFocus = 'mapping' | 'domain';
export type DataStudioAssistProvider = 'ollama' | 'openai_compatible';
export type DataStudioAssistStatus = 'ok' | 'unavailable' | 'invalid_response';
export type DataStudioIssueSeverity = 'blocker' | 'warning' | 'info';

export interface DataStudioRecipeSummary {
    id: string;
    name: string;
    task_profile: string;
    adapter_id?: string;
    default_input_column?: string;
    default_output_column?: string;
}

export interface DataStudioDomainSummary {
    profile_id?: string | null;
    profile_source?: string | null;
    pack_id?: string | null;
    pack_source?: string | null;
    display_name?: string | null;
}

export interface DataStudioRowCounts {
    trainable: number;
    raw: number;
    cleaned: number;
    gold: number;
    synthetic_total: number;
    synthetic_pending: number;
    synthetic_accepted: number;
    prepared: number;
    train: number;
    validation: number;
    test: number;
}

export interface DataStudioSourceSummary {
    dataset_count: number;
    documents_total: number;
    documents_accepted: number;
    documents_processing: number;
    documents_pending: number;
    documents_error: number;
}

export interface DataStudioIssue {
    id: string;
    severity: DataStudioIssueSeverity;
    title: string;
    message: string;
    action_label: string;
    target_tab: string;
}

export interface DataStudioPrimaryAction {
    label: string;
    target_tab: string;
    reason: string;
}

export interface DataStudioOverview {
    project_id: number;
    verdict: DataStudioVerdict;
    recipe: DataStudioRecipeSummary | null;
    domain: DataStudioDomainSummary;
    row_counts: DataStudioRowCounts;
    source_summary: DataStudioSourceSummary;
    issues: DataStudioIssue[];
    primary_action: DataStudioPrimaryAction;
}

export async function getDataStudioOverview(projectId: number): Promise<DataStudioOverview> {
    const resp = await api.get(`/projects/${projectId}/data-studio/overview`);
    return resp.data as DataStudioOverview;
}

export interface DataStudioSourceTotals {
    dataset_count: number;
    document_count: number;
    row_count: number;
    accepted_documents: number;
    pending_documents: number;
    processing_documents: number;
    error_documents: number;
    rejected_documents: number;
}

export interface DataStudioDatasetGroup {
    dataset_type: string;
    dataset_count: number;
    row_count: number;
    locked_count: number;
    with_file_count: number;
}

export interface DataStudioRecentDocument {
    id: number;
    dataset_id: number;
    dataset_name: string;
    dataset_type: string;
    filename: string;
    file_type: string;
    status: string;
    source: string;
    sensitivity: string;
    file_size_bytes: number;
    chunk_count: number;
    quality_score?: number | null;
    ingested_at?: string | null;
}

export interface DataStudioSources {
    project_id: number;
    verdict: DataStudioSourcesVerdict;
    totals: DataStudioSourceTotals;
    dataset_groups: DataStudioDatasetGroup[];
    recent_documents: DataStudioRecentDocument[];
    issues: DataStudioIssue[];
}

export async function getDataStudioSources(projectId: number): Promise<DataStudioSources> {
    const resp = await api.get(`/projects/${projectId}/data-studio/sources`);
    return resp.data as DataStudioSources;
}

export interface DataStudioMappingPreference {
    source: string;
    adapter_id: string;
    task_profile?: string | null;
    field_mapping: Record<string, string>;
    field_mapping_count: number;
}

export interface DataStudioEffectiveMapping {
    source: string;
    adapter_id: string;
    requested_adapter_id?: string;
    task_profile?: string | null;
    requested_task_profile?: string | null;
    adapter_config: Record<string, unknown>;
    field_mapping: Record<string, string>;
    auto_apply?: Record<string, unknown>;
}

export interface DataStudioMappingSource {
    dataset_type: string;
    dataset_id?: number | null;
    dataset_name?: string | null;
    document_id?: number | null;
    document_name?: string | null;
    document_count?: number;
    row_count?: number;
}

export interface DataStudioRequiredFieldCoverage {
    field: string;
    present: number;
    missing: number;
    ratio: number;
}

export interface DataStudioMappingSummary {
    sampled_records: number;
    mapped_records: number;
    dropped_records: number;
    error_count: number;
    mapping_success_rate: number;
    contract_pass: boolean;
    required_fields: string[];
    required_fields_below_100: string[];
    required_field_coverage: DataStudioRequiredFieldCoverage[];
}

export interface DataStudioMappingPreviewRow {
    index: number;
    raw: Record<string, unknown>;
    mapped: Record<string, unknown>;
}

export interface DataStudioMappingDiagnostics {
    adapter_contract?: Record<string, unknown>;
    validation_report?: Record<string, unknown>;
    detection_scores?: Record<string, number>;
    auto_fix_suggestions?: Array<Record<string, unknown>>;
    compatibility_warnings?: string[];
    inferred_task_profiles?: string[];
}

export interface DataStudioMappingPreview {
    project_id: number;
    verdict: DataStudioMappingVerdict;
    recipe: DataStudioRecipeSummary | null;
    preference: DataStudioMappingPreference;
    effective_mapping: DataStudioEffectiveMapping;
    source: DataStudioMappingSource | null;
    summary: DataStudioMappingSummary;
    preview_rows: DataStudioMappingPreviewRow[];
    diagnostics: DataStudioMappingDiagnostics;
    issues: DataStudioIssue[];
}

export async function getDataStudioMappingPreview(projectId: number): Promise<DataStudioMappingPreview> {
    const resp = await api.get(`/projects/${projectId}/data-studio/mapping-preview`);
    return resp.data as DataStudioMappingPreview;
}

export interface DataStudioDetectedDomain {
    id: string;
    label: string;
    confidence: number;
    confidence_label: string;
    source: string;
    summary?: string | null;
    matched_keywords: string[];
    matched_fields: string[];
    recommended_recipes: string[];
}

export interface DataStudioAppliedDomain {
    profile_id?: string | null;
    profile_source?: string | null;
    profile_display_name?: string | null;
    profile_version?: string | null;
    pack_id?: string | null;
    pack_source?: string | null;
    pack_display_name?: string | null;
    pack_version?: string | null;
    pack_default_profile_id?: string | null;
}

export interface DataStudioDomainSource {
    dataset_type: string;
    dataset_id?: number | null;
    dataset_name?: string | null;
    document_id?: number | null;
    document_name?: string | null;
    document_count?: number;
    row_count?: number;
    sampled_records?: number;
}

export interface DataStudioDomainEvidence {
    id: string;
    title: string;
    message: string;
    score: number;
}

export interface DataStudioDomainAction {
    id: string;
    label: string;
    target_tab: string;
}

export interface DataStudioDomainRisk {
    id: string;
    severity: DataStudioIssueSeverity;
    title: string;
    message: string;
}

export interface DataStudioDomainDetection {
    project_id: number;
    verdict: DataStudioDomainVerdict;
    detected_domain: DataStudioDetectedDomain;
    applied: DataStudioAppliedDomain;
    recipe: DataStudioRecipeSummary | null;
    source: DataStudioDomainSource | null;
    evidence: DataStudioDomainEvidence[];
    suggested_actions: DataStudioDomainAction[];
    risks: DataStudioDomainRisk[];
    issues: DataStudioIssue[];
    power_details: Record<string, unknown>;
}

export async function getDataStudioDomainDetection(projectId: number): Promise<DataStudioDomainDetection> {
    const resp = await api.get(`/projects/${projectId}/data-studio/domain-detection`);
    return resp.data as DataStudioDomainDetection;
}

export interface DataStudioGoldSetValidation {
    status: string;
    trusted_examples: number;
    review_needed: number;
    locked_gold_sets: number;
    locked_versions: number;
}

export interface DataStudioGoldSetTotals {
    gold_set_count: number;
    example_count: number;
    trusted_examples: number;
    review_needed: number;
    approved_rows: number;
    pending_rows: number;
    in_review_rows: number;
    changes_requested_rows: number;
    rejected_rows: number;
    queue_pending: number;
    queue_in_progress: number;
    locked_gold_sets: number;
    draft_versions: number;
    locked_versions: number;
}

export interface DataStudioGoldSetFieldCoverage {
    field: string;
    present: number;
    missing: number;
    ratio: number;
}

export interface DataStudioGoldSetCoverage {
    source_rows: number;
    input_fields: DataStudioGoldSetFieldCoverage[];
    expected_fields: DataStudioGoldSetFieldCoverage[];
    label_fields: DataStudioGoldSetFieldCoverage[];
    field_counts: {
        input: number;
        expected: number;
        labels: number;
    };
}

export interface DataStudioGoldSetVersionSummary {
    count: number;
    draft_count: number;
    locked_count: number;
    latest?: Record<string, unknown> | null;
    active_draft?: Record<string, unknown> | null;
    latest_locked?: Record<string, unknown> | null;
}

export interface DataStudioGoldSetDataset {
    id: number;
    name: string;
    dataset_type: string;
    record_count: number;
    example_count: number;
    trusted_examples: number;
    review_needed: number;
    is_locked: boolean;
    validation_status: string;
    coverage_source: string;
    row_status_counts: Record<string, number>;
    queue_status_counts: Record<string, number>;
    versions: DataStudioGoldSetVersionSummary;
    coverage: DataStudioGoldSetCoverage;
    updated_at?: string | null;
}

export interface DataStudioGoldSetSample {
    dataset_id: number;
    dataset_name: string;
    source: string;
    status: string;
    input_preview: string;
    expected_preview: string;
}

export interface DataStudioGoldSetEntryPoint {
    label: string;
    target_tab: string;
    reason: string;
}

export interface DataStudioGoldSetWorkbench {
    project_id: number;
    verdict: DataStudioGoldSetVerdict;
    read_only: boolean;
    minimum_recommended_examples: number;
    validation: DataStudioGoldSetValidation;
    totals: DataStudioGoldSetTotals;
    datasets: DataStudioGoldSetDataset[];
    trusted_examples: DataStudioGoldSetSample[];
    coverage: DataStudioGoldSetCoverage;
    issues: DataStudioIssue[];
    entry_point: DataStudioGoldSetEntryPoint;
}

export async function getDataStudioGoldSetWorkbench(projectId: number): Promise<DataStudioGoldSetWorkbench> {
    const resp = await api.get(`/projects/${projectId}/data-studio/gold-set`);
    return resp.data as DataStudioGoldSetWorkbench;
}

export interface DataStudioAssistRequest {
    focus: DataStudioAssistFocus;
    provider: DataStudioAssistProvider;
    api_url?: string;
    api_key?: string;
    model_name: string;
}

export interface DataStudioAssistProviderSummary {
    provider: DataStudioAssistProvider;
    api_url?: string;
    model_name: string;
    api_key_configured: boolean;
    tokens_used?: number;
}

export interface DataStudioAssistSuggestion {
    id: string;
    type: string;
    title: string;
    confidence: number;
    rationale: string;
    evidence: string[];
    target_tab: string;
    requires_user_confirmation: boolean;
    suggested_field_mapping?: Record<string, string>;
}

export interface DataStudioAssistResponse {
    project_id: number;
    focus: DataStudioAssistFocus;
    status: DataStudioAssistStatus;
    provider: DataStudioAssistProviderSummary;
    source_of_truth: string;
    auto_apply: boolean;
    summary: string;
    suggestions: DataStudioAssistSuggestion[];
    deterministic_context: Record<string, unknown>;
    warnings: string[];
}

export async function runDataStudioAssist(
    projectId: number,
    payload: DataStudioAssistRequest,
): Promise<DataStudioAssistResponse> {
    const resp = await api.post(`/projects/${projectId}/data-studio/assist`, payload);
    return resp.data as DataStudioAssistResponse;
}

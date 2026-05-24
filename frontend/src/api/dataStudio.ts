import api from './client';

export type DataStudioVerdict = 'blocked' | 'needs_work' | 'ready';
export type DataStudioSourcesVerdict = 'empty' | 'attention' | 'healthy';
export type DataStudioMappingVerdict = 'empty' | 'attention' | 'ready';
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

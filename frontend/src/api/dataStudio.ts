import api from './client';

export type DataStudioVerdict = 'blocked' | 'needs_work' | 'ready';
export type DataStudioCoachVerdict = 'blocked' | 'attention' | 'ready';
export type DataStudioSourcesVerdict = 'empty' | 'attention' | 'healthy';
export type DataStudioMappingVerdict = 'empty' | 'attention' | 'ready';
export type DataStudioDomainVerdict = 'unknown' | 'attention' | 'confirmed';
export type DataStudioGoldSetVerdict = 'empty' | 'attention' | 'ready';
export type DataStudioSyntheticPlaybookVerdict = 'empty' | 'attention' | 'ready';
export type DataStudioSyntheticRecommendationVerdict = 'empty' | 'attention' | 'ready';
export type DataStudioReviewQueueVerdict = 'empty' | 'attention' | 'ready';
export type DataStudioPrepareDatasetVerdict = 'blocked' | 'attention' | 'ready';
export type DataStudioDatasetVersionVerdict = 'empty' | 'attention' | 'ready';
export type DataStudioQualitySafetyVerdict = 'blocked' | 'attention' | 'ready';
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

export interface DataStudioCoachSummary {
    blocker_count: number;
    warning_count: number;
    info_count: number;
    section_count: number;
    ready_section_count: number;
    empty_section_count: number;
    next_action_target?: string | null;
}

export interface DataStudioCoachAction {
    id: string;
    section_id: string;
    section_label: string;
    severity: DataStudioIssueSeverity;
    priority: 'high' | 'medium' | 'low' | string;
    title: string;
    message: string;
    action_label: string;
    target_tab: string;
    requires_user_confirmation: boolean;
}

export interface DataStudioCoachCheck {
    id: string;
    label: string;
    status: 'blocked' | 'attention' | 'ready' | 'empty' | string;
    verdict: string;
    target_tab: string;
    action_label: string;
    message: string;
    blocker_count: number;
    warning_count: number;
    info_count: number;
}

export interface DataStudioCoachEntryPoint {
    label: string;
    target_tab: string;
    reason: string;
    requires_confirmation: boolean;
}

export interface DataStudioCoachRail {
    project_id: number;
    verdict: DataStudioCoachVerdict;
    read_only: boolean;
    auto_apply: boolean;
    source_of_truth: string;
    summary: DataStudioCoachSummary;
    next_action: DataStudioCoachAction;
    next_steps: DataStudioCoachAction[];
    checks: DataStudioCoachCheck[];
    issues: DataStudioCoachAction[];
    entry_points: DataStudioCoachEntryPoint[];
    power_details: Record<string, unknown>;
}

export async function getDataStudioCoachRail(projectId: number): Promise<DataStudioCoachRail> {
    const resp = await api.get(`/projects/${projectId}/data-studio/coach`);
    return resp.data as DataStudioCoachRail;
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

export interface DataStudioDomainSetupGuidance {
    id: string;
    title: string;
    recommendation: string;
    why: string;
}

export interface DataStudioDomainSetupChoice {
    id: string;
    label: string;
    target: string;
    detail: string;
}

export interface DataStudioDomainSetupPreview {
    available: boolean;
    recommended: boolean;
    reason: string;
    read_only: boolean;
    requires_confirmation: boolean;
    create_mode: string;
    detected_domain_id: string;
    detected_domain_label: string;
    profile_id: string;
    pack_id: string;
    profile_exists: boolean;
    pack_exists: boolean;
    profile_status?: string | null;
    pack_status?: string | null;
    can_create_profile: boolean;
    can_create_pack: boolean;
    guidance: DataStudioDomainSetupGuidance[];
    choices: DataStudioDomainSetupChoice[];
    profile_contract: Record<string, unknown>;
    pack_contract: Record<string, unknown>;
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
    domain_setup?: DataStudioDomainSetupPreview | null;
    power_details: Record<string, unknown>;
}

export async function getDataStudioDomainDetection(projectId: number): Promise<DataStudioDomainDetection> {
    const resp = await api.get(`/projects/${projectId}/data-studio/domain-detection`);
    return resp.data as DataStudioDomainDetection;
}

export interface DataStudioDomainSetupCreateResponse {
    status: 'created' | 'already_exists' | string;
    project_id: number;
    detected_domain_id: string;
    detected_domain_label: string;
    created_profile: boolean;
    created_pack: boolean;
    assigned_to_project: boolean;
    profile: {
        profile_id: string;
        display_name: string;
        status: string;
        version: string;
    };
    pack: {
        pack_id: string;
        display_name: string;
        status: string;
        version: string;
        default_profile_id?: string | null;
    };
    next_targets: string[];
}

export async function createDataStudioDomainSetup(
    projectId: number,
): Promise<DataStudioDomainSetupCreateResponse> {
    const resp = await api.post(`/projects/${projectId}/data-studio/domain-detection/domain-setup`, {
        confirm: true,
    });
    return resp.data as DataStudioDomainSetupCreateResponse;
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

export interface DataStudioSyntheticPlaybook {
    recipe_id: string;
    mode: string;
    label: string;
}

export interface DataStudioSyntheticCatalog {
    total_playbooks: number;
    compatible_playbooks: number;
    preview_playbooks: DataStudioSyntheticPlaybook[];
    supported_recipes: string[];
    compatible_modes: string[];
}

export interface DataStudioSyntheticBackend {
    name: string;
    available: boolean;
    describe: string;
    is_default: boolean;
    is_local: boolean;
    paid_required: boolean;
}

export interface DataStudioSyntheticRecommendedBackend {
    name: string;
    available: boolean;
    describe: string;
    local_default: boolean;
    paid_required: boolean;
}

export interface DataStudioSyntheticPrerequisite {
    id: string;
    label: string;
    status: 'met' | 'attention' | 'missing' | string;
    message: string;
    target_tab: string;
}

export interface DataStudioSyntheticReviewGroup {
    synth_source: string;
    count: number;
    truncated: boolean;
}

export interface DataStudioSyntheticReviewQueue {
    dataset_id?: number | null;
    total_rows: number;
    total_pending: number;
    total_accepted: number;
    pending_group_count: number;
    accepted_group_count: number;
    top_pending_groups: DataStudioSyntheticReviewGroup[];
    top_accepted_groups: DataStudioSyntheticReviewGroup[];
}

export interface DataStudioSyntheticEntryPoint {
    label: string;
    target_tab: string;
    reason: string;
}

export interface DataStudioSyntheticPlaybookCenter {
    project_id: number;
    verdict: DataStudioSyntheticPlaybookVerdict;
    read_only: boolean;
    recipe: DataStudioRecipeSummary | null;
    catalog: DataStudioSyntheticCatalog;
    backends: DataStudioSyntheticBackend[];
    recommended_backend: DataStudioSyntheticRecommendedBackend;
    prerequisites: DataStudioSyntheticPrerequisite[];
    review_queue: DataStudioSyntheticReviewQueue;
    issues: DataStudioIssue[];
    entry_point: DataStudioSyntheticEntryPoint;
}

export async function getDataStudioSyntheticPlaybookCenter(
    projectId: number,
): Promise<DataStudioSyntheticPlaybookCenter> {
    const resp = await api.get(`/projects/${projectId}/data-studio/synthetic-playbooks`);
    return resp.data as DataStudioSyntheticPlaybookCenter;
}

export interface DataStudioSyntheticRecommendationDomain {
    id: string;
    label: string;
    confidence: number;
    source?: string | null;
}

export interface DataStudioSyntheticRecommendationSignals {
    mapping_verdict?: string;
    mapping_required_gaps: string[];
    gold_trusted_examples: number;
    gold_review_needed: number;
    gold_label_field_count: number;
    synthetic_pending: number;
    synthetic_accepted: number;
    compatible_playbook_modes: string[];
    ollama_available: boolean;
}

export interface DataStudioSyntheticRecommendationPath {
    backend: string;
    available: boolean;
    describe: string;
    local_default: boolean;
    paid_required: boolean;
}

export interface DataStudioSyntheticRecommendationItem {
    id: string;
    title: string;
    strategy: string;
    priority: 'high' | 'medium' | 'low' | string;
    target_tab: string;
    action_label: string;
    rationale: string;
    domain_reason: string;
    evidence: string[];
    confidence: number;
    playbook_mode?: string | null;
    playbook_available: boolean;
    requires_user_confirmation: boolean;
    generation_path: DataStudioSyntheticRecommendationPath;
}

export interface DataStudioSyntheticRecommendationEntryPoint {
    label: string;
    target_tab: string;
    reason: string;
}

export interface DataStudioSyntheticRecommendations {
    project_id: number;
    verdict: DataStudioSyntheticRecommendationVerdict;
    read_only: boolean;
    auto_apply: boolean;
    source_of_truth: string;
    domain: DataStudioSyntheticRecommendationDomain;
    recipe: DataStudioRecipeSummary | null;
    signals: DataStudioSyntheticRecommendationSignals;
    recommendations: DataStudioSyntheticRecommendationItem[];
    issues: DataStudioIssue[];
    entry_points: DataStudioSyntheticRecommendationEntryPoint[];
    power_details: Record<string, unknown>;
}

export async function getDataStudioSyntheticRecommendations(
    projectId: number,
): Promise<DataStudioSyntheticRecommendations> {
    const resp = await api.get(`/projects/${projectId}/data-studio/synthetic-recommendations`);
    return resp.data as DataStudioSyntheticRecommendations;
}

export interface DataStudioReviewQueueDomain {
    id: string;
    label: string;
    confidence: number;
    source?: string | null;
}

export interface DataStudioReviewQueueTotals {
    open_review_items: number;
    accepted_or_promoted: number;
    synthetic_pending: number;
    synthetic_accepted: number;
    gold_review_needed: number;
    gold_trusted_examples: number;
    annotation_jobs: number;
    annotation_review_needed: number;
    annotation_labeled: number;
    annotation_labeled_unpromoted: number;
    annotation_promoted: number;
}

export interface DataStudioReviewQueueTriageItem {
    id: string;
    title: string;
    priority: 'high' | 'medium' | 'low' | string;
    count: number;
    message: string;
    action_label: string;
    target_tab: string;
    requires_user_confirmation: boolean;
    evidence: string[];
}

export interface DataStudioReviewQueueGroup {
    key: string;
    label: string;
    kind: string;
    status: string;
    count: number;
    target_tab: string;
}

export interface DataStudioReviewQueueStatusGroup {
    status: string;
    label: string;
    count: number;
    target_tab: string;
    kind: string;
}

export interface DataStudioReviewQueueDomainGroup {
    domain_id: string;
    domain_label: string;
    confidence: number;
    open_review_items: number;
    accepted_or_promoted: number;
    source?: string | null;
}

export interface DataStudioReviewQueueAnnotationJob {
    id: number;
    name: string;
    label_type: string;
    status: string;
    target_rows?: number | null;
    total: number;
    assigned: number;
    unlabeled: number;
    labeled: number;
    labeled_unpromoted: number;
    promoted: number;
    review_needed: number;
    updated_at?: string | null;
}

export interface DataStudioReviewQueueEntryPoint {
    label: string;
    target_tab: string;
    reason: string;
}

export interface DataStudioReviewQueue {
    project_id: number;
    verdict: DataStudioReviewQueueVerdict;
    read_only: boolean;
    auto_apply: boolean;
    source_of_truth: string;
    domain: DataStudioReviewQueueDomain;
    totals: DataStudioReviewQueueTotals;
    synthetic: DataStudioSyntheticReviewQueue;
    gold_set: {
        validation?: DataStudioGoldSetValidation;
        totals?: Partial<DataStudioGoldSetTotals>;
        datasets: DataStudioGoldSetDataset[];
    };
    annotation: {
        totals: Record<string, number>;
        jobs: DataStudioReviewQueueAnnotationJob[];
    };
    triage: DataStudioReviewQueueTriageItem[];
    groupings: {
        by_source: DataStudioReviewQueueGroup[];
        by_status: DataStudioReviewQueueStatusGroup[];
        by_domain: DataStudioReviewQueueDomainGroup[];
    };
    issues: DataStudioIssue[];
    entry_points: DataStudioReviewQueueEntryPoint[];
    power_details: Record<string, unknown>;
}

export async function getDataStudioReviewQueue(
    projectId: number,
): Promise<DataStudioReviewQueue> {
    const resp = await api.get(`/projects/${projectId}/data-studio/review-queue`);
    return resp.data as DataStudioReviewQueue;
}

export interface DataStudioQualitySafetySummary {
    scanned_rows: number;
    sampled_rows: number;
    blocker_count: number;
    warning_count: number;
    info_count: number;
    pii_pci_signal_count: number;
    duplicate_signal_count: number;
    leakage_overlap_count: number;
    low_quality_signal_count: number;
    pending_review_count: number;
    domain_signal_count: number;
    domain_authored_check_count?: number;
    domain_authored_warning_count?: number;
    domain_authored_blocker_count?: number;
}

export interface DataStudioQualitySafetyDomain {
    id: string;
    label: string;
    confidence: number;
    source?: string | null;
}

export interface DataStudioQualitySafetyCheck {
    id: string;
    label: string;
    category: string;
    status: 'blocked' | 'attention' | 'ready' | string;
    severity: DataStudioIssueSeverity;
    message: string;
    count: number;
    target_tab: string;
    workflow_owner: string;
    source: string;
    domain_id: string;
    domain_label: string;
    evidence: string[];
    action_label: string;
    domain_authored?: boolean;
    read_only_preview?: boolean;
}

export interface DataStudioQualitySafetyGroup {
    key: string;
    label: string;
    blocker_count: number;
    warning_count: number;
    info_count: number;
    total: number;
    target_tab: string;
}

export interface DataStudioQualitySafetyStatusGroup {
    status: 'blocked' | 'attention' | 'ready' | string;
    label: string;
    count: number;
    target_tab: string;
}

export interface DataStudioQualitySafetyEntryPoint {
    label: string;
    target_tab: string;
    reason: string;
    requires_confirmation: boolean;
}

export interface DataStudioQualitySafetyAssist {
    available: boolean;
    default_provider: DataStudioAssistProvider;
    openai_compatible_supported: boolean;
    purpose: string;
    auto_apply: boolean;
    target_tab: string;
}

export interface DataStudioQualitySafetyDomainAuthored {
    available: boolean;
    preview_only: boolean;
    applied_profile_id?: string | null;
    applied_profile_source?: string | null;
    applied_pack_id?: string | null;
    applied_pack_source?: string | null;
    check_count: number;
    failing_count: number;
    blocker_count: number;
    warning_count: number;
    ready_count: number;
    supported_sources: string[];
}

export interface DataStudioQualitySafety {
    project_id: number;
    verdict: DataStudioQualitySafetyVerdict;
    read_only: boolean;
    auto_apply: boolean;
    source_of_truth: string;
    summary: DataStudioQualitySafetySummary;
    domain: DataStudioQualitySafetyDomain;
    domain_authored: DataStudioQualitySafetyDomainAuthored;
    checks: DataStudioQualitySafetyCheck[];
    findings_by_source: DataStudioQualitySafetyGroup[];
    findings_by_status: DataStudioQualitySafetyStatusGroup[];
    findings_by_domain: DataStudioQualitySafetyGroup[];
    findings_by_owner: DataStudioQualitySafetyGroup[];
    issues: DataStudioIssue[];
    entry_points: DataStudioQualitySafetyEntryPoint[];
    assist: DataStudioQualitySafetyAssist;
    power_details: Record<string, unknown>;
}

export async function getDataStudioQualitySafety(
    projectId: number,
): Promise<DataStudioQualitySafety> {
    const resp = await api.get(`/projects/${projectId}/data-studio/quality-safety`);
    return resp.data as DataStudioQualitySafety;
}

export interface DataStudioPrepareRecipe {
    status: 'met' | 'attention' | 'missing' | string;
    selected: DataStudioRecipeSummary | null;
    message: string;
}

export interface DataStudioPrepareMapping {
    status: 'met' | 'attention' | 'missing' | string;
    message: string;
    verdict?: string | null;
    contract_pass: boolean;
    source: DataStudioMappingSource | null;
    adapter_id?: string | null;
    task_profile?: string | null;
    mapping_success_rate: number;
    sampled_records: number;
    mapped_records: number;
    required_fields: string[];
    required_fields_below_100: string[];
}

export interface DataStudioPrepareSplitVersion {
    id: number;
    version: number;
    record_count: number;
    file_path: string;
    created_at?: string | null;
    manifest: Record<string, unknown>;
}

export interface DataStudioPrepareSplitItem {
    key: string;
    manifest_key: string;
    label: string;
    dataset_type: string;
    dataset_id?: number | null;
    exists: boolean;
    row_count: number;
    file_path: string;
    file_exists: boolean;
    manifest_count: number;
    manifest_version?: number | null;
    version_count: number;
    latest_version?: DataStudioPrepareSplitVersion | null;
}

export interface DataStudioPrepareSplits {
    status: 'ready' | 'partial' | 'missing' | string;
    total_prepared_rows: number;
    required_splits: string[];
    items: DataStudioPrepareSplitItem[];
}

export interface DataStudioPrepareManifest {
    status: 'ready' | 'attention' | 'missing' | string;
    exists: boolean;
    readable: boolean;
    path: string;
    error?: string | null;
    created_at?: string | null;
    total_entries: number;
    splits: Record<string, number>;
    ratios: Record<string, number>;
    included_types: string[];
    adapter_id?: string | null;
    task_profile?: string | null;
    dataset_versions: Record<string, number>;
    missing_dataset_version_splits: string[];
    missing_manifest_version_splits: string[];
}

export interface DataStudioPrepareInclusion {
    trainable_rows: number;
    raw_rows: number;
    cleaned_rows: number;
    gold_rows: number;
    synthetic_total: number;
    synthetic_pending: number;
    synthetic_accepted: number;
    synthetic_pending_excluded: boolean;
    gold_trusted_examples: number;
    gold_review_needed: number;
    included_source_types: string[];
}

export interface DataStudioPrepareReviewBlocker {
    id: string;
    label: string;
    count: number;
    severity: DataStudioIssueSeverity;
    message: string;
    target_tab: string;
}

export interface DataStudioPrepareCheck {
    id: string;
    label: string;
    status: 'met' | 'attention' | 'missing' | 'partial' | string;
    message: string;
    target_tab: string;
}

export interface DataStudioPrepareEntryPoint {
    label: string;
    target_tab: string;
    reason: string;
    requires_confirmation: boolean;
}

export interface DataStudioPrepareDataset {
    project_id: number;
    verdict: DataStudioPrepareDatasetVerdict;
    can_prepare: boolean;
    read_only: boolean;
    auto_apply: boolean;
    source_of_truth: string;
    recipe: DataStudioPrepareRecipe;
    mapping: DataStudioPrepareMapping;
    splits: DataStudioPrepareSplits;
    manifest: DataStudioPrepareManifest;
    inclusion: DataStudioPrepareInclusion;
    review_blockers: DataStudioPrepareReviewBlocker[];
    checks: DataStudioPrepareCheck[];
    issues: DataStudioIssue[];
    entry_point: DataStudioPrepareEntryPoint;
    power_details: Record<string, unknown>;
}

export async function getDataStudioPrepareDataset(
    projectId: number,
): Promise<DataStudioPrepareDataset> {
    const resp = await api.get(`/projects/${projectId}/data-studio/prepare-dataset`);
    return resp.data as DataStudioPrepareDataset;
}

export interface DataStudioDatasetVersionSummary {
    prepared_dataset_count: number;
    total_version_count: number;
    latest_total_rows: number;
    latest_created_at?: string | null;
    manifest_exists: boolean;
    manifest_readable: boolean;
    manifest_version_ref_count: number;
    training_reuse_ready: boolean;
    eval_reuse_ready: boolean;
}

export interface DataStudioDatasetVersionRecord {
    id: number;
    version: number;
    record_count: number;
    file_path: string;
    file_exists: boolean;
    created_at?: string | null;
    manifest_split?: string | null;
    manifest_count?: number | null;
    manifest: Record<string, unknown>;
}

export interface DataStudioDatasetVersionArtifact {
    key: string;
    manifest_key: string;
    label: string;
    dataset_type: string;
    dataset_id?: number | null;
    dataset_name?: string | null;
    row_count: number;
    file_path: string;
    file_exists: boolean;
    version_count: number;
    latest_version?: DataStudioDatasetVersionRecord | null;
    latest_version_number?: number | null;
    manifest_count: number;
    manifest_version?: number | null;
    manifest_file_path: string;
    manifest_file_hash: string;
    version_matches_manifest: boolean;
    row_count_matches_manifest: boolean;
}

export interface DataStudioDatasetVersionHistoryItem {
    dataset_id: number;
    dataset_name: string;
    dataset_type: string;
    row_count: number;
    file_path: string;
    file_exists: boolean;
    is_locked: boolean;
    created_at?: string | null;
    updated_at?: string | null;
    version_count: number;
    latest_version?: DataStudioDatasetVersionRecord | null;
    versions: DataStudioDatasetVersionRecord[];
}

export interface DataStudioDatasetVersionManifest {
    exists: boolean;
    readable: boolean;
    path: string;
    error?: string | null;
    created_at?: string | null;
    seed?: number | null;
    total_entries: number;
    splits: Record<string, number>;
    ratios: Record<string, number>;
    file_hashes: Record<string, string>;
    dataset_versions: Record<string, number>;
    included_types: string[];
    chat_template?: string | null;
    adapter_id?: string | null;
    task_profile?: string | null;
}

export interface DataStudioDatasetVersionSourceContext {
    recipe: DataStudioRecipeSummary | null;
    domain: DataStudioAppliedDomain;
    domain_runtime: Record<string, unknown>;
    adapter_id?: string | null;
    task_profile?: string | null;
    included_source_types: string[];
}

export interface DataStudioDatasetVersionReuseTarget {
    status: 'ready' | 'attention' | 'missing' | string;
    target_tab: string;
    message: string;
}

export interface DataStudioDatasetVersionSignal {
    id: string;
    label: string;
    status: 'met' | 'attention' | 'missing' | string;
    message: string;
    target_tab: string;
}

export interface DataStudioDatasetVersionEntryPoint {
    label: string;
    target_tab: string;
    reason: string;
    requires_confirmation: boolean;
}

export interface DataStudioDatasetVersions {
    project_id: number;
    verdict: DataStudioDatasetVersionVerdict;
    read_only: boolean;
    auto_apply: boolean;
    source_of_truth: string;
    summary: DataStudioDatasetVersionSummary;
    latest_artifacts: DataStudioDatasetVersionArtifact[];
    version_history: DataStudioDatasetVersionHistoryItem[];
    manifest: DataStudioDatasetVersionManifest;
    source_context: DataStudioDatasetVersionSourceContext;
    reuse_readiness: {
        training: DataStudioDatasetVersionReuseTarget;
        evaluation: DataStudioDatasetVersionReuseTarget;
    };
    reproducibility: DataStudioDatasetVersionSignal[];
    issues: DataStudioIssue[];
    entry_points: DataStudioDatasetVersionEntryPoint[];
    power_details: Record<string, unknown>;
}

export async function getDataStudioDatasetVersions(
    projectId: number,
): Promise<DataStudioDatasetVersions> {
    const resp = await api.get(`/projects/${projectId}/data-studio/dataset-versions`);
    return resp.data as DataStudioDatasetVersions;
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

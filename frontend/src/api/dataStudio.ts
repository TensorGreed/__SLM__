import api from './client';

export type DataStudioVerdict = 'blocked' | 'needs_work' | 'ready';
export type DataStudioIssueSeverity = 'blocker' | 'warning' | 'info';

export interface DataStudioRecipeSummary {
    id: string;
    name: string;
    task_profile: string;
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

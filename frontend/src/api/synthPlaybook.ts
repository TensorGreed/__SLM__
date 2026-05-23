/**
 * Typed API wrapper for the synth playbook framework
 * (USER-SUCCESS Epic 2).
 */

import api from './client';

export type SynthMode =
    | 'positives_paraphrase'
    | 'hard_negatives'
    | 'class_balance_fill'
    | 'edge_cases'
    | 'refusals'
    | 'format_robustness'
    | 'cluster_targeted';

export interface PlaybookCatalogEntry {
    recipe_id: string;
    mode: SynthMode;
}

export interface PlaybookCatalogResponse {
    project_id: number;
    recipe_id: string | null;
    playbooks: PlaybookCatalogEntry[];
}

export interface SynthRow {
    payload: Record<string, unknown>;
    synth_confidence: number;
    synth_source: string;
}

export interface PlaybookResult {
    rows: SynthRow[];
    backend_used: string;
    elapsed_sec: number;
    prompt_snippet: string;
}

export async function listPlaybooks(projectId: number): Promise<PlaybookCatalogResponse> {
    const resp = await api.get(`/projects/${projectId}/synthetic/playbooks`);
    return resp.data as PlaybookCatalogResponse;
}

export interface RunPlaybookArgs {
    mode: SynthMode;
    targetCount: number;
    targetClass?: string | null;
    backend?: string | null;
}

export async function runPlaybook(
    projectId: number,
    args: RunPlaybookArgs,
): Promise<PlaybookResult> {
    const resp = await api.post(`/projects/${projectId}/synthetic/run-playbook`, {
        mode: args.mode,
        target_count: args.targetCount,
        target_class: args.targetClass ?? null,
        backend: args.backend ?? null,
    });
    return resp.data as PlaybookResult;
}


// ─────────────────────────────────────────────────────────────────────
// USER-SUCCESS Epic 2b — cluster-augment + review queue.
// ─────────────────────────────────────────────────────────────────────

export interface AugmentFromClusterArgs {
    evalResultId: number;
    clusterId: string;
    targetCount?: number;
    backend?: string | null;
}

export async function augmentFromCluster(
    projectId: number,
    args: AugmentFromClusterArgs,
): Promise<PlaybookResult> {
    const params: Record<string, unknown> = {
        target_count: args.targetCount ?? 30,
    };
    if (args.backend) {
        params.backend = args.backend;
    }
    const resp = await api.post(
        `/projects/${projectId}/evaluation/${args.evalResultId}/clusters/${args.clusterId}/augment`,
        null,
        { params },
    );
    return resp.data as PlaybookResult;
}


export interface ReviewQueueEntry {
    id: number;
    synth_confidence: number;
    preview: string;
    payload: Record<string, unknown>;
}

export interface ReviewQueueGroup {
    synth_source: string;
    count: number;
    rows: ReviewQueueEntry[];
}

export interface ReviewQueueResponse {
    project_id: number;
    dataset_id: number | null;
    total_pending: number;
    groups: ReviewQueueGroup[];
}

export interface BulkUpdateResult {
    accepted: number;
    rejected: number;
    not_found: number;
    not_pending: number;
    total_remaining_pending: number;
}

export async function listSynthReviewQueue(projectId: number): Promise<ReviewQueueResponse> {
    const resp = await api.get(`/projects/${projectId}/synthetic/review-queue`);
    return resp.data as ReviewQueueResponse;
}

export async function bulkUpdateSynthReviewQueue(
    projectId: number,
    args: { rowIds: number[]; action: 'accept' | 'reject' },
): Promise<BulkUpdateResult> {
    const resp = await api.post(`/projects/${projectId}/synthetic/review-queue/bulk-update`, {
        row_ids: args.rowIds,
        action: args.action,
    });
    return resp.data as BulkUpdateResult;
}

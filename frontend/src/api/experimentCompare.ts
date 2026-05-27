/**
 * Typed client for the eval-aware experiment comparison endpoint (E3).
 *
 * Pairs with the existing rerun-from-manifest endpoint — the "Fix
 * the gap" CTA on the comparison page posts there directly with the
 * winning experiment id.
 */

import api from './client';


export type CompareDirection =
    | 'improved'
    | 'regressed'
    | 'unchanged'
    | 'new'
    | 'removed';


export type CompareWinner = 'a' | 'b' | 'tie' | 'unknown';


export interface CompareExperimentSummary {
    experiment_id: number;
    name: string;
    base_model: string;
    training_mode: string;
    status: string;
    started_at: string | null;
    completed_at: string | null;
    eval_result_id: number | null;
    eval_pass_rate: number | null;
    eval_type: string | null;
    dataset_name: string | null;
    metrics: Record<string, number | null>;
}


export interface MetricDelta {
    metric_id: string;
    a_value: number | null;
    b_value: number | null;
    delta: number | null;
    direction: CompareDirection;
    higher_is_better: boolean;
}


export interface ClusterDiffRow {
    reason_code: string;
    output_pattern: string;
    failure_count?: number;
    a_count?: number;
    b_count?: number;
    delta?: number;
}


export interface ClusterDiff {
    a_total: number;
    b_total: number;
    only_in_a: ClusterDiffRow[];
    only_in_b: ClusterDiffRow[];
    shared: ClusterDiffRow[];
}


export interface ConfigDiffRow {
    field: string;
    a_value: unknown;
    b_value: unknown;
    changed: boolean;
    primary: boolean;
}


export interface CompareResponse {
    project_id: number;
    a: CompareExperimentSummary;
    b: CompareExperimentSummary;
    metric_deltas: MetricDelta[];
    cluster_diff: ClusterDiff;
    config_diff: ConfigDiffRow[];
    winner: CompareWinner;
    regressed: boolean;
}


export async function fetchExperimentCompare(
    projectId: number,
    a: number,
    b: number,
): Promise<CompareResponse> {
    const resp = await api.get(`/projects/${projectId}/evaluation/compare`, {
        params: { a, b },
    });
    return resp.data as CompareResponse;
}


export interface RerunResponse {
    id: number;
    name: string;
    base_model: string;
    status: string;
}


/** "Fix the gap" rollback — reuses the existing rerun-from-manifest
 *  flow with the winning experiment id. The destination endpoint
 *  returns 404 with ``manifest_not_captured`` when the winner never
 *  completed a real training run; callers should surface that. */
export async function rerunExperimentFromManifest(
    projectId: number,
    experimentId: number,
    options: { runName?: string; description?: string } = {},
): Promise<RerunResponse> {
    const resp = await api.post(
        `/projects/${projectId}/training/runs/${experimentId}/rerun-from-manifest`,
        {
            run_name: options.runName ?? null,
            description: options.description ?? null,
        },
    );
    return resp.data as RerunResponse;
}

/**
 * Quality-Lift phase 8 slice 1 — seed-group drill-down client.
 *
 * The EvalPanel renders an ``AggregateRunBadge`` in the result header
 * when the row carries ``is_aggregate=true``; the badge surfaces the
 * mean ± std for the primary metric. Clicking the "View per-seed
 * runs" expander triggers a fetch through this client so the table
 * can show every child experiment's scalar value (picked-data-
 * provenance rule — the badge's mean is verifiable against the rows
 * that produced it).
 */

import api from './client';


export interface SeedGroupChildEvalResult {
    eval_result_id: number;
    experiment_id: number;
    seed_value: number | null;
    experiment_status: string;
    metrics: Record<string, unknown>;
    pass_rate: number | null;
}


export interface SeedGroupDrillDownResponse {
    seed_group_id: string;
    dataset_name: string;
    eval_type: string;
    aggregate_eval_result_id: number | null;
    leader_experiment_id: number | null;
    children: SeedGroupChildEvalResult[];
}


export async function fetchSeedGroupDrillDown(
    projectId: number,
    seedGroupId: string,
    options: { datasetName?: string; evalType?: string } = {},
): Promise<SeedGroupDrillDownResponse> {
    const params: string[] = [];
    if (options.datasetName) params.push(`dataset_name=${encodeURIComponent(options.datasetName)}`);
    if (options.evalType) params.push(`eval_type=${encodeURIComponent(options.evalType)}`);
    const qs = params.length ? `?${params.join('&')}` : '';
    const url = `/projects/${projectId}/evaluation/seed-group/${encodeURIComponent(seedGroupId)}${qs}`;
    const res = await api.get<SeedGroupDrillDownResponse>(url);
    return res.data;
}

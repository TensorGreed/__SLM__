/**
 * Typed wrapper for the Theme 8 Epic 4 "Did SFT help?" lift summary.
 *
 * Endpoint: GET /api/projects/{id}/evaluation/sft-lift-summary
 */

import api from './client';

export interface SftLiftExperimentRef {
    experiment_id: number;
    experiment_name: string;
    base_model: string;
    training_mode: string | null;
    completed_at: string | null;
    eval_result_id: number;
    dataset_name: string;
    eval_type: string;
    metrics: Record<string, number>;
    pass_rate: number | null;
}

export interface SftLiftMetricRow {
    metric_id: string;
    baseline_value: number;
    trained_value: number;
    absolute_delta: number;
    /** null when baseline was zero — "infinite" relative lift is
     * not meaningful; UI renders "new" instead. */
    relative_delta_pct: number | null;
    direction: 'improved' | 'regressed' | 'unchanged';
    is_headline: boolean;
}

export type SftLiftGateStatus =
    | 'cleared'
    | 'still_failing'
    | 'regressed'
    | 'always_passed'
    | 'incomplete';

export interface SftLiftGateRow {
    gate_id: string;
    metric_id: string;
    threshold: number;
    operator: 'gte' | 'lte' | string;
    required: boolean;
    baseline_value: number | null;
    trained_value: number | null;
    baseline_passes: boolean | null;
    trained_passes: boolean | null;
    delta_to_threshold: number | null;
    status: SftLiftGateStatus;
}

export type SftLiftStatus =
    | 'ok'
    | 'no_baseline'
    | 'no_trained'
    | 'no_overlap';

export interface SftLiftSummary {
    status: SftLiftStatus;
    project_id: number;
    message?: string | null;
    baseline: SftLiftExperimentRef | null;
    trained: SftLiftExperimentRef | null;
    metric_lifts: SftLiftMetricRow[];
    gate_status: SftLiftGateRow[];
    eval_pack_id?: string;
    task_profile_used?: string | null;
}

export async function fetchSftLiftSummary(
    projectId: number,
): Promise<SftLiftSummary> {
    const res = await api.get<SftLiftSummary>(
        `/projects/${projectId}/evaluation/sft-lift-summary`,
    );
    return res.data;
}

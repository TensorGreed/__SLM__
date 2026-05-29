/**
 * Typed wrapper for the Track 1 Epic D SLM-vs-frontier benchmark report.
 *
 * Endpoint: GET /api/projects/{id}/evaluation/frontier-comparison/{experimentId}
 *           [?frontier_model_id=gpt-4o-mini][?frontier_run_id=N]
 *
 * Quality is a pure ratio over stored EvalResults (soft-fallback when no
 * frontier baseline eval); cost + latency carry explicit provenance (published
 * frontier reference vs the project's benchmark-sweep estimate).
 */

import api from './client';

export type FrontierQualityStatus = 'ok' | 'no_slm_eval' | 'no_frontier_eval' | 'no_overlap';
export type Provenance = 'measured' | 'estimated' | 'unavailable';

export interface FrontierMetricRow {
    metric_id: string;
    slm_value: number;
    frontier_value: number;
    /** slm / frontier; null when frontier scored 0 (UI renders "exceeds"). */
    quality_pct: number | null;
    direction: 'matches_or_better' | 'behind' | 'exceeds';
    is_headline: boolean;
}

export interface FrontierExperimentRef {
    experiment_id: number;
    experiment_name: string;
    base_model: string;
    eval_result_id?: number;
    dataset_name?: string;
    eval_type?: string;
    metrics?: Record<string, number>;
    pass_rate?: number | null;
}

export interface FrontierComparison {
    project_id: number;
    frontier_model: { id: string; display_name: string; source: string; as_of: string };
    slm: FrontierExperimentRef;
    frontier: FrontierExperimentRef | null;
    quality: {
        status: FrontierQualityStatus;
        metric_comparisons: FrontierMetricRow[];
        headline_quality_pct: number | null;
        frontier_baseline_run_id: number | null;
        message?: string | null;
    };
    cost: {
        frontier_usd_per_1m_tokens: number;
        frontier_source: string;
        slm_usd_per_1m_tokens: number | null;
        cost_pct: number | null;
        provenance: Provenance;
        source?: string;
        gpu_hourly_usd?: number;
        message?: string | null;
    };
    latency: {
        frontier_latency_ms: number;
        frontier_source: string;
        slm_latency_ms: number | null;
        latency_ratio: number | null;
        provenance: Provenance;
        source?: string;
        message?: string | null;
    };
    headline: string;
}

export async function fetchFrontierComparison(
    projectId: number,
    experimentId: number,
    opts?: { frontierModelId?: string; frontierRunId?: number | null },
): Promise<FrontierComparison> {
    const params: Record<string, string | number> = {};
    if (opts?.frontierModelId) params.frontier_model_id = opts.frontierModelId;
    if (opts?.frontierRunId != null) params.frontier_run_id = opts.frontierRunId;
    const res = await api.get<FrontierComparison>(
        `/projects/${projectId}/evaluation/frontier-comparison/${experimentId}`,
        Object.keys(params).length ? { params } : undefined,
    );
    return res.data;
}

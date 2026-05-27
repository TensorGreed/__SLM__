/**
 * Typed API wrapper for the trainability forecast endpoint
 * (USER-SUCCESS Epic 1).
 */

import api from './client';

export type ForecastVerdict = 'likely_pass' | 'borderline' | 'likely_fail';
export type ForecastSeverity = 'ok' | 'warn' | 'block';

export type SuggestedActionKind =
    | 'synth_augment'
    | 'synth_balance'
    | 'synth_diversify'
    | 'fix_gold_rows';

export interface SuggestedAction {
    kind: SuggestedActionKind;
    params: Record<string, unknown>;
}

export interface ForecastSignal {
    id: string;
    severity: ForecastSeverity;
    headline: string;
    detail: string;
    suggested_action: SuggestedAction | null;
}

export interface ForecastResult {
    overall: ForecastVerdict;
    confidence_pct: number;
    signals: ForecastSignal[];
    computed_at: string;
    cache_key: string;
    cache_hit: boolean;
}

export async function fetchTrainingForecast(
    projectId: number,
    options: { refresh?: boolean } = {},
): Promise<ForecastResult> {
    const params = options.refresh ? { refresh: true } : undefined;
    const resp = await api.get(`/projects/${projectId}/training/forecast`, { params });
    return resp.data as ForecastResult;
}

/** One row from the per-project forecast history (T2). Shape mirrors
 *  ``ForecastResult`` minus ``cache_hit`` (always false for persisted
 *  rows) so the panel can render historical entries with the same
 *  code path it uses for live results. */
export interface ForecastSnapshot {
    id: number;
    cache_key: string;
    computed_at: string;
    overall: ForecastVerdict;
    confidence_pct: number;
    signals: ForecastSignal[];
}

export interface ForecastHistoryResponse {
    project_id: number;
    snapshots: ForecastSnapshot[];
}

export async function fetchTrainingForecastHistory(
    projectId: number,
    options: { limit?: number } = {},
): Promise<ForecastHistoryResponse> {
    const params = options.limit !== undefined ? { limit: options.limit } : undefined;
    const resp = await api.get(
        `/projects/${projectId}/training/forecast/history`,
        { params },
    );
    return resp.data as ForecastHistoryResponse;
}

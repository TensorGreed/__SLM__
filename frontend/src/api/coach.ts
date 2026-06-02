/**
 * Coach suggestion types + async fetcher for stage-specific coaching recommendations.
 */

import api from './client';

export type CoachStage = 'data' | 'cleaning' | 'gold_set' | 'training' | 'eval';
export type CoachSeverity = 'info' | 'warning' | 'critical';

export type CoachActionKind =
    | 'run_playbook'
    | 'navigate'
    | 'augment_from_cluster';

export interface CoachAction {
    kind: CoachActionKind;
    label: string;
    // params is intentionally a free-form bag because each action kind
    // carries a different shape (run_playbook → {mode, target_count,
    // target_class}; navigate → {target}). The consumer branches on
    // ``kind`` before reading ``params``.
    params: Record<string, unknown>;
}

export interface CoachSuggestion {
    id: string;
    title: string;
    body: string;
    severity: CoachSeverity;
    action: CoachAction;
    // Decision-trace fields (Arc 4) — surface WHY a suggestion is
    // firing, not just what it suggests. ``context`` carries the
    // signal values the coach rule observed (row counts, ratios,
    // thresholds, etc.). ``rule_id`` (when set) names the specific
    // decision rule that matched, so the UI can label the trace
    // ("Rule: low-gold-row-count fired because gold_rows=12, threshold=50").
    // Both optional so non-enriched suggestion builders still work.
    context?: Record<string, unknown>;
    rule_id?: string;
}

export interface CoachStageResponse {
    project_id: number;
    stage: CoachStage;
    suggestions: CoachSuggestion[];
    handler_available: boolean;
}

export async function fetchCoachSuggestions(
    projectId: number,
    stage: CoachStage,
): Promise<CoachStageResponse> {
    const res = await api.get<CoachStageResponse>(
        `/projects/${projectId}/coach/${stage}`,
    );
    return res.data;
}

import api from './client';

export type CoachStage = 'data' | 'cleaning' | 'gold_set' | 'training' | 'eval';
export type CoachSeverity = 'info' | 'warning' | 'critical';

export type CoachActionKind = 'run_playbook' | 'navigate';

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
    // Free-form context the backend attaches so the UI can show
    // numbers / thresholds inline if it wants to (e.g. the row count
    // that triggered the suggestion). Currently unused by the strip
    // but reserved so the contract is stable across phases.
    context?: Record<string, unknown>;
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

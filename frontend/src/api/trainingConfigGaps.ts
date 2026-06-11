/**
 * Training Config Gaps API client — Coach-stage-2 phase 1.
 *
 * One read-only endpoint. The payload mirrors data-health's shape so
 * the panel rendering can share the same severity vocab.
 */

import api from './client';

export type GapSeverity = 'ok' | 'warn' | 'block';

export interface GapSuggestedAction {
    kind?: string;
    label?: string;
    target?: string;
    params?: Record<string, unknown>;
}

export interface GapSignal {
    id: string;
    severity: GapSeverity;
    headline: string;
    plain_english: string;
    why_it_matters: string;
    suggested_action: GapSuggestedAction | null;
    context: Record<string, unknown>;
}

export interface GapGroup {
    id: string;
    title: string;
    subtitle: string;
    signals: GapSignal[];
}

export interface TrainingConfigGapReport {
    project_id: number;
    computed_at: string;
    overall: GapSeverity;
    severity_summary: { ok: number; warn: number; block: number };
    total_signals: number;
    groups: GapGroup[];
}

export async function fetchTrainingConfigGaps(
    projectId: number,
): Promise<TrainingConfigGapReport> {
    const res = await api.get<TrainingConfigGapReport>(
        `/projects/${projectId}/training-config-gaps`,
    );
    return res.data;
}

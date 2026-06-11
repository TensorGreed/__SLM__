/**
 * Eval Gaps API client — Coach-stage-2 phase 3.
 *
 * One read-only endpoint. Shares the gap-payload shape with the
 * training-config gap panel so the panel rendering can reuse the same
 * severity vocab.
 */

import api from './client';

import type { GapSeverity, GapSuggestedAction } from './trainingConfigGaps';

export type EvalGapSeverity = GapSeverity;

export interface EvalGapSignal {
    id: string;
    severity: EvalGapSeverity;
    headline: string;
    plain_english: string;
    why_it_matters: string;
    suggested_action: GapSuggestedAction | null;
    context: Record<string, unknown>;
}

export interface EvalGapGroup {
    id: string;
    title: string;
    subtitle: string;
    signals: EvalGapSignal[];
}

export interface EvalGapReport {
    project_id: number;
    computed_at: string;
    overall: EvalGapSeverity;
    severity_summary: { ok: number; warn: number; block: number };
    total_signals: number;
    groups: EvalGapGroup[];
}

export async function fetchEvalGaps(
    projectId: number,
): Promise<EvalGapReport> {
    const res = await api.get<EvalGapReport>(
        `/projects/${projectId}/eval-gaps`,
    );
    return res.data;
}

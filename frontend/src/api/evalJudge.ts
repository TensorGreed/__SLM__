import api from './client';
import type { EvalJudgeTrustSummary } from './evalJudgeCalibration';

export type JudgeBilling = 'local' | 'metered' | 'subscription';

export interface EvalJudgeOption {
    /** The pin to store ("" = auto). */
    value: string;
    label: string;
    billing: JudgeBilling | string;
    available: boolean;
}

/** GET/PUT /projects/{id}/eval-judge. */
export interface EvalJudgeState {
    enabled: boolean;
    backend: string | null;
    /** Set when the EVAL_JUDGE_BACKEND env var overrides this setting. */
    env_override: string | null;
    resolved: { describe: string; local: boolean; billing: JudgeBilling } | null;
    options: EvalJudgeOption[];
    /** Judge calibration V4: how far the judge in use can be trusted (absent on older backends). */
    trust?: EvalJudgeTrustSummary | null;
}

export async function fetchEvalJudge(projectId: number): Promise<EvalJudgeState> {
    const res = await api.get<EvalJudgeState>(`/projects/${projectId}/eval-judge`);
    return res.data;
}

export async function saveEvalJudge(
    projectId: number,
    body: { enabled: boolean; backend: string | null },
): Promise<EvalJudgeState> {
    const res = await api.put<EvalJudgeState>(`/projects/${projectId}/eval-judge`, body);
    return res.data;
}

export const BILLING_LABELS: Record<string, string> = {
    local: 'local - free',
    metered: 'API credits - counts against the budget',
    subscription: 'subscription - not billed per call',
};

/**
 * The Eval tab's default card (Wave 3b).
 * GET /api/projects/{id}/evaluation/summary?experiment_id=
 */

import api from './client';

export type EvalVerdict =
    | 'better'
    | 'worse'
    | 'same'
    | 'no_baseline'
    | 'no_comparison'
    | 'not_evaluated'
    | 'no_trained_run';

export interface EvalSummaryHeadline {
    metric_id: string;
    baseline_value: number;
    trained_value: number;
    absolute_delta: number;
    relative_delta_pct: number | null;
    direction: 'improved' | 'regressed' | 'unchanged';
}

export interface EvalSummaryFailure {
    prompt: string;
    reference: string;
    prediction: string;
    row_exact_match?: number | null;
    row_f1?: number | null;
}

export interface EvalSummary {
    project_id: number;
    experiment_id: number | null;
    verdict: EvalVerdict;
    message?: string | null;
    headline: EvalSummaryHeadline | null;
    baseline?: { experiment_id: number; base_model: string } | null;
    trained?: { experiment_id: number; experiment_name: string } | null;
    eval_result_id?: number;
    eval_type?: string;
    evaluated_samples?: number | null;
    failures: EvalSummaryFailure[];
    failed_count?: number | null;
}

export async function fetchEvalSummary(
    projectId: number,
    experimentId?: number | null,
): Promise<EvalSummary> {
    const res = await api.get<EvalSummary>(
        `/projects/${projectId}/evaluation/summary`,
        experimentId ? { params: { experiment_id: experimentId } } : undefined,
    );
    return res.data;
}

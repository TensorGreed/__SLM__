/**
 * The Eval tab's default card (Wave 3b).
 * GET /api/projects/{id}/evaluation/summary?experiment_id=
 */

import api from './client';
import type { LiftEvidence, SeedEvidence } from '../components/evaluation/liftEvidence';

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
    /** Multi-seed run: the headline is the mean across seeds. */
    trained_std?: number | null;
    n_seeds?: number;
}

export interface EvalSummaryFailure {
    prompt: string;
    reference: string;
    prediction: string;
    row_exact_match?: number | null;
    row_f1?: number | null;
    /** LLM-judge verdict for long-answer tasks (answer_judge_service). */
    row_judge_verdict?: 'correct' | 'partial' | 'wrong' | null;
    row_judge_reason?: string | null;
}

/** The LLM judge's snapshot for the fine-tuned eval: who judged, how many
 *  answers were correct / partial / wrong. Null when the task isn't a
 *  long-answer one or no judge was reachable (F1 stays the headline). */
export interface EvalSummaryJudge {
    judge: string | null;
    score: number;
    judged: number;
    unjudged: number;
    correct: number;
    partial: number;
    wrong: number;
    judge_calls?: number | null;
    judge_cached?: number | null;
}

export interface EvalSummary {
    project_id: number;
    experiment_id: number | null;
    verdict: EvalVerdict;
    message?: string | null;
    headline: EvalSummaryHeadline | null;
    /** Rows better / worse / same vs the base model + noise verdict for the
     *  headline lift; null for older results or non-row metrics. */
    evidence?: LiftEvidence | null;
    /** Multi-seed run: per-seed headlines + the spread across seeds. */
    seed_evidence?: SeedEvidence | null;
    seeds?: Array<{ experiment_id: number; seed_value: number | null; headline: EvalSummaryHeadline | null }> | null;
    n_seeds?: number;
    representative_experiment_id?: number | null;
    baseline?: { experiment_id: number; base_model: string } | null;
    trained?: { experiment_id: number; experiment_name: string } | null;
    eval_result_id?: number;
    eval_type?: string;
    evaluated_samples?: number | null;
    failures: EvalSummaryFailure[];
    failed_count?: number | null;
    judge?: EvalSummaryJudge | null;
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

export interface CheckSeedsResponse {
    status: string;
    experiment_id: number;
    experiment_name: string;
    source_experiment_id: number;
    num_seeds: number;
}

/** Re-train a run's config with ``numSeeds`` seeds so the lift check can
 *  report run-to-run variance. */
export async function checkLiftAcrossSeeds(
    projectId: number,
    experimentId: number,
    numSeeds = 3,
): Promise<CheckSeedsResponse> {
    const res = await api.post<CheckSeedsResponse>(
        `/projects/${projectId}/evaluation/summary/check-seeds`,
        { experiment_id: experimentId, num_seeds: numSeeds },
    );
    return res.data;
}

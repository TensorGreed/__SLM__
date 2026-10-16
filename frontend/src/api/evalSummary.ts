/**
 * The Eval tab's default card (Wave 3b).
 * GET /api/projects/{id}/evaluation/summary?experiment_id=
 */

import api from './client';
import type { LiftEvidence, SeedEvidence } from '../components/evaluation/liftEvidence';
import type { RetrievalSplit } from '../components/evaluation/retrievalSplit';

export type EvalVerdict =
    | 'better'
    | 'worse'
    | 'same'
    | 'no_baseline'
    | 'no_comparison'
    | 'not_evaluated'
    | 'no_trained_run'
    /** span_eval: a token-classification run scored on its own (no base). */
    | 'evaluated';

/** One compliance regime's span scores (recall = the redaction headline). */
export interface SpanRegimeScore {
    regime: string;
    label: string;
    custom: boolean;
    types: string[];
    unscored_types: string[];
    tp: number;
    fp: number;
    fn: number;
    support: number;
    precision: number | null;
    recall: number | null;
    f1: number | null;
}

/** One span_set evaluation of a token-classification run. */
export interface SpanEvalBlock {
    eval_result_id: number;
    dataset_name: string;
    precision: number | null;
    recall: number | null;
    f1: number | null;
    rows: number | null;
    weakest_types: Array<{ type: string; f1: number; support: number }>;
    /** Micro P/R/F1 per compliance regime (project's type→regimes map),
     *  weakest recall first; [] when the project tags no types. */
    by_regime?: SpanRegimeScore[];
    /** answer_key only: who wrote it (e.g. cloud:anthropic:claude-haiku-4-5…). */
    written_by?: string | null;
    /** answer_key only: the human review of the key (null until a row was
     *  checked) — counts, and this eval re-totalled over checked vs
     *  unchecked rows. ``stale``: a correction / rejection is newer than
     *  the eval, so re-evaluate to score the corrected key. */
    review?: AnswerKeyReview | null;
}

export interface ReviewSplitScores {
    precision: number | null;
    recall: number | null;
    f1: number | null;
    rows: number;
}

export interface AnswerKeyReview {
    total: number;
    reviewed: number;
    accepted: number;
    corrected: number;
    rejected: number;
    unreviewed: number;
    reviewed_scores: ReviewSplitScores | null;
    unreviewed_scores: ReviewSplitScores | null;
    stale: boolean;
    changed_since_eval: number;
}

/** "12 of 43 rows checked by a person · F1 0.91 on the checked rows, 0.96
 *  on the rest" (+ a re-evaluate note when the eval predates a change). */
export function answerKeyReviewText(review: AnswerKeyReview): string {
    const f = (v: number | null | undefined) => (typeof v === 'number' ? v.toFixed(2) : '—');
    const parts: string[] = [];
    if (review.corrected) parts.push(`${review.corrected} corrected`);
    if (review.rejected) parts.push(`${review.rejected} rejected`);
    let text = `${review.reviewed} of ${review.total} rows checked by a person${parts.length ? ` (${parts.join(', ')})` : ''}`;
    const checked = review.reviewed_scores;
    const rest = review.unreviewed_scores;
    if (checked && checked.rows > 0 && checked.f1 != null) {
        text += ` · F1 ${f(checked.f1)} on the ${checked.rows} checked row${checked.rows === 1 ? '' : 's'}`;
        if (rest && rest.rows > 0 && rest.f1 != null) text += `, ${f(rest.f1)} on the ${rest.rows} unchecked`;
    }
    if (review.stale) {
        text += ` · ${review.changed_since_eval} row${review.changed_since_eval === 1 ? '' : 's'} changed since this eval — re-evaluate to score the corrected key`;
    }
    return text + '.';
}

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
    /** rag_passages: the retrieval judge — did the passages hold the answer? */
    row_retrieval_verdict?: 'found' | 'missed' | null;
    row_retrieval_passage?: number | null;
    row_retrieval_reason?: string | null;
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
    /** rag_passages: retrieval vs reading (retrievalSplit.ts). */
    retrieval?: RetrievalSplit | null;
}

export interface EvalSummary {
    project_id: number;
    experiment_id: number | null;
    /** External benchmarks with this run's latest score on each (G-B1). */
    benchmarks?: import('./benchmarks').BenchmarkInfo[];
    /** 'rag_passages': a RAG-first project — the "run" is the base model +
     *  document-passage retrieval, scored by the judge against the base
     *  model alone (no training). Absent for trained runs. */
    kind?: 'rag_passages' | 'span_eval' | 'head_eval';
    /** kind 'head_eval': a classifier head on an encoder, scored on its own. */
    head?: { accuracy: number | null; macro_f1: number | null; rows: number | null; per_class?: Record<string, { f1?: number }> };
    /** span_eval: scores on the answer key written by a second model (the
     *  headline when present) and on the test examples. */
    answer_key?: SpanEvalBlock | null;
    test?: SpanEvalBlock | null;
    dataset_name?: string | null;
    split?: string | null;
    /** rag_passages: the retrieval scored and served (top-k + reranker,
     *  and after the staged sweep possibly a candidate pool / passage size). */
    retrieval?: { k: number; reranker: string | null; pool?: number | null; chunk_size?: number | null } | null;
    /** rag_passages after a retrieval sweep: every config tried with the
     *  judge's score; `chosen` is the one served. */
    retrieval_sweep?: Array<{
        label: string;
        stage?: string | null;
        judge_score: number | null;
        on_mean_f1: number;
        chosen: boolean;
        current?: boolean;
        /** Paired row evidence vs the incumbent retrieval (null for the incumbent). */
        vs_current?: { verdict: string; n: number; better: number; worse: number; same: number } | null;
    }> | null;
    /** Why the sweep kept or switched the served retrieval. */
    retrieval_decision?: { current: { k: number; reranker: string | null }; chosen: { k: number; reranker: string | null }; switched: boolean; reason: string } | null;
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
    /** True when the summarised run is a multi-seed leader ("Ship this seed"). */
    seed_group_leader?: boolean;
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

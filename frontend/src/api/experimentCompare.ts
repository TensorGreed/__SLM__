/**
 * Typed client for the eval-aware experiment comparison endpoint (E3).
 *
 * Pairs with the existing rerun-from-manifest endpoint — the "Fix
 * the gap" CTA on the comparison page posts there directly with the
 * winning experiment id.
 */

import api from './client';
import type { EvalSizing } from '../components/evaluation/liftEvidence';


export type CompareDirection =
    | 'improved'
    | 'regressed'
    | 'unchanged'
    | 'new'
    | 'removed';


export type CompareWinner = 'a' | 'b' | 'tie' | 'unknown';


export interface CompareExperimentSummary {
    experiment_id: number;
    name: string;
    base_model: string;
    training_mode: string;
    status: string;
    started_at: string | null;
    completed_at: string | null;
    eval_result_id: number | null;
    eval_pass_rate: number | null;
    eval_type: string | null;
    dataset_name: string | null;
    metrics: Record<string, number | null>;
    /** What this side's eval was scored against (Epic W1). Absent on
     *  older servers; null when it could not be read. */
    provenance?: CompareProvenance | null;
}


/** Which answer key an eval scored against: the split version (recorded at
 *  eval time, or inferred from the version that was newest then) and the
 *  test file's hash. */
export interface CompareAnswerKey {
    dataset_id: number | null;
    dataset_type: string | null;
    version: number | null;
    version_source: 'recorded' | 'inferred' | null;
    /** Only a hash the eval recorded; an inferred one is in ``inferred_sha256``. */
    sha256: string | null;
    inferred_sha256?: string | null;
}


export interface CompareJudge {
    answer: string | null;
    rubric: string | null;
    rubric_sha: string | null;
    code_example: string | null;
}


export interface CompareProvenance {
    experiment_id: number;
    experiment_name: string;
    eval_result_id: number | null;
    evaluated_at: string | null;
    answer_key: CompareAnswerKey | null;
    judge: CompareJudge | null;
    scoring: {
        eval_type: string | null;
        decoding?: Record<string, unknown> | null;
        scoring_mode: string | null;
        fingerprints: Record<string, string>;
    } | null;
    has_row_scores: boolean;
}


export type RowEvidenceVerdict =
    | 'better'
    | 'worse'
    | 'within_noise'
    | 'too_few_rows'
    | 'not_comparable';


/** Per-row B-vs-A evidence for one metric (backend
 *  ``paired_difference_evidence`` + pairing counts). */
export interface RowMetricEvidence {
    metric_id: string;
    score_key: string;
    n: number;
    better: number;
    worse: number;
    same: number;
    mean_diff: number | null;
    ci_low: number | null;
    ci_high: number | null;
    verdict: Exclude<RowEvidenceVerdict, 'not_comparable'>;
    only_in_a: number;
    only_in_b: number;
    unscored: number;
    /** Epic W2: detectable difference on these paired rows (absent on older servers). */
    sizing?: EvalSizing | null;
}


export interface RescoreOffer {
    side: 'a' | 'b';
    experiment_id: number;
    experiment_name: string;
    why: 'not_comparable' | 'no_row_scores' | 'missing_eval';
    available: boolean;
    /** A lift check for this run is already queued / running. */
    in_progress?: boolean;
    job_id?: number | null;
    unavailable_reason: string | null;
    instruction: string;
}


/** Is B's change more than noise? Rows paired by key; headline metric
 *  fields at the top level, every per-row metric under ``metrics``. */
export interface CompareRowEvidence {
    verdict: RowEvidenceVerdict;
    reason: 'not_comparable' | 'no_row_scores' | 'no_row_metric' | 'missing_eval' | null;
    message: string | null;
    metric_id: string | null;
    score_key: string | null;
    n: number;
    better: number;
    worse: number;
    same: number;
    mean_diff: number | null;
    ci_low: number | null;
    ci_high: number | null;
    only_in_a: number;
    only_in_b: number;
    unscored: number;
    metrics: RowMetricEvidence[];
    comparability: {
        comparable: boolean;
        /** False when every difference rests on an inferred answer key ("may differ"). */
        certain?: boolean;
        differences: { kind: 'answer_key' | 'judge' | 'scoring'; message: string; a: unknown; b: unknown; inferred?: boolean }[];
        notes: string[];
        /** Non-blocking, e.g. different decoding settings. */
        caveats?: string[];
    } | null;
    rescore: RescoreOffer | null;
    /** Epic W2: the headline metric's detectable difference (absent on older servers). */
    sizing?: EvalSizing | null;
}


export interface MetricDelta {
    metric_id: string;
    a_value: number | null;
    b_value: number | null;
    delta: number | null;
    direction: CompareDirection;
    higher_is_better: boolean;
}


export interface ClusterDiffRow {
    reason_code: string;
    output_pattern: string;
    failure_count?: number;
    a_count?: number;
    b_count?: number;
    delta?: number;
}


export interface ClusterDiff {
    a_total: number;
    b_total: number;
    only_in_a: ClusterDiffRow[];
    only_in_b: ClusterDiffRow[];
    shared: ClusterDiffRow[];
}


export interface ConfigDiffRow {
    field: string;
    a_value: unknown;
    b_value: unknown;
    changed: boolean;
    primary: boolean;
}


export interface CompareResponse {
    project_id: number;
    a: CompareExperimentSummary;
    b: CompareExperimentSummary;
    metric_deltas: MetricDelta[];
    cluster_diff: ClusterDiff;
    config_diff: ConfigDiffRow[];
    winner: CompareWinner;
    regressed: boolean;
    /** Epic W1 — absent on older servers; null when it could not be built. */
    row_evidence?: CompareRowEvidence | null;
}


export async function fetchExperimentCompare(
    projectId: number,
    a: number,
    b: number,
): Promise<CompareResponse> {
    const resp = await api.get(`/projects/${projectId}/evaluation/compare`, {
        params: { a, b },
    });
    return resp.data as CompareResponse;
}


export interface RerunResponse {
    id: number;
    name: string;
    base_model: string;
    status: string;
}


/** "Fix the gap" rollback — reuses the existing rerun-from-manifest
 *  flow with the winning experiment id. The destination endpoint
 *  returns 404 with ``manifest_not_captured`` when the winner never
 *  completed a real training run; callers should surface that. */
export async function rerunExperimentFromManifest(
    projectId: number,
    experimentId: number,
    options: { runName?: string; description?: string } = {},
): Promise<RerunResponse> {
    const resp = await api.post(
        `/projects/${projectId}/training/runs/${experimentId}/rerun-from-manifest`,
        {
            run_name: options.runName ?? null,
            description: options.description ?? null,
        },
    );
    return resp.data as RerunResponse;
}


/** Re-score a run on the current answer key and judge: the lift check
 *  (the same Job as "Evaluate this run"), tracked by the bell. */
export async function rescoreExperiment(projectId: number, experimentId: number): Promise<void> {
    await api.post(`/projects/${projectId}/evaluation/summary/lift-check`, {
        experiment_id: experimentId,
    });
}

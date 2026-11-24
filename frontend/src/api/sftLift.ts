/**
 * Typed wrapper for the Theme 8 Epic 4 "Did SFT help?" lift summary.
 *
 * Endpoint: GET /api/projects/{id}/evaluation/sft-lift-summary
 */

import api from './client';
import type { JudgedSubset, LiftEvidence } from '../components/evaluation/liftEvidence';
import type { CheckedDecodingSummary } from './checkedDecoding';
import type { BaselineDecodingBar } from './evalSummary';

export interface SftLiftExperimentRef {
    experiment_id: number;
    experiment_name: string;
    base_model: string;
    training_mode: string | null;
    completed_at: string | null;
    eval_result_id: number;
    dataset_name: string;
    eval_type: string;
    metrics: Record<string, number>;
    pass_rate: number | null;
    /** Opt-in checked decoding: present only when this eval ran with it. */
    checked_decoding?: CheckedDecodingSummary | null;
    /** The base side only: which decoding set the bar when the lift check scored the base model both ways
     *  (plain and the model's own ``generation_config``); null otherwise (absent on older servers). */
    decoding_bar?: BaselineDecodingBar | null;
}

export interface SftLiftMetricRow {
    metric_id: string;
    baseline_value: number;
    trained_value: number;
    absolute_delta: number;
    /** null when baseline was zero — "infinite" relative lift is
     * not meaningful; UI renders "new" instead. */
    relative_delta_pct: number | null;
    direction: 'improved' | 'regressed' | 'unchanged';
    is_headline: boolean;
    /** Rows better / worse / same vs the base model + noise verdict; null
     * for metrics that aren't a per-row mean or for older results. */
    evidence?: LiftEvidence | null;
    /** Judge-based metric resting on the judged rows only (per side); absent otherwise. */
    judged_subset?: JudgedSubset | null;
}

export type SftLiftGateStatus =
    | 'cleared'
    | 'still_failing'
    | 'regressed'
    | 'always_passed'
    | 'incomplete';

export interface SftLiftGateRow {
    gate_id: string;
    metric_id: string;
    threshold: number;
    operator: 'gte' | 'lte' | string;
    required: boolean;
    baseline_value: number | null;
    trained_value: number | null;
    baseline_passes: boolean | null;
    trained_passes: boolean | null;
    delta_to_threshold: number | null;
    status: SftLiftGateStatus;
}

export type SftLiftStatus =
    | 'ok'
    | 'no_baseline'
    | 'no_trained'
    | 'no_overlap'
    | 'not_comparable';

export interface SftLiftSummary {
    status: SftLiftStatus;
    project_id: number;
    /** Set when the lift was asked for on one eval set (``eval_set:<slug>@v<N>[+<variant>]``). */
    dataset_name?: string | null;
    /** The base leg's label when it was asked for on another variant of the same version. */
    base_dataset_name?: string | null;
    /** Base and fine-tuned on different prompt-format variants of one eval-set version (same rows):
     * paired on the shared rows; ``label`` says both the model and the prompt differ. */
    cross_variant?: { slug: string; version: number; variant_a: string | null; variant_b: string | null; label: string } | null;
    message?: string | null;
    baseline: SftLiftExperimentRef | null;
    trained: SftLiftExperimentRef | null;
    metric_lifts: SftLiftMetricRow[];
    gate_status: SftLiftGateRow[];
    eval_pack_id?: string;
    task_profile_used?: string | null;
    /** The two evals ran under different checked-decoding policies (not like for like). */
    checked_decoding_differs?: boolean;
    /** span_set results scored by different scorer versions (null = before versions were recorded). */
    scoring_version_differs?: ScoringVersionDiffers;
}

export interface ScoringVersionDiffers {
    base: number | null;
    tuned: number | null;
    message: string;
}

/** ``experimentId`` pins the trained side to that run (and the baseline to
 * its base model); omitted → the project's latest trained run. ``datasetName``
 * pairs both evals on exactly that eval set + version + variant; omitted → the
 * built-in sets (unchanged request). ``baseDatasetName``: the base leg on another prompt-format variant of
 * the same eval-set version (a labelled cross-variant pair). */
export async function fetchSftLiftSummary(
    projectId: number,
    experimentId?: number | null,
    datasetName?: string | null,
    baseDatasetName?: string | null,
): Promise<SftLiftSummary> {
    const params: Record<string, string | number> = {};
    if (experimentId) params.experiment_id = experimentId;
    if (datasetName) params.dataset_name = datasetName;
    if (datasetName && baseDatasetName && baseDatasetName !== datasetName) params.base_dataset_name = baseDatasetName;
    const res = await api.get<SftLiftSummary>(
        `/projects/${projectId}/evaluation/sft-lift-summary`,
        Object.keys(params).length ? { params } : undefined,
    );
    return res.data;
}

/**
 * A ``fetchSftLiftSummary`` that answers repeated identical requests from one cache (in flight or done), so
 * two panels showing the same lift (the Verdict card and Verdict › Details) make one request. A failed
 * request is dropped from the cache so a later call retries. Make a new one to refresh.
 */
export function createSharedLiftFetch(): (
    projectId: number,
    experimentId?: number | null,
    datasetName?: string | null,
    baseDatasetName?: string | null,
) => Promise<SftLiftSummary> {
    const cache = new Map<string, Promise<SftLiftSummary>>();
    return (projectId, experimentId, datasetName, baseDatasetName) => {
        const key = JSON.stringify([projectId, experimentId ?? null, datasetName ?? null, baseDatasetName ?? null]);
        let pending = cache.get(key);
        if (!pending) {
            pending = baseDatasetName
                ? fetchSftLiftSummary(projectId, experimentId, datasetName, baseDatasetName)
                : datasetName
                    ? fetchSftLiftSummary(projectId, experimentId, datasetName)
                    : fetchSftLiftSummary(projectId, experimentId);
            cache.set(key, pending);
            pending.catch(() => { cache.delete(key); });
        }
        return pending;
    };
}

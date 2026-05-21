/**
 * Typed wrapper for the Theme 8 Epic 2 active-learning recommender.
 *
 * Endpoints (mounted under the existing evaluation router):
 *   GET  /api/projects/{id}/evaluation/active-learning/{exp_id}/proposal
 *   POST /api/projects/{id}/evaluation/active-learning/{exp_id}/promote
 */

import api from './client';

export interface ActiveLearningCandidate {
    row_index: number;
    failure_reason: string;
    prompt: string;
    prediction: string;
    reference: string;
    row_score: number | null;
    already_promoted: boolean;
}

export interface ActiveLearningProposal {
    eval_result_id: number | null;
    experiment_id: number;
    candidates: ActiveLearningCandidate[];
    total_failures: number;
    total_predictions: number;
    max_rows: number;
    dataset_name?: string | null;
    promoted_count: number;
    message?: string;
}

export interface ActiveLearningPromoteResult {
    status: 'ok';
    experiment_id: number;
    promoted_count: number;
    skipped_already_promoted: number;
    skipped_invalid_indexes: number;
    target_dataset_id: number | null;
    target_dataset_path: string | null;
    total_promoted_lifetime: number;
}

export async function fetchActiveLearningProposal(
    projectId: number,
    experimentId: number,
    maxRows = 20,
): Promise<ActiveLearningProposal> {
    const res = await api.get<ActiveLearningProposal>(
        `/projects/${projectId}/evaluation/active-learning/${experimentId}/proposal`,
        { params: { max_rows: maxRows } },
    );
    return res.data;
}

export async function promoteActiveLearningRows(
    projectId: number,
    experimentId: number,
    rowIndexes: number[],
): Promise<ActiveLearningPromoteResult> {
    const res = await api.post<ActiveLearningPromoteResult>(
        `/projects/${projectId}/evaluation/active-learning/${experimentId}/promote`,
        { row_indexes: rowIndexes },
    );
    return res.data;
}

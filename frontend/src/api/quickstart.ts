/**
 * Typed wrappers for the project-guide quickstart endpoints
 * (Theme 1 Epic 4).
 */

import api from './client';

export interface ImportSampleSummary {
    slug: string;
    created: boolean;
    project_id: number;
    project_name: string;
    source_dataset_id: number;
    source_row_count: number;
    gold_set_id: number;
    gold_version_id: number;
    gold_row_count: number;
    prepared_train_path: string;
    prepared_train_rows: number;
    prepared_val_rows: number;
    prepared_test_rows: number;
    prepared_dataset_ids: Record<string, number>;
    adapter_id: string;
    task_profile: string;
    suggested_brief: string;
}

export interface ImportSampleResponse {
    status: 'ok';
    summary: ImportSampleSummary;
}

export async function quickstartImportSample(
    projectId: number,
    slug?: string,
): Promise<ImportSampleResponse> {
    const res = await api.post<ImportSampleResponse>(
        `/projects/${projectId}/quickstart/import-sample`,
        slug ? { slug } : {},
    );
    return res.data;
}

export interface TrainDefaultResponse {
    status: 'training_started';
    experiment_id: number;
    experiment_name: string;
    base_model: string;
    training_mode: string | null;
    recipe_id: string | null;
    start_result: unknown;
}

export async function quickstartTrainDefault(
    projectId: number,
): Promise<TrainDefaultResponse> {
    const res = await api.post<TrainDefaultResponse>(
        `/projects/${projectId}/quickstart/train-default`,
    );
    return res.data;
}

export interface EvaluateLatestResponse {
    status: 'evaluation_complete';
    experiment_id: number;
    eval_type: string;
    result: Record<string, unknown>;
}

export async function quickstartEvaluateLatest(
    projectId: number,
): Promise<EvaluateLatestResponse> {
    const res = await api.post<EvaluateLatestResponse>(
        `/projects/${projectId}/quickstart/evaluate-latest`,
    );
    return res.data;
}

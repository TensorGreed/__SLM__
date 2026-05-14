/**
 * Typed wrappers around the Story 1.1 annotation API.
 *
 * Backend endpoints (under /api/projects/{id}/label-jobs):
 *   POST   /                                 — create label job
 *   GET    /                                 — list jobs
 *   GET    /{job_id}                         — detail + stats
 *   PATCH  /{job_id}                         — update mutable fields
 *   DELETE /{job_id}                         — drop job + cascade rows
 *   POST   /{job_id}/seed-from-dataset       — seed N work units
 *   POST   /{job_id}/next-row                — assign one unlabeled row
 *   POST   /{job_id}/rows/{row_id}/skip      — clear assignment
 *   POST   /{job_id}/rows/{row_id}/submit    — persist reviewer label
 */

import api from './client';

export type LabelType = 'classification' | 'span' | 'preference_pair';
export type JobStatus = 'active' | 'paused' | 'completed';

export interface LabelJob {
    id: number;
    project_id: number;
    name: string;
    label_type: LabelType;
    label_schema: {
        allowed_labels?: string[];
        span_types?: string[];
        [key: string]: unknown;
    };
    instructions: string | null;
    status: JobStatus;
    target_rows: number | null;
    created_at: string | null;
    updated_at: string | null;
}

export interface JobStats {
    job_id: number;
    name: string;
    label_type: LabelType;
    status: JobStatus;
    target_rows: number | null;
    total: number;
    labeled: number;
    assigned: number;
    unlabeled: number;
}

export interface LabelJobDetail extends LabelJob {
    stats: JobStats;
}

export interface LabelRow {
    id: number;
    job_id: number;
    source_row_id: string | null;
    raw_payload: Record<string, unknown>;
    assigned_to: number | null;
    assigned_at: string | null;
    label_payload: Record<string, unknown> | null;
    labeled_at: string | null;
    reviewer_notes: string | null;
}

export interface NextRowResponse {
    row: LabelRow | null;
    queue_empty: boolean;
}

export interface SubmitLabelBody {
    label_payload: Record<string, unknown>;
    reviewer_notes?: string | null;
}

export interface CreateJobBody {
    name: string;
    label_type: LabelType;
    label_schema?: Record<string, unknown>;
    instructions?: string | null;
    target_rows?: number | null;
}

export async function listLabelJobs(projectId: number): Promise<LabelJob[]> {
    const res = await api.get<{ jobs: LabelJob[] }>(
        `/projects/${projectId}/label-jobs/`,
    );
    return res.data?.jobs ?? [];
}

export async function getLabelJob(
    projectId: number,
    jobId: number,
): Promise<LabelJobDetail> {
    const res = await api.get<LabelJobDetail>(
        `/projects/${projectId}/label-jobs/${jobId}`,
    );
    return res.data;
}

export async function createLabelJob(
    projectId: number,
    body: CreateJobBody,
): Promise<LabelJob> {
    const res = await api.post<LabelJob>(
        `/projects/${projectId}/label-jobs/`,
        body,
    );
    return res.data;
}

export async function deleteLabelJob(
    projectId: number,
    jobId: number,
): Promise<void> {
    await api.delete(`/projects/${projectId}/label-jobs/${jobId}`);
}

export async function seedFromDataset(
    projectId: number,
    jobId: number,
    body: { dataset_id: number; n: number },
): Promise<{ seeded: number }> {
    const res = await api.post<{ seeded: number }>(
        `/projects/${projectId}/label-jobs/${jobId}/seed-from-dataset`,
        body,
    );
    return res.data;
}

export async function fetchNextRow(
    projectId: number,
    jobId: number,
    userId: number | null,
): Promise<NextRowResponse> {
    const res = await api.post<NextRowResponse>(
        `/projects/${projectId}/label-jobs/${jobId}/next-row`,
        { user_id: userId },
    );
    return res.data;
}

export async function submitLabel(
    projectId: number,
    jobId: number,
    rowId: number,
    body: SubmitLabelBody,
): Promise<LabelRow> {
    const res = await api.post<LabelRow>(
        `/projects/${projectId}/label-jobs/${jobId}/rows/${rowId}/submit`,
        body,
    );
    return res.data;
}

export async function skipRow(
    projectId: number,
    jobId: number,
    rowId: number,
): Promise<LabelRow> {
    const res = await api.post<LabelRow>(
        `/projects/${projectId}/label-jobs/${jobId}/rows/${rowId}/skip`,
    );
    return res.data;
}

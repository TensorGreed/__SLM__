/**
 * Typed client for the Jobs framework (Hardening Phase H1).
 *
 * The notification bell polls ``listActiveJobs`` every 3-5s while
 * any job is in-flight (or the bell dropdown is open). The shapes
 * here mirror ``backend/app/services/jobs_service.serialize_job``
 * exactly — keep in lockstep.
 */

import api from './client';


export type JobStatus =
    | 'queued'
    | 'running'
    | 'succeeded'
    | 'failed'
    | 'cancelled';


export interface Job {
    id: number;
    kind: string;
    title: string;
    status: JobStatus;
    progress: number | null;
    progress_message: string | null;
    project_id: number | null;
    user_id: number | null;
    params: Record<string, unknown>;
    result: Record<string, unknown> | null;
    error: string | null;
    queued_at: string | null;
    started_at: string | null;
    completed_at: string | null;
    dismissed_at: string | null;
}


export interface ActiveJobsResponse {
    count: number;
    jobs: Job[];
}


export async function listActiveJobs(opts: {
    projectId?: number | null;
    includeRecentlyCompleted?: boolean;
    limit?: number;
} = {}): Promise<ActiveJobsResponse> {
    const params: Record<string, string> = {};
    if (opts.projectId !== undefined && opts.projectId !== null) {
        params.project_id = String(opts.projectId);
    }
    if (opts.includeRecentlyCompleted !== undefined) {
        params.include_recently_completed = String(opts.includeRecentlyCompleted);
    }
    if (opts.limit !== undefined) {
        params.limit = String(opts.limit);
    }
    const query = new URLSearchParams(params).toString();
    const url = query ? `/jobs/active?${query}` : '/jobs/active';
    const resp = await api.get<ActiveJobsResponse>(url);
    return resp.data;
}


export async function getJob(jobId: number): Promise<Job> {
    const resp = await api.get<Job>(`/jobs/${jobId}`);
    return resp.data;
}


export async function dismissJob(jobId: number): Promise<Job> {
    const resp = await api.post<Job>(`/jobs/${jobId}/dismiss`);
    return resp.data;
}


export async function cancelJob(jobId: number): Promise<Job> {
    const resp = await api.post<Job>(`/jobs/${jobId}/cancel`);
    return resp.data;
}

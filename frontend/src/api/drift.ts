/**
 * Typed client for the drift API (E4 UI).
 *
 * Wraps the four endpoints the DriftReviewQueuePanel consults:
 *   - GET  /drift/settings    → opt-in flag + count
 *   - PUT  /drift/settings    → mutate opt-in + count
 *   - GET  /drift/review-queue → list queue rows
 *   - POST /drift/refresh-traps → manual generate trigger
 *   - POST /drift/review-queue/{row}/triage → accept / reject one row
 */

import api from './client';


export type DriftQueueStatus = 'pending' | 'accepted' | 'rejected';


export interface DriftQueueRow {
    id: number;
    project_id: number;
    source_drift_check_id: number | null;
    cluster_reason_code: string | null;
    cluster_signature: string | null;
    payload: Record<string, unknown>;
    status: DriftQueueStatus;
    source_confidence: string;
    triage_note: string | null;
    created_at: string;
    triaged_at: string | null;
}


export interface DriftQueueListResponse {
    project_id: number;
    rows: DriftQueueRow[];
}


export interface DriftSettings {
    project_id: number;
    enabled: boolean;
    count: number;
}


export interface RefreshTrapsResponse {
    project_id: number;
    generated: number;
    clusters_targeted: string[];
    simulated: boolean;
    row_ids: number[];
}


export interface DriftTriageResponse {
    id: number;
    status: DriftQueueStatus;
    triaged_at: string | null;
    triage_note: string | null;
}


export async function fetchDriftSettings(projectId: number): Promise<DriftSettings> {
    const resp = await api.get(`/projects/${projectId}/drift/settings`);
    return resp.data as DriftSettings;
}


export async function updateDriftSettings(
    projectId: number,
    patch: { enabled?: boolean; count?: number },
): Promise<DriftSettings> {
    const resp = await api.put(`/projects/${projectId}/drift/settings`, patch);
    return resp.data as DriftSettings;
}


export async function listDriftReviewQueue(
    projectId: number,
    options: { status?: DriftQueueStatus; limit?: number } = {},
): Promise<DriftQueueListResponse> {
    const params: Record<string, string | number> = {};
    if (options.status !== undefined) params.status = options.status;
    if (options.limit !== undefined) params.limit = options.limit;
    const resp = await api.get(`/projects/${projectId}/drift/review-queue`, { params });
    return resp.data as DriftQueueListResponse;
}


export async function refreshDriftTraps(
    projectId: number,
    options: { count?: number; simulate?: boolean } = {},
): Promise<RefreshTrapsResponse> {
    const params: Record<string, string | number | boolean> = {};
    if (options.count !== undefined) params.count = options.count;
    if (options.simulate !== undefined) params.simulate = options.simulate;
    const resp = await api.post(
        `/projects/${projectId}/drift/refresh-traps`,
        null,
        { params },
    );
    return resp.data as RefreshTrapsResponse;
}


export async function triageDriftRow(
    projectId: number,
    rowId: number,
    payload: { accept: boolean; note?: string },
): Promise<DriftTriageResponse> {
    const resp = await api.post(
        `/projects/${projectId}/drift/review-queue/${rowId}/triage`,
        payload,
    );
    return resp.data as DriftTriageResponse;
}

/**
 * Typed wrapper for the SQL execution scoring settings API.
 *
 *   GET /projects/{id}/sql-execution-config
 *   PUT /projects/{id}/sql-execution-config
 */

import api from './client';

export type SqlOrderMode = 'auto' | 'always' | 'never';

export interface SqlExecutionConfig {
    enabled?: boolean;
    db_paths?: string[];
    timeout_ms?: number;
    row_cap?: number;
    float_rel_tol?: number;
    order?: SqlOrderMode;
    refusal_marker?: string;
}

export interface SqlExecutionConfigResponse {
    config: SqlExecutionConfig & { updated_at?: string | null };
    effective: SqlExecutionConfig & { active?: boolean };
    databases: Array<{ path: string; ok: boolean; error: string | null; size_bytes: number | null }>;
    data_dir: string;
}

export async function getSqlExecutionConfig(projectId: number): Promise<SqlExecutionConfigResponse> {
    const res = await api.get<SqlExecutionConfigResponse>(`/projects/${projectId}/sql-execution-config`);
    return res.data;
}

export async function putSqlExecutionConfig(projectId: number, body: SqlExecutionConfig): Promise<SqlExecutionConfigResponse> {
    const res = await api.put<SqlExecutionConfigResponse>(`/projects/${projectId}/sql-execution-config`, body);
    return res.data;
}

/** The 422 `metadata.errors` list (INVALID_SQL_EXECUTION_CONFIG), else the message. */
export function sqlExecutionErrors(err: unknown): string[] {
    const data = (err as { response?: { data?: { detail?: unknown; message?: string; metadata?: { errors?: unknown } } } })?.response?.data;
    const list = data?.metadata?.errors;
    if (Array.isArray(list) && list.length > 0) return list.map(String);
    if (typeof data?.detail === 'string') return [data.detail];
    if (typeof data?.message === 'string') return [data.message];
    return ['Could not save the settings.'];
}

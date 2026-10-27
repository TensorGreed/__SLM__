/**
 * Typed wrapper for the tool-scoring settings API.
 *
 *   GET /projects/{id}/tool-scoring-config
 *   PUT /projects/{id}/tool-scoring-config
 */

import api from './client';

export type ToolScoringMode = 'tool_call_valid' | 'tool_args_match' | 'plan_order_valid';

export const TOOL_SCORING_MODES: Array<{ id: ToolScoringMode; label: string; hint: string }> = [
    { id: 'tool_call_valid', label: 'Valid tool call', hint: 'every call names a known, allowed tool and its arguments fit the tool schema' },
    { id: 'tool_args_match', label: 'Arguments match the answer key', hint: 'same tool and matching argument values as the answer key' },
    { id: 'plan_order_valid', label: 'Plan order valid', hint: 'unique step ids; every dependency exists and there are no cycles (plans only)' },
];

export interface ToolScoringConfig {
    enabled?: boolean;
    modes?: ToolScoringMode[];
    steps_pointer?: string;
    tool_field?: string;
    args_field?: string;
    id_field?: string;
    depends_field?: string;
    arg_keys?: string[];
    dry_run?: boolean;
}

export interface ToolScoringConfigResponse {
    config: ToolScoringConfig;
    effective: ToolScoringConfig | null;
    catalog: { version: string; tool_count: number; source_kind: string } | null;
}

export async function getToolScoringConfig(projectId: number): Promise<ToolScoringConfigResponse> {
    const res = await api.get<ToolScoringConfigResponse>(`/projects/${projectId}/tool-scoring-config`);
    return res.data;
}

export async function putToolScoringConfig(projectId: number, body: ToolScoringConfig): Promise<ToolScoringConfigResponse> {
    const res = await api.put<ToolScoringConfigResponse>(`/projects/${projectId}/tool-scoring-config`, body);
    return res.data;
}

/** The 422 `metadata.errors` list (INVALID_TOOL_SCORING_CONFIG), else the message. */
export function toolScoringErrors(err: unknown): string[] {
    const data = (err as { response?: { data?: { detail?: unknown; message?: string; metadata?: { errors?: unknown } } } })?.response?.data;
    const list = data?.metadata?.errors;
    if (Array.isArray(list) && list.length > 0) return list.map(String);
    if (typeof data?.detail === 'string') return [data.detail];
    if (typeof data?.message === 'string') return [data.message];
    return ['Could not save the settings.'];
}

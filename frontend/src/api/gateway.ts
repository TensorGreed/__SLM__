/**
 * Serving gateway client (G11): API keys, per-project gateway settings, the
 * "Use from Claude Code" snippets, telemetry, and the multi-LoRA serve plan.
 */

import api from './client';

export interface GatewayKey {
    id: number;
    name: string;
    key_prefix: string;
    project_ids: number[];
    rate_limit_per_minute: number;
    is_active: boolean;
    created_at: string | null;
    last_used_at: string | null;
    revoked_at: string | null;
}

export interface GatewayTool {
    name: string;
    description: string;
    input_schema: Record<string, unknown>;
    /** Model id suffix: "" (project default), ":base" or ":run-N". */
    model?: string;
    prompt_template: string;
    system_prompt?: string | null;
    retrieval?: boolean;
    max_tokens?: number | null;
}

export type GatewayUpstream = 'in_process' | 'openai_compatible';
export type GatewayUpstreamKind = 'vllm' | 'ollama' | 'openai';

export interface GatewaySettings {
    upstream: GatewayUpstream;
    base_url?: string | null;
    upstream_kind?: GatewayUpstreamKind | null;
    upstream_model_map?: Record<string, string>;
    default_run_id?: number | null;
    tools?: GatewayTool[];
}

export interface ServableRun {
    id: number;
    name: string;
    base_model: string | null;
    passage_trained: boolean;
    adapter: boolean;
}

export interface GatewaySettingsResponse {
    settings: GatewaySettings;
    slug: string;
    model_ids: string[];
    base_model: string;
    default_run_resolved: number | null;
    default_resolves_to: { kind: 'run' | 'base_with_retrieval' | 'none'; run_id: number | null };
    guided_json_available: boolean;
    servable_runs: ServableRun[];
}

export interface GatewaySnippets {
    gateway_url: string;
    model_ids: string[];
    mcp_json: Record<string, unknown>;
    curl: string;
    openai_python: string;
}

export interface GatewayTelemetry {
    summary: {
        by_model: { model_id: string; requests: number; error_rate: number; p50_ms: number | null; p95_ms: number | null }[];
        total: number;
    };
    recent: {
        ts: string;
        model_id: string;
        run_id: number | null;
        status_code: number;
        latency_ms: number;
        retrieval_applied: boolean;
        guided_json_mode: string | null;
        key_prefix?: string | null;
    }[];
}

export interface MultiLoraPlan {
    base_model: string;
    adapters: { run_id: number; name: string; module_name: string; path: string; rank: number | null }[];
    template: { command?: string; [k: string]: unknown };
    gateway_settings_hint: { upstream: GatewayUpstream; base_url: string; upstream_kind: GatewayUpstreamKind };
}

export interface FieldError { field: string; message: string }

/** Message + per-field errors from a 400 (structured envelope or plain). */
export function parseGatewayError(err: unknown): { message: string; errors: FieldError[] } {
    const data = (err as { response?: { data?: Record<string, unknown> } })?.response?.data;
    if (!data) return { message: err instanceof Error ? err.message : 'Request failed', errors: [] };
    const meta = data.metadata as { errors?: FieldError[] } | undefined;
    const errors = (data.errors as FieldError[] | undefined) ?? meta?.errors ?? [];
    const detail = data.detail;
    const message = typeof detail === 'string' ? detail : 'Request failed';
    return { message, errors: Array.isArray(errors) ? errors : [] };
}

export async function listGatewayKeys(): Promise<GatewayKey[]> {
    const res = await api.get<{ keys: GatewayKey[] }>('/gateway/keys');
    return res.data.keys ?? [];
}

export async function createGatewayKey(body: {
    name: string; project_ids: number[]; rate_limit_per_minute?: number;
}): Promise<{ key: string; record: GatewayKey }> {
    const res = await api.post<{ key: string; record: GatewayKey }>('/gateway/keys', body);
    return res.data;
}

export async function revokeGatewayKey(id: number): Promise<GatewayKey> {
    const res = await api.delete<{ record: GatewayKey }>(`/gateway/keys/${id}`);
    return res.data.record;
}

export async function getGatewaySettings(projectId: number): Promise<GatewaySettingsResponse> {
    const res = await api.get<GatewaySettingsResponse>(`/projects/${projectId}/gateway/settings`);
    return res.data;
}

export async function putGatewaySettings(
    projectId: number, body: Partial<GatewaySettings>,
): Promise<GatewaySettingsResponse> {
    const res = await api.put<GatewaySettingsResponse>(`/projects/${projectId}/gateway/settings`, body);
    return res.data;
}

export async function getGatewaySnippets(projectId: number): Promise<GatewaySnippets> {
    const res = await api.get<GatewaySnippets>(`/projects/${projectId}/gateway/snippets`);
    return res.data;
}

export async function getGatewayTelemetry(projectId: number): Promise<GatewayTelemetry> {
    const res = await api.get<GatewayTelemetry>(`/projects/${projectId}/gateway/telemetry`);
    return res.data;
}

export async function buildMultiLoraPlan(
    projectId: number, body: { experiment_ids: number[]; host?: string; port?: number },
): Promise<MultiLoraPlan> {
    const res = await api.post<MultiLoraPlan>(`/projects/${projectId}/serve/multi-lora-plan`, body);
    return res.data;
}

export async function startMultiLoraRun(
    projectId: number, body: { experiment_ids: number[]; host?: string; port?: number },
): Promise<Record<string, unknown>> {
    const res = await api.post<Record<string, unknown>>(`/projects/${projectId}/serve/multi-lora-runs/start`, body);
    return res.data;
}

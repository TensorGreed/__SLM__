/**
 * Typed wrapper for the project tool catalog API (imported MCP / OpenAPI tools).
 *
 *   GET    /projects/{id}/tool-catalog
 *   POST   /projects/{id}/tool-catalog/import/mcp        (?async_job=true returns a Job)
 *   POST   /projects/{id}/tool-catalog/import/openapi    { filename, content }
 *   PUT    /projects/{id}/tool-catalog/tools             { tools }
 *   PATCH  /projects/{id}/tool-catalog/flags             { flags: { name: { allowed?, read_only? } } }
 *   POST   /projects/{id}/tool-catalog/restore           { version }
 *   DELETE /projects/{id}/tool-catalog
 *
 * Secret values are write-only: the API returns env / header NAMES, never values.
 */

import api from './client';

export type ToolSourceKind = 'mcp_stdio' | 'mcp_http' | 'openapi' | 'manual' | 'domain_pack';

export interface ToolCatalogSource {
    kind: ToolSourceKind | string;
    label?: string;
    command?: string[];
    env_names?: string[];
    url?: string;
    header_names?: string[];
    secret_provider?: string;
    filename?: string;
    title?: string;
    version?: string;
    pack_id?: string | number;
}

export interface ToolCatalogServer {
    name?: string;
    version?: string;
    protocol_version?: string;
}

export interface CatalogTool {
    name: string;
    description?: string;
    input_schema?: Record<string, unknown>;
    annotations?: Record<string, unknown>;
    read_only: boolean | null;
    destructive: boolean | null;
    schema_error?: string;
    /** Effective flags, resolved by the server. */
    allowed: boolean;
    read_only_effective: boolean;
    mutating: boolean;
    /** Compact one-line signature when the server provides it. */
    signature?: string;
}

export interface ToolCatalog {
    tools: CatalogTool[];
    version: string;
    flags_version?: string;
    source: ToolCatalogSource | null;
    server?: ToolCatalogServer | null;
    imported_at?: string | null;
}

export interface ToolCatalogHistoryEntry {
    version: string;
    imported_at?: string | null;
    source_kind?: string;
    tool_count: number;
    current: boolean;
    restorable: boolean;
}

export interface ToolCatalogStatus {
    catalog: ToolCatalog | null;
    history: ToolCatalogHistoryEntry[];
}

export interface ToolFlagPatch {
    allowed?: boolean;
    read_only?: boolean | null;
}

export interface McpStdioRequest {
    kind: 'mcp_stdio';
    command: string[];
    /** NAME -> value; null keeps the stored secret. */
    env: Record<string, string | null>;
    label?: string;
}

export interface McpHttpRequest {
    kind: 'mcp_http';
    url: string;
    headers: Record<string, string | null>;
    label?: string;
}

export type McpImportRequest = McpStdioRequest | McpHttpRequest;

export async function getToolCatalog(projectId: number): Promise<ToolCatalogStatus> {
    const res = await api.get<ToolCatalogStatus>(`/projects/${projectId}/tool-catalog`);
    return res.data;
}

export async function importMcpTools(projectId: number, body: McpImportRequest): Promise<ToolCatalogStatus> {
    const res = await api.post<ToolCatalogStatus>(`/projects/${projectId}/tool-catalog/import/mcp`, body);
    return res.data;
}

export async function importOpenApiTools(projectId: number, filename: string, content: string): Promise<ToolCatalogStatus> {
    const res = await api.post<ToolCatalogStatus>(`/projects/${projectId}/tool-catalog/import/openapi`, { filename, content });
    return res.data;
}

export async function patchToolFlags(projectId: number, flags: Record<string, ToolFlagPatch>): Promise<ToolCatalogStatus> {
    const res = await api.patch<ToolCatalogStatus>(`/projects/${projectId}/tool-catalog/flags`, { flags });
    return res.data;
}

export async function restoreToolCatalog(projectId: number, version: string): Promise<ToolCatalogStatus> {
    const res = await api.post<ToolCatalogStatus>(`/projects/${projectId}/tool-catalog/restore`, { version });
    return res.data;
}

export async function clearToolCatalog(projectId: number): Promise<void> {
    await api.delete(`/projects/${projectId}/tool-catalog`);
}

/** A readable one-line signature when the server did not send one: `name(arg: type, other?: type)`. */
export function toolSignature(tool: CatalogTool): string {
    if (tool.signature) return tool.signature;
    const schema = (tool.input_schema ?? {}) as { properties?: Record<string, { type?: unknown; enum?: unknown[] }>; required?: string[] };
    const required = new Set(schema.required ?? []);
    const parts = Object.entries(schema.properties ?? {}).map(([key, prop]) => {
        const values = Array.isArray(prop?.enum) ? prop.enum.map(String).slice(0, 8).join('|') : null;
        const type = values ?? (Array.isArray(prop?.type) ? prop.type.join('|') : String(prop?.type ?? 'any'));
        return `${key}${required.has(key) ? '' : '?'}: ${type}`;
    });
    return `${tool.name}(${parts.join(', ')})`;
}

/** The risk badge text for a tool. */
export function toolRiskLabel(tool: CatalogTool): 'destructive' | 'mutating' | 'read-only' {
    if (tool.destructive === true) return 'destructive';
    return tool.mutating ? 'mutating' : 'read-only';
}

export interface ToolCatalogErrorInfo {
    code: string | null;
    message: string;
    errors: string[];
}

/** Reads the structured-error envelope (error_code / metadata hoisted to the top level). */
export function toolCatalogError(err: unknown, fallback = 'The request failed.'): ToolCatalogErrorInfo {
    const data = (err as { response?: { data?: { error_code?: string; detail?: unknown; message?: string; metadata?: { errors?: unknown } } } })?.response?.data;
    const list = data?.metadata?.errors;
    const errors = Array.isArray(list) ? list.map(String) : [];
    const message = typeof data?.detail === 'string' ? data.detail : typeof data?.message === 'string' ? data.message : fallback;
    return { code: typeof data?.error_code === 'string' ? data.error_code : null, message, errors };
}

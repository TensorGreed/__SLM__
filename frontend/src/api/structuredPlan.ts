/**
 * Typed wrapper for the structured-plan settings API (the `structured-plan` recipe).
 *
 *   GET  /projects/{id}/structured-plan-config
 *   PUT  /projects/{id}/structured-plan-config   (partial merge; null clears a key)
 *   POST /projects/{id}/structured-plan/validate
 */

import api from './client';

export interface StructuredPlanConfig {
    /** file name -> parsed JSON-Schema document */
    schema_files?: Record<string, unknown>;
    root_schema?: string;
    rubric_yaml?: string | null;
    pointers?: string[];
    judge_enabled?: boolean;
    retrieve_passages?: boolean | null;
    top_k?: number;
    updated_at?: string;
}

export interface StructuredPlanSchemaInfo {
    root: string;
    files: string[];
    sha256: string;
    valid: boolean;
    errors: string[];
}

export interface StructuredPlanRubricCriterion {
    id: string;
    description: string;
    weight: number;
    critical: boolean;
}

export interface StructuredPlanRubricInfo {
    id: string;
    title: string;
    sha256: string;
    criteria: StructuredPlanRubricCriterion[];
}

export interface StructuredPlanConfigResponse {
    config: StructuredPlanConfig;
    effective: {
        judge_enabled?: boolean;
        retrieve_passages?: boolean;
        top_k?: number;
        pointers?: string[];
    };
    schema: StructuredPlanSchemaInfo | null;
    rubric: StructuredPlanRubricInfo | null;
    guided_json_present: boolean;
}

/** One finding of `SchemaBundle.validate`. */
export interface StructuredPlanValidationError {
    pointer: string;
    message: string;
    validator?: string;
    schema_path?: string;
}

export interface StructuredPlanValidateResponse {
    valid: boolean;
    errors: StructuredPlanValidationError[];
}

/** Partial body for the PUT: a null value clears that key. */
export type StructuredPlanConfigPatch = {
    [K in keyof StructuredPlanConfig]?: StructuredPlanConfig[K] | null;
};

export async function getStructuredPlanConfig(projectId: number): Promise<StructuredPlanConfigResponse> {
    const res = await api.get<StructuredPlanConfigResponse>(`/projects/${projectId}/structured-plan-config`);
    return res.data;
}

export async function putStructuredPlanConfig(
    projectId: number,
    patch: StructuredPlanConfigPatch,
): Promise<StructuredPlanConfigResponse> {
    const res = await api.put<StructuredPlanConfigResponse>(`/projects/${projectId}/structured-plan-config`, patch);
    return res.data;
}

export async function validateStructuredPlanJson(
    projectId: number,
    value: unknown,
): Promise<StructuredPlanValidateResponse> {
    const res = await api.post<StructuredPlanValidateResponse>(`/projects/${projectId}/structured-plan/validate`, { value });
    return res.data;
}

export function shortSha(sha: string | null | undefined): string {
    return sha ? sha.slice(0, 8) : '';
}

/** The 422 `metadata.errors` list (error envelope hoists metadata to the top level), else the message. */
export function structuredPlanErrors(err: unknown): string[] {
    const data = (err as { response?: { data?: { detail?: unknown; message?: string; metadata?: { errors?: unknown } } } })?.response?.data;
    const list = data?.metadata?.errors;
    if (Array.isArray(list) && list.length > 0) return list.map(String);
    if (typeof data?.detail === 'string') return [data.detail];
    if (typeof data?.message === 'string') return [data.message];
    return ['Could not save the settings.'];
}

/**
 * Slice-definitions CRUD client.
 *
 * Mirrors the phase 2 slice 1 endpoint shape:
 *   GET    /api/projects/{id}/slice-definitions
 *   PUT    /api/projects/{id}/slice-definitions
 *   DELETE /api/projects/{id}/slice-definitions
 *
 * Type shapes match the Pydantic schemas in backend/app/schemas/project.py
 * — kept narrow so the editor's TS contract reflects the validator's
 * actual contract instead of drifting.
 */

import api from './client';


// Quality-Lift phase 2 slice 1 closed op set (see
// backend/app/services/slice_definitions_service.SLICE_OPERATORS).
// Mirrored on the frontend so the editor's op picker is exhaustive
// without a server round-trip. The op-pickers in slice 1 of phase 7
// are dropdowns sourced from this exact tuple; if the backend tuple
// ever grows, this constant must update in lockstep — the validator
// will reject the unknown op otherwise.
export const SLICE_OPERATORS = [
    'eq', 'neq',
    'gt', 'gte', 'lt', 'lte',
    'in', 'not_in',
    'contains',
    'regex',
    'exists',
] as const;
export type SliceOperator = typeof SLICE_OPERATORS[number];

// Quality-Lift phase 2 slice 2 platform fields (see
// backend/app/services/slice_evaluator_service.PLATFORM_FIELDS).
// Mirrored so the field-picker autocompletes the platform-computed
// fields the evaluator injects on every row. Users can still type
// arbitrary dataset field names — these are just the documented
// helpful defaults.
export const PLATFORM_FIELDS: ReadonlyArray<{ name: string; description: string }> = [
    { name: 'input_length',      description: 'Char count of the prompt/input.' },
    { name: 'input_token_count', description: 'Token count via the eval tokenizer (word count fallback).' },
    { name: 'prediction_length', description: 'Char count of the model\'s prediction.' },
    { name: 'reference_length',  description: 'Char count of the gold reference.' },
    { name: 'latency_ms',        description: 'Per-row inference latency (when available).' },
    { name: '_dataset_index',    description: 'Row position in the eval set; useful for sampling slices.' },
] as const;

export interface SliceClause {
    field: string;
    op: SliceOperator;
    value: unknown;
}

export interface SliceDefinition {
    slice_id: string;
    display_name: string;
    where: SliceClause[];
}

export interface SliceDefinitionsPayload {
    slices: SliceDefinition[];
}

export interface SliceDefinitionsResponse {
    project_id: number;
    slice_definitions: SliceDefinitionsPayload;
}


export async function fetchSliceDefinitions(projectId: number): Promise<SliceDefinitionsResponse> {
    const resp = await api.get<SliceDefinitionsResponse>(
        `/projects/${projectId}/slice-definitions`,
    );
    return resp.data;
}

export async function saveSliceDefinitions(
    projectId: number,
    payload: SliceDefinitionsPayload,
): Promise<SliceDefinitionsResponse> {
    const resp = await api.put<SliceDefinitionsResponse>(
        `/projects/${projectId}/slice-definitions`,
        payload,
    );
    return resp.data;
}

export async function clearSliceDefinitions(projectId: number): Promise<SliceDefinitionsResponse> {
    const resp = await api.delete<SliceDefinitionsResponse>(
        `/projects/${projectId}/slice-definitions`,
    );
    return resp.data;
}

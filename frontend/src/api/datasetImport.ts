/**
 * Typed wrappers around the Phase A–E dataset-import API.
 *
 * Endpoints:
 *   GET  /api/dataset-import/sources             — list registered source ids
 *   GET  /api/dataset-import/mappers             — list registered mapper ids
 *   POST /api/dataset-import/introspect          — sniff + propose mapping
 *   POST /api/projects/{id}/dataset-import/preview — dry-run a mapping
 *   POST /api/projects/{id}/dataset-import/run     — persist accepted rows
 *
 * Shared types are exported so the wizard components can reuse them
 * without redefining payload shapes.
 */

import api from './client';

// ── Catalog ──────────────────────────────────────────────────────────

export interface SourceListResponse {
    sources: string[];
}

export interface MapperListResponse {
    mappers: string[];
}

export async function listSources(): Promise<string[]> {
    const res = await api.get<SourceListResponse>('/dataset-import/sources');
    return res.data?.sources ?? [];
}

export async function listMappers(): Promise<string[]> {
    const res = await api.get<MapperListResponse>('/dataset-import/mappers');
    return res.data?.mappers ?? [];
}

// ── Introspect ───────────────────────────────────────────────────────

export interface ColumnSignature {
    name: string;
    column_type: string;
    confidence: number;
    unique_values: string[];
    sample_value: unknown;
    notes: string;
}

export interface ShapeHypothesisDict {
    mapper_id: string;
    target_task_profile: string;
    field_map: Record<string, unknown>;
    confidence: number;
    rationale: string;
    warnings: string[];
}

export interface ProposalDict {
    target_task_profile: string;
    mapper_id: string;
    field_map: Record<string, unknown>;
    confidence: number;
    rationale: string;
    warnings: string[];
    needs_force: boolean;
}

export interface IntrospectResponse {
    source_id: string;
    locator: string;
    resolved_path: string | null;
    approximate_total_rows: number | null;
    columns: string[];
    sample_rows: Array<Record<string, unknown>>;
    column_signatures: ColumnSignature[];
    hypotheses: ShapeHypothesisDict[];
    proposal: ProposalDict | null;
    confidence_threshold: number;
}

export async function introspectLocator(
    locator: string,
    sample_size = 20,
): Promise<IntrospectResponse> {
    const res = await api.post<IntrospectResponse>('/dataset-import/introspect', {
        locator,
        sample_size,
    });
    return res.data;
}

// ── Preview / Run ────────────────────────────────────────────────────

export interface ImportRowSample {
    payload: Record<string, unknown>;
    row_key: string | null;
    warnings: string[];
}

export interface RejectedRowSample {
    reason: string;
    detail: string;
    row_index: number | null;
    raw_row: Record<string, unknown>;
}

export interface ImportResultDict {
    accepted_count: number;
    rejected_count: number;
    source_id: string;
    mapper_id: string;
    target_task_profile: string;
    locator: string;
    written_path: string | null;
    dry_run: boolean;
    rejection_counts: Record<string, number>;
    warnings: string[];
    accepted_sample: ImportRowSample[];
    rejected_sample: RejectedRowSample[];
}

export interface PreviewRequestBody {
    locator: string;
    mapper_id: string;
    field_map: Record<string, unknown>;
    limit?: number | null;
    drop_reasons?: string[];
    sample_cap?: number;
}

export interface RunRequestBody {
    locator: string;
    mapper_id: string;
    field_map: Record<string, unknown>;
    limit?: number | null;
    drop_reasons?: string[];
}

export async function previewImport(
    projectId: number,
    body: PreviewRequestBody,
): Promise<ImportResultDict> {
    const res = await api.post<ImportResultDict>(
        `/projects/${projectId}/dataset-import/preview`,
        body,
    );
    return res.data;
}

export async function runImport(
    projectId: number,
    body: RunRequestBody,
): Promise<ImportResultDict> {
    const res = await api.post<ImportResultDict>(
        `/projects/${projectId}/dataset-import/run`,
        body,
    );
    return res.data;
}

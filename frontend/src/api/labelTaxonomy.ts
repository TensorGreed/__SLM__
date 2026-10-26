/**
 * Typed client for the project label taxonomy (a hierarchical label set with
 * per-namespace crosswalk references).
 *
 *   GET    /projects/{id}/label-taxonomy
 *   PUT    /projects/{id}/label-taxonomy
 *   POST   /projects/{id}/label-taxonomy/validate
 *   POST   /projects/{id}/label-taxonomy/restore
 *   DELETE /projects/{id}/label-taxonomy
 *   GET    /projects/{id}/label-taxonomy/resolve?label=...
 */

import api from './client';

export interface TaxonomyEntry {
    id: string;
    title: string;
    parent: string | null;
    aliases: string[];
    attrs: Record<string, unknown>;
    crosswalk: Record<string, string[]>;
}

export interface Taxonomy {
    name: string | null;
    source_version: string | null;
    entries: TaxonomyEntry[];
}

export interface TaxonomyHistoryItem {
    version: string;
    entry_count: number;
    saved_at: string;
    current: boolean;
    /** False for older versions whose snapshot is no longer kept. */
    restorable: boolean;
}

export interface TaxonomyState {
    taxonomy: Taxonomy | null;
    version: string | null;
    entry_count: number;
    namespaces: string[];
    history: TaxonomyHistoryItem[];
}

export interface TaxonomyIssue {
    code: string;
    id?: string | null;
    message: string;
}

export interface TaxonomyValidation {
    valid: boolean;
    errors: TaxonomyIssue[];
    warnings: TaxonomyIssue[];
    entry_count: number;
    version: string | null;
}

export interface TaxonomyResolution {
    raw: string;
    id: string | null;
    matched_by: 'id' | 'alias' | 'title' | 'ambiguous' | null;
    title: string | null;
    /** Entry ids a label could mean when `matched_by` is 'ambiguous'. */
    candidates: string[];
}

/** Exactly one of `content` / `entries`. */
export interface TaxonomyBody {
    content?: string;
    entries?: TaxonomyEntry[];
    format?: 'auto' | 'yaml' | 'json';
    crosswalk_from?: string[];
    name?: string;
}

export async function getTaxonomy(projectId: number): Promise<TaxonomyState> {
    return (await api.get<TaxonomyState>(`/projects/${projectId}/label-taxonomy`)).data;
}

export async function saveTaxonomy(projectId: number, body: TaxonomyBody): Promise<TaxonomyState> {
    return (await api.put<TaxonomyState>(`/projects/${projectId}/label-taxonomy`, body)).data;
}

export async function validateTaxonomy(projectId: number, body: TaxonomyBody): Promise<TaxonomyValidation> {
    return (await api.post<TaxonomyValidation>(`/projects/${projectId}/label-taxonomy/validate`, body)).data;
}

export async function restoreTaxonomy(projectId: number, version: string): Promise<TaxonomyState> {
    return (await api.post<TaxonomyState>(`/projects/${projectId}/label-taxonomy/restore`, { version })).data;
}

export async function deleteTaxonomy(projectId: number): Promise<TaxonomyState> {
    return (await api.delete<TaxonomyState>(`/projects/${projectId}/label-taxonomy`)).data;
}

export async function resolveLabel(projectId: number, label: string): Promise<TaxonomyResolution> {
    return (await api.get<TaxonomyResolution>(`/projects/${projectId}/label-taxonomy/resolve`, { params: { label } })).data;
}

/** Pull the structured 422 issues out of an axios error (INVALID_TAXONOMY). */
export function taxonomyErrorIssues(err: unknown): TaxonomyIssue[] | null {
    const body = (err as { response?: { data?: { metadata?: { errors?: TaxonomyIssue[] } } } })?.response?.data;
    const errors = body?.metadata?.errors;
    return Array.isArray(errors) ? errors : null;
}

/** The server's message for a failed call: the envelope's `detail` string (or `message`), else the error's own text. */
export function taxonomyErrorMessage(err: unknown, fallback: string): string {
    const data = (err as { response?: { data?: { detail?: unknown; message?: unknown } } })?.response?.data;
    if (typeof data?.detail === 'string' && data.detail) return data.detail;
    if (typeof data?.message === 'string' && data.message) return data.message;
    return err instanceof Error && err.message ? err.message : fallback;
}

/**
 * Domain-pack content API client: import (preview + commit), per-pack content,
 * instantiate into a new project, and export a project back to a pack zip.
 *
 * A pack's overlay / hooks are managed by `DomainPackManager`; this client
 * covers the content sections (taxonomy, checks, rubrics, schemas, documents,
 * gold rows, project template).
 */

import api from './client';
import type { DomainPackResponse, Project } from '../types';

export interface PackIssue {
    code: string;
    path?: string | null;
    message: string;
}

export interface PackSummary {
    taxonomy_entries: number;
    probe_packs: number;
    probes: number;
    critical_probes: number;
    rubrics: number;
    schemas: number;
    tool_catalog: boolean;
    documents: number;
    gold_dev_rows: number;
    gold_test_rows: number;
    project_template: boolean;
}

export interface IdDiff {
    added: string[];
    removed: string[];
    changed: string[];
}

export interface PackDiff {
    version: { from: string | null; to: string | null };
    sections: {
        taxonomy?: IdDiff;
        probes?: IdDiff;
        rubrics?: IdDiff;
        schemas?: IdDiff;
        documents?: IdDiff;
        gold?: {
            dev?: { from: number; to: number; changed?: boolean };
            test?: { from: number; to: number; changed?: boolean };
        };
        project_template?: { changed_fields: string[] };
        overlay?: { changed: boolean };
        tool_catalog?: { changed: boolean };
    };
}

export interface ImportPreview {
    valid: boolean;
    errors: PackIssue[];
    warnings: PackIssue[];
    summary: PackSummary | null;
    manifest: Record<string, unknown> | null;
    content_hash: string | null;
    existing: { version: string; content_hash: string } | null;
    proposed_version: string | null;
    unchanged: boolean;
    diff: PackDiff | null;
}

export interface ImportResult {
    pack: DomainPackResponse;
    version: string;
    unchanged: boolean;
    diff: PackDiff | null;
    warnings?: PackIssue[];
}

export interface PackContent {
    pack_id: string;
    version: string;
    versions: { version: string; content_hash?: string; imported_at?: string }[];
    manifest: Record<string, unknown>;
    summary: PackSummary;
    files: { path: string; sha256: string; size: number }[];
}

export interface InstantiateSummary {
    pack_id: string;
    version: string;
    content_hash: string;
    recipe_id: string | null;
    base_model: string | null;
    taxonomy_entries: number;
    probe_packs: number;
    probes: number;
    critical_probes: number;
    documents: number;
    gold_dev_rows: number;
    gold_test_rows: number;
    rubrics: number;
    schemas: number;
    tool_catalog: boolean;
    warnings: PackIssue[];
}

export interface InstantiateResult {
    project: Project;
    summary: InstantiateSummary;
}

export type IncludeDocuments = 'pack' | 'all' | 'none';

function zipForm(file: File): FormData {
    const form = new FormData();
    form.append('file', file);
    return form;
}

export async function previewPackImport(file: File): Promise<ImportPreview> {
    const res = await api.post<ImportPreview>('/domain-packs/import/preview', zipForm(file));
    return res.data;
}

export async function importPack(file: File): Promise<ImportResult> {
    const res = await api.post<ImportResult>('/domain-packs/import', zipForm(file));
    return res.data;
}

/** Content of a pack, or null when the pack has none (404). */
export async function fetchPackContent(
    packId: string,
    version?: string,
): Promise<PackContent | null> {
    try {
        const res = await api.get<PackContent>(
            `/domain-packs/${encodeURIComponent(packId)}/content`,
            version ? { params: { version } } : undefined,
        );
        return res.data;
    } catch (err) {
        const status = (err as { response?: { status?: number } })?.response?.status;
        if (status === 404) return null;
        throw err;
    }
}

export async function instantiatePack(
    packId: string,
    body: { project_name?: string; version?: string },
): Promise<InstantiateResult> {
    const res = await api.post<InstantiateResult>(
        `/domain-packs/${encodeURIComponent(packId)}/instantiate`,
        body,
    );
    return res.data;
}

export async function exportProjectPack(
    projectId: number,
    includeDocuments: IncludeDocuments,
): Promise<{ blob: Blob; filename: string }> {
    const res = await api.get<Blob>(`/projects/${projectId}/domain-pack/export`, {
        params: { include_documents: includeDocuments },
        responseType: 'blob',
    });
    const disposition = String(
        (res.headers as Record<string, unknown> | undefined)?.['content-disposition'] ?? '',
    );
    const match = /filename="?([^";]+)"?/.exec(disposition);
    return { blob: res.data, filename: match ? match[1] : `project-${projectId}-pack.zip` };
}

/** Group issues by code, keeping first-seen order. */
export function groupIssuesByCode(issues: PackIssue[]): [string, PackIssue[]][] {
    const groups = new Map<string, PackIssue[]>();
    for (const issue of issues) {
        const list = groups.get(issue.code) ?? [];
        list.push(issue);
        groups.set(issue.code, list);
    }
    return [...groups.entries()];
}

/** Pull `metadata.errors` (structured envelope) or a message out of a failed call. */
export function extractPackErrors(err: unknown): { message: string; errors: PackIssue[] } {
    const data = (err as { response?: { data?: Record<string, unknown> } })?.response?.data;
    const metadata = (data?.metadata ?? {}) as { errors?: PackIssue[] };
    const errors = Array.isArray(metadata.errors) ? metadata.errors : [];
    const detail = data?.detail;
    const message =
        typeof detail === 'string' ? detail : err instanceof Error ? err.message : 'Request failed';
    return { message, errors };
}

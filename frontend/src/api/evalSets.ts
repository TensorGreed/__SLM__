/**
 * Typed client for named, versioned eval sets + prompt-format variants (every task shape).
 *
 *   GET    /projects/{id}/eval-sets                          list (versions, variants, last eval per label)
 *   POST   /projects/{id}/eval-sets                          multipart: name, file, drop_lines -> new set
 *   POST   /projects/{id}/eval-sets/from-source              { name, source: {kind: split|eval_set, ...}, variant? }
 *   GET    /projects/{id}/eval-sets/variant-support          ?experiment_id -> { supported, reason }
 *   POST   /projects/{id}/eval-sets/{slug}/versions          multipart: file -> new version
 *   POST   /projects/{id}/eval-sets/{slug}/rows/delete       { positions, version? } -> new version
 *   GET    /projects/{id}/eval-sets/{slug}/rows              ?version&offset&limit
 *   POST   /projects/{id}/eval-sets/{slug}/variants          { name, system: strip|replace|add, text? }
 *   DELETE /projects/{id}/eval-sets/{slug}/variants/{name}
 *   POST   /projects/{id}/eval-sets/{slug}/lock
 *   DELETE /projects/{id}/eval-sets/{slug}
 *
 * An eval runs on a set through ``/evaluation/run-heldout`` with ``dataset_name = eval_set:<slug>@v<N>``
 * (+ ``prompt_variant``); the result's ``dataset_name`` is ``eval_set:<slug>@v<N>[+<variant>]``.
 */

import api from './client';
import type { GoldRejects } from './goldFiles';

export const EVAL_SET_PREFIX = 'eval_set:';
export type VariantMode = 'strip' | 'replace' | 'add';
export type SplitSource = 'test' | 'validation' | 'gold_test' | 'gold_dev';

export interface EvalSetVariant {
    name: string;
    spec: { system: VariantMode; text?: string };
    label: string;
    digest?: string | null;
    created_at?: string | null;
}

export interface EvalSetVersion {
    version: number;
    rows: number;
    sha256?: string | null;
    content_sha256?: string | null;
    created_at?: string | null;
    source?: Record<string, unknown> | null;
    dropped_lines?: number[] | null;
}

export interface EvalSetLastEval {
    eval_result_id: number;
    experiment_id: number;
    pass_rate: number | null;
    created_at?: string | null;
    /** The result's rows (matches the version's content hash; the server leaves out any other). */
    content_sha256?: string | null;
}

export interface EvalSet {
    id: number;
    slug: string;
    name: string;
    description?: string;
    current_version: number;
    rows: number;
    is_locked: boolean;
    versions: EvalSetVersion[];
    variants: EvalSetVariant[];
    dataset_name: string | null;
    last_evals?: Record<string, EvalSetLastEval>;
}

export interface EvalSetRowsPage {
    slug: string;
    version: number;
    total: number;
    offset: number;
    limit: number;
    rows: { position: number; row: Record<string, unknown> }[];
    /** Backend-reported (`rcs.reads_report`): which row_context kinds the project's task reads. */
    effective_task_profile?: string | null;
    row_context_reads?: string[];
    row_context_schema_mode?: 'extraction' | 'lead' | null;
}

export async function listEvalSets(projectId: number): Promise<EvalSet[]> {
    const res = await api.get<{ eval_sets?: EvalSet[] }>(`/projects/${projectId}/eval-sets`);
    const sets = res?.data?.eval_sets;
    return Array.isArray(sets) ? sets : [];
}

/** Returned (not thrown) when the file has rows that cannot be used; nothing was written. */
export interface EvalSetImportRejected {
    rejected: GoldRejects;
    usable: number;
}

function rejectedFrom(err: unknown): EvalSetImportRejected | null {
    const data = (err as {
        response?: { data?: { error_code?: string; metadata?: { rejected?: GoldRejects; row_counts?: { usable?: number } } } };
    })?.response?.data;
    if (data?.error_code === 'EVAL_SET_IMPORT_REJECTED_ROWS' && data.metadata?.rejected) {
        return { rejected: data.metadata.rejected, usable: data.metadata.row_counts?.usable ?? 0 };
    }
    return null;
}

/** Non-blocking import warnings, grouped by kind (e.g. `row_context_schema_large`: a row's own output schema
 *  over ~4k tokens is rendered into every prompt). The rows still import. */
export type ImportWarnings = Record<string, { label: string; count: number; lines: number[]; max_tokens?: number }>;

export async function uploadEvalSet(
    projectId: number,
    file: File,
    opts: { name?: string; slug?: string; dropLines?: number[] },
): Promise<{ set: EvalSet; warnings?: ImportWarnings } | EvalSetImportRejected> {
    const form = new FormData();
    form.append('file', file);
    if (!opts.slug) form.append('name', opts.name ?? file.name);
    if (opts.dropLines && opts.dropLines.length > 0) form.append('drop_lines', JSON.stringify(opts.dropLines));
    const url = opts.slug
        ? `/projects/${projectId}/eval-sets/${opts.slug}/versions`
        : `/projects/${projectId}/eval-sets`;
    try {
        const res = await api.post<EvalSet>(url, form);
        return { set: res.data, warnings: (res.data as { warnings?: ImportWarnings }).warnings };
    } catch (err) {
        const rejected = rejectedFrom(err);
        if (rejected) return rejected;
        throw err;
    }
}

export interface VariantInput {
    name: string;
    system: VariantMode;
    text?: string;
}

/** Copy a built-in set or another eval set into a new eval set; ``variant`` also defines one on it. */
export async function createEvalSetFromSource(
    projectId: number,
    name: string,
    source: { kind: 'split'; split: SplitSource } | { kind: 'eval_set'; slug: string; version?: number },
    variant?: VariantInput,
): Promise<EvalSet> {
    const res = await api.post<EvalSet>(`/projects/${projectId}/eval-sets/from-source`, {
        name,
        source,
        ...(variant ? { variant } : {}),
    });
    return res.data;
}

export interface VariantSupport {
    experiment_id: number;
    supported: boolean;
    reason: string | null;
}

/** Whether a prompt-format variant can apply to evals of this run (not for classifier-head,
 *  token-classification, seq2seq or multimodal runs: they render no system message). */
export async function fetchVariantSupport(projectId: number, experimentId: number): Promise<VariantSupport> {
    const res = await api.get<VariantSupport>(`/projects/${projectId}/eval-sets/variant-support`, {
        params: { experiment_id: experimentId },
    });
    return res.data;
}

export async function deleteEvalSetRows(projectId: number, slug: string, positions: number[], version?: number) {
    const res = await api.post<{ deleted: number; eval_set: EvalSet }>(
        `/projects/${projectId}/eval-sets/${slug}/rows/delete`,
        version ? { positions, version } : { positions },
    );
    return res.data;
}

export async function fetchEvalSetRows(projectId: number, slug: string, version: number, offset = 0, limit = 50) {
    const res = await api.get<EvalSetRowsPage>(`/projects/${projectId}/eval-sets/${slug}/rows`, {
        params: { version, offset, limit },
    });
    return res.data;
}

export async function addEvalSetVariant(
    projectId: number,
    slug: string,
    name: string,
    system: VariantMode,
    text?: string,
) {
    const res = await api.post<{ eval_set: EvalSet }>(`/projects/${projectId}/eval-sets/${slug}/variants`, {
        name,
        system,
        ...(system === 'strip' ? {} : { text }),
    });
    return res.data.eval_set;
}

export async function deleteEvalSetVariant(projectId: number, slug: string, name: string) {
    const res = await api.delete<EvalSet>(`/projects/${projectId}/eval-sets/${slug}/variants/${name}`);
    return res.data;
}

export async function lockEvalSet(projectId: number, slug: string) {
    const res = await api.post<EvalSet>(`/projects/${projectId}/eval-sets/${slug}/lock`);
    return res.data;
}

export async function deleteEvalSet(projectId: number, slug: string) {
    await api.delete(`/projects/${projectId}/eval-sets/${slug}`);
}

/** ``eval_set:<slug>@v<N>`` (+ ``+<variant>``). */
export function evalSetDatasetName(slug: string, version: number, variant?: string | null): string {
    return `${EVAL_SET_PREFIX}${slug}@v${version}${variant ? `+${variant}` : ''}`;
}

export function isEvalSetName(name: string | null | undefined): boolean {
    return typeof name === 'string' && name.startsWith(EVAL_SET_PREFIX);
}

/** "eval set fresh v1 (native)" for an eval-set dataset name; the name itself otherwise. */
export function evalDatasetLabel(name: string | null | undefined): string {
    if (!isEvalSetName(name)) return name ?? '';
    const rest = (name as string).slice(EVAL_SET_PREFIX.length);
    const [setPart, variant] = rest.split('+', 2);
    const [slug, ver] = setPart.split('@', 2);
    return `eval set ${slug}${ver ? ` ${ver}` : ''}${variant ? ` (${variant})` : ''}`;
}

interface ResultLike {
    dataset_name: string;
    metrics?: object | null;
    details?: object | null;
}

/** Picked-data provenance for one result: set, version, variant, rows, content hash, prompt format. */
export function evalResultDatasetLabel(result: ResultLike): string {
    if (!isEvalSetName(result.dataset_name)) return result.dataset_name;
    const details = (result.details ?? null) as Record<string, unknown> | null;
    const metrics = (result.metrics ?? null) as Record<string, unknown> | null;
    const dataset = (details?.dataset ?? null) as Record<string, unknown> | null;
    const block = (dataset?.eval_set ?? null) as Record<string, unknown> | null;
    const parts = [evalDatasetLabel(result.dataset_name)];
    if (block?.name) parts[0] = `${parts[0]} "${String(block.name)}"`;
    if (typeof block?.rows === 'number') parts.push(`${block.rows} rows`);
    const content = typeof dataset?.content_sha256 === 'string' ? dataset.content_sha256 : null;
    if (content) parts.push(`content ${content.slice(0, 8)}`);
    const fmt = metrics?.prompt_format_version;
    parts.push(`prompt format v${typeof fmt === 'number' ? fmt : 1}`);
    return parts.join(' · ');
}

/**
 * The newest result on a built-in set (decision D3): "latest eval" readers never pick an eval-set
 * result. ``results`` newest first, as the results list is.
 */
export function latestBuiltInResult<T extends { dataset_name: string | null }>(results: T[]): T | null {
    return results.find((r) => !isEvalSetName(r.dataset_name)) ?? null;
}

/** Every eval-set label that has at least one result (for Compare / lift pickers). */
export function evaluatedSetLabels(sets: EvalSet[]): string[] {
    const out: string[] = [];
    for (const s of sets) {
        for (const label of Object.keys(s.last_evals ?? {}).sort()) out.push(label);
    }
    return out;
}

export function evalSetErrorMessage(err: unknown, fallback = 'The eval-set change failed.'): string {
    const data = (err as { response?: { data?: { message?: string; detail?: unknown } } })?.response?.data;
    if (typeof data?.message === 'string') return data.message;
    if (typeof data?.detail === 'string') return data.detail;
    return fallback;
}

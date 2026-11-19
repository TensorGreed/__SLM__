/**
 * Typed client for named, versioned eval sets + prompt-format variants (every task shape).
 *
 *   GET    /projects/{id}/eval-sets                          list (versions, variants, last eval per label)
 *   POST   /projects/{id}/eval-sets                          multipart: name, file, drop_lines, keep_leak_lines -> new set
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
    /** Rows the user kept although the leak scan flagged them (scored as seen data). */
    kept_despite_leak?: KeptDespiteLeak | null;
}

/** Rows of an eval-set version kept on purpose although they leak training data: their count, 0-based
 *  positions, ids (the row's own `id`, else `#<position>`) and, for an upload, the file lines. */
export interface KeptDespiteLeak {
    count: number;
    row_indexes?: number[];
    row_ids?: string[];
    lines?: number[];
}

/** The reject reason of a row that leaks training data: the only one an upload may keep. */
export const LEAK_REASON = 'leaks_training_data';

export interface EvalSetLastEval {
    eval_result_id: number;
    experiment_id: number;
    pass_rate: number | null;
    created_at?: string | null;
    /** The result's rows (matches the version's content hash; the server leaves out any other). */
    content_sha256?: string | null;
    /** True when this newest result is a baseline run's (the base model) — no trained run has one. */
    baseline?: boolean;
    /** The newest base-model (baseline run) result on this label, when there is one. */
    base?: { eval_result_id: number; experiment_id: number; pass_rate: number | null; base_model?: string | null } | null;
}

/** "last result #265 (run #214) · pass 87.7% · base model Qwen/X: pass 78.5% (#263)" — which model
 *  answered each side, so two results on one label are never told apart by id alone. */
export function lastEvalText(ev: EvalSetLastEval): string {
    const pct = (v: number | null | undefined) => (v != null ? `${(v * 100).toFixed(1)}%` : '—');
    const who = ev.baseline ? `base model, run #${ev.experiment_id}` : `run #${ev.experiment_id}`;
    let text = `last result #${ev.eval_result_id} (${who})${ev.pass_rate != null ? ` · pass ${pct(ev.pass_rate)}` : ''}`;
    if (ev.base && !ev.baseline) {
        text += ` · base model${ev.base.base_model ? ` ${ev.base.base_model}` : ''}: pass ${pct(ev.base.pass_rate)} (#${ev.base.eval_result_id})`;
    }
    return text;
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
    opts: { name?: string; slug?: string; dropLines?: number[]; keepLeakLines?: number[] },
): Promise<{ set: EvalSet; warnings?: ImportWarnings } | EvalSetImportRejected> {
    const form = new FormData();
    form.append('file', file);
    if (!opts.slug) form.append('name', opts.name ?? file.name);
    if (opts.dropLines && opts.dropLines.length > 0) form.append('drop_lines', JSON.stringify(opts.dropLines));
    if (opts.keepLeakLines && opts.keepLeakLines.length > 0) {
        form.append('keep_leak_lines', JSON.stringify(opts.keepLeakLines));
    }
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
/** "answered by the base model Qwen/X" / "answered by model path /m" for a result evaluated with an
 *  explicit model path (``metrics.evaluated_model``); '' otherwise (the run's own weights answered). */
export function evaluatedModelText(metrics: Record<string, unknown> | null | undefined): string {
    const record = (metrics?.evaluated_model ?? null) as {
        model_path?: string;
        kind?: string;
        generation?: { max_new_tokens?: number; temperature?: number; decoding?: string };
    } | null;
    if (!record || !record.model_path) return '';
    const text = record.kind === 'base' ? `answered by the base model ${record.model_path}` : `answered by model path ${record.model_path}`;
    const gen = record.generation;
    if (!gen || typeof gen.max_new_tokens !== 'number') return text;
    const temp = typeof gen.temperature === 'number' ? `, temperature ${gen.temperature}` : '';
    const dec = gen.decoding ? `, ${gen.decoding} decoding` : '';
    return `${text} (max ${gen.max_new_tokens} new tokens${temp}${dec})`;
}

export function evalResultDatasetLabel(result: ResultLike): string {
    const metrics = (result.metrics ?? null) as Record<string, unknown> | null;
    const answeredBy = evaluatedModelText(metrics);
    if (!isEvalSetName(result.dataset_name)) return answeredBy ? `${result.dataset_name} · ${answeredBy}` : result.dataset_name;
    const details = (result.details ?? null) as Record<string, unknown> | null;
    const dataset = (details?.dataset ?? null) as Record<string, unknown> | null;
    const block = (dataset?.eval_set ?? null) as Record<string, unknown> | null;
    const parts = [evalDatasetLabel(result.dataset_name)];
    if (block?.name) parts[0] = `${parts[0]} "${String(block.name)}"`;
    if (typeof block?.rows === 'number') parts.push(`${block.rows} rows`);
    const content = typeof dataset?.content_sha256 === 'string' ? dataset.content_sha256 : null;
    if (content) parts.push(`content ${content.slice(0, 8)}`);
    const fmt = metrics?.prompt_format_version;
    parts.push(`prompt format v${typeof fmt === 'number' ? fmt : 1}`);
    const kept = keptLeakText(block?.kept_despite_leak as KeptDespiteLeak | null | undefined);
    if (kept) parts.push(kept);
    if (answeredBy) parts.push(answeredBy);
    return parts.join(' · ');
}

/** "2 rows kept despite a training-data leak flag (q-7, #12)", or '' when none were kept. */
export function keptLeakText(kept: KeptDespiteLeak | null | undefined): string {
    if (!kept || !kept.count) return '';
    const ids = (kept.row_ids ?? []).slice(0, 5);
    const more = (kept.row_ids?.length ?? 0) > ids.length ? ', …' : '';
    return `${kept.count} row${kept.count === 1 ? '' : 's'} kept despite a training-data leak flag`
        + (ids.length ? ` (${ids.join(', ')}${more})` : '');
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

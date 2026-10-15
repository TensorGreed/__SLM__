/** External benchmarks (Epic G-B1): a labelled set someone else made, mapped
 *  onto the project's labels, eval only. See external_benchmark_service. */
import api from './client';

export type BenchmarkHeadline = 'accuracy' | 'f1' | 'exact_match' | 'judge_correct';

/** The answer judge on a generative benchmark (Q&A, summarization). */
export interface BenchmarkJudge {
    judge?: string;
    score?: number | null;
    judged?: number;
    counts?: { correct: number; partial: number; wrong: number };
    /** not_long_answer_task / no_judge_available / local_only_cloud_judge / error: … */
    skipped?: string;
    message?: string;
}

export interface BenchmarkResult {
    experiment_id: number;
    experiment_name?: string;
    evaluated_at: string;
    rows: number;
    headline: BenchmarkHeadline;
    score: number | null;
    exact_match?: number | null;
    judge_correct?: number | null;
    judge?: BenchmarkJudge | null;
    precision?: number | null;
    recall?: number | null;
    f1?: number | null;
    accuracy?: number | null;
    macro_f1?: number | null;
    per_type?: { type: string; precision: number | null; recall: number | null; f1: number | null; support: number }[];
    failures?: { prompt: string; reference: string; prediction: string }[];
}

export type BenchmarkTaskProfile = 'classification' | 'structured_extraction' | 'qa' | 'summarization';
export type BenchmarkMappingValue = string | string[];

export interface BenchmarkHistoryPoint {
    experiment_id: number;
    experiment_name?: string | null;
    evaluated_at: string;
    headline: BenchmarkHeadline;
    score: number | null;
    rows?: number;
}

export interface BenchmarkInfo {
    id: number;
    name: string;
    slug: string;
    task_profile: BenchmarkTaskProfile;
    source: { locator: string; source_id: string; dataset_id: string; split: string | null; revision: string | null };
    licence: string | null;
    local_only: boolean;
    rows: number;
    /** Source label → project label | a group of project labels | "ignore". Empty for Q&A / summarization. */
    mapping: Record<string, BenchmarkMappingValue>;
    /** Labels scored; a group reads "a|b". */
    scored_labels: string[];
    /** Group name → its project labels (one-to-many mappings). */
    groups?: Record<string, string[]>;
    /** Q&A context folded into the question (the project doesn't answer from a context). */
    context_folded?: boolean;
    mapping_summary?: { rows_in: number; rows_kept: number; rows_dropped_by_label: Record<string, number>; entities_ignored: Record<string, number> };
    leakage?: { checked_against: number; exact: number; near: number };
    created_at: string;
    /** This run's latest score (null: not scored yet). */
    result: BenchmarkResult | null;
    latest_result: BenchmarkResult | null;
}

export interface BenchmarkPreview {
    source: BenchmarkInfo['source'];
    licence: string | null;
    task_profile: BenchmarkInfo['task_profile'];
    format: string;
    fields: Record<string, string>;
    project: { task_profile: string | null; labels: string[] };
    shape_matches_project: boolean;
    /** False for Q&A / summarization: no labels to map. */
    labels_to_map?: boolean;
    source_labels: Record<string, number>;
    suggested_mapping: Record<string, string>;
    sample: Record<string, unknown>[];
    rows_scanned: number;
}

export async function listBenchmarks(projectId: number, experimentId: number | null): Promise<BenchmarkInfo[]> {
    const res = await api.get<{ benchmarks: BenchmarkInfo[] }>(`/projects/${projectId}/benchmarks`, {
        params: experimentId != null ? { experiment_id: experimentId } : {},
    });
    return res.data?.benchmarks ?? [];
}

export async function previewBenchmark(projectId: number, locator: string, limit: number): Promise<BenchmarkPreview> {
    const res = await api.post<BenchmarkPreview>(`/projects/${projectId}/benchmarks/preview`, { locator, limit });
    return res.data;
}

export async function createBenchmark(projectId: number, body: {
    locator: string; name: string; limit: number; mapping: Record<string, BenchmarkMappingValue>;
    local_only: boolean; licence?: string | null; join_adjacent: boolean;
}): Promise<BenchmarkInfo> {
    const res = await api.post<BenchmarkInfo>(`/projects/${projectId}/benchmarks`, body);
    return res.data;
}

export async function evaluateBenchmark(projectId: number, benchmarkId: number, experimentId: number | null): Promise<{ id: number }> {
    const res = await api.post<{ id: number }>(`/projects/${projectId}/benchmarks/${benchmarkId}/evaluate`, {
        experiment_id: experimentId,
    });
    return res.data;
}

export async function deleteBenchmark(projectId: number, benchmarkId: number): Promise<void> {
    await api.delete(`/projects/${projectId}/benchmarks/${benchmarkId}`);
}

export async function getBenchmarkHistory(projectId: number, benchmarkId: number): Promise<BenchmarkHistoryPoint[]> {
    const res = await api.get<{ history: BenchmarkHistoryPoint[] }>(`/projects/${projectId}/benchmarks/${benchmarkId}/history`);
    return Array.isArray(res.data?.history) ? res.data.history : [];
}

/** Display name of a benchmark headline metric. */
export function benchmarkHeadlineLabel(headline: string | undefined | null): string {
    switch (headline) {
        case 'f1': return 'F1';
        case 'exact_match': return 'exact match';
        case 'judge_correct': return 'judged correct';
        default: return 'accuracy';
    }
}

/** Why a generative benchmark's answers were not judged (null: judged, or nothing worth saying). */
export function benchmarkJudgeNote(result: BenchmarkResult | null): string | null {
    const skipped = result?.judge?.skipped;
    if (skipped === 'local_only_cloud_judge') return `not judged: local-only benchmark, cloud judge (${result?.judge?.judge ?? 'cloud'})`;
    if (skipped === 'no_judge_available') return 'no answer judge available';
    return null;
}

/** "accuracy 0.91" / "F1 0.78" / "judged correct 0.62" — the benchmark's headline for one result. */
export function benchmarkScoreText(result: BenchmarkResult | null): string {
    if (!result || result.score == null) return 'not scored yet';
    const note = benchmarkJudgeNote(result);
    return `${benchmarkHeadlineLabel(result.headline)} ${result.score.toFixed(2)} on ${result.rows} rows${note ? ` · ${note}` : ''}`;
}

/** "date_of_birth|visit_date" → "date_of_birth + visit_date". */
export function benchmarkLabelText(label: string): string {
    return label.split('|').join(' + ');
}

/** Benchmark label → its project labels, from a mapping value ("ignore" → []). */
export function mappingMembers(value: BenchmarkMappingValue | undefined): string[] {
    if (Array.isArray(value)) return value.filter((v) => v && v !== 'ignore');
    return value && value !== 'ignore' ? [value] : [];
}

/** A project label picked in two different groups — the backend refuses it. Null when fine. */
export function mappingGroupConflict(mapping: Record<string, BenchmarkMappingValue>): string | null {
    const groups = new Set<string>();
    for (const value of Object.values(mapping)) {
        const members = [...new Set(mappingMembers(value))].sort();
        if (members.length) groups.add(members.join('|'));
    }
    const owners: Record<string, string[]> = {};
    for (const g of groups) for (const label of g.split('|')) (owners[label] ??= []).push(g);
    const clash = Object.entries(owners).find(([, gs]) => gs.length > 1);
    return clash ? `“${clash[0]}” is in two groups (${clash[1].map(benchmarkLabelText).join('; ')}) — a label can belong to one group only.` : null;
}

/** Where the benchmark came from, for provenance lines. */
export function benchmarkSourceText(b: BenchmarkInfo): string {
    const src = b.source?.dataset_id || b.source?.locator || '';
    const split = b.source?.split ? `:${b.source.split}` : '';
    return `${src}${split} · licence ${b.licence || 'unknown'}${b.local_only ? ' · local only' : ''}`;
}

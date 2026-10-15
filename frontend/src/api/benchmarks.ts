/** External benchmarks (Epic G-B1): a labelled set someone else made, mapped
 *  onto the project's labels, eval only. See external_benchmark_service. */
import api from './client';

export interface BenchmarkResult {
    experiment_id: number;
    experiment_name?: string;
    evaluated_at: string;
    rows: number;
    headline: 'accuracy' | 'f1';
    score: number | null;
    precision?: number | null;
    recall?: number | null;
    f1?: number | null;
    accuracy?: number | null;
    macro_f1?: number | null;
    per_type?: { type: string; precision: number | null; recall: number | null; f1: number | null; support: number }[];
    failures?: { prompt: string; reference: string; prediction: string }[];
}

export interface BenchmarkInfo {
    id: number;
    name: string;
    slug: string;
    task_profile: 'classification' | 'structured_extraction';
    source: { locator: string; source_id: string; dataset_id: string; split: string | null; revision: string | null };
    licence: string | null;
    local_only: boolean;
    rows: number;
    mapping: Record<string, string>;
    scored_labels: string[];
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
    locator: string; name: string; limit: number; mapping: Record<string, string>;
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

/** "accuracy 0.91" / "F1 0.78" — the benchmark's headline for one result. */
export function benchmarkScoreText(result: BenchmarkResult | null): string {
    if (!result || result.score == null) return 'not scored yet';
    return `${result.headline === 'f1' ? 'F1' : 'accuracy'} ${result.score.toFixed(2)} on ${result.rows} rows`;
}

/** Where the benchmark came from, for provenance lines. */
export function benchmarkSourceText(b: BenchmarkInfo): string {
    const src = b.source?.dataset_id || b.source?.locator || '';
    const split = b.source?.split ? `:${b.source.split}` : '';
    return `${src}${split} · licence ${b.licence || 'unknown'}${b.local_only ? ' · local only' : ''}`;
}

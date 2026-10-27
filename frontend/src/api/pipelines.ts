/**
 * Pipeline evaluation client (G10): multi-stage pipelines that chain any
 * projects, a shared held-out set, and runs scored per stage in two views.
 * Shapes mirror the backend's files-on-disk pipeline / run JSON.
 */

import api from './client';
import type { Job } from './jobs';
import { parseGatewayError, type FieldError } from './gateway';

export type { FieldError };
export const parsePipelineError = parseGatewayError;

export type InputFrom = 'row' | 'previous' | 'item';

export interface StageInput { from: InputFrom; pointer: string }

export interface StagePrompt { mode: 'recipe' | 'template'; template?: string; system?: string }

export interface StageFanOut { pointer: string; item_key: string[] }

export interface PipelineStage {
    name: string;
    project_id: number;
    /** "" (project default), ":base" or ":run-<id>". */
    model: string;
    prompt: StagePrompt;
    inputs: Record<string, StageInput>;
    fan_out?: StageFanOut | null;
    retrieval?: boolean | null;
    max_tokens?: number;
    output: 'json' | 'text';
}

export interface PipelineDefinition {
    name: string;
    description: string;
    stages: PipelineStage[];
}

export interface Pipeline extends PipelineDefinition {
    id: string;
    created_at?: string;
    updated_at?: string;
}

/** Run status on disk: "stale" = still "running" but its Job ended or is gone. */
export type RunStatus = 'running' | 'done' | 'failed' | 'cancelled' | 'stale';

export interface PipelineSummary {
    id: string;
    name: string;
    /** `project_missing`: the member project was deleted after the pipeline was saved. */
    stages: { name: string; project_id: number; project_name: string | null; project_missing?: boolean }[];
    heldout_rows: number;
    last_run: {
        id: string; started_at: string | null; share_fully_correct: number | null;
        status?: RunStatus; error?: string | null;
    } | null;
}

export interface HeldoutUploadResult {
    accepted: number;
    rejected_by_reason: Record<string, { line: number; id?: string | null }[]>;
}

export interface HeldoutPreview {
    rows: { id: string; input: Record<string, unknown>; gold: Record<string, unknown> }[];
    total?: number;
}

export interface RunListItem {
    id: string;
    status?: RunStatus;
    error?: string | null;
    started_at: string | null;
    rows: number | null;
    share_fully_correct: number | null;
    truncated: boolean;
}

export interface ViewMetrics {
    scored: number;
    correct: number;
    accuracy: number | null;
    /** Not scored: no gold, or (with a reason) the scoring context / judge / scorer was missing. */
    unscored: number;
    unscored_by_reason?: Record<string, number>;
    spurious: number;
    missed_upstream: number;
    lost_to_error?: number;
    missing_item_key?: number;
    duplicate_item_key?: number;
    errors: number;
    /** Isolated view only: rows without gold for this stage or the one before. */
    skipped_rows?: number;
}

export interface StageJudge {
    kind: 'answer' | 'rubric' | string;
    label: string | null;
    ran: boolean;
    units: number;
    reason?: string;
}

export interface RunStage {
    name: string;
    project_id: number;
    project_name: string;
    model_id: string;
    run_id: number | null;
    scoring_mode?: string | null;
    handler?: string | null;
    scoring_error?: string | null;
    judge?: StageJudge | null;
    judge_not_run?: boolean;
    /** The stage producing the fan-out items: how its items were checked against gold. */
    item_compare?: 'deep_equal' | 'key_only' | 'mixed';
    item_compare_counts?: { deep_equal: number; key_only: number };
    end_to_end: ViewMetrics;
    isolated: ViewMetrics | null;
}

export interface RunRowOutput {
    item_key_values: unknown;
    correct: boolean | null;
    unscored_reason?: string | null;
    error: string | null;
}

export interface RunRowDetail {
    id: string;
    stages: { name: string; inputs_count: number; outputs: RunRowOutput[] }[];
    final: {
        gold_item_key: unknown; fully_correct: boolean | null; first_wrong_stage: string | null; unscored_reason?: string;
    }[];
}

/** A run file. While running, or after a failure / cancel, only the first fields are present. */
export interface PipelineRun {
    run_id: string;
    pipeline_id: string;
    status?: RunStatus;
    error?: string | null;
    started_at: string | null;
    finished_at?: string | null;
    rows?: number;
    truncated?: boolean;
    truncated_at_row?: string | null;
    truncated_message?: string;
    gold_mismatch?: Record<string, string[]>;
    calls?: number;
    errors_by_reason?: Record<string, number>;
    stages?: RunStage[];
    end_to_end?: {
        gold_final_items: number; fully_correct: number; share_fully_correct: number | null;
        unscored_items?: number; unscored_by_reason?: Record<string, number>;
    };
    attribution?: {
        first_wrong_stage: Record<string, number>;
        correct: number;
        missed_upstream_at: Record<string, number>;
    };
    rows_detail?: RunRowDetail[];
}

/** A run that finished with results (status done and stage blocks present). */
export type FinishedPipelineRun = PipelineRun & Required<Pick<PipelineRun, 'stages' | 'end_to_end' | 'attribution' | 'rows_detail'>>;

export function isFinishedRun(run: PipelineRun): run is FinishedPipelineRun {
    return (run.status ?? 'done') === 'done' && Array.isArray(run.stages) && !!run.end_to_end && !!run.attribution;
}

/** Plain words for the reasons a unit or item was left unscored. */
export function unscoredReasonLabel(reason: string): string {
    const words: Record<string, string> = {
        no_gold: 'no gold',
        row_context_unavailable: 'scoring context missing',
        needs_judge: 'needs a judge',
        scoring_error: 'scorer failed',
    };
    return words[reason] ?? reason.replace(/_/g, ' ');
}

export function runStatusLabel(status: RunStatus | undefined): string {
    const words: Record<RunStatus, string> = {
        running: 'running', done: 'done', failed: 'failed', cancelled: 'cancelled', stale: 'stale (stopped without a result)',
    };
    return words[status ?? 'done'];
}

export async function listPipelines(): Promise<PipelineSummary[]> {
    const res = await api.get<{ pipelines: PipelineSummary[] }>('/pipelines');
    return res.data.pipelines ?? [];
}

export async function getPipeline(id: string): Promise<Pipeline> {
    return (await api.get<Pipeline>(`/pipelines/${id}`)).data;
}

export async function createPipeline(body: PipelineDefinition): Promise<Pipeline> {
    return (await api.post<Pipeline>('/pipelines', body)).data;
}

export async function updatePipeline(id: string, body: PipelineDefinition): Promise<Pipeline> {
    return (await api.put<Pipeline>(`/pipelines/${id}`, body)).data;
}

export async function deletePipeline(id: string): Promise<void> {
    await api.delete(`/pipelines/${id}`);
}

export async function uploadHeldout(id: string, file: File): Promise<HeldoutUploadResult> {
    const form = new FormData();
    form.append('file', file);
    return (await api.put<HeldoutUploadResult>(`/pipelines/${id}/heldout`, form)).data;
}

export async function getHeldout(id: string, limit = 50): Promise<HeldoutPreview> {
    return (await api.get<HeldoutPreview>(`/pipelines/${id}/heldout`, { params: { limit } })).data;
}

export async function startPipelineRun(id: string, maxRows?: number): Promise<Job> {
    const body = maxRows ? { max_rows: maxRows } : {};
    return (await api.post<Job>(`/pipelines/${id}/runs?async_job=true`, body)).data;
}

export async function listPipelineRuns(id: string): Promise<RunListItem[]> {
    const res = await api.get<{ runs: RunListItem[] }>(`/pipelines/${id}/runs`);
    return res.data.runs ?? [];
}

export async function getPipelineRun(id: string, runId: string): Promise<PipelineRun> {
    return (await api.get<PipelineRun>(`/pipelines/${id}/runs/${runId}`)).data;
}

export async function listProjectPipelines(projectId: number): Promise<{ id: string; name: string }[]> {
    const res = await api.get<{ pipelines: { id: string; name: string }[] }>(`/projects/${projectId}/pipelines`);
    return res.data.pipelines ?? [];
}

export function pipelinePath(id: string, runId?: string): string {
    return runId ? `/pipelines/${id}?run=${encodeURIComponent(runId)}` : `/pipelines/${id}`;
}

export function percent(x: number | null | undefined): string {
    return typeof x === 'number' ? `${Math.round(x * 100)}%` : 'n/a';
}

/** Stage name with the most first-wrong counts, or null when none. */
export function topFirstWrongStage(counts: Record<string, number> | undefined): string | null {
    let best: string | null = null;
    let n = 0;
    for (const [stage, c] of Object.entries(counts ?? {})) {
        if (c > n) { best = stage; n = c; }
    }
    return best;
}

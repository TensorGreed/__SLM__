/**
 * Typed API wrapper for the synth playbook framework
 * (USER-SUCCESS Epic 2).
 */

import api from './client';

export type SynthMode =
    | 'positives_paraphrase'
    | 'hard_negatives'
    | 'class_balance_fill'
    | 'edge_cases'
    | 'refusals'
    | 'format_robustness'
    | 'cluster_targeted';

export interface PlaybookCatalogEntry {
    recipe_id: string;
    mode: SynthMode;
}

export interface PlaybookCatalogResponse {
    project_id: number;
    recipe_id: string | null;
    /** True when the project has no `selected_recipe` set yet —
     *  legacy projects pre-dating the auto-apply-on-create fix.
     *  Optional for backwards-compat with mocks that pre-date the
     *  flag; treat missing as ``false``. */
    recipe_required?: boolean;
    playbooks: PlaybookCatalogEntry[];
}

export interface SynthRow {
    payload: Record<string, unknown>;
    synth_confidence: number;
    synth_source: string;
}

export interface PlaybookResult {
    rows: SynthRow[];
    backend_used: string;
    elapsed_sec: number;
    prompt_snippet: string;
}

export async function listPlaybooks(projectId: number): Promise<PlaybookCatalogResponse> {
    const resp = await api.get(`/projects/${projectId}/synthetic/playbooks`);
    return resp.data as PlaybookCatalogResponse;
}


// ─────────────────────────────────────────────────────────────────────
// USER-SUCCESS Epic 5 Phase 5a — synth backend picker.
// ─────────────────────────────────────────────────────────────────────

export interface SynthBackendInfo {
    name: string;
    available: boolean;
    describe: string;
    /** Phase 5c — True for backends that forward the playbook's
     *  response_schema as response_format=json_schema (NeMo, vLLM).
     *  False for backends that silently ignore it (Ollama, Teacher). */
    schema_aware?: boolean;
}

export interface SynthBackendsResponse {
    project_id: number;
    backends: SynthBackendInfo[];
}

export async function listSynthBackends(
    projectId: number,
): Promise<SynthBackendsResponse> {
    const resp = await api.get(`/projects/${projectId}/synthetic/backends`);
    return resp.data as SynthBackendsResponse;
}

export interface RunPlaybookArgs {
    mode: SynthMode;
    targetCount: number;
    targetClass?: string | null;
    backend?: string | null;
}

export async function runPlaybook(
    projectId: number,
    args: RunPlaybookArgs,
): Promise<PlaybookResult> {
    const resp = await api.post(`/projects/${projectId}/synthetic/run-playbook`, {
        mode: args.mode,
        target_count: args.targetCount,
        target_class: args.targetClass ?? null,
        backend: args.backend ?? null,
    });
    return resp.data as PlaybookResult;
}


// ─────────────────────────────────────────────────────────────────────
// Pre-flight dry-run (P2). The picker calls this BEFORE kicking the
// async job so a refusal or empty-output failure surfaces inline in
// <10s, not after the user waits 60-180s for the real job to fail.
// ─────────────────────────────────────────────────────────────────────

export interface DryRunPlaybookResult {
    /** Convenience flag: true only when accepted_count >= 1 AND
     *  refusal_detected is false. The panel uses this as a single
     *  go/no-go signal before kicking the real run. */
    ok: boolean;
    accepted_count: number;
    /** Heuristic: the LLM returned a short non-JSON apology
     *  ("I cannot generate malicious examples", etc.). When true,
     *  the panel surfaces a "Retry with a less-restricted model"
     *  affordance with Qwen 2.5 as the suggested fallback. */
    refusal_detected: boolean;
    /** First ~280 chars of the model's raw response — so the user
     *  can see exactly what came back, not just "0 rows accepted". */
    raw_llm_snippet: string;
    backend_used: string;
    elapsed_sec: number;
    prompt_snippet: string;
    /** Set when the backend was unavailable (e.g. Ollama daemon
     *  down). The panel renders this as a "no backend installed"
     *  affordance. */
    error?: string;
    rows: SynthRow[];
}

export async function dryRunPlaybook(
    projectId: number,
    args: RunPlaybookArgs,
): Promise<DryRunPlaybookResult> {
    const resp = await api.post(
        `/projects/${projectId}/synthetic/run-playbook/dry-run`,
        {
            mode: args.mode,
            target_count: args.targetCount,
            target_class: args.targetClass ?? null,
            backend: args.backend ?? null,
        },
    );
    return resp.data as DryRunPlaybookResult;
}


// ─────────────────────────────────────────────────────────────────────
// Ollama-models list (P3). Powers the model-picker dropdown.
// ─────────────────────────────────────────────────────────────────────

export interface OllamaModelInfo {
    name: string;
    size_bytes: number;
    parameter_size: string;
    family: string;
}

export interface OllamaModelsResponse {
    project_id: number;
    /** Auto-pick tag the platform would default to given the
     *  installed models + the backend's PREFERRED_MODEL_PATTERNS. */
    default: string | null;
    models: OllamaModelInfo[];
    ollama_available: boolean;
    error?: string;
}

export async function listOllamaModels(
    projectId: number,
): Promise<OllamaModelsResponse> {
    const resp = await api.get(
        `/projects/${projectId}/synthetic/backends/ollama/models`,
    );
    return resp.data as OllamaModelsResponse;
}


// Hardening Phase H1 — async-job variant. Returns the Job stub (202).
// Caller starts polling via useJobsStore and shows progress in the
// top-bar notification bell instead of blocking on the LLM call.
import type { Job } from './jobs';

export async function runPlaybookAsync(
    projectId: number,
    args: RunPlaybookArgs,
): Promise<Job> {
    const resp = await api.post(
        `/projects/${projectId}/synthetic/run-playbook?async_job=true`,
        {
            mode: args.mode,
            target_count: args.targetCount,
            target_class: args.targetClass ?? null,
            backend: args.backend ?? null,
        },
    );
    return resp.data as Job;
}


// ─────────────────────────────────────────────────────────────────────
// USER-SUCCESS Epic 2b — cluster-augment + review queue.
// ─────────────────────────────────────────────────────────────────────

export interface AugmentFromClusterArgs {
    evalResultId: number;
    clusterId: string;
    targetCount?: number;
    backend?: string | null;
}

export async function augmentFromCluster(
    projectId: number,
    args: AugmentFromClusterArgs,
): Promise<PlaybookResult> {
    const params: Record<string, unknown> = {
        target_count: args.targetCount ?? 30,
    };
    if (args.backend) {
        params.backend = args.backend;
    }
    const resp = await api.post(
        `/projects/${projectId}/evaluation/${args.evalResultId}/clusters/${args.clusterId}/augment`,
        null,
        { params },
    );
    return resp.data as PlaybookResult;
}


// Hardening — async-job variant. Returns the Job stub (202).
// Caller starts polling via useJobsStore and shows progress in
// the top-bar notification bell instead of blocking on the LLM
// call for 30-180s.
export async function augmentFromClusterAsync(
    projectId: number,
    args: AugmentFromClusterArgs,
): Promise<Job> {
    const params: Record<string, unknown> = {
        target_count: args.targetCount ?? 30,
        async_job: true,
    };
    if (args.backend) {
        params.backend = args.backend;
    }
    const resp = await api.post(
        `/projects/${projectId}/evaluation/${args.evalResultId}/clusters/${args.clusterId}/augment`,
        null,
        { params },
    );
    return resp.data as Job;
}


export interface ReviewQueueEntry {
    id: number;
    synth_confidence: number;
    preview: string;
    payload: Record<string, unknown>;
}

export interface ReviewQueueGroup {
    synth_source: string;
    count: number;
    /** True when `rows` is a truncated sample of the group (legacy
     *  buckets with thousands of rows hit the cap). */
    truncated?: boolean;
    rows: ReviewQueueEntry[];
}

export interface ReviewQueueResponse {
    project_id: number;
    dataset_id: number | null;
    /** Total rows in synthetic.jsonl regardless of review_status. */
    total_rows: number;
    total_pending: number;
    total_accepted: number;
    groups: ReviewQueueGroup[];
    /** Accepted rows (passed review or pre-Epic-2a legacy rows).
     *  Surfaces what's queued for the next dataset prep. */
    accepted_groups: ReviewQueueGroup[];
}

export interface BulkUpdateResult {
    accepted: number;
    rejected: number;
    not_found: number;
    not_pending: number;
    total_remaining_pending: number;
}

export async function listSynthReviewQueue(projectId: number): Promise<ReviewQueueResponse> {
    const resp = await api.get(`/projects/${projectId}/synthetic/review-queue`);
    return resp.data as ReviewQueueResponse;
}

export async function bulkUpdateSynthReviewQueue(
    projectId: number,
    args: { rowIds: number[]; action: 'accept' | 'reject' },
): Promise<BulkUpdateResult> {
    const resp = await api.post(`/projects/${projectId}/synthetic/review-queue/bulk-update`, {
        row_ids: args.rowIds,
        action: args.action,
    });
    return resp.data as BulkUpdateResult;
}

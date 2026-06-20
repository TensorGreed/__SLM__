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


// ─────────────────────────────────────────────────────────────────────
// Cloud-LLM catalog (OpenAI / Anthropic / Deepseek). The platform
// ships the same backend wrapper for all three; the only per-project
// state is whether an API key is saved under the relevant secret.
// ─────────────────────────────────────────────────────────────────────

export interface CloudModelEntry {
    id: string;
    label: string;
}

export interface CloudProviderEntry {
    provider: 'openai' | 'anthropic' | 'deepseek';
    /** True when a key is saved at ``cloud_llm_<provider>:api_key``
     *  on this project. The picker uses this to badge providers with
     *  a green check + enable the model dropdown. Providers without
     *  a saved key show a 'Save key first' affordance instead. */
    key_saved: boolean;
    models: CloudModelEntry[];
}

export interface CloudModelsResponse {
    project_id: number;
    providers: CloudProviderEntry[];
}

export async function listCloudModels(
    projectId: number,
): Promise<CloudModelsResponse> {
    const resp = await api.get(
        `/projects/${projectId}/synthetic/backends/cloud/models`,
    );
    return resp.data as CloudModelsResponse;
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
    /** Arc 5 — soft-reject. Rejected rows stay on disk with
     *  ``review_status="rejected"`` so they can be reviewed and
     *  bulk-purged by reason. */
    total_rejected?: number;
    groups: ReviewQueueGroup[];
    /** Accepted rows (passed review or pre-Epic-2a legacy rows).
     *  Surfaces what's queued for the next dataset prep. */
    accepted_groups: ReviewQueueGroup[];
    /** Arc 5 — rejected rows grouped by ``synth_source``. The
     *  per-row ``payload`` carries ``reject_reason`` when set by
     *  the bulk-update endpoint. */
    rejected_groups?: ReviewQueueGroup[];
}


export interface PurgeRejectedResult {
    purged: number;
    retained: number;
    total_rows: number;
}

export interface BulkUpdateResult {
    accepted: number;
    rejected: number;
    not_found: number;
    not_pending: number;
    total_remaining_pending: number;
}

export interface BulkUpdateBySourceResult extends BulkUpdateResult {
    /** Echo of the targeted synth_source group key. */
    source: string;
    /** Pending rows in the group at call time. */
    matched: number;
}

export async function listSynthReviewQueue(
    projectId: number,
    groupBy: 'source' | 'class' = 'source',
): Promise<ReviewQueueResponse> {
    const resp = await api.get(`/projects/${projectId}/synthetic/review-queue`, {
        params: groupBy === 'class' ? { group_by: 'class' } : undefined,
    });
    return resp.data as ReviewQueueResponse;
}

export async function bulkUpdateSynthReviewQueue(
    projectId: number,
    args: {
        rowIds: number[];
        action: 'accept' | 'reject';
        // Arc 5 — optional reason label stamped on each rejected
        // row (e.g. 'duplicate', 'schema_invalid', 'low_confidence').
        // Ignored when action='accept'.
        rejectReason?: string | null;
    },
): Promise<BulkUpdateResult> {
    const resp = await api.post(`/projects/${projectId}/synthetic/review-queue/bulk-update`, {
        row_ids: args.rowIds,
        action: args.action,
        reject_reason: args.rejectReason ?? null,
    });
    return resp.data as BulkUpdateResult;
}


/**
 * Epic E — accept/reject every pending row in one synth_source group by key
 * (no row-id enumeration). Powers the Data Studio review-queue panel's
 * one-click "Accept all (N)" / "Reject all (N)" on a pending group.
 */
export async function bulkUpdateSynthReviewBySource(
    projectId: number,
    args: {
        source: string;
        action: 'accept' | 'reject';
        rejectReason?: string | null;
    },
): Promise<BulkUpdateBySourceResult> {
    const resp = await api.post(
        `/projects/${projectId}/synthetic/review-queue/bulk-update-by-source`,
        {
            source: args.source,
            action: args.action,
            reject_reason: args.rejectReason ?? null,
        },
    );
    return resp.data as BulkUpdateBySourceResult;
}


/**
 * Arc 5 — physically remove rejected synth rows from
 * synthetic.jsonl. Soft-reject keeps them on disk for review;
 * this is the explicit "drop the pile" step. ``reasons`` filters
 * by ``reject_reason`` cohort; omit to purge every rejected row.
 */
export async function purgeRejectedSynthRows(
    projectId: number,
    args: { reasons?: string[] | null } = {},
): Promise<PurgeRejectedResult> {
    const resp = await api.post(
        `/projects/${projectId}/synthetic/review-queue/purge`,
        { reasons: args.reasons ?? null },
    );
    return resp.data as PurgeRejectedResult;
}

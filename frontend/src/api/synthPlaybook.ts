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

/**
 * Guided flows — several pipeline steps chained as one background Job.
 * Today: documents → generated Q&A → answer key → split → train → lift check.
 */

import api from './client';
import type { Job } from './jobs';

export interface DocumentsQaFlowPreview {
    project_id: number;
    eligible: boolean;
    blockers: string[];
    passages: number;
    backend: string | null;
    current_recipe_id: string | null;
    already_generated: number;
    plan: {
        max_passages: number;
        pairs_per_passage: number;
        estimated_training_pairs: number;
        estimated_answer_key_rows: number;
        llm_calls: number;
    };
}

export interface DocumentsQaFlowResult {
    project_id: number;
    passages_total: number;
    passages_used: number;
    passages_failed: number;
    training_pairs: number;
    answer_key_rows: number;
    backend: string;
    split: { train: number | null; val: number | null; test: number | null; dedup_dropped: number | null } | null;
    experiment_id: number | null;
    stopped_reason: string | null;
    warnings: string[];
    /** The pre-training gate: what the base model + document passages got
     *  right on the test examples, judged, before any training. */
    passages_gate?: {
        status: 'retrieval_ready' | 'retrieval_weak' | 'not_judged' | 'not_run' | 'error';
        reason?: string | null;
        score?: number | null;
        correct?: number;
        partial?: number;
        wrong?: number;
        judged?: number;
        judge?: string | null;
        split?: string | null;
    } | null;
}

export async function fetchDocumentsQaFlowPreview(projectId: number): Promise<DocumentsQaFlowPreview> {
    const res = await api.get<DocumentsQaFlowPreview>(`/projects/${projectId}/flows/documents-to-qa/preview`);
    return res.data;
}

export async function startDocumentsQaFlow(
    projectId: number,
    options: {
        maxPassages?: number;
        pairsPerPassage?: number;
        train?: boolean;
        reuseExisting?: boolean;
        passagesCheck?: boolean;
        trainIfRetrievalReady?: boolean;
    } = {},
): Promise<Job> {
    const body: Record<string, unknown> = {};
    if (options.maxPassages != null) body.max_passages = options.maxPassages;
    if (options.pairsPerPassage != null) body.pairs_per_passage = options.pairsPerPassage;
    if (options.train != null) body.train = options.train;
    if (options.reuseExisting) body.reuse_existing = true;
    if (options.passagesCheck != null) body.passages_check = options.passagesCheck;
    if (options.trainIfRetrievalReady) body.train_if_retrieval_ready = true;
    const res = await api.post<Job>(`/projects/${projectId}/flows/documents-to-qa`, body);
    return res.data;
}

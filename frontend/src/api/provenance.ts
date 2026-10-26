/**
 * Typed client for the licence & provenance ledger. Mirrors
 * ``backend/app/api/provenance.py``: one row per data source (upload, crawl,
 * repo, hub dataset, ...) with its licence, verification and usage.
 */

import api from './client';

export type ProvenanceUsage = 'any' | 'train' | 'eval' | 'rag';

export const USAGE_LABEL: Record<ProvenanceUsage, string> = {
    any: 'Any use',
    train: 'Train only',
    eval: 'Eval only',
    rag: 'Retrieval only',
};

export interface ProvenanceSource {
    id: number;
    source_kind: string;
    title: string;
    locator: string;
    source_ref: string | null;
    licence: string | null;
    licence_verified: boolean;
    licence_verified_by: string | null;
    licence_verified_at: string | null;
    retrieved_at: string | null;
    usage: ProvenanceUsage;
    document_count: number;
    row_count: number;
    /** 'unknown' | 'non_commercial' */
    licence_flags: string[];
    in_prepared?: { train: number; val: number; test: number; excluded: number } | null;
    /** True for predefined-split sources: usage is governed by their split role. */
    usage_locked: boolean;
}

export interface ProvenanceTotals {
    sources: number;
    documents: number;
    unknown_licence: number;
    non_commercial: number;
}

export interface ProvenanceList {
    sources: ProvenanceSource[];
    totals: ProvenanceTotals;
}

export interface ProvenanceUpdate {
    source_ids: number[];
    licence?: string | null;
    licence_verified?: boolean;
    usage?: ProvenanceUsage;
}

/** The `provenance` section of the prepared manifest. */
export interface PreparedProvenance {
    by_source?: Record<string, {
        title: string; kind: string; licence: string | null; usage: string;
        train: number; val: number; test: number; excluded: number;
    }>;
    unattributed?: { train: number; val: number; test: number };
    excluded_rag?: number;
    eval_routed_to_test?: number;
    eval_not_added_pinned_test?: number;
    licence_warnings?: PreparedLicenceWarning[];
    /** Notes such as "eval-only rows were not added: the test split is pinned". */
    warnings?: string[];
    /** Training-pool rows removed as exact / near-duplicates of eval-only rows. */
    eval_duplicates_removed?: number;
}

export interface PreparedLicenceWarning {
    source_id: number;
    title: string | null;
    flag: 'unknown' | 'non_commercial' | string;
    licence: string | null;
    message: string;
}

export async function listProvenanceSources(projectId: number): Promise<ProvenanceList> {
    const resp = await api.get(`/projects/${projectId}/provenance/sources`);
    return resp.data as ProvenanceList;
}

export async function updateProvenanceSources(projectId: number, body: ProvenanceUpdate): Promise<ProvenanceSource[]> {
    const resp = await api.patch(`/projects/${projectId}/provenance/sources`, body);
    const data = resp.data;
    return (Array.isArray(data) ? data : data?.sources ?? []) as ProvenanceSource[];
}

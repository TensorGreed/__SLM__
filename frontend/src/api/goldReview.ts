/**
 * Human review of the answer key (P6 Trust) —
 * GET / PUT /api/projects/{id}/gold/review.
 *
 * Verdicts live on the gold rows themselves: ``accepted`` keeps the
 * model's labels, ``corrected`` replaces them (the original is kept so a
 * clear restores it), ``rejected`` drops the row from every eval.
 */

import api from './client';

export type ReviewVerdict = 'accepted' | 'corrected' | 'rejected';

export interface ReviewSpan {
    type: string;
    start: number;
    end: number;
    text?: string;
}

export interface RowReview {
    verdict: ReviewVerdict;
    reviewed_at: string;
    reviewed_by: string | null;
    note: string;
    /** Fields as the model wrote them, before a correction. */
    original?: Record<string, unknown>;
}

export interface ReviewQueueRow {
    index: number;
    /** Hash of the text; sent back with a verdict so a stale queue is refused. */
    key: string;
    text: string;
    /** Span rows: the key's spans. null for other row shapes. */
    entities: ReviewSpan[] | null;
    /** Non-span answer fields (answer / label / summary). */
    fields: Record<string, string>;
    source: string | null;
    review: RowReview | null;
    /** The trained model's spans for this text (null without a span run). */
    predicted: ReviewSpan[] | null;
    /** Strict span F1 between key and model; null when either is missing. */
    agreement: number | null;
}

export interface ReviewCounts {
    total: number;
    reviewed: number;
    accepted: number;
    corrected: number;
    rejected: number;
    unreviewed: number;
}

export interface ReviewQueue extends ReviewCounts {
    project_id: number;
    dataset_type: 'gold_test' | 'gold_dev';
    model_experiment_id: number | null;
    last_reviewed_at: string | null;
    rows: ReviewQueueRow[];
}

export interface ReviewVerdictResponse extends ReviewCounts {
    row: ReviewQueueRow;
    last_reviewed_at: string | null;
}

export async function fetchReviewQueue(
    projectId: number,
    opts: { datasetType?: 'gold_test' | 'gold_dev'; experimentId?: number | null; compare?: boolean } = {},
): Promise<ReviewQueue> {
    const params: Record<string, string> = { dataset_type: opts.datasetType ?? 'gold_test' };
    if (opts.experimentId != null) params.experiment_id = String(opts.experimentId);
    if (opts.compare === false) params.compare = 'false';
    const res = await api.get(`/projects/${projectId}/gold/review`, { params });
    return res.data as ReviewQueue;
}

export async function setReviewVerdict(
    projectId: number,
    body: {
        dataset_type?: 'gold_test' | 'gold_dev';
        index: number;
        key: string;
        verdict: ReviewVerdict | null;
        corrected?: { entities?: ReviewSpan[]; answer?: string; label?: string; summary?: string };
        note?: string;
    },
): Promise<ReviewVerdictResponse> {
    const res = await api.put(`/projects/${projectId}/gold/review`, { dataset_type: 'gold_test', ...body });
    return res.data as ReviewVerdictResponse;
}

/** "3 of 43 checked · 1 corrected · 1 rejected". */
export function reviewProgressText(c: ReviewCounts): string {
    const parts = [`${c.reviewed} of ${c.total} checked`];
    if (c.corrected) parts.push(`${c.corrected} corrected`);
    if (c.rejected) parts.push(`${c.rejected} rejected`);
    return parts.join(' · ');
}

/** Why a row sits where it does in the queue. */
export function agreementText(row: ReviewQueueRow): string {
    if (row.review) return `reviewed: ${row.review.verdict}`;
    if (row.agreement == null) return 'no model to compare';
    if (row.agreement >= 1) return 'model agrees';
    return `model disagrees (agreement ${row.agreement.toFixed(2)})`;
}

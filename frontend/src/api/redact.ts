/**
 * POST /api/projects/{id}/redact — tag a text with a span-tagging
 * (token classification) run and mask / tag what it finds.
 */

import api from './client';

export type RedactMode = 'mask' | 'tag' | 'spans';

export interface RedactSpan {
    type: string;
    start: number;
    end: number;
    text: string;
    /** Mean per-token probability of the span's type. */
    score: number;
    /** The weakest token's probability. */
    min_score: number;
}

export interface RedactResponse {
    project_id: number;
    experiment_id: number;
    experiment_name: string | null;
    mode: RedactMode;
    spans: RedactSpan[];
    text: string;
    offsets: Array<{ type: string; start: number; end: number; out_start: number; out_end: number }>;
    settings: {
        min_score: number | null;
        recall_bias: number;
        per_type: Record<string, number>;
        thresholds_file: boolean;
        max_length: number;
    };
}

export interface RedactRequest {
    text: string;
    experiment_id?: number | null;
    mode?: RedactMode;
    min_score?: number | null;
    recall_bias?: number | null;
}

export async function redactText(projectId: number, body: RedactRequest): Promise<RedactResponse> {
    const res = await api.post<RedactResponse>(`/projects/${projectId}/redact`, body);
    return res.data;
}

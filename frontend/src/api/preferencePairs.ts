import api from './client';

export type PairVerdict = 'wrong' | 'partial';

export interface PreferencePairExample {
    prompt?: string;
    chosen?: string;
    rejected?: string;
    verdict?: string;
    row_index?: number;
    source_split?: string | null;
}

export interface PairsFromEvalResponse {
    /** Always null today: the pairs are a file, not a registered dataset. */
    dataset_id: number | null;
    dataset_name: string;
    /** Where the pairs file was written. */
    dataset_path?: string;
    source_split?: string | null;
    /** Set when the test-split override was used. */
    warning?: string | null;
    created: number;
    skipped: Record<string, number>;
    examples: PreferencePairExample[];
}

/** Skipped-row reasons the backend reports, with the wording shown to the user. */
export const SKIP_REASON_LABELS: Record<string, string> = {
    not_judged: 'Not judged',
    correct: 'Judged correct',
    prediction_equals_reference: 'Prediction equals the answer key',
    empty_prediction: 'Empty prediction',
    empty_reference: 'Empty answer key',
    no_prediction_stored: 'No prediction stored',
    not_included: 'Verdict not included',
    not_stored: 'Judged but not stored (past the stored-row cap or history cut)',
};

/** Splits whose pairs would teach the model rows it is later scored on. */
export const TEST_SPLITS = ['test', 'gold_test'];

export async function buildPairsFromEval(
    projectId: number,
    body: { eval_result_id: number; include: PairVerdict[]; dataset_name?: string; allow_test_split?: boolean },
): Promise<PairsFromEvalResponse> {
    const res = await api.post<PairsFromEvalResponse>(
        `/projects/${projectId}/alignment/preference-pairs/from-eval`,
        body,
    );
    return res.data;
}

/** Pull the skip breakdown out of a 400 body (top level, metadata, or a detail object). */
export function skipBreakdownFromError(error: unknown): Record<string, number> | null {
    const data = (error as { response?: { data?: Record<string, unknown> } })?.response?.data;
    if (!data || typeof data !== 'object') return null;
    const meta = (data.metadata && typeof data.metadata === 'object' ? data.metadata : {}) as Record<string, unknown>;
    const detail = (data.detail && typeof data.detail === 'object' ? data.detail : {}) as Record<string, unknown>;
    const found = data.skipped ?? meta.skipped ?? detail.skipped;
    return found && typeof found === 'object' ? (found as Record<string, number>) : null;
}

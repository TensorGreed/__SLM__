/**
 * Checked decoding (Epic Y): opt-in verify-and-retry at eval time.
 *
 * After the model answers a test row, quick automatic checks look at the answer and the
 * question only (never the reference answer). When one fails, the row is tried again — sampled
 * again ("resample") or asked to fix the listed problems ("repair") — up to N tries. Off by
 * default. Backend: ``checked_decoding_service`` and ``GET/PUT .../evaluation/checked-decoding``.
 */

import api from './client';

export type CheckedDecodingMode = 'resample' | 'repair';

export interface CheckedDecodingSettings {
    enabled: boolean;
    mode: CheckedDecodingMode;
    max_attempts: number;
    verifiers: string[] | null;
    sample_temperature: number;
    seed: number;
}

export interface CheckedDecodingPayload {
    settings: CheckedDecodingSettings;
    label: string;
    modes: CheckedDecodingMode[];
    verifier_ids: string[];
    max_attempts_limit: number;
}

/** The policy an eval ran under (``metrics.inference.decoding.checked_decoding``). */
export interface CheckedDecodingPolicy {
    mode: CheckedDecodingMode;
    max_attempts: number;
    verifiers: string[] | 'auto';
    sample_temperature?: number;
    seed?: number;
}

/** ``checked_decoding_summary``: what the retries did in one eval. */
export interface CheckedDecodingSummary {
    policy: CheckedDecodingPolicy | null;
    label?: string;
    skipped?: string;
    rows?: number;
    rows_checked?: number;
    rows_retried?: number;
    rows_fixed_by_retry?: number;
    rows_still_failing?: number;
    mean_attempts?: number;
    extra_generation_seconds?: number;
}

export async function fetchCheckedDecoding(projectId: number): Promise<CheckedDecodingPayload> {
    const res = await api.get<CheckedDecodingPayload>(`/projects/${projectId}/evaluation/checked-decoding`);
    return res.data;
}

export async function saveCheckedDecoding(
    projectId: number,
    settings: Partial<CheckedDecodingSettings>,
): Promise<CheckedDecodingPayload> {
    const res = await api.put<CheckedDecodingPayload>(`/projects/${projectId}/evaluation/checked-decoding`, settings);
    return res.data;
}

/** "Checked decoding · up to 3 tries · resample" (null when the eval ran without it). */
export function checkedDecodingBadgeText(policy: CheckedDecodingPolicy | null | undefined): string | null {
    if (!policy) return null;
    return `Checked decoding · up to ${policy.max_attempts} tries · ${policy.mode}`;
}

/** "2 rows fixed by a retry" / "1 row fixed by a retry · 3 still failing the checks". */
export function rowsFixedText(summary: CheckedDecodingSummary | null | undefined): string | null {
    if (!summary || !summary.policy) return null;
    if (summary.skipped === 'engine_unsupported') {
        return 'not applied: this model type answers without generated text to check';
    }
    const fixed = summary.rows_fixed_by_retry ?? 0;
    let text = `${fixed} ${fixed === 1 ? 'row' : 'rows'} fixed by a retry`;
    const still = summary.rows_still_failing ?? 0;
    if (still > 0) text += ` · ${still} still failing the checks`;
    return text;
}

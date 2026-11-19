/**
 * Typed wrapper for answer-key (gold) file management. Works for every task shape: rows are stored
 * as given.
 *
 *   POST   /projects/{id}/gold/import-file            multipart: file, dataset_type, mode, drop_lines
 *   POST   /projects/{id}/gold/{type}/rows/delete     { row_ids }
 *   DELETE /projects/{id}/gold/{type}                 clear the set
 */

import api from './client';

export type GoldType = 'gold_dev' | 'gold_test';
export type GoldImportMode = 'append' | 'replace';

export interface GoldDatasetLabel {
    id: number;
    name: string;
    dataset_type: GoldType;
    record_count: number;
    is_locked: boolean;
}

export interface GoldRejectRow {
    line: number;
    preview: string;
}

export interface GoldRejectGroup {
    label: string;
    count: number;
    lines: number[];
    rows: GoldRejectRow[];
}

export interface GoldRejects {
    count: number;
    by_reason: Record<string, GoldRejectGroup>;
}

export interface GoldImportResult {
    imported: number;
    total: number;
    mode: GoldImportMode;
    source_filename: string;
    dropped_rows: number;
    dropped_lines: number[];
    snapshot_version?: number | null;
    dataset: GoldDatasetLabel;
}

/** Returned (not thrown) when the file has rows that cannot be used; nothing was written. */
export interface GoldImportRejected {
    rejected: GoldRejects;
    usable: number;
}

export async function importGoldFile(
    projectId: number,
    file: File,
    opts: { datasetType: GoldType; mode: GoldImportMode; dropLines?: number[] },
): Promise<{ result: GoldImportResult } | GoldImportRejected> {
    const form = new FormData();
    form.append('file', file);
    form.append('dataset_type', opts.datasetType);
    form.append('mode', opts.mode);
    if (opts.dropLines && opts.dropLines.length > 0) form.append('drop_lines', JSON.stringify(opts.dropLines));
    try {
        const res = await api.post<GoldImportResult>(`/projects/${projectId}/gold/import-file`, form);
        return { result: res.data };
    } catch (err) {
        const data = (err as {
            response?: { data?: { error_code?: string; metadata?: { rejected?: GoldRejects; row_counts?: { usable?: number } } } };
        })?.response?.data;
        if (data?.error_code === 'GOLD_IMPORT_REJECTED_ROWS' && data.metadata?.rejected) {
            return { rejected: data.metadata.rejected, usable: data.metadata.row_counts?.usable ?? 0 };
        }
        throw err;
    }
}

export async function deleteGoldRows(projectId: number, datasetType: GoldType, rowIds: number[]) {
    const res = await api.post<{ deleted: number; remaining: number; missing: number[]; dataset: GoldDatasetLabel }>(
        `/projects/${projectId}/gold/${datasetType}/rows/delete`,
        { row_ids: rowIds },
    );
    return res.data;
}

export async function clearGoldSet(projectId: number, datasetType: GoldType) {
    const res = await api.delete<{ deleted: number; remaining: number; dataset: GoldDatasetLabel }>(
        `/projects/${projectId}/gold/${datasetType}`,
    );
    return res.data;
}

/** Message for a failed gold call (409 locked, 400 unreadable file, ...). */
export function goldErrorMessage(err: unknown): string {
    const data = (err as { response?: { data?: { message?: string; detail?: unknown } } })?.response?.data;
    if (typeof data?.message === 'string') return data.message;
    if (typeof data?.detail === 'string') return data.detail;
    return 'The answer-key change failed.';
}

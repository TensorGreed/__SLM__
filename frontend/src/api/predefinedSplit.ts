/** Predefined-split import: the user's own train / val / test files with
 *  exact membership (POST /projects/{id}/dataset/predefined-split). */

import api from './client';

export type OnLeak = 'fail' | 'drop_from_train';

export interface PredefinedSplitSourceFile {
    filename: string;
    stored_as: string;
    document_id: number;
    source_rows: number;
    sha256: string;
}

export interface PredefinedSplitLeakPair {
    pair: string;
    counts: Record<string, number>;
    /** 1-based row numbers in each file (header excluded), capped at 20. */
    sample_rows: Record<string, number[]>;
}

export interface PredefinedSplitManifest {
    split_source: 'predefined';
    total_entries: number;
    splits: Record<string, number>;
    file_hashes: Record<string, string>;
    dataset_versions: Record<string, number>;
    prepared_version: number | null;
    source_files: Record<string, PredefinedSplitSourceFile>;
    duplicate_report: Record<string, number>;
    leak_report: {
        on_leak: OnLeak;
        pairs: PredefinedSplitLeakPair[];
        dropped_from_train: number;
        dropped_sample_rows?: number[];
    };
}

export async function importPredefinedSplit(
    projectId: number,
    files: { train: File; val?: File | null; test?: File | null },
    opts: { onLeak: OnLeak; fieldMapping?: string },
): Promise<PredefinedSplitManifest> {
    const form = new FormData();
    form.append('train', files.train);
    if (files.val) form.append('val', files.val);
    if (files.test) form.append('test', files.test);
    form.append('on_leak', opts.onLeak);
    if (opts.fieldMapping && opts.fieldMapping.trim()) form.append('field_mapping', opts.fieldMapping.trim());
    const res = await api.post<PredefinedSplitManifest>(
        `/projects/${projectId}/dataset/predefined-split`,
        form,
    );
    return res.data;
}

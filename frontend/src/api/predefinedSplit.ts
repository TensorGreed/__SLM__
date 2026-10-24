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
    /** Set when drop_from_train filtered the train file: the rows actually registered. */
    registered_rows?: number;
    registered_sha256?: string;
    dropped_row_lines?: number[];
}

export interface PredefinedSplitLeakPair {
    pair: string;
    counts: Record<string, number>;
    /** File line numbers in each file (a CSV header is line 1), capped at 20. */
    sample_rows: Record<string, number[]>;
}

export interface PredefinedSplitManifest {
    split_source: 'predefined';
    /** Present (true) when a test file was imported. */
    predefined_test_active?: boolean;
    /** Present when a confirmed re-import retired the project's earlier own test file(s). */
    dropped_existing_test_document_ids?: number[];
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
    opts: { onLeak: OnLeak; fieldMapping?: string; chatTemplate?: string | null; dropExistingTest?: boolean },
): Promise<PredefinedSplitManifest> {
    const form = new FormData();
    form.append('train', files.train);
    if (files.val) form.append('val', files.val);
    if (files.test) form.append('test', files.test);
    form.append('on_leak', opts.onLeak);
    if (opts.fieldMapping && opts.fieldMapping.trim()) form.append('field_mapping', opts.fieldMapping.trim());
    if (opts.chatTemplate) form.append('chat_template', opts.chatTemplate);
    if (opts.dropExistingTest) form.append('drop_existing_test', 'true');
    const res = await api.post<PredefinedSplitManifest>(
        `/projects/${projectId}/dataset/predefined-split`,
        form,
    );
    return res.data;
}

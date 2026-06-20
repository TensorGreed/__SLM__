import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioDatasetVersionsPanel from './DataStudioDatasetVersionsPanel';

const versionPayload = {
    project_id: 1,
    verdict: 'ready',
    read_only: true,
    auto_apply: false,
    source_of_truth: 'deterministic_data_studio_checks',
    summary: {
        prepared_dataset_count: 3,
        total_version_count: 4,
        latest_total_rows: 120,
        latest_created_at: '2026-05-24T12:30:00Z',
        manifest_exists: true,
        manifest_readable: true,
        manifest_version_ref_count: 3,
        training_reuse_ready: true,
        eval_reuse_ready: true,
    },
    latest_artifacts: [
        {
            key: 'train',
            manifest_key: 'train',
            label: 'Train',
            dataset_type: 'train',
            dataset_id: 21,
            dataset_name: 'Train Set',
            row_count: 96,
            file_path: '/tmp/train.jsonl',
            file_exists: true,
            version_count: 2,
            latest_version: {
                id: 202,
                version: 2,
                record_count: 96,
                file_path: '/tmp/train.jsonl',
                file_exists: true,
                created_at: '2026-05-24T12:30:00Z',
                manifest_split: 'train',
                manifest_count: 96,
                manifest: {},
            },
            latest_version_number: 2,
            manifest_count: 96,
            manifest_version: 2,
            manifest_file_path: '/tmp/train.jsonl',
            manifest_file_hash: 'hash-train',
            version_matches_manifest: true,
            row_count_matches_manifest: true,
        },
        {
            key: 'validation',
            manifest_key: 'val',
            label: 'Validation',
            dataset_type: 'validation',
            dataset_id: 22,
            dataset_name: 'Validation Set',
            row_count: 12,
            file_path: '/tmp/val.jsonl',
            file_exists: true,
            version_count: 1,
            latest_version: {
                id: 203,
                version: 1,
                record_count: 12,
                file_path: '/tmp/val.jsonl',
                file_exists: true,
                created_at: '2026-05-24T12:10:00Z',
                manifest_split: 'val',
                manifest_count: 12,
                manifest: {},
            },
            latest_version_number: 1,
            manifest_count: 12,
            manifest_version: 1,
            manifest_file_path: '/tmp/val.jsonl',
            manifest_file_hash: 'hash-val',
            version_matches_manifest: true,
            row_count_matches_manifest: true,
        },
        {
            key: 'test',
            manifest_key: 'test',
            label: 'Test',
            dataset_type: 'test',
            dataset_id: 23,
            dataset_name: 'Test Set',
            row_count: 12,
            file_path: '/tmp/test.jsonl',
            file_exists: true,
            version_count: 1,
            latest_version: {
                id: 204,
                version: 1,
                record_count: 12,
                file_path: '/tmp/test.jsonl',
                file_exists: true,
                created_at: '2026-05-24T12:10:00Z',
                manifest_split: 'test',
                manifest_count: 12,
                manifest: {},
            },
            latest_version_number: 1,
            manifest_count: 12,
            manifest_version: 1,
            manifest_file_path: '/tmp/test.jsonl',
            manifest_file_hash: 'hash-test',
            version_matches_manifest: true,
            row_count_matches_manifest: true,
        },
    ],
    version_history: [
        {
            dataset_id: 21,
            dataset_name: 'Train Set',
            dataset_type: 'train',
            row_count: 96,
            file_path: '/tmp/train.jsonl',
            file_exists: true,
            is_locked: false,
            created_at: '2026-05-24T12:00:00Z',
            updated_at: '2026-05-24T12:30:00Z',
            version_count: 2,
            latest_version: {
                id: 202,
                version: 2,
                record_count: 96,
                file_path: '/tmp/train.jsonl',
                file_exists: true,
                created_at: '2026-05-24T12:30:00Z',
                manifest_split: 'train',
                manifest_count: 96,
                manifest: {},
            },
            versions: [],
        },
        {
            dataset_id: 22,
            dataset_name: 'Validation Set',
            dataset_type: 'validation',
            row_count: 12,
            file_path: '/tmp/val.jsonl',
            file_exists: true,
            is_locked: false,
            created_at: '2026-05-24T12:00:00Z',
            updated_at: '2026-05-24T12:10:00Z',
            version_count: 1,
            latest_version: {
                id: 203,
                version: 1,
                record_count: 12,
                file_path: '/tmp/val.jsonl',
                file_exists: true,
                created_at: '2026-05-24T12:10:00Z',
                manifest_split: 'val',
                manifest_count: 12,
                manifest: {},
            },
            versions: [],
        },
    ],
    manifest: {
        exists: true,
        readable: true,
        path: '/tmp/manifest.json',
        error: null,
        created_at: '2026-05-24T12:30:00Z',
        seed: 42,
        total_entries: 120,
        splits: { train: 96, val: 12, test: 12 },
        ratios: { train: 0.8, val: 0.1, test: 0.1 },
        file_hashes: { train: 'hash-train', val: 'hash-val', test: 'hash-test' },
        dataset_versions: { train: 2, val: 1, test: 1 },
        included_types: ['cleaned', 'gold_dev', 'synthetic'],
        chat_template: 'llama3',
        adapter_id: 'classification-label',
        task_profile: 'classification',
    },
    source_context: {
        recipe: {
            id: 'classification',
            name: 'Text Classifier',
            task_profile: 'classification',
            adapter_id: 'classification-label',
            default_input_column: 'text',
            default_output_column: 'label',
        },
        domain: {
            profile_id: 'support-faq-profile-v1',
            profile_source: 'project',
            profile_display_name: 'Support FAQ Profile',
            profile_version: '1.0.0',
            pack_id: 'support-faq-pack-v1',
            pack_source: 'project',
            pack_display_name: 'Support FAQ Pack',
            pack_version: '1.0.0',
            pack_default_profile_id: 'support-faq-profile-v1',
        },
        domain_runtime: {},
        adapter_id: 'classification-label',
        task_profile: 'classification',
        included_source_types: ['cleaned', 'gold_dev', 'synthetic'],
    },
    reuse_readiness: {
        training: {
            status: 'ready',
            target_tab: 'training',
            message: 'Prepared train/validation/test versions are reusable for training.',
        },
        evaluation: {
            status: 'ready',
            target_tab: 'eval',
            message: 'Validation and test artifacts are available for evaluation.',
        },
    },
    reproducibility: [
        {
            id: 'manifest',
            label: 'Prepared manifest',
            status: 'met',
            message: 'Prepared manifest is readable.',
            target_tab: 'dataprep',
        },
        {
            id: 'version_refs',
            label: 'Manifest version refs',
            status: 'met',
            message: 'Latest versions match manifest references.',
            target_tab: 'dataprep',
        },
    ],
    issues: [],
    entry_points: [
        {
            label: 'Open Dataset Prep',
            target_tab: 'dataprep',
            reason: 'Create or refresh prepared dataset versions.',
            requires_confirmation: true,
        },
        {
            label: 'Open Training',
            target_tab: 'training',
            reason: 'Use prepared split versions for training runs.',
            requires_confirmation: false,
        },
        {
            label: 'Open Eval',
            target_tab: 'eval',
            reason: 'Use validation/test artifacts for evaluation.',
            requires_confirmation: false,
        },
    ],
    power_details: {},
};

describe('DataStudioDatasetVersionsPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders reusable dataset versions and routes to training/eval workflows', async () => {
        apiMock.get.mockResolvedValueOnce({ data: versionPayload });
        const onOpenTarget = vi.fn();

        render(<DataStudioDatasetVersionsPanel projectId={1} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-dataset-versions')).toBeInTheDocument();
        });

        expect(screen.getByText('Reusable')).toBeInTheDocument();
        expect(screen.getByText('Text Classifier')).toBeInTheDocument();
        expect(screen.getByText('Support FAQ Profile')).toBeInTheDocument();
        expect(screen.getByText('Train')).toBeInTheDocument();
        expect(screen.getByText('Validation')).toBeInTheDocument();
        expect(screen.getByText('Test')).toBeInTheDocument();
        expect(screen.getByText('Prepared manifest')).toBeInTheDocument();
        expect(screen.getByText('Training reuse')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: /Open Training/i }));
        fireEvent.click(screen.getByRole('button', { name: /^Eval$/i }));
        expect(onOpenTarget).toHaveBeenCalledWith('training');
        expect(onOpenTarget).toHaveBeenCalledWith('eval');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/dataset-versions');
    });

    it('exports a prepared split as JSONL via the export endpoint', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: versionPayload })   // initial load
            .mockResolvedValueOnce({ data: new Blob(['{}\n']) }); // the export blob
        const createObjectURL = vi.fn(() => 'blob://stub');
        const revokeObjectURL = vi.fn();
        const origCreate = URL.createObjectURL;
        const origRevoke = URL.revokeObjectURL;
        URL.createObjectURL = createObjectURL as typeof URL.createObjectURL;
        URL.revokeObjectURL = revokeObjectURL as typeof URL.revokeObjectURL;
        try {
            render(<DataStudioDatasetVersionsPanel projectId={1} onOpenTarget={vi.fn()} />);
            await waitFor(() => {
                expect(screen.getByTestId('data-studio-dataset-versions')).toBeInTheDocument();
            });
            // Each file-present artifact gets an Export JSONL button (train/val/test).
            const exportButtons = screen.getAllByRole('button', { name: /Export JSONL/i });
            expect(exportButtons.length).toBeGreaterThan(0);
            fireEvent.click(exportButtons[0]);
            await waitFor(() => {
                expect(apiMock.get).toHaveBeenCalledWith(
                    '/projects/1/data-studio/dataset-versions/train/export',
                    { responseType: 'blob' },
                );
            });
            expect(createObjectURL).toHaveBeenCalled();
        } finally {
            URL.createObjectURL = origCreate;
            URL.revokeObjectURL = origRevoke;
        }
    });

    it('renders empty version state without mutating', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...versionPayload,
                verdict: 'empty',
                summary: {
                    prepared_dataset_count: 0,
                    total_version_count: 0,
                    latest_total_rows: 0,
                    latest_created_at: null,
                    manifest_exists: false,
                    manifest_readable: false,
                    manifest_version_ref_count: 0,
                    training_reuse_ready: false,
                    eval_reuse_ready: false,
                },
                latest_artifacts: versionPayload.latest_artifacts.map((item) => ({
                    ...item,
                    dataset_id: null,
                    dataset_name: null,
                    row_count: 0,
                    file_path: '',
                    file_exists: false,
                    version_count: 0,
                    latest_version: null,
                    latest_version_number: null,
                    manifest_count: 0,
                    manifest_version: null,
                    manifest_file_path: '',
                    manifest_file_hash: '',
                    version_matches_manifest: false,
                    row_count_matches_manifest: true,
                })),
                version_history: [],
                manifest: {
                    ...versionPayload.manifest,
                    exists: false,
                    readable: false,
                    total_entries: 0,
                    splits: {},
                    file_hashes: {},
                    dataset_versions: {},
                    included_types: [],
                    adapter_id: null,
                    task_profile: null,
                },
                reuse_readiness: {
                    training: {
                        status: 'missing',
                        target_tab: 'training',
                        message: 'Refresh prepared versions before treating this dataset as reusable for training.',
                    },
                    evaluation: {
                        status: 'missing',
                        target_tab: 'eval',
                        message: 'Prepare validation and test artifacts before relying on evaluation reuse.',
                    },
                },
                reproducibility: [
                    {
                        id: 'manifest',
                        label: 'Prepared manifest',
                        status: 'missing',
                        message: 'Create or refresh the prepared manifest in Dataset Prep.',
                        target_tab: 'dataprep',
                    },
                ],
                issues: [
                    {
                        id: 'dataset_versions_empty',
                        severity: 'info',
                        title: 'No prepared dataset versions yet',
                        message: 'Run Dataset Prep to create versioned train, validation, and test artifacts.',
                        action_label: 'Open Dataset Prep',
                        target_tab: 'dataprep',
                    },
                ],
            },
        });
        const onOpenTarget = vi.fn();

        render(<DataStudioDatasetVersionsPanel projectId={2} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-dataset-versions')).toBeInTheDocument();
        });

        expect(screen.getByText('No versions')).toBeInTheDocument();
        expect(screen.getByText('No prepared dataset versions yet')).toBeInTheDocument();
        expect(screen.getByText('Version history appears after Dataset Prep writes prepared split versions.')).toBeInTheDocument();
        expect(screen.getByText('Read-only check')).toBeInTheDocument();

        fireEvent.click(screen.getAllByRole('button', { name: /Open Dataset Prep/i })[0]);
        expect(onOpenTarget).toHaveBeenCalledWith('dataprep');
    });

    // ─────────────────────────────────────────────────────────────────
    // Arc A — inline "Re-prepare to fix drift" CTA. Only renders when
    // an artifact's version/row-count disagrees with the manifest; on
    // the all-clean payload above the button must be hidden so the
    // toolbar doesn't pollute itself with an action that would re-
    // build identical files.
    // ─────────────────────────────────────────────────────────────────

    it('hides the re-prepare button when artifacts all match the manifest', async () => {
        apiMock.get.mockResolvedValueOnce({ data: versionPayload });
        render(<DataStudioDatasetVersionsPanel projectId={1} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-dataset-versions')).toBeInTheDocument();
        });
        expect(
            screen.queryByTestId('data-studio-versions-re-prepare'),
        ).not.toBeInTheDocument();
    });

    it('shows the re-prepare button when an artifact disagrees with the manifest', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...versionPayload,
                verdict: 'attention',
                latest_artifacts: [
                    {
                        ...versionPayload.latest_artifacts[0],
                        version_matches_manifest: false,
                    },
                    ...versionPayload.latest_artifacts.slice(1),
                ],
            },
        });
        render(<DataStudioDatasetVersionsPanel projectId={1} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('data-studio-versions-re-prepare'),
            ).toBeInTheDocument();
        });
    });

    it('POSTs to /dataset/split and refreshes on re-prepare success', async () => {
        const driftedPayload = {
            ...versionPayload,
            verdict: 'attention',
            latest_artifacts: [
                {
                    ...versionPayload.latest_artifacts[0],
                    row_count_matches_manifest: false,
                },
                ...versionPayload.latest_artifacts.slice(1),
            ],
        };
        apiMock.get.mockResolvedValueOnce({ data: driftedPayload });
        apiMock.post.mockResolvedValueOnce({
            data: { train_count: 96, val_count: 12, test_count: 12 },
        });
        apiMock.get.mockResolvedValueOnce({ data: versionPayload });

        render(<DataStudioDatasetVersionsPanel projectId={9} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('data-studio-versions-re-prepare'),
            ).toBeInTheDocument();
        });

        fireEvent.click(screen.getByTestId('data-studio-versions-re-prepare'));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/9/dataset/split',
                {},
            );
        });
        await waitFor(() => {
            expect(
                screen.getByTestId('data-studio-versions-run-flash'),
            ).toHaveTextContent(/Re-prepared 120 rows/);
        });
        expect(apiMock.get).toHaveBeenCalledTimes(2);
    });

    it('activates a prepared snapshot and refreshes', async () => {
        const withSnapshots = {
            ...versionPayload,
            prepared_versions: {
                available: [
                    { version: 2, is_active: false },
                    { version: 1, is_active: true },
                ],
                active: 1,
                latest_prepared_version: 2,
            },
        };
        apiMock.get.mockResolvedValueOnce({ data: withSnapshots });
        apiMock.post.mockResolvedValueOnce({
            data: { project_id: 9, active_prepared_version: 2, restored_counts: { train: 96, val: 12, test: 12 } },
        });
        apiMock.get.mockResolvedValueOnce({ data: withSnapshots });

        render(<DataStudioDatasetVersionsPanel projectId={9} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('prepared-version-snapshots')).toBeInTheDocument();
        });
        // v1 is active (badge); v2 has an enabled "Make active".
        expect(screen.getByTestId('snapshot-active-1')).toBeInTheDocument();
        const makeActiveButtons = screen.getAllByRole('button', { name: /Make active/i });
        fireEvent.click(makeActiveButtons[0]); // v2 (listed first)

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/9/data-studio/dataset-versions/2/activate',
            );
        });
        await waitFor(() => {
            expect(screen.getByTestId('activate-flash')).toHaveTextContent(/v2 is now the active/i);
        });
    });

    it('compares two prepared snapshots and shows the diff', async () => {
        const withSnapshots = {
            ...versionPayload,
            prepared_versions: {
                available: [
                    { version: 2, is_active: true },
                    { version: 1, is_active: false },
                ],
                active: 2,
                latest_prepared_version: 2,
            },
        };
        apiMock.get.mockImplementation((url: string) => {
            if (url.includes('/dataset-versions/compare')) {
                return Promise.resolve({
                    data: {
                        project_id: 9,
                        a: { version: 1, total_entries: 100, splits: { train: 80 }, ratios: {}, included_types: ['cleaned'] },
                        b: { version: 2, total_entries: 150, splits: { train: 120 }, ratios: {}, included_types: ['cleaned', 'synthetic'] },
                        diff: {
                            total_delta: 50,
                            split_deltas: { train: 40 },
                            sources_added: ['synthetic'],
                            sources_removed: [],
                            seed_changed: true,
                            ratios_changed: false,
                            strategy_changed: false,
                        },
                    },
                });
            }
            return Promise.resolve({ data: withSnapshots });
        });

        render(<DataStudioDatasetVersionsPanel projectId={9} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('version-compare')).toBeInTheDocument();
        });
        fireEvent.change(screen.getByLabelText('Compare version A'), { target: { value: '1' } });
        fireEvent.change(screen.getByLabelText('Compare version B'), { target: { value: '2' } });
        fireEvent.click(screen.getByRole('button', { name: /^Compare$/ }));
        const result = await screen.findByTestId('version-compare-result');
        expect(result).toHaveTextContent(/\+50.*rows total/);
        expect(result).toHaveTextContent(/train \+40/);
        expect(result).toHaveTextContent(/Sources added: synthetic/);
        expect(result).toHaveTextContent(/seed/);
    });

    it('retrain-from-version activates then opens Training', async () => {
        const withSnapshots = {
            ...versionPayload,
            prepared_versions: {
                available: [{ version: 1, is_active: true }],
                active: 1,
                latest_prepared_version: 1,
            },
        };
        apiMock.get.mockResolvedValue({ data: withSnapshots });
        apiMock.post.mockResolvedValueOnce({
            data: { project_id: 9, active_prepared_version: 1, restored_counts: { train: 96 } },
        });
        const onOpenTarget = vi.fn();
        render(<DataStudioDatasetVersionsPanel projectId={9} onOpenTarget={onOpenTarget} />);
        await waitFor(() => {
            expect(screen.getByTestId('prepared-version-snapshots')).toBeInTheDocument();
        });
        fireEvent.click(screen.getByRole('button', { name: /Retrain from this/i }));
        await waitFor(() => {
            expect(onOpenTarget).toHaveBeenCalledWith('training');
        });
    });
});

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioPrepareDatasetPanel from './DataStudioPrepareDatasetPanel';

const preparePayload = {
    project_id: 1,
    verdict: 'ready',
    can_prepare: true,
    read_only: true,
    auto_apply: false,
    source_of_truth: 'deterministic_data_studio_checks',
    recipe: {
        status: 'met',
        selected: {
            id: 'classification',
            name: 'Text Classifier',
            task_profile: 'classification',
            adapter_id: 'classification-label',
            default_input_column: 'text',
            default_output_column: 'label',
        },
        message: 'Text Classifier is selected.',
    },
    mapping: {
        status: 'met',
        message: 'Adapter mapping passes the required field contract for the selected recipe.',
        verdict: 'ready',
        contract_pass: true,
        source: {
            dataset_type: 'cleaned',
            dataset_id: 12,
            dataset_name: 'Cleaned Rows',
            row_count: 120,
        },
        adapter_id: 'classification-label',
        task_profile: 'classification',
        mapping_success_rate: 1,
        sampled_records: 100,
        mapped_records: 100,
        required_fields: ['text', 'label'],
        required_fields_below_100: [],
    },
    splits: {
        status: 'ready',
        total_prepared_rows: 120,
        required_splits: ['train', 'validation', 'test'],
        items: [
            {
                key: 'train',
                manifest_key: 'train',
                label: 'Train',
                dataset_type: 'train',
                dataset_id: 21,
                exists: true,
                row_count: 96,
                file_path: '/tmp/train.jsonl',
                file_exists: true,
                manifest_count: 96,
                manifest_version: 1,
                version_count: 1,
                latest_version: {
                    id: 101,
                    version: 1,
                    record_count: 96,
                    file_path: '/tmp/train.jsonl',
                    created_at: '2026-05-24T12:00:00Z',
                    manifest: {},
                },
            },
            {
                key: 'validation',
                manifest_key: 'val',
                label: 'Validation',
                dataset_type: 'validation',
                dataset_id: 22,
                exists: true,
                row_count: 12,
                file_path: '/tmp/val.jsonl',
                file_exists: true,
                manifest_count: 12,
                manifest_version: 1,
                version_count: 1,
                latest_version: {
                    id: 102,
                    version: 1,
                    record_count: 12,
                    file_path: '/tmp/val.jsonl',
                    created_at: '2026-05-24T12:00:00Z',
                    manifest: {},
                },
            },
            {
                key: 'test',
                manifest_key: 'test',
                label: 'Test',
                dataset_type: 'test',
                dataset_id: 23,
                exists: true,
                row_count: 12,
                file_path: '/tmp/test.jsonl',
                file_exists: true,
                manifest_count: 12,
                manifest_version: 1,
                version_count: 1,
                latest_version: {
                    id: 103,
                    version: 1,
                    record_count: 12,
                    file_path: '/tmp/test.jsonl',
                    created_at: '2026-05-24T12:00:00Z',
                    manifest: {},
                },
            },
        ],
    },
    manifest: {
        status: 'ready',
        exists: true,
        readable: true,
        path: '/tmp/manifest.json',
        error: null,
        created_at: '2026-05-24T12:00:00Z',
        total_entries: 120,
        splits: { train: 96, val: 12, test: 12 },
        ratios: { train: 0.8, val: 0.1, test: 0.1 },
        included_types: ['cleaned', 'gold_dev', 'synthetic'],
        adapter_id: 'classification-label',
        task_profile: 'classification',
        dataset_versions: { train: 1, val: 1, test: 1 },
        missing_dataset_version_splits: [],
        missing_manifest_version_splits: [],
    },
    inclusion: {
        trainable_rows: 120,
        raw_rows: 0,
        cleaned_rows: 100,
        gold_rows: 10,
        synthetic_total: 15,
        synthetic_pending: 5,
        synthetic_accepted: 10,
        synthetic_pending_excluded: true,
        gold_trusted_examples: 10,
        gold_review_needed: 0,
        included_source_types: ['cleaned', 'gold_dev', 'synthetic'],
    },
    review_blockers: [],
    checks: [
        {
            id: 'recipe',
            label: 'Recipe readiness',
            status: 'met',
            message: 'Text Classifier is selected.',
            target_tab: 'data',
        },
        {
            id: 'mapping_contract',
            label: 'Mapping contract',
            status: 'met',
            message: 'Adapter mapping passes the required field contract for the selected recipe.',
            target_tab: 'dataprep',
        },
        {
            id: 'split_files',
            label: 'Prepared split files',
            status: 'met',
            message: 'Train, validation, and test splits are present.',
            target_tab: 'dataprep',
        },
    ],
    issues: [],
    entry_point: {
        label: 'Open Dataset Prep',
        target_tab: 'dataprep',
        reason: 'Confirm adapter and split settings before writing prepared files.',
        requires_confirmation: true,
    },
    power_details: {},
};

describe('DataStudioPrepareDatasetPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders split readiness and routes mutating work to Dataset Prep', async () => {
        apiMock.get.mockResolvedValueOnce({ data: preparePayload });
        const onOpenTarget = vi.fn();

        render(<DataStudioPrepareDatasetPanel projectId={1} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-prepare-dataset')).toBeInTheDocument();
        });

        expect(screen.getByText('Ready')).toBeInTheDocument();
        expect(screen.getByText('Text Classifier')).toBeInTheDocument();
        expect(screen.getByText('Pass')).toBeInTheDocument();
        expect(screen.getAllByText('120').length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText('Train')).toBeInTheDocument();
        expect(screen.getByText('Validation')).toBeInTheDocument();
        expect(screen.getByText('Test')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: /Open Dataset Prep/i }));
        expect(onOpenTarget).toHaveBeenCalledWith('dataprep');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/prepare-dataset');
    });

    it('shows blockers, review gates, and read-only status without preparing data', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...preparePayload,
                verdict: 'blocked',
                can_prepare: false,
                recipe: {
                    status: 'missing',
                    selected: null,
                    message: 'Choose a recipe to make split and adapter checks recipe-aware.',
                },
                mapping: {
                    ...preparePayload.mapping,
                    status: 'attention',
                    contract_pass: false,
                    required_fields_below_100: ['label'],
                },
                splits: {
                    ...preparePayload.splits,
                    status: 'missing',
                    total_prepared_rows: 0,
                    items: preparePayload.splits.items.map((item) => ({
                        ...item,
                        exists: false,
                        row_count: 0,
                        file_exists: false,
                        manifest_count: 0,
                        manifest_version: null,
                        version_count: 0,
                        latest_version: null,
                    })),
                },
                manifest: {
                    ...preparePayload.manifest,
                    status: 'missing',
                    exists: false,
                    readable: false,
                    total_entries: 0,
                    splits: {},
                    dataset_versions: {},
                },
                review_blockers: [
                    {
                        id: 'synthetic_pending_review',
                        label: 'Synthetic rows pending review',
                        count: 5,
                        severity: 'warning',
                        message: 'Pending generated rows are excluded from dataset prep until accepted.',
                        target_tab: 'synthetic',
                    },
                ],
                checks: [
                    {
                        id: 'recipe',
                        label: 'Recipe readiness',
                        status: 'missing',
                        message: 'Choose a recipe to make split and adapter checks recipe-aware.',
                        target_tab: 'data',
                    },
                    {
                        id: 'mapping_contract',
                        label: 'Mapping contract',
                        status: 'attention',
                        message: 'Adapter mapping needs review before creating prepared split files.',
                        target_tab: 'dataprep',
                    },
                ],
                issues: [
                    {
                        id: 'prepare_missing_recipe',
                        severity: 'blocker',
                        title: 'Recipe not selected',
                        message: 'Pick a recipe before preparing splits so BrewSLM knows the training shape.',
                        action_label: 'Choose recipe',
                        target_tab: 'data',
                    },
                ],
            },
        });
        const onOpenTarget = vi.fn();

        render(<DataStudioPrepareDatasetPanel projectId={2} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-prepare-dataset')).toBeInTheDocument();
        });

        expect(screen.getByText('Blocked')).toBeInTheDocument();
        expect(screen.getByText('No recipe')).toBeInTheDocument();
        expect(screen.getByText('Synthetic rows pending review')).toBeInTheDocument();
        expect(screen.getByText('Recipe not selected')).toBeInTheDocument();
        expect(screen.getByText('Read-only check')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: /Synthetic rows pending review/i }));
        expect(onOpenTarget).toHaveBeenCalledWith('synthetic');
    });

    // ─────────────────────────────────────────────────────────────────
    // Arc A — inline "Run prepare now" button. Verdict drives whether
    // it's enabled; click POSTs to /dataset/split with NO overrides so
    // the backend resolves ratios/adapter from project preferences.
    // ─────────────────────────────────────────────────────────────────

    it('enables "Run prepare now" when can_prepare is true', async () => {
        apiMock.get.mockResolvedValueOnce({ data: preparePayload });
        render(<DataStudioPrepareDatasetPanel projectId={1} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-prepare-run')).toBeInTheDocument();
        });
        const runBtn = screen.getByTestId('data-studio-prepare-run') as HTMLButtonElement;
        expect(runBtn.disabled).toBe(false);
        expect(runBtn.textContent).toMatch(/Run prepare now/);
    });

    it('passes seed + ratio overrides from the Configure-splits controls', async () => {
        apiMock.get.mockImplementation((url: string) => {
            if (url.includes('/prepare-dataset')) return Promise.resolve({ data: preparePayload });
            return Promise.resolve(coverageNA);
        });
        apiMock.post.mockResolvedValueOnce({ data: { train_count: 70, val_count: 20, test_count: 10 } });

        render(<DataStudioPrepareDatasetPanel projectId={7} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-prepare-run')).toBeInTheDocument();
        });

        fireEvent.click(screen.getByTestId('data-studio-prepare-config-toggle'));
        // Invalid sum (0.5 + 0.1 + 0.1) → client-side error, no POST.
        fireEvent.change(screen.getByLabelText('Train ratio'), { target: { value: '0.5' } });
        fireEvent.click(screen.getByTestId('data-studio-prepare-run'));
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-prepare-config-error')).toHaveTextContent(/sum to 1\.0/);
        });
        expect(apiMock.post).not.toHaveBeenCalled();

        // Fix the ratios + set a seed → POST carries the overrides.
        fireEvent.change(screen.getByLabelText('Train ratio'), { target: { value: '0.8' } });
        fireEvent.change(screen.getByLabelText('Seed'), { target: { value: '123' } });
        fireEvent.click(screen.getByTestId('data-studio-prepare-run'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/7/dataset/split', {
                seed: 123, train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1,
            });
        });
    });

    it('disables "Run prepare now" when can_prepare is false (blocked verdict)', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...preparePayload,
                verdict: 'blocked',
                can_prepare: false,
            },
        });
        render(<DataStudioPrepareDatasetPanel projectId={1} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-prepare-run')).toBeInTheDocument();
        });
        const runBtn = screen.getByTestId('data-studio-prepare-run') as HTMLButtonElement;
        expect(runBtn.disabled).toBe(true);
    });

    // Each mount also fires a best-effort split-class-coverage GET; mock it
    // (applicable:false) so it doesn't consume the readiness mock in the chain.
    const coverageNA = { data: { applicable: false, label_field: 'label', splits: {} } };
    const prepareCalls = () =>
        apiMock.get.mock.calls.filter((c: any[]) => String(c[0]).includes('prepare-dataset')).length;

    it('POSTs an empty split body and refreshes the panel on success', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: preparePayload }) // mount: readiness
            .mockResolvedValueOnce(coverageNA)               // mount: coverage
            .mockResolvedValueOnce({ data: preparePayload }) // refresh: readiness
            .mockResolvedValueOnce(coverageNA);              // refresh: coverage
        apiMock.post.mockResolvedValueOnce({
            data: { train_count: 96, val_count: 12, test_count: 12 },
        });

        render(<DataStudioPrepareDatasetPanel projectId={7} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-prepare-run')).toBeInTheDocument();
        });

        fireEvent.click(screen.getByTestId('data-studio-prepare-run'));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/dataset/split',
                {},
            );
        });
        await waitFor(() => {
            expect(
                screen.getByTestId('data-studio-prepare-run-flash'),
            ).toHaveTextContent(/Prepared 120 rows/);
        });
        // Refresh fetched the readiness payload again (twice total).
        await waitFor(() => expect(prepareCalls()).toBe(2));
    });

    it('surfaces a backend error inline without refreshing', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: preparePayload }) // mount: readiness
            .mockResolvedValueOnce(coverageNA);              // mount: coverage
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'recipe not selected' } },
        });

        render(<DataStudioPrepareDatasetPanel projectId={1} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-prepare-run')).toBeInTheDocument();
        });

        fireEvent.click(screen.getByTestId('data-studio-prepare-run'));
        await waitFor(() => {
            expect(
                screen.getByTestId('data-studio-prepare-run-error'),
            ).toHaveTextContent(/recipe not selected/);
        });
        // No refresh on failure — readiness fetched once.
        expect(prepareCalls()).toBe(1);
    });

    it('renders per-class coverage warnings when a split is missing a class', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: preparePayload })
            .mockResolvedValueOnce({
                data: {
                    applicable: true,
                    label_field: 'label',
                    class_count: 2,
                    splits: {
                        train: { prepared: true, total: 3, by_label: { billing: 2, technical: 1 } },
                        val: { prepared: true, total: 1, by_label: { technical: 1 } },
                        test: { prepared: true, total: 2, by_label: { billing: 1, technical: 1 } },
                    },
                    warnings: [
                        {
                            severity: 'warning', split: 'val', label: 'billing', train_count: 2,
                            message: 'Your val set has no “billing” examples — train has 2.',
                        },
                    ],
                },
            });

        render(<DataStudioPrepareDatasetPanel projectId={1} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-prepare-coverage')).toBeInTheDocument();
        });
        expect(screen.getByText(/no “billing” examples/)).toBeInTheDocument();
        // The coverage table lists both classes.
        expect(screen.getByText('billing')).toBeInTheDocument();
        expect(screen.getByText('technical')).toBeInTheDocument();
    });
});

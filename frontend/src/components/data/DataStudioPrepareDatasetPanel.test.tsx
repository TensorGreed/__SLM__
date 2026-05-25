import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
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
});

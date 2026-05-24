import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioMappingPreviewPanel from './DataStudioMappingPreviewPanel';

const mappingPayload = {
    project_id: 1,
    verdict: 'ready',
    recipe: {
        id: 'classification',
        name: 'Text Classifier',
        task_profile: 'classification',
        adapter_id: 'classification-label',
        default_input_column: 'text',
        default_output_column: 'label',
    },
    preference: {
        source: 'default',
        adapter_id: 'default-canonical',
        task_profile: null,
        field_mapping: {},
        field_mapping_count: 0,
    },
    effective_mapping: {
        source: 'recipe',
        adapter_id: 'classification-label',
        requested_adapter_id: 'classification-label',
        task_profile: 'classification',
        requested_task_profile: 'classification',
        adapter_config: {},
        field_mapping: {},
        auto_apply: {},
    },
    source: {
        dataset_type: 'raw',
        dataset_id: 10,
        dataset_name: 'Ticket exports',
        document_id: 20,
        document_name: 'tickets.jsonl',
        document_count: 1,
        row_count: 2,
    },
    summary: {
        sampled_records: 2,
        mapped_records: 2,
        dropped_records: 0,
        error_count: 0,
        mapping_success_rate: 1,
        contract_pass: true,
        required_fields: ['text', 'source_text', 'target_text', 'label'],
        required_fields_below_100: [],
        required_field_coverage: [
            { field: 'text', present: 2, missing: 0, ratio: 1 },
            { field: 'label', present: 2, missing: 0, ratio: 1 },
        ],
    },
    preview_rows: [
        {
            index: 0,
            raw: { text: 'Refund requested after renewal', label: 'billing' },
            mapped: {
                text: 'Refund requested after renewal',
                source_text: 'Refund requested after renewal',
                target_text: 'billing',
                label: 'billing',
            },
        },
    ],
    diagnostics: {
        adapter_contract: {},
        validation_report: {},
        detection_scores: { 'classification-label': 1 },
        auto_fix_suggestions: [],
        compatibility_warnings: [],
        inferred_task_profiles: ['classification'],
    },
    issues: [],
};

describe('DataStudioMappingPreviewPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders recipe-aware adapter, coverage, and canonical preview', async () => {
        apiMock.get.mockResolvedValueOnce({ data: mappingPayload });

        render(<DataStudioMappingPreviewPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-mapping')).toBeInTheDocument();
        });

        expect(screen.getByText('Ready')).toBeInTheDocument();
        expect(screen.getAllByText('classification-label').length).toBeGreaterThan(0);
        expect(screen.getAllByText('classification').length).toBeGreaterThan(0);
        expect(screen.getByText('2 / 2 mapped')).toBeInTheDocument();
        expect(screen.getAllByText('100%').length).toBeGreaterThan(0);
        expect(screen.getAllByText('tickets.jsonl', { exact: false }).length).toBeGreaterThan(0);
        expect(screen.getAllByText('billing', { exact: false }).length).toBeGreaterThan(0);
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/mapping-preview');
    });

    it('renders empty mapping guidance when no preview source exists', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...mappingPayload,
                verdict: 'empty',
                recipe: null,
                source: null,
                summary: {
                    ...mappingPayload.summary,
                    sampled_records: 0,
                    mapped_records: 0,
                    required_field_coverage: [],
                },
                preview_rows: [],
                issues: [
                    {
                        id: 'no_mapping_source',
                        severity: 'blocker',
                        title: 'No previewable rows',
                        message: 'Add an accepted raw document first.',
                        action_label: 'Add sources',
                        target_tab: 'data',
                    },
                ],
            },
        });

        render(<DataStudioMappingPreviewPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByText('No preview')).toBeInTheDocument();
        });

        expect(screen.getByText('No previewable rows')).toBeInTheDocument();
        expect(screen.getByText(/Canonical rows will appear/i)).toBeInTheDocument();
    });
});

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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
        raw_field_frequency: {
            text: 2,
            label: 2,
            category: 1,
        },
    },
    mapping_templates: {
        read_only: true,
        template_count: 2,
        recommended_template_id: 'recipe-classification',
        detected_fields: [
            { field: 'text', count: 2 },
            { field: 'label', count: 2 },
            { field: 'category', count: 1 },
        ],
        missing_field_count: 1,
        ambiguous_field_count: 1,
        templates: [
            {
                id: 'recipe-classification',
                label: 'Text Classifier recipe defaults',
                description: 'Uses the selected recipe columns as the starter mapping.',
                source: 'recipe',
                status: 'ready',
                recommended: true,
                confidence: 1,
                adapter_id: 'classification-label',
                task_profile: 'classification',
                field_mapping: {
                    text: 'text',
                    label: 'label',
                },
                fields: [
                    {
                        canonical_field: 'text',
                        recommended_source: 'text',
                        current_source: null,
                        status: 'available',
                        required: true,
                        detected_candidates: ['text'],
                        candidate_sources: ['text', 'content'],
                        note: 'recipe-defaults',
                    },
                    {
                        canonical_field: 'label',
                        recommended_source: 'label',
                        current_source: null,
                        status: 'available',
                        required: true,
                        detected_candidates: ['label'],
                        candidate_sources: ['label', 'category'],
                        note: 'recipe-defaults',
                    },
                ],
                summary: {
                    total_fields: 2,
                    applied_count: 0,
                    available_count: 2,
                    missing_count: 0,
                    ambiguous_count: 0,
                },
                apply_action: {
                    label: 'Open Data Prep to apply',
                    target_tab: 'dataprep',
                    requires_confirmation: true,
                    description: 'Review and save/apply this mapping in Data Prep. Data Studio does not mutate project mapping.',
                },
            },
            {
                id: 'domain-applied-contract',
                label: 'Applied domain contract',
                description: 'Uses aliases from the applied domain contract.',
                source: 'domain',
                status: 'attention',
                recommended: false,
                confidence: 0.6,
                adapter_id: 'classification-label',
                task_profile: 'classification',
                field_mapping: {
                    text: 'text',
                    label: 'category',
                    rationale: 'rationale',
                },
                fields: [
                    {
                        canonical_field: 'text',
                        recommended_source: 'text',
                        current_source: null,
                        status: 'ambiguous',
                        required: true,
                        detected_candidates: ['text', 'category'],
                        candidate_sources: ['text', 'category'],
                        note: 'domain-contract',
                    },
                    {
                        canonical_field: 'rationale',
                        recommended_source: 'rationale',
                        current_source: null,
                        status: 'missing',
                        required: true,
                        detected_candidates: [],
                        candidate_sources: ['rationale'],
                        note: 'domain-contract',
                    },
                ],
                summary: {
                    total_fields: 2,
                    applied_count: 0,
                    available_count: 0,
                    missing_count: 1,
                    ambiguous_count: 1,
                },
                apply_action: {
                    label: 'Open Data Prep to apply',
                    target_tab: 'dataprep',
                    requires_confirmation: true,
                    description: 'Review and save/apply this mapping in Data Prep. Data Studio does not mutate project mapping.',
                },
            },
        ],
        entry_points: [
            {
                label: 'Open Data Prep',
                target_tab: 'dataprep',
                reason: 'Save or apply mapping templates only after reviewing them in Data Prep.',
                requires_confirmation: true,
            },
        ],
    },
    issues: [],
};

describe('DataStudioMappingPreviewPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders recipe-aware adapter, coverage, and canonical preview', async () => {
        apiMock.get.mockResolvedValueOnce({ data: mappingPayload });
        const onOpenTarget = vi.fn();

        render(<DataStudioMappingPreviewPanel projectId={1} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-mapping')).toBeInTheDocument();
        });

        expect(screen.getAllByText('Ready').length).toBeGreaterThan(0);
        expect(screen.getAllByText('classification-label').length).toBeGreaterThan(0);
        expect(screen.getAllByText('classification').length).toBeGreaterThan(0);
        expect(screen.getByText('2 / 2 mapped')).toBeInTheDocument();
        expect(screen.getAllByText('100%').length).toBeGreaterThan(0);
        expect(screen.getAllByText('tickets.jsonl', { exact: false }).length).toBeGreaterThan(0);
        expect(screen.getAllByText('billing', { exact: false }).length).toBeGreaterThan(0);
        expect(screen.getByText('Mapping templates')).toBeInTheDocument();
        expect(screen.getAllByText('Text Classifier recipe defaults').length).toBeGreaterThan(0);
        expect(screen.getByText('Applied domain contract')).toBeInTheDocument();
        expect(screen.getByText(/Recommended:/)).toBeInTheDocument();
        expect(screen.getByText('3 detected fields')).toBeInTheDocument();
        expect(screen.getAllByText('rationale').length).toBeGreaterThan(0);

        fireEvent.click(screen.getAllByRole('button', { name: /Open Data Prep to apply/i })[0]);

        expect(onOpenTarget).toHaveBeenCalledWith('dataprep');
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
                mapping_templates: {
                    read_only: true,
                    template_count: 0,
                    recommended_template_id: null,
                    detected_fields: [],
                    missing_field_count: 0,
                    ambiguous_field_count: 0,
                    templates: [],
                    entry_points: [],
                },
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

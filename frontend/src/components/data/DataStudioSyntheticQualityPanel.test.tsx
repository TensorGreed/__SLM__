import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioSyntheticQualityPanel from './DataStudioSyntheticQualityPanel';

const qualityPayload = {
    project_id: 1,
    verdict: 'attention',
    read_only: true,
    auto_apply: false,
    source_of_truth: 'deterministic_synthetic_quality_checks',
    domain: {
        id: 'support_faq',
        label: 'Support FAQ',
        confidence: 0.86,
        source: 'sampled_data',
    },
    recipe: {
        id: 'classification',
        name: 'Classification',
        task_profile: 'classification',
        adapter_id: 'classification-label',
    },
    summary: {
        total_rows: 4,
        pending_rows: 2,
        accepted_rows: 2,
        rejected_rows: 0,
        source_count: 2,
        avg_confidence: 0.78,
        low_confidence_rows: 1,
        unknown_confidence_rows: 1,
        missing_required_rows: 1,
        duplicate_signal_rows: 2,
        low_quality_rows: 1,
        avg_gold_similarity: 0.42,
        high_gold_similarity_rows: 1,
        low_gold_similarity_rows: 1,
        gold_anchor_rows: 3,
    },
    quality_bands: {
        confidence: {
            high: 1,
            medium: 1,
            low: 1,
            unknown: 1,
            average: 0.78,
        },
        duplicates: {
            exact_duplicate_rows: 1,
            near_duplicate_pairs: 1,
            affected_rows: 2,
            ratio: 0.5,
        },
        required_fields: {
            required_fields: ['text', 'label'],
            missing_rows: 1,
            ratio: 0.25,
        },
        gold_similarity: {
            average: 0.42,
            high_overlap_rows: 1,
            low_similarity_rows: 1,
            gold_anchor_rows: 3,
        },
    },
    review_outcomes: {
        total_pending: 2,
        total_accepted: 2,
        top_pending_groups: [
            { synth_source: 'playbook:classification:positives_paraphrase', count: 2, truncated: false },
        ],
        top_accepted_groups: [
            { synth_source: 'playbook:classification:hard_negatives', count: 2, truncated: false },
        ],
    },
    source_groups: [
        {
            key: 'playbook-classification-positives-paraphrase',
            source: 'playbook:classification:positives_paraphrase',
            count: 2,
            pending: 2,
            accepted: 0,
            rejected: 0,
            other_status: 0,
            low_confidence: 1,
            unknown_confidence: 0,
            missing_required: 1,
            duplicate_signal_count: 1,
            avg_confidence: 0.66,
            avg_gold_similarity: 0.34,
            target_tab: 'synthetic',
        },
        {
            key: 'playbook-classification-hard-negatives',
            source: 'playbook:classification:hard_negatives',
            count: 2,
            pending: 0,
            accepted: 2,
            rejected: 0,
            other_status: 0,
            low_confidence: 0,
            unknown_confidence: 1,
            missing_required: 0,
            duplicate_signal_count: 1,
            avg_confidence: 0.9,
            avg_gold_similarity: 0.5,
            target_tab: 'synthetic',
        },
    ],
    status_groups: [
        { status: 'pending', label: 'Pending review', count: 2, target_tab: 'synthetic' },
        { status: 'accepted', label: 'Accepted', count: 2, target_tab: 'synthetic' },
        { status: 'rejected', label: 'Rejected', count: 0, target_tab: 'synthetic' },
    ],
    domain_groups: [
        {
            domain_id: 'support_faq',
            domain_label: 'Support FAQ',
            confidence: 0.86,
            synthetic_rows: 4,
            pending_rows: 2,
            accepted_rows: 2,
            source: 'sampled_data',
            target_tab: 'domain',
        },
    ],
    findings: [
        {
            id: 'synthetic_quality_pending_review',
            label: 'Pending synthetic review',
            severity: 'warning',
            status: 'attention',
            message: '2 synthetic rows are pending review before they can enter prepared datasets.',
            count: 2,
            target_tab: 'synthetic',
            workflow_owner: 'Synthetic Review',
            evidence: ['1 pending source group(s).'],
            action_label: 'Review synthetic rows',
        },
        {
            id: 'synthetic_quality_missing_required_fields',
            label: 'Missing required fields',
            severity: 'blocker',
            status: 'blocked',
            message: '1 synthetic row is missing required recipe fields.',
            count: 1,
            target_tab: 'dataprep',
            workflow_owner: 'Data Prep',
            evidence: ['Required fields: text, label.'],
            action_label: 'Review mapping',
        },
    ],
    preview_rows: [
        {
            source: 'playbook:classification:positives_paraphrase',
            source_type: 'synthetic',
            target_tab: 'synthetic',
            row_index: 0,
            file_name: 'synthetic.jsonl',
            redacted_text: 'Refund requested after renewal',
            fields: [
                { field: 'text', value: 'Refund requested after renewal' },
                { field: 'label', value: '(empty)' },
            ],
            reason: 'Synthetic quality analytics preview',
        },
    ],
    issues: [
        {
            id: 'synthetic_quality_pending_review',
            severity: 'warning',
            title: 'Pending synthetic review',
            message: '2 synthetic rows are pending review before they can enter prepared datasets.',
            action_label: 'Review synthetic rows',
            target_tab: 'synthetic',
        },
    ],
    entry_points: [
        {
            label: 'Open Synthetic review',
            target_tab: 'synthetic',
            reason: 'Accept, reject, or inspect synthetic rows.',
        },
        {
            label: 'Open Gold Set',
            target_tab: 'goldset',
            reason: 'Strengthen trusted anchors.',
        },
    ],
    assist: {
        available: true,
        read_only: true,
        status: 'not_invoked',
        default_provider: 'ollama',
        supported_providers: ['ollama', 'openai_compatible'],
        purpose: 'explanations_only',
        message: 'Synthetic quality analytics are deterministic by default.',
    },
    power_details: {},
};

describe('DataStudioSyntheticQualityPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders deterministic synthetic quality analytics and read-only workflow routes', async () => {
        apiMock.get.mockResolvedValueOnce({ data: qualityPayload });
        const onOpenTarget = vi.fn();

        render(<DataStudioSyntheticQualityPanel projectId={1} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-synthetic-quality')).toBeInTheDocument();
        });

        expect(screen.getByText('Needs review')).toBeInTheDocument();
        expect(screen.getByText('Synthetic quality analytics')).toBeInTheDocument();
        expect(screen.getByText('Support FAQ')).toBeInTheDocument();
        expect(screen.getByText('78%')).toBeInTheDocument();
        expect(screen.getByText('42%')).toBeInTheDocument();
        expect(screen.getByText('Pending synthetic review')).toBeInTheDocument();
        expect(screen.getByText('Missing required fields')).toBeInTheDocument();
        expect(screen.getAllByText('playbook:classification:positives_paraphrase').length).toBeGreaterThan(0);
        expect(screen.getByText('ollama explanations optional')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: /Review mapping/i }));
        expect(onOpenTarget).toHaveBeenCalledWith('dataprep');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/synthetic-quality');
    });

    it('renders an empty state when no synthetic rows exist', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...qualityPayload,
                verdict: 'empty',
                summary: {
                    ...qualityPayload.summary,
                    total_rows: 0,
                    pending_rows: 0,
                    accepted_rows: 0,
                    source_count: 0,
                },
                source_groups: [],
                findings: [],
                preview_rows: [],
            },
        });

        render(<DataStudioSyntheticQualityPanel projectId={1} onOpenTarget={vi.fn()} />);

        await waitFor(() => {
            expect(screen.getByText('No rows')).toBeInTheDocument();
        });
        expect(screen.getByText(/Source-level analytics appear/i)).toBeInTheDocument();
        expect(screen.getByText(/No synthetic quality findings/i)).toBeInTheDocument();
    });
});

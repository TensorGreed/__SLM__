import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioGoldSetWorkbenchPanel from './DataStudioGoldSetWorkbenchPanel';

const goldPayload = {
    project_id: 1,
    verdict: 'attention',
    read_only: true,
    minimum_recommended_examples: 5,
    validation: {
        status: 'needs_review',
        trusted_examples: 1,
        review_needed: 2,
        locked_gold_sets: 0,
        locked_versions: 0,
    },
    totals: {
        gold_set_count: 1,
        example_count: 3,
        trusted_examples: 1,
        review_needed: 2,
        approved_rows: 1,
        pending_rows: 1,
        in_review_rows: 0,
        changes_requested_rows: 1,
        rejected_rows: 0,
        queue_pending: 1,
        queue_in_progress: 1,
        locked_gold_sets: 0,
        draft_versions: 1,
        locked_versions: 0,
    },
    datasets: [
        {
            id: 3,
            name: 'Support Gold Dev',
            dataset_type: 'gold_dev',
            record_count: 3,
            example_count: 3,
            trusted_examples: 1,
            review_needed: 2,
            is_locked: false,
            validation_status: 'needs_review',
            coverage_source: 'workbench_rows',
            row_status_counts: {
                pending: 1,
                in_review: 0,
                approved: 1,
                rejected: 0,
                changes_requested: 1,
            },
            queue_status_counts: {
                pending: 1,
                in_progress: 1,
                completed: 0,
                skipped: 0,
            },
            versions: {
                count: 1,
                draft_count: 1,
                locked_count: 0,
                latest: { version: 1, status: 'draft' },
                active_draft: { version: 1, status: 'draft' },
                latest_locked: null,
            },
            coverage: {
                source_rows: 3,
                input_fields: [{ field: 'question', present: 3, missing: 0, ratio: 1 }],
                expected_fields: [{ field: 'answer', present: 3, missing: 0, ratio: 1 }],
                label_fields: [{ field: 'category', present: 3, missing: 0, ratio: 1 }],
                field_counts: { input: 1, expected: 1, labels: 1 },
            },
            updated_at: '2026-05-24T12:00:00Z',
        },
    ],
    trusted_examples: [
        {
            dataset_id: 3,
            dataset_name: 'Support Gold Dev',
            source: 'workbench_row',
            status: 'approved',
            input_preview: '{"question":"How do I reset my password?"}',
            expected_preview: '{"answer":"Use the password reset flow."}',
        },
    ],
    coverage: {
        source_rows: 3,
        input_fields: [{ field: 'question', present: 3, missing: 0, ratio: 1 }],
        expected_fields: [{ field: 'answer', present: 3, missing: 0, ratio: 1 }],
        label_fields: [{ field: 'category', present: 3, missing: 0, ratio: 1 }],
        field_counts: { input: 1, expected: 1, labels: 1 },
    },
    issues: [
        {
            id: 'gold_rows_need_review',
            severity: 'warning',
            title: 'Gold rows need review',
            message: '2 gold rows are pending, in review, or waiting on changes.',
            action_label: 'Review Gold Set',
            target_tab: 'goldset',
        },
    ],
    entry_point: {
        label: 'Open Gold Set workflow',
        target_tab: 'goldset',
        reason: 'Use the existing Gold Set panel to add, review, sample, or lock trusted examples.',
    },
};

describe('DataStudioGoldSetWorkbenchPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders Gold Set summary and routes into the existing workflow', async () => {
        apiMock.get.mockResolvedValueOnce({ data: goldPayload });
        const onOpenGoldSet = vi.fn();

        render(<DataStudioGoldSetWorkbenchPanel projectId={1} onOpenGoldSet={onOpenGoldSet} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-gold')).toBeInTheDocument();
        });

        expect(screen.getByText('Needs review')).toBeInTheDocument();
        expect(screen.getByText('1 / 5')).toBeInTheDocument();
        expect(screen.getAllByText('2').length).toBeGreaterThan(0);
        expect(screen.getAllByText('Support Gold Dev').length).toBeGreaterThan(0);
        expect(screen.getByText('Gold rows need review')).toBeInTheDocument();
        expect(screen.getAllByText(/password/i).length).toBeGreaterThan(0);

        fireEvent.click(screen.getByRole('button', { name: /Open Gold Set workflow/i }));
        expect(onOpenGoldSet).toHaveBeenCalledTimes(1);
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/gold-set');
    });

    it('renders the empty Gold Set entry state', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...goldPayload,
                verdict: 'empty',
                validation: {
                    ...goldPayload.validation,
                    status: 'empty',
                    trusted_examples: 0,
                    review_needed: 0,
                },
                totals: {
                    ...goldPayload.totals,
                    gold_set_count: 0,
                    example_count: 0,
                    trusted_examples: 0,
                    review_needed: 0,
                },
                datasets: [],
                trusted_examples: [],
                coverage: {
                    source_rows: 0,
                    input_fields: [],
                    expected_fields: [],
                    label_fields: [],
                    field_counts: { input: 0, expected: 0, labels: 0 },
                },
                issues: [
                    {
                        id: 'no_gold_sets',
                        severity: 'blocker',
                        title: 'No gold set yet',
                        message: 'Create a small trusted gold set before relying on evaluations.',
                        action_label: 'Open Gold Set',
                        target_tab: 'goldset',
                    },
                ],
            },
        });

        render(<DataStudioGoldSetWorkbenchPanel projectId={1} onOpenGoldSet={vi.fn()} />);

        await waitFor(() => {
            expect(screen.getByText('No gold set')).toBeInTheDocument();
        });
        expect(screen.getByText(/No Gold Set dataset has been created/i)).toBeInTheDocument();
        expect(screen.getByText('No gold set yet')).toBeInTheDocument();
    });
});

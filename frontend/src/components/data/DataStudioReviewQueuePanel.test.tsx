import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioReviewQueuePanel from './DataStudioReviewQueuePanel';

const reviewQueuePayload = {
    project_id: 1,
    verdict: 'attention',
    read_only: true,
    auto_apply: false,
    source_of_truth: 'deterministic_data_studio_checks',
    domain: {
        id: 'support_faq',
        label: 'Support FAQ',
        confidence: 0.86,
        source: 'sampled_data',
    },
    totals: {
        open_review_items: 6,
        accepted_or_promoted: 3,
        synthetic_pending: 2,
        synthetic_accepted: 1,
        gold_review_needed: 1,
        gold_trusted_examples: 1,
        annotation_jobs: 1,
        annotation_review_needed: 2,
        annotation_labeled: 2,
        annotation_labeled_unpromoted: 1,
        annotation_promoted: 1,
    },
    synthetic: {
        dataset_id: 9,
        total_rows: 3,
        total_pending: 2,
        total_accepted: 1,
        pending_group_count: 1,
        accepted_group_count: 1,
        top_pending_groups: [
            {
                synth_source: 'playbook:classification:positives_paraphrase',
                count: 2,
                truncated: false,
            },
        ],
        top_accepted_groups: [
            {
                synth_source: 'playbook:classification:hard_negatives',
                count: 1,
                truncated: false,
            },
        ],
    },
    gold_set: {
        validation: {
            status: 'needs_review',
            trusted_examples: 1,
            review_needed: 1,
            locked_gold_sets: 0,
            locked_versions: 0,
        },
        totals: {
            review_needed: 1,
            trusted_examples: 1,
        },
        datasets: [],
    },
    annotation: {
        totals: {
            job_count: 1,
            assigned: 1,
            unlabeled: 1,
            labeled: 2,
            labeled_unpromoted: 1,
            promoted: 1,
        },
        jobs: [
            {
                id: 12,
                name: 'Support annotation pass',
                label_type: 'classification',
                status: 'active',
                target_rows: 5,
                total: 4,
                assigned: 1,
                unlabeled: 1,
                labeled: 2,
                labeled_unpromoted: 1,
                promoted: 1,
                review_needed: 2,
                updated_at: '2026-05-24T12:00:00Z',
            },
        ],
    },
    triage: [
        {
            id: 'review_pending_synthetic_rows',
            title: 'Review pending synthetic rows',
            priority: 'high',
            count: 2,
            message: 'Accept good rows or reject weak rows before they enter the next prepared dataset.',
            action_label: 'Open Synthetic review',
            target_tab: 'synthetic',
            requires_user_confirmation: true,
            evidence: ['playbook:classification:positives_paraphrase (2)'],
        },
        {
            id: 'promote_labeled_annotation_rows',
            title: 'Promote labeled annotation rows',
            priority: 'medium',
            count: 1,
            message: 'Labels remain advisory until promoted.',
            action_label: 'Open Annotation',
            target_tab: 'annotate',
            requires_user_confirmation: true,
            evidence: ['2 labeled row(s).', '1 promoted row(s).'],
        },
    ],
    groupings: {
        by_source: [
            {
                key: 'synthetic:pending:playbook:classification:positives_paraphrase',
                label: 'playbook:classification:positives_paraphrase',
                kind: 'synthetic',
                status: 'pending',
                count: 2,
                target_tab: 'synthetic',
            },
            {
                key: 'annotation:12:promotion',
                label: 'Support annotation pass',
                kind: 'annotation',
                status: 'needs_promotion',
                count: 1,
                target_tab: 'annotate',
            },
        ],
        by_status: [
            {
                status: 'synthetic_pending',
                label: 'Synthetic pending review',
                count: 2,
                target_tab: 'synthetic',
                kind: 'synthetic',
            },
            {
                status: 'annotation_needs_promotion',
                label: 'Annotation needs promotion',
                count: 1,
                target_tab: 'annotate',
                kind: 'annotation',
            },
        ],
        by_domain: [
            {
                domain_id: 'support_faq',
                domain_label: 'Support FAQ',
                confidence: 0.86,
                open_review_items: 6,
                accepted_or_promoted: 3,
                source: 'sampled_data',
            },
        ],
    },
    issues: [
        {
            id: 'review_queue_synthetic_pending',
            severity: 'warning',
            title: 'Synthetic rows need review',
            message: '2 synthetic row(s) are pending accept/reject before dataset prep can use them.',
            action_label: 'Review synthetic rows',
            target_tab: 'synthetic',
        },
    ],
    entry_points: [
        {
            label: 'Open Synthetic review',
            target_tab: 'synthetic',
            reason: 'Accept or reject pending synthetic rows.',
        },
        {
            label: 'Open Annotation workspace',
            target_tab: 'annotate',
            reason: 'Label, skip, or promote annotation rows.',
        },
    ],
    power_details: {},
};

describe('DataStudioReviewQueuePanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders cross-workflow review triage and routes to existing actions', async () => {
        apiMock.get.mockResolvedValueOnce({ data: reviewQueuePayload });
        const onOpenTarget = vi.fn();

        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-review-queue')).toBeInTheDocument();
        });

        expect(screen.getByText('Needs review')).toBeInTheDocument();
        expect(screen.getByText('Support FAQ')).toBeInTheDocument();
        expect(screen.getByText('86% domain confidence')).toBeInTheDocument();
        expect(screen.getByText('Review pending synthetic rows')).toBeInTheDocument();
        expect(screen.getByText('Promote labeled annotation rows')).toBeInTheDocument();
        expect(screen.getByText('Support annotation pass')).toBeInTheDocument();
        expect(screen.getByText('Synthetic rows need review')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: /Open Annotation workspace/i }));
        expect(onOpenTarget).toHaveBeenCalledWith('annotate');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/review-queue');
    });

    it('renders the empty review queue state without mutating', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...reviewQueuePayload,
                verdict: 'empty',
                totals: {
                    ...reviewQueuePayload.totals,
                    open_review_items: 0,
                    accepted_or_promoted: 0,
                    synthetic_pending: 0,
                    synthetic_accepted: 0,
                    gold_review_needed: 0,
                    gold_trusted_examples: 0,
                    annotation_jobs: 0,
                    annotation_review_needed: 0,
                    annotation_labeled: 0,
                    annotation_labeled_unpromoted: 0,
                    annotation_promoted: 0,
                },
                triage: [
                    {
                        id: 'create_review_source',
                        title: 'Create a review source',
                        priority: 'low',
                        count: 0,
                        message: 'Review queues appear after synthetic generation, Gold Set sampling, or annotation seeding.',
                        action_label: 'Open Synthetic',
                        target_tab: 'synthetic',
                        requires_user_confirmation: true,
                        evidence: [],
                    },
                ],
                groupings: {
                    by_source: [],
                    by_status: [],
                    by_domain: reviewQueuePayload.groupings.by_domain,
                },
                issues: [
                    {
                        id: 'review_queue_no_review_sources',
                        severity: 'info',
                        title: 'No review queue yet',
                        message: 'Generate synthetic rows, add Gold Set examples, or create annotation jobs to start review flow.',
                        action_label: 'Open Synthetic',
                        target_tab: 'synthetic',
                    },
                ],
            },
        });
        const onOpenTarget = vi.fn();

        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByText('No queue')).toBeInTheDocument();
        });

        expect(screen.getByText('Create a review source')).toBeInTheDocument();
        expect(screen.getByText('No review queue yet')).toBeInTheDocument();

        fireEvent.click(screen.getAllByRole('button', { name: /^Open Synthetic$/i })[0]);
        expect(onOpenTarget).toHaveBeenCalledWith('synthetic');
    });
});

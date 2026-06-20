import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
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
                synth_source: 'playbook:classification:positives_paraphrase',
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
        apiMock.post.mockReset();
    });

    it('bulk-accepts a synthetic pending group in place and refreshes', async () => {
        // Route GETs by URL so the best-effort cards (active-learning,
        // label-noise, prepared-version-preview) can't shift a strict chain.
        apiMock.get.mockImplementation((url: string) => {
            if (url.includes('/data-studio/review-queue')) {
                return Promise.resolve({ data: reviewQueuePayload });
            }
            return Promise.reject(new Error('not mocked (best-effort)'));
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                accepted: 2, rejected: 0, not_found: 0, not_pending: 0,
                total_remaining_pending: 0,
                source: 'playbook:classification:positives_paraphrase', matched: 2,
            },
        });
        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={vi.fn()} />);

        const acceptBtn = await screen.findByRole('button', { name: /Accept all \(2\)/i });
        fireEvent.click(acceptBtn);

        // Hits the bulk-update-by-source endpoint with the group's synth_source.
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/review-queue/bulk-update-by-source',
                {
                    source: 'playbook:classification:positives_paraphrase',
                    action: 'accept',
                    reject_reason: null,
                },
            );
        });
        // Result flash (from the action result) + the queue is refetched.
        await waitFor(() => {
            expect(screen.getByTestId('bulk-flash')).toHaveTextContent(/Accepted 2 rows/i);
        });
        const reviewQueueCalls = apiMock.get.mock.calls.filter(
            (c: any[]) => String(c[0]).includes('/data-studio/review-queue'),
        ).length;
        expect(reviewQueueCalls).toBeGreaterThanOrEqual(2); // mount + reload
    });

    it('renders the "what version will include this" preview', async () => {
        apiMock.get.mockImplementation((url: string) => {
            if (url.includes('/data-studio/review-queue')) {
                return Promise.resolve({ data: reviewQueuePayload });
            }
            if (url.includes('/prepared-version-preview')) {
                return Promise.resolve({
                    data: {
                        project_id: 1,
                        next_version: 3,
                        has_existing_versions: true,
                        staged: { synthetic_accepted: 12, synthetic_pending: 5, gold: 20, cleaned: 80 },
                        trainable_total: 112,
                    },
                });
            }
            return Promise.reject(new Error('not mocked'));
        });
        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={vi.fn()} />);
        const card = await screen.findByTestId('version-preview');
        expect(card).toHaveTextContent(/next prepared dataset/i);
        expect(card).toHaveTextContent(/v3/);
        expect(card).toHaveTextContent(/12/);   // accepted synthetic
        expect(card).toHaveTextContent(/5 pending rows excluded/i);
    });

    it('does not render bulk actions for non-actionable groups (no synth_source)', async () => {
        apiMock.get
            .mockResolvedValueOnce({
                data: {
                    ...reviewQueuePayload,
                    groupings: {
                        ...reviewQueuePayload.groupings,
                        by_source: [
                            {
                                key: 'annotation:12:promotion',
                                label: 'Support annotation pass',
                                kind: 'annotation',
                                status: 'needs_promotion',
                                count: 1,
                                target_tab: 'annotate',
                            },
                        ],
                    },
                },
            })
            .mockRejectedValueOnce(new Error('no AL'))
            .mockRejectedValueOnce(new Error('no noise'));
        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-review-queue')).toBeInTheDocument();
        });
        expect(screen.queryByRole('button', { name: /Accept all/i })).toBeNull();
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

    // ────────────────────────────────────────────────────────────────────
    // Quality-Lift phase 3 slice 3 — Active-learning card
    // ────────────────────────────────────────────────────────────────────

    const buildAlSnapshot = (overrides: Record<string, unknown> = {}) => ({
        project_id: 1,
        snapshot: {
            scored_at: '2026-06-09T12:00:00Z',
            model_experiment_id: 42,
            task_type: 'classification',
            uncertainty_metric: 'entropy',
            pool_size_total: 2000,
            pool_size_scored: 500,
            skipped_reason: null,
            top_k: [
                { label_row_id: 101, label_job_id: 7, uncertainty_score: 1.32, text_preview: 'is this a benign or attack?', labeled: false },
                { label_row_id: 102, label_job_id: 7, uncertainty_score: 1.28, text_preview: 'classify the following …', labeled: false },
                { label_row_id: 103, label_job_id: 7, uncertainty_score: 1.10, text_preview: 'previously-labeled row', labeled: true },
                { label_row_id: 104, label_job_id: 7, uncertainty_score: 0.95, text_preview: null, labeled: false },
                { label_row_id: 105, label_job_id: 7, uncertainty_score: 0.90, text_preview: 'last visible preview', labeled: false },
            ],
        },
        experiment_id: 42,
        experiment_name: 'exp-42',
        top_k_size: 5,
        labeled_count: 1,
        unlabeled_count: 4,
        staleness_ratio: 0.2,
        is_stale: false,
        no_snapshot_reason: null,
        staleness_threshold: 0.8,
        dominant_label_job_id: 7,
        ...overrides,
    });

    it('renders the active-learning card with top-K rows + provenance + sample fraction', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: reviewQueuePayload })
            .mockResolvedValueOnce({ data: buildAlSnapshot() });
        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={vi.fn()} />);

        await waitFor(() => {
            expect(screen.getByText(/Active-learning queue/)).toBeInTheDocument();
        });
        // Provenance: experiment id surfaced so user can correlate.
        expect(screen.getByText(/scored by exp #42/)).toBeInTheDocument();
        // Sample fraction: 500 of 2000 = 25%.
        expect(screen.getByText(/sampled 500 of 2000/)).toBeInTheDocument();
        expect(screen.getByText(/25%/)).toBeInTheDocument();
        // First two rows visible by preview text.
        expect(screen.getByText(/is this a benign or attack/)).toBeInTheDocument();
        expect(screen.getByText(/classify the following/)).toBeInTheDocument();
        // Missing-text row falls back to "(no text)".
        expect(screen.getByText(/no text/)).toBeInTheDocument();
        // Open label queue button surfaces.
        expect(screen.getByRole('button', { name: /Open label queue/i })).toBeInTheDocument();
        // Confirms the two fetches.
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/active-learning/latest');
    });

    it('renders a quiet empty state when no snapshot exists', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: reviewQueuePayload })
            .mockResolvedValueOnce({
                data: buildAlSnapshot({
                    snapshot: null,
                    top_k_size: 0,
                    experiment_id: null,
                    experiment_name: null,
                    dominant_label_job_id: null,
                    no_snapshot_reason: 'no_completed_experiment_with_snapshot',
                }),
            });
        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={vi.fn()} />);

        await waitFor(() => {
            expect(screen.getByText(/Active-learning queue/)).toBeInTheDocument();
        });
        // Friendly explainer instead of the row table.
        expect(
            screen.getByText(/No completed training run has scored/i),
        ).toBeInTheDocument();
        // The row table should not render. Match the AL-card-specific
        // ``sampled N of M`` string rather than a loose ``sampled``
        // regex that would match anywhere in the panel.
        expect(screen.queryByText(/sampled \d+ of \d+/)).toBeNull();
        expect(screen.queryByRole('button', { name: /Open label queue/i })).toBeNull();
    });

    it('renders the stale state advisory when ≥80% of the snapshot is labeled', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: reviewQueuePayload })
            .mockResolvedValueOnce({
                data: buildAlSnapshot({
                    labeled_count: 4,
                    unlabeled_count: 1,
                    staleness_ratio: 0.8,
                    is_stale: true,
                }),
            });
        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={vi.fn()} />);

        await waitFor(() => {
            expect(screen.getByText(/Active-learning queue/)).toBeInTheDocument();
        });
        expect(
            screen.getByText(/consider re-training to score a fresh batch/i),
        ).toBeInTheDocument();
    });

    // ────────────────────────────────────────────────────────────────────
    // Quality-Lift phase 4 slice 2 — Suspected-mislabels card
    // ────────────────────────────────────────────────────────────────────

    const buildNoiseScan = (overrides: Record<string, unknown> = {}) => ({
        id: 9,
        project_id: 1,
        base_experiment_id: 42,
        status: 'succeeded' as const,
        label_count_at_scan: 387,
        suspected_count: 12,
        confidence_threshold: 0.85,
        given_label_floor: 0.15,
        result_payload: {
            scored_at: '2026-06-09T12:00:00Z',
            base_experiment_id: 42,
            label_count_total: 387,
            label_count_scored: 387,
            suspected_count: 12,
            confidence_threshold: 0.85,
            given_label_floor: 0.15,
            skipped_reason: null,
            top_k: [
                { label_row_id: 1234, label_job_id: 7, given_label: 'benign', predicted_label: 'attack', predicted_prob: 0.92, given_label_prob: 0.06, mislabel_score: 0.86, text_preview: 'this looks like an obvious attack to me' },
                { label_row_id: 5678, label_job_id: 7, given_label: 'A', predicted_label: 'B', predicted_prob: 0.88, given_label_prob: 0.08, mislabel_score: 0.80, text_preview: 'classify the following …' },
                { label_row_id: 9012, label_job_id: 7, given_label: 'A', predicted_label: 'B', predicted_prob: 0.86, given_label_prob: 0.10, mislabel_score: 0.76, text_preview: null },
            ],
        },
        error: null,
        job_id: 11,
        created_at: '2026-06-09T11:00:00Z',
        completed_at: '2026-06-09T12:00:00Z',
        ...overrides,
    });

    const noScanReviewQueue = () =>
        apiMock.get
            .mockResolvedValueOnce({ data: reviewQueuePayload })  // /review-queue
            .mockResolvedValueOnce({                              // /active-learning/latest
                data: buildAlSnapshot({ snapshot: null, top_k_size: 0 }),
            });

    it('renders the suspected-mislabels card with given/predicted badges + open-review button', async () => {
        noScanReviewQueue();
        apiMock.get.mockResolvedValueOnce({  // /label-noise/latest
            data: { project_id: 1, scan: buildNoiseScan(), no_scan_reason: null },
        });
        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={vi.fn()} />);

        await waitFor(() => {
            expect(screen.getByText(/Suspected mislabels/)).toBeInTheDocument();
        });
        // Provenance — scan id + base experiment id surface.
        expect(screen.getByText(/scan #9/)).toBeInTheDocument();
        expect(screen.getByText(/exp #42/)).toBeInTheDocument();
        // Meta line — threshold + floor as percentages.
        expect(screen.getByText(/confidence ≥ 85%/)).toBeInTheDocument();
        expect(screen.getByText(/given label ≤ 15%/)).toBeInTheDocument();
        // First row's given/predicted badges + text preview.
        expect(screen.getByText('benign')).toBeInTheDocument();
        expect(screen.getAllByText('attack').length).toBeGreaterThan(0);
        expect(screen.getByText(/this looks like an obvious attack/)).toBeInTheDocument();
        // Missing-text row → "(no text)".
        expect(screen.getByText(/no text/i)).toBeInTheDocument();
        // Open-review button is present.
        expect(
            screen.getByRole('button', { name: /Review suspected mislabels/i }),
        ).toBeInTheDocument();
        // Three label-noise endpoint reads confirmed.
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/label-noise/latest');
    });

    it('renders the clean state when a scan ran but found no suspects', async () => {
        noScanReviewQueue();
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 1,
                scan: buildNoiseScan({
                    suspected_count: 0,
                    result_payload: {
                        ...buildNoiseScan().result_payload,
                        suspected_count: 0,
                        top_k: [],
                    },
                }),
                no_scan_reason: null,
            },
        });
        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByText(/Suspected mislabels/)).toBeInTheDocument();
        });
        // Win-condition copy — no review button rendered.
        expect(
            screen.getByText(/your labels look clean/i),
        ).toBeInTheDocument();
        expect(
            screen.queryByRole('button', { name: /Review suspected mislabels/i }),
        ).toBeNull();
    });

    it('renders the empty state when no scan has ever run', async () => {
        noScanReviewQueue();
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 1, scan: null, no_scan_reason: 'no_succeeded_scan_yet' },
        });
        render(<DataStudioReviewQueuePanel projectId={1} onOpenTarget={vi.fn()} />);
        await waitFor(() => {
            expect(screen.getByText(/Suspected mislabels/)).toBeInTheDocument();
        });
        expect(
            screen.getByText(/Cleaning Coach will nudge when it's worth scanning/i),
        ).toBeInTheDocument();
        // No review button on the empty state either.
        expect(
            screen.queryByRole('button', { name: /Review suspected mislabels/i }),
        ).toBeNull();
    });
});

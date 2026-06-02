import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import SynthReviewQueue from './SynthReviewQueue';


const SAMPLE_PAYLOAD = {
    project_id: 1,
    dataset_id: 12,
    total_rows: 3,
    total_pending: 3,
    total_accepted: 0,
    groups: [
        {
            synth_source: 'playbook:classification:positives_paraphrase',
            count: 2,
            truncated: false,
            rows: [
                { id: 1, synth_confidence: 0.9, preview: '{"text": "row a", "label": "billing"}', payload: { text: 'row a', label: 'billing' } },
                { id: 2, synth_confidence: 0.85, preview: '{"text": "row b", "label": "billing"}', payload: { text: 'row b', label: 'billing' } },
            ],
        },
        {
            synth_source: 'playbook:classification:hard_negatives:vs=billing',
            count: 1,
            truncated: false,
            rows: [
                { id: 3, synth_confidence: 0.95, preview: '{"text": "row c", "label": "technical"}', payload: { text: 'row c', label: 'technical' } },
            ],
        },
    ],
    accepted_groups: [],
};


describe('SynthReviewQueue', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders the empty state when no rows are pending or accepted', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                dataset_id: null,
                total_rows: 0,
                total_pending: 0,
                total_accepted: 0,
                groups: [],
                accepted_groups: [],
            },
        });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-empty')).toBeInTheDocument();
        });
    });

    it('renders the totals strip with all 3 numbers', async () => {
        apiMock.get.mockResolvedValue({
            data: { ...SAMPLE_PAYLOAD, total_rows: 8, total_accepted: 5, total_pending: 3 },
        });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-totals')).toBeInTheDocument();
        });
        // Each cell shows its number prominently.
        const totalRowsCell = screen.getByTestId('synth-review-queue-total-rows');
        expect(totalRowsCell.textContent).toMatch(/8/);
        expect(totalRowsCell.textContent).toMatch(/total in synthetic.jsonl/i);
        const totalAcceptedCell = screen.getByTestId('synth-review-queue-total-accepted');
        expect(totalAcceptedCell.textContent).toMatch(/5/);
        expect(totalAcceptedCell.textContent).toMatch(/queued for training/i);
        const totalPendingCell = screen.getByTestId('synth-review-queue-total-pending');
        expect(totalPendingCell.textContent).toMatch(/3/);
        expect(totalPendingCell.textContent).toMatch(/awaiting review/i);
    });

    it('renders the "queued for training" summary when only accepted rows exist', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                dataset_id: 12,
                total_rows: 5,
                total_pending: 0,
                total_accepted: 5,
                groups: [],
                accepted_groups: [
                    { synth_source: 'playbook:qa-sft:positives_paraphrase', count: 3, truncated: false, rows: [] },
                    { synth_source: 'playbook:qa-sft:cluster_targeted:cluster=cluster-2', count: 2, truncated: false, rows: [] },
                ],
            },
        });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue')).toBeInTheDocument();
        });
        const root = screen.getByTestId('synth-review-queue');
        expect(root.textContent).toMatch(/5 rows accepted/);
        const accepted = screen.getByTestId('synth-review-queue-accepted');
        expect(accepted).toBeInTheDocument();
        expect(
            screen.getByTestId('synth-review-queue-accepted-group-playbook:qa-sft:positives_paraphrase'),
        ).toBeInTheDocument();
        expect(
            screen.getByTestId('synth-review-queue-accepted-group-playbook:qa-sft:cluster_targeted:cluster=cluster-2'),
        ).toBeInTheDocument();
    });

    it('mentions the accepted-count alongside pending count when both exist', async () => {
        apiMock.get.mockResolvedValue({
            data: { ...SAMPLE_PAYLOAD, total_rows: 7, total_accepted: 4, accepted_groups: [
                { synth_source: 'playbook:classification:positives_paraphrase', count: 4, truncated: false, rows: [] },
            ] },
        });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue')).toBeInTheDocument();
        });
        const root = screen.getByTestId('synth-review-queue');
        expect(root.textContent).toMatch(/3 rows awaiting review/);
        expect(root.textContent).toMatch(/4 already accepted/);
        expect(screen.getByTestId('synth-review-queue-accepted')).toBeInTheDocument();
    });

    it('renders accepted-group rows + a truncated footer for capped groups', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                dataset_id: 12,
                total_rows: 1000,
                total_pending: 0,
                total_accepted: 1000,
                groups: [],
                accepted_groups: [
                    {
                        synth_source: 'legacy:teacher_model',
                        count: 1000,
                        truncated: true,
                        rows: [
                            { id: 1, synth_confidence: 0.9, preview: '{"question": "Q1"}', payload: { question: 'Q1' } },
                            { id: 2, synth_confidence: 0.85, preview: '{"question": "Q2"}', payload: { question: 'Q2' } },
                        ],
                    },
                ],
            },
        });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-accepted')).toBeInTheDocument();
        });
        const group = screen.getByTestId('synth-review-queue-accepted-group-legacy:teacher_model');
        expect(group.textContent).toMatch(/legacy:teacher_model/);
        expect(group.textContent).toMatch(/1000/);
        // Once expanded (via the rendered <details>), the rows should be in the DOM.
        expect(screen.getByTestId('synth-review-queue-accepted-row-1')).toBeInTheDocument();
        expect(screen.getByTestId('synth-review-queue-accepted-row-2')).toBeInTheDocument();
        // Truncated footer is present.
        expect(group.textContent).toMatch(/Showing 2 of 1000/);
    });

    it('renders the queue grouped by synth_source with bulk action buttons disabled when nothing is selected', async () => {
        apiMock.get.mockResolvedValue({ data: SAMPLE_PAYLOAD });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue')).toBeInTheDocument();
        });
        // Both groups render.
        expect(screen.getByTestId('synth-review-queue-group-playbook:classification:positives_paraphrase')).toBeInTheDocument();
        expect(screen.getByTestId('synth-review-queue-group-playbook:classification:hard_negatives:vs=billing')).toBeInTheDocument();
        // Accept/reject disabled.
        expect(screen.getByTestId('synth-review-queue-accept')).toBeDisabled();
        expect(screen.getByTestId('synth-review-queue-reject')).toBeDisabled();
    });

    it('selecting individual rows enables bulk actions and posts the chosen IDs on accept', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: SAMPLE_PAYLOAD })
            .mockResolvedValueOnce({
                data: { ...SAMPLE_PAYLOAD, total_pending: 2 },
            });
        apiMock.post.mockResolvedValue({
            data: { accepted: 1, rejected: 0, not_found: 0, not_pending: 0, total_remaining_pending: 2 },
        });

        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-row-1')).toBeInTheDocument();
        });

        // Click row 1's checkbox.
        const row1 = screen.getByTestId('synth-review-queue-row-1');
        const cb1 = row1.querySelector('input[type="checkbox"]') as HTMLInputElement;
        await userEvent.click(cb1);
        expect(cb1.checked).toBe(true);

        const acceptBtn = screen.getByTestId('synth-review-queue-accept');
        expect(acceptBtn).not.toBeDisabled();
        expect(acceptBtn.textContent).toContain('1');

        await userEvent.click(acceptBtn);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-flash').textContent).toMatch(/Accepted 1 row/);
        });

        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/1/synthetic/review-queue/bulk-update',
            { row_ids: [1], action: 'accept', reject_reason: null },
        );
    });

    it('group checkbox selects all rows in that group', async () => {
        apiMock.get.mockResolvedValue({ data: SAMPLE_PAYLOAD });
        render(<SynthReviewQueue projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue')).toBeInTheDocument();
        });

        const group = screen.getByTestId('synth-review-queue-group-playbook:classification:positives_paraphrase');
        const groupCheckbox = group.querySelector('.synth-review-queue__group-toggle input[type="checkbox"]') as HTMLInputElement;

        await userEvent.click(groupCheckbox);
        // Both rows in this group should now be selected. The bulk
        // action button reflects the count (2).
        expect(screen.getByTestId('synth-review-queue-accept').textContent).toContain('2');
    });

    it('clicking reject posts the action=reject body', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: SAMPLE_PAYLOAD })
            .mockResolvedValueOnce({
                data: { ...SAMPLE_PAYLOAD, total_pending: 2, groups: SAMPLE_PAYLOAD.groups.slice(1) },
            });
        apiMock.post.mockResolvedValue({
            data: { accepted: 0, rejected: 1, not_found: 0, not_pending: 0, total_remaining_pending: 2 },
        });

        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-row-1')).toBeInTheDocument();
        });

        const row1 = screen.getByTestId('synth-review-queue-row-1');
        const cb1 = row1.querySelector('input[type="checkbox"]') as HTMLInputElement;
        await userEvent.click(cb1);
        await userEvent.click(screen.getByTestId('synth-review-queue-reject'));

        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-flash').textContent).toMatch(/Rejected 1 row/);
        });
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/1/synthetic/review-queue/bulk-update',
            { row_ids: [1], action: 'reject', reject_reason: null },
        );
    });

    // ── Focused-source banner (Phase 5c — Coach landing) ────────────

    it('renders the focused-source banner with the right row count when focusSource matches a group', async () => {
        apiMock.get.mockResolvedValue({ data: SAMPLE_PAYLOAD });
        render(
            <SynthReviewQueue
                projectId={1}
                focusSource="playbook:classification:positives_paraphrase"
            />,
        );
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-focus-banner')).toBeInTheDocument();
        });
        // Source label is rendered exactly.
        expect(
            screen.getByTestId('synth-review-queue-focus-source').textContent,
        ).toBe('playbook:classification:positives_paraphrase');
        // 2 rows in the matching group → button copy reflects the count.
        const acceptAll = screen.getByTestId('synth-review-queue-focus-accept-all');
        expect(acceptAll.textContent).toMatch(/Accept all 2 rows/i);
        expect(acceptAll).not.toBeDisabled();
    });

    it('hides the focused banner when no focusSource prop is set', async () => {
        apiMock.get.mockResolvedValue({ data: SAMPLE_PAYLOAD });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue')).toBeInTheDocument();
        });
        expect(
            screen.queryByTestId('synth-review-queue-focus-banner'),
        ).not.toBeInTheDocument();
    });

    it('focus banner with no matching group renders an empty state + disabled button', async () => {
        // Defensive: Coach can race with a queue mutation, so the
        // focusSource may no longer match any pending group by the
        // time the user lands. Render the banner with "Nothing to
        // accept" rather than crashing or silently dropping it.
        apiMock.get.mockResolvedValue({ data: SAMPLE_PAYLOAD });
        render(
            <SynthReviewQueue
                projectId={1}
                focusSource="playbook:does-not-exist"
            />,
        );
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-focus-banner')).toBeInTheDocument();
        });
        const acceptAll = screen.getByTestId('synth-review-queue-focus-accept-all');
        expect(acceptAll.textContent).toMatch(/Nothing to accept/i);
        expect(acceptAll).toBeDisabled();
    });

    it('one-click Accept all posts exactly the focused group ids + dismisses the banner', async () => {
        // Round-trip: the bulk-accept endpoint is called with ONLY the
        // ids of the focused group (not all pending rows — important
        // so the user doesn't accidentally accept a sibling source).
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_PAYLOAD });
        // After accept, the queue refetches → return a payload with
        // the focused source drained.
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...SAMPLE_PAYLOAD,
                total_pending: 1,
                groups: [SAMPLE_PAYLOAD.groups[1]],  // only hard_negatives left
            },
        });
        apiMock.post.mockResolvedValue({
            data: {
                accepted: 2,
                rejected: 0,
                total_remaining_pending: 1,
            },
        });
        render(
            <SynthReviewQueue
                projectId={1}
                focusSource="playbook:classification:positives_paraphrase"
            />,
        );
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-focus-accept-all')).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('synth-review-queue-focus-accept-all'),
        );
        await waitFor(() => {
            // Only the ids in the focused group (1, 2) — NOT 3 from
            // the sibling hard_negatives group.
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/review-queue/bulk-update',
                { row_ids: [1, 2], action: 'accept', reject_reason: null },
            );
        });
        // Banner auto-dismisses once the source bucket is drained.
        await waitFor(() => {
            expect(
                screen.queryByTestId('synth-review-queue-focus-banner'),
            ).not.toBeInTheDocument();
        });
        // Flash names the focused source so the user knows what shipped.
        expect(screen.getByTestId('synth-review-queue-flash').textContent).toMatch(
            /Accepted 2 rows from playbook:classification:positives_paraphrase/,
        );
    });

    it('Clear focus button dismisses the banner without firing an API call', async () => {
        apiMock.get.mockResolvedValue({ data: SAMPLE_PAYLOAD });
        render(
            <SynthReviewQueue
                projectId={1}
                focusSource="playbook:classification:positives_paraphrase"
            />,
        );
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-focus-banner')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('synth-review-queue-focus-clear'));
        await waitFor(() => {
            expect(
                screen.queryByTestId('synth-review-queue-focus-banner'),
            ).not.toBeInTheDocument();
        });
        // No bulk-update POST.
        expect(apiMock.post).not.toHaveBeenCalled();
    });

    it('shows an error + retry when the list endpoint fails', async () => {
        apiMock.get.mockRejectedValueOnce({ response: { status: 500, data: { detail: 'boom' } } });
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_PAYLOAD });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-error')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByRole('button', { name: /Retry/i }));
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue')).toBeInTheDocument();
        });
    });


    // ─────────────────────────────────────────────────────────────────
    // Arc 5 — soft-reject. Rejected rows stay on disk until the
    // explicit purge step; the section groups them by reject_reason
    // so a noisy synth run becomes one card per cause.
    // ─────────────────────────────────────────────────────────────────

    const REJECTED_PAYLOAD = {
        project_id: 1,
        dataset_id: 12,
        total_rows: 6,
        total_pending: 0,
        total_accepted: 0,
        total_rejected: 4,
        groups: [],
        accepted_groups: [],
        rejected_groups: [
            {
                synth_source: 'playbook:classification:positives_paraphrase',
                count: 3,
                truncated: false,
                rows: [
                    {
                        id: 10,
                        synth_confidence: 0.4,
                        preview: '{"text": "bad row a"}',
                        payload: { text: 'bad row a', reject_reason: 'low_confidence' },
                    },
                    {
                        id: 11,
                        synth_confidence: 0.3,
                        preview: '{"text": "bad row b"}',
                        payload: { text: 'bad row b', reject_reason: 'low_confidence' },
                    },
                    {
                        id: 12,
                        synth_confidence: 0.5,
                        preview: '{"text": "dup row"}',
                        payload: { text: 'dup row', reject_reason: 'duplicate' },
                    },
                ],
            },
            {
                synth_source: 'playbook:classification:hard_negatives',
                count: 1,
                truncated: false,
                rows: [
                    {
                        id: 13,
                        synth_confidence: 0.2,
                        preview: '{"text": "unlabelled row"}',
                        payload: { text: 'unlabelled row' },
                    },
                ],
            },
        ],
    };

    it('renders the rejected section grouped by reject_reason', async () => {
        apiMock.get.mockResolvedValue({ data: REJECTED_PAYLOAD });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-rejected')).toBeInTheDocument();
        });
        const section = screen.getByTestId('synth-review-queue-rejected');
        expect(section.textContent).toMatch(/4\s+rejected rows/);
        // Three distinct reason buckets: low_confidence, duplicate, (no reason).
        expect(
            screen.getByTestId('synth-review-queue-rejected-reason-low_confidence'),
        ).toBeInTheDocument();
        expect(
            screen.getByTestId('synth-review-queue-rejected-reason-duplicate'),
        ).toBeInTheDocument();
        expect(
            screen.getByTestId('synth-review-queue-rejected-reason-(no reason)'),
        ).toBeInTheDocument();
    });

    it('does not render the rejected section when no rejected rows exist', async () => {
        apiMock.get.mockResolvedValue({ data: SAMPLE_PAYLOAD });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue')).toBeInTheDocument();
        });
        expect(
            screen.queryByTestId('synth-review-queue-rejected'),
        ).not.toBeInTheDocument();
    });

    it('purges all rejected rows when "Purge all" is clicked', async () => {
        apiMock.get.mockResolvedValueOnce({ data: REJECTED_PAYLOAD });
        apiMock.post.mockResolvedValueOnce({
            data: { purged: 4, retained: 0, total_rows: 0 },
        });
        // After the purge, the next list call returns the empty state.
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...REJECTED_PAYLOAD,
                total_rejected: 0,
                rejected_groups: [],
            },
        });

        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-purge-all')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('synth-review-queue-purge-all'));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/review-queue/purge',
                { reasons: null },
            );
        });
        // Confirmation flash surfaces what landed.
        await waitFor(() => {
            expect(
                screen.getByTestId('synth-review-queue-purge-flash'),
            ).toHaveTextContent(/Purged 4 rows/);
        });
        // List re-fetched after the purge — section disappears.
        await waitFor(() => {
            expect(
                screen.queryByTestId('synth-review-queue-rejected'),
            ).not.toBeInTheDocument();
        });
    });

    it('purges a single reason cohort when "Purge this reason" is clicked', async () => {
        apiMock.get.mockResolvedValueOnce({ data: REJECTED_PAYLOAD });
        apiMock.post.mockResolvedValueOnce({
            data: { purged: 2, retained: 2, total_rows: 2 },
        });
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...REJECTED_PAYLOAD,
                total_rejected: 2,
                rejected_groups: [
                    {
                        synth_source: 'playbook:classification:positives_paraphrase',
                        count: 1,
                        truncated: false,
                        rows: [REJECTED_PAYLOAD.rejected_groups[0].rows[2]],
                    },
                    REJECTED_PAYLOAD.rejected_groups[1],
                ],
            },
        });

        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('synth-review-queue-purge-reason-low_confidence'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('synth-review-queue-purge-reason-low_confidence'),
        );

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/review-queue/purge',
                { reasons: ['low_confidence'] },
            );
        });
        // low_confidence card gone, duplicate + (no reason) remain.
        await waitFor(() => {
            expect(
                screen.queryByTestId('synth-review-queue-rejected-reason-low_confidence'),
            ).not.toBeInTheDocument();
        });
        expect(
            screen.getByTestId('synth-review-queue-rejected-reason-duplicate'),
        ).toBeInTheDocument();
    });

    it('purges the no-reason cohort with reasons=[""]', async () => {
        // Rows that were rejected without a tag still need a way to
        // be purged. The backend treats reasons=[""] as the
        // empty-reason filter.
        apiMock.get.mockResolvedValueOnce({ data: REJECTED_PAYLOAD });
        apiMock.post.mockResolvedValueOnce({
            data: { purged: 1, retained: 3, total_rows: 3 },
        });
        apiMock.get.mockResolvedValueOnce({ data: REJECTED_PAYLOAD });

        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('synth-review-queue-purge-reason-(no reason)'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('synth-review-queue-purge-reason-(no reason)'),
        );

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/review-queue/purge',
                { reasons: [''] },
            );
        });
    });

    it('surfaces a backend error from the purge endpoint', async () => {
        apiMock.get.mockResolvedValueOnce({ data: REJECTED_PAYLOAD });
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'synthetic.jsonl missing' } },
        });

        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue-purge-all')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('synth-review-queue-purge-all'));
        await waitFor(() => {
            expect(
                screen.getByTestId('synth-review-queue-purge-error'),
            ).toHaveTextContent(/synthetic.jsonl missing/);
        });
    });
});

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
    total_pending: 3,
    total_accepted: 0,
    groups: [
        {
            synth_source: 'playbook:classification:positives_paraphrase',
            count: 2,
            rows: [
                { id: 1, synth_confidence: 0.9, preview: '{"text": "row a", "label": "billing"}', payload: { text: 'row a', label: 'billing' } },
                { id: 2, synth_confidence: 0.85, preview: '{"text": "row b", "label": "billing"}', payload: { text: 'row b', label: 'billing' } },
            ],
        },
        {
            synth_source: 'playbook:classification:hard_negatives:vs=billing',
            count: 1,
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

    it('renders the "queued for training" summary when only accepted rows exist', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                dataset_id: 12,
                total_pending: 0,
                total_accepted: 5,
                groups: [],
                accepted_groups: [
                    { synth_source: 'playbook:qa-sft:positives_paraphrase', count: 3, rows: [] },
                    { synth_source: 'playbook:qa-sft:cluster_targeted:cluster=cluster-2', count: 2, rows: [] },
                ],
            },
        });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue')).toBeInTheDocument();
        });
        // Headline reports accepted count (text is split across a
        // <strong> tag, so assert on the section's combined textContent).
        const root = screen.getByTestId('synth-review-queue');
        expect(root.textContent).toMatch(/5 rows accepted/);
        // Accepted section appears with both source groups.
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
            data: { ...SAMPLE_PAYLOAD, total_accepted: 4, accepted_groups: [
                { synth_source: 'playbook:classification:positives_paraphrase', count: 4, rows: [] },
            ] },
        });
        render(<SynthReviewQueue projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-review-queue')).toBeInTheDocument();
        });
        // Subtitle mentions both pending + accepted counts (text is
        // interleaved with <strong> tags).
        const root = screen.getByTestId('synth-review-queue');
        expect(root.textContent).toMatch(/3 rows awaiting review/);
        expect(root.textContent).toMatch(/4 already accepted/);
        // Accepted section is rendered as well.
        expect(screen.getByTestId('synth-review-queue-accepted')).toBeInTheDocument();
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
            { row_ids: [1], action: 'accept' },
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
            { row_ids: [1], action: 'reject' },
        );
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
});

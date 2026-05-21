import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({
    default: apiMock,
}));

import ActiveLearningPanel from './ActiveLearningPanel';

const SAMPLE_PROPOSAL = {
    eval_result_id: 42,
    experiment_id: 7,
    candidates: [
        {
            row_index: 1,
            failure_reason: 'row_exact_match=0',
            prompt: 'How do I reset my password?',
            prediction: 'WRONG prediction',
            reference: 'Settings → Security → Reset password.',
            row_score: 0.0,
            already_promoted: false,
        },
        {
            row_index: 2,
            failure_reason: 'row_f1=0.20<0.5',
            prompt: 'Where do I download my invoice?',
            prediction: 'invoices live somewhere',
            reference: 'Billing → History → Download all.',
            row_score: 0.2,
            already_promoted: false,
        },
        {
            row_index: 5,
            failure_reason: 'row_exact_match=0',
            prompt: 'Can I close my account?',
            prediction: 'no',
            reference: 'Yes — Account → Close.',
            row_score: 0.0,
            already_promoted: true,
        },
    ],
    total_failures: 3,
    total_predictions: 20,
    max_rows: 50,
    dataset_name: 'Gold (dev)',
    promoted_count: 1,
};

describe('ActiveLearningPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders nothing actionable while loading the proposal', () => {
        apiMock.get.mockReturnValue(new Promise(() => undefined));
        render(<ActiveLearningPanel projectId={1} experimentId={7} />);
        expect(screen.getByTestId('active-learning-panel-loading')).toBeInTheDocument();
    });

    it('shows the empty-state when the eval result has no failures', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...SAMPLE_PROPOSAL,
                candidates: [],
                total_failures: 0,
            },
        });
        render(<ActiveLearningPanel projectId={1} experimentId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('active-learning-panel-empty')).toBeInTheDocument();
        });
        expect(
            screen.getByText(/No failed eval rows to learn from/),
        ).toBeInTheDocument();
    });

    it('renders candidate rows with truncated prompt/prediction/reference', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_PROPOSAL });
        render(<ActiveLearningPanel projectId={1} experimentId={7} />);

        await waitFor(() => {
            expect(screen.getByTestId('active-learning-panel')).toBeInTheDocument();
        });
        expect(screen.getByTestId('active-learning-row-1')).toBeInTheDocument();
        expect(screen.getByTestId('active-learning-row-2')).toBeInTheDocument();
        expect(screen.getByTestId('active-learning-row-5')).toBeInTheDocument();

        // Already-promoted row shows a badge + checkbox disabled.
        expect(screen.getByTestId('active-learning-row-5-promoted')).toBeInTheDocument();
        const row5Checkbox = screen
            .getByTestId('active-learning-row-5')
            .querySelector('input[type="checkbox"]') as HTMLInputElement;
        expect(row5Checkbox.disabled).toBe(true);

        // Headline counts come from the proposal.
        expect(screen.getByText(/3 failing examples/i)).toBeInTheDocument();
        expect(screen.getByText(/1 already promoted/)).toBeInTheDocument();
    });

    it('defaults selection to every actionable candidate and posts on Add click', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_PROPOSAL });
        apiMock.post.mockResolvedValueOnce({
            data: {
                status: 'ok',
                experiment_id: 7,
                promoted_count: 2,
                skipped_already_promoted: 0,
                skipped_invalid_indexes: 0,
                target_dataset_id: 11,
                target_dataset_path: '/tmp/synthetic.jsonl',
                total_promoted_lifetime: 3,
            },
        });
        // After promote, the panel reloads — return an updated proposal
        // where rows 1+2 now show already_promoted=true.
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...SAMPLE_PROPOSAL,
                candidates: SAMPLE_PROPOSAL.candidates.map((c) =>
                    c.row_index === 1 || c.row_index === 2
                        ? { ...c, already_promoted: true }
                        : c,
                ),
                promoted_count: 3,
            },
        });

        render(<ActiveLearningPanel projectId={1} experimentId={7} />);
        const button = (await screen.findByTestId(
            'active-learning-promote',
        )) as HTMLButtonElement;

        // Default selection covers rows 1 + 2 (NOT the already-promoted row 5).
        expect(button).toHaveTextContent('Add 2 to training');

        const user = userEvent.setup();
        await user.click(button);

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/evaluation/active-learning/7/promote',
                { row_indexes: expect.arrayContaining([1, 2]) },
            );
        });
        // After reload, both rows show the promoted badge.
        await waitFor(() => {
            expect(screen.getByTestId('active-learning-row-1-promoted')).toBeInTheDocument();
        });
        expect(screen.getByTestId('active-learning-row-2-promoted')).toBeInTheDocument();
    });

    it('toggling a checkbox updates the Add button count', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_PROPOSAL });
        render(<ActiveLearningPanel projectId={1} experimentId={7} />);

        const button = (await screen.findByTestId(
            'active-learning-promote',
        )) as HTMLButtonElement;
        expect(button).toHaveTextContent('Add 2 to training');

        const user = userEvent.setup();
        const row1Checkbox = screen
            .getByTestId('active-learning-row-1')
            .querySelector('input[type="checkbox"]') as HTMLInputElement;
        await user.click(row1Checkbox);

        expect(button).toHaveTextContent('Add 1 to training');
    });

    it('shows error inline when proposal fetch fails', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { data: { detail: 'eval_result_not_found:7' } },
        });
        render(<ActiveLearningPanel projectId={1} experimentId={7} />);

        await waitFor(() => {
            expect(screen.getByTestId('active-learning-panel-error')).toBeInTheDocument();
        });
        expect(screen.getByTestId('active-learning-panel-error')).toHaveTextContent(
            'eval_result_not_found:7',
        );
    });

    it('show-all toggle reveals candidates past the default top-5 cap', async () => {
        const many = Array.from({ length: 8 }, (_, i) => ({
            row_index: i,
            failure_reason: 'row_exact_match=0',
            prompt: `q ${i}`,
            prediction: `wrong ${i}`,
            reference: `right ${i}`,
            row_score: 0.0,
            already_promoted: false,
        }));
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...SAMPLE_PROPOSAL,
                candidates: many,
                total_failures: 8,
                promoted_count: 0,
            },
        });
        render(<ActiveLearningPanel projectId={1} experimentId={7} />);

        // Default = top 5 shown.
        await waitFor(() => {
            expect(screen.getByTestId('active-learning-row-0')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('active-learning-row-7')).not.toBeInTheDocument();

        const user = userEvent.setup();
        await user.click(screen.getByTestId('active-learning-toggle-all'));

        expect(screen.getByTestId('active-learning-row-7')).toBeInTheDocument();
    });
});

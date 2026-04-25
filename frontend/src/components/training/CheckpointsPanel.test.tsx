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

import CheckpointsPanel from './CheckpointsPanel';

const RUNNING_EXP = { id: 7, name: 'phi-2 sft', status: 'running' };
const PAUSED_EXP = { id: 7, name: 'phi-2 sft', status: 'paused' };
const COMPLETED_EXP = { id: 7, name: 'phi-2 sft', status: 'completed' };

const LIST_RESPONSE = {
    project_id: 5,
    experiment_id: 7,
    run_status: 'running',
    count: 3,
    checkpoints: [
        {
            id: 1,
            run_id: 7,
            experiment_id: 7,
            step: 50,
            epoch: 1,
            loss: 1.42,
            train_loss: 1.42,
            eval_loss: null,
            metrics_blob: {},
            path: '/tmp/ckpts/step-50.json',
            is_best: false,
            promoted_at: null,
            created_at: '2026-04-25T13:00:00Z',
        },
        {
            id: 2,
            run_id: 7,
            experiment_id: 7,
            step: 100,
            epoch: 1,
            loss: 1.18,
            train_loss: 1.18,
            eval_loss: 1.31,
            metrics_blob: {},
            path: '/tmp/ckpts/step-100.json',
            is_best: false,
            promoted_at: null,
            created_at: '2026-04-25T13:05:00Z',
        },
        {
            id: 3,
            run_id: 7,
            experiment_id: 7,
            step: 150,
            epoch: 2,
            loss: 0.95,
            train_loss: 0.95,
            eval_loss: 1.09,
            metrics_blob: {},
            path: '/tmp/ckpts/step-150.json',
            is_best: true,
            promoted_at: '2026-04-25T13:15:00Z',
            created_at: '2026-04-25T13:10:00Z',
        },
    ],
};

describe('CheckpointsPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('lists checkpoints for the run with step / loss / best flag', async () => {
        apiMock.get.mockResolvedValueOnce({ data: LIST_RESPONSE });

        render(<CheckpointsPanel projectId={5} experiment={RUNNING_EXP} />);

        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/5/training/runs/7/checkpoints',
            );
        });

        // Three rows + the best badge on step 150.
        expect(screen.getByText('50')).toBeInTheDocument();
        expect(screen.getByText('100')).toBeInTheDocument();
        expect(screen.getByText('150')).toBeInTheDocument();
        expect(screen.getByText('best')).toBeInTheDocument();
        expect(screen.getByText('1.4200')).toBeInTheDocument();
    });

    it('shows Pause button for RUNNING and POSTs to the pause endpoint', async () => {
        apiMock.get.mockResolvedValueOnce({ data: LIST_RESPONSE });
        apiMock.post.mockResolvedValueOnce({ data: { pause_requested: true } });
        // After pause, the panel re-fetches the list:
        apiMock.get.mockResolvedValueOnce({
            data: { ...LIST_RESPONSE, run_status: 'running' },
        });

        const user = userEvent.setup();
        render(<CheckpointsPanel projectId={5} experiment={RUNNING_EXP} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(1));

        await user.click(screen.getByRole('button', { name: /^Pause$/i }));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/5/training/runs/7/pause',
            );
        });
        // Re-fetch after the mutation.
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(2));
    });

    it('shows Resume button for PAUSED and POSTs to the resume endpoint', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { ...LIST_RESPONSE, run_status: 'paused' },
        });
        apiMock.post.mockResolvedValueOnce({
            data: { resumed_from_step: 150, status: 'running' },
        });
        apiMock.get.mockResolvedValueOnce({
            data: { ...LIST_RESPONSE, run_status: 'running' },
        });

        const user = userEvent.setup();
        render(<CheckpointsPanel projectId={5} experiment={PAUSED_EXP} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(1));

        await user.click(screen.getByRole('button', { name: /^Resume$/i }));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/5/training/runs/7/resume',
            );
        });
    });

    it('hides Pause + Resume for terminal statuses', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { ...LIST_RESPONSE, run_status: 'completed' },
        });

        render(<CheckpointsPanel projectId={5} experiment={COMPLETED_EXP} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        expect(screen.queryByRole('button', { name: /^Pause$/i })).toBeNull();
        expect(screen.queryByRole('button', { name: /^Resume$/i })).toBeNull();
    });

    it('promotes a checkpoint via POST and re-fetches the list', async () => {
        apiMock.get.mockResolvedValueOnce({ data: LIST_RESPONSE });
        apiMock.post.mockResolvedValueOnce({
            data: { step: 100, is_best: true, promoted_at: '2026-04-25T13:30:00Z' },
        });
        apiMock.get.mockResolvedValueOnce({ data: LIST_RESPONSE });

        const user = userEvent.setup();
        render(<CheckpointsPanel projectId={5} experiment={RUNNING_EXP} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(1));

        // The "Promote" buttons live inside each row — we want the one on
        // step 100. Find by exact row containing "100".
        const promoteButtons = screen.getAllByRole('button', { name: /^Promote$/i });
        // 3 rows → 3 promote buttons (one per checkpoint).
        expect(promoteButtons.length).toBe(3);
        await user.click(promoteButtons[1]); // step 100 (zero-indexed)

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/5/training/runs/7/checkpoints/100/promote',
            );
        });
        // After promotion, list refetched.
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(2));
    });

    it('forks a new run from a checkpoint via resume-from-step and notifies parent', async () => {
        apiMock.get.mockResolvedValueOnce({ data: LIST_RESPONSE });
        apiMock.post.mockResolvedValueOnce({
            data: { experiment_id: 99, status: 'pending' },
        });
        apiMock.get.mockResolvedValueOnce({ data: LIST_RESPONSE });

        const onForked = vi.fn();
        const user = userEvent.setup();
        render(
            <CheckpointsPanel
                projectId={5}
                experiment={RUNNING_EXP}
                onForked={onForked}
            />,
        );
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(1));

        const forkButtons = screen.getAllByRole('button', { name: /^Resume from…$/i });
        await user.click(forkButtons[2]); // step 150

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/5/training/runs/7/resume-from/150',
                {},
            );
        });
        expect(onForked).toHaveBeenCalledWith(99);
    });

    it('surfaces a friendly error when pause is rejected by the API', async () => {
        apiMock.get.mockResolvedValueOnce({ data: LIST_RESPONSE });
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'not_running' } },
        });

        const user = userEvent.setup();
        render(<CheckpointsPanel projectId={5} experiment={RUNNING_EXP} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());

        await user.click(screen.getByRole('button', { name: /^Pause$/i }));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalled();
        });
        expect(await screen.findByText('not_running')).toBeInTheDocument();
    });

    it('renders an empty state when the run has no checkpoints yet', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 5,
                experiment_id: 7,
                run_status: 'running',
                count: 0,
                checkpoints: [],
            },
        });

        render(<CheckpointsPanel projectId={5} experiment={RUNNING_EXP} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        expect(
            screen.getByText(/No checkpoints yet/i),
        ).toBeInTheDocument();
    });
});

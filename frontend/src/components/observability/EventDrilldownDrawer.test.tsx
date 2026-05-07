import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import EventDrilldownDrawer from './EventDrilldownDrawer';

beforeEach(() => {
    apiMock.get.mockReset();
});

describe('EventDrilldownDrawer', () => {
    it('returns null when runId is null', () => {
        const { container } = render(
            <EventDrilldownDrawer runId={null} onClose={vi.fn()} />,
        );
        expect(container).toBeEmptyDOMElement();
    });

    it('fetches per-run events on runId change and renders them', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                run_id: 'exp-1',
                limit: 200,
                events: [
                    {
                        id: 1,
                        run_id: 'exp-1',
                        parent_run_id: null,
                        stage: 'training',
                        severity: 'info',
                        reason_code: null,
                        actor: 'system',
                        summary: 'Training started',
                        payload: {},
                        ts: '2026-05-07T10:00:00Z',
                        created_at: '2026-05-07T10:00:00Z',
                    },
                    {
                        id: 2,
                        run_id: 'exp-1',
                        parent_run_id: null,
                        stage: 'training',
                        severity: 'error',
                        reason_code: 'training_runtime_error',
                        actor: 'system',
                        summary: 'CUDA OOM at step 1200',
                        payload: { step: 1200 },
                        ts: '2026-05-07T10:00:30Z',
                        created_at: '2026-05-07T10:00:30Z',
                    },
                ],
            },
        });
        render(<EventDrilldownDrawer runId="exp-1" onClose={vi.fn()} />);
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith('/run-events/run/exp-1');
        });
        expect(await screen.findByText('Training started')).toBeInTheDocument();
        expect(screen.getByText('CUDA OOM at step 1200')).toBeInTheDocument();
        expect(screen.getByText('training_runtime_error')).toBeInTheDocument();
    });

    it('renders empty state when there are no events', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { run_id: 'exp-1', limit: 200, events: [] },
        });
        render(<EventDrilldownDrawer runId="exp-1" onClose={vi.fn()} />);
        expect(
            await screen.findByText(/No events for this run id/i),
        ).toBeInTheDocument();
    });

    it('surfaces API errors in an alert', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 500, data: { detail: 'server_error' } },
        });
        render(<EventDrilldownDrawer runId="exp-1" onClose={vi.fn()} />);
        const alert = await screen.findByRole('alert');
        expect(alert).toHaveTextContent('server_error');
    });

    it('Close button fires onClose', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { run_id: 'exp-1', limit: 200, events: [] },
        });
        const onClose = vi.fn();
        render(<EventDrilldownDrawer runId="exp-1" onClose={onClose} />);
        const user = userEvent.setup();
        await user.click(
            await screen.findByRole('button', { name: /Close drilldown/i }),
        );
        expect(onClose).toHaveBeenCalledTimes(1);
    });
});

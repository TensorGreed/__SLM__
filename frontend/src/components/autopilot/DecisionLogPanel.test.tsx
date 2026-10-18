import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DecisionLogPanel from './DecisionLogPanel';

const row = (id: number, run_id: string, extra: Record<string, unknown> = {}) => ({
    id,
    run_id,
    project_id: 31,
    sequence: id,
    stage: 'start_training',
    status: 'applied',
    action: 'applied',
    reason_code: null,
    confidence: 0.9,
    rationale: 'rows were enough',
    summary: `step ${id}`,
    actor: 'autopilot',
    changed: true,
    safe: true,
    blocker: false,
    dry_run: false,
    intent: null,
    payload: {},
    created_at: '2026-10-18T10:00:00Z',
    ...extra,
});

describe('DecisionLogPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('fetches the project decisions on mount and groups them by run', async () => {
        apiMock.get.mockResolvedValue({
            data: { items: [row(1, 'run-aaaaaaaa-1'), row(2, 'run-aaaaaaaa-1'), row(3, 'run-bbbbbbbb-2', { blocker: true, status: 'blocked' })], limit: 200, offset: 0, returned: 3 },
        });
        render(<DecisionLogPanel projectId={31} />);
        expect(screen.getByTestId('loading-state')).toBeInTheDocument();
        await waitFor(() => expect(screen.getAllByTestId('decision-log-run')).toHaveLength(2));
        expect(apiMock.get).toHaveBeenCalledWith('/autopilot/decisions', { params: { project_id: 31, limit: 200 } });
        expect(screen.getByText('2 steps')).toBeInTheDocument();
        expect(screen.getByText('BLOCKED')).toBeInTheDocument();
    });

    it('expands a row to its detail and refetches when a filter changes', async () => {
        const user = userEvent.setup();
        apiMock.get.mockResolvedValue({ data: { items: [row(7, 'run-x', { reason_code: 'STRICT_MODE', payload: { k: 1 } })], limit: 200, offset: 0, returned: 1 } });
        render(<DecisionLogPanel projectId={31} />);
        await user.click(await screen.findByTestId('decision-log-row-7'));
        expect(screen.getByTestId('decision-log-detail-7')).toHaveTextContent('STRICT_MODE');
        expect(screen.getByTestId('decision-log-detail-7')).toHaveTextContent('"k": 1');

        await user.selectOptions(screen.getByTestId('decision-log-status'), 'blocked');
        await waitFor(() =>
            expect(apiMock.get).toHaveBeenLastCalledWith('/autopilot/decisions', { params: { project_id: 31, limit: 200, status: 'blocked' } }),
        );
    });

    it('shows an empty state when nothing was recorded', async () => {
        apiMock.get.mockResolvedValue({ data: { items: [], limit: 200, offset: 0, returned: 0 } });
        render(<DecisionLogPanel projectId={31} />);
        expect(await screen.findByTestId('empty-state')).toHaveTextContent('No Autopilot decisions yet');
    });

    it('surfaces a load error', async () => {
        apiMock.get.mockRejectedValue({ response: { data: { detail: 'decisions unavailable' } } });
        render(<DecisionLogPanel projectId={31} />);
        expect(await screen.findByRole('alert')).toHaveTextContent('decisions unavailable');
    });
});

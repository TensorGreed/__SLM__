import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import SpanThresholdTuner from './SpanThresholdTuner';

const FLOORS = {
    floors: {},
    default: null,
    types: ['credit_card', 'email'],
    suggestions: { credit_card: { floor: 0.95, regimes: ['pci_dss'] } },
};

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.put.mockReset();
    apiMock.get.mockResolvedValue({ data: FLOORS });
});

describe('SpanThresholdTuner', () => {
    it('offers compliance suggestions only on click, then saves the floors', async () => {
        const user = userEvent.setup();
        render(<SpanThresholdTuner projectId={7} experimentId={59} />);
        const cc = await screen.findByTestId('span-tuner-floor-credit_card');
        expect(cc).toHaveValue(null);
        expect(screen.getByText('suggested 0.95 (pci_dss)')).toBeInTheDocument();
        await user.click(screen.getByTestId('span-tuner-use-suggested'));
        expect(cc).toHaveValue(0.95);
        expect(apiMock.put).not.toHaveBeenCalled();

        await user.type(screen.getByTestId('span-tuner-floor-default'), '0.8');
        apiMock.put.mockResolvedValue({ data: { ...FLOORS, floors: { credit_card: 0.95 }, default: 0.8 } });
        await user.click(screen.getByTestId('span-tuner-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/7/span-recall-floors', {
            floors: { credit_card: 0.95 }, default: 0.8,
        }));
        expect(await screen.findByTestId('span-tuner-saved')).toHaveTextContent('saved on project');
    });

    it('refuses a floor outside 0–1 before calling the server', async () => {
        const user = userEvent.setup();
        render(<SpanThresholdTuner projectId={7} experimentId={59} />);
        await user.type(await screen.findByTestId('span-tuner-floor-email'), '1.5');
        await user.click(screen.getByTestId('span-tuner-save'));
        expect(screen.getByTestId('span-tuner-error')).toHaveTextContent('Floors must be between 0 and 1 (email)');
        expect(apiMock.put).not.toHaveBeenCalled();
    });

    it('tunes with the edited floors and the recall-bias opt-in, and shows met floors', async () => {
        const user = userEvent.setup();
        const onTuned = vi.fn();
        render(<SpanThresholdTuner projectId={7} experimentId={59} onTuned={onTuned} />);
        await user.type(await screen.findByTestId('span-tuner-floor-email'), '0.9');
        await user.click(screen.getByTestId('span-tuner-apply-bias'));
        const result = {
            experiment_id: 59, split: 'val', rows: 40, micro_f1_before: 0.8, micro_f1_after: 0.82, raised: { email: 0.4 },
            report: {
                email: {
                    threshold: 0.4, f1_before: 0.8, f1_after: 0.82, precision_before: 0.7, precision_after: 0.8,
                    recall_before: 0.85, recall_after: 0.92, support: 20, predicted: 24, floor: 0.9, floor_status: 'met',
                },
            },
            floors_met: ['email'], floors_not_reachable: [], recall_bias: 0.1, recall_bias_suggestion: null,
        };
        apiMock.post.mockResolvedValue({ data: result });
        await user.click(screen.getByTestId('eval-summary-tune-run'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/7/span-thresholds/tune', {
            experiment_id: 59, floors: { email: 0.9 }, apply_recall_bias: true,
        }));
        expect(await screen.findByTestId('span-tuner-row-email')).toHaveTextContent('email0.900.85 → 0.920.70 → 0.800.4met');
        expect(screen.getByTestId('span-tuner-bias-applied')).toHaveTextContent('Recall bias 0.1 applied');
        expect(onTuned).toHaveBeenCalledWith(result);
    });
});

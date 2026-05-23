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

import TrainabilityForecastPanel from './TrainabilityForecastPanel';
import type { ForecastResult } from '../../api/trainabilityForecast';


function makeForecast(overrides: Partial<ForecastResult> = {}): ForecastResult {
    return {
        overall: 'likely_pass',
        confidence_pct: 72,
        signals: [
            {
                id: 'row_count_below_minimum',
                severity: 'ok',
                headline: '200 training rows — above recipe minimum.',
                detail: 'Recipe minimum is 50; you\'re comfortably above.',
                suggested_action: null,
            },
            {
                id: 'gate_pass_probability',
                severity: 'ok',
                headline: 'Predicted gate-pass probability: ~72%',
                detail: 'Most signals look healthy.',
                suggested_action: null,
            },
        ],
        computed_at: '2026-05-23T00:00:00Z',
        cache_key: 'abc123',
        cache_hit: false,
        ...overrides,
    };
}


describe('TrainabilityForecastPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders the verdict badge and confidence band on success', async () => {
        apiMock.get.mockResolvedValue({ data: makeForecast() });
        render(<TrainabilityForecastPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('trainability-forecast')).toBeInTheDocument();
        });
        expect(screen.getByText('Likely to pass gates')).toBeInTheDocument();
        // Confidence chip in the header (one of two occurrences of "~72%"
        // — the other lives inside the gate_pass_probability signal row).
        const confidenceChip = screen.getByText('Predicted gate-pass:').parentElement;
        expect(confidenceChip?.textContent).toMatch(/~72%/);
    });

    it('renders each signal row with severity-coded marker', async () => {
        apiMock.get.mockResolvedValue({
            data: makeForecast({
                overall: 'likely_fail',
                confidence_pct: 28,
                signals: [
                    {
                        id: 'row_count_below_minimum',
                        severity: 'block',
                        headline: 'Only 16 training rows — recipe recommends at least 50.',
                        detail: 'You\'re 34 rows short.',
                        suggested_action: {
                            kind: 'synth_augment',
                            params: { target_rows: 50 },
                        },
                    },
                    {
                        id: 'gate_pass_probability',
                        severity: 'warn',
                        headline: 'Predicted gate-pass probability: ~28%',
                        detail: 'Multiple signals suggest under-resourcing.',
                        suggested_action: null,
                    },
                ],
            }),
        });
        render(<TrainabilityForecastPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('trainability-forecast-signal-row_count_below_minimum')).toBeInTheDocument();
        });
        const blockSignal = screen.getByTestId('trainability-forecast-signal-row_count_below_minimum');
        expect(blockSignal).toHaveAttribute('data-severity', 'block');
        const warnSignal = screen.getByTestId('trainability-forecast-signal-gate_pass_probability');
        expect(warnSignal).toHaveAttribute('data-severity', 'warn');
    });

    it('fires onActionClicked when the suggested action button is clicked', async () => {
        apiMock.get.mockResolvedValue({
            data: makeForecast({
                overall: 'likely_fail',
                signals: [
                    {
                        id: 'row_count_below_minimum',
                        severity: 'block',
                        headline: 'Only 16 rows.',
                        detail: '',
                        suggested_action: {
                            kind: 'synth_augment',
                            params: { target_rows: 50 },
                        },
                    },
                ],
            }),
        });
        const onAction = vi.fn();
        render(<TrainabilityForecastPanel projectId={1} onActionClicked={onAction} />);

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Generate more training rows/i })).toBeInTheDocument();
        });
        await userEvent.click(screen.getByRole('button', { name: /Generate more training rows/i }));
        expect(onAction).toHaveBeenCalledWith('synth_augment', { target_rows: 50 });
    });

    it('refreshes the forecast and bypasses cache when Refresh is clicked', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: makeForecast({ cache_hit: false }) })
            .mockResolvedValueOnce({ data: makeForecast({ cache_hit: false, confidence_pct: 80 }) });

        render(<TrainabilityForecastPanel projectId={5} />);

        await waitFor(() => {
            const chip = screen.getByText('Predicted gate-pass:').parentElement;
            expect(chip?.textContent).toMatch(/~72%/);
        });
        // First call: no refresh param.
        expect(apiMock.get).toHaveBeenNthCalledWith(1, '/projects/5/training/forecast', { params: undefined });

        await userEvent.click(screen.getByRole('button', { name: /Refresh forecast/i }));
        await waitFor(() => {
            const chip = screen.getByText('Predicted gate-pass:').parentElement;
            expect(chip?.textContent).toMatch(/~80%/);
        });
        // Second call: refresh=true bypasses cache.
        expect(apiMock.get).toHaveBeenNthCalledWith(2, '/projects/5/training/forecast', { params: { refresh: true } });
    });

    it('renders a cache-hit note when result was served from cache', async () => {
        apiMock.get.mockResolvedValue({ data: makeForecast({ cache_hit: true }) });
        render(<TrainabilityForecastPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByText(/Cached result/i)).toBeInTheDocument();
        });
    });

    it('returns null silently when the project has no recipe (400)', async () => {
        apiMock.get.mockRejectedValue({
            response: { status: 400, data: { detail: 'Project has no selected recipe' } },
        });
        const { container } = render(<TrainabilityForecastPanel projectId={99} />);
        // Wait for the loading state to clear.
        await waitFor(() => {
            expect(screen.queryByTestId('trainability-forecast-loading')).not.toBeInTheDocument();
        });
        // Panel collapses to nothing — nothing visible from the component.
        expect(container.querySelector('.trainability-forecast')).toBeNull();
    });

    it('surfaces a retry option when the request fails with a non-400 status', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 500, data: { detail: 'boom' } },
        });
        apiMock.get.mockResolvedValueOnce({ data: makeForecast() });
        render(<TrainabilityForecastPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByText(/boom/)).toBeInTheDocument();
        });
        await userEvent.click(screen.getByRole('button', { name: /Retry/i }));
        await waitFor(() => {
            expect(screen.getByText('Likely to pass gates')).toBeInTheDocument();
        });
    });
});

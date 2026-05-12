import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { TelemetryAggregate } from '../../types/deployment';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import TelemetryPanel from './TelemetryPanel';

const HEALTHY_AGGREGATE: TelemetryAggregate = {
    deployment_version_id: 99,
    window_start: '2026-05-04T00:00:00Z',
    window_end: '2026-05-04T01:00:00Z',
    window_seconds: 3600,
    sample_count: 120,
    request_volume: { total: 120, per_second: 0.0333, per_minute: 2.0 },
    latency_ms: { p50: 50, p95: 120, p99: 320, min: 10, max: 410, mean: 65 },
    errors: { count: 1, rate: 0.0083 },
    tokens: {
        input_total: 12000,
        output_total: 4800,
        input_per_second: 3.33,
        output_per_second: 1.33,
        total_per_second: 4.66,
    },
};

const EMPTY_AGGREGATE: TelemetryAggregate = {
    deployment_version_id: 99,
    window_start: '2026-05-04T00:00:00Z',
    window_end: '2026-05-04T01:00:00Z',
    window_seconds: 3600,
    sample_count: 0,
    request_volume: { total: 0, per_second: 0, per_minute: 0 },
    latency_ms: { p50: 0, p95: 0, p99: 0, min: 0, max: 0, mean: 0 },
    errors: { count: 0, rate: 0 },
    tokens: {
        input_total: 0,
        output_total: 0,
        input_per_second: 0,
        output_per_second: 0,
        total_per_second: 0,
    },
};

beforeEach(() => {
    apiMock.get.mockReset();
});

describe('TelemetryPanel', () => {
    it('renders KPIs and percentile chart for a healthy aggregate', async () => {
        apiMock.get.mockResolvedValue({ data: HEALTHY_AGGREGATE });
        render(<TelemetryPanel deploymentVersionId={99} />);

        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/deployments/99/telemetry',
                { params: { window_seconds: 3600 } },
            );
        });

        expect(await screen.findByText('120')).toBeInTheDocument(); // total
        expect(screen.getByText('120.0')).toBeInTheDocument(); // p95 KPI
        expect(screen.getByText(/p50 50.0/)).toBeInTheDocument();
        expect(screen.getByText(/0.83%/)).toBeInTheDocument(); // error rate
        // Chart bars rendered as SVG with role=img
        expect(screen.getByRole('img', { name: /Latency percentile bars/i })).toBeInTheDocument();
    });

    it('renders empty state when no samples in the window', async () => {
        apiMock.get.mockResolvedValue({ data: EMPTY_AGGREGATE });
        render(<TelemetryPanel deploymentVersionId={99} />);
        expect(
            await screen.findByText(/No telemetry samples yet/i),
        ).toBeInTheDocument();
    });

    it('window picker re-fetches with the chosen window_seconds', async () => {
        apiMock.get.mockResolvedValue({ data: HEALTHY_AGGREGATE });
        const user = userEvent.setup();
        render(<TelemetryPanel deploymentVersionId={99} />);

        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledTimes(1);
        });
        await user.click(screen.getByRole('button', { name: '5m' }));
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenLastCalledWith(
                '/deployments/99/telemetry',
                { params: { window_seconds: 300 } },
            );
        });
    });

    it('surfaces API errors inline', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 400, data: { detail: 'invalid_window' } },
        });
        render(<TelemetryPanel deploymentVersionId={99} />);
        expect(await screen.findByRole('alert')).toHaveTextContent('invalid_window');
    });
});

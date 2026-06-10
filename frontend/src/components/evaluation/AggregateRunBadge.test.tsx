/**
 * Quality-Lift phase 8 slice 1 — AggregateRunBadge tests.
 *
 * Pins:
 *   * Renders the curated headline metrics in declared order
 *     (pass_rate, accuracy, f1, …) so the per-result layout is
 *     stable across runs.
 *   * Mean ± std formatting strips trailing zeros (0.83, not 0.830).
 *   * Drill-down is lazy — no GET fired until the user expands.
 *   * Expanded state renders a per-seed table with seed_value rows
 *     sorted ascending, columns mirroring the headline metric order.
 *   * Failed children render with the ``--failed`` class and ``—``
 *     in metric columns when the child has no scalar value.
 *   * ``initiallyExpanded`` prop skips the click + immediately
 *     triggers the load.
 *   * Long-tail variance metrics (top-level keys not in the curated
 *     list) flow through after the curated rows.
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import AggregateRunBadge, {
    extractTopLevelVarianceMetrics,
    isVarianceBlock,
} from './AggregateRunBadge';


const METRICS_3_SEEDS = {
    pass_rate: { mean: 0.83, std: 0.04, min: 0.78, max: 0.87, n: 3 },
    accuracy: { mean: 0.85, std: 0.03, min: 0.82, max: 0.88, n: 3 },
    f1: { mean: 0.83, std: 0.04, min: 0.78, max: 0.87, n: 3 },
};


function mockChildren(children: Array<Record<string, unknown>>) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/seed-group/')) {
            return {
                data: {
                    seed_group_id: 'grp-abcd1234',
                    dataset_name: 'goldset',
                    eval_type: 'classification',
                    aggregate_eval_result_id: 99,
                    leader_experiment_id: 7,
                    children,
                },
            };
        }
        return { data: {} };
    });
}


describe('extractTopLevelVarianceMetrics', () => {
    it('returns curated headline metrics in declared order', () => {
        const out = extractTopLevelVarianceMetrics(METRICS_3_SEEDS);
        expect(out.map((r) => r.name)).toEqual(['pass_rate', 'accuracy', 'f1']);
    });

    it('appends long-tail variance metrics after the curated set', () => {
        const out = extractTopLevelVarianceMetrics({
            ...METRICS_3_SEEDS,
            macro_f1: { mean: 0.81, std: 0.02, min: 0.79, max: 0.83, n: 3 },
        });
        expect(out.map((r) => r.name)).toEqual(['pass_rate', 'accuracy', 'f1', 'macro_f1']);
    });

    it('skips non-variance scalar values', () => {
        const out = extractTopLevelVarianceMetrics({
            pass_rate: { mean: 0.83, std: 0.04, min: 0.78, max: 0.87, n: 3 },
            accuracy: 0.85, // scalar — must be excluded
        });
        expect(out.map((r) => r.name)).toEqual(['pass_rate']);
    });
});


describe('isVarianceBlock', () => {
    it('accepts {mean, std, n} shape', () => {
        expect(isVarianceBlock({ mean: 0.8, std: 0.1, n: 3 })).toBe(true);
    });
    it('rejects scalars + null + non-numeric shapes', () => {
        expect(isVarianceBlock(0.85)).toBe(false);
        expect(isVarianceBlock(null)).toBe(false);
        expect(isVarianceBlock({ mean: '0.8', std: 0.1, n: 3 })).toBe(false);
        expect(isVarianceBlock({ mean: 0.8 })).toBe(false);
    });
});


describe('AggregateRunBadge', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders headline metrics with mean ± std formatting', () => {
        render(
            <AggregateRunBadge
                projectId={1}
                seedGroupId="grp-abcd1234"
                datasetName="goldset"
                evalType="classification"
                metrics={METRICS_3_SEEDS}
            />,
        );
        // n=3 sourced from the first headline metric.
        expect(screen.getByTestId('agg-run-badge-count').textContent).toBe('n=3');
        // pass_rate row + value strips trailing zeros.
        const row = screen.getByTestId('agg-run-badge-headline-pass_rate');
        expect(within(row).getByText('0.83 ± 0.04')).toBeInTheDocument();
    });

    it('does NOT fetch per-seed runs until the user expands', async () => {
        render(
            <AggregateRunBadge
                projectId={1}
                seedGroupId="grp-abcd1234"
                datasetName="goldset"
                evalType="classification"
                metrics={METRICS_3_SEEDS}
            />,
        );
        // No drill panel rendered yet.
        expect(screen.queryByTestId('agg-run-badge-drill')).toBeNull();
        // And critically — no network call to the seed-group endpoint.
        expect(apiMock.get).not.toHaveBeenCalled();
    });

    it('toggle opens the drill-down + fetches with the right URL params', async () => {
        mockChildren([
            { eval_result_id: 1, experiment_id: 11, seed_value: 42, experiment_status: 'completed', metrics: { pass_rate: 0.78, accuracy: 0.82, f1: 0.78 }, pass_rate: 0.78 },
            { eval_result_id: 2, experiment_id: 12, seed_value: 43, experiment_status: 'completed', metrics: { pass_rate: 0.83, accuracy: 0.85, f1: 0.83 }, pass_rate: 0.83 },
            { eval_result_id: 3, experiment_id: 13, seed_value: 44, experiment_status: 'completed', metrics: { pass_rate: 0.87, accuracy: 0.88, f1: 0.87 }, pass_rate: 0.87 },
        ]);
        const user = userEvent.setup();
        render(
            <AggregateRunBadge
                projectId={1}
                seedGroupId="grp-abcd1234"
                datasetName="goldset"
                evalType="classification"
                metrics={METRICS_3_SEEDS}
            />,
        );
        await user.click(screen.getByTestId('agg-run-badge-toggle'));
        await waitFor(() => {
            expect(screen.getByTestId('agg-run-badge-table')).toBeInTheDocument();
        });
        // GET URL scoped to dataset + eval_type so the drill-down
        // matches the badge row.
        const url = apiMock.get.mock.calls[0][0] as string;
        expect(url).toContain('/projects/1/evaluation/seed-group/grp-abcd1234');
        expect(url).toContain('dataset_name=goldset');
        expect(url).toContain('eval_type=classification');
        // Three rows in seed-value order.
        const rows = screen.getAllByTestId(/agg-run-badge-row-/);
        expect(rows.length).toBe(3);
        expect(within(rows[0]).getByText('42')).toBeInTheDocument();
        expect(within(rows[1]).getByText('43')).toBeInTheDocument();
        expect(within(rows[2]).getByText('44')).toBeInTheDocument();
    });

    it('failed children render with the failed class + dash for missing metrics', async () => {
        mockChildren([
            { eval_result_id: 1, experiment_id: 11, seed_value: 42, experiment_status: 'completed', metrics: { pass_rate: 0.83, accuracy: 0.85, f1: 0.83 }, pass_rate: 0.83 },
            { eval_result_id: -1, experiment_id: 12, seed_value: 43, experiment_status: 'failed', metrics: {}, pass_rate: null },
        ]);
        const user = userEvent.setup();
        render(
            <AggregateRunBadge
                projectId={1}
                seedGroupId="grp-abcd1234"
                datasetName="goldset"
                evalType="classification"
                metrics={METRICS_3_SEEDS}
            />,
        );
        await user.click(screen.getByTestId('agg-run-badge-toggle'));
        await waitFor(() => {
            expect(screen.getByTestId('agg-run-badge-table')).toBeInTheDocument();
        });
        const failedRow = screen.getByTestId('agg-run-badge-row-43');
        expect(failedRow.className).toContain('agg-run-badge__row--failed');
        // Status column shows "failed" + every metric column shows "—"
        // because the child has no scalar value.
        expect(within(failedRow).getByText('failed')).toBeInTheDocument();
        const dashCells = within(failedRow).getAllByText('—');
        // 3 metric columns (pass_rate, accuracy, f1) all show "—".
        expect(dashCells.length).toBeGreaterThanOrEqual(3);
    });

    it('initiallyExpanded loads immediately without a toggle click', async () => {
        mockChildren([
            { eval_result_id: 1, experiment_id: 11, seed_value: 42, experiment_status: 'completed', metrics: { pass_rate: 0.83 }, pass_rate: 0.83 },
        ]);
        render(
            <AggregateRunBadge
                projectId={1}
                seedGroupId="grp-abcd1234"
                datasetName="goldset"
                evalType="classification"
                metrics={METRICS_3_SEEDS}
                initiallyExpanded
            />,
        );
        await waitFor(() => {
            expect(screen.getByTestId('agg-run-badge-table')).toBeInTheDocument();
        });
        expect(apiMock.get).toHaveBeenCalledTimes(1);
    });

    it('renders an error + retry button when the drill-down fetch fails', async () => {
        apiMock.get.mockRejectedValueOnce(new Error('boom: 500'));
        const user = userEvent.setup();
        render(
            <AggregateRunBadge
                projectId={1}
                seedGroupId="grp-abcd1234"
                datasetName="goldset"
                evalType="classification"
                metrics={METRICS_3_SEEDS}
            />,
        );
        await user.click(screen.getByTestId('agg-run-badge-toggle'));
        await waitFor(() => {
            expect(screen.getByText(/boom: 500/i)).toBeInTheDocument();
        });
        expect(screen.getByRole('button', { name: /Retry/i })).toBeInTheDocument();
    });
});

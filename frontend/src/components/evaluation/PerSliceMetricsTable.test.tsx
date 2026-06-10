/**
 * Quality-Lift phase 8 slice 2 — PerSliceMetricsTable tests.
 *
 * Pins:
 *   * Loads slice_definitions on mount via the /slice-definitions
 *     endpoint and renders rows in definition order.
 *   * Renders nothing (no DOM) when there's no per_slice payload AND
 *     no slice definitions — the EvalPanel sits silent on projects
 *     that haven't engaged with phase 2.
 *   * ``showEmptyStateWhenNoSlices`` flips the silent default into a
 *     nudge string.
 *   * Columns: ``support`` first (when any slice has it), then the
 *     union of numeric metric names, alphabetized.
 *   * Cell formatter handles scalar + variance-block + missing.
 *   * Failing gate highlights the matching cell ``--failed``.
 *   * Bare metric_id and eval-type-scoped metric_id both match.
 *   * Support=0 row gets the ``--empty`` class + ``support=0`` tag.
 *   * Orphan slice (in per_slice but not in definitions) gets the
 *     ``orphan`` tag + sorts after defined slices.
 *   * Slice defined but missing from per_slice still renders (all-dash
 *     row) so the user sees "no data for this slice this run."
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import PerSliceMetricsTable, {
    type GateCheckLike,
} from './PerSliceMetricsTable';


const TWO_SLICE_DEFS = {
    project_id: 1,
    slice_definitions: {
        slices: [
            { slice_id: 'long_input', display_name: 'Long inputs', where: [{ field: 'input_length', op: 'gte', value: 100 }] },
            { slice_id: 'short_input', display_name: 'Short inputs', where: [{ field: 'input_length', op: 'lt', value: 100 }] },
        ],
    },
};

const EMPTY_SLICE_DEFS = {
    project_id: 1,
    slice_definitions: { slices: [] },
};


function mockSliceDefs(defs: unknown) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.endsWith('/slice-definitions')) return { data: defs };
        return { data: {} };
    });
}


describe('PerSliceMetricsTable', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders nothing when no slice defs AND no per_slice data', async () => {
        mockSliceDefs(EMPTY_SLICE_DEFS);
        const { container } = render(
            <PerSliceMetricsTable projectId={1} metrics={{ accuracy: 0.8 }} />,
        );
        // Wait for the fetch to settle so we know the component had a
        // chance to render an empty state if it wanted to.
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        // Nothing in the DOM — silent on uninitialized projects.
        expect(container.querySelector('[data-testid="per-slice-metrics-table"]')).toBeNull();
        expect(container.querySelector('[data-testid="per-slice-empty"]')).toBeNull();
    });

    it('renders an empty-state hint when showEmptyStateWhenNoSlices is set', async () => {
        mockSliceDefs(EMPTY_SLICE_DEFS);
        render(
            <PerSliceMetricsTable
                projectId={1}
                metrics={{ accuracy: 0.8 }}
                showEmptyStateWhenNoSlices
            />,
        );
        await screen.findByTestId('per-slice-empty');
    });

    it('renders rows in slice-definition order with support column first', async () => {
        mockSliceDefs(TWO_SLICE_DEFS);
        render(
            <PerSliceMetricsTable
                projectId={1}
                metrics={{
                    per_slice: {
                        long_input: { accuracy: 0.80, f1: 0.78, support: 50 },
                        short_input: { accuracy: 0.86, f1: 0.84, support: 80 },
                    },
                }}
            />,
        );
        await screen.findByTestId('per-slice-metrics-table');
        const table = screen.getByTestId('per-slice-metrics-table-table');
        // Rows render in definition order: long_input before short_input.
        const rows = within(table).getAllByRole('row');
        // 1 header + 2 body rows
        expect(rows.length).toBe(3);
        const dataCells = within(rows[1]).getAllByRole('cell');
        // Column order: support, accuracy, f1 (support first; rest
        // alphabetized).
        expect(dataCells[0].textContent).toBe('50');
        expect(dataCells[1].textContent).toBe('0.8');
        expect(dataCells[2].textContent).toBe('0.78');
    });

    it('formats variance blocks as mean ± std with trailing-zero strip', async () => {
        mockSliceDefs(TWO_SLICE_DEFS);
        render(
            <PerSliceMetricsTable
                projectId={1}
                metrics={{
                    per_slice: {
                        long_input: {
                            accuracy: { mean: 0.83, std: 0.04, min: 0.78, max: 0.87, n: 3 },
                            support: 50,
                        },
                        short_input: { accuracy: 0.86, support: 80 },
                    },
                }}
            />,
        );
        await screen.findByTestId('per-slice-metrics-table');
        const cell = screen.getByTestId('per-slice-cell-long_input-accuracy');
        expect(cell.textContent).toBe('0.83 ± 0.04');
    });

    it('highlights cells whose matching gate failed (bare + scoped metric_id)', async () => {
        mockSliceDefs(TWO_SLICE_DEFS);
        const gateChecks: GateCheckLike[] = [
            // Bare form
            {
                metric_id: 'per_slice.long_input.accuracy',
                threshold: 0.85,
                operator: 'gte',
                passed: false,
            },
            // Eval-type-scoped form
            {
                metric_id: 'classification.per_slice.short_input.f1',
                threshold: 0.85,
                operator: 'gte',
                passed: false,
            },
            // Passing gate — should NOT highlight.
            {
                metric_id: 'per_slice.short_input.accuracy',
                threshold: 0.85,
                operator: 'gte',
                passed: true,
            },
        ];
        render(
            <PerSliceMetricsTable
                projectId={1}
                metrics={{
                    per_slice: {
                        long_input: { accuracy: 0.80, f1: 0.78, support: 50 },
                        short_input: { accuracy: 0.86, f1: 0.78, support: 80 },
                    },
                }}
                gateChecks={gateChecks}
            />,
        );
        await screen.findByTestId('per-slice-metrics-table');
        // long_input.accuracy: failed bare gate.
        expect(
            screen.getByTestId('per-slice-cell-long_input-accuracy').className,
        ).toContain('per-slice-table__cell--failed');
        // short_input.f1: failed scoped gate.
        expect(
            screen.getByTestId('per-slice-cell-short_input-f1').className,
        ).toContain('per-slice-table__cell--failed');
        // short_input.accuracy: passing gate → NOT highlighted.
        expect(
            screen.getByTestId('per-slice-cell-short_input-accuracy').className,
        ).not.toContain('per-slice-table__cell--failed');
    });

    it('marks support=0 rows with the empty tag and class', async () => {
        mockSliceDefs(TWO_SLICE_DEFS);
        render(
            <PerSliceMetricsTable
                projectId={1}
                metrics={{
                    per_slice: {
                        long_input: { support: 0 },
                        short_input: { accuracy: 0.86, support: 80 },
                    },
                }}
            />,
        );
        await screen.findByTestId('per-slice-metrics-table');
        const emptyRow = screen.getByTestId('per-slice-row-long_input');
        expect(emptyRow.className).toContain('per-slice-table__row--empty');
        // The "support=0" tag is rendered for the user.
        expect(within(emptyRow).getByText('support=0')).toBeInTheDocument();
    });

    it('orphan slice (in metrics but not in defs) sorts after defined slices + gets a tag', async () => {
        mockSliceDefs(TWO_SLICE_DEFS);
        render(
            <PerSliceMetricsTable
                projectId={1}
                metrics={{
                    per_slice: {
                        long_input: { accuracy: 0.80, support: 50 },
                        // ``zombie_slice`` was deleted from the definitions
                        // but the run still has its metrics.
                        zombie_slice: { accuracy: 0.90, support: 30 },
                    },
                }}
            />,
        );
        await screen.findByTestId('per-slice-metrics-table');
        const rows = within(
            screen.getByTestId('per-slice-metrics-table-table'),
        ).getAllByRole('row');
        // header + long_input + short_input + zombie_slice → 4 rows.
        expect(rows.length).toBe(4);
        // Zombie slice is last (orphans go after defined).
        const lastRow = rows[rows.length - 1];
        expect(lastRow.getAttribute('data-testid')).toBe('per-slice-row-zombie_slice');
        // Orphan tag rendered.
        expect(screen.getByTestId('per-slice-orphan-zombie_slice')).toBeInTheDocument();
    });

    it('renders a defined slice with no metrics as an all-dash row', async () => {
        mockSliceDefs(TWO_SLICE_DEFS);
        render(
            <PerSliceMetricsTable
                projectId={1}
                metrics={{
                    per_slice: {
                        // Only long_input has data; short_input was added
                        // after this run completed.
                        long_input: { accuracy: 0.80, support: 50 },
                    },
                }}
            />,
        );
        await screen.findByTestId('per-slice-metrics-table');
        const shortRow = screen.getByTestId('per-slice-row-short_input');
        // Every cell in the short_input row shows the long-dash
        // placeholder.
        const dashCells = within(shortRow).getAllByText('—');
        expect(dashCells.length).toBeGreaterThan(0);
    });
});

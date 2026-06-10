/**
 * Quality-Lift phase 8 slice 3 — BehavioralResultsTable tests.
 *
 * Pins:
 *   * Renders nothing when ``metrics["behavioral"]`` is missing or
 *     empty — the eval-panel is silent on projects without tests.
 *   * One row per test_id in alphabetical order (stable layout).
 *   * Pass-rate scalar formatted with trailing-zero strip; variance
 *     blocks (multi-seed aggregates) format as ``mean ± std``.
 *   * Per-slice expander hidden when ``per_slice`` is absent.
 *   * Per-slice expander click reveals sub-rows under the parent,
 *     scoped to that test only (other tests' expanders stay closed).
 *   * total=0 row gets the empty class + tag — matches the
 *     PerSliceMetricsTable support=0 convention.
 *   * ``capped_at_budget`` flag renders a "capped @ N" tag in the
 *     notes column so the user knows the score is a sample, not the
 *     full population.
 *   * No invented thresholds — no color coding based on pass-rate;
 *     the cell just shows the number (honest-metrics-no-vanity).
 */

import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import BehavioralResultsTable, {
    extractBehavioralBlocks,
} from './BehavioralResultsTable';


describe('extractBehavioralBlocks', () => {
    it('returns empty array on missing / non-object metrics["behavioral"]', () => {
        expect(extractBehavioralBlocks({})).toEqual([]);
        expect(extractBehavioralBlocks({ behavioral: null })).toEqual([]);
        expect(extractBehavioralBlocks({ behavioral: 'oops' })).toEqual([]);
    });

    it('returns test blocks in alphabetical order', () => {
        const blocks = extractBehavioralBlocks({
            behavioral: {
                typo_invariance: { kind: 'INV', passed: 90, total: 100, pass_rate: 0.9 },
                abuse_dir: { kind: 'DIR', passed: 50, total: 60, pass_rate: 0.83 },
            },
        });
        expect(blocks.map((b) => b.testId)).toEqual(['abuse_dir', 'typo_invariance']);
    });
});


describe('BehavioralResultsTable', () => {
    it('renders nothing when no behavioral metrics present', () => {
        const { container } = render(
            <BehavioralResultsTable metrics={{ accuracy: 0.85 }} />,
        );
        expect(container.querySelector('[data-testid="behavioral-results-table"]')).toBeNull();
    });

    it('renders one row per test_id with formatted scalars', () => {
        render(
            <BehavioralResultsTable
                metrics={{
                    behavioral: {
                        typo_invariance: { kind: 'INV', passed: 90, total: 100, pass_rate: 0.9 },
                        boundary_dir: { kind: 'DIR', passed: 12, total: 20, pass_rate: 0.6 },
                    },
                }}
            />,
        );
        const table = screen.getByTestId('behavioral-results-table');
        expect(table).toBeInTheDocument();
        const typo = screen.getByTestId('behavioral-test-row-typo_invariance');
        expect(within(typo).getByText('INV')).toBeInTheDocument();
        // Trailing-zero strip: 0.900 -> 0.9.
        expect(screen.getByTestId('behavioral-test-rate-typo_invariance').textContent).toBe('0.9');
        // Count columns render as integers.
        expect(within(typo).getByText('90')).toBeInTheDocument();
        expect(within(typo).getByText('100')).toBeInTheDocument();
    });

    it('formats variance-block leaves as mean ± std (multi-seed aggregate row)', () => {
        render(
            <BehavioralResultsTable
                metrics={{
                    behavioral: {
                        typo_invariance: {
                            kind: 'INV',
                            passed: { mean: 88, std: 2.5, n: 3 },
                            total: { mean: 100, std: 0, n: 3 },
                            pass_rate: { mean: 0.88, std: 0.025, n: 3 },
                        },
                    },
                }}
            />,
        );
        // Pass-rate variance block uses the trailing-zero strip too.
        expect(screen.getByTestId('behavioral-test-rate-typo_invariance').textContent).toBe('0.88 ± 0.025');
    });

    it('does NOT render a per-slice expander when per_slice is absent', () => {
        render(
            <BehavioralResultsTable
                metrics={{
                    behavioral: {
                        typo_invariance: { kind: 'INV', passed: 90, total: 100, pass_rate: 0.9 },
                    },
                }}
            />,
        );
        expect(screen.queryByTestId('behavioral-test-expand-typo_invariance')).toBeNull();
    });

    it('per-slice expander reveals sub-rows scoped to the parent test', async () => {
        const user = userEvent.setup();
        render(
            <BehavioralResultsTable
                metrics={{
                    behavioral: {
                        typo_invariance: {
                            kind: 'INV',
                            passed: 90, total: 100, pass_rate: 0.9,
                            per_slice: {
                                long_input: { kind: 'INV', passed: 40, total: 50, pass_rate: 0.8 },
                                short_input: { kind: 'INV', passed: 50, total: 50, pass_rate: 1.0 },
                            },
                        },
                        // Separate test with its own per_slice; its
                        // expander should be independent of the typo one.
                        boundary_dir: {
                            kind: 'DIR',
                            passed: 12, total: 20, pass_rate: 0.6,
                            per_slice: {
                                long_input: { kind: 'DIR', passed: 6, total: 10, pass_rate: 0.6 },
                            },
                        },
                    },
                }}
            />,
        );
        // Collapsed by default — no per-slice rows yet.
        expect(screen.queryByTestId('behavioral-slice-row-long_input')).toBeNull();
        // Open typo only — boundary_dir's slice stays hidden.
        await user.click(screen.getByTestId('behavioral-test-expand-typo_invariance'));
        const sliceRows = screen.getAllByTestId(/^behavioral-slice-row-/);
        // Two rows belonging to typo_invariance, NOT boundary_dir.
        expect(sliceRows.length).toBe(2);
        expect(screen.getByTestId('behavioral-slice-row-long_input')).toBeInTheDocument();
        expect(screen.getByTestId('behavioral-slice-row-short_input')).toBeInTheDocument();
    });

    it('total=0 row gets the empty class + total=0 tag', () => {
        render(
            <BehavioralResultsTable
                metrics={{
                    behavioral: {
                        empty_test: { kind: 'MFT', passed: 0, total: 0, pass_rate: 0.0 },
                    },
                }}
            />,
        );
        const row = screen.getByTestId('behavioral-test-row-empty_test');
        expect(row.className).toContain('behavioral-results__row--empty');
        expect(within(row).getByText('total=0')).toBeInTheDocument();
    });

    it('capped_at_budget flag renders a "capped @ N" tag', () => {
        render(
            <BehavioralResultsTable
                metrics={{
                    behavioral: {
                        big_test: {
                            kind: 'INV',
                            passed: 1800, total: 2000, pass_rate: 0.9,
                            capped_at_budget: 2000,
                        },
                    },
                }}
            />,
        );
        const tag = screen.getByTestId('behavioral-test-capped-big_test');
        expect(tag).toBeInTheDocument();
        expect(tag.textContent).toContain('2000');
    });

    it('does NOT color cells by pass-rate (honest-metrics rule)', () => {
        // The cell rendering must stay neutral — no ad-hoc thresholds
        // ("green >= 0.95, yellow >= 0.80") get baked in here because
        // the table is independent of gates. The user opted-in a
        // threshold elsewhere (the gate); this table is the catalog.
        render(
            <BehavioralResultsTable
                metrics={{
                    behavioral: {
                        flaky: { kind: 'INV', passed: 30, total: 100, pass_rate: 0.30 },
                        solid: { kind: 'INV', passed: 99, total: 100, pass_rate: 0.99 },
                    },
                }}
            />,
        );
        const flakyCell = screen.getByTestId('behavioral-test-rate-flaky');
        const solidCell = screen.getByTestId('behavioral-test-rate-solid');
        // Neither cell carries any "--good" / "--bad" modifier.
        expect(flakyCell.className).not.toMatch(/(good|bad|warn|danger|success)/i);
        expect(solidCell.className).not.toMatch(/(good|bad|warn|danger|success)/i);
    });
});

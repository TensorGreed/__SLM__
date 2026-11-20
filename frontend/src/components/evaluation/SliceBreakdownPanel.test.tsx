import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import SliceBreakdownPanel, { rateText, sliceBreakdownOf } from './SliceBreakdownPanel';

const block = {
    version: 1,
    overall: { rows: 300, scored: 300, pass: 181, pass_ci: [0.546, 0.655] as [number, number],
        strict: { scored: 300, pass: 170, rate: 0.5667, ci: [0.51, 0.62] as [number, number] } },
    slices: [
        { id: 'slice=unseen', source: 'row_tag', field: 'slice', label: 'unseen', rows: 200, scored: 200, pass: 174,
            pass_ci: [0.816, 0.91] as [number, number], strict: { scored: 200, pass: 170, rate: 0.85, ci: [0.79, 0.89] as [number, number] } },
        { id: 'none_rows', source: 'slice_definition', label: 'Should not call', rows: 100, scored: 100, pass: 7, pass_ci: null },
    ],
};

describe('SliceBreakdownPanel', () => {
    it('reads the block only when there are slices', () => {
        expect(sliceBreakdownOf({})).toBeNull();
        expect(sliceBreakdownOf({ slice_breakdown: { slices: [] } })).toBeNull();
        expect(sliceBreakdownOf({ slice_breakdown: block })).not.toBeNull();
    });

    it('formats counts, rates and the Wilson interval', () => {
        expect(rateText(174, 200, [0.816, 0.91])).toBe('174 / 200 · 87.0% (95% CI 81.6%–91.0%)');
        expect(rateText(0, 0, null)).toBe('no scored rows');
    });

    it('is collapsed by default and shows pass + strict per slice', () => {
        render(<SliceBreakdownPanel breakdown={block} provenance="Run #1 · eval set fresh v1 · result 9" />);
        expect(screen.getByTestId('slice-breakdown').hasAttribute('open')).toBe(false);
        expect(screen.getByText('By slice (2)')).toBeTruthy();
        expect(screen.getByTestId('slice-pass-slice=unseen').textContent).toContain('174 / 200');
        expect(screen.getByTestId('slice-strict-slice=unseen').textContent).toContain('170 / 200');
        expect(screen.getByTestId('slice-strict-none_rows').textContent).toBe('not computed');
        expect(screen.getByTestId('slice-row-overall').textContent).toContain('181 / 300');
        expect(screen.getByText('project slice')).toBeTruthy();
        expect(screen.getByText('row slice')).toBeTruthy();
    });
});

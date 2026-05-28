import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import ParetoComparisonPanel, { computeParetoOptimal, type ParetoRow } from './ParetoComparisonPanel';

// A dominates B on latency (B worse quality AND slower); C is the cheap/fast
// frontier point. On the VRAM axis the frontier shifts (B becomes optimal,
// C becomes dominated), which exercises the client-side recompute on toggle.
const MATRIX: ParetoRow[] = [
    { model_id: 'A', estimated_quality_score: 0.9, estimated_latency_ms: 100, estimated_min_vram_gb: 12, params_b: 1.5 },
    { model_id: 'B', estimated_quality_score: 0.8, estimated_latency_ms: 140, estimated_min_vram_gb: 4, params_b: 0.5 },
    { model_id: 'C', estimated_quality_score: 0.6, estimated_latency_ms: 40, estimated_min_vram_gb: 6, params_b: 0.8 },
];

function classOf(testId: string): string {
    return screen.getByTestId(testId).getAttribute('class') || '';
}

describe('computeParetoOptimal', () => {
    it('finds the latency frontier', () => {
        const opt = computeParetoOptimal(MATRIX, 'estimated_latency_ms');
        expect(opt.has('A')).toBe(true);
        expect(opt.has('C')).toBe(true);
        expect(opt.has('B')).toBe(false); // dominated by A
    });

    it('shifts the frontier when the cost axis changes', () => {
        const opt = computeParetoOptimal(MATRIX, 'estimated_min_vram_gb');
        expect(opt.has('A')).toBe(true);
        expect(opt.has('B')).toBe(true); // cheapest VRAM at high quality
        expect(opt.has('C')).toBe(false); // now dominated by B
    });
});

describe('ParetoComparisonPanel', () => {
    it('renders points classified by frontier membership', () => {
        render(<ParetoComparisonPanel matrix={MATRIX} onPromote={vi.fn()} />);
        expect(classOf('pareto-point-A')).toMatch(/is-optimal/);
        expect(classOf('pareto-point-C')).toMatch(/is-optimal/);
        expect(classOf('pareto-point-B')).toMatch(/is-dominated/);
    });

    it('recomputes the frontier when the cost axis is toggled', async () => {
        const user = userEvent.setup();
        render(<ParetoComparisonPanel matrix={MATRIX} onPromote={vi.fn()} />);
        expect(classOf('pareto-point-B')).toMatch(/is-dominated/);

        await user.click(screen.getByRole('button', { name: 'VRAM' }));
        expect(classOf('pareto-point-B')).toMatch(/is-optimal/);
        expect(classOf('pareto-point-C')).toMatch(/is-dominated/);
    });

    it('promotes the best-balance frontier winner by default', async () => {
        const onPromote = vi.fn();
        const user = userEvent.setup();
        render(
            <ParetoComparisonPanel matrix={MATRIX} bestBalanceModelId="A" onPromote={onPromote} />,
        );
        await user.click(screen.getByRole('button', { name: /Promote A/ }));
        expect(onPromote).toHaveBeenCalledTimes(1);
        expect(onPromote.mock.calls[0][0].model_id).toBe('A');
    });

    it('disables promotion when the winner is already the base model', () => {
        render(
            <ParetoComparisonPanel matrix={MATRIX} bestBalanceModelId="A" currentBaseModel="A" onPromote={vi.fn()} />,
        );
        expect(screen.getByRole('button', { name: /Already the base model/ })).toBeDisabled();
    });

    it('shows an empty hint with no matrix', () => {
        render(<ParetoComparisonPanel matrix={[]} onPromote={vi.fn()} />);
        expect(screen.getByText(/Run a benchmark sweep/i)).toBeInTheDocument();
    });
});

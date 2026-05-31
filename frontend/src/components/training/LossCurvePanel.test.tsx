import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import LossCurvePanel from './LossCurvePanel';

// Build a synthetic "healthy" run: train + eval both monotonically
// decreasing across 5 steps. Useful baseline for the no-divergence path.
function healthyMetrics() {
    return [
        { step: 100, train_loss: 2.4, eval_loss: 2.5 },
        { step: 200, train_loss: 1.8, eval_loss: 2.0 },
        { step: 300, train_loss: 1.4, eval_loss: 1.7 },
        { step: 400, train_loss: 1.1, eval_loss: 1.5 },
        { step: 500, train_loss: 0.9, eval_loss: 1.4 },
    ];
}

// Build a synthetic overfit: train keeps falling, eval bottoms at step
// 300 then climbs by +0.2 by step 500.
function overfitMetrics() {
    return [
        { step: 100, train_loss: 2.4, eval_loss: 2.5 },
        { step: 200, train_loss: 1.8, eval_loss: 2.0 },
        { step: 300, train_loss: 1.4, eval_loss: 1.6 },
        { step: 400, train_loss: 1.0, eval_loss: 1.7 },
        { step: 500, train_loss: 0.6, eval_loss: 1.8 },
    ];
}

describe('LossCurvePanel', () => {
    it('shows an empty hint when no metrics have arrived', () => {
        render(<LossCurvePanel metrics={[]} />);
        const node = screen.getByTestId('loss-curve');
        expect(node.className).toMatch(/--empty/);
        expect(node.textContent).toMatch(/Loss curve will render here/);
    });

    it('renders train + eval paths and the best-eval marker on a healthy run', () => {
        render(<LossCurvePanel metrics={healthyMetrics()} />);
        expect(screen.getByTestId('loss-curve-train-path')).toBeInTheDocument();
        expect(screen.getByTestId('loss-curve-eval-path')).toBeInTheDocument();
        expect(screen.getByTestId('loss-curve-best-marker')).toBeInTheDocument();
        // Best eval = step 500 (lowest eval_loss on the healthy run).
        const head = screen.getByTestId('loss-curve').textContent || '';
        expect(head).toMatch(/best eval @ step.*500/);
        expect(head).toMatch(/1\.400/);
        // No divergence on the healthy run.
        expect(screen.queryByTestId('loss-curve-divergence')).not.toBeInTheDocument();
        expect(screen.queryByTestId('loss-curve-divergence-note')).not.toBeInTheDocument();
    });

    it('shades the overfitting region and surfaces the divergence note when eval climbs after its minimum', () => {
        render(<LossCurvePanel metrics={overfitMetrics()} />);
        expect(screen.getByTestId('loss-curve-divergence')).toBeInTheDocument();
        const note = screen.getByTestId('loss-curve-divergence-note');
        // Best eval was at step 300 (eval=1.6); the largest later climb is to 1.8 → delta +0.2.
        expect(note.textContent).toMatch(/\+0\.200/);
        expect(note.textContent).toMatch(/step 300/);
        // Honest framing: name the gate-aligned action ("promote step 300, not the final").
        expect(note.textContent).toMatch(/promote-the-winner pick is\s+step 300/);
    });

    it('renders train-only when eval points are absent (early in a run)', () => {
        // First two steps have train_loss but no eval_loss yet.
        render(
            <LossCurvePanel
                metrics={[
                    { step: 50, train_loss: 2.8, eval_loss: null },
                    { step: 100, train_loss: 2.4, eval_loss: null },
                ]}
            />,
        );
        expect(screen.getByTestId('loss-curve-train-path')).toBeInTheDocument();
        expect(screen.queryByTestId('loss-curve-eval-path')).not.toBeInTheDocument();
        expect(screen.queryByTestId('loss-curve-best-marker')).not.toBeInTheDocument();
        expect(screen.queryByTestId('loss-curve-divergence')).not.toBeInTheDocument();
    });

    it('ignores divergence jitter below the noise threshold', () => {
        // Eval bottoms at 1.50 and only climbs to 1.52 — below the 0.05 threshold.
        render(
            <LossCurvePanel
                metrics={[
                    { step: 100, train_loss: 2.0, eval_loss: 1.80 },
                    { step: 200, train_loss: 1.6, eval_loss: 1.50 },
                    { step: 300, train_loss: 1.2, eval_loss: 1.52 },
                ]}
            />,
        );
        // Best marker still renders (we always know the min) — but no
        // overfitting region is drawn because the climb is jitter.
        expect(screen.getByTestId('loss-curve-best-marker')).toBeInTheDocument();
        expect(screen.queryByTestId('loss-curve-divergence')).not.toBeInTheDocument();
        expect(screen.queryByTestId('loss-curve-divergence-note')).not.toBeInTheDocument();
    });
});

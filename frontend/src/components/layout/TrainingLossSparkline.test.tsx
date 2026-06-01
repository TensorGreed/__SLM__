import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import TrainingLossSparkline from './TrainingLossSparkline';


describe('TrainingLossSparkline', () => {
    it('renders the empty placeholder when no train_loss points exist', () => {
        // First few seconds of a training run — checkpoint exists
        // but trainer hasn't logged loss yet. The bell should show
        // a dashed flat line so the row layout doesn't shift when
        // the first real point lands a second later.
        render(<TrainingLossSparkline points={[]} />);
        const svg = screen.getByTestId('training-loss-sparkline');
        expect(svg).toHaveAttribute('data-trend', 'empty');
        // Placeholder is a single <line> element, no polyline.
        expect(svg.querySelector('polyline')).toBeNull();
    });

    it('tints the polyline green when loss is trending down', () => {
        // Typical happy path mid-run — loss decreasing across the
        // window. The colour-coded tint is the load-bearing
        // affordance: user reads the trend at a glance.
        render(
            <TrainingLossSparkline
                points={[
                    { step: 100, train_loss: 0.5 },
                    { step: 200, train_loss: 0.4 },
                    { step: 300, train_loss: 0.3 },
                    { step: 400, train_loss: 0.25 },
                    { step: 500, train_loss: 0.2 },
                ]}
            />,
        );
        const svg = screen.getByTestId('training-loss-sparkline');
        expect(svg).toHaveAttribute('data-trend', 'down');
        // ARIA label carries the trend symbol for screen readers.
        const aria = svg.getAttribute('aria-label') || '';
        expect(aria).toContain('↘');
        expect(aria).toContain('5 points');
    });

    it('tints amber when loss is flat-lining', () => {
        // ~1% drift over the window — within the dead-zone. Amber
        // is the warning state: the user should investigate
        // whether learning rate / data quality is the bottleneck.
        render(
            <TrainingLossSparkline
                points={[
                    { step: 100, train_loss: 0.500 },
                    { step: 200, train_loss: 0.501 },
                    { step: 300, train_loss: 0.499 },
                    { step: 400, train_loss: 0.502 },
                ]}
            />,
        );
        const svg = screen.getByTestId('training-loss-sparkline');
        expect(svg).toHaveAttribute('data-trend', 'flat');
    });

    it('tints red when loss is trending up (worse than chance)', () => {
        // Diverging run — the worst-case signal we surface
        // proactively rather than letting the user discover at the
        // end. Red tint nudges them toward kill+restart.
        render(
            <TrainingLossSparkline
                points={[
                    { step: 100, train_loss: 0.20 },
                    { step: 200, train_loss: 0.30 },
                    { step: 300, train_loss: 0.40 },
                    { step: 400, train_loss: 0.50 },
                    { step: 500, train_loss: 0.60 },
                ]}
            />,
        );
        const svg = screen.getByTestId('training-loss-sparkline');
        expect(svg).toHaveAttribute('data-trend', 'up');
    });

    it('plots a polyline with one point per train_loss value', () => {
        // Eval rows (eval_loss only) don't count toward the line —
        // sparkline currently shows train_loss as the primary
        // signal. Regression guard so a future enhancement adding
        // eval markers doesn't accidentally double-count.
        render(
            <TrainingLossSparkline
                points={[
                    { step: 100, train_loss: 0.4 },
                    { step: 200, eval_loss: 0.42 },
                    { step: 200, train_loss: 0.3 },
                    { step: 300, train_loss: 0.25 },
                ]}
            />,
        );
        const polyline = screen
            .getByTestId('training-loss-sparkline')
            .querySelector('polyline');
        expect(polyline).not.toBeNull();
        // 3 train_loss points → 3 coordinate pairs in the points
        // attribute (space-separated).
        const pairs = (polyline!.getAttribute('points') || '').trim().split(/\s+/);
        expect(pairs).toHaveLength(3);
    });

    it('marks the last point with a dot so the eye lands on the latest step', () => {
        render(
            <TrainingLossSparkline
                points={[
                    { step: 100, train_loss: 0.4 },
                    { step: 200, train_loss: 0.3 },
                ]}
            />,
        );
        expect(
            screen.getByTestId('training-loss-sparkline-last-dot'),
        ).toBeInTheDocument();
    });

    it('handles all-equal losses without crashing (zero-span guard)', () => {
        // Edge case — early training where loss happens to land on
        // the same value for several consecutive logging steps.
        // Naive (loss - min) / (max - min) would divide by zero;
        // the component snaps to the midline instead.
        render(
            <TrainingLossSparkline
                points={[
                    { step: 100, train_loss: 0.5 },
                    { step: 200, train_loss: 0.5 },
                    { step: 300, train_loss: 0.5 },
                ]}
            />,
        );
        // No throw → test passes. Trend resolves to flat.
        expect(
            screen.getByTestId('training-loss-sparkline'),
        ).toHaveAttribute('data-trend', 'flat');
    });
});
